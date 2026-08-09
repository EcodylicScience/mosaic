"""Reconcile the transcode kind directory against the links that reach into it.

A transcode derivative is reachable through exactly one route: a forward-link
cell (``analysis_derivative_path`` / ``playback_derivative_path``) on an
originals row. Nothing else addresses one -- ``Dataset._derivative_facts``
matches rows *to* a path the link cell produced, and mosaic-api's playback
stream opens the same cell. So **a file no link names is reachable by nothing**,
and that is the whole liveness rule here.

Files accumulate in that state and nothing removes them. Retuning a recipe
writes a new derivative and overwrites the link cell; ``_set_back_link`` drops
only the row matching the *new* path, so the previous file and its row both
survive, referenced by nothing. Re-probing never removes a row, and a rescan
only ever clears a link whose file is already gone. This module is what removes
them.

**Deleting an unreferenced derivative costs a re-encode that was going to
happen anyway.** The transcode op's reuse gate is a conjunction --
``dest.exists() and already_linked`` -- so an unlinked file at a path the op is
about to write is *re-encoded over*, not reused. Keeping it buys nothing; the
value is in ``relink``, which writes the cell and makes the next run skip.

**What this cannot reach, and it is the larger population.** Derivatives written
before the content-addressed scheme sit *directly under the media root* as
siblings of ``index.csv`` and ``frames/``, and their links are usually still
live. Confining deletion to the transcode kind directory -- the only defensible
blast radius, since every other child of ``media/`` belongs to another kind --
means this module structurally cannot see them. ``scripts/clear_transcode_
derivatives.py`` is what clears those, and the two reaches are disjoint rather
than overlapping.

**Four things it refuses to touch**, each because being wrong there costs more
than the disk it would reclaim: a derivative whose source is no longer indexed
(it may be the only surviving copy of an archived video), a linked file with no
row (deleting it breaks a working read path and rebuilding the row needs a
probe), a row addressing a file outside the kind directory, and a row carrying
no ``recipe_hash`` at all. The last is the one that matters most -- see
:func:`classify_rows`.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from mosaic_media.transcode import Target

from mosaic.core.media.facts_columns import (
    MEDIA_INDEX_COLUMNS,
    derivative_column_for_target,
    media_row_uuid,
    read_link_cell,
)
from mosaic.core.pipeline.index_lock import index_lock
from mosaic.core.pipeline.media_index import (
    frame_from_rows,
    read_media_index,
    write_media_index_rows,
)
from mosaic.core.stored_paths import resolve_stored_path

# Every target, spelled out rather than read back from `get_args(Target)`: the
# runtime form of a Literal is untyped, and a target has to stay a `Target` all
# the way from a parsed filename to `derivative_column_for_target`. Adding one
# upstream makes this tuple the single place that needs the new member, and the
# annotation is what says so on the day it happens.
TARGETS: tuple[Target, ...] = ("analysis", "playback")

# Every forward-link column, derived from the target map rather than restated.
# Four copies of this pair already exist in the tree and a fifth would be one
# more place to forget a target if one is ever added.
LINK_COLUMNS: tuple[str, ...] = tuple(
    derivative_column_for_target(target) for target in TARGETS
)

# The name a derivative is written under: <video_uuid>.<recipe_hash>.<target>.mp4
# (transcode.TranscodeOp.run). A canonical UUID carries no dot, so a real
# derivative name splits into exactly these four parts.
_NAME_PARTS = 4
_MP4_SUFFIX = "mp4"

PruneClass = Literal[
    # Reachable and current. Untouched.
    "live",
    # Reachable, but under a recipe no current run would produce. Untouched and
    # named: reads are being served from a derivative the op would now rebuild.
    "live_legacy_recipe",
    # Unreferenced, but its name matches a live recipe over a live source, so
    # one link cell turns it back into a skipped encode. `relink` writes it.
    "relinkable",
    # Unreferenced and unreachable: no live recipe would produce this name.
    # The state this module exists for.
    "superseded",
    # A link cell naming a file that is not there. `relink` clears the cell.
    "dangling",
    # A row with neither a link nor a file left.
    "vanished",
    # A row whose abs_path cell is empty. It addresses nothing, so dropping it
    # cannot strand anything -- and it is what makes a re-probe abort.
    "unaddressed",
    # Unreferenced, but its source uuid is in no originals row. Refused.
    "unsourced",
    # A linked file with no row describing it. Refused, both directions.
    "unrowed",
    # A row naming a file outside the kind directory -- the pre-content-address
    # layout. Refused; the one-off sweep is what reaches these.
    "outside_kind_directory",
    # An entry under the kind directory that is not a derivative: an interrupted
    # encode's hidden temp, a subdirectory, a symlink, a non-mp4.
    "stray",
    # A row carrying no recipe_hash. Not a derivative at all. Refused.
    "foreign",
]

# Which classes a run is allowed to delete, and under which flag. Everything
# absent from here is reported and left alone.
_DELETABLE: frozenset[PruneClass] = frozenset({"superseded"})
_ROW_ONLY_DROPS: frozenset[PruneClass] = frozenset({"vanished", "unaddressed"})

DeclineReason = Literal[
    "no-media-root",
    "single-root",
    "one-index",
    "nested-root",
]

_DECLINE_TEXT: Mapping[DeclineReason, str] = {
    "no-media-root": (
        "this dataset has no media root, so there is no derivative index to prune"
    ),
    "single-root": (
        "this dataset has no media_raw root, so media/index.csv is its originals "
        "index; nothing here is a derivative and nothing will ever be pruned"
    ),
    "one-index": (
        "the media and media_raw roots resolve to one directory, so the two "
        "indexes are one file; nothing here is a derivative"
    ),
    "nested-root": (
        "a source or frames root resolves inside the transcode kind directory, "
        "so pruning it would delete originals"
    ),
}


def decline_text(reason: DeclineReason) -> str:
    """The operator-facing sentence for a declined run."""
    return _DECLINE_TEXT[reason]


@dataclass(frozen=True)
class PruneEntry:
    """One reconciled path, and what the run decided about it."""

    path: Path
    verdict: PruneClass
    # Empty when no row describes this path, which is exactly class `unrowed`
    # and the `stray` entries.
    row_index: int | None = None
    # The uuid and recipe read out of the filename, empty when it does not parse.
    video_uuid: str = ""
    recipe_hash: str = ""
    size_bytes: int = 0
    # Set when a decision was overruled by the age window, so the report can say
    # "would have, but it is too new" rather than silently keeping the file.
    held_for_age: bool = False


@dataclass(frozen=True)
class PruneReport:
    """What the run found, and what it did about it."""

    # False when a gate declined to look at all. Reported apart from a dry run,
    # which looked and would act: "would prune 0" reads as "run it again with
    # --apply", and on a dataset that can never hold a derivative that is a lie.
    considered: bool = False
    declined: DeclineReason | None = None
    applied: bool = False
    changed: bool = False
    backups: list[Path] = field(default_factory=list)
    entries: list[PruneEntry] = field(default_factory=list)
    files_deleted: list[Path] = field(default_factory=list)
    rows_dropped: int = 0
    links_relinked: list[str] = field(default_factory=list)
    links_cleared: list[str] = field(default_factory=list)
    bytes_reclaimed: int = 0
    held_for_age: int = 0
    # The recipes the run treated as current. Printed because they are the one
    # input an operator cannot see and can silently differ from the worker's --
    # media_thresholds() reads the environment, and the recipe folds it whole.
    live_recipes: dict[str, str] = field(default_factory=dict)

    def of(self, verdict: PruneClass) -> list[PruneEntry]:
        """Every entry the run put in *verdict*, in the order it decided them."""
        return [entry for entry in self.entries if entry.verdict == verdict]

    def counts(self) -> dict[str, int]:
        """How many entries landed in each class, classes with none omitted."""
        tally: dict[str, int] = {}
        for entry in self.entries:
            tally[entry.verdict] = tally.get(entry.verdict, 0) + 1
        return tally

    def payload(self) -> dict[str, object]:
        """The ``--json`` document: one flat object, no nested optionals."""
        return {
            "considered": self.considered,
            "declined": self.declined or "",
            "applied": self.applied,
            "changed": self.changed,
            "counts": self.counts(),
            "files_deleted_count": len(self.files_deleted),
            "files_deleted": [str(path) for path in self.files_deleted],
            "rows_dropped": self.rows_dropped,
            "bytes_reclaimed": self.bytes_reclaimed,
            "links_relinked": self.links_relinked,
            "links_cleared": self.links_cleared,
            "held_for_age": self.held_for_age,
            "live_recipes": self.live_recipes,
            "backups": [str(path) for path in self.backups],
            # Every refused class listed by path, because each names a repair a
            # person has to make rather than a thing this run can do.
            "unsourced": [str(e.path) for e in self.of("unsourced")],
            "unrowed": [str(e.path) for e in self.of("unrowed")],
            "outside_kind_directory": [
                str(e.path) for e in self.of("outside_kind_directory")
            ],
            "foreign": [str(e.path) for e in self.of("foreign")],
            "stray": [str(e.path) for e in self.of("stray")],
            "live_legacy_recipe": [str(e.path) for e in self.of("live_legacy_recipe")],
            "relinkable": [str(e.path) for e in self.of("relinkable")],
        }


def declined_report(reason: DeclineReason) -> PruneReport:
    """A report for a run that a gate stopped before it read anything."""
    return PruneReport(considered=False, declined=reason)


def parse_derivative_name(name: str) -> tuple[str, str, Target] | None:
    """Split ``<uuid>.<recipe>.<target>.mp4`` into its three parts, or ``None``.

    A leading dot fails the parse rather than yielding an empty uuid, which is
    what keeps an interrupted encode's hidden temp file -- written as
    ``.<stem>.XXXXXXXX.mp4`` beside its destination -- out of every class that
    can be deleted. It has more parts than this anyway; the empty first part is
    the belt to that braces.
    """
    parts = name.split(".")
    if len(parts) != _NAME_PARTS:
        return None
    video_uuid, recipe_hash, target, suffix = parts
    if not video_uuid or not recipe_hash or suffix != _MP4_SUFFIX:
        return None
    for known in TARGETS:
        if target == known:
            return video_uuid, recipe_hash, known
    return None


def live_link_targets(
    originals: Sequence[Mapping[str, str]], media_root: Path
) -> dict[Path, list[str]]:
    """Every path a forward-link cell names, mapped to the uuids naming it.

    The **union** over all originals rows, never per row: ``set_forward_link``
    assigns by uuid mask, so two byte-identical originals share one uuid and both
    carry the cell. A liveness count taken per row or per entry misses the second
    holder and deletes a file the first still resolves.

    Anchored at the media root, because that is what a link cell is relative to
    -- a derivative row's ``abs_path`` is relative to the dataset base directory
    instead, so the two must both be resolved before they can be compared.
    """
    live: dict[Path, list[str]] = {}
    for row in originals:
        uuid = media_row_uuid(row)
        for column in LINK_COLUMNS:
            cell = read_link_cell(row, column)
            if not cell:
                continue
            live.setdefault((media_root / cell).resolve(), []).append(uuid)
    return live


def classify_rows(
    derivative_rows: Sequence[Mapping[str, str]],
    *,
    base_directory: Path,
    transcode_root: Path,
) -> dict[int, tuple[Path | None, PruneClass | None]]:
    """Resolve each derivative row and reject the ones that are not derivatives.

    Returns ``{row index: (resolved path, refusal)}``; a ``None`` refusal means
    the row reached the reconciliation and its class comes from there.

    **The ``recipe_hash`` gate is the one that prevents the worst outcome this
    module could produce.** ``Dataset.resolve_media_root`` answers ``media_raw``
    the moment that root is *set*, so adding one to a dataset that never had it
    reinterprets an index still full of originals as the derivative index. The
    discriminator is cheap and sound: the transcode job is the only writer that
    fills ``recipe_hash``, a fresh probe always leaves it empty, and a re-probe
    is forbidden from ever writing it. A row without one is not a derivative,
    whatever index it is sitting in.
    """
    verdicts: dict[int, tuple[Path | None, PruneClass | None]] = {}
    resolved_kind_root = transcode_root.resolve()
    for position, row in enumerate(derivative_rows):
        if not read_link_cell(row, "recipe_hash"):
            verdicts[position] = (None, "foreign")
            continue
        stored = read_link_cell(row, "abs_path")
        if not stored:
            verdicts[position] = (None, "unaddressed")
            continue
        path = resolve_stored_path(stored, base_directory).resolve()
        if resolved_kind_root not in path.parents:
            verdicts[position] = (path, "outside_kind_directory")
            continue
        verdicts[position] = (path, None)
    return verdicts


def _too_young(path: Path, cutoff: datetime) -> bool:
    """Was *path* modified after the age window opened?

    An in-flight encode's temp file and a derivative registered a second ago are
    indistinguishable from stranded ones by name, so the window is what keeps a
    prune from racing a running job. A file that has vanished under us is not
    young; it is gone, and the delete that follows is a no-op either way.
    """
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return False
    return modified > cutoff


def reconcile(
    *,
    originals: Sequence[Mapping[str, str]],
    derivative_rows: Sequence[Mapping[str, str]],
    media_root: Path,
    transcode_root: Path,
    base_directory: Path,
    live_recipes: Mapping[str, str],
    cutoff: datetime,
) -> list[PruneEntry]:
    """Decide every path reachable from the links, the rows, or the directory.

    Pure: reads the filesystem for existence, size and mtime, and writes
    nothing. The union of the three sources is walked in sorted path order so
    two runs over one dataset produce identical reports.
    """
    live = live_link_targets(originals, media_root)
    row_verdicts = classify_rows(
        derivative_rows, base_directory=base_directory, transcode_root=transcode_root
    )
    row_by_path: dict[Path, int] = {
        path: position
        for position, (path, refusal) in row_verdicts.items()
        if path is not None and refusal is None
    }
    source_uuids = {uuid for row in originals if (uuid := media_row_uuid(row)) and uuid}
    files = (
        sorted(p.resolve() for p in transcode_root.iterdir())
        if transcode_root.is_dir()
        else []
    )

    entries: list[PruneEntry] = []
    # Rows refused before reconciliation: they are decided by the row alone and
    # never appear in the path walk, because their path may not exist or may not
    # be under the kind directory at all.
    for position, (path, refusal) in sorted(row_verdicts.items()):
        if refusal is None:
            continue
        entries.append(
            PruneEntry(
                path=path if path is not None else Path(),
                verdict=refusal,
                row_index=position,
            )
        )

    for path in sorted(set(files) | set(row_by_path) | set(live)):
        # A live link may name a file outside the kind directory -- the
        # pre-content-address layout, whose links are usually still live. It is
        # reachable, so it is not this module's business either way.
        if path in live and transcode_root.resolve() not in path.parents:
            continue
        position = row_by_path.get(path)
        exists = path.is_file()
        parsed = parse_derivative_name(path.name)
        entries.append(
            _decide(
                path=path,
                position=position,
                linked=path in live,
                exists=exists,
                parsed=parsed,
                is_dir_or_link=path.is_dir() or path.is_symlink(),
                source_uuids=source_uuids,
                live_recipes=live_recipes,
                cutoff=cutoff,
            )
        )
    return entries


def _decide(
    *,
    path: Path,
    position: int | None,
    linked: bool,
    exists: bool,
    parsed: tuple[str, str, Target] | None,
    is_dir_or_link: bool,
    source_uuids: frozenset[str] | set[str],
    live_recipes: Mapping[str, str],
    cutoff: datetime,
) -> PruneEntry:
    """The class of one reconciled path.

    The (link, row, file) triple decides it, plus one reachability question for
    the unlinked-but-present case. A row's presence never changes the *file*
    decision -- only whether a row is dropped alongside it -- which is what lets
    a run interrupted between the row write and the unlink finish its own work
    on the next pass.
    """
    video_uuid, recipe_hash = ("", "")
    if parsed is not None:
        video_uuid, recipe_hash, _ = parsed
    size = path.stat().st_size if exists else 0

    if not exists:
        # A directory or a symlink under the kind directory is neither a
        # derivative nor missing; it is something this module does not own.
        if is_dir_or_link:
            return PruneEntry(path=path, verdict="stray", row_index=position)
        verdict: PruneClass = "dangling" if linked else "vanished"
        return PruneEntry(path=path, verdict=verdict, row_index=position)

    if is_dir_or_link or parsed is None:
        return PruneEntry(
            path=path, verdict="stray", row_index=position, size_bytes=size
        )

    _, _, target = parsed
    current = live_recipes.get(target, "")
    if linked:
        live_now = recipe_hash == current
        if position is None:
            return PruneEntry(
                path=path,
                verdict="unrowed",
                video_uuid=video_uuid,
                recipe_hash=recipe_hash,
                size_bytes=size,
            )
        return PruneEntry(
            path=path,
            verdict="live" if live_now else "live_legacy_recipe",
            row_index=position,
            video_uuid=video_uuid,
            recipe_hash=recipe_hash,
            size_bytes=size,
        )

    if video_uuid not in source_uuids:
        return PruneEntry(
            path=path,
            verdict="unsourced",
            row_index=position,
            video_uuid=video_uuid,
            recipe_hash=recipe_hash,
            size_bytes=size,
        )
    if recipe_hash == current:
        return PruneEntry(
            path=path,
            verdict="relinkable",
            row_index=position,
            video_uuid=video_uuid,
            recipe_hash=recipe_hash,
            size_bytes=size,
        )
    return PruneEntry(
        path=path,
        verdict="superseded",
        row_index=position,
        video_uuid=video_uuid,
        recipe_hash=recipe_hash,
        size_bytes=size,
        held_for_age=_too_young(path, cutoff),
    )


def _backup_index(index_path: Path) -> Path:
    """Copy *index_path* aside under a UTC timestamp before it is modified.

    The same scheme the re-probe command uses, and deliberately ``with_name``
    rather than ``with_suffix``: replacing ``.csv`` would put a second backup
    spelling in one media root. The copy is the only route back from a prune --
    the rows it drops are not otherwise recoverable, though the files they name
    are, by re-running the transcode.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = index_path.with_name(f"{index_path.name}.{stamp}.backup")
    _ = shutil.copy2(index_path, backup)
    return backup


def _relink_originals(
    originals: list[dict[str, str]],
    entries: Sequence[PruneEntry],
    media_root: Path,
) -> tuple[list[str], list[str]]:
    """Write the cells `relink` fills and clear the ones it empties, in memory.

    Returns ``(relinked, cleared)`` as media-root-relative cell values. A
    ``relinkable`` file whose target cell already names something is left alone
    and reported: overwriting a live link to adopt an unreferenced file would
    strand the one it displaced, turning a repair into the thing being repaired.
    """
    relinked: list[str] = []
    cleared: list[str] = []
    by_uuid: dict[str, list[dict[str, str]]] = {}
    for row in originals:
        uuid = media_row_uuid(row)
        if uuid:
            by_uuid.setdefault(uuid, []).append(row)

    for entry in entries:
        if entry.verdict == "dangling":
            for row in originals:
                for column in LINK_COLUMNS:
                    cell = read_link_cell(row, column)
                    if cell and (media_root / cell).resolve() == entry.path:
                        row[column] = ""
                        cleared.append(cell)
            continue
        if entry.verdict != "relinkable":
            continue
        parsed = parse_derivative_name(entry.path.name)
        if parsed is None:
            continue
        _, _, target = parsed
        column = derivative_column_for_target(target)
        cell = entry.path.relative_to(media_root.resolve()).as_posix()
        for row in by_uuid.get(entry.video_uuid, []):
            if read_link_cell(row, column):
                continue
            row[column] = cell
            relinked.append(cell)
    return relinked, cleared


def _project(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    """Widen string records onto the full schema for the writer."""
    return [
        {column: row.get(column, "") for column in MEDIA_INDEX_COLUMNS} for row in rows
    ]


def prune_media(
    originals_index_path: Path,
    *,
    derivative_index_path: Path,
    media_root: Path,
    transcode_root: Path,
    base_directory: Path,
    live_recipes: Mapping[str, str],
    apply: bool,
    min_age_hours: float = 24.0,
    relink: bool = False,
    include_stray: bool = False,
    now: datetime | None = None,
) -> PruneReport:
    """Delete the transcode derivatives nothing reaches, and optionally repair.

    *live_recipes* maps each target to the recipe hash a run would produce
    today; a file named under any other recipe, and referenced by no link, is
    what this deletes. *now* is injectable so the age window can be tested
    without sleeping.

    Dry-run unless *apply*. When nothing needs changing it writes nothing at
    all, so a second run over a reconciled dataset leaves both indexes
    byte-identical.

    **Write order, and it is the safe one of the two.** Two sequential locked
    blocks -- originals first, derivatives second -- never nested, for
    lock-ordering hygiene rather than safety: ``index_lock``'s sidecar survives
    the atomic rename, so a block that wrote twice would keep its grip. The
    unlinks come after both. A crash between the row drop and the unlink leaves
    files on disk that no row and no link describes, which the next run decides
    identically and removes: the file predicate never consults the row. The
    reverse order would leave rows addressing files that are already gone.

    The originals index is written only when a cell actually changes, so the
    ordinary case -- a retuned recipe, whose links all still name live files --
    does not touch it at all.
    """
    moment = now or datetime.now(timezone.utc)
    cutoff = moment - timedelta(hours=min_age_hours)
    originals = [dict(row) for row in read_media_index(originals_index_path)]
    derivative_rows = [dict(row) for row in read_media_index(derivative_index_path)]

    entries = reconcile(
        originals=originals,
        derivative_rows=derivative_rows,
        media_root=media_root,
        transcode_root=transcode_root,
        base_directory=base_directory,
        live_recipes=live_recipes,
        cutoff=cutoff,
    )

    delete_files = [
        entry
        for entry in entries
        if entry.verdict in _DELETABLE and not entry.held_for_age
    ]
    if include_stray:
        # Only a stray that is a file, and only one old enough to be past an
        # in-flight encode: an interrupted encode's temp is exactly the shape
        # this sweeps, and a running one is the same shape.
        delete_files += [
            entry
            for entry in entries
            if entry.verdict == "stray"
            and entry.path.is_file()
            and not _too_young(entry.path, cutoff)
        ]
    drop_positions = {
        entry.row_index
        for entry in [
            *delete_files,
            *(e for e in entries if e.verdict in _ROW_ONLY_DROPS),
        ]
        if entry.row_index is not None
    }
    if relink:
        drop_positions |= {
            entry.row_index
            for entry in entries
            if entry.verdict == "dangling" and entry.row_index is not None
        }

    relinked, cleared = (
        _relink_originals(originals, entries, media_root) if relink else ([], [])
    )
    changed = bool(delete_files or drop_positions or relinked or cleared)
    backups: list[Path] = []

    if apply and changed:
        if (relinked or cleared) and originals_index_path.exists():
            backups.append(_backup_index(originals_index_path))
            with index_lock(originals_index_path):
                write_media_index_rows(
                    originals_index_path, frame_from_rows(_project(originals))
                )
            # Deliberately no sequences.csv re-projection. A media composition is
            # the ordered uids of a sequence's members; a forward-link cell is
            # not one of its terms, so clearing one cannot move the digest. The
            # repair path if that is ever wrong is rebuild_sequence_index.
        if drop_positions and derivative_index_path.exists():
            backups.append(_backup_index(derivative_index_path))
            with index_lock(derivative_index_path):
                kept = [
                    row
                    for position, row in enumerate(derivative_rows)
                    if position not in drop_positions
                ]
                write_media_index_rows(
                    derivative_index_path, frame_from_rows(_project(kept))
                )
        for entry in delete_files:
            entry.path.unlink(missing_ok=True)

    return PruneReport(
        considered=True,
        applied=apply and changed,
        changed=changed,
        backups=backups,
        entries=entries,
        files_deleted=[entry.path for entry in delete_files],
        rows_dropped=len(drop_positions),
        links_relinked=sorted(relinked),
        links_cleared=sorted(cleared),
        bytes_reclaimed=sum(entry.size_bytes for entry in delete_files),
        held_for_age=sum(1 for entry in entries if entry.held_for_age),
        live_recipes=dict(live_recipes),
    )
