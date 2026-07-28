"""Re-probe an existing media index in place, minting per-file identity.

Scanning a media tree builds a media index; projecting explicit sequence
assignments writes one. This module is the third mode, for a media index that
already exists and is authoritative: its ``(group, sequence)`` assignments, its
``video_order`` and its transcode links are curated data, and only the measured
cells are stale. Every file the index lists is re-probed and the fresh
measurement written back into that row. No directory is scanned, no row is
constructed, and no row is added or removed.

The entry point takes plain paths, never a dataset. Deciding which roots an index
lives under is the caller's business; this module receives those answers and does
the measuring, so it stays a cheap leaf that a path and a CSV are enough to
drive.

It is also the migration path for an index written before a column existed. The
production datasets this serves have index files whose header lacks the identity
columns outright, so each record is read as exact strings with
``read_media_index`` and widened against ``MEDIA_INDEX_COLUMNS`` in Python.
``load_media_index_frame`` would be the obvious reader and is unusable here: its
``pd.read_csv`` applies pandas' default NA literals, so a curated ``group="NA"``
or ``camera="NA"`` is destroyed before the NaN handling runs. Writing back
through ``frame_from_rows`` never re-parses, so every unpatched cell survives
byte-identical -- the same pipeline ``write_media_index`` uses for the rows it
preserves.

``video_uuid`` is derived from the probed content and timing, not minted from
entropy, so re-probing an unchanged file reproduces its exact identity. That is
what makes the fresh probe safe to treat as authoritative for the identity
columns, and what makes a second run over an unchanged dataset a no-op.
"""

from __future__ import annotations

import csv
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from mosaic_media import MediaProbeError

from mosaic.core.helpers import to_safe_name
from mosaic.core.pipeline.media_index import (
    frame_from_rows,
    mtime_iso,
    read_media_index,
    write_media_index_rows,
)
from mosaic.core.stored_paths import resolve_stored_path

from .facts_columns import (
    FACTS_JSON_COLUMN,
    MEDIA_INDEX_COLUMNS,
    ProbeMetadata,
    read_link_cell,
    row_facts_or_none,
)
from .imgstore_io import imgstore_probe, is_imgstore
from .probe_row import probe_video_metadata

# The five cells a fresh probe always leaves empty because they record a
# transcode decision rather than a measurement. The patch must never write them:
# doing so would erase the media index's link graph and the recipe a derivative
# was produced under.
UNMEASURED_LINK_COLUMNS = frozenset(
    {
        "analysis_derivative_path",
        "playback_derivative_path",
        "source_path",
        "source_video_uuid",
        "recipe_hash",
    }
)

IdentityChange = Literal[
    "minted",
    "unchanged",
    "content_digest_changed",
    "video_uuid_changed",
    "unmintable",
]


class ReprobeAbort(Exception):
    """The media index cannot be re-probed. Raised before anything is written.

    Names every offending row, so one run tells the operator the whole repair
    rather than one item of it. The two unreadable conditions stay apart here
    exactly as they do in the report: media archived years ago and media that is
    present but will not probe call for different responses, and one merged list
    would hide which is which.
    """

    def __init__(
        self,
        message: str,
        *,
        missing: Sequence[str] = (),
        unprobeable: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.missing: list[str] = list(missing)
        self.unprobeable: list[str] = list(unprobeable)


@dataclass(frozen=True)
class MediaIndex:
    """A media index read as exact string cells, widened to the full schema.

    ``rows`` holds one full-schema mapping per row, in file order, with
    every cell exactly as the file spelled it. ``present_columns`` is the file's
    real header -- the only place the pre-widen truth survives, and what says
    whether the schema needs migrating.
    """

    index_path: Path
    rows: list[dict[str, object]]
    present_columns: list[str]

    @property
    def missing_columns(self) -> list[str]:
        """Schema columns the file on disk does not have."""
        present = set(self.present_columns)
        return [column for column in MEDIA_INDEX_COLUMNS if column not in present]

    @property
    def unknown_columns(self) -> list[str]:
        """Columns the file has that the schema does not, and a write will drop."""
        return sorted(set(self.present_columns) - set(MEDIA_INDEX_COLUMNS))


@dataclass(frozen=True)
class ProbedRow:
    """One index row's fresh measurement, and what it implies for that row."""

    row_number: int
    stored_path: str
    resolved_path: Path
    probe: ProbeMetadata
    size_bytes: int
    modified_at: str
    change: IdentityChange
    needs_patch: bool
    facts_rebuilt: bool


@dataclass(frozen=True)
class ProbeOutcome:
    """Every row that was measured, and the two ways a row can fail to be.

    ``missing`` and ``unprobeable`` stay apart the whole way to the report. They
    mean different things to an operator: media archived or deleted years ago is
    expected on an old corpus, while media that is present and will not probe is
    a corruption signal that may warrant investigating the storage rather than
    skipping past it.
    """

    probed: list[ProbedRow] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unprobeable: list[str] = field(default_factory=list)


def _empty_patches() -> dict[int, dict[str, object]]:
    """Typed default for :attr:`DerivativePlan.patches` (a bare ``dict`` as a
    default_factory infers ``dict[Unknown, Unknown]`` under strict typing)."""
    return {}


@dataclass(frozen=True)
class DerivativePlan:
    """The derivative index's share of the work, computed before any write.

    Its rows are classified exactly as the ``media_raw`` index's are, so a
    derivative whose measured identity has drifted from its recorded one is
    counted and named here too. A derivative is a file on disk like any other:
    it can be re-encoded or replaced, and a run that reported drift only for
    originals would stay silent about it.
    """

    media_index: MediaIndex | None = None
    rows: int = 0
    minted: int = 0
    relinked: int = 0
    facts_rebuilt: int = 0
    missing_columns: list[str] = field(default_factory=list)
    unknown_columns: list[str] = field(default_factory=list)
    content_digest_changed: list[str] = field(default_factory=list)
    video_uuid_changed: list[str] = field(default_factory=list)
    patches: Mapping[int, dict[str, object]] = field(default_factory=_empty_patches)
    unresolved: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unprobeable: list[str] = field(default_factory=list)

    @property
    def has_work(self) -> bool:
        return bool(self.patches or self.missing_columns)


@dataclass(frozen=True)
class ReprobeReport:
    """What the run found, and what it did about it."""

    index_path: Path
    applied: bool
    changed: bool
    backups: list[Path]
    rows_total: int
    minted: int
    unchanged: int
    unmintable: int
    content_digest_changed: list[str]
    video_uuid_changed: list[str]
    rows_patched: int
    facts_rebuilt: int
    missing: list[str]
    unprobeable: list[str]
    schema_columns_added: list[str]
    unknown_columns: list[str]
    derivative: DerivativePlan

    def payload(self) -> dict[str, object]:
        """The ``--json`` document: one flat object, no nested optionals."""
        derivative_media_index = self.derivative.media_index
        derivative_index = (
            None
            if derivative_media_index is None
            else str(derivative_media_index.index_path)
        )
        return {
            "applied": self.applied,
            "changed": self.changed,
            "index": str(self.index_path),
            "backups": [str(path) for path in self.backups],
            "rows_total": self.rows_total,
            "rows_patched": self.rows_patched,
            "facts_rebuilt": self.facts_rebuilt,
            "identity_minted": self.minted,
            "identity_unchanged": self.unchanged,
            "identity_unmintable": self.unmintable,
            "content_digest_changed": self.content_digest_changed,
            "video_uuid_changed": self.video_uuid_changed,
            # The two unreadable conditions are counted and listed apart: a
            # missing file is expected on an old corpus, an unprobeable one is a
            # corruption signal.
            "media_missing_count": len(self.missing),
            "media_missing": self.missing,
            "media_unprobeable_count": len(self.unprobeable),
            "media_unprobeable": self.unprobeable,
            "schema_columns_added": self.schema_columns_added,
            "unknown_columns_dropped": self.unknown_columns,
            "derivative_index": derivative_index,
            "derivative_rows": self.derivative.rows,
            "derivative_identity_minted": self.derivative.minted,
            "derivative_facts_rebuilt": self.derivative.facts_rebuilt,
            "derivative_content_digest_changed": self.derivative.content_digest_changed,
            "derivative_video_uuid_changed": self.derivative.video_uuid_changed,
            "derivative_schema_columns_added": self.derivative.missing_columns,
            "derivative_unknown_columns_dropped": self.derivative.unknown_columns,
            "derivative_links_rekeyed": self.derivative.relinked,
            "derivative_links_unresolved": self.derivative.unresolved,
            "derivative_media_missing_count": len(self.derivative.missing),
            "derivative_media_missing": self.derivative.missing,
            "derivative_media_unprobeable_count": len(self.derivative.unprobeable),
            "derivative_media_unprobeable": self.derivative.unprobeable,
        }


def measured_cells(probed: ProbedRow) -> dict[str, object]:
    """The cells a re-probe is entitled to overwrite on an existing row.

    The probe's own cells minus the five link columns it always leaves empty,
    plus the two filesystem facts the probe does not carry.
    """
    cells: dict[str, object] = {
        key: value
        for key, value in probed.probe.items()
        if key not in UNMEASURED_LINK_COLUMNS
    }
    cells["size_bytes"] = probed.size_bytes
    cells["mtime_iso"] = probed.modified_at
    return cells


def _derivable_cells(row: Mapping[str, object]) -> dict[str, object]:
    """The safe-name cells to fill when the row has none.

    ``group_safe`` and ``sequence_safe`` are pure functions of ``group`` and
    ``sequence``, which the row already holds, and an empty one degrades the
    safe-name fallback in ``Dataset._match_media_rows``. Filled only when absent: a stored
    value that differs from the derivation is a curated choice this command does
    not overrule.
    """
    cells: dict[str, object] = {}
    group = read_link_cell(row, "group")
    sequence = read_link_cell(row, "sequence")
    if group and not read_link_cell(row, "group_safe"):
        cells["group_safe"] = to_safe_name(group)
    if sequence and not read_link_cell(row, "sequence_safe"):
        cells["sequence_safe"] = to_safe_name(sequence)
    return cells


def _facts_cell_is_stale(row: Mapping[str, object]) -> bool:
    """True when the row stores a facts cell that no longer reconstructs.

    A cell written under an older required field set still holds well-formed
    JSON, so the only way to see that it has gone stale is to attempt the
    reconstruction and watch it fail.
    """
    if not read_link_cell(row, FACTS_JSON_COLUMN):
        return False
    return row_facts_or_none(row) is None


def _classify(row: Mapping[str, object], probe: ProbeMetadata) -> IdentityChange:
    """Compare a row's recorded identity against the freshly probed one.

    Absence is tested before divergence and wins: a row that recorded no identity
    has nothing for the measurement to diverge from, so it is minted, not
    drifted. Treating an absent value as a differing one would report every row
    of a pre-identity media index as drift on the one run where a clean report
    matters.

    **The absence test is split by what the row's regime actually mints**, and
    that split is not optional. A store carries a uuid (its ``__store.uuid``,
    open item O5) and never a ``content_digest`` -- there is no elementary stream
    to hash. Under a single "either is empty" test a store would stay permanently
    ``unmintable``, so ``reprobe --apply`` could never backfill the uuid onto an
    already-indexed row. Under a naive fix it would classify ``"minted"`` on
    every run, because ``recorded_digest`` is empty forever, and
    :func:`_needs_patch` would then rewrite the index and drop a backup on every
    run without bound -- the failure that function's docstring already warns
    about. Branching on whether the *probe* produced a digest gives each regime
    its own settled state.
    """
    if not probe["video_uuid"]:
        # No identity at all: a store whose metadata carries no uuid, or a probe
        # that produced nothing. Neither has anything to mint or compare.
        return "unmintable"
    recorded_uuid = read_link_cell(row, "video_uuid")
    if not probe["content_digest"]:
        # A store: uuid-only. It settles on the uuid alone, and a re-recorded
        # store (a fresh mint in the same directory) is detected. Chunks edited
        # in place are not -- see STORE_IDENTITY_SCHEME, which exists to say so.
        if not recorded_uuid:
            return "minted"
        if recorded_uuid != probe["video_uuid"]:
            return "video_uuid_changed"
        return "unchanged"
    recorded_digest = read_link_cell(row, "content_digest")
    if not recorded_uuid or not recorded_digest:
        return "minted"
    if recorded_digest != probe["content_digest"]:
        return "content_digest_changed"
    if recorded_uuid != probe["video_uuid"]:
        return "video_uuid_changed"
    return "unchanged"


def _needs_patch(row: Mapping[str, object], change: IdentityChange) -> bool:
    """True when this row's line would not be written back identically.

    ``"unmintable"`` means identity can never call this row settled; were that
    treated as needing a write, every applied run over a media index holding one
    would rewrite it and drop another backup, without bound. Its stored
    MediaFacts cell decides instead, and that cell is empty exactly once -- on
    the run that first measures the file. A store used to live here permanently;
    since open item O5 it mints its ``__store.uuid`` and settles on
    ``"unchanged"`` like anything else, and only a store whose metadata carries
    no uuid at all still lands in this branch.

    A stale facts cell needs a write even when identity classifies the row
    ``"unchanged"``: the same bytes still mint the same uuid and digest, so
    identity comparison cannot see the cell has gone stale and
    :func:`_facts_cell_is_stale` is the only detector. Checked ahead of the
    ``"unchanged"`` return, which would otherwise treat the row as settled and
    skip it.
    """
    if _derivable_cells(row) or _facts_cell_is_stale(row):
        return True
    if change == "unchanged":
        return False
    if change == "unmintable":
        return not read_link_cell(row, FACTS_JSON_COLUMN)
    return True


def _rebuilds_facts(row: Mapping[str, object], change: IdentityChange) -> bool:
    """True when healing a stale facts cell is what forces this row's write.

    The count this feeds exists to explain a rewrite no other report line does,
    so a row whose identity also drifted stays out of it: that rewrite is already
    named by its own ``content_digest`` or ``video_uuid`` line, and counting the
    row twice would read as two rewritten rows. An ``"unmintable"`` row belongs in
    the count -- a store's identity is never settled, so it can never explain a
    rewrite -- and reaches it only with a cell present, since a stale cell is a
    present one.
    """
    return _facts_cell_is_stale(row) and change in {"unchanged", "unmintable"}


def _header_columns(index_path: Path) -> list[str]:
    """The file's real header line, before any widening."""
    no_header: list[str] = []
    with index_path.open(newline="") as handle:
        return next(csv.reader(handle), no_header)


def _read_widened_media_index(index_path: Path) -> MediaIndex:
    """Read a media index as exact string cells, widened to the full schema.

    ``load_media_index_frame`` is deliberately not used: its ``pd.read_csv``
    applies pandas' default NA literals, so a curated ``NA`` / ``None`` /
    ``NULL`` cell is already destroyed by the time its NaN handling runs, and a
    ``video_order`` column with one blank cell comes back as ``float64``.
    """
    if not index_path.exists():
        raise ReprobeAbort(f"no media index to re-probe at {index_path}")
    records = read_media_index(index_path)
    return MediaIndex(
        index_path=index_path,
        rows=[
            {column: record.get(column, "") for column in MEDIA_INDEX_COLUMNS}
            for record in records
        ],
        present_columns=_header_columns(index_path),
    )


def _unreadable_abort(missing: list[str], unprobeable: list[str]) -> ReprobeAbort:
    """The abort for unreadable media, naming whichever condition was hit.

    Both conditions when both are present, so one run tells the operator the
    whole repair and which part of it is a corruption signal rather than an
    archived file. The two lists ride on the exception separately, so a caller
    reports them apart exactly as the report does.
    """
    conditions: list[str] = []
    if missing:
        conditions.append(f"{len(missing)} row(s) point at media that is not on disk")
    if unprobeable:
        conditions.append(
            f"{len(unprobeable)} row(s) point at media that cannot be probed"
        )
    message = (
        f"{' and '.join(conditions)}; pass --skip-unreadable to leave them "
        f"untouched and migrate the rest"
    )
    return ReprobeAbort(message, missing=missing, unprobeable=unprobeable)


def _probe_media_index(
    media_index: MediaIndex, label: str, *, base_directory: Path
) -> ProbeOutcome:
    """Resolve and probe every row once, collecting the rows that cannot be.

    The single measurement of the run: it feeds the classification, the change
    decision, and the patch. Resolution reads ``abs_path`` and nothing else, so
    it behaves identically on a media index with no identity columns. A row that
    cannot be measured lands in ``missing`` or ``unprobeable`` -- never a shared
    bucket -- so the report can keep an archived file and a corrupt one apart.

    Never aborts: the caller probes every index first and then decides, so one
    run can name the whole repair across both of them.
    """
    probed: list[ProbedRow] = []
    missing: list[str] = []
    unprobeable: list[str] = []
    for row_number, row in enumerate(media_index.rows):
        stored = read_link_cell(row, "abs_path")
        if not stored:
            # A row with no path is a malformed index row, not archived media.
            # Counting it as missing would inflate the one number whose meaning
            # is "deleted long ago, and skipping past it is the right move".
            unprobeable.append(f"{label} row {row_number}: no abs_path cell to resolve")
            continue
        path = resolve_stored_path(stored, base_directory)
        if not path.exists():
            missing.append(
                f"{label} row {row_number}: {stored} is not on disk ({path})"
            )
            continue
        try:
            # The curated cell first, then the filesystem. An index old enough to
            # lack the identity columns -- the corpus this exists for -- may also
            # predate media_type, and probing a store as a plain video fails and
            # aborts the whole run. is_imgstore only reads metadata.yaml, so the
            # fallback costs a stat on a path that is already being probed.
            stored_type = read_link_cell(row, "media_type")
            is_store = stored_type == "imgstore" or (
                not stored_type and is_imgstore(path)
            )
            probe = imgstore_probe(path) if is_store else probe_video_metadata(path)
        except (OSError, MediaProbeError) as error:
            unprobeable.append(
                f"{label} row {row_number}: {path} cannot be probed: {error}"
            )
            continue
        stat_result = path.stat()
        change = _classify(row, probe)
        probed.append(
            ProbedRow(
                row_number=row_number,
                stored_path=stored,
                resolved_path=path.resolve(),
                probe=probe,
                size_bytes=stat_result.st_size,
                modified_at=mtime_iso(stat_result.st_mtime),
                change=change,
                needs_patch=_needs_patch(row, change),
                facts_rebuilt=_rebuilds_facts(row, change),
            )
        )
    return ProbeOutcome(probed=probed, missing=missing, unprobeable=unprobeable)


def _patches_for(
    media_index: MediaIndex, outcome: ProbeOutcome
) -> dict[int, dict[str, object]]:
    """The cells to write per row: the fresh measurement plus any derivable name."""
    return {
        row.row_number: {
            **measured_cells(row),
            **_derivable_cells(media_index.rows[row.row_number]),
        }
        for row in outcome.probed
        if row.needs_patch
    }


def _plan_derivatives(
    index_path: Path | None,
    uuid_by_path: Mapping[Path, str],
    *,
    label: str,
    media_raw_root: Path,
    base_directory: Path,
) -> DerivativePlan:
    """Plan the derivative index's re-probe and back-link re-key.

    The derivative rows carry ``source_video_uuid``, which nothing else in a
    rewrite touches, and a fresh probe leaves it empty. Re-keying it here is what
    keeps the link graph resolvable by uuid once the originals are minted.

    *index_path* is ``None`` when there is no ``media`` index distinct from the
    ``media_raw`` one, which makes the whole pass a no-op. A path that names
    no file is equally a no-op rather than an abort: a dataset that has never
    transcoded anything is a normal state, not a broken one.
    """
    if index_path is None or not index_path.exists():
        return DerivativePlan()

    media_index = _read_widened_media_index(index_path)
    outcome = _probe_media_index(media_index, label, base_directory=base_directory)
    patches = _patches_for(media_index, outcome)

    relinked = 0
    unresolved: list[str] = []
    # Only rows that were actually probed: a row left untouched because its own
    # media is missing or unprobeable stays verbatim, back-link included. Writing
    # one would break that promise for no gain, since a derivative whose file is
    # absent never reaches the uuid pass that reads it.
    for probed in outcome.probed:
        row = media_index.rows[probed.row_number]
        source_cell = read_link_cell(row, "source_path")
        if not source_cell:
            continue
        source = resolve_stored_path(source_cell, media_raw_root).resolve()
        source_uuid = uuid_by_path.get(source)
        if source_uuid is None:
            unresolved.append(f"{label} row {probed.row_number}: {source_cell}")
            continue
        if read_link_cell(row, "source_video_uuid") == source_uuid:
            continue
        patches.setdefault(probed.row_number, {})["source_video_uuid"] = source_uuid
        relinked += 1

    changed = bool(patches or media_index.missing_columns)
    return DerivativePlan(
        media_index=media_index,
        rows=len(media_index.rows),
        minted=sum(1 for row in outcome.probed if row.change == "minted"),
        relinked=relinked,
        facts_rebuilt=sum(1 for row in outcome.probed if row.facts_rebuilt),
        missing_columns=media_index.missing_columns,
        # Nothing is dropped by a run that does not rewrite this file.
        unknown_columns=media_index.unknown_columns if changed else [],
        content_digest_changed=[
            row.stored_path
            for row in outcome.probed
            if row.change == "content_digest_changed"
        ],
        video_uuid_changed=[
            row.stored_path
            for row in outcome.probed
            if row.change == "video_uuid_changed"
        ],
        patches=patches,
        unresolved=unresolved,
        missing=outcome.missing,
        unprobeable=outcome.unprobeable,
    )


def _backup_index(index_path: Path) -> Path:
    """Copy *index_path* aside under a UTC timestamp before it is modified.

    This rewrite changes the file's schema, not only its values, so the copy is
    the only record of the media index in its original shape -- including any
    non-schema column the write drops. The ``.backup`` suffix is not a media
    extension, so a backup can never be picked up as media by a later scan.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = index_path.with_name(f"{index_path.name}.{stamp}.backup")
    _ = shutil.copy2(index_path, backup)
    return backup


def _write_media_index(
    media_index: MediaIndex, patches: Mapping[int, dict[str, object]]
) -> None:
    """Apply *patches* to the index's rows and rewrite the file atomically.

    The rows are the exact strings the file held, widened to the schema, so every
    cell this command does not patch is written back byte-identical -- including
    the values a re-parse would have destroyed. Writing at all also completes the
    schema, because the rows were widened on read and the writer projects onto
    MEDIA_INDEX_COLUMNS; that is why the caller invokes this for an empty patch
    set when the file's header is short of the schema.
    """
    rows = [dict(row) for row in media_index.rows]
    for row_number, cells in patches.items():
        rows[row_number].update(cells)
    write_media_index_rows(media_index.index_path, frame_from_rows(rows))


def reprobe_media(
    index_path: Path,
    *,
    derivative_index_path: Path | None,
    media_raw_root: Path,
    base_directory: Path,
    apply: bool,
    skip_unreadable: bool = False,
) -> ReprobeReport:
    """Re-probe the media an existing index lists, minting per-file identity.

    *index_path* is the ``media_raw`` index to re-probe.
    *derivative_index_path* is the ``media`` index, holding one row per transcode
    derivative, or ``None`` when there is no ``media`` index distinct from the
    ``media_raw`` one. *media_raw_root* anchors a derivative row's
    ``source_path`` back onto its original; *base_directory* anchors the
    root-relative ``abs_path`` cells of both indexes.

    Dry-run unless *apply*. Migrates an index written before a column existed to
    the full current schema. Aborts, having written nothing, when an indexed
    file is missing or cannot be probed -- the two reported apart, and both
    indexes probed first so one run names the whole repair -- unless
    *skip_unreadable* leaves those rows verbatim. Reports, without failing, when
    a file's measured identity has drifted from a recorded one. When nothing
    needs changing, writes nothing at all -- so a second run over an unchanged
    media index leaves it byte-identical.
    """
    media_index = _read_widened_media_index(index_path)
    outcome = _probe_media_index(
        media_index, "media_raw index", base_directory=base_directory
    )
    patches = _patches_for(media_index, outcome)
    uuid_by_path = {
        row.resolved_path: row.probe["video_uuid"]
        for row in outcome.probed
        if row.probe["video_uuid"]
    }
    derivative = _plan_derivatives(
        derivative_index_path,
        uuid_by_path,
        label="media index",
        media_raw_root=media_raw_root,
        base_directory=base_directory,
    )
    # Both indexes are probed before this decision, so an operator with
    # unreadable media in each is told about both in one run rather than
    # discovering the second only after repairing the first.
    missing = [*outcome.missing, *derivative.missing]
    unprobeable = [*outcome.unprobeable, *derivative.unprobeable]
    if (missing or unprobeable) and not skip_unreadable:
        raise _unreadable_abort(missing, unprobeable)

    missing_columns = media_index.missing_columns
    changed = bool(patches or missing_columns or derivative.has_work)

    # Each backup sits inside the branch that rewrites its own file, so a run
    # whose only work is in the derivative index never backs up an untouched
    # originals index.
    backups: list[Path] = []
    if changed and apply:
        if patches or missing_columns:
            backups.append(_backup_index(media_index.index_path))
            _write_media_index(media_index, patches)
        if derivative.media_index is not None and derivative.has_work:
            backups.append(_backup_index(derivative.media_index.index_path))
            _write_media_index(derivative.media_index, derivative.patches)

    return ReprobeReport(
        index_path=media_index.index_path,
        applied=apply and changed,
        changed=changed,
        backups=backups,
        rows_total=len(media_index.rows),
        minted=sum(1 for row in outcome.probed if row.change == "minted"),
        unchanged=sum(1 for row in outcome.probed if row.change == "unchanged"),
        unmintable=sum(1 for row in outcome.probed if row.change == "unmintable"),
        content_digest_changed=[
            row.stored_path
            for row in outcome.probed
            if row.change == "content_digest_changed"
        ],
        video_uuid_changed=[
            row.stored_path
            for row in outcome.probed
            if row.change == "video_uuid_changed"
        ],
        rows_patched=len(patches),
        facts_rebuilt=sum(1 for row in outcome.probed if row.facts_rebuilt),
        missing=outcome.missing,
        unprobeable=outcome.unprobeable,
        schema_columns_added=missing_columns,
        # Nothing is dropped by a run that does not rewrite this file, so an
        # unchanged run reports no dropped column rather than a warning about a
        # write it is not going to perform.
        unknown_columns=media_index.unknown_columns if changed else [],
        derivative=derivative,
    )
