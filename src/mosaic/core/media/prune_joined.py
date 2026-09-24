"""Reclaim the joined exports no tracker will read again.

A joined export (:mod:`mosaic.core.pipeline.joined_export`) has no index row and
no forward link. It is found by the clip set it holds, and a tracker reads it
only under a *current* recipe. So when :attr:`JoinedExportOp.version` moves,
every join already on disk becomes superseded at once. Re-running the op writes
the current one beside it and nothing removes the old file: ``prune-media``
walks the transcode kind directory and ``sweep-tracking`` the tracker roots, and
neither reaches ``media/joined/``. This module does.

**The liveness rule is the tracker's own.** A superseded join of a clip set some
entry still resolves to is read by nothing, because
:func:`~mosaic.core.pipeline.joined_export.joins_of` hands the consumer current
joins only. Deleting one costs nothing a tracker uses, which is why
``superseded`` is the only class deleted by default.

**Two things it refuses to touch.** A join of a clip set no entry resolves to
(``unsourced``) is kept, because the name is a digest and cannot say whether the
clips were rearranged or deleted, and in the second case the join may be the
last copy of the session's pixels. Two current joins of one clip set
(``competing``) are kept too: both are valid, and which one a tracker should
read is a person's decision.

**Pure where it decides.** :func:`reconcile_joined` reads names, sizes and
mtimes, and writes nothing. :func:`prune_joined` performs what it decided, and
``Dataset.prune_joined_exports`` resolves the roots and the clip sets. That is
the split ``core/media/prune.py`` uses.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final, Literal

from mosaic.core.media.prune import modified_after
from mosaic.core.pipeline.joined_export import parse_joined_name

__all__ = [
    "JoinedDeclineReason",
    "JoinedPruneClass",
    "JoinedPruneEntry",
    "JoinedPruneReport",
    "declined_joined_report",
    "deletable",
    "joined_decline_text",
    "prune_joined",
    "reconcile_joined",
]

JoinedPruneClass = Literal[
    # A current join, and the only current join of its clip set. What a
    # tracker reads. Untouched.
    "live",
    # A current join of a clip set that has another. Untouched and listed: the
    # tracker refuses the entry until a person deletes one.
    "competing",
    # A join named under a recipe the op no longer writes, of a clip set some
    # entry still resolves to. Read by nothing. The class this module exists for.
    "superseded",
    # A join of a clip set no entry resolves to. Refused: it may be the last
    # copy of a session whose clips are gone.
    "unsourced",
    # Not a join: a partial kept after a failed check, a concat listing, a
    # normalised clip from an interrupted run, a subdirectory, a symlink.
    "stray",
]

_DELETABLE: Final[frozenset[JoinedPruneClass]] = frozenset({"superseded"})
"""The only class deleted by default. Strays are deleted only on request."""

JoinedDeclineReason = Literal["no-media-root", "nested-root"]

_DECLINE_TEXT: Final[Mapping[JoinedDeclineReason, str]] = {
    "no-media-root": (
        "this dataset has no media root, so there is no joined export to prune"
    ),
    "nested-root": (
        "another root resolves inside the joined kind directory, so pruning it "
        "could delete files that are not joined exports"
    ),
}


def deletable(verdict: JoinedPruneClass) -> bool:
    """Does *verdict* authorize deleting the file without ``--include-stray``?"""
    return verdict in _DELETABLE


def joined_decline_text(reason: JoinedDeclineReason) -> str:
    """The operator-facing sentence for a declined run."""
    return _DECLINE_TEXT[reason]


@dataclass(frozen=True)
class JoinedPruneEntry:
    """One path under the joined kind directory, and what the run decided."""

    path: Path
    verdict: JoinedPruneClass
    # The entry whose clips this join holds, as "group/sequence". Empty when no
    # entry resolves to its clip set, and for a stray.
    entry: str = ""
    source_uid: str = ""
    recipe_hash: str = ""
    size_bytes: int = 0
    # Set when a deletion was overruled by the age window, so the report can
    # say "would have, but it is too new" rather than silently keeping it.
    held_for_age: bool = False


@dataclass(frozen=True)
class JoinedPruneReport:
    """What the run found, and what it did about it."""

    # False when a gate declined to look at all. Reported apart from a dry run,
    # which looked and would act.
    considered: bool = False
    declined: JoinedDeclineReason | None = None
    applied: bool = False
    entries: list[JoinedPruneEntry] = field(default_factory=list)
    files_deleted: list[Path] = field(default_factory=list)
    bytes_reclaimed: int = 0
    held_for_age: int = 0
    # The recipes the run treated as current. Printed because a shell running a
    # different mosaic version from the worker's would call fresh joins
    # superseded, and this is what makes that visible.
    current_recipes: list[str] = field(default_factory=list)
    # Entries whose media could not be resolved, with the reason. Their joins
    # read as unsourced and are kept, so a failure here only prevents deletion.
    unresolved: list[str] = field(default_factory=list)

    def of(self, verdict: JoinedPruneClass) -> list[JoinedPruneEntry]:
        """Every entry the run put in *verdict*, in the order it decided them."""
        return [entry for entry in self.entries if entry.verdict == verdict]

    def counts(self) -> dict[str, int]:
        """How many entries landed in each class, classes with none omitted."""
        return dict(Counter(entry.verdict for entry in self.entries))

    def payload(self) -> dict[str, object]:
        """The ``--json`` document: one flat object, no nested optionals."""
        return {
            "considered": self.considered,
            "declined": self.declined or "",
            "applied": self.applied,
            "counts": self.counts(),
            "files_deleted_count": len(self.files_deleted),
            "files_deleted": [str(path) for path in self.files_deleted],
            "bytes_reclaimed": self.bytes_reclaimed,
            "held_for_age": self.held_for_age,
            "current_recipes": self.current_recipes,
            "unresolved": self.unresolved,
            "competing": [str(e.path) for e in self.of("competing")],
            "unsourced": [str(e.path) for e in self.of("unsourced")],
            "stray": [str(e.path) for e in self.of("stray")],
        }


def declined_joined_report(reason: JoinedDeclineReason) -> JoinedPruneReport:
    """A report for a run that a gate stopped before it read anything."""
    return JoinedPruneReport(considered=False, declined=reason)


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def reconcile_joined(
    joined_root: Path,
    *,
    clip_sets: Mapping[str, str],
    current_recipes: Set[str],
    cutoff: datetime,
) -> list[JoinedPruneEntry]:
    """Decide every path directly under *joined_root*.

    *clip_sets* maps the source uid of every clip set some entry resolves to
    onto that entry's label. *current_recipes* is every recipe the op writes
    today. Walked in sorted order, so two runs over one dataset report alike.
    """
    if not joined_root.is_dir():
        return []
    children = sorted(joined_root.iterdir())

    def as_join(path: Path) -> tuple[str, str] | None:
        # A symlink is never a join here, even one pointing at a video: what it
        # points at is not this directory's to judge.
        if path.is_symlink() or not path.is_file():
            return None
        return parse_joined_name(path.name)

    parsed = {path: as_join(path) for path in children}
    current_per_clip_set = Counter(
        name[0]
        for name in parsed.values()
        if name is not None and name[1] in current_recipes
    )

    entries: list[JoinedPruneEntry] = []
    for path in children:
        name = parsed[path]
        if name is None:
            stray_size = _size(path) if path.is_file() and not path.is_symlink() else 0
            entries.append(
                JoinedPruneEntry(path=path, verdict="stray", size_bytes=stray_size)
            )
            continue
        source_uid, recipe_hash = name
        size = _size(path)
        entry = clip_sets.get(source_uid, "")
        verdict: JoinedPruneClass
        if not entry:
            verdict = "unsourced"
        elif recipe_hash in current_recipes:
            verdict = "competing" if current_per_clip_set[source_uid] > 1 else "live"
        else:
            verdict = "superseded"
        entries.append(
            JoinedPruneEntry(
                path=path,
                verdict=verdict,
                entry=entry,
                source_uid=source_uid,
                recipe_hash=recipe_hash,
                size_bytes=size,
                held_for_age=verdict == "superseded" and modified_after(path, cutoff),
            )
        )
    return entries


def prune_joined(
    joined_root: Path,
    *,
    clip_sets: Mapping[str, str],
    current_recipes: Set[str],
    apply: bool,
    min_age_hours: float = 24.0,
    include_stray: bool = False,
    unresolved: list[str] | None = None,
    now: datetime | None = None,
) -> JoinedPruneReport:
    """Delete the superseded joins under *joined_root*, and optionally the strays.

    Dry-run unless *apply*. *min_age_hours* holds back anything modified inside
    the window: a join an old worker published a moment ago, or a partial the
    op is still writing, looks by name exactly like one left behind. *now* is
    injectable so the window can be tested without sleeping.

    No index is read or written. A join has no row, so a deletion here is the
    whole of the change and a crash part-way leaves a state the next run decides
    the same way.
    """
    moment = now or datetime.now(timezone.utc)
    cutoff = moment - timedelta(hours=min_age_hours)
    entries = reconcile_joined(
        joined_root,
        clip_sets=clip_sets,
        current_recipes=current_recipes,
        cutoff=cutoff,
    )
    delete = [
        entry
        for entry in entries
        if deletable(entry.verdict) and not entry.held_for_age
    ]
    if include_stray:
        # Files only, and only past the window: an in-flight partial is exactly
        # the shape this sweeps. A subdirectory or a symlink is never deleted.
        delete += [
            entry
            for entry in entries
            if entry.verdict == "stray"
            and entry.path.is_file()
            and not entry.path.is_symlink()
            and not modified_after(entry.path, cutoff)
        ]
    if apply:
        for entry in delete:
            entry.path.unlink(missing_ok=True)
    return JoinedPruneReport(
        considered=True,
        applied=apply and bool(delete),
        entries=entries,
        files_deleted=[entry.path for entry in delete],
        bytes_reclaimed=sum(entry.size_bytes for entry in delete),
        held_for_age=sum(1 for entry in entries if entry.held_for_age),
        current_recipes=sorted(current_recipes),
        unresolved=list(unresolved or []),
    )
