"""Reclaiming tracker intermediates -- item 8.4, and the milestone's second gate.

`_tracking` holds what a tracking stage produced on the way to a result: a
`.pv` and its settings, a `.slp`, a predictions CSV, an audit parquet. It is
generated and safe to delete, and nothing deletes it, so it grows without bound.
This decides what may go.

**The whole deletion authority is two frozensets.** Everything a walk classifies
is reported; only a class named in `_DELETABLE` is unlinked and only one in
`_ROW_ONLY_DROPS` loses its index row. That is `prune_media`'s shape, and it is
what makes this milestone's gate -- *the sweeper does not delete in-flight
work* -- a one-line assertion rather than a review of every branch.

**Three ways a directory can be alive while looking finished**, all present in
the producers today, and each is why a class exists rather than being folded into
another:

- The `tracks/` bridge runs *after* the in-flight claim is released, so a
  completed-but-unbridged directory reads as free. It is `complete` only once
  its completion marker is written, which the producers write after the bridge.
- Index rows are appended after the whole batch on some producers, so mid-run
  most finished directories are *unrowed*. Unrowed is therefore refused, never
  deleted -- the opposite of what "no row names it" suggests.
- A claim held on another host cannot be verified at all, only aged out. That is
  what the marker's self-carried expiry is for (item 8.3), and it is the only
  cross-host authority this consults.

**Age is the fallback, promotion is the signal.** Item 8.4 puts it that way round
because a corrected track set in `tracks_raw` makes the run it was corrected from
reclaimable immediately, whatever its age. `promoted` is that class.

**Pure.** Nothing here writes; it reads markers, mtimes and rows and returns a
verdict per directory. `Dataset.sweep_tracking` resolves roots, applies the
decline gates and performs what this decides -- the same three-layer split
`core/media/prune.py` uses, for the same reason: a decision module you can test
without a filesystem full of half-finished tracker runs.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

from mosaic.core.pipeline.markers import (
    MARKER_PREFIX,
    InflightState,
    inflight_state,
    read_inflight,
    read_phase_marker,
)
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS, RetentionClass

__all__ = [
    "REFUSED_NOTES",
    "DeclineReason",
    "SweepClass",
    "SweepEntry",
    "SweepReport",
    "classify_entry",
    "declined_sweep",
    "deletable",
    "summarize",
    "decline_text",
    "retention_days",
]

SweepClass = Literal[
    # Held by a live execution. REFUSED -- this is the gate.
    "inflight",
    # A claim past its own expiry, or whose execution the run-log proves dead.
    # Deletable: the directory is abandoned, whatever it holds.
    "expired_claim",
    # Complete, and inside its retention window. Held, and reported as held --
    # "would delete 0" must not read as "there was nothing here".
    "complete_young",
    # Complete and past its window. The ordinary reclaim.
    "complete_aged",
    # Superseded by a promoted correction (item 8.6). Age does not apply.
    "promoted",
    # No completion marker and no live claim: an attempt that died partway.
    # Deletable -- it is not a result, and re-running rebuilds it.
    "incomplete",
    # A directory this root's index does not name. REFUSED: index rows are
    # appended after the batch on some producers, so mid-run this is most of a
    # healthy run.
    "unrowed",
    # A row whose directory is gone. Row-only drop.
    "orphan_row",
    # Not shaped like a run/entry directory at all.
    "stray",
    # Carries no mosaic marker anywhere. REFUSED -- someone pointed a tracker
    # root at a directory that is not one.
    "foreign",
]

_DELETABLE: Final[frozenset[SweepClass]] = frozenset(
    {"complete_aged", "promoted", "expired_claim", "incomplete"}
)
"""The only classes whose files are removed.

``inflight`` is absent, and that absence *is* the milestone gate. Asserted
directly by the suite as well as behaviourally, because the behavioural test can
only prove the paths it exercises and this states the rule for all of them.
"""

_ROW_ONLY_DROPS: Final[frozenset[SweepClass]] = frozenset({"orphan_row"})
"""Classes where the row goes and there is no file left to unlink."""

REFUSED_NOTES: Final[Mapping[SweepClass, str]] = {
    "inflight": "held by a live execution",
    "unrowed": "not named by this root's index; run `mosaic reindex`",
    "foreign": "carries no mosaic marker -- is this root a tracker root?",
    "stray": "not shaped like a run/entry directory",
    "complete_young": "finished, inside the retention window",
}
"""Every class a run reports but does not act on, and what a reader should do.

Beside the policy sets rather than in the command, so a class that stops being
deletable cannot become one nothing prints. Each names a *different* repair --
held work will be reclaimable later, a foreign directory never will, an unrowed
one wants a reindex -- which is why they are grouped rather than summed.
"""

DeclineReason = Literal[
    "no-tracking-root",
    "root-outside-dataset",
    "legacy-layout",
]

_DECLINE_TEXT: Final[Mapping[DeclineReason, str]] = {
    "no-tracking-root": (
        "this dataset declares no _tracking root, so there is nothing to sweep. "
        "A dataset written before that root existed keeps its own layout; "
        "`mosaic reindex` and the tracker runners will not move it."
    ),
    "root-outside-dataset": (
        "a tracker root resolves outside the dataset directory. Refusing rather "
        "than deleting: item 9.1 pins roots inside the dataset precisely so a "
        "destructive pass cannot be pointed at somebody else's files."
    ),
    "legacy-layout": (
        "a tracker root still points inside tracks_raw (the pre-item-8.1 "
        "layout), where user uploads live. Deleting under it would delete "
        "source data, so this run does nothing at all."
    ),
}


def decline_text(reason: DeclineReason) -> str:
    """Operator-facing prose for a gate that stopped the run."""
    return _DECLINE_TEXT[reason]


# Retention per artifact class, from the registry rather than a branch per tool
# (item 8.4's table). A tracker's output is expensive, correctable and the thing
# a promotion supersedes; inference output is audit-only and neither reused nor
# edited, so it goes sooner.
_DEFAULT_RETENTION_DAYS: Final[Mapping[RetentionClass, float]] = {
    "tracker": 14.0,
    "inference": 3.0,
}


def retention_days(
    root_key: str, overrides: Mapping[RetentionClass, float] | None = None
) -> float:
    """How long *root_key*'s finished output is kept, in days."""
    table = {**_DEFAULT_RETENTION_DAYS, **(overrides or {})}
    return table[TRACKING_ROOTS[root_key].retention]


def deletable(verdict: SweepClass) -> bool:
    """Does this class authorize removing the directory's files?"""
    return verdict in _DELETABLE


@dataclass(frozen=True, slots=True)
class SweepEntry:
    """One entry working directory, and what the sweep decided about it."""

    root_key: str
    run_id: str
    entry: str
    path: Path
    verdict: SweepClass
    bytes_on_disk: int = 0
    held_for_age: bool = False
    detail: str = ""

    @property
    def rowed(self) -> bool:
        """Was this directory named by its root's index?"""
        return self.verdict != "unrowed"


@dataclass(frozen=True)
class SweepReport:
    """What the sweep found, and what it did about it."""

    # False when a gate declined to look at all. Reported apart from a dry run,
    # which looked and would act: "would delete 0" reads as "run it with
    # --apply", and on a dataset that must not be swept that is a lie.
    considered: bool = False
    declined: DeclineReason | None = None
    applied: bool = False
    entries: list[SweepEntry] = field(default_factory=list)
    removed: list[Path] = field(default_factory=list)
    rows_dropped: int = 0
    bytes_reclaimed: int = 0
    held_for_age: int = 0

    def of(self, verdict: SweepClass) -> list[SweepEntry]:
        """Every entry in *verdict*, in the order the walk decided them."""
        return [entry for entry in self.entries if entry.verdict == verdict]

    def counts(self) -> dict[str, int]:
        """How many entries landed in each class; empty classes omitted."""
        tally: dict[str, int] = {}
        for entry in self.entries:
            tally[entry.verdict] = tally.get(entry.verdict, 0) + 1
        return tally

    def payload(self) -> dict[str, object]:
        """The ``--json`` document: one flat object, no nested optionals."""
        refused: dict[str, list[str]] = {
            verdict: [str(e.path) for e in self.of(verdict)]
            for verdict in ("inflight", "unrowed", "foreign", "stray", "complete_young")
        }
        return {
            "considered": self.considered,
            "declined": self.declined or "",
            "applied": self.applied,
            "counts": self.counts(),
            "removed_count": len(self.removed),
            "removed": [str(path) for path in self.removed],
            "rows_dropped": self.rows_dropped,
            "bytes_reclaimed": self.bytes_reclaimed,
            "held_for_age": self.held_for_age,
            **refused,
        }


def declined_sweep(reason: DeclineReason) -> SweepReport:
    """A report for a run a gate stopped before it read anything."""
    return SweepReport(considered=False, declined=reason)


def _directory_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _has_marker(path: Path) -> bool:
    """Does anything under *path* carry mosaic's marker prefix?"""
    return any(child.name.startswith(MARKER_PREFIX) for child in path.rglob("*"))


def classify_entry(
    path: Path,
    *,
    root_key: str,
    run_id: str,
    entry: str,
    run_log_base: Path,
    execution_id: str,
    rowed: bool,
    promoted: bool,
    max_age_days: float,
    now: datetime.datetime | None = None,
) -> SweepEntry:
    """Decide one entry working directory. Reads; never writes.

    The order is load-bearing and each step forecloses the next:

    1. A live claim wins over everything. A directory can be complete *and*
       claimed -- a producer that finished one phase and is running the next --
       and deleting it because the first phase looks done is the failure this
       whole module exists to prevent.
    2. An expired or orphaned claim means abandoned, whatever else is true.
    3. No mosaic marker anywhere means this is not a tracker directory. Refused,
       so a root pointed at the wrong place reclaims nothing.
    4. Unrowed is refused *before* completeness is consulted, because on some
       producers rows arrive only after the whole batch: mid-run, most finished
       directories are unrowed and every one of them is live work.
    5. Only then does completeness decide, and promotion outranks age.
    """
    moment = now or datetime.datetime.now(datetime.timezone.utc)
    verdict, detail, held = _decide(
        path,
        root_key=root_key,
        run_log_base=run_log_base,
        execution_id=execution_id,
        rowed=rowed,
        promoted=promoted,
        max_age_days=max_age_days,
        moment=moment,
    )
    return SweepEntry(
        root_key=root_key,
        run_id=run_id,
        entry=entry,
        path=path,
        verdict=verdict,
        bytes_on_disk=_directory_bytes(path),
        held_for_age=held,
        detail=detail,
    )


def _decide(
    path: Path,
    *,
    root_key: str,
    run_log_base: Path,
    execution_id: str,
    rowed: bool,
    promoted: bool,
    max_age_days: float,
    moment: datetime.datetime,
) -> tuple[SweepClass, str, bool]:
    """The verdict alone, so ``classify_entry`` builds its record once."""
    claim: InflightState = inflight_state(
        read_inflight(path),
        run_log_base=run_log_base,
        execution_id=execution_id,
        now=moment,
    )
    if claim == "live":
        return "inflight", "held by a live execution", False
    if claim in ("expired", "orphaned"):
        return "expired_claim", f"claim is {claim}", False
    if not _has_marker(path):
        return "foreign", "carries no mosaic marker", False
    if not rowed:
        return "unrowed", "this root's index does not name it", False

    completed = _completed_at(path, root_key)
    if completed is None:
        return "incomplete", "not every declared phase completed", False
    if promoted:
        return "promoted", "superseded by a promoted correction", False

    age_days = (moment - completed).total_seconds() / 86400.0
    if age_days < max_age_days:
        return (
            "complete_young",
            f"finished {age_days:.1f}d ago, window is {max_age_days:.0f}d",
            True,
        )
    return "complete_aged", f"finished {age_days:.1f}d ago", False


def _completed_at(path: Path, root_key: str) -> datetime.datetime | None:
    """When this directory's *last* declared phase finished, else ``None``.

    **Every phase the producer declares must have a marker**, which is why the
    registry carries the list rather than this reading whichever markers happen
    to be present. A TREx directory whose conversion completed and whose
    tracking was killed holds exactly one marker; taking that as the answer would
    read it as a finished run and reclaim it at its age, throwing away a
    conversion someone is still using. Asked the other way -- "are all of
    ``('convert', 'track')`` there?" -- it is unfinished, and unfinished is not
    reclaimed on a clock.

    The *last* stamp, not the first, because that is when the directory as a
    whole stopped changing.
    """
    stamps: list[datetime.datetime] = []
    for phase in TRACKING_ROOTS[root_key].phases:
        marker = read_phase_marker(path, phase)
        if marker is None or not marker.completed_at:
            return None
        parsed = _instant(marker.completed_at)
        if parsed is None:
            return None
        stamps.append(parsed)
    return max(stamps) if stamps else None


def _instant(stamp: str) -> datetime.datetime | None:
    try:
        moment = datetime.datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=datetime.timezone.utc)


def summarize(entries: Sequence[SweepEntry], *, applied: bool) -> SweepReport:
    """Fold decided entries into a report, without performing anything."""
    return SweepReport(
        considered=True,
        applied=applied,
        entries=list(entries),
        held_for_age=sum(1 for entry in entries if entry.held_for_age),
    )
