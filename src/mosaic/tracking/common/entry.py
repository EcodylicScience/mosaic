"""Decide which completed phases in one entry's working directory are still valid.

The per-entry protocol that every tracker follows: claim the directory, decide per
phase whether the recorded marker still proves the work, clear what is stale, run
what is left, record what completed, and release the claim whatever happened.
Taking, refreshing and releasing the claim is
:mod:`mosaic.core.pipeline.entry_claim`. The reuse decision, the clearing, the
recording and the adoption of a pre-marker directory are here, once. They were
written three times, with the copies already drifting.

**A marker answers "did this phase complete", never "where is its artifact".**
The two are separate calls here because a phase that completed and produced
nothing -- a video with no detected individuals -- is genuinely reusable, and
folding an existence check into the marker test would re-run it forever. A phase
whose *successor consumes its output* does need both, and says so by calling
:func:`reusable_output`.

**Unknown is not mismatched.** An empty ``source``, ``source_uid`` or
``params_hash`` on a marker means the marker cannot say, which is not grounds for
a recompute: a marker adopted from a directory that predates markers cannot know
any of them, and treating silence as disagreement would re-run every such entry
once, forever.

An entry of several clips is the exception. A marker's ``source`` is the first
clip's path, which says nothing of the clips after it. For such an entry only a
``source_uid`` equal to the entry's proves the output, and a directory that
predates markers is never adopted.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.core.pipeline.markers import (
    PhaseMarker,
    PhaseName,
    read_phase_marker,
    write_phase_marker,
)
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS
from mosaic.runlog import now_iso

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext
    from mosaic.tracking.common.scope import TrackerWorkItem

__all__ = [
    "AdoptEvidence",
    "adopt_completed_directory",
    "clear_outputs",
    "record_phase",
    "reusable_marker",
    "reusable_output",
]


# --- what still holds ------------------------------------------------------


def _same_video(ds: Dataset, stored: str, video_path: Path) -> bool:
    """Is *stored* the path *video_path* now resolves to? Both sides are resolved.

    A stored value may be root-relative (the portable form) or a legacy absolute,
    and the dataset may have moved. A raw string comparison would call every
    relocated dataset a source change.
    """
    return ds.resolve_path(stored).resolve() == video_path.resolve()


def reusable_marker(
    ds: Dataset,
    work_dir: Path,
    phase: PhaseName,
    *,
    params_hash: str,
    item: TrackerWorkItem,
) -> PhaseMarker | None:
    """Return the marker proving *phase* need not run again for *item*, or ``None``.

    The source comparison is uid-first with the path as fallback. The uid is the
    item's ``source_uid``, the identity of the whole input: one clip's uuid, or
    the ordered composition of several. It answers "are these the same bytes",
    which is what a durable cache needs, and it catches the case a path
    comparison cannot see at all: a video replaced in place, same path, different
    content. Three populations record no uid (markers backfilled by adoption,
    media indexed before the identity columns existed, and directories written
    before ``source_uid`` did), and the path fallback is their only relocation
    guard.

    The path is the first clip's, so it proves nothing for an item of several
    clips. Such an item reuses a marker only when both uids are recorded and
    equal.
    """
    marker = read_phase_marker(work_dir, phase)
    if marker is None:
        return None
    if marker.params_hash and marker.params_hash != params_hash:
        return None
    if marker.source_uid and item.source_uid:
        if marker.source_uid != item.source_uid:
            return None
    elif item.n_sources > 1:
        return None
    elif marker.source and not _same_video(ds, marker.source, item.video_path):
        return None
    return marker


def reusable_output(
    ds: Dataset,
    work_dir: Path,
    phase: PhaseName,
    *,
    params_hash: str,
    item: TrackerWorkItem,
) -> tuple[PhaseMarker, Path] | None:
    """The marker *and its still-present recorded output*, or ``None``.

    For a phase whose successor consumes what it produced. Where the output is
    comes from the marker rather than a glob, because a tool may leave it outside
    the working directory -- TREx can write its ``.pv`` beside the source video.
    """
    marker = reusable_marker(ds, work_dir, phase, params_hash=params_hash, item=item)
    if marker is None or not marker.recorded_output:
        return None
    output = ds.resolve_path(marker.recorded_output)
    if not output.exists():
        return None
    return marker, output


def record_phase(
    ds: Dataset,
    work_dir: Path,
    phase: PhaseName,
    *,
    ctx: JobContext,
    run_id: str,
    params_hash: str,
    item: TrackerWorkItem,
    output: Path | None,
) -> PhaseMarker:
    """Write *phase*'s completion marker for *item*, once its outputs are on disk.

    The marker records what :func:`reusable_marker` compares: the item's
    ``source_uid`` and its first clip's path.
    """
    marker = PhaseMarker(
        phase=phase,
        run_id=run_id,
        params_hash=params_hash,
        execution_id=ctx.execution_id,
        completed_at=now_iso(),
        source=ds.relative_to_root(item.video_path),
        source_uid=item.source_uid,
        recorded_output=ds.relative_to_root(output) if output is not None else "",
    )
    write_phase_marker(work_dir, marker)
    return marker


def clear_outputs(work_dir: Path, kind: str, phase: PhaseName) -> None:
    """Delete what *phase* owns, before re-running it.

    Reads the globs declared for this root in
    :data:`~mosaic.core.pipeline.tracking_roots.TRACKING_ROOTS`. A glob that
    matches a directory removes it as a tree, so a tool whose phase output is a
    session directory is expressible without a special case here.
    """
    root = TRACKING_ROOTS.get(kind)
    if root is None:
        return
    for pattern in root.clear_globs(phase):
        for path in sorted(work_dir.glob(pattern)):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)


# --- adopting a directory that predates markers ----------------------------


@dataclass(frozen=True, slots=True)
class AdoptEvidence:
    """One phase to backfill, and the glob naming the output its marker records."""

    phase: PhaseName
    output_glob: str


def adopt_completed_directory(
    ds: Dataset,
    work_dir: Path,
    run_id: str,
    *,
    item: TrackerWorkItem,
    required: Sequence[str],
    record: Sequence[AdoptEvidence],
) -> None:
    """Mark a pre-marker directory complete when it demonstrably holds a finished run.

    Without this, every sequence tracked before markers existed re-runs once --
    hours of tracking apiece on a real dataset.

    *required* are globs that must **all** match. They have to include whatever
    the tool writes *last*, because the outputs that appear as processing
    proceeds cannot distinguish a finished run from one killed partway. A tracker
    whose single output cannot make that distinction should not adopt at all, and
    says so by not calling this.

    The backfilled markers record no source and no parameter hash. Nothing on
    disk says what the directory was computed from, and an honest unknown beats a
    confident guess that would then serve as a cache key. The consequence is
    inherent rather than a gap: an adopted directory is not protected by the
    source-video guard, because there is nothing to compare against.

    An *item* of several clips is never adopted. The directory cannot say how many
    clips it covered, and its shape is a single clip's, so it may hold one clip's
    output, which a marker would then record as done for the whole entry.
    """
    if item.n_sources > 1:
        return
    if any(read_phase_marker(work_dir, ev.phase) is not None for ev in record):
        return
    if not all(sorted(work_dir.glob(pattern)) for pattern in required):
        return

    stamp = now_iso()
    for evidence in record:
        matches = sorted(work_dir.glob(evidence.output_glob))
        write_phase_marker(
            work_dir,
            PhaseMarker(
                phase=evidence.phase,
                run_id=run_id,
                completed_at=stamp,
                recorded_output=ds.relative_to_root(matches[0]) if matches else "",
                backfilled=True,
            ),
        )
