"""Claiming one entry's working directory, and deciding what still holds.

The per-entry protocol every tracker follows: take the directory if nobody else
holds it, decide per phase whether the recorded marker still proves the work,
clear what is stale, run what is left, record what completed, and release the
claim whatever happened. Written three times, with the copies already drifting.

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
"""

from __future__ import annotations

import os
import shutil
import socket
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mosaic.core.pipeline.markers import (
    InflightMarker,
    PhaseMarker,
    PhaseName,
    clear_inflight,
    inflight_state,
    new_inflight,
    try_create_inflight,
    read_inflight,
    read_phase_marker,
    refresh_inflight,
    write_inflight,
    write_phase_marker,
)
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS
from mosaic.runlog import now_iso

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext

__all__ = [
    "INFLIGHT_REFRESH_SECONDS",
    "AdoptEvidence",
    "adopt_completed_directory",
    "claim",
    "clear_outputs",
    "open_entry",
    "phase_activity",
    "record_phase",
    "release_entry",
    "reusable_marker",
    "reusable_output",
]

INFLIGHT_REFRESH_SECONDS: Final = 15.0
"""How often the activity callback re-stamps the claim and the heartbeat.

A tool's progress bar can redraw many times a second, so the throttle keeps that
from becoming a per-line disk write, while staying well inside the run-log
heartbeat cadence and any plausible idle window.
"""


# --- the claim -------------------------------------------------------------


def claim(
    ctx: JobContext, work_dir: Path, phase: PhaseName | None, idle_seconds: float
) -> InflightMarker:
    """Write this execution's in-flight claim on *work_dir*, and return it."""
    marker = new_inflight(
        execution_id=ctx.execution_id,
        host=socket.gethostname(),
        pid=os.getpid(),
        phase=phase,
        idle_seconds=idle_seconds,
    )
    write_inflight(work_dir, marker)
    return marker


def phase_activity(
    ctx: JobContext, work_dir: Path, marker: InflightMarker, idle_seconds: float
) -> Callable[[str], None]:
    """The per-output-line liveness callback for a running phase.

    Every line the tool prints is proof the phase is alive. On a throttle it
    advances the run-log heartbeat, so the queue reaper does not read a live
    multi-hour subprocess as lost, and re-stamps the in-flight claim, so a
    concurrent execution does not read the working directory as abandoned. Both
    are best-effort: a missed refresh only shortens the claim, never aborts the
    run. Runs on the subprocess reader thread.
    """
    last_refresh = [0.0]

    def on_line(_line: str) -> None:
        now = time.monotonic()
        if now - last_refresh[0] < INFLIGHT_REFRESH_SECONDS:
            return
        last_refresh[0] = now
        ctx.heartbeat()
        try:
            _ = refresh_inflight(work_dir, marker, idle_seconds)
        except OSError:
            pass

    return on_line


def open_entry(
    ds: Dataset,
    ctx: JobContext,
    run_root: Path,
    key: str,
    *,
    kind: str,
    overwrite: bool,
    idle_seconds: float = 0.0,
) -> tuple[Path, InflightMarker] | None:
    """Create this entry's working directory and take it, or ``None`` if held.

    Takes it for real, with an exclusive create, before anything else touches the
    directory. The claim used to be *read* here and written hundreds of lines later
    inside per-phase code, so two executions could both see a free directory and
    proceed -- and the reuse-hit path wrote none at all, leaving the entry
    unprotected for its whole run.

    ``overwrite``'s tree removal happens after the claim succeeds: clearing first
    would delete a peer's work and the claim saying so. A contended entry is
    skipped, not raised, so one sequence cannot end a batch. An expired or orphaned
    claim is stealable -- otherwise a killed run locks its directory forever -- so
    it is unlinked and the create retried once.
    """
    work_dir = run_root / key
    work_dir.mkdir(parents=True, exist_ok=True)
    marker = new_inflight(
        execution_id=ctx.execution_id,
        host=socket.gethostname(),
        pid=os.getpid(),
        phase=None,
        idle_seconds=idle_seconds,
    )
    for attempt in (0, 1):
        if try_create_inflight(work_dir, marker):
            break
        state = inflight_state(
            read_inflight(work_dir),
            run_log_base=ds.base_dir,
            execution_id=ctx.execution_id,
        )
        if state == "mine":
            break
        if state in {"expired", "orphaned"} and attempt == 0:
            clear_inflight(work_dir)
            continue
        print(
            f"[{kind}] {key} is held by another execution; skipping it.",
            file=sys.stderr,
        )
        return None

    if overwrite:
        shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        _ = try_create_inflight(work_dir, marker)
    return work_dir, marker


def release_entry(work_dir: Path, execution_id: str = "") -> None:
    """Release *our* claim. Belongs in the caller's ``finally``.

    Ownership-checked: this runs whether or not this execution ever held the
    directory, so an unchecked unlink deleted a live peer's claim.
    """
    clear_inflight(work_dir, execution_id=execution_id)


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
    video_path: Path,
    video_uid: str = "",
) -> PhaseMarker | None:
    """The marker proving *phase* need not run again, or ``None``.

    The source comparison is uid-first with the path as fallback. The uid answers
    "are these the same bytes", which is what a durable cache needs, and it
    catches the case a path comparison cannot see at all: a video replaced in
    place, same path, different content. The path fallback is not decoration --
    three populations carry no uid (markers backfilled by adoption, media indexed
    before the identity columns existed, and directories written before
    ``source_uid`` did), and dropping it would remove the relocation guard from
    exactly those datasets.

    Pass ``video_uid=""`` to compare on the path alone.
    """
    marker = read_phase_marker(work_dir, phase)
    if marker is None:
        return None
    if marker.params_hash and marker.params_hash != params_hash:
        return None
    if marker.source_uid and video_uid:
        if marker.source_uid != video_uid:
            return None
    elif marker.source and not _same_video(ds, marker.source, video_path):
        return None
    return marker


def reusable_output(
    ds: Dataset,
    work_dir: Path,
    phase: PhaseName,
    *,
    params_hash: str,
    video_path: Path,
    video_uid: str = "",
) -> tuple[PhaseMarker, Path] | None:
    """The marker *and its still-present recorded output*, or ``None``.

    For a phase whose successor consumes what it produced. Where the output is
    comes from the marker rather than a glob, because a tool may leave it outside
    the working directory -- TREx can write its ``.pv`` beside the source video.
    """
    marker = reusable_marker(
        ds,
        work_dir,
        phase,
        params_hash=params_hash,
        video_path=video_path,
        video_uid=video_uid,
    )
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
    video_path: Path,
    video_uid: str,
    output: Path | None,
) -> PhaseMarker:
    """Write *phase*'s completion marker. Call only after its outputs are on disk."""
    marker = PhaseMarker(
        phase=phase,
        run_id=run_id,
        params_hash=params_hash,
        execution_id=ctx.execution_id,
        completed_at=now_iso(),
        source=ds.relative_to_root(video_path),
        source_uid=video_uid,
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
    """
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
