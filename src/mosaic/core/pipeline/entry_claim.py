"""Claim one entry's working directory, and keep the claim alive.

A stage that writes into a per-entry directory takes it with an exclusive create
before touching anything, re-stamps the claim while its work runs, and releases the
claim in a ``finally`` whatever happened. The trackers and the ``infer-*`` ops
follow this lifecycle. It lives in ``core`` because ``core`` may not import
``tracking``, and a stage in ``core`` that works per entry needs the same claim.
The refresh callbacks also keep a one-shot op's run-root claim alive. A training
op attaches :func:`phase_activity` to its tool's output, or
:class:`ClaimRefreshingProgress` to an in-process trainer's callbacks.

The markers, and the rule deciding whether a claim is live, expired or orphaned,
are in :mod:`mosaic.core.pipeline.markers`. This module is the lifecycle built on
them.
"""

from __future__ import annotations

import os
import shutil
import socket
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mosaic.core.pipeline.markers import (
    InflightMarker,
    PhaseName,
    clear_inflight,
    inflight_state,
    new_inflight,
    read_inflight,
    refresh_inflight,
    try_create_inflight,
    write_inflight,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext

__all__ = [
    "INFLIGHT_REFRESH_SECONDS",
    "ClaimRefreshingProgress",
    "claim",
    "open_entry",
    "phase_activity",
    "release_entry",
    "throttled_refresh",
]

INFLIGHT_REFRESH_SECONDS: Final = 15.0
"""How often the activity callback re-stamps the claim and the heartbeat.

A tool's progress bar can redraw many times a second, so the throttle keeps that
from becoming a per-line disk write, while staying well inside the run-log
heartbeat cadence and any plausible idle window.
"""


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
    run.

    Pass it as ``run_supervised``'s *on_activity* rather than its *on_output*.
    What proves a phase alive is that the child spoke at all, and a tool whose
    progress goes to standard error -- which is where a PyTorch Lightning
    trainer's may land -- is invisible to the stdout-only callback. Taking both
    streams also makes the claim outlive exactly the window ``idle_timeout``
    measures, so the watchdog and the claim agree about when a tool is dead.

    Runs on both subprocess reader threads, hence the lock inside
    :func:`throttled_refresh`.
    """
    refresh = throttled_refresh(work_dir, marker, idle_seconds)

    def on_line(_line: str) -> None:
        if refresh():
            ctx.heartbeat()

    return on_line


def throttled_refresh(
    work_dir: Path, marker: InflightMarker, idle_seconds: float
) -> Callable[[], bool]:
    """Re-stamp *marker* on :data:`INFLIGHT_REFRESH_SECONDS`, reporting whether it did.

    The throttle is a read-then-write across whatever threads the activity
    signal arrives on -- two subprocess readers, or a trainer's own -- so the
    lock is what keeps a burst of lines from becoming a burst of disk writes.
    Best-effort: a refusal by the filesystem shortens the claim and never aborts
    the run.

    Returns:
        Whether this call was the one that acted, so a caller with a second
        thing to do on the same cadence -- the run-log heartbeat -- can hang it
        off the same decision.
    """
    last_refresh = [0.0]
    throttle = threading.Lock()

    def refresh() -> bool:
        now = time.monotonic()
        with throttle:
            if now - last_refresh[0] < INFLIGHT_REFRESH_SECONDS:
                return False
            last_refresh[0] = now
        try:
            _ = refresh_inflight(work_dir, marker, idle_seconds)
        except OSError:
            pass
        return True

    return refresh


class ClaimRefreshingProgress:
    """Keeps a one-shot op's claim alive from an in-process trainer's callbacks.

    :func:`phase_activity` is this guard for a tool mosaic runs as a subprocess:
    a line arrives, the claim is re-stamped. A trainer running *in* this process
    prints no line mosaic reads -- it calls a progress callback instead -- so
    that is where its claim refresh has to hang, and without one its run root is
    read as abandoned ``idle_seconds`` after it started.

    Every method refreshes, because every one of them is equally proof the
    trainer is alive, and none reports anything: compose it beside
    ``ctx.progress`` with
    :class:`~mosaic.core.pipeline.progress.CompositeProgressCallback` so
    reporting stays one object's job and the claim another's. The run-log needs
    no separate heartbeat here -- it advances liveness from the timestamp of
    whatever event the reporting half wrote.
    """

    def __init__(
        self, work_dir: Path, marker: InflightMarker, idle_seconds: float
    ) -> None:
        self._refresh = throttled_refresh(work_dir, marker, idle_seconds)

    def on_epoch_end(
        self, epoch: int, total_epochs: int, metrics: dict[str, float]
    ) -> None:
        _ = self._refresh()

    def on_class_start(
        self, class_idx: int, total_classes: int, class_name: str
    ) -> None:
        _ = self._refresh()

    def on_phase(self, phase: str, message: str) -> None:
        _ = self._refresh()

    def on_entry_start(self, index: int, total: int, key: str) -> None:
        _ = self._refresh()

    def on_entry_end(self, index: int, total: int, key: str) -> None:
        _ = self._refresh()


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
