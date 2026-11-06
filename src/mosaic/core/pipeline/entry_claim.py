"""Claim one entry's working directory or one run's root, and keep the claim alive.

A stage that writes into a per-entry directory takes it with an exclusive create
before touching anything, re-stamps the claim while its work runs, and releases the
claim in a ``finally`` whatever happened. The trackers and the ``infer-*`` ops
follow this lifecycle. It lives in ``core`` because ``core`` may not import
``tracking``, and a stage in ``core`` that works per entry needs the same claim.

A one-shot op is the whole batch, so it takes its run root through
:func:`claim_run_root` instead, which raises :class:`RunRootHeld` when another
execution holds the root rather than skipping it. That claim has no release: a
holder whose run-log went terminal reads as ``orphaned``, so a finished or dead
execution frees the root by itself. :func:`renew_run_root_claim` re-stamps it, and
:func:`empty_claimed_run_root` clears a root without removing the claim. The
refresh callbacks keep that claim alive too. A training op attaches
:func:`phase_activity` to its tool's output, or :class:`ClaimRefreshingProgress` to
an in-process trainer's callbacks.

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
    INFLIGHT_MARKER_NAME,
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
from mosaic.core.pipeline.refusal import Refusal

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.json_value import JsonValue
    from mosaic.core.pipeline.job import JobContext

__all__ = [
    "INFLIGHT_REFRESH_SECONDS",
    "ClaimRefreshingProgress",
    "RunRootHeld",
    "claim",
    "claim_run_root",
    "discard_if_empty",
    "empty_claimed_run_root",
    "open_entry",
    "phase_activity",
    "release_entry",
    "renew_run_root_claim",
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


def discard_if_empty(directory: Path) -> None:
    """Remove *directory* if nothing was ever written into it.

    A claim can only be taken inside a directory, so claiming creates one, and
    an attempt that then wrote nothing leaves it behind empty. The sweeper reads
    an empty directory as carrying no marker, classifies it ``foreign`` and
    refuses it permanently, telling the operator that a directory mosaic created
    is not one of its own. Call only after the claim is released.
    """
    try:
        if not any(directory.iterdir()):
            directory.rmdir()
    except OSError:
        pass


def _holder_detail(held: InflightMarker | None) -> dict[str, JsonValue]:
    """The holder of a refused claim, as the keys a refusal's ``error_json`` spreads.

    Prefixed, because ``error_json`` puts the detail beside ``reason``, ``step`` and
    ``message``. A claim that could not be read names no holder: its text fields
    are empty and its pid is ``None``.
    """
    if held is None:
        return {
            "holder_execution_id": "",
            "holder_host": "",
            "holder_expires_at": "",
            "holder_pid": None,
        }
    return {
        "holder_execution_id": held.execution_id,
        "holder_host": held.host,
        "holder_expires_at": held.expires_at,
        "holder_pid": held.pid,
    }


class RunRootHeld(Refusal, RuntimeError):
    """Another execution is already producing this run, so this one must not.

    Attributes:
        held: The claim that refused this one, or ``None`` when it could not be
            read.
    """

    def __init__(self, message: str, *, held: InflightMarker | None) -> None:
        super().__init__(message, reason="run_root_held", detail=_holder_detail(held))
        self.held: InflightMarker | None = held


def claim_run_root(
    ds: Dataset,
    execution_id: str,
    run_root: Path,
    kind: str,
    idle_seconds: float,
    *,
    doing: str,
) -> InflightMarker:
    """Take *run_root* exclusively for a one-shot op, or raise.

    Per-entry work skips a contended item so one sequence cannot end a batch; a
    one-shot op *is* the batch, and returning its run_id would hand back a model
    another execution is mid-write. Two executions of one identifier are not merely
    wasted: a nondeterministic trainer interleaves ``best.pt`` / ``last.pt`` /
    ``results.csv`` into one root, which is corrupt rather than slow.

    No ``finally`` release -- ``inflight_state`` reads a holder whose run-log went
    terminal as ``orphaned``, so a dead execution frees the root by itself.

    Args:
        ds: The dataset whose run-logs say whether a holder is still running.
        execution_id: This execution.
        run_root: The directory to claim. It must exist.
        kind: The op's kind, named in the refusal.
        idle_seconds: How long the claim lasts without a refresh.
        doing: What this execution would otherwise do to the root, as a gerund
            for the refusal's sentence: ``"training"``, ``"scoring"``.

    Returns:
        The claim that was taken, so a caller with output to read can keep it
        alive. A claim expires ``idle_seconds`` plus a grace after it is written,
        and a one-shot op that outruns that window has its root read as abandoned
        by the next execution along -- which then clears it and starts writing
        into the same directory. Only a caller with lines arriving from its tool
        can refresh it, through :func:`phase_activity`, and it cannot do that
        without the marker.

    Raises:
        RunRootHeld: Another execution holds *run_root*. It is a
            :class:`~mosaic.core.pipeline.refusal.Refusal`, so a verb reports it
            as one.
    """
    marker = new_inflight(
        execution_id=execution_id,
        host=socket.gethostname(),
        pid=os.getpid(),
        phase=None,
        idle_seconds=idle_seconds,
    )
    held: InflightMarker | None = None
    for attempt in (0, 1):
        if try_create_inflight(run_root, marker):
            return marker
        held = read_inflight(run_root)
        state = inflight_state(
            held, run_log_base=ds.base_dir, execution_id=execution_id
        )
        if state == "mine":
            return marker
        if state in {"expired", "orphaned"} and attempt == 0:
            clear_inflight(run_root)
            continue
        break
    where = f"{held.host}:{held.pid}" if held is not None else "another host"
    raise RunRootHeld(
        f"[{kind}] {run_root.name} is being produced by execution "
        f"{held.execution_id if held else '?'} on {where}; not {doing} it again.",
        held=held,
    )


def renew_run_root_claim(
    run_root: Path, marker: InflightMarker, kind: str, idle_seconds: float
) -> InflightMarker:
    """Re-stamp this execution's claim on *run_root* before it writes, or raise.

    An op whose work in this process gives no activity signal calls this between
    that work and its first write. When the work outlived the claim and another
    execution took the root over, that execution keeps it and this one writes
    nothing.

    Returns:
        The re-stamped claim.

    Raises:
        RunRootHeld: Another execution holds *run_root*.
    """
    renewed = refresh_inflight(run_root, marker, idle_seconds)
    if renewed is not None:
        return renewed
    held = read_inflight(run_root)
    holder = held.execution_id if held is not None else "?"
    where = f"{held.host}:{held.pid}" if held is not None else "another host"
    msg = (
        f"[{kind}] execution {holder} on {where} took {run_root.name} over while "
        "this execution worked; not writing into it."
    )
    raise RunRootHeld(msg, held=held)


def empty_claimed_run_root(run_root: Path) -> None:
    """Remove everything under *run_root* except the claim on it.

    An op calls this before writing into a root it has just claimed. Removing the
    root itself, or everything in it, would remove the claim too and leave the
    directory free for another execution mid-clear.

    Raises:
        OSError: A child could not be removed.
    """
    for child in run_root.iterdir():
        if child.name == INFLIGHT_MARKER_NAME:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
