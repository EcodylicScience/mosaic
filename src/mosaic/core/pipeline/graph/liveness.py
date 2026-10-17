"""Whether the process of a step attempt may still be running, read from its run-log.

One rule, :func:`attempt_liveness`, answers for every reader: the status reports,
the driver's refusal to resume, a cancel and the agent's views of runs. It reads
what an attempt's ``started`` event recorded (its host, pid, boot, pid namespace
and liveness interval) and when the attempt last wrote a line, and it answers
``alive``, ``unknown``, ``dead`` or ``ended`` (:data:`~mosaic.runlog.AttemptLiveness`).

A pid is checked only where it means something: on the host that recorded it, in
the same pid namespace. Another host is judged by the attempt's ``alive`` events,
which a job writes after each :data:`~mosaic.runlog.LIVENESS_INTERVAL_SECONDS` of
silence, and is dead after :data:`~mosaic.runlog.LIVENESS_STALE_INTERVALS` of them.
A run-log written before mosaic recorded liveness declares no interval, and a
process on another host is then ``unknown``.

:func:`may_still_run` is the yes-or-no answer that refusals use. It keeps the
answer that mosaic gave before liveness existed for such a run-log: a process on
another host may still run, and a local attempt without a pid may not.

Importing this module imports pydantic and the standard library alone
(``tests/meta/test_status_imports.py``).
"""

from __future__ import annotations

import datetime
import socket
from typing import TYPE_CHECKING

from mosaic.runlog import (
    LIVENESS_STALE_INTERVALS,
    TERMINAL_STATUSES,
    AttemptLiveness,
    RunLogSnapshot,
)

from ..markers import (
    boots_differ,
    current_boot_id,
    current_pid_namespace,
    parse_instant,
    process_running_since,
)

if TYPE_CHECKING:
    from .rollup import StepAttempt

__all__ = [
    "attempt_liveness",
    "is_local",
    "may_still_run",
    "snapshot_liveness",
    "step_liveness",
]


def is_local(host: str, pid_ns: str) -> bool:
    """Return whether a pid recorded on *host* in *pid_ns* names a process here.

    A host left unrecorded counts as this one. A namespace counts as this one's
    unless both are recorded and they differ, as for two containers that share a
    host name.
    """
    if host not in ("", socket.gethostname()):
        return False
    here = current_pid_namespace()
    return not pid_ns or not here or pid_ns == here


def attempt_liveness(
    *,
    status: str,
    host: str,
    pid: int,
    started_at: str,
    heartbeat_at: str,
    boot_id: str = "",
    pid_ns: str = "",
    liveness_interval: float = 0.0,
    now: datetime.datetime | None = None,
) -> AttemptLiveness:
    """Return whether an attempt's process may still be running.

    The rules, in the order they decide:

    1. A terminal status is ``ended``.
    2. An attempt recorded on this host in an earlier boot is ``dead``.
    3. An attempt whose pid names a process here is ``alive`` while that process
       runs and began by the attempt's start, and ``dead`` otherwise
       (:func:`~mosaic.core.pipeline.markers.process_running_since`).
    4. An attempt that declared a liveness interval is ``dead`` after
       :data:`~mosaic.runlog.LIVENESS_STALE_INTERVALS` intervals without a line,
       and ``alive`` before.
    5. Anything else is ``unknown``: a process on another host whose run-log
       declares no interval, or a local attempt without a pid.

    Args:
        status: The run-log status.
        host: The host the process recorded.
        pid: The process, 0 when unrecorded.
        started_at: When the process started, ISO-8601.
        heartbeat_at: When the attempt last wrote a line, ISO-8601.
        boot_id: The boot the process recorded (:func:`~.markers.current_boot_id`).
        pid_ns: The pid namespace the process recorded.
        liveness_interval: The seconds between ``alive`` events it promised, 0
            for none.
        now: The current instant. ``None`` reads the clock.
    """
    if status in TERMINAL_STATUSES:
        return "ended"
    if host == socket.gethostname() and boots_differ(boot_id, current_boot_id()):
        return "dead"
    if pid > 0 and is_local(host, pid_ns):
        return "alive" if process_running_since(pid, started_at) else "dead"
    if liveness_interval > 0:
        last = parse_instant(heartbeat_at) or parse_instant(started_at)
        if last is None:
            return "unknown"
        moment = now or datetime.datetime.now(datetime.UTC)
        silent = (moment - last).total_seconds()
        return (
            "dead" if silent > LIVENESS_STALE_INTERVALS * liveness_interval else "alive"
        )
    return "unknown"


def snapshot_liveness(
    snapshot: RunLogSnapshot, *, now: datetime.datetime | None = None
) -> AttemptLiveness:
    """Return :func:`attempt_liveness` of the attempt that *snapshot* folds."""
    return attempt_liveness(
        status=snapshot["status"],
        host=snapshot["host"],
        pid=snapshot["pid"],
        started_at=snapshot["started_at"],
        heartbeat_at=snapshot["heartbeat_at"],
        boot_id=snapshot["boot_id"],
        pid_ns=snapshot["pid_ns"],
        liveness_interval=snapshot["liveness_interval"],
        now=now,
    )


def step_liveness(
    attempt: StepAttempt, *, now: datetime.datetime | None = None
) -> AttemptLiveness:
    """Return :func:`attempt_liveness` of one step's attempt in a rollup."""
    return attempt_liveness(
        status=attempt.status,
        host=attempt.host,
        pid=attempt.pid,
        started_at=attempt.started_at,
        heartbeat_at=attempt.heartbeat_at,
        boot_id=attempt.boot_id,
        pid_ns=attempt.pid_ns,
        liveness_interval=attempt.liveness_interval,
        now=now,
    )


def may_still_run(
    attempt: StepAttempt, *, now: datetime.datetime | None = None
) -> bool:
    """Return whether a step's attempt may still be running, for a refusal to act.

    ``alive`` may run. ``unknown`` may run on another host, which this process
    cannot check, and may not here, where an attempt without a pid has no process
    to wait for. ``dead`` and ``ended`` may not.
    """
    liveness = step_liveness(attempt, now=now)
    if liveness == "alive":
        return True
    return liveness == "unknown" and not is_local(attempt.host, attempt.pid_ns)
