"""The lease that the process driving a pipeline request keeps current.

A driver (:func:`~.run.run_request`) creates ``.mosaic/drivers/<request_id>.lease.json``
before it runs a step, rewrites its ``refreshed_at`` every
:data:`DRIVER_LEASE_REFRESH_SECONDS` from a thread, and deletes it once the request
has ended. While a step's process runs, the lease names the step and the process.
The output of the driver and of its steps is appended to ``<request_id>.log``
beside the lease.

A lease is stale when its process is gone from this host, or when it was not
refreshed within :data:`DRIVER_LEASE_GRACE_SECONDS`. The second covers a driver on
another host and a host that rebooted. A request whose rollup is ``running``, whose
lease is stale and none of whose step processes may still run reads ``orphaned``
(:func:`~.status.request_verdict_of`). Running the request again resumes it.

A driver that runs its steps in its own process (:func:`~.run.run_pipeline`) takes
an ``in_process`` lease, which names its own pid as the running step's process.

Reading a lease imports pydantic and the standard library alone, as the status
reports that read it do (``tests/meta/test_status_imports.py``). Deciding whether
a step process runs on this host also imports psutil.
"""

from __future__ import annotations

import datetime
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final

from pydantic import BaseModel, ConfigDict, ValidationError

from mosaic.runlog import (
    LIVENESS_INTERVAL_SECONDS,
    LIVENESS_STALE_INTERVALS,
    AttemptLiveness,
    now_iso,
)

from ..error_kind import ClassifiedError, ErrorKind
from ..index_lock import index_lock
from ..markers import (
    create_exclusive,
    parse_instant,
    process_running,
    process_running_since,
)
from .liveness import may_still_run, step_liveness

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from mosaic.core.json_value import JsonValue

    from .rollup import RequestRollup

__all__ = [
    "DRIVER_LEASE_GRACE_SECONDS",
    "DRIVER_LEASE_REFRESH_SECONDS",
    "STALE_STEP_SECONDS",
    "DriverLease",
    "DriverRunning",
    "LeaseKeeper",
    "LeaseLost",
    "LeaseNotRefreshed",
    "StepProcess",
    "driver_alive",
    "driver_lease_path",
    "driver_log_path",
    "drivers_root",
    "lease_is_live",
    "lease_step_may_run",
    "live_step_process",
    "read_driver_lease",
    "refuse_live_driver",
    "release_driver_lease",
    "take_driver_lease",
]

DRIVER_LEASE_REFRESH_SECONDS: Final = 10.0
"""How often a driver rewrites its lease."""

DRIVER_LEASE_GRACE_SECONDS: Final = 60.0
"""How long after its last refresh a lease stays live.

Six refresh intervals. A driver that is slow to write while its host is busy
keeps its lease, and a reader on another host with a modest clock skew reads it
live.
"""

DRIVER_LEASE_UNREFRESHED_LIMIT_SECONDS: Final = DRIVER_LEASE_GRACE_SECONDS / 2
"""How long a driver may go without refreshing its lease before it stops.

Half the grace, so a driver stops before another can read its lease stale, and a
step that it starts has the other half to record its process in its run-log.
"""

STALE_STEP_SECONDS: Final = LIVENESS_STALE_INTERVALS * LIVENESS_INTERVAL_SECONDS
"""How long after its last refresh a stale lease of another host names a live step.

A step on another host that a stale lease names has not written its run-log, so
nothing else tells whether it runs. It is given as long as a run-log's silence is
(:mod:`~.liveness`).
"""


class DriverLease(BaseModel):
    """The lease of the process driving one request.

    Attributes:
        request_id: The request it drives.
        host: The host the process runs on.
        pid: The process.
        started_at: When it took the lease, ISO-8601 UTC.
        refreshed_at: When it last rewrote the lease, ISO-8601 UTC.
        step_id: The step whose process the driver is running, ``""`` between
            steps.
        step_pid: That process, 0 between steps. A step runs in a session of its
            own and outlives a driver that is killed, so a driver taking over a
            stale lease is refused while this process runs.
        in_process: Whether the driver runs its steps in its own process, as
            :func:`~.run.run_pipeline` does. ``step_pid`` is then the driver's
            own pid, and a cancel signals the running step and records the
            others itself, because no driver forwards it. A reader that predates
            the field drops it.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    request_id: str
    host: str
    pid: int
    started_at: str
    refreshed_at: str
    step_id: str = ""
    step_pid: int = 0
    in_process: bool = False

    def same_holder(self, other: DriverLease) -> bool:
        """Return whether *other* was taken by the same driver as this lease.

        One process may take a request's lease twice, one run after the other, so
        the time it was taken is compared as well as the host and the pid.
        """
        return (self.host, self.pid, self.started_at) == (
            other.host,
            other.pid,
            other.started_at,
        )


class DriverRunning(ClassifiedError):
    """Refuse to drive a request whose lease another live process holds.

    Attributes:
        lease: The live lease.
    """

    error_kind: ClassVar[ErrorKind] = "conflict"

    def __init__(self, lease: DriverLease) -> None:
        message = (
            f"request {lease.request_id!r} is driven by pid {lease.pid} on host "
            f"{lease.host!r}, which refreshed its lease at {lease.refreshed_at}. "
            f"Poll 'mosaic pipeline status --request {lease.request_id}' until it "
            f"ends, or stop that process first."
        )
        super().__init__(message)
        self.lease: DriverLease = lease

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the request, the host and the pid of the live driver."""
        return {
            "request_id": self.lease.request_id,
            "host": self.lease.host,
            "pid": self.lease.pid,
            "refreshed_at": self.lease.refreshed_at,
        }


class LeaseLost(ClassifiedError):
    """Stop a driver whose request's lease was deleted while it ran.

    Another driver took the lease over after this one went stale, finished the
    request and deleted the lease.

    Attributes:
        request_id: The request.
    """

    error_kind: ClassVar[ErrorKind] = "conflict"

    def __init__(self, request_id: str) -> None:
        message = (
            f"the lease of request {request_id!r} was deleted while this process "
            f"drove it. Read 'mosaic pipeline status --request {request_id}' "
            f"before running it again."
        )
        super().__init__(message)
        self.request_id: str = request_id

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the request."""
        return {"request_id": self.request_id}


class LeaseNotRefreshed(ClassifiedError):
    """Stop a driver that has failed to refresh its request's lease for too long.

    Attributes:
        request_id: The request.
        refreshed_at: When the driver last refreshed the lease, ISO-8601 UTC.
        error: The failure of the last refresh.
    """

    error_kind: ClassVar[ErrorKind] = "conflict"

    def __init__(self, request_id: str, refreshed_at: str, error: str) -> None:
        message = (
            f"the lease of request {request_id!r} was last refreshed at "
            f"{refreshed_at}, and refreshing it failed: {error}. Another driver "
            f"could take it over, so this one stops. Run the request again once "
            f"its lease can be written."
        )
        super().__init__(message)
        self.request_id: str = request_id
        self.refreshed_at: str = refreshed_at
        self.error: str = error

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the request, the last refresh and the failure."""
        return {
            "request_id": self.request_id,
            "refreshed_at": self.refreshed_at,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class StepProcess:
    """The process of one of a request's steps that may still be running.

    Attributes:
        step_id: The step.
        execution_id: Its attempt.
        host: The host that its run-log, or its driver's lease, names.
        pid: The process.
        liveness: How its run-log reads (:mod:`~.liveness`). ``unknown`` for a
            step on another host that only a stale lease names.
    """

    step_id: str
    execution_id: str
    host: str
    pid: int
    liveness: AttemptLiveness = "alive"


def drivers_root(base_dir: Path | str) -> Path:
    """Return ``<base_dir>/.mosaic/drivers``, the leases and logs of the drivers."""
    return Path(base_dir) / ".mosaic" / "drivers"


def driver_lease_path(base_dir: Path | str, request_id: str) -> Path:
    """Return the path of the lease of the driver of *request_id*."""
    return drivers_root(base_dir) / f"{request_id}.lease.json"


def driver_log_path(base_dir: Path | str, request_id: str) -> Path:
    """Return the log that the driver of *request_id* and its steps append to."""
    return drivers_root(base_dir) / f"{request_id}.log"


def read_driver_lease(base_dir: Path | str, request_id: str) -> DriverLease | None:
    """Return the lease of the driver of *request_id*.

    Returns:
        The lease, or ``None`` when the file is absent or does not parse.
    """
    try:
        text = driver_lease_path(base_dir, request_id).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        return DriverLease.model_validate_json(text)
    except ValidationError:
        return None


def lease_is_live(lease: DriverLease, *, now: datetime.datetime | None = None) -> bool:
    """Return whether *lease* is held by a running driver.

    Args:
        lease: The lease.
        now: The current instant. ``None`` reads the clock.

    Returns:
        ``False`` when the process is gone from this host, when ``refreshed_at``
        does not parse, and when it is older than
        :data:`DRIVER_LEASE_GRACE_SECONDS`.
    """
    if lease.host == socket.gethostname() and not process_running(lease.pid):
        return False
    refreshed = parse_instant(lease.refreshed_at)
    if refreshed is None:
        return False
    moment = now or datetime.datetime.now(datetime.UTC)
    return (moment - refreshed).total_seconds() <= DRIVER_LEASE_GRACE_SECONDS


def driver_alive(base_dir: Path | str, request_id: str) -> bool | None:
    """Return whether a running driver holds the lease of *request_id*.

    Returns:
        ``None`` when the request has no lease, as for a request whose steps are
        run by a queue, a shell loop or a driver that ended. Otherwise
        :func:`lease_is_live`.
    """
    lease = read_driver_lease(base_dir, request_id)
    return None if lease is None else lease_is_live(lease)


def live_step_process(
    rollup: RequestRollup,
    lease: DriverLease | None,
    *,
    now: datetime.datetime | None = None,
) -> StepProcess | None:
    """Return a process of one of the request's steps that may still be running.

    A step whose run-log is not terminal is judged by its run-log
    (:func:`~.liveness.may_still_run`): its pid on this host, its ``alive`` events
    on another, and a process on another host whose run-log declares no liveness
    interval counts as running. A pid reused by a process that began after the
    attempt's recorded start does not
    (:func:`~mosaic.core.pipeline.markers.process_running_since`).

    A step without a run-log is checked by the step process that *lease* names,
    because a process spawned moments before its driver was killed has not
    written its run-log yet. On this host its pid decides. On another host it
    counts as running until the lease is :data:`STALE_STEP_SECONDS` old.

    Args:
        rollup: The request's rollup.
        lease: The request's driver lease, or ``None`` without one.
        now: The current instant. ``None`` reads the clock.

    Returns:
        The first such process in the request's step order, or ``None``.
    """
    for attempt in rollup.steps:
        if attempt.terminal:
            continue
        if attempt.started:
            if may_still_run(attempt, now=now):
                return StepProcess(
                    step_id=attempt.step_id,
                    execution_id=attempt.execution_id,
                    host=attempt.host,
                    pid=attempt.pid,
                    liveness=step_liveness(attempt, now=now),
                )
        elif lease is not None and lease.step_id == attempt.step_id:
            if lease_step_may_run(lease, now=now):
                local = lease.host in ("", socket.gethostname())
                return StepProcess(
                    step_id=attempt.step_id,
                    execution_id=attempt.execution_id,
                    host=lease.host,
                    pid=lease.step_pid,
                    liveness="alive" if local else "unknown",
                )
    return None


def lease_step_may_run(
    lease: DriverLease, *, now: datetime.datetime | None = None
) -> bool:
    """Return whether the step process that *lease* names may still be running.

    On this host the pid decides, counted only when its process began by the
    lease's last refresh. On another host the process counts as running until
    the lease is :data:`STALE_STEP_SECONDS` old.
    """
    if lease.host in ("", socket.gethostname()):
        return process_running_since(lease.step_pid, lease.refreshed_at)
    refreshed = parse_instant(lease.refreshed_at)
    if refreshed is None:
        return False
    moment = now or datetime.datetime.now(datetime.UTC)
    return (moment - refreshed).total_seconds() <= STALE_STEP_SECONDS


def refuse_live_driver(base_dir: Path | str, request_id: str) -> None:
    """Raise when a running driver holds the lease of *request_id*.

    Raises:
        DriverRunning: The lease is live (:func:`lease_is_live`).
    """
    lease = read_driver_lease(base_dir, request_id)
    if lease is not None and lease_is_live(lease):
        raise DriverRunning(lease)


def take_driver_lease(
    base_dir: Path | str, request_id: str, *, in_process: bool = False
) -> DriverLease:
    """Create the lease of *request_id* for this process.

    The lease is published whole (:func:`~mosaic.core.pipeline.markers.create_exclusive`),
    so a driver never reads a lease that another is still writing. A stale lease is
    replaced. The replacement holds :func:`_takeover_lock`, and of two drivers
    replacing one stale lease on one mount, the second finds the first one's live
    lease.

    Args:
        base_dir: The dataset root.
        request_id: The request to drive.
        in_process: Whether this process runs the steps itself
            (:attr:`DriverLease.in_process`).

    Returns:
        The lease taken.

    Raises:
        DriverRunning: A live driver holds the lease.
    """
    stamp = now_iso()
    lease = DriverLease(
        request_id=request_id,
        host=socket.gethostname(),
        pid=os.getpid(),
        started_at=stamp,
        refreshed_at=stamp,
        in_process=in_process,
    )
    path = driver_lease_path(base_dir, request_id)
    text = lease.model_dump_json(indent=2)
    if create_exclusive(path, text):
        return lease
    with _takeover_lock(base_dir):
        held = read_driver_lease(base_dir, request_id)
        if held is not None and lease_is_live(held):
            raise DriverRunning(held)
        path.unlink(missing_ok=True)
        if create_exclusive(path, text):
            return lease
    held = read_driver_lease(base_dir, request_id)
    raise DriverRunning(lease if held is None else held)


def release_driver_lease(base_dir: Path | str, lease: DriverLease) -> None:
    """Delete *lease* when this process still holds it.

    A lease that another driver took over after this one went stale is kept.
    """
    with _takeover_lock(base_dir):
        held = read_driver_lease(base_dir, lease.request_id)
        if held is not None and held.same_holder(lease):
            driver_lease_path(base_dir, lease.request_id).unlink(missing_ok=True)


def _takeover_lock(base_dir: Path | str) -> AbstractContextManager[None]:
    """Return the lock under which a lease is replaced, rewritten or deleted.

    Holding it makes reading a lease and rewriting it one step, so a driver that
    went stale cannot overwrite the lease of the driver that took it over.
    """
    return index_lock(drivers_root(base_dir) / ".takeover")


class LeaseKeeper:
    """Keeps one lease current while this process holds it.

    A daemon thread rewrites ``refreshed_at`` every
    :data:`DRIVER_LEASE_REFRESH_SECONDS`, and stops once another driver has taken
    the lease over. A rewrite that raises is reported on standard error and
    retried at the next interval. :meth:`record_step` names the step whose
    process runs, and :meth:`ensure_held` raises once the lease has gone
    unrefreshed for :data:`DRIVER_LEASE_UNREFRESHED_LIMIT_SECONDS`.

    Args:
        base_dir: The dataset root.
        lease: The lease that this process has just taken or refreshed.
    """

    def __init__(self, base_dir: Path | str, lease: DriverLease) -> None:
        self._base_dir: Path = Path(base_dir)
        self._lease: DriverLease = lease
        self._writing: threading.Lock = threading.Lock()
        self._stopped: threading.Event = threading.Event()
        self._written_at: float = time.monotonic()
        self._last_error: str = ""
        self._thread: threading.Thread = threading.Thread(
            target=self._refresh_until_stopped,
            name=f"lease-{lease.request_id}",
            daemon=True,
        )

    @property
    def refreshing(self) -> bool:
        """Whether the thread still refreshes the lease."""
        return self._thread.is_alive()

    def start(self) -> None:
        """Start refreshing."""
        self._thread.start()

    def stop(self) -> None:
        """Stop refreshing, and wait for the thread to end."""
        self._stopped.set()
        if self._thread.is_alive():
            self._thread.join()

    def ensure_held(self) -> None:
        """Raise unless this process still holds the lease and keeps it current.

        Raises:
            LeaseNotRefreshed: The lease went unrefreshed for
                :data:`DRIVER_LEASE_UNREFRESHED_LIMIT_SECONDS`.
            DriverRunning: Another driver took the lease over.
            LeaseLost: The lease was deleted.
        """
        unrefreshed = time.monotonic() - self._written_at
        if unrefreshed > DRIVER_LEASE_UNREFRESHED_LIMIT_SECONDS:
            raise LeaseNotRefreshed(
                self._lease.request_id, self._lease.refreshed_at, self._last_error
            )
        held = read_driver_lease(self._base_dir, self._lease.request_id)
        if held is None:
            raise LeaseLost(self._lease.request_id)
        if not held.same_holder(self._lease):
            raise DriverRunning(held)

    def record_step(self, step_id: str, pid: int) -> None:
        """Name the step whose process runs, or clear it with ``("", 0)``.

        The process is the step's own, or this process for an in-process run.
        A write that raises is reported as a refresh's is.
        """
        _ = self._try_rewrite(step_id=step_id, step_pid=pid)

    def _refresh_until_stopped(self) -> None:
        """Rewrite the lease every interval while this process holds it."""
        while not self._stopped.wait(DRIVER_LEASE_REFRESH_SECONDS):
            if not self._try_rewrite():
                return

    def _try_rewrite(self, **update: str | int) -> bool:
        """Rewrite the lease, reporting a write that raises on standard error.

        Returns:
            ``False`` when this process no longer holds the lease, and ``True``
            otherwise, including after a write that raised.
        """
        try:
            return self._rewrite(**update)
        except Exception as error:
            self._last_error = f"{type(error).__name__}: {error}"
            print(
                f"[mosaic] {now_iso()} refreshing the lease of request "
                f"{self._lease.request_id} failed: {self._last_error}",
                file=sys.stderr,
                flush=True,
            )
            return True

    def _rewrite(self, **update: str | int) -> bool:
        """Write the lease with ``refreshed_at`` now and *update* applied.

        Returns:
            ``False``, writing nothing, when this process no longer holds the
            lease.
        """
        # Imported in the call because its module loads pandas, and reading a
        # lease leaves pandas unloaded.
        from .._utils import atomic_write

        path = driver_lease_path(self._base_dir, self._lease.request_id)
        with self._writing, _takeover_lock(self._base_dir):
            held = read_driver_lease(self._base_dir, self._lease.request_id)
            if held is None or not held.same_holder(self._lease):
                return False
            lease = self._lease.model_copy(update={**update, "refreshed_at": now_iso()})
            text = lease.model_dump_json(indent=2)
            atomic_write(path, lambda temp: temp.write_text(text, encoding="utf-8"))
            self._lease = lease
            self._written_at = time.monotonic()
        return True
