"""Cancelling a running attempt or a pipeline request from another process.

A cancel sends SIGTERM to the process that a run-log or a driver lease records.
``mosaic run`` and ``mosaic pipeline run`` turn SIGTERM into a cooperative cancel
(``install_signal_handler``). An attempt stops at its next checkpoint and records
itself ``cancelled``. A driver forwards SIGTERM to its running step's process
group and records every step without a terminal record ``cancelled``.

A run-log has one writer. A cancel therefore records an attempt itself only when
the attempt's process has exited, or when its step has not started and a driver
will not start it. A process on another host is not signalled, and a request with
one is left as it is.

Importing this module leaves pandas, cv2, av and the feature library unloaded
(``tests/meta/test_status_imports.py``).
"""

from __future__ import annotations

import os
import signal
import socket
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from mosaic.runlog import (
    TERMINAL_STATUSES,
    read_run,
    record_exited_attempt,
    run_log_dir,
)

from .error_kind import ClassifiedError, ErrorKind
from .graph.driver_lease import lease_is_live, read_driver_lease
from .graph.rollup import RequestStatus, request_rollup
from .graph.store import load_request
from .markers import process_running_since

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.json_value import JsonValue

    from .graph.driver_lease import DriverLease
    from .graph.model import Request
    from .graph.rollup import StepAttempt

__all__ = [
    "AttemptCancel",
    "AttemptNotFound",
    "CancelOutcome",
    "CannotSignal",
    "RequestCancel",
    "RequestNotFound",
    "cancel_attempt",
    "cancel_request",
    "read_request",
]

type CancelOutcome = Literal["signalled", "recorded", "ended", "other_host"]
"""What a cancel did.

``signalled``
    SIGTERM was delivered. The process records how it ended.
``recorded``
    The attempt's process had exited, or its step had not started. The cancel
    recorded the attempt ``cancelled``.
``ended``
    The attempt or request was already over. The cancel changed nothing.
``other_host``
    A process runs on another host, which a signal sent from this host does not
    arrive at. The cancel changed nothing.
"""


class AttemptCancel(BaseModel):
    """What a cancel did to one attempt.

    Attributes:
        execution_id: The attempt.
        outcome: What the cancel did.
        status: The attempt's status after the cancel. ``cancelling`` once
            signalled, ``cancelled`` once recorded, and the run-log's status
            otherwise.
        signalled: Whether SIGTERM was delivered.
        pid: The process that the run-log or the driver lease records, 0 when
            neither records one.
        host: The host of that process, ``""`` when unrecorded.
        message: One sentence saying what was done.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    execution_id: str
    outcome: CancelOutcome
    status: str
    signalled: bool
    pid: int
    host: str
    message: str


class RequestCancel(BaseModel):
    """What a cancel did to one pipeline request.

    Attributes:
        request_id: The request.
        outcome: What the cancel did. ``signalled`` when it signalled the driver
            or a step's process, and ``recorded`` when every process had exited
            and it recorded the unended steps ``cancelled``.
        status: The request's rollup status after the cancel. It reads
            ``running`` when only the driver was signalled, and ``cancelled``
            once any step is recorded ``cancelled``, even while a signalled step
            is still stopping.
        driver_pid: The driver that received SIGTERM, 0 when none did.
        steps: The attempts that the cancel signalled or recorded itself, in the
            request's step order. Empty when the driver was signalled, because
            the driver forwards SIGTERM to its running step and records the
            others.
        message: One sentence saying what was done.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    outcome: CancelOutcome
    status: RequestStatus
    driver_pid: int
    steps: tuple[AttemptCancel, ...]
    message: str


class AttemptNotFound(ClassifiedError, LookupError):
    """An execution id without a run-log.

    Attributes:
        execution_id: The attempt looked up.
    """

    error_kind: ClassVar[ErrorKind] = "not_found"

    def __init__(self, execution_id: str) -> None:
        super().__init__(f"No run found with execution_id={execution_id}.")
        self.execution_id: str = execution_id

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the attempt looked up."""
        return {"execution_id": self.execution_id}


class RequestNotFound(ClassifiedError, LookupError):
    """A request id without a request file.

    Attributes:
        request_id: The request looked up.
    """

    error_kind: ClassVar[ErrorKind] = "not_found"

    def __init__(self, request_id: str, message: str) -> None:
        super().__init__(message)
        self.request_id: str = request_id

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the request looked up."""
        return {"request_id": self.request_id}


class CannotSignal(ClassifiedError):
    """A running process that this process may not signal, or a run-log without one.

    Attributes:
        execution_id: The attempt, ``""`` for a driver.
        pid: The process, 0 when the run-log records none.
    """

    error_kind: ClassVar[ErrorKind] = "conflict"

    def __init__(self, message: str, *, execution_id: str, pid: int) -> None:
        super().__init__(message)
        self.execution_id: str = execution_id
        self.pid: int = pid

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the attempt and the process."""
        return {"execution_id": self.execution_id, "pid": self.pid}


def cancel_attempt(run_dir: Path | str, execution_id: str) -> AttemptCancel:
    """Cancel one running attempt by sending SIGTERM to the pid its run-log records.

    An attempt whose process has exited is recorded ``cancelled``
    (:func:`~mosaic.runlog.record_exited_attempt`). So is one whose pid names a
    process that began after the attempt's ``started`` record, as a pid reused
    after a reboot does, and that process is not signalled. An attempt whose
    process received the signal keeps its run-log as it is, because that process
    records how it ended.

    Args:
        run_dir: The ``.mosaic/runs`` directory.
        execution_id: The attempt.

    Returns:
        What the cancel did. An attempt on another host is left as it is.

    Raises:
        AttemptNotFound: The attempt's run-log is absent.
        CannotSignal: The run-log records a pid of 0, or this process may not
            signal the pid.
    """
    snapshot = read_run(run_dir, execution_id)
    if snapshot is None:
        raise AttemptNotFound(execution_id)
    status, host, pid = snapshot["status"], snapshot["host"], snapshot["pid"]
    if status in TERMINAL_STATUSES:
        return _attempt_result(
            execution_id,
            "ended",
            status,
            pid=pid,
            host=host,
            message=f"run {execution_id} already {status}; nothing to cancel.",
        )
    if host and host != socket.gethostname():
        return _attempt_result(
            execution_id,
            "other_host",
            status,
            pid=pid,
            host=host,
            message=_other_host_message("Run", host),
        )
    if pid <= 0:
        raise CannotSignal(
            "No pid recorded for this run; cannot signal it.",
            execution_id=execution_id,
            pid=pid,
        )
    running = process_running_since(pid, snapshot["started_at"])
    if running and _send_sigterm(pid, execution_id=execution_id):
        return _attempt_result(
            execution_id,
            "signalled",
            "cancelling",
            pid=pid,
            host=host,
            message=f"sent SIGTERM to pid {pid} for run {execution_id}.",
        )
    record_exited_attempt(run_dir, execution_id, "cancelled")
    return _attempt_result(
        execution_id,
        "recorded",
        "cancelled",
        pid=pid,
        host=host,
        message=(
            f"pid {pid} for run {execution_id} had already exited; recorded "
            "the run as cancelled."
        ),
    )


def cancel_request(ds: Dataset, request_id: str) -> RequestCancel:
    """Cancel a pipeline request that has not ended.

    A request whose driver runs on this host receives SIGTERM through the pid of
    its lease (:mod:`~mosaic.core.pipeline.graph.driver_lease`), and the driver
    stops its running step and records the rest. Without a running driver, each
    started step is cancelled by :func:`cancel_attempt`, a step process that a
    stale lease names receives SIGTERM, and every other step that has not started
    is recorded ``cancelled``. The rollup then reads ``cancelled``.

    A request run in one process (:func:`~mosaic.core.pipeline.graph.run.run_pipeline`)
    holds an ``in_process`` lease, and no driver process forwards a cancel. Its
    running step is cancelled by :func:`cancel_attempt`, which signals that one
    process, and its steps that have not started are recorded ``cancelled``, which
    the run reads before it starts each step. A notebook kernel that runs a
    request receives SIGTERM like any other process.

    A request is left as it is when its live driver runs on another host, or when
    one of its running step processes does.

    The request is one that a local driver or a person runs step by step. A queue
    that dispatches its steps holds no lease, and a step that the queue starts
    after the cancel appends its own ending after the recorded ``cancelled``.

    Args:
        ds: The dataset the request was submitted against.
        request_id: The request.

    Returns:
        What the cancel did.

    Raises:
        RequestNotFound: The request file is absent.
        RequestUnreadable: The request file is not valid JSON or not a valid
            request.
        CannotSignal: A running step's run-log records a pid of 0, or this
            process may not signal the driver or a step. A step refused before
            any is signalled leaves the request as it was.
    """
    request = read_request(ds.base_dir, request_id)
    rollup = request_rollup(ds.base_dir, request)
    if rollup.is_terminal:
        return RequestCancel(
            request_id=request_id,
            outcome="ended",
            status=rollup.status,
            driver_pid=0,
            steps=(),
            message=f"request {request_id} already {rollup.status}; nothing to cancel.",
        )
    here = socket.gethostname()
    lease = read_driver_lease(ds.base_dir, request_id)
    elsewhere = _other_host(lease, rollup.steps, here)
    if elsewhere:
        return RequestCancel(
            request_id=request_id,
            outcome="other_host",
            status=rollup.status,
            driver_pid=0,
            steps=(),
            message=elsewhere,
        )
    if (
        lease is not None
        and lease.host == here
        and not lease.in_process
        and process_running_since(lease.pid, lease.started_at)
        and _send_sigterm(lease.pid, execution_id="")
    ):
        return RequestCancel(
            request_id=request_id,
            outcome="signalled",
            status=rollup.status,
            driver_pid=lease.pid,
            steps=(),
            message=(
                f"sent SIGTERM to pid {lease.pid}, the driver of request {request_id}."
            ),
        )
    for attempt in rollup.steps:
        if attempt.started and not attempt.terminal:
            _check_signallable(attempt)
    logs = run_log_dir(ds.base_dir)
    steps = tuple(
        cancel_attempt(logs, attempt.execution_id)
        if attempt.started
        else _cancel_unstarted(ds, request, attempt, lease)
        for attempt in rollup.steps
        if not attempt.terminal
    )
    signalled = any(step.signalled for step in steps)
    after = request_rollup(ds.base_dir, request).status
    driver = (
        f"request {request_id} runs in process {lease.pid}; "
        if lease is not None and lease.in_process and lease_is_live(lease)
        else f"the driver of request {request_id} is not running; "
    )
    return RequestCancel(
        request_id=request_id,
        outcome="signalled" if signalled else "recorded",
        status=after,
        driver_pid=0,
        steps=steps,
        message=driver
        + (
            "sent SIGTERM to its running steps."
            if signalled
            else f"recorded {len(steps)} unended steps as cancelled."
        ),
    )


def read_request(base_dir: Path, request_id: str) -> Request:
    """Read the request *request_id* of the dataset at *base_dir*.

    Raises:
        RequestNotFound: The dataset holds no such request.
        RequestUnreadable: Its file is not valid JSON or not a valid request.
    """
    try:
        return load_request(base_dir, request_id)
    except FileNotFoundError as missing:
        raise RequestNotFound(request_id, str(missing)) from missing


def _other_host(
    lease: DriverLease | None, steps: tuple[StepAttempt, ...], here: str
) -> str:
    """Return why a process of the request is on another host, ``""`` when none is.

    A live lease of another host names a running driver there, and its steps run
    there too. A stale one names a step process there that may still run, as
    :func:`~mosaic.core.pipeline.graph.driver_lease.live_step_process` counts it.
    """
    if lease is not None and lease.host != here:
        if lease_is_live(lease):
            return _other_host_message(
                f"Request {lease.request_id} is driven by pid {lease.pid}, which",
                lease.host,
            )
        if lease.step_pid > 0 and any(
            attempt.step_id == lease.step_id and not attempt.started
            for attempt in steps
        ):
            return _other_host_message(
                f"Step {lease.step_id!r} runs as pid {lease.step_pid}, which",
                lease.host,
            )
    for attempt in steps:
        if attempt.started and not attempt.terminal and attempt.host not in ("", here):
            return _other_host_message(
                f"Step {attempt.step_id!r} runs as pid {attempt.pid}, which",
                attempt.host,
            )
    return ""


def _check_signallable(attempt: StepAttempt) -> None:
    """Raise :class:`CannotSignal` when a started step's process cannot be signalled.

    Every started step of a request without a running driver is checked before
    any is signalled or recorded, so a refusal leaves the request as it was. A
    process that has exited, or that began after the step started, is not
    signalled and passes.
    """
    if attempt.pid <= 0:
        raise CannotSignal(
            f"No pid recorded for step {attempt.step_id!r}; cannot signal it.",
            execution_id=attempt.execution_id,
            pid=attempt.pid,
        )
    if not process_running_since(attempt.pid, attempt.started_at):
        return
    try:
        os.kill(attempt.pid, 0)
    except ProcessLookupError:
        return
    except PermissionError as denied:
        raise CannotSignal(
            f"Not permitted to signal process {attempt.pid}.",
            execution_id=attempt.execution_id,
            pid=attempt.pid,
        ) from denied


def _other_host_message(subject: str, host: str) -> str:
    """Return the sentence that refuses to signal a process on *host*."""
    return (
        f"{subject} is on host {host!r}; a cancel signals processes on this host only."
    )


def _cancel_unstarted(
    ds: Dataset, request: Request, attempt: StepAttempt, lease: DriverLease | None
) -> AttemptCancel:
    """Cancel a step whose run-log is absent, when the request's driver has exited.

    A step process that the stale lease names may not have written its run-log
    yet, and receives SIGTERM. Any other step is recorded ``cancelled``.
    """
    if (
        lease is not None
        and lease.step_id == attempt.step_id
        and process_running_since(lease.step_pid, lease.refreshed_at)
        and _send_sigterm(lease.step_pid, execution_id=attempt.execution_id)
    ):
        return _attempt_result(
            attempt.execution_id,
            "signalled",
            "cancelling",
            pid=lease.step_pid,
            host=lease.host,
            message=(
                f"sent SIGTERM to pid {lease.step_pid} for run {attempt.execution_id}."
            ),
        )
    # Imported in the call because its module loads pandas, and a cancel that
    # signals a process leaves pandas unloaded.
    from .graph.step import record_step_ended

    record_step_ended(
        ds,
        attempt.execution_id,
        step_id=attempt.step_id,
        owner=request.owner,
        status="cancelled",
    )
    return _attempt_result(
        attempt.execution_id,
        "recorded",
        "cancelled",
        pid=0,
        host="",
        message=(
            f"step {attempt.step_id} had not started; recorded run "
            f"{attempt.execution_id} as cancelled."
        ),
    )


def _send_sigterm(pid: int, *, execution_id: str) -> bool:
    """Send SIGTERM to *pid*, returning ``False`` when the process has exited.

    Raises:
        CannotSignal: This process may not signal *pid*.
    """
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    except PermissionError as denied:
        raise CannotSignal(
            f"Not permitted to signal process {pid}.",
            execution_id=execution_id,
            pid=pid,
        ) from denied
    return True


def _attempt_result(
    execution_id: str,
    outcome: CancelOutcome,
    status: str,
    *,
    pid: int,
    host: str,
    message: str,
) -> AttemptCancel:
    """Return the :class:`AttemptCancel` of *outcome*."""
    return AttemptCancel(
        execution_id=execution_id,
        outcome=outcome,
        status=status,
        signalled=outcome == "signalled",
        pid=pid,
        host=host,
        message=message,
    )
