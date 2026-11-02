"""Waiting for several pipeline requests to end, for as long as a caller asks.

:func:`wait_for_requests` reads each request through
:func:`~.status.request_status_wire`, whose own wait stops at
:data:`~.status.MAX_WAIT_SECONDS`, and reads again until the requests end. A
request ends when its verdict is anything but ``running``: ``finished``,
``partial``, ``failed``, ``cancelled``, or ``orphaned``, which ends the wait
without being a run-log status. ``mosaic wait`` prints the result.

The module imports what :mod:`~.status` imports, so a wait leaves the feature
library, pandas, cv2 and av unloaded (``tests/meta/test_status_imports.py``).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

from .model import GraphModel
from .status import MAX_WAIT_SECONDS, RequestStatusWire, request_status_wire

if TYPE_CHECKING:
    from pathlib import Path

    from .model import Request

__all__ = ["RequestWaitWire", "WaitOutcome", "WaitUntil", "wait_for_requests"]

type WaitUntil = Literal["any", "all"]
"""Whether a wait ends with the first request to end, or with the last."""

type WaitOutcome = Literal["finished", "failed", "timed_out"]
"""How a wait ended.

``finished`` when every request that ended reads ``finished``, ``failed`` when
one reads anything else, and ``timed_out`` when the time ran out first.
"""


class RequestWaitWire(GraphModel):
    """The requests a wait was given, as they stood when it ended.

    Attributes:
        until: Whether the wait ended with the first request or with the last.
        outcome: How the wait ended.
        requests: One brief report per request, in the order given.
    """

    until: WaitUntil
    outcome: WaitOutcome
    requests: tuple[RequestStatusWire, ...]


def wait_for_requests(
    base_dir: Path | str,
    requests: Sequence[Request],
    *,
    until: WaitUntil,
    timeout: float | None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> RequestWaitWire:
    """Wait until any or all of *requests* end, or *timeout* seconds pass.

    Each round waits on every running request in turn, for an equal share of
    :data:`~.status.MAX_WAIT_SECONDS`. A read returns early when a step of its
    request ends, so the request being read is seen to end at once and every
    other within about that cap.

    Args:
        base_dir: The dataset root containing ``.mosaic/runs``.
        requests: The submissions to wait for, at least one.
        until: ``any`` to end with the first request that ends, ``all`` with
            the last.
        timeout: The most seconds to wait, ``None`` for no limit.
        sleep: How a read pauses, a seam for tests.
        monotonic: The clock the wait is timed by, a seam for tests.

    Returns:
        Each request's brief report and how the wait ended.

    Raises:
        ValueError: *requests* is empty.
    """
    if not requests:
        msg = "a wait needs at least one request"
        raise ValueError(msg)
    deadline = math.inf if timeout is None else monotonic() + timeout
    reports = [
        request_status_wire(base_dir, request, brief=True) for request in requests
    ]
    while not _ended(reports, until):
        running = [index for index, report in enumerate(reports) if _running(report)]
        share = MAX_WAIT_SECONDS / len(running)
        for index in running:
            remaining = deadline - monotonic()
            if remaining <= 0:
                return RequestWaitWire(
                    until=until, outcome="timed_out", requests=tuple(reports)
                )
            reports[index] = request_status_wire(
                base_dir,
                requests[index],
                brief=True,
                wait_seconds=min(share, remaining),
                sleep=sleep,
                monotonic=monotonic,
            )
            if _ended(reports, until):
                break
    ended = [report for report in reports if not _running(report)]
    outcome: WaitOutcome = (
        "finished"
        if all(report.verdict == "finished" for report in ended)
        else "failed"
    )
    return RequestWaitWire(until=until, outcome=outcome, requests=tuple(reports))


def _running(report: RequestStatusWire) -> bool:
    return report.verdict == "running"


def _ended(reports: Sequence[RequestStatusWire], until: WaitUntil) -> bool:
    """Whether the wait for *reports* is over under *until*."""
    over = (not _running(report) for report in reports)
    return any(over) if until == "any" else all(over)
