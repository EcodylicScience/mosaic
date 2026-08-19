"""The inactivity watchdog in :func:`run_supervised`.

A fixed total-wall-clock limit cannot fit a tool whose healthy runtime spans
seconds to hours (TREx). The watchdog kills only after a window of *no* output,
so a run that keeps printing survives regardless of length while a wedged one is
reclaimed. These tests drive real short-lived child processes so the reader
threads, the poll loop, and the group kill are all exercised together.

The cancel predicate is covered here too, and deliberately: it is the other way
out of the poll loop, it is what every external tool passes, and its answer is a
kill. Anything that wants a tool to stop *cooperatively* has to be built beside
it rather than on top of it, so what it does is written down.

Every child prints faster than, or sleeps longer than, the idle window with a
wide margin, so a loaded machine's scheduling jitter does not flip the outcome.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from mosaic.core.pipeline.subprocess_util import (
    IdleTimeoutExpired,
    ProcessCancelled,
    run_supervised,
)

# A child that prints one line every 0.1 s -- an order of magnitude below any
# idle window used here, so it must never trip the watchdog.
CHATTY_STDOUT = (
    "import time\n[(print(i, flush=True), time.sleep(0.1)) for i in range(12)]\n"
)
# The same cadence, but on stderr -- output on either stream is liveness.
CHATTY_STDERR = (
    "import time, sys\n"
    "[(print(i, file=sys.stderr, flush=True), time.sleep(0.1)) for i in range(12)]\n"
)
# One line, then a long silence -- the hang the watchdog exists to catch.
GOES_SILENT = "import time\nprint('start', flush=True)\ntime.sleep(30)\n"
# Never stops printing -- only an absolute ceiling can stop it.
NEVER_STOPS = (
    "import time\n[(print(i, flush=True), time.sleep(0.1)) for i in range(3000)]\n"
)


def _argv(program: str) -> list[str]:
    return [sys.executable, "-c", program]


def test_a_chatty_child_survives_past_the_idle_window() -> None:
    """A run that keeps printing is never idle, however long it lasts."""
    seen: list[str] = []
    stdout, _stderr, rc = run_supervised(
        _argv(CHATTY_STDOUT),
        idle_timeout=0.6,
        poll_interval=0.05,
        on_output=seen.append,
    )

    assert rc == 0, "a healthy chatty child must exit on its own, not be killed"
    assert "0" in stdout and "11" in stdout, "all output is captured"
    assert seen, "on_output receives the stdout lines"


def test_output_on_stderr_also_counts_as_activity() -> None:
    """Liveness is any output; a child that only writes stderr is still alive."""
    _stdout, stderr, rc = run_supervised(
        _argv(CHATTY_STDERR),
        idle_timeout=0.6,
        poll_interval=0.05,
    )

    assert rc == 0, "stderr activity must reset the idle timer, not be ignored"
    assert "11" in stderr


def test_a_silent_child_is_idle_killed_with_partial_output() -> None:
    """The core fix: a hung run is reclaimed, and what it printed is preserved."""
    with pytest.raises(IdleTimeoutExpired) as excinfo:
        run_supervised(
            _argv(GOES_SILENT),
            idle_timeout=0.4,
            poll_interval=0.05,
        )

    exc = excinfo.value
    assert isinstance(exc, subprocess.TimeoutExpired), (
        "an existing except TimeoutExpired must still catch a hang"
    )
    assert exc.timeout == 0.4, "the exception carries the idle window"
    assert exc.output is not None and "start" in exc.output, (
        "the partial output before the hang is attached"
    )


def test_the_absolute_ceiling_still_fires_independently() -> None:
    """A run that never idles is bounded by the optional total wall-clock cap."""
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        run_supervised(
            _argv(NEVER_STOPS),
            timeout=0.5,
            idle_timeout=None,
            poll_interval=0.05,
        )

    assert not isinstance(excinfo.value, IdleTimeoutExpired), (
        "hitting the absolute ceiling is not an inactivity kill"
    )


def test_no_watchdog_by_default_lets_a_short_child_finish() -> None:
    """Both bounds default off, matching the prior no-limit behavior."""
    stdout, _stderr, rc = run_supervised(_argv("print('done')\n"))

    assert rc == 0
    assert "done" in stdout


# A child that appends to a file forever -- a cancel has to actually stop it, and
# the file is what proves the process is gone rather than merely disowned.
def _appender(path: str) -> str:
    return (
        "import time\n"
        f"handle = open({path!r}, 'a')\n"
        "while True:\n"
        "    handle.write('x')\n"
        "    handle.flush()\n"
        "    print('tick', flush=True)\n"
        "    time.sleep(0.05)\n"
    )


def _fires_after(calls: int) -> Callable[[], bool]:
    """A cancel predicate that answers True from its *calls*-th question on."""
    asked = [0]

    def check() -> bool:
        asked[0] += 1
        return asked[0] >= calls

    return check


def test_a_cancelled_child_raises_process_cancelled() -> None:
    """The documented cancel contract, which nothing exercised.

    ``run_supervised`` is the single supervision primitive behind every external
    tool, and its answer to a fired predicate is a process-group kill. That is
    correct for a tracker, whose unit of loss is one video it will redo, and it
    is what a cooperative epoch-boundary stop has to be built *beside* rather
    than on top of -- so the behavior is pinned here before anything relies on
    it.
    """
    with pytest.raises(ProcessCancelled) as excinfo:
        run_supervised(
            _argv(NEVER_STOPS),
            cancel_check=_fires_after(2),
            poll_interval=0.05,
        )

    assert excinfo.value.argv[0] == sys.executable, (
        "the exception carries the argv it cancelled, for the message"
    )


def test_a_cancelled_child_stops_writing(tmp_path: Path) -> None:
    """A cancel kills the child rather than merely abandoning it.

    Asserted against the file the child is appending to, not against its exit
    status: a process that survived the kill and went on working would still
    give the caller a ``ProcessCancelled``, and the damage -- a tool still
    writing into a directory mosaic has released -- would be invisible.
    """
    scratch = tmp_path / "ticks.txt"
    with pytest.raises(ProcessCancelled):
        run_supervised(
            _argv(_appender(str(scratch))),
            cancel_check=_fires_after(2),
            poll_interval=0.05,
        )

    settled = scratch.stat().st_size
    time.sleep(0.6)  # a dozen writes' worth, at the child's 0.05 s cadence
    assert scratch.stat().st_size == settled, (
        "the child went on writing after the cancel, so the group was not killed"
    )
