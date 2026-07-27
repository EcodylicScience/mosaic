"""The inactivity watchdog in :func:`run_supervised`.

A fixed total-wall-clock limit cannot fit a tool whose healthy runtime spans
seconds to hours (TREx). The watchdog kills only after a window of *no* output,
so a run that keeps printing survives regardless of length while a wedged one is
reclaimed. These tests drive real short-lived child processes so the reader
threads, the poll loop, and the group kill are all exercised together.

Every child prints faster than, or sleeps longer than, the idle window with a
wide margin, so a loaded machine's scheduling jitter does not flip the outcome.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from mosaic.core.pipeline.subprocess_util import IdleTimeoutExpired, run_supervised

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
