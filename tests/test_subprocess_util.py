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
    command_summary,
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


def test_the_activity_callback_sees_stderr_that_on_output_never_does() -> None:
    """The two callbacks answer two different questions, so they take two streams.

    *on_output* is for reading what the child says, which is written against one
    stream's format. *on_activity* is for knowing it said anything, and that is
    the same signal the idle watchdog measures -- so a caller hanging a claim
    refresh on it keeps the claim alive for exactly as long as the watchdog
    considers the child alive. A tool whose progress bar goes to standard error
    reaches only the second, and until it existed such a tool read as silent.
    """
    parsed: list[str] = []
    alive: list[str] = []
    _stdout, stderr, rc = run_supervised(
        _argv(CHATTY_STDERR),
        idle_timeout=0.6,
        poll_interval=0.05,
        on_output=parsed.append,
        on_activity=alive.append,
    )

    assert rc == 0
    assert "11" in stderr, "the child really did write only to stderr"
    assert not parsed, "on_output is stdout only"
    assert len(alive) == 12, "on_activity sees every line, whichever stream it is on"


def test_the_activity_callback_also_sees_stdout() -> None:
    """Both streams, not the other one: a caller wires it once and stops caring."""
    alive: list[str] = []
    _stdout, _stderr, rc = run_supervised(
        _argv(CHATTY_STDOUT),
        idle_timeout=0.6,
        poll_interval=0.05,
        on_activity=alive.append,
    )

    assert rc == 0
    assert len(alive) == 12


def test_a_raising_activity_callback_does_not_end_the_drain() -> None:
    """It runs on the reader thread, where a raise reaches nobody.

    Worse than useless: the raise would end the loop draining the pipe, and an
    undrained pipe deadlocks a chatty child.
    """

    def explode(_line: str) -> None:
        raise RuntimeError("the liveness callback is not the child's problem")

    stdout, _stderr, rc = run_supervised(
        _argv(CHATTY_STDOUT),
        idle_timeout=0.6,
        poll_interval=0.05,
        on_activity=explode,
    )

    assert rc == 0
    assert "11" in stdout, "every line was still drained and captured"


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


# A tool's environment runs a probe as ``python -c <program> <path>``, so from a
# bin placement the program is the third token of the argv. A message about the
# run is recorded in the run-log's error_json and returned in an API error body,
# and must not quote the program.
_PROGRAM_MARKER = "# the program that no message quotes"


def _program_argv(program: str) -> list[str]:
    """*program* as a tool's environment runs it, with its path argument."""
    return [sys.executable, "-c", f"{_PROGRAM_MARKER}\n{program}", "clip.mp4"]


def _assert_names_the_program_without_quoting_it(message: str) -> None:
    assert f"{sys.executable} -c <program> clip.mp4" in message
    assert _PROGRAM_MARKER not in message


def test_a_cancel_message_does_not_quote_the_program() -> None:
    with pytest.raises(ProcessCancelled) as excinfo:
        run_supervised(
            _program_argv(NEVER_STOPS),
            cancel_check=_fires_after(2),
            poll_interval=0.05,
        )

    _assert_names_the_program_without_quoting_it(str(excinfo.value))


def test_an_idle_timeout_message_does_not_quote_the_program() -> None:
    with pytest.raises(IdleTimeoutExpired) as excinfo:
        run_supervised(
            _program_argv(GOES_SILENT),
            idle_timeout=0.4,
            poll_interval=0.05,
        )

    message = str(excinfo.value)
    _assert_names_the_program_without_quoting_it(message)
    assert "produced no output for 0.4 seconds" in message


def test_a_wall_clock_timeout_message_does_not_quote_the_program() -> None:
    """Still a ``TimeoutExpired``, so a caller catching that is unaffected."""
    argv = _program_argv(NEVER_STOPS)
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        run_supervised(argv, timeout=0.5, poll_interval=0.05)

    message = str(excinfo.value)
    _assert_names_the_program_without_quoting_it(message)
    assert "timed out after 0.5 seconds" in message
    assert excinfo.value.cmd == argv


# A decoder library can write a byte that is not UTF-8 before the tool answers,
# as an OpenCV or ffmpeg warning quoting a raw tag does.
NOT_UTF8_FIRST = (
    "import sys, time\n"
    "sys.stderr.buffer.write(b'[h264 @ 0x1] warning: \\xff\\xfe tag\\n')\n"
    "sys.stderr.flush()\n"
    "time.sleep(0.2)\n"
    "print('read frame 0', flush=True)\n"
    "sys.stderr.write('a later warning\\n')\n"
)


def test_output_that_is_not_utf8_is_read_to_the_end() -> None:
    """A byte that is not UTF-8 is replaced, and both streams are still drained."""
    stdout, stderr, rc = run_supervised(_argv(NOT_UTF8_FIRST), poll_interval=0.05)

    assert rc == 0
    assert stdout == "read frame 0\n"
    assert stderr == "[h264 @ 0x1] warning: \ufffd\ufffd tag\na later warning\n"


def test_a_summary_without_a_head_keeps_every_token_but_the_program() -> None:
    argv = ["/env/bin/python", "-c", "import sys", "a", "b", "c", "d", "e", "f"]

    assert command_summary(argv) == "/env/bin/python -c <program> a b c d e f"
    assert command_summary(argv, 3) == "/env/bin/python -c <program> ..."


def test_any_token_after_dash_c_is_a_program_whatever_the_executable() -> None:
    assert command_summary(["sh", "-c", "echo hi", "x"]) == "sh -c <program> x"
