"""``run_child``, ``run_fresh`` and ``read_line`` bound the child, so a hung child fails.

Without a bound, a child that never exits holds the whole run with no output.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.helpers import CHILD_TIMEOUT_SECONDS, read_line, run_child, run_fresh

_HANG = "import time\ntime.sleep(60)\n"
"""A program that outlives any timeout this module passes."""

_SMALL_TIMEOUT_SECONDS = 0.5

_PROMPT_SECONDS = 20.0
"""A generous cap on how long the timed-out call takes, the interpreter start included."""


def test_a_child_past_the_timeout_fails_naming_its_code_and_the_timeout() -> None:
    started = time.monotonic()

    with pytest.raises(AssertionError) as raised:
        _ = run_fresh(_HANG, timeout=_SMALL_TIMEOUT_SECONDS)

    assert time.monotonic() - started < _PROMPT_SECONDS
    message = str(raised.value)
    assert repr(_HANG) in message
    assert f"{_SMALL_TIMEOUT_SECONDS} s" in message


def test_a_child_that_finishes_within_the_timeout_returns_its_stdout() -> None:
    assert run_fresh("print('done')", timeout=CHILD_TIMEOUT_SECONDS) == "done"


def test_run_fresh_passes_arguments_the_directory_and_the_environment(
    tmp_path: Path,
) -> None:
    code = "import os, sys\nprint(os.getcwd(), os.environ['PROBE'], *sys.argv[1:])"

    printed = run_fresh(
        code,
        args=["one", "two"],
        cwd=tmp_path,
        env={"PROBE": "set"},
    )

    assert printed == f"{tmp_path.resolve()} set one two"


def test_run_child_returns_a_nonzero_exit_for_the_caller_to_read() -> None:
    completed = run_child([sys.executable, "-c", "import sys\nsys.exit(3)"])

    assert completed.returncode == 3


def test_a_command_past_the_timeout_fails_naming_the_command() -> None:
    argv = [sys.executable, "-c", _HANG]

    with pytest.raises(AssertionError) as raised:
        _ = run_child(argv, timeout=_SMALL_TIMEOUT_SECONDS)

    assert repr(argv) in str(raised.value)


def test_read_line_returns_the_first_line_of_a_child() -> None:
    with subprocess.Popen(
        [sys.executable, "-c", "print('first', flush=True)"],
        stdout=subprocess.PIPE,
        text=True,
    ) as process:
        assert read_line(process) == "first\n"


def test_a_child_that_prints_nothing_within_the_timeout_is_killed_and_closed() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time\ntime.sleep(60)\nprint('late')"],
        stdout=subprocess.PIPE,
        text=True,
    )
    started = time.monotonic()

    with pytest.raises(AssertionError) as raised:
        _ = read_line(process, timeout=_SMALL_TIMEOUT_SECONDS)

    assert time.monotonic() - started < _PROMPT_SECONDS
    assert f"{_SMALL_TIMEOUT_SECONDS} s" in str(raised.value)
    assert process.returncode is not None
    assert process.returncode != 0
    assert process.stdout is not None
    assert process.stdout.closed
