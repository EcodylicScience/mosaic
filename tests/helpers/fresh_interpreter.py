"""Programs run in a fresh interpreter, for the checks of what an import loads.

An in-process probe answers about whatever an earlier test already imported, so
each check starts its own interpreter.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from tests.helpers.processes import CHILD_TIMEOUT_SECONDS, run_child

__all__ = ["modules_loaded_by", "run_fresh"]


def run_fresh(
    code: str,
    *,
    args: Sequence[str] = (),
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = CHILD_TIMEOUT_SECONDS,
) -> str:
    """Run *code* in a fresh interpreter and return its stripped stdout.

    A test that reads the exit status or the standard error itself calls
    :func:`tests.helpers.run_child` with ``[sys.executable, "-c", code]``.

    Args:
        code: The program to run.
        args: The arguments the program reads as ``sys.argv[1:]``.
        cwd: The working directory of the interpreter, this process's by default.
        env: The environment of the interpreter, this process's by default.
        timeout: Seconds the program has to finish. A longer run is killed.

    Raises:
        AssertionError: The program exits nonzero, with its stderr as the message,
            or runs past *timeout*, with the program and the timeout.
    """
    completed = run_child(
        [sys.executable, "-c", code, *args], cwd=cwd, env=env, timeout=timeout
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def modules_loaded_by(statement: str, candidates: Sequence[str]) -> list[str]:
    """Return the *candidates* that a fresh interpreter holds after *statement*, sorted."""
    code = (
        "import json, sys\n"
        f"{statement}\n"
        f"print(json.dumps(sorted(m for m in {tuple(candidates)!r} if m in sys.modules)))"
    )
    loaded: list[str] = json.loads(run_fresh(code))
    return loaded
