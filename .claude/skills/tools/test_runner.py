"""Detect the test runner of a repository from its lockfile."""

from pathlib import Path
from typing import Literal

from quiet_paths import present

Runner = Literal["pytest", "vitest"]

PYTEST_MARKER = "uv.lock"
VITEST_MARKER = "package-lock.json"


class RunnerError(Exception):
    """The runner of a repository cannot be picked."""


def detect_runner(toplevel: Path) -> Runner:
    """Pick the runner by its lockfile, which a clone without installed
    dependencies has too."""
    has_pytest = present((toplevel / PYTEST_MARKER).is_file)
    has_vitest = present((toplevel / VITEST_MARKER).is_file)
    if has_pytest and has_vitest:
        raise RunnerError(
            f"cannot pick a test runner: {toplevel} has both {PYTEST_MARKER} and {VITEST_MARKER};"
            + " delete the one that does not belong to the repository"
        )
    if not has_pytest and not has_vitest:
        raise RunnerError(
            f"cannot pick a test runner: {toplevel} has neither {PYTEST_MARKER} nor {VITEST_MARKER};"
            + f" run uv lock for pytest or npm install for vitest in {toplevel}"
        )
    return "pytest" if has_pytest else "vitest"
