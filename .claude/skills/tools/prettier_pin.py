"""Pin the prettier that rewrap-md runs to one version, under a dedicated
prefix.

Prettiers found on PATH belong to other projects, at other versions.
"""

import os
from pathlib import Path

from tool_interpreter import version_report

PRETTIER_VERSION = "3.9.6"


def prettier_prefix() -> Path:
    return Path.home() / ".local" / "share" / "ecodylic-rewrap"


def prettier_program(prefix: Path) -> Path:
    return prefix / "node_modules" / ".bin" / "prettier"


def prettier_fault(program: Path) -> str | None:
    """Return why `program` is not the pinned prettier, or None when it is."""
    if not os.access(program, os.X_OK):
        return "prettier is missing"
    have = version_report(program, merge_error=False)
    if have != PRETTIER_VERSION:
        return f'expected prettier {PRETTIER_VERSION}, found "{have}"'
    return None
