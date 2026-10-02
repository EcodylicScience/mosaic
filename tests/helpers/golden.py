"""Golden files: reading them, rewriting them, and saying how to rewrite them.

A golden file pins output that something outside the suite depends on, such as
run identifiers already written into datasets or a format other tools read. Each
is a JSON object under ``tests/data/``, written indented with sorted keys so that
a deliberate change reads as a diff. ``MOSAIC_UPDATE_GOLDEN=1`` turns a golden
test into the one that rewrites its file.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from pydantic import RootModel

from tests.helpers.paths import GOLDEN_DIR, REPO_ROOT

UPDATE_GOLDEN_ENV: Final = "MOSAIC_UPDATE_GOLDEN"
"""Set to ``1`` to rewrite the golden files instead of comparing with them."""


class _Golden(RootModel[dict[str, object]]):
    pass


class _StringGolden(RootModel[dict[str, str]]):
    pass


def updating_golden() -> bool:
    """True when this run rewrites the golden files instead of comparing."""
    return os.environ.get(UPDATE_GOLDEN_ENV) == "1"


def golden_path(name: str) -> Path:
    """The golden file *name* under ``tests/data/``."""
    return GOLDEN_DIR / name


def read_golden(name: str) -> dict[str, object]:
    """The golden object *name*, or an empty one when the file does not exist yet."""
    path = golden_path(name)
    if not path.exists():
        return {}
    return _Golden.model_validate_json(path.read_text()).root


def read_string_golden(name: str) -> dict[str, str]:
    """The golden map *name* of case ids to strings, empty when it does not exist."""
    path = golden_path(name)
    if not path.exists():
        return {}
    return _StringGolden.model_validate_json(path.read_text()).root


def write_golden(name: str, document: Mapping[str, object]) -> None:
    """Rewrite the golden file *name*, indented with sorted keys."""
    path = golden_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def regenerate_command(module: str) -> str:
    """The command that rewrites the golden files of the test module named *module*.

    *module* is the test module's ``__name__``, so the command names the file
    wherever it sits in the suite.
    """
    path = Path(*module.split(".")).with_suffix(".py")
    assert (REPO_ROOT / path).is_file(), f"{module} is not a test module"
    return f"{UPDATE_GOLDEN_ENV}=1 pytest {path.as_posix()}"
