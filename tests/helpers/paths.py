"""Where the repository and the suite's own data are, wherever a test file sits.

A test that finds the repository by counting parents from its own ``__file__``
breaks when the file moves into a subdirectory, and some break silently: a grep
pointed at a directory that no longer exists finds nothing and passes. Every
path the suite needs outside a test's ``tmp_path`` is derived here instead, from
this module's fixed place in ``tests/helpers/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

TESTS_ROOT: Final = Path(__file__).resolve().parent.parent
"""The ``tests`` directory."""

REPO_ROOT: Final = TESTS_ROOT.parent
"""The repository checkout the suite belongs to."""

SOURCE_ROOT: Final = REPO_ROOT / "src" / "mosaic"
"""The ``mosaic`` package's source directory in that checkout."""

GOLDEN_DIR: Final = TESTS_ROOT / "data"
"""The golden files that pin identifiers and converter output."""
