"""Every reference to a test module, and every link into this repository, resolves.

Docstrings, comments and documents name the test that enforces a rule, and a reader
follows the name to it. The documentation site publishes links into the
repository on GitHub. Both stop resolving when a file moves or is renamed, and
no other check fails when they do.

The scan reads the repository's prose: ``src``, ``tests``, ``docs``,
``notebooks``, ``.github`` and the top-level documents. It leaves out
``CHANGELOG.md``, whose entries describe the tree at each release, and the
plans, specs, issues and drafts under ``docs/``, which describe it as it was
when each was written.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Final

from tests.helpers import REPO_ROOT, TESTS_ROOT

_SCANNED_FILES: Final = ("CLAUDE.md", "CONTRIBUTING.md", "README.md", "pyproject.toml")
_SCANNED_TREES: Final = ("src", "tests", "docs", "notebooks", ".github")
_UNSCANNED_TREES: Final = frozenset(
    Path("docs", name) for name in ("plans", "specs", "issues", "drafts")
)
_UNSCANNED_DIRECTORIES: Final = frozenset(
    {"__pycache__", "site-packages", "node_modules"}
)
"""Directories of generated or installed files, skipped at any depth."""

_PROSE_SUFFIXES: Final = frozenset({".py", ".md", ".toml", ".yml", ".yaml", ".ipynb"})

_TEST_MODULE_PATH: Final = re.compile(r"(?<![\w/.-])(tests/[\w/]+\.py)\b")
_TEST_MODULE_NAME: Final = re.compile(r"(?<![\w/.-])(test_\w+\.py)\b")
_REPOSITORY_LINK: Final = re.compile(
    r"github\.com/EcodylicScience/mosaic/(?:blob|tree)/main/([\w./-]*[\w/-])"
)


def _skipped(directory: Path) -> bool:
    """True for a directory whose files the scan does not read."""
    return (
        directory.name.startswith(".")
        or directory.name in _UNSCANNED_DIRECTORIES
        or directory.relative_to(REPO_ROOT) in _UNSCANNED_TREES
    )


def _scanned_paths() -> Iterator[Path]:
    """Each file of prose that the scan reads.

    A tool environment built under ``src/`` is a hidden directory of thousands
    of files. The walk prunes each skipped directory before entering it.
    """
    for name in _SCANNED_FILES:
        yield REPO_ROOT / name
    for tree in _SCANNED_TREES:
        for directory, subdirectories, files in (REPO_ROOT / tree).walk():
            subdirectories[:] = [
                name for name in subdirectories if not _skipped(directory / name)
            ]
            yield from (
                directory / name
                for name in sorted(files)
                if Path(name).suffix in _PROSE_SUFFIXES
            )


@cache
def _scanned_texts() -> tuple[tuple[str, str], ...]:
    """Each scanned file's path relative to the repository, with its text."""
    return tuple(
        (path.relative_to(REPO_ROOT).as_posix(), path.read_text(encoding="utf-8"))
        for path in _scanned_paths()
    )


def _references(pattern: re.Pattern[str]) -> Iterator[tuple[str, str]]:
    """Each match of *pattern*'s group in the scanned prose, with its file and line."""
    for name, text in _scanned_texts():
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            yield f"{name}:{line}", match.group(1)


def _test_modules_by_name() -> dict[str, str]:
    """Each test module's basename, mapped to its path relative to the repository."""
    return {
        path.name: path.relative_to(REPO_ROOT).as_posix()
        for path in TESTS_ROOT.rglob("test_*.py")
        if "__pycache__" not in path.parts
    }


def test_the_scan_reads_the_documents_that_name_tests() -> None:
    names = {name for name, _ in _scanned_texts()}

    assert {
        "CLAUDE.md",
        "pyproject.toml",
        ".github/workflows/ci.yml",
        "docs/guides/tracking/write-a-converter.md",
        "src/mosaic/core/params.py",
        "tests/meta/test_repository_references.py",
    } <= names
    assert not any(name.startswith("docs/plans/") for name in names)


def test_every_test_module_path_names_a_file() -> None:
    modules = _test_modules_by_name()
    stale = [
        f"{where} names {path}"
        + (f", now {modules[name]}" if (name := Path(path).name) in modules else "")
        for where, path in _references(_TEST_MODULE_PATH)
        if not (REPO_ROOT / path).is_file()
    ]

    assert not stale, f"rewrite each path to the module's current one: {stale}"


def test_every_test_module_name_names_a_module() -> None:
    """A bare basename resolves because basenames are unique across the suite."""
    modules = _test_modules_by_name()
    unknown = [
        f"{where} names {name}"
        for where, name in _references(_TEST_MODULE_NAME)
        if name not in modules
    ]

    assert not unknown, f"name a test module that exists: {unknown}"


def test_every_link_into_the_repository_names_a_path() -> None:
    broken = [
        f"{where} links {path}"
        for where, path in _references(_REPOSITORY_LINK)
        if not (REPO_ROOT / path).exists()
    ]

    assert not broken, f"link a path that exists on main: {broken}"
