"""What a test module may reach for, wherever its file sits in the suite.

A test module that imports another test module, or that finds a file by counting
directories up from its own ``__file__``, depends on where both files sit. Moved,
the first fails to import, and the second can go on passing while it checks
nothing: a scan pointed at a directory that is no longer there finds nothing to
complain about. So the suite shares code only through ``tests.helpers``, and
finds the repository only through the paths that the helpers export.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

from tests.helpers import TESTS_ROOT, source_tree


def _suite_modules() -> list[Path]:
    """Every test module and conftest under ``tests/``, outside ``tests/helpers``."""
    helpers = TESTS_ROOT / "helpers"
    return sorted(
        path
        for path in TESTS_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
        and helpers not in path.parents
        and (path.name.startswith("test_") or path.name == "conftest.py")
    )


def _imported_modules(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """Each absolute import in *tree*, as its line and the module it names."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.lineno, node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name


def _where(path: Path, line: int) -> str:
    return f"{path.relative_to(TESTS_ROOT).as_posix()}:{line}"


def test_the_scan_reads_every_test_module() -> None:
    names = {path.name for path in _suite_modules()}

    assert {"conftest.py", "test_suite_layout.py", "test_pytest_config.py"} <= names


def test_a_test_module_imports_the_suite_only_through_the_helpers() -> None:
    """Shared test code lives in ``tests.helpers`` and is imported from it alone.

    Importing another test module ties the two files' locations together, and
    importing a helper submodule bypasses the facade that lets a helper move
    between submodules without touching a caller.
    """
    offenders = [
        f"{_where(path, line)} imports {module}"
        for path in _suite_modules()
        for line, module in _imported_modules(source_tree(path))
        if module.split(".")[0] == "tests" and module != "tests.helpers"
    ]

    assert not offenders, (
        "import shared test code as `from tests.helpers import X`, moving it into "
        f"tests/helpers/ and exporting it there if it is not yet: {offenders}"
    )


def test_no_test_module_finds_a_path_from_its_own_file() -> None:
    """A path derived from ``__file__`` changes meaning when the file moves.

    ``REPO_ROOT``, ``TESTS_ROOT`` and ``GOLDEN_DIR`` from ``tests.helpers`` name
    the same places from wherever a test sits.
    """
    offenders = [
        _where(path, node.lineno)
        for path in _suite_modules()
        for node in ast.walk(source_tree(path))
        if isinstance(node, ast.Name) and node.id == "__file__"
    ]

    assert not offenders, (
        "use REPO_ROOT, TESTS_ROOT or GOLDEN_DIR from tests.helpers instead of "
        f"a path derived from __file__: {offenders}"
    )
