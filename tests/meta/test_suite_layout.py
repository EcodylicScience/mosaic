"""Checks where each test module is placed, and which code and paths it may use.

A test module is placed in the directory that mirrors the ``src/mosaic``
package it exercises. The tests of one package are then adjacent, and a new
module has one correct directory. "Where a test goes" in ``CLAUDE.md`` states
the rule, and the tests below fail on a module or a directory outside it.

A test module that imports another test module, or that finds a file by counting
directories up from its own ``__file__``, depends on where both files sit. Moved,
the first fails to import, and the second can go on passing while it checks
nothing: a scan pointed at a directory that is no longer there finds nothing to
complain about. So the suite shares code only through ``tests.helpers``, and
finds the repository only through the paths that the helpers export.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from tests.helpers import SOURCE_ROOT, TESTS_ROOT, source_tree

_SUPPORT_DIRECTORIES: Final = frozenset({"helpers", "data"})
"""The directories of ``tests/`` for shared code and data files."""

_UNMIRRORED_AREA: Final = Path("meta")
"""The one area without a source package: the repository, its configuration and
the suite itself."""

_TOP_LEVEL_MODULES: Final = frozenset({"__init__.py", "conftest.py"})

_PLACEMENT_RULE: Final = 'See "Where a test goes" in CLAUDE.md.'


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


def _area_directories() -> list[Path]:
    """Every directory below ``tests/`` for test modules, relative to ``tests/``."""
    return sorted(
        relative
        for path in TESTS_ROOT.rglob("*")
        if path.is_dir()
        and "__pycache__" not in path.parts
        and (relative := path.relative_to(TESTS_ROOT)).parts[0]
        not in _SUPPORT_DIRECTORIES
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
    assert {Path("core", "pipeline"), _UNMIRRORED_AREA} <= set(_area_directories())


def test_only_the_package_and_the_root_conftest_sit_at_the_top() -> None:
    """A test module is placed in the directory of the package that it exercises."""
    stray = sorted(
        path.name
        for path in TESTS_ROOT.glob("*.py")
        if path.name not in _TOP_LEVEL_MODULES
    )

    assert not stray, (
        "move each into the directory that mirrors the src/mosaic package it "
        f"exercises. {_PLACEMENT_RULE} {stray}"
    )


def test_every_directory_holding_modules_is_a_package() -> None:
    """A directory without ``__init__.py`` has its modules imported by basename.

    pytest then puts the directory itself on ``sys.path``. Two modules of one name
    in different directories collide, and ``regenerate_command(__name__)``
    cannot find the module's file.
    """
    directories = {
        path.parent
        for path in TESTS_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    missing = sorted(
        directory.relative_to(TESTS_ROOT).as_posix()
        for directory in directories
        if not (directory / "__init__.py").is_file()
    )

    assert not missing, f"add an empty __init__.py to each: {missing}"


def test_every_area_mirrors_a_source_package() -> None:
    """Each directory of tests names a package of ``src/mosaic``, apart from ``meta``.

    A directory without a source counterpart is a category that only the suite
    knows, and a reader looking for a package's tests beside its path does not
    find the tests filed there.
    """
    unmirrored = [
        area.as_posix()
        for area in _area_directories()
        if area != _UNMIRRORED_AREA and not (SOURCE_ROOT / area).is_dir()
    ]

    assert not unmirrored, (
        "name each directory after the src/mosaic package its tests exercise, "
        f"or move its tests to one that is. {_PLACEMENT_RULE} {unmirrored}"
    )


def test_a_nested_conftest_implements_no_hooks() -> None:
    """Only the root ``conftest.py`` implements pytest hooks.

    ``pytest_collection_modifyitems`` in a nested conftest receives every item of
    the session, including items outside its directory. A nested
    ``pytest_configure`` runs only when pytest collects that directory, after the
    root one has accepted the environment. A nested conftest declares fixtures.
    """
    hooks = [
        f"{_where(path, node.lineno)} implements {node.name}"
        for path in _suite_modules()
        if path.name == "conftest.py" and path.parent != TESTS_ROOT
        for node in ast.iter_child_nodes(source_tree(path))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name.startswith("pytest_")
    ]

    assert not hooks, f"move each hook to tests/conftest.py: {hooks}"


def test_test_module_names_are_unique() -> None:
    """One basename names one test module across the suite.

    Prose and recorded references name a test module by its basename, and
    ``test_unwired_fields`` resolves the ones it records that way.
    """
    counts = Counter(
        path.name for path in _suite_modules() if path.name != "conftest.py"
    )
    shared = sorted(name for name, count in counts.items() if count > 1)

    assert not shared, f"rename all but one module of each name: {shared}"


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
