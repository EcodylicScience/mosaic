"""Select the test files that import changed Python files."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import PurePosixPath

from python_imports import ImportStatement, parse_module

# pytest's default `python_files`.
TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")
CONFTEST = "conftest.py"
SOURCE_ROOT = "src"


def is_test_file(path: PurePosixPath) -> bool:
    return any(fnmatchcase(path.name, pattern) for pattern in TEST_FILE_PATTERNS)


def module_name(path: PurePosixPath) -> str | None:
    """Return the module that `path`, relative to the toplevel, defines: its
    path below `src/` or the toplevel, dotted, without `.py` and a trailing
    `__init__`. None for a path not ending in `.py` and for a toplevel
    `__init__.py`."""
    if path.suffix != ".py":
        return None
    parts = list(path.with_suffix("").parts)
    if parts[0] == SOURCE_ROOT and len(parts) > 1:
        parts = parts[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or None


def own_package(path: PurePosixPath) -> list[str]:
    """Return the package that a relative import in `path` resolves against:
    the module itself for an `__init__.py`, else its parent."""
    name = module_name(path)
    parts = name.split(".") if name else []
    return parts if path.name == "__init__.py" else parts[:-1]


def with_parents(name: str) -> list[str]:
    """Return `name` and each package above it: `a`, `a.b` and `a.b.c` for
    `a.b.c`."""
    parts = name.split(".")
    return [".".join(parts[: count + 1]) for count in range(len(parts))]


def imported_modules(path: PurePosixPath, statement: ImportStatement) -> list[str]:
    """Return every module that `statement` in `path` runs, present in the tree
    or not."""
    if statement.kind == "import":
        return [module for name in statement.names for module in with_parents(name)]
    package = own_package(path)
    if statement.level > 0:
        if statement.level - 1 >= len(package):
            return []
        package = package[: len(package) - (statement.level - 1)]
        base = ".".join([*package, *([statement.module] if statement.module else [])])
    else:
        base = statement.module or ""
    if not base:
        return []
    return [*with_parents(base), *(f"{base}.{name}" for name in statement.names)]


def import_edges(
    sources: Mapping[PurePosixPath, bytes], present: set[str]
) -> tuple[dict[str, set[PurePosixPath]], set[PurePosixPath]]:
    """Return the files that import each present module, and the files that do
    not parse."""
    importers: dict[str, set[PurePosixPath]] = {}
    unparsable: set[PurePosixPath] = set()
    for path, source in sources.items():
        parsed = parse_module(source)
        if parsed is None:
            unparsable.add(path)
            continue
        modules = [module for statement in parsed.statements for module in imported_modules(path, statement)]
        for module in [*modules, *parsed.plugins]:
            if module in present:
                importers.setdefault(module, set()).add(path)
    return importers, unparsable


@dataclass(frozen=True)
class GraphSelection:
    """The test files that each changed path selects, and the changed paths
    whose source does not parse."""

    selected: Mapping[PurePosixPath, frozenset[PurePosixPath]]
    unparsable: frozenset[PurePosixPath]


def tests_below(conftest: PurePosixPath, sources: Mapping[PurePosixPath, bytes]) -> set[PurePosixPath]:
    """Return every test file in the directory of `conftest` and below it."""
    return {path for path in sources if is_test_file(path) and conftest.parent in path.parents}


def tests_selected(
    changed: PurePosixPath, sources: Mapping[PurePosixPath, bytes], importers: Mapping[str, set[PurePosixPath]]
) -> frozenset[PurePosixPath]:
    """Return the tests that `changed` selects: itself when it is a test file,
    and the reverse closure of its module over `importers`, which stops at
    every conftest. A conftest that is changed or imports the changed module
    directly selects every test file in its directory and below it."""
    selected: set[PurePosixPath] = set()
    if changed in sources and is_test_file(changed):
        selected.add(changed)
    if changed.name == CONFTEST:
        return frozenset(selected | tests_below(changed, sources))
    start = module_name(changed)
    if start is None:
        return frozenset(selected)
    visited = {start}
    pending = [start]
    while pending:
        module = pending.pop()
        for importer in importers.get(module, set()):
            if importer.name == CONFTEST:
                if module == start:
                    selected |= tests_below(importer, sources)
                continue
            if is_test_file(importer):
                selected.add(importer)
            name = module_name(importer)
            if name is not None and name not in visited:
                visited.add(name)
                pending.append(name)
    return frozenset(selected)


def select_tests(sources: Mapping[PurePosixPath, bytes], changed: Sequence[PurePosixPath]) -> GraphSelection:
    """Select the tests for each of `changed` over the import graph of
    `sources`, which maps each Python file's path relative to the toplevel to
    its source. A changed path's module counts as present whether or not it is
    in `sources`."""
    present = {name for path in [*sources, *changed] if (name := module_name(path)) is not None}
    importers, unparsable = import_edges(sources, present)
    return GraphSelection(
        selected={path: tests_selected(path, sources, importers) for path in changed},
        unparsable=frozenset(path for path in changed if path in unparsable),
    )
