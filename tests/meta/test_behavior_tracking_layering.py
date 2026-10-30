"""``mosaic.behavior`` and ``mosaic.tracking`` import ``core`` and never each other.

The two packages exchange data only through files on disk. An import across
the boundary makes one package's code depend on the other's, and a change to
either can then break the other with no file format in between to say so.

The scan is :func:`~tests.helpers.source_scan.imports_reaching`. It reads each
file's syntax tree rather than importing it, so an import deferred inside a
function counts the same as one at the top of the module. A relative import is
resolved against the file's own package, and a call of ``import_module`` or
``__import__`` that names its module with a literal counts as the import it
performs. So does a string naming a module of the other package, the spelling
of a ``lazy_export`` table and of ``REGISTERING_MODULES``. A module name built
at run time is out of reach of any static scan.

The deprecated ``behavior.tracking`` alias is the one exception: the module
``__getattr__`` of ``behavior/__init__.py`` imports ``mosaic.tracking`` when it
is asked for ``tracking``. An import of the alias itself, such as ``from
mosaic.behavior import tracking``, reaches ``mosaic.tracking`` through it and
counts as an import across.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from tests.helpers import (
    SOURCE_ROOT,
    ReachedImport,
    imports_reaching,
    inside_a_virtualenv,
    package_of,
    source_tree,
)

ACROSS: Final = {"behavior": "mosaic.tracking", "tracking": "mosaic.behavior"}
"""Each package under ``src/mosaic``, mapped to the package it may not import."""

ALIASES: Final = {"mosaic.behavior.tracking": "mosaic.tracking"}
"""Each deprecated alias module, mapped to the module its access imports."""

DEPRECATED_ALIAS: Final = (
    "behavior/__init__.py",
    ("__getattr__", "if name == 'tracking'"),
    "from mosaic import tracking",
)
"""The one import across: its file, the scope it sits in and its statement.

Matched whole, so the same statement elsewhere in the file, another import in
the ``tracking`` branch, or a second name in the statement is an offender.
"""


@dataclass(frozen=True, slots=True)
class Import:
    """One name a file imports, and the file it sits in.

    Attributes:
        path: The file, relative to ``src/mosaic``.
        found: The import, with each alias module it reaches read as the module
            the alias imports.
    """

    path: str
    found: ReachedImport


def _unaliased(module: str) -> str:
    """*module*, spelled through the module its alias imports, if it has one."""
    for alias, target in ALIASES.items():
        if module == alias or module.startswith(f"{alias}."):
            return target + module[len(alias) :]
    return module


def imports(tree: ast.Module, path: Path) -> list[Import]:
    """Every name that *tree*, the source of the file at *path*, imports.

    *path* places the file in its package, which a relative import and the
    ``__package__`` of an ``import_module`` call resolve against.
    """
    relative = path.relative_to(SOURCE_ROOT)
    return [
        Import(
            relative.as_posix(),
            ReachedImport(
                found.line,
                found.scope,
                found.statement,
                tuple(_unaliased(module) for module in found.reaches),
            ),
        )
        for found in imports_reaching(
            tree, package_of(path), literal_packages=sorted(ACROSS.values())
        )
    ]


def _exempt(imported: Import) -> bool:
    """Whether *imported* is the deprecated alias, at its place and as written."""
    found = imported.found
    return (imported.path, found.scope, found.statement) == DEPRECATED_ALIAS


def _across(where: str, source: str) -> list[Import]:
    """The imports across of *source*, scanned as the file at *where*."""
    path = SOURCE_ROOT / where
    other = ACROSS[Path(where).parts[0]]
    return [
        imported
        for imported in imports(ast.parse(source), path)
        if imported.found.into(other)
    ]


@pytest.mark.parametrize("package", sorted(ACROSS))
def test_no_module_imports_the_other_analysis_package(package: str) -> None:
    """An environment built for an external tool inside the package is skipped.

    It holds installed third-party source, some of which does not parse under
    the warnings-as-errors configuration of the suite.
    """
    other = ACROSS[package]
    paths = [
        path
        for path in sorted((SOURCE_ROOT / package).rglob("*.py"))
        if not inside_a_virtualenv(path, SOURCE_ROOT)
    ]
    offenders = [
        f"{imported.path}:{imported.found.line} imports {module} "
        f"({imported.found.statement})"
        for path in paths
        for imported in imports(source_tree(path), path)
        if (module := imported.found.into(other)) and not _exempt(imported)
    ]

    assert paths, f"no module found under {SOURCE_ROOT / package}"
    assert offenders == [], (
        f"mosaic.{package} imports {other}, and the two exchange data only "
        f"through files on disk: {offenders}"
    )


def test_the_exemption_names_the_deprecated_alias() -> None:
    """The exemption lasts only as long as the alias it was written for.

    Left behind after the alias goes, it would admit a new import of the same
    spelling at the same place.
    """
    path = SOURCE_ROOT / DEPRECATED_ALIAS[0]
    found = [
        imported.found.into(ACROSS["behavior"])
        for imported in imports(source_tree(path), path)
        if _exempt(imported)
    ]

    assert found == ["mosaic.tracking"], (
        f"{DEPRECATED_ALIAS[0]} no longer imports {DEPRECATED_ALIAS[2]!r} in "
        f"{' / '.join(DEPRECATED_ALIAS[1])}; remove DEPRECATED_ALIAS with the alias"
    )


@pytest.mark.parametrize(
    ("where", "source", "expected"),
    [
        pytest.param(
            "behavior/feature_library/example.py",
            "import mosaic.tracking.sleap as sleap",
            "mosaic.tracking.sleap",
            id="import",
        ),
        pytest.param(
            "behavior/feature_library/example.py",
            "from mosaic.tracking.ops import predict_frames",
            "mosaic.tracking.ops",
            id="from-import",
        ),
        pytest.param(
            "behavior/feature_library/example.py",
            "from mosaic import core, tracking",
            "mosaic.tracking",
            id="from-mosaic-import",
        ),
        pytest.param(
            "behavior/feature_library/example.py",
            "class Feature:\n    def fit(self):\n        import mosaic.tracking",
            "mosaic.tracking",
            id="inside-a-method",
        ),
        pytest.param(
            "behavior/feature_library/example.py",
            "if TYPE_CHECKING:\n    from mosaic.tracking import Track",
            "mosaic.tracking",
            id="type-checking-block",
        ),
        pytest.param(
            "behavior/feature_library/example.py",
            "from ...tracking.common import x",
            "mosaic.tracking.common",
            id="relative-module",
        ),
        pytest.param(
            "behavior/feature_library/example.py",
            "def f():\n    from ... import tracking",
            "mosaic.tracking",
            id="relative-name",
        ),
        pytest.param(
            "tracking/sleap/example.py",
            "from ...behavior.feature_library import FEATURES",
            "mosaic.behavior.feature_library",
            id="relative-from-tracking",
        ),
        pytest.param(
            "tracking/sleap/example.py",
            "def f():\n    return importlib.import_module('mosaic.behavior')",
            "mosaic.behavior",
            id="import-module",
        ),
        pytest.param(
            "tracking/example.py",
            "import_module(name='mosaic.behavior.feature_library')",
            "mosaic.behavior.feature_library",
            id="import-module-keyword",
        ),
        pytest.param(
            "tracking/sleap/example.py",
            "importlib.import_module('..behavior', 'mosaic.tracking')",
            "mosaic.behavior",
            id="import-module-relative",
        ),
        pytest.param(
            "behavior/example.py",
            "importlib.import_module('..tracking.ops', __package__)",
            "mosaic.tracking.ops",
            id="import-module-package",
        ),
        pytest.param(
            "tracking/example.py",
            "__import__('mosaic.behavior.label_library')",
            "mosaic.behavior.label_library",
            id="dunder-import",
        ),
        pytest.param(
            "tracking/example.py",
            "__import__('mosaic', fromlist=['behavior'])",
            "mosaic.behavior",
            id="dunder-import-fromlist",
        ),
        pytest.param(
            "behavior/feature_library/example.py",
            "from mosaic.behavior import tracking",
            "mosaic.tracking",
            id="through-the-alias",
        ),
        pytest.param(
            "behavior/feature_library/example.py",
            "from .. import tracking",
            "mosaic.tracking",
            id="through-the-alias-relative",
        ),
        pytest.param(
            "tracking/__init__.py",
            '_MODULE_OF = {"X": "mosaic.behavior.feature_library"}',
            "mosaic.behavior.feature_library",
            id="lazy-export-table",
        ),
        pytest.param(
            "tracking/__init__.py",
            '_MODULE_OF = {"X": "mosaic.behavior.feature_library:FEATURES"}',
            "mosaic.behavior.feature_library",
            id="lazy-export-table-attribute",
        ),
        pytest.param(
            "tracking/registration.py",
            'MODULES = ("mosaic.behavior.label_library",)\n'
            "for module in MODULES:\n"
            "    import_module(module)",
            "mosaic.behavior.label_library",
            id="module-tuple",
        ),
    ],
)
def test_the_scan_finds_each_spelling_of_an_import_across(
    where: str, source: str, expected: str
) -> None:
    other = ACROSS[Path(where).parts[0]]

    assert [imported.found.into(other) for imported in _across(where, source)] == [
        expected
    ]


@pytest.mark.parametrize(
    ("where", "source"),
    [
        pytest.param(
            "behavior/feature_library/example.py",
            "from .tracking import x",
            id="sibling-named-like-the-package",
        ),
        pytest.param(
            "behavior/example.py",
            "import mosaic.trackings",
            id="name-sharing-a-prefix",
        ),
        pytest.param(
            "tracking/example.py",
            "from mosaic.core import tracks",
            id="core",
        ),
        pytest.param(
            "tracking/example.py",
            "importlib.import_module(name)",
            id="name-built-at-run-time",
        ),
        pytest.param(
            "behavior/example.py",
            "warnings.warn(\"Use 'from mosaic import tracking' directly.\")",
            id="string-that-is-not-an-import",
        ),
    ],
)
def test_the_scan_passes_over_an_import_inside_the_boundary(
    where: str, source: str
) -> None:
    assert _across(where, source) == []


@pytest.mark.parametrize(
    ("source", "exempt"),
    [
        pytest.param(
            "def __getattr__(name):\n"
            '    if name == "tracking":\n'
            "        from mosaic import tracking\n"
            "        return tracking",
            True,
            id="the-alias",
        ),
        pytest.param("from mosaic import tracking", False, id="module-level"),
        pytest.param(
            "def __getattr__(name):\n"
            '    if name == "media":\n'
            "        from mosaic import tracking",
            False,
            id="another-branch",
        ),
        pytest.param(
            "def __getattr__(name):\n"
            '    if name == "tracking":\n'
            "        import mosaic.tracking",
            False,
            id="another-spelling",
        ),
        pytest.param(
            "def __getattr__(name):\n"
            '    if name == "tracking":\n'
            "        from mosaic.tracking import sleap",
            False,
            id="a-submodule",
        ),
    ],
)
def test_the_exemption_admits_the_alias_alone(source: str, exempt: bool) -> None:
    found = _across(DEPRECATED_ALIAS[0], source)

    assert [_exempt(imported) for imported in found] == [exempt]
