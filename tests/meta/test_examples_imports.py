"""The example modules are leaves that the analysis packages do not import.

``mosaic examples list`` and a notebook's first cell read the catalog and fetch an
example before anything imports pandas, so importing either loads no
``mosaic.core``, numpy, pandas or cv2. The library packages do not import
``mosaic.examples``, which keeps the layering of CLAUDE.md free of an edge to it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.helpers import (
    SOURCE_ROOT,
    imports_reaching,
    inside_a_virtualenv,
    modules_loaded_by,
    package_of,
    source_tree,
)

HEAVY_MODULES = ("mosaic.core", "numpy", "pandas", "cv2")
"""The modules that importing an example module must not load."""

EXAMPLE_MODULES = ("mosaic.examples.catalog", "mosaic.examples.fetch")
"""The modules of the example package that the import budget covers."""

LIBRARY_PACKAGES = ("core", "behavior", "tracking", "agent")
"""The packages under ``src/mosaic`` that may not import ``mosaic.examples``."""

EXAMPLES = "mosaic.examples"


@pytest.mark.parametrize("module", EXAMPLE_MODULES)
def test_an_example_module_loads_no_heavy_module(module: str) -> None:
    assert modules_loaded_by(f"import {module}", HEAVY_MODULES) == []


def _example_imports(tree: ast.Module, path: Path) -> list[int]:
    """The lines at which *tree*, the source of the file at *path*, reaches the examples."""
    return [
        imported.line
        for imported in imports_reaching(
            tree, package_of(path), literal_packages=(EXAMPLES,)
        )
        if imported.into(EXAMPLES)
    ]


@pytest.mark.parametrize("package", LIBRARY_PACKAGES)
def test_no_library_module_imports_the_examples_package(package: str) -> None:
    """An environment built for an external tool inside the package is skipped."""
    offenders = [
        f"{path.relative_to(SOURCE_ROOT)}:{line}"
        for path in sorted((SOURCE_ROOT / package).rglob("*.py"))
        if not inside_a_virtualenv(path, SOURCE_ROOT)
        for line in _example_imports(source_tree(path), path)
    ]

    assert offenders == [], (
        f"mosaic.{package} imports mosaic.examples, a leaf above it: {offenders}"
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("from mosaic.examples import catalog", id="absolute-name"),
        pytest.param("import mosaic.examples.catalog", id="absolute-module"),
        pytest.param("from ...examples.catalog import CATALOG", id="relative-module"),
        pytest.param("from ... import examples", id="relative-name"),
    ],
)
def test_the_scan_finds_each_spelling_of_an_import_of_the_examples(
    source: str,
) -> None:
    path = SOURCE_ROOT / "core" / "pipeline" / "run.py"

    assert _example_imports(ast.parse(source), path) == [1]
