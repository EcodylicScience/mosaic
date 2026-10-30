"""``mosaic.agent`` is an adapter beside ``mosaic.cli``, and the analysis packages do not import it.

The agent package may import ``core``, ``behavior`` and ``tracking``. An import
the other way would make the library depend on its MCP adapter, and every
process importing the library would load the adapter's dependencies.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.helpers import (
    SOURCE_ROOT,
    imports_reaching,
    inside_a_virtualenv,
    source_tree,
)

LIBRARY_PACKAGES = ("core", "behavior", "tracking")
"""The packages under ``src/mosaic`` that may not import ``mosaic.agent``."""

AGENT = "mosaic.agent"


def _package(path: Path) -> str:
    """The package of the file at *path*, which a relative import resolves against."""
    return ".".join(("mosaic", *path.relative_to(SOURCE_ROOT).parent.parts))


def _agent_imports(tree: ast.Module, path: Path) -> list[int]:
    """The lines at which *tree*, the source of the file at *path*, reaches the agent.

    The scan is :func:`~tests.helpers.source_scan.imports_reaching`, so a
    relative import, a literal ``import_module`` or ``__import__`` call and a
    string naming an agent module all count.
    """
    return [
        imported.line
        for imported in imports_reaching(
            tree, _package(path), literal_packages=(AGENT,)
        )
        if imported.into(AGENT)
    ]


@pytest.mark.parametrize("package", LIBRARY_PACKAGES)
def test_no_library_module_imports_the_agent_package(package: str) -> None:
    """An environment built for an external tool inside the package is skipped.

    It holds installed third-party source, some of which does not parse under
    the warnings-as-errors configuration of the suite.
    """
    offenders = [
        f"{path.relative_to(SOURCE_ROOT)}:{line}"
        for path in sorted((SOURCE_ROOT / package).rglob("*.py"))
        if not inside_a_virtualenv(path, SOURCE_ROOT)
        for line in _agent_imports(source_tree(path), path)
    ]

    assert offenders == [], (
        f"mosaic.{package} imports mosaic.agent, an adapter above it: {offenders}"
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("from ...agent.mcp import spec", id="relative-module"),
        pytest.param("from ... import agent", id="relative-name"),
        pytest.param("import_module('mosaic.agent.mcp')", id="import-module"),
        pytest.param('_MODULE_OF = {"X": "mosaic.agent.mcp:spec"}', id="literal"),
    ],
)
def test_the_scan_finds_each_spelling_of_an_import_of_the_agent(source: str) -> None:
    path = SOURCE_ROOT / "core" / "pipeline" / "run.py"

    assert _agent_imports(ast.parse(source), path) == [1]


def test_the_scan_passes_over_a_relative_import_inside_core() -> None:
    path = SOURCE_ROOT / "core" / "pipeline" / "run.py"

    assert _agent_imports(ast.parse("from .. import agent"), path) == []
