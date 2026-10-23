"""``mosaic.agent`` is an adapter beside ``mosaic.cli``, and the analysis packages do not import it.

The agent package may import ``core``, ``behavior`` and ``tracking``. An import
the other way would make the library depend on its MCP adapter, and every
process importing the library would load the adapter's dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import SOURCE_ROOT, imports_of, inside_a_virtualenv

LIBRARY_PACKAGES = ("core", "behavior", "tracking")
"""The packages under ``src/mosaic`` that may not import ``mosaic.agent``."""

AGENT = "mosaic.agent"


def _absolute(module: str, path: Path) -> str:
    """Return the absolute module that *module*, as written in *path*, names."""
    if not module.startswith("."):
        return module
    level = len(module) - len(module.lstrip("."))
    package = ("mosaic", *path.relative_to(SOURCE_ROOT).parent.parts)
    base = package[: len(package) - level + 1]
    rest = module[level:]
    return ".".join((*base, rest) if rest else base)


def _names_agent(module: str, name: str | None) -> bool:
    """Whether importing *name* from *module* reaches ``mosaic.agent``."""
    if module == AGENT or module.startswith(f"{AGENT}."):
        return True
    return module == "mosaic" and name == "agent"


@pytest.mark.parametrize("package", LIBRARY_PACKAGES)
def test_no_library_module_imports_the_agent_package(package: str) -> None:
    """An environment built for an external tool inside the package is skipped.

    It holds installed third-party source, some of which does not parse under
    the warnings-as-errors configuration of the suite.
    """
    offenders = [
        f"{path.relative_to(SOURCE_ROOT)}:{imported.line}"
        for path in sorted((SOURCE_ROOT / package).rglob("*.py"))
        if not inside_a_virtualenv(path, SOURCE_ROOT)
        for imported in imports_of(path, scope="anywhere", relative=True)
        if _names_agent(_absolute(imported.module, path), imported.name)
    ]

    assert offenders == [], (
        f"mosaic.{package} imports mosaic.agent, an adapter above it: {offenders}"
    )


def test_the_scan_resolves_a_relative_import_to_the_agent_package() -> None:
    path = SOURCE_ROOT / "core" / "pipeline" / "run.py"

    assert _absolute("...agent.mcp", path) == "mosaic.agent.mcp"
    assert _absolute("..", path) == "mosaic.core"
    assert _names_agent(_absolute("...", path), "agent")
