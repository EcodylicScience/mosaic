"""The ``graph`` package must be importable without the feature library.

Importing ``FEATURES`` costs seconds of wall clock -- it imports every feature
module, and through them scipy, sklearn and pywt. That is fine to pay once in a
process that is about to compute something, and it is not fine on the paths that
read a graph rather than run one: validating a recipe, ordering it, listing a
step's parents, deciding a lane, rendering a status view, evaluating a release
gate, cancelling. The gate runs far more often than a submit does, so a
three-second floor there is the shape of mistake that only shows up in
production.

Two habits hold the line, and this checks both. Only ``resolve`` reaches the
registry at all, and it does so **inside its functions** rather than at module
scope -- so importing the package, which re-exports through ``plan``, costs
nothing. Both are easy to undo by accident while editing.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import mosaic.core.pipeline.graph as graph_pkg

_REGISTRY_MODULE = "mosaic.behavior.feature_library"

_IMPORT_WITH_REGISTRY_BLOCKED = f"""
import importlib.abc
import sys


class NoFeatureLibrary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "{_REGISTRY_MODULE}" or fullname.startswith("{_REGISTRY_MODULE}."):
            raise ImportError(
                f"the feature library is blocked for this test (asked for {{fullname}})"
            )
        return None


sys.meta_path.insert(0, NoFeatureLibrary())

import mosaic.core.pipeline.graph  # noqa: F401

assert "{_REGISTRY_MODULE}" not in sys.modules, "the feature library was imported"
print("OK")
"""


def test_importing_the_graph_package_does_not_import_the_feature_library() -> None:
    """A fresh interpreter must import the package with the registry blocked."""
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITH_REGISTRY_BLOCKED],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "importing mosaic.core.pipeline.graph reaches the feature library, which "
        "puts a multi-second floor on every read path.\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "OK" in proc.stdout


def _module_level_imports(source: str) -> set[str]:
    """Every module imported at module scope, ignoring imports inside functions.

    A walk of the top-level body rather than ``ast.walk``, which is the whole
    point: an import inside a function is deferred and costs nothing until it is
    called, and that is the sanctioned way for ``resolve`` to reach the registry.
    """
    tree = ast.parse(source)
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def test_only_resolve_reaches_the_registry_and_only_inside_a_function() -> None:
    """No module in the package may import the feature library at module scope."""
    package_root = Path(graph_pkg.__file__).parent
    offenders: list[str] = []
    for module in sorted(package_root.glob("*.py")):
        imported = _module_level_imports(module.read_text())
        if any(name.startswith(_REGISTRY_MODULE) for name in imported):
            offenders.append(module.name)
    assert not offenders, (
        f"{', '.join(offenders)} import the feature library at module scope. "
        f"Move the import inside the function that needs it, as resolve.py does."
    )


def test_the_registry_is_reached_from_exactly_one_module() -> None:
    """Whichever module mentions the registry, there must be only the one."""
    package_root = Path(graph_pkg.__file__).parent
    mention = {
        module.name
        for module in sorted(package_root.glob("*.py"))
        if _REGISTRY_MODULE in module.read_text() and module.name != "__init__.py"
    }
    assert mention <= {"resolve.py"}, (
        f"{', '.join(sorted(mention - {'resolve.py'}))} reach the feature "
        f"registry. Only resolve.py may; everything else takes declarations."
    )
