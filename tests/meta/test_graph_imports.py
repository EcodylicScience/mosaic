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

import subprocess
import sys
from pathlib import Path

import mosaic.core.pipeline.graph as graph_pkg

from tests.helpers import imports_of

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


def test_only_resolve_reaches_the_registry_and_only_inside_a_function() -> None:
    """No module in the package may import the feature library at module scope.

    Only the statements of the module body are read. An import inside a function
    is deferred and costs nothing until the function is called. ``resolve``
    imports the registry that way.
    """
    package_root = Path(graph_pkg.__file__).parent
    offenders = [
        module.name
        for module in sorted(package_root.glob("*.py"))
        if any(
            imported.in_package(_REGISTRY_MODULE)
            for imported in imports_of(module, scope="module", relative=False)
        )
    ]
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


def _run(code: str) -> str:
    """Run *code* in a fresh interpreter, returning its stdout."""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def test_every_exported_name_of_the_graph_package_resolves() -> None:
    names = sorted(graph_pkg.__all__)
    code = (
        f"from mosaic.core.pipeline.graph import {', '.join(names)}\n"
        "import mosaic.core.pipeline.graph as package\n"
        f"missing = [n for n in {names!r} if not hasattr(package, n)]\n"
        "print(missing or 'ok')"
    )
    assert _run(code) == "ok"


def test_the_graph_package_body_imports_no_submodule() -> None:
    """``graph/__init__.py`` binds its names on access, as ``core.pipeline`` does."""
    code = (
        "import sys\n"
        "import mosaic.core.pipeline.graph\n"
        "prefix = 'mosaic.core.pipeline.graph.'\n"
        "print(sorted(m for m in sys.modules if m.startswith(prefix)) or 'none')"
    )
    assert _run(code) == "none"
