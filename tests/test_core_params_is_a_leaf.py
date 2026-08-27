""":mod:`mosaic.core.entry`, :mod:`mosaic.core.params` and
:mod:`mosaic.core.strict_model` declare the vocabulary every parameter model in
mosaic shares, and each stays importable without pulling in the rest of mosaic.

The checks below read each file's own ast rather than measuring an import at
run time. ``core/__init__.py`` imports ``Dataset`` eagerly, so pandas and the
whole pipeline machinery already sit in ``sys.modules`` by the time any leaf
module is reachable, whatever that module itself imports -- a runtime probe
would report the same pollution regardless of which module earned it. Walking
the full tree, not just the module body, also catches an import deferred
inside a function.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType

import pytest

import mosaic.core.entry as entry_module
import mosaic.core.params as params_module
import mosaic.core.strict_model as strict_model_module

LEAF_MODULES = (entry_module, params_module, strict_model_module)
"""The three modules the leaf rule covers, for the guard that reads all of
them the same way."""


def _mosaic_imports(path: Path) -> set[str]:
    """Every mosaic module named by an import anywhere in the file.

    Walks the whole tree rather than the module body alone, so an import
    deferred inside a function counts the same as a top-level one. A relative
    import counts too, spelled with its dot count, since a module living
    inside the mosaic package reaches back into it at any level.
    """
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(
                alias.name
                for alias in node.names
                if alias.name == "mosaic" or alias.name.startswith("mosaic.")
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                found.add("." * node.level + (node.module or ""))
            elif node.module and (
                node.module == "mosaic" or node.module.startswith("mosaic.")
            ):
                found.add(node.module)
    return found


def _dynamic_import_names(path: Path) -> set[str]:
    """Whichever of ``importlib`` and ``__import__`` the file names anywhere.

    A bare name match over every identifier in the tree, not an import check.
    :func:`_mosaic_imports` reads ``ast.Import`` and ``ast.ImportFrom`` only.
    ``importlib.import_module("mosaic.core.dataset")`` inside a function is a
    call to it, which that walk reports as no import at all. None of the three
    modules has a use for either name. The coarse match therefore costs nothing
    and covers every spelling: the module imported, the module imported from,
    the attribute reached and the builtin called.
    """
    dynamic = {"importlib", "__import__"}
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Name):
            spelled = {node.id}
        elif isinstance(node, ast.Attribute):
            spelled = {node.attr}
        elif isinstance(node, ast.alias):
            spelled = {node.name.split(".", 1)[0]}
        elif isinstance(node, ast.ImportFrom):
            spelled = {(node.module or "").split(".", 1)[0]}
        else:
            continue
        found |= spelled & dynamic
    return found


def test_entry_has_no_mosaic_imports() -> None:
    """``Entry`` and ``CameraEntry`` import no mosaic module at all.

    ``core/entry.py`` opens by claiming no dependencies at all. ``OpParams``
    imports it to name the entries a run covers. An import from here back into
    ``core.pipeline.inventory``, where both aliases began, is the cycle that
    move broke.
    """
    assert _mosaic_imports(Path(entry_module.__file__)) == set()


def test_strict_model_has_no_mosaic_imports() -> None:
    """``StrictModel`` imports pydantic and the standard library only."""
    assert _mosaic_imports(Path(strict_model_module.__file__)) == set()


def test_params_imports_only_strict_model() -> None:
    """``Params`` imports exactly one mosaic module: ``mosaic.core.strict_model``."""
    assert _mosaic_imports(Path(params_module.__file__)) == {"mosaic.core.strict_model"}


@pytest.mark.parametrize("module", LEAF_MODULES, ids=lambda module: module.__name__)
def test_no_leaf_defers_an_import_through_importlib(module: ModuleType) -> None:
    """A dynamic import is the way past the three guards above.

    ``importlib.import_module("mosaic.core.dataset")`` in a function body is a
    call rather than an import. The whole pipeline arrives behind it.
    ``__import__`` does the same. Naming either in one of these three modules
    fails here.
    """
    named = _dynamic_import_names(Path(module.__file__ or ""))
    deferred = (
        f"{module.__name__} names {sorted(named)}. A dynamic import reaches "
        f"whatever string it is given. The guards above read spelled imports "
        f"only."
    )
    assert not named, deferred
