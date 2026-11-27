""":mod:`mosaic.core.entry`, :mod:`mosaic.core.params` and
:mod:`mosaic.core.strict_model` declare the vocabulary a mosaic model declares
its fields with, and each stays importable without dragging in the rest of the
package. :mod:`mosaic.core.regions` declares shapes and regions of interest on
that vocabulary and stays as light, and :mod:`mosaic.core.region_geometry`
adds numpy to it alone. :mod:`mosaic.core.individuals` splits a tracks table
into its individuals for both ``core.pipeline`` and the feature library, and
imports neither.

The checks below read each file's own ast rather than measuring an import at
run time, because an ast walk covers the full tree and so catches an import
deferred inside a function, which a runtime probe does not reach.

They once read the ast for a second reason that no longer holds:
``core/__init__.py`` imported ``Dataset`` eagerly, so pandas sat in
``sys.modules`` by the time any leaf was reachable and a runtime probe reported
the same pollution whichever module earned it. That file now binds its names on
access, and ``tests/meta/test_core_lazy_exports.py`` makes the runtime measurement
the ast cannot: that importing a leaf loads neither pandas nor h5py.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType

import pytest

import mosaic.core.arenas as arenas_module
import mosaic.core.entry as entry_module
import mosaic.core.expressions as expressions_module
import mosaic.core.individuals as individuals_module
import mosaic.core.params as params_module
import mosaic.core.region_geometry as region_geometry_module
import mosaic.core.regions as regions_module
import mosaic.core.strict_model as strict_model_module
import mosaic.entry_points as entry_points_module
from tests.helpers import (
    imports_of,
    imports_reaching,
    modules_loaded_by,
    package_of,
    source_tree,
)

LEAF_MODULES = (
    entry_module,
    individuals_module,
    params_module,
    region_geometry_module,
    regions_module,
    strict_model_module,
)
"""The modules the leaf rule covers, for the guard that reads all of them the
same way."""


def _mosaic_imports(path: Path) -> set[str]:
    """Every mosaic module named by an import anywhere in the file.

    Walks the whole tree rather than the module body alone, so an import
    deferred inside a function counts the same as a top-level one. A relative
    import counts too, spelled with its dot count, since a module living
    inside the mosaic package reaches back into it at any level.
    """
    return {
        imported.module
        for imported in imports_of(path, scope="anywhere", relative=True)
        if imported.module.startswith(".") or imported.in_package("mosaic")
    }


def _dynamic_import_names(path: Path) -> set[str]:
    """Whichever of ``importlib`` and ``__import__`` the file names anywhere.

    A bare name match over every identifier in the tree, not an import check.
    :func:`_mosaic_imports` reads ``ast.Import`` and ``ast.ImportFrom`` only.
    ``importlib.import_module("mosaic.core.dataset")`` inside a function is a
    call to it. That walk does not count a call as an import. None of these
    modules has a use for either name. The coarse match therefore costs nothing
    and covers every spelling: the module imported, the module imported from,
    the attribute reached and the builtin called.
    """
    dynamic = {"importlib", "__import__"}
    found: set[str] = set()
    for node in ast.walk(source_tree(path)):
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

    ``core/entry.py`` opens by claiming the standard library only. A scope imports
    it to name the entries a run covers. An import from here back into
    ``core.pipeline.inventory``, where both aliases began, is the cycle that
    move broke.
    """
    assert _mosaic_imports(Path(entry_module.__file__)) == set()


def test_individuals_has_no_mosaic_imports() -> None:
    """``core.individuals`` imports only modules outside mosaic.

    ``core.pipeline.resample_tracks`` and the features both import it. An
    import of ``COLUMNS`` from ``core.pipeline.types`` puts the pipeline package on
    the import path of a module that the pipeline imports.
    """
    assert _mosaic_imports(Path(individuals_module.__file__)) == set()


def test_strict_model_has_no_mosaic_imports() -> None:
    """``StrictModel`` imports pydantic and the standard library only."""
    assert _mosaic_imports(Path(strict_model_module.__file__)) == set()


def test_entry_points_imports_the_standard_library_only() -> None:
    """``mosaic.entry_points`` imports no mosaic module and no third-party one.

    ``core`` imports it at module level to find installed plugins, so it carries
    nothing beyond the standard library. It stays out of ``LEAF_MODULES``, because
    ``EntryPoint.load`` is a dynamic import by design, which the leaf rule below
    refuses.
    """
    path = Path(entry_points_module.__file__)
    assert _mosaic_imports(path) == set()
    outside = sorted(
        {
            name.module.split(".", 1)[0]
            for name in imports_of(path, scope="anywhere", relative=True)
            if name.module.startswith(".")
            or name.module.split(".", 1)[0] not in sys.stdlib_module_names
        }
    )
    assert not outside, f"mosaic/entry_points.py imports {outside}."


def test_params_imports_only_strict_model() -> None:
    """``Params`` imports exactly one mosaic module: ``mosaic.core.strict_model``."""
    assert _mosaic_imports(Path(params_module.__file__)) == {"mosaic.core.strict_model"}


def test_regions_imports_only_pydantic_the_standard_library_and_two_core_leaves() -> (
    None
):
    """``core.regions`` reaches pydantic, the standard library, ``params`` and
    ``strict_model``.

    ``core.arenas`` and the agent tool contract import it to declare a field over
    a boundary or a region of interest, so it carries none of the numpy geometry
    or the rest of ``core``.
    """
    allowed_mosaic = {"mosaic.core.params", "mosaic.core.strict_model"}
    imported = imports_of(
        Path(regions_module.__file__), scope="anywhere", relative=True
    )
    outside = sorted(
        {
            name.module
            for name in imported
            if name.module.startswith(".")
            or (
                name.module.split(".", 1)[0]
                not in {*sys.stdlib_module_names, "pydantic", "mosaic"}
            )
            or (name.in_package("mosaic") and name.module not in allowed_mosaic)
        }
    )
    assert not outside, (
        f"core/regions.py imports {outside}. It may import pydantic, the standard "
        f"library, mosaic.core.params and mosaic.core.strict_model only."
    )


def test_region_geometry_imports_only_numpy_the_standard_library_and_regions() -> None:
    """``core.region_geometry`` reaches numpy, the standard library and ``regions``.

    The feature library imports it for the geometry of a row's arena, so it
    carries none of pandas or the rest of ``core``.
    """
    path = Path(region_geometry_module.__file__)
    assert _mosaic_imports(path) == {"mosaic.core.regions"}
    third_party = {
        name.module.split(".", 1)[0]
        for name in imports_of(path, scope="anywhere", relative=True)
        if not name.module.startswith(".")
        and name.module.split(".", 1)[0] not in {*sys.stdlib_module_names, "mosaic"}
    }
    assert third_party == {"numpy"}


def test_arenas_imports_only_entry_recording_and_regions() -> None:
    """``core.arenas`` reaches ``entry``, ``recording`` and ``regions`` in mosaic.

    ``arenas`` is a pure model that the agent server imports lazily and whose
    schema the API routes publish, so it carries none of the numpy geometry or
    the rest of ``core``. The check reads each module an import may reach, so
    ``from mosaic.core import region_geometry`` fails it, as
    ``import mosaic.core.region_geometry`` does.
    """
    path = Path(arenas_module.__file__)
    reached = {
        module
        for imported in imports_reaching(source_tree(path), package_of(path))
        for module in imported.reaches
        if module.split(".", 1)[0] == "mosaic"
    }
    allowed = {"mosaic.core.entry", "mosaic.core.recording", "mosaic.core.regions"}
    outside = sorted(
        module
        for module in reached
        if not any(module == name or module.startswith(f"{name}.") for name in allowed)
    )
    assert not outside, (
        f"core/arenas.py reaches {outside}. Its mosaic imports are entry, recording "
        f"and regions."
    )
    assert {module for module in reached if module in allowed} == allowed


def test_the_agent_tool_contract_does_not_import_region_geometry() -> None:
    """The MCP contract declares regions through ``core.regions`` alone.

    Each import is checked for every module it may reach, so
    ``from mosaic.core import region_geometry`` is caught as well.
    """
    spec = Path(regions_module.__file__).parents[1] / "agent" / "mcp" / "spec.py"
    reaching = [
        imported.statement
        for imported in imports_reaching(source_tree(spec), package_of(spec))
        if "mosaic.core.region_geometry" in imported.reaches
    ]
    assert not reaching, f"agent/mcp/spec.py imports region_geometry: {reaching}"


@pytest.mark.parametrize("module", LEAF_MODULES, ids=lambda module: module.__name__)
def test_no_leaf_defers_an_import_through_importlib(module: ModuleType) -> None:
    """A dynamic import is the way past the guards above.

    ``importlib.import_module("mosaic.core.dataset")`` in a function body is a
    call rather than an import. The whole pipeline arrives behind it.
    ``__import__`` does the same. Naming either in one of these modules fails
    here.
    """
    named = _dynamic_import_names(Path(module.__file__ or ""))
    deferred = (
        f"{module.__name__} names {sorted(named)}. A dynamic import reaches "
        f"whatever string it is given. The guards above read spelled imports "
        f"only."
    )
    assert not named, deferred


def test_expressions_imports_only_the_standard_library_at_module_scope() -> None:
    """``core.expressions`` parses names with the standard library alone.

    The column contract reads an expression's names without numpy, and numpy
    arrives inside ``evaluate``. A function body is therefore outside this
    check, and ``LEAF_MODULES`` leaves the module out because its walk reads
    function bodies too. The statements read here are the module body, any
    ``try`` or ``if`` block in it and a class body, and an ``if TYPE_CHECKING``
    block is passed over.
    """
    outside: set[str] = set()
    pending: list[ast.AST] = list(source_tree(Path(expressions_module.__file__)).body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.If) and ast.unparse(node.test) in {
            "TYPE_CHECKING",
            "typing.TYPE_CHECKING",
        }:
            pending.extend(node.orelse)
            continue
        if isinstance(node, ast.Import):
            outside.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            outside.add("." * node.level + (node.module or ""))
        pending.extend(ast.iter_child_nodes(node))
    outside = {
        name
        for name in outside
        if name.startswith(".") or name.split(".", 1)[0] not in sys.stdlib_module_names
    }
    assert not outside, (
        f"core/expressions.py imports {sorted(outside)} at module scope. It may "
        f"import the standard library only, and numpy inside evaluate."
    )


def test_importing_expressions_leaves_numpy_unloaded() -> None:
    """Importing ``mosaic.core.expressions`` in a fresh interpreter loads no numpy."""
    assert modules_loaded_by("import mosaic.core.expressions", ("numpy",)) == []
