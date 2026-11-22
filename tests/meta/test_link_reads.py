"""The row merge is a leaf, and no write or removal path reads through a link.

A dataset never writes, renames, locks or deletes a path under a linked
dataset. The sites that write, remove or reconcile therefore stay on the
dataset's own addresses, and none of them calls a lookup that merges a link's
data. The guard reads the named bodies with ``ast`` rather than measuring a
run, because a call in a branch that no test takes is as wrong as one that
runs.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from tests.helpers import (
    SOURCE_ROOT,
    ImportedName,
    imports_in,
    imports_of,
    modules_loaded_by,
    source_tree,
)

LINKED_ROWS: Final = SOURCE_ROOT / "core" / "pipeline" / "linked_rows.py"
"""The module that merges the rows of a dataset and of its links."""

LINK_LOOKUPS: Final = frozenset(
    {
        "lookup_order",
        "merge_linked_rows",
        "tracks_rows",
        "located_tracks_table",
        "tracks_variant_location",
        "feature_rows",
        "feature_run_rows",
        "has_feature_index",
        "feature_run_roots",
        "feature_run_file",
        "readable_feature_storages",
        "latest_feature_run_root",
        "run_coverage",
        "frame_rows",
        "frame_run_file",
        "frame_run_roots",
        "latest_frame_run_id",
        "linked_frames_row_files",
        "label_series_rows",
        "resolve_series_refs",
        "labels_rows",
        "readable_label_kinds",
        "linked_training_complete",
        "open_enclosing",
        "linked_run_params",
    }
)
"""Every function that reads a link's data, by the name a call spells."""


def _imported_modules(path: Path) -> set[str]:
    """Every module the file at *path* imports, relative ones with dots."""
    return {
        imported.module
        for imported in imports_of(path, scope="anywhere", relative=True)
    }


def test_the_merge_imports_only_pandas_and_the_standard_library() -> None:
    allowed = {"pandas", *sys.stdlib_module_names}

    outside = sorted(
        module
        for module in _imported_modules(LINKED_ROWS)
        if module.split(".")[0] not in allowed
    )

    assert outside == []


def test_importing_the_merge_loads_no_dataset() -> None:
    loaded = modules_loaded_by(
        "import mosaic.core.pipeline.linked_rows",
        ["mosaic.core.dataset", "mosaic.core.pipeline.index"],
    )

    assert loaded == []


@pytest.mark.parametrize(
    ("path", "module"),
    [
        ("core/dataset_discovery.py", "mosaic.core.dataset"),
        ("tracking/model_refs.py", "mosaic.core.pipeline.linked_rows"),
    ],
    ids=["discovery-imports-no-dataset", "models-lookup-merges-no-rows"],
)
def test_a_module_below_a_lookup_does_not_import_it(path: str, module: str) -> None:
    """``from a import b`` reaches ``a`` and ``a.b``, since ``b`` may be a module."""
    reached = {
        reach
        for imported in imports_of(SOURCE_ROOT / path, scope="anywhere", relative=False)
        for reach in (
            imported.module,
            f"{imported.module}.{imported.name}" if imported.name else imported.module,
        )
    }

    assert module not in reached


@dataclass(frozen=True, slots=True)
class GuardedSite:
    """One definition the guard reads: a function, a method or a whole class.

    Attributes:
        path: The module under ``src/mosaic``, as a path relative to it.
        qualified: ``name`` for a module-level function or class, and
            ``Class.name`` for one of its methods. A class named alone covers
            every method and nested function in it.
    """

    path: str
    qualified: str


GUARDED_SITES: Final = (
    GuardedSite("core/pipeline/feature_run_removal.py", "remove_feature_run"),
    GuardedSite("core/pipeline/feature_run_removal.py", "finish_feature_run_removal"),
    GuardedSite("core/pipeline/feature_run_removal.py", "_checked_run_root"),
    GuardedSite("core/pipeline/graph/cleanup.py", "_remove_approved"),
    GuardedSite("core/pipeline/graph/cleanup.py", "remove_approved_runs"),
    GuardedSite("core/pipeline/graph/retire.py", "run_retirement"),
    GuardedSite("core/pipeline/graph/cleanup.py", "_keepers"),
    GuardedSite("core/pipeline/delete_set.py", "_apply"),
    GuardedSite("core/dataset.py", "Dataset.drop_entries"),
    GuardedSite("core/dataset.py", "Dataset.reindex_features"),
    GuardedSite("core/media/prune.py", "prune_media"),
    GuardedSite("core/pipeline/reconcile_features.py", "FeatureReconciler"),
    GuardedSite("core/pipeline/reconcile_variants.py", "TracksReconciler"),
    GuardedSite("core/pipeline/reconcile_variants.py", "_VariantReconciler"),
    GuardedSite("core/pipeline/reconcile_variants.py", "LabelsReconciler"),
    GuardedSite("core/pipeline/labels_index.py", "write_labels_row"),
    GuardedSite("core/pipeline/labels_migration.py", "revert_labels"),
    GuardedSite("core/pipeline/tracks_index.py", "backfill_frame_extents"),
    GuardedSite("core/pipeline/tracks_index.py", "backfill_pair_frames"),
    GuardedSite("core/pipeline/tracks_index.py", "_backfill_cell"),
    GuardedSite("core/pipeline/tracks_index.py", "index_tracks_variant"),
    GuardedSite("core/pipeline/upgrade_tracks.py", "upgrade_trex_tables"),
    GuardedSite("core/pipeline/pipeline.py", "Pipeline.clean"),
    GuardedSite("core/pipeline/run.py", "_run_feature_impl"),
    GuardedSite("tracking/frame_extraction/dataset_runs.py", "_run_extract_frames"),
)
"""The write, removal and reconcile sites that stay on own addresses."""


Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
"""A statement that defines a function, a method or a class."""


def _definition(tree: ast.Module, qualified: str) -> Definition | None:
    """The function, class or method that *qualified* names, or ``None``."""
    scope: list[ast.stmt] = tree.body
    found: Definition | None = None
    for name in qualified.split("."):
        found = next(
            (
                node
                for node in scope
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
                and node.name == name
            ),
            None,
        )
        if found is None:
            return None
        scope = found.body if isinstance(found, ast.ClassDef) else []
    return found


def _import_aliases(imports: list[ImportedName]) -> dict[str, str]:
    """Maps each name an ``import ... as`` binds to the name it imports."""
    return {
        imported.asname: imported.name or imported.module
        for imported in imports
        if imported.asname is not None
    }


def lookups_called(tree: ast.Module, qualified: str) -> set[str]:
    """The lookups in :data:`LINK_LOOKUPS` that the body of *qualified* calls.

    A call through an object counts by its final name. A name bound by ``import
    ... as`` resolves to the name it imports, from the imports of the module
    and of the definition, before it is matched.

    Raises:
        AssertionError: *qualified* is not defined in *tree*, so that a rename
            fails the guard instead of emptying it.
    """
    definition = _definition(tree, qualified)
    assert definition is not None, f"{qualified} is not defined; the guard misses it"
    aliases = _import_aliases(
        [
            *imports_in(tree, scope="module", relative=True),
            *imports_in(
                ast.Module(body=[definition], type_ignores=[]),
                scope="anywhere",
                relative=True,
            ),
        ]
    )
    called: set[str] = set()
    for node in ast.walk(definition):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else ""
        )
        called.add(aliases.get(name, name))
    return called & LINK_LOOKUPS


@pytest.mark.parametrize(
    "site", GUARDED_SITES, ids=lambda site: f"{site.path}::{site.qualified}"
)
def test_a_write_path_calls_no_lookup_through_a_link(site: GuardedSite) -> None:
    tree = source_tree(SOURCE_ROOT / site.path)

    assert lookups_called(tree, site.qualified) == set()


SOURCE: Final = """
from mosaic.core.pipeline.index import latest_feature_run_root as _latest

def reads_directly(ds):
    return ds.lookup_order("feature")

def reads_through_an_alias(ds):
    return _latest(ds, "x")

def reads_through_a_local_alias(ds):
    from somewhere import feature_rows as rows
    return rows(ds)

def reads_nothing(ds):
    return ds.feature_run_root("x")

class Holder:
    def method(self, ds):
        return merge_linked_rows(ds)

    def other(self):
        def inner(ds):
            return ds.tracks_rows()
        return inner
"""


@pytest.mark.parametrize(
    ("qualified", "expected"),
    [
        ("reads_directly", {"lookup_order"}),
        ("reads_through_an_alias", {"latest_feature_run_root"}),
        ("reads_through_a_local_alias", {"feature_rows"}),
        ("reads_nothing", set[str]()),
        ("Holder.method", {"merge_linked_rows"}),
        ("Holder", {"merge_linked_rows", "tracks_rows"}),
    ],
)
def test_the_guard_finds_a_lookup_however_it_is_spelled(
    qualified: str, expected: set[str]
) -> None:
    assert lookups_called(ast.parse(SOURCE), qualified) == expected


def test_the_guard_fails_on_a_definition_that_is_gone() -> None:
    with pytest.raises(AssertionError, match="is not defined"):
        lookups_called(ast.parse(SOURCE), "renamed_away")
