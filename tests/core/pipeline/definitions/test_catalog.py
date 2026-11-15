"""The catalogs joined with the definitions of the scope in force."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.definitions.define import define_feature
from mosaic.core.pipeline.definitions.store import DEFINITIONS_ROOT, DefinitionCorrupt
from mosaic.core.pipeline.definitions.scope import (
    clear_definition_caches,
    definition_scope,
)
from mosaic.core.pipeline.graph import (
    Recipe,
    check_recipe,
    declaration_catalog,
    describe_step_type,
    editor_catalog,
    lane_for_step,
)
from mosaic.core.pipeline.graph.wire import CatalogEntry
from mosaic.core.pipeline.graph.lanes import DEFAULT_LANE, GPU_INFER_LANE
from mosaic.core.pipeline.graph.resolve import (
    builtin_declaration_catalog,
    builtin_editor_catalog,
    catalog_index,
    step_type_detail,
)
from tests.helpers import (
    GLOBAL_NAME,
    GLOBAL_SOURCE,
    MINIMAL_NAME,
    MINIMAL_SOURCE,
    defined_dataset,
)

SLUG: Final = f"def:{MINIMAL_NAME}"
GLOBAL_SLUG: Final = f"def:{GLOBAL_NAME}"
SCALED_SOURCE: Final = MINIMAL_SOURCE.replace(
    "scale: float = 1.0", "scale: float = 2.0"
)
TRIPLED_SOURCE: Final = MINIMAL_SOURCE.replace(
    "scale: float = 1.0", "scale: float = 3.0"
)
GPU_NAME: Final = "gpu-distance"
GPU_SOURCE: Final = MINIMAL_SOURCE.replace(
    f'name = "{MINIMAL_NAME}"', f'name = "{GPU_NAME}"'
).replace(
    "    consumed_roots = ()\n", '    consumed_roots = ()\n    resource_class = "gpu"\n'
)


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _scale_default(entry: CatalogEntry) -> JsonValue:
    """The default of ``scale`` in the params schema of *entry*."""
    schema = entry.params_schema
    assert schema is not None
    properties = schema["properties"]
    assert isinstance(properties, dict)
    scale = properties["scale"]
    assert isinstance(scale, dict)
    return scale["default"]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    """A dataset whose ``origin-distance`` has two versions, the scaled one current."""
    defined = defined_dataset(
        tmp_path_factory.mktemp("catalog"),
        {
            MINIMAL_NAME: MINIMAL_SOURCE,
            GLOBAL_NAME: GLOBAL_SOURCE,
            GPU_NAME: GPU_SOURCE,
        },
    )
    _ = define_feature(defined, MINIMAL_NAME, SCALED_SOURCE)
    return defined


@pytest.fixture(scope="module")
def other_dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    """Another dataset, whose ``origin-distance`` is a third source."""
    return defined_dataset(
        tmp_path_factory.mktemp("other"), {MINIMAL_NAME: TRIPLED_SOURCE}
    )


@pytest.fixture(autouse=True)
def forget_loaded_definitions() -> Iterator[None]:
    """Remove the definition modules a test imported, so each test loads afresh."""
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith("mosaic_definitions."):
            _ = sys.modules.pop(name)


def _recipe(feature: str, **params: object) -> Recipe:
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [
                {
                    "id": "d",
                    "type": "feature",
                    "feature": feature,
                    "inputs": ["tracks"],
                    "params": params,
                }
            ],
        }
    )


# --- a scope lists its definitions --------------------------------------------


def test_the_editor_catalog_lists_a_definition_with_what_its_sidecar_declares(
    dataset: Dataset,
) -> None:
    with definition_scope(dataset, load=False):
        entries = {entry.name: entry for entry in editor_catalog().entries}
    entry = entries[SLUG]
    assert entry.type == "feature"
    assert entry.version == "1"
    assert entry.summary == "Each row's distance from the origin."
    assert _scale_default(entry) == 2.0
    assert entry.definition is not None
    assert entry.definition.name == MINIMAL_NAME
    assert entry.definition.digest == _digest(SCALED_SOURCE)
    assert entries[GLOBAL_SLUG].declaration.fits is True
    assert [name for name in entries] == sorted(entries)


def test_a_built_in_entry_has_no_definition(dataset: Dataset) -> None:
    with definition_scope(dataset, load=False):
        entries = editor_catalog().entries
    assert all(
        entry.definition is None
        for entry in entries
        if not entry.name.startswith("def:")
    )
    assert all(entry.definition is None for entry in builtin_editor_catalog().entries)


def test_the_readers_of_the_editor_catalog_include_a_definition(
    dataset: Dataset,
) -> None:
    with definition_scope(dataset, load=False):
        described = describe_step_type(SLUG)
        indexed = {entry.name: entry for entry in catalog_index().entries}[SLUG]
        detail = step_type_detail(SLUG)
        global_detail = step_type_detail(GLOBAL_SLUG, detail="schema")
        indexed_global = {entry.name: entry for entry in catalog_index().entries}[
            GLOBAL_SLUG
        ]
    assert described.definition == indexed.definition == detail.definition
    assert indexed.parallelizable is True
    assert indexed_global.parallelizable is False
    assert detail.scope_dependent is False
    assert [row.path for row in detail.params] == ["scale"]
    assert global_detail.scope_dependent is True
    assert global_detail.params_schema is not None


def test_the_catalogs_import_no_definition_under_a_load_scope(
    dataset: Dataset,
) -> None:
    with definition_scope(dataset, load=True):
        _ = editor_catalog()
        _ = catalog_index()
        _ = step_type_detail(GLOBAL_SLUG)
        assert declaration_catalog().get(SLUG) is not None
    assert not [name for name in sys.modules if name.startswith("mosaic_definitions.")]


def test_the_lane_of_a_definition_follows_its_attributes(dataset: Dataset) -> None:
    with definition_scope(dataset, load=False):
        assert lane_for_step(SLUG) == DEFAULT_LANE
        assert lane_for_step(f"def:{GPU_NAME}") == GPU_INFER_LANE
        lanes = {entry.name: entry.lane for entry in catalog_index().entries}
    assert lanes[SLUG] == DEFAULT_LANE
    assert lanes[f"def:{GPU_NAME}"] == GPU_INFER_LANE
    with pytest.raises(KeyError):
        _ = lane_for_step(SLUG)


def test_an_entry_is_derived_once_per_version(dataset: Dataset) -> None:
    with definition_scope(dataset, load=False):
        first = describe_step_type(SLUG)
        assert describe_step_type(SLUG) is first
        clear_definition_caches()
        assert describe_step_type(SLUG) is not first


def test_a_pin_names_the_version_a_bare_slug_lists(dataset: Dataset) -> None:
    pinned = _digest(MINIMAL_SOURCE)
    with definition_scope(dataset, load=False, pins={MINIMAL_NAME: pinned}):
        entry = describe_step_type(SLUG)
    assert entry.definition is not None
    assert entry.definition.digest == pinned
    assert _scale_default(entry) == 1.0


# --- no scope, a disabled scope, and the built-in catalogs --------------------


def test_without_a_scope_the_catalogs_are_the_built_in_catalogs() -> None:
    assert declaration_catalog().entries == builtin_declaration_catalog().entries
    assert editor_catalog().entries == builtin_editor_catalog().entries
    assert declaration_catalog().get(SLUG) is None
    assert SLUG not in declaration_catalog()


def test_a_scope_without_a_dataset_lists_nothing() -> None:
    with definition_scope(None, load=False):
        assert editor_catalog().entries == builtin_editor_catalog().entries
        assert declaration_catalog().entries == builtin_declaration_catalog().entries


def test_a_disabled_scope_gives_the_built_in_catalogs(dataset: Dataset) -> None:
    with definition_scope(dataset, load=False, allowed=False):
        assert declaration_catalog().entries == builtin_declaration_catalog().entries
        assert editor_catalog().entries == builtin_editor_catalog().entries
        assert declaration_catalog().get(SLUG) is None
        assert SLUG not in declaration_catalog()


def test_the_built_in_editor_catalog_is_built_once_across_scoped_calls(
    dataset: Dataset,
) -> None:
    builtin_editor_catalog.cache_clear()
    try:
        with definition_scope(dataset, load=False):
            _ = editor_catalog()
            _ = editor_catalog()
        assert builtin_editor_catalog.cache_info().misses == 1
    finally:
        builtin_editor_catalog.cache_clear()


def test_a_definition_added_between_two_calls_is_listed_at_the_second(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    growing = defined_dataset(tmp_path_factory.mktemp("growing"), {})
    with definition_scope(growing, load=False):
        before = {entry.name for entry in editor_catalog().entries}
        _ = define_feature(growing, MINIMAL_NAME, MINIMAL_SOURCE)
        after = {entry.name for entry in editor_catalog().entries}
        assert SLUG in declaration_catalog()
    assert after - before == {SLUG}


# --- a catalog passed on -------------------------------------------------------


def test_a_catalog_answers_for_the_scope_it_is_asked_under(
    dataset: Dataset, other_dataset: Dataset
) -> None:
    tripled = f"{SLUG}@{_digest(TRIPLED_SOURCE)}"
    scaled = f"{SLUG}@{_digest(SCALED_SOURCE)}"
    with definition_scope(dataset, load=False):
        catalog = declaration_catalog()
        assert catalog.get(scaled) is not None
        assert catalog.get(tripled) is None
    with definition_scope(other_dataset, load=False):
        assert catalog.get(tripled) is not None
        assert catalog.get(scaled) is None
        assert catalog.get(SLUG) is not None
    assert catalog.get(tripled) is None


# --- recipe validation ---------------------------------------------------------


def test_a_definition_step_without_a_scope_is_a_problem_naming_the_step() -> None:
    (problem,) = check_recipe(_recipe(SLUG))
    assert (problem.step, problem.where) == ("d", "feature")
    assert "dataset" in problem.message


def test_a_definition_step_validates_under_a_scope(dataset: Dataset) -> None:
    with definition_scope(dataset, load=False):
        assert check_recipe(_recipe(SLUG, scale=4.0)) == ()


def test_a_param_the_rebuilt_model_refuses_is_a_problem(dataset: Dataset) -> None:
    with definition_scope(dataset, load=False):
        problems = check_recipe(_recipe(SLUG, scale="wide"))
        unknown = check_recipe(_recipe(SLUG, shade=1))
    assert [problem.step for problem in problems] == ["d"]
    assert problems[0].where.startswith("params")
    assert [problem.step for problem in unknown] == ["d"]


def test_a_pinned_version_that_is_not_current_validates_and_finds_its_lane(
    dataset: Dataset,
) -> None:
    older = f"{SLUG}@{_digest(MINIMAL_SOURCE)[:12]}"
    with definition_scope(dataset, load=False):
        assert check_recipe(_recipe(older, scale=1.5)) == ()
        assert lane_for_step(older) == DEFAULT_LANE


def test_a_scope_without_definitions_allowed_says_so_for_a_definition_step(
    dataset: Dataset,
) -> None:
    with definition_scope(dataset, load=False, allowed=False):
        (problem,) = check_recipe(_recipe(SLUG))
    assert (problem.step, problem.where) == ("d", "feature")
    assert "not allowed" in problem.message


# --- a definition that cannot be read ----------------------------------------


@pytest.fixture
def broken(tmp_path: Path) -> Dataset:
    """A dataset with a good definition and one whose sidecar is not JSON."""
    held = defined_dataset(
        tmp_path, {MINIMAL_NAME: MINIMAL_SOURCE, GLOBAL_NAME: GLOBAL_SOURCE}
    )
    for sidecar in (held.get_root(DEFINITIONS_ROOT) / GLOBAL_NAME).glob("*.json"):
        _ = sidecar.write_text("{not json")
    return held


def test_a_definition_that_cannot_be_read_leaves_the_other_catalog_reads_working(
    broken: Dataset,
) -> None:
    builtin_recipe = _recipe("speed-angvel")
    with definition_scope(broken, load=False):
        assert check_recipe(builtin_recipe) == ()
        assert check_recipe(_recipe(SLUG)) == ()
        entries = {entry.name: entry for entry in editor_catalog().entries}
        indexed = {entry.name: entry for entry in catalog_index().entries}
        declared = declaration_catalog()
        good = declared.get(SLUG)
        detail = step_type_detail(GLOBAL_SLUG)
    assert entries[SLUG].unavailable is None
    assert good is not None
    assert GLOBAL_SLUG not in declared.entries
    reason = entries[GLOBAL_SLUG].unavailable
    assert reason is not None
    assert GLOBAL_NAME in reason
    assert indexed[GLOBAL_SLUG].unavailable == reason
    assert detail.definition is not None
    assert detail.definition.name == GLOBAL_NAME
    assert detail.scope_dependent is False


def test_looking_up_a_definition_that_cannot_be_read_raises_its_error(
    broken: Dataset,
) -> None:
    with definition_scope(broken, load=False):
        with pytest.raises(DefinitionCorrupt):
            _ = declaration_catalog().get(GLOBAL_SLUG)
        with pytest.raises(DefinitionCorrupt):
            _ = check_recipe(_recipe(GLOBAL_SLUG))
