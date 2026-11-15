"""A dataset that declares its own roots and names no ``feature_definitions`` root."""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.definitions.scope import (
    current_scope,
    definition_scope,
    resolve_definition_slug,
)
from mosaic.core.pipeline.definitions.store import (
    DEFINITIONS_ROOT,
    DefinitionNotFound,
    FeatureDefinitionsNotAllowed,
)
from mosaic.core.pipeline.graph import Recipe, plan_pipeline
from mosaic.core.pipeline.run import run_feature

RECIPE = Recipe.model_validate(
    {
        "schema_version": 1,
        "name": "speed",
        "steps": [
            {
                "id": "speed",
                "type": "feature",
                "feature": "speed-angvel",
                "inputs": ["tracks"],
            }
        ],
    }
)


@pytest.fixture
def declared(tmp_path: Path) -> Dataset:
    """A dataset whose roots are the ones its owner declared, with no definitions."""
    ds = Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={
            "tracks": str(tmp_path / "tracks"),
            "tracks_raw": str(tmp_path / "tracks_raw"),
            "features": str(tmp_path / "features"),
        },
    )
    ds.ensure_roots()
    ds.save()
    assert not ds.has_root(DEFINITIONS_ROOT)
    return ds


def test_a_scope_of_a_dataset_without_the_root_holds_no_definitions(
    declared: Dataset,
) -> None:
    with definition_scope(declared, load=False) as scope:
        assert scope.definitions_dir is None
        assert current_scope() is scope
        with pytest.raises(DefinitionNotFound):
            _ = resolve_definition_slug("def:anything")
    assert not (declared.base_dir / DEFINITIONS_ROOT).exists()


def test_a_disabled_scope_of_such_a_dataset_still_refuses(declared: Dataset) -> None:
    with definition_scope(declared, load=False, allowed=False):
        with definition_scope(declared, load=False):
            with pytest.raises(FeatureDefinitionsNotAllowed):
                _ = resolve_definition_slug("def:anything")


def test_a_built_in_feature_runs_on_such_a_dataset(declared: Dataset) -> None:
    _ = run_feature(declared, SpeedAngvel())
    assert not (declared.base_dir / DEFINITIONS_ROOT).exists()


def test_a_plan_of_a_built_in_feature_reads_such_a_dataset(declared: Dataset) -> None:
    plan = plan_pipeline(declared, RECIPE)
    assert plan.step("speed").run_id
    assert not (declared.base_dir / DEFINITIONS_ROOT).exists()
