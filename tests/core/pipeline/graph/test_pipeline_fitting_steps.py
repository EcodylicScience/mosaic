"""``fitting_steps`` names the feature steps of a recipe that fit.

It is the rule an apply locks by (``derive_applied_fit``) and a retirement reads,
applied to a recipe and a declaration catalog: a step fits when its feature
declares it and the step names no ``model`` to apply instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    Recipe,
    derive_applied_fit,
    fitting_steps,
    run_pipeline,
)
from mosaic.core.pipeline.graph.resolve import builtin_declaration_catalog
from mosaic.core.scope import Scope
from tests.helpers import CONVERT_TRACKS_STEP, calms21_dataset

TRAIN = Scope(groups=["train"])

SPEED: dict[str, JsonValue] = {
    "id": "speed",
    "type": "feature",
    "feature": "speed-angvel",
    "inputs": ["tracks"],
}
TEMPLATES: dict[str, JsonValue] = {
    "id": "templates",
    "type": "feature",
    "feature": "extract-templates",
    "inputs": [{"step": "speed"}],
    "params": {"n_templates": 4},
}
KMEANS: dict[str, JsonValue] = {
    "id": "kmeans",
    "type": "feature",
    "feature": "global-kmeans",
    "inputs": [{"step": "speed"}],
    "params": {"templates": {"step": "templates"}, "k": 2},
}


def _recipe(*steps: dict[str, JsonValue]) -> Recipe:
    return Recipe.model_validate(
        {"schema_version": 1, "name": "fitting", "steps": list(steps)}
    )


def test_a_fitting_feature_step_is_listed_with_its_scope_dependence() -> None:
    recipe = _recipe(SPEED, TEMPLATES, KMEANS)

    assert fitting_steps(recipe, builtin_declaration_catalog()) == {
        "templates": True,
        "kmeans": False,
    }


def test_a_step_given_a_model_applies_it_and_fits_nothing() -> None:
    applied: dict[str, JsonValue] = {
        **TEMPLATES,
        "params": {"n_templates": 4, "model": "a-run-id"},
    }
    recipe = _recipe(SPEED, applied, KMEANS)

    assert fitting_steps(recipe, builtin_declaration_catalog()) == {"kmeans": False}


def test_a_step_that_is_not_a_feature_is_not_listed() -> None:
    recipe = _recipe(CONVERT_TRACKS_STEP, SPEED)

    assert fitting_steps(recipe, builtin_declaration_catalog()) == {}


def test_a_step_whose_feature_the_catalog_lacks_is_not_listed() -> None:
    unknown = {**TEMPLATES, "id": "unknown", "feature": "no-such-feature"}
    recipe = _recipe(SPEED, TEMPLATES, unknown, KMEANS)

    assert fitting_steps(recipe, builtin_declaration_catalog()) == {
        "templates": True,
        "kmeans": False,
    }


@pytest.fixture
def split(tmp_path: Path) -> Dataset:
    """Converted tracks of a train group of two sequences and a test group of one."""
    ds = calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": {"s1": 40, "s2": 30}}, "test": {"a": {"s3": 30}}},
        seed=5,
    )
    _ = ds.convert_all_tracks()
    return ds


def test_the_steps_are_what_an_apply_locks_as_fitting(split: Dataset) -> None:
    recipe = _recipe(SPEED, TEMPLATES, KMEANS)
    catalog = builtin_declaration_catalog()
    run = run_pipeline(split, recipe, scope=TRAIN)

    applied = derive_applied_fit(
        split, recipe, run.request_id, tracks_run_id=None, catalog=catalog
    )

    assert fitting_steps(recipe, catalog) == {"templates": True, "kmeans": False}
    assert applied.locked == ("speed", "templates", "kmeans")
    assert set(applied.pins) == {"templates"}
