"""Recipes that a graph test validates, plans or runs.

The guppies recipe is the largest one the suite holds: twenty steps of the
collective-behavior analysis, with time filters, tag columns and the
``nn-delta-response`` chain that reads ``ANGLE``. The speed recipe is the
smallest that runs a feature: one ``speed-angvel`` step over the tracks. The chain
recipe adds a second step that reads the first, for a driver to stop between, and
the chain with a sibling adds a step that reads only the tracks, for a failure of
the chain to leave running.
A converting recipe starts by converting a dataset's CalMS21 tracks and labels.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import Recipe, read_recipe
from tests.helpers.golden import golden_path, read_golden

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.graph import PipelineRun

SPEED_RECIPE: dict[str, object] = {
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
"""One ``speed-angvel`` step over the tracks, as a recipe document."""

CHAIN_RECIPE: dict[str, object] = {
    "schema_version": 1,
    "name": "speed then templates",
    "steps": [
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "inputs": ["tracks"],
        },
        {
            "id": "templates",
            "type": "feature",
            "feature": "extract-templates",
            "inputs": [{"step": "speed"}],
            "params": {"n_templates": 4},
        },
    ],
}
"""A ``speed-angvel`` step and an ``extract-templates`` step that reads it."""

GUPPIES_RECIPE = "guppies_pipeline_v5.recipe.json"
"""The guppies recipe's file under ``tests/data/``."""


RECIPE_SAVED_BY_0_17 = "recipe_saved_by_0_17.json"
"""A recipe file as mosaic 0.17's ``save_recipe`` wrote it, under ``tests/data/``.

Its params hold whole-number floats, which 0.17 kept and later versions write as
integers. It is a static input, not a golden file: it records what an earlier
version wrote, so it is never regenerated. Its name in a dataset was
``a382f3f695561521.json``.
"""


def recipe_saved_by_0_17() -> dict[str, JsonValue]:
    """The 0.17 recipe file's document, read as written."""
    document: dict[str, JsonValue] = json.loads(
        golden_path(RECIPE_SAVED_BY_0_17).read_text()
    )
    return document


def guppies_recipe() -> Recipe:
    """The guppies recipe, parsed."""
    return read_recipe(read_golden(GUPPIES_RECIPE))


def speed_recipe() -> Recipe:
    """The speed recipe, parsed."""
    return Recipe.model_validate(SPEED_RECIPE)


def speed_recipe_stepping(step_size: int) -> Recipe:
    """The speed recipe with ``step_size`` set, so each value names its own run."""
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "speed",
            "steps": [
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                    "params": {"step_size": step_size},
                }
            ],
        }
    )


def named_recipe(recipe: Recipe, name: str) -> Recipe:
    """*recipe* under *name*, which makes it a different recipe with its own digest."""
    return Recipe.model_validate({**recipe.model_dump(mode="json"), "name": name})


def run_id_of(done: PipelineRun, step_id: str) -> str:
    """The run that step *step_id* of a finished recipe run named."""
    [found] = [o.run_id for o in done.outcomes if o.step_id == step_id and o.run_id]
    return found


def speed_heading_recipe(name: str = "old") -> Recipe:
    """``speed-angvel`` and ``heading`` over the tracks, each a storage of its own.

    A cleanup that plans ``speed_recipe_stepping`` names only the speed storage,
    which leaves ``heading__from__tracks`` as a storage the planned recipe does not
    name.
    """
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": name,
            "steps": [
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                    "params": {"step_size": 1},
                },
                {
                    "id": "heading",
                    "type": "feature",
                    "feature": "heading",
                    "inputs": ["tracks"],
                    "params": {},
                },
            ],
        }
    )


def classifier_recipe(*classifiers: str) -> Recipe:
    """CalMS21 conversions, pair features, labeled templates and one classifier per id.

    The ``labeled`` step fits on the entries it covers, so a train/test split
    scopes it to the training groups. Each classifier is an ``xgboost`` fitted
    from those templates, scope-free, with its own seed, so it is the step a
    held-out score is read from.

    Args:
        classifiers: The ids of the classifier steps, one per seed.

    Returns:
        The parsed recipe.
    """
    steps: list[dict[str, JsonValue]] = [
        CONVERT_TRACKS_STEP,
        CONVERT_LABELS_STEP,
        {
            "id": "heading",
            "type": "feature",
            "feature": "heading",
            "inputs": ["tracks"],
            "tracks": {"step": "tracks"},
            "params": {"method": "two_point", "front_idx": 0, "rear_idx": 3},
        },
        {
            "id": "pairs",
            "type": "feature",
            "feature": "pair-position",
            "inputs": ["tracks", {"step": "heading"}],
            "tracks": {"step": "tracks"},
        },
        {
            "id": "labeled",
            "type": "feature",
            "feature": "extract-labeled-templates",
            "inputs": [{"step": "pairs"}],
            "params": {"labels": {"step": "labels"}, "n_total": 200},
        },
    ]
    for seed, step in enumerate(classifiers, start=1):
        steps.append(
            {
                "id": step,
                "type": "feature",
                "feature": "xgboost",
                "inputs": [{"step": "pairs"}],
                "params": {
                    "templates": {"step": "labeled"},
                    "default_class": 0,
                    "n_estimators": 5,
                    "n_jobs": 1,
                    "random_state": seed,
                },
            }
        )
    return Recipe.model_validate(
        {"schema_version": 1, "name": "classified", "steps": steps}
    )


def run_speed(dataset: Dataset, step_size: int) -> str:
    """Run ``speed-angvel`` at *step_size* outside any request, and return its run id.

    Each value is a run of its own in the one storage
    ``speed-angvel__from__tracks``. No request is recorded, so no request holds
    the run.
    """
    from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
    from mosaic.core.pipeline.run import run_feature

    run_id = run_feature(dataset, SpeedAngvel(params={"step_size": step_size})).run_id
    assert run_id is not None
    return run_id


def chain_recipe() -> Recipe:
    """The chain recipe, parsed."""
    return Recipe.model_validate(CHAIN_RECIPE)


def chain_with_sibling_recipe() -> Recipe:
    """The chain recipe and a ``sibling`` step over the tracks that nothing reads.

    ``sibling`` runs ``speed-angvel`` at step size 2, a run apart from ``speed``.
    It comes after ``speed`` and before ``templates`` in topological order.
    """
    sibling: dict[str, JsonValue] = {
        "id": "sibling",
        "type": "feature",
        "feature": "speed-angvel",
        "inputs": ["tracks"],
        "params": {"step_size": 2},
    }
    return Recipe.model_validate(
        {**CHAIN_RECIPE, "steps": [*chain_recipe().model_dump()["steps"], sibling]}
    )


CONVERT_TRACKS_STEP: dict[str, JsonValue] = {
    "id": "tracks",
    "type": "op",
    "kind": "convert-tracks",
    "params": {"source_format": "calms21_npy"},
}
"""A step converting a dataset's CalMS21 tracks, with the id ``tracks``."""

CONVERT_LABELS_STEP: dict[str, JsonValue] = {
    "id": "labels",
    "type": "op",
    "kind": "convert-labels",
    "params": {"source_format": "calms21_npy"},
}
"""A step converting a dataset's CalMS21 behavior labels, with the id ``labels``."""

SPEED_OF_CONVERTED_STEP: dict[str, JsonValue] = {
    "id": "speed",
    "type": "feature",
    "feature": "speed-angvel",
    "inputs": ["tracks"],
    "tracks": {"step": "tracks"},
}
"""``speed-angvel`` over the tracks that the ``tracks`` step converts."""


def converting_recipe(
    *steps: Mapping[str, JsonValue], labels: bool = False, name: str = "converting"
) -> Recipe:
    """A recipe whose first step converts CalMS21 tracks, then *steps*.

    Args:
        steps: The steps after the conversion, as recipe documents.
        labels: Whether a ``labels`` step converting the behavior labels
            follows the tracks step.
        name: The recipe's name.

    Returns:
        The parsed recipe.
    """
    head: list[Mapping[str, JsonValue]] = [CONVERT_TRACKS_STEP]
    if labels:
        head.append(CONVERT_LABELS_STEP)
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": name,
            "steps": [dict(step) for step in (*head, *steps)],
        }
    )
