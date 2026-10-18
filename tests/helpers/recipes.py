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
