"""Recipes that a graph test validates, plans or runs.

The guppies recipe is the largest one the suite holds: twenty steps of the
collective-behavior analysis, with time filters, tag columns and the
``nn-delta-response`` chain that reads ``ANGLE``. The speed recipe is the
smallest that runs a feature: one ``speed-angvel`` step over the tracks. The chain
recipe adds a second step that reads the first, for a driver to stop between.
"""

from __future__ import annotations

import json

from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import Recipe, read_recipe
from tests.helpers.golden import golden_path, read_golden

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


def chain_recipe() -> Recipe:
    """The chain recipe, parsed."""
    return Recipe.model_validate(CHAIN_RECIPE)
