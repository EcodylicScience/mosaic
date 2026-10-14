"""Recipes that a graph test validates, plans or runs.

The guppies recipe is the largest one the suite holds: twenty steps of the
collective-behavior analysis, with time filters, tag columns and the
``nn-delta-response`` chain that reads ``ANGLE``. The speed recipe is the
smallest that runs a feature: one ``speed-angvel`` step over the tracks. The chain
recipe adds a second step that reads the first, for a driver to stop between.
"""

from __future__ import annotations

from mosaic.core.pipeline.graph import Recipe, read_recipe
from tests.helpers.golden import read_golden

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


def guppies_recipe() -> Recipe:
    """The guppies recipe, parsed."""
    return read_recipe(read_golden(GUPPIES_RECIPE))


def speed_recipe() -> Recipe:
    """The speed recipe, parsed."""
    return Recipe.model_validate(SPEED_RECIPE)


def chain_recipe() -> Recipe:
    """The chain recipe, parsed."""
    return Recipe.model_validate(CHAIN_RECIPE)
