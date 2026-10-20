"""The recipes and commands of docs/guides/analysis/train-a-classifier.md, read from the page.

A recipe or a command that a guide shows must run as written. The tests that run them
take the blocks from the page itself, so that a recipe that no longer validates, plans
or runs fails here before a reader copies it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import PipelineRun, Recipe
from mosaic.core.scope import Scope
from tests.helpers.paths import REPO_ROOT
from tests.helpers.tracks import calms21_dataset

__all__ = [
    "GUIDE",
    "GUIDE_CLASSES",
    "guide_blocks",
    "guide_commands",
    "guide_dataset",
    "guide_recipe",
    "run_ids",
    "step_scopes",
    "step_states",
    "with_params",
]

GUIDE = REPO_ROOT / "docs" / "guides" / "analysis" / "train-a-classifier.md"
GUIDE_CLASSES = 4
"""The classes of the dataset the guide's recipes run on: they name ``default_class: 3``."""


def guide_blocks(language: str) -> list[str]:
    """Return the fenced blocks of *language* on the page, in order."""
    return re.findall(rf"```{language}\n(.*?)```", GUIDE.read_text(), re.S)


def guide_recipe(name: str) -> Recipe:
    """Return the recipe of the one JSON block whose ``name`` is *name*."""
    (block,) = [b for b in guide_blocks("json") if f'"name": "{name}"' in b]
    return Recipe.model_validate_json(block)


def guide_commands(recipe_file: str) -> list[str]:
    """Return the ``--step-scope`` commands that run ``@<recipe_file>``, in order."""
    return [
        block
        for block in guide_blocks("bash")
        if "--step-scope" in block and f"@{recipe_file}" in block
    ]


def step_scopes(command: str) -> dict[str, Scope]:
    """Return the scopes that a ``mosaic pipeline run`` command gives its steps."""
    found = re.findall(r"--step-scope '(\w+)=(\{.*?\})'", command)
    assert found, command
    return {step: Scope.model_validate(json.loads(scope)) for step, scope in found}


def with_params(
    recipe: Recipe, updates: Mapping[str, Mapping[str, JsonValue]]
) -> Recipe:
    """Return *recipe* with *updates* laid over the params of the steps they name.

    A guide's sizes suit a recording, and a test runs the same recipe on a few
    seconds of tracks.
    """
    steps = [
        step.model_copy(update={"params": {**step.params, **updates[step.id]}})
        if step.id in updates
        else step
        for step in recipe.steps
    ]
    return recipe.model_copy(update={"steps": steps})


def guide_dataset(base: Path) -> Dataset:
    """Return a CalMS21 dataset of six training sequences and two test sequences."""
    return calms21_dataset(
        base,
        {
            "train": {"a": {f"s{i}": 60 for i in range(1, 7)}},
            "test": {"a": {"t1": 60, "t2": 60}},
        },
        seed=5,
        n_classes=GUIDE_CLASSES,
    )


def step_states(run: PipelineRun) -> dict[str, str]:
    """Return what each step of *run* did: ``ran``, ``cached`` and so on."""
    return {outcome.step_id: outcome.state for outcome in run.outcomes}


def run_ids(run: PipelineRun) -> dict[str, str]:
    """Return the run id of each step of *run*."""
    return {outcome.step_id: str(outcome.run_id) for outcome in run.outcomes}
