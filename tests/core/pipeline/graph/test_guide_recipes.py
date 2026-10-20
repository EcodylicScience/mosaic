"""The recipes of docs/guides/analysis/train-a-classifier.md run as written.

The guide's JSON and shell blocks are read from the page itself, so a recipe that no
longer validates, plans or runs fails here before a reader copies it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from mosaic.core.pipeline.graph import Recipe, run_pipeline
from tests.helpers import (
    LA_WINDOW,
    guide_blocks,
    guide_dataset,
    install_lightning_action_stub,
    run_ids,
    step_scopes,
    step_states,
)


@pytest.fixture(scope="module")
def supervised() -> dict[str, Any]:
    (block,) = [b for b in guide_blocks("json") if '"name": "supervised"' in b]
    recipe: dict[str, Any] = json.loads(block)
    return recipe


@pytest.fixture(scope="module")
def sequence() -> dict[str, Any]:
    (block,) = [
        b for b in guide_blocks("json") if '"name": "lightning-action-sequence"' in b
    ]
    recipe: dict[str, Any] = json.loads(block)
    return recipe


@pytest.fixture(scope="module")
def cross_validated(supervised: dict[str, Any]) -> dict[str, Any]:
    """The supervised recipe, with the two steps that the cross-validation section changes."""
    (block,) = [
        b for b in guide_blocks("json") if '"xgboost"' in b and '"cv_folds"' in b
    ]
    changed: dict[str, dict[str, Any]] = {
        step["id"]: step for step in json.loads(f"[{block}]")
    }
    steps: list[dict[str, Any]] = supervised["steps"]
    return {
        **supervised,
        "name": "cross-validated",
        "steps": [changed.get(step["id"], step) for step in steps],
    }


@pytest.fixture(scope="module")
def commands() -> list[str]:
    return [b for b in guide_blocks("bash") if "--step-scope" in b]


def test_the_supervised_recipe_trains_and_scores_as_written(
    tmp_path: Path, supervised: dict[str, Any], commands: list[str]
) -> None:
    run = run_pipeline(
        guide_dataset(tmp_path / "ds"),
        Recipe.model_validate(supervised),
        step_scopes=step_scopes(commands[0]),
    )

    assert set(step_states(run).values()) == {"ran"}
    assert "evaluate" in step_states(run)


def test_the_cross_validation_commands_screen_and_then_score_the_test_groups(
    tmp_path: Path, cross_validated: dict[str, Any], commands: list[str]
) -> None:
    ds = guide_dataset(tmp_path / "ds")
    recipe = Recipe.model_validate(cross_validated)

    screening = run_pipeline(ds, recipe, step_scopes=step_scopes(commands[1]))
    final = run_pipeline(ds, recipe, step_scopes=step_scopes(commands[2]))

    assert set(step_states(screening).values()) == {"ran"}
    assert run_ids(final)["classifier"] == run_ids(screening)["classifier"]
    assert step_states(final)["evaluate"] == "ran"


def test_the_sequence_recipe_trains_on_the_train_group_and_predicts_every_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sequence: dict[str, Any],
    commands: list[str],
) -> None:
    package = install_lightning_action_stub(monkeypatch)
    recipe = copy.deepcopy(sequence)
    (classifier,) = [s for s in recipe["steps"] if s["id"] == "classifier"]
    # The guide's sizes suit a recording, not the two minutes of the fixture. The
    # stub trains on windows of its own length.
    classifier["params"].update(
        sequence_length=LA_WINDOW,
        num_epochs=1,
        num_layers=1,
        num_lags=1,
        device="cpu",
    )
    (command,) = [c for c in commands if "@sequence.json" in c]

    run = run_pipeline(
        guide_dataset(tmp_path / "ds"),
        Recipe.model_validate(recipe),
        step_scopes=step_scopes(command),
    )

    assert set(step_states(run).values()) == {"ran"}
    # One resident-focal run of each of the six training sequences, and no test group.
    assert package.trainings == 1
    assert len(package.trained) == 6
