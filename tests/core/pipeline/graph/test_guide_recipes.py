"""The recipes of docs/guides/analysis/train-a-classifier.md run as written.

The guide's JSON and shell blocks are read from the page itself, so a recipe that no
longer validates, plans or runs fails here before a reader copies it.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import PipelineRun, Recipe, run_pipeline
from mosaic.core.scope import Scope
from tests.helpers import (
    LA_WINDOW,
    REPO_ROOT,
    calms21_dataset,
    install_lightning_action_stub,
)

GUIDE = REPO_ROOT / "docs" / "guides" / "analysis" / "train-a-classifier.md"
N_CLASSES = 4  # the guide's recipe names `default_class: 3`


def _blocks(language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", GUIDE.read_text(), re.S)


def _step_scopes(command: str) -> dict[str, Scope]:
    """The scopes that a `mosaic pipeline run` command of the guide gives its steps."""
    found = re.findall(r"--step-scope '(\w+)=(\{.*?\})'", command)
    assert found, command
    return {step: Scope.model_validate(json.loads(scope)) for step, scope in found}


@pytest.fixture(scope="module")
def supervised() -> dict[str, Any]:
    (block,) = [b for b in _blocks("json") if '"name": "supervised"' in b]
    recipe: dict[str, Any] = json.loads(block)
    return recipe


@pytest.fixture(scope="module")
def sequence() -> dict[str, Any]:
    (block,) = [b for b in _blocks("json") if '"name": "sequence-classifier"' in b]
    recipe: dict[str, Any] = json.loads(block)
    return recipe


@pytest.fixture(scope="module")
def cross_validated(supervised: dict[str, Any]) -> dict[str, Any]:
    """The supervised recipe, with the two steps that the cross-validation section changes."""
    (block,) = [b for b in _blocks("json") if '"cv_folds"' in b]
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
    return [b for b in _blocks("bash") if "--step-scope" in b]


def _dataset(base: Path) -> Dataset:
    return calms21_dataset(
        base,
        {
            "train": {"a": {f"s{i}": 60 for i in range(1, 7)}},
            "test": {"a": {"t1": 60, "t2": 60}},
        },
        seed=5,
        n_classes=N_CLASSES,
    )


def _states(run: PipelineRun) -> dict[str, str]:
    return {outcome.step_id: outcome.state for outcome in run.outcomes}


def _run_ids(run: PipelineRun) -> dict[str, str]:
    return {outcome.step_id: str(outcome.run_id) for outcome in run.outcomes}


def test_the_supervised_recipe_trains_and_scores_as_written(
    tmp_path: Path, supervised: dict[str, Any], commands: list[str]
) -> None:
    run = run_pipeline(
        _dataset(tmp_path / "ds"),
        Recipe.model_validate(supervised),
        step_scopes=_step_scopes(commands[0]),
    )

    assert set(_states(run).values()) == {"ran"}
    assert "evaluate" in _states(run)


def test_the_cross_validation_commands_screen_and_then_score_the_test_groups(
    tmp_path: Path, cross_validated: dict[str, Any], commands: list[str]
) -> None:
    ds = _dataset(tmp_path / "ds")
    recipe = Recipe.model_validate(cross_validated)

    screening = run_pipeline(ds, recipe, step_scopes=_step_scopes(commands[1]))
    final = run_pipeline(ds, recipe, step_scopes=_step_scopes(commands[2]))

    assert set(_states(screening).values()) == {"ran"}
    assert _run_ids(final)["classifier"] == _run_ids(screening)["classifier"]
    assert _states(final)["evaluate"] == "ran"


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
        _dataset(tmp_path / "ds"),
        Recipe.model_validate(recipe),
        step_scopes=_step_scopes(command),
    )

    assert set(_states(run).values()) == {"ran"}
    # One resident-focal run of each of the six training sequences, and no test group.
    assert package.trainings == 1
    assert len(package.trained) == 6
