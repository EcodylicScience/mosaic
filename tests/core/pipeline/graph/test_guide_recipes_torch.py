"""The sequence-network recipes of docs/guides/analysis/train-a-classifier.md run as written.

`test_guide_recipes.py` runs the recipes that need no torch. These train real networks
on the CPU, so CI's `identity` job runs them. The guide's sizes suit a recording, and
the dataset here has eight sequences of two seconds, so the tests lay smaller ones over
the networks' params and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import Recipe, run_pipeline
from tests.helpers import (
    guide_commands,
    guide_dataset,
    guide_recipe,
    run_ids,
    step_scopes,
    step_states,
    with_params,
)

pytestmark = pytest.mark.identity

SMALL: dict[str, JsonValue] = {
    "device": "cpu",
    "steps": 20,
    "batch_size": 4,
    "crop_length": 30,
    "min_run": 8,
    "cv_folds": 3,
    "learning_rate": 0.02,
}
GRU: dict[str, JsonValue] = {**SMALL, "head_config": {"hidden": 8, "layers": 1}}
TCN: dict[str, JsonValue] = {**SMALL, "head_config": {"hidden": 8, "blocks": 2}}
NETWORKS = ("gru_1", "gru_2", "tcn_1", "tcn_2")


def _network() -> Recipe:
    return with_params(guide_recipe("sequence-network"), {"classifier": GRU})


def _blend() -> Recipe:
    return with_params(
        guide_recipe("blend"),
        {"gru_1": GRU, "gru_2": GRU, "tcn_1": TCN, "tcn_2": TCN},
    )


def test_the_network_commands_screen_and_then_score_the_test_groups(
    tmp_path: Path,
) -> None:
    ds = guide_dataset(tmp_path / "ds")
    screening_command, final_command = guide_commands("network.json")

    screening = run_pipeline(ds, _network(), step_scopes=step_scopes(screening_command))
    final = run_pipeline(ds, _network(), step_scopes=step_scopes(final_command))

    assert set(step_states(screening).values()) == {"ran"}
    # The classifier fits from its own inputs, so its scope is part of its run id.
    assert run_ids(final)["classifier"] != run_ids(screening)["classifier"]
    assert step_states(final)["classifier"] == "ran"
    assert step_states(final)["evaluate"] == "ran"


def test_the_blend_commands_screen_and_then_score_the_test_groups(
    tmp_path: Path,
) -> None:
    ds = guide_dataset(tmp_path / "ds")
    screening_command, final_command = guide_commands("blend.json")

    screening = run_pipeline(ds, _blend(), step_scopes=step_scopes(screening_command))
    final = run_pipeline(ds, _blend(), step_scopes=step_scopes(final_command))

    assert set(step_states(screening).values()) == {"ran"}
    assert {
        step_states(final)[step] for step in (*NETWORKS, "gru", "tcn", "blend")
    } == {"ran"}
    assert step_states(final)["evaluate"] == "ran"
    assert run_ids(final)["blend"] != run_ids(screening)["blend"]
