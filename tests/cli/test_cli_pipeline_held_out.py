"""``mosaic pipeline plan`` on a dataset with roles: the guard and ``--assume-added``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import PlanWire
from tests.helpers import calms21_dataset, invoke_json

runner = CliRunner()

RECIPE = {
    "schema_version": 1,
    "name": "held-out",
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


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": {"s1": 40, "s2": 30}}, "test": {"a": {"s3": 30}}},
        seed=5,
    )
    _ = ds.convert_all_tracks()
    ds.set_splits("train", ["train"])
    ds.set_splits("test", ["test"])
    return ds


@pytest.fixture
def recipe(tmp_path: Path) -> str:
    path = tmp_path / "held-out.recipe.json"
    _ = path.write_text(json.dumps(RECIPE))
    return f"@{path}"


def _plan(dataset: Dataset, recipe: str, *flags: str) -> list[str]:
    return [
        "pipeline",
        "plan",
        "--recipe",
        recipe,
        "--manifest",
        str(dataset.manifest_path),
        *flags,
    ]


def test_a_fit_over_the_test_group_exits_one_and_names_the_group(
    dataset: Dataset, recipe: str
) -> None:
    result = runner.invoke(app, _plan(dataset, recipe))

    assert result.exit_code == 1
    assert "templates" in result.output
    assert "held-out group(s) 'test' (test)" in result.output


def test_the_same_fit_scoped_to_the_train_group_plans(
    dataset: Dataset, recipe: str
) -> None:
    scoped = ["--step-scope", 'templates={"groups": ["train"]}']

    plan = PlanWire.model_validate(
        invoke_json(_plan(dataset, recipe, *scoped, "--json"))
    )

    assert plan.problems == ()


def test_assume_added_notes_the_refits_in_the_cautions(
    dataset: Dataset, recipe: str
) -> None:
    scoped = ["--step-scope", 'templates={"groups": ["train"]}']
    plan = PlanWire.model_validate(
        invoke_json(
            _plan(
                dataset,
                recipe,
                *scoped,
                "--assume-added",
                "train2",
                "--assume-added",
                "elsewhere",
                "--json",
            )
        )
    )
    unassumed = PlanWire.model_validate(
        invoke_json(_plan(dataset, recipe, *scoped, "--json"))
    )

    [caution] = [c for c in plan.cautions if c.where == "assume_added"]
    assert caution.message == (
        "if groups 'train2', 'elsewhere' joined, no step's run would change"
    )
    assert plan.fingerprint == unassumed.fingerprint
    assert plan.steps == unassumed.steps


def test_assume_added_names_a_refit_in_the_plain_output(
    dataset: Dataset, recipe: str
) -> None:
    # The request names the group it assumes, so the fit would take it.
    result = runner.invoke(
        app,
        _plan(
            dataset,
            recipe,
            "--group",
            "train",
            "--group",
            "later",
            "--assume-added",
            "later",
        ),
    )

    assert result.exit_code == 0, result.output
    assert (
        "caution: assume_added: if group 'later' joined, these steps "
        "would refit: templates" in result.stderr
    )


def test_a_group_already_in_the_plan_is_reported_and_left_out(
    dataset: Dataset, recipe: str
) -> None:
    result = runner.invoke(
        app, _plan(dataset, recipe, "--group", "train", "--assume-added", "train")
    )

    assert result.exit_code == 0, result.output
    assert "already in this plan's scope" in result.stderr
