"""``mosaic pipeline verify``: the exit status and the ``--json`` report of a verification.

The library's cases are in ``tests/core/pipeline/test_verify_plan.py``. These
check what the command adds: the plan options it shares with ``pipeline plan``,
the exit code, the refusals, and that it leaves the dataset alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import Recipe, plan_pipeline, run_pipeline
from mosaic.core.scope import Scope
from tests.helpers import (
    assert_tree_unchanged,
    calms21_dataset,
    classifier_recipe,
    tree_snapshot,
)

runner = CliRunner()

SCOPES = {"labeled": Scope(groups=["train"])}


def _recipe() -> Recipe:
    base = classifier_recipe("clf").model_dump(mode="json")
    evaluation = {
        "id": "score",
        "type": "feature",
        "feature": "evaluate-labels",
        "inputs": [{"step": "clf"}],
        "params": {"labels": {"step": "labels"}, "perspective": 0},
    }
    return Recipe.model_validate({**base, "steps": [*base["steps"], evaluation]})


@pytest.fixture
def study(tmp_path: Path) -> Dataset:
    ds = calms21_dataset(
        tmp_path / "study",
        {"train": {"a": {"s1": 90, "s2": 90}}, "other": {"a": {"s3": 90}}},
        seed=11,
        n_classes=2,
    )
    _ = run_pipeline(ds, _recipe(), step_scopes=SCOPES)
    return ds


@pytest.fixture
def recipe_file(tmp_path: Path) -> str:
    path = tmp_path / "recipe.json"
    _ = path.write_text(_recipe().model_dump_json())
    return f"@{path}"


def _verify(study: Dataset, recipe_file: str, *flags: str) -> Any:
    return runner.invoke(
        app,
        [
            "pipeline",
            "verify",
            "--manifest",
            str(study.manifest_path),
            "--recipe",
            recipe_file,
            "--step-scope",
            'labeled={"groups": ["train"]}',
            *flags,
        ],
    )


def test_a_matching_recipe_exits_zero_and_reports_every_step(
    study: Dataset, recipe_file: str
) -> None:
    result = _verify(study, recipe_file, "--json")

    assert result.exit_code == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["atol"] == 1e-5
    assert report["n_threads"] == 1
    by_step = {step["step"]: step for step in report["steps"]}
    assert by_step["clf"]["status"] == "ok"
    assert by_step["clf"]["argmax_agreement"] == 1.0
    assert by_step["score"]["role"] == "evaluation"


def test_a_changed_output_exits_one_and_the_text_report_names_the_entry(
    study: Dataset, recipe_file: str
) -> None:
    planned = plan_pipeline(study, _recipe(), step_scopes=SCOPES).step("clf")
    root = study.get_root("features") / planned.storage_name / str(planned.run_id)
    [path] = sorted(root.glob("train__s1*.parquet"))
    stored = pd.read_parquet(path)
    stored["prob_1"] = stored["prob_1"] + 0.5
    stored.to_parquet(path)

    result = _verify(study, recipe_file)

    assert result.exit_code == 1
    assert "differences found" in result.stderr
    assert "train/s1 differs" in result.stderr
    assert "clf\txgboost\tfit\tdiffers" in result.stdout


def test_a_step_scope_that_the_run_did_not_use_verifies_nothing_and_exits_one(
    study: Dataset, recipe_file: str
) -> None:
    """Without the step scope the plan is another run, which has not been made."""
    result = runner.invoke(
        app,
        [
            "pipeline",
            "verify",
            "--manifest",
            str(study.manifest_path),
            "--recipe",
            recipe_file,
            "--json",
        ],
    )

    assert result.exit_code == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert report["verified"] == 0
    by_step = {step["step"]: step for step in report["steps"]}
    assert by_step["labeled"]["status"] == "skipped"
    assert "no complete run" in by_step["labeled"]["reason"]


def test_a_scratch_directory_inside_the_dataset_exits_one(
    study: Dataset, recipe_file: str
) -> None:
    result = _verify(study, recipe_file, "--scratch", str(study.base_dir / "scratch"))

    assert result.exit_code == 1
    assert "inside the dataset" in result.stderr


def test_verifying_writes_nothing_under_the_dataset(
    study: Dataset, recipe_file: str
) -> None:
    before = tree_snapshot(study.base_dir)

    result = _verify(study, recipe_file, "--json")

    assert result.exit_code == 0, result.stderr
    assert_tree_unchanged(study.base_dir, before)
