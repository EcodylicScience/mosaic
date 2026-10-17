"""``mosaic pipeline`` with ``--cleanup superseded``: plan, submit, run and status."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import CleanupPolicy, PlanWire
from mosaic.core.pipeline.index import feature_run_root
from tests.helpers import (
    add_tracks_variant,
    invoke_json,
    make_dataset,
    run_speed,
    speed_recipe_stepping,
)

VARIANT = "convert-trex.0.2-1111111111"

STORAGE = "speed-angvel__from__tracks"

LOOSE = ["--cleanup", "superseded", "--keep-newest", "0", "--min-age-hours", "0"]

runner = CliRunner()


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


@pytest.fixture
def recipe(tmp_path: Path) -> str:
    path = tmp_path / "speed.recipe.json"
    _ = path.write_text(speed_recipe_stepping(2).model_dump_json())
    return f"@{path}"


def _manifest(ds: Dataset) -> list[str]:
    return ["--manifest", str(ds.manifest_path)]


def test_plan_json_reports_superseded_runs(dataset: Dataset, recipe: str) -> None:
    old = run_speed(dataset, 1)

    plan = PlanWire.model_validate(
        invoke_json(
            [
                "pipeline",
                "plan",
                "--recipe",
                recipe,
                *_manifest(dataset),
                *LOOSE,
                "--json",
            ]
        )
    )
    plain = PlanWire.model_validate(
        invoke_json(
            ["pipeline", "plan", "--recipe", recipe, *_manifest(dataset), "--json"]
        )
    )

    [step] = plan.steps
    assert [run.run_id for run in step.superseded] == [old]
    assert plan.cleanup is not None
    assert plan.cleanup.policy == CleanupPolicy(keep_newest=0, min_age_hours=0)
    assert (plan.cleanup.removable_runs, plan.cleanup.kept_runs) == (1, 0)
    assert plain.cleanup is None
    assert plan.fingerprint != plain.fingerprint


def test_plan_prints_the_superseded_runs(dataset: Dataset, recipe: str) -> None:
    old = run_speed(dataset, 1)

    result = runner.invoke(
        app, ["pipeline", "plan", "--recipe", recipe, *_manifest(dataset), *LOOSE]
    )

    assert result.exit_code == 0, result.output
    assert "superseded runs" in result.stdout
    assert old in result.stdout
    assert "1 superseded run(s) removable" in result.stderr


def test_submit_cleanup_without_expect_plan_fails(
    dataset: Dataset, recipe: str
) -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            "submit",
            "--recipe",
            recipe,
            *_manifest(dataset),
            *LOOSE,
            "--json",
        ],
    )

    assert result.exit_code == 1
    envelope = json.loads(result.stdout)
    assert envelope["reason"] == "cleanup_needs_plan"
    assert not (dataset.base_dir / ".mosaic" / "pipelines" / "requests").exists()


def test_run_with_cleanup_removes_superseded_runs_and_status_reports_them(
    dataset: Dataset, recipe: str
) -> None:
    old = run_speed(dataset, 1)
    plan = invoke_json(
        ["pipeline", "plan", "--recipe", recipe, *_manifest(dataset), *LOOSE, "--json"]
    )

    ran = invoke_json(
        [
            "pipeline",
            "run",
            "--recipe",
            recipe,
            *_manifest(dataset),
            *LOOSE,
            "--expect-plan",
            str(plan["fingerprint"]),
            "--json",
        ]
    )
    status = invoke_json(
        [
            "pipeline",
            "status",
            "--request",
            str(ran["request_id"]),
            *_manifest(dataset),
            "--json",
        ]
    )

    assert not feature_run_root(dataset, STORAGE, old).exists()
    cleanup = ran["cleanup"]
    assert isinstance(cleanup, dict)
    assert (cleanup["status"], cleanup["removed_runs"]) == ("finished", 1)
    assert status["cleanup"] == cleanup
    assert status["verdict"] == "finished"


def test_run_refuses_cleanup_flags_with_a_request(dataset: Dataset) -> None:
    result = runner.invoke(
        app,
        ["pipeline", "run", "--request", "R1", *_manifest(dataset), *LOOSE],
    )

    assert result.exit_code != 0
    assert "--cleanup" in result.output
