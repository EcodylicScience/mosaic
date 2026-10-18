"""``mosaic pipeline`` with ``--cleanup superseded``: plan, submit, run and status."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    CleanupPolicy,
    CleanupStatusWire,
    PlanWire,
    run_pipeline,
)
from mosaic.core.pipeline.index import feature_run_root
from tests.helpers import (
    add_tracks_variant,
    invoke_json,
    make_dataset,
    run_id_of,
    run_speed,
    speed_heading_recipe,
    speed_recipe_stepping,
)

VARIANT = "convert-trex.0.2-1111111111"

STORAGE = "speed-angvel__from__tracks"

HEADING = "heading__from__tracks"

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


@pytest.mark.parametrize("flag", ["--cleanup", "--retire"])
def test_run_refuses_cleanup_flags_with_a_request(dataset: Dataset, flag: str) -> None:
    flags = LOOSE if flag == "--cleanup" else ["--retire", "0123456789abcdef"]
    result = runner.invoke(
        app,
        ["pipeline", "run", "--request", "R1", *_manifest(dataset), *flags],
    )

    assert result.exit_code != 0
    assert flag in result.output


def test_run_cleanup_without_expect_plan_fails_and_removes_nothing(
    dataset: Dataset, recipe: str
) -> None:
    old = run_speed(dataset, 1)

    result = runner.invoke(
        app,
        ["pipeline", "run", "--recipe", recipe, *_manifest(dataset), *LOOSE, "--json"],
    )

    assert result.exit_code == 1
    assert json.loads(result.stdout)["reason"] == "cleanup_needs_plan"
    assert feature_run_root(dataset, STORAGE, old).is_dir()
    assert not (dataset.base_dir / ".mosaic" / "pipelines" / "requests").exists()


# --- retiring a recipe ----------------------------------------------------------------


def _plan(dataset: Dataset, recipe: str, *extra: str) -> dict[str, object]:
    return invoke_json(
        [
            "pipeline",
            "plan",
            "--recipe",
            recipe,
            *_manifest(dataset),
            *LOOSE,
            *extra,
            "--json",
        ]
    )


def test_plan_lists_the_recipes_whose_runs_can_be_retired(
    dataset: Dataset, recipe: str
) -> None:
    ran = run_pipeline(dataset, speed_heading_recipe())

    plan = PlanWire.model_validate(_plan(dataset, recipe))
    plain = PlanWire.model_validate(
        invoke_json(
            ["pipeline", "plan", "--recipe", recipe, *_manifest(dataset), "--json"]
        )
    )
    text = runner.invoke(
        app, ["pipeline", "plan", "--recipe", recipe, *_manifest(dataset), *LOOSE]
    )

    [row] = plan.retirable
    assert (row.digest, row.name, row.runs, row.open) == (
        ran.recipe_digest,
        "old",
        2,
        False,
    )
    assert plain.retirable == ()
    assert "retirable recipes (pass a digest to --retire)" in text.stdout
    assert ran.recipe_digest in text.stdout


def test_plan_with_retire_lists_the_retired_runs(
    dataset: Dataset, recipe: str, lapsed_request_holds: None
) -> None:
    ran = run_pipeline(dataset, speed_heading_recipe())

    plan = PlanWire.model_validate(
        _plan(dataset, recipe, "--retire", ran.recipe_digest)
    )
    text = runner.invoke(
        app,
        [
            "pipeline",
            "plan",
            "--recipe",
            recipe,
            *_manifest(dataset),
            *LOOSE,
            "--retire",
            ran.recipe_digest,
        ],
    )

    assert plan.cleanup is not None
    assert plan.cleanup.retired == (ran.recipe_digest,)
    assert [run.storage for run in plan.cleanup.retired_runs] == [HEADING]
    assert plan.cleanup.removable_runs == 2
    assert "retired runs" in text.stdout
    assert HEADING in text.stdout


def test_run_retires_the_recipe_and_status_reports_the_runs(
    dataset: Dataset, recipe: str, lapsed_request_holds: None
) -> None:
    ran = run_pipeline(dataset, speed_heading_recipe())
    heading = run_id_of(ran, "heading")
    retire = ["--retire", ran.recipe_digest]
    plan = _plan(dataset, recipe, *retire)

    done = invoke_json(
        [
            "pipeline",
            "run",
            "--recipe",
            recipe,
            *_manifest(dataset),
            *LOOSE,
            *retire,
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
            str(done["request_id"]),
            *_manifest(dataset),
            "--json",
        ]
    )

    assert not feature_run_root(dataset, HEADING, heading).exists()
    cleanup = done["cleanup"]
    assert isinstance(cleanup, dict)
    assert (cleanup["status"], cleanup["removed_runs"]) == ("finished", 2)
    assert status["cleanup"] == cleanup
    report = CleanupStatusWire.model_validate(cleanup)
    assert {run.retired for run in report.removed} == {ran.recipe_digest}


def test_submit_records_the_retirement(
    dataset: Dataset, recipe: str, lapsed_request_holds: None
) -> None:
    ran = run_pipeline(dataset, speed_heading_recipe())
    retire = ["--retire", ran.recipe_digest]
    plan = _plan(dataset, recipe, *retire)

    submitted = invoke_json(
        [
            "pipeline",
            "submit",
            "--recipe",
            recipe,
            *_manifest(dataset),
            *LOOSE,
            *retire,
            "--expect-plan",
            str(plan["fingerprint"]),
            "--json",
        ]
    )

    cleanup = submitted["cleanup"]
    assert isinstance(cleanup, dict)
    assert cleanup["retired"] == [ran.recipe_digest]


@pytest.mark.parametrize("verb", ["plan", "submit", "run"])
def test_retire_without_a_cleanup_fails_and_writes_nothing(
    dataset: Dataset, recipe: str, verb: str
) -> None:
    ran = run_pipeline(dataset, speed_heading_recipe())
    before = sorted((dataset.base_dir / ".mosaic").rglob("*"))

    result = runner.invoke(
        app,
        [
            "pipeline",
            verb,
            "--recipe",
            recipe,
            *_manifest(dataset),
            "--retire",
            ran.recipe_digest,
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.stdout)["reason"] == "retire_needs_cleanup"
    assert sorted((dataset.base_dir / ".mosaic").rglob("*")) == before


def test_retiring_a_recipe_the_dataset_does_not_hold_fails(
    dataset: Dataset, recipe: str
) -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            "plan",
            "--recipe",
            recipe,
            *_manifest(dataset),
            *LOOSE,
            "--retire",
            "0123456789abcdef",
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.stdout)["reason"] == "retire_unknown_recipe"
