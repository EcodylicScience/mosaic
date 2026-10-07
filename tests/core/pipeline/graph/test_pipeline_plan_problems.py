"""The problems a dataset finds with a recipe that validates.

Validation decides from declarations alone. A time filter is valid on any
dataset there, and a column that the track schemas allow is only a caution. A plan reads
the dataset. A time filter without a frame rate, and a required column that every
tracks table lacks, are problems there, each at the place in the recipe that would
change. ``plan_pipeline`` raises them together as ``RecipeRefused``, and a step
that finds one of its own when it re-plans itself is refused and recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    FeatureStepSpec,
    Recipe,
    RecipeInvalid,
    RecipeRefused,
    RecipeRefusedAtStep,
    Request,
    execute_step,
    plan_pipeline,
    plan_wire,
    read_recipe,
    request_rollup,
    submit_request,
)
from mosaic.core.pipeline.graph.columns import measured_columns
from mosaic.core.pipeline.refusal import REFUSED_EXIT_CODE
from mosaic.runlog import read_run, run_log_dir
from tests.helpers import add_tracks_variant, guppies_recipe, make_dataset

runner = CliRunner()

VARIANT = "convert-trex.0.2-1111111111"

FILTER_PROBLEM = "a time filter needs the dataset's frame rate"


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two ``mosaic_v1`` tables with keypoints and no ``ANGLE``, and no frame rate."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return dataset


def _delta(*, with_heading: bool) -> Recipe:
    """``nn-delta-response`` over smoothed tracks, which require ``ANGLE``.

    ``heading`` writes ``ANGLE`` and the identity columns. It joins the merge as
    one more input when *with_heading* is set.
    """
    heading: list[dict[str, JsonValue]] = (
        [{"id": "heading", "type": "feature", "feature": "heading"}]
        if with_heading
        else []
    )
    extra: list[JsonValue] = [{"step": "heading"}] if with_heading else []
    return read_recipe(
        {
            "steps": [
                *heading,
                {"id": "smooth", "type": "feature", "feature": "trajectory-smooth"},
                {
                    "id": "nn",
                    "type": "feature",
                    "feature": "nearest-neighbor",
                    "inputs": [{"step": "smooth"}],
                },
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": [{"step": "smooth"}],
                },
                {
                    "id": "delta",
                    "type": "feature",
                    "feature": "nn-delta-response",
                    "inputs": [
                        {"step": "smooth"},
                        {"step": "nn"},
                        {"step": "speed"},
                        *extra,
                    ],
                },
            ]
        }
    )


def _refused(dataset: Dataset, recipe: Recipe) -> RecipeRefused:
    with pytest.raises(RecipeRefused) as caught:
        _ = plan_pipeline(dataset, recipe)
    return caught.value


# --- a time filter without a frame rate --------------------------------------------


def test_a_time_filter_without_a_frame_rate_is_a_problem_at_the_filter(
    tracked: Dataset,
) -> None:
    recipe = guppies_recipe()
    filtered = {
        step.id
        for step in recipe.steps
        if isinstance(step, FeatureStepSpec) and step.run.filter_start_time is not None
    }

    refusal = _refused(tracked, recipe)

    at_filter = [p for p in refusal.problems if p.where == "run.filter_start_time"]
    assert {problem.step for problem in at_filter} == filtered
    assert all(FILTER_PROBLEM in problem.message for problem in at_filter)
    assert all("meta.fps_default" in problem.message for problem in at_filter)
    # Each filtered step is reported, and none of them has an identity.
    assert {
        step for step, run_id in refusal.plan.run_ids.items() if run_id is None
    } >= (filtered)


def test_a_refusal_is_a_recipe_refusal_and_lists_every_problem(
    tracked: Dataset,
) -> None:
    refusal = _refused(tracked, guppies_recipe())

    assert isinstance(refusal, RecipeInvalid)
    assert str(refusal).startswith("this recipe cannot be run on this dataset: ")
    assert f"{len(refusal.problems)} problems" in str(refusal)


def test_a_frame_rate_clears_the_filter_problems(tracked: Dataset) -> None:
    tracked.meta["fps_default"] = 30.0

    refusal = _refused(tracked, guppies_recipe())

    assert {problem.where for problem in refusal.problems} == {"inputs"}


def test_a_frame_and_a_time_on_one_boundary_are_refused_at_parse() -> None:
    document: dict[str, JsonValue] = {
        "steps": [
            {
                "id": "speed",
                "type": "feature",
                "feature": "speed-angvel",
                "run": {"filter_start_frame": 10, "filter_start_time": 1.0},
            }
        ]
    }

    with pytest.raises(RecipeInvalid) as caught:
        _ = read_recipe(document)

    (problem,) = caught.value.problems
    assert (problem.step, problem.where) == ("speed", "run")
    assert "one of filter_start_frame and filter_start_time" in problem.message


# --- a column that the tracks tables lack ------------------------------------------


def test_a_column_every_table_lacks_is_a_problem_naming_its_provider(
    tracked: Dataset,
) -> None:
    refusal = _refused(tracked, _delta(with_heading=False))

    (problem,) = refusal.problems
    assert (problem.step, problem.where) == ("delta", "inputs")
    assert "needs 'ANGLE'" in problem.message
    assert "heading provides it" in problem.message


def test_the_column_s_provider_as_an_input_plans(tracked: Dataset) -> None:
    plan = plan_pipeline(tracked, _delta(with_heading=True))

    assert plan.problems == ()
    assert all(run_id is not None for run_id in plan.run_ids.values())


def test_the_tables_decide_where_the_schemas_allow(tracked: Dataset) -> None:
    """Validation knows only what a schema allows, so ``ANGLE`` is a caution there."""
    from mosaic.core.pipeline.graph import validation_report

    report = validation_report(_delta(with_heading=False).model_dump(mode="json"))

    assert report.valid
    assert [(caution.step, caution.where) for caution in report.cautions] == [
        ("delta", "inputs")
    ]


def test_measured_columns_guarantee_what_every_table_holds() -> None:
    columns = measured_columns([("frame", "X", "ANGLE"), ("frame", "X")])

    assert columns is not None
    assert columns.guaranteed == {"frame", "X"}
    assert columns.possible == {"frame", "X", "ANGLE"}
    assert not columns.admits("Y")
    assert measured_columns([]) is None


# --- the plan, the wire and the CLI ------------------------------------------------


def test_an_unrefused_plan_carries_the_problems_onto_the_wire(
    tracked: Dataset,
) -> None:
    plan = plan_pipeline(tracked, _delta(with_heading=False), refuse=False)

    wire = plan_wire(plan)

    assert len(plan.problems) == 1
    assert wire.problems == plan.problems
    assert plan_wire(plan_pipeline(tracked, _delta(with_heading=True))).problems == ()


def test_the_cli_plan_prints_the_refused_plan_and_exits_one(
    tracked: Dataset, tmp_path: Path
) -> None:
    recipe_path = tmp_path / "recipe.json"
    _ = recipe_path.write_text(
        json.dumps(_delta(with_heading=False).model_dump(mode="json"))
    )

    result = runner.invoke(
        app,
        [
            "pipeline",
            "plan",
            "--recipe",
            f"@{recipe_path}",
            "--manifest",
            str(tracked.manifest_path),
            "--json",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert [(found["step"], found["where"]) for found in payload["problems"]] == [
        ("delta", "inputs")
    ]


# --- a step that runs ----------------------------------------------------------------


def _losing_its_frame_rate(dataset: Dataset) -> Request:
    """A request submitted with a frame rate, on a dataset that then loses it.

    ``speed`` has no time filter and ``late`` has one.
    """
    dataset.meta["fps_default"] = 30.0
    recipe = read_recipe(
        {
            "steps": [
                {"id": "speed", "type": "feature", "feature": "speed-angvel"},
                {
                    "id": "late",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "run": {"filter_start_time": 0.5},
                },
            ]
        }
    )
    request = submit_request(dataset, recipe).request
    del dataset.meta["fps_default"]
    return request


def test_a_running_step_refuses_only_its_own_problem(tracked: Dataset) -> None:
    """The dataset can change between a submission and a step's attempt."""
    request = _losing_its_frame_rate(tracked)

    ran = execute_step(tracked, request, "speed")
    with pytest.raises(RecipeRefusedAtStep) as caught:
        _ = execute_step(tracked, request, "late")

    assert ran.run_id is not None
    assert caught.value.reason == "recipe_refused"
    assert [(p.step, p.where) for p in caught.value.problems] == [
        ("late", "run.filter_start_time")
    ]


def test_a_refused_running_step_records_its_attempt(tracked: Dataset) -> None:
    """A refusal with no run-log is a step, and a request, that reads as running."""
    request = _losing_its_frame_rate(tracked)
    _ = execute_step(tracked, request, "speed")

    with pytest.raises(RecipeRefusedAtStep):
        _ = execute_step(tracked, request, "late")

    logged = read_run(run_log_dir(tracked.base_dir), request.execution_of("late"))
    assert logged is not None
    assert logged["status"] == "failed"
    recorded = json.loads(logged["error_json"])
    assert recorded["reason"] == "recipe_refused"
    assert recorded["step"] == "late"
    assert [(p["step"], p["where"]) for p in recorded["problems"]] == [
        ("late", "run.filter_start_time")
    ]
    assert request_rollup(tracked.base_dir, request).status == "failed"


def test_the_cli_exits_with_the_refused_code_for_a_refused_step(
    tracked: Dataset,
) -> None:
    request = _losing_its_frame_rate(tracked)

    result = runner.invoke(
        app,
        [
            "run",
            "--json",
            "--manifest",
            str(tracked.manifest_path),
            "--graph-request",
            request.request_id,
            "--step",
            "late",
            "--execution-id",
            request.execution_of("late"),
        ],
    )

    assert result.exit_code == REFUSED_EXIT_CODE, result.output
    payload = json.loads(result.stdout)
    assert (payload["status"], payload["reason"]) == ("refused", "recipe_refused")
