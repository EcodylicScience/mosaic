"""A request narrows single steps: the train/test split as one request.

``Request.step_scopes`` narrows a step to part of what the request covers. The
step's target is the request's narrowed by its scope, its coverage and its
identity are measured over that target, and a step reading another's entries
through ``inputs`` or ``tracks`` must be planned over a subset of them. A params
reference carries no such rule, so a test-scoped step reads what a train-scoped
step fitted.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    BoundRef,
    CoverageShortfall,
    Plan,
    Recipe,
    RecipeRefused,
    Request,
    plan_fingerprint,
    plan_pipeline,
    plan_wire,
    request_path,
    run_pipeline,
    submit_request,
)
from mosaic.core.pipeline.graph.step_preflight import refuse_shortfall
from mosaic.core.pipeline.index import feature_index_path
from mosaic.core.scope import Scope
from tests.helpers import calms21_dataset

TRAIN = Scope(groups=["train"])
TEST = Scope(groups=["test"])

SPEED: dict[str, JsonValue] = {
    "id": "speed",
    "type": "feature",
    "feature": "speed-angvel",
    "inputs": ["tracks"],
}
TEMPLATES: dict[str, JsonValue] = {
    "id": "templates",
    "type": "feature",
    "feature": "extract-templates",
    "inputs": [{"step": "speed"}],
    "params": {"n_templates": 4},
}
KMEANS: dict[str, JsonValue] = {
    "id": "kmeans",
    "type": "feature",
    "feature": "global-kmeans",
    "inputs": [{"step": "speed"}],
    "params": {"templates": {"step": "templates"}, "k": 2},
}


def _recipe(*steps: dict[str, JsonValue]) -> Recipe:
    return Recipe.model_validate(
        {"schema_version": 1, "name": "split", "steps": list(steps)}
    )


@pytest.fixture
def split(tmp_path: Path) -> Dataset:
    """Converted tracks of a train group of two sequences and a test group of one."""
    ds = calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": {"s1": 40, "s2": 30}}, "test": {"a": {"s3": 30}}},
        seed=5,
    )
    _ = ds.convert_all_tracks()
    return ds


def _where(plan: Plan) -> list[tuple[str, str]]:
    return [(problem.step, problem.where) for problem in plan.problems]


# --- planning ---------------------------------------------------------------


def test_a_step_scope_narrows_only_its_step(split: Dataset) -> None:
    plan = plan_pipeline(
        split, _recipe(SPEED, TEMPLATES, KMEANS), step_scopes={"templates": TRAIN}
    )

    assert plan.problems == ()
    assert len(plan.scope) == 3
    assert plan.step("templates").coverage.target == {("train", "s1"), ("train", "s2")}
    assert len(plan.step("speed").coverage.target) == 3
    assert len(plan.step("kmeans").coverage.target) == 3
    wire = plan_wire(plan)
    narrowed = {step.step: step.step_scope for step in wire.steps}
    assert narrowed == {"speed": None, "templates": TRAIN, "kmeans": None}


def test_a_scope_dependent_step_hashes_its_step_scope(split: Dataset) -> None:
    recipe = _recipe(SPEED, TEMPLATES)

    whole = plan_pipeline(split, recipe)
    trained = plan_pipeline(split, recipe, step_scopes={"templates": TRAIN})
    speed_trained = plan_pipeline(
        split, recipe, step_scopes={"speed": TRAIN, "templates": TRAIN}
    )

    assert whole.step("templates").run_id != trained.step("templates").run_id
    assert whole.step("speed").run_id == speed_trained.step("speed").run_id


def test_a_step_scope_naming_no_step_is_a_problem(split: Dataset) -> None:
    plan = plan_pipeline(
        split, _recipe(SPEED), step_scopes={"nope": TRAIN}, refuse=False
    )

    assert _where(plan) == [("", "step_scopes.nope")]
    with pytest.raises(RecipeRefused):
        _ = plan_pipeline(split, _recipe(SPEED), step_scopes={"nope": TRAIN})


def test_a_step_scope_on_an_unscoped_op_or_a_bound_step_is_a_problem(
    split: Dataset,
) -> None:
    converting = _recipe(
        {
            "id": "tracks",
            "type": "op",
            "kind": "convert-tracks",
            "params": {"source_format": "calms21_npy"},
        },
        SPEED,
    )
    op_plan = plan_pipeline(
        split, converting, step_scopes={"tracks": TRAIN}, refuse=False
    )
    request = Request(
        request_id="r",
        recipe_digest="d",
        bind={"speed": BoundRef(feature="speed-angvel", run_id="0.2-0123456789")},
        step_scopes={"speed": TRAIN},
    )
    bound_plan = plan_pipeline(split, _recipe(SPEED), request=request, refuse=False)

    assert _where(op_plan) == [("tracks", "step_scopes.tracks")]
    assert "takes no entry scope" in op_plan.problems[0].message
    assert _where(bound_plan) == [("speed", "step_scopes.speed")]


def test_a_step_scope_selecting_nothing_is_a_problem(split: Dataset) -> None:
    plan = plan_pipeline(
        split,
        _recipe(SPEED),
        step_scopes={"speed": Scope(groups=["absent"])},
        refuse=False,
    )

    assert _where(plan) == [("speed", "step_scopes.speed")]
    assert "selects no entry" in plan.problems[0].message


def test_an_input_narrower_than_its_reader_is_a_problem(split: Dataset) -> None:
    plan = plan_pipeline(
        split, _recipe(SPEED, TEMPLATES), step_scopes={"speed": TRAIN}, refuse=False
    )

    (problem,) = plan.problems
    assert (problem.step, problem.where) == ("templates", "inputs[0]")
    assert "1 entries of 'templates' are outside what 'speed'" in problem.message
    assert "('test', 's3')" in problem.message


def test_a_params_reference_to_a_narrower_step_is_allowed(split: Dataset) -> None:
    """A model fitted on the training entries is applied to every entry."""
    plan = plan_pipeline(
        split,
        _recipe(SPEED, TEMPLATES, KMEANS),
        step_scopes={"templates": TRAIN, "kmeans": TEST},
    )

    assert plan.problems == ()
    assert plan.step("kmeans").status_parents == ("templates",)
    assert plan.step("kmeans").coverage.target == {("test", "s3")}


# --- the fingerprint ----------------------------------------------------------


def test_step_scopes_change_the_fingerprint(split: Dataset) -> None:
    recipe = _recipe(SPEED, TEMPLATES)

    train = plan_pipeline(split, recipe, step_scopes={"templates": TRAIN})
    test = plan_pipeline(split, recipe, step_scopes={"templates": TEST})

    assert plan_fingerprint(train) != plan_fingerprint(test)


def test_the_order_a_step_scope_lists_its_groups_in_names_nothing(
    split: Dataset,
) -> None:
    recipe = _recipe(SPEED, TEMPLATES)

    one = plan_pipeline(
        split, recipe, step_scopes={"templates": Scope(groups=["train", "test"])}
    )
    other = plan_pipeline(
        split, recipe, step_scopes={"templates": Scope(groups=["test", "train"])}
    )

    assert plan_fingerprint(one) == plan_fingerprint(other)


def test_a_plan_without_step_scopes_keeps_its_fingerprint(split: Dataset) -> None:
    """No step scope adds no key, so a plan's fingerprint is what it was."""
    from mosaic.core.pipeline.graph.digest import document_digest
    from mosaic.core.pipeline.graph.serialize import (
        _step_fingerprint,  # pyright: ignore[reportPrivateUsage]
    )

    plan = plan_pipeline(split, _recipe(SPEED, TEMPLATES), step_scopes={})
    legacy = document_digest(
        {
            "dataset": plan.dataset_uuid,
            "recipe": plan.recipe_digest,
            "steps": [
                _step_fingerprint(planned, plan)
                for planned in sorted(plan.steps, key=lambda step: step.step_id)
            ],
        }
    )

    assert plan_fingerprint(plan) == legacy


# --- the request ----------------------------------------------------------------


def test_step_scopes_are_recorded_on_the_request(split: Dataset) -> None:
    submitted = submit_request(
        split, _recipe(SPEED, TEMPLATES), step_scopes={"templates": TRAIN}
    )

    written = json.loads(
        request_path(split.base_dir, submitted.request.request_id).read_text()
    )
    assert written["step_scopes"] == {"templates": TRAIN.model_dump(mode="json")}
    assert submitted.request.step_scope("templates") == TRAIN
    assert submitted.request.step_scope("speed") is None
    assert submitted.plan.step("templates").coverage.target == {
        ("train", "s1"),
        ("train", "s2"),
    }


def test_a_request_without_step_scopes_omits_the_key(split: Dataset) -> None:
    """An older mosaic, which forbids unknown keys, still reads the file."""
    submitted = submit_request(split, _recipe(SPEED))

    written = json.loads(
        request_path(split.base_dir, submitted.request.request_id).read_text()
    )
    assert "step_scopes" not in written


def test_camera_entries_in_a_step_scope_are_a_plan_problem(split: Dataset) -> None:
    """The plan refuses what the request would, rather than the submission raising."""
    plan = plan_pipeline(
        split,
        _recipe(SPEED),
        step_scopes={"speed": Scope(entries=[("train", "s1", "cam0")])},
        refuse=False,
    )

    assert _where(plan) == [("speed", "step_scopes.speed")]
    assert "names cameras" in plan.problems[0].message


def test_camera_entries_in_a_step_scope_are_refused() -> None:
    with pytest.raises(ValidationError, match="names cameras"):
        _ = Request(
            request_id="r",
            recipe_digest="d",
            step_scopes={"speed": Scope(entries=[("train", "s1", "cam0")])},
        )


# --- execution ----------------------------------------------------------------


def _feature_entries(ds: Dataset, storage: str) -> set[tuple[str, str]]:
    import pandas as pd

    index = pd.read_csv(
        feature_index_path(ds, storage), dtype=str, keep_default_na=False
    )
    return set(zip(index["group"], index["sequence"], strict=True))


def test_execution_honours_the_step_scope(split: Dataset) -> None:
    _ = run_pipeline(split, _recipe(SPEED), step_scopes={"speed": TEST})

    assert _feature_entries(split, "speed-angvel__from__tracks") == {("test", "s3")}


def test_shortfall_is_measured_against_the_step_target(split: Dataset) -> None:
    """Templates fitted on the training entries read speed over those alone."""
    run = run_pipeline(
        split,
        _recipe(SPEED, TEMPLATES),
        step_scopes={"speed": TRAIN, "templates": TRAIN},
    )

    templates = next(
        outcome for outcome in run.outcomes if outcome.step_id == "templates"
    )
    assert (templates.covered, templates.target) == (2, 2)
    assert _feature_entries(split, "speed-angvel__from__tracks") == {
        ("train", "s1"),
        ("train", "s2"),
    }


def test_a_params_parent_narrower_than_the_step_is_not_a_shortfall(
    split: Dataset,
) -> None:
    """A parent read through a params reference must be complete, not cover the step.

    Read through ``inputs``, a parent holding only the training entries leaves a
    test-scoped step short. Read through a params reference, the same parent
    answers for what it was fitted on.
    """
    _ = run_pipeline(split, _recipe(SPEED), step_scopes={"speed": TRAIN})
    plan = plan_pipeline(
        split,
        _recipe(SPEED, TEMPLATES),
        step_scopes={"speed": TRAIN, "templates": TEST},
        refuse=False,
    )
    assert _where(plan) == [("templates", "inputs[0]")]
    templates = plan.step("templates")
    assert plan.step("speed").is_complete

    with pytest.raises(CoverageShortfall, match=r"missing \('test', 's3'\)"):
        refuse_shortfall(plan, templates)
    refuse_shortfall(plan, replace(templates, status_parents=("speed",)))


def test_an_after_parent_is_asked_only_to_be_complete(split: Dataset) -> None:
    """An ordering edge reads no entry, so a narrower parent named by it is not short."""
    recipe = _recipe(
        SPEED,
        {**SPEED, "id": "ordered"},
        {**TEMPLATES, "inputs": [{"step": "ordered"}], "after": ["speed"]},
    )
    _ = run_pipeline(split, _recipe(SPEED), step_scopes={"speed": TRAIN})
    _ = run_pipeline(split, _recipe({**SPEED, "id": "ordered"}))
    plan = plan_pipeline(split, recipe, step_scopes={"speed": TRAIN})

    assert plan.problems == ()
    templates = plan.step("templates")
    assert templates.status_parents == ("speed",)
    refuse_shortfall(plan, templates)
