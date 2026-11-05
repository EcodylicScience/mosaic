"""An apply names a finished request and reuses its fit over other sequences.

``applies_request_id`` names the request whose fit an apply reuses. mosaic
derives the apply's pins from that request's run-logs, takes its tracks
variant, and locks every fitting step and the steps those read. A locked step
that differs from the named request's recipe is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    RecipeInvalid,
    Recipe,
    Request,
    StepRefused,
    execute_step,
    load_request,
    plan_pipeline,
    plan_wire,
    request_path,
    requests_root,
    run_pipeline,
    submit_request,
    topological_order,
)
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.scope import Scope
from mosaic.runlog import read_run, run_log_dir
from tests.helpers import CONVERT_TRACKS_STEP, calms21_dataset

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
SMOOTHED: dict[str, JsonValue] = {
    "id": "smoothed",
    "type": "feature",
    "feature": "speed-angvel",
    "inputs": ["tracks"],
    "params": {"smooth_window": 3},
}


def _recipe(*steps: dict[str, JsonValue]) -> Recipe:
    return Recipe.model_validate(
        {"schema_version": 1, "name": "apply", "steps": list(steps)}
    )


def _with_params(
    step: dict[str, JsonValue], **params: JsonValue
) -> dict[str, JsonValue]:
    current = step.get("params")
    merged: dict[str, JsonValue] = dict(current) if isinstance(current, dict) else {}
    merged.update(params)
    return {**step, "params": merged}


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


def _variant(ds: Dataset) -> str:
    variants = sorted(set(read_tracks_index(ds)["run_id"]))
    assert len(variants) == 1
    return variants[0]


def _fit(ds: Dataset, recipe: Recipe, *, tracks_run_id: str | None = None) -> Request:
    """Run *recipe* over the train group, and return its finished request."""
    run = run_pipeline(ds, recipe, scope=TRAIN, tracks_run_id=tracks_run_id)
    assert all(outcome.run_id for outcome in run.outcomes)
    return load_request(ds.base_dir, run.request_id)


def _request_files(ds: Dataset) -> set[str]:
    return {path.name for path in requests_root(ds.base_dir).glob("*.json")}


def _refused(
    ds: Dataset, recipe: Recipe, applies: str, *, tracks_run_id: str | None = None
) -> StepRefused:
    before = _request_files(ds)
    with pytest.raises(StepRefused) as raised:
        _ = submit_request(
            ds,
            recipe,
            scope=TEST,
            applies_request_id=applies,
            tracks_run_id=tracks_run_id,
        )
    assert _request_files(ds) == before
    return raised.value


FIT = _recipe(SPEED, TEMPLATES, KMEANS)


# --- an apply reuses the fit ---------------------------------------------------


def test_an_apply_reuses_the_fit_over_new_sequences(split: Dataset) -> None:
    fit = _fit(split, FIT)
    pin = _attempt_run_id(split, fit, "templates")

    submitted = submit_request(
        split, FIT, scope=TEST, applies_request_id=fit.request_id
    )
    run = run_pipeline(split, FIT, request=submitted.request)

    assert submitted.request.pins == {"templates": pin}
    assert submitted.plan.step("templates").run_id == pin
    outcomes = {outcome.step_id: outcome for outcome in run.outcomes}
    assert (outcomes["templates"].state, outcomes["templates"].run_id) == (
        "cached",
        pin,
    )
    assert outcomes["kmeans"].run_id == _attempt_run_id(split, fit, "kmeans")
    wire = plan_wire(submitted.plan)
    assert wire.locked_steps == ("speed", "templates", "kmeans")


def test_an_apply_takes_the_fit_tracks_when_it_names_none(split: Dataset) -> None:
    variant = _variant(split)
    fit = _fit(split, FIT, tracks_run_id=variant)

    submitted = submit_request(
        split, FIT, scope=TEST, applies_request_id=fit.request_id
    )

    assert submitted.request.tracks_run_id == variant
    assert submitted.plan.tracks_run_id == variant


def test_a_change_below_the_locked_steps_is_still_an_apply(split: Dataset) -> None:
    fit = _fit(split, _recipe(SPEED, TEMPLATES, KMEANS, SMOOTHED))
    changed = _recipe(
        SPEED,
        _with_params(TEMPLATES, n_templates=4.0),
        KMEANS,
        _with_params(SMOOTHED, smooth_window=5),
    )

    submitted = submit_request(
        split, changed, scope=TEST, applies_request_id=fit.request_id
    )

    assert submitted.request.applies_request_id == fit.request_id
    assert submitted.request.locked == ["speed", "templates", "kmeans"]
    assert "smoothed" not in submitted.plan.locked


# --- what an apply refuses -----------------------------------------------------


def test_an_apply_of_a_missing_request_is_refused(split: Dataset) -> None:
    refused = _refused(split, FIT, "01NOSUCHREQUEST")

    assert (refused.reason, refused.step_id) == ("apply_request_missing", "")


def test_an_apply_does_not_read_a_request_file_outside_the_requests(
    split: Dataset,
) -> None:
    """``../x`` is not a request id, so a request file at ``pipelines/x.json`` stays unread.

    Read, the planted request would be refused as unfinished instead.
    """
    unrun = submit_request(split, FIT, scope=TRAIN).request
    planted = requests_root(split.base_dir).parent / "x.json"
    _ = planted.write_bytes(request_path(split.base_dir, unrun.request_id).read_bytes())

    refused = _refused(split, FIT, "../x")

    assert (refused.reason, refused.step_id) == ("apply_request_missing", "")


def test_an_apply_of_an_unfinished_request_is_refused(split: Dataset) -> None:
    unrun = submit_request(split, FIT, scope=TRAIN).request

    refused = _refused(split, FIT, unrun.request_id)

    assert (refused.reason, refused.step_id) == ("apply_request_unfinished", "")


def test_an_apply_with_a_changed_locked_step_is_refused(split: Dataset) -> None:
    fit = _fit(split, FIT)
    changed = _recipe(_with_params(SPEED, step_size=2), TEMPLATES, KMEANS)

    refused = _refused(split, changed, fit.request_id)

    assert (refused.reason, refused.step_id) == ("apply_step_changed", "speed")


def test_an_apply_with_a_changed_run_filter_is_refused(split: Dataset) -> None:
    fit = _fit(split, FIT)
    filtered: dict[str, JsonValue] = {**SPEED, "run": {"filter_end_frame": 20}}

    refused = _refused(split, _recipe(filtered, TEMPLATES, KMEANS), fit.request_id)

    assert (refused.reason, refused.step_id) == ("apply_step_changed", "speed")


def test_an_apply_with_a_changed_tracks_reference_is_refused(split: Dataset) -> None:
    fit = _fit(split, _recipe(CONVERT_TRACKS_STEP, SPEED, TEMPLATES, KMEANS))
    referenced: dict[str, JsonValue] = {**SPEED, "tracks": {"step": "tracks"}}
    changed = _recipe(CONVERT_TRACKS_STEP, referenced, TEMPLATES, KMEANS)

    refused = _refused(split, changed, fit.request_id)

    assert (refused.reason, refused.step_id) == ("apply_step_changed", "speed")


def test_a_cyclic_recipe_naming_an_apply_is_refused_as_invalid(
    split: Dataset,
) -> None:
    fit = _fit(split, FIT)
    looped: dict[str, JsonValue] = {**TEMPLATES, "inputs": [{"step": "kmeans"}]}
    before = _request_files(split)

    with pytest.raises(RecipeInvalid, match="cycle"):
        _ = submit_request(
            split,
            _recipe(SPEED, looped, KMEANS),
            scope=TEST,
            applies_request_id=fit.request_id,
        )
    assert _request_files(split) == before


def test_an_apply_of_a_step_the_named_recipe_lacks_is_refused(
    split: Dataset,
) -> None:
    fit = _fit(split, _recipe(SPEED))

    refused = _refused(split, FIT, fit.request_id)

    assert (refused.reason, refused.step_id) == ("apply_step_changed", "templates")


def test_an_apply_with_other_tracks_is_refused(split: Dataset) -> None:
    fit = _fit(split, FIT, tracks_run_id=_variant(split))

    refused = _refused(split, FIT, fit.request_id, tracks_run_id="other-variant")

    assert (refused.reason, refused.step_id) == ("apply_tracks_changed", "")


def test_pins_beside_applies_request_id_raise(split: Dataset) -> None:
    fit = _fit(split, FIT)
    pins = {"templates": _attempt_run_id(split, fit, "templates")}

    with pytest.raises(ValueError, match="applies_request_id"):
        _ = submit_request(
            split, FIT, scope=TEST, applies_request_id=fit.request_id, pins=pins
        )
    with pytest.raises(ValueError, match="applies_request_id"):
        _ = plan_pipeline(
            split, FIT, scope=TEST, applies_request_id=fit.request_id, pins=pins
        )


def test_empty_pins_beside_applies_request_id_are_no_pins(split: Dataset) -> None:
    fit = _fit(split, FIT)
    pin = _attempt_run_id(split, fit, "templates")

    plan = plan_pipeline(
        split, FIT, scope=TEST, applies_request_id=fit.request_id, pins={}
    )
    submitted = submit_request(
        split, FIT, scope=TEST, applies_request_id=fit.request_id, pins={}
    )

    assert plan.step("templates").run_id == pin
    assert submitted.request.pins == {"templates": pin}


def test_a_blank_applies_request_id_is_refused() -> None:
    with pytest.raises(ValueError, match="applies_request_id names no request"):
        _ = Request(request_id="R", recipe_digest="0" * 16, applies_request_id=" ")


# --- a step of an applied request ----------------------------------------------


def test_a_step_of_an_applied_request_runs_from_its_recorded_pins(
    split: Dataset,
) -> None:
    fit = _fit(split, FIT)
    pin = _attempt_run_id(split, fit, "templates")
    request = submit_request(
        split, FIT, scope=TEST, applies_request_id=fit.request_id
    ).request
    request_path(split.base_dir, fit.request_id).unlink()

    outcomes = {
        step.id: execute_step(split, request, step.id)
        for step in topological_order(FIT)
    }

    assert (outcomes["templates"].state, outcomes["templates"].run_id) == (
        "cached",
        pin,
    )
    assert outcomes["speed"].state != "cached"
    assert outcomes["kmeans"].state != "cached"


def test_an_applied_request_records_its_locked_steps_and_omits_them_otherwise(
    split: Dataset,
) -> None:
    fit = _fit(split, FIT)
    fit_written = json.loads(request_path(split.base_dir, fit.request_id).read_text())

    applied = submit_request(split, FIT, scope=TEST, applies_request_id=fit.request_id)

    assert "applies_request_id" not in fit_written
    assert "locked" not in fit_written
    written = json.loads(applied.request_path.read_text())
    assert written["applies_request_id"] == fit.request_id
    assert written["locked"] == ["speed", "templates", "kmeans"]
    loaded = load_request(split.base_dir, applied.request.request_id)
    assert loaded.locked == ["speed", "templates", "kmeans"]
    assert loaded.pins == applied.request.pins


def _attempt_run_id(ds: Dataset, request: Request, step_id: str) -> str:
    """The run id that *step_id*'s attempt of *request* recorded."""
    snapshot = read_run(run_log_dir(ds.base_dir), request.execution_of(step_id))
    assert snapshot is not None
    return snapshot["run_id"]
