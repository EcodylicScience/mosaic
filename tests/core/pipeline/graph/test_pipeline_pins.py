"""A request pins a fitted step to a run, which then serves other sequences.

``Request.pins`` maps a step that fits over its scope (``fits`` and
``scope_dependent``) to a finished run of its storage. The pinned step is not
executed: it reads complete over any scope, and the steps below it derive
their identity from the pinned run.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    BoundRef,
    Recipe,
    Request,
    StepRefused,
    execute_step,
    load_request,
    plan_fingerprint,
    plan_pipeline,
    requests_root,
    run_pipeline,
    submit_request,
)
from mosaic.core.pipeline.index import feature_index_path
from mosaic.core.pipeline.inventory.cache import InventoryCache
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
PCA_FIT: dict[str, JsonValue] = {
    "id": "fit",
    "type": "feature",
    "feature": "pair-posedistance-pca",
    "inputs": ["tracks"],
    "params": {"n_components": 2, "pose": {"pose_n": 3}},
}
ARHMM_FIT: dict[str, JsonValue] = {
    "id": "fit",
    "type": "feature",
    "feature": "arhmm",
    "inputs": [{"step": "speed"}],
    "params": {"n_states": 2, "n_iter": 2},
}
ARHMM_APPLY: dict[str, JsonValue] = {
    "id": "apply",
    "type": "feature",
    "feature": "arhmm",
    "inputs": [{"step": "speed"}],
    "params": {"n_states": 2, "n_iter": 2, "model": {"step": "fit"}},
}


def _recipe(*steps: dict[str, JsonValue]) -> Recipe:
    return Recipe.model_validate(
        {"schema_version": 1, "name": "pins", "steps": list(steps)}
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


def _fit_templates(ds: Dataset) -> str:
    """Fit the templates over the train group, and return their run id."""
    run = run_pipeline(ds, _recipe(SPEED, TEMPLATES), scope=TRAIN)
    fitted = next(outcome for outcome in run.outcomes if outcome.step_id == "templates")
    assert fitted.run_id
    return fitted.run_id


def _templates_root(ds: Dataset, run_id: str) -> Path:
    """The run root of a templates run, whichever storage holds it."""
    return next(Path(ds.get_root("features")).glob(f"*/{run_id}"))


def _request_files(ds: Dataset) -> set[str]:
    return {path.name for path in requests_root(ds.base_dir).glob("*.json")}


# --- a pin serves its run ------------------------------------------------------


def test_a_pinned_template_step_serves_its_run_over_other_sequences(
    split: Dataset,
) -> None:
    pin = _fit_templates(split)
    recipe = _recipe(SPEED, TEMPLATES, KMEANS)

    submitted = submit_request(split, recipe, scope=TEST, pins={"templates": pin})
    run = run_pipeline(split, recipe, request=submitted.request)

    plan = submitted.plan
    assert plan.pinned == {"templates"}
    assert plan.step("templates").run_id == pin
    assert plan.step("templates").status == "complete"
    templates_params = plan.step("kmeans").spec.params["templates"]
    assert isinstance(templates_params, dict)
    assert templates_params["run_id"] == pin
    outcomes = {outcome.step_id: outcome for outcome in run.outcomes}
    assert outcomes["templates"].state == "cached"
    assert outcomes["templates"].run_id == pin
    assert outcomes["kmeans"].run_id == plan.step("kmeans").run_id
    assert outcomes["kmeans"].state != "cached"


def test_a_pinned_step_is_served_from_the_cache_under_overwrite(
    split: Dataset,
) -> None:
    pin = _fit_templates(split)
    request = submit_request(
        split, _recipe(SPEED, TEMPLATES), scope=TEST, pins={"templates": pin}
    ).request
    runs = sorted(path.name for path in _templates_root(split, pin).parent.iterdir())

    outcome = execute_step(split, request, "templates", overwrite=True)

    assert (outcome.state, outcome.run_id) == ("cached", pin)
    after = sorted(path.name for path in _templates_root(split, pin).parent.iterdir())
    assert after == runs


def test_a_pin_reads_complete_when_the_held_inventory_was_built_scoped(
    split: Dataset,
) -> None:
    pin = _fit_templates(split)
    held = InventoryCache(split)
    _ = held.get(scope=Scope(entries=[("test", "s3")]))

    plan = plan_pipeline(
        split,
        _recipe(SPEED, TEMPLATES),
        scope=TEST,
        inventory=held,
        pins={"templates": pin},
    )

    assert plan.step("templates").status == "complete"
    assert plan.step("templates").spec.entries.is_unset


def test_a_request_records_its_pins(split: Dataset) -> None:
    pin = _fit_templates(split)

    submitted = submit_request(
        split, _recipe(SPEED, TEMPLATES), scope=TEST, pins={"templates": pin}
    )

    written = json.loads(submitted.request_path.read_text())
    assert written["pins"] == {"templates": pin}
    loaded = load_request(split.base_dir, submitted.request.request_id)
    assert loaded.pins == {"templates": pin}


# --- what a pin refuses ----------------------------------------------------------


def _refused(
    ds: Dataset, recipe: Recipe, pins: dict[str, str], scope: Scope | None = None
) -> StepRefused:
    before = _request_files(ds)
    with pytest.raises(StepRefused) as raised:
        _ = submit_request(ds, recipe, scope=scope, pins=pins)
    assert _request_files(ds) == before
    return raised.value


def test_a_pin_on_an_unknown_step_is_refused(split: Dataset) -> None:
    refused = _refused(split, _recipe(SPEED, TEMPLATES), {"nope": "0.3-aaaaaaaaaa"})

    assert refused.reason == "pin_unknown_step"
    assert refused.step_id == "nope"


def test_a_pin_on_a_scope_free_fitter_is_refused(split: Dataset) -> None:
    refused = _refused(
        split, _recipe(SPEED, TEMPLATES, KMEANS), {"kmeans": "0.1-aaaaaaaaaa"}
    )

    assert refused.reason == "pin_not_fitted"
    assert refused.step_id == "kmeans"


def test_a_pin_on_a_model_given_fitter_is_refused(split: Dataset) -> None:
    refused = _refused(
        split,
        _recipe(SPEED, ARHMM_FIT, ARHMM_APPLY),
        {"apply": "0.1-aaaaaaaaaa"},
    )

    assert refused.reason == "pin_not_fitted"
    assert refused.step_id == "apply"


def test_a_pin_to_a_run_the_index_lacks_is_refused(split: Dataset) -> None:
    refused = _refused(
        split, _recipe(SPEED, TEMPLATES, KMEANS), {"templates": "0.3-aaaaaaaaaa"}
    )

    assert refused.reason == "pin_run_missing"
    assert refused.step_id == "templates"


def test_a_pin_to_an_unfinished_run_is_refused(split: Dataset) -> None:
    pin = _fit_templates(split)
    storage = _templates_root(split, pin).parent.name
    index = feature_index_path(split, storage)
    rows = pd.read_csv(index, dtype=str, keep_default_na=False)
    rows.loc[rows["run_id"] == pin, "finished_at"] = ""
    rows.to_csv(index, index=False)

    refused = _refused(split, _recipe(SPEED, TEMPLATES), {"templates": pin}, scope=TEST)

    assert refused.reason == "pin_run_unfinished"
    assert refused.step_id == "templates"


def test_reading_a_pinned_step_through_inputs_is_a_problem_at_inputs(
    split: Dataset,
) -> None:
    run = run_pipeline(split, _recipe(PCA_FIT), scope=TRAIN)
    pin = run.outcomes[0].run_id
    reader: dict[str, JsonValue] = {
        "id": "hmm",
        "type": "feature",
        "feature": "arhmm",
        "inputs": [{"step": "fit"}],
        "params": {"n_states": 2, "n_iter": 2},
    }

    plan = plan_pipeline(
        split, _recipe(PCA_FIT, reader), pins={"fit": pin}, refuse=False
    )

    [problem] = [found for found in plan.problems if found.step == "hmm"]
    assert problem.where == "inputs"
    assert "model" in problem.message


def test_a_pinned_fit_over_a_test_group_is_not_refused_as_held_out(
    split: Dataset,
) -> None:
    pin = _fit_templates(split)
    split.set_splits("train", ["train"])
    split.set_splits("test", ["test"])

    plan = plan_pipeline(
        split, _recipe(SPEED, TEMPLATES), pins={"templates": pin}, refuse=False
    )

    assert [(found.step, found.where) for found in plan.problems] == []
    assert plan.step("templates").run_id == pin


# --- the fingerprint and the request file ----------------------------------------


def test_pins_enter_the_fingerprint_only_when_present(split: Dataset) -> None:
    """The same steps and runs, without the pin, are a plan of their own.

    A plan without pins adds no key, which ``test_pipeline_plan_fingerprint.py``
    checks against the document of an earlier mosaic.
    """
    pin = _fit_templates(split)
    pinned = plan_pipeline(
        split, _recipe(SPEED, TEMPLATES), scope=TEST, pins={"templates": pin}
    )
    plain = plan_pipeline(split, _recipe(SPEED, TEMPLATES), scope=TEST)

    assert plain.pinned == frozenset()
    assert plan_fingerprint(pinned) != plan_fingerprint(
        replace(pinned, pinned=frozenset())
    )


def test_a_request_without_pins_writes_no_pins_key(split: Dataset) -> None:
    submitted = submit_request(split, _recipe(SPEED))

    assert "pins" not in json.loads(submitted.request_path.read_text())
    assert submitted.request.pins == {}


@pytest.mark.parametrize(
    "pins",
    [{"templates": ""}, {"templates": "  "}, {"": "0.3-aaaaaaaaaa"}],
    ids=["empty-run-id", "blank-run-id", "empty-step"],
)
def test_a_request_refuses_a_pin_that_names_nothing(pins: dict[str, str]) -> None:
    with pytest.raises(ValueError, match=r"pins(\[.*\] names no run_id| holds a run)"):
        _ = Request(request_id="R", recipe_digest="0" * 16, pins=pins)


def test_a_request_refuses_a_step_both_bound_and_pinned() -> None:
    with pytest.raises(ValueError, match="both bound and pinned"):
        _ = Request(
            request_id="R",
            recipe_digest="0" * 16,
            bind={"templates": BoundRef(feature="extract-templates", run_id="0.3-a")},
            pins={"templates": "0.3-b"},
        )
