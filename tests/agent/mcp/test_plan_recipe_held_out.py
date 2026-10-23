"""``plan_recipe`` on a dataset with roles: the held-out guard and ``assume_added``.

The guard's findings travel as ordinary problems and cautions, and ``assume_added``
adds cautions only, because the tool's output schema has no room for a field.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.agent.mcp.local import LocalBackend
from mosaic.agent.mcp.spec import call_tool
from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.graph import PlanWire
from tests.helpers import RecordingStarter, calms21_dataset

RECIPE: dict[str, JsonValue] = {
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
TRAIN_ONLY: dict[str, JsonValue] = {"templates": {"groups": ["train"]}}


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
def backend(dataset: Dataset) -> LocalBackend:
    return LocalBackend(
        [dataset.manifest_path], warm=False, start_driver=RecordingStarter()
    )


def _plan(backend: LocalBackend, dataset: Dataset, **arguments: object) -> PlanWire:
    answered = call_tool(
        backend,
        "plan_recipe",
        {"dataset": dataset.manifest.uuid, "recipe": RECIPE, **arguments},
    )
    assert isinstance(answered, PlanWire), answered
    return answered


def test_a_fit_over_the_test_group_is_a_problem_the_plan_shows(
    backend: LocalBackend, dataset: Dataset
) -> None:
    plan = _plan(backend, dataset)

    assert [(problem.step, problem.where) for problem in plan.problems] == [
        ("templates", "scope")
    ]
    assert "'test' (test)" in plan.problems[0].message
    assert [caution.where for caution in plan.cautions] == ["scope"]
    assert "'test' (test), 'train' (train)" in plan.cautions[0].message


def test_the_fit_scoped_to_the_train_group_has_no_problem(
    backend: LocalBackend, dataset: Dataset
) -> None:
    plan = _plan(backend, dataset, step_entries=TRAIN_ONLY)

    assert plan.problems == ()


def test_assume_added_reports_through_the_cautions_and_changes_nothing_else(
    backend: LocalBackend, dataset: Dataset
) -> None:
    plain = _plan(backend, dataset, step_entries=TRAIN_ONLY)
    assumed = _plan(
        backend, dataset, step_entries=TRAIN_ONLY, assume_added=["later", "latest"]
    )

    assert assumed.fingerprint == plain.fingerprint
    assert assumed.steps == plain.steps
    assert assumed.problems == plain.problems
    [extra] = [c for c in assumed.cautions if c.where == "assume_added"]
    assert extra.message == (
        "if groups 'later', 'latest' joined, no step's run would change"
    )


def test_assume_added_names_the_refit_of_an_unscoped_fit(
    backend: LocalBackend, dataset: Dataset
) -> None:
    plan = _plan(backend, dataset, entries={"groups": ["train"]}, assume_added=["x"])

    [extra] = [c for c in plan.cautions if c.where == "assume_added"]
    assert extra.message == ("if group 'x' joined, no step's run would change")
    named = _plan(
        backend, dataset, entries={"groups": ["train", "x"]}, assume_added=["x"]
    )
    [taken] = [c for c in named.cautions if c.where == "assume_added"]
    assert taken.message == ("if group 'x' joined, these steps would refit: templates")


def test_assume_added_is_refused_past_its_cap(
    backend: LocalBackend, dataset: Dataset
) -> None:
    answered = call_tool(
        backend,
        "plan_recipe",
        {
            "dataset": dataset.manifest.uuid,
            "recipe": RECIPE,
            "assume_added": [f"g{index}" for index in range(17)],
        },
    )

    assert isinstance(answered, ErrorEnvelope)
