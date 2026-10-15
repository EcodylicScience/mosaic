"""The guards on the tools that change a dataset: confirmation, plan currency, managers.

Each submission here starts a recording stand-in for the detached driver, so a
request is recorded and nothing runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from mosaic.agent.mcp.local import LocalBackend
from mosaic.agent.mcp.spec import (
    ConfirmationCondition,
    DatasetList,
    SubmittedPipeline,
    call_tool,
)
from mosaic.core.dataset import Dataset
from mosaic.core.dataset_summary import DatasetSummary
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline import graph
from mosaic.core.pipeline.cancel import RequestCancel
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.graph import (
    DetachedDriver,
    PlanWire,
    Recipe,
    Request,
    read_recipe,
    recipe_digest,
    save_recipe,
    save_request,
)
from mosaic.runlog import JsonlRunLog, new_execution_id, run_log_path
from tests.helpers import SPEED_RECIPE, add_tracks_variant, make_dataset

VARIANT = "convert-trex.0.2-1111111111"

ENTRIES: list[JsonValue] = [["", "seq_a"], ["", "seq_b"]]
"""Both entries of the fixture, named explicitly."""


@dataclass
class RecordingStarter:
    """A stand-in for the detached driver that records each request it is given."""

    started: list[str] = field(default_factory=list)

    def __call__(self, dataset: Dataset, request: Request) -> DetachedDriver:
        self.started.append(request.request_id)
        return DetachedDriver(
            request_id=request.request_id,
            pid=4242,
            log_path=dataset.base_dir / "driver.log",
        )


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """Two schema-valid track tables, and nothing computed."""
    ds = make_dataset(tmp_path / "tracked", name="tracked")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


@pytest.fixture
def starter() -> RecordingStarter:
    return RecordingStarter()


@pytest.fixture
def backend(dataset: Dataset, starter: RecordingStarter) -> LocalBackend:
    return LocalBackend([dataset.manifest_path], warm=False, start_driver=starter)


def _uuid(dataset: Dataset) -> str:
    uuid = dataset.manifest.uuid
    assert uuid is not None
    return uuid


def _call[T](
    backend: LocalBackend, tool: str, expected: type[T], /, **arguments: object
) -> T:
    """Call *tool* through the dispatcher and require an output of *expected*."""
    answered = call_tool(backend, tool, arguments)
    assert isinstance(answered, expected), answered
    return answered


def _submit(
    backend: LocalBackend,
    dataset: Dataset,
    recipe: Mapping[str, object],
    **arguments: object,
) -> SubmittedPipeline | ErrorEnvelope:
    """Plan *recipe* over the entries that *arguments* name, then submit that plan."""
    entries = arguments.get("entries")
    plan = _call(
        backend,
        "plan_recipe",
        PlanWire,
        dataset=_uuid(dataset),
        recipe=recipe,
        entries=entries,
    )
    answered = call_tool(
        backend,
        "submit_pipeline",
        {
            "dataset": _uuid(dataset),
            "recipe": recipe,
            "plan_fingerprint": plan.fingerprint,
            **arguments,
        },
    )
    assert isinstance(answered, SubmittedPipeline | ErrorEnvelope)
    return answered


def _conditions(refused: SubmittedPipeline | ErrorEnvelope) -> set[str]:
    """The confirmation conditions that a refusal lists."""
    assert isinstance(refused, ErrorEnvelope), refused
    assert (refused.kind, refused.reason) == ("invalid", "confirmation_required")
    assert refused.detail is not None
    listed = refused.detail["conditions"]
    assert isinstance(listed, list)
    return {str(condition) for condition in listed}


def _op_recipe(kind: str, params: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "schema_version": 1,
        "steps": [{"id": "op", "type": "op", "kind": kind, "params": params}],
    }


def _requests(dataset: Dataset) -> list[Path]:
    root = dataset.base_dir / ".mosaic" / "pipelines" / "requests"
    return sorted(root.glob("*.json")) if root.exists() else []


def test_a_submission_meeting_no_condition_is_recorded_and_driven(
    backend: LocalBackend, dataset: Dataset, starter: RecordingStarter
) -> None:
    submitted = _submit(backend, dataset, SPEED_RECIPE, entries=ENTRIES)

    assert isinstance(submitted, SubmittedPipeline)
    assert starter.started == [submitted.request_id]
    assert (submitted.driver_pid, submitted.poll_after_seconds) == (4242, 10)
    assert len(_requests(dataset)) == 1


@pytest.mark.parametrize(
    ("recipe", "arguments", "expected"),
    [
        (
            _op_recipe("trex", {}),
            {"entries": ENTRIES},
            {"gpu_lane", "tracking_op"},
        ),
        (SPEED_RECIPE, {}, {"whole_dataset"}),
        (SPEED_RECIPE, {"entries": ENTRIES, "allow_partial": True}, {"allow_partial"}),
        (
            SPEED_RECIPE,
            {
                "entries": ENTRIES,
                "bind": {"speed": {"feature": "speed-angvel", "run_id": "0.2-x"}},
            },
            {"bind"},
        ),
    ],
    ids=["tracker", "whole-dataset", "allow-partial", "bind"],
)
def test_each_condition_refuses_until_confirmed(
    backend: LocalBackend,
    dataset: Dataset,
    starter: RecordingStarter,
    recipe: Mapping[str, object],
    arguments: dict[str, object],
    expected: set[ConfirmationCondition],
) -> None:
    refused = _submit(backend, dataset, recipe, **arguments)

    assert _conditions(refused) == expected
    assert (starter.started, _requests(dataset)) == ([], [])

    confirmed = _submit(backend, dataset, recipe, confirm=True, **arguments)

    assert isinstance(confirmed, SubmittedPipeline), confirmed
    assert starter.started == [confirmed.request_id]


def test_training_asks_to_confirm_and_is_polled_less_often(
    backend: LocalBackend, dataset: Dataset, tmp_path: Path
) -> None:
    recipe = _op_recipe("train-pose", {"data": str(tmp_path)})

    refused = _submit(backend, dataset, recipe, entries=ENTRIES)
    confirmed = _submit(backend, dataset, recipe, entries=ENTRIES, confirm=True)

    assert _conditions(refused) == {"gpu_lane", "training_op"}
    assert isinstance(confirmed, SubmittedPipeline), confirmed
    assert confirmed.poll_after_seconds == 60


def test_a_step_asked_for_more_than_a_hundred_entries_asks_to_confirm(
    tmp_path: Path, starter: RecordingStarter
) -> None:
    sequences = [f"seq_{index:03d}" for index in range(101)]
    large = make_dataset(tmp_path / "large")
    add_tracks_variant(large, VARIANT, *sequences, n_rows=2, std_format="mosaic_v1")
    backend = LocalBackend([large.manifest_path], warm=False, start_driver=starter)
    entries: list[JsonValue] = [["", sequence] for sequence in sequences]

    refused = _submit(backend, large, SPEED_RECIPE, entries=entries)

    assert _conditions(refused) == {"large_scope"}


def test_an_open_request_of_the_recipe_asks_to_confirm_a_second(
    backend: LocalBackend, dataset: Dataset
) -> None:
    first = _submit(backend, dataset, SPEED_RECIPE, entries=ENTRIES)
    second = _submit(backend, dataset, SPEED_RECIPE, entries=ENTRIES)

    assert isinstance(first, SubmittedPipeline)
    assert _conditions(second) == {"open_duplicate"}
    assert isinstance(second, ErrorEnvelope)
    assert first.request_id in second.message


def _no_open_requests(*_: object) -> tuple[str, ...]:
    return ()


def test_an_unconfirmed_duplicate_written_after_the_check_refuses_the_submission(
    backend: LocalBackend,
    dataset: Dataset,
    starter: RecordingStarter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _submit(backend, dataset, SPEED_RECIPE, entries=ENTRIES)
    # The confirmation check reads no open request, as when another process
    # writes the first one after the check and before this submission's write.
    monkeypatch.setattr(graph, "open_requests_matching", _no_open_requests)

    second = _submit(backend, dataset, SPEED_RECIPE, entries=ENTRIES)

    assert isinstance(first, SubmittedPipeline)
    assert isinstance(second, ErrorEnvelope)
    assert second.kind == "conflict"
    assert first.request_id in second.message
    assert starter.started == [first.request_id]


def test_a_plan_that_changed_since_it_was_reviewed_refuses_the_submission(
    backend: LocalBackend, dataset: Dataset, starter: RecordingStarter
) -> None:
    plan = _call(
        backend,
        "plan_recipe",
        PlanWire,
        dataset=_uuid(dataset),
        recipe=SPEED_RECIPE,
    )
    add_tracks_variant(dataset, VARIANT, "seq_c", std_format="mosaic_v1")

    refused = _call(
        backend,
        "submit_pipeline",
        ErrorEnvelope,
        dataset=_uuid(dataset),
        recipe=SPEED_RECIPE,
        plan_fingerprint=plan.fingerprint,
        confirm=True,
    )

    assert refused.kind == "conflict"
    assert (starter.started, _requests(dataset)) == ([], [])


def _mark_managed(dataset: Dataset) -> None:
    with dataset.manifest_path.open("a", encoding="utf-8") as manifest:
        _ = manifest.write("managed_by: mosaic-api\n")


def test_a_managed_dataset_is_read_and_not_changed(
    backend: LocalBackend, dataset: Dataset, starter: RecordingStarter
) -> None:
    _ = _call(backend, "list_datasets", DatasetList)
    _mark_managed(dataset)

    listed = _call(backend, "list_datasets", DatasetList)
    summary = _call(backend, "describe_dataset", DatasetSummary, dataset=_uuid(dataset))
    submitted = _submit(backend, dataset, SPEED_RECIPE, entries=ENTRIES, confirm=True)
    cancel = _call(
        backend,
        "cancel_request",
        ErrorEnvelope,
        dataset=_uuid(dataset),
        request_id="R0",
    )

    assert listed.items[0].api_managed
    assert summary.api_managed
    for refused in (submitted, cancel):
        assert isinstance(refused, ErrorEnvelope)
        assert (refused.kind, refused.reason) == ("unavailable", "dataset_managed")
    assert (starter.started, _requests(dataset)) == ([], [])


def test_a_backend_started_to_allow_it_changes_a_managed_dataset(
    dataset: Dataset, starter: RecordingStarter
) -> None:
    _mark_managed(dataset)
    backend = LocalBackend(
        [dataset.manifest_path], warm=False, allow_managed=True, start_driver=starter
    )

    listed = _call(backend, "list_datasets", DatasetList)
    submitted = _submit(backend, dataset, SPEED_RECIPE, entries=ENTRIES)

    assert listed.items[0].api_managed
    assert isinstance(submitted, SubmittedPipeline), submitted


def _running_request(dataset: Dataset, recipe: Recipe) -> Request:
    """Record *recipe*'s one step as started and not ended, in a request of its own."""
    step = recipe.steps[0]
    request = Request(
        request_id=new_execution_id(),
        recipe_digest=recipe_digest(recipe),
        step_executions={step.id: new_execution_id()},
        step_versions={step.id: "0"},
    )
    _ = save_recipe(dataset.base_dir, recipe)
    _ = save_request(dataset.base_dir, request)
    execution_id = request.execution_of(step.id)
    with JsonlRunLog(run_log_path(dataset.base_dir, execution_id), execution_id) as log:
        log.started(kind="feature", target=step.id)
    return request


@pytest.mark.parametrize(
    "step",
    [
        {"id": "fit", "type": "feature", "feature": "kpms", "inputs": ["tracks"]},
        {"id": "fit", "type": "op", "kind": "train-pose", "params": {"data": "d"}},
    ],
    ids=["runner-feature", "training-op"],
)
def test_cancelling_a_step_that_loses_its_work_asks_to_confirm(
    backend: LocalBackend, dataset: Dataset, step: dict[str, JsonValue]
) -> None:
    request = _running_request(
        dataset, read_recipe({"schema_version": 1, "steps": [step]})
    )

    refused = _call(
        backend,
        "cancel_request",
        ErrorEnvelope,
        dataset=_uuid(dataset),
        request_id=request.request_id,
    )

    assert _conditions(refused) == {"hard_cancel"}


def test_a_request_that_never_started_is_cancelled_without_confirmation(
    backend: LocalBackend, dataset: Dataset
) -> None:
    submitted = _submit(backend, dataset, SPEED_RECIPE, entries=ENTRIES)
    assert isinstance(submitted, SubmittedPipeline)

    cancelled = _call(
        backend,
        "cancel_request",
        RequestCancel,
        dataset=_uuid(dataset),
        request_id=submitted.request_id,
    )

    assert (cancelled.outcome, cancelled.status) == ("recorded", "cancelled")
