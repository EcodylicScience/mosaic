"""Every shared tool answered by the local backend over a fixture dataset.

The submission test starts a real detached driver, which runs each step as a
process of its own, and polls the request as a client would.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.agent.mcp.local import LocalBackend
from mosaic.agent.mcp.outputs import StepSuggestions, StepTypePage
from mosaic.agent.mcp.spec import (
    ActivityPage,
    DatasetList,
    FailureDiagnosis,
    SequencePage,
    SubmittedPipeline,
    call_tool,
)
from mosaic.core.dataset import Dataset
from mosaic.core.dataset_summary import DatasetSummary
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.graph import (
    QUARANTINE_AFTER,
    CatalogEntry,
    CoverageShortfall,
    FileFailureStore,
    PlanWire,
    RequestStatusWire,
    ValidationReport,
    request_path,
    submit_request,
)
from mosaic.core.pipeline.inventory import InventoryWire
from mosaic.core.pipeline.markers import process_running
from mosaic.core.pipeline.preview import ArtifactPreview
from mosaic.runlog import JsonlRunLog, run_log_path
from tests.helpers import (
    CHAIN_RECIPE,
    SPEED_RECIPE,
    add_tracks_variant,
    chain_recipe,
    make_dataset,
    wait_for,
)

VARIANT = "convert-trex.0.2-1111111111"

ENTRIES = [["", "seq_a"], ["", "seq_b"]]
"""Both entries of the fixture, named explicitly."""

REQUEST_SECONDS = 360.0
"""The longest the submission test waits for its two step processes to finish."""

STORAGE = "speed-angvel__from__tracks"


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """Two schema-valid track tables, and nothing computed."""
    ds = make_dataset(tmp_path / "tracked", name="tracked")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


@pytest.fixture
def backend(dataset: Dataset) -> LocalBackend:
    return LocalBackend([dataset.manifest_path], warm=False)


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


def _refusal(backend: LocalBackend, tool: str, /, **arguments: object) -> ErrorEnvelope:
    return _call(backend, tool, ErrorEnvelope, **arguments)


def test_the_datasets_are_listed_by_uuid_a_page_at_a_time(
    tmp_path: Path, dataset: Dataset
) -> None:
    other = make_dataset(tmp_path / "other", name="other")
    backend = LocalBackend(
        [dataset.manifest_path, other.manifest_path, dataset.manifest_path],
        warm=False,
    )

    first = _call(backend, "list_datasets", DatasetList, limit=1)
    second = _call(
        backend, "list_datasets", DatasetList, limit=1, cursor=first.next_cursor
    )

    assert first.total == 2
    assert [item.name for item in (*first.items, *second.items)] == ["other", "tracked"]
    assert second.items[0].uuid == _uuid(dataset)
    assert (second.items[0].n_sequences, second.items[0].api_managed) == (2, False)
    assert second.next_cursor is None


def test_the_datasets_below_a_directory_are_served_once_warm(
    tmp_path: Path, dataset: Dataset
) -> None:
    backend = LocalBackend.discover(tmp_path)
    backend.wait_until_warm()

    listed = _call(backend, "list_datasets", DatasetList)

    assert [item.uuid for item in listed.items] == [_uuid(dataset)]


def test_a_cursor_no_page_returned_is_invalid(backend: LocalBackend) -> None:
    refused = _refusal(backend, "list_datasets", cursor="not json")

    assert (refused.kind, refused.reason) == ("invalid", "invalid_cursor")


def test_a_dataset_is_listed_with_the_sequences_it_is_described_with(
    backend: LocalBackend, dataset: Dataset
) -> None:
    (missing,) = (dataset.base_dir / "tracks").rglob("*seq_b*.parquet")
    missing.unlink()

    listed = _call(backend, "list_datasets", DatasetList)
    summary = _call(backend, "describe_dataset", DatasetSummary, dataset=_uuid(dataset))

    assert listed.items[0].n_sequences == summary.counts.sequences == 1


def test_a_dataset_outside_the_allowlist_is_not_found(backend: LocalBackend) -> None:
    refused = _refusal(backend, "describe_dataset", dataset="no-such-uuid")

    assert (refused.kind, refused.reason) == ("not_found", "unknown_dataset")


def test_a_dataset_is_described_from_its_manifest_and_indexes(
    backend: LocalBackend, dataset: Dataset
) -> None:
    summary = _call(backend, "describe_dataset", DatasetSummary, dataset=_uuid(dataset))

    assert (summary.name, summary.uuid) == ("tracked", _uuid(dataset))
    assert summary.counts.sequences == 2
    assert summary.counts.tracks_variants == 1


def test_the_sequences_are_listed_with_their_tracks_tables(
    backend: LocalBackend, dataset: Dataset
) -> None:
    first = _call(
        backend, "list_sequences", SequencePage, dataset=_uuid(dataset), limit=1
    )
    rest = _call(
        backend,
        "list_sequences",
        SequencePage,
        dataset=_uuid(dataset),
        cursor=first.next_cursor,
    )

    rows = [*first.items, *rest.items]
    assert [(row.group, row.sequence) for row in rows] == [("", "seq_a"), ("", "seq_b")]
    assert {row.tracks_variant for row in rows} == {VARIANT}
    assert {row.std_format for row in rows} == {"mosaic_v1"}
    assert all(row.frame_min is not None and row.n_keypoints for row in rows)
    assert (first.total, rest.next_cursor) == (2, None)


def test_the_inventory_leaves_out_complete_artifacts_unless_asked(
    backend: LocalBackend, dataset: Dataset
) -> None:
    default = _call(backend, "get_inventory", InventoryWire, dataset=_uuid(dataset))
    every = _call(
        backend, "get_inventory", InventoryWire, dataset=_uuid(dataset), status_not=[]
    )
    variant = next(a for a in every.artifacts if a.kind == "tracks-variant")
    detailed = _call(
        backend, "get_inventory", InventoryWire, dataset=_uuid(dataset), ref=variant.ref
    )

    assert variant.status == "complete"
    assert variant.ref not in {artifact.ref for artifact in default.artifacts}
    assert detailed.artifacts[0].detail is not None


def test_step_types_are_searched_by_words_and_paged(backend: LocalBackend) -> None:
    found = _call(backend, "search_step_types", StepTypePage, query="speed angvel")
    ops = _call(backend, "search_step_types", StepTypePage, type="op", limit=3)
    more = _call(
        backend, "search_step_types", StepTypePage, type="op", cursor=ops.next_cursor
    )

    assert "speed-angvel" in {entry.name for entry in found.entries}
    assert len(ops.entries) == 3
    assert ops.total == len(ops.entries) + len(more.entries)
    assert {entry.type for entry in (*ops.entries, *more.entries)} == {"op"}


def test_one_step_type_is_described_whole(backend: LocalBackend) -> None:
    entry = _call(backend, "describe_step_type", CatalogEntry, name="speed-angvel")
    missing = _refusal(backend, "describe_step_type", name="no-such-step")

    assert entry.params_schema is not None
    assert missing.kind == "not_found"


def test_a_recipe_is_validated_and_a_broken_one_reported(backend: LocalBackend) -> None:
    valid = _call(backend, "validate_recipe", ValidationReport, recipe=SPEED_RECIPE)
    broken = _call(
        backend,
        "validate_recipe",
        ValidationReport,
        recipe={"schema_version": 1, "steps": [{"id": "x", "type": "feature"}]},
    )

    assert valid.valid
    assert not broken.valid
    assert broken.problems


def test_suggestions_list_the_allowed_choices_and_count_the_refused(
    backend: LocalBackend,
) -> None:
    below = _call(
        backend,
        "suggest_steps",
        StepSuggestions,
        recipe=SPEED_RECIPE,
        mode="attach",
        step="speed",
    )
    inputs = _call(
        backend,
        "suggest_steps",
        StepSuggestions,
        recipe=CHAIN_RECIPE,
        mode="input",
        step="templates",
    )
    unnamed = _refusal(backend, "suggest_steps", recipe=SPEED_RECIPE, mode="input")

    assert below.steps
    assert all(attachment.allowed for attachment in below.steps.values())
    assert below.refused > 0
    assert all(candidate.allowed for candidate in inputs.inputs.values())
    assert (unnamed.kind, unnamed.reason) == ("invalid", "step_required")


def test_a_plan_names_its_fingerprint_and_a_malformed_recipe_is_invalid(
    backend: LocalBackend, dataset: Dataset
) -> None:
    plan = _call(
        backend, "plan_recipe", PlanWire, dataset=_uuid(dataset), recipe=SPEED_RECIPE
    )
    malformed = _refusal(
        backend, "plan_recipe", dataset=_uuid(dataset), recipe={"steps": "none"}
    )

    assert plan.fingerprint
    assert [(step.step, step.status, step.asked) for step in plan.steps] == [
        ("speed", "absent", 2)
    ]
    assert malformed.kind == "invalid"


def test_a_submitted_request_runs_detached_and_its_output_is_previewed(
    backend: LocalBackend, dataset: Dataset
) -> None:
    uuid = _uuid(dataset)
    plan = _call(
        backend,
        "plan_recipe",
        PlanWire,
        dataset=uuid,
        recipe=CHAIN_RECIPE,
        entries=ENTRIES,
    )

    submitted = _call(
        backend,
        "submit_pipeline",
        SubmittedPipeline,
        dataset=uuid,
        recipe=CHAIN_RECIPE,
        entries=ENTRIES,
        plan_fingerprint=plan.fingerprint,
    )

    assert submitted.driver_pid > 0
    assert submitted.poll_after_seconds == 10
    assert [step.step for step in submitted.steps] == ["speed", "templates"]
    wait_for(
        lambda: (
            _call(
                backend,
                "get_request_status",
                RequestStatusWire,
                dataset=uuid,
                request_id=submitted.request_id,
            ).terminal
        ),
        REQUEST_SECONDS,
        "the detached driver never ended the request",
    )
    status = _call(
        backend,
        "get_request_status",
        RequestStatusWire,
        dataset=uuid,
        request_id=submitted.request_id,
    )
    assert status.verdict == "finished", status
    # The driver outlives this test's process otherwise, and its handle warns
    # when the worker exits.
    wait_for(
        lambda: not process_running(submitted.driver_pid),
        REQUEST_SECONDS,
        "the detached driver never exited",
    )
    speed = status.steps[0]
    preview = _call(
        backend,
        "preview_output",
        ArtifactPreview,
        dataset=uuid,
        ref=f"feature/{STORAGE}/{speed.run_id}",
        group="",
        sequence="seq_a",
        limit=5,
    )
    assert (len(preview.rows), preview.has_more) == (5, True)
    requests = _call(backend, "list_activity", ActivityPage, dataset=uuid)
    runs = _call(backend, "list_activity", ActivityPage, dataset=uuid, view="runs")
    assert [item.request_id for item in requests.requests] == [submitted.request_id]
    assert {run.execution_id for run in runs.runs} == {
        step.execution_id for step in submitted.steps
    }
    diagnosis = _call(
        backend,
        "diagnose_failures",
        FailureDiagnosis,
        dataset=uuid,
        request_id=submitted.request_id,
    )
    assert (diagnosis.verdict, diagnosis.attempts) == ("finished", ())


def test_a_diagnosis_says_where_allow_partial_answers_a_failure(
    backend: LocalBackend, dataset: Dataset
) -> None:
    request = submit_request(dataset, chain_recipe()).request
    base = dataset.base_dir
    speed, templates = (
        request.execution_of("speed"),
        request.execution_of("templates"),
    )
    with JsonlRunLog(run_log_path(base, speed), speed) as log:
        log.started(kind="feature", target=STORAGE)
        log.set_run_id("0.2-abcdef0123")
        log.entry_failed("seq_a", '{"type": "ValueError", "message": "bad"}')
        log.finished()
    store = FileFailureStore(base)
    for _ in range(QUARANTINE_AFTER):
        _ = store.note_entry_failure(
            (STORAGE, "0.2-abcdef0123", "", "seq_a"), error="bad"
        )
    refusal = CoverageShortfall("templates", 1, 2, "one entry is quarantined")
    with JsonlRunLog(run_log_path(base, templates), templates) as log:
        log.started(kind="feature", target="extract-templates__from__speed")
        log.failed(refusal.error_json())

    diagnosis = _call(
        backend,
        "diagnose_failures",
        FailureDiagnosis,
        dataset=_uuid(dataset),
        request_id=request.request_id,
    )
    one = _call(
        backend,
        "diagnose_failures",
        FailureDiagnosis,
        dataset=_uuid(dataset),
        execution_id=speed,
    )

    assert diagnosis.verdict == "failed"
    lost, refused = diagnosis.attempts
    assert (lost.step, lost.outcome, lost.allow_partial_applies) == (
        "speed",
        "partial",
        True,
    )
    assert lost.entry_errors is not None
    assert lost.entry_errors.held_back_by_quarantine
    assert (refused.step, refused.outcome, refused.allow_partial_applies) == (
        "templates",
        "failed",
        True,
    )
    assert refused.error is not None
    assert refused.error.refusal is not None
    assert one.attempts == (lost.model_copy(update={"step": ""}),)


def test_a_diagnosis_names_one_request_or_one_attempt(
    backend: LocalBackend, dataset: Dataset
) -> None:
    neither = _refusal(backend, "diagnose_failures", dataset=_uuid(dataset))
    unknown = _refusal(
        backend, "diagnose_failures", dataset=_uuid(dataset), execution_id="E0"
    )

    assert (neither.kind, neither.reason) == ("invalid", "one_id_required")
    assert unknown.kind == "not_found"


def test_an_unknown_request_is_not_found(
    backend: LocalBackend, dataset: Dataset
) -> None:
    refused = _refusal(
        backend, "get_request_status", dataset=_uuid(dataset), request_id="R0"
    )

    assert (refused.kind, refused.detail) == ("not_found", {"request_id": "R0"})


def test_a_preview_of_another_kind_is_invalid(
    backend: LocalBackend, dataset: Dataset
) -> None:
    refused = _refusal(
        backend,
        "preview_output",
        dataset=_uuid(dataset),
        ref="trained-model/x",
        group="",
        sequence="seq_a",
    )

    assert refused.kind == "invalid"


def test_a_damaged_request_file_is_reported_by_every_tool_that_reads_it(
    backend: LocalBackend, dataset: Dataset
) -> None:
    damaged = request_path(dataset.base_dir, "R0")
    damaged.parent.mkdir(parents=True)
    _ = damaged.write_text('{"bad": 1}')
    uuid = _uuid(dataset)

    refusals = [
        _refusal(backend, "list_activity", dataset=uuid),
        _refusal(backend, "get_request_status", dataset=uuid, request_id="R0"),
        _refusal(backend, "cancel_request", dataset=uuid, request_id="R0"),
        _refusal(backend, "diagnose_failures", dataset=uuid, request_id="R0"),
    ]

    for refused in refusals:
        assert (refused.kind, refused.reason) == ("unavailable", "request_unreadable")
        assert refused.detail == {"request_id": "R0", "path": str(damaged)}


@pytest.mark.parametrize(
    ("tool", "field"),
    [
        ("list_activity", "cursor"),
        ("get_request_status", "request_id"),
        ("cancel_request", "request_id"),
        ("diagnose_failures", "execution_id"),
    ],
)
def test_an_id_that_is_not_one_file_name_stem_is_invalid(
    backend: LocalBackend, dataset: Dataset, tool: str, field: str
) -> None:
    refused = _refusal(
        backend, tool, **{"dataset": _uuid(dataset), field: "../../outside"}
    )

    assert (refused.kind, refused.reason) == ("invalid", "invalid_arguments")
