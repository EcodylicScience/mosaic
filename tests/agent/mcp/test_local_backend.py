"""Every shared tool answered by the local backend over a fixture dataset.

The submission test starts a real detached driver, which runs each step as a
process of its own, and polls the request as a client would.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.agent.mcp.local import POLL_AFTER_MIN_SECONDS, LocalBackend
from mosaic.agent.mcp.outputs import StepSuggestions, StepTypes
from mosaic.agent.mcp.summaries import summarize
from mosaic.agent.mcp.spec import (
    LISTING_MISSING_SAMPLE,
    MAX_PAGE_SIZE,
    MAX_RESULT_CHARS,
    MAX_STEP_TYPE_DETAIL_CHARS,
    PLAN_MISSING_SAMPLE,
    PREVIEW_MAX_COLUMNS,
    RECIPE_EXAMPLE,
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
    POLL_AFTER_MIN_SECONDS as GRAPH_POLL_AFTER_MIN_SECONDS,
)
from mosaic.core.pipeline.graph import (
    QUARANTINE_AFTER,
    CoverageShortfall,
    FileFailureStore,
    PlanWire,
    RequestStatusWire,
    Recipe,
    ValidationReport,
    open_requests_matching,
    request_path,
    run_pipeline,
    submit_request,
)
from mosaic.core.pipeline.inventory import InventoryWire
from mosaic.core.pipeline.media_index import frame_from_rows, write_media_index_rows
from mosaic.core.pipeline.markers import process_running
from mosaic.core.pipeline.parallelism import ParallelLimits, host_cpus
from mosaic.core.pipeline.preview import ArtifactPreview
from mosaic.core.scope import Scope
from mosaic.runlog import JsonlRunLog, run_log_path
from tests.helpers import (
    CHAIN_RECIPE,
    CONVERT_TRACKS_STEP,
    SPEED_OF_CONVERTED_STEP,
    SPEED_RECIPE,
    MediaClip,
    RecordingStarter,
    add_tracks_table,
    add_tracks_variant,
    calms21_dataset,
    chain_recipe,
    exited_pid,
    make_dataset,
    speed_recipe,
    track_table,
    wait_for,
    write_media_index,
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


def test_a_media_only_entry_is_listed_with_null_tracks_fields(
    dataset: Dataset,
) -> None:
    """An entry the media index names and no tracks table covers is listed too."""
    write_media_index(dataset, [MediaClip(sequence="pilot")])
    served = LocalBackend([dataset.manifest_path], warm=False)

    page = _call(served, "list_sequences", SequencePage, dataset=_uuid(dataset))
    pinned = _call(
        served,
        "list_sequences",
        SequencePage,
        dataset=_uuid(dataset),
        tracks_variant=VARIANT,
    )

    pilot = next(row for row in page.items if row.sequence == "pilot")
    assert (pilot.group, pilot.recording) == ("", "pilot")
    assert (pilot.tracks_variant, pilot.producer, pilot.std_format) == (
        None,
        None,
        None,
    )
    assert (pilot.n_rows, pilot.frame_min, pilot.frame_max) == (None, None, None)
    assert page.total == 3
    assert [row.sequence for row in pinned.items] == ["seq_a", "seq_b"]


def test_the_sequences_and_the_summary_name_each_entry_s_recording(
    dataset: Dataset,
) -> None:
    write_media_index(
        dataset,
        [
            MediaClip(group="night", sequence="s1", filename="night_s1.mp4"),
            MediaClip(group="night", sequence="s2", filename="night_s2.mp4"),
        ],
    )
    for sequence in ("s1", "s2"):
        add_tracks_table(dataset, VARIANT, track_table("night", sequence))
    dataset.set_continuous_groups(["night"])
    served = LocalBackend([dataset.manifest_path], warm=False)

    page = _call(served, "list_sequences", SequencePage, dataset=_uuid(dataset))
    summary = _call(served, "describe_dataset", DatasetSummary, dataset=_uuid(dataset))

    assert [(row.sequence, row.recording) for row in page.items] == [
        ("seq_a", "seq_a"),
        ("seq_b", "seq_b"),
        ("s1", "night"),
        ("s2", "night"),
    ]
    assert (summary.counts.recordings, summary.counts.divided_recordings) == (3, 1)
    assert [(recording.name, recording.kind) for recording in summary.recordings] == [
        ("seq_a", "undivided"),
        ("seq_b", "undivided"),
        ("night", "divided"),
    ]
    assert "3 recording(s) (1 divided)" in summarize(summary)


def test_a_group_s_sequences_are_listed_with_its_media_only_entries(
    dataset: Dataset,
) -> None:
    """A divided recording's sequences are read from the listing, by its group."""
    write_media_index(
        dataset,
        [
            MediaClip(
                group="night", sequence=sequence, filename=f"night_{sequence}.mp4"
            )
            for sequence in ("s1", "s2", "s3")
        ],
    )
    for sequence in ("s1", "s2"):
        add_tracks_table(dataset, VARIANT, track_table("night", sequence))
    dataset.set_continuous_groups(["night"])
    served = LocalBackend([dataset.manifest_path], warm=False)

    first = _call(
        served,
        "list_sequences",
        SequencePage,
        dataset=_uuid(dataset),
        group="night",
        limit=2,
    )
    rest = _call(
        served,
        "list_sequences",
        SequencePage,
        dataset=_uuid(dataset),
        group="night",
        cursor=first.next_cursor,
    )

    rows = [*first.items, *rest.items]
    assert [(row.group, row.sequence, row.recording) for row in rows] == [
        ("night", "s1", "night"),
        ("night", "s2", "night"),
        ("night", "s3", "night"),
    ]
    assert [row.frame_min is not None for row in rows] == [True, True, False]
    assert (first.total, rest.next_cursor) == (3, None)


SUMMARY_RECORDINGS = 600
"""The recordings of the dataset that ``describe_dataset`` must describe whole."""


def test_a_summary_of_hundreds_of_recordings_is_returned_whole(
    tmp_path: Path,
) -> None:
    """The media index names each sequence, and no media file is written."""
    ds = make_dataset(tmp_path / "many")
    divided = [(f"night{n}", f"s{m:02d}") for n in range(2) for m in range(50)]
    undivided = [("", f"session_{n:04d}") for n in range(SUMMARY_RECORDINGS - 2)]
    write_media_index_rows(
        ds.get_root("media_raw") / "index.csv",
        frame_from_rows(
            [
                {"group": group, "sequence": sequence, "fps": 30.0}
                for group, sequence in [*divided, *undivided]
            ]
        ),
    )
    ds.set_continuous_groups(["night0", "night1"])
    served = LocalBackend([ds.manifest_path], warm=False)

    summary = _call(served, "describe_dataset", DatasetSummary, dataset=_uuid(ds))

    assert summary.counts.recordings == len(summary.recordings) == SUMMARY_RECORDINGS
    assert len(summary.model_dump_json()) <= MAX_RESULT_CHARS


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
    found = _call(backend, "describe_step_types", StepTypes, query="speed angvel")
    ops = _call(backend, "describe_step_types", StepTypes, type="op", limit=3)
    more = _call(
        backend,
        "describe_step_types",
        StepTypes,
        type="op",
        cursor=ops.next_cursor,
        limit=MAX_PAGE_SIZE,
    )

    assert "speed-angvel" in {entry.name for entry in found.entries}
    assert len(ops.entries) == 3
    assert ops.total == len(ops.entries) + len(more.entries)
    assert {entry.type for entry in (*ops.entries, *more.entries)} == {"op"}


def test_step_types_are_ranked_and_paged_in_rank_order(backend: LocalBackend) -> None:
    whole = _call(backend, "describe_step_types", StepTypes, query="speed", limit=100)
    paged: list[str] = []
    cursor: str | None = None
    while True:
        page = _call(
            backend,
            "describe_step_types",
            StepTypes,
            query="speed",
            limit=1,
            cursor=cursor,
        )
        paged.extend(entry.name for entry in page.entries)
        cursor = page.next_cursor
        if cursor is None:
            break

    assert whole.entries[0].name == "speed-angvel"
    assert len(whole.entries) > 1
    assert paged == [entry.name for entry in whole.entries]


def test_named_step_types_are_described_as_param_tables(
    backend: LocalBackend,
) -> None:
    found = _call(
        backend, "describe_step_types", StepTypes, names=["speed-angvel", "trex"]
    )
    schema = _call(
        backend,
        "describe_step_types",
        StepTypes,
        names=["speed-angvel"],
        detail="schema",
    )
    rows = _call(
        backend, "describe_step_types", StepTypes, names=["trex"], detail="summary"
    )
    missing = _refusal(backend, "describe_step_types", names=["no-such-step"])

    assert found.detail == "fields"
    assert [detail.name for detail in found.details] == ["speed-angvel", "trex"]
    assert "step_size" in {row.path for row in found.details[0].params}
    assert found.details[0].params_schema is None
    assert found.entries == ()
    assert schema.details[0].params_schema is not None
    assert [entry.name for entry in rows.entries] == ["trex"]
    assert rows.details == ()
    assert missing.kind == "not_found"
    assert missing.detail == {"names": ["no-such-step"]}


def test_unknown_names_are_reported_beside_the_known(backend: LocalBackend) -> None:
    found = _call(
        backend,
        "describe_step_types",
        StepTypes,
        names=["no-such-step", "speed-angvel"],
    )

    assert found.unknown == ("no-such-step",)
    assert [detail.name for detail in found.details] == ["speed-angvel"]
    assert found.total == 1


def test_every_step_type_detail_stays_under_its_budget(backend: LocalBackend) -> None:
    every = _call(backend, "describe_step_types", StepTypes, limit=100)
    names = [entry.name for entry in every.entries]
    while every.next_cursor is not None:
        every = _call(
            backend, "describe_step_types", StepTypes, cursor=every.next_cursor
        )
        names.extend(entry.name for entry in every.entries)

    over = {
        name: size
        for name in names
        if (
            size := len(
                _call(backend, "describe_step_types", StepTypes, names=[name])
                .details[0]
                .model_dump_json()
            )
        )
        > MAX_STEP_TYPE_DETAIL_CHARS
    }

    assert len(names) > 50
    assert not over, f"step types over {MAX_STEP_TYPE_DETAIL_CHARS} characters: {over}"


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


def test_validate_plan_and_a_cli_file_name_one_digest(
    backend: LocalBackend, dataset: Dataset
) -> None:
    """A client sends ``1`` where a hand-written file says ``1.0``: one recipe."""
    as_sent = json.loads(json.dumps(SPEED_RECIPE))
    as_sent["steps"][0]["params"] = {"step_size": 1}
    as_written = json.loads(json.dumps(SPEED_RECIPE))
    as_written["steps"][0]["params"] = {"step_size": 1.0}
    submitted = submit_request(dataset, Recipe.model_validate(as_written))

    valid = _call(backend, "validate_recipe", ValidationReport, recipe=as_sent)
    plan = _call(
        backend, "plan_recipe", PlanWire, dataset=_uuid(dataset), recipe=as_sent
    )

    assert valid.digest == plan.recipe_digest == submitted.request.recipe_digest
    assert open_requests_matching(dataset.base_dir, plan.recipe_digest, Scope()) == (
        submitted.request.request_id,
    )


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


def test_a_plan_lists_the_media_that_awaits_an_analysis_transcode(
    tmp_path: Path,
) -> None:
    awaiting = make_dataset(tmp_path / "awaiting", name="awaiting")
    write_media_index(
        awaiting, [MediaClip(sequence="vid1", video_uuid="u1", rotation=90)]
    )
    backend = LocalBackend([awaiting.manifest_path], warm=False)
    tracker = {
        "id": "trex",
        "type": "op",
        "kind": "trex",
        "params": {"track_max_individuals": 2},
    }

    plan = _call(
        backend,
        "plan_recipe",
        PlanWire,
        dataset=_uuid(awaiting),
        recipe={"steps": [tracker]},
    )

    assert [(problem.step, problem.where) for problem in plan.problems] == [
        ("trex", "after")
    ]


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


def test_a_submission_with_workers_runs_them_and_reports_the_memory_used(
    dataset: Dataset,
) -> None:
    """The workers travel from the tool to the step process and back into its status."""
    backend = LocalBackend(
        [dataset.manifest_path],
        warm=False,
        parallel_limits=ParallelLimits(max_workers=2),
    )
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
        workers=2,
    )

    assert submitted.workers == 2
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
    wait_for(
        lambda: not process_running(submitted.driver_pid),
        REQUEST_SECONDS,
        "the detached driver never exited",
    )
    assert status.verdict == "finished", status
    speed, templates = status.steps
    assert speed.workers == min(2, host_cpus())
    assert speed.peak_rss_bytes > 0
    # A fit across every entry is not parallelizable, so it runs in turn.
    assert templates.workers == 1


def test_plan_and_submit_a_recipe_whose_first_step_converts(tmp_path: Path) -> None:
    """Conversion is a step, so one request converts and computes on the result."""
    converting = calms21_dataset(tmp_path / "calms21", {"train": {"a": {"s1": 20}}})
    backend = LocalBackend([converting.manifest_path], warm=False)
    uuid = _uuid(converting)
    recipe = {
        "schema_version": 1,
        "name": "converting",
        "steps": [CONVERT_TRACKS_STEP, SPEED_OF_CONVERTED_STEP],
    }
    entries = [["train", "s1"]]
    plan = _call(
        backend, "plan_recipe", PlanWire, dataset=uuid, recipe=recipe, entries=entries
    )
    assert [(step.step, step.run_id is not None) for step in plan.steps] == [
        ("tracks", True),
        ("speed", True),
    ]

    submitted = _call(
        backend,
        "submit_pipeline",
        SubmittedPipeline,
        dataset=uuid,
        recipe=recipe,
        entries=entries,
        plan_fingerprint=plan.fingerprint,
    )

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
    wait_for(
        lambda: not process_running(submitted.driver_pid),
        REQUEST_SECONDS,
        "the detached driver never exited",
    )
    status = _call(
        backend,
        "get_request_status",
        RequestStatusWire,
        dataset=uuid,
        request_id=submitted.request_id,
    )
    assert status.verdict == "finished", status
    assert [step.run_id for step in status.steps] == [
        step.run_id for step in plan.steps
    ]


SPLIT_RECIPE: dict[str, object] = {
    "schema_version": 1,
    "name": "split",
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
"""Speed over every entry and templates fitted on part of them."""


@pytest.fixture
def split(tmp_path: Path) -> Dataset:
    """Converted tracks of a train group of two sequences and a test group of one."""
    ds = calms21_dataset(
        tmp_path / "split",
        {"train": {"a": {"s1": 20, "s2": 20}}, "test": {"a": {"s3": 20}}},
        seed=4,
    )
    _ = ds.convert_all_tracks()
    return ds


def test_plan_recipe_step_entries_fingerprint_matches_submit(split: Dataset) -> None:
    starter = RecordingStarter()
    backend = LocalBackend([split.manifest_path], warm=False, start_driver=starter)
    uuid = _uuid(split)
    arguments: dict[str, object] = {
        "dataset": uuid,
        "recipe": SPLIT_RECIPE,
        "entries": {"groups": ["train", "test"]},
        "step_entries": {"templates": {"groups": ["train"]}},
    }
    plan = _call(backend, "plan_recipe", PlanWire, **arguments)
    other = _call(
        backend,
        "plan_recipe",
        PlanWire,
        **{**arguments, "step_entries": {"templates": {"groups": ["test"]}}},
    )

    submitted = _call(
        backend,
        "submit_pipeline",
        SubmittedPipeline,
        **arguments,
        plan_fingerprint=plan.fingerprint,
    )

    assert plan.fingerprint != other.fingerprint
    assert submitted.plan_fingerprint == plan.fingerprint
    assert [step.coverage.target for step in plan.steps] == [3, 2]
    assert starter.started == [submitted.request_id]


def test_submit_pipeline_records_step_scopes(split: Dataset) -> None:
    from mosaic.core.pipeline.graph import load_request

    backend = LocalBackend(
        [split.manifest_path], warm=False, start_driver=RecordingStarter()
    )
    uuid = _uuid(split)
    step_entries = {"templates": [["train", "s1"], ["train", "s2"]]}
    plan = _call(
        backend,
        "plan_recipe",
        PlanWire,
        dataset=uuid,
        recipe=SPLIT_RECIPE,
        entries={"groups": ["train", "test"]},
        step_entries=step_entries,
    )

    submitted = _call(
        backend,
        "submit_pipeline",
        SubmittedPipeline,
        dataset=uuid,
        recipe=SPLIT_RECIPE,
        entries={"groups": ["train", "test"]},
        step_entries=step_entries,
        plan_fingerprint=plan.fingerprint,
    )

    request = load_request(split.base_dir, submitted.request_id)
    assert request.scope == Scope(groups=["train", "test"])
    assert request.step_scopes == {
        "templates": Scope(entries=[("train", "s1"), ("train", "s2")])
    }


def test_get_inventory_accepts_a_groups_selector(split: Dataset) -> None:
    backend = LocalBackend([split.manifest_path], warm=False)

    listing = _call(
        backend,
        "get_inventory",
        InventoryWire,
        dataset=_uuid(split),
        kinds=["tracks-variant"],
        entries={"groups": ["test"]},
        status_not=[],
    )

    (record,) = listing.artifacts
    assert record.coverage.target == 1


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


def _dead_step(dataset: Dataset, execution_id: str) -> None:
    """A step that started on this host as a process that has exited."""
    with JsonlRunLog(run_log_path(dataset.base_dir, execution_id), execution_id) as log:
        log.started(
            kind="feature", target=STORAGE, host=socket.gethostname(), pid=exited_pid()
        )
        log.on_phase("fit", "2 entries")


def test_the_runs_view_reports_each_attempts_liveness(
    backend: LocalBackend, dataset: Dataset
) -> None:
    request = submit_request(dataset, chain_recipe()).request
    speed, templates = request.execution_of("speed"), request.execution_of("templates")
    _dead_step(dataset, speed)
    with JsonlRunLog(run_log_path(dataset.base_dir, templates), templates) as log:
        log.started(kind="feature", target="extract-templates__from__speed")
        log.finished()

    page = _call(
        backend, "list_activity", ActivityPage, dataset=_uuid(dataset), view="runs"
    )

    liveness = {run.execution_id: (run.liveness, run.phase) for run in page.runs}
    assert liveness == {speed: ("dead", "fit"), templates: ("ended", "")}
    assert all(run.host for run in page.runs if run.execution_id == speed)


def test_a_dead_request_reads_alike_in_status_activity_and_plan(
    backend: LocalBackend, dataset: Dataset
) -> None:
    """No driver lease, as for a request run by 'mosaic pipeline run --recipe'."""
    request = submit_request(dataset, speed_recipe()).request
    _dead_step(dataset, request.execution_of("speed"))

    status = _call(
        backend,
        "get_request_status",
        RequestStatusWire,
        dataset=_uuid(dataset),
        request_id=request.request_id,
    )
    activity = _call(
        backend, "list_activity", ActivityPage, dataset=_uuid(dataset), view="requests"
    )
    plan = _call(
        backend, "plan_recipe", PlanWire, dataset=_uuid(dataset), recipe=SPEED_RECIPE
    )

    assert status.verdict == "orphaned"
    assert [(item.request_id, item.verdict) for item in activity.requests] == [
        (request.request_id, "orphaned")
    ]
    assert [(item.request_id, item.verdict) for item in plan.open_requests] == [
        (request.request_id, "orphaned")
    ]


def test_request_status_reports_poll_after_seconds(
    backend: LocalBackend, dataset: Dataset
) -> None:
    request = submit_request(dataset, chain_recipe()).request

    status = _call(
        backend,
        "get_request_status",
        RequestStatusWire,
        dataset=_uuid(dataset),
        request_id=request.request_id,
    )

    assert status.poll_after_seconds == POLL_AFTER_MIN_SECONDS
    assert POLL_AFTER_MIN_SECONDS == GRAPH_POLL_AFTER_MIN_SECONDS


def test_a_status_call_waits_for_a_step_to_end_and_lists_only_what_changed(
    backend: LocalBackend, dataset: Dataset
) -> None:
    request = submit_request(dataset, chain_recipe()).request
    execution_id = request.execution_of("speed")

    def run_the_step() -> None:
        path = run_log_path(dataset.base_dir, execution_id)
        with JsonlRunLog(path, execution_id) as log:
            log.started(kind="feature", target=STORAGE)
            log.finished()

    ender = threading.Timer(1.0, run_the_step)
    ender.start()
    began = time.monotonic()
    try:
        status = _call(
            backend,
            "get_request_status",
            RequestStatusWire,
            dataset=_uuid(dataset),
            request_id=request.request_id,
            wait_seconds=30,
            brief=True,
        )
    finally:
        ender.join()

    assert time.monotonic() - began < 20
    assert (status.verdict, status.terminal) == ("running", False)
    assert [(step.step, step.outcome) for step in status.steps] == [
        ("speed", "finished")
    ]
    assert (status.omitted_finished, status.omitted_pending) == (0, 1)


def test_a_status_call_with_nothing_to_report_returns_when_its_wait_runs_out(
    backend: LocalBackend, dataset: Dataset
) -> None:
    request = submit_request(dataset, chain_recipe()).request
    execution_id = request.execution_of("speed")
    path = run_log_path(dataset.base_dir, execution_id)
    with JsonlRunLog(path, execution_id) as log:
        log.started(kind="feature", target=STORAGE)

    began = time.monotonic()
    status = _call(
        backend,
        "get_request_status",
        RequestStatusWire,
        dataset=_uuid(dataset),
        request_id=request.request_id,
        wait_seconds=1,
    )
    waited = time.monotonic() - began

    assert 0.9 <= waited < 10
    assert (status.verdict, status.terminal) == ("running", False)
    assert [(step.step, step.outcome) for step in status.steps] == [
        ("speed", "running"),
        ("templates", "pending"),
    ]


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


def test_the_recipe_example_validates(backend: LocalBackend) -> None:
    report = _call(backend, "validate_recipe", ValidationReport, recipe=RECIPE_EXAMPLE)

    assert report.valid, report.problems
    assert not report.cautions


def test_a_wide_feature_preview_stays_under_fifteen_thousand_characters(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "wide", name="wide")
    base = track_table("", "seq_a", n_rows=40, n_ids=2)
    payload = pd.DataFrame(
        {f"W_{i:03d}_f{i % 3}": np.linspace(0, 1, 40) for i in range(400)}
    )
    add_tracks_table(
        ds, VARIANT, pd.concat([base, payload], axis=1), std_format="mosaic_v1"
    )
    backend = LocalBackend([ds.manifest_path], warm=False)

    preview = _call(
        backend,
        "preview_output",
        ArtifactPreview,
        dataset=_uuid(ds),
        ref=f"tracks-variant/{VARIANT}",
        group="",
        sequence="seq_a",
    )

    assert len(preview.read_columns) == PREVIEW_MAX_COLUMNS
    assert {"frame", "id"} <= set(preview.read_columns)
    assert len(preview.columns) == PREVIEW_MAX_COLUMNS
    assert preview.columns_omitted_has_more
    assert len(preview.rows) == 5
    assert len(preview.model_dump_json()) < 15_000


def test_an_inventory_listing_leaves_params_to_the_ref(
    backend: LocalBackend, dataset: Dataset
) -> None:
    _ = run_pipeline(dataset, speed_recipe())
    uuid = _uuid(dataset)

    listed = _call(
        backend,
        "get_inventory",
        InventoryWire,
        dataset=uuid,
        kinds=["feature"],
        status_not=[],
    )
    [run] = listed.artifacts
    detailed = _call(backend, "get_inventory", InventoryWire, dataset=uuid, ref=run.ref)

    assert run.params is None
    assert run.params_state == "present"
    assert detailed.artifacts[0].params is not None


def test_a_plan_names_three_missing_entries_per_step(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "five", name="five")
    sequences = [f"seq_{index}" for index in range(LISTING_MISSING_SAMPLE + 1)]
    add_tracks_variant(ds, VARIANT, *sequences, std_format="mosaic_v1")
    backend = LocalBackend([ds.manifest_path], warm=False)

    plan = _call(
        backend, "plan_recipe", PlanWire, dataset=_uuid(ds), recipe=SPEED_RECIPE
    )

    [step] = plan.steps
    assert step.coverage.target == len(sequences)
    assert len(step.coverage.missing_sample) == PLAN_MISSING_SAMPLE
    assert step.coverage.has_more
