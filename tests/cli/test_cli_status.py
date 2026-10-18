"""``mosaic status``, ``mosaic pipeline status|list`` and ``mosaic runs`` print the status wires."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from pydantic import TypeAdapter
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    MAX_WAIT_SECONDS,
    EntryErrorsWire,
    FileFailureStore,
    Recipe,
    RequestPageWire,
    RequestStatusWire,
    RunStatusWire,
    request_status_wire,
    submit_request,
)
from mosaic.core.pipeline.refusal import Refusal
from mosaic.runlog import JsonlRunLog, run_log_path
from tests.helpers import invoke_json, invoke_ok

RECIPE: dict[str, object] = {
    "schema_version": 1,
    "steps": [
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "inputs": ["tracks"],
        }
    ],
}


def test_status_prints_a_run_status_report(dataset: tuple[Path, Dataset]) -> None:
    manifest, _ = dataset
    ran = invoke_json(
        ["run", "-m", str(manifest), "--feature", "speed-angvel", "--json"]
    )

    payload = invoke_json(
        [
            "status",
            "-m",
            str(manifest),
            "--execution-id",
            str(ran["execution_id"]),
            "--json",
        ]
    )

    report = RunStatusWire.model_validate(payload)
    assert (report.status, report.outcome, report.terminal) == (
        "finished",
        "finished",
        True,
    )
    assert report.run_id == ran["run_id"]
    assert report.entries_written == 2
    assert report.failed_entries == ()
    assert report.error is None


def test_pipeline_status_prints_a_request_status_report(
    dataset: tuple[Path, Dataset],
) -> None:
    manifest, ds = dataset
    submitted = submit_request(ds, Recipe.model_validate(RECIPE))
    command = [
        "pipeline",
        "status",
        "-m",
        str(manifest),
        "--request",
        submitted.request.request_id,
        "--json",
    ]

    before = RequestStatusWire.model_validate(invoke_json(command))
    (step,) = before.steps
    assert (before.status, before.verdict) == ("running", "running")
    assert (step.step, step.status, step.outcome) == ("speed", "", "pending")

    _ = invoke_json(
        [
            "run",
            "-m",
            str(manifest),
            "--graph-request",
            submitted.request.request_id,
            "--step",
            "speed",
            "--execution-id",
            step.execution_id,
            "--json",
        ]
    )

    after = RequestStatusWire.model_validate(invoke_json(command))
    assert (after.status, after.verdict, after.terminal) == (
        "finished",
        "finished",
        True,
    )
    assert after.steps[0].outcome == "finished"
    assert after.steps[0].entries_written == 2


def test_pipeline_status_prints_phase_and_liveness(
    dataset: tuple[Path, Dataset],
) -> None:
    manifest, ds = dataset
    request = submit_request(ds, Recipe.model_validate(RECIPE)).request
    execution_id = request.execution_of("speed")
    with JsonlRunLog(run_log_path(ds.base_dir, execution_id), execution_id) as log:
        log.started(
            kind="feature",
            target="speed",
            host="another-host.invalid",
            pid=7,
            liveness_interval=60.0,
        )
        log.on_phase("fit", "2 entries")

    result = invoke_ok(
        ["pipeline", "status", "-m", str(manifest), "--request", request.request_id]
    )

    (row,) = [line for line in result.stdout.splitlines() if line.startswith("speed")]
    assert row.split()[-2:] == ["fit", "alive"]
    assert "note: poll again in 10 s" in result.stderr


def test_pipeline_status_brief_leaves_out_the_steps_that_have_not_started(
    dataset: tuple[Path, Dataset],
) -> None:
    manifest, ds = dataset
    request = submit_request(ds, Recipe.model_validate(RECIPE)).request
    command = [
        "pipeline",
        "status",
        "-m",
        str(manifest),
        "--request",
        request.request_id,
        "--brief",
    ]

    brief = RequestStatusWire.model_validate(invoke_json([*command, "--json"]))
    plain = invoke_ok(command)

    assert brief == request_status_wire(ds.base_dir, request, brief=True)
    assert brief.steps == ()
    assert (brief.omitted_finished, brief.omitted_pending) == (0, 1)
    assert "note: 0 finished and 1 not started step(s) are not listed" in plain.stderr


def test_pipeline_status_waits_until_a_step_ends_and_prints_the_same_model(
    dataset: tuple[Path, Dataset],
) -> None:
    manifest, ds = dataset
    request = submit_request(ds, Recipe.model_validate(RECIPE)).request
    execution_id = request.execution_of("speed")

    def run_the_step() -> None:
        with JsonlRunLog(run_log_path(ds.base_dir, execution_id), execution_id) as log:
            log.started(kind="feature", target="speed")
            log.finished()

    ender = threading.Timer(0.3, run_the_step)
    ender.start()
    began = time.monotonic()
    try:
        waited = RequestStatusWire.model_validate(
            invoke_json(
                [
                    "pipeline",
                    "status",
                    "-m",
                    str(manifest),
                    "--request",
                    request.request_id,
                    "--wait-seconds",
                    "30",
                    "--brief",
                    "--json",
                ]
            )
        )
    finally:
        ender.join()

    assert time.monotonic() - began < 20
    assert (waited.verdict, waited.terminal) == ("finished", True)
    assert [(step.step, step.outcome) for step in waited.steps] == [
        ("speed", "finished")
    ]
    assert waited == request_status_wire(ds.base_dir, request)


def test_pipeline_status_refuses_a_wait_above_the_cap(
    dataset: tuple[Path, Dataset],
) -> None:
    manifest, ds = dataset
    request = submit_request(ds, Recipe.model_validate(RECIPE)).request

    result = CliRunner().invoke(
        app,
        [
            "pipeline",
            "status",
            "-m",
            str(manifest),
            "--request",
            request.request_id,
            "--wait-seconds",
            str(MAX_WAIT_SECONDS + 1),
        ],
    )

    assert result.exit_code == 2
    assert f"0<=x<={MAX_WAIT_SECONDS}" in result.output


def test_plain_status_prints_the_error_and_each_failed_entry_as_text(
    dataset: tuple[Path, Dataset],
) -> None:
    manifest, ds = dataset
    crash = json.dumps({"type": "ValueError", "message": "no frames"})
    refusal = Refusal("short by 2", reason="coverage_shortfall", step_id="fit")
    with JsonlRunLog(run_log_path(ds.base_dir, "E1"), "E1") as log:
        log.started(kind="feature", target="speed")
        log.entry_failed("g__s1", crash)
        log.failed(refusal.error_json())

    lines = invoke_ok(
        ["status", "-m", str(manifest), "--execution-id", "E1"]
    ).stdout.splitlines()

    assert "error: refused (coverage_shortfall): short by 2" in lines
    assert "failed entry g__s1: ValueError: no frames" in lines
    assert not any(line.startswith(("progress:", "failed_entries:")) for line in lines)


def test_pipeline_list_prints_a_page_of_requests_newest_first(
    dataset: tuple[Path, Dataset],
) -> None:
    manifest, ds = dataset
    newer, older = sorted(
        (submit_request(ds, Recipe.model_validate(RECIPE)).request for _ in range(2)),
        key=lambda request: request.request_id,
        reverse=True,
    )
    command = ["pipeline", "list", "-m", str(manifest), "--json"]

    first = RequestPageWire.model_validate(invoke_json([*command, "--limit", "1"]))
    assert first.next_after is not None
    second = RequestPageWire.model_validate(
        invoke_json([*command, "--limit", "1", "--after", first.next_after])
    )

    assert [item.request_id for item in first.items] == [newer.request_id]
    assert [item.request_id for item in second.items] == [older.request_id]
    assert second.next_after is None
    (item,) = second.items
    assert (item.step_count, item.verdict) == (1, "running")
    assert item.recipe_digest == older.recipe_digest


def test_plain_pipeline_list_prints_a_row_per_request_and_the_next_page(
    dataset: tuple[Path, Dataset],
) -> None:
    manifest, ds = dataset
    for _ in range(2):
        _ = submit_request(ds, Recipe.model_validate(RECIPE))

    result = invoke_ok(["pipeline", "list", "-m", str(manifest), "--limit", "1"])

    header, row = result.stdout.splitlines()
    assert header.split("\t") == [
        "request_id",
        "created_at",
        "recipe",
        "steps",
        "verdict",
    ]
    assert row.endswith("\t1\trunning")
    assert "--after" in result.stderr


def _two_runs(manifest: Path) -> list[str]:
    """Two attempts' execution ids, the older first by id."""
    return sorted(
        str(
            invoke_json(
                ["run", "-m", str(manifest), "--feature", "speed-angvel", "--json"]
            )["execution_id"]
        )
        for _ in range(2)
    )


def test_runs_with_a_limit_prints_a_page_object(dataset: tuple[Path, Dataset]) -> None:
    manifest, _ = dataset
    older, newer = _two_runs(manifest)
    command = ["runs", "-m", str(manifest), "--json", "--limit", "1"]

    first = invoke_json(command)
    second = invoke_json([*command, "--after", str(first["next_after"])])

    assert set(first) == {"items", "next_after"}
    assert [run["execution_id"] for run in _rows(first)] == [newer]
    assert first["next_after"] == newer
    assert [run["execution_id"] for run in _rows(second)] == [older]
    assert second["next_after"] is None


def test_runs_without_paging_flags_prints_the_array_of_every_run(
    dataset: tuple[Path, Dataset],
) -> None:
    manifest, _ = dataset
    older, newer = _two_runs(manifest)

    rows = json.loads(invoke_ok(["runs", "-m", str(manifest), "--json"]).stdout)

    assert [run["execution_id"] for run in rows] == [newer, older]


_ROWS = TypeAdapter(list[dict[str, object]])
"""A page's ``items``, each run an object."""


def _rows(page: dict[str, object]) -> list[dict[str, object]]:
    return _ROWS.validate_python(page["items"])


def _lost_two(ds: Dataset) -> None:
    """Attempt E1 of a run that lost two entities, the first failed once since."""
    crash = json.dumps({"type": "ValueError", "message": "no frames"})
    with JsonlRunLog(run_log_path(ds.base_dir, "E1"), "E1") as log:
        log.started(kind="feature", target="speed")
        log.set_run_id("0.1-abc")
        log.entry_failed("g__s1", crash)
        log.entry_failed("g__s2", crash)
        log.finished()
    _ = FileFailureStore(ds.base_dir).note_entry_failure(
        ("speed", "0.1-abc", "g", "s1"), error="no frames"
    )


def test_runs_errors_prints_the_entry_errors_wire(
    dataset: tuple[Path, Dataset],
) -> None:
    manifest, ds = dataset
    _lost_two(ds)

    report = EntryErrorsWire.model_validate(
        invoke_json(["runs", "errors", "E1", "-m", str(manifest), "--json"])
    )

    assert (report.total, report.has_more, report.outcome) == (2, False, "partial")
    assert [(e.sequence, e.retry) for e in report.entries] == [
        ("s1", "waiting"),
        ("s2", "unrecorded"),
    ]


def test_runs_errors_takes_the_manifest_given_before_it_and_a_limit(
    dataset: tuple[Path, Dataset],
) -> None:
    manifest, ds = dataset
    _lost_two(ds)

    result = invoke_ok(["runs", "-m", str(manifest), "errors", "E1", "--limit", "1"])

    lines = result.stdout.splitlines()
    assert "total: 2" in lines
    assert lines[-2].split("\t") == [
        "key",
        "retry",
        "attempts",
        "not_before",
        "error",
    ]
    key, retry, attempts, _, error = lines[-1].split("\t")
    assert (key, retry, attempts, error) == (
        "g__s1",
        "waiting",
        "1",
        "ValueError: no frames",
    )
    assert "1 of 2 listed" in result.stderr


def test_runs_errors_of_an_unknown_attempt_fails(dataset: tuple[Path, Dataset]) -> None:
    manifest, _ = dataset

    result = CliRunner().invoke(app, ["runs", "errors", "MISSING", "-m", str(manifest)])

    assert result.exit_code == 1
    assert "No run found with execution_id=MISSING" in result.stderr


def test_a_listing_option_before_runs_errors_is_refused(
    dataset: tuple[Path, Dataset],
) -> None:
    manifest, _ = dataset

    result = CliRunner().invoke(
        app, ["runs", "-m", str(manifest), "--limit", "1", "errors", "E1"]
    )

    assert result.exit_code == 2
    assert "--limit given before 'errors'" in result.stderr
