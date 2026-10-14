"""``mosaic status``, ``mosaic pipeline status|list`` and ``mosaic runs`` print the status wires."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    EntryErrorsWire,
    FileFailureStore,
    Recipe,
    RequestPageWire,
    RequestStatusWire,
    RunStatusWire,
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


def _rows(page: dict[str, object]) -> list[dict[str, object]]:
    items = page["items"]
    assert isinstance(items, list)
    return [row for row in items if isinstance(row, dict)]


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
