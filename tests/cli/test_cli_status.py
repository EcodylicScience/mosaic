"""``mosaic status`` and ``mosaic pipeline status`` print the status wire models."""

from __future__ import annotations

import json
from pathlib import Path

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    Recipe,
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
