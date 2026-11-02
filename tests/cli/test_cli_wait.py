"""``mosaic wait`` prints each request's verdict and exits by how the wait ended."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.cli.wait import WAIT_TIMED_OUT_EXIT_CODE
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    Request,
    RequestWaitWire,
    request_path,
    save_request,
)
from mosaic.runlog import JsonlRunLog, run_log_path
from tests.helpers import make_dataset


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """Three requests of one step each: finished, failed and running."""
    ds = make_dataset(tmp_path / "ds")
    for request_id, ending in (("R1", "finished"), ("R2", "failed"), ("R3", None)):
        execution_id = f"E{request_id}"
        _ = save_request(
            ds.base_dir,
            Request(
                request_id=request_id,
                recipe_digest="d" * 64,
                step_executions={"speed": execution_id},
            ),
        )
        path = run_log_path(ds.base_dir, execution_id)
        with JsonlRunLog(path, execution_id) as log:
            log.started(kind="feature", target="speed")
            if ending == "finished":
                log.finished()
            elif ending == "failed":
                log.failed('{"type": "ValueError", "message": "bad"}')
    return ds


def _wait(dataset: Dataset, *arguments: str) -> tuple[int, str]:
    result = CliRunner().invoke(
        app, ["wait", *arguments, "--manifest", str(dataset.manifest_path)]
    )
    return result.exit_code, result.stdout


def test_a_finished_request_exits_zero_with_its_verdict(dataset: Dataset) -> None:
    code, stdout = _wait(dataset, "R1", "--json")
    waited = RequestWaitWire.model_validate(json.loads(stdout))

    assert code == 0
    assert waited.outcome == "finished"
    assert [(report.request_id, report.verdict) for report in waited.requests] == [
        ("R1", "finished")
    ]


def test_a_failed_request_exits_one(dataset: Dataset) -> None:
    code, stdout = _wait(dataset, "R1", "R2")

    assert code == 1
    assert "request R1: finished" in stdout
    assert "request R2: failed" in stdout


def test_any_exits_with_the_first_request_that_ended(dataset: Dataset) -> None:
    code, stdout = _wait(dataset, "R3", "R1", "--until", "any", "--timeout", "0")

    assert code == 0
    assert "request R3: running" in stdout


def test_a_wait_that_runs_out_exits_with_its_own_code(dataset: Dataset) -> None:
    code, stdout = _wait(dataset, "R1", "R3", "--timeout", "0", "--json")
    waited = RequestWaitWire.model_validate(json.loads(stdout))

    assert code == WAIT_TIMED_OUT_EXIT_CODE
    assert waited.outcome == "timed_out"
    assert [report.verdict for report in waited.requests] == ["finished", "running"]


def test_an_unknown_request_is_refused(dataset: Dataset) -> None:
    code, _ = _wait(dataset, "R1", "NOPE")

    assert code == 2


def test_an_unreadable_request_is_refused_naming_its_file(dataset: Dataset) -> None:
    damaged = request_path(dataset.base_dir, "R9")
    _ = damaged.write_text('{"bad": 1}')

    result = CliRunner().invoke(
        app, ["wait", "R1", "R9", "--manifest", str(dataset.manifest_path)]
    )

    assert result.exit_code == 2
    assert str(damaged) in result.stderr
