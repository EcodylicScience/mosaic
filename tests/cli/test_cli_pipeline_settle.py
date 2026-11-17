"""``mosaic pipeline settle``: record a step that needs no work, and report it.

The verb is a write verb that opens its dataset without creating the declared root
directories. Every outcome but ``settled`` leaves the dataset as it was.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import pytest
from pydantic import TypeAdapter
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.definitions.scope import clear_definition_caches
from mosaic.core.pipeline.graph import (
    Recipe,
    Request,
    execute_step,
    submit_request,
)
from mosaic.core.pipeline.graph.settle import StepSettleWire
from mosaic.core.pipeline.job import Cancelled
from mosaic.core.pipeline.refusal import REFUSED_EXIT_CODE
from mosaic.runlog import run_log_path
from tests.helpers import (
    CANARY_NAME,
    CANARY_SOURCE,
    CANARY_VARIABLE,
    add_tracks_variant,
    assert_tree_unchanged,
    chain_recipe,
    defined_dataset,
    forget_imported_definitions,
    imported_definitions,
    invoke_json,
    invoke_ok,
    make_dataset,
    tree_snapshot,
)

pytestmark = pytest.mark.usefixtures("fresh_definitions")

VARIANT = "convert-trex.0.2-1111111111"

SEQUENCES = ("seq_a", "seq_b")

PAYLOAD = TypeAdapter(dict[str, object])


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two tables of tracks, and nothing computed."""
    dataset = make_dataset(tmp_path / "tracked")
    for sequence in SEQUENCES:
        add_tracks_variant(
            dataset, VARIANT, sequence, n_rows=12, std_format="mosaic_v1"
        )
    return dataset


def _settle_args(dataset: Dataset, request: Request, *flags: str) -> list[str]:
    return [
        "pipeline",
        "settle",
        "--manifest",
        str(dataset.manifest_path),
        "--request",
        request.request_id,
        "--step",
        "speed",
        *flags,
    ]


def _remove_empty_roots(dataset: Dataset) -> list[Path]:
    """Remove each empty declared root, so that a verb that creates one adds a path."""
    empty_roots = [
        path
        for path in (dataset.get_root(key) for key in dataset.roots)
        if path.is_dir() and not any(path.iterdir())
    ]
    assert empty_roots
    for path in empty_roots:
        path.rmdir()
    return empty_roots


def _complete_step_and_a_new_request(tracked: Dataset) -> Request:
    """Compute ``speed`` under one request, and submit another that has not run."""
    recipe = chain_recipe()
    _ = execute_step(tracked, submit_request(tracked, recipe).request, "speed")
    return submit_request(tracked, recipe).request


def test_a_complete_step_settles_and_reports_its_size_and_the_outcome(
    tracked: Dataset,
) -> None:
    request = _complete_step_and_a_new_request(tracked)
    sized = invoke_json(
        [
            "pipeline",
            "input-size",
            "--manifest",
            str(tracked.manifest_path),
            "--request",
            request.request_id,
            "--step",
            "speed",
            "--json",
        ]
    )

    payload = invoke_json(_settle_args(tracked, request, "--owner", "gate", "--json"))

    assert {key: payload[key] for key in sized} == sized
    report = StepSettleWire.model_validate(payload)
    assert report.settle.outcome == "settled"
    assert report.settle.reason == ""
    assert report.settle.state == "cached"
    assert report.settle.run_id == sized["run_id"]
    assert report.settle.execution_id == request.execution_of("speed")
    assert run_log_path(tracked.base_dir, request.execution_of("speed")).exists()


def test_a_fresh_step_is_not_attempted_and_the_tree_is_unchanged(
    tracked: Dataset,
) -> None:
    request = submit_request(tracked, chain_recipe()).request
    before = tree_snapshot(tracked.base_dir)

    payload = invoke_json(_settle_args(tracked, request, "--json"))

    assert_tree_unchanged(tracked.base_dir, before)
    settle = StepSettleWire.model_validate(payload).settle
    assert (settle.outcome, settle.reason) == ("not_attempted", "entries_to_compute")


def test_a_step_with_a_run_log_is_declined_and_the_tree_is_unchanged(
    tracked: Dataset,
) -> None:
    recipe = chain_recipe()
    request = submit_request(tracked, recipe).request
    _ = execute_step(tracked, request, "speed")
    before = tree_snapshot(tracked.base_dir)

    payload = invoke_json(_settle_args(tracked, request, "--json"))

    assert_tree_unchanged(tracked.base_dir, before)
    settle = StepSettleWire.model_validate(payload).settle
    assert (settle.outcome, settle.reason) == ("declined", "attempt_recorded")


def test_a_refusal_of_the_size_read_exits_65_with_the_envelope(
    tracked: Dataset,
) -> None:
    request = submit_request(tracked, chain_recipe()).request
    log_path = run_log_path(tracked.base_dir, request.execution_of("speed"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _ = log_path.write_text(
        json.dumps({"t": "2026-01-01T00:00:00+00:00", "ev": "started"})
        + "\n"
        + json.dumps({"t": "2026-01-01T00:00:01+00:00", "ev": "finished"})
        + "\n"
    )
    before = tree_snapshot(tracked.base_dir)
    args = _settle_args(tracked, request, "--json")
    args[args.index("speed")] = "templates"

    result = CliRunner().invoke(app, args)

    assert result.exit_code == REFUSED_EXIT_CODE, result.output
    payload = PAYLOAD.validate_json(result.stdout)
    assert payload["status"] == "refused"
    assert payload["step"] == "templates"
    assert payload["execution_id"] == ""
    assert payload["reason"] == "parent_unrecorded"
    assert_tree_unchanged(tracked.base_dir, before)


def test_an_unknown_step_exits_1(tracked: Dataset) -> None:
    request = submit_request(tracked, chain_recipe()).request
    args = _settle_args(tracked, request)
    args[args.index("speed")] = "no-such-step"
    _ = _remove_empty_roots(tracked)
    before = tree_snapshot(tracked.base_dir)

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 1, result.output
    assert "no-such-step" in result.stderr
    assert_tree_unchanged(tracked.base_dir, before)


def test_an_unknown_request_exits_1(tracked: Dataset) -> None:
    request = submit_request(tracked, chain_recipe()).request
    args = _settle_args(tracked, request)
    args[args.index(request.request_id)] = "no-such-request"
    _ = _remove_empty_roots(tracked)
    before = tree_snapshot(tracked.base_dir)

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 1, result.output
    assert_tree_unchanged(tracked.base_dir, before)


def test_a_definition_step_settles_without_importing_its_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canary = tmp_path / "canary.jsonl"
    monkeypatch.setenv(CANARY_VARIABLE, str(canary))
    dataset = defined_dataset(tmp_path, {CANARY_NAME: CANARY_SOURCE})
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "definitions",
            "steps": [
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": f"def:{CANARY_NAME}",
                    "inputs": ["tracks"],
                }
            ],
        }
    )
    ran = execute_step(dataset, submit_request(dataset, recipe).request, "speed")
    assert ran.state == "ran"
    assert canary.exists()
    request = submit_request(dataset, recipe).request
    canary.unlink()
    forget_imported_definitions()
    clear_definition_caches()

    payload = invoke_json(_settle_args(dataset, request, "--json"))

    assert StepSettleWire.model_validate(payload).settle.outcome == "settled"
    assert not canary.exists()
    assert not imported_definitions()


def test_a_settle_creates_no_root_directory(tracked: Dataset) -> None:
    request = _complete_step_and_a_new_request(tracked)
    empty_roots = _remove_empty_roots(tracked)
    before = tree_snapshot(tracked.base_dir)

    _ = invoke_ok(_settle_args(tracked, request, "--json"))

    after = tree_snapshot(tracked.base_dir)
    added = set(after) - set(before)
    run_log = run_log_path(tracked.base_dir, request.execution_of("speed"))
    expected = {run_log.relative_to(tracked.base_dir).as_posix()}
    if ".mosaic/runs" not in before:
        expected.add(".mosaic/runs")
    assert added == expected
    assert not any(path.exists() for path in empty_roots)
    assert set(before) - set(after) == set()


def test_a_cancelled_settle_exits_130_and_prints_nothing(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = submit_request(tracked, chain_recipe()).request

    def cancelled(*_args: object, **_kwargs: object) -> NoReturn:
        raise Cancelled

    monkeypatch.setattr("mosaic.core.pipeline.graph.settle.settle_step", cancelled)
    _ = _remove_empty_roots(tracked)
    before = tree_snapshot(tracked.base_dir)

    result = CliRunner().invoke(app, _settle_args(tracked, request, "--json"))

    assert result.exit_code == 130, result.output
    assert result.stdout == ""
    assert_tree_unchanged(tracked.base_dir, before)


@pytest.mark.parametrize("flag", ["--overwrite", "--execution-id"])
def test_a_flag_that_a_settle_does_not_take_is_a_usage_error(
    tracked: Dataset, flag: str
) -> None:
    request = submit_request(tracked, chain_recipe()).request
    args = _settle_args(tracked, request, flag)
    if flag == "--execution-id":
        args.append("some-id")

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 2, result.output


def test_the_text_output_names_the_outcome_after_the_size(tracked: Dataset) -> None:
    request = _complete_step_and_a_new_request(tracked)

    result = invoke_ok(_settle_args(tracked, request))

    lines = result.stdout.splitlines()
    assert lines[0].startswith("step speed (feature) of request ")
    assert lines[-1].startswith("settled")
    assert "cached" in lines[-1]
