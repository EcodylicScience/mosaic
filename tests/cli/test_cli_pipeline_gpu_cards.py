"""``mosaic pipeline run --gpu-cards``: the cards a request's GPU steps may lease.

The flag limits a request's driver. It belongs to the machine and enters no
recipe, request or run id, so it goes with ``--request`` and an in-process
``--recipe`` run refuses it. A card the host does not show ends the command
before a step is recorded or a driver is started.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import mosaic.core.pipeline.graph as graph_module
from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import DetachedDriver, Request, submit_request
from mosaic.core.pipeline.job import CancelToken
from mosaic.runlog import read_run, run_log_dir
from tests.helpers import (
    CHAIN_RECIPE,
    TWO_CARDS,
    add_tracks_variant,
    chain_recipe,
    install_fake_nvidia_smi,
    make_dataset,
)

runner = CliRunner()

VARIANT = "convert-trex.0.2-1111111111"


class _Stopped(Exception):
    """Ends a command at the call under test, after its arguments are seen."""


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two schema-valid track tables, and nothing computed."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return dataset


@pytest.fixture
def submitted(tracked: Dataset) -> Request:
    """The chain, submitted and not run."""
    return submit_request(tracked, chain_recipe()).request


def _run_request(
    dataset: Dataset, request: Request, *flags: str
) -> tuple[int, str, str]:
    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--manifest",
            str(dataset.manifest_path),
            "--request",
            request.request_id,
            *flags,
        ],
    )
    return result.exit_code, result.stdout, result.stderr


def _no_step_was_recorded(dataset: Dataset, request: Request) -> bool:
    return all(
        read_run(run_log_dir(dataset.base_dir), request.execution_of(step)) is None
        for step in ("speed", "templates")
    )


def test_a_card_limit_with_a_recipe_is_refused_before_anything_is_written(
    tracked: Dataset,
) -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--manifest",
            str(tracked.manifest_path),
            "--recipe",
            json.dumps(CHAIN_RECIPE),
            "--gpu-cards",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert "--gpu-cards" in result.stderr
    assert "--request" in result.stderr
    assert not (tracked.base_dir / ".mosaic" / "pipelines" / "requests").exists()


def test_a_request_naming_a_hidden_card_ends_before_any_step(
    tracked: Dataset,
    submitted: Request,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS)

    code, stdout, stderr = _run_request(tracked, submitted, "--gpu-cards", "0,5")

    assert code == 1
    assert "the card limit names '5'" in stderr
    assert "gpu_cards" not in stderr
    assert stdout == ""
    assert _no_step_was_recorded(tracked, submitted)


def test_an_empty_card_limit_is_refused(tracked: Dataset, submitted: Request) -> None:
    code, _stdout, stderr = _run_request(tracked, submitted, "--gpu-cards", "")

    assert code == 1
    assert "the card limit names no card" in stderr
    assert _no_step_was_recorded(tracked, submitted)


def test_a_detached_request_naming_a_hidden_card_starts_no_driver(
    tracked: Dataset,
    submitted: Request,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS)
    started: list[object] = []

    def start(*args: object, **kwargs: object) -> DetachedDriver:
        started.append((args, kwargs))
        raise _Stopped

    monkeypatch.setattr(graph_module, "start_detached_driver", start)

    code, _stdout, stderr = _run_request(
        tracked, submitted, "--detach", "--gpu-cards", "3"
    )

    assert code == 1
    assert "the card limit names '3'" in stderr
    assert started == []


def test_a_detached_request_hands_the_driver_its_card_limit(
    tracked: Dataset,
    submitted: Request,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS)
    seen: dict[str, object] = {}

    def start(*_args: object, **kwargs: object) -> DetachedDriver:
        seen.update(kwargs)
        return DetachedDriver(
            request_id=submitted.request_id,
            pid=4242,
            log_path=tracked.base_dir / "driver.log",
        )

    monkeypatch.setattr(graph_module, "start_detached_driver", start)

    code, _stdout, stderr = _run_request(
        tracked, submitted, "--detach", "--gpu-cards", "1, 0"
    )

    assert code == 0, stderr
    assert seen["gpu_cards"] == ("1", "0")


def test_a_request_run_here_hands_the_run_its_card_limit(
    tracked: Dataset, submitted: Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def run_request(*_args: object, **kwargs: object) -> None:
        seen.update(kwargs)
        raise _Stopped

    monkeypatch.setattr(graph_module, "run_request", run_request)

    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--manifest",
            str(tracked.manifest_path),
            "--request",
            submitted.request_id,
            "--gpu-cards",
            "GPU-aaaa,0",
        ],
    )

    assert isinstance(result.exception, _Stopped)
    assert seen["gpu_cards"] == ("GPU-aaaa", "0")
    assert isinstance(seen["cancel_token"], CancelToken)


def test_without_the_flag_the_run_names_no_card_limit(
    tracked: Dataset, submitted: Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def run_request(*_args: object, **kwargs: object) -> None:
        seen.update(kwargs)
        raise _Stopped

    monkeypatch.setattr(graph_module, "run_request", run_request)

    _ = _run_request(tracked, submitted)

    assert seen["gpu_cards"] is None
