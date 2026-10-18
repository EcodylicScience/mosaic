"""The commands that run work take the worker count, and refuse what it cannot mean.

``--workers`` belongs to the machine, so ``mosaic run`` and ``mosaic pipeline run``
accept it and a recipe does not. Each case drives the real command.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import mosaic.core.pipeline.graph as graph_module
import mosaic.core.pipeline.run as run_module
from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.parallelism import MAX_WORKERS_CEILING, ParallelSettings
from mosaic.core.pipeline.types import Result
from tests.helpers import CHAIN_RECIPE, make_dataset

runner = CliRunner()


class _Stopped(Exception):
    """Ends a command at the call under test, after its arguments are seen."""


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return make_dataset(tmp_path / "ds")


def test_mosaic_run_hands_the_worker_flags_to_the_feature_run(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def run_feature(*_args: object, **kwargs: object) -> Result[str]:
        seen.update(kwargs)
        return Result(feature="speed-angvel", run_id="r1", execution_id="e1")

    monkeypatch.setattr(run_module, "run_feature", run_feature)

    result = runner.invoke(
        app,
        [
            "run",
            "--manifest",
            str(dataset.manifest_path),
            "--feature",
            "speed-angvel",
            "--workers",
            "4",
            "--parallel-mode",
            "process",
            "--min-free-memory-fraction",
            "0.2",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["parallel_workers"] == 4
    assert seen["parallel_mode"] == "process"
    assert seen["min_free_memory_fraction"] == 0.2


def test_mosaic_run_without_workers_runs_in_turn(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def run_feature(*_args: object, **kwargs: object) -> Result[str]:
        seen.update(kwargs)
        return Result(feature="speed-angvel", run_id="r1", execution_id="e1")

    monkeypatch.setattr(run_module, "run_feature", run_feature)

    result = runner.invoke(
        app,
        [
            "run",
            "--manifest",
            str(dataset.manifest_path),
            "--feature",
            "speed-angvel",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["parallel_workers"] is None
    assert seen["min_free_memory_fraction"] == 0.0


def test_a_count_beyond_the_ceiling_is_refused_by_the_option(
    dataset: Dataset,
) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--manifest",
            str(dataset.manifest_path),
            "--feature",
            "speed-angvel",
            "--workers",
            str(MAX_WORKERS_CEILING + 1),
        ],
    )

    assert result.exit_code == 2


@pytest.mark.parametrize(
    "flags",
    [["--parallel-mode", "process"], ["--min-free-memory-fraction", "0.2"]],
    ids=["mode", "floor"],
)
def test_the_mode_and_the_floor_need_several_workers(
    dataset: Dataset, flags: list[str]
) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--manifest",
            str(dataset.manifest_path),
            "--feature",
            "speed-angvel",
            *flags,
        ],
    )

    assert result.exit_code == 1
    assert "--workers above 1" in result.stderr


def test_an_op_run_takes_no_workers(dataset: Dataset) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--manifest",
            str(dataset.manifest_path),
            "--kind",
            "transcode",
            "--workers",
            "2",
        ],
    )

    assert result.exit_code == 1
    assert "--feature or --graph-request" in result.stderr


def test_pipeline_run_hands_the_worker_flags_to_the_run(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def run_pipeline(*_args: object, **kwargs: object) -> None:
        seen.update(kwargs)
        raise _Stopped

    monkeypatch.setattr(graph_module, "run_pipeline", run_pipeline)

    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--manifest",
            str(dataset.manifest_path),
            "--recipe",
            json.dumps(CHAIN_RECIPE),
            "--workers",
            "3",
            "--min-free-memory-fraction",
            "0.1",
        ],
    )

    assert isinstance(result.exception, _Stopped)
    assert seen["parallel"] == ParallelSettings(
        workers=3, mode="thread", min_free_memory_fraction=0.1
    )
