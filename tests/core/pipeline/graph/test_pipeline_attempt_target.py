"""What a step's attempt records of the run it names.

A cleanup that reaches a retired recipe's runs reads the storage and the process
of each attempt, so a rollup carries both: the ``target`` the run-log names, and
whether a process wrote the run or the attempt found it already done.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import load_request, request_rollup, run_pipeline
from tests.helpers import add_tracks_variant, make_dataset, speed_recipe_stepping

VARIANT = "convert-trex.0.2-1111111111"

STORAGE = "speed-angvel__from__tracks"


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


def test_an_attempt_names_the_storage_it_wrote_and_the_process_that_wrote_it(
    dataset: Dataset,
) -> None:
    ran = run_pipeline(dataset, speed_recipe_stepping(1))

    request = load_request(dataset.base_dir, ran.request_id)
    [attempt] = request_rollup(dataset.base_dir, request).steps

    assert attempt.target == STORAGE
    assert attempt.run_id
    assert not attempt.cache_hit
    assert attempt.pid != 0


def test_an_attempt_that_found_its_run_done_records_no_process(
    dataset: Dataset,
) -> None:
    first = run_pipeline(dataset, speed_recipe_stepping(1))
    again = run_pipeline(dataset, speed_recipe_stepping(1))

    [wrote] = request_rollup(
        dataset.base_dir, load_request(dataset.base_dir, first.request_id)
    ).steps
    [found] = request_rollup(
        dataset.base_dir, load_request(dataset.base_dir, again.request_id)
    ).steps

    assert (found.target, found.run_id) == (wrote.target, wrote.run_id)
    assert found.cache_hit
    assert found.pid == 0
