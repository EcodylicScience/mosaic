"""Removing a whole feature run: the record first, then the rows, then the files.

``remove_feature_run`` refuses a run outside the ``features`` root, records the
run's ``params.json`` in the cleanup's run-log before it touches anything, and
drops the index rows before it deletes the directory. The record is enough to
compute the run again under the same run id.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline import feature_run_removal
from mosaic.core.pipeline.feature_run_removal import (
    finish_feature_run_removal,
    remove_feature_run,
    run_bytes,
)
from mosaic.core.pipeline.index import (
    feature_index,
    feature_index_path,
    feature_run_root,
)
from mosaic.core.pipeline.inventory.params import RunParams, read_run_params
from mosaic.core.pipeline.run import run_feature
from mosaic.runlog import (
    JsonlRunLog,
    new_execution_id,
    read_removal_records,
    run_log_dir,
    run_log_path,
)
from tests.helpers import add_tracks_variant, make_dataset, run_speed

VARIANT = "convert-trex.0.2-1111111111"

STORAGE = "speed-angvel__from__tracks"


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


def _params(ds: Dataset, run_id: str) -> RunParams:
    read = read_run_params(feature_run_root(ds, STORAGE, run_id))
    assert read.params is not None
    return read.params


def _rows(ds: Dataset, run_id: str) -> pd.DataFrame:
    frame = feature_index(feature_index_path(ds, STORAGE)).read()
    return frame[frame["run_id"] == run_id]


def _log(ds: Dataset) -> JsonlRunLog:
    execution_id = new_execution_id()
    return JsonlRunLog(run_log_path(ds.base_dir, execution_id), execution_id)


def test_a_run_is_recorded_then_its_rows_and_files_are_removed(
    dataset: Dataset,
) -> None:
    old = run_speed(dataset, 1)
    kept = run_speed(dataset, 2)
    run_root = feature_run_root(dataset, STORAGE, old)
    size, files = run_bytes(run_root)

    with _log(dataset) as log:
        removed = remove_feature_run(
            dataset,
            STORAGE,
            old,
            params=_params(dataset, old),
            log=log,
            reason="superseded",
            superseded_by=kept,
        )
    [record] = read_removal_records(run_log_dir(dataset.base_dir), log.execution_id)

    assert (removed.bytes, removed.files, removed.rows_dropped) == (size, files, 2)
    assert size > 0
    assert not run_root.exists()
    assert _rows(dataset, old).empty
    assert len(_rows(dataset, kept)) == 2
    assert feature_run_root(dataset, STORAGE, kept).is_dir()
    assert (record["state"], record["reason"], record["superseded_by"]) == (
        "removed",
        "superseded",
        kept,
    )
    assert record["bytes"] == size
    assert record["params"]["_params"]["step_size"] == 1


def test_the_intent_precedes_rows_and_rows_precede_files(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = run_speed(dataset, 1)
    run_root = feature_run_root(dataset, STORAGE, old)

    def refuse(path: Path) -> None:
        raise OSError(f"cannot remove {path}")

    monkeypatch.setattr(feature_run_removal.shutil, "rmtree", refuse)
    with _log(dataset) as log, pytest.raises(OSError, match="cannot remove"):
        _ = remove_feature_run(
            dataset,
            STORAGE,
            old,
            params=_params(dataset, old),
            log=log,
            reason="superseded",
            superseded_by="",
        )
    [record] = read_removal_records(run_log_dir(dataset.base_dir), log.execution_id)

    assert record["state"] == "interrupted"
    assert record["params"]
    assert _rows(dataset, old).empty
    assert run_root.is_dir()


def test_an_interrupted_removal_is_finished(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = run_speed(dataset, 1)

    def refuse(path: Path) -> None:
        raise OSError(f"cannot remove {path}")

    with _log(dataset) as log:
        with monkeypatch.context() as patched:
            patched.setattr(feature_run_removal.shutil, "rmtree", refuse)
            with pytest.raises(OSError):
                _ = remove_feature_run(
                    dataset,
                    STORAGE,
                    old,
                    params=_params(dataset, old),
                    log=log,
                    reason="superseded",
                    superseded_by="",
                )
        finished = finish_feature_run_removal(dataset, STORAGE, old, log=log)
    [record] = read_removal_records(run_log_dir(dataset.base_dir), log.execution_id)

    assert finished.bytes > 0
    assert record["state"] == "removed"
    assert not feature_run_root(dataset, STORAGE, old).exists()


@pytest.mark.parametrize(
    ("storage", "run_id"),
    [("..", "x"), (STORAGE, ".."), (f"{STORAGE}/..", "x"), ("", "x")],
    ids=["storage-up", "run-up", "storage-slash", "storage-empty"],
)
def test_a_run_outside_the_features_root_is_refused(
    dataset: Dataset, storage: str, run_id: str
) -> None:
    kept = run_speed(dataset, 1)

    with _log(dataset) as log, pytest.raises(RuntimeError, match="refusing"):
        _ = remove_feature_run(
            dataset,
            storage,
            run_id,
            params=_params(dataset, kept),
            log=log,
            reason="superseded",
            superseded_by="",
        )

    assert read_removal_records(run_log_dir(dataset.base_dir), log.execution_id) == []
    assert feature_run_root(dataset, STORAGE, kept).is_dir()


def test_a_row_pointing_outside_the_run_root_is_refused(
    dataset: Dataset, tmp_path: Path
) -> None:
    old = run_speed(dataset, 1)
    index_path = feature_index_path(dataset, STORAGE)
    frame = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    outside = tmp_path / "elsewhere.parquet"
    outside.write_bytes(b"")
    frame.loc[0, "abs_path"] = str(outside)
    frame.to_csv(index_path, index=False)

    with _log(dataset) as log, pytest.raises(RuntimeError, match="outside the run"):
        _ = remove_feature_run(
            dataset,
            STORAGE,
            old,
            params=_params(dataset, old),
            log=log,
            reason="superseded",
            superseded_by="",
        )

    assert outside.exists()
    assert len(_rows(dataset, old)) == 2
    assert read_removal_records(run_log_dir(dataset.base_dir), log.execution_id) == []


def test_the_record_rebuilds_the_feature_to_its_run_id(dataset: Dataset) -> None:
    old = run_speed(dataset, 3)
    with _log(dataset) as log:
        _ = remove_feature_run(
            dataset,
            STORAGE,
            old,
            params=_params(dataset, old),
            log=log,
            reason="superseded",
            superseded_by="",
        )
    [record] = read_removal_records(run_log_dir(dataset.base_dir), log.execution_id)
    recorded = RunParams.model_validate(record["params"])

    rebuilt = SpeedAngvel(
        SpeedAngvel.Inputs.model_validate(recorded.inputs), recorded.params
    )
    again = run_feature(dataset, rebuilt)

    assert again.run_id == old
    assert feature_run_root(dataset, STORAGE, old).is_dir()
