"""A run that computes several entries at once.

The worker count belongs to the machine and not to the run. It changes neither the
run id nor the outputs, and the failures it adds are the host's: a worker the host
kills, and a host that has run low on memory. Each ends the run in a state a later
run resumes, and says what to change.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline import memory_watch
from mosaic.core.pipeline import run as run_module
from mosaic.core.pipeline.memory_watch import MemoryWatch
from mosaic.core.pipeline.parallelism import MemoryFloorReached, WorkerProcessDied
from mosaic.core.pipeline.run import run_feature
from mosaic.runlog import read_run, run_log_dir
from tests.helpers import ParallelFeature, add_tracks_variant, make_dataset

VARIANT = "convert-trex.0.2-1111111111"
_SEQUENCES = ("seq_a", "seq_b", "seq_c")


class _KillsOnSeqB(ParallelFeature):
    """Ends its process on ``seq_b``, whether it runs in a thread or a worker."""

    name = "prov-parallel-kills"
    kill_on = ("seq_b",)


def _outputs(ds: Dataset, storage: str, run_id: str) -> dict[str, pd.DataFrame]:
    run_root = ds.get_root("features") / storage / run_id
    return {
        path.name: pd.read_parquet(path) for path in sorted(run_root.glob("*.parquet"))
    }


def _dataset(tmp_path: Path, name: str = "ds") -> Dataset:
    ds = make_dataset(tmp_path / name)
    add_tracks_variant(ds, VARIANT, *_SEQUENCES, std_format="mosaic_v1")
    return ds


def test_workers_change_neither_the_run_id_nor_the_outputs(tmp_path: Path) -> None:
    alone = _dataset(tmp_path, "alone")
    together = _dataset(tmp_path, "together")

    first = run_feature(alone, ParallelFeature(), parallel_workers=1)
    second = run_feature(
        together, ParallelFeature(), parallel_workers=3, parallel_mode="thread"
    )

    assert first.run_id is not None
    assert second.run_id == first.run_id
    storage = "prov-parallel__from__tracks"
    expected = _outputs(alone, storage, first.run_id)
    found = _outputs(together, storage, first.run_id)
    assert expected.keys() == found.keys()
    assert len(found) == len(_SEQUENCES)
    for name, table in expected.items():
        pd.testing.assert_frame_equal(found[name], table)


def test_a_dead_worker_ends_the_run_with_a_remedy_and_a_rerun_resumes(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path)
    execution_id = "TESTPARALLEL1"

    with pytest.raises(WorkerProcessDied) as raised:
        _ = run_feature(
            ds,
            _KillsOnSeqB(),
            parallel_workers=2,
            parallel_mode="process",
            execution_id=execution_id,
        )

    message = str(raised.value)
    assert "workers=2, mode=process" in message
    assert "fewer workers" in message
    assert "thread mode" in message
    snapshot = read_run(run_log_dir(ds.base_dir), execution_id)
    assert snapshot is not None
    assert snapshot["status"] == "failed"
    assert "WorkerProcessDied" in snapshot["error_json"]
    assert (snapshot["workers"], snapshot["parallel_mode"]) == (2, "process")
    assert snapshot["peak_rss_bytes"] > 0


def test_a_rerun_after_a_dead_worker_finishes_the_same_run(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    with pytest.raises(WorkerProcessDied):
        _ = run_feature(ds, _KillsOnSeqB(), parallel_workers=2, parallel_mode="process")

    class _Survives(_KillsOnSeqB):
        kill_on = ()

    result = run_feature(ds, _Survives(), parallel_workers=1)

    # ``kill_on`` is not a parameter, so both features name one run.
    assert result.run_id is not None
    kept = _outputs(ds, "prov-parallel-kills__from__tracks", result.run_id)
    assert len(kept) == len(_SEQUENCES)


def _free_fractions(values: list[float]) -> Callable[[MemoryWatch], float]:
    """A ``MemoryWatch.free_fraction`` that returns *values* in turn, then the last."""
    remaining = list(values)

    def free_fraction(self: MemoryWatch) -> float:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return free_fraction


def test_the_memory_floor_waits_for_the_entries_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    # Plenty at the first entry, little at the second, plenty again once the first
    # has been written.
    monkeypatch.setattr(
        memory_watch.MemoryWatch, "free_fraction", _free_fractions([0.9, 0.01, 0.9])
    )

    result = run_feature(
        ds,
        ParallelFeature(),
        parallel_workers=3,
        parallel_mode="thread",
        min_free_memory_fraction=0.5,
    )

    assert result.failed_entries == ()
    assert result.entries_written == len(_SEQUENCES)


def test_the_memory_floor_fails_the_run_when_nothing_is_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    execution_id = "TESTPARALLEL2"
    monkeypatch.setattr(
        memory_watch.MemoryWatch, "free_fraction", _free_fractions([0.01])
    )

    with pytest.raises(MemoryFloorReached, match="fewer workers"):
        _ = run_feature(
            ds,
            ParallelFeature(),
            parallel_workers=2,
            parallel_mode="thread",
            min_free_memory_fraction=0.5,
            execution_id=execution_id,
        )

    snapshot = read_run(run_log_dir(ds.base_dir), execution_id)
    assert snapshot is not None
    assert snapshot["status"] == "failed"
    assert "MemoryFloorReached" in snapshot["error_json"]


def test_the_floor_applies_to_a_run_with_several_workers_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    monkeypatch.setattr(
        memory_watch.MemoryWatch, "free_fraction", _free_fractions([0.01])
    )

    result = run_feature(ds, ParallelFeature(), min_free_memory_fraction=0.5)

    assert result.entries_written == len(_SEQUENCES)


@pytest.mark.parametrize("fraction", [-0.1, 1.0])
def test_a_floor_outside_zero_to_one_is_refused(
    tmp_path: Path, fraction: float
) -> None:
    ds = _dataset(tmp_path)

    with pytest.raises(ValueError, match="min_free_memory_fraction"):
        _ = run_feature(ds, ParallelFeature(), min_free_memory_fraction=fraction)


@pytest.mark.parametrize(
    ("workers", "mode"), [(1, ""), (2, "thread")], ids=["inline", "thread"]
)
def test_a_run_records_its_workers_and_the_memory_it_used(
    tmp_path: Path, workers: int, mode: str
) -> None:
    ds = _dataset(tmp_path)
    execution_id = f"TESTPARALLEL3{workers}"

    _ = run_feature(
        ds,
        ParallelFeature(),
        parallel_workers=workers,
        parallel_mode="thread",
        execution_id=execution_id,
    )

    snapshot = read_run(run_log_dir(ds.base_dir), execution_id)
    assert snapshot is not None
    assert (snapshot["workers"], snapshot["parallel_mode"]) == (workers, mode)
    assert snapshot["peak_rss_bytes"] > 0
    assert snapshot["min_available_bytes"] > 0
    assert snapshot["total_memory_bytes"] >= snapshot["min_available_bytes"]


def test_the_pool_is_shut_down_when_the_run_fails_for_another_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    shutdowns: list[bool] = []

    class _Recording(ThreadPoolExecutor):
        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
            shutdowns.append(cancel_futures)
            super().shutdown(wait=wait, cancel_futures=cancel_futures)

    def full_disk(*_args: object, **_kwargs: object) -> int:
        msg = "No space left on device"
        raise OSError(msg)

    monkeypatch.setattr(run_module, "ThreadPoolExecutor", _Recording)
    monkeypatch.setattr(run_module, "write_output", full_disk)

    with pytest.raises(OSError, match="No space left"):
        _ = run_feature(
            ds, ParallelFeature(), parallel_workers=2, parallel_mode="thread"
        )

    assert True in shutdowns
