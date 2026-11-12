"""A run that computes several entries at once.

The worker count belongs to the machine and not to the run. It changes neither the
run id nor the outputs, and the failures it adds are the host's: a worker the host
kills, and a host that has run low on memory. Each ends the run in a state a later
run resumes, and says what to change.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline import entry_pool, memory_watch
from mosaic.core.pipeline import manifest as manifest_module
from mosaic.core.pipeline import run as run_module
from mosaic.core.pipeline._loaders import LoadSpec
from mosaic.core.pipeline.admission import RowCostEstimate
from mosaic.core.pipeline.column_contract import MissingInputColumns
from mosaic.core.pipeline.input_size import EntryInputSize
from mosaic.core.pipeline.manifest import ManifestEntry
from mosaic.core.pipeline.memory_limit import MemoryLimit
from mosaic.core.pipeline.memory_watch import MemoryWatch
from mosaic.core.pipeline.parallelism import (
    EntryOverMemoryLimit,
    MemoryFloorReached,
    WorkerProcessDied,
)
from mosaic.core.pipeline.run import run_feature
from mosaic.runlog import read_run, run_log_dir, run_log_path
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


def test_the_memory_floor_collects_garbage_before_it_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    collections: list[int] = []
    # Under the floor until garbage has been collected, as when finished entries'
    # tables are held by reference cycles.
    free = _free_fractions([0.01, 0.9])

    def counting_collect(generation: int = 2) -> int:
        collections.append(generation)
        return 0

    monkeypatch.setattr(memory_watch.MemoryWatch, "free_fraction", free)
    monkeypatch.setattr(entry_pool.gc, "collect", counting_collect)

    result = run_feature(
        ds,
        ParallelFeature(),
        parallel_workers=2,
        parallel_mode="thread",
        min_free_memory_fraction=0.5,
    )

    assert result.failed_entries == ()
    assert collections


def test_the_floor_applies_to_a_run_with_several_workers_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    monkeypatch.setattr(
        memory_watch.MemoryWatch, "free_fraction", _free_fractions([0.01])
    )

    result = run_feature(ds, ParallelFeature(), min_free_memory_fraction=0.5)

    assert result.entries_written == len(_SEQUENCES)


def _recorded_loads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the sequence of every table the run loads, in the order loaded."""
    loads: list[str] = []
    real = manifest_module.load_entry_data

    def load(
        file_specs: list[tuple[Path, LoadSpec]],
        filters: Iterable[Callable[[pd.DataFrame], pd.DataFrame]] = (),
        *,
        cross_join: bool = False,
        expands_individuals: bool = False,
    ) -> pd.DataFrame | None:
        loads.extend(
            name
            for name in _SEQUENCES
            if any(name in str(path) for path, _ in file_specs)
        )
        return real(
            file_specs,
            filters,
            cross_join=cross_join,
            expands_individuals=expands_individuals,
        )

    monkeypatch.setattr(manifest_module, "load_entry_data", load)
    return loads


def test_the_memory_floor_holds_the_load_and_not_only_the_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    loads = _recorded_loads(monkeypatch)
    monkeypatch.setattr(
        memory_watch.MemoryWatch, "free_fraction", _free_fractions([0.01])
    )

    with pytest.raises(MemoryFloorReached):
        _ = run_feature(
            ds,
            ParallelFeature(),
            parallel_workers=2,
            parallel_mode="thread",
            min_free_memory_fraction=0.5,
        )

    assert loads == []


@pytest.mark.parametrize("fraction", [-0.1, 1.0])
def test_a_floor_outside_zero_to_one_is_refused(
    tmp_path: Path, fraction: float
) -> None:
    ds = _dataset(tmp_path)

    with pytest.raises(ValueError, match="min_free_memory_fraction"):
        _ = run_feature(ds, ParallelFeature(), min_free_memory_fraction=fraction)


@pytest.mark.parametrize("margin", [-1.0, 0.5, 4.5])
def test_an_admission_margin_outside_one_to_four_is_refused(
    tmp_path: Path, margin: float
) -> None:
    ds = _dataset(tmp_path)

    with pytest.raises(ValueError, match="admission_margin"):
        _ = run_feature(
            ds, ParallelFeature(), parallel_workers=2, admission_margin=margin
        )


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

    monkeypatch.setattr(entry_pool, "ThreadPoolExecutor", _Recording)
    monkeypatch.setattr(run_module, "write_output", full_disk)

    with pytest.raises(OSError, match="No space left"):
        _ = run_feature(
            ds, ParallelFeature(), parallel_workers=2, parallel_mode="thread"
        )

    assert True in shutdowns


# --- the admission gate -----------------------------------------------------------

_GIB = 1024**3


def _events(ds: Dataset, execution_id: str, kind: str) -> list[dict[str, object]]:
    lines = run_log_path(ds.base_dir, execution_id).read_text().splitlines()
    return [record for record in map(json.loads, lines) if record["ev"] == kind]


def _sequence_of(record: dict[str, object]) -> str:
    key = str(record["key"])
    (sequence,) = [name for name in _SEQUENCES if key.endswith(name)]
    return sequence


def _in_flight(ds: Dataset, execution_id: str) -> dict[str, int]:
    """The most entries in flight beside each entry, by sequence."""
    found: dict[str, int] = {}
    for end in _events(ds, execution_id, "entry_end"):
        count = end["in_flight"]
        assert isinstance(count, int)
        found[_sequence_of(end)] = count
    return found


def _rows(monkeypatch: pytest.MonkeyPatch, rows: dict[str, int]) -> None:
    """Give each sequence's inputs the rows *rows* names."""

    def entry_input_size(entry: ManifestEntry) -> EntryInputSize:
        return EntryInputSize(rows=rows[entry.sequence], columns=1)

    monkeypatch.setattr(run_module, "entry_input_size", entry_input_size)


def _limit(
    monkeypatch: pytest.MonkeyPatch, *, limit_gib: float, in_use_gib: float
) -> None:
    """Put the run under a container limit of which *in_use_gib* is in use."""
    limit = MemoryLimit(
        limit_bytes=int(limit_gib * _GIB),
        source="cgroup_v2",
        in_use_bytes=int(in_use_gib * _GIB),
        peak_bytes=0,
        host_total_bytes=512 * _GIB,
        host_available_bytes=256 * _GIB,
    )
    monkeypatch.setattr(memory_watch, "read_memory_limit", lambda: limit)


class _OneGibARow(RowCostEstimate):
    """Costs 1 GiB an input row and nothing fixed, once an entry with rows is in flight."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: bool = False

    def observe(self, tree_rss: int, in_flight_rows: int, live_workers: int) -> None:
        super().observe(tree_rss, in_flight_rows, live_workers)
        if in_flight_rows > 0:
            self.seen = True

    @property
    def bytes_per_row(self) -> int | None:
        return _GIB if self.seen else None

    def predicted_peak(
        self, rows_in_flight: int, rows_next: int, workers_after: int, margin: float
    ) -> int:
        _ = workers_after
        return int(margin * _GIB * (rows_in_flight + rows_next))


def test_with_a_margin_the_first_entry_runs_alone_and_the_largest_starts_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    _rows(monkeypatch, {"seq_a": 10, "seq_b": 30, "seq_c": 20})
    loads = _recorded_loads(monkeypatch)

    result = run_feature(
        ds,
        ParallelFeature(),
        parallel_workers=2,
        parallel_mode="thread",
        admission_margin=1.5,
        execution_id="TESTGATEORDER",
    )

    assert result.entries_written == len(_SEQUENCES)
    assert loads == ["seq_b", "seq_c", "seq_a"]
    assert _in_flight(ds, "TESTGATEORDER")["seq_b"] == 1
    resources = _events(ds, "TESTGATEORDER", "resources")[-1]
    assert resources["admission_margin"] == 1.5


def test_without_a_margin_the_entries_keep_their_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    _rows(monkeypatch, {"seq_a": 10, "seq_b": 30, "seq_c": 20})
    loads = _recorded_loads(monkeypatch)

    _ = run_feature(ds, ParallelFeature(), parallel_workers=2, parallel_mode="thread")

    assert loads == list(_SEQUENCES)


def test_an_entry_predicted_over_the_budget_beside_another_waits_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    _rows(monkeypatch, {"seq_a": 40, "seq_b": 40, "seq_c": 40})
    # One entry is 60 GiB with the margin and two are 120 GiB, over 100 GiB.
    _limit(monkeypatch, limit_gib=100, in_use_gib=1)
    monkeypatch.setattr(run_module, "RowCostEstimate", _OneGibARow)

    result = run_feature(
        ds,
        ParallelFeature(),
        parallel_workers=2,
        parallel_mode="thread",
        admission_margin=1.5,
        execution_id="TESTGATEWAITS",
    )

    assert result.entries_written == len(_SEQUENCES)
    assert _in_flight(ds, "TESTGATEWAITS") == dict.fromkeys(_SEQUENCES, 1)
    waits = _events(ds, "TESTGATEWAITS", "resources")[-1]["admission_waits"]
    assert isinstance(waits, int)
    assert waits >= 2


def test_an_entry_that_fits_the_limit_but_not_what_is_available_starts_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    _rows(monkeypatch, {"seq_a": 40, "seq_b": 40, "seq_c": 40})
    # 40 GiB an entry fits the 100 GiB limit, and not the half GiB available.
    _limit(monkeypatch, limit_gib=100, in_use_gib=99.5)
    monkeypatch.setattr(run_module, "RowCostEstimate", _OneGibARow)

    result = run_feature(
        ds,
        ParallelFeature(),
        parallel_workers=2,
        parallel_mode="thread",
        admission_margin=1.5,
        execution_id="TESTGATEBUSY",
    )

    assert result.entries_written == len(_SEQUENCES)
    assert _in_flight(ds, "TESTGATEBUSY") == dict.fromkeys(_SEQUENCES, 1)


def test_an_entry_predicted_over_the_limit_fails_the_run_resumably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    _rows(monkeypatch, {"seq_a": 40, "seq_b": 40, "seq_c": 40})
    # 40 GiB an entry is over a 30 GiB limit; the first runs before the estimate.
    _limit(monkeypatch, limit_gib=30, in_use_gib=1)
    monkeypatch.setattr(run_module, "RowCostEstimate", _OneGibARow)

    with pytest.raises(EntryOverMemoryLimit) as raised:
        _ = run_feature(
            ds,
            ParallelFeature(),
            parallel_workers=2,
            parallel_mode="thread",
            admission_margin=1.5,
            execution_id="TESTGATEOVER",
        )

    assert "Run again with a larger memory limit." in str(raised.value)

    snapshot = read_run(run_log_dir(ds.base_dir), "TESTGATEOVER")
    assert snapshot is not None
    assert snapshot["status"] == "failed"
    assert "EntryOverMemoryLimit" in snapshot["error_json"]
    assert snapshot["entries_written"] == 1

    monkeypatch.undo()
    again = run_feature(
        ds, ParallelFeature(), parallel_workers=2, execution_id="TESTGATERESUME"
    )

    assert again.failed_entries == ()
    assert again.entries_written == len(_SEQUENCES)
    computed = [
        end
        for end in _events(ds, "TESTGATERESUME", "entry_end")
        if "wall_seconds" in end
    ]
    assert len(computed) == len(_SEQUENCES) - 1


def test_an_entry_predicted_over_the_limit_in_process_mode_offers_thread_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    _rows(monkeypatch, {"seq_a": 40, "seq_b": 40, "seq_c": 40})
    _limit(monkeypatch, limit_gib=30, in_use_gib=1)
    monkeypatch.setattr(run_module, "RowCostEstimate", _OneGibARow)

    with pytest.raises(EntryOverMemoryLimit) as raised:
        _ = run_feature(
            ds,
            ParallelFeature(),
            parallel_workers=2,
            parallel_mode="process",
            admission_margin=1.5,
            execution_id="TESTGATEOVERPROCESS",
        )

    assert "Run again with a larger memory limit, or in thread mode." in str(
        raised.value
    )


def test_an_entry_without_a_table_never_stalls_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    real = manifest_module.load_entry_data

    def load(
        file_specs: list[tuple[Path, LoadSpec]],
        filters: Iterable[Callable[[pd.DataFrame], pd.DataFrame]] = (),
        *,
        cross_join: bool = False,
        expands_individuals: bool = False,
    ) -> pd.DataFrame | None:
        if any("seq_b" in str(path) for path, _ in file_specs):
            return None
        return real(
            file_specs,
            filters,
            cross_join=cross_join,
            expands_individuals=expands_individuals,
        )

    monkeypatch.setattr(manifest_module, "load_entry_data", load)

    result = run_feature(
        ds,
        ParallelFeature(),
        parallel_workers=2,
        parallel_mode="thread",
        admission_margin=1.5,
    )

    assert result.entries_written == len(_SEQUENCES) - 1


def test_an_entry_failing_its_column_check_never_stalls_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    calls: list[int] = []

    def fail_first(feature: object, columns: object, *, numeric: object) -> None:
        _ = (feature, columns, numeric)
        calls.append(1)
        if len(calls) == 1:
            raise MissingInputColumns("prov-parallel", (), ("tracks",))

    monkeypatch.setattr(run_module, "check_input_columns", fail_first)

    result = run_feature(
        ds,
        ParallelFeature(),
        parallel_workers=2,
        parallel_mode="thread",
        admission_margin=1.5,
    )

    assert len(result.failed_entries) == 1
    assert result.entries_written == len(_SEQUENCES) - 1


def test_the_gate_changes_neither_the_run_id_nor_the_outputs(tmp_path: Path) -> None:
    open_run = _dataset(tmp_path, "open")
    gated_run = _dataset(tmp_path, "gated")

    first = run_feature(open_run, ParallelFeature(), parallel_workers=1)
    second = run_feature(
        gated_run,
        ParallelFeature(),
        parallel_workers=3,
        parallel_mode="thread",
        admission_margin=2.0,
    )

    assert first.run_id is not None
    assert second.run_id == first.run_id
    storage = "prov-parallel__from__tracks"
    expected = _outputs(open_run, storage, first.run_id)
    found = _outputs(gated_run, storage, first.run_id)
    assert expected.keys() == found.keys()
    for name, table in expected.items():
        pd.testing.assert_frame_equal(found[name], table)
