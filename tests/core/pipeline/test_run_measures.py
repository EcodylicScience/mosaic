"""What a feature run measures of its own cost, per entry and for the whole run.

Each computed entry's ``entry_end`` carries its wall, apply and CPU seconds, its input
rows and how many entries were in flight beside it. The ``resources`` event carries the
run's CPU and wall time, its memory limit and base, the fit's cost and the per-row
estimate, and is written during a fit as during apply.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import threading
import time
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.pipeline import manifest as manifest_module
from mosaic.core.pipeline import memory_watch
from mosaic.core.pipeline.memory_watch import MemorySample, MemoryWatch
from mosaic.core.pipeline import run as run_module
from mosaic.core.pipeline.column_contract import MissingInputColumns
from mosaic.core.pipeline.loading import PairRowsMissing
from mosaic.core.pipeline.memory_limit import MemoryLimit
from mosaic.core.pipeline.admission import (
    EntriesInFlight,
    PeakSample,
    RowCostEstimate,
    WorkerBase,
    WorkerMeasure,
)
from mosaic.core.pipeline.run import ApplyOutcome, process_apply_worker, run_feature
from mosaic.core.pipeline._loaders import LoadSpec
from mosaic.core.pipeline.types import DependencyLookup
from mosaic.runlog import read_run, run_log_dir, run_log_path
from tests.helpers import (
    FitControl,
    FittingFeature,
    ParallelFeature,
    add_tracks_variant,
    make_dataset,
    run_fresh,
)

VARIANT = "convert-trex.0.2-1111111111"
_SEQUENCES = ("seq_a", "seq_b", "seq_c")
_MEASURE_FIELDS = (
    "wall_seconds",
    "apply_seconds",
    "input_rows",
    "in_flight",
    "cpu_seconds",
)
_SAMPLER = "mosaic-memory-sampler"
_GIB = 1024**3


def _dataset(tmp_path: Path, name: str = "ds") -> Dataset:
    ds = make_dataset(tmp_path / name)
    add_tracks_variant(ds, VARIANT, *_SEQUENCES, std_format="mosaic_v1")
    return ds


def _events_at(base_dir: Path, execution_id: str, kind: str) -> list[dict[str, object]]:
    lines = run_log_path(base_dir, execution_id).read_text().splitlines()
    return [record for record in map(json.loads, lines) if record["ev"] == kind]


def _events(ds: Dataset, execution_id: str, kind: str) -> list[dict[str, object]]:
    return _events_at(ds.base_dir, execution_id, kind)


def _last_resources(ds: Dataset, execution_id: str) -> dict[str, object]:
    return _events(ds, execution_id, "resources")[-1]


def _number(record: dict[str, object], key: str) -> float:
    value = record[key]
    assert isinstance(value, int | float)
    return float(value)


def _no_sampler_alive() -> bool:
    return not any(thread.name == _SAMPLER for thread in threading.enumerate())


class _Scoped(ParallelFeature):
    """A parallelizable feature that needs its scope, so it cannot run in workers."""

    name = "prov-parallel-scoped"

    def set_scope(self, scope: object) -> None:
        _ = scope


# --- per-entry measures ------------------------------------------------------


@pytest.mark.parametrize(
    ("workers", "least_in_flight"), [(1, 1), (2, 2)], ids=["inline", "thread"]
)
def test_every_computed_entry_records_its_cost(
    tmp_path: Path, workers: int, least_in_flight: int
) -> None:
    ds = _dataset(tmp_path)
    execution_id = f"TESTMEASURE{workers}"

    _ = run_feature(
        ds,
        ParallelFeature(),
        parallel_workers=workers,
        parallel_mode="thread",
        execution_id=execution_id,
    )

    ends = _events(ds, execution_id, "entry_end")
    assert len(ends) == len(_SEQUENCES)
    for end in ends:
        assert all(field in end for field in _MEASURE_FIELDS)
        assert _number(end, "input_rows") > 0
        assert _number(end, "wall_seconds") >= _number(end, "apply_seconds") >= 0
        assert _number(end, "cpu_seconds") >= 0
    in_flight = [int(_number(end, "in_flight")) for end in ends]
    if workers == 1:
        assert in_flight == [1, 1, 1]
    else:
        assert all(least_in_flight <= count <= workers + 1 for count in in_flight)


def test_a_cache_hit_records_no_cost(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    _ = run_feature(ds, ParallelFeature())

    _ = run_feature(ds, ParallelFeature(), execution_id="TESTMEASUREHIT")

    ends = _events(ds, "TESTMEASUREHIT", "entry_end")
    assert len(ends) == len(_SEQUENCES)
    assert all("wall_seconds" not in end for end in ends)


def test_progress_and_the_heartbeat_reach_the_log_and_an_injected_callback(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path)
    calls: list[tuple[int, int, str]] = []

    class _Recorder:
        def on_epoch_end(
            self, epoch: int, total_epochs: int, metrics: dict[str, float]
        ) -> None:
            pass

        def on_class_start(
            self, class_idx: int, total_classes: int, class_name: str
        ) -> None:
            pass

        def on_phase(self, phase: str, message: str) -> None:
            pass

        def on_entry_start(self, index: int, total: int, key: str) -> None:
            pass

        def on_entry_end(self, index: int, total: int, key: str) -> None:
            calls.append((index, total, key))

    _ = run_feature(
        ds, ParallelFeature(), progress_callback=_Recorder(), execution_id="TESTPROG"
    )

    assert [(done, total) for done, total, _key in calls] == [(1, 3), (2, 3), (3, 3)]
    ends = _events(ds, "TESTPROG", "entry_end")
    assert [end["key"] for end in ends] == [key for _d, _t, key in calls]
    snapshot = read_run(run_log_dir(ds.base_dir), "TESTPROG")
    assert snapshot is not None
    assert (snapshot["progress_done"], snapshot["progress_total"]) == (3, 3)


# --- the run's resources ------------------------------------------------------


def test_the_resources_carry_the_run_cost_and_what_ran(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)

    _ = run_feature(
        ds,
        ParallelFeature(),
        parallel_workers=2,
        parallel_mode="thread",
        execution_id="TESTRESOURCES",
    )

    report = _last_resources(ds, "TESTRESOURCES")
    assert _number(report, "cpu_seconds") > 0
    assert _number(report, "wall_seconds") > 0
    assert _number(report, "apply_wall_seconds") > 0
    assert _number(report, "base_rss_bytes") > 0
    assert _number(report, "memory_limit_bytes") > 0
    assert report["memory_limit_source"] in {"cgroup_v2", "cgroup_v1", "host"}
    assert (report["feature"], report["feature_version"]) == ("prov-parallel", "0.1")
    assert report["process_capable"] is True
    assert _number(report, "fit_seconds") == 0
    assert _number(report, "worker_base_rss_bytes") == 0
    assert _no_sampler_alive()


class _RecordingEstimate(RowCostEstimate):
    """Keeps the rows in flight of every observation a run hands its estimate."""

    rows_seen: list[int] = []

    def observe(
        self, measured_bytes: int, in_flight_rows: int, live_workers: int
    ) -> None:
        _RecordingEstimate.rows_seen.append(in_flight_rows)
        super().observe(measured_bytes, in_flight_rows, live_workers)


@pytest.mark.parametrize("workers", [1, 2], ids=["inline", "thread"])
def test_every_entry_observes_the_estimate_with_its_rows_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    ds = _dataset(tmp_path)
    _RecordingEstimate.rows_seen = []
    monkeypatch.setattr(run_module, "RowCostEstimate", _RecordingEstimate)

    _ = run_feature(
        ds,
        ParallelFeature(),
        parallel_workers=workers,
        parallel_mode="thread",
        execution_id=f"TESTOBSERVED{workers}",
    )

    # Each entry's apply ends with an observation while its rows are in flight,
    # however short the entry, so the estimate exists after the first.
    with_rows = [rows for rows in _RecordingEstimate.rows_seen if rows > 0]
    assert len(with_rows) >= len(_SEQUENCES)


def test_a_feature_that_needs_its_scope_is_not_process_capable(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path)

    _ = run_feature(ds, _Scoped(), execution_id="TESTSCOPED")

    assert _last_resources(ds, "TESTSCOPED")["process_capable"] is False


# --- the in-flight record -------------------------------------------------------


class _Recording(EntriesInFlight):
    """Keeps every in-flight record a run makes, for the test to read after it."""

    made: list[EntriesInFlight] = []

    def __init__(self) -> None:
        super().__init__()
        _Recording.made.append(self)


@pytest.fixture
def recorded_in_flight(monkeypatch: pytest.MonkeyPatch) -> list[EntriesInFlight]:
    _Recording.made = []
    monkeypatch.setattr(run_module, "EntriesInFlight", _Recording)
    return _Recording.made


def _on_seq_b(
    monkeypatch: pytest.MonkeyPatch, answer: Callable[[], pd.DataFrame | None]
) -> None:
    """Make loading ``seq_b`` return or raise what *answer* gives."""
    real = manifest_module.load_entry_data

    def load(
        file_specs: list[tuple[Path, LoadSpec]],
        filters: Iterable[Callable[[pd.DataFrame], pd.DataFrame]] = (),
        *,
        cross_join: bool = False,
        expands_individuals: bool = False,
    ) -> pd.DataFrame | None:
        if any("seq_b" in str(path) for path, _ in file_specs):
            return answer()
        return real(
            file_specs,
            filters,
            cross_join=cross_join,
            expands_individuals=expands_individuals,
        )

    monkeypatch.setattr(manifest_module, "load_entry_data", load)


def _raise_pairs_missing() -> pd.DataFrame | None:
    raise PairRowsMissing("no pair rows")


def _no_table() -> pd.DataFrame | None:
    return None


@pytest.mark.parametrize(
    "answer", [_raise_pairs_missing, _no_table], ids=["pairs-missing", "no-table"]
)
def test_an_entry_that_never_reaches_apply_leaves_the_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_in_flight: list[EntriesInFlight],
    answer: Callable[[], pd.DataFrame | None],
) -> None:
    ds = _dataset(tmp_path)
    _on_seq_b(monkeypatch, answer)

    _ = run_feature(ds, ParallelFeature(), execution_id="TESTSKIPPED")

    (record,) = recorded_in_flight
    assert record.count() == 0
    ends = _events(ds, "TESTSKIPPED", "entry_end")
    assert [int(_number(end, "in_flight")) for end in ends] == [1, 1]


def test_an_entry_failing_its_column_check_leaves_the_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_in_flight: list[EntriesInFlight],
) -> None:
    ds = _dataset(tmp_path)
    calls: list[int] = []

    def fail_first(feature: object, columns: object, *, numeric: object) -> None:
        _ = (feature, columns, numeric)
        calls.append(1)
        if len(calls) == 1:
            raise MissingInputColumns("prov-parallel", (), ("tracks",))

    monkeypatch.setattr(run_module, "check_input_columns", fail_first)

    result = run_feature(ds, ParallelFeature(), execution_id="TESTCOLUMNS")

    assert len(result.failed_entries) == 1
    (record,) = recorded_in_flight
    assert record.count() == 0
    ends = _events(ds, "TESTCOLUMNS", "entry_end")
    assert [int(_number(end, "in_flight")) for end in ends] == [1, 1]


class _FailsOnSeqB(ParallelFeature):
    """Raises from ``apply`` on ``seq_b``."""

    name = "prov-parallel-fails"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty and str(df["sequence"].iloc[0]) == "seq_b":
            raise RuntimeError("a lost entry")
        return df


@pytest.mark.parametrize("workers", [1, 2], ids=["inline", "thread"])
def test_an_apply_that_raises_leaves_the_record(
    tmp_path: Path, recorded_in_flight: list[EntriesInFlight], workers: int
) -> None:
    ds = _dataset(tmp_path)

    result = run_feature(
        ds,
        _FailsOnSeqB(),
        parallel_workers=workers,
        parallel_mode="thread",
    )

    assert len(result.failed_entries) == 1
    (record,) = recorded_in_flight
    assert record.count() == 0


# --- the fit ------------------------------------------------------------------


def test_a_fit_records_its_cost_and_what_it_was_offered(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)

    _ = run_feature(ds, FittingFeature(FitControl()), execution_id="TESTFIT")

    report = _last_resources(ds, "TESTFIT")
    assert _number(report, "fit_seconds") > 0
    assert _number(report, "fit_peak_rss_bytes") > 0
    assert _number(report, "fit_entries") == len(_SEQUENCES)
    rows = sum(
        _number(end, "input_rows") for end in _events(ds, "TESTFIT", "entry_end")
    )
    assert _number(report, "fit_input_rows") == rows


def test_a_fit_in_progress_writes_its_resources(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    release = threading.Event()
    control = FitControl(finish_fit=release)
    failures: list[BaseException] = []

    def run() -> None:
        try:
            _ = run_feature(ds, FittingFeature(control), execution_id="TESTFITHELD")
        except BaseException as exc:  # reported to the test's thread
            failures.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert control.fitting.wait(30)
        path = run_log_path(ds.base_dir, "TESTFITHELD")
        deadline = time.monotonic() + 15
        written: list[dict[str, object]] = []
        while not written and time.monotonic() < deadline:
            written = _events(ds, "TESTFITHELD", "resources") if path.exists() else []
            time.sleep(0.1)
        assert written, "no resources event was written while the fit ran"
        assert _number(written[-1], "fit_peak_rss_bytes") > 0
    finally:
        release.set()
        worker.join(60)
    assert failures == []


def test_a_fit_that_raises_leaves_its_resources_and_no_sampler(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path)
    control = FitControl(fit_error=RuntimeError("the fit failed"))

    with pytest.raises(RuntimeError, match="the fit failed"):
        _ = run_feature(ds, FittingFeature(control), execution_id="TESTFITRAISES")

    report = _last_resources(ds, "TESTFITRAISES")
    assert _number(report, "wall_seconds") > 0
    assert _number(report, "fit_peak_rss_bytes") > 0
    assert _no_sampler_alive()


# --- the process's high-water mark -------------------------------------------


def test_a_high_water_mark_from_before_the_run_moves_no_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    earlier = 10**15
    monkeypatch.setattr(memory_watch, "max_rss_bytes", lambda: earlier)

    _ = run_feature(ds, FittingFeature(FitControl()), execution_id="TESTMAXRSS")

    report = _last_resources(ds, "TESTMAXRSS")
    assert 0 < _number(report, "peak_rss_bytes") < earlier
    assert 0 < _number(report, "fit_peak_rss_bytes") < earlier


def test_a_high_water_mark_that_rises_during_the_fit_raises_its_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    readings = iter([1, 10**15])
    monkeypatch.setattr(memory_watch, "max_rss_bytes", lambda: next(readings, 10**15))

    _ = run_feature(ds, FittingFeature(FitControl()), execution_id="TESTMAXRISE")

    assert _number(_last_resources(ds, "TESTMAXRISE"), "fit_peak_rss_bytes") == 10**15


# --- a process worker's own memory --------------------------------------------


def test_a_process_capable_feature_has_one_worker_measured(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)

    _ = run_feature(
        ds, ParallelFeature(), execution_id="TESTWORKERBASE", measure_worker_base=True
    )

    *_, frozen, measured = _events(ds, "TESTWORKERBASE", "resources")
    assert _number(frozen, "worker_base_rss_bytes") == 0
    assert _number(measured, "worker_base_rss_bytes") > 0
    assert _number(measured, "cpu_seconds") == _number(frozen, "cpu_seconds")
    assert "worker_base_nonreclaimable_bytes" in measured
    bases = {"t", "worker_base_rss_bytes", "worker_base_nonreclaimable_bytes"}
    assert {k: v for k, v in measured.items() if k not in bases} == {
        k: v for k, v in frozen.items() if k not in bases
    }


def test_a_process_workers_base_leaves_out_the_table_it_is_handed(
    tmp_path: Path,
) -> None:
    """The base of a worker handed a table is the base of one handed none.

    A process worker receives its entry's table as an argument, so the table is
    resident before ``apply``. Counted in the base, it would be read as memory
    every worker holds, and the probe's figure, measured without a table, would
    mean something else.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np

    rows = 5_000_000
    table = pd.DataFrame(
        {
            "X": np.arange(rows, dtype=np.float64),
            "Y": np.arange(rows, dtype=np.float64),
            "sequence": pd.Categorical(["seq_a"] * rows),
        }
    )
    table_bytes = int(table.memory_usage(deep=True).sum())
    feature = ParallelFeature()
    module, name = type(feature).__module__, type(feature).__name__
    inputs, params = feature.inputs.model_dump(), feature.params.model_dump()
    paths: dict[str, str] = {}
    lookups: dict[str, DependencyLookup] = {}

    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        empty = pool.submit(
            run_module.process_worker_base,
            module,
            name,
            inputs,
            params,
            str(tmp_path),
            paths,
            lookups,
        ).result()
        _, _, measure = pool.submit(
            run_module.process_apply_worker,
            module,
            name,
            inputs,
            params,
            str(tmp_path),
            paths,
            lookups,
            table,
        ).result()

    assert measure is not None
    assert empty.rss_bytes > 0
    assert abs(measure.base_rss_bytes - empty.rss_bytes) < table_bytes // 4


def test_a_feature_that_needs_its_scope_has_no_worker_measured(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path)

    _ = run_feature(
        ds, _Scoped(), execution_id="TESTWORKERSCOPED", measure_worker_base=True
    )

    assert (
        _number(_last_resources(ds, "TESTWORKERSCOPED"), "worker_base_rss_bytes") == 0
    )


def test_no_worker_is_measured_when_the_limit_has_less_left_than_the_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    nearly_full = MemoryLimit(
        limit_bytes=8 * 1024**3,
        source="cgroup_v2",
        in_use_bytes=8 * 1024**3 - 1,
        peak_bytes=0,
        host_total_bytes=64 * 1024**3,
        host_available_bytes=32 * 1024**3,
    )
    monkeypatch.setattr(memory_watch, "read_memory_limit", lambda: nearly_full)

    _ = run_feature(
        ds, ParallelFeature(), execution_id="TESTWORKERFULL", measure_worker_base=True
    )

    assert _number(_last_resources(ds, "TESTWORKERFULL"), "worker_base_rss_bytes") == 0


_APPLIED_IN = "MOSAIC_TEST_APPLIED_IN"
"""Names a directory where a process worker marks each sequence it has applied."""

_HELD_MIB = 64


class HoldsMemory(ParallelFeature):
    """Holds 64 MiB for a second while it applies every entry but ``seq_a``.

    A table of a few rows costs almost nothing beyond the base, so a run of them
    can estimate 0 bytes a row. The memory an entry holds while it computes is what
    the estimate is there to see. ``seq_a`` holds none, so what the run learns
    while only it is in flight is noise.

    Where the environment names a directory, a worker marks each sequence there
    once it has applied it, so the run's loads can wait for it.
    """

    name = "prov-parallel-holds"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if str(df["sequence"].iloc[0]) != "seq_a":
            # Random bytes, which a memory compressor cannot shrink out of
            # resident memory under pressure.
            held = os.urandom(_HELD_MIB * 1024**2)
            time.sleep(1.0)
            del held
        applied = os.environ.get(_APPLIED_IN)
        if applied:
            (Path(applied) / str(df["sequence"].iloc[0])).touch()
        return df


_LEARN_PER_ROW = """
import sys
import time
from pathlib import Path

from mosaic.core.pipeline import manifest
from mosaic.core.pipeline import run
from mosaic.core.pipeline.admission import RowCostEstimate
from tests.core.pipeline import test_run_measures as stand_ins

mode = sys.argv[2]
applied = Path(sys.argv[3]) if len(sys.argv) > 3 else None
real = manifest.load_entry_data


PREVIOUS = {"seq_b": "seq_a", "seq_c": "seq_b"}


def load(file_specs, filters=(), *, cross_join=False, expands_individuals=False):
    # Each load waits for the sequence before it to have been applied, so the pool
    # finds its one worker idle at every submission and never starts a second.
    waits = [
        before
        for sequence, before in PREVIOUS.items()
        if any(sequence in str(path) for path, _ in file_specs)
    ]
    if applied is not None and waits:
        deadline = time.monotonic() + 60
        while not (applied / waits[0]).exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)
    return real(
        file_specs,
        filters,
        cross_join=cross_join,
        expands_individuals=expands_individuals,
    )


live_workers = [0]


class Recording(RowCostEstimate):
    def observe(self, measured_bytes, in_flight_rows, live_workers_now):
        live_workers[0] = max(live_workers[0], live_workers_now)
        super().observe(measured_bytes, in_flight_rows, live_workers_now)


manifest.load_entry_data = load
run.RowCostEstimate = Recording
run.run_feature(
    stand_ins._dataset(Path(sys.argv[1])),
    stand_ins.HoldsMemory(),
    parallel_workers=2,
    parallel_mode=mode,
    execution_id="TESTPERROW",
)
print(live_workers[0])
"""


def _learn_per_row(tmp_path: Path, mode: str) -> tuple[dict[str, object], int, int]:
    """Run :class:`HoldsMemory` over three entries in a fresh interpreter.

    In a fresh interpreter, so the measured tree holds this run alone. A test
    process carries the memory and the child processes of earlier tests, which may
    shrink or exit while the run measures. In process mode each load waits for the
    entry before it to have been applied, so the pool keeps one worker.

    Returns:
        The run's last ``resources`` event, the largest input rows of an entry and
        the most live workers an observation of the estimate counted.
    """
    env = dict(os.environ)
    args = [str(tmp_path), mode]
    if mode == "process":
        applied = tmp_path / "applied"
        applied.mkdir()
        env[_APPLIED_IN] = str(applied)
        args.append(str(applied))

    printed = run_fresh(_LEARN_PER_ROW, args=args, env=env)

    base_dir = tmp_path / "ds"
    ends = _events_at(base_dir, "TESTPERROW", "entry_end")
    rows = max(int(_number(end, "input_rows")) for end in ends)
    most_live = int(printed.splitlines()[-1])
    return _events_at(base_dir, "TESTPERROW", "resources")[-1], rows, most_live


def test_a_thread_run_learns_what_its_entries_cost_per_row(tmp_path: Path) -> None:
    report, _rows, most_live = _learn_per_row(tmp_path, "thread")

    assert _number(report, "peak_bytes_per_row") > 0, json.dumps(report)
    assert most_live == 0


def test_a_process_run_counts_the_one_worker_its_pool_started(tmp_path: Path) -> None:
    report, rows, most_live = _learn_per_row(tmp_path, "process")

    # The pool started one worker, and no observation took off a second's base.
    assert most_live == 1

    # An entry holds 64 MiB over its own rows, which its worker's peak shows. Half
    # of it leaves room for the worker's own noise. Counting a second worker the
    # pool never started takes its base off every sample, and the estimate then
    # sees none of the held memory.
    per_row = _number(report, "peak_bytes_per_row")
    assert per_row * rows >= _HELD_MIB * 1024**2 // 2, json.dumps(report)


def test_a_process_mode_run_measures_its_workers_base(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path)

    _ = run_feature(
        ds,
        ParallelFeature(),
        parallel_workers=2,
        parallel_mode="process",
        execution_id="TESTPROCESS",
    )

    report = _last_resources(ds, "TESTPROCESS")
    assert _number(report, "worker_base_rss_bytes") > 0


# --- the entries in flight keep their largest sample ----------------------------


def test_a_flight_keeps_its_largest_sample() -> None:
    in_flight = EntriesInFlight()
    in_flight.enter("a", 10, started=0.0, cpu=0.0)
    in_flight.observe_memory(500)
    in_flight.enter("b", 20, started=1.0, cpu=0.0)
    in_flight.observe_memory(900)
    in_flight.observe_memory(700)

    span_a = in_flight.leave("a")
    in_flight.observe_memory(800)
    span_b = in_flight.leave("b")

    assert span_a is not None
    assert span_a.peak_nonreclaimable_bytes == 900
    assert span_b is not None
    assert span_b.peak_nonreclaimable_bytes == 900


def test_an_entry_that_saw_no_sample_reads_none() -> None:
    in_flight = EntriesInFlight()
    in_flight.enter("a", 10, started=0.0, cpu=0.0)
    in_flight.observe_memory(500)
    _ = in_flight.leave("a")
    in_flight.enter("b", 20, started=1.0, cpu=0.0)

    span = in_flight.leave("b")

    assert span is not None
    assert span.peak_nonreclaimable_bytes is None


def test_an_entry_that_entered_after_a_sample_began_takes_no_figure_from_it() -> None:
    in_flight = EntriesInFlight()
    in_flight.enter("early", 10, started=1.0, cpu=0.0)
    in_flight.enter("late", 20, started=3.0, cpu=0.0)

    in_flight.observe_memory(900, sampled_from=2.0)
    early = in_flight.leave("early")
    late = in_flight.leave("late")

    assert early is not None
    assert early.peak_nonreclaimable_bytes == 900
    assert late is not None
    assert late.peak_nonreclaimable_bytes is None


def test_an_entry_that_entered_when_a_sample_began_takes_its_figure() -> None:
    in_flight = EntriesInFlight()
    in_flight.enter("a", 10, started=2.0, cpu=0.0)

    in_flight.observe_memory(900, sampled_from=2.0)

    span = in_flight.leave("a")
    assert span is not None
    assert span.peak_nonreclaimable_bytes == 900


def test_a_sample_counts_only_the_entries_that_entered_before_it_began() -> None:
    in_flight = EntriesInFlight()
    in_flight.enter("early", 10, started=1.0, cpu=0.0)
    in_flight.enter("late", 20, started=3.0, cpu=0.0)

    assert in_flight.totals(sampled_from=2.0) == (1, 10)
    assert in_flight.totals(sampled_from=3.0) == (2, 30)
    assert in_flight.totals() == (2, 30)


def test_the_entries_and_rows_in_flight_are_read_together() -> None:
    in_flight = EntriesInFlight()
    assert in_flight.totals() == (0, 0)
    in_flight.enter("a", 10, started=0.0, cpu=0.0)
    in_flight.enter("b", 20, started=0.0, cpu=0.0)

    assert in_flight.totals() == (2, 30)

    _ = in_flight.leave("a")

    assert in_flight.totals() == (1, 20)


# --- the run's measure -----------------------------------------------------------

_BASE = 7 * _GIB + 12_345
"""A non-reclaimable figure no tree of this test's size reaches."""


class _Spy(RowCostEstimate):
    """Keeps every figure the run hands its estimate."""

    bases: list[int] = []
    worker_bases: list[int] = []
    worker_flags: list[bool] = []
    observed: list[int] = []
    peak_samples: list[PeakSample] = []
    base_fixed: threading.Event = threading.Event()

    @property
    def bytes_per_row(self) -> int | None:
        return 1

    def parent_base(self, bytes_: int) -> None:
        _Spy.bases.append(bytes_)
        _Spy.base_fixed.set()
        super().parent_base(bytes_)

    def worker_base(self, bytes_: int) -> None:
        _Spy.worker_bases.append(bytes_)
        super().worker_base(bytes_)

    def observe_worker(
        self, measure: WorkerMeasure, rows: int, *, nonreclaimable: bool
    ) -> None:
        _Spy.worker_flags.append(nonreclaimable)
        super().observe_worker(measure, rows, nonreclaimable=nonreclaimable)

    def observe(
        self, measured_bytes: int, in_flight_rows: int, live_workers: int
    ) -> None:
        _Spy.observed.append(measured_bytes)
        super().observe(measured_bytes, in_flight_rows, live_workers)


class _Sleeps(ParallelFeature):
    """Holds each entry for a moment, so a later one loads beside it."""

    name = "prov-parallel-sleeps"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        time.sleep(0.4)
        return df


@pytest.fixture
def spied(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[int]]:
    """The figures a run hands its estimate and its budget, and its peak samples.

    The estimate reports 1 byte a row once asked, so the gate computes a budget
    whenever an entry loads beside another.
    """
    _Spy.bases, _Spy.worker_bases, _Spy.worker_flags = [], [], []
    _Spy.observed, _Spy.peak_samples = [], []
    _Spy.base_fixed = threading.Event()
    held: list[int] = []
    peaks: list[int] = []
    real_budget = run_module.admission_budget

    def budget(
        *, available_bytes: int, held_bytes: int, limit_bytes: int, floor: float
    ) -> int:
        held.append(held_bytes)
        return real_budget(
            available_bytes=available_bytes,
            held_bytes=held_bytes,
            limit_bytes=limit_bytes,
            floor=floor,
        )

    real_peak = run_module.larger_peak

    def larger(
        peak: PeakSample | None,
        *,
        bytes_: int,
        entries: int,
        rows: int,
        live_workers: int,
    ) -> PeakSample | None:
        peaks.append(bytes_)
        found = real_peak(
            peak, bytes_=bytes_, entries=entries, rows=rows, live_workers=live_workers
        )
        if found is not None:
            _Spy.peak_samples.append(found)
        return found

    monkeypatch.setattr(run_module, "RowCostEstimate", _Spy)
    monkeypatch.setattr(run_module, "admission_budget", budget)
    monkeypatch.setattr(run_module, "larger_peak", larger)
    return {
        "bases": _Spy.bases,
        "observed": _Spy.observed,
        "held": held,
        "peaks": peaks,
    }


def _figure_after_the_base(
    monkeypatch: pytest.MonkeyPatch,
    before: Callable[[], int],
    after: Callable[[], int],
) -> None:
    """Make the limit's non-reclaimable figure *before* until the run fixes its base.

    The estimate takes the run's base once, right after the base sample, so the
    figure switches to *after* from then on.
    """

    def read() -> MemoryLimit:
        return MemoryLimit(
            limit_bytes=64 * _GIB,
            source="cgroup_v2",
            in_use_bytes=_GIB,
            peak_bytes=0,
            host_total_bytes=128 * _GIB,
            host_available_bytes=64 * _GIB,
            nonreclaimable_bytes=after() if _Spy.base_fixed.is_set() else before(),
        )

    monkeypatch.setattr(memory_watch, "read_memory_limit", read)


def _run_gated(ds: Dataset, execution_id: str) -> None:
    _ = run_feature(
        ds,
        _Sleeps(),
        parallel_workers=2,
        parallel_mode="thread",
        admission_margin=1.5,
        execution_id=execution_id,
    )


def test_a_run_whose_base_measured_the_figure_gates_on_it_throughout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spied: dict[str, list[int]]
) -> None:
    ds = _dataset(tmp_path)
    _figure_after_the_base(monkeypatch, lambda: _BASE, lambda: _BASE + _GIB)

    _run_gated(ds, "TESTMEASURENR")

    assert spied["bases"] == [_BASE]
    assert set(spied["observed"]) <= {_BASE, _BASE + _GIB}
    assert _BASE + _GIB in spied["observed"]
    assert spied["held"], "the gate never computed a budget"
    assert set(spied["held"]) == {_BASE + _GIB}
    assert max(spied["peaks"]) == _BASE + _GIB


def test_a_run_whose_base_had_no_figure_gates_on_tree_rss_throughout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spied: dict[str, list[int]]
) -> None:
    ds = _dataset(tmp_path)
    # The figure appears after the base sample, which the run has already measured
    # as tree RSS.
    _figure_after_the_base(monkeypatch, lambda: 0, lambda: _BASE)

    _run_gated(ds, "TESTMEASURERSS")

    base_rss = _number(_last_resources(ds, "TESTMEASURERSS"), "base_rss_bytes")
    assert spied["bases"] == [int(base_rss)]
    assert _BASE not in spied["observed"]
    assert spied["observed"]
    assert spied["held"], "the gate never computed a budget"
    assert _BASE not in spied["held"]
    assert spied["peaks"] == []


def test_a_sample_without_the_figure_is_never_read_as_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spied: dict[str, list[int]]
) -> None:
    ds = _dataset(tmp_path)
    readings = iter(range(10**6))
    # After the base, every second sample has no figure.
    _figure_after_the_base(
        monkeypatch,
        lambda: _BASE,
        lambda: 0 if next(readings) % 2 else _BASE + _GIB,
    )

    _run_gated(ds, "TESTMEASUREGAP")

    assert 0 not in spied["observed"]
    assert 0 not in spied["held"]
    assert 0 not in spied["peaks"]
    assert _BASE + _GIB in spied["observed"]


def test_a_process_run_takes_its_worker_figures_in_the_runs_measure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spied: dict[str, list[int]]
) -> None:
    _ = spied
    ds = _dataset(tmp_path)
    _figure_after_the_base(monkeypatch, lambda: _BASE, lambda: _BASE + _GIB)

    _ = run_feature(
        ds,
        ParallelFeature(),
        parallel_workers=2,
        parallel_mode="process",
        execution_id="TESTPROCESSMEASURE",
    )

    # Every worker's measure was taken in the run's measure, whether or not the
    # worker could read its figure: without /proc (macOS) it cannot, and then the
    # estimate takes no worker base, rather than the worker's resident memory.
    assert _Spy.worker_flags
    assert all(_Spy.worker_flags)
    if not Path("/proc/self/status").exists():
        assert _Spy.worker_bases == []
    else:
        assert _Spy.worker_bases
    report = _last_resources(ds, "TESTPROCESSMEASURE")
    assert _number(report, "worker_base_rss_bytes") > 0


def test_the_peak_sample_has_an_entry_in_flight_and_never_comes_from_the_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spied: dict[str, list[int]]
) -> None:
    ds = _dataset(tmp_path)
    # The fit and the base sample see a large figure, and the apply a smaller one
    # that grows with each sample.
    growth = iter(range(10**6))
    _figure_after_the_base(
        monkeypatch, lambda: 50 * _GIB, lambda: _BASE + next(growth) * 1024**2
    )

    _ = run_feature(
        ds,
        FittingFeature(FitControl()),
        execution_id="TESTMEASUREPEAK",
    )

    assert spied["bases"] == [50 * _GIB]
    assert _Spy.peak_samples
    for sample in _Spy.peak_samples:
        assert sample.entries >= 1
        assert sample.rows > 0
        assert _BASE <= sample.bytes_ < 50 * _GIB


def _non_reclaimable_growth(
    base: int | None,
    peak: int | None,
    module_name: str,
    class_name: str,
    inputs_dump: dict[str, object],
    params_dump: dict[str, object],
    run_root_str: str,
    artifact_paths_str: dict[str, str],
    dependency_lookups: dict[str, DependencyLookup],
    df: pd.DataFrame,
    ds_manifest_path: str | None = None,
    tracklets: Mapping[Entry, bool | None] | None = None,
) -> ApplyOutcome:
    """Apply as a process worker does, and report the given non-reclaimable pair."""
    output, notices, measure = process_apply_worker(
        module_name,
        class_name,
        inputs_dump,
        params_dump,
        run_root_str,
        artifact_paths_str,
        dependency_lookups,
        df,
        ds_manifest_path,
        tracklets,
    )
    assert measure is not None
    measure = dataclasses.replace(
        measure, base_nonreclaimable_bytes=base, peak_nonreclaimable_bytes=peak
    )
    return output, notices, measure


def _grown_ends(ds: Dataset, execution_id: str) -> list[tuple[int, object]]:
    return [
        (int(_number(end, "in_flight")), end.get("peak_grown_bytes"))
        for end in _events(ds, execution_id, "entry_end")
    ]


def test_the_resources_carry_the_peak_sample_and_its_growth_over_the_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spied: dict[str, list[int]]
) -> None:
    _ = spied
    ds = _dataset(tmp_path)
    _figure_after_the_base(monkeypatch, lambda: _BASE, lambda: _BASE + _GIB)

    _run_gated(ds, "TESTPEAKGROWN")

    report = _last_resources(ds, "TESTPEAKGROWN")
    assert report["base_nonreclaimable_bytes"] == _BASE
    assert report["nonreclaimable_bytes"] == _BASE + _GIB
    assert report["peak_nonreclaimable_bytes"] == _BASE + _GIB
    assert report["peak_grown_bytes"] == _GIB
    peak = _Spy.peak_samples[-1]
    assert peak.entries >= 1
    assert peak.rows > 0
    assert report["peak_in_flight_entries"] == peak.entries
    assert report["peak_in_flight_rows"] == peak.rows


@pytest.mark.parametrize("workers", [1, 2], ids=["inline", "thread"])
def test_an_alone_thread_entry_carries_its_growth_and_a_paired_one_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spied: dict[str, list[int]],
    workers: int,
) -> None:
    _ = spied
    ds = _dataset(tmp_path)
    _figure_after_the_base(monkeypatch, lambda: _BASE, lambda: _BASE + _GIB)

    _ = run_feature(
        ds,
        _Sleeps(),
        parallel_workers=workers,
        parallel_mode="thread",
        execution_id=f"TESTGROWTHALONE{workers}",
    )

    ends = _grown_ends(ds, f"TESTGROWTHALONE{workers}")
    assert len(ends) == len(_SEQUENCES)
    for in_flight, grown in ends:
        assert grown == (_GIB if in_flight == 1 else None)
    assert any(grown is None for _, grown in ends) == (workers == 2)
    if workers == 2:
        assert any(in_flight > 1 for in_flight, _ in ends)


def test_a_run_that_measures_tree_rss_records_no_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spied: dict[str, list[int]]
) -> None:
    _ = spied
    ds = _dataset(tmp_path)
    _figure_after_the_base(monkeypatch, lambda: 0, lambda: 0)

    _ = run_feature(ds, ParallelFeature(), execution_id="TESTGROWTHRSS")

    report = _last_resources(ds, "TESTGROWTHRSS")
    assert [grown for _, grown in _grown_ends(ds, "TESTGROWTHRSS")] == [None] * 3
    for field in (
        "nonreclaimable_bytes",
        "peak_nonreclaimable_bytes",
        "base_nonreclaimable_bytes",
        "worker_base_nonreclaimable_bytes",
        "peak_grown_bytes",
    ):
        assert report[field] == 0
    assert _number(report, "peak_in_flight_entries") == 0


def test_every_process_entry_carries_its_workers_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spied: dict[str, list[int]]
) -> None:
    _ = spied
    ds = _dataset(tmp_path)
    samples = iter(range(10**6))
    _figure_after_the_base(
        monkeypatch, lambda: _BASE, lambda: _BASE + _GIB + next(samples) * 1024
    )
    worker_base = _GIB // 4
    monkeypatch.setattr(
        run_module,
        "process_apply_worker",
        functools.partial(
            _non_reclaimable_growth, worker_base, worker_base + _GIB // 2
        ),
    )

    _ = run_feature(
        ds,
        ParallelFeature(),
        parallel_workers=2,
        parallel_mode="process",
        execution_id="TESTGROWTHPROCESS",
    )

    ends = _grown_ends(ds, "TESTGROWTHPROCESS")
    assert [grown for _, grown in ends] == [_GIB // 2] * 3
    report = _last_resources(ds, "TESTGROWTHPROCESS")
    assert report["worker_base_nonreclaimable_bytes"] == worker_base
    # The parent's base and one or two workers' bases come off the peak sample.
    assert _GIB // 2 <= _number(report, "peak_grown_bytes") <= _GIB * 3 // 4 + 10**6


def test_a_process_entry_whose_worker_gave_no_figure_carries_no_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spied: dict[str, list[int]]
) -> None:
    _ = spied
    ds = _dataset(tmp_path)
    samples = iter(range(10**6))
    _figure_after_the_base(
        monkeypatch, lambda: _BASE, lambda: _BASE + _GIB + next(samples) * 1024
    )
    monkeypatch.setattr(
        run_module,
        "process_apply_worker",
        functools.partial(_non_reclaimable_growth, None, None),
    )

    _ = run_feature(
        ds,
        ParallelFeature(),
        parallel_workers=2,
        parallel_mode="process",
        admission_margin=1.5,
        execution_id="TESTGROWTHBLIND",
    )

    ends = _grown_ends(ds, "TESTGROWTHBLIND")
    # The gate starts the first entry alone, where the parent's own sample would
    # give a figure if the entry were not left to its worker.
    assert any(in_flight == 1 for in_flight, _ in ends)
    assert [grown for _, grown in ends] == [None] * 3
    report = _last_resources(ds, "TESTGROWTHBLIND")
    assert report["worker_base_nonreclaimable_bytes"] == 0
    # A worker base that no worker gave is not subtracted as 0.
    assert report["peak_grown_bytes"] == 0


# --- a process worker's two bases ---------------------------------------------------


def _base_of_a_worker(tmp_path: Path) -> WorkerBase:
    feature = ParallelFeature()
    return run_module.process_worker_base(
        type(feature).__module__,
        type(feature).__name__,
        feature.inputs.model_dump(),
        feature.params.model_dump(),
        str(tmp_path),
        {},
        {},
    )


def test_a_process_worker_reports_both_bases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_module, "process_nonreclaimable_bytes", lambda: 123_456)

    base = _base_of_a_worker(tmp_path)

    assert base.rss_bytes > 0
    assert base.nonreclaimable_bytes == 123_456


def test_a_process_worker_without_the_figure_reports_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_module, "process_nonreclaimable_bytes", lambda: None)

    base = _base_of_a_worker(tmp_path)

    assert base.rss_bytes > 0
    assert base.nonreclaimable_bytes is None


_SAMPLING: dict[str, bool] = {"active": False, "during_apply": False}


class _SeesTheSampler(ParallelFeature):
    name = "prov-parallel-sees-sampler"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        _SAMPLING["during_apply"] = _SAMPLING["active"]
        return df


def test_a_process_worker_measures_its_non_reclaimable_growth_around_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = 5 * _GIB
    monkeypatch.setattr(run_module, "process_nonreclaimable_bytes", lambda: reader)

    @contextmanager
    def peak_while() -> Generator[memory_watch.NonreclaimablePeak]:
        _SAMPLING["active"] = True
        try:
            yield memory_watch.NonreclaimablePeak(peak=reader + 4096)
        finally:
            _SAMPLING["active"] = False

    monkeypatch.setattr(run_module, "peak_nonreclaimable_while", peak_while)
    _SAMPLING["during_apply"] = False
    feature = _SeesTheSampler()
    table = pd.DataFrame({"sequence": ["seq_a"] * 100, "value": range(100)})
    handed = int(table.memory_usage(deep=True).sum())

    _, _, measure = run_module.process_apply_worker(
        type(feature).__module__,
        type(feature).__name__,
        feature.inputs.model_dump(),
        feature.params.model_dump(),
        str(tmp_path),
        {},
        {},
        table,
    )

    assert _SAMPLING["during_apply"]
    assert isinstance(measure, WorkerMeasure)
    assert measure.base_nonreclaimable_bytes == reader - handed
    assert measure.peak_nonreclaimable_bytes == reader + 4096


def test_a_process_worker_without_the_figure_leaves_both_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_module, "process_nonreclaimable_bytes", lambda: None)

    @contextmanager
    def peak_while() -> Generator[memory_watch.NonreclaimablePeak]:
        yield memory_watch.NonreclaimablePeak()

    monkeypatch.setattr(run_module, "peak_nonreclaimable_while", peak_while)
    feature = ParallelFeature()
    table = pd.DataFrame({"sequence": ["seq_a"] * 10})

    _, _, measure = run_module.process_apply_worker(
        type(feature).__module__,
        type(feature).__name__,
        feature.inputs.model_dump(),
        feature.params.model_dump(),
        str(tmp_path),
        {},
        {},
        table,
    )

    assert isinstance(measure, WorkerMeasure)
    assert measure.base_nonreclaimable_bytes is None
    assert measure.peak_nonreclaimable_bytes is None


def test_a_sample_that_began_before_every_entry_gives_none_a_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spied: dict[str, list[int]]
) -> None:
    _ = spied
    ds = _dataset(tmp_path)
    _figure_after_the_base(monkeypatch, lambda: _BASE, lambda: _BASE + _GIB)
    real_sample = MemoryWatch.sample

    def began_first(self: MemoryWatch) -> MemorySample | None:
        sample = real_sample(self)
        if sample is None:
            return None
        return dataclasses.replace(sample, started_at=-1.0)

    monkeypatch.setattr(MemoryWatch, "sample", began_first)

    _ = run_feature(ds, _Sleeps(), execution_id="TESTSAMPLEBEFORE")

    assert {grown for _, grown in _grown_ends(ds, "TESTSAMPLEBEFORE")} == {None}
    report = _last_resources(ds, "TESTSAMPLEBEFORE")
    assert report["peak_in_flight_entries"] == 0
    assert report["peak_in_flight_rows"] == 0
