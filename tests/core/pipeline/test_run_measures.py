"""What a feature run measures of its own cost, per entry and for the whole run.

Each computed entry's ``entry_end`` carries its wall, apply and CPU seconds, its input
rows and how many entries were in flight beside it. The ``resources`` event carries the
run's CPU and wall time, its memory limit and base, the fit's cost and the per-row
estimate, and is written during a fit as during apply.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline import manifest as manifest_module
from mosaic.core.pipeline import memory_watch
from mosaic.core.pipeline import run as run_module
from mosaic.core.pipeline.column_contract import MissingInputColumns
from mosaic.core.pipeline.loading import PairRowsMissing
from mosaic.core.pipeline.memory_limit import MemoryLimit
from mosaic.core.pipeline.admission import EntriesInFlight, RowCostEstimate
from mosaic.core.pipeline.run import run_feature
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

    def observe(self, tree_rss: int, in_flight_rows: int, live_workers: int) -> None:
        _RecordingEstimate.rows_seen.append(in_flight_rows)
        super().observe(tree_rss, in_flight_rows, live_workers)


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
    assert {
        k: v for k, v in measured.items() if k not in {"t", "worker_base_rss_bytes"}
    } == {k: v for k, v in frozen.items() if k not in {"t", "worker_base_rss_bytes"}}


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
    assert empty > 0
    assert abs(measure.base_rss_bytes - empty) < table_bytes // 4


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
    def observe(self, tree_rss, in_flight_rows, live_workers_now):
        live_workers[0] = max(live_workers[0], live_workers_now)
        super().observe(tree_rss, in_flight_rows, live_workers_now)


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
