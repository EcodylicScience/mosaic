"""What a feature run measures of its own cost, per entry and for the whole run.

Each computed entry's ``entry_end`` carries its wall, apply and CPU seconds, its input
rows and how many entries were in flight beside it. The ``resources`` event carries the
run's CPU and wall time, its memory limit and base, the fit's cost and the per-row
estimate, and is written during a fit as during apply.
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
import json
import os
import threading
import time
from collections.abc import Callable, Collection, Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import override

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pydantic import TypeAdapter

from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry, parse_entry_key
from mosaic.core.filter_bounds import TimeWindow
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline import memory_watch
from mosaic.core.pipeline.memory_watch import MemorySample, MemoryWatch
from mosaic.core.pipeline import entry_task as entry_task_module
from mosaic.core.pipeline import run as run_module
from mosaic.core.pipeline.column_contract import MissingInputColumns
from mosaic.core.pipeline.loading import ColumnProjection, PairRowsMissing
from mosaic.core.pipeline.memory_limit import MemoryLimit
from mosaic.core.pipeline.admission import (
    EntriesInFlight,
    PeakSample,
    RowCostEstimate,
    WorkerBase,
    WorkerMeasure,
)
from mosaic.core.pipeline.entry_pool import EntryFailure
from mosaic.core.pipeline.entry_task import EntryLoad, EntryReport, EntryTask
from mosaic.core.pipeline.input_size import EntryCovariates, entry_covariates
from mosaic.core.pipeline.params_facts import (
    params_equal_defaults,
    params_identity_digest,
)
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.pipeline.types import DependencyLookup, InputStream
from mosaic.core.pipeline.run import (
    FeatureBuild,
    FeatureWorker,
    build_feature_meta,
    run_feature,
)
from mosaic.core.pipeline._loaders import LoadSpec, ParquetLoadSpec
from mosaic.runlog import (
    FlightSpan,
    InFlightLevel,
    read_run,
    run_log_dir,
    run_log_path,
    sweep_levels,
)
from tests.helpers import (
    HOLD_DIRECTORY_VARIABLE,
    CountsItsBuilds,
    FitControl,
    FittingFeature,
    HoldsUntilReleased,
    ParallelFeature,
    add_tracks_table,
    add_tracks_variant,
    logged_events,
    make_dataset,
    report_of,
    run_fresh,
    track_table,
    wrapped_write,
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


def _last_resources(ds: Dataset, execution_id: str) -> dict[str, object]:
    return logged_events(ds.base_dir, execution_id, "resources")[-1]


def _number(record: dict[str, object], key: str) -> float:
    value = record[key]
    assert isinstance(value, int | float)
    return float(value)


def _no_sampler_alive() -> bool:
    return not any(thread.name == _SAMPLER for thread in threading.enumerate())


class _NeedsScope(ParallelFeature):
    """A parallelizable feature whose ``apply`` fails on an instance without its scope.

    A process worker rebuilds the feature from its class and parameters, and
    ``run_feature`` calls ``set_scope`` on the instance it holds alone.
    """

    name = "prov-parallel-needs-scope"
    resolved: object | None = None

    def set_scope(self, scope: object) -> None:
        self.resolved = scope

    @override
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.resolved is None:
            raise RuntimeError("apply ran without the run's scope")
        return super().apply(df)


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

    ends = logged_events(ds.base_dir, execution_id, "entry_end")
    assert len(ends) == len(_SEQUENCES)
    for end in ends:
        assert all(field in end for field in _MEASURE_FIELDS)
        assert _number(end, "input_rows") > 0
        assert _number(end, "wall_seconds") >= _number(end, "apply_seconds") >= 0
        assert _number(end, "cpu_seconds") >= 0
    in_flight = {
        parse_entry_key(str(end["key"]))[1]: int(_number(end, "in_flight"))
        for end in ends
    }
    if workers == 1:
        assert in_flight == dict.fromkeys(_SEQUENCES, 1)
    else:
        # seq_a and seq_b are dispatched together and always overlap. seq_c waits
        # for a worker, and runs alone when both finish before that wait drains.
        assert all(count <= workers + 1 for count in in_flight.values())
        assert all(
            in_flight[sequence] >= least_in_flight for sequence in ("seq_a", "seq_b")
        )


class _SleepsHalfASecond(ParallelFeature):
    """Holds each entry for half a second in ``apply``."""

    name = "prov-parallel-sleeps-half"
    apply_sleep_seconds = 0.5

    @override
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        time.sleep(self.apply_sleep_seconds)
        return df


class _SleepsASecond(_SleepsHalfASecond):
    """Holds each entry for a second in ``apply``."""

    name = "prov-parallel-sleeps-second"
    apply_sleep_seconds = 1.0


@pytest.mark.parametrize("mode", ["thread", "process"])
def test_no_more_entries_than_workers_are_in_flight(tmp_path: Path, mode: str) -> None:
    ds = make_dataset(tmp_path / "ds")
    add_tracks_variant(ds, VARIANT, *_SEQUENCES, "seq_d", std_format="mosaic_v1")
    execution_id = f"TESTATMOSTWORKERS{mode.upper()}"

    _ = run_feature(
        ds,
        _SleepsHalfASecond(),
        parallel_workers=2,
        parallel_mode=mode,
        execution_id=execution_id,
    )

    ends = logged_events(ds.base_dir, execution_id, "entry_end")
    assert [end["mode"] for end in ends] == [mode] * (len(_SEQUENCES) + 1)
    counts = [int(_number(end, "in_flight")) for end in ends]
    assert max(counts) <= 2, counts
    levels = [level.in_flight for level in _written_levels(ds, execution_id)]
    assert max(levels) <= 2, levels


def test_an_entrys_flight_leaves_out_its_wait_for_a_worker(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)

    _ = run_feature(
        ds,
        _SleepsASecond(),
        parallel_workers=2,
        parallel_mode="thread",
        execution_id="TESTFLIGHTFROMDISPATCH",
    )

    ends = logged_events(ds.base_dir, "TESTFLIGHTFROMDISPATCH", "entry_end")
    assert len(ends) == len(_SEQUENCES)
    # What the flight holds beyond the dispatch to the end of ``apply`` is the
    # write. An entry that waited for a worker inside its flight holds that wait
    # too, about the length of the sleep.
    for end in ends:
        assert _number(end, "wall_seconds") - _number(end, "apply_seconds") < 0.5


def test_a_cache_hit_records_no_cost(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    _ = run_feature(ds, ParallelFeature())

    _ = run_feature(ds, ParallelFeature(), execution_id="TESTMEASUREHIT")

    ends = logged_events(ds.base_dir, "TESTMEASUREHIT", "entry_end")
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
    ends = logged_events(ds.base_dir, "TESTPROG", "entry_end")
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

    _ = run_feature(ds, _NeedsScope(), execution_id="TESTSCOPED")

    assert _last_resources(ds, "TESTSCOPED")["process_capable"] is False


@pytest.mark.parametrize("mode", ["process", "Process"])
def test_a_feature_that_needs_its_scope_computes_in_threads_when_processes_are_asked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mode: str
) -> None:
    ds = _dataset(tmp_path)

    result = run_feature(
        ds,
        _NeedsScope(),
        parallel_workers=2,
        parallel_mode=mode,
        execution_id="TESTSCOPEDPROCESS",
    )

    assert result.failed_entries == ()
    report = _last_resources(ds, "TESTSCOPEDPROCESS")
    assert report["parallel_mode"] == "thread"
    assert report["workers"] == 2
    assert report["process_capable"] is False
    assert _number(report, "worker_base_rss_bytes") == 0
    assert (
        "[feature:prov-parallel-needs-scope] parallel_mode process requested but "
        "feature cannot run in worker processes; running in threads."
    ) in capsys.readouterr().err


def test_a_phase_that_asks_for_processes_runs_a_scope_feature_in_threads(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path)
    entries = tuple(("", sequence) for sequence in _SEQUENCES)

    result = run_feature(
        ds,
        _NeedsScope(),
        phases=[
            run_module.ApplyPhase(
                entries=entries,
                workers=2,
                mode="process",
                admission_margin=0.0,
                measure_worker_base=False,
            )
        ],
        execution_id="TESTSCOPEDPHASE",
    )

    assert result.failed_entries == ()
    assert _last_resources(ds, "TESTSCOPEDPHASE")["parallel_mode"] == "thread"


def test_a_feature_without_a_scope_still_computes_in_processes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds = _dataset(tmp_path)

    _ = run_feature(
        ds,
        ParallelFeature(),
        parallel_workers=2,
        parallel_mode="process",
        execution_id="TESTPLAINPROCESS",
    )

    assert _last_resources(ds, "TESTPLAINPROCESS")["parallel_mode"] == "process"
    assert "running in threads" not in capsys.readouterr().err


# --- the in-flight record -------------------------------------------------------


class _Recording(EntriesInFlight):
    """Keeps every in-flight record a run makes, for the test to read after it."""

    made: list[EntriesInFlight] = []

    def __init__(
        self,
        *,
        cpu_seconds: Callable[[], float],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(cpu_seconds=cpu_seconds, clock=clock)
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
    real = entry_task_module.load_entry_data

    def load(
        file_specs: list[tuple[Path, LoadSpec]],
        filters: Iterable[Callable[[pd.DataFrame], pd.DataFrame]] = (),
        *,
        cross_join: bool = False,
        expands_individuals: bool = False,
        projection: ColumnProjection | None = None,
    ) -> pd.DataFrame | None:
        if any("seq_b" in str(path) for path, _ in file_specs):
            return answer()
        return real(
            file_specs,
            filters,
            cross_join=cross_join,
            expands_individuals=expands_individuals,
            projection=projection,
        )

    monkeypatch.setattr(entry_task_module, "load_entry_data", load)


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
    ends = logged_events(ds.base_dir, "TESTSKIPPED", "entry_end")
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

    monkeypatch.setattr(entry_task_module, "check_input_columns", fail_first)

    result = run_feature(ds, ParallelFeature(), execution_id="TESTCOLUMNS")

    assert len(result.failed_entries) == 1
    (record,) = recorded_in_flight
    assert record.count() == 0
    ends = logged_events(ds.base_dir, "TESTCOLUMNS", "entry_end")
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


_LATE_START_FRAME = 100
"""The first frame of ``seq_b`` in the fixture of an entry with no rows."""


def _seq_b_after_the_bounds(tmp_path: Path) -> Dataset:
    """A dataset whose ``seq_b`` starts after the frames of ``seq_a`` and ``seq_c``."""
    ds = make_dataset(tmp_path / "ds")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_c", std_format="mosaic_v1")
    late = track_table("", "seq_b", n_ids=2, start_frame=_LATE_START_FRAME)
    add_tracks_table(ds, VARIANT, late, std_format="mosaic_v1")
    return ds


@pytest.mark.parametrize(
    ("workers", "mode"),
    [(1, "thread"), (2, "thread"), (2, "process")],
    ids=["serial", "thread", "process"],
)
def test_an_entry_with_no_rows_in_its_bounds_leaves_the_record(
    tmp_path: Path,
    recorded_in_flight: list[EntriesInFlight],
    workers: int,
    mode: str,
) -> None:
    ds = _seq_b_after_the_bounds(tmp_path)
    execution_id = f"TESTNOROWS{workers}{mode.upper()}"

    result = run_feature(
        ds,
        ParallelFeature(),
        parallel_workers=workers,
        parallel_mode=mode,
        filter_end_frame=_LATE_START_FRAME // 2,
        execution_id=execution_id,
    )

    assert result.entries_written == 2
    assert result.failed_entries == ()
    assert logged_events(ds.base_dir, execution_id, "entry_error") == []
    (record,) = recorded_in_flight
    assert record.count() == 0


_SLOW_WRITE_SECONDS = 1.0
"""How long the write of each entry is slowed in the tests of ``apply_seconds``."""


class _WritesSlowly(HoldsUntilReleased):
    """Waits :data:`_SLOW_WRITE_SECONDS` before the write of each entry, in its worker."""

    fault = "slow-write"
    fault_on = _SEQUENCES
    write_seconds = _SLOW_WRITE_SECONDS


@pytest.mark.parametrize(
    ("workers", "mode"),
    [(1, "thread"), (2, "thread"), (2, "process")],
    ids=["serial", "thread", "process"],
)
def test_the_apply_seconds_of_an_entry_leave_out_its_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int, mode: str
) -> None:
    ds = _dataset(tmp_path)
    marks = tmp_path / "marks"
    marks.mkdir()
    monkeypatch.setenv(HOLD_DIRECTORY_VARIABLE, str(marks))
    in_processes = mode == "process"
    if not in_processes:
        monkeypatch.setattr(
            entry_task_module,
            "write_output",
            wrapped_write(lambda _meta: time.sleep(_SLOW_WRITE_SECONDS)),
        )
    execution_id = f"TESTAPPLYNOWRITE{workers}{mode.upper()}"

    _ = run_feature(
        ds,
        _WritesSlowly() if in_processes else HoldsUntilReleased(),
        parallel_workers=workers,
        parallel_mode=mode,
        execution_id=execution_id,
    )

    ends = logged_events(ds.base_dir, execution_id, "entry_end")
    assert [end["mode"] for end in ends] == [mode] * len(_SEQUENCES)
    for end in ends:
        apply_seconds = _number(end, "apply_seconds")
        assert _number(end, "wall_seconds") - apply_seconds >= _SLOW_WRITE_SECONDS
        # The first entry of a process worker also holds the worker's start and
        # the feature's build, so only the other modes bound the apply.
        if not in_processes:
            assert apply_seconds < 0.5


# --- the fit ------------------------------------------------------------------


def test_a_fit_records_its_cost_and_what_it_was_offered(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)

    _ = run_feature(ds, FittingFeature(FitControl()), execution_id="TESTFIT")

    report = _last_resources(ds, "TESTFIT")
    assert _number(report, "fit_seconds") > 0
    assert _number(report, "fit_peak_rss_bytes") > 0
    assert _number(report, "fit_entries") == len(_SEQUENCES)
    rows = sum(
        _number(end, "input_rows")
        for end in logged_events(ds.base_dir, "TESTFIT", "entry_end")
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
            written = (
                logged_events(ds.base_dir, "TESTFITHELD", "resources")
                if path.exists()
                else []
            )
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


def _task(run_root: Path, source: pd.DataFrame | EntryLoad) -> EntryTask:
    """The task of ``seq_a`` from *source*, writing its output under *run_root*."""
    return EntryTask(meta=build_feature_meta("", "seq_a", run_root), source=source)


def test_a_process_capable_feature_has_one_worker_measured(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)

    _ = run_feature(
        ds, ParallelFeature(), execution_id="TESTWORKERBASE", measure_worker_base=True
    )

    *_, frozen, measured = logged_events(ds.base_dir, "TESTWORKERBASE", "resources")
    assert _number(frozen, "worker_base_rss_bytes") == 0
    assert _number(measured, "worker_base_rss_bytes") > 0
    assert _number(measured, "cpu_seconds") == _number(frozen, "cpu_seconds")
    assert "worker_base_nonreclaimable_bytes" in measured
    bases = {"t", "worker_base_rss_bytes", "worker_base_nonreclaimable_bytes"}
    assert {k: v for k, v in measured.items() if k not in bases} == {
        k: v for k, v in frozen.items() if k not in bases
    }


_BIG_ROWS = 5_000_000
"""The rows of a table whose bytes stand out of a worker's base."""


def _big_table() -> pd.DataFrame:
    import numpy as np

    return pd.DataFrame(
        {
            "X": np.arange(_BIG_ROWS, dtype=np.float64),
            "Y": np.arange(_BIG_ROWS, dtype=np.float64),
            "sequence": pd.Categorical(["seq_a"] * _BIG_ROWS),
        }
    )


def _build_of(feature: ParallelFeature, run_root: Path) -> FeatureBuild:
    """The build a process worker makes *feature* from, its state in *run_root*."""
    return FeatureBuild.of(
        feature,
        state_root=run_root,
        artifact_paths={},
        dependency_lookups={},
        manifest_path=None,
        tracklets=None,
    )


def _spawned_worker_bases(
    run_root: Path, source: pd.DataFrame | EntryLoad
) -> tuple[WorkerBase, WorkerMeasure | None]:
    """The base of a spawned worker handed nothing, and its measure of *source*'s task."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    build = _build_of(ParallelFeature(), run_root)

    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=mp.get_context("spawn"),
        initializer=run_module.start_feature_worker,
        initargs=(build,),
    ) as pool:
        empty = pool.submit(run_module.process_worker_base, build).result()
        measure = report_of(
            pool.submit(
                run_module.process_entry_worker, _task(run_root, source)
            ).result()
        ).worker
    return empty, measure


def test_a_process_workers_base_leaves_out_the_table_it_is_handed(
    tmp_path: Path,
) -> None:
    """The base of a worker handed a table is the base of one handed none.

    A process worker receives its entry's table as an argument, so the table is
    resident before ``apply``. Counted in the base, it would be read as memory
    every worker holds, and the probe's figure, measured without a table, would
    mean something else.
    """
    table = _big_table()
    table_bytes = int(table.memory_usage(deep=True).sum())

    empty, measure = _spawned_worker_bases(tmp_path, table)

    assert measure is not None
    assert empty.rss_bytes > 0
    assert abs(measure.base_rss_bytes - empty.rss_bytes) < table_bytes // 4


def test_a_process_worker_that_loads_its_input_reads_its_base_before_the_load(
    tmp_path: Path,
) -> None:
    """A worker given a load holds no table before it, and its base is its own."""
    table = _big_table()
    table_bytes = int(table.memory_usage(deep=True).sum())
    tracks = tmp_path / "tracks.parquet"
    table.to_parquet(tracks)
    del table
    load = EntryLoad(
        entry_key=make_entry_key("", "seq_a"),
        file_specs=((tracks, ParquetLoadSpec()),),
        projection=None,
        expands_individuals=False,
        frame_start=None,
        frame_end=None,
        time_window=TimeWindow(),
        footers=(),
    )

    empty, measure = _spawned_worker_bases(tmp_path, load)

    assert measure is not None
    assert empty.rss_bytes > 0
    assert abs(measure.base_rss_bytes - empty.rss_bytes) < table_bytes // 4


def test_a_feature_that_needs_its_scope_has_no_worker_measured(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path)

    _ = run_feature(
        ds, _NeedsScope(), execution_id="TESTWORKERSCOPED", measure_worker_base=True
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
    once it has applied it, so the run's submissions can wait for it.
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
import dataclasses
import sys
import time
from pathlib import Path

from mosaic.core.pipeline import memory_watch, run
from mosaic.core.pipeline.admission import RowCostEstimate
from mosaic.core.pipeline.memory_limit import read_memory_limit
from tests.core.pipeline import test_run_measures as stand_ins


def host_limit():
    # A cgroup's figure counts every process in the cgroup, the suite's other
    # workers included, so the run reads its limit as the host's and sums the
    # memory of its own processes.
    limit = read_memory_limit()
    if limit is None:
        return None
    return dataclasses.replace(limit, source="host", nonreclaimable_bytes=0)


memory_watch.read_memory_limit = host_limit

mode = sys.argv[2]
applied = Path(sys.argv[3]) if len(sys.argv) > 3 else None
real = run.EntryPool.submit


PREVIOUS = {"seq_b": "seq_a", "seq_c": "seq_b"}


def submit(self, tag, fn, *args):
    # Each submission waits for the sequence before it to have been applied, so
    # the pool finds its one worker idle at every submission and never starts a
    # second.
    before = PREVIOUS.get(tag.sequence)
    if applied is not None and before is not None:
        deadline = time.monotonic() + 60
        while not (applied / before).exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)
    real(self, tag, fn, *args)


live_workers = [0]


class Recording(RowCostEstimate):
    def observe(self, measured_bytes, in_flight_rows, live_workers_now):
        live_workers[0] = max(live_workers[0], live_workers_now)
        super().observe(measured_bytes, in_flight_rows, live_workers_now)


run.EntryPool.submit = submit
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
    shrink or exit while the run measures. The run reads its memory limit as the
    host's, so it sums the memory of its own processes: a cgroup's figure also
    counts the suite's other workers, whose memory may shrink while it measures. In
    process mode each submission waits for the entry before it to have been applied,
    so the pool keeps one worker.

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
    ends = logged_events(base_dir, "TESTPERROW", "entry_end")
    rows = max(int(_number(end, "input_rows")) for end in ends)
    most_live = int(printed.splitlines()[-1])
    return logged_events(base_dir, "TESTPERROW", "resources")[-1], rows, most_live


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


_LEVELS = TypeAdapter(list[InFlightLevel])
_COVARIATE_FIELDS = (
    "ids_per_frame",
    "pair_frames",
    "pair_frames_bound",
    "frames",
    "keypoints",
    "fps",
)


def _indexed_covariates(ds: Dataset) -> dict[str, EntryCovariates]:
    """What the tracks index records for each sequence, read as the run reads it."""
    index = read_tracks_index(ds)
    return {
        str(row["sequence"]): entry_covariates(row, fps=None)
        for _, row in index.iterrows()
    }


@pytest.mark.parametrize(
    ("workers", "mode", "recorded"),
    [
        (1, "thread", "thread"),
        (2, "thread", "thread"),
        (1, "process", "thread"),
        (2, "process", "process"),
    ],
    ids=["inline", "thread", "inline-process-request", "process"],
)
def test_each_entry_records_its_covariates_and_mode(
    tmp_path: Path, workers: int, mode: str, recorded: str
) -> None:
    ds = _dataset(tmp_path)
    execution_id = f"TESTCOVARIATES{workers}{mode}"
    expected = _indexed_covariates(ds)
    assert {covariates.frames for covariates in expected.values()} != {None}

    _ = run_feature(
        ds,
        ParallelFeature(),
        parallel_workers=workers,
        parallel_mode=mode,
        execution_id=execution_id,
    )

    ends = logged_events(ds.base_dir, execution_id, "entry_end")
    assert len(ends) == len(_SEQUENCES)
    for end in ends:
        sequence = str(end["key"]).rsplit("__", 1)[-1]
        covariates = expected[sequence]
        assert end["mode"] == recorded
        for name in _COVARIATE_FIELDS:
            value = getattr(covariates, name)
            if value is None or value is False:
                assert name not in end
            else:
                assert end[name] == value
        assert _number(end, "frames") > 0
        assert _number(end, "keypoints") > 0


class _SleepsAndFailsOnSeqB(ParallelFeature):
    """Holds each entry for a while, and raises from ``apply`` on ``seq_b``."""

    name = "prov-parallel-sleeps-fails"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty and str(df["sequence"].iloc[0]) == "seq_b":
            time.sleep(0.2)
            raise RuntimeError("a lost entry")
        time.sleep(0.4)
        return df


@pytest.mark.parametrize("workers", [1, 2], ids=["inline", "thread"])
def test_the_final_resources_levels_match_the_written_entries(
    tmp_path: Path, workers: int
) -> None:
    ds = _dataset(tmp_path)
    execution_id = f"TESTLEVELS{workers}"

    result = run_feature(
        ds,
        _SleepsAndFailsOnSeqB(),
        parallel_workers=workers,
        parallel_mode="thread",
        execution_id=execution_id,
    )

    assert len(result.failed_entries) == 1
    ends = logged_events(ds.base_dir, execution_id, "entry_end")
    assert len(ends) == len(_SEQUENCES) - 1
    levels = _written_levels(ds, execution_id)
    counts = [level.in_flight for level in levels]
    assert counts == sorted(set(counts))
    assert 1 <= min(counts) and max(counts) <= workers + 1
    assert sum(level.rows for level in levels) == pytest.approx(
        sum(_number(end, "input_rows") for end in ends)
    )
    flights = logged_events(ds.base_dir, execution_id, "entry_end")
    spent = sum(level.seconds for level in levels)
    assert spent >= max(_number(end, "wall_seconds") for end in flights)
    assert spent <= sum(_number(end, "wall_seconds") for end in flights)
    assert all(level.cpu_seconds >= 0 for level in levels)


def test_the_final_resources_levels_are_those_an_old_log_rebuilds(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path)

    _ = run_feature(
        ds,
        _Sleeps(),
        parallel_workers=2,
        parallel_mode="thread",
        execution_id="TESTLEVELSREBUILT",
    )

    _assert_the_levels_are_those_its_entries_rebuild(ds, "TESTLEVELSREBUILT")


def _assert_the_levels_are_those_its_entries_rebuild(
    ds: Dataset, execution_id: str
) -> None:
    """The run's levels have the counts and totals its ``entry_end`` events rebuild."""
    ends = logged_events(ds.base_dir, execution_id, "entry_end")
    spans = [
        FlightSpan(
            start=_stamp(end) - _number(end, "wall_seconds"),
            end=_stamp(end),
            rows=int(_number(end, "input_rows")),
        )
        for end in ends
    ]
    rebuilt = sweep_levels(spans, lambda _start, _end, _flights: 0.0)
    written = _written_levels(ds, execution_id)
    assert [level.in_flight for level in written] == [
        level.in_flight for level in rebuilt
    ]
    # The event's time follows the entry's stamp by however long the host took to
    # write it, so only the totals, which that shift leaves alone, are compared.
    assert sum(level.rows for level in written) == pytest.approx(
        sum(level.rows for level in rebuilt)
    )
    assert sum(level.seconds for level in written) == pytest.approx(
        sum(level.seconds for level in rebuilt), abs=0.5
    )


def _written_levels(ds: Dataset, execution_id: str) -> list[InFlightLevel]:
    """The levels the run's last ``resources`` event carries."""
    return _LEVELS.validate_python(
        _last_resources(ds, execution_id)["in_flight_levels"]
    )


def _stamp(record: dict[str, object]) -> float:
    return datetime.datetime.fromisoformat(str(record["t"])).timestamp()


def test_reading_the_covariates_is_in_no_entrys_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    real = run_module.tracks_covariates
    read_returned: list[float] = []
    flight_started: list[float] = []

    def noting(
        ds: Dataset,
        entries: Collection[Entry],
        *,
        tracks_run_id: str | None,
        variants: Collection[str] | None,
    ) -> dict[Entry, EntryCovariates]:
        read = real(ds, entries, tracks_run_id=tracks_run_id, variants=variants)
        read_returned.append(time.monotonic())
        return read

    class _NotingStarts(EntriesInFlight):
        @override
        def enter(
            self,
            key: str,
            rows: int,
            *,
            pair_frames: int | None,
            started: float,
            cpu: float,
            pair_frames_bound: bool = False,
        ) -> None:
            flight_started.append(started)
            super().enter(
                key,
                rows,
                pair_frames=pair_frames,
                started=started,
                cpu=cpu,
                pair_frames_bound=pair_frames_bound,
            )

    monkeypatch.setattr(run_module, "tracks_covariates", noting)
    monkeypatch.setattr(run_module, "EntriesInFlight", _NotingStarts)

    _ = run_feature(ds, ParallelFeature(), execution_id="TESTCOVARIATETIMING")

    assert len(read_returned) == 1
    assert flight_started
    assert min(flight_started) >= read_returned[0]


def test_a_full_cache_hit_reads_no_covariates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path)
    _ = run_feature(ds, ParallelFeature())

    def refused(*_args: object, **_kwargs: object) -> dict[Entry, EntryCovariates]:
        raise AssertionError("a cached run reads no covariates")

    monkeypatch.setattr(run_module, "tracks_covariates", refused)

    _ = run_feature(ds, ParallelFeature(), execution_id="TESTCOVARIATEHIT")


def test_the_resources_carry_the_params_digest_and_emits(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    feature = ParallelFeature()

    _ = run_feature(ds, feature, execution_id="TESTPARAMSFACTS")

    report = _last_resources(ds, "TESTPARAMSFACTS")
    assert report["params_digest"] == params_identity_digest(feature.params)
    assert report["params_default"] is params_equal_defaults(feature.params)
    assert report["params_default"] is True
    assert report["emits"] == "as-input"


# --- the entries in flight keep their largest sample ----------------------------


def _in_flight() -> EntriesInFlight:
    return EntriesInFlight(cpu_seconds=lambda: 0.0)


def test_a_flight_keeps_its_largest_sample() -> None:
    in_flight = _in_flight()
    in_flight.enter("a", 10, pair_frames=None, started=0.0, cpu=0.0)
    in_flight.observe_memory(500)
    in_flight.enter("b", 20, pair_frames=None, started=1.0, cpu=0.0)
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
    in_flight = _in_flight()
    in_flight.enter("a", 10, pair_frames=None, started=0.0, cpu=0.0)
    in_flight.observe_memory(500)
    _ = in_flight.leave("a")
    in_flight.enter("b", 20, pair_frames=None, started=1.0, cpu=0.0)

    span = in_flight.leave("b")

    assert span is not None
    assert span.peak_nonreclaimable_bytes is None


def test_an_entry_that_entered_after_a_sample_began_takes_no_figure_from_it() -> None:
    in_flight = _in_flight()
    in_flight.enter("early", 10, pair_frames=None, started=1.0, cpu=0.0)
    in_flight.enter("late", 20, pair_frames=None, started=3.0, cpu=0.0)

    in_flight.observe_memory(900, sampled_from=2.0)
    early = in_flight.leave("early")
    late = in_flight.leave("late")

    assert early is not None
    assert early.peak_nonreclaimable_bytes == 900
    assert late is not None
    assert late.peak_nonreclaimable_bytes is None


def test_an_entry_that_entered_when_a_sample_began_takes_its_figure() -> None:
    in_flight = _in_flight()
    in_flight.enter("a", 10, pair_frames=None, started=2.0, cpu=0.0)

    in_flight.observe_memory(900, sampled_from=2.0)

    span = in_flight.leave("a")
    assert span is not None
    assert span.peak_nonreclaimable_bytes == 900


def test_a_sample_counts_only_the_entries_that_entered_before_it_began() -> None:
    in_flight = _in_flight()
    in_flight.enter("early", 10, pair_frames=None, started=1.0, cpu=0.0)
    in_flight.enter("late", 20, pair_frames=None, started=3.0, cpu=0.0)

    assert in_flight.totals(sampled_from=2.0) == (1, 10, 0, False)
    assert in_flight.totals(sampled_from=3.0) == (2, 30, 0, False)
    assert in_flight.totals() == (2, 30, 0, False)


def test_the_entries_and_rows_in_flight_are_read_together() -> None:
    in_flight = _in_flight()
    assert in_flight.totals() == (0, 0, 0, False)
    in_flight.enter("a", 10, pair_frames=None, started=0.0, cpu=0.0)
    in_flight.enter("b", 20, pair_frames=None, started=0.0, cpu=0.0)

    assert in_flight.totals() == (2, 30, 0, False)

    _ = in_flight.leave("a")

    assert in_flight.totals() == (1, 20, 0, False)


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
        pair_frames: int,
        pair_frames_bound: bool,
        live_workers: int,
    ) -> PeakSample | None:
        peaks.append(bytes_)
        found = real_peak(
            peak,
            bytes_=bytes_,
            entries=entries,
            rows=rows,
            pair_frames=pair_frames,
            pair_frames_bound=pair_frames_bound,
            live_workers=live_workers,
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
    base: int | None, peak: int | None, task: EntryTask
) -> EntryReport | EntryFailure:
    """Apply as a process worker does, and report the given non-reclaimable pair."""
    outcome = run_module.process_entry_worker(task)
    measure = report_of(outcome).worker
    assert measure is not None
    measure = dataclasses.replace(
        measure, base_nonreclaimable_bytes=base, peak_nonreclaimable_bytes=peak
    )
    return dataclasses.replace(outcome, worker=measure)


def _grown_ends(ds: Dataset, execution_id: str) -> list[tuple[int, object]]:
    return [
        (int(_number(end, "in_flight")), end.get("peak_grown_bytes"))
        for end in logged_events(ds.base_dir, execution_id, "entry_end")
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


class _WithPairFrames(EntriesInFlight):
    """Gives every entry 7 pair frames, the bound among them, as it enters."""

    @override
    def enter(
        self,
        key: str,
        rows: int,
        *,
        pair_frames: int | None,
        started: float,
        cpu: float,
        pair_frames_bound: bool = False,
    ) -> None:
        super().enter(
            key,
            rows,
            pair_frames=7,
            pair_frames_bound=True,
            started=started,
            cpu=cpu,
        )


def test_the_peak_sample_sums_the_pair_frames_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spied: dict[str, list[int]]
) -> None:
    _ = spied
    ds = _dataset(tmp_path)
    monkeypatch.setattr(run_module, "EntriesInFlight", _WithPairFrames)
    _figure_after_the_base(monkeypatch, lambda: _BASE, lambda: _BASE + _GIB)

    _run_gated(ds, "TESTPEAKPAIRS")

    assert _Spy.peak_samples
    for sample in _Spy.peak_samples:
        assert sample.pair_frames == 7 * sample.entries
        assert sample.pair_frames_bound


def test_the_resources_carry_the_pair_frames_in_flight_at_the_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spied: dict[str, list[int]]
) -> None:
    _ = spied
    ds = _dataset(tmp_path)
    covariates = _indexed_covariates(ds)
    assert len({(c.pair_frames, c.pair_frames_bound) for c in covariates.values()}) == 1
    one = covariates["seq_a"]
    assert one.pair_frames is not None
    _figure_after_the_base(monkeypatch, lambda: _BASE, lambda: _BASE + _GIB)

    _run_gated(ds, "TESTPEAKPAIRFRAMES")

    peak = _Spy.peak_samples[-1]
    assert peak.pair_frames == peak.entries * one.pair_frames
    assert peak.pair_frames_bound is one.pair_frames_bound
    report = _last_resources(ds, "TESTPEAKPAIRFRAMES")
    assert report["peak_in_flight_pair_frames"] == peak.pair_frames
    assert report["peak_in_flight_pair_frames_bound"] is peak.pair_frames_bound


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
        "process_entry_worker",
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
        "process_entry_worker",
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
    return run_module.process_worker_base(_build_of(ParallelFeature(), tmp_path))


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
    monkeypatch.setattr(
        entry_task_module, "process_nonreclaimable_bytes", lambda: reader
    )

    @contextmanager
    def peak_while() -> Generator[memory_watch.NonreclaimablePeak]:
        _SAMPLING["active"] = True
        try:
            yield memory_watch.NonreclaimablePeak(peak=reader + 4096)
        finally:
            _SAMPLING["active"] = False

    monkeypatch.setattr(entry_task_module, "peak_nonreclaimable_while", peak_while)
    _SAMPLING["during_apply"] = False
    feature = _SeesTheSampler()
    table = pd.DataFrame({"sequence": ["seq_a"] * 100, "value": range(100)})
    handed = int(table.memory_usage(deep=True).sum())

    measure = report_of(
        FeatureWorker(_build_of(feature, tmp_path)).run(_task(tmp_path, table))
    ).worker

    assert _SAMPLING["during_apply"]
    assert isinstance(measure, WorkerMeasure)
    assert measure.base_nonreclaimable_bytes == reader - handed
    assert measure.peak_nonreclaimable_bytes == reader + 4096


def test_a_process_worker_without_the_figure_leaves_both_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(entry_task_module, "process_nonreclaimable_bytes", lambda: None)

    @contextmanager
    def peak_while() -> Generator[memory_watch.NonreclaimablePeak]:
        yield memory_watch.NonreclaimablePeak()

    monkeypatch.setattr(entry_task_module, "peak_nonreclaimable_while", peak_while)
    feature = ParallelFeature()
    table = pd.DataFrame({"sequence": ["seq_a"] * 10})

    measure = report_of(
        FeatureWorker(_build_of(feature, tmp_path)).run(_task(tmp_path, table))
    ).worker

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


# --- the build of a process worker ---------------------------------------------------


@pytest.mark.parametrize("mode", ["thread", "process"])
def test_the_entry_whose_worker_built_the_feature_records_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Each build in a process worker gives one entry its seconds, and threads none."""
    ds = _dataset(tmp_path)
    marks = tmp_path / "marks"
    marks.mkdir()
    monkeypatch.setenv(HOLD_DIRECTORY_VARIABLE, str(marks))
    execution_id = f"TESTBUILDSECONDS{mode.upper()}"

    _ = run_feature(
        ds,
        CountsItsBuilds(),
        parallel_workers=2,
        parallel_mode=mode,
        execution_id=execution_id,
    )

    ends = logged_events(ds.base_dir, execution_id, "entry_end")
    assert len(ends) == len(_SEQUENCES)
    built = [end for end in ends if "build_seconds" in end]
    builds = marks / "builds"
    expected = len(builds.read_text().split()) if builds.exists() else 0
    assert len(built) == expected
    assert (expected > 0) == (mode == "process")
    for end in built:
        assert 0.0 < _number(end, "build_seconds") <= _number(end, "apply_seconds")


# --- a fitted state built outside an entry's growth -----------------------------------


_STATE_BYTES = 256 * 1024**2
"""The bytes of the state :class:`_HoldsALargeState` loads."""

_STATE_SEQUENCES = ("seq_a", "seq_b", "seq_c", "seq_d")

_BUILD_SECONDS = 0.75
"""How long the build of :class:`_HoldsALargeState` takes at least.

Longer than the tolerance on the seconds of rebuilt levels, so levels that leave
out the build differ from those that contain it on any host.
"""


class _HoldsALargeState(_Sleeps):
    """A fitted feature whose loaded state holds 256 MiB.

    Its fit saves a small file, and once that file exists ``load_state`` takes at
    least ``_BUILD_SECONDS`` and keeps an array of 256 MiB on the instance. The array
    is filled with ones, so every page of it is resident, which the pages of an array
    of zeros are not.
    """

    name = "prov-parallel-large-state"
    state_file = "large-state.json"
    state: npt.NDArray[np.uint8] | None = None

    @override
    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        if not (run_root / self.state_file).is_file():
            return False
        time.sleep(_BUILD_SECONDS)
        self.state = np.ones(_STATE_BYTES, dtype=np.uint8)
        return True

    @override
    def fit(self, inputs: InputStream) -> None:
        pass

    @override
    def save_state(self, run_root: Path) -> None:
        (run_root / self.state_file).write_text("{}")


@pytest.fixture(scope="module")
def large_state_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Dataset, str]:
    """A dataset and the execution id of a process run of :class:`_HoldsALargeState`."""
    ds = make_dataset(tmp_path_factory.mktemp("large-state") / "ds")
    add_tracks_variant(ds, VARIANT, *_STATE_SEQUENCES, std_format="mosaic_v1")
    execution_id = "TESTLARGESTATE"
    _ = run_feature(
        ds,
        _HoldsALargeState(),
        parallel_workers=2,
        parallel_mode="process",
        execution_id=execution_id,
    )
    return ds, execution_id


@pytest.mark.skipif(
    memory_watch.process_nonreclaimable_bytes() is None,
    reason="this host does not give a process's anonymous and shared memory",
)
def test_a_workers_fitted_state_is_in_its_base_and_not_in_an_entrys_growth(
    large_state_run: tuple[Dataset, str],
) -> None:
    """The first entry of each worker grows by its own memory, not by the state built.

    A worker builds the feature at its first task, and reads its base after the
    build, so the loaded state is memory the worker holds for every entry.
    """
    ds, execution_id = large_state_run

    ends = logged_events(ds.base_dir, execution_id, "entry_end")

    assert len(ends) == len(_STATE_SEQUENCES)
    built = [end for end in ends if "build_seconds" in end]
    assert built
    assert all(end.get("peak_grown_bytes") is not None for end in built)
    grown = [
        _number(end, "peak_grown_bytes")
        for end in ends
        if end.get("peak_grown_bytes") is not None
    ]
    assert all(bytes_ < _STATE_BYTES // 2 for bytes_ in grown), grown


def test_a_run_whose_workers_build_a_large_state_has_the_levels_its_entries_rebuild(
    large_state_run: tuple[Dataset, str],
) -> None:
    _assert_the_levels_are_those_its_entries_rebuild(*large_state_run)
