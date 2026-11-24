"""One entry's work: check, apply, write its output, and report what it wrote.

Every mode and worker count runs an entry through ``run_entry``, and a process
worker through ``run_entry_in_process``, which adds the worker's own memory. The
task writes the entry's output and nothing else, and once the output is renamed
into place nothing it does may raise.
"""

from __future__ import annotations

import pickle
import time
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import ClassVar, NoReturn

import pandas as pd
import pyarrow as pa
import pytest

from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline import entry_task
from mosaic.core.pipeline._utils import FeatureMeta
from mosaic.core.pipeline.admission import WorkerMeasure
from mosaic.core.pipeline.arena_guard import ArenaBlindFeature
from mosaic.core.pipeline.column_contract import (
    Column,
    MissingInputColumns,
    Requirement,
)
from mosaic.core.pipeline.entry_pool import EntryFailure
from mosaic.core.pipeline.entry_task import (
    EntryReport,
    EntryTask,
    run_entry,
    run_entry_in_process,
)
from mosaic.core.pipeline.manifest import CoreSelector
from mosaic.core.pipeline.memory_watch import NonreclaimablePeak
from mosaic.core.pipeline.writers import FeatureOutput, write_output
from mosaic.core.timebase import IrregularNotice, uniform_grid_rate
from tests.helpers import ParallelFeature, dropped_frame_tracks


class _NotesUnevenTime(ParallelFeature):
    """Returns its input after reading its time axis as evenly spaced."""

    name = "test-notes-uneven-time"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        _ = uniform_grid_rate(df, self.name)
        return df


class _Raises(ParallelFeature):
    name = "test-raises"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        msg = "apply failed"
        raise RuntimeError(msg)


class _NeedsAColumn(ParallelFeature):
    name = "test-needs-a-column"
    requires: ClassVar[tuple[Requirement, ...]] = (Column("needed"),)


class _Unpicklable(Exception):
    def __init__(self, first: str, second: str) -> None:
        super().__init__(first)
        self.second: str = second


def _task(
    run_root: Path,
    table: pd.DataFrame | None = None,
    *,
    selector: CoreSelector | None = None,
    arena_count: int | None = None,
) -> EntryTask:
    meta = FeatureMeta(group="g", sequence="s", out_path=run_root / "g__s.parquet")
    return EntryTask(
        meta=meta,
        table=dropped_frame_tracks() if table is None else table,
        selector=selector,
        arena_count=arena_count,
    )


def _frames(count: int) -> pd.DataFrame:
    return pd.DataFrame({"frame": range(count), "value": range(count), "sequence": "s"})


def _report(outcome: EntryReport | EntryFailure) -> EntryReport:
    assert isinstance(outcome, EntryReport), outcome
    return outcome


def _failure(outcome: EntryReport | EntryFailure) -> EntryFailure:
    assert isinstance(outcome, EntryFailure), outcome
    return outcome


def _raise_unpicklable(_meta: FeatureMeta, _output: FeatureOutput, **_: object) -> int:
    raise _Unpicklable("one", "two")


def _raise_os_error(_meta: FeatureMeta, _output: FeatureOutput, **_: object) -> int:
    msg = "No space left on device"
    raise OSError(28, msg)


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    return root


# --- the output and the report ---------------------------------------------------------


def test_a_task_writes_the_bytes_the_writer_writes_and_reports_them(
    run_root: Path, tmp_path: Path
) -> None:
    task = _task(run_root)
    expected = FeatureMeta(group="g", sequence="s", out_path=tmp_path / "x.parquet")
    _ = write_output(expected, task.table)

    report = _report(run_entry(_NotesUnevenTime(), task, None))

    assert task.meta.out_path.read_bytes() == expected.out_path.read_bytes()
    assert report.rows == len(task.table)
    assert report.output_bytes == int(task.table.memory_usage(index=True).sum())
    [notice] = report.notices
    assert notice.who == _NotesUnevenTime.name
    assert report.worker is None


def test_a_task_under_overlap_writes_the_rows_its_selector_keeps(
    run_root: Path,
) -> None:
    table = _frames(10)
    selector = CoreSelector(order_col="frame", first=3, last=6, entry_key="g__s")

    report = _report(
        run_entry(ParallelFeature(), _task(run_root, table, selector=selector), None)
    )

    written = pd.read_parquet(run_root / "g__s.parquet")
    assert list(written["frame"]) == [3, 4, 5, 6]
    assert report.rows == 4


def test_a_task_of_no_output_writes_an_empty_table_of_no_bytes(
    run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_output(
        _apply: Callable[[pd.DataFrame], FeatureOutput], _df: pd.DataFrame
    ) -> tuple[FeatureOutput, tuple[IrregularNotice, ...]]:
        return None, ()

    monkeypatch.setattr(entry_task, "apply_noting_irregular_time", no_output)

    report = _report(run_entry(ParallelFeature(), _task(run_root), None))

    assert (report.rows, report.output_bytes) == (0, 0)
    assert pd.read_parquet(run_root / "g__s.parquet").empty


def test_after_apply_runs_once_between_apply_and_the_write(run_root: Path) -> None:
    applied: list[bool] = []
    calls: list[bool] = []

    class _Marks(ParallelFeature):
        def apply(self, df: pd.DataFrame) -> pd.DataFrame:
            applied.append(True)
            return df

    task = _task(run_root)

    def after_apply() -> None:
        calls.append(bool(applied) and not task.meta.out_path.exists())

    _ = _report(run_entry(_Marks(), task, after_apply))

    assert calls == [True]
    assert task.meta.out_path.exists()


@pytest.mark.parametrize("runner", ["serial", "process"])
def test_the_seconds_after_apply_cover_a_slow_write(
    run_root: Path, monkeypatch: pytest.MonkeyPatch, runner: str
) -> None:
    written = entry_task.write_output

    def slow_write(meta: FeatureMeta, output: FeatureOutput, **kwargs: bool) -> int:
        time.sleep(0.3)
        return written(meta, output, **kwargs)

    monkeypatch.setattr(entry_task, "write_output", slow_write)
    task = _task(run_root)

    outcome = (
        run_entry(ParallelFeature(), task, None)
        if runner == "serial"
        else run_entry_in_process(ParallelFeature(), task)
    )

    assert _report(outcome).seconds_after_apply >= 0.3


# --- entry errors raise ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("feature", "arena_count", "raised"),
    [
        pytest.param(_NeedsAColumn(), None, MissingInputColumns, id="column-check"),
        pytest.param(ParallelFeature(), 2, ArenaBlindFeature, id="arena-refusal"),
        pytest.param(_Raises(), None, RuntimeError, id="apply"),
    ],
)
def test_an_entry_error_raises_and_writes_nothing(
    run_root: Path,
    feature: ParallelFeature,
    arena_count: int | None,
    raised: type[Exception],
) -> None:
    with pytest.raises(raised):
        _ = run_entry(feature, _task(run_root, arena_count=arena_count), None)

    assert list(run_root.iterdir()) == []


def test_an_arena_count_of_one_lets_the_entry_through(run_root: Path) -> None:
    _ = _report(run_entry(ParallelFeature(), _task(run_root, arena_count=1), None))


def test_a_notice_that_does_not_pickle_raises_before_the_write(
    run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unpicklable_notice(
        apply: Callable[[pd.DataFrame], FeatureOutput], df: pd.DataFrame
    ) -> tuple[FeatureOutput, tuple[object, ...]]:
        return apply(df), (lambda: None,)

    monkeypatch.setattr(entry_task, "apply_noting_irregular_time", unpicklable_notice)

    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        _ = run_entry(ParallelFeature(), _task(run_root), None)

    assert list(run_root.iterdir()) == []


# --- output failures return --------------------------------------------------------------


def test_a_trim_error_returns_a_failure_and_writes_nothing(run_root: Path) -> None:
    table = _frames(5)
    selector = CoreSelector(order_col="frame", first=50, last=60, entry_key="g__s")

    failure = _failure(
        run_entry(ParallelFeature(), _task(run_root, table, selector=selector), None)
    )

    assert failure.key == make_entry_key("g", "s")
    assert isinstance(failure.error, ValueError)
    assert list(run_root.iterdir()) == []


def test_a_write_error_returns_a_failure_and_writes_nothing(
    run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(entry_task, "write_output", _raise_os_error)

    failure = _failure(run_entry(ParallelFeature(), _task(run_root), None))

    assert failure.key == make_entry_key("g", "s")
    assert isinstance(failure.error, OSError)
    assert list(run_root.iterdir()) == []


def test_a_write_error_that_does_not_pickle_is_replaced_only_in_a_process(
    run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(entry_task, "write_output", _raise_unpicklable)

    kept = _failure(run_entry(ParallelFeature(), _task(run_root), None))
    replaced = _failure(run_entry_in_process(ParallelFeature(), _task(run_root)))

    assert isinstance(kept.error, _Unpicklable)
    assert type(replaced.error) is RuntimeError
    assert str(replaced.error) == "_Unpicklable: one"
    assert list(run_root.iterdir()) == []


def test_a_write_into_a_missing_run_root_fails_and_creates_nothing(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "removed"

    failure = _failure(run_entry(ParallelFeature(), _task(run_root), None))

    assert isinstance(failure.error, FileNotFoundError)
    assert not run_root.exists()


# --- after the rename ------------------------------------------------------------------


class _RefusesToRelease:
    def release_unused(self) -> NoReturn:
        msg = "release failed"
        raise RuntimeError(msg)


def test_a_failed_release_after_the_rename_keeps_the_report_and_the_output(
    run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pa, "default_memory_pool", _RefusesToRelease)
    task = _task(run_root)

    report = _report(run_entry(ParallelFeature(), task, None))

    assert report.rows == len(task.table)
    assert task.meta.out_path.exists()


# --- a process worker's memory ---------------------------------------------------------


def _sampler(
    seen: list[bool],
    out_path: Path,
    figure: int | None,
    *,
    exit_raises: bool = False,
    exit_seconds: float = 0.0,
) -> Callable[[], AbstractContextManager[NonreclaimablePeak]]:
    @contextmanager
    def sampler() -> Generator[NonreclaimablePeak]:
        holder = NonreclaimablePeak()
        yield holder
        time.sleep(exit_seconds)
        seen.append(out_path.exists())
        holder.record(figure)
        if exit_raises:
            msg = "the sampler failed"
            raise RuntimeError(msg)

    return sampler


def test_a_process_worker_measures_its_base_less_the_table_and_its_peak_after_the_write(
    run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task(run_root)
    handed = int(task.table.memory_usage(deep=True).sum())
    seen: list[bool] = []
    monkeypatch.setattr(
        entry_task, "process_nonreclaimable_bytes", lambda: handed + 5_000
    )
    monkeypatch.setattr(
        entry_task,
        "peak_nonreclaimable_while",
        _sampler(seen, task.meta.out_path, 9_000),
    )

    worker = _report(run_entry_in_process(ParallelFeature(), task)).worker

    assert worker is not None
    assert worker.base_nonreclaimable_bytes == 5_000
    assert worker.peak_nonreclaimable_bytes == 9_000
    assert seen == [True]
    assert worker.peak_rss_bytes >= worker.peak_before_rss_bytes


def test_a_process_worker_without_the_figure_measures_neither(
    run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task(run_root)
    monkeypatch.setattr(entry_task, "process_nonreclaimable_bytes", lambda: None)
    monkeypatch.setattr(
        entry_task, "peak_nonreclaimable_while", _sampler([], task.meta.out_path, None)
    )

    worker = _report(run_entry_in_process(ParallelFeature(), task)).worker

    assert worker is not None
    assert worker.base_nonreclaimable_bytes is None
    assert worker.peak_nonreclaimable_bytes is None


def test_a_process_workers_seconds_after_apply_run_to_its_return(
    run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sampler's exit runs after :func:`run_entry` returned, and its time counts."""
    task = _task(run_root)
    monkeypatch.setattr(
        entry_task,
        "peak_nonreclaimable_while",
        _sampler([], task.meta.out_path, 1, exit_seconds=0.3),
    )

    report = _report(run_entry_in_process(ParallelFeature(), task))

    assert report.seconds_after_apply >= 0.3


def _measure_refused(**_: object) -> NoReturn:
    msg = "the measure failed"
    raise RuntimeError(msg)


@pytest.mark.parametrize("fault", ["measure", "sampler-exit"])
def test_a_failed_measure_after_the_write_keeps_the_report_without_a_worker(
    run_root: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    task = _task(run_root)
    if fault == "measure":
        monkeypatch.setattr(entry_task, "WorkerMeasure", _measure_refused)
    else:
        monkeypatch.setattr(
            entry_task,
            "peak_nonreclaimable_while",
            _sampler([], task.meta.out_path, 1, exit_raises=True),
        )

    report = _report(run_entry_in_process(ParallelFeature(), task))

    assert report.worker is None
    assert report.rows == len(task.table)
    assert task.meta.out_path.exists()


# --- values that cross the process boundary --------------------------------------------


def _crossed[T](value: T) -> T:
    crossed: T = pickle.loads(pickle.dumps(value))
    return crossed


def test_a_task_survives_pickle(run_root: Path) -> None:
    selector = CoreSelector(order_col="frame", first=1, last=4, entry_key="g__s")
    task = _task(run_root, selector=selector, arena_count=1)

    crossed = _crossed(task)

    assert crossed.meta == task.meta
    pd.testing.assert_frame_equal(crossed.table, task.table)
    assert crossed.selector == task.selector
    assert crossed.arena_count == task.arena_count
    assert crossed.key == task.key == make_entry_key("g", "s")


def test_a_report_survives_pickle(run_root: Path) -> None:
    report = _report(run_entry_in_process(_NotesUnevenTime(), _task(run_root)))
    assert report.notices
    assert isinstance(report.worker, WorkerMeasure)

    assert _crossed(report) == report


def _failures(
    run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[EntryFailure]:
    with monkeypatch.context() as patched:
        patched.setattr(entry_task, "write_output", _raise_os_error)
        yield _failure(run_entry(ParallelFeature(), _task(run_root), None))
        patched.setattr(entry_task, "write_output", _raise_unpicklable)
        yield _failure(run_entry_in_process(ParallelFeature(), _task(run_root)))
    table = _frames(5)
    selector = CoreSelector(order_col="frame", first=50, last=60, entry_key="g__s")
    yield _failure(
        run_entry(ParallelFeature(), _task(run_root, table, selector=selector), None)
    )
    yield _failure(run_entry(ParallelFeature(), _task(run_root / "removed"), None))


def test_a_failure_of_each_kind_survives_pickle(
    run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failures = list(_failures(run_root, monkeypatch))

    assert [type(failure.error) for failure in failures] == [
        OSError,
        RuntimeError,
        ValueError,
        FileNotFoundError,
    ]
    for failure in failures:
        crossed = _crossed(failure)
        assert crossed.key == failure.key
        assert type(crossed.error) is type(failure.error)
        assert crossed.error.args == failure.error.args
        assert crossed.trace == failure.trace
