"""One entry's work: check, apply, write its output, and report what it wrote.

Every mode and worker count runs an entry through ``run_entry``, and a process
worker through ``run_entry_in_process``, which adds the worker's own memory. The
task writes the entry's output and nothing else, and once the output is renamed
into place nothing it does may raise.
"""

from __future__ import annotations

import errno
import pickle
import shutil
import time
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import ClassVar, NoReturn

import pandas as pd
import pyarrow as pa
import pytest

from mosaic.core.filter_bounds import TimeWindow
from mosaic.core.helpers import filter_time_range, make_entry_key
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline import entry_pool, entry_task, parquet_reader
from mosaic.core.pipeline._loaders import LoadSpec, ParquetLoadSpec
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
    EntryLoad,
    EntryReport,
    EntryTask,
    PairFilter,
    run_entry,
    run_entry_in_process,
)
from mosaic.core.pipeline.loading import (
    ColumnProjection,
    MultiInputAlignmentError,
    PairRowsMissing,
    load_entry_data,
    nn_pair_filter,
)
from mosaic.core.pipeline.manifest import CoreSelector
from mosaic.core.pipeline.memory_watch import NonreclaimablePeak
from mosaic.core.pipeline.parquet_reader import FooterRecord, ParquetFooterMemo
from mosaic.core.pipeline.types import NNResult, Result
from mosaic.core.pipeline.writers import FeatureOutput, write_output
from mosaic.core.timebase import IrregularNotice, uniform_grid_rate
from tests.helpers import (
    ParallelFeature,
    TracksFeature,
    corrupt_output,
    dropped_frame_tracks,
    make_pair_df,
    make_sequence_df,
    neighbor_pairs,
    observed,
    ordered_pairs,
    pair_run_results,
    record_path_calls,
    report_of,
    result_run_root,
    wrapped_write,
)


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
        source=dropped_frame_tracks() if table is None else table,
        selector=selector,
        arena_count=arena_count,
    )


def _table_of(task: EntryTask) -> pd.DataFrame:
    assert isinstance(task.source, pd.DataFrame), task.source
    return task.source


def _inputs(directory: Path, table: pd.DataFrame | None = None) -> tuple[Path, Path]:
    """Two tracks files under *directory* that merge back into *table*'s rows.

    The first holds every column but ``ANGLE``, the second the frame, the
    individual and ``ANGLE``.
    """
    whole = dropped_frame_tracks(sequence="s") if table is None else table
    directory.mkdir(exist_ok=True)
    first, second = directory / "first.parquet", directory / "second.parquet"
    whole.drop(columns=["ANGLE"]).to_parquet(first)
    whole[["frame", "id", "ANGLE"]].to_parquet(second)
    return first, second


def _specs(paths: tuple[Path, ...]) -> tuple[tuple[Path, LoadSpec], ...]:
    return tuple((path, ParquetLoadSpec()) for path in paths)


def _load_task(
    run_root: Path,
    paths: tuple[Path, ...],
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
    time_window: TimeWindow = TimeWindow(),
    projection: ColumnProjection | None = None,
    expands_individuals: bool = False,
    footers: tuple[FooterRecord, ...] = (),
    pair_filter: PairFilter | None = None,
) -> EntryTask:
    meta = FeatureMeta(group="g", sequence="s", out_path=run_root / "g__s.parquet")
    load = EntryLoad(
        entry_key=make_entry_key("g", "s"),
        file_specs=_specs(paths),
        projection=projection,
        expands_individuals=expands_individuals,
        pair_filter=pair_filter,
        frame_start=frame_start,
        frame_end=frame_end,
        time_window=time_window,
        footers=footers,
    )
    return EntryTask(meta=meta, source=load)


def _frames(count: int) -> pd.DataFrame:
    return pd.DataFrame({"frame": range(count), "value": range(count), "sequence": "s"})


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
    _ = write_output(expected, _table_of(task))

    report = report_of(run_entry(_NotesUnevenTime(), task, None))

    assert task.meta.out_path.read_bytes() == expected.out_path.read_bytes()
    assert report.rows == len(_table_of(task))
    assert report.output_bytes == int(_table_of(task).memory_usage(index=True).sum())
    [notice] = report.notices
    assert notice.who == _NotesUnevenTime.name
    assert report.worker is None


def test_a_task_under_overlap_writes_the_rows_its_selector_keeps(
    run_root: Path,
) -> None:
    table = _frames(10)
    selector = CoreSelector(order_col="frame", first=3, last=6, entry_key="g__s")

    report = report_of(
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

    report = report_of(run_entry(ParallelFeature(), _task(run_root), None))

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

    _ = report_of(run_entry(_Marks(), task, after_apply))

    assert calls == [True]
    assert task.meta.out_path.exists()


@pytest.mark.parametrize("runner", ["serial", "process"])
def test_the_seconds_after_apply_cover_a_slow_write(
    run_root: Path, monkeypatch: pytest.MonkeyPatch, runner: str
) -> None:
    monkeypatch.setattr(
        entry_task, "write_output", wrapped_write(lambda _meta: time.sleep(0.3))
    )
    task = _task(run_root)

    outcome = (
        run_entry(ParallelFeature(), task, None)
        if runner == "serial"
        else run_entry_in_process(ParallelFeature(), task)
    )

    assert report_of(outcome).seconds_after_apply >= 0.3


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
    _ = report_of(run_entry(ParallelFeature(), _task(run_root, arena_count=1), None))


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

    report = report_of(run_entry(ParallelFeature(), task, None))

    assert report.rows == len(_table_of(task))
    assert task.meta.out_path.exists()


# --- a task that loads its input ------------------------------------------------------


def test_a_load_task_writes_what_a_table_task_holding_its_loaded_table_writes(
    run_root: Path, tmp_path: Path
) -> None:
    paths = _inputs(tmp_path / "tracks")
    loaded = load_entry_data(list(_specs(paths)))
    assert loaded is not None
    table_root = tmp_path / "table"
    table_root.mkdir()

    from_load = report_of(
        run_entry(ParallelFeature(), _load_task(run_root, paths), None)
    )
    from_table = report_of(
        run_entry(ParallelFeature(), _task(table_root, loaded), None)
    )

    out = "g__s.parquet"
    assert (run_root / out).read_bytes() == (table_root / out).read_bytes()
    assert from_load.input_bytes == int(loaded.memory_usage(index=True).sum()) > 0
    assert from_table.input_bytes == 0
    assert from_load.written and from_table.written
    assert from_load.rows == from_table.rows == len(loaded)


def test_frame_bounds_and_a_time_window_filter_the_rows_as_the_run_filters_them(
    run_root: Path, tmp_path: Path
) -> None:
    paths = _inputs(tmp_path / "tracks")
    loaded = load_entry_data(list(_specs(paths)))
    assert loaded is not None
    window = TimeWindow(0.2, 1.0)
    expected = filter_time_range(
        filter_time_range(loaded, filter_start_frame=3, filter_end_frame=20),
        filter_start_time=window.start,
        filter_end_time=window.end,
    )
    assert 0 < len(expected) < len(loaded)

    task = _load_task(run_root, paths, frame_start=3, frame_end=20, time_window=window)
    report = report_of(run_entry(ParallelFeature(), task, None))

    pd.testing.assert_frame_equal(pd.read_parquet(task.meta.out_path), expected)
    assert report.rows == len(expected)


def test_a_time_window_over_rows_without_time_returns_the_runs_error(
    run_root: Path, tmp_path: Path
) -> None:
    paths = _inputs(
        tmp_path / "tracks", dropped_frame_tracks(sequence="s").drop(columns=["time"])
    )

    failure = _failure(
        run_entry(
            ParallelFeature(),
            _load_task(run_root, paths, time_window=TimeWindow(0.0, 1.0)),
            None,
        )
    )

    assert failure.key == make_entry_key("g", "s")
    assert isinstance(failure.error, ValueError)
    assert str(failure.error) == (
        "a time filter reads the 'time' column, and the rows of 'g__s' have none. "
        "Filter by frame with filter_start_frame and filter_end_frame"
    )
    assert list(run_root.iterdir()) == []


def _raise_eio(*_: object, **__: object) -> NoReturn:
    raise OSError(errno.EIO, "input/output error")


@pytest.mark.parametrize("fault", ["corrupt-input", "os-error"])
def test_an_input_failure_returns_a_failure_and_writes_nothing(
    run_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    paths = _inputs(tmp_path / "tracks")
    if fault == "corrupt-input":
        _ = paths[1].write_bytes(b"not a parquet file")
    else:
        monkeypatch.setattr(entry_task, "load_entry_data", _raise_eio)

    failure = _failure(run_entry(ParallelFeature(), _load_task(run_root, paths), None))

    assert failure.key == make_entry_key("g", "s")
    if fault == "os-error":
        assert isinstance(failure.error, OSError)
        assert failure.error.errno == errno.EIO
    assert list(run_root.iterdir()) == []


def test_missing_pair_rows_raise_as_the_entrys_error(
    run_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def pairs_missing(*_: object, **__: object) -> NoReturn:
        msg = "input 0 holds no pair rows"
        raise PairRowsMissing(msg)

    monkeypatch.setattr(entry_task, "load_entry_data", pairs_missing)

    with pytest.raises(PairRowsMissing):
        _ = run_entry(
            ParallelFeature(), _load_task(run_root, _inputs(tmp_path / "t")), None
        )

    assert list(run_root.iterdir()) == []


def test_a_load_of_no_rows_reports_nothing_written_and_skips_after_apply(
    run_root: Path, tmp_path: Path
) -> None:
    calls: list[bool] = []
    task = _load_task(run_root, _inputs(tmp_path / "tracks"), frame_start=10**6)

    report = report_of(run_entry(ParallelFeature(), task, lambda: calls.append(True)))

    assert report == EntryReport(
        written=False,
        rows=0,
        output_bytes=0,
        input_bytes=0,
        notices=(),
        seconds_after_apply=0.0,
    )
    assert calls == []
    assert list(run_root.iterdir()) == []


# --- a task that widens individual rows onto pair rows -----------------------------------


class _Echoes(ParallelFeature):
    """Writes the table it is applied to."""

    name = "test-echoes"


def _widening_inputs(directory: Path, *, pair_frames: int) -> tuple[Path, Path]:
    """A pair-level file of *pair_frames* frames and an individual-level file of three.

    The individual file holds both individuals of the pair in every frame.
    """
    directory.mkdir(exist_ok=True)
    pairs, individuals = directory / "pairs.parquet", directory / "individuals.parquet"
    make_pair_df(pair_frames, 1).to_parquet(pairs)
    one = make_sequence_df(3, 1)
    pd.concat([one.assign(id=0), one.assign(id=1)], ignore_index=True).to_parquet(
        individuals
    )
    return pairs, individuals


def test_a_widening_load_gives_the_table_the_run_loads(
    run_root: Path, tmp_path: Path
) -> None:
    paths = _widening_inputs(tmp_path / "inputs", pair_frames=3)
    loaded = load_entry_data(list(_specs(paths)), expands_individuals=True)
    assert loaded is not None
    assert "id" not in loaded.columns
    unwidened = tmp_path / "unwidened"
    unwidened.mkdir()

    task = _load_task(run_root, paths, expands_individuals=True)
    report = report_of(run_entry(_Echoes(), task, None))
    failure = _failure(run_entry(_Echoes(), _load_task(unwidened, paths), None))

    pd.testing.assert_frame_equal(pd.read_parquet(task.meta.out_path), loaded)
    assert report.rows == len(loaded)
    assert isinstance(failure.error, MultiInputAlignmentError)
    assert not isinstance(failure.error, PairRowsMissing)
    assert list(unwidened.iterdir()) == []


def test_a_widening_load_without_pair_rows_raises_as_the_entrys_error(
    run_root: Path, tmp_path: Path
) -> None:
    paths = _widening_inputs(tmp_path / "inputs", pair_frames=0)
    task = _load_task(run_root, paths, expands_individuals=True)

    with pytest.raises(PairRowsMissing) as raised:
        _ = run_entry(_Echoes(), task, None)

    message = (
        "input 0 holds no pair rows for this entry, so the individual rows of "
        "input 1 have none to widen onto"
    )
    assert str(raised.value) == message
    crossed = _crossed(raised.value)
    assert type(crossed) is PairRowsMissing
    assert str(crossed) == message
    assert list(run_root.iterdir()) == []


def test_a_task_handed_the_footers_parses_none_to_find_missing_pair_rows(
    run_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _widening_inputs(tmp_path / "inputs", pair_frames=0)
    records = _crossed_records(paths)
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")

    handed = _load_task(run_root, paths, expands_individuals=True, footers=records)
    with pytest.raises(PairRowsMissing):
        _ = run_entry(_Echoes(), handed, None)
    assert parsed == []

    bare = _load_task(run_root, paths, expands_individuals=True)
    with pytest.raises(PairRowsMissing):
        _ = run_entry(_Echoes(), bare, None)
    assert paths[0] in parsed


@pytest.mark.parametrize("pair_frames", [3, 0], ids=["rows", "no-rows"])
def test_a_one_input_load_reads_alike_with_and_without_the_flag(
    tmp_path: Path, pair_frames: int
) -> None:
    pairs, _ = _widening_inputs(tmp_path / "inputs", pair_frames=pair_frames)
    reports: list[EntryReport] = []
    outputs: list[Path] = []
    for expands in (True, False):
        root = tmp_path / f"run-{expands}"
        root.mkdir()
        task = _load_task(root, (pairs,), expands_individuals=expands)
        reports.append(report_of(run_entry(_Echoes(), task, None)))
        outputs.append(task.meta.out_path)

    assert [report.written for report in reports] == [pair_frames > 0] * 2
    if pair_frames:
        assert outputs[0].read_bytes() == outputs[1].read_bytes()
    else:
        assert not outputs[0].exists() and not outputs[1].exists()


# --- a task that filters its pair rows to the nearest neighbors ------------------------


def _pair_filtered_task(
    run_root: Path,
    ds: Dataset,
    pairs: Result,
    neighbors: NNResult,
    *,
    dataset: Dataset | str | None = None,
) -> EntryTask:
    """A load of the pair rows of entry ``g__s``, filtered by the neighbor run."""
    path = result_run_root(ds, pairs) / "g__s.parquet"
    pair_filter = PairFilter(
        dataset=ds if dataset is None else dataset,
        spec=neighbors,
        group="g",
        sequence="s",
    )
    return _load_task(run_root, (path,), pair_filter=pair_filter)


def _loaded_by_the_run(
    ds: Dataset, pairs: Result, neighbors: NNResult | None
) -> pd.DataFrame:
    """The pair rows of ``g__s`` as the run loads them, filtered when *neighbors* is given."""
    filter_ = None if neighbors is None else nn_pair_filter(ds, "g", "s", neighbors)
    table = load_entry_data(
        [(result_run_root(ds, pairs) / "g__s.parquet", ParquetLoadSpec())],
        filters=[] if filter_ is None else [filter_],
        expands_individuals=False,
        projection=None,
    )
    assert table is not None
    return table


def _bytes_written_for(tmp_path: Path, table: pd.DataFrame) -> bytes:
    """The bytes ``TracksFeature`` writes for *table*."""
    meta = FeatureMeta(group="g", sequence="s", out_path=tmp_path / "expected.parquet")
    _ = write_output(meta, TracksFeature().apply(table))
    return meta.out_path.read_bytes()


def test_a_pair_filtered_load_gives_the_table_the_run_filters(
    run_root: Path, tmp_path: Path
) -> None:
    ds, pairs, _, neighbors = pair_run_results(tmp_path / "data", "s")
    task = _pair_filtered_task(run_root, ds, pairs, neighbors)
    unfiltered = _loaded_by_the_run(ds, pairs, None)
    filtered = _loaded_by_the_run(ds, pairs, neighbors)
    assert len(filtered) < len(unfiltered)

    _ = report_of(run_entry(TracksFeature(), task, None))

    assert task.meta.out_path.read_bytes() == _bytes_written_for(tmp_path, filtered)
    written = pd.read_parquet(task.meta.out_path)
    assert len(written) == len(filtered)
    assert ordered_pairs(written) == neighbor_pairs(ds, neighbors, "g", "s")


def test_a_pickled_pair_filter_opens_the_dataset_by_its_manifest(
    run_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, pairs, _, neighbors = pair_run_results(tmp_path / "data", "s")
    reference = _pair_filtered_task(run_root, ds, pairs, neighbors)
    _ = report_of(run_entry(TracksFeature(), reference, None))
    untouched = ds.get_root("media_raw")
    assert untouched.exists()
    shutil.rmtree(untouched)
    opened: dict[tuple[str, bool], Dataset] = {}
    monkeypatch.setattr(entry_pool, "_OPENED", opened)
    manifest = str(ds.manifest_path)
    crossed = _crossed(
        _pair_filtered_task(
            tmp_path / "crossed", ds, pairs, neighbors, dataset=manifest
        )
    )
    (tmp_path / "crossed").mkdir()

    _ = report_of(run_entry(TracksFeature(), crossed, None))

    assert crossed.meta.out_path.read_bytes() == reference.meta.out_path.read_bytes()
    assert list(opened) == [(manifest, False)]
    assert not untouched.exists()


def test_a_neighbor_table_that_cannot_be_read_is_the_loads_failure(
    run_root: Path, tmp_path: Path
) -> None:
    ds, pairs, _, neighbors = pair_run_results(tmp_path / "data", "s")
    corrupt_output(ds, neighbors, "s")
    task = _pair_filtered_task(run_root, ds, pairs, neighbors)

    failure = _failure(run_entry(TracksFeature(), task, None))

    assert failure.key == make_entry_key("g", "s")
    assert type(failure.error).__name__ == "ArrowInvalid"
    assert list(run_root.iterdir()) == []


def test_a_load_with_an_empty_lookup_filters_nothing(
    run_root: Path, tmp_path: Path
) -> None:
    ds, pairs, _, neighbors = pair_run_results(
        tmp_path / "data", "s", "t", neighbors_of=["t"]
    )
    task = _pair_filtered_task(run_root, ds, pairs, neighbors)

    _ = report_of(run_entry(TracksFeature(), task, None))

    assert task.meta.out_path.read_bytes() == _bytes_written_for(
        tmp_path, _loaded_by_the_run(ds, pairs, None)
    )


# --- footers handed to the task --------------------------------------------------------


_PROJECTION = ColumnProjection(frozenset({"frame", "time", "id", "X", "sequence"}))
"""A projection, so that the load reads each input's footer to select columns."""


def _crossed_records(paths: tuple[Path, ...]) -> tuple[FooterRecord, ...]:
    """The records of a memo that read every input, passed through ``pickle``."""
    memo = ParquetFooterMemo()
    for path in paths:
        _ = memo.footer(path)
    return _crossed(memo.records(paths))


def test_a_task_handed_the_footers_parses_no_input_whose_stamp_did_not_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _inputs(tmp_path / "tracks")
    records = _crossed_records(paths)
    assert len(records) == len(paths)
    handed, unhanded = tmp_path / "handed", tmp_path / "unhanded"
    handed.mkdir()
    unhanded.mkdir()
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")

    task = _load_task(handed, paths, projection=_PROJECTION, footers=records)
    _ = report_of(run_entry(ParallelFeature(), task, None))
    assert parsed == []

    bare = _load_task(unhanded, paths, projection=_PROJECTION)
    _ = report_of(run_entry(ParallelFeature(), bare, None))
    assert parsed == list(paths)
    assert task.meta.out_path.read_bytes() == bare.meta.out_path.read_bytes()


def test_a_task_handed_the_footers_parses_an_input_rewritten_since_they_were_read(
    run_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _inputs(tmp_path / "tracks")
    records = _crossed_records(paths)
    rewritten = dropped_frame_tracks(sequence="s", n_rows=12)
    rewritten[["frame", "id", "ANGLE"]].to_parquet(paths[1])
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")

    task = _load_task(run_root, paths, projection=_PROJECTION, footers=records)
    _ = report_of(run_entry(ParallelFeature(), task, None))

    assert parsed == [paths[1]]


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
    handed = int(_table_of(task).memory_usage(deep=True).sum())
    seen: list[bool] = []
    monkeypatch.setattr(
        entry_task, "process_nonreclaimable_bytes", lambda: handed + 5_000
    )
    monkeypatch.setattr(
        entry_task,
        "peak_nonreclaimable_while",
        _sampler(seen, task.meta.out_path, 9_000),
    )

    worker = report_of(run_entry_in_process(ParallelFeature(), task)).worker

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

    worker = report_of(run_entry_in_process(ParallelFeature(), task)).worker

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

    report = report_of(run_entry_in_process(ParallelFeature(), task))

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

    report = report_of(run_entry_in_process(ParallelFeature(), task))

    assert report.worker is None
    assert report.rows == len(_table_of(task))
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
    pd.testing.assert_frame_equal(_table_of(crossed), _table_of(task))
    assert crossed.selector == task.selector
    assert crossed.arena_count == task.arena_count
    assert crossed.key == task.key == make_entry_key("g", "s")


def test_a_report_survives_pickle(run_root: Path) -> None:
    report = report_of(run_entry_in_process(_NotesUnevenTime(), _task(run_root)))
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


def test_a_load_task_and_its_report_survive_pickle(
    run_root: Path, tmp_path: Path
) -> None:
    paths = _inputs(tmp_path / "tracks")
    task = _load_task(
        run_root,
        paths,
        frame_start=1,
        frame_end=30,
        time_window=TimeWindow(0.0, 2.0),
        projection=_PROJECTION,
        footers=_crossed_records(paths),
    )
    load = task.source
    assert isinstance(load, EntryLoad)

    crossed = _crossed(task)
    crossed_load = crossed.source
    report = report_of(run_entry_in_process(ParallelFeature(), crossed))

    assert crossed.meta == task.meta
    assert crossed.selector is None
    assert crossed.arena_count is None
    assert isinstance(crossed_load, EntryLoad)
    assert crossed_load.entry_key == load.entry_key
    assert crossed_load.file_specs == load.file_specs
    assert crossed_load.projection == load.projection
    assert crossed_load.frame_start == load.frame_start
    assert crossed_load.frame_end == load.frame_end
    assert crossed_load.time_window == load.time_window
    assert crossed_load.footers == load.footers
    assert crossed_load == load
    assert report.written
    assert report.input_bytes > 0
    assert _crossed(report) == report


def test_a_process_worker_given_a_load_reads_its_base_before_it_and_subtracts_nothing(
    run_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []

    def resident() -> int:
        order.append("base")
        return 7_000

    def on_load(*_: object, **__: object) -> None:
        order.append("load")

    monkeypatch.setattr(entry_task, "process_rss_bytes", resident)
    monkeypatch.setattr(entry_task, "process_nonreclaimable_bytes", lambda: 5_000)
    monkeypatch.setattr(
        entry_task, "load_entry_data", observed(load_entry_data, on_load)
    )

    report = report_of(
        run_entry_in_process(
            ParallelFeature(), _load_task(run_root, _inputs(tmp_path / "tracks"))
        )
    )

    assert order == ["base", "load"]
    assert report.worker is not None
    assert report.worker.base_rss_bytes == 7_000
    assert report.worker.base_nonreclaimable_bytes == 5_000
