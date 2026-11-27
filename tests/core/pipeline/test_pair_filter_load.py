"""A run whose ``pair_filter`` names a nearest-neighbor run keeps only the neighbor pairs.

Each run goes in three modes: serial (one worker), thread (two workers) and
process (two workers), and each test asserts that the entries it computed ran in
the mode asked for, so a fallback to threads cannot pass. The stubs are built as
``cls(inputs, params)`` so that a process worker can build them.

The dataset holds three individuals per frame. The tracks offset individual
``k`` by ``k`` on both axes, so the neighbor of individuals 0 and 2 is 1, the
neighbor of 1 is 0, and the pairs ``(0, 2)`` and ``(2, 0)`` are nobody's
neighbor pairs. A run over the pair rows therefore drops 2 of the 6 ordered
pairs of each frame, and a test that kept every pair would fail where one over
two individuals, each the other's neighbor, would pass.

A run with overlap reads each entry with the frames of its neighbors. Its
``apply`` counts the frames and the pair rows of the whole table it receives,
before the trim, so a test sees that the neighbor's frame was filtered by the
neighbor's own table.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from mosaic.behavior.feature_library.nearestneighbor import NearestNeighbor
from mosaic.core.dataset import Dataset
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline import loading, parquet_reader
from mosaic.core.pipeline._utils import FeatureMeta
from mosaic.core.pipeline.entry_pool import EntryPool
from mosaic.core.pipeline.entry_task import EntryLoad, EntryTask
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import NNResult, Result
from tests.helpers import (
    EMPTY_SEQUENCE,
    ROWS,
    EchoFeature,
    Pair,
    PairRowsFeature,
    RunMode,
    add_pair_tracks,
    assert_ended_without_an_entry_error,
    assert_lost_the_entry_without_pair_rows,
    assert_ran_in_mode,
    copy_of,
    corrupt_output,
    latest_events,
    make_dataset,
    neighbor_pairs,
    observed,
    ordered_pairs,
    pair_run_results,
    record_path_calls,
    result_outputs,
    result_run_root,
    run_in_mode,
    run_modes,
)

_MODES: Final = run_modes(2)

_RunPool = EntryPool[FeatureMeta, object]
"""The pool of a feature run."""

_KEPT_PER_FRAME: Final = 4
"""The ordered pairs of a frame that are neighbor pairs."""

_PAIRS_PER_FRAME: Final = 6
"""The ordered pairs of a frame."""


class _FilteredEcho(EchoFeature):
    """An ``as-input`` consumer that filters its pair rows to a neighbor run's pairs."""

    name = "test-filtered-echo"

    class Params(EchoFeature.Params):
        pair_filter: NNResult | None = None


class _FilteredEchoWithOverlap(_FilteredEcho):
    """A filtered consumer that accepts overlap and counts the table it receives.

    Each output row carries the frames and the pair rows of the whole table
    ``apply`` received, before the trim to the entry's own frames.
    """

    name = "test-filtered-echo-overlap"
    accepts_overlap = True

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        counted = df.copy()
        counted["n_frames"] = df["frame"].nunique()
        counted["n_pair_rows"] = len(df)
        return counted


def _kept(ds: Dataset, neighbors: Result, sequence: str) -> list[Pair]:
    """The ordered pairs of *sequence* where one individual's neighbor is the other."""
    return neighbor_pairs(ds, neighbors, "g", sequence)


def _filtered(pairs: Result, neighbors: NNResult) -> _FilteredEcho:
    return _FilteredEcho(_FilteredEcho.Inputs((pairs,)), {"pair_filter": neighbors})


@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_a_pair_filtered_run_keeps_the_pairs_of_nearest_neighbors(
    tmp_path: Path, mode: RunMode
) -> None:
    ds, pairs, _, neighbors = pair_run_results(tmp_path, "s1", "s2")

    result = run_in_mode(ds, _filtered(pairs, neighbors), mode)

    assert_ran_in_mode(ds, mode, entries=2)
    outputs = result_outputs(ds, result)
    assert set(outputs) == {make_entry_key("g", s) for s in ("s1", "s2")}
    held = result_outputs(ds, pairs)
    for sequence in ("s1", "s2"):
        key = make_entry_key("g", sequence)
        assert len(held[key]) == ROWS * _PAIRS_PER_FRAME
        assert len(outputs[key]) == ROWS * _KEPT_PER_FRAME
        assert ordered_pairs(outputs[key]) == _kept(ds, neighbors, sequence)


@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_an_entry_the_neighbor_run_lacks_keeps_every_pair(
    tmp_path: Path, mode: RunMode
) -> None:
    ds, pairs, _, neighbors = pair_run_results(
        tmp_path, "s1", "s2", neighbors_of=["s1"]
    )

    result = run_in_mode(ds, _filtered(pairs, neighbors), mode)

    assert_ran_in_mode(ds, mode, entries=2)
    outputs = result_outputs(ds, result)
    assert len(outputs[make_entry_key("g", "s1")]) == ROWS * _KEPT_PER_FRAME
    assert len(outputs[make_entry_key("g", "s2")]) == ROWS * _PAIRS_PER_FRAME
    assert ordered_pairs(outputs[make_entry_key("g", "s1")]) == _kept(
        ds, neighbors, "s1"
    )


@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_a_neighbor_run_of_a_linked_dataset_filters_the_copys_run(
    tmp_path: Path, mode: RunMode
) -> None:
    source = make_dataset(tmp_path / "source")
    add_pair_tracks(source, ("s1", "s2"))
    ran = run_feature(source, NearestNeighbor())
    neighbors = NNResult(feature=ran.feature, run_id=ran.run_id)
    copy = copy_of(source, tmp_path / "copy")
    add_pair_tracks(copy, ("s1", "s2"))
    held = run_feature(copy, PairRowsFeature())
    pairs = Result(feature=held.feature, run_id=held.run_id)

    result = run_in_mode(copy, _filtered(pairs, neighbors), mode)

    assert_ran_in_mode(copy, mode, entries=2)
    outputs = result_outputs(copy, result)
    for sequence in ("s1", "s2"):
        output = outputs[make_entry_key("g", sequence)]
        assert len(output) == ROWS * _KEPT_PER_FRAME
        assert ordered_pairs(output) == _kept(source, neighbors, sequence)


@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_a_pair_filtered_widening_entry_without_pair_rows_is_an_entry_error(
    tmp_path: Path, mode: RunMode
) -> None:
    ds, pairs, individuals, neighbors = pair_run_results(tmp_path, "s1", EMPTY_SEQUENCE)

    result = run_in_mode(
        ds,
        _FilteredEcho(
            _FilteredEcho.Inputs((pairs, individuals)), {"pair_filter": neighbors}
        ),
        mode,
    )

    assert_ran_in_mode(ds, mode, entries=1)
    assert_lost_the_entry_without_pair_rows(ds, result)
    (output,) = result_outputs(ds, result).values()
    assert {"speed__id1", "speed__id2"} <= set(output.columns)
    assert len(output) == ROWS * _KEPT_PER_FRAME
    assert ordered_pairs(output) == _kept(ds, neighbors, "s1")


@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_a_corrupt_neighbor_table_ends_the_run(tmp_path: Path, mode: RunMode) -> None:
    """The first entry sorts before the corrupt one, so it computes before the run ends."""
    ds, pairs, _, neighbors = pair_run_results(tmp_path, "s1", "s2")
    corrupt_output(ds, neighbors, "s2")

    with pytest.raises(
        ValueError, match="Parquet magic bytes not found in footer"
    ) as raised:
        _ = run_in_mode(ds, _filtered(pairs, neighbors), mode)

    assert type(raised.value).__name__ == "ArrowInvalid"
    assert_ran_in_mode(ds, mode, entries=1)
    assert_ended_without_an_entry_error(ds, raised.value)


@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_a_time_window_over_a_pair_filtered_input_without_time_ends_the_run(
    tmp_path: Path, mode: RunMode
) -> None:
    ds, pairs, _, neighbors = pair_run_results(tmp_path, "s1", "s2")

    with pytest.raises(
        ValueError, match="a time filter reads the 'time' column"
    ) as raised:
        _ = run_in_mode(ds, _filtered(pairs, neighbors), mode, filter_start_time=0.0)

    assert_ran_in_mode(ds, mode, entries=0)
    assert_ended_without_an_entry_error(ds, raised.value)


def test_a_pair_filtered_run_parses_each_input_footer_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, pairs, _, neighbors = pair_run_results(tmp_path, "s1", "s2")
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")

    _ = run_feature(
        ds,
        _filtered(pairs, neighbors),
        parallel_workers=2,
        parallel_mode="thread",
    )

    assert len(latest_events(ds, "entry_end")) == 2
    assert parsed
    assert len(parsed) == len(set(parsed)), sorted(parsed)
    for sequence in ("s1", "s2"):
        output = result_run_root(ds, neighbors) / (
            f"{make_entry_key('g', sequence)}.parquet"
        )
        assert parsed.count(output) == 1, output


def test_a_pair_filtered_widening_run_parses_each_input_footer_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The footer of an empty input 0, which the load reads, comes from the handed records.

    The admission gate makes the run read each entry's input footers to size it.
    """
    ds, pairs, individuals, neighbors = pair_run_results(tmp_path, "s1", EMPTY_SEQUENCE)
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")

    _ = run_feature(
        ds,
        _FilteredEcho(
            _FilteredEcho.Inputs((pairs, individuals)), {"pair_filter": neighbors}
        ),
        parallel_workers=2,
        parallel_mode="thread",
        admission_margin=1.5,
    )

    assert len(latest_events(ds, "entry_end")) == 1
    assert len(parsed) == len(set(parsed)), sorted(parsed)
    empty_pairs = result_run_root(ds, pairs) / (
        f"{make_entry_key('g', EMPTY_SEQUENCE)}.parquet"
    )
    assert parsed.count(empty_pairs) == 1, empty_pairs


@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_an_overlapped_pair_filtered_run_filters_each_entry_and_its_neighbors(
    tmp_path: Path, mode: RunMode
) -> None:
    """An entry's output holds its own kept pairs, and ``apply`` saw its neighbor's too."""
    ds, pairs, _, neighbors = pair_run_results(tmp_path, "s1", "s2", continuous=True)

    result = run_in_mode(
        ds,
        _FilteredEchoWithOverlap(
            _FilteredEchoWithOverlap.Inputs((pairs,)), {"pair_filter": neighbors}
        ),
        mode,
        overlap_frames=1,
    )

    assert_ran_in_mode(ds, mode, entries=2)
    outputs = result_outputs(ds, result)
    for sequence in ("s1", "s2"):
        output = outputs[make_entry_key("g", sequence)]
        assert ordered_pairs(output) == _kept(ds, neighbors, sequence)
        frames = ROWS + 1
        assert set(output["n_frames"]) == {frames}
        assert set(output["n_pair_rows"]) == {frames * _KEPT_PER_FRAME}


def _record_lookup_threads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the name of the thread of each read of a neighbor run's rows.

    ``feature_run_rows`` is read by ``build_nn_lookup`` alone, which the run's
    process reaches through ``loading``, so the list holds one name per lookup
    built in this process.
    """
    threads: list[str] = []

    def record(*_: object, **__: object) -> None:
        threads.append(threading.current_thread().name)

    monkeypatch.setattr(
        loading, "feature_run_rows", observed(loading.feature_run_rows, record)
    )
    return threads


def test_a_pair_filtered_entry_builds_its_lookup_in_its_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without overlap each lookup is built by a pool thread, and with it by the run."""
    ds, pairs, _, neighbors = pair_run_results(tmp_path / "free", "s1", "s2")
    threads = _record_lookup_threads(monkeypatch)

    _ = run_feature(
        ds, _filtered(pairs, neighbors), parallel_workers=2, parallel_mode="thread"
    )

    assert len(threads) == 2, threads
    assert all(name.startswith("ThreadPoolExecutor") for name in threads), threads

    ds, pairs, _, neighbors = pair_run_results(
        tmp_path / "overlap", "s1", "s2", continuous=True
    )
    threads.clear()

    _ = run_feature(
        ds,
        _FilteredEchoWithOverlap(
            _FilteredEchoWithOverlap.Inputs((pairs,)), {"pair_filter": neighbors}
        ),
        parallel_workers=2,
        parallel_mode="thread",
        overlap_frames=1,
    )

    assert len(threads) >= 2, threads
    assert set(threads) == {threading.main_thread().name}, threads


def test_a_process_worker_builds_its_lookup_from_the_dataset_it_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run's process reads no neighbor table: the worker does, in its own process."""
    mode = _MODES[2]
    ds, pairs, _, neighbors = pair_run_results(tmp_path, "s1", "s2")

    def refuse(*_: object, **__: object) -> pd.DataFrame:
        msg = "the run's process built a lookup"
        raise AssertionError(msg)

    monkeypatch.setattr(loading, "feature_run_rows", refuse)

    result = run_in_mode(ds, _filtered(pairs, neighbors), mode)

    assert_ran_in_mode(ds, mode, entries=2)
    outputs = result_outputs(ds, result)
    for sequence in ("s1", "s2"):
        output = outputs[make_entry_key("g", sequence)]
        assert ordered_pairs(output) == _kept(ds, neighbors, sequence)


@pytest.mark.parametrize("mode", _MODES[1:], ids=lambda mode: mode.name)
def test_a_task_names_its_dataset_as_its_mode_hands_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: RunMode
) -> None:
    ds, pairs, _, neighbors = pair_run_results(tmp_path, "s1", "s2")
    tasks: list[EntryTask] = []

    real = _RunPool.submit

    def submit(
        self: _RunPool, tag: FeatureMeta, fn: Callable[..., object], *args: object
    ) -> None:
        tasks.extend(arg for arg in args if isinstance(arg, EntryTask))
        real(self, tag, fn, *args)

    monkeypatch.setattr(EntryPool, "submit", submit)

    _ = run_in_mode(ds, _filtered(pairs, neighbors), mode)

    assert_ran_in_mode(ds, mode, entries=2)
    assert len(tasks) == 2
    for task in tasks:
        assert isinstance(task.source, EntryLoad)
        assert task.source.pair_filter is not None
        expected = str(ds.manifest_path) if mode.kind == "process" else ds
        assert task.source.pair_filter.dataset == expected
        if mode.kind == "thread":
            assert task.source.pair_filter.dataset is ds
