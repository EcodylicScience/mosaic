"""A run widens an individual-level input onto the pair rows of its input 0.

The loader's rule is pinned in ``test_multi_input_alignment.py``. These run
``run_feature`` over a pair-level and an individual-level result, so the flag
reaches the loader from the feature's ``emits``, and an entry whose pair input
holds no rows is reported as lost.

Each run goes in three modes: serial (one worker), thread (two workers) and
process (two workers), and each test asserts that the entries it computed ran in
the mode asked for, so a fallback to threads cannot pass. A run with overlap
widens too, and loses the entry without pair rows as a run without overlap does.
The stubs are built as ``cls(inputs, params)`` so that a process worker can
build them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.helpers import make_entry_key
from mosaic.core.params import Params
from mosaic.core.pipeline import entry_task, parquet_reader
from mosaic.core.pipeline.loading import MultiInputAlignmentError
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import (
    Feature,
    Inputs,
    InputStream,
    Result,
    TrackInput,
)
from mosaic.core.pipeline.feature_levels import EmitsLevel
from tests.helpers import (
    RunMode,
    add_track_sequences,
    all_logged_events,
    assert_ran_in_mode,
    latest_events,
    latest_snapshot,
    make_dataset,
    observed,
    record_path_calls,
    result_outputs,
    result_run_root,
    run_modes,
)

_EMPTY = "empty"

_NO_PAIR_ROWS_MESSAGE: Final = (
    "input 0 holds no pair rows for this entry, so the individual rows of "
    "input 1 have none to widen onto"
)
"""What the entry without pair rows reports."""


_MODES: Final = run_modes(2)


class _Stub:
    """The boilerplate of a test feature; subclasses set the inputs and ``apply``."""

    name = "test-stub"
    version = "0.1"
    category = "per-frame"
    emits: EmitsLevel = "individual"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()

    class Params(Params):
        pass

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self.params: _Stub.Params = self.Params.from_overrides(params)

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        return None

    def save_state(self, run_root: Path) -> None:
        return None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class _OverTracks(_Stub):
    """A stub reading the tracks."""

    class Inputs(Inputs[TrackInput]):
        pass

    def __init__(self) -> None:
        super().__init__()
        self.inputs: _OverTracks.Inputs = self.Inputs(("tracks",))


class _PairRows(_OverTracks):
    """Both orderings of every pair of individuals per frame, none for ``_EMPTY``."""

    name = "test-pair-rows"
    emits: EmitsLevel = "pair"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        ids = sorted(int(i) for i in df["id"].unique())
        rows = [
            {"frame": int(frame), "id1": a, "id2": b, "perspective": int(a > b)}
            for frame in sorted(df["frame"].unique())
            for a in ids
            for b in ids
            if a != b
        ]
        table = pd.DataFrame(rows, columns=["frame", "id1", "id2", "perspective"])
        table["dist"] = 1.0
        if (df["sequence"] == _EMPTY).all():
            return table.iloc[0:0]
        return table


class _IndividualRows(_OverTracks):
    """``speed`` names its cell: ten times the frame plus the id."""

    name = "test-individual-rows"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "frame": df["frame"],
                "id": df["id"],
                "speed": 10.0 * df["frame"] + df["id"],
            }
        )


class _Echo(_Stub):
    """An ``as-input`` consumer that writes its merged input."""

    name = "test-echo"
    emits: EmitsLevel = "as-input"

    class Inputs(Inputs[Result]):
        pass

    def __init__(
        self, inputs: _Echo.Inputs, params: dict[str, object] | None = None
    ) -> None:
        super().__init__(params)
        self.inputs: _Echo.Inputs = inputs


class _FitsEcho(_Echo):
    """An ``as-input`` consumer that fits by reading every entry of its stream."""

    name = "test-fits-echo"

    def __init__(
        self, inputs: _Echo.Inputs, params: dict[str, object] | None = None
    ) -> None:
        super().__init__(inputs, params)
        self.fitted_entries: list[str] = []

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        return False

    def fit(self, inputs: InputStream) -> None:
        self.fitted_entries = [entry_key for entry_key, _ in inputs()]


class _EchoWithOverlap(_Echo):
    """An ``as-input`` consumer that accepts overlap."""

    name = "test-echo-overlap"
    accepts_overlap = True


class _EchoPerIndividual(_Echo):
    name = "test-echo-individual"
    emits: EmitsLevel = "individual"


def _results(tmp_path: Path, *sequences: str) -> tuple[Dataset, Result, Result]:
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, *(("g", s) for s in sequences), n_rows=6, n_ids=2)
    pairs = run_feature(ds, _PairRows())
    individuals = run_feature(ds, _IndividualRows())
    return (
        ds,
        Result(feature=pairs.feature, run_id=pairs.run_id),
        Result(feature=individuals.feature, run_id=individuals.run_id),
    )


def _entry_errors(ds: Dataset) -> list[tuple[object, dict[str, str]]]:
    """The key and the parsed error of each ``entry_error`` of the last attempt."""
    errors: list[tuple[object, dict[str, str]]] = []
    for event in latest_events(ds, "entry_error"):
        error = event["error"]
        assert isinstance(error, str)
        errors.append((event["key"], json.loads(error)))
    return errors


def _lost_the_entry_without_pair_rows(ds: Dataset, result: Result) -> None:
    """Assert that the run lost the entry of ``_EMPTY``, recorded it and went on."""
    assert result.failed_entries == (make_entry_key("g", _EMPTY),)
    errors = _entry_errors(ds)
    assert len(errors) == 1
    key, error = errors[0]
    assert key == make_entry_key("g", _EMPTY)
    assert error["type"] == "PairRowsMissing"
    assert error["message"] == _NO_PAIR_ROWS_MESSAGE
    assert latest_snapshot(ds)["status"] == "finished"


def _run_in(
    ds: Dataset,
    feature: Feature,
    mode: RunMode,
    *,
    overlap_frames: int = 0,
    filter_start_time: float | None = None,
) -> Result:
    return run_feature(
        ds,
        feature,
        **mode.run_kwargs(),
        overlap_frames=overlap_frames,
        filter_start_time=filter_start_time,
    )


@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_an_as_input_run_reads_each_pair_row_with_both_individuals(
    tmp_path: Path, mode: RunMode
) -> None:
    ds, pairs, individuals = _results(tmp_path, "s1")

    echoed = _run_in(ds, _Echo(_Echo.Inputs((pairs, individuals))), mode)

    assert_ran_in_mode(ds, mode, entries=1)
    (table,) = result_outputs(ds, echoed).values()
    assert {"id1", "id2", "dist", "speed__id1", "speed__id2"} <= set(table.columns)
    assert "id" not in table.columns
    assert len(table) == 12
    for slot in ("id1", "id2"):
        expected = 10.0 * table["frame"] + table[slot]
        assert list(table[f"speed__{slot}"]) == list(expected)


def test_an_individual_level_consumer_keeps_the_refusal(tmp_path: Path) -> None:
    ds, pairs, individuals = _results(tmp_path, "s1")

    with pytest.raises(MultiInputAlignmentError, match="pair every row"):
        _ = run_feature(
            ds, _EchoPerIndividual(_EchoPerIndividual.Inputs((pairs, individuals)))
        )


@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_an_entry_without_pair_rows_is_an_entry_error(
    tmp_path: Path, mode: RunMode
) -> None:
    """The entry is lost and reported, and the run keeps the others."""
    ds, pairs, individuals = _results(tmp_path, "s1", _EMPTY)

    result = _run_in(ds, _Echo(_Echo.Inputs((pairs, individuals))), mode)

    assert_ran_in_mode(ds, mode)
    _lost_the_entry_without_pair_rows(ds, result)
    assert set(result_outputs(ds, result)) == {make_entry_key("g", "s1")}
    assert [end["key"] for end in latest_events(ds, "entry_end")] == [
        make_entry_key("g", "s1")
    ]


def test_an_entry_without_pair_rows_leaves_the_gate(tmp_path: Path) -> None:
    """The lost entry leaves the in-flight record before the next one enters."""
    ds, pairs, individuals = _results(tmp_path, _EMPTY, "s1", "s2")
    execution_id = "TESTWIDENGATE"

    result = run_feature(
        ds, _Echo(_Echo.Inputs((pairs, individuals))), execution_id=execution_id
    )

    assert result.failed_entries == (make_entry_key("g", _EMPTY),)
    events = all_logged_events(ds.base_dir, execution_id)
    names = [event["ev"] for event in events]
    assert names.index("entry_error") < names.index("entry_end")
    ends = [event for event in events if event["ev"] == "entry_end"]
    assert [end["key"] for end in ends] == [
        make_entry_key("g", "s1"),
        make_entry_key("g", "s2"),
    ]
    assert [end["in_flight"] for end in ends] == [1, 1]


@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_an_overlapped_widening_run_records_the_entry_without_pair_rows(
    tmp_path: Path, mode: RunMode
) -> None:
    """A run that reads in the parent records the lost entry as a worker's run does."""
    ds, pairs, individuals = _results(tmp_path, "s1", _EMPTY)

    result = _run_in(
        ds,
        _EchoWithOverlap(_EchoWithOverlap.Inputs((pairs, individuals))),
        mode,
        overlap_frames=1,
    )

    assert_ran_in_mode(ds, mode)
    _lost_the_entry_without_pair_rows(ds, result)


def _corrupt(ds: Dataset, result: Result, sequence: str) -> None:
    """Overwrite the output of *result* for *sequence* with bytes that are not parquet."""
    path = result_run_root(ds, result) / f"{make_entry_key('g', sequence)}.parquet"
    assert path.exists()
    _ = path.write_bytes(b"not a parquet file")


def _ended_without_an_entry_error(ds: Dataset, raised: BaseException) -> None:
    """Assert that the run failed with *raised* and wrote no ``entry_error``."""
    snapshot = latest_snapshot(ds)
    assert snapshot["status"] == "failed"
    assert json.loads(snapshot["error_json"])["type"] == type(raised).__name__
    assert latest_events(ds, "entry_error") == []


@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_a_corrupt_input_of_a_widening_run_ends_the_run(
    tmp_path: Path, mode: RunMode
) -> None:
    ds, pairs, individuals = _results(tmp_path, "s1", "s2")
    _corrupt(ds, individuals, "s2")

    with pytest.raises(ValueError, match="magic bytes not found") as raised:
        _ = _run_in(ds, _Echo(_Echo.Inputs((pairs, individuals))), mode)

    assert_ran_in_mode(ds, mode)
    _ended_without_an_entry_error(ds, raised.value)


@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_a_time_window_over_a_widening_input_without_time_ends_the_run(
    tmp_path: Path, mode: RunMode
) -> None:
    """The entry of ``_EMPTY`` is left out, since its missing pair rows raise first."""
    ds, pairs, individuals = _results(tmp_path, "s1", "s2")

    with pytest.raises(
        ValueError, match="a time filter reads the 'time' column"
    ) as raised:
        _ = _run_in(
            ds,
            _Echo(_Echo.Inputs((pairs, individuals))),
            mode,
            filter_start_time=0.0,
        )

    assert_ran_in_mode(ds, mode)
    _ended_without_an_entry_error(ds, raised.value)


def test_a_widening_run_parses_each_input_footer_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, pairs, individuals = _results(tmp_path, _EMPTY, "s1")
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")

    result = run_feature(
        ds,
        _Echo(_Echo.Inputs((pairs, individuals))),
        parallel_workers=2,
        parallel_mode="thread",
    )

    assert result.failed_entries == (make_entry_key("g", _EMPTY),)
    assert parsed
    assert len(parsed) == len(set(parsed)), sorted(parsed)


@pytest.mark.parametrize(
    ("feature_class", "overlap_frames", "loaded_by_the_task"),
    [(_Echo, 0, True), (_EchoWithOverlap, 1, False)],
    ids=["widening", "overlap"],
)
def test_a_widening_entry_is_loaded_by_its_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feature_class: type[_Echo],
    overlap_frames: int,
    loaded_by_the_task: bool,
) -> None:
    """A run with overlap loads in the parent, so its tasks load nothing."""
    ds, pairs, individuals = _results(tmp_path, "s1", "s2")
    widened: list[object] = []

    def on_load(*_: object, **keywords: object) -> None:
        widened.append(keywords["expands_individuals"])

    monkeypatch.setattr(
        entry_task,
        "load_entry_data",
        observed(entry_task.load_entry_data, on_load),
    )

    _ = run_feature(
        ds,
        feature_class(feature_class.Inputs((pairs, individuals))),
        parallel_workers=2,
        parallel_mode="thread",
        overlap_frames=overlap_frames,
    )

    assert len(latest_events(ds, "entry_end")) == 2
    assert widened == ([True, True] if loaded_by_the_task else [])


def test_a_fit_skips_an_entry_without_pair_rows_and_apply_reports_it(
    tmp_path: Path,
) -> None:
    """The fit stream leaves the entry out instead of aborting the run."""
    ds, pairs, individuals = _results(tmp_path, "s1", _EMPTY)
    feature = _FitsEcho(_Echo.Inputs((pairs, individuals)))

    result = run_feature(ds, feature)

    assert feature.fitted_entries == [make_entry_key("g", "s1")]
    assert result.failed_entries == (make_entry_key("g", _EMPTY),)
