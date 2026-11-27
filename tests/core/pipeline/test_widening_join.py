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

from pathlib import Path
from typing import Final

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline import entry_task, parquet_reader
from mosaic.core.pipeline.loading import MultiInputAlignmentError
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import (
    InputStream,
    Result,
)
from mosaic.core.pipeline.feature_levels import EmitsLevel
from tests.helpers import (
    EMPTY_SEQUENCE,
    EchoFeature,
    IndividualRowsFeature,
    PairRowsFeature,
    RunMode,
    add_track_sequences,
    all_logged_events,
    assert_ended_without_an_entry_error,
    assert_lost_the_entry_without_pair_rows,
    assert_ran_in_mode,
    corrupt_output,
    latest_events,
    make_dataset,
    observed,
    record_path_calls,
    result_outputs,
    run_in_mode,
    run_modes,
)

_MODES: Final = run_modes(2)


class _FitsEcho(EchoFeature):
    """An ``as-input`` consumer that fits by reading every entry of its stream."""

    name = "test-fits-echo"

    def __init__(
        self, inputs: EchoFeature.Inputs, params: dict[str, object] | None = None
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


class _EchoWithOverlap(EchoFeature):
    """An ``as-input`` consumer that accepts overlap."""

    name = "test-echo-overlap"
    accepts_overlap = True


class _EchoPerIndividual(EchoFeature):
    name = "test-echo-individual"
    emits: EmitsLevel = "individual"


def _results(tmp_path: Path, *sequences: str) -> tuple[Dataset, Result, Result]:
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, *(("g", s) for s in sequences), n_rows=6, n_ids=2)
    pairs = run_feature(ds, PairRowsFeature())
    individuals = run_feature(ds, IndividualRowsFeature())
    return (
        ds,
        Result(feature=pairs.feature, run_id=pairs.run_id),
        Result(feature=individuals.feature, run_id=individuals.run_id),
    )


@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_an_as_input_run_reads_each_pair_row_with_both_individuals(
    tmp_path: Path, mode: RunMode
) -> None:
    ds, pairs, individuals = _results(tmp_path, "s1")

    echoed = run_in_mode(
        ds, EchoFeature(EchoFeature.Inputs((pairs, individuals))), mode
    )

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
    ds, pairs, individuals = _results(tmp_path, "s1", EMPTY_SEQUENCE)

    result = run_in_mode(
        ds, EchoFeature(EchoFeature.Inputs((pairs, individuals))), mode
    )

    assert_ran_in_mode(ds, mode)
    assert_lost_the_entry_without_pair_rows(ds, result)
    assert set(result_outputs(ds, result)) == {make_entry_key("g", "s1")}
    assert [end["key"] for end in latest_events(ds, "entry_end")] == [
        make_entry_key("g", "s1")
    ]


def test_an_entry_without_pair_rows_leaves_the_gate(tmp_path: Path) -> None:
    """The lost entry leaves the in-flight record before the next one enters."""
    ds, pairs, individuals = _results(tmp_path, EMPTY_SEQUENCE, "s1", "s2")
    execution_id = "TESTWIDENGATE"

    result = run_feature(
        ds,
        EchoFeature(EchoFeature.Inputs((pairs, individuals))),
        execution_id=execution_id,
    )

    assert result.failed_entries == (make_entry_key("g", EMPTY_SEQUENCE),)
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
    ds, pairs, individuals = _results(tmp_path, "s1", EMPTY_SEQUENCE)

    result = run_in_mode(
        ds,
        _EchoWithOverlap(_EchoWithOverlap.Inputs((pairs, individuals))),
        mode,
        overlap_frames=1,
    )

    assert_ran_in_mode(ds, mode)
    assert_lost_the_entry_without_pair_rows(ds, result)


@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_a_corrupt_input_of_a_widening_run_ends_the_run(
    tmp_path: Path, mode: RunMode
) -> None:
    ds, pairs, individuals = _results(tmp_path, "s1", "s2")
    corrupt_output(ds, individuals, "s2")

    with pytest.raises(ValueError, match="magic bytes not found") as raised:
        _ = run_in_mode(ds, EchoFeature(EchoFeature.Inputs((pairs, individuals))), mode)

    assert_ran_in_mode(ds, mode)
    assert_ended_without_an_entry_error(ds, raised.value)


@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_a_time_window_over_a_widening_input_without_time_ends_the_run(
    tmp_path: Path, mode: RunMode
) -> None:
    """The stub pair rows of ``s1`` and ``s2`` carry no ``time``, so the time filter raises."""
    ds, pairs, individuals = _results(tmp_path, "s1", "s2")

    with pytest.raises(
        ValueError, match="a time filter reads the 'time' column"
    ) as raised:
        _ = run_in_mode(
            ds,
            EchoFeature(EchoFeature.Inputs((pairs, individuals))),
            mode,
            filter_start_time=0.0,
        )

    assert_ran_in_mode(ds, mode)
    assert_ended_without_an_entry_error(ds, raised.value)


def test_a_widening_run_parses_each_input_footer_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, pairs, individuals = _results(tmp_path, EMPTY_SEQUENCE, "s1")
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")

    result = run_feature(
        ds,
        EchoFeature(EchoFeature.Inputs((pairs, individuals))),
        parallel_workers=2,
        parallel_mode="thread",
    )

    assert result.failed_entries == (make_entry_key("g", EMPTY_SEQUENCE),)
    assert parsed
    assert len(parsed) == len(set(parsed)), sorted(parsed)


@pytest.mark.parametrize(
    ("feature_class", "overlap_frames", "loaded_by_the_task"),
    [(EchoFeature, 0, True), (_EchoWithOverlap, 1, False)],
    ids=["widening", "overlap"],
)
def test_a_widening_entry_is_loaded_by_its_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feature_class: type[EchoFeature],
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
    ds, pairs, individuals = _results(tmp_path, "s1", EMPTY_SEQUENCE)
    feature = _FitsEcho(EchoFeature.Inputs((pairs, individuals)))

    result = run_feature(ds, feature)

    assert feature.fitted_entries == [make_entry_key("g", "s1")]
    assert result.failed_entries == (make_entry_key("g", EMPTY_SEQUENCE),)
