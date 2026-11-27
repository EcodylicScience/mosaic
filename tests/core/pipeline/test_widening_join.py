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

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline import entry_task, parquet_reader
from mosaic.core.pipeline.loading import MultiInputAlignmentError, load_entry_data
from mosaic.core.pipeline.manifest import ManifestEntry, iter_manifest
from mosaic.core.pipeline.run import AllEntriesFailed, run_feature
from mosaic.core.pipeline.types import (
    InputStream,
    LoadSpec,
    ParquetLoadSpec,
    Result,
)
from mosaic.core.pipeline.feature_levels import EmitsLevel
from tests.helpers import (
    EMPTY_SEQUENCE,
    NO_INDIVIDUAL_ROWS,
    EchoFeature,
    IndividualRowsFeature,
    PairRowsFeature,
    RunMode,
    add_track_sequences,
    all_logged_events,
    assert_ended_without_an_entry_error,
    assert_lost_the_entry_without_individual_rows,
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
def test_an_entry_without_individual_rows_is_an_entry_error(
    tmp_path: Path, mode: RunMode
) -> None:
    """The entry would hold pair rows without the widened columns, so it is lost."""
    ds, pairs, individuals = _results(tmp_path, "s1", NO_INDIVIDUAL_ROWS)

    result = run_in_mode(
        ds, EchoFeature(EchoFeature.Inputs((pairs, individuals))), mode
    )

    assert_ran_in_mode(ds, mode)
    assert_lost_the_entry_without_individual_rows(ds, result)
    assert set(result_outputs(ds, result)) == {make_entry_key("g", "s1")}
    written = result_outputs(ds, result)[make_entry_key("g", "s1")]
    assert {"speed__id1", "speed__id2"} <= set(written.columns)


@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_an_overlapped_run_records_the_entry_without_individual_rows(
    tmp_path: Path, mode: RunMode
) -> None:
    ds, pairs, individuals = _results(tmp_path, "s1", NO_INDIVIDUAL_ROWS)

    result = run_in_mode(
        ds,
        _EchoWithOverlap(_EchoWithOverlap.Inputs((pairs, individuals))),
        mode,
        overlap_frames=1,
    )

    assert_ran_in_mode(ds, mode)
    assert_lost_the_entry_without_individual_rows(ds, result)


def test_every_entry_without_individual_rows_is_a_failed_run(tmp_path: Path) -> None:
    ds, pairs, individuals = _results(tmp_path, NO_INDIVIDUAL_ROWS)

    with pytest.raises(AllEntriesFailed):
        _ = run_feature(ds, EchoFeature(EchoFeature.Inputs((pairs, individuals))))


def test_a_fit_skips_an_entry_without_individual_rows_and_apply_reports_it(
    tmp_path: Path,
) -> None:
    ds, pairs, individuals = _results(tmp_path, "s1", NO_INDIVIDUAL_ROWS)
    feature = _FitsEcho(EchoFeature.Inputs((pairs, individuals)))

    result = run_feature(ds, feature)

    assert feature.fitted_entries == [make_entry_key("g", "s1")]
    assert result.failed_entries == (make_entry_key("g", NO_INDIVIDUAL_ROWS),)


def test_a_neighbor_without_individual_rows_gives_no_context(tmp_path: Path) -> None:
    """The entry is reported, and its neighbor reads it as an absent neighbor."""
    names = ("s1", "s2", "s3")
    frames = 4
    manifest: dict[str, ManifestEntry] = {}
    specs: dict[str, list[tuple[Path, LoadSpec]]] = {}
    for index, name in enumerate(names):
        start = index * frames
        pairs = tmp_path / f"{name}.pairs.parquet"
        pd.DataFrame(
            {
                "frame": [start + f for f in range(frames) for _ in range(2)],
                "id1": [0, 1] * frames,
                "id2": [1, 0] * frames,
                "dist": 1.0,
            }
        ).to_parquet(pairs)
        individuals = tmp_path / f"{name}.individuals.parquet"
        rows = 0 if name == "s1" else frames
        pd.DataFrame(
            {
                "frame": [start + f for f in range(rows)] * 2,
                "id": [0] * rows + [1] * rows,
                "speed": 1.0,
            }
        ).to_parquet(individuals)
        specs[name] = [(pairs, ParquetLoadSpec()), (individuals, ParquetLoadSpec())]
    for index, name in enumerate(names):
        before = names[index - 1] if index > 0 else None
        after = names[index + 1] if index < len(names) - 1 else None
        manifest[f"g1__{name}"] = ManifestEntry(
            file_specs=specs[name],
            prev_file_specs=None if before is None else specs[before],
            prev_entry_key=None if before is None else f"g1__{before}",
            next_file_specs=None if after is None else specs[after],
            next_entry_key=None if after is None else f"g1__{after}",
            entry_key=f"g1__{name}",
            group="g1",
            sequence=name,
            continuous=True,
            core_extent=(index * frames, (index + 1) * frames - 1),
            prev_extent=None
            if before is None
            else (index * frames - frames, index * frames - 1),
            next_extent=None
            if after is None
            else ((index + 1) * frames, (index + 2) * frames - 1),
        )
    missing: list[str] = []

    loaded = list(
        iter_manifest(
            manifest,
            overlap_frames=2,
            expands_individuals=True,
            on_rows_missing=lambda key, exc: missing.append(
                f"{key}: {type(exc).__name__}"
            ),
        )
    )

    assert missing == ["g1__s1: IndividualRowsMissing"]
    assert [key for key, _, _ in loaded] == ["g1__s2", "g1__s3"]
    _key, df, selector = loaded[0]
    assert not (df["frame"] < selector.first).any()
    assert (df["frame"] > selector.last).any()


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


# --- widening by id where no order column is shared -----------------------------

_BINS: Final = ((0, 10), (10, 20))


def _merged_by_entry(tmp_path: Path, *tables: pd.DataFrame) -> pd.DataFrame | None:
    """What the loader merges of *tables*, written as one entry's inputs, when it widens."""
    specs: list[tuple[Path, LoadSpec]] = []
    for position, table in enumerate(tables):
        path = tmp_path / f"{position}.parquet"
        table.to_parquet(path)
        specs.append((path, ParquetLoadSpec()))
    return load_entry_data(specs, expands_individuals=True)


def _pair_summary() -> pd.DataFrame:
    """Both orderings of the pair of individuals 0 and 1, in each bin and an arena."""
    return pd.DataFrame(
        [
            {
                "id1": first,
                "id2": second,
                "bin_start": start,
                "bin_end": end,
                "arena": "a",
                "dist": float(start + first),
            }
            for start, end in _BINS
            for first, second in ((0, 1), (1, 0))
        ]
    )


def _id_summary() -> pd.DataFrame:
    """One row per individual in each bin and an arena, ``speed`` naming the cell."""
    return pd.DataFrame(
        [
            {
                "id": identity,
                "bin_start": start,
                "bin_end": end,
                "arena": "a",
                "speed": float(10 * start + identity),
            }
            for start, end in _BINS
            for identity in (0, 1)
        ]
    )


def test_a_per_id_summary_widens_onto_pair_rows_by_id(tmp_path: Path) -> None:
    pairs = pd.DataFrame(
        {
            "frame": [0, 0, 1, 1],
            "id1": [0, 1, 0, 1],
            "id2": [1, 0, 1, 0],
            "perspective": [0, 1, 0, 1],
            "dist": 1.0,
        }
    )
    per_id = pd.DataFrame({"id": [0, 1], "speed": [3.0, 5.0]})

    merged = _merged_by_entry(tmp_path, pairs, per_id)

    assert merged is not None
    assert len(merged) == 4
    assert list(merged["speed__id1"]) == [3.0, 5.0, 3.0, 5.0]
    assert list(merged["speed__id2"]) == [5.0, 3.0, 5.0, 3.0]
    assert "id" not in merged.columns


def test_a_per_id_summary_widens_onto_pair_summaries_by_id(tmp_path: Path) -> None:
    merged = _merged_by_entry(
        tmp_path,
        _pair_summary().drop(columns=["bin_start", "bin_end", "arena"]),
        pd.DataFrame({"id": [0, 1], "speed": [3.0, 5.0]}),
    )

    assert merged is not None
    assert list(merged["speed__id1"]) == [3.0, 5.0, 3.0, 5.0]


def test_a_binned_per_id_summary_widens_onto_binned_pair_summaries(
    tmp_path: Path,
) -> None:
    merged = _merged_by_entry(tmp_path, _pair_summary(), _id_summary())

    assert merged is not None
    assert len(merged) == 4
    for slot in ("id1", "id2"):
        expected = 10 * merged["bin_start"] + merged[slot]
        assert list(merged[f"speed__{slot}"]) == list(expected.astype(float))
    assert {"arena", "arena__id1", "arena__id2"} <= set(merged.columns)
    assert {"bin_start", "bin_end", "bin_start__id1", "bin_end__id2"} <= set(
        merged.columns
    )


def test_a_summary_key_that_is_missing_refuses_the_widening(tmp_path: Path) -> None:
    per_id = _id_summary().astype({"bin_start": float})
    per_id.loc[1, "bin_start"] = float("nan")

    with pytest.raises(
        MultiInputAlignmentError,
        match=(
            "cannot merge input 1: bin_start is missing in 1 rows of input 1, "
            "and a summary key names a bin or an arena"
        ),
    ):
        _ = _merged_by_entry(tmp_path, _pair_summary(), per_id)


def test_a_repeated_key_row_refuses_the_widening_by_id(tmp_path: Path) -> None:
    per_id = pd.concat([_id_summary(), _id_summary().iloc[:1]], ignore_index=True)

    with pytest.raises(
        MultiInputAlignmentError,
        match=(
            r"cannot merge input 1: it repeats a \(id, bin_start, bin_end, arena\) "
            "row, so a pair row would match more than one"
        ),
    ):
        _ = _merged_by_entry(tmp_path, _pair_summary(), per_id)


def test_a_repeated_id_refuses_a_widening_that_shares_no_key(tmp_path: Path) -> None:
    per_id = pd.DataFrame({"id": [0, 0, 1], "speed": [1.0, 2.0, 3.0]})

    with pytest.raises(MultiInputAlignmentError, match=r"it repeats a \(id\) row"):
        _ = _merged_by_entry(tmp_path, _pair_summary().drop(columns=["arena"]), per_id)


def test_a_per_id_summary_before_pair_rows_is_refused(tmp_path: Path) -> None:
    with pytest.raises(
        MultiInputAlignmentError, match="needs the pair-level input first"
    ):
        _ = _merged_by_entry(tmp_path, _id_summary(), _pair_summary())
