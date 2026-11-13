"""Tests for the SequenceAggregate feature: one row per entry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.sequence_aggregate import SequenceAggregate
from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import Recipe, plan_pipeline, run_pipeline
from mosaic.core.pipeline.inventory import FeatureRunRef
from mosaic.core.pipeline.preview import preview_artifact
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.run_matrix import load_run_matrix
from mosaic.core.pipeline.types import Result
from mosaic.core.pipeline.types.feature import EmitsLevel
from mosaic.core.pipeline.writers import write_parquet_atomic
from tests.helpers import (
    MIXED,
    STEADY,
    ROI_VALUES,
    TIME_SENTENCE,
    PlainFeature,
    add_track_sequences,
    add_tracks_variant,
    make_dataset,
    record_applied_sequences,
    result_outputs,
    run_over_edited_tracks,
    without_time,
    without_time_in_mixed,
    constant_time_in_mixed,
    run_over_labeled_tracks,
    stepped_individual,
    write_tracktor_recording,
)

VARIANT = "convert-trex.0.2-1111111111"
type Document = dict[str, object]


def _apply(params: dict[str, object], df: pd.DataFrame) -> pd.DataFrame:
    return SequenceAggregate(params=params).apply(df)


def _individual_table(group: str = "G", sequence: str = "S1") -> pd.DataFrame:
    """Three individuals over two frames."""
    values = {0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]}
    return pd.DataFrame(
        [
            {
                "frame": frame,
                "time": frame * 0.02,
                "id": identity,
                "value": value,
                "group": group,
                "sequence": sequence,
            }
            for frame, row in values.items()
            for identity, value in enumerate(row)
        ]
    )


def _pair_table() -> pd.DataFrame:
    """Two pairs over two frames, each pair emitted once per perspective."""
    rows: list[dict[str, float | int | str]] = []
    for frame in (0, 1):
        for perspective in (0, 1):
            for base in (3.0, 5.0):
                rows.append(
                    {
                        "frame": frame,
                        "perspective": perspective,
                        "AB_dist": base + frame,
                        "sequence": "S1",
                        "group": "G",
                    }
                )
    return pd.DataFrame(rows)


def test_one_row_per_entry_holds_group_sequence_and_the_value() -> None:
    out = _apply({"columns": ["value"], "aggs": ["mean"]}, _individual_table())

    assert list(out.columns) == ["group", "sequence", "value_mean"]
    assert len(out) == 1
    assert out["group"].iloc[0] == "G"
    assert out["sequence"].iloc[0] == "S1"
    assert out["value_mean"].iloc[0] == pytest.approx(3.5)


def test_an_entry_with_an_empty_group_keeps_its_row() -> None:
    out = _apply({"columns": ["value"], "aggs": ["max"]}, _individual_table(group=""))

    assert out["group"].tolist() == [""]
    assert out["value_max"].tolist() == [6.0]


def test_each_column_is_reduced_by_each_aggregation() -> None:
    table = _individual_table().assign(other=lambda df: df["value"] * 10)

    out = _apply({"columns": ["value", "other"], "aggs": ["sum", "max"]}, table)

    assert list(out.columns) == [
        "group",
        "sequence",
        "value_sum",
        "value_max",
        "other_sum",
        "other_max",
    ]
    assert out.iloc[0][["value_sum", "value_max"]].tolist() == [21.0, 6.0]
    assert out.iloc[0][["other_sum", "other_max"]].tolist() == [210.0, 60.0]


def test_a_column_listed_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="value more than once"):
        _ = SequenceAggregate(params={"columns": ["value", "value"]})


def test_an_arena_column_gives_one_row_per_arena() -> None:
    table = _individual_table().assign(
        arena=lambda df: np.where(df["id"] < 2, "a", "b")
    )

    out = _apply({"columns": ["value"], "aggs": ["mean"]}, table)

    assert list(out.columns) == ["group", "sequence", "arena", "value_mean"]
    assert out["arena"].tolist() == ["a", "b"]
    assert out["value_mean"].tolist() == [3.0, 4.5]


def test_subgroup_col_splits_an_entry_and_drops_null_values() -> None:
    table = _individual_table().assign(event=[1.0, 1.0, np.nan, 2.0, 2.0, 2.0])

    out = _apply(
        {"columns": ["value"], "aggs": ["count"], "subgroup_col": "event"}, table
    )

    assert list(out.columns) == ["group", "sequence", "event", "value_count"]
    assert out["event"].tolist() == [1.0, 2.0]
    assert out["value_count"].tolist() == [2, 3]


def test_once_halves_the_sum_and_not_the_mean_of_one_step() -> None:
    out = _apply(
        {"columns": ["AB_dist"], "aggs": ["sum", "mean"], "pairs": "once"},
        _pair_table(),
    )

    assert out["AB_dist_sum"].tolist() == [18.0]
    assert out["AB_dist_mean"].iloc[0] == pytest.approx(4.5)


def test_sum_over_pair_input_with_once_counts_each_pair_once() -> None:
    out = _apply(
        {"columns": ["AB_dist"], "aggs": ["sum"], "pairs": "once"}, _pair_table()
    )

    # Perspective 0 only: (3 + 5) at frame 0 and (4 + 6) at frame 1.
    assert out["AB_dist_sum"].tolist() == [18.0]


def test_auto_over_pair_input_reads_every_row() -> None:
    out = _apply({"columns": ["AB_dist"], "aggs": ["sum"]}, _pair_table())

    assert out["AB_dist_sum"].tolist() == [36.0]


def test_count_over_pair_input_with_once_counts_each_pair_once() -> None:
    out = _apply(
        {"columns": ["AB_dist"], "aggs": ["count"], "pairs": "once"}, _pair_table()
    )

    assert out["AB_dist_count"].tolist() == [4]


def test_mean_over_pair_input_reads_every_row() -> None:
    out = _apply(
        {"columns": ["AB_dist"], "aggs": ["mean"], "pairs": "once"}, _pair_table()
    )

    assert out["AB_dist_mean"].iloc[0] == pytest.approx(4.5)


def test_a_filter_that_keeps_nothing_gives_zero_counts_and_nan() -> None:
    out = _apply(
        {
            "columns": ["value"],
            "aggs": ["count", "sum", "mean"],
            "filter_expr": "id > 100",
        },
        _individual_table(),
    )

    assert list(out.columns) == [
        "group",
        "sequence",
        "value_count",
        "value_sum",
        "value_mean",
    ]
    assert out["group"].tolist() == ["G"]
    assert out["sequence"].tolist() == ["S1"]
    assert out["value_count"].tolist() == [0]
    assert out["value_sum"].tolist() == [0]
    assert np.isnan(out["value_mean"].iloc[0])


def test_a_blank_entry_has_the_columns_of_an_entry_that_keeps_rows() -> None:
    params: dict[str, object] = {
        "columns": ["value"],
        "aggs": ["count", "mean"],
        "subgroup_col": "id",
    }
    kept = _apply(params, _individual_table())

    blank = _apply({**params, "filter_expr": "id > 100"}, _individual_table())

    assert list(blank.columns) == list(kept.columns)
    assert blank["id"].tolist() == [0, 1, 2]
    assert blank["value_count"].tolist() == [0, 0, 0]


def test_empty_input_gives_no_output() -> None:
    assert _apply({"columns": ["value"]}, pd.DataFrame()).empty


def test_nan_values_are_skipped() -> None:
    df = _individual_table()
    df.loc[0, "value"] = np.nan

    out = _apply({"columns": ["value"], "aggs": ["count"]}, df)

    assert out["value_count"].tolist() == [5]


# --- keep_columns ----------------------------------------------------------------------


def _kept(**overrides: object) -> dict[str, object]:
    return {
        "columns": ["value"],
        "aggs": ["mean"],
        "keep_columns": ["strain"],
    } | overrides


def test_a_kept_label_is_written_after_the_keys_with_its_value() -> None:
    table = _individual_table().assign(strain="wild")

    out = _apply(_kept(), table)

    assert list(out.columns) == ["group", "sequence", "strain", "value_mean"]
    assert out["strain"].tolist() == ["wild"]


def test_a_kept_label_follows_the_subgroup_key() -> None:
    table = _individual_table().assign(
        strain="wild", event=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    )

    out = _apply(_kept(subgroup_col="event"), table)

    assert list(out.columns) == [
        "group",
        "sequence",
        "event",
        "strain",
        "value_mean",
    ]
    assert out["strain"].tolist() == ["wild", "wild"]


def test_a_kept_label_that_varies_across_subgroups_is_kept_in_each() -> None:
    table = _individual_table().assign(
        strain=["a", "a", "a", "b", "b", "b"], event=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    )

    out = _apply(_kept(subgroup_col="event"), table)

    assert out["strain"].tolist() == ["a", "b"]


def test_a_kept_label_may_vary_among_the_rows_a_filter_drops() -> None:
    table = _individual_table().assign(strain=["a", "a", "a", "b", "a", "a"])

    out = _apply(_kept(filter_expr="frame == 0"), table)

    assert out["strain"].tolist() == ["a"]


def test_an_entry_the_filter_empties_keeps_its_label_from_the_rows_before_it() -> None:
    table = _individual_table().assign(strain="wild")
    kept = _apply(_kept(aggs=["count", "mean"]), table)

    blank = _apply(_kept(aggs=["count", "mean"], filter_expr="id > 100"), table)

    assert list(blank.columns) == list(kept.columns)
    assert blank["strain"].tolist() == ["wild"]
    assert blank["value_count"].tolist() == [0]
    assert np.isnan(blank["value_mean"].iloc[0])


def test_an_entry_the_filter_empties_keeps_each_subgroup_label() -> None:
    table = _individual_table().assign(
        strain=["a", "a", "a", "b", "b", "b"], event=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    )

    blank = _apply(_kept(subgroup_col="event", filter_expr="id > 100"), table)

    assert blank["event"].tolist() == [1.0, 2.0]
    assert blank["strain"].tolist() == ["a", "b"]


def test_a_label_that_varies_before_a_filter_that_empties_the_entry_is_refused() -> (
    None
):
    table = _individual_table().assign(strain=["a", "a", "b", "a", "a", "a"])

    with pytest.raises(ValueError) as refused:
        _ = _apply(_kept(filter_expr="id > 100"), table)

    message = str(refused.value)
    assert "'strain'" in message
    assert "group='G', sequence='S1'" in message


@pytest.mark.parametrize(
    ("strain", "values"),
    [
        (["a", "a", "b", "a", "a", "a"], "'a', 'b'"),
        (["a", "a", None, "a", "a", "a"], "'a', null"),
    ],
    ids=["two values", "a value and a null"],
)
def test_a_kept_label_that_varies_within_a_group_is_refused(
    strain: list[str | None], values: str
) -> None:
    table = _individual_table().assign(strain=pd.Series(strain, dtype=object))

    with pytest.raises(ValueError) as refused:
        _ = _apply(_kept(), table)

    message = str(refused.value)
    assert "'strain'" in message
    assert "group='G', sequence='S1'" in message
    assert values in message


def test_a_refusal_lists_at_most_five_values() -> None:
    table = _individual_table().assign(strain=list("abcdef"))

    with pytest.raises(ValueError) as refused:
        _ = _apply(_kept(), table)

    message = str(refused.value)
    assert "takes 6 values" in message
    assert "'a', 'b', 'c', 'd', 'e', ..." in message
    assert "'f'" not in message


def test_a_kept_label_the_input_lacks_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="'strain'.*not in the input"):
        _ = _apply(_kept(), _individual_table())


def test_a_kept_label_the_input_lacks_is_refused_when_the_filter_drops_every_row() -> (
    None
):
    with pytest.raises(ValueError, match="'strain'.*not in the input"):
        _ = _apply(_kept(filter_expr="id > 100"), _individual_table())


def test_a_nullable_kept_label_is_written_as_it_is() -> None:
    table = _individual_table().assign(
        strain=pd.array([1, 1, 1, 1, 1, 1], dtype="Int64")
    )

    out = _apply(_kept(), table)

    assert out["strain"].tolist() == [1]


def test_an_entry_whose_label_varies_is_lost_and_the_other_is_written(
    tmp_path: Path,
) -> None:
    run = run_over_labeled_tracks(
        tmp_path, SequenceAggregate(params=_kept(columns=["feat_a"]))
    )

    assert run.result.failed_entries == (make_entry_key(*MIXED),)
    assert run.written == [make_entry_key(*STEADY)]
    (error,) = run.errors
    assert error["key"] == make_entry_key(*MIXED)
    assert "'strain'" in error["message"]


def test_a_kept_name_listed_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="strain more than once"):
        _ = SequenceAggregate(params=_kept(keep_columns=["strain", "strain"]))


def test_a_kept_name_that_is_also_aggregated_is_refused() -> None:
    with pytest.raises(ValueError, match="value.*in columns"):
        _ = SequenceAggregate(params=_kept(keep_columns=["value"]))


def test_a_kept_name_equal_to_the_subgroup_column_is_refused() -> None:
    with pytest.raises(ValueError, match="event.*subgroup_col"):
        _ = SequenceAggregate(
            params=_kept(keep_columns=["event"], subgroup_col="event")
        )


@pytest.mark.parametrize(
    "name",
    ["group", "sequence", "frame", "time", "id", "id1", "id2", "perspective", "arena"],
)
def test_a_kept_name_the_reduction_writes_itself_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match=f"{name}.*writes itself"):
        _ = SequenceAggregate(params=_kept(keep_columns=[name]))


# --- the plan and a run ----------------------------------------------------------------


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two schema-valid track tables and nothing computed."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return dataset


def _recipe(steps: list[Document]) -> Recipe:
    return Recipe.model_validate({"steps": steps})


def test_a_recipe_sum_of_ab_dist_is_planned_once(tracked: Dataset) -> None:
    recipe = _recipe(
        [
            {
                "id": "heading",
                "type": "feature",
                "feature": "heading",
                "inputs": ["tracks"],
                "params": {"method": "two_point", "front_idx": 0, "rear_idx": 3},
            },
            {
                "id": "pairs",
                "type": "feature",
                "feature": "pair-position",
                "inputs": ["tracks", {"step": "heading"}],
            },
            {
                "id": "total",
                "type": "feature",
                "feature": "sequence-aggregate",
                "inputs": [{"step": "pairs"}],
                "params": {"columns": ["AB_dist"], "aggs": ["sum"]},
            },
        ]
    )

    plan = plan_pipeline(tracked, recipe)

    total = {planned.step_id: planned for planned in plan.steps}["total"]
    assert total.pairs == "once"
    assert total.spec.params["pairs"] == "once"


def test_a_recipe_of_speed_then_sequence_aggregate_runs(tracked: Dataset) -> None:
    recipe = _recipe(
        [
            {
                "id": "speed",
                "type": "feature",
                "feature": "speed-angvel",
                "inputs": ["tracks"],
            },
            {
                "id": "mean_speed",
                "type": "feature",
                "feature": "sequence-aggregate",
                "inputs": [{"step": "speed"}],
                "params": {"columns": ["speed"], "aggs": ["mean"]},
            },
        ]
    )

    done = run_pipeline(tracked, recipe)

    outcome = {o.step_id: o for o in done.outcomes}["mean_speed"]
    assert outcome.run_id is not None
    storage = "sequence-aggregate__from__speed-angvel__from__tracks"
    tables = {
        sequence: pd.read_parquet(
            preview_artifact(
                tracked, FeatureRunRef(storage, outcome.run_id), ("", sequence)
            ).path
        )
        for sequence in ("seq_a", "seq_b")
    }
    for sequence, table in tables.items():
        assert list(table.columns) == ["group", "sequence", "speed_mean"]
        assert table["sequence"].tolist() == [sequence]
        assert table["speed_mean"].notna().all()


def test_an_entry_the_filter_empties_loads_beside_one_it_keeps(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "filtered")
    add_track_sequences(dataset, ("", "seq_a"), ("", "seq_b"), n_rows=10, n_ids=2)
    aggregate = SequenceAggregate(
        params={
            "columns": ["X"],
            "aggs": ["count", "mean"],
            "filter_expr": "sequence == 'seq_a'",
        }
    )

    result = dataset.run_feature(aggregate)
    matrix = load_run_matrix(dataset, result)

    assert list(matrix.columns) == ["X_count", "X_mean"]
    counts = dict(zip(matrix.index["sequence"], matrix.values[:, 0], strict=True))
    assert counts == {"seq_a": 20.0, "seq_b": 0.0}
    means = dict(zip(matrix.index["sequence"], matrix.values[:, 1], strict=True))
    assert np.isfinite(means["seq_a"])
    assert np.isnan(means["seq_b"])


# --- duration -------------------------------------------------------------------------


def _timed_table(
    frames: list[int], times: list[float], **columns: object
) -> pd.DataFrame:
    """One individual's rows at *frames* and *times*, with the given columns."""
    return pd.DataFrame(
        {
            "frame": frames,
            "time": times,
            "id": 0,
            "group": "G",
            "sequence": "S1",
            **columns,
        }
    )


_SKIPPED = ([0, 1, 3, 4], [0.0, 0.1, 0.3, 0.4])
"""A frame number skipped at a steady 0.1 s per frame."""

_DROPPED = ([0, 1, 2, 3], [0.0, 0.1, 2.1, 2.2])
"""Contiguous frames whose time jumps 2 s after the second: a dropped stretch."""


def _duration(params: dict[str, object], df: pd.DataFrame) -> float:
    out = _apply({"aggs": ["duration"], **params}, df)
    assert len(out) == 1
    return float(out["value_duration"].iloc[0])


def test_duration_of_a_thresholded_column_sums_the_seconds_of_the_rows_above_it() -> (
    None
):
    in_center = pd.array([True, True, False, None], dtype="boolean")
    table = _timed_table(*_SKIPPED, value=in_center)

    seconds = _duration({"columns": ["value"], "threshold": 0.5}, table)

    # Frames 0 and 1 are in the center, 0.1 s each. A null does not count.
    assert seconds == pytest.approx(0.2)


def test_a_dropped_stretch_counts_in_full_for_the_row_before_it() -> None:
    table = _timed_table(*_DROPPED, value=[True, True, False, True])

    seconds = _duration({"columns": ["value"], "threshold": 0.5}, table)

    # Frame 1 lasts until the stretch ends: 0.1 + 2.0 + 0.1.
    assert seconds == pytest.approx(2.2)


def test_duration_without_a_threshold_sums_the_seconds_of_the_non_null_rows() -> None:
    table = _timed_table(*_DROPPED, value=[1.0, np.nan, 3.0, 4.0])

    seconds = _duration({"columns": ["value"]}, table)

    assert seconds == pytest.approx(0.3)


def test_duration_is_a_float_column() -> None:
    out = _apply(
        {"columns": ["value"], "aggs": ["duration"]},
        _timed_table(*_SKIPPED, value=[1.0, 1.0, 1.0, 1.0]),
    )

    assert out["value_duration"].dtype == np.float64


def test_a_filter_leaves_a_kept_row_the_seconds_it_had_before_it() -> None:
    table = _timed_table(*_DROPPED, value=[1.0, 1.0, 1.0, 1.0])

    around = _duration(
        {"columns": ["value"], "filter_expr": "frame != 0 and frame != 2"}, table
    )
    alone = _duration({"columns": ["value"], "filter_expr": "frame == 1"}, table)

    assert around == pytest.approx(2.1)
    assert alone == pytest.approx(2.0)


def test_a_duration_filtered_to_a_condition_reads_the_seconds_where_it_holds() -> None:
    table = _timed_table(*_DROPPED, value=[1.0, 1.0, 1.0, 1.0], in_center=[1, 1, 0, 1])

    seconds = _duration({"columns": ["value"], "filter_expr": "in_center == 1"}, table)

    assert seconds == pytest.approx(2.2)


def _timed_pair_table() -> pd.DataFrame:
    """Pairs 0-1 and 0-2 over two frames at 0.1 s, once per perspective."""
    return pd.DataFrame(
        [
            {
                "frame": frame,
                "time": frame * 0.1,
                "id1": id1,
                "id2": id2,
                "perspective": perspective,
                "value": 1.0,
                "group": "G",
                "sequence": "S1",
            }
            for frame in (0, 1)
            for id1, id2, perspective in [(0, 1, 0), (0, 2, 0), (1, 0, 1), (2, 0, 1)]
        ]
    )


def test_duration_over_pair_input_with_once_counts_each_pairs_time_once() -> None:
    seconds = _duration({"columns": ["value"], "pairs": "once"}, _timed_pair_table())

    # Two pairs, 0.2 s each.
    assert seconds == pytest.approx(0.4)


def test_duration_over_pair_input_with_each_perspective_counts_it_twice() -> None:
    seconds = _duration(
        {"columns": ["value"], "pairs": "each_perspective"}, _timed_pair_table()
    )

    assert seconds == pytest.approx(0.8)


def test_auto_over_pair_input_reads_every_row_for_duration() -> None:
    seconds = _duration({"columns": ["value"]}, _timed_pair_table())

    assert seconds == pytest.approx(0.8)


def test_duration_splits_by_subgroup_after_each_row_has_its_seconds() -> None:
    table = _timed_table(*_DROPPED, value=1.0, tag=["a", "b", "a", "b"])

    out = _apply(
        {"columns": ["value"], "aggs": ["duration"], "subgroup_col": "tag"}, table
    )

    assert out["tag"].tolist() == ["a", "b"]
    assert out["value_duration"].tolist() == pytest.approx([0.2, 2.1])


def test_an_id_reused_in_two_arenas_keeps_each_arenas_seconds_apart() -> None:
    table = pd.concat(
        [
            _timed_table([0, 1, 2], [0.0, 0.1, 0.2], value=1.0, arena="a"),
            _timed_table([0, 1, 2], [0.0, 0.5, 1.0], value=1.0, arena="b"),
        ]
    )

    out = _apply({"columns": ["value"], "aggs": ["duration"]}, table)

    assert out["arena"].tolist() == ["a", "b"]
    assert out["value_duration"].tolist() == pytest.approx([0.3, 1.5])


def test_a_kept_label_is_written_beside_a_duration() -> None:
    table = _timed_table(*_SKIPPED, value=1.0, strain="s")

    out = _apply(_kept(aggs=["duration"]), table)

    assert list(out.columns) == ["group", "sequence", "strain", "value_duration"]
    assert out["value_duration"].tolist() == pytest.approx([0.4])


def test_a_filter_that_keeps_nothing_gives_zero_duration() -> None:
    out = _apply(
        {"columns": ["value"], "aggs": ["duration"], "filter_expr": "id > 100"},
        _timed_table(*_SKIPPED, value=1.0),
    )

    assert out["group"].tolist() == ["G"]
    assert out["value_duration"].tolist() == [0.0]
    assert out["value_duration"].dtype == np.float64


def test_a_mean_alone_reads_no_time() -> None:
    table = _timed_table(*_SKIPPED, value=[1.0, 2.0, 3.0, 4.0]).drop(columns="time")

    out = _apply({"columns": ["value"], "aggs": ["mean"]}, table)

    assert out["value_mean"].tolist() == [2.5]


def test_duration_over_input_without_time_names_the_column_and_what_to_do() -> None:
    table = _timed_table(*_SKIPPED, value=1.0).drop(columns="time")

    with pytest.raises(
        ValueError, match="reads its rate off the 'time' column"
    ) as raised:
        _ = _apply({"columns": ["value"], "aggs": ["duration"]}, table)

    message = str(raised.value)
    assert message.startswith("sequence-aggregate reads its rate off the 'time'")
    assert message.endswith(TIME_SENTENCE)


def test_a_run_whose_time_does_not_advance_loses_the_entry_with_the_sentence(
    tmp_path: Path,
) -> None:
    feature = SequenceAggregate(params={"columns": ["X"], "aggs": ["duration"]})

    run = run_over_edited_tracks(tmp_path, feature, constant_time_in_mixed)

    assert run.result.failed_entries == (make_entry_key(*MIXED),)
    assert run.written == [make_entry_key(*STEADY)]
    (error,) = run.errors
    assert "does not advance" in error["message"]
    assert error["message"].endswith(TIME_SENTENCE)


def test_a_run_over_tracks_without_time_lists_the_entry_as_failed(
    tmp_path: Path,
) -> None:
    feature = SequenceAggregate(params={"columns": ["X"], "aggs": ["duration"]})

    run = run_over_edited_tracks(tmp_path, feature, without_time_in_mixed)

    assert run.result.failed_entries == (make_entry_key(*MIXED),)
    assert run.written == [make_entry_key(*STEADY)]


@pytest.mark.parametrize("agg", ["duration", "share"])
def test_a_run_over_an_input_without_time_fails_the_entry_before_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agg: str
) -> None:
    applied = record_applied_sequences(monkeypatch, SequenceAggregate)
    feature = SequenceAggregate(params={"columns": ["frame"], "aggs": [agg]})

    run = run_over_edited_tracks(tmp_path, feature, without_time_in_mixed)

    assert run.result.failed_entries == (make_entry_key(*MIXED),)
    assert applied == [STEADY[1]]
    (error,) = run.errors
    assert error["error_type"] == "MissingInputColumns"
    assert "'time'" in error["message"]


def test_a_run_of_means_over_tracks_without_time_succeeds(tmp_path: Path) -> None:
    feature = SequenceAggregate(params={"columns": ["X"], "aggs": ["mean"]})

    run = run_over_edited_tracks(tmp_path, feature, without_time)

    assert run.result.failed_entries == ()


# --- share ----------------------------------------------------------------------------


def _share(params: dict[str, object], df: pd.DataFrame) -> pd.DataFrame:
    return _apply({"aggs": ["share"], **params}, df)


def _share_values(out: pd.DataFrame) -> dict[str, float]:
    """The share columns of a one-row result, by name."""
    assert len(out) == 1
    return {name: float(out[name].iloc[0]) for name in out.columns if "_share_" in name}


def test_share_is_each_values_seconds_over_the_seconds_of_the_non_null_rows() -> None:
    out = _share({"columns": ["roi"]}, stepped_individual({"roi": ROI_VALUES}))

    # Four non-null rows of 0.1 s: center twice, none and feeder once.
    assert list(out.columns) == [
        "group",
        "sequence",
        "roi_share_center",
        "roi_share_feeder",
        "roi_share_none",
    ]
    assert _share_values(out) == pytest.approx(
        {"roi_share_center": 0.5, "roi_share_feeder": 0.25, "roi_share_none": 0.25}
    )
    assert sum(_share_values(out).values()) == pytest.approx(1.0)


def test_a_group_of_null_values_only_is_nan_in_each_share_column() -> None:
    table = stepped_individual(
        {"roi": ROI_VALUES + [None, None], "tag": list("aaaaabb")}
    )

    out = _share({"columns": ["roi"], "subgroup_col": "tag"}, table)

    assert out["tag"].tolist() == ["a", "b"]
    assert out["roi_share_center"].iloc[0] == pytest.approx(0.5)
    assert out.drop(columns=["group", "sequence", "tag"]).iloc[1].isna().all()


def test_a_value_the_group_lacks_is_zero() -> None:
    table = stepped_individual({"roi": ["a", "a", "b", "b"], "tag": list("xxyy")})

    out = _share({"columns": ["roi"], "subgroup_col": "tag"}, table)

    assert out["roi_share_a"].tolist() == pytest.approx([1.0, 0.0])
    assert out["roi_share_b"].tolist() == pytest.approx([0.0, 1.0])


def test_a_share_column_is_float64() -> None:
    out = _share({"columns": ["roi"]}, stepped_individual({"roi": ROI_VALUES}))

    assert out["roi_share_center"].dtype == np.float64


@pytest.mark.parametrize("dtype", [object, "string", "str"])
def test_text_gives_the_same_shares_in_any_text_dtype(dtype: str | type) -> None:
    roi = pd.Series(ROI_VALUES, dtype=dtype)
    roi = roi.where(roi.notna(), np.nan) if dtype == "str" else roi

    out = _share({"columns": ["roi"]}, stepped_individual({"roi": roi}))

    assert _share_values(out) == pytest.approx(
        {"roi_share_center": 0.5, "roi_share_feeder": 0.25, "roi_share_none": 0.25}
    )


def test_text_read_back_from_a_parquet_file_gives_the_same_shares(
    tmp_path: Path,
) -> None:
    path = tmp_path / "roi.parquet"
    _ = write_parquet_atomic(stepped_individual({"roi": ROI_VALUES}), path)

    out = _share({"columns": ["roi"]}, pd.read_parquet(path))

    assert _share_values(out) == pytest.approx(
        {"roi_share_center": 0.5, "roi_share_feeder": 0.25, "roi_share_none": 0.25}
    )


def test_a_categorical_column_contributes_the_values_that_occur() -> None:
    roi = pd.Categorical(
        ["center", "none", None, "center"], categories=["nest", "none", "center"]
    )

    out = _share({"columns": ["roi"]}, stepped_individual({"roi": roi}))

    assert _share_values(out) == pytest.approx(
        {"roi_share_center": 2 / 3, "roi_share_none": 1 / 3}
    )


@pytest.mark.parametrize("dtype", ["boolean", "bool"])
def test_a_boolean_column_writes_false_before_true(dtype: str) -> None:
    in_center = pd.Series([True, False, True, True], dtype=dtype)

    out = _share(
        {"columns": ["in_center"]}, stepped_individual({"in_center": in_center})
    )

    assert list(out.columns)[2:] == ["in_center_share_false", "in_center_share_true"]
    assert out["in_center_share_false"].tolist() == pytest.approx([0.25])
    assert out["in_center_share_true"].tolist() == pytest.approx([0.75])


def test_a_nullable_boolean_column_leaves_its_nulls_out_of_both_sums() -> None:
    in_center = pd.array([True, False, None, True, True], dtype="boolean")

    out = _share(
        {"columns": ["in_center"]}, stepped_individual({"in_center": in_center})
    )

    assert _share_values(out) == pytest.approx(
        {"in_center_share_false": 0.25, "in_center_share_true": 0.75}
    )


@pytest.mark.parametrize("dtype", ["int64", "Int64", "uint8"])
def test_an_integer_column_writes_decimal_names_in_numeric_order(dtype: str) -> None:
    values = [3, 10, 3, 1] if dtype == "uint8" else [3, 10, 3, -1]
    smallest = "1" if dtype == "uint8" else "-1"
    table = stepped_individual({"x": pd.Series(values, dtype=dtype)})

    out = _share({"columns": ["x"]}, table)

    assert list(out.columns)[2:] == [f"x_share_{smallest}", "x_share_3", "x_share_10"]
    assert out["x_share_3"].tolist() == pytest.approx([0.5])


def test_categories_pin_the_columns_in_their_order_and_write_zero_for_a_value_absent() -> (
    None
):
    out = _share(
        {"columns": ["roi"], "categories": ["feeder", "center", "none", "nest"]},
        stepped_individual({"roi": ROI_VALUES}),
    )

    assert list(out.columns)[2:] == [
        "roi_share_feeder",
        "roi_share_center",
        "roi_share_none",
        "roi_share_nest",
    ]
    assert out["roi_share_nest"].tolist() == [0.0]
    assert out["roi_share_center"].tolist() == pytest.approx([0.5])


def test_a_value_outside_the_categories_fails_the_entry_by_name() -> None:
    with pytest.raises(ValueError, match="'feeder'") as raised:
        _ = _share(
            {"columns": ["roi"], "categories": ["center", "none"]},
            stepped_individual({"roi": ROI_VALUES}),
        )

    assert "center" in str(raised.value)
    assert "none" in str(raised.value)


def test_a_value_outside_the_categories_in_a_row_the_filter_drops_is_ignored() -> None:
    out = _share(
        {
            "columns": ["roi"],
            "categories": ["center", "none"],
            "filter_expr": "roi != 'feeder'",
        },
        stepped_individual({"roi": ROI_VALUES}),
    )

    assert _share_values(out) == pytest.approx(
        {"roi_share_center": 2 / 3, "roi_share_none": 1 / 3}
    )


def test_a_value_of_a_row_outside_every_group_names_no_column() -> None:
    table = stepped_individual({"roi": ["a", "b"], "tag": ["x", None]})

    out = _share({"columns": ["roi"], "subgroup_col": "tag"}, table)

    assert list(out.columns) == ["group", "sequence", "tag", "roi_share_a"]


def test_unset_categories_let_two_entries_write_different_sorted_columns() -> None:
    first = _share({"columns": ["roi"]}, stepped_individual({"roi": ["b", "a"]}))
    second = _share({"columns": ["roi"]}, stepped_individual({"roi": ["d", "c", "b"]}))

    assert list(first.columns)[2:] == ["roi_share_a", "roi_share_b"]
    assert list(second.columns)[2:] == ["roi_share_b", "roi_share_c", "roi_share_d"]


def test_each_column_expands_in_place_of_share_in_column_major_order() -> None:
    table = stepped_individual({"a": ["u", "v"], "b": ["x", "x"]})

    out = _apply({"columns": ["a", "b"], "aggs": ["count", "share"]}, table)

    assert list(out.columns)[2:] == [
        "a_count",
        "a_share_u",
        "a_share_v",
        "b_count",
        "b_share_x",
    ]


def test_a_float_column_fails_the_entry_naming_the_column_and_its_dtype() -> None:
    with pytest.raises(ValueError, match="'value'") as raised:
        _ = _share({"columns": ["value"]}, stepped_individual({"value": [1.0, 2.0]}))

    assert "float64" in str(raised.value)


def test_an_object_column_of_text_and_integers_fails_the_entry_naming_its_kinds() -> (
    None
):
    with pytest.raises(ValueError, match="'roi'") as raised:
        _ = _share(
            {"columns": ["roi"]},
            stepped_individual({"roi": pd.Series(["a", 1, "b"], dtype=object)}),
        )

    assert "int" in str(raised.value)
    assert "str" in str(raised.value)


@pytest.mark.parametrize("value", ["a b", ""])
def test_a_value_outside_the_safe_characters_fails_the_entry_and_points_upstream(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="recode") as raised:
        _ = _share({"columns": ["roi"]}, stepped_individual({"roi": ["a", value]}))

    assert repr(value) in str(raised.value)


def test_a_share_name_equal_to_another_output_name_fails_the_entry_naming_both() -> (
    None
):
    table = stepped_individual({"a": ["count", "x"], "a_share": [1, 2]})

    with pytest.raises(ValueError, match="'a_share_count'") as raised:
        _ = _apply({"columns": ["a", "a_share"], "aggs": ["share", "count"]}, table)

    assert "share of column 'a'" in str(raised.value)
    assert "count of column 'a_share'" in str(raised.value)


def test_a_seconds_weighted_share_differs_from_the_share_of_rows() -> None:
    table = _timed_table(*_DROPPED, roi=["a", "b", "b", "b"])

    out = _share({"columns": ["roi"]}, table)

    # Row 0 lasts 0.1 s and row 1 lasts 2.0 s, which a count of rows would not say.
    assert out["roi_share_a"].tolist() == pytest.approx([0.1 / 2.3])
    assert out["roi_share_b"].tolist() == pytest.approx([2.2 / 2.3])


def test_a_filter_leaves_a_kept_row_its_seconds_in_a_share() -> None:
    table = _timed_table(*_DROPPED, roi=["a", "b", "a", "b"])

    out = _share({"columns": ["roi"], "filter_expr": "frame != 0"}, table)

    # Frame 1 lasts 2.0 s and frame 2 lasts 0.1 s, as before the filter.
    assert out["roi_share_a"].tolist() == pytest.approx([0.1 / 2.2])
    assert out["roi_share_b"].tolist() == pytest.approx([2.1 / 2.2])


def test_share_splits_by_subgroup_after_each_row_has_its_seconds() -> None:
    table = _timed_table(*_DROPPED, roi=["a", "a", "a", "b"], tag=["x", "x", "y", "y"])

    out = _share({"columns": ["roi"], "subgroup_col": "tag"}, table)

    assert out["roi_share_a"].tolist() == pytest.approx([1.0, 0.1 / 0.2])
    assert out["roi_share_b"].tolist() == pytest.approx([0.0, 0.1 / 0.2])


def test_an_id_reused_in_two_arenas_keeps_each_arenas_share_apart() -> None:
    table = pd.concat(
        [
            _timed_table([0, 1, 2], [0.0, 0.1, 0.2], roi=["a", "a", "b"], arena="p"),
            _timed_table([0, 1, 2], [0.0, 0.5, 1.0], roi=["a", "b", "b"], arena="q"),
        ]
    )

    out = _share({"columns": ["roi"]}, table)

    assert out["arena"].tolist() == ["p", "q"]
    assert out["roi_share_a"].tolist() == pytest.approx([2 / 3, 0.5 / 1.5])


def test_a_kept_label_is_written_beside_the_share_columns() -> None:
    table = stepped_individual({"value": ["a", "b"], "strain": "s"})

    out = _share(_kept(aggs=["share"]), table)

    assert list(out.columns) == [
        "group",
        "sequence",
        "strain",
        "value_share_a",
        "value_share_b",
    ]


def test_share_reads_every_row_of_a_pair_table_under_either_pairs_rule() -> None:
    table = _timed_pair_table().assign(
        region=lambda df: np.where(df[["id1", "id2"]].eq(1).any(axis=1), "near", "far")
    )

    once = _share({"columns": ["region"], "pairs": "once"}, table)
    each = _share({"columns": ["region"], "pairs": "each_perspective"}, table)

    pd.testing.assert_frame_equal(once, each)
    assert each["region_share_near"].tolist() == pytest.approx([0.5])


def test_a_filter_that_keeps_nothing_gives_nan_for_each_category() -> None:
    kept = _share(
        {"columns": ["roi"], "categories": ["a", "b"]},
        stepped_individual({"roi": ["a", "b"]}),
    )

    blank = _share(
        {"columns": ["roi"], "categories": ["a", "b"], "filter_expr": "id > 100"},
        stepped_individual({"roi": ["a", "b"]}),
    )

    assert list(blank.columns) == list(kept.columns)
    assert blank["roi_share_a"].isna().all()
    assert blank["roi_share_b"].isna().all()
    assert blank["roi_share_a"].dtype == np.float64


def test_a_filter_that_keeps_nothing_writes_no_share_column_without_categories() -> (
    None
):
    out = _apply(
        {"columns": ["roi"], "aggs": ["count", "share"], "filter_expr": "id > 100"},
        stepped_individual({"roi": ["a", "b"]}),
    )

    assert list(out.columns) == ["group", "sequence", "roi_count"]


def test_share_over_input_without_time_names_the_column_and_what_to_do() -> None:
    table = stepped_individual({"roi": ["a", "b"]}).drop(columns="time")

    with pytest.raises(ValueError, match="reads its rate off the 'time'") as raised:
        _ = _share({"columns": ["roi"]}, table)

    assert str(raised.value).endswith(TIME_SENTENCE)


def test_a_run_loses_the_entry_whose_value_is_outside_the_categories(
    tmp_path: Path,
) -> None:
    feature = SequenceAggregate(
        params={"columns": ["strain"], "aggs": ["share"], "categories": ["a"]}
    )

    run = run_over_labeled_tracks(tmp_path, feature)

    assert run.result.failed_entries == (make_entry_key(*MIXED),)
    assert run.written == [make_entry_key(*STEADY)]
    (error,) = run.errors
    assert "'b'" in error["message"]


# --- share and categories: validation ---------------------------------------------------


def _share_params(**overrides: object) -> dict[str, object]:
    return {"columns": ["roi"], "aggs": ["share"], **overrides}


def test_share_with_a_threshold_is_refused() -> None:
    with pytest.raises(ValueError, match="share.*threshold"):
        _ = SequenceAggregate(params=_share_params(threshold=0.5))


def test_share_with_the_abs_transform_is_refused() -> None:
    with pytest.raises(ValueError, match="share.*abs"):
        _ = SequenceAggregate(params=_share_params(transform="abs"))


def test_categories_without_share_are_refused() -> None:
    with pytest.raises(ValueError, match="categories.*share"):
        _ = SequenceAggregate(
            params={"columns": ["roi"], "aggs": ["mean"], "categories": ["a"]}
        )


@pytest.mark.parametrize(
    ("categories", "match"),
    [
        ([], "at least 1"),
        (["a", "b", "a"], "lists a more than once"),
        (["a b"], "'a b'"),
        ([""], "''"),
        (["caf\u00e9"], "caf"),
    ],
)
def test_categories_that_are_empty_repeated_or_unsafe_are_refused(
    categories: list[str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _ = SequenceAggregate(params=_share_params(categories=categories))


def test_share_with_the_none_transform_and_categories_validates() -> None:
    feature = SequenceAggregate(
        params=_share_params(transform="none", categories=["a", "none"])
    )

    assert feature.params.categories == ["a", "none"]


def test_unset_categories_are_none_and_leave_the_identity_dump() -> None:
    params = SequenceAggregate(params=_share_params()).params

    assert params.categories is None
    assert "categories" not in params.identity_dump()


def test_categories_enter_the_identity_dump_when_set() -> None:
    params = SequenceAggregate(params=_share_params(categories=["a"])).params

    assert params.identity_dump()["categories"] == ["a"]


# --- labels of the tracks table ------------------------------------------------------

_TRACKTOR_RECORDINGS = {
    "20211117111002_900s_no12-DGRP101-female-2to4-6-1": ("DGRP101", "female"),
    "20211117121500_900s_no07-CantonS-male-2to4-6-1": ("CantonS", "male"),
}
"""Each recording's stem, with the ``strain`` and ``sex`` its name carries."""


def _tracktor_dataset(tmp_path: Path) -> Dataset:
    """A dataset of the two recordings, converted to ``mosaic_v1`` tracks."""
    dataset = make_dataset((tmp_path / "ds").resolve())
    for stem in _TRACKTOR_RECORDINGS:
        _ = write_tracktor_recording(dataset.base_dir / "raw", stem=stem)
    _ = dataset.index_tracks_raw(
        [dataset.base_dir / "raw"],
        patterns=["*_tracked_fixed_final.tsv"],
        src_format="tracktor_corrected_tsv",
    )
    _ = dataset.convert_all_tracks()
    return dataset


def test_a_tracktor_label_reaches_each_entry_through_the_tracks_input(
    tmp_path: Path,
) -> None:
    dataset = _tracktor_dataset(tmp_path)
    speed = run_feature(dataset, SpeedAngvel())
    feature = SequenceAggregate(
        inputs=SequenceAggregate.Inputs(
            (Result(feature=speed.feature, run_id=str(speed.run_id)), "tracks")
        ),
        params={
            "columns": ["speed"],
            "aggs": ["mean"],
            "keep_columns": ["strain", "sex"],
        },
    )

    result = run_feature(dataset, feature)

    assert result.failed_entries == ()
    outputs = result_outputs(dataset, result)
    assert len(outputs) == len(_TRACKTOR_RECORDINGS)
    for stem, (strain, sex) in _TRACKTOR_RECORDINGS.items():
        row = outputs[make_entry_key("", stem)]
        assert row.columns.tolist() == [
            "group",
            "sequence",
            "strain",
            "sex",
            "speed_mean",
        ]
        assert row["strain"].tolist() == [strain]
        assert row["sex"].tolist() == [sex]


class _ShiftedPositions(PlainFeature):
    """Writes ``X`` with values that differ from the tracks' ``X``."""

    name = "shifted-positions"
    accepts_overlap = False
    emits: EmitsLevel = "as-input"
    shift = 1000.0

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "frame": df["frame"],
                "id": df["id"],
                "X": df["X"] + self.shift,
                "group": df["group"],
                "sequence": df["sequence"],
            }
        )


def test_a_payload_column_the_feature_shares_with_the_tracks_is_read_from_the_feature(
    tmp_path: Path,
) -> None:
    """The tracks come second, so their ``X`` arrives as ``X__1``."""
    dataset = _tracktor_dataset(tmp_path)
    shifted = run_feature(dataset, _ShiftedPositions())
    inputs = SequenceAggregate.Inputs(
        (Result(feature=shifted.feature, run_id=str(shifted.run_id)), "tracks")
    )

    def mean_of(column: str) -> dict[str, float]:
        params: dict[str, object] = {
            "columns": [column],
            "aggs": ["mean"],
            "keep_columns": ["strain"],
        }
        result = run_feature(dataset, SequenceAggregate(inputs=inputs, params=params))
        assert result.failed_entries == ()
        return {
            stem: float(
                result_outputs(dataset, result)[make_entry_key("", stem)][
                    f"{column}_mean"
                ].iloc[0]
            )
            for stem in _TRACKTOR_RECORDINGS
        }

    own = mean_of("X")
    tracks = mean_of("X__1")

    for stem in _TRACKTOR_RECORDINGS:
        tracks_x = dataset.load_tracks("", stem)["X"].mean()
        assert tracks[stem] == pytest.approx(tracks_x)
        assert own[stem] == pytest.approx(tracks_x + _ShiftedPositions.shift)
