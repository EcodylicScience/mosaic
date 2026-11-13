"""Tests for the IndividualAggregate feature: one row per individual."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.individual_aggregate import IndividualAggregate
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import Recipe, plan_pipeline
from mosaic.core.pipeline.run import OverlapUnsupported, run_feature
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.core.helpers import make_entry_key
from tests.helpers import (
    MIXED,
    STEADY,
    ROI_VALUES,
    TIME_SENTENCE,
    add_track_sequences,
    add_tracks_variant,
    make_dataset,
    record_applied_sequences,
    run_over_edited_tracks,
    without_time,
    without_time_in_mixed,
    constant_time_in_mixed,
    run_over_labeled_tracks,
    stepped_individual,
)


def _apply(params: dict[str, object], df: pd.DataFrame) -> pd.DataFrame:
    return IndividualAggregate(params=params).apply(df)


def _individual_table() -> pd.DataFrame:
    """Three individuals over two frames."""
    values = {0: [1.0, 2.0, 3.0], 1: [4.0, 6.0, 8.0]}
    return pd.DataFrame(
        [
            {
                "frame": frame,
                "id": identity,
                "value": value,
                "group": "G",
                "sequence": "S1",
            }
            for frame, row in values.items()
            for identity, value in enumerate(row)
        ]
    )


def _pair_table() -> pd.DataFrame:
    """Three individuals, each ordered pair once, on one frame.

    ``A_speed`` is the focal's own, so the two rows of a pair differ.
    """
    speed = {
        (0, 1): 1.0,
        (1, 0): 10.0,
        (0, 2): 2.0,
        (2, 0): 20.0,
        (1, 2): 3.0,
        (2, 1): 30.0,
    }
    return pd.DataFrame(
        [
            {
                "frame": 0,
                "id1": id1,
                "id2": id2,
                "perspective": int(id1 > id2),
                "A_speed": value,
                "group": "G",
                "sequence": "S1",
            }
            for (id1, id2), value in speed.items()
        ]
    )


def test_one_row_per_id_holds_group_sequence_id_and_the_value() -> None:
    out = _apply({"columns": ["value"], "aggs": ["mean"]}, _individual_table())

    assert list(out.columns) == ["group", "sequence", "id", "value_mean"]
    assert out["id"].tolist() == [0, 1, 2]
    assert out["value_mean"].tolist() == [2.5, 4.0, 5.5]
    assert set(out["group"]) == {"G"}
    assert set(out["sequence"]) == {"S1"}


def test_an_arena_column_keys_each_individual_by_its_arena() -> None:
    table = _individual_table().assign(
        arena=lambda df: np.where(df["id"] < 2, "a", "b")
    )

    out = _apply({"columns": ["value"], "aggs": ["mean", "max"]}, table)

    assert list(out.columns) == [
        "group",
        "sequence",
        "id",
        "arena",
        "value_mean",
        "value_max",
    ]
    assert out["arena"].tolist() == ["a", "a", "b"]
    assert out["value_max"].tolist() == [4.0, 6.0, 8.0]


def test_subgroup_col_splits_an_individuals_rows() -> None:
    table = _individual_table().assign(phase=["x", "x", "x", "y", "y", "y"])

    out = _apply({"columns": ["value"], "subgroup_col": "phase"}, table)

    assert out[["id", "phase"]].values.tolist() == [
        [0, "x"],
        [0, "y"],
        [1, "x"],
        [1, "y"],
        [2, "x"],
        [2, "y"],
    ]
    assert out["value_mean"].tolist() == [1.0, 4.0, 2.0, 6.0, 3.0, 8.0]


def test_sum_over_individual_input_adds_each_frame() -> None:
    out = _apply({"columns": ["value"], "aggs": ["sum"]}, _individual_table())

    assert out["value_sum"].tolist() == [5.0, 8.0, 11.0]


def test_pair_input_counts_each_ordered_pair_once_per_focal() -> None:
    out = _apply({"columns": ["A_speed"], "aggs": ["sum"]}, _pair_table())

    # Individual 2 is the focal of two rows only because the higher-id focal's
    # rows count: perspective 0 alone would give it nothing.
    assert out["id"].tolist() == [0, 1, 2]
    assert out["A_speed_sum"].tolist() == [3.0, 13.0, 50.0]


def test_pair_input_count_is_the_number_of_partners() -> None:
    out = _apply({"columns": ["A_speed"], "aggs": ["count"]}, _pair_table())

    assert out["A_speed_count"].tolist() == [2, 2, 2]


def test_pair_input_mean_reads_every_focal_row() -> None:
    out = _apply({"columns": ["A_speed"], "aggs": ["mean"]}, _pair_table())

    assert out["A_speed_mean"].tolist() == [1.5, 6.5, 25.0]


@pytest.mark.parametrize("dtype", ["boolean", "Int64"])
def test_a_nullable_column_skips_its_nulls_in_every_aggregation(
    dtype: Literal["boolean", "Int64"],
) -> None:
    table = pd.DataFrame(
        {"frame": [0, 1, 2, 0, 1], "id": [0, 0, 0, 1, 1]}, dtype="int64"
    ).assign(group="G", sequence="S1")
    table["value"] = (
        pd.array([True, False, None, True, True], dtype="boolean")
        if dtype == "boolean"
        else pd.array([1, 0, None, 1, 1], dtype="Int64")
    )

    out = _apply(
        {"columns": ["value"], "aggs": ["sum", "std", "max", "median", "mean"]}, table
    )

    assert out["value_sum"].tolist() == [1.0, 2.0]
    assert out["value_std"].tolist() == pytest.approx([2**-0.5, 0.0])
    assert out["value_max"].tolist() == [1.0, 1.0]
    assert out["value_median"].tolist() == [0.5, 1.0]
    assert out["value_mean"].tolist() == [0.5, 1.0]
    for agg in ("sum", "std", "max", "median", "mean"):
        assert out[f"value_{agg}"].dtype == np.float64


def test_the_params_have_no_pairs_field() -> None:
    assert "pairs" not in IndividualAggregate.Params.model_fields
    with pytest.raises(ValueError, match="pairs"):
        _ = IndividualAggregate(params={"columns": ["value"], "pairs": "once"})


def test_empty_input_gives_no_output() -> None:
    assert _apply({"columns": ["value"]}, pd.DataFrame()).empty


def test_overlap_is_refused(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds")
    add_track_sequences(dataset, ("g", "s1"), ("g", "s2"), n_rows=10, n_ids=2)
    feature = IndividualAggregate(params={"columns": ["X"], "aggs": ["mean"]})

    with pytest.raises(OverlapUnsupported, match="accepts_overlap = False"):
        _ = run_feature(dataset, feature, overlap_frames=3)


def test_a_filter_names_the_focal_of_pair_input_as_id1() -> None:
    out = _apply(
        {"columns": ["A_speed"], "aggs": ["sum"], "filter_expr": "id1 < 2"},
        _pair_table(),
    )

    assert out["id"].tolist() == [0, 1]
    assert out["A_speed_sum"].tolist() == [3.0, 13.0]


def test_a_recipe_reduces_pair_rows_to_individuals(tmp_path: Path) -> None:
    """The plan takes pair input, and leaves no pair rule to resolve."""
    dataset: Dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(
        dataset, "convert-trex.0.2-1111111111", "seq_a", std_format="mosaic_v1"
    )
    recipe = Recipe.model_validate(
        {
            "steps": [
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
                    "id": "per_focal",
                    "type": "feature",
                    "feature": "individual-aggregate",
                    "inputs": [{"step": "pairs"}],
                    "params": {"columns": ["A_speed"], "aggs": ["sum"]},
                },
            ]
        }
    )

    plan = plan_pipeline(dataset, recipe)

    per_focal = {planned.step_id: planned for planned in plan.steps}["per_focal"]
    assert per_focal.pairs is None
    assert "pairs" not in per_focal.spec.params


def _kept(**overrides: object) -> dict[str, object]:
    return {
        "columns": ["value"],
        "aggs": ["mean"],
        "keep_columns": ["strain"],
    } | overrides


def test_a_kept_label_is_written_after_the_id_with_its_value() -> None:
    table = _individual_table().assign(strain=["a", "b", "c", "a", "b", "c"])

    out = _apply(_kept(), table)

    assert list(out.columns) == ["group", "sequence", "id", "strain", "value_mean"]
    assert out["strain"].tolist() == ["a", "b", "c"]


@pytest.mark.parametrize(
    ("strain", "values"),
    [
        (["a", "a", "a", "b", "a", "a"], "'a', 'b'"),
        (["a", "a", "a", None, "a", "a"], "'a', null"),
    ],
    ids=["two values", "a value and a null"],
)
def test_a_kept_label_that_varies_within_an_individual_is_refused(
    strain: list[str | None], values: str
) -> None:
    table = _individual_table().assign(strain=pd.Series(strain, dtype=object))

    with pytest.raises(ValueError) as refused:
        _ = _apply(_kept(), table)

    message = str(refused.value)
    assert "'strain'" in message
    assert "id=0" in message
    assert values in message


def test_a_kept_label_the_input_lacks_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="'strain'.*not in the input"):
        _ = _apply(_kept(), _individual_table())


def test_an_entry_whose_label_varies_is_lost_and_the_other_is_written(
    tmp_path: Path,
) -> None:
    run = run_over_labeled_tracks(
        tmp_path, IndividualAggregate(params=_kept(columns=["feat_a"]))
    )

    assert run.result.failed_entries == (make_entry_key(*MIXED),)
    assert run.written == [make_entry_key(*STEADY)]
    (error,) = run.errors
    assert error["key"] == make_entry_key(*MIXED)
    assert "'strain'" in error["message"]


# --- duration -------------------------------------------------------------------------


def _timed_individuals() -> pd.DataFrame:
    """Two individuals over frames 0 to 3, whose time jumps 2 s after frame 1.

    Individual 1 has no value at frame 3.
    """
    times = {0: 0.0, 1: 0.1, 2: 2.1, 3: 2.2}
    return pd.DataFrame(
        [
            {
                "frame": frame,
                "time": time,
                "id": identity,
                "value": np.nan if (identity == 1 and frame == 3) else 1.0,
                "tag": "a" if frame % 2 == 0 else "b",
                "group": "G",
                "sequence": "S1",
            }
            for frame, time in times.items()
            for identity in (0, 1)
        ]
    )


def _timed_pair_table() -> pd.DataFrame:
    """Three individuals, each ordered pair once, over two frames at 0.1 s."""
    return pd.DataFrame(
        [
            {
                "frame": frame,
                "time": frame * 0.1,
                "id1": id1,
                "id2": id2,
                "perspective": int(id1 > id2),
                "A_speed": 1.0,
                "group": "G",
                "sequence": "S1",
            }
            for frame in (0, 1)
            for id1, id2 in [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
        ]
    )


def test_duration_sums_each_individuals_own_seconds() -> None:
    out = _apply({"columns": ["value"], "aggs": ["duration"]}, _timed_individuals())

    assert out["id"].tolist() == [0, 1]
    # Individual 1's last row has no value, so its 0.1 s does not count.
    assert out["value_duration"].tolist() == pytest.approx([2.3, 2.2])


def test_duration_of_a_thresholded_column_is_the_time_the_individual_is_above_it() -> (
    None
):
    table = _timed_individuals().assign(in_center=lambda df: df["frame"] != 2)

    out = _apply(
        {"columns": ["in_center"], "aggs": ["duration"], "threshold": 0.5}, table
    )

    assert out["in_center_duration"].tolist() == pytest.approx([0.1 + 2.0 + 0.1] * 2)


def test_a_filter_leaves_a_kept_row_the_seconds_it_had_before_it() -> None:
    out = _apply(
        {
            "columns": ["value"],
            "aggs": ["duration"],
            "filter_expr": "frame != 0 and frame != 2",
        },
        _timed_individuals(),
    )

    assert out["value_duration"].tolist() == pytest.approx([2.1, 2.0])


def test_duration_over_pair_input_sums_a_focals_time_over_its_partners() -> None:
    out = _apply({"columns": ["A_speed"], "aggs": ["duration"]}, _timed_pair_table())

    # Each focal has two partners, each for 0.2 s.
    assert out["id"].tolist() == [0, 1, 2]
    assert out["A_speed_duration"].tolist() == pytest.approx([0.4, 0.4, 0.4])


def test_duration_splits_by_subgroup_after_each_row_has_its_seconds() -> None:
    out = _apply(
        {"columns": ["value"], "aggs": ["duration"], "subgroup_col": "tag"},
        _timed_individuals(),
    )

    assert out[["id", "tag"]].values.tolist() == [
        [0, "a"],
        [0, "b"],
        [1, "a"],
        [1, "b"],
    ]
    assert out["value_duration"].tolist() == pytest.approx([0.2, 2.1, 0.2, 2.0])


def test_an_id_reused_in_two_arenas_keeps_each_arenas_seconds_apart() -> None:
    table = pd.concat(
        [
            _timed_individuals().iloc[::2].assign(arena="a"),
            _timed_individuals()
            .iloc[::2]
            .assign(arena="b", time=lambda d: d["time"] * 2),
        ]
    )

    out = _apply({"columns": ["value"], "aggs": ["duration"]}, table)

    assert out["arena"].tolist() == ["a", "b"]
    assert out["value_duration"].tolist() == pytest.approx([2.3, 4.6])


def test_a_kept_label_is_written_beside_a_duration() -> None:
    table = _timed_individuals().assign(strain="s")

    out = _apply(_kept(aggs=["duration"]), table)

    assert list(out.columns) == ["group", "sequence", "id", "strain", "value_duration"]
    assert out["value_duration"].tolist() == pytest.approx([2.3, 2.2])


def test_a_filter_that_keeps_nothing_gives_an_empty_table_for_duration() -> None:
    out = _apply(
        {"columns": ["value"], "aggs": ["duration"], "filter_expr": "id > 100"},
        _timed_individuals(),
    )

    assert out.empty


def test_a_mean_alone_reads_no_time() -> None:
    out = _apply(
        {"columns": ["value"], "aggs": ["mean"]},
        _timed_individuals().drop(columns="time"),
    )

    assert out["value_mean"].tolist() == [1.0, 1.0]


def test_duration_over_input_without_time_names_the_column_and_what_to_do() -> None:
    table = _timed_individuals().drop(columns="time")

    with pytest.raises(
        ValueError, match="reads its rate off the 'time' column"
    ) as raised:
        _ = _apply({"columns": ["value"], "aggs": ["duration"]}, table)

    message = str(raised.value)
    assert message.startswith("individual-aggregate reads its rate off the 'time'")
    assert message.endswith(TIME_SENTENCE)


def test_a_run_whose_time_does_not_advance_loses_the_entry_with_the_sentence(
    tmp_path: Path,
) -> None:
    feature = IndividualAggregate(params={"columns": ["X"], "aggs": ["duration"]})

    run = run_over_edited_tracks(tmp_path, feature, constant_time_in_mixed)

    assert run.result.failed_entries == (make_entry_key(*MIXED),)
    assert run.written == [make_entry_key(*STEADY)]
    (error,) = run.errors
    assert "does not advance" in error["message"]
    assert error["message"].endswith(TIME_SENTENCE)


def test_a_run_over_tracks_without_time_lists_the_entry_as_failed(
    tmp_path: Path,
) -> None:
    feature = IndividualAggregate(params={"columns": ["X"], "aggs": ["duration"]})

    run = run_over_edited_tracks(tmp_path, feature, without_time_in_mixed)

    assert run.result.failed_entries == (make_entry_key(*MIXED),)
    assert run.written == [make_entry_key(*STEADY)]


@pytest.mark.parametrize("agg", ["duration", "share"])
def test_a_run_over_an_input_without_time_fails_the_entry_before_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agg: str
) -> None:
    applied = record_applied_sequences(monkeypatch, IndividualAggregate)
    feature = IndividualAggregate(params={"columns": ["frame"], "aggs": [agg]})

    run = run_over_edited_tracks(tmp_path, feature, without_time_in_mixed)

    assert run.result.failed_entries == (make_entry_key(*MIXED),)
    assert applied == [STEADY[1]]
    (error,) = run.errors
    assert error["error_type"] == "MissingInputColumns"
    assert "'time'" in error["message"]


def test_a_run_of_means_over_tracks_without_time_succeeds(tmp_path: Path) -> None:
    feature = IndividualAggregate(params={"columns": ["X"], "aggs": ["mean"]})

    run = run_over_edited_tracks(tmp_path, feature, without_time)

    assert run.result.failed_entries == ()


# --- share ----------------------------------------------------------------------------


def _share(params: dict[str, object], df: pd.DataFrame) -> pd.DataFrame:
    return _apply({"aggs": ["share"], **params}, df)


def _individuals(**by_id: Sequence[object]) -> pd.DataFrame:
    """One table of the individuals ``id0``, ``id1``, ... with each one's ``roi``."""
    return pd.concat(
        [
            stepped_individual({"roi": values}).assign(id=int(name.removeprefix("id")))
            for name, values in by_id.items()
        ],
        ignore_index=True,
    )


def test_share_is_each_individuals_own_time_fraction_and_nan_for_a_null_only_one() -> (
    None
):
    out = _share({"columns": ["roi"]}, _individuals(id0=ROI_VALUES, id1=[None] * 5))

    assert list(out.columns) == [
        "group",
        "sequence",
        "id",
        "roi_share_center",
        "roi_share_feeder",
        "roi_share_none",
    ]
    assert out["id"].tolist() == [0, 1]
    assert out.iloc[0, 3:].tolist() == pytest.approx([0.5, 0.25, 0.25])
    assert out.iloc[1, 3:].isna().to_numpy().all()
    assert out["roi_share_center"].dtype == np.float64


def test_a_value_an_individual_lacks_is_zero() -> None:
    out = _share({"columns": ["roi"]}, _individuals(id0=["a", "a"], id1=["b", "a"]))

    assert out["roi_share_a"].tolist() == pytest.approx([1.0, 0.5])
    assert out["roi_share_b"].tolist() == pytest.approx([0.0, 0.5])


@pytest.mark.parametrize("dtype", [object, "string", "str"])
def test_text_gives_the_same_shares_in_any_text_dtype(dtype: str | type) -> None:
    table = _individuals(id0=ROI_VALUES)
    roi = pd.Series(ROI_VALUES, dtype=dtype)
    table["roi"] = roi.where(roi.notna(), np.nan) if dtype == "str" else roi

    out = _share({"columns": ["roi"]}, table)

    assert out.iloc[0, 3:].tolist() == pytest.approx([0.5, 0.25, 0.25])


def test_text_read_back_from_a_parquet_file_gives_the_same_shares(
    tmp_path: Path,
) -> None:
    path = tmp_path / "roi.parquet"
    _ = write_parquet_atomic(_individuals(id0=ROI_VALUES), path)

    out = _share({"columns": ["roi"]}, pd.read_parquet(path))

    assert out.iloc[0, 3:].tolist() == pytest.approx([0.5, 0.25, 0.25])


def test_a_nullable_boolean_column_writes_false_before_true() -> None:
    in_center = pd.array([True, False, None, True, True], dtype="boolean")

    out = _share(
        {"columns": ["in_center"]}, stepped_individual({"in_center": in_center})
    )

    assert list(out.columns)[3:] == ["in_center_share_false", "in_center_share_true"]
    assert out.iloc[0, 3:].tolist() == pytest.approx([0.25, 0.75])


def test_an_integer_column_writes_decimal_names_in_numeric_order() -> None:
    out = _share(
        {"columns": ["x"]}, stepped_individual({"x": pd.array([3, 10, 3, -1], "Int64")})
    )

    assert list(out.columns)[3:] == ["x_share_-1", "x_share_3", "x_share_10"]
    assert out["x_share_3"].tolist() == pytest.approx([0.5])


def test_categories_pin_the_columns_in_their_order_and_write_zero_for_a_value_absent() -> (
    None
):
    out = _share(
        {"columns": ["roi"], "categories": ["feeder", "center", "none", "nest"]},
        _individuals(id0=ROI_VALUES, id1=["none"] * 3),
    )

    assert list(out.columns)[3:] == [
        "roi_share_feeder",
        "roi_share_center",
        "roi_share_none",
        "roi_share_nest",
    ]
    assert out["roi_share_nest"].tolist() == [0.0, 0.0]
    assert out["roi_share_none"].tolist() == pytest.approx([0.25, 1.0])


def test_a_value_outside_the_categories_fails_the_entry_by_name() -> None:
    with pytest.raises(ValueError, match="'feeder'") as raised:
        _ = _share(
            {"columns": ["roi"], "categories": ["center", "none"]},
            _individuals(id0=ROI_VALUES),
        )

    assert "center" in str(raised.value)


def test_unset_categories_let_two_entries_write_different_sorted_columns() -> None:
    first = _share({"columns": ["roi"]}, _individuals(id0=["b", "a"]))
    second = _share({"columns": ["roi"]}, _individuals(id0=["d", "c", "b"]))

    assert list(first.columns)[3:] == ["roi_share_a", "roi_share_b"]
    assert list(second.columns)[3:] == ["roi_share_b", "roi_share_c", "roi_share_d"]


def test_a_float_column_fails_the_entry_naming_the_column_and_its_dtype() -> None:
    with pytest.raises(ValueError, match="'value'") as raised:
        _ = _share({"columns": ["value"]}, stepped_individual({"value": [1.0, 2.0]}))

    assert "float64" in str(raised.value)


def test_an_object_column_of_text_and_integers_fails_the_entry_naming_its_kinds() -> (
    None
):
    table = _individuals(id0=["a", "b", "c"])
    table["roi"] = pd.Series(["a", 1, "b"], dtype=object)

    with pytest.raises(ValueError, match="'roi'") as raised:
        _ = _share({"columns": ["roi"]}, table)

    assert "int" in str(raised.value)
    assert "str" in str(raised.value)


@pytest.mark.parametrize("value", ["a b", ""])
def test_a_value_outside_the_safe_characters_fails_the_entry_and_points_upstream(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="recode") as raised:
        _ = _share({"columns": ["roi"]}, _individuals(id0=["a", value]))

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
    table = stepped_individual({"roi": ["a", "b", "b", "b"]}).assign(
        time=[0.0, 0.1, 2.1, 2.2]
    )

    out = _share({"columns": ["roi"]}, table)

    # Row 0 lasts 0.1 s and row 1 lasts 2.0 s, which a count of rows would not say.
    assert out["roi_share_a"].tolist() == pytest.approx([0.1 / 2.3])
    assert out["roi_share_b"].tolist() == pytest.approx([2.2 / 2.3])


def test_share_splits_an_individual_by_subgroup() -> None:
    table = stepped_individual(
        {"roi": ["a", "a", "a", "b"], "tag": ["x", "x", "y", "y"]}
    )

    out = _share({"columns": ["roi"], "subgroup_col": "tag"}, table)

    assert out["tag"].tolist() == ["x", "y"]
    assert out["roi_share_a"].tolist() == pytest.approx([1.0, 0.5])
    assert out["roi_share_b"].tolist() == pytest.approx([0.0, 0.5])


def test_an_id_reused_in_two_arenas_keeps_each_arenas_share_apart() -> None:
    table = pd.concat(
        [
            stepped_individual({"roi": ["a", "a", "b"]}).assign(arena="p"),
            stepped_individual({"roi": ["a", "b", "b"]}, step=0.5).assign(arena="q"),
        ]
    )

    out = _share({"columns": ["roi"]}, table)

    assert out["arena"].tolist() == ["p", "q"]
    assert out["roi_share_a"].tolist() == pytest.approx([2 / 3, 1 / 3])


def test_a_kept_label_is_written_beside_the_share_columns() -> None:
    table = stepped_individual({"value": ["a", "b"], "strain": "s"})

    out = _share(_kept(aggs=["share"]), table)

    assert list(out.columns) == [
        "group",
        "sequence",
        "id",
        "strain",
        "value_share_a",
        "value_share_b",
    ]


def test_share_reads_every_focal_row_of_a_pair_table() -> None:
    table = _timed_pair_table().assign(
        region=lambda df: np.where(df["id2"] == 0, "near", "far")
    )

    out = _share({"columns": ["region"]}, table)

    # Focal 0 has two partners, both far. Focals 1 and 2 have one near and one far.
    assert out["id"].tolist() == [0, 1, 2]
    assert out["region_share_far"].tolist() == pytest.approx([1.0, 0.5, 0.5])
    assert out["region_share_near"].tolist() == pytest.approx([0.0, 0.5, 0.5])


def test_a_filter_that_keeps_nothing_gives_an_empty_table_for_share() -> None:
    out = _share(
        {"columns": ["roi"], "categories": ["a"], "filter_expr": "id > 100"},
        _individuals(id0=["a", "b"]),
    )

    assert out.empty


def test_share_over_input_without_time_names_the_column_and_what_to_do() -> None:
    table = _individuals(id0=["a", "b"]).drop(columns="time")

    with pytest.raises(ValueError, match="reads its rate off the 'time'") as raised:
        _ = _share({"columns": ["roi"]}, table)

    assert str(raised.value).endswith(TIME_SENTENCE)


def test_a_run_loses_the_entry_whose_value_is_outside_the_categories(
    tmp_path: Path,
) -> None:
    feature = IndividualAggregate(
        params={"columns": ["strain"], "aggs": ["share"], "categories": ["a"]}
    )

    run = run_over_labeled_tracks(tmp_path, feature)

    assert run.result.failed_entries == (make_entry_key(*MIXED),)
    assert run.written == [make_entry_key(*STEADY)]
    (error,) = run.errors
    assert "'b'" in error["message"]


def test_share_with_a_threshold_or_the_abs_transform_is_refused() -> None:
    with pytest.raises(ValueError, match="share.*threshold"):
        _ = IndividualAggregate(
            params={"columns": ["roi"], "aggs": ["share"], "threshold": 0.5}
        )
    with pytest.raises(ValueError, match="share.*abs"):
        _ = IndividualAggregate(
            params={"columns": ["roi"], "aggs": ["share"], "transform": "abs"}
        )


def test_categories_without_share_are_refused() -> None:
    with pytest.raises(ValueError, match="categories.*share"):
        _ = IndividualAggregate(
            params={"columns": ["roi"], "aggs": ["mean"], "categories": ["a"]}
        )


@pytest.mark.parametrize(
    ("categories", "match"),
    [([], "at least 1"), (["a", "a"], "lists a more than once"), (["a b"], "'a b'")],
)
def test_categories_that_are_empty_repeated_or_unsafe_are_refused(
    categories: list[str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _ = IndividualAggregate(
            params={"columns": ["roi"], "aggs": ["share"], "categories": categories}
        )
