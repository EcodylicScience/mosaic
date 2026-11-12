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
from tests.helpers import (
    MIXED,
    STEADY,
    PlainFeature,
    add_track_sequences,
    add_tracks_variant,
    make_dataset,
    result_outputs,
    run_over_labeled_tracks,
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
