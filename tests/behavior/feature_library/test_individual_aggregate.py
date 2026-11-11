"""Tests for the IndividualAggregate feature: one row per individual."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.individual_aggregate import IndividualAggregate
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import Recipe, plan_pipeline
from mosaic.core.pipeline.run import OverlapUnsupported, run_feature
from tests.helpers import add_track_sequences, add_tracks_variant, make_dataset


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
