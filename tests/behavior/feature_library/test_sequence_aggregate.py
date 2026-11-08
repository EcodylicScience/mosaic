"""Tests for the SequenceAggregate feature: one row per entry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.sequence_aggregate import SequenceAggregate
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import Recipe, plan_pipeline, run_pipeline
from mosaic.core.pipeline.inventory import FeatureRunRef
from mosaic.core.pipeline.preview import preview_artifact
from tests.helpers import add_tracks_variant, make_dataset

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


def test_a_filter_that_removes_every_row_gives_no_output() -> None:
    out = _apply(
        {"columns": ["value"], "aggs": ["mean"], "filter_expr": "id > 100"},
        _individual_table(),
    )

    assert out.empty


def test_empty_input_gives_no_output() -> None:
    assert _apply({"columns": ["value"]}, pd.DataFrame()).empty


def test_nan_values_are_skipped() -> None:
    df = _individual_table()
    df.loc[0, "value"] = np.nan

    out = _apply({"columns": ["value"], "aggs": ["count"]}, df)

    assert out["value_count"].tolist() == [5]


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
