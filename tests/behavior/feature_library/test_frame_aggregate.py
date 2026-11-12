"""Tests for the generic FrameAggregate feature."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.frame_aggregate import FrameAggregate
from mosaic.core.helpers import make_entry_key
from tests.helpers import MIXED, STEADY, run_over_labeled_tracks


def _make_df(
    values_per_frame: dict[int, list[float]],
    extra_cols: dict[str, list] | None = None,
    seq: str = "S1",
    group: str = "G",
) -> pd.DataFrame:
    """Build a tracks-shaped DataFrame.

    values_per_frame[frame] is the list of per-id values that frame.
    """
    rows = []
    for frame, vals in values_per_frame.items():
        for i, v in enumerate(vals):
            rows.append(
                {
                    "frame": int(frame),
                    "time": float(frame) * 0.02,
                    "id": i,
                    "value": v,
                    "sequence": seq,
                    "group": group,
                }
            )
    df = pd.DataFrame(rows)
    if extra_cols:
        for k, vs in extra_cols.items():
            assert len(vs) == len(df)
            df[k] = vs
    return df


def _apply(params: dict, df: pd.DataFrame) -> pd.DataFrame:
    feat = FrameAggregate(params=params)
    return feat.apply(df)


# ---------------------------------------------------------------------------
# Aggregation modes
# ---------------------------------------------------------------------------


def test_mean():
    df = _make_df({0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0], 2: [10.0, 20.0, 30.0]})
    out = _apply({"columns": ["value"], "aggs": ["mean"]}, df)
    assert list(out["frame"]) == [0, 1, 2]
    np.testing.assert_allclose(out["value_mean"], [2.0, 5.0, 20.0])
    assert {name: str(dtype) for name, dtype in out.dtypes.items()} == {
        "frame": "int64",
        "time": "float64",
        "value_mean": "float64",
        "sequence": "str",
        "group": "str",
    }
    assert list(out.columns) == ["frame", "time", "value_mean", "sequence", "group"]


def test_median():
    df = _make_df({0: [1.0, 2.0, 100.0], 1: [4.0, 5.0, 6.0]})
    out = _apply({"columns": ["value"], "aggs": ["median"]}, df)
    np.testing.assert_allclose(out["value_median"], [2.0, 5.0])


@pytest.mark.parametrize(
    "agg,expected",
    [
        ("min", [1.0, 4.0]),
        ("max", [3.0, 6.0]),
        ("sum", [6.0, 15.0]),
        ("count", [3, 3]),
    ],
)
def test_min_max_sum_count(agg, expected):
    df = _make_df({0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]})
    out = _apply({"columns": ["value"], "aggs": [agg]}, df)
    np.testing.assert_allclose(out[f"value_{agg}"], expected)


def test_std():
    df = _make_df({0: [1.0, 2.0, 3.0]})
    out = _apply({"columns": ["value"], "aggs": ["std"]}, df)
    np.testing.assert_allclose(out["value_std"], [np.std([1.0, 2.0, 3.0], ddof=1)])


# ---------------------------------------------------------------------------
# Transforms / threshold
# ---------------------------------------------------------------------------


def test_transform_abs():
    df = _make_df({0: [-1.0, -2.0, 3.0], 1: [-4.0, 5.0, -6.0]})
    out = _apply({"columns": ["value"], "aggs": ["mean"], "transform": "abs"}, df)
    np.testing.assert_allclose(out["value_mean"], [2.0, 5.0])


def test_threshold_mean():
    # Boolean fraction-of-ids-exceeding-threshold
    df = _make_df({0: [1.0, 1.5, 2.0, 3.0], 1: [0.5, 0.7, 0.9, 1.1]})
    out = _apply({"columns": ["value"], "aggs": ["mean"], "threshold": 1.0}, df)
    # frame 0: 3 of 4 > 1.0 = 0.75 ; frame 1: 1 of 4 > 1.0 = 0.25
    np.testing.assert_allclose(out["value_mean"], [0.75, 0.25])


# ---------------------------------------------------------------------------
# filter_expr
# ---------------------------------------------------------------------------


def test_filter_expr_perspective_dedup_gives_the_same_sum_as_once():
    """``once`` keeps perspective 0, so the explicit filter is the same sum and is
    not applied twice."""
    # 2 pairs (a,b) and (a,c) with mirrored perspectives, frame 0 only.
    df = pd.DataFrame(
        {
            "frame": [0, 0, 0, 0],
            "time": [0.0, 0.0, 0.0, 0.0],
            "perspective": [0, 1, 0, 1],
            "AB_dist": [3.0, 3.0, 5.0, 5.0],  # mirror values
            "sequence": "S1",
            "group": "G",
        }
    )
    out_once = _apply({"columns": ["AB_dist"], "aggs": ["sum"], "pairs": "once"}, df)
    out_filtered = _apply(
        {
            "columns": ["AB_dist"],
            "aggs": ["sum"],
            "pairs": "once",
            "filter_expr": "perspective == 0",
        },
        df,
    )
    assert float(out_once["AB_dist_sum"].iloc[0]) == pytest.approx(8.0)
    assert float(out_filtered["AB_dist_sum"].iloc[0]) == pytest.approx(8.0)


def _pair_table() -> pd.DataFrame:
    """Two pairs over two frames, each pair emitted once per perspective."""
    rows: list[dict[str, float | int | str]] = []
    for frame in (0, 1):
        for perspective in (0, 1):
            for base in (3.0, 5.0):
                rows.append(
                    {
                        "frame": frame,
                        "time": frame * 0.02,
                        "perspective": perspective,
                        "AB_dist": base + frame,
                        "sequence": "S1",
                        "group": "G",
                    }
                )
    return pd.DataFrame(rows)


def test_sum_over_pair_input_counts_each_pair_once():
    out = _apply(
        {"columns": ["AB_dist"], "aggs": ["sum"], "pairs": "once"}, _pair_table()
    )
    assert list(out["frame"]) == [0, 1]
    # Each pair counts once: (3 + 5) at frame 0 and (4 + 6) at frame 1.
    np.testing.assert_allclose(out["AB_dist_sum"], [8.0, 10.0])


def test_count_over_pair_input_counts_each_pair_once():
    out = _apply(
        {"columns": ["AB_dist"], "aggs": ["count"], "pairs": "once"}, _pair_table()
    )
    np.testing.assert_allclose(out["AB_dist_count"], [2, 2])


def test_auto_over_pair_input_reads_every_row():
    """Outside a plan nothing resolves ``auto``, and it reads both perspectives."""
    total = _apply({"columns": ["AB_dist"], "aggs": ["sum"]}, _pair_table())
    count = _apply({"columns": ["AB_dist"], "aggs": ["count"]}, _pair_table())
    np.testing.assert_allclose(total["AB_dist_sum"], [16.0, 20.0])
    np.testing.assert_allclose(count["AB_dist_count"], [4, 4])


def test_each_perspective_reads_every_row():
    out = _apply(
        {"columns": ["AB_dist"], "aggs": ["sum"], "pairs": "each_perspective"},
        _pair_table(),
    )
    np.testing.assert_allclose(out["AB_dist_sum"], [16.0, 20.0])


def test_mean_over_pair_input_is_unchanged():
    out = _apply({"columns": ["AB_dist"], "aggs": ["mean"]}, _pair_table())
    np.testing.assert_allclose(out["AB_dist_mean"], [4.0, 5.0])


@pytest.mark.parametrize(
    "agg,expected",
    [
        ("median", [4.0, 5.0]),
        ("min", [3.0, 4.0]),
        ("max", [5.0, 6.0]),
        ("std", [float(np.std([3.0, 3.0, 5.0, 5.0], ddof=1))] * 2),
    ],
)
def test_the_other_aggregations_over_pair_input_read_every_row(
    agg: str, expected: list[float]
):
    out = _apply({"columns": ["AB_dist"], "aggs": [agg]}, _pair_table())
    np.testing.assert_allclose(out[f"AB_dist_{agg}"], expected)


def test_input_without_perspective_is_unchanged():
    df = _make_df({0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]})
    out_sum = _apply({"columns": ["value"], "aggs": ["sum"]}, df)
    out_count = _apply({"columns": ["value"], "aggs": ["count"]}, df)
    np.testing.assert_allclose(out_sum["value_sum"], [6.0, 15.0])
    np.testing.assert_allclose(out_count["value_count"], [3, 3])


def test_a_pair_input_filtered_to_one_perspective_is_not_filtered_twice():
    out = _apply(
        {"columns": ["AB_dist"], "aggs": ["sum"], "filter_expr": "perspective == 0"},
        _pair_table(),
    )
    np.testing.assert_allclose(out["AB_dist_sum"], [8.0, 10.0])


def test_filter_expr_drops_rows():
    df = _make_df({0: [1.0, 2.0, 3.0, 4.0], 1: [10.0, 20.0, 30.0, 40.0]})
    out = _apply({"columns": ["value"], "aggs": ["mean"], "filter_expr": "id < 2"}, df)
    # Only ids 0,1 keep -> mean is [(1+2)/2, (10+20)/2]
    np.testing.assert_allclose(out["value_mean"], [1.5, 15.0])


# ---------------------------------------------------------------------------
# Metadata / output_column / empty
# ---------------------------------------------------------------------------


def test_metadata_attached():
    df = _make_df({0: [1.0, 2.0]}, seq="OCI_1", group="OCI")
    out = _apply({"columns": ["value"], "aggs": ["mean"]}, df)
    assert (out["sequence"] == "OCI_1").all()
    assert (out["group"] == "OCI").all()
    assert "time" in out.columns
    np.testing.assert_allclose(out["time"], [0.0])


def test_a_table_without_frames_is_aggregated_by_its_time():
    """The time orders the rows and keys the result, and is not aggregated twice."""
    df = _make_df({0: [1.0, 3.0], 1: [4.0, 6.0]}).drop(columns="frame")
    out = _apply({"columns": ["value"], "aggs": ["mean"]}, df)
    assert list(out.columns) == ["time", "value_mean", "sequence", "group"]
    np.testing.assert_allclose(out["time"], [0.0, 0.02])
    np.testing.assert_allclose(out["value_mean"], [2.0, 5.0])


def test_an_arena_column_gives_one_row_per_frame_and_arena():
    df = _make_df(
        {0: [1.0, 3.0, 10.0], 1: [4.0, 6.0, 20.0]},
        extra_cols={"arena": ["a", "a", "b", "a", "a", "b"]},
    )
    out = _apply({"columns": ["value"], "aggs": ["mean", "count"]}, df)
    assert list(out.columns) == [
        "frame",
        "arena",
        "time",
        "value_mean",
        "value_count",
        "sequence",
        "group",
    ]
    assert out["arena"].tolist() == ["a", "b", "a", "b"]
    np.testing.assert_allclose(out["value_mean"], [2.0, 10.0, 5.0, 20.0])
    np.testing.assert_allclose(out["value_count"], [2, 1, 2, 1])


def test_empty_input():
    df = pd.DataFrame()
    out = _apply({"columns": ["value"], "aggs": ["mean"]}, df)
    assert out.empty


def test_filter_eliminates_all_rows():
    df = _make_df({0: [1.0, 2.0]})
    out = _apply(
        {"columns": ["value"], "aggs": ["mean"], "filter_expr": "id > 100"}, df
    )
    assert out.empty


# ---------------------------------------------------------------------------
# keep_columns
# ---------------------------------------------------------------------------


def _kept(**overrides: object) -> dict[str, object]:
    return {
        "columns": ["value"],
        "aggs": ["mean"],
        "keep_columns": ["strain"],
    } | overrides


def test_a_kept_label_is_written_after_time_with_its_value() -> None:
    df = _make_df({0: [1.0, 2.0], 1: [3.0, 4.0]}, {"strain": ["wild"] * 4})

    out = _apply(_kept(), df)

    assert list(out.columns) == [
        "frame",
        "time",
        "strain",
        "value_mean",
        "sequence",
        "group",
    ]
    assert out["strain"].tolist() == ["wild", "wild"]


def test_a_kept_label_may_differ_between_frames() -> None:
    df = _make_df({0: [1.0, 2.0], 1: [3.0, 4.0]}, {"strain": ["a", "a", "b", "b"]})

    out = _apply(_kept(), df)

    assert out["strain"].tolist() == ["a", "b"]


@pytest.mark.parametrize(
    ("strain", "values"),
    [
        (["a", "b", "a", "a"], "'a', 'b'"),
        (["a", None, "a", "a"], "'a', null"),
    ],
    ids=["two values", "a value and a null"],
)
def test_a_kept_label_that_varies_within_a_frame_is_refused(
    strain: list[str | None], values: str
) -> None:
    df = _make_df({0: [1.0, 2.0], 1: [3.0, 4.0]}, {"strain": strain})

    with pytest.raises(ValueError) as refused:
        _ = _apply(_kept(), df)

    message = str(refused.value)
    assert "'strain'" in message
    assert "frame=0" in message
    assert values in message


def test_a_kept_label_the_input_lacks_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="'strain'.*not in the input"):
        _ = _apply(_kept(), _make_df({0: [1.0, 2.0]}))


def test_an_entry_whose_label_varies_is_lost_and_the_other_is_written(
    tmp_path: Path,
) -> None:
    run = run_over_labeled_tracks(
        tmp_path, FrameAggregate(params=_kept(columns=["feat_a"]))
    )

    assert run.result.failed_entries == (make_entry_key(*MIXED),)
    assert run.written == [make_entry_key(*STEADY)]
    (error,) = run.errors
    assert error["key"] == make_entry_key(*MIXED)
    assert "'strain'" in error["message"]


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_nan_skipped_in_mean():
    df = _make_df({0: [1.0, np.nan, 3.0], 1: [np.nan, np.nan, 6.0]})
    out = _apply({"columns": ["value"], "aggs": ["mean"]}, df)
    # mean of [1, NaN, 3] = 2.0 ; mean of [NaN, NaN, 6] = 6.0
    np.testing.assert_allclose(out["value_mean"], [2.0, 6.0])


def test_count_excludes_nan():
    df = _make_df({0: [1.0, np.nan, 3.0]})
    out = _apply({"columns": ["value"], "aggs": ["count"]}, df)
    assert int(out["value_count"].iloc[0]) == 2
