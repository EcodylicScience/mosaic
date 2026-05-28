"""Tests for the generic FrameAggregate feature."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.frame_aggregate import FrameAggregate


def _make_df(values_per_frame: dict[int, list[float]],
             extra_cols: dict[str, list] | None = None,
             seq: str = "S1", group: str = "G") -> pd.DataFrame:
    """Build a tracks-shaped DataFrame.

    values_per_frame[frame] is the list of per-id values that frame.
    """
    rows = []
    for frame, vals in values_per_frame.items():
        for i, v in enumerate(vals):
            rows.append({
                "frame": int(frame),
                "time": float(frame) * 0.02,
                "id": i,
                "value": v,
                "sequence": seq,
                "group": group,
            })
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
    out = _apply({"column": "value", "agg": "mean"}, df)
    assert list(out["frame"]) == [0, 1, 2]
    np.testing.assert_allclose(out["value_mean"], [2.0, 5.0, 20.0])


def test_median():
    df = _make_df({0: [1.0, 2.0, 100.0], 1: [4.0, 5.0, 6.0]})
    out = _apply({"column": "value", "agg": "median"}, df)
    np.testing.assert_allclose(out["value_median"], [2.0, 5.0])


@pytest.mark.parametrize("agg,expected", [
    ("min", [1.0, 4.0]),
    ("max", [3.0, 6.0]),
    ("sum", [6.0, 15.0]),
    ("count", [3, 3]),
])
def test_min_max_sum_count(agg, expected):
    df = _make_df({0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]})
    out = _apply({"column": "value", "agg": agg}, df)
    np.testing.assert_allclose(out[f"value_{agg}"], expected)


def test_std():
    df = _make_df({0: [1.0, 2.0, 3.0]})
    out = _apply({"column": "value", "agg": "std"}, df)
    np.testing.assert_allclose(out["value_std"], [np.std([1.0, 2.0, 3.0], ddof=1)])


# ---------------------------------------------------------------------------
# Transforms / threshold
# ---------------------------------------------------------------------------


def test_transform_abs():
    df = _make_df({0: [-1.0, -2.0, 3.0], 1: [-4.0, 5.0, -6.0]})
    out = _apply({"column": "value", "agg": "mean", "transform": "abs"}, df)
    np.testing.assert_allclose(out["value_mean"], [2.0, 5.0])


def test_threshold_mean():
    # Boolean fraction-of-ids-exceeding-threshold
    df = _make_df({0: [1.0, 1.5, 2.0, 3.0], 1: [0.5, 0.7, 0.9, 1.1]})
    out = _apply({"column": "value", "agg": "mean", "threshold": 1.0}, df)
    # frame 0: 3 of 4 > 1.0 = 0.75 ; frame 1: 1 of 4 > 1.0 = 0.25
    np.testing.assert_allclose(out["value_mean"], [0.75, 0.25])


# ---------------------------------------------------------------------------
# filter_expr
# ---------------------------------------------------------------------------


def test_filter_expr_perspective_dedup():
    """Pair-perspective duplication: sum without filter double-counts;
    sum with filter_expr='perspective == 0' gives the right answer."""
    # 2 pairs (a,b) and (a,c) with mirrored perspectives, frame 0 only.
    df = pd.DataFrame({
        "frame": [0, 0, 0, 0],
        "time": [0.0, 0.0, 0.0, 0.0],
        "perspective": [0, 1, 0, 1],
        "AB_dist": [3.0, 3.0, 5.0, 5.0],   # mirror values
        "sequence": "S1",
        "group": "G",
    })
    out_no_dedup = _apply({"column": "AB_dist", "agg": "sum"}, df)
    out_dedup = _apply({"column": "AB_dist", "agg": "sum",
                        "filter_expr": "perspective == 0"}, df)
    assert float(out_no_dedup["AB_dist_sum"].iloc[0]) == pytest.approx(16.0)  # 2x
    assert float(out_dedup["AB_dist_sum"].iloc[0]) == pytest.approx(8.0)


def test_filter_expr_drops_rows():
    df = _make_df({0: [1.0, 2.0, 3.0, 4.0], 1: [10.0, 20.0, 30.0, 40.0]})
    out = _apply({"column": "value", "agg": "mean",
                  "filter_expr": "id < 2"}, df)
    # Only ids 0,1 keep -> mean is [(1+2)/2, (10+20)/2]
    np.testing.assert_allclose(out["value_mean"], [1.5, 15.0])


# ---------------------------------------------------------------------------
# Metadata / output_column / empty
# ---------------------------------------------------------------------------


def test_metadata_attached():
    df = _make_df({0: [1.0, 2.0]}, seq="OCI_1", group="OCI")
    out = _apply({"column": "value", "agg": "mean"}, df)
    assert (out["sequence"] == "OCI_1").all()
    assert (out["group"] == "OCI").all()
    assert "time" in out.columns
    np.testing.assert_allclose(out["time"], [0.0])


def test_output_column_override():
    df = _make_df({0: [1.0, 3.0]})
    out = _apply({"column": "value", "agg": "mean",
                  "output_column": "custom_name"}, df)
    assert "custom_name" in out.columns
    assert "value_mean" not in out.columns
    np.testing.assert_allclose(out["custom_name"], [2.0])


def test_empty_input():
    df = pd.DataFrame()
    out = _apply({"column": "value", "agg": "mean"}, df)
    assert out.empty


def test_filter_eliminates_all_rows():
    df = _make_df({0: [1.0, 2.0]})
    out = _apply({"column": "value", "agg": "mean",
                  "filter_expr": "id > 100"}, df)
    assert out.empty


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_nan_skipped_in_mean():
    df = _make_df({0: [1.0, np.nan, 3.0], 1: [np.nan, np.nan, 6.0]})
    out = _apply({"column": "value", "agg": "mean"}, df)
    # mean of [1, NaN, 3] = 2.0 ; mean of [NaN, NaN, 6] = 6.0
    np.testing.assert_allclose(out["value_mean"], [2.0, 6.0])


def test_count_excludes_nan():
    df = _make_df({0: [1.0, np.nan, 3.0]})
    out = _apply({"column": "value", "agg": "count"}, df)
    assert int(out["value_count"].iloc[0]) == 2
