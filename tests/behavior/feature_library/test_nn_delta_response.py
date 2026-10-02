"""Tests for ``nn-delta-response``, focused on the 0.2 window extension.

The load-bearing one is :func:`test_defaults_are_a_no_op_against_the_uniform_shift`:
0.2 adds a per-row lag and a backward window, and both must leave the default
path bit-identical, because every published force map was produced by it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.nn_delta_response import (
    NearestNeighborDelta,
    _row_lags,
    _windowed_delta,
)


def _track(
    n_frames: int = 40,
    n_ids: int = 3,
    *,
    drop_frames: set[int] | None = None,
) -> pd.DataFrame:
    """A synthetic merged track frame with the columns the feature needs."""
    drop_frames = drop_frames or set()
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(0)
    for fid in range(n_ids):
        for frame in range(n_frames):
            if frame in drop_frames:
                continue
            rows.append(
                {
                    "frame": frame,
                    "id": fid,
                    "group": "g",
                    "sequence": "s",
                    "X": float(frame) + fid * 10.0,
                    "Y": float(fid),
                    "ANGLE": 0.01 * frame + fid,
                    "speed": 1.0 + 0.1 * frame + rng.normal(0, 0.01),
                    "nn_id": float((fid + 1) % n_ids),
                    "nn_delta_x_ego": 1.0 + fid,
                    "nn_delta_y_ego": -0.5 + fid,
                }
            )
    return pd.DataFrame(rows)


def _run(df: pd.DataFrame, **params: object) -> pd.DataFrame:
    feat = NearestNeighborDelta(params={"speed_col": "speed", **params})
    return feat.apply(df.copy())


# --------------------------------------------------------------------------- #
# (a) the regression test: defaults must not move
# --------------------------------------------------------------------------- #


def _uniform_shift_reference(df: pd.DataFrame, diff_n: int) -> pd.DataFrame:
    """The 0.1 implementation of the forward window, kept verbatim as an oracle."""
    out: list[pd.DataFrame] = []
    cols = ["X", "Y", "ANGLE", "speed", "frame"]
    for _, g in df.sort_values(["frame", "id"]).groupby("id", sort=False):
        g = g.sort_values("frame")
        delta = g[cols].shift(-diff_n) - g[cols]
        valid = delta["frame"].notna() & (delta["frame"] == diff_n)
        if not valid.any():
            continue
        rows = g.loc[valid, ["frame", "id"]].copy()
        rows["dx"] = delta.loc[valid, "X"].to_numpy()
        rows["dy"] = delta.loc[valid, "Y"].to_numpy()
        rows["dt"] = delta.loc[valid, "frame"].to_numpy()
        rows["dspeed"] = delta.loc[valid, "speed"].to_numpy()
        out.append(rows)
    return pd.concat(out, ignore_index=True).sort_values(["id", "frame"])


@pytest.mark.parametrize("diff_n", [1, 4, 10])
@pytest.mark.parametrize("drop", [None, {7, 8, 21}])
def test_defaults_are_a_no_op_against_the_uniform_shift(
    diff_n: int, drop: set[int] | None
) -> None:
    """0.2's positional gather reproduces 0.1's ``shift(-n)``, gaps included."""
    df = _track(drop_frames=drop)
    got = _run(df, diff_numframes=diff_n).sort_values(["id", "frame"])
    want = _uniform_shift_reference(df, diff_n)

    assert len(got) == len(want)
    for col in ("frame", "id", "dx", "dy", "dt", "dspeed"):
        np.testing.assert_allclose(
            got[col].to_numpy(dtype=float),
            want[col].to_numpy(dtype=float),
            err_msg=f"column {col!r} moved",
        )


def test_defaults_emit_no_new_columns() -> None:
    """A caller on 0.1 params sees the 0.1 schema -- no stray dx_back/dt_frames."""
    out = _run(_track())
    assert "dx_back" not in out.columns
    assert "dy_back" not in out.columns
    assert "dt_frames" not in out.columns


# --------------------------------------------------------------------------- #
# (b) the backward window
# --------------------------------------------------------------------------- #


def test_backward_window_matches_forward_on_constant_velocity() -> None:
    """Constant velocity => equal ground covered both ways, so Model B's target is 0."""
    out = _run(_track(), diff_numframes=4, emit_backward=True)
    fwd = np.hypot(out["dx"], out["dy"])
    back = np.hypot(out["dx_back"], out["dy_back"])
    finite = np.isfinite(back)
    assert finite.any()
    np.testing.assert_allclose(fwd[finite], back[finite], atol=1e-12)


def test_backward_window_is_nan_only_at_the_start_of_a_track() -> None:
    """Rows keep their forward answer even when they have no valid past."""
    diff_n = 4
    out = _run(_track(), diff_numframes=diff_n, emit_backward=True)
    for fid, g in out.groupby("id"):
        g = g.sort_values("frame")
        missing = g.loc[~np.isfinite(g["dx_back"]), "frame"].to_numpy()
        assert set(missing) == set(range(diff_n)), f"id {fid}: {missing}"
        # and the forward answer survived for those rows
        assert np.isfinite(g["dx"].to_numpy()).all()


def test_backward_window_refuses_to_straddle_a_frame_gap() -> None:
    """A positional gather across a gap would report the wrong elapsed time."""
    diff_n = 4
    df = _track(drop_frames={20})
    out = _run(df, diff_numframes=diff_n, emit_backward=True)
    one = out[out["id"] == 0].sort_values("frame")
    invalid = set(one.loc[~np.isfinite(one["dx_back"]), "frame"].to_numpy())
    # the first tau frames, plus every row whose tau-window would span frame 20
    assert invalid == set(range(diff_n)) | {21, 22, 23, 24}


# --------------------------------------------------------------------------- #
# (c) the per-row lag
# --------------------------------------------------------------------------- #


def test_constant_lag_column_equals_the_uniform_path() -> None:
    df = _track()
    df["my_lag"] = 4
    got = _run(df, diff_numframes=99, diff_numframes_col="my_lag").sort_values(
        ["id", "frame"]
    )
    want = _run(_track(), diff_numframes=4).sort_values(["id", "frame"])
    assert len(got) == len(want)
    for col in ("dx", "dy", "dt", "dangle", "dspeed"):
        np.testing.assert_allclose(
            got[col].to_numpy(dtype=float), want[col].to_numpy(dtype=float)
        )
    assert (got["dt_frames"] == 4).all()


def test_per_row_lag_varies_the_window_and_is_reported() -> None:
    """Rows with a longer lag cover proportionally more ground on a constant track."""
    df = _track()
    df["my_lag"] = np.where(df["frame"] < 20, 2, 6)
    out = _run(df, diff_numframes=4, diff_numframes_col="my_lag")
    short = out[out["dt_frames"] == 2]
    long = out[out["dt_frames"] == 6]
    assert len(short) > 0 and len(long) > 0
    # X advances 1 unit/frame in the fixture, so |dx| == lag
    np.testing.assert_allclose(short["dx"].to_numpy(), 2.0)
    np.testing.assert_allclose(long["dx"].to_numpy(), 6.0)
    np.testing.assert_allclose(out["dt"].to_numpy(), out["dt_frames"].to_numpy())


def test_dangle_is_divided_by_the_per_row_lag() -> None:
    """`divide_dangle_by_frames` must use the row's own lag, not the nominal one."""
    df = _track()
    df["my_lag"] = np.where(df["frame"] < 20, 2, 6)
    out = _run(
        df,
        diff_numframes=4,
        diff_numframes_col="my_lag",
        scale_dangle_by_fps=False,
    )
    # ANGLE advances 0.01 rad/frame, so dangle-per-frame is 0.01 for every lag
    np.testing.assert_allclose(out["dangle"].to_numpy(), 0.01, atol=1e-12)


def test_lag_column_falls_back_where_it_is_unusable() -> None:
    """NaN / sub-frame lags use the nominal delay rather than dropping the row."""
    df = _track()
    df["my_lag"] = np.where(df["id"] == 1, np.nan, 3.0)
    out = _run(df, diff_numframes=5, diff_numframes_col="my_lag")
    assert (out.loc[out["id"] == 1, "dt_frames"] == 5).all()
    assert (out.loc[out["id"] != 1, "dt_frames"] == 3).all()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def test_row_lags_rounds_and_floors() -> None:
    g = pd.DataFrame({"lag": [2.4, 2.6, 0.4, np.inf, np.nan, -3.0]})
    np.testing.assert_array_equal(_row_lags(g, "lag", 7), [2, 3, 7, 7, 7, 7])
    np.testing.assert_array_equal(_row_lags(g, None, 7), [7] * 6)


def test_windowed_delta_boundary_rows_are_never_valid() -> None:
    vals = np.column_stack([np.arange(6.0), np.arange(6.0)])  # value, frame
    lags = np.full(6, 2, dtype=np.int64)
    _, ok_fwd = _windowed_delta(vals, lags, frame_idx=1, forward=True)
    _, ok_back = _windowed_delta(vals, lags, frame_idx=1, forward=False)
    np.testing.assert_array_equal(ok_fwd, [1, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(ok_back, [0, 0, 1, 1, 1, 1])
