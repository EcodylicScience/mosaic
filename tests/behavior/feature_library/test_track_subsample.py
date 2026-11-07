"""``track-subsample`` and the rows ``drop_nan`` keeps out of its output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.track_subsample import TrackSubsample

_POSE_N = 2
_REQUIRED = ["X", "Y", "ANGLE", "poseX0", "poseY0", "poseX1", "poseY1"]


def _tracks(n_frames: int, nan_frames: tuple[int, ...] = ()) -> pd.DataFrame:
    """One individual over *n_frames* frames, with ``X`` NaN on *nan_frames*."""
    rng = np.random.default_rng(0)
    frames = np.arange(n_frames)
    x = rng.normal(size=n_frames)
    y = rng.normal(size=n_frames)
    angle = rng.uniform(-np.pi, np.pi, size=n_frames)
    table = pd.DataFrame(
        {
            "frame": frames,
            "id": 0,
            "sequence": "s",
            "X": x,
            "Y": y,
            "ANGLE": angle,
            "poseX0": x + rng.normal(size=n_frames),
            "poseY0": y + rng.normal(size=n_frames),
            "poseX1": x + rng.normal(size=n_frames),
            "poseY1": y + rng.normal(size=n_frames),
            "speed": rng.normal(size=n_frames),
        }
    )
    table.loc[list(nan_frames), "X"] = np.nan
    return table


def _subsample(table: pd.DataFrame, **params: object) -> pd.DataFrame:
    feature = TrackSubsample(params={"pose": {"pose_n": _POSE_N}, **params})
    return feature.apply(table)


def _all_required_finite(table: pd.DataFrame) -> bool:
    return bool(np.isfinite(table[_REQUIRED].to_numpy(dtype=float)).all())


@pytest.mark.parametrize("method", ["kmeans", "uniform", "clips"])
def test_drop_nan_removes_non_finite_required_rows(method: str) -> None:
    table = _tracks(200, nan_frames=tuple(range(0, 200, 7)))

    out = _subsample(table, method=method, target_frames=40, clip_len=4)

    assert len(out) > 0
    assert _all_required_finite(out)


@pytest.mark.parametrize("method", ["kmeans", "uniform", "clips"])
def test_drop_nan_applies_to_a_short_input(method: str) -> None:
    table = _tracks(3, nan_frames=(1,))

    dropped = _subsample(table, method=method, target_frames=10, clip_len=4)
    kept = _subsample(
        table, method=method, target_frames=10, clip_len=4, drop_nan=False
    )

    assert list(dropped["frame"]) == [0, 2]
    assert list(kept["frame"]) == [0, 1, 2]


def test_drop_nan_keeps_a_row_whose_other_columns_are_not_finite() -> None:
    table = _tracks(3)
    table.loc[1, "speed"] = np.nan

    out = _subsample(table, method="uniform", target_frames=10)

    assert list(out["frame"]) == [0, 1, 2]


def test_drop_nan_keeps_each_clip_on_consecutive_frames() -> None:
    clip_len = 5
    table = _tracks(120, nan_frames=(13, 14, 40, 57, 58, 59, 81, 99))

    out = _subsample(table, method="clips", target_frames=60, clip_len=clip_len)

    assert len(out) > 0
    assert len(out) % clip_len == 0
    for start in range(0, len(out), clip_len):
        clip = out.iloc[start : start + clip_len]
        assert list(np.diff(clip["frame"].to_numpy())) == [1] * (clip_len - 1)
        assert _all_required_finite(clip)
