"""Excluding flagged rows before a summary reduces them: ``exclude_mask`` and its scopes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mosaic.behavior.feature_library.ffgroups_metrics import FFGroupsMetrics
from mosaic.behavior.feature_library.helpers import apply_exclude_cols, exclude_mask


def _rows(n_frames: int, ids: tuple[int, ...] = (0, 1, 2)) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frames = np.repeat(np.arange(n_frames), len(ids))
    count = len(frames)
    return pd.DataFrame(
        {
            "frame": frames,
            "id": np.tile(ids, n_frames),
            "sequence": "s",
            "group": "g",
            "X": rng.normal(size=count),
            "Y": rng.normal(size=count),
            "speed": rng.uniform(1.0, 3.0, size=count),
            "event": 0,
            "flag": False,
        }
    )


def test_row_scope_flags_only_the_flagged_row() -> None:
    df = _rows(3)
    df.loc[(df["frame"] == 1) & (df["id"] == 0), "flag"] = True
    assert (
        exclude_mask(df, ["flag"]).tolist()
        == [False] * 3 + [True, False, False] + [False] * 3
    )


def test_frame_scope_flags_the_whole_frame() -> None:
    df = _rows(3)
    df.loc[(df["frame"] == 1) & (df["id"] == 0), "flag"] = True
    assert (
        exclude_mask(df, ["flag"], "frame").tolist()
        == [False] * 3 + [True] * 3 + [False] * 3
    )


def test_frame_scope_stays_within_an_arena() -> None:
    df = _rows(2)
    df["arena"] = df["id"].map({0: "a", 1: "a", 2: "b"})
    df.loc[(df["frame"] == 0) & (df["id"] == 0), "flag"] = True
    assert (
        exclude_mask(df, ["flag"], "frame").tolist()
        == [True, True, False] + [False] * 3
    )


def test_absent_columns_and_missing_flags_exclude_nothing() -> None:
    df = _rows(2)
    assert not exclude_mask(df, ["absent"], "frame").any()
    assert not exclude_mask(df, None).any()
    assert apply_exclude_cols(df, ["absent"]) is df


def _ffgroups_metrics(df: pd.DataFrame, **params: object) -> pd.DataFrame:
    return FFGroupsMetrics(params=params).apply(df)


def test_ffgroups_metrics_row_scope_matches_dropping_the_row() -> None:
    df = _rows(8)
    flagged = (df["frame"] == 3) & (df["id"] == 1)
    df.loc[flagged, "flag"] = True
    pd.testing.assert_frame_equal(
        _ffgroups_metrics(df, exclude_cols=["flag"]),
        _ffgroups_metrics(df[~flagged]),
    )


def test_ffgroups_metrics_frame_scope_matches_dropping_the_frame() -> None:
    df = _rows(8)
    df.loc[(df["frame"] == 3) & (df["id"] == 1), "flag"] = True
    excluded = _ffgroups_metrics(df, exclude_cols=["flag"], exclude_scope="frame")
    pd.testing.assert_frame_equal(excluded, _ffgroups_metrics(df[df["frame"] != 3]))
    assert not excluded.equals(_ffgroups_metrics(df, exclude_cols=["flag"]))
