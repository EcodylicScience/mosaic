"""Body length, and the conversion into it, for comparing animals of different sizes.

``body-length`` measures a length along a chain of keypoints, or reads one a
tracker measured, and reports each individual's median over the sequence.
``scale-to-bl`` divides every length column by one length per sequence, so a
table of animals of different ages can be compared in body lengths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.body_length import BodyLength
from mosaic.behavior.feature_library.scale_to_bl import ScaleToBodyLength


def _fish(lengths: dict[int, float], n: int = 5) -> pd.DataFrame:
    """Straight fish along +x, three keypoints from nose to tail, per id."""
    rows: list[dict[str, object]] = []
    for fid, length in lengths.items():
        for frame in range(n):
            nose = (10.0 * fid + frame, 5.0)
            rows.append(
                {
                    "frame": frame,
                    "time": frame / 10.0,
                    "id": fid,
                    "group": "g",
                    "sequence": "s",
                    "X": nose[0] - length / 2,
                    "Y": nose[1],
                    # nose, mid-body, tail: two segments that sum to the length
                    "poseX0": nose[0],
                    "poseY0": nose[1],
                    "poseX1": nose[0] - length / 2,
                    "poseY1": nose[1],
                    "poseX2": nose[0] - length,
                    "poseY2": nose[1],
                }
            )
    return pd.DataFrame(rows)


# --- body-length ----------------------------------------------------------------


def test_length_along_the_whole_chain() -> None:
    out = BodyLength().apply(_fish({0: 4.0, 1: 12.0}))
    per_id = out.groupby("id")["body_length"].first()
    assert per_id.to_dict() == pytest.approx({0: 4.0, 1: 12.0})
    assert out["body_length_frame"].to_numpy() == pytest.approx(
        out["body_length"].to_numpy()
    )


def test_a_bent_body_is_measured_along_the_bend() -> None:
    df = _fish({0: 4.0}, n=1)
    # Fold the tail 90 degrees at the mid-body: the polyline keeps its length,
    # the nose-to-tail chord shrinks to sqrt(2) * 2.
    df.loc[0, "poseX2"] = df.loc[0, "poseX1"]
    df.loc[0, "poseY2"] = df.loc[0, "poseY1"] + 2.0
    out = BodyLength().apply(df)
    assert out["body_length"].iloc[0] == pytest.approx(4.0)
    chord = BodyLength(params={"keypoints": [0, 2]}).apply(df)
    assert chord["body_length"].iloc[0] == pytest.approx(2.0 * np.sqrt(2.0))


def test_keypoints_by_name() -> None:
    out = BodyLength(
        params={
            "keypoints": ["nose", "tail"],
            "keypoint_names": ["nose", "mid", "tail"],
        }
    ).apply(_fish({0: 6.0}))
    assert out["body_length"].iloc[0] == pytest.approx(6.0)


def test_a_missing_point_leaves_the_frame_unmeasured_not_shorter() -> None:
    df = _fish({0: 4.0}, n=3)
    df.loc[1, "poseX2"] = np.nan
    out = BodyLength().apply(df)
    assert out["frame"].tolist() == [0, 2]
    assert out["body_length"].tolist() == pytest.approx([4.0, 4.0])


def test_the_reported_length_is_the_median_over_frames() -> None:
    df = _fish({0: 4.0}, n=5)
    # One frame with a mistracked tail at twice the length.
    df.loc[2, "poseX2"] = df.loc[2, "poseX0"] - 8.0
    out = BodyLength(params={"keypoints": [0, 2]}).apply(df)
    assert out["body_length_frame"].max() == pytest.approx(8.0)
    assert out["body_length"].to_numpy() == pytest.approx(np.full(5, 4.0))


def test_a_length_column_the_tracker_measured() -> None:
    df = _fish({0: 4.0}, n=3).drop(columns=["poseX0", "poseY0"])
    df["body_length"] = [70.0, 77.0, 80.0]
    out = BodyLength(params={"source": "column"}).apply(df)
    assert out["body_length"].iloc[0] == pytest.approx(77.0)


def test_a_centroid_only_table_refuses_with_a_way_out() -> None:
    df = _fish({0: 4.0}).drop(columns=[c for c in _fish({0: 4.0}) if "pose" in c])
    with pytest.raises(ValueError, match="source='column'"):
        BodyLength().apply(df)


def test_a_chain_of_one_keypoint_refuses_at_construction() -> None:
    with pytest.raises(ValueError, match="fewer than two"):
        BodyLength(params={"keypoints": [0]})


def test_a_name_without_names_refuses_at_construction() -> None:
    with pytest.raises(ValueError, match="keypoint_names"):
        BodyLength(params={"keypoints": ["nose", "tail"]})


def test_an_index_past_the_table_refuses() -> None:
    with pytest.raises(ValueError, match="nose|range|index|0..2"):
        BodyLength(params={"keypoints": [0, 7]}).apply(_fish({0: 4.0}))


# --- scale-to-bl ----------------------------------------------------------------


def test_an_explicit_length_divides_every_length_column() -> None:
    df = _fish({0: 4.0})
    out = ScaleToBodyLength(params={"length": 4.0}).apply(df)
    assert out["X_bl"].to_numpy() == pytest.approx(df["X"].to_numpy() / 4.0)
    assert out["poseX2_bl"].to_numpy() == pytest.approx(df["poseX2"].to_numpy() / 4.0)
    assert "time_bl" not in out.columns
    assert "frame_bl" not in out.columns


def test_convert_mode_is_track_shaped() -> None:
    df = _fish({0: 4.0})
    out = ScaleToBodyLength(params={"length": 2.0, "mode": "convert"}).apply(df)
    assert list(out.columns) == list(df.columns)
    assert out["Y"].to_numpy() == pytest.approx(df["Y"].to_numpy() / 2.0)
    assert out["time"].equals(df["time"])


def test_a_run_needs_a_length() -> None:
    with pytest.raises(ValueError, match="needs a length"):
        ScaleToBodyLength()


def test_a_run_takes_one_length() -> None:
    with pytest.raises(ValueError, match="both set"):
        ScaleToBodyLength(
            params={"length": 1.0, "body_length": {"feature": "body-length"}}
        )


def test_a_non_positive_length_refuses() -> None:
    with pytest.raises(ValueError, match="not a usable body length"):
        ScaleToBodyLength(params={"length": 0.0}).apply(_fish({0: 4.0}))


def test_the_length_is_one_per_sequence_from_body_length(tmp_path: Path) -> None:
    """Every individual is divided by the median of the individuals' lengths."""
    df = _fish({0: 4.0, 1: 6.0, 2: 20.0})
    lengths = BodyLength().apply(df)
    path = tmp_path / "s.parquet"
    lengths.to_parquet(path)

    feat = ScaleToBodyLength(params={"body_length": {"feature": "body-length"}})
    assert feat.load_state(tmp_path, {}, {"body_length": {("g", "s"): path}})
    out = feat.apply(df)
    # median(4, 6, 20) = 6 for every individual, so relative positions hold.
    assert out["X_bl"].to_numpy() == pytest.approx(df["X"].to_numpy() / 6.0)


def test_a_missing_body_length_run_refuses_once_at_load(tmp_path: Path) -> None:
    feat = ScaleToBodyLength(params={"body_length": {"feature": "body-length"}})
    with pytest.raises(ValueError, match="resolved to no entries"):
        feat.load_state(tmp_path, {}, {})


def test_an_entry_body_length_did_not_measure_refuses(tmp_path: Path) -> None:
    feat = ScaleToBodyLength(params={"body_length": {"feature": "body-length"}})
    feat.load_state(tmp_path, {}, {"body_length": {("g", "other"): tmp_path / "x"}})
    with pytest.raises(ValueError, match="no output for"):
        feat.apply(_fish({0: 4.0}))
