"""Per-keypoint speed, acceleration and distances, on paths whose answers are known.

``keypoint-kinematics`` differentiates every keypoint coordinate with a
Savitzky-Golay fit. A quadratic path is the sharpest check of that: a fit of
order 2 reproduces it exactly, edges included, so its speed and acceleration
are known to float precision. A helix checks the 3D case and the vertical
velocity, and a loop-per-stretch reference written the slow, obvious way checks
the gap handling, which is where a vectorised implementation can go quietly
wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy.signal import savgol_filter

from mosaic.behavior.feature_library.keypoint_kinematics import KeypointKinematics
from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.tracks_identity import tracks_variant_root
from mosaic.core.pipeline.tracks_index import write_tracks_row
from mosaic.core.schema import ensure_track_schema

type FloatArray = npt.NDArray[np.float64]

FPS = 100.0


def _table(
    keypoints: list[FloatArray],
    *,
    frames: npt.NDArray[np.int64] | None = None,
    fps: float = FPS,
    individual: int = 0,
    sequence: str = "seq",
) -> pd.DataFrame:
    """A one-individual tracks table from ``(n, 2)`` or ``(n, 3)`` keypoint arrays.

    The body centre is the first keypoint; nothing here reads it.
    """
    n = keypoints[0].shape[0]
    frame = np.arange(n, dtype=np.int64) if frames is None else frames
    n_axes = keypoints[0].shape[1]
    columns: dict[str, object] = {
        "frame": frame,
        "time": frame / fps,
        "id": np.full(n, individual, dtype=np.int64),
        "group": [""] * n,
        "sequence": [sequence] * n,
    }
    for axis, name in zip(range(n_axes), "XYZ", strict=False):
        columns[name] = keypoints[0][:, axis]
    for k, points in enumerate(keypoints):
        for axis, name in zip(range(n_axes), "XYZ", strict=False):
            columns[f"pose{name}{k}"] = points[:, axis]
    return pd.DataFrame(columns)


def _apply(table: pd.DataFrame, **params: object) -> pd.DataFrame:
    return KeypointKinematics(params=params).apply(table)


def _walk(n: int, n_axes: int, seed: int = 0) -> FloatArray:
    """A random walk, so no smoothing reproduces it by accident."""
    return np.random.default_rng(seed).normal(size=(n, n_axes)).cumsum(axis=0)


# --- Known answers ---


def test_a_quadratic_path_has_exact_speed_and_acceleration() -> None:
    """x = a t^2, y = b t: velocity (2at, b), acceleration (2a, 0), everywhere."""
    t = np.arange(200) / FPS
    a, b = 3.0, -1.5
    out = _apply(_table([np.column_stack([a * t**2, b * t])]), fps=FPS)
    np.testing.assert_allclose(out["speed__kp0"], np.hypot(2 * a * t, b), rtol=1e-5)
    np.testing.assert_allclose(out["accel__kp0"], 2 * a, rtol=1e-5)
    assert "vz__kp0" not in out.columns


def test_a_helix_has_its_known_speed_acceleration_and_climb() -> None:
    """Radius r at angular rate w, climbing at c: speed hypot(rw, c), accel rw^2."""
    t = np.arange(2000) / FPS
    r, w, c = 0.3, 2 * np.pi * 0.25, 0.05
    helix = np.column_stack([r * np.cos(w * t), r * np.sin(w * t), c * t])
    out = _apply(_table([helix]), fps=FPS)
    np.testing.assert_allclose(out["speed__kp0"], np.hypot(r * w, c), rtol=1e-3)
    np.testing.assert_allclose(out["accel__kp0"], r * w**2, rtol=1e-3)
    np.testing.assert_allclose(out["vz__kp0"], c, rtol=1e-4)


def test_a_falling_keypoint_has_negative_vertical_velocity() -> None:
    """+Z is up, so the sign says which way it moved."""
    t = np.arange(50) / FPS
    falling = np.column_stack([np.zeros_like(t), np.zeros_like(t), 2.0 - 0.7 * t])
    out = _apply(_table([falling]), fps=FPS)
    np.testing.assert_allclose(out["vz__kp0"], -0.7, rtol=1e-5)
    np.testing.assert_allclose(out["speed__kp0"], 0.7, rtol=1e-5)


def test_an_unbroken_track_matches_scipy() -> None:
    """Edges included: the ends are fitted as ``mode="interp"`` fits them."""
    walk = _walk(300, 3)
    out = _apply(_table([walk]), fps=FPS, window=9, polyorder=3)
    velocity = savgol_filter(walk, 9, 3, deriv=1, delta=1 / FPS, axis=0)
    acceleration = savgol_filter(walk, 9, 3, deriv=2, delta=1 / FPS, axis=0)
    np.testing.assert_allclose(
        out["speed__kp0"], np.linalg.norm(velocity, axis=1), rtol=1e-5
    )
    np.testing.assert_allclose(
        out["accel__kp0"], np.linalg.norm(acceleration, axis=1), rtol=1e-5
    )
    np.testing.assert_allclose(out["vz__kp0"], velocity[:, 2], rtol=1e-5, atol=1e-3)


# --- Gaps ---


def _reference(
    frames: npt.NDArray[np.int64],
    points: FloatArray,
    window: int,
    polyorder: int,
    fps: float,
) -> dict[str, FloatArray]:
    """The method written the obvious way, for one individual.

    One Python loop per gap and per stretch, on the individual's own contiguous
    frame range, so nothing in it can reach across to another individual.

    Returns:
        ``speed``, ``accel`` and ``vz`` (the last axis's velocity) on each row.
    """
    span = int(frames[-1] - frames[0]) + 1
    dense = np.full((span, points.shape[1]), np.nan)
    dense[frames - frames[0]] = points
    missing = ~np.isfinite(dense).all(axis=1)
    dense[missing] = np.nan
    i = 0
    while i < span:
        if not missing[i]:
            i += 1
            continue
        j = i
        while j < span and missing[j]:
            j += 1
        if i > 0 and j < span and j - i <= window:
            for row in range(i, j):
                share = (row - (i - 1)) / (j - (i - 1))
                dense[row] = dense[i - 1] + share * (dense[j] - dense[i - 1])
        i = j
    filled_missing = ~np.isfinite(dense).all(axis=1)
    velocity = np.full_like(dense, np.nan)
    acceleration = np.full_like(dense, np.nan)
    i = 0
    while i < span:
        if filled_missing[i]:
            i += 1
            continue
        j = i
        while j < span and not filled_missing[j]:
            j += 1
        if j - i >= window:
            stretch = dense[i:j]
            delta = 1 / fps
            velocity[i:j] = savgol_filter(stretch, window, polyorder, 1, delta, axis=0)
            acceleration[i:j] = savgol_filter(
                stretch, window, polyorder, 2, delta, axis=0
            )
        i = j
    rows = frames - frames[0]
    absent = ~np.isfinite(points).all(axis=1)
    out = {
        "speed": np.linalg.norm(velocity[rows], axis=1),
        "accel": np.linalg.norm(acceleration[rows], axis=1),
        "vz": velocity[rows, -1],
    }
    for values in out.values():
        values[absent] = np.nan
    return out


def _assert_matches_reference(
    out: pd.DataFrame, frames: npt.NDArray[np.int64], points: FloatArray, **kw: int
) -> None:
    window, polyorder = kw.get("window", 7), kw.get("polyorder", 2)
    expected = _reference(frames, points, window, polyorder, FPS)
    np.testing.assert_array_equal(out["frame"], frames)
    names = ["speed", "accel"] + (["vz"] if points.shape[1] == 3 else [])
    for name in names:
        got = out[f"{name}__kp0"].to_numpy(dtype=np.float64)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(expected[name]), name)
        np.testing.assert_allclose(got, expected[name], rtol=1e-4, atol=1e-6)


def _gappy(n: int, n_axes: int, seed: int) -> tuple[npt.NDArray[np.int64], FloatArray]:
    """A walk with NaN rows of every length and rows absent from the table."""
    rng = np.random.default_rng(seed)
    points = _walk(n, n_axes, seed=seed)
    for start, length in [(20, 1), (50, 4), (90, 7), (130, 8), (200, 30), (400, 2)]:
        points[start : start + length] = np.nan
    points[300:304, 1] = np.nan  # one axis missing is the keypoint missing
    points[310, 0] = np.inf  # not finite is missing too
    keep = np.ones(n, dtype=bool)
    keep[rng.choice(np.arange(5, n - 5), size=40, replace=False)] = False
    keep[450:458] = False  # eight absent frames: one more than the window
    keep[500:507] = False  # seven: bridged
    return np.arange(n, dtype=np.int64)[keep], points[keep]


def test_gaps_and_dropped_frames_match_the_per_stretch_reference() -> None:
    frames, points = _gappy(600, 2, seed=4)
    out = _apply(_table([points], frames=frames), fps=FPS)
    assert len(out) == len(frames)
    _assert_matches_reference(out, frames, points)


def test_interleaved_individuals_each_match_the_reference_in_3d() -> None:
    """Shuffled rows, different frame ranges, one range abutting the next.

    The individuals are filtered in one pass along one axis, so a boundary
    between two of them that let a window or an interpolation reach across
    would show here, as would an individual's rows placed at another's frames.
    """
    gappy_frames, gappy_points = _gappy(600, 3, seed=5)
    tracks: dict[int, tuple[npt.NDArray[np.int64], FloatArray]] = {
        # Individual 4 starts on the frame after individual 2 ends.
        4: (gappy_frames + 100, gappy_points),
        2: (np.arange(0, 100, dtype=np.int64), _walk(100, 3, seed=6) + 50.0),
        9: (np.arange(0, 5, dtype=np.int64), _walk(5, 3, seed=7)),
        1: (np.arange(40, 90, dtype=np.int64), np.full((50, 3), np.nan)),
    }
    tables = [
        _table([points], frames=frames, individual=individual)
        for individual, (frames, points) in tracks.items()
    ]
    table = pd.concat(tables).sample(frac=1.0, random_state=0)
    out = _apply(table, fps=FPS, window=9, polyorder=3)
    assert out["id"].tolist() == sorted(out["id"].tolist())
    for individual, (frames, points) in tracks.items():
        mine = out[out["id"] == individual]
        _assert_matches_reference(mine, frames, points, window=9, polyorder=3)
    assert out.loc[out["id"] == 9, "speed__kp0"].isna().all()
    assert out.loc[out["id"] == 1, "speed__kp0"].isna().all()


def test_absent_frames_longer_than_the_window_split_the_track() -> None:
    """Rows missing from the table, not NaN rows: the gap is measured in frames."""
    t = np.arange(100) / FPS
    line = np.column_stack([0.4 * t, 0.3 * t])
    # Eight frames absent, then five rows, then seventeen absent.
    frames = np.r_[0:30, 38:43, 60:100].astype(np.int64)
    table = _table([line[frames]], frames=frames)
    split = _apply(table, fps=FPS).set_index("frame")["speed__kp0"]
    assert split.loc[38:42].isna().all()
    np.testing.assert_allclose(split.drop(range(38, 43)), 0.5, rtol=1e-5)
    bridged = _apply(table, fps=FPS, window=9).set_index("frame")["speed__kp0"]
    np.testing.assert_allclose(bridged.loc[:42], 0.5, rtol=1e-5)
    assert bridged.loc[38:42].notna().all()


def test_a_short_gap_is_bridged_and_still_reported_missing() -> None:
    """At constant velocity the bridge is exact, so every measured row is too."""
    t = np.arange(100) / FPS
    line = np.column_stack([0.4 * t, 0.3 * t])
    line[40:43] = np.nan
    frames = np.delete(np.arange(100, dtype=np.int64), [60, 61])
    out = _apply(_table([np.delete(line, [60, 61], axis=0)], frames=frames), fps=FPS)
    assert 60 not in out["frame"].to_numpy()
    measured = out[~out["frame"].between(40, 42)]
    np.testing.assert_allclose(measured["speed__kp0"], 0.5, rtol=1e-5)
    np.testing.assert_allclose(measured["accel__kp0"], 0.0, atol=1e-3)
    assert out.loc[out["frame"].between(40, 42), "speed__kp0"].isna().all()


def test_a_long_gap_splits_the_track_and_a_short_stretch_has_no_estimate() -> None:
    t = np.arange(100) / FPS
    line = np.column_stack([0.4 * t, 0.3 * t])
    line[30:38] = np.nan  # one longer than the window: not bridged
    line[43:60] = np.nan  # leaves 38..42, five rows, shorter than the window
    out = _apply(_table([line]), fps=FPS)
    speed = out.set_index("frame")["speed__kp0"]
    np.testing.assert_allclose(speed.loc[0:29], 0.5, rtol=1e-5)
    np.testing.assert_allclose(speed.loc[60:], 0.5, rtol=1e-5)
    assert speed.loc[30:59].isna().all()


def test_the_window_sets_the_longest_bridged_gap() -> None:
    t = np.arange(100) / FPS
    line = np.column_stack([0.4 * t, 0.3 * t])
    line[30:38] = np.nan  # eight rows
    speed = _apply(_table([line]), fps=FPS, window=9).set_index("frame")["speed__kp0"]
    np.testing.assert_allclose(speed.drop(range(30, 38)), 0.5, rtol=1e-5)


# --- Distances, names and normalization ---


def test_distances_in_two_and_three_dimensions() -> None:
    n = 20
    origin2, origin3 = np.zeros((n, 2)), np.zeros((n, 3))
    out2 = _apply(
        _table([origin2, np.tile([3.0, 4.0], (n, 1))]), fps=FPS, distances=[(0, 1)]
    )
    out3 = _apply(
        _table([origin3, np.tile([1.0, 2.0, 2.0], (n, 1))]),
        fps=FPS,
        distances=[(1, 0)],
    )
    np.testing.assert_allclose(out2["dist__kp0__kp1"], 5.0)
    np.testing.assert_allclose(out3["dist__kp1__kp0"], 3.0)


def test_a_distance_is_missing_where_either_keypoint_is() -> None:
    n = 20
    far = np.tile([3.0, 4.0], (n, 1))
    far[5] = np.nan
    out = _apply(_table([np.zeros((n, 2)), far]), fps=FPS, distances=[(0, 1)])
    assert np.isnan(out["dist__kp0__kp1"][5])
    assert np.isfinite(out["dist__kp0__kp1"].drop(5)).all()


def test_names_label_the_columns_and_resolve_references() -> None:
    t = np.arange(40) / FPS
    still = np.zeros((40, 3))
    moving = np.column_stack([0.6 * t, np.zeros_like(t), 0.8 * t])
    out = _apply(
        _table([still, moving, still + [0.0, 0.5, 0.0]]),
        fps=FPS,
        keypoint_names=["nose", "left_wrist", "right_wrist"],
        keypoints=["left_wrist"],
        distances=[("left_wrist", 2)],
    )
    measures = [c for c in out.columns if "__" in c]
    assert measures == [
        "speed__left_wrist",
        "accel__left_wrist",
        "vz__left_wrist",
        "dist__left_wrist__right_wrist",
    ]
    np.testing.assert_allclose(out["speed__left_wrist"], 1.0, rtol=1e-5)
    np.testing.assert_allclose(out["vz__left_wrist"], 0.8, rtol=1e-5)
    expected = np.hypot(0.6 * t, np.hypot(0.5, 0.8 * t))
    np.testing.assert_allclose(
        out["dist__left_wrist__right_wrist"], expected, rtol=1e-5
    )


def test_an_empty_keypoint_list_reports_only_distances() -> None:
    n = 10
    out = _apply(
        _table([np.zeros((n, 2)), np.ones((n, 2))]),
        fps=FPS,
        keypoints=[],
        distances=[(0, 1)],
    )
    assert [c for c in out.columns if "__" in c] == ["dist__kp0__kp1"]


def test_normalizing_by_a_segment_makes_two_body_sizes_agree() -> None:
    """The same motion at three times the size: identical once normalized."""
    walk = _walk(200, 3, seed=7)
    head = walk + [0.0, 0.0, 0.2]
    small = _table([walk, head])
    large = _table([3 * walk, 3 * head])
    params: dict[str, object] = {"fps": FPS, "distances": [(0, 1)]}
    raw_small = _apply(small, **params)
    norm_small = _apply(small, normalize_by=(0, 1), **params)
    norm_large = _apply(large, normalize_by=(0, 1), **params)
    measures = [c for c in raw_small.columns if "__" in c]
    np.testing.assert_allclose(norm_small[measures], norm_large[measures], rtol=1e-4)
    np.testing.assert_allclose(norm_small["dist__kp0__kp1"], 1.0, rtol=1e-5)
    np.testing.assert_allclose(
        norm_small["speed__kp0"], raw_small["speed__kp0"] / 0.2, rtol=1e-5
    )


def test_each_individual_is_normalized_by_its_own_segment() -> None:
    """Two body sizes in one table: one divisor for both would leave them apart."""
    walk = _walk(200, 3, seed=8)
    head = walk + [0.0, 0.0, 0.2]
    small = _table([walk, head], individual=1)
    large = _table([3 * walk, 3 * head], individual=2)
    out = _apply(pd.concat([small, large]), fps=FPS, normalize_by=(0, 1))
    one, two = (out[out["id"] == i].reset_index(drop=True) for i in (1, 2))
    measures = [c for c in out.columns if "__" in c]
    np.testing.assert_allclose(one[measures], two[measures], rtol=1e-4)
    # The sign of a vertical velocity survives the division.
    alone = _apply(small, fps=FPS)
    np.testing.assert_allclose(one["vz__kp0"], alone["vz__kp0"] / 0.2, rtol=1e-4)


def test_an_individual_never_showing_the_segment_is_nan_when_normalized() -> None:
    n = 30
    hidden = np.full((n, 2), np.nan)
    out = _apply(_table([_walk(n, 2), hidden]), fps=FPS, normalize_by=(0, 1))
    assert out["speed__kp0"].isna().all()


# --- Rows, individuals and the frame rate ---


def test_individuals_are_measured_apart_and_come_back_in_order() -> None:
    first = _table([_walk(80, 2, seed=1)], individual=1)
    second = _table([_walk(80, 2, seed=2)], individual=2)
    alone = pd.concat([_apply(first, fps=FPS), _apply(second, fps=FPS)])
    shuffled = pd.concat([second, first]).sample(frac=1.0, random_state=0)
    together = _apply(shuffled, fps=FPS)
    pd.testing.assert_frame_equal(together, alone.reset_index(drop=True))


def test_the_measures_are_float32_and_identity_is_carried() -> None:
    out = _apply(_table([_walk(30, 3)]), fps=FPS)
    assert {"frame", "time", "id", "group", "sequence"} <= set(out.columns)
    assert all(out[c].dtype == np.float32 for c in out.columns if "__" in c)


def test_the_frame_rate_is_read_off_the_time_column() -> None:
    """Dropped frames do not distort it: the step is per frame, not per row."""
    frames = np.delete(np.arange(120, dtype=np.int64), [10, 11, 50])
    table = _table([_walk(frames.size, 2)], frames=frames, fps=50.0)
    pd.testing.assert_frame_equal(_apply(table), _apply(table, fps=50.0))


def test_without_a_frame_rate_or_a_time_column_it_refuses() -> None:
    table = _table([_walk(30, 2)]).drop(columns="time")
    with pytest.raises(ValueError, match="keypoint-kinematics needs a frame rate"):
        _ = _apply(table)


def test_a_time_column_that_never_advances_refuses() -> None:
    table = _table([_walk(30, 2)])
    table["time"] = 0.0
    with pytest.raises(ValueError, match="could not read a frame rate"):
        _ = _apply(table)


def test_a_repeated_frame_refuses() -> None:
    table = _table([_walk(30, 2)])
    with pytest.raises(ValueError, match="repeated frames"):
        _ = _apply(pd.concat([table, table.iloc[[4]]]), fps=FPS)


# --- Refusals ---


def test_a_table_without_keypoints_refuses() -> None:
    table = _table([_walk(30, 2)])
    table = table.drop(columns=["poseX0", "poseY0"])
    with pytest.raises(ValueError, match="keypoint-kinematics needs pose keypoint"):
        _ = _apply(table, fps=FPS)


def test_a_table_with_a_z_axis_and_planar_keypoints_refuses() -> None:
    table = _table([_walk(30, 2)])
    table["Z"] = 0.0
    with pytest.raises(ValueError, match="no poseZ column"):
        _ = _apply(table, fps=FPS)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"window": 8}, "must be odd"),
        ({"window": 3, "polyorder": 3}, "must exceed"),
        ({"polyorder": 1}, "greater than or equal to 2"),
        ({"keypoints": ["wrist"]}, "not given keypoint names"),
        ({"keypoint_names": ["a", "b"], "keypoints": ["c"]}, "no keypoint named"),
        ({"keypoint_names": ["a", "a"]}, "repeats a name"),
        ({"keypoint_names": ["left__wrist"]}, "contain '__'"),
        ({"distances": [(0, -1)]}, "negative"),
        ({"keypoints": []}, "nothing to compute"),
        ({"distances": [(0, 0)]}, "a keypoint and itself"),
        ({"distances": [(0, 1), (1, 0)]}, "requested twice"),
        ({"normalize_by": (1, 1)}, "a keypoint and itself"),
        ({"keypoints": [1, 1]}, "names a keypoint twice"),
        # An index and a name are one keypoint where they agree.
        ({"keypoint_names": ["a", "b"], "keypoints": [0, "a"]}, "twice"),
        ({"keypoint_names": ["a", "b"], "distances": [(0, "b"), ("a", 1)]}, "twice"),
        ({"keypoint_names": ["a", "b"], "normalize_by": ("b", 1)}, "itself"),
        # The names fix the table's keypoint count, and so bound every index.
        ({"keypoint_names": ["a", "b"], "keypoints": [2]}, "outside"),
    ],
)
def test_a_parameter_no_table_could_satisfy_is_refused_at_construction(
    params: dict[str, object], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        _ = KeypointKinematics(params=params)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"keypoints": [2]}, "outside the table's 2 keypoints"),
        ({"distances": [(0, 5)]}, "outside the table's 2 keypoints"),
        ({"normalize_by": (0, 2)}, "outside the table's 2 keypoints"),
        ({"keypoint_names": ["a", "b", "c"]}, "has 3 names"),
    ],
)
def test_a_parameter_this_table_cannot_satisfy_is_refused_at_apply(
    params: dict[str, object], match: str
) -> None:
    feature = KeypointKinematics(params={"fps": FPS, **params})
    table = _table([_walk(30, 2), _walk(30, 2, seed=1)])
    with pytest.raises(ValueError, match=match):
        _ = feature.apply(table)


# --- Through run_feature ---


def _dataset(base: Path, table: pd.DataFrame, std_format: str) -> Dataset:
    """A dataset holding one entry, ``seq``, whose tracks row records *std_format*."""
    _ = ensure_track_schema(table, std_format, strict=True)
    manifest = new_dataset_manifest(name="kinematics", base_dir=base / "dataset")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    run_id = "convert-x.0.1-aaaaaaaaaa"
    root = tracks_variant_root(dataset.get_root("tracks"), run_id)
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{make_entry_key('', 'seq')}.parquet"
    table.to_parquet(out)
    write_tracks_row(
        dataset,
        run_id=run_id,
        group="",
        sequence="seq",
        out_path=out,
        producer="convert-x",
        std_format=std_format,
        n_rows=len(table),
    )
    return dataset


def _run_output(dataset: Dataset, feature: KeypointKinematics) -> pd.DataFrame:
    result = run_feature(dataset, feature)
    assert result.run_id
    parquets = sorted(
        feature_run_root(dataset, result.feature, result.run_id).glob("*.parquet")
    )
    assert len(parquets) == 1
    return pd.read_parquet(parquets[0])


def test_run_feature_measures_a_table_with_a_z_axis(tmp_path: Path) -> None:
    """A 3D table is measured in its own unit; here metres, so metres per second."""
    t = np.arange(200) / FPS
    wrist = np.column_stack([0.3 * t, np.zeros_like(t), 1.0 + 0.4 * t])
    elbow = wrist + [0.0, 0.0, -0.25]
    dataset = _dataset(tmp_path, _table([wrist, elbow]), "mosaic_v1")
    out = _run_output(
        dataset,
        KeypointKinematics(
            params={"keypoint_names": ["wrist", "elbow"], "distances": [(0, 1)]}
        ),
    )
    np.testing.assert_allclose(out["speed__wrist"], 0.5, rtol=1e-4)
    np.testing.assert_allclose(out["vz__elbow"], 0.4, rtol=1e-4)
    np.testing.assert_allclose(out["dist__wrist__elbow"], 0.25, rtol=1e-5)
    assert len(out) == 200


def test_run_feature_measures_a_pixel_table(tmp_path: Path) -> None:
    t = np.arange(100) / FPS
    path = np.column_stack([30.0 * t, 40.0 * t])
    dataset = _dataset(tmp_path, _table([path]), "mosaic_v1")
    out = _run_output(dataset, KeypointKinematics(params={"fps": FPS}))
    np.testing.assert_allclose(out["speed__kp0"], 50.0, rtol=1e-5)
    assert "vz__kp0" not in out.columns
