"""``pose-egocentric`` removes position and heading and keeps posture.

Every test here builds a body whose answer is known before the feature runs: a
fixed skeleton in its own frame, turned by a chosen yaw and set down at a chosen
point. The egocentric coordinates must then come back as the skeleton, the yaw
as the yaw and the origin as the point, whatever the pose was placed at -- which
is the whole claim the feature makes, and the one a clustering of its output
depends on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.pose_egocentric import PoseEgocentric
from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.track_space import TrackSpaceRefused, feature_track_spaces
from mosaic.core.pipeline.tracks_identity import tracks_variant_root
from mosaic.core.pipeline.tracks_index import write_tracks_row

# A small quadruped in its own frame: +x forward, +y its left, +z up, metres.
# The hips' midpoint is the origin, so each keypoint's egocentric coordinates
# are exactly its row here.
SKELETON = np.array(
    [
        [0.00, 0.03, 0.00],  # 0 left hip
        [0.00, -0.03, 0.00],  # 1 right hip
        [0.12, 0.00, 0.02],  # 2 nose
        [0.08, 0.025, 0.01],  # 3 left shoulder
        [0.08, -0.025, 0.01],  # 4 right shoulder
        [-0.06, 0.00, 0.00],  # 5 tail base
    ]
)
NAMES = ["hip_l", "hip_r", "nose", "shoulder_l", "shoulder_r", "tail"]
BASE_PARAMS: dict[str, object] = {"origin": [0, 1], "lateral": [0, 1]}


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _place(skeleton: np.ndarray, yaw: float, at: np.ndarray) -> np.ndarray:
    """The skeleton turned by *yaw* about world up and set down at *at*."""
    return skeleton @ _yaw_matrix(yaw).T + at


def _table(
    points: np.ndarray, *, identity: int = 0, sequence: str = "seq"
) -> pd.DataFrame:
    """A ``mosaic_world_v1`` table for one individual, from ``(n, K, 3)`` points."""
    n, n_keypoints, _ = points.shape
    frame = np.arange(n, dtype=np.int64)
    columns: dict[str, object] = {
        "frame": frame,
        "time": frame / 30.0,
        "id": np.full(n, identity, dtype=np.int64),
        "group": [""] * n,
        "sequence": [sequence] * n,
        "X": points[:, :, 0].mean(axis=1),
        "Y": points[:, :, 1].mean(axis=1),
        "Z": points[:, :, 2].mean(axis=1),
    }
    for k in range(n_keypoints):
        for a, axis in enumerate("XYZ"):
            columns[f"pose{axis}{k}"] = points[:, k, a]
    return pd.DataFrame(columns)


def _still(yaw: float, at: tuple[float, float, float], n: int = 4) -> pd.DataFrame:
    """The skeleton held still at one yaw and one place for *n* frames."""
    placed = _place(SKELETON, yaw, np.asarray(at))
    return _table(np.repeat(placed[None], n, axis=0))


def _ego(out: pd.DataFrame, k: int) -> np.ndarray:
    """Keypoint *k*'s ``(n, 3)`` egocentric coordinates."""
    return out[[f"ego_x{k}", f"ego_y{k}", f"ego_z{k}"]].to_numpy(dtype=np.float64)


def _wrapped(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


# --- the body frame -----------------------------------------------------------


@pytest.mark.parametrize(
    "yaw", [0.0, 0.7, np.pi / 2, 2.5, -1.2, np.pi - 1e-3, -np.pi + 1e-3]
)
def test_a_turned_and_moved_body_comes_back_as_its_skeleton(yaw: float) -> None:
    at = (1.5, -0.4, 0.25)
    out = PoseEgocentric(params=BASE_PARAMS).apply(_still(yaw, at))

    for k in range(len(SKELETON)):
        np.testing.assert_allclose(
            _ego(out, k), np.tile(SKELETON[k], (4, 1)), atol=1e-6
        )
    np.testing.assert_allclose(
        _wrapped(out["body_yaw"].to_numpy() - yaw), 0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        out[["origin_x", "origin_y", "origin_z"]].to_numpy(),
        np.tile(at, (4, 1)),
        atol=1e-6,
    )


def test_one_posture_in_two_places_and_headings_is_one_posture() -> None:
    first = PoseEgocentric(params=BASE_PARAMS).apply(_still(0.3, (0.0, 0.0, 0.0)))
    second = PoseEgocentric(params=BASE_PARAMS).apply(_still(-2.4, (3.0, 1.0, 0.5)))
    ego = [c for c in first.columns if c.startswith("ego_")]
    np.testing.assert_allclose(first[ego].to_numpy(), second[ego].to_numpy(), atol=1e-6)


def test_a_body_travelling_a_helix_holds_its_posture() -> None:
    """Facing along its path while climbing: the world moves, the posture does not."""
    n, radius, omega, climb = 200, 0.5, 0.05, 0.002
    phase = omega * np.arange(n)
    path = np.column_stack(
        [radius * np.cos(phase), radius * np.sin(phase), climb * np.arange(n)]
    )
    yaw = phase + np.pi / 2
    points = np.stack([_place(SKELETON, yaw[i], path[i]) for i in range(n)])
    out = PoseEgocentric(params=BASE_PARAMS).apply(_table(points))

    for k in range(len(SKELETON)):
        np.testing.assert_allclose(
            _ego(out, k), np.tile(SKELETON[k], (n, 1)), atol=1e-6
        )
    np.testing.assert_allclose(
        _wrapped(out["body_yaw"].to_numpy() - yaw), 0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        out[["origin_x", "origin_y", "origin_z"]].to_numpy(), path, atol=1e-6
    )


def test_the_frame_is_right_handed_with_forward_as_y_cross_z() -> None:
    """Left along world +Y puts forward along world +X, not -X."""
    out = PoseEgocentric(params=BASE_PARAMS).apply(_still(0.0, (0.0, 0.0, 0.0)))
    assert out["ego_x2"].iloc[0] == pytest.approx(0.12, abs=1e-6)
    assert out["ego_y0"].iloc[0] == pytest.approx(0.03, abs=1e-6)
    assert out["body_yaw"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_rearing_stays_in_the_posture() -> None:
    """The frame turns about world up only, so a raised nose reads as raised."""
    pitch = 0.6
    c, s = np.cos(pitch), np.sin(pitch)
    nose_up = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    reared = SKELETON @ nose_up.T
    placed = _place(reared, 1.1, np.array([0.2, 0.3, 0.0]))
    out = PoseEgocentric(params=BASE_PARAMS).apply(_table(placed[None]))

    np.testing.assert_allclose(_ego(out, 2)[0], reared[2], atol=1e-6)
    assert out["ego_z2"].iloc[0] > SKELETON[2, 2]
    assert out["body_yaw"].iloc[0] == pytest.approx(1.1, abs=1e-6)


def test_a_rolled_body_takes_its_left_from_the_horizontal_part_of_the_pair() -> None:
    """Lifting the left hip tilts the pair; the yaw is the same, the height shows."""
    rolled = SKELETON.copy()
    rolled[0, 2] = 0.02
    out = PoseEgocentric(params=BASE_PARAMS).apply(
        _table(_place(rolled, -0.8, np.zeros(3))[None])
    )
    assert out["body_yaw"].iloc[0] == pytest.approx(-0.8, abs=1e-6)
    np.testing.assert_allclose(_ego(out, 0)[0], [0.0, 0.03, 0.01], atol=1e-6)


# --- where it is undefined ----------------------------------------------------


def _on_its_side() -> np.ndarray:
    """The skeleton rolled a quarter turn about its forward axis, left hip up.

    ``cos(pi / 2)`` is not zero in floating point, so the hips keep a horizontal
    separation of about ``1e-18`` m: rounding error, which a direction normalised
    from it would report as a heading.
    """
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    roll = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return SKELETON @ roll.T


def _upright() -> np.ndarray:
    """The skeleton with its hips stacked exactly one above the other."""
    upright = SKELETON.copy()
    upright[0] = [0.0, 0.0, 0.03]
    upright[1] = [0.0, 0.0, -0.03]
    return upright


@pytest.mark.parametrize("body", [_upright(), _on_its_side()], ids=["exact", "rolled"])
def test_a_vertical_lateral_pair_defines_no_heading(body: np.ndarray) -> None:
    assert abs(body[0, 1] - body[1, 1]) < 1e-15  # the pair is vertical
    out = PoseEgocentric(params=BASE_PARAMS).apply(_table(body[None]))
    ego = [c for c in out.columns if c.startswith("ego_")]
    assert out[ego].isna().all(axis=None)
    assert np.isnan(out["body_yaw"].iloc[0])
    # The origin is still where the hips are; only the heading is undefined.
    np.testing.assert_allclose(
        out[["origin_x", "origin_y", "origin_z"]].iloc[0], 0.0, atol=1e-6
    )


def test_a_body_facing_exactly_backwards_has_one_yaw() -> None:
    """``-pi`` whichever sign the zero in its lateral vector carries.

    Facing world -X, the hips differ in ``x`` by a zero, and ``arctan2`` reads
    ``+0.0`` and ``-0.0`` as the two ends of the circle. One heading spelled two
    ways is two values to anything that clusters or differences it.
    """
    backwards = _place(SKELETON, np.pi, np.zeros(3))
    points = np.repeat(backwards[None], 2, axis=0)
    points[:, [0, 1], 0] = 0.0
    points[1, 0, 0] = -0.0
    out = PoseEgocentric(params=BASE_PARAMS).apply(_table(points))
    yaw = out["body_yaw"].to_numpy(dtype=np.float64)
    assert yaw[0] == yaw[1]
    assert yaw[0] == pytest.approx(-np.pi, abs=1e-6)


def test_a_zero_lateral_vector_defines_no_heading() -> None:
    collapsed = SKELETON.copy()
    collapsed[1] = collapsed[0]
    out = PoseEgocentric(params=BASE_PARAMS).apply(_table(collapsed[None]))
    assert out.filter(like="ego_").isna().all(axis=None)
    assert np.isnan(out["body_yaw"].iloc[0])


def test_a_missing_origin_keypoint_leaves_the_posture_and_origin_undefined() -> None:
    """Never the remaining hip as the origin: that would read as a lurch sideways."""
    table = _still(0.4, (1.0, 1.0, 0.0), n=3)
    table.loc[1, ["poseX0", "poseY0", "poseZ0"]] = np.nan
    out = PoseEgocentric(params={"origin": [0, 1], "lateral": [3, 4]}).apply(table)

    assert out.filter(like="ego_").iloc[1].isna().all()
    assert out[["origin_x", "origin_y", "origin_z"]].iloc[1].isna().all()
    # The shoulders still fix which way the body faces.
    assert out["body_yaw"].iloc[1] == pytest.approx(0.4, abs=1e-6)
    assert out.filter(like="ego_").iloc[[0, 2]].notna().all(axis=None)


def test_a_keypoint_missing_one_axis_is_missing() -> None:
    table = _still(0.0, (0.0, 0.0, 0.0), n=2)
    table.loc[0, "poseZ2"] = np.nan
    out = PoseEgocentric(params=BASE_PARAMS).apply(table)
    assert out[["ego_x2", "ego_y2", "ego_z2"]].iloc[0].isna().all()
    np.testing.assert_allclose(_ego(out, 2)[1], SKELETON[2], atol=1e-6)
    np.testing.assert_allclose(_ego(out, 5)[0], SKELETON[5], atol=1e-6)


# --- the output ---------------------------------------------------------------


def test_the_output_is_one_float32_row_per_input_row_sorted_by_id_and_frame() -> None:
    first = _still(0.0, (0.0, 0.0, 0.0), n=3)
    second = _still(1.0, (1.0, 0.0, 0.0), n=3)
    second["id"] = 1
    table = pd.concat([second, first], ignore_index=True).iloc[::-1]
    out = PoseEgocentric(
        params={**BASE_PARAMS, "keypoints": [2, "tail"], "keypoint_names": NAMES}
    ).apply(table)

    assert len(out) == len(table)
    assert list(zip(out["id"], out["frame"], strict=True)) == [
        (i, f) for i in (0, 1) for f in range(3)
    ]
    measured = [
        "ego_x2",
        "ego_y2",
        "ego_z2",
        "ego_x5",
        "ego_y5",
        "ego_z5",
        "body_yaw",
        "origin_x",
        "origin_y",
        "origin_z",
    ]
    assert [
        c for c in out.columns if c not in {"frame", "time", "id", "group", "sequence"}
    ] == measured
    assert set(out.dtypes[measured]) == {np.dtype(np.float32)}
    assert {"frame", "time", "id", "group", "sequence"} <= set(out.columns)


def test_interleaved_shuffled_rows_keep_their_own_values() -> None:
    """Each output row is computed from its own input row, whatever the order.

    Three individuals at irregular frames, each turned and placed differently on
    every frame, handed over in a random order. A value that followed the sort
    to a neighbouring row would read as that neighbour's heading.
    """
    rng = np.random.default_rng(7)
    tables: list[pd.DataFrame] = []
    for identity in range(3):
        frames = np.sort(rng.choice(200, size=30, replace=False))
        yaws = 0.05 * frames + 2.0 * identity
        at = np.column_stack(
            [np.cos(0.1 * frames), np.full(30, identity), 0.01 * frames]
        )
        points = np.stack(
            [_place(SKELETON, yaw, row) for yaw, row in zip(yaws, at, strict=True)]
        )
        table = _table(points, identity=identity)
        table["frame"] = frames
        table["time"] = frames / 30.0
        tables.append(table)
    shuffled = pd.concat(tables, ignore_index=True).sample(frac=1.0, random_state=3)
    out = PoseEgocentric(params=BASE_PARAMS).apply(shuffled)

    frame = out["frame"].to_numpy(dtype=np.float64)
    identity = out["id"].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(
        _wrapped(out["body_yaw"].to_numpy() - (0.05 * frame + 2.0 * identity)),
        0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(out["origin_x"], np.cos(0.1 * frame), atol=1e-6)
    np.testing.assert_allclose(out["origin_y"], identity, atol=1e-6)
    for k in range(len(SKELETON)):
        np.testing.assert_allclose(
            _ego(out, k), np.tile(SKELETON[k], (len(out), 1)), atol=1e-6
        )


def test_by_default_every_keypoint_is_emitted() -> None:
    out = PoseEgocentric(params=BASE_PARAMS).apply(_still(0.0, (0.0, 0.0, 0.0)))
    assert {f"ego_{a}{k}" for a in "xyz" for k in range(len(SKELETON))} <= set(
        out.columns
    )


def test_keypoints_are_named_by_their_column_number_across_a_gap() -> None:
    """A table without keypoint 1 still calls keypoint 2 ``ego_x2``."""
    table = _still(0.0, (0.0, 0.0, 0.0)).drop(columns=["poseX1", "poseY1", "poseZ1"])
    out = PoseEgocentric(params={"origin": [0], "lateral": [3, 4]}).apply(table)
    assert "ego_x2" in out.columns
    assert "ego_x1" not in out.columns
    assert out["ego_x2"].iloc[0] == pytest.approx(0.12, abs=1e-6)


def test_names_and_indices_resolve_to_the_same_output() -> None:
    by_index = PoseEgocentric(
        params={"origin": [0, 1], "lateral": [0, 1], "keypoints": [2]}
    )
    by_name = PoseEgocentric(
        params={
            "keypoint_names": NAMES,
            "origin": ["hip_l", "hip_r"],
            "lateral": ["hip_l", 1],
            "keypoints": ["nose"],
        }
    )
    table = _still(2.0, (0.5, 0.5, 0.1))
    pd.testing.assert_frame_equal(by_index.apply(table), by_name.apply(table))


# --- scale_by -----------------------------------------------------------------


def test_scale_by_divides_by_the_hip_to_shoulder_distance() -> None:
    params = {**BASE_PARAMS, "scale_by": [[0, 1], [3, 4]]}
    out = PoseEgocentric(params=params).apply(_still(0.9, (1.0, 2.0, 0.0)))
    body = np.linalg.norm(SKELETON[[3, 4]].mean(axis=0) - SKELETON[[0, 1]].mean(axis=0))
    np.testing.assert_allclose(
        _ego(out, 2), np.tile(SKELETON[2] / body, (4, 1)), atol=1e-6
    )
    # Context is not posture and is not scaled.
    np.testing.assert_allclose(out["origin_x"], 1.0, atol=1e-6)


def test_scale_by_makes_two_sizes_of_one_posture_agree() -> None:
    small = _still(0.0, (0.0, 0.0, 0.0))
    large = _table(
        np.repeat(_place(SKELETON * 1.8, 2.0, np.ones(3))[None], 4, axis=0), identity=1
    )
    params = {
        **BASE_PARAMS,
        "scale_by": [["hip_l", "hip_r"], ["shoulder_l", "shoulder_r"]],
        "keypoint_names": NAMES,
    }
    out = PoseEgocentric(params=params).apply(
        pd.concat([small, large], ignore_index=True)
    )
    ego = [c for c in out.columns if c.startswith("ego_")]
    np.testing.assert_allclose(
        out.loc[out["id"] == 0, ego].to_numpy(),
        out.loc[out["id"] == 1, ego].to_numpy(),
        atol=1e-6,
    )


def test_the_body_size_is_measured_per_sequence() -> None:
    """One id in two sequences is two measurements, not one median over both.

    The same id number in two recordings need not be the same animal, and a
    median over both would scale one recording by the other's body.
    """
    small = _still(0.0, (0.0, 0.0, 0.0))
    large = _table(
        np.repeat(_place(SKELETON * 1.8, 2.0, np.ones(3))[None], 4, axis=0),
        sequence="seq_b",
    )
    params = {**BASE_PARAMS, "scale_by": [[0, 1], [3, 4]]}
    out = PoseEgocentric(params=params).apply(
        pd.concat([large, small], ignore_index=True)
    )
    ego = [c for c in out.columns if c.startswith("ego_")]
    np.testing.assert_allclose(
        out.loc[out["sequence"] == "seq", ego].to_numpy(),
        out.loc[out["sequence"] == "seq_b", ego].to_numpy(),
        atol=1e-6,
    )


def test_the_body_size_is_a_median_and_does_not_follow_one_frame() -> None:
    """A per-row divisor would cancel a stretch; the median keeps it visible."""
    points = np.repeat(SKELETON[None], 5, axis=0).copy()
    points[2, [3, 4], 0] = 0.16  # one frame stretched: shoulders twice as far forward
    params = {**BASE_PARAMS, "scale_by": [[0, 1], [3, 4]]}
    out = PoseEgocentric(params=params).apply(_table(points))
    body = np.linalg.norm(SKELETON[[3, 4]].mean(axis=0))
    assert out["ego_x3"].iloc[2] == pytest.approx(0.16 / body, abs=1e-5)
    assert out["ego_x3"].iloc[0] == pytest.approx(0.08 / body, abs=1e-5)


# --- max_gap ------------------------------------------------------------------


def _walking(n: int = 8) -> pd.DataFrame:
    """The skeleton walking straight along world +X at 1 cm per frame."""
    points = np.stack(
        [_place(SKELETON, 0.0, np.array([0.01 * i, 0.0, 0.0])) for i in range(n)]
    )
    return _table(points)


def _blank(table: pd.DataFrame, frames: list[int], keypoint: int = 2) -> pd.DataFrame:
    table = table.copy()
    table.loc[table["frame"].isin(frames), [f"pose{a}{keypoint}" for a in "XYZ"]] = (
        np.nan
    )
    return table


def test_a_short_interior_run_is_interpolated() -> None:
    out = PoseEgocentric(params={**BASE_PARAMS, "max_gap": 2}).apply(
        _blank(_walking(), [3, 4])
    )
    np.testing.assert_allclose(_ego(out, 2), np.tile(SKELETON[2], (8, 1)), atol=1e-6)


def test_a_run_longer_than_max_gap_is_left_whole() -> None:
    """Not filled at its ends: a partly filled gap is a partly invented one."""
    out = PoseEgocentric(params={**BASE_PARAMS, "max_gap": 2}).apply(
        _blank(_walking(), [2, 3, 4])
    )
    assert out.loc[out["frame"].isin([2, 3, 4]), "ego_x2"].isna().all()


def test_without_max_gap_nothing_is_filled() -> None:
    out = PoseEgocentric(params=BASE_PARAMS).apply(_blank(_walking(), [3]))
    assert np.isnan(out.loc[out["frame"] == 3, "ego_x2"].iloc[0])


def test_a_run_at_either_end_is_never_filled() -> None:
    out = PoseEgocentric(params={**BASE_PARAMS, "max_gap": 5}).apply(
        _blank(_walking(), [0, 7])
    )
    assert out.loc[out["frame"].isin([0, 7]), "ego_x2"].isna().all()


def test_missing_rows_count_toward_the_gap() -> None:
    """Frames 3 to 5 have no row at all, so one NaN row beside them is a run of 4."""
    table = _blank(_walking(), [2])
    table = table[~table["frame"].isin([3, 4, 5])]
    out = PoseEgocentric(params={**BASE_PARAMS, "max_gap": 2}).apply(table)
    assert np.isnan(out.loc[out["frame"] == 2, "ego_x2"].iloc[0])


def test_a_short_gap_is_filled_along_a_straight_line() -> None:
    """Every column of a lost frame, filled from the values on either side.

    The left hip is lost on frames 3 and 4, which takes the origin and the
    heading with it. The body walks at 1 cm per frame, so a fill that repeats a
    neighbour instead of interpolating puts the origin a centimetre off.
    """
    table = _walking()
    table.loc[table["frame"].isin([3, 4]), ["poseX0", "poseY0", "poseZ0"]] = np.nan
    out = PoseEgocentric(params={**BASE_PARAMS, "max_gap": 2}).apply(table)

    np.testing.assert_allclose(out["origin_x"], 0.01 * np.arange(8), atol=1e-6)
    np.testing.assert_allclose(out[["origin_y", "origin_z"]], 0.0, atol=1e-6)
    np.testing.assert_allclose(out["body_yaw"], 0.0, atol=1e-6)
    for k in range(len(SKELETON)):
        np.testing.assert_allclose(
            _ego(out, k), np.tile(SKELETON[k], (8, 1)), atol=1e-6
        )


@pytest.mark.parametrize("split", ["id", "sequence"])
def test_a_gap_is_never_filled_across_a_track_boundary(split: str) -> None:
    """Not across two individuals, nor across two sequences of one individual.

    The second track picks up at the frame after the first one ends, so the two
    lost frames either side of the boundary are a gap of two frames between
    defined values -- short enough to fill, were the tracks one.
    """
    first = _blank(_walking(4), [3])
    second = _blank(_walking(4), [0])
    second["frame"] += 4
    second["time"] = second["frame"] / 30.0
    if split == "id":
        second["id"] = 1
    else:
        second["sequence"] = "seq_b"
    out = PoseEgocentric(params={**BASE_PARAMS, "max_gap": 5}).apply(
        pd.concat([first, second], ignore_index=True)
    )
    assert out.loc[out["frame"].isin([3, 4]), "ego_x2"].isna().all()
    assert out.loc[~out["frame"].isin([3, 4]), "ego_x2"].notna().all()


def test_the_heading_is_interpolated_along_the_shorter_arc() -> None:
    """From 3.0 to -3.0 is 0.28 rad through pi, not 6 rad through zero."""
    yaws = [3.0, 0.0, 0.0, -3.0]
    points = np.stack([_place(SKELETON, yaw, np.zeros(3)) for yaw in yaws])
    points[[1, 2], 0] = np.nan  # the left hip lost: no frame on frames 1 and 2
    out = PoseEgocentric(params={**BASE_PARAMS, "max_gap": 2}).apply(_table(points))
    step = (2 * np.pi - 6.0) / 3
    expected = _wrapped(np.array([3.0, 3.0 + step, 3.0 + 2 * step, -3.0]))
    np.testing.assert_allclose(
        _wrapped(out["body_yaw"].to_numpy() - expected), 0.0, atol=1e-6
    )


# --- refusals -----------------------------------------------------------------


def test_a_planar_table_is_refused_by_name() -> None:
    planar = _still(0.0, (0.0, 0.0, 0.0)).drop(columns=[f"poseZ{k}" for k in range(6)])
    with pytest.raises(ValueError, match=r"pose-egocentric.*2D ones only"):
        _ = PoseEgocentric(params=BASE_PARAMS).apply(planar)


def test_a_centroid_only_table_is_refused_by_name() -> None:
    bare = _still(0.0, (0.0, 0.0, 0.0))
    bare = bare.drop(columns=[c for c in bare.columns if c.startswith("pose")])
    with pytest.raises(ValueError, match=r"pose-egocentric.*carries none"):
        _ = PoseEgocentric(params=BASE_PARAMS).apply(bare)


def test_a_table_of_named_3d_keypoints_is_refused_by_name() -> None:
    """Not as a 2D table, which it is not."""
    named = _still(0.0, (0.0, 0.0, 0.0)).rename(
        columns={f"pose{a}{k}": f"pose{a}{NAMES[k]}" for a in "XYZ" for k in range(6)}
    )
    with pytest.raises(ValueError, match=r"pose-egocentric needs numbered 3D"):
        _ = PoseEgocentric(params=BASE_PARAMS).apply(named)


def test_a_table_without_an_id_column_is_refused_by_name() -> None:
    table = _still(0.0, (0.0, 0.0, 0.0)).drop(columns=["id"])
    with pytest.raises(ValueError, match=r"pose-egocentric needs the columns \['id'\]"):
        _ = PoseEgocentric(params=BASE_PARAMS).apply(table)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"origin": [0], "lateral": [0, 9]}, "outside the table's 6 keypoints"),
        (
            {"origin": ["pelvis"], "lateral": [0, 1], "keypoint_names": NAMES},
            "no keypoint named 'pelvis'",
        ),
        ({"origin": ["hip_l"], "lateral": [0, 1]}, "no keypoint names"),
        (
            {"origin": [0], "lateral": [0, 1], "keypoint_names": NAMES[:5]},
            "lists 5 names",
        ),
        (
            {"origin": [0], "lateral": ["hip_l", 0], "keypoint_names": NAMES},
            "more than once",
        ),
        (
            {
                "origin": [0],
                "lateral": [0, 1],
                "keypoints": [2, "nose"],
                "keypoint_names": NAMES,
            },
            "more than once",
        ),
        (
            {"origin": [0], "lateral": [0, 1], "scale_by": [[0, 1], [1, 0]]},
            "zero on every row",
        ),
    ],
)
def test_a_bad_keypoint_reference_is_refused_against_the_table(
    params: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = PoseEgocentric(params=params).apply(_still(0.0, (0.0, 0.0, 0.0)))


def test_a_reference_to_a_keypoint_the_table_lacks_is_refused() -> None:
    table = _still(0.0, (0.0, 0.0, 0.0)).drop(columns=["poseX2", "poseY2", "poseZ2"])
    with pytest.raises(
        ValueError, match=r"no poseX/poseY/poseZ triplet for keypoint\(s\) \[2\]"
    ):
        _ = PoseEgocentric(params={**BASE_PARAMS, "keypoints": [2]}).apply(table)


@pytest.mark.parametrize(
    "params",
    [
        {"origin": [], "lateral": [0, 1]},
        {"origin": [0], "lateral": [1, 1]},
        {"origin": [0], "lateral": [0, 1, 2]},
        {"origin": [0], "lateral": [0, 1], "keypoints": []},
        {"origin": [0], "lateral": [0, 1], "scale_by": [[], [1]]},
        {"origin": [0], "lateral": [0, 1], "max_gap": -1},
        {"origin": [0], "lateral": [0, 1], "keypoint_names": ["a", "a"]},
        {"lateral": [0, 1]},
    ],
)
def test_a_malformed_parameter_set_is_refused_at_construction(
    params: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _ = PoseEgocentric(params=params)


# --- through run_feature ------------------------------------------------------


def _dataset(base: Path, table: pd.DataFrame, std_format: str) -> Dataset:
    """A dataset holding one entry, ``seq``, whose tracks row records *std_format*.

    Written through the real index writer under the variant layout a conversion
    produces. ``add_tracks_variant`` would write a planar table; this one carries
    whatever *table* holds.
    """
    manifest = new_dataset_manifest(name="egocentric", base_dir=base / "dataset")
    ds = Dataset(manifest_path=manifest).load(ensure_roots=True)
    run_id = "convert-x.0.1-aaaaaaaaaa"
    root = tracks_variant_root(ds.get_root("tracks"), run_id)
    root.mkdir(parents=True, exist_ok=True)
    out = root / "seq.parquet"
    table.to_parquet(out)
    write_tracks_row(
        ds,
        run_id=run_id,
        group="",
        sequence="seq",
        out_path=out,
        producer="convert-x",
        std_format=std_format,
        n_rows=len(table),
        consumed_source_roots=("tracks_raw",),
    )
    return ds


def test_the_feature_declares_the_world_space_only() -> None:
    assert feature_track_spaces(PoseEgocentric(params=BASE_PARAMS)) == {"world_m"}


def test_run_feature_computes_the_posture_of_a_world_table(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, _still(1.3, (0.6, -0.2, 0.05), n=6), "mosaic_world_v1")
    result = run_feature(ds, PoseEgocentric(params=BASE_PARAMS))
    assert result.run_id is not None

    written = sorted(
        feature_run_root(ds, result.feature, result.run_id).glob("*.parquet")
    )
    assert len(written) == 1
    out = pd.read_parquet(written[0])
    assert len(out) == 6
    np.testing.assert_allclose(_ego(out, 2), np.tile(SKELETON[2], (6, 1)), atol=1e-6)
    np.testing.assert_allclose(out["body_yaw"], 1.3, atol=1e-6)


def test_run_feature_refuses_a_pixel_table(tmp_path: Path) -> None:
    """Refused for its space before any keypoint is read, whatever columns it has."""
    ds = _dataset(tmp_path, _still(0.0, (0.0, 0.0, 0.0)), "mosaic_v1")
    with pytest.raises(TrackSpaceRefused):
        _ = run_feature(ds, PoseEgocentric(params=BASE_PARAMS))
    assert not any(ds.get_root("features").rglob("*.parquet"))
