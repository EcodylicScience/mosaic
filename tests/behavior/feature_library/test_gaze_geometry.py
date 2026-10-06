"""``ray-target`` and ``head-eye`` measure gaze and head geometry in a world frame.

Every case is built so its answer is known in closed form -- a ray aimed exactly
at a target, a head yawed by a stated angle, a gaze a stated number of degrees
left of and above head-forward, a rotation turning at a stated rate -- so a
convention flipped anywhere (a transposed rotation matrix, a scalar-last
quaternion read scalar first, a pitch whose sign means "up") shows up as a wrong
number rather than as a plausible one.

The sign conventions are pinned physically as well as numerically: positive
pitch is asserted to tip the nose *down*, positive roll to lift the *left* ear,
and positive azimuth to look to the subject's *left*. Those are the readings a
person analysing the output will rely on without opening the source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from mosaic.behavior.feature_library.gaze_geometry import HeadEye, RayTarget
from mosaic.core.dataset import Dataset
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.track_space import TrackSpaceRefused, feature_track_spaces
from mosaic.core.pipeline.tracks_identity import tracks_variant_root
from mosaic.core.pipeline.tracks_index import write_tracks_row
from mosaic.core.schema import ensure_track_schema

from tests.helpers import add_tracks_variant, make_dataset

FPS = 30.0
EYE_HEIGHT = 1.5

type Floats = npt.NDArray[np.float64]


# --- fixtures -----------------------------------------------------------------


def _world_table(
    n: int = 4,
    *,
    keypoints: Floats | None = None,
    gaze_direction: Floats | None = None,
    head_quaternion: Floats | None = None,
    frames: Floats | None = None,
    ident: int = 0,
) -> pd.DataFrame:
    """A schema-valid ``mosaic_world_v1`` table for one individual.

    Args:
        n: Rows, when *frames* is not given.
        keypoints: ``(n, k, 3)`` keypoint positions, or ``None`` for none.
        gaze_direction: ``(n, 3)`` or ``(3,)`` gaze directions; ``None`` writes
            no gaze columns. The origin is the eye, at ``(0, 0, EYE_HEIGHT)``.
        head_quaternion: ``(n, 4)`` or ``(4,)`` scalar-first quaternions;
            ``None`` writes no head columns.
        frames: The frame numbers, ``0..n-1`` by default.
        ident: The individual's id.
    """
    frame = np.arange(n, dtype=np.float64) if frames is None else frames
    n = len(frame)
    table = pd.DataFrame(
        {
            "frame": frame.astype(np.int64),
            "time": frame / FPS,
            "id": np.full(n, ident, dtype=np.int64),
            "group": [""] * n,
            "sequence": ["seq"] * n,
            "X": np.zeros(n),
            "Y": np.zeros(n),
            "Z": np.full(n, 1.0),
        }
    )
    if keypoints is not None:
        for k in range(keypoints.shape[1]):
            table[f"poseX{k}"] = keypoints[:, k, 0]
            table[f"poseY{k}"] = keypoints[:, k, 1]
            table[f"poseZ{k}"] = keypoints[:, k, 2]
    if gaze_direction is not None:
        direction = np.broadcast_to(gaze_direction, (n, 3))
        table["gazeX"] = 0.0
        table["gazeY"] = 0.0
        table["gazeZ"] = EYE_HEIGHT
        table["gazeDX"] = direction[:, 0]
        table["gazeDY"] = direction[:, 1]
        table["gazeDZ"] = direction[:, 2]
    if head_quaternion is not None:
        quaternion = np.broadcast_to(head_quaternion, (n, 4))
        table["headX"] = 0.0
        table["headY"] = 0.0
        table["headZ"] = EYE_HEIGHT
        table["headQW"] = quaternion[:, 0]
        table["headQX"] = quaternion[:, 1]
        table["headQY"] = quaternion[:, 2]
        table["headQZ"] = quaternion[:, 3]
    _ = ensure_track_schema(table, "mosaic_world_v1", strict=True)
    return table


def _points(*xyz: tuple[float, float, float], n: int = 4) -> Floats:
    """``(n, len(xyz), 3)`` keypoints standing still at the given positions."""
    return np.broadcast_to(np.asarray(xyz, dtype=np.float64), (n, len(xyz), 3)).copy()


def _scalar_first(rotation: Rotation) -> Floats:
    """A scipy rotation's quaternion(s), scalar first as the schema stores them."""
    quaternion = np.asarray(rotation.as_quat(), dtype=np.float64)
    return np.roll(quaternion, 1, axis=-1)


def _about(axis: tuple[float, float, float], angle: float) -> Floats:
    """The scalar-first quaternion turning *angle* about the unit *axis*."""
    half = angle / 2.0
    return np.array([np.cos(half), *(np.sin(half) * np.asarray(axis))])


def _column(out: pd.DataFrame, name: str) -> Floats:
    return out[name].to_numpy(dtype=np.float64)


# Targets around an eye at (0, 0, EYE_HEIGHT) looking along +X.
_AHEAD = (2.0, 0.1, EYE_HEIGHT)
_AHEAD_TOO = (2.0, -0.1, EYE_HEIGHT)
_LEFT = (0.0, 1.0, EYE_HEIGHT)
_BEHIND = (-1.0, 0.0, EYE_HEIGHT)
_ABOVE_AHEAD = (1.0, 0.0, EYE_HEIGHT + 1.0)


def _ray_table(n: int = 4, direction: Floats | None = None) -> pd.DataFrame:
    return _world_table(
        n,
        keypoints=_points(_AHEAD, _AHEAD_TOO, _LEFT, _BEHIND, _ABOVE_AHEAD, n=n),
        gaze_direction=np.array([3.0, 0.0, 0.0]) if direction is None else direction,
    )


_RAY_TARGETS: dict[str, object] = {
    "ahead": [0, 1],
    "left": [2],
    "behind": [3],
    "above": [4],
}


# --- ray-target: the angles and distances -------------------------------------


def test_a_ray_aimed_at_a_target_reads_zero_and_the_rest_read_their_angle() -> None:
    """``ahead`` is the mean of two keypoints straddling the ray, so exactly on it.

    The direction is three units long: the feature renormalises it, so a
    producer's not-quite-unit vector does not scale the result.
    """
    out = RayTarget(params={"targets": _RAY_TARGETS}).apply(_ray_table())

    np.testing.assert_allclose(_column(out, "angle__ahead"), 0.0, atol=1e-6)
    np.testing.assert_allclose(_column(out, "dist__ahead"), 2.0, rtol=1e-6)
    np.testing.assert_allclose(_column(out, "angle__left"), np.pi / 2, rtol=1e-6)
    np.testing.assert_allclose(_column(out, "dist__left"), 1.0, rtol=1e-6)
    np.testing.assert_allclose(_column(out, "angle__behind"), np.pi, rtol=1e-6)
    np.testing.assert_allclose(_column(out, "angle__above"), np.pi / 4, rtol=1e-6)
    np.testing.assert_allclose(_column(out, "dist__above"), np.sqrt(2.0), rtol=1e-6)


def test_a_ray_off_every_axis_hits_its_target_and_neither_mirror_of_it() -> None:
    """Every component of the eye and of the aim is distinct and nonzero.

    The ray tests above all aim along +X from an eye on the Z axis, where a
    direction with its Y and Z read swapped, or its Z negated, still reads the
    same angles. Here the decoys are the target mirrored in Y and in Z about the
    eye, so a ray read with either axis flipped hits a decoy rather than the
    target, and an origin with X and Y swapped hits nothing.
    """
    eye = np.array([0.3, -0.2, EYE_HEIGHT])
    aim = np.array([2.0, 1.0, -2.0])  # 3 m off, to the left and below the eye
    table = _world_table(
        keypoints=_points(
            tuple(eye + aim),
            tuple(eye + aim * [1.0, -1.0, 1.0]),
            tuple(eye + aim * [1.0, 1.0, -1.0]),
        ),
        gaze_direction=aim * 2.5,  # renormalised
    )
    table["gazeX"], table["gazeY"] = eye[0], eye[1]
    out = RayTarget(
        params={"targets": {"on": [0], "mirror_y": [1], "mirror_z": [2]}}
    ).apply(table)

    np.testing.assert_allclose(_column(out, "angle__on"), 0.0, atol=1e-6)
    np.testing.assert_allclose(_column(out, "dist__on"), 3.0, rtol=1e-6)
    # (2, 1, -2) . (2, -1, -2) = 7 and (2, 1, -2) . (2, 1, 2) = 1, over 3 * 3.
    np.testing.assert_allclose(
        _column(out, "angle__mirror_y"), np.arccos(7.0 / 9.0), rtol=1e-6
    )
    np.testing.assert_allclose(
        _column(out, "angle__mirror_z"), np.arccos(1.0 / 9.0), rtol=1e-6
    )


def test_the_output_columns_are_in_target_name_order_and_float32() -> None:
    """The recipe's key order is not part of the run identity, so not of the output."""
    out = RayTarget(params={"targets": dict(reversed(_RAY_TARGETS.items()))}).apply(
        _ray_table()
    )
    measured = [c for c in out.columns if c not in {"frame", "time", "id"}]
    measured = [c for c in measured if c not in {"group", "sequence"}]
    assert measured == [
        "angle__above",
        "dist__above",
        "angle__ahead",
        "dist__ahead",
        "angle__behind",
        "dist__behind",
        "angle__left",
        "dist__left",
        "ray_angular_speed",
    ]
    assert all(out[c].dtype == np.float32 for c in measured)
    assert list(out["frame"]) == [0, 1, 2, 3]


def test_a_missing_keypoint_drops_out_of_its_target() -> None:
    """The target is the mean of what is finite, so it moves to the survivor."""
    table = _ray_table()
    table.loc[1, "poseZ1"] = np.nan  # one axis missing drops the keypoint whole
    out = RayTarget(params={"targets": {"ahead": [0, 1]}}).apply(table)

    survivor = np.array(_AHEAD) - np.array([0.0, 0.0, EYE_HEIGHT])
    expected = np.arctan2(np.hypot(survivor[1], survivor[2]), survivor[0])
    assert _column(out, "angle__ahead")[1] == pytest.approx(expected, rel=1e-5)
    assert _column(out, "dist__ahead")[1] == pytest.approx(
        float(np.linalg.norm(survivor)), rel=1e-6
    )
    np.testing.assert_allclose(_column(out, "angle__ahead")[[0, 2, 3]], 0.0, atol=1e-6)


def test_a_target_with_no_finite_keypoint_is_nan() -> None:
    table = _ray_table()
    table.loc[2, ["poseX2", "poseY2", "poseZ2"]] = np.nan
    out = RayTarget(params={"targets": {"left": [2]}}).apply(table)
    assert np.isnan(_column(out, "angle__left")[2])
    assert np.isnan(_column(out, "dist__left")[2])
    assert np.isfinite(_column(out, "angle__left")[[0, 1, 3]]).all()


@pytest.mark.parametrize(
    "direction", [(0.0, 0.0, 0.0), (np.nan, 0.0, 1.0)], ids=["zero", "nan"]
)
def test_an_unusable_direction_is_nan_never_a_default(
    direction: tuple[float, float, float],
) -> None:
    directions = np.tile([1.0, 0.0, 0.0], (4, 1))
    directions[2] = direction
    out = RayTarget(params={"targets": {"ahead": [0, 1]}}).apply(
        _ray_table(direction=directions)
    )
    angle = _column(out, "angle__ahead")
    assert np.isnan(angle[2])
    np.testing.assert_allclose(angle[[0, 1, 3]], 0.0, atol=1e-6)
    # The distance does not read the direction, so it stays.
    np.testing.assert_allclose(_column(out, "dist__ahead"), 2.0, rtol=1e-6)
    # Neither step touching the bad row has a direction change.
    speed = _column(out, "ray_angular_speed")
    assert np.isnan(speed[[0, 2, 3]]).all()
    assert speed[1] == pytest.approx(0.0, abs=1e-6)


def test_targets_resolve_by_name_as_by_index() -> None:
    names = ["nose_left", "nose_right", "left_hand", "tail", "crown"]
    by_name = RayTarget(
        params={
            "keypoint_names": names,
            "targets": {"ahead": ["nose_left", "nose_right"], "left": ["left_hand"]},
        }
    ).apply(_ray_table())
    by_index = RayTarget(params={"targets": {"ahead": [0, 1], "left": [2]}}).apply(
        _ray_table()
    )
    pd.testing.assert_frame_equal(by_name, by_index)


# --- ray-target: the angular speed --------------------------------------------


def _turning_directions(omega: float, frames: Floats) -> Floats:
    theta = omega * frames / FPS
    return np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])


def test_a_ray_turning_at_a_known_rate_reads_that_rate() -> None:
    omega = 1.2  # rad/s
    frames = np.arange(6, dtype=np.float64)
    table = _world_table(
        keypoints=_points(_AHEAD, n=6),
        gaze_direction=_turning_directions(omega, frames),
        frames=frames,
    )
    out = RayTarget(params={"targets": {"ahead": [0]}}).apply(table)
    speed = _column(out, "ray_angular_speed")
    assert np.isnan(speed[0])
    np.testing.assert_allclose(speed[1:], omega, rtol=1e-5)


def test_the_angular_speed_does_not_bridge_a_missing_frame_or_an_individual() -> None:
    """A gap's two-frame turn is not reported as a one-frame one.

    Two individuals turning at different rates arrive interleaved and unsorted,
    so a speed computed across the boundary between them, or in arrival order,
    would show up as a wrong value.
    """
    frames_a = np.array([0, 1, 2, 4, 5], dtype=np.float64)  # frame 3 missing
    frames_b = np.arange(5, dtype=np.float64)
    a = _world_table(
        keypoints=_points(_AHEAD, n=5),
        gaze_direction=_turning_directions(1.0, frames_a),
        frames=frames_a,
        ident=0,
    )
    b = _world_table(
        keypoints=_points(_AHEAD, n=5),
        gaze_direction=_turning_directions(-2.0, frames_b),
        frames=frames_b,
        ident=1,
    )
    shuffled = pd.concat([b, a], ignore_index=True).sample(frac=1.0, random_state=0)
    out = RayTarget(params={"targets": {"ahead": [0]}}).apply(shuffled)

    assert list(out["id"]) == [0] * 5 + [1] * 5
    speed_a = _column(out, "ray_angular_speed")[:5]
    speed_b = _column(out, "ray_angular_speed")[5:]
    assert list(out["frame"])[:5] == [0, 1, 2, 4, 5]
    assert np.isnan(speed_a[[0, 3]]).all()
    np.testing.assert_allclose(speed_a[[1, 2, 4]], 1.0, rtol=1e-5)
    assert np.isnan(speed_b[0])
    np.testing.assert_allclose(speed_b[1:], 2.0, rtol=1e-5)


def test_rows_without_an_id_continue_nothing() -> None:
    """Two unidentified detections in consecutive frames are not one individual.

    The rows without an id sort to the end, together and one frame apart, so a
    speed that compared ids by their position in the sort alone would report the
    difference between two of them as a turn.
    """
    frames = np.arange(3, dtype=np.float64)
    known = _world_table(
        keypoints=_points(_AHEAD, n=3),
        gaze_direction=_turning_directions(1.0, frames),
        frames=frames,
    )
    unknown = known.copy()
    unknown["id"] = np.nan
    unknown[["gazeDX", "gazeDY", "gazeDZ"]] = _turning_directions(5.0, frames)
    out = RayTarget(params={"targets": {"ahead": [0]}}).apply(
        pd.concat([unknown, known], ignore_index=True)
    )

    speed = _column(out, "ray_angular_speed")
    assert out["id"].isna().tolist() == [False] * 3 + [True] * 3
    assert np.isnan(speed[0])
    np.testing.assert_allclose(speed[1:3], 1.0, rtol=1e-5)
    assert np.isnan(speed[3:]).all()
    # The rows are still measured; only their speed has no predecessor.
    np.testing.assert_allclose(
        _column(out, "dist__ahead"), np.hypot(2.0, 0.1), rtol=1e-6
    )


def test_the_frame_rate_comes_from_the_time_column_unless_given() -> None:
    frames = np.arange(5, dtype=np.float64)
    table = _world_table(
        keypoints=_points(_AHEAD, n=5),
        gaze_direction=_turning_directions(1.0, frames),
        frames=frames,
    )
    targets: dict[str, object] = {"targets": {"ahead": [0]}}
    measured = RayTarget(params=targets).apply(table)
    stated = RayTarget(params={**targets, "fps": FPS}).apply(table.drop(columns="time"))
    doubled = RayTarget(params={**targets, "fps": 2 * FPS}).apply(table)

    np.testing.assert_allclose(
        _column(measured, "ray_angular_speed")[1:], 1.0, rtol=1e-5
    )
    np.testing.assert_allclose(_column(stated, "ray_angular_speed")[1:], 1.0, rtol=1e-5)
    np.testing.assert_allclose(
        _column(doubled, "ray_angular_speed")[1:], 2.0, rtol=1e-5
    )


def test_no_frame_rate_at_all_refuses() -> None:
    table = _ray_table().drop(columns="time")
    with pytest.raises(ValueError, match="ray-target needs a frame rate"):
        _ = RayTarget(params={"targets": {"ahead": [0]}}).apply(table)


def test_a_table_with_no_consecutive_frames_needs_no_frame_rate() -> None:
    """Every speed is NaN whatever the rate, so asking for one would refuse for nothing."""
    table = _ray_table(n=1).drop(columns="time")
    out = RayTarget(params={"targets": {"ahead": [0, 1]}}).apply(table)
    assert np.isnan(_column(out, "ray_angular_speed")).all()
    assert _column(out, "angle__ahead")[0] == pytest.approx(0.0, abs=1e-6)


# --- ray-target: what it refuses ----------------------------------------------


def test_targets_are_required() -> None:
    with pytest.raises(ValidationError, match="targets"):
        _ = RayTarget()


@pytest.mark.parametrize(
    ("targets", "match"),
    [
        ({}, "at least 1"),
        ({"left hand": [0]}, "identifier"),
        ({"1st": [0]}, "identifier"),
        ({"hand": []}, "names no keypoints"),
        ({"hand": [2, 0, 2]}, "more than once"),
    ],
    ids=["empty", "space", "leading-digit", "empty-group", "repeated-index"],
)
def test_malformed_targets_are_refused_at_construction(
    targets: dict[str, list[int]], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        _ = RayTarget(params={"targets": targets})


def test_a_ray_prefix_must_be_a_column_prefix() -> None:
    with pytest.raises(ValidationError, match="column prefix"):
        _ = RayTarget(params={"ray": "gaze.", "targets": {"a": [0]}})


def test_a_named_keypoint_needs_a_name_list_that_holds_it() -> None:
    with pytest.raises(ValidationError, match="no keypoint_names"):
        _ = RayTarget(params={"targets": {"hand": ["left_hand"]}})
    with pytest.raises(ValidationError, match="no keypoint named 'left_hnad'"):
        _ = RayTarget(
            params={"keypoint_names": ["left_hand"], "targets": {"h": ["left_hnad"]}}
        )


def test_a_keypoint_named_and_indexed_in_one_target_is_refused() -> None:
    """``"nose"`` and ``0`` are one keypoint; listing both would weight it twice."""
    with pytest.raises(ValidationError, match="more than once"):
        _ = RayTarget(
            params={"keypoint_names": ["nose", "tail"], "targets": {"h": ["nose", 0]}}
        )


def test_a_name_list_of_another_skeleton_is_refused() -> None:
    """Names are matched by position, so two names for five keypoints mislabel them."""
    feature = RayTarget(
        params={"keypoint_names": ["nose", "left_hand"], "targets": {"h": ["nose"]}}
    )
    with pytest.raises(
        ValueError, match="ray-target: keypoint_names lists 2 names and this table "
    ):
        _ = feature.apply(_ray_table())


def test_a_keypoint_index_outside_the_table_is_refused() -> None:
    with pytest.raises(ValueError, match="ray-target target 'far'.*outside"):
        _ = RayTarget(params={"targets": {"far": [9]}}).apply(_ray_table())


def test_a_table_without_the_ray_is_refused() -> None:
    table = _world_table(keypoints=_points(_AHEAD))
    with pytest.raises(ValueError, match="ray-target needs the 'gaze' ray columns"):
        _ = RayTarget(params={"targets": {"ahead": [0]}}).apply(table)


def test_another_ray_prefix_reads_its_own_columns() -> None:
    table = _ray_table().rename(columns=lambda c: c.replace("gaze", "laser"))
    out = RayTarget(params={"ray": "laser", "targets": {"ahead": [0, 1]}}).apply(table)
    np.testing.assert_allclose(_column(out, "angle__ahead"), 0.0, atol=1e-6)


def test_planar_keypoints_are_refused() -> None:
    """``poseX``/``poseY`` without ``poseZ`` is a 2D table; a target there has no height."""
    table = _ray_table()
    table = table.drop(columns=[c for c in table.columns if c.startswith("poseZ")])
    with pytest.raises(ValueError, match="ray-target needs 3D keypoint columns"):
        _ = RayTarget(params={"targets": {"ahead": [0]}}).apply(table)


# --- head-eye: orientation ----------------------------------------------------


def test_a_yawed_head_reads_its_yaw_and_nothing_else() -> None:
    yaw = 0.7
    table = _world_table(
        gaze_direction=np.array([np.cos(yaw), np.sin(yaw), 0.0]),
        head_quaternion=_about((0.0, 0.0, 1.0), yaw),
    )
    out = HeadEye().apply(table)
    np.testing.assert_allclose(_column(out, "head_yaw"), yaw, rtol=1e-6)
    np.testing.assert_allclose(_column(out, "head_pitch"), 0.0, atol=1e-6)
    np.testing.assert_allclose(_column(out, "head_roll"), 0.0, atol=1e-6)
    # Gaze along the yawed nose is straight ahead in the head.
    np.testing.assert_allclose(_column(out, "gaze_azimuth"), 0.0, atol=1e-6)
    np.testing.assert_allclose(_column(out, "eye_eccentricity"), 0.0, atol=1e-6)


@pytest.mark.parametrize(
    ("yaw", "pitch", "roll"),
    [(0.4, 0.3, -0.2), (-2.5, -1.1, 2.9), (3.0, 0.05, -3.0), (0.0, 1.4, 0.0)],
)
def test_the_angles_are_the_intrinsic_zyx_decomposition(
    yaw: float, pitch: float, roll: float
) -> None:
    """Checked against scipy's own ``"ZYX"`` (intrinsic) construction."""
    rotation = Rotation.from_euler("ZYX", [yaw, pitch, roll])
    table = _world_table(
        gaze_direction=np.asarray(rotation.apply([1.0, 0.0, 0.0])),
        head_quaternion=_scalar_first(rotation),
    )
    out = HeadEye().apply(table)
    np.testing.assert_allclose(_column(out, "head_yaw"), yaw, atol=1e-5)
    np.testing.assert_allclose(_column(out, "head_pitch"), pitch, atol=1e-5)
    np.testing.assert_allclose(_column(out, "head_roll"), roll, atol=1e-5)


def test_positive_pitch_tips_the_nose_down() -> None:
    """The right-hand rule about a leftward +Y carries the nose toward -Z.

    A gaze straight along that lowered nose reads as straight ahead in the head,
    which is what fixes the sign physically rather than by formula.
    """
    pitch = 0.3
    nose = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])  # below the horizon
    out = HeadEye().apply(
        _world_table(
            gaze_direction=nose, head_quaternion=_about((0.0, 1.0, 0.0), pitch)
        )
    )
    np.testing.assert_allclose(_column(out, "head_pitch"), pitch, rtol=1e-6)
    np.testing.assert_allclose(_column(out, "gaze_azimuth"), 0.0, atol=1e-6)
    np.testing.assert_allclose(_column(out, "gaze_elevation"), 0.0, atol=1e-6)


def test_positive_roll_lifts_the_left_ear() -> None:
    """The head's +Y (left) axis rises; a gaze along it reads as straight left."""
    roll = 0.2
    left_ear = np.array([0.0, np.cos(roll), np.sin(roll)])  # above the horizon
    out = HeadEye().apply(
        _world_table(
            gaze_direction=left_ear, head_quaternion=_about((1.0, 0.0, 0.0), roll)
        )
    )
    np.testing.assert_allclose(_column(out, "head_roll"), roll, rtol=1e-6)
    np.testing.assert_allclose(_column(out, "gaze_azimuth"), np.pi / 2, rtol=1e-6)
    np.testing.assert_allclose(_column(out, "gaze_elevation"), 0.0, atol=1e-6)


def test_a_gaze_left_and_up_of_head_forward_reads_its_azimuth_and_elevation() -> None:
    """10 degrees left and 5 up of the nose, on a head that is itself turned."""
    azimuth, elevation = np.radians(10.0), np.radians(5.0)
    in_head = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    head = Rotation.from_euler("ZYX", [1.1, -0.4, 0.25])
    out = HeadEye().apply(
        _world_table(
            gaze_direction=np.asarray(head.apply(in_head)) * 4.0,  # renormalised
            head_quaternion=_scalar_first(head) * 2.0,  # renormalised
        )
    )
    np.testing.assert_allclose(_column(out, "gaze_azimuth"), azimuth, rtol=1e-5)
    np.testing.assert_allclose(_column(out, "gaze_elevation"), elevation, rtol=1e-5)
    np.testing.assert_allclose(
        _column(out, "eye_eccentricity"),
        np.arccos(np.cos(azimuth) * np.cos(elevation)),
        rtol=1e-5,
    )


def test_an_unusable_quaternion_is_nan_never_the_identity() -> None:
    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (4, 1))
    quaternions[2] = 0.0
    out = HeadEye().apply(
        _world_table(
            gaze_direction=np.array([1.0, 0.0, 0.0]), head_quaternion=quaternions
        )
    )
    for column in (
        "head_yaw",
        "head_pitch",
        "head_roll",
        "gaze_azimuth",
        "gaze_elevation",
        "eye_eccentricity",
    ):
        values = _column(out, column)
        assert np.isnan(values[2]), column
        np.testing.assert_allclose(values[[0, 1, 3]], 0.0, atol=1e-6)
    head_speed = _column(out, "head_angular_speed")
    assert np.isnan(head_speed[[0, 2, 3]]).all()
    # The gaze in space did not read the head, so its speed survives.
    np.testing.assert_allclose(_column(out, "gaze_angular_speed")[1:], 0.0, atol=1e-6)


def test_an_unusable_gaze_is_nan_and_leaves_the_head_alone() -> None:
    yaw = 0.4
    gaze = np.tile([np.cos(yaw), np.sin(yaw), 0.0], (4, 1))
    gaze[2] = 0.0
    out = HeadEye().apply(
        _world_table(gaze_direction=gaze, head_quaternion=_about((0.0, 0.0, 1.0), yaw))
    )
    for column in ("gaze_azimuth", "gaze_elevation", "eye_eccentricity"):
        values = _column(out, column)
        assert np.isnan(values[2]), column
        np.testing.assert_allclose(values[[0, 1, 3]], 0.0, atol=1e-6)
    np.testing.assert_allclose(_column(out, "head_yaw"), yaw, rtol=1e-6)
    gaze_speed = _column(out, "gaze_angular_speed")
    assert np.isnan(gaze_speed[[0, 2, 3]]).all()
    assert gaze_speed[1] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(_column(out, "head_angular_speed")[1:], 0.0, atol=1e-6)


# --- head-eye: angular speed --------------------------------------------------


def test_a_pure_rotation_reads_its_angular_speed_through_quaternion_sign_flips() -> (
    None
):
    """``q`` and ``-q`` are one rotation; alternating them must read as no extra turn.

    The head turns about a tilted axis at a stated rate. The gaze is fixed along
    head +X, and the axis is chosen perpendicular to it, so the gaze in space
    turns at the same rate.
    """
    omega = 2.0  # rad/s
    axis = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)  # perpendicular to +X
    frames = np.arange(8, dtype=np.float64)
    rotvecs: Floats = (omega * frames / FPS)[:, None] * axis
    rotations = Rotation.from_rotvec(rotvecs)
    quaternions = _scalar_first(rotations)
    quaternions[1::2] *= -1.0
    table = _world_table(
        gaze_direction=np.asarray(rotations.apply([1.0, 0.0, 0.0])),
        head_quaternion=quaternions,
        frames=frames,
    )
    out = HeadEye().apply(table)
    head_speed = _column(out, "head_angular_speed")
    gaze_speed = _column(out, "gaze_angular_speed")
    assert np.isnan(head_speed[0]) and np.isnan(gaze_speed[0])
    np.testing.assert_allclose(head_speed[1:], omega, rtol=1e-4)
    np.testing.assert_allclose(gaze_speed[1:], omega, rtol=1e-4)
    # Eyes fixed in the head: the eye-in-head angles do not move.
    np.testing.assert_allclose(_column(out, "eye_eccentricity"), 0.0, atol=1e-6)


def test_head_angular_speed_does_not_bridge_a_missing_frame() -> None:
    frames = np.array([0, 1, 3, 4], dtype=np.float64)
    quaternions = np.stack([_about((0.0, 0.0, 1.0), 0.5 * f / FPS) for f in frames])
    out = HeadEye().apply(
        _world_table(
            gaze_direction=np.array([1.0, 0.0, 0.0]),
            head_quaternion=quaternions,
            frames=frames,
        )
    )
    speed = _column(out, "head_angular_speed")
    assert np.isnan(speed[[0, 2]]).all()
    np.testing.assert_allclose(speed[[1, 3]], 0.5, rtol=1e-4)


def test_head_eye_output_columns() -> None:
    out = HeadEye().apply(
        _world_table(
            gaze_direction=np.array([1.0, 0.0, 0.0]),
            head_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        )
    )
    assert list(out.columns) == [
        "frame",
        "group",
        "id",
        "sequence",
        "time",
        "head_yaw",
        "head_pitch",
        "head_roll",
        "gaze_azimuth",
        "gaze_elevation",
        "eye_eccentricity",
        "head_angular_speed",
        "gaze_angular_speed",
    ]
    assert all(out[c].dtype == np.float32 for c in out.columns[5:])


@pytest.mark.parametrize(
    ("drop", "match"),
    [("head", "head-eye needs the head-pose columns"), ("gaze", "the gaze columns")],
)
def test_head_eye_refuses_a_table_without_either_family(drop: str, match: str) -> None:
    table = _world_table(
        gaze_direction=np.array([1.0, 0.0, 0.0]),
        head_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    table = table.drop(columns=[c for c in table.columns if c.startswith(drop)])
    with pytest.raises(ValueError, match=match):
        _ = HeadEye().apply(table)


# --- through run_feature, on a dataset ----------------------------------------


def test_both_features_compute_in_the_world_frame_only() -> None:
    assert feature_track_spaces(HeadEye()) == {"world_m"}
    assert feature_track_spaces(RayTarget(params={"targets": {"a": [0]}})) == {
        "world_m"
    }


def _world_dataset(tmp_path: Path, table: pd.DataFrame) -> Dataset:
    """A dataset holding *table* as one ``mosaic_world_v1`` entry, ``seq``."""
    dataset = make_dataset(tmp_path / "dataset", name="world")
    run_id = "convert-world.0.1-aaaaaaaaaa"
    root = tracks_variant_root(dataset.get_root("tracks"), run_id)
    root.mkdir(parents=True, exist_ok=True)
    out_path = root / f"{make_entry_key('', 'seq')}.parquet"
    table.to_parquet(out_path)
    write_tracks_row(
        dataset,
        run_id=run_id,
        group="",
        sequence="seq",
        out_path=out_path,
        producer="convert-world",
        std_format="mosaic_world_v1",
        n_rows=len(table),
    )
    return dataset


def _feature_output(dataset: Dataset, feature: HeadEye | RayTarget) -> pd.DataFrame:
    result = run_feature(dataset, feature)
    assert result.run_id
    (parquet,) = feature_run_root(dataset, result.feature, result.run_id).glob(
        "*.parquet"
    )
    return pd.read_parquet(parquet)


def test_both_features_run_on_a_world_dataset(tmp_path: Path) -> None:
    yaw = 0.6
    frames = np.arange(5, dtype=np.float64)
    quaternions = np.stack([_about((0.0, 0.0, 1.0), yaw + f / FPS) for f in frames])
    table = _world_table(
        keypoints=_points(_AHEAD, _AHEAD_TOO, _LEFT, n=5),
        gaze_direction=np.array([1.0, 0.0, 0.0]),
        head_quaternion=quaternions,
        frames=frames,
    )
    dataset = _world_dataset(tmp_path, table)

    rays = _feature_output(
        dataset, RayTarget(params={"targets": {"ahead": [0, 1], "left": [2]}})
    )
    np.testing.assert_allclose(_column(rays, "angle__ahead"), 0.0, atol=1e-6)
    np.testing.assert_allclose(_column(rays, "angle__left"), np.pi / 2, rtol=1e-6)
    np.testing.assert_allclose(_column(rays, "dist__ahead"), 2.0, rtol=1e-6)

    heads = _feature_output(dataset, HeadEye())
    np.testing.assert_allclose(
        _column(heads, "head_yaw"), yaw + frames / FPS, rtol=1e-5
    )
    np.testing.assert_allclose(_column(heads, "head_angular_speed")[1:], 1.0, rtol=1e-4)
    # The head turns left while the gaze stays on world +X: the eyes look right.
    np.testing.assert_allclose(
        _column(heads, "gaze_azimuth"), -(yaw + frames / FPS), rtol=1e-5
    )


def test_a_pixel_table_is_refused_before_any_work(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "dataset", name="planar")
    add_tracks_variant(
        dataset, "convert-x.0.1-0000000000", "seq_0", std_format="mosaic_v1"
    )
    with pytest.raises(TrackSpaceRefused):
        _ = run_feature(dataset, HeadEye())
    with pytest.raises(TrackSpaceRefused):
        _ = run_feature(dataset, RayTarget(params={"targets": {"ahead": [0]}}))
    assert not any(dataset.get_root("features").rglob("*.parquet"))
