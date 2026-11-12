"""``pair-world`` measures each pair of individuals in the world frame.

Every expected value here is worked out by hand from positions and headings
chosen for it. The world frame is right-handed with +Z up, so seen from above a
positive bearing is counterclockwise: a partner on the focal's left.

Each fish of these tables carries three keypoints: 0 (``head``) at its position
plus its heading vector, 1 (``pec``) at its position, and 2 (``tail``) at its
position minus its heading vector. ``front=0, rear=1`` therefore reads the
heading vector exactly, and the body center ``X``/``Y``/``Z`` is the position.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.pair_world import PAIR_WORLD_PROVIDES, PairWorld
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.column_contract import Column
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.track_space import TrackSpaceRefused, feature_track_spaces

from tests.helpers import add_tracks_table, make_dataset

PARAMS: dict[str, object] = {"front": 0, "rear": 1}
NAMES = ["head", "pec", "tail"]
FPS = 140.0
PI = np.pi
ROOT_HALF = np.sqrt(0.5)

MEASURES = [
    "dist",
    "dist_xy",
    "dz",
    "bearing",
    "elevation",
    "rel_yaw_cos",
    "rel_yaw_sin",
    "rel_pitch",
    "heading_angle",
    "pitch",
    "speed",
    "approach_speed",
    "closing_speed",
]
"""Every measured column, in output order."""

DERIVED = ["speed", "approach_speed", "closing_speed"]

Vector = Sequence[float]


def _fish(
    identity: int,
    positions: np.ndarray,
    headings: np.ndarray,
    *,
    frames: Sequence[int] | None = None,
    fps: float = FPS,
) -> pd.DataFrame:
    """One fish's ``mosaic_world_v1`` rows from ``(n, 3)`` positions and headings."""
    n = len(positions)
    frame = np.arange(n, dtype=np.int64) if frames is None else np.asarray(frames)
    columns: dict[str, object] = {
        "frame": frame,
        "time": frame / fps,
        "id": np.full(n, identity, dtype=np.int64),
        "group": ["g"] * n,
        "sequence": ["s"] * n,
        "X": positions[:, 0],
        "Y": positions[:, 1],
        "Z": positions[:, 2],
    }
    keypoints = (positions + headings, positions, positions - headings)
    for k, points in enumerate(keypoints):
        for a, axis in enumerate("XYZ"):
            columns[f"pose{axis}{k}"] = points[:, a]
    return pd.DataFrame(columns)


def _still(identity: int, at: Vector, heading: Vector, n: int = 5) -> pd.DataFrame:
    """A fish held at *at*, facing along *heading*, for *n* frames."""
    positions = np.tile(np.asarray(at, dtype=np.float64), (n, 1))
    headings = np.tile(np.asarray(heading, dtype=np.float64), (n, 1))
    return _fish(identity, positions, headings)


def _moving(
    identity: int,
    start: Vector,
    velocity: Vector,
    *,
    n: int = 12,
    fps: float = FPS,
    steps_at: float = FPS,
) -> pd.DataFrame:
    """A fish moving at a constant *velocity* in m/s, facing along it.

    Its position on frame ``f`` is ``start + velocity * f / steps_at`` and its
    ``time`` is ``f / fps``, so with *steps_at* left at 140 a table timed at
    another rate holds the same positions on a different clock.
    """
    frame = np.arange(n, dtype=np.float64)[:, None]
    positions = np.asarray(start) + np.asarray(velocity) * frame / steps_at
    headings = np.tile(np.asarray(velocity, dtype=np.float64), (n, 1))
    return _fish(identity, positions, headings, fps=fps)


def _table(*fish: pd.DataFrame) -> pd.DataFrame:
    """The fish's rows, frame-major as a tracker writes them."""
    return (
        pd.concat(fish, ignore_index=True)
        .sort_values(["frame", "id"], kind="stable")
        .reset_index(drop=True)
    )


def _apply(table: pd.DataFrame, params: dict[str, object] = PARAMS) -> pd.DataFrame:
    return PairWorld(params=params).apply(table)


Row = dict[str, float]
"""One output row's measurements, by column."""


def _rows(out: pd.DataFrame, focal: int, partner: int) -> dict[int, Row]:
    """Each frame's measurements of *partner* from *focal*, as float64."""
    rows = out[(out["id1"] == focal) & (out["id2"] == partner)]
    frames: list[int] = rows["frame"].to_numpy(dtype=np.int64).tolist()
    values = rows[MEASURES].to_numpy(dtype=np.float64)
    return {
        frame: dict(zip(MEASURES, measured.tolist(), strict=True))
        for frame, measured in zip(frames, values, strict=True)
    }


def _at(out: pd.DataFrame, focal: int, partner: int, frame: int = 2) -> Row:
    """One frame's measurements of *partner* from *focal*."""
    return _rows(out, focal, partner)[frame]


def _series(rows: dict[int, Row], column: str, frames: Sequence[int]) -> list[float]:
    """*column* on each of *frames*."""
    return [rows[frame][column] for frame in frames]


def _undefined(row: Row, columns: Sequence[str]) -> bool:
    return all(np.isnan(row[column]) for column in columns)


def _check(row: Row, expected: dict[str, float]) -> None:
    for column, value in expected.items():
        if np.isnan(value):
            assert np.isnan(row[column]), f"{column} is {row[column]}, not NaN"
        else:
            np.testing.assert_allclose(row[column], value, atol=1e-6, err_msg=column)


INTERIOR = range(1, 11)
"""The frames of a 12-frame track with a frame on either side."""


STILL = {"speed": 0.0, "approach_speed": 0.0, "closing_speed": 0.0}
"""The motion of two still fish, on a frame with a frame on either side."""


# --- geometry of two still fish -------------------------------------------------


def test_a_partner_ahead_left_and_above_in_both_perspectives() -> None:
    """Fish 0 at (1, 1, 1) faces 45 degrees left of +X on the level.

    Fish 1 sits 2 m to +Y and 2 m up, facing -X and pitched 45 degrees down.
    The offset (0, 2, 2) lies 45 degrees left of fish 0's heading and 45 degrees
    above its horizontal plane. The headings (1, 1, 0) and (-1, 0, -1) have a
    dot product of -1 and lengths of sqrt(2) each, so the angle between them is
    arccos(-1/2) = 2 pi / 3. Fish 1's yaw is pi, 3 pi / 4 from fish 0's.

    Seen from fish 1, facing -X with its left toward -Y, the offset (0, -2, -2)
    points straight to its left and 45 degrees down.
    """
    out = _apply(
        _table(_still(0, (1, 1, 1), (1, 1, 0)), _still(1, (1, 3, 3), (-1, 0, -1)))
    )

    _check(
        _at(out, 0, 1),
        {
            "dist": 2 * np.sqrt(2),
            "dist_xy": 2.0,
            "dz": 2.0,
            "bearing": PI / 4,
            "elevation": PI / 4,
            "rel_yaw_cos": -ROOT_HALF,
            "rel_yaw_sin": ROOT_HALF,
            "rel_pitch": -PI / 4,
            "heading_angle": 2 * PI / 3,
            "pitch": 0.0,
            **STILL,
        },
    )
    _check(
        _at(out, 1, 0),
        {
            "dist": 2 * np.sqrt(2),
            "dist_xy": 2.0,
            "dz": -2.0,
            "bearing": PI / 2,
            "elevation": -PI / 4,
            "rel_yaw_cos": -ROOT_HALF,
            "rel_yaw_sin": -ROOT_HALF,
            "rel_pitch": PI / 4,
            "heading_angle": 2 * PI / 3,
            "pitch": -PI / 4,
            **STILL,
        },
    )


def test_a_partner_to_the_left_sees_the_focal_on_its_right() -> None:
    """Fish 0 faces +X with fish 1 at (0, 2, 0); fish 1 faces +X too, 2 m to its left."""
    out = _apply(
        _table(_still(0, (0, 0, 0), (1, 0, 0)), _still(1, (0, 2, 0), (2, 0, 0)))
    )

    _check(_at(out, 0, 1), {"bearing": PI / 2, "rel_yaw_cos": 1.0, "rel_yaw_sin": 0.0})
    _check(_at(out, 1, 0), {"bearing": -PI / 2, "heading_angle": 0.0})


@pytest.mark.parametrize("y", [0.0, -0.0], ids=["positive-zero", "negative-zero"])
def test_a_partner_directly_behind_has_one_bearing(y: float) -> None:
    """Straight behind is -pi, whichever sign the zero lateral offset carries."""
    out = _apply(
        _table(_still(0, (0, 0, 0), (1, 0, 0)), _still(1, (-3, y, 0), (0, 1, 1)))
    )

    behind = _series(_rows(out, 0, 1), "bearing", range(5))
    assert behind == [float(np.float32(-PI))] * 5
    _check(
        _at(out, 0, 1),
        {
            "dist": 3.0,
            "dist_xy": 3.0,
            "dz": 0.0,
            "elevation": 0.0,
            "rel_yaw_cos": 0.0,
            "rel_yaw_sin": 1.0,
            "rel_pitch": PI / 4,
            "heading_angle": PI / 2,
        },
    )
    # Fish 1 faces +Y, so fish 0 at (+3, 0, 0) from it is on its right.
    _check(_at(out, 1, 0), {"bearing": -PI / 2, "pitch": PI / 4, "rel_yaw_sin": -1.0})


def test_a_partner_straight_above_has_no_bearing() -> None:
    out = _apply(
        _table(_still(0, (0, 0, 0), (1, 0, 0)), _still(1, (0, 0, 3), (1, 0, 0)))
    )

    _check(
        _at(out, 0, 1),
        {
            "dist": 3.0,
            "dist_xy": 0.0,
            "dz": 3.0,
            "bearing": np.nan,
            "elevation": PI / 2,
            "heading_angle": 0.0,
            **STILL,
        },
    )
    _check(_at(out, 1, 0), {"dz": -3.0, "bearing": np.nan, "elevation": -PI / 2})


@pytest.mark.parametrize(("up", "pitch"), [(2.0, PI / 2), (-0.5, -PI / 2)])
def test_a_vertical_focal_heading_fixes_a_pitch_and_no_yaw(
    up: float, pitch: float
) -> None:
    out = _apply(
        _table(_still(0, (0, 0, 0), (0, 0, up)), _still(1, (3, 4, 0), (1, 0, 0)))
    )

    _check(
        _at(out, 0, 1),
        {
            "dist": 5.0,
            "bearing": np.nan,
            "elevation": 0.0,
            "rel_yaw_cos": np.nan,
            "rel_yaw_sin": np.nan,
            "rel_pitch": -pitch,
            "heading_angle": PI / 2,
            "pitch": pitch,
        },
    )
    # Fish 1 keeps its own bearing toward fish 0 and loses only the relative yaw.
    _check(
        _at(out, 1, 0),
        {
            "bearing": float(np.arctan2(-4.0, -3.0)),
            "rel_yaw_cos": np.nan,
            "rel_yaw_sin": np.nan,
            "rel_pitch": pitch,
            "pitch": 0.0,
        },
    )


def test_a_heading_within_the_tolerance_of_vertical_fixes_no_yaw() -> None:
    nearly = _apply(
        _table(_still(0, (0, 0, 0), (1e-9, 0, 1)), _still(1, (1, 0, 0), (1, 0, 0)))
    )
    tilted = _apply(
        _table(_still(0, (0, 0, 0), (1e-3, 0, 1)), _still(1, (1, 0, 0), (1, 0, 0)))
    )

    _check(_at(nearly, 0, 1), {"bearing": np.nan, "rel_yaw_cos": np.nan})
    _check(_at(tilted, 0, 1), {"bearing": 0.0, "rel_yaw_cos": 1.0, "rel_yaw_sin": 0.0})


def test_coincident_fish_have_no_bearing_elevation_or_approach() -> None:
    out = _apply(
        _table(_still(0, (1, 2, 3), (1, 0, 0)), _still(1, (1, 2, 3), (0, 1, 0)))
    )

    for focal, partner in ((0, 1), (1, 0)):
        _check(
            _at(out, focal, partner),
            {
                "dist": 0.0,
                "dist_xy": 0.0,
                "dz": 0.0,
                "bearing": np.nan,
                "elevation": np.nan,
                "approach_speed": np.nan,
                "speed": 0.0,
                "closing_speed": 0.0,
                "heading_angle": PI / 2,
            },
        )


def test_a_zero_length_heading_has_no_pitch_or_heading_angle() -> None:
    out = _apply(
        _table(_still(0, (0, 0, 0), (0, 0, 0)), _still(1, (0, 2, 1), (1, 0, 1)))
    )

    _check(
        _at(out, 0, 1),
        {
            "dist": np.sqrt(5.0),
            "elevation": float(np.arctan2(1.0, 2.0)),
            "bearing": np.nan,
            "rel_yaw_cos": np.nan,
            "rel_yaw_sin": np.nan,
            "rel_pitch": np.nan,
            "heading_angle": np.nan,
            "pitch": np.nan,
        },
    )
    # Fish 1 faces +X pitched up 45 degrees, with fish 0 at (0, -2, -1) on its right.
    _check(
        _at(out, 1, 0),
        {
            "bearing": -PI / 2,
            "pitch": PI / 4,
            "rel_yaw_cos": np.nan,
            "rel_pitch": np.nan,
            "heading_angle": np.nan,
        },
    )


def test_parallel_and_opposite_headings_are_zero_and_pi_apart() -> None:
    parallel = _apply(
        _table(_still(0, (0, 0, 0), (1, 1, 1)), _still(1, (5, 0, 0), (2, 2, 2)))
    )
    opposite = _apply(
        _table(_still(0, (0, 0, 0), (1, 1, 1)), _still(1, (5, 0, 0), (-3, -3, -3)))
    )

    _check(_at(parallel, 0, 1), {"heading_angle": 0.0, "rel_pitch": 0.0})
    _check(_at(opposite, 0, 1), {"heading_angle": PI, "rel_yaw_cos": -1.0})


# --- motion ---------------------------------------------------------------------


def _chase(*, fps: float = FPS) -> pd.DataFrame:
    """Fish 0 swims at 3 m/s along (1, 2, 2) / 3 toward fish 1, ahead on that line.

    Fish 1 swims away along the same line at 1.5 m/s, so the gap closes at
    1.5 m/s. Fish 0's velocity points straight at fish 1 (approach 3 m/s), and
    fish 1's straight away from fish 0 (approach -1.5 m/s).
    """
    return _table(
        _moving(0, (0, 0, 0), (1, 2, 2), fps=fps),
        _moving(1, (10, 20, 20), (0.5, 1, 1), fps=fps),
    )


def test_constant_velocity_gives_the_hand_speeds_on_interior_frames() -> None:
    out = _apply(_chase())

    for focal, partner, speed, approach in ((0, 1, 3.0, 3.0), (1, 0, 1.5, -1.5)):
        rows = _rows(out, focal, partner)
        expected = {"speed": speed, "approach_speed": approach, "closing_speed": 1.5}
        for column, value in expected.items():
            measured = _series(rows, column, INTERIOR)
            np.testing.assert_allclose(measured, value, rtol=1e-5, err_msg=column)
        for frame in (0, 11):
            assert _undefined(rows[frame], DERIVED), (focal, frame)


def test_derivatives_read_the_time_column() -> None:
    """One path on a clock half as fast is half as fast."""
    at_140 = _rows(_apply(_chase(fps=140.0)), 0, 1)
    at_70 = _rows(_apply(_chase(fps=70.0)), 0, 1)

    for column in DERIVED:
        fast = np.asarray(_series(at_140, column, INTERIOR))
        slow = np.asarray(_series(at_70, column, INTERIOR))
        np.testing.assert_allclose(slow, fast / 2, rtol=1e-5, err_msg=column)
    np.testing.assert_allclose(_series(at_70, "speed", INTERIOR), 1.5, rtol=1e-5)


def test_a_missing_frame_leaves_the_derivatives_beside_it_undefined() -> None:
    chase = _chase()
    table = chase[~((chase["id"] == 0) & (chase["frame"] == 5))]

    out = _apply(table)

    forward, backward = _rows(out, 0, 1), _rows(out, 1, 0)
    assert 5 not in forward and 5 not in backward
    for frame in (4, 6):
        assert _undefined(forward[frame], DERIVED), frame
        # Fish 1 has every frame, so its own velocity is defined beside the gap.
        _check(
            backward[frame],
            {"speed": 1.5, "approach_speed": -1.5, "closing_speed": np.nan},
        )
    np.testing.assert_allclose(_series(forward, "speed", (3, 7)), 3.0, rtol=1e-5)
    np.testing.assert_allclose(
        _series(forward, "closing_speed", (3, 7)), 1.5, rtol=1e-5
    )


# --- gaps -----------------------------------------------------------------------

HEADING_READERS = [
    "bearing",
    "rel_yaw_cos",
    "rel_yaw_sin",
    "rel_pitch",
    "heading_angle",
    "pitch",
]


@pytest.mark.parametrize("column", ["poseX0", "poseZ0", "poseY1"])
def test_a_missing_heading_keypoint_is_not_interpolated(column: str) -> None:
    table = _table(_still(0, (0, 0, 0), (1, 0, 0)), _still(1, (0, 2, 0), (1, 0, 0)))
    table.loc[(table["id"] == 0) & (table["frame"] == 2), column] = np.nan

    out = _apply(table)

    forward, backward = _rows(out, 0, 1), _rows(out, 1, 0)
    assert _undefined(forward[2], HEADING_READERS)
    assert _undefined(
        backward[2], ["rel_yaw_cos", "rel_yaw_sin", "rel_pitch", "heading_angle"]
    )
    _check(backward[2], {"bearing": -PI / 2, "pitch": 0.0})
    _check(forward[2], {"dist": 2.0, "dist_xy": 2.0, "dz": 0.0})
    for frame in (1, 3):
        assert not np.isnan(list(forward[frame].values())).any(), frame


def test_a_position_missing_one_axis_is_missing() -> None:
    table = _chase()
    table.loc[(table["id"] == 0) & (table["frame"] == 5), "Z"] = np.nan

    out = _apply(table)

    forward = _rows(out, 0, 1)
    assert _undefined(
        forward[5], ["dist", "dist_xy", "dz", "elevation", "bearing", "approach_speed"]
    )
    for frame in (4, 6):
        assert _undefined(forward[frame], ["speed", "closing_speed"]), frame
    np.testing.assert_allclose(_series(forward, "speed", (3, 7)), 3.0, rtol=1e-5)


# --- layout ---------------------------------------------------------------------


def _provided() -> list[str]:
    return [ref.name for ref in PAIR_WORLD_PROVIDES.adds if isinstance(ref, Column)]


def test_the_output_holds_the_declared_columns_sorted_by_pair_and_frame() -> None:
    out = _apply(_chase())

    assert list(out.columns) == [*_provided(), "group", "time"]
    assert _provided() == ["frame", "perspective", "id1", "id2", "sequence", *MEASURES]
    assert all(out[column].dtype == np.float32 for column in MEASURES)
    assert (out["sequence"] == "s").all() and (out["group"] == "g").all()
    expected = out.sort_values(["id1", "id2", "perspective", "frame"])
    pd.testing.assert_frame_equal(out, expected.reset_index(drop=True))
    assert out[["id1", "id2", "perspective"]].drop_duplicates().values.tolist() == [
        [0, 1, 0],
        [1, 0, 1],
    ]


def test_a_table_without_a_group_writes_none() -> None:
    out = _apply(_chase().drop(columns="group"))

    assert list(out.columns) == [*_provided(), "time"]


def test_three_fish_give_three_pairs_of_two_rows_a_frame() -> None:
    out = _apply(
        _table(
            _still(0, (0, 0, 0), (1, 0, 0)),
            _still(1, (1, 0, 0), (1, 0, 0)),
            _still(2, (0, 1, 0), (1, 0, 0)),
        )
    )

    assert len(out) == 3 * 2 * 5
    pairs = {tuple(pair) for pair in out[["id1", "id2"]].to_numpy().tolist()}
    assert pairs == {(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)}
    assert (out.groupby(["id1", "id2"]).size() == 5).all()


def test_fish_in_different_arenas_form_no_pair() -> None:
    table = _table(
        _still(0, (0, 0, 0), (1, 0, 0)),
        _still(1, (1, 0, 0), (1, 0, 0)),
        _still(2, (0, 1, 0), (1, 0, 0)),
    )
    table["arena"] = table["id"].map({0: "left", 1: "right", 2: "left"})

    out = _apply(table)

    pairs = {tuple(pair) for pair in out[["id1", "id2"]].to_numpy().tolist()}
    assert pairs == {(0, 2), (2, 0)}


def test_a_lone_fish_writes_no_rows() -> None:
    out = _apply(_still(0, (0, 0, 0), (1, 0, 0)))

    assert out.empty
    assert list(out.columns) == _provided()


# --- keypoint references --------------------------------------------------------


def test_a_keypoint_position_measures_from_that_keypoint() -> None:
    """The tails sit one heading vector behind the body centers.

    Fish 0's tail is at (-1, 0, 0) and fish 1's at (1, 0, 0), 2 m apart where
    the body centers are 3 m apart.
    """
    table = _table(_still(0, (0, 0, 0), (1, 0, 0)), _still(1, (3, 0, 0), (2, 0, 0)))

    out = _apply(table, {**PARAMS, "position": 2})

    _check(_at(out, 0, 1), {"dist": 2.0, "bearing": 0.0})


def test_names_and_indices_resolve_to_the_same_output() -> None:
    table = _chase()
    by_index = _apply(table, {"front": 0, "rear": 1, "position": 2})
    by_name = _apply(
        table,
        {"front": "head", "rear": "pec", "position": "tail", "keypoint_names": NAMES},
    )

    pd.testing.assert_frame_equal(by_index, by_name)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"front": 5, "rear": 1}, r"pair-world front.*outside the table's 3 keypoints"),
        ({**PARAMS, "position": 3}, r"pair-world position.*outside the table's 3"),
        ({**PARAMS, "keypoint_names": ["a", "b"]}, r"pair-world.*has 2 names"),
    ],
)
def test_a_reference_the_table_cannot_resolve_is_refused_by_name(
    params: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = _apply(_chase(), params)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"front": 1, "rear": 1}, "points nowhere"),
        ({"front": "head", "rear": 0, "keypoint_names": NAMES}, "points nowhere"),
        ({"front": 0}, "rear"),
        ({"front": -1, "rear": 0}, "negative"),
        ({"front": "head", "rear": 1}, "not given keypoint names"),
        (
            {**PARAMS, "position": "fin", "keypoint_names": NAMES},
            "no keypoint named 'fin'",
        ),
        ({**PARAMS, "keypoint_names": ["a", "a", "b"]}, "repeats"),
    ],
)
def test_a_malformed_parameter_set_is_refused_at_construction(
    params: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _ = PairWorld(params=params)


def test_a_planar_table_is_refused_by_name() -> None:
    planar = _chase().drop(columns=["Z", "poseZ0", "poseZ1", "poseZ2"])
    with pytest.raises(ValueError, match=r"pair-world.*2D"):
        _ = _apply(planar)


def test_a_table_without_time_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match=r"pair-world needs the columns \['time'\]"):
        _ = _apply(_chase().drop(columns="time"))


# --- through run_feature --------------------------------------------------------


def _dataset(base: Path, table: pd.DataFrame, std_format: str) -> Dataset:
    """A dataset holding one entry, ``s``, whose tracks row records *std_format*."""
    dataset = make_dataset(base / "dataset")
    add_tracks_table(dataset, "convert-x.0.1-aaaaaaaaaa", table, std_format=std_format)
    return dataset


def test_the_feature_declares_the_world_space_only() -> None:
    assert feature_track_spaces(PairWorld(params=PARAMS)) == {"world_m"}


def test_run_feature_measures_the_pairs_of_a_world_table(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, _chase(), "mosaic_world_v1")

    result = run_feature(ds, PairWorld(params=PARAMS))

    assert result.run_id is not None
    written = sorted(
        feature_run_root(ds, result.feature, result.run_id).glob("*.parquet")
    )
    assert len(written) == 1
    out = pd.read_parquet(written[0])
    assert len(out) == 2 * 12
    np.testing.assert_allclose(
        _series(_rows(out, 0, 1), "closing_speed", INTERIOR), 1.5, rtol=1e-5
    )


def test_run_feature_refuses_a_pixel_table(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, _chase(), "mosaic_v1")

    with pytest.raises(TrackSpaceRefused):
        _ = run_feature(ds, PairWorld(params=PARAMS))
    assert not any(ds.get_root("features").rglob("*.parquet"))
