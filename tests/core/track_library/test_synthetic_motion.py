"""The synthetic walk is a seeded correlated random walk in a reflecting arena.

Each generated sequence is a function of its file alone, whatever the chunk size
that advances it, and every individual stays inside the arena.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from mosaic.core.track_library.synthetic_motion import (
    CHUNK_FRAMES,
    STREAM_NAMES,
    SyntheticColumns,
    simulate,
    turn_headings,
    zone_step,
)
from mosaic.core.track_library.synthetic_spec import (
    SyntheticSpec,
    ZoneSpec,
    read_synthetic_spec,
)
from tests.helpers import write_synthetic_spec

Floats = NDArray[np.float64]
Bools = NDArray[np.bool_]

BOX_2D: dict[str, object] = {}
DISK_2D: dict[str, object] = {
    "arena": {"shape": "disk", "radius": 400.0, "width": None, "height": None}
}
BOX_3D: dict[str, object] = {"dimensions": 3, "arena": {"depth": 300.0}}
DISK_3D: dict[str, object] = {
    "dimensions": 3,
    "arena": {
        "shape": "disk",
        "radius": 400.0,
        "width": None,
        "height": None,
        "depth": 300.0,
    },
}
ARENAS = {"box_2d": BOX_2D, "disk_2d": DISK_2D, "box_3d": BOX_3D, "disk_3d": DISK_3D}


def make_spec(tmp_path: Path, **overrides: object) -> SyntheticSpec:
    return read_synthetic_spec(
        write_synthetic_spec(tmp_path / "walk.toml", **overrides)
    )


def position_arrays(columns: SyntheticColumns) -> list[Floats]:
    axes = [columns.X, columns.Y]
    if columns.Z is not None:
        axes.append(columns.Z)
    return axes


def assert_columns_equal(left: SyntheticColumns, right: SyntheticColumns) -> None:
    left_columns = dict(left.named_columns())
    right_columns = dict(right.named_columns())
    assert list(left_columns) == list(right_columns)
    for name, array in left_columns.items():
        np.testing.assert_array_equal(array, right_columns[name], err_msg=name)


@pytest.mark.parametrize("arena", ARENAS.values(), ids=ARENAS.keys())
def test_the_same_spec_gives_equal_arrays_whatever_the_chunk_size(
    tmp_path: Path, arena: dict[str, object]
) -> None:
    spec = make_spec(
        tmp_path,
        **arena,
        frames=60,
        individuals=4,
        burn_in_frames=11,
        motion={"speed_sd": 30.0},
    )

    first = simulate(spec)
    assert_columns_equal(first, simulate(spec))
    assert_columns_equal(first, simulate(spec, chunk_frames=1))
    assert_columns_equal(first, simulate(spec, chunk_frames=7))
    assert CHUNK_FRAMES == 4096


def test_the_columns_are_contiguous_one_dimensional_arrays_in_frame_then_id_order(
    tmp_path: Path,
) -> None:
    columns = simulate(make_spec(tmp_path, frames=20, individuals=3))

    assert columns.Z is None
    for array in (columns.frame, columns.id, *position_arrays(columns)):
        assert array.ndim == 1
        assert array.shape == (60,)
    assert columns.frame.dtype == columns.id.dtype == np.int64
    assert columns.X.dtype == columns.Y.dtype == np.float64
    np.testing.assert_array_equal(columns.frame, np.repeat(np.arange(20), 3))
    np.testing.assert_array_equal(columns.id, np.tile(np.arange(3), 20))


def test_a_3d_file_returns_a_z_column(tmp_path: Path) -> None:
    columns = simulate(make_spec(tmp_path, **BOX_3D, frames=10))

    assert columns.Z is not None
    assert columns.Z.dtype == np.float64
    assert columns.Z.shape == columns.X.shape


def test_a_different_seed_gives_different_positions(tmp_path: Path) -> None:
    one = simulate(make_spec(tmp_path, seed=1))
    other = simulate(make_spec(tmp_path, seed=2))

    assert not np.array_equal(one.X, other.X)
    assert not np.array_equal(one.Y, other.Y)


def test_the_eight_streams_are_named_in_a_fixed_order() -> None:
    assert STREAM_NAMES == (
        "start_position",
        "start_heading",
        "speed",
        "heading",
        "keypoint",
        "confidence",
        "id_chain",
        "gap_chain",
    )


@pytest.mark.parametrize("arena", ARENAS.values(), ids=ARENAS.keys())
@pytest.mark.parametrize("extreme_speed", [False, True])
def test_every_position_stays_inside_the_arena(
    tmp_path: Path, arena: dict[str, object], extreme_speed: bool
) -> None:
    probe = make_spec(tmp_path, **arena)
    motion: dict[str, float] = {"speed_sd": 30.0, "mean_speed": 100.0}
    if extreme_speed:
        extent = probe.arena.smallest_extent(probe.dimensions)
        motion["speed_sd"] = (probe.fps * extent - motion["mean_speed"]) / 4
    spec = make_spec(
        tmp_path, **arena, frames=800, individuals=12, seed=5, motion=motion
    )

    columns = simulate(spec)

    if spec.arena.shape == "box":
        assert spec.arena.width is not None and spec.arena.height is not None
        assert columns.X.min() >= 0 and columns.X.max() <= spec.arena.width
        assert columns.Y.min() >= 0 and columns.Y.max() <= spec.arena.height
    else:
        assert spec.arena.radius is not None
        radius = spec.arena.radius
        distance = np.hypot(columns.X - radius, columns.Y - radius)
        assert distance.max() <= radius * (1 + 1e-12)
    if columns.Z is not None:
        assert spec.arena.depth is not None
        assert columns.Z.min() >= 0 and columns.Z.max() <= spec.arena.depth
    assert np.ptp(columns.X) > 0


def test_a_walk_starts_within_start_radius_of_the_footprint_center(
    tmp_path: Path,
) -> None:
    spec = make_spec(
        tmp_path,
        dimensions=3,
        arena={"width": 600.0, "height": 400.0, "depth": 300.0, "start_radius": 50.0},
        frames=1,
        individuals=300,
    )

    columns = simulate(spec)

    assert columns.Z is not None
    assert np.hypot(columns.X - 300.0, columns.Y - 200.0).max() <= 50.0
    assert columns.Z.min() >= 0 and columns.Z.max() <= 300.0
    assert np.ptp(columns.Z) > 150.0


def test_a_disk_starts_across_its_whole_footprint_without_start_radius(
    tmp_path: Path,
) -> None:
    spec = make_spec(tmp_path, **DISK_2D, frames=1, individuals=300)

    columns = simulate(spec)

    distance = np.hypot(columns.X - 400.0, columns.Y - 400.0)
    assert distance.max() <= 400.0
    assert distance.max() > 340.0


def test_speed_is_an_ornstein_uhlenbeck_process_with_the_declared_moments(
    tmp_path: Path,
) -> None:
    spec = make_spec(
        tmp_path,
        frames=50_000,
        individuals=1,
        arena={"width": 1e7, "height": 1e7, "start_radius": 100.0},
        motion={
            "mean_speed": 100.0,
            "speed_sd": 30.0,
            "speed_tau_s": 1.0,
            "turning_sd": 0.0,
        },
    )

    columns = simulate(spec)

    step = np.hypot(np.diff(columns.X), np.diff(columns.Y))
    speed = step * spec.fps
    centered = speed - speed.mean()
    lag_one = float(centered[1:] @ centered[:-1] / (centered @ centered))
    assert lag_one == pytest.approx(np.exp(-1 / spec.fps / 1.0), abs=0.02)
    assert speed.mean() == pytest.approx(100.0, rel=0.05)
    assert speed.std() == pytest.approx(30.0, rel=0.05)


def test_the_heading_change_has_the_declared_variance(tmp_path: Path) -> None:
    spec = make_spec(
        tmp_path,
        frames=50_000,
        individuals=1,
        arena={"width": 1e7, "height": 1e7, "start_radius": 100.0},
        motion={"mean_speed": 100.0, "speed_sd": 0.0, "turning_sd": 0.5},
    )

    columns = simulate(spec)

    dx, dy = np.diff(columns.X), np.diff(columns.Y)
    change = np.angle((dx[1:] + 1j * dy[1:]) / (dx[:-1] + 1j * dy[:-1]))
    assert change.var() == pytest.approx(0.5**2 / spec.fps, rel=0.05)
    assert change.mean() == pytest.approx(0.0, abs=0.003)


def test_a_zero_turning_noise_walks_straight_away_from_the_walls(
    tmp_path: Path,
) -> None:
    spec = make_spec(
        tmp_path,
        frames=100,
        individuals=2,
        arena={"width": 1e7, "height": 1e7, "start_radius": 100.0},
        motion={"mean_speed": 90.0, "turning_sd": 0.0},
    )

    columns = simulate(spec)

    for individual in range(2):
        rows = columns.id == individual
        direction = np.stack([np.diff(columns.X[rows]), np.diff(columns.Y[rows])])
        np.testing.assert_allclose(np.hypot(*direction), 90.0 / spec.fps, rtol=1e-9)
        np.testing.assert_allclose(
            direction, np.broadcast_to(direction[:, :1], direction.shape), rtol=1e-6
        )


def test_a_3d_step_has_the_speed_length_whatever_the_turning(tmp_path: Path) -> None:
    spec = make_spec(
        tmp_path,
        dimensions=3,
        frames=2_000,
        individuals=2,
        arena={"width": 1e7, "height": 1e7, "depth": 1e7, "start_radius": 100.0},
        motion={"mean_speed": 90.0, "speed_sd": 0.0, "turning_sd": 3.0},
    )

    columns = simulate(spec)

    assert columns.Z is not None
    rows = columns.id == 0
    step = np.hypot(
        np.hypot(np.diff(columns.X[rows]), np.diff(columns.Y[rows])),
        np.diff(columns.Z[rows]),
    )
    np.testing.assert_allclose(step, 90.0 / spec.fps, rtol=1e-9)


def test_turning_keeps_a_heading_a_unit_vector_in_three_dimensions() -> None:
    rng = np.random.default_rng(3)
    headings = rng.standard_normal((200, 3))
    headings /= np.linalg.norm(headings, axis=1, keepdims=True)

    for _ in range(5_000):
        headings = turn_headings(headings, rng.standard_normal((200, 3)) * 0.4)

    np.testing.assert_allclose(np.linalg.norm(headings, axis=1), 1.0, atol=1e-12)


def test_turning_is_a_rotation_by_the_tangent_noise_in_the_plane() -> None:
    headings = np.array([[1.0, 0.0]])

    turned = turn_headings(headings, np.array([[5.0, 0.3]]))

    np.testing.assert_allclose(turned, [[np.cos(0.3), np.sin(0.3)]], atol=1e-15)


def test_turning_by_no_noise_leaves_a_heading_unchanged() -> None:
    headings = np.array([[0.0, 0.6, 0.8]])

    np.testing.assert_array_equal(turn_headings(headings, np.zeros((1, 3))), headings)


@pytest.mark.parametrize("arena", [BOX_2D, DISK_3D], ids=["box_2d", "disk_3d"])
def test_burn_in_runs_first_and_writes_nothing(
    tmp_path: Path, arena: dict[str, object]
) -> None:
    burned = simulate(
        make_spec(tmp_path, **arena, frames=40, burn_in_frames=25, individuals=3)
    )
    longer = simulate(make_spec(tmp_path, **arena, frames=65, individuals=3))

    kept = longer.frame >= 25
    assert burned.X.shape == (40 * 3,)
    np.testing.assert_array_equal(burned.frame, longer.frame[kept] - 25)
    np.testing.assert_array_equal(burned.id, longer.id[kept])
    np.testing.assert_array_equal(burned.X, longer.X[kept])
    np.testing.assert_array_equal(burned.Y, longer.Y[kept])
    if burned.Z is not None and longer.Z is not None:
        np.testing.assert_array_equal(burned.Z, longer.Z[kept])


def test_the_generator_holds_the_columns_and_one_chunk_of_draws(tmp_path: Path) -> None:
    spec = make_spec(
        tmp_path,
        frames=5_000,
        individuals=100,
        motion={"speed_sd": 30.0},
    )
    chunk = 64

    tracemalloc.start()
    try:
        columns = simulate(spec, chunk_frames=chunk)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    returned = sum(
        array.nbytes for array in (columns.frame, columns.id, *position_arrays(columns))
    )
    assert returned == 4 * 500_000 * 8
    assert peak <= 1.05 * returned + 32 * chunk * 100 * 8


def zones(**overrides: float) -> ZoneSpec:
    values = {"repulsion": 10.0, "orientation": 60.0, "attraction": 150.0}
    values.update(overrides)
    return ZoneSpec.model_validate(values)


def unit(*components: float) -> Floats:
    vector = np.array(components)
    return vector / np.linalg.norm(vector)


def test_two_individuals_inside_repulsion_turn_apart() -> None:
    positions = np.array([[0.0, 0.0], [5.0, 0.0]])
    headings = np.array([[0.0, 1.0], [0.0, 1.0]])

    turned = zone_step(positions, headings, zones(), 1 / 30)

    angle = 6.0 / 30
    np.testing.assert_allclose(
        turned, [[-np.sin(angle), np.cos(angle)], [np.sin(angle), np.cos(angle)]]
    )


def test_a_neighbor_behind_the_field_of_view_is_not_perceived() -> None:
    positions = np.array([[0.0, 0.0], [-100.0, 0.0]])
    headings = np.array([[1.0, 0.0], [0.0, 1.0]])

    blind = zone_step(positions, headings, zones(field_of_view_deg=270.0), 1.0)
    seeing = zone_step(positions, headings, zones(field_of_view_deg=360.0), 1.0)

    np.testing.assert_array_equal(blind[0], headings[0])
    np.testing.assert_allclose(seeing[0], [-1.0, 0.0], atol=1e-12)


def test_the_field_of_view_edge_is_perceived() -> None:
    positions = np.array([[0.0, 0.0], [-100.0, 100.0]])
    headings = np.array([[1.0, 0.0], [1.0, 0.0]])

    turned = zone_step(positions, headings, zones(field_of_view_deg=270.0), 1.0)

    np.testing.assert_allclose(turned[0], unit(-1.0, 1.0))


def test_a_neighbor_in_the_orientation_zone_aligns_the_focal_heading() -> None:
    positions = np.array([[0.0, 0.0], [30.0, 0.0]])
    headings = np.array([[1.0, 0.0], [0.0, 1.0]])

    turned = zone_step(positions, headings, zones(), 1.0)

    np.testing.assert_allclose(turned[0], unit(1.0, 1.0))


def test_a_neighbor_in_the_attraction_zone_draws_the_focal_heading_toward_it() -> None:
    positions = np.array([[0.0, 0.0], [100.0, 50.0]])
    headings = np.array([[1.0, 0.0], [0.0, -1.0]])

    turned = zone_step(positions, headings, zones(), 1.0)

    np.testing.assert_allclose(turned[0], unit(100.0, 50.0))


def test_orientation_and_attraction_combine_into_one_direction() -> None:
    positions = np.array([[0.0, 0.0], [30.0, 0.0], [0.0, 100.0]])
    headings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

    turned = zone_step(positions, headings, zones(), 1.0)

    np.testing.assert_allclose(turned[0], unit(2.0, 1.0))


def test_repulsion_overrides_alignment_and_attraction() -> None:
    positions = np.array([[0.0, 0.0], [3.0, 4.0], [30.0, 0.0], [100.0, 0.0]])
    headings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    turned = zone_step(positions, headings, zones(), 1.0)

    np.testing.assert_allclose(turned[0], [-0.6, -0.8])


def test_no_neighbor_in_any_zone_keeps_the_heading() -> None:
    positions = np.array([[0.0, 0.0], [500.0, 0.0]])
    headings = unit(1.0, 2.0)[None, :].repeat(2, axis=0)

    turned = zone_step(positions, headings, zones(), 1.0)

    np.testing.assert_allclose(turned, headings, rtol=0, atol=1e-15)


def test_a_zero_desired_direction_keeps_the_heading() -> None:
    positions = np.array([[0.0, 0.0], [0.0, 5.0], [0.0, -5.0]])
    headings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

    turned = zone_step(positions, headings, zones(), 1.0)

    np.testing.assert_array_equal(turned[0], headings[0])


def test_a_desired_direction_cancelling_up_to_rounding_keeps_the_heading() -> None:
    focal = np.array([0.1, 0.2])
    offset = np.array([0.3, 0.7])
    positions = np.stack([focal, focal + offset, focal - offset])
    sideways = unit(0.7, -0.3)
    headings = np.stack([sideways, sideways, sideways])
    repulsion = positions[1:] - focal
    residue = (repulsion / np.linalg.norm(repulsion, axis=1, keepdims=True)).sum(0)

    turned = zone_step(positions, headings, zones(), 1.0)

    assert np.any(residue != 0)
    np.testing.assert_allclose(turned[0], sideways, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ((1.0, 0.0), (0.0, 1.0)),
        ((1.0, 1.0), (1.0, -1.0)),
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        ((1.0, 2.0, 3.0), (13.0, -2.0, -3.0)),
    ],
    ids=["2d_axis", "2d_diagonal", "3d_axis", "3d_oblique"],
)
def test_an_antiparallel_desired_direction_turns_toward_the_least_aligned_axis(
    heading: tuple[float, ...], expected: tuple[float, ...]
) -> None:
    focal = unit(*heading)
    positions = np.stack([np.zeros_like(focal), 5.0 * focal])
    headings = np.stack([focal, focal])
    angle = 0.2

    turned = zone_step(positions, headings, zones(max_turn_rate=angle), 1.0)

    np.testing.assert_allclose(
        turned[0], np.cos(angle) * focal + np.sin(angle) * unit(*expected)
    )


@pytest.mark.parametrize(
    ("positions", "heading", "expected"),
    [
        ([[0.0, 0.0], [0.0, 100.0]], (1.0, 0.0), (np.cos(0.2), np.sin(0.2))),
        (
            [[0.0, 0.0, 0.0], [0.0, 0.0, 100.0]],
            (1.0, 0.0, 0.0),
            (np.cos(0.2), 0.0, np.sin(0.2)),
        ),
    ],
    ids=["2d", "3d"],
)
def test_the_turn_per_frame_is_clipped_in_the_plane_of_the_two_directions(
    positions: list[list[float]],
    heading: tuple[float, ...],
    expected: tuple[float, ...],
) -> None:
    headings = np.array([heading, heading])

    turned = zone_step(np.array(positions), headings, zones(max_turn_rate=6.0), 1 / 30)

    np.testing.assert_allclose(turned[0], expected, atol=1e-15)


@pytest.mark.parametrize("dimensions", [2, 3])
def test_zones_give_equal_arrays_whatever_the_chunk_size(
    tmp_path: Path, dimensions: int
) -> None:
    spec = make_spec(
        tmp_path,
        dimensions=dimensions,
        frames=60,
        individuals=6,
        burn_in_frames=11,
        arena={"start_radius": 100.0, "depth": 300.0 if dimensions == 3 else None},
        zones={"repulsion": 10.0, "orientation": 60.0, "attraction": 150.0},
    )

    first = simulate(spec)
    assert_columns_equal(first, simulate(spec, chunk_frames=1))
    assert_columns_equal(first, simulate(spec, chunk_frames=7))


def frame_positions(columns: SyntheticColumns, individuals: int) -> Floats:
    """The positions as an array of shape ``(frames, individuals, dimensions)``."""
    axes = position_arrays(columns)
    return np.stack(axes, axis=1).reshape(-1, individuals, len(axes))


def mean_nearest_neighbor_distance(positions: Floats) -> float:
    individuals = positions.shape[1]
    offsets = positions[:, :, None, :] - positions[:, None, :, :]
    distance = np.linalg.norm(offsets, axis=3)
    distance[:, np.arange(individuals), np.arange(individuals)] = np.inf
    return float(distance.min(axis=2).mean())


def test_zones_hold_a_group_closer_than_independent_walkers(tmp_path: Path) -> None:
    file: dict[str, object] = {
        "frames": 5_000,
        "individuals": 30,
        "seed": 11,
        "arena": {"start_radius": 100.0},
    }
    walkers = frame_positions(simulate(make_spec(tmp_path, **file)), 30)
    school = frame_positions(
        simulate(
            make_spec(
                tmp_path,
                **file,
                zones={"repulsion": 10.0, "orientation": 60.0, "attraction": 150.0},
            )
        ),
        30,
    )

    assert mean_nearest_neighbor_distance(
        school[2_500:]
    ) < 0.5 * mean_nearest_neighbor_distance(walkers[2_500:])


def test_noisy_turning_lets_groups_form_and_split(tmp_path: Path) -> None:
    """Scattered walkers with zones and noisy turning join one group, then split.

    Ten individuals start scattered over an 800 by 800 box and turn with
    ``turning_sd = 2``, four times the default. A group is a connected
    component of the graph joining individuals within the attraction radius.
    Four squarings of its reachability matrix cover paths of 16 edges, more
    than ten individuals need. The box is small beside the 150 attraction
    radius, so scattered individuals meet, and the noise is large, so a joined
    group sheds members.
    """
    spec = make_spec(
        tmp_path,
        frames=20_000,
        individuals=10,
        seed=1,
        arena={"width": 800.0, "height": 800.0},
        motion={"turning_sd": 2.0},
        zones={"repulsion": 10.0, "orientation": 60.0, "attraction": 150.0},
    )

    positions = frame_positions(simulate(spec), 10)

    offsets = positions[:, :, None, :] - positions[:, None, :, :]
    reach = (np.linalg.norm(offsets, axis=3) <= 150.0).astype(np.int64)
    for _ in range(4):
        reach = np.minimum(reach @ reach, 1)
    one_group = np.flatnonzero(reach.all(axis=(1, 2)))
    assert one_group.size > 0
    assert one_group[0] > 0
    assert not reach[one_group[0] :].all(axis=(1, 2)).all()


BODY = {"keypoints": 5, "length": 30.0, "keypoint_sd": 0.0}


def body_spec(tmp_path: Path, **overrides: object) -> SyntheticSpec:
    body = {**BODY, **cast("dict[str, object]", overrides.pop("body", {}))}
    return make_spec(tmp_path, body=body, **overrides)


def keypoint_array(columns: SyntheticColumns, axis: str, keypoints: int) -> Floats:
    """The ``poseX``, ``poseY`` or ``poseZ`` columns stacked as ``(rows, keypoints)``."""
    return np.stack([columns.pose[f"pose{axis}{k}"] for k in range(keypoints)], axis=1)


def test_without_a_body_table_there_is_no_pose_array(tmp_path: Path) -> None:
    columns = simulate(make_spec(tmp_path, frames=10))

    assert not columns.pose
    assert [name for name, _ in columns.named_columns()] == ["frame", "id", "X", "Y"]


@pytest.mark.parametrize("keypoints", [0, 1])
def test_zero_keypoints_write_no_pose_array_and_one_sits_at_the_center(
    tmp_path: Path, keypoints: int
) -> None:
    columns = simulate(
        make_spec(
            tmp_path,
            frames=30,
            body={"keypoints": keypoints, "length": 30.0, "keypoint_sd": 0.0},
        )
    )

    if keypoints == 0:
        assert not columns.pose
    else:
        np.testing.assert_array_equal(columns.pose["poseX0"], columns.X)
        np.testing.assert_array_equal(columns.pose["poseY0"], columns.Y)
        assert list(columns.pose) == ["poseX0", "poseY0", "poseP0"]


@pytest.mark.parametrize("dimensions", [2, 3])
def test_pose_columns_follow_the_table_order_with_z_only_in_3d(
    tmp_path: Path, dimensions: int
) -> None:
    columns = simulate(
        body_spec(
            tmp_path,
            frames=10,
            dimensions=dimensions,
            arena={"depth": 300.0 if dimensions == 3 else None},
            body={"keypoints": 3},
        )
    )

    axes = ["X", "Y", "Z"][:dimensions]
    expected = ["frame", "id", *axes]
    for k in range(3):
        expected += [f"pose{axis}{k}" for axis in axes] + [f"poseP{k}"]
    named = list(columns.named_columns())
    assert [name for name, _ in named] == expected
    assert ("poseZ0" in columns.pose) == (dimensions == 3)
    for name, array in named:
        assert array.ndim == 1
        assert array.shape == columns.X.shape
        assert array.flags.c_contiguous
        assert array.dtype == (np.int64 if name in ("frame", "id") else np.float64)


def test_noise_free_keypoints_average_to_the_center_and_are_evenly_spaced(
    tmp_path: Path,
) -> None:
    keypoints = 5
    columns = simulate(body_spec(tmp_path, frames=60, individuals=4))

    x = keypoint_array(columns, "X", keypoints)
    y = keypoint_array(columns, "Y", keypoints)
    np.testing.assert_allclose(x.mean(axis=1), columns.X, atol=1e-9)
    np.testing.assert_allclose(y.mean(axis=1), columns.Y, atol=1e-9)
    gaps = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))
    np.testing.assert_allclose(gaps, 30.0 / (keypoints - 1), atol=1e-9)


def test_the_head_is_first_and_points_along_the_step_that_arrived_at_the_position(
    tmp_path: Path,
) -> None:
    """Keypoints use the heading of the written position.

    The individual arrived at frame ``t``'s position along the heading that
    frame writes, so the tail-to-head direction is the unit step from frame
    ``t - 1``. The walk turns, stays far from the walls and has constant speed.
    """
    individuals = 3
    columns = simulate(
        body_spec(
            tmp_path,
            frames=40,
            individuals=individuals,
            arena={"start_radius": 50.0},
            motion={"speed_sd": 0.0, "turning_sd": 2.0},
        )
    )

    x = keypoint_array(columns, "X", 5)
    y = keypoint_array(columns, "Y", 5)
    head_minus_tail = np.stack([x[:, 0] - x[:, -1], y[:, 0] - y[:, -1]], axis=1)
    direction = head_minus_tail / 30.0
    step_by_frame = np.stack([columns.X, columns.Y], axis=1).reshape(40, individuals, 2)
    steps = np.diff(step_by_frame, axis=0)
    unit_steps = steps / np.linalg.norm(steps, axis=2, keepdims=True)
    heading = direction.reshape(40, individuals, 2)
    np.testing.assert_allclose(heading[1:], unit_steps, atol=1e-9)


def test_a_3d_body_lies_along_the_3d_heading(tmp_path: Path) -> None:
    individuals = 2
    columns = simulate(
        body_spec(
            tmp_path,
            frames=30,
            individuals=individuals,
            dimensions=3,
            arena={"depth": 300.0, "start_radius": 50.0},
            motion={"speed_sd": 0.0, "turning_sd": 1.0},
        )
    )

    axes = ["X", "Y", "Z"]
    head_minus_tail = np.stack(
        [
            keypoint_array(columns, axis, 5)[:, 0]
            - keypoint_array(columns, axis, 5)[:, -1]
            for axis in axes
        ],
        axis=1,
    )
    np.testing.assert_allclose(np.linalg.norm(head_minus_tail, axis=1), 30.0, atol=1e-9)
    centers = np.stack(position_arrays(columns), axis=1).reshape(30, individuals, 3)
    steps = np.diff(centers, axis=0)
    unit_steps = steps / np.linalg.norm(steps, axis=2, keepdims=True)
    np.testing.assert_allclose(
        (head_minus_tail / 30.0).reshape(30, individuals, 3)[1:], unit_steps, atol=1e-9
    )


def test_keypoint_noise_has_the_declared_standard_deviation(tmp_path: Path) -> None:
    arguments: dict[str, object] = {"frames": 2_000, "individuals": 3}
    exact = simulate(body_spec(tmp_path, **arguments))
    noisy = simulate(body_spec(tmp_path, **arguments, body={"keypoint_sd": 2.0}))

    for axis in ("X", "Y"):
        residual = keypoint_array(noisy, axis, 5) - keypoint_array(exact, axis, 5)
        assert residual.std() == pytest.approx(2.0, rel=0.03)
        assert residual.mean() == pytest.approx(0.0, abs=0.05)
    np.testing.assert_array_equal(noisy.X, exact.X)


def test_confidence_lies_in_the_unit_interval_and_clips_at_zero(tmp_path: Path) -> None:
    arguments: dict[str, object] = {"frames": 400, "individuals": 3}
    default = simulate(body_spec(tmp_path, **arguments))
    wide = simulate(body_spec(tmp_path, **arguments, body={"confidence_sd": 1.0}))
    exact = simulate(body_spec(tmp_path, **arguments, body={"confidence_sd": 0.0}))

    confidence = np.concatenate([default.pose[f"poseP{k}"] for k in range(5)])
    assert confidence.min() >= 0.0
    assert confidence.max() <= 1.0
    assert confidence.mean() == pytest.approx(1 - 0.05 * np.sqrt(2 / np.pi), abs=0.003)
    spread = np.concatenate([wide.pose[f"poseP{k}"] for k in range(5)])
    assert spread.min() == 0.0
    assert spread.max() <= 1.0
    assert (spread == 0.0).mean() == pytest.approx(0.317, abs=0.03)
    for k in range(5):
        np.testing.assert_array_equal(exact.pose[f"poseP{k}"], 1.0)


def test_adding_a_body_leaves_the_trajectories_unchanged(tmp_path: Path) -> None:
    arguments: dict[str, object] = {
        "frames": 50,
        "individuals": 4,
        "burn_in_frames": 9,
        "zones": {"repulsion": 10.0, "orientation": 60.0, "attraction": 150.0},
    }
    bare = simulate(make_spec(tmp_path, **arguments))

    for keypoints in (1, 3, 8):
        body = {"keypoints": keypoints, "length": 20.0, "keypoint_sd": 3.0}
        with_body = simulate(make_spec(tmp_path, **arguments, body=body))
        np.testing.assert_array_equal(with_body.frame, bare.frame)
        np.testing.assert_array_equal(with_body.id, bare.id)
        np.testing.assert_array_equal(with_body.X, bare.X)
        np.testing.assert_array_equal(with_body.Y, bare.Y)


def test_keypoint_and_confidence_draws_come_from_their_own_streams(
    tmp_path: Path,
) -> None:
    base = simulate(body_spec(tmp_path, frames=40, body={"keypoint_sd": 1.0}))
    other_confidence = simulate(
        body_spec(tmp_path, frames=40, body={"keypoint_sd": 1.0, "confidence_sd": 0.3})
    )
    other_noise = simulate(
        body_spec(tmp_path, frames=40, body={"keypoint_sd": 4.0, "confidence_sd": 0.05})
    )

    for k in range(5):
        np.testing.assert_array_equal(
            base.pose[f"poseX{k}"], other_confidence.pose[f"poseX{k}"]
        )
        np.testing.assert_array_equal(
            base.pose[f"poseP{k}"], other_noise.pose[f"poseP{k}"]
        )
    assert not np.array_equal(base.pose["poseP0"], other_confidence.pose["poseP0"])
    assert not np.array_equal(base.pose["poseX0"], other_noise.pose["poseX0"])


@pytest.mark.parametrize("arena", [BOX_2D, BOX_3D], ids=["box_2d", "box_3d"])
def test_keypoints_and_confidence_do_not_depend_on_the_chunk_size(
    tmp_path: Path, arena: dict[str, object]
) -> None:
    spec = body_spec(
        tmp_path,
        **arena,
        frames=60,
        individuals=4,
        burn_in_frames=11,
        body={"keypoint_sd": 1.5},
    )

    first = simulate(spec)
    assert first.pose
    assert_columns_equal(first, simulate(spec, chunk_frames=1))
    assert_columns_equal(first, simulate(spec, chunk_frames=7))


def test_burn_in_frames_draw_noise_and_confidence_like_any_other_frame(
    tmp_path: Path,
) -> None:
    body = {"keypoint_sd": 1.5}
    burned = simulate(body_spec(tmp_path, frames=20, burn_in_frames=15, body=body))
    longer = simulate(body_spec(tmp_path, frames=35, body=body))

    kept = longer.frame >= 15
    for name, array in burned.pose.items():
        np.testing.assert_array_equal(array, longer.pose[name][kept], err_msg=name)


def test_the_generator_holds_the_pose_columns_and_one_chunk_of_draws(
    tmp_path: Path,
) -> None:
    keypoints = 5
    spec = body_spec(
        tmp_path,
        frames=2_000,
        individuals=100,
        body={"keypoints": keypoints, "keypoint_sd": 1.0},
    )
    chunk = 64

    tracemalloc.start()
    try:
        columns = simulate(spec, chunk_frames=chunk)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    returned = sum(array.nbytes for _, array in columns.named_columns())
    assert returned == (2 + 2 + 3 * keypoints) * 200_000 * 8
    per_chunk_frame = 100 * 8 * (3 * 2 + 4 * keypoints * 2 + 4 * keypoints)
    assert peak <= 1.05 * returned + chunk * per_chunk_frame


ID_DROPOUTS = {"id_rate": 0.05, "id_mean_frames": 10.0}
GAP_DROPOUTS = {"gap_rate": 0.1, "gap_mean_frames": 5.0}
BOTH_DROPOUTS = {
    "id_rate": 0.1,
    "id_mean_frames": 4.0,
    "gap_rate": 0.05,
    "gap_mean_frames": 3.0,
}


def presence(columns: SyntheticColumns, frames: int, individuals: int) -> Bools:
    """Which ``(frame, individual)`` cells a run writes, for ids in range."""
    written = np.zeros((frames, individuals), dtype=np.bool_)
    written[columns.frame, columns.id] = True
    return written


def run_lengths(flags: Bools) -> list[int]:
    """The lengths of the runs of ``True`` down each column of ``flags``."""
    padded = np.zeros((flags.shape[0] + 2, flags.shape[1]), dtype=np.int8)
    padded[1:-1] = flags
    edges = np.diff(padded.T.reshape(-1))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return (stops - starts).tolist()


def test_the_missing_fraction_and_run_length_of_an_id_follow_its_chain(
    tmp_path: Path,
) -> None:
    frames, individuals = 10_000, 20
    columns = simulate(
        make_spec(
            tmp_path, frames=frames, individuals=individuals, dropouts=ID_DROPOUTS
        )
    )

    missing = ~presence(columns, frames, individuals)
    stationary = 0.05 / (0.05 + 1 / 10.0)
    assert missing.mean() == pytest.approx(stationary, rel=0.1)
    assert np.mean(run_lengths(missing)) == pytest.approx(10.0, rel=0.1)
    assert columns.X.shape == ((~missing).sum(),)


def test_frame_gaps_remove_whole_frames_at_their_stationary_rate(
    tmp_path: Path,
) -> None:
    frames = 40_000
    columns = simulate(
        make_spec(tmp_path, frames=frames, individuals=1, dropouts=GAP_DROPOUTS)
    )

    in_gap = ~presence(columns, frames, 1)
    stationary = 0.1 / (0.1 + 1 / 5.0)
    assert in_gap.mean() == pytest.approx(stationary, rel=0.1)
    assert np.mean(run_lengths(in_gap)) == pytest.approx(5.0, rel=0.1)


def test_a_frame_gap_removes_every_individual_of_that_frame(tmp_path: Path) -> None:
    frames, individuals = 500, 6
    columns = simulate(
        make_spec(
            tmp_path, frames=frames, individuals=individuals, dropouts=GAP_DROPOUTS
        )
    )

    written = presence(columns, frames, individuals)
    assert written.all(axis=1).sum() + (~written).all(axis=1).sum() == frames
    assert (~written).all(axis=1).any()


def test_zero_rates_write_every_row_like_a_file_without_dropouts(
    tmp_path: Path,
) -> None:
    arguments: dict[str, object] = {"frames": 40, "individuals": 4}
    plain = simulate(make_spec(tmp_path, **arguments))
    quiet = simulate(
        make_spec(tmp_path, **arguments, dropouts={"id_rate": 0.0, "gap_rate": 0.0}),
        relabel_after_gap=True,
    )

    assert_columns_equal(plain, quiet)


@pytest.mark.parametrize("arena", [BOX_2D, DISK_3D], ids=["box_2d", "disk_3d"])
def test_dropouts_drop_rows_and_leave_the_positions_of_the_others_unchanged(
    tmp_path: Path, arena: dict[str, object]
) -> None:
    arguments: dict[str, object] = {
        **arena,
        "frames": 120,
        "individuals": 5,
        "burn_in_frames": 7,
        "body": {"keypoints": 3, "length": 20.0, "keypoint_sd": 1.0},
        "zones": {"repulsion": 10.0, "orientation": 60.0, "attraction": 150.0},
    }
    plain = simulate(make_spec(tmp_path, **arguments))
    dropped = simulate(make_spec(tmp_path, **arguments, dropouts=BOTH_DROPOUTS))

    keys = plain.frame * 5 + plain.id
    kept = np.isin(keys, dropped.frame * 5 + dropped.id)
    assert 0 < kept.sum() < keys.size
    plain_columns = dict(plain.named_columns())
    for name, array in dropped.named_columns():
        np.testing.assert_array_equal(array, plain_columns[name][kept], err_msg=name)


def reference_rows(written: Bools) -> list[tuple[int, int, int]]:
    """The ``(frame, id, individual)`` of each row a run writes with relabeling.

    An individual missing at its previous frame takes the next unused id at its
    next written row, and every individual starts present at the first frame.
    Rows are ordered by frame and then id, and ``individual`` is the index whose
    walk the row carries.
    """
    frames, individuals = written.shape
    current = list(range(individuals))
    absent = [False] * individuals
    next_id = individuals
    rows: list[tuple[int, int, int]] = []
    for frame in range(frames):
        frame_ids: list[tuple[int, int]] = []
        for individual in range(individuals):
            if not written[frame, individual]:
                continue
            if absent[individual]:
                current[individual] = next_id
                next_id += 1
            frame_ids.append((current[individual], individual))
        rows.extend((frame, identity, owner) for identity, owner in sorted(frame_ids))
        absent = [not written[frame, individual] for individual in range(individuals)]
    return rows


@pytest.mark.parametrize("arena", [BOX_2D, BOX_3D], ids=["box_2d", "box_3d"])
def test_relabeled_rows_are_ordered_by_frame_and_then_id_with_every_column_carried(
    tmp_path: Path, arena: dict[str, object]
) -> None:
    arguments: dict[str, object] = {
        **arena,
        "frames": 300,
        "individuals": 5,
        "burn_in_frames": 6,
        "body": {"keypoints": 3, "length": 20.0, "keypoint_sd": 1.0},
        "dropouts": BOTH_DROPOUTS,
    }
    plain = simulate(make_spec(tmp_path, **arguments))
    relabeled = simulate(make_spec(tmp_path, **arguments), relabel_after_gap=True)

    same_frame = np.diff(relabeled.frame) == 0
    assert same_frame.any()
    assert (np.diff(relabeled.id)[same_frame] > 0).all()
    assert (np.diff(relabeled.frame) >= 0).all()

    frame_id_owner = np.array(reference_rows(presence(plain, 300, 5)), dtype=np.int64)
    np.testing.assert_array_equal(relabeled.frame, frame_id_owner[:, 0])
    np.testing.assert_array_equal(relabeled.id, frame_id_owner[:, 1])
    plain_keys = plain.frame * 5 + plain.id
    source = np.searchsorted(
        plain_keys, frame_id_owner[:, 0] * 5 + frame_id_owner[:, 2]
    )
    np.testing.assert_array_equal(
        plain_keys[source], relabeled.frame * 5 + frame_id_owner[:, 2]
    )
    plain_columns = dict(plain.named_columns())
    for name, array in relabeled.named_columns():
        if name not in ("frame", "id"):
            np.testing.assert_array_equal(
                array, plain_columns[name][source], err_msg=name
            )


@pytest.mark.parametrize("burn_in", [0, 15], ids=["no_burn_in", "burn_in"])
def test_an_individual_returning_after_a_dropout_or_a_gap_takes_the_next_unused_id(
    tmp_path: Path, burn_in: int
) -> None:
    frames, individuals = 300, 4
    arguments: dict[str, object] = {
        "frames": frames,
        "individuals": individuals,
        "burn_in_frames": burn_in,
        "dropouts": BOTH_DROPOUTS,
    }
    plain = simulate(make_spec(tmp_path, **arguments))
    relabeled = simulate(make_spec(tmp_path, **arguments), relabel_after_gap=True)

    written = presence(plain, frames, individuals)
    frame_id_owner = np.array(reference_rows(written), dtype=np.int64)
    np.testing.assert_array_equal(relabeled.frame, frame_id_owner[:, 0])
    np.testing.assert_array_equal(relabeled.id, frame_id_owner[:, 1])
    np.testing.assert_array_equal(relabeled.frame, plain.frame)
    assert relabeled.id.max() > individuals


def test_a_relabeled_id_is_one_unbroken_run_and_the_new_ids_count_up_without_a_hole(
    tmp_path: Path,
) -> None:
    columns = simulate(
        make_spec(tmp_path, frames=400, individuals=5, dropouts=BOTH_DROPOUTS),
        relabel_after_gap=True,
    )

    unique = np.unique(columns.id)
    new_ids = unique[unique >= 5]
    np.testing.assert_array_equal(new_ids, np.arange(5, new_ids.size + 5))
    assert new_ids.size > 5
    for identity in unique:
        frames = columns.frame[columns.id == identity]
        assert (np.diff(frames) == 1).all(), identity
    assert np.bincount(columns.frame).max() <= 5


def test_without_relabeling_ids_stay_in_range(tmp_path: Path) -> None:
    columns = simulate(
        make_spec(tmp_path, frames=400, individuals=5, dropouts=BOTH_DROPOUTS)
    )

    assert columns.id.min() == 0
    assert columns.id.max() == 4
    gaps = [np.diff(columns.frame[columns.id == i]) for i in range(5)]
    assert any((gap > 1).any() for gap in gaps)


def test_relabeling_without_dropouts_changes_nothing(tmp_path: Path) -> None:
    spec = make_spec(tmp_path, frames=30, individuals=3)

    assert_columns_equal(simulate(spec), simulate(spec, relabel_after_gap=True))


@pytest.mark.parametrize("relabel", [False, True], ids=["ids", "relabeled"])
@pytest.mark.parametrize("arena", [BOX_2D, BOX_3D], ids=["box_2d", "box_3d"])
def test_dropouts_and_relabeling_do_not_depend_on_the_chunk_size(
    tmp_path: Path, arena: dict[str, object], relabel: bool
) -> None:
    spec = body_spec(
        tmp_path,
        **arena,
        frames=90,
        individuals=4,
        burn_in_frames=11,
        body={"keypoint_sd": 1.5},
        dropouts=BOTH_DROPOUTS,
    )

    first = simulate(spec, relabel_after_gap=relabel)
    assert first.frame.size < 90 * 4
    assert_columns_equal(
        first, simulate(spec, relabel_after_gap=relabel, chunk_frames=1)
    )
    assert_columns_equal(
        first, simulate(spec, relabel_after_gap=relabel, chunk_frames=7)
    )


def test_burn_in_frames_advance_the_chains_like_any_other_frame(
    tmp_path: Path,
) -> None:
    arguments: dict[str, object] = {
        "individuals": 4,
        "body": {"keypoints": 2, "length": 10.0, "keypoint_sd": 1.0},
        "dropouts": BOTH_DROPOUTS,
    }
    burned = simulate(make_spec(tmp_path, **arguments, frames=60, burn_in_frames=25))
    longer = simulate(make_spec(tmp_path, **arguments, frames=85))

    kept = longer.frame >= 25
    np.testing.assert_array_equal(burned.frame, longer.frame[kept] - 25)
    longer_columns = dict(longer.named_columns())
    for name, array in burned.named_columns():
        if name != "frame":
            np.testing.assert_array_equal(
                array, longer_columns[name][kept], err_msg=name
            )


def test_the_chains_enter_the_first_written_frame_in_their_stationary_state(
    tmp_path: Path,
) -> None:
    arguments: dict[str, object] = {
        "frames": 1,
        "individuals": 300,
        "dropouts": ID_DROPOUTS,
    }
    cold = simulate(make_spec(tmp_path, **arguments))
    warm = simulate(make_spec(tmp_path, **arguments, burn_in_frames=200))

    assert 1 - cold.id.size / 300 < 0.12
    assert 1 - warm.id.size / 300 == pytest.approx(1 / 3, abs=0.13)


def test_the_generator_holds_the_columns_and_one_chunk_with_dropouts(
    tmp_path: Path,
) -> None:
    spec = make_spec(
        tmp_path,
        frames=5_000,
        individuals=100,
        motion={"speed_sd": 30.0},
        dropouts=BOTH_DROPOUTS,
    )
    chunk = 64

    tracemalloc.start()
    try:
        columns = simulate(spec, chunk_frames=chunk, relabel_after_gap=True)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    allocated = 4 * 500_000 * 8
    assert columns.X.size < 500_000
    assert peak <= 1.05 * allocated + 40 * chunk * 100 * 8
