"""Partner keypoints and velocities in the focal's body frame, on scenes with known answers.

``pair-keypoints`` places each partner keypoint, and its velocity, in the focal
animal's body frame, and measures chosen focal-to-partner keypoint distances. A
partner standing straight ahead and a partner crossing the focal's path check
the axes; a rotated scene checks that nothing depends on the world frame; and a
focal with a gap in its track checks that the partner's velocity is taken on
the partner's own frames, not on the frames the two share.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.keypoint_kinematics import KeypointKinematics
from mosaic.behavior.feature_library.pair_keypoints import PairKeypoints
from mosaic.core.pipeline.column_contract import Prefix

type FloatArray = npt.NDArray[np.float64]

FPS = 100.0
PARTNER_PREFIXES = ("partner_x__", "partner_y__", "partner_vx__", "partner_vy__")


def _individual(
    keypoints: list[FloatArray],
    individual: int,
    *,
    frames: npt.NDArray[np.int64] | None = None,
    arena: str | None = None,
) -> pd.DataFrame:
    """One individual's rows from ``(n, 2)`` or ``(n, 3)`` keypoint arrays."""
    n = keypoints[0].shape[0]
    frame = np.arange(n, dtype=np.int64) if frames is None else frames
    n_axes = keypoints[0].shape[1]
    columns: dict[str, object] = {
        "frame": frame,
        "time": frame / FPS,
        "id": np.full(n, individual, dtype=np.int64),
        "group": [""] * n,
        "sequence": ["seq"] * n,
    }
    if arena is not None:
        columns["arena"] = [arena] * n
    for axis, name in zip(range(n_axes), "XYZ", strict=False):
        columns[name] = keypoints[0][:, axis]
    for k, points in enumerate(keypoints):
        for axis, name in zip(range(n_axes), "XYZ", strict=False):
            columns[f"pose{name}{k}"] = points[:, axis]
    return pd.DataFrame(columns)


def _scene(*individuals: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(individuals, ignore_index=True)


def _apply(table: pd.DataFrame, **params: object) -> pd.DataFrame:
    return PairKeypoints(params=params).apply(table)


def _walk(n: int, seed: int) -> FloatArray:
    """A random walk, so no smoothing reproduces it by accident."""
    return np.random.default_rng(seed).normal(size=(n, 2)).cumsum(axis=0)


def _body(center: FloatArray, angle: float) -> list[FloatArray]:
    """Neck, tail base and a flank keypoint of a body 0.4 long facing *angle*.

    The flank keypoint sits 0.1 to the left of the body center.
    """
    along = np.array([np.cos(angle), np.sin(angle)])
    left = np.array([-np.sin(angle), np.cos(angle)])
    return [center + 0.2 * along, center - 0.2 * along, center + 0.1 * left]


def _perspective(out: pd.DataFrame, perspective: int) -> pd.DataFrame:
    return out[out["perspective"] == perspective].reset_index(drop=True)


def _values(out: pd.DataFrame, column: str) -> FloatArray:
    return out[column].to_numpy(dtype=np.float64)


def _measures(out: pd.DataFrame) -> list[str]:
    return [c for c in out.columns if c.startswith((*PARTNER_PREFIXES, "inter_dist__"))]


# --- Known answers ---


@pytest.mark.parametrize("angle", [0.0, 0.9, 2.6, -1.7])
def test_a_partner_straight_ahead_is_on_the_heading_axis(angle: float) -> None:
    n, d = 30, 3.0
    along = np.array([np.cos(angle), np.sin(angle)])
    focal_center = np.tile([1.0, -2.0], (n, 1))
    partner_center = focal_center + d * along
    table = _scene(
        _individual(_body(focal_center, angle), 0),
        _individual(_body(partner_center, angle), 1),
    )
    out = _perspective(_apply(table, fps=FPS, heading=(0, 1)), 0)

    np.testing.assert_allclose(out["partner_x__kp0"], d + 0.2, rtol=1e-5)
    np.testing.assert_allclose(out["partner_x__kp1"], d - 0.2, rtol=1e-5)
    np.testing.assert_allclose(out["partner_x__kp2"], d, rtol=1e-5)
    np.testing.assert_allclose(out["partner_y__kp0"], 0.0, atol=1e-5)
    np.testing.assert_allclose(out["partner_y__kp1"], 0.0, atol=1e-5)
    np.testing.assert_allclose(out["partner_y__kp2"], 0.1, rtol=1e-5)


def test_a_rotated_scene_has_the_same_columns() -> None:
    focal = [_walk(120, 1), _walk(120, 2), _walk(120, 3)]
    partner = [_walk(120, 4), _walk(120, 5), _walk(120, 6)]
    angle = 1.1
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )

    def turned(points: list[FloatArray]) -> list[FloatArray]:
        return [p @ rotation.T for p in points]

    params: dict[str, object] = {
        "fps": FPS,
        "heading": (0, 1),
        "inter_distances": [(0, 2), (2, 2)],
    }
    out = _apply(_scene(_individual(focal, 0), _individual(partner, 1)), **params)
    out_turned = _apply(
        _scene(_individual(turned(focal), 0), _individual(turned(partner), 1)),
        **params,
    )
    columns = _measures(out)
    assert len(columns) == 14
    np.testing.assert_allclose(out_turned[columns], out[columns], rtol=1e-4, atol=1e-4)


def test_a_partner_crossing_the_focals_path_moves_to_its_right() -> None:
    """The focal faces world +Y and the partner moves along world +X at speed v."""
    n, v = 50, 0.8
    t = np.arange(n) / FPS
    focal_center = np.tile([0.0, 0.0], (n, 1))
    partner_center = np.column_stack([v * t, np.full(n, 2.0)])
    table = _scene(
        _individual(_body(focal_center, np.pi / 2), 0),
        _individual(_body(partner_center, 0.0), 1),
    )
    out = _perspective(_apply(table, fps=FPS, heading=(0, 1)), 0)

    for k in range(3):
        np.testing.assert_allclose(out[f"partner_vx__kp{k}"], 0.0, atol=1e-5)
        np.testing.assert_allclose(out[f"partner_vy__kp{k}"], -v, rtol=1e-5)


def test_the_partners_velocity_is_taken_on_the_partners_own_frames() -> None:
    n = 60
    focal_frames = np.setdiff1d(np.arange(n), np.arange(20, 30)).astype(np.int64)
    focal = [_walk(n, s)[focal_frames] for s in (1, 2, 3)]
    partner = [_walk(n, s) for s in (4, 5, 6)]
    partner_rows = _individual(partner, 1)
    table = _scene(_individual(focal, 0, frames=focal_frames), partner_rows)
    params: dict[str, object] = {"fps": FPS, "window": 7, "polyorder": 2}

    out = _perspective(_apply(table, heading=(0, 1), **params), 0)
    alone = KeypointKinematics(params=params).apply(partner_rows)

    np.testing.assert_array_equal(out["frame"], focal_frames)
    speed = alone.set_index("frame").loc[focal_frames]
    for k in range(3):
        own = np.hypot(
            _values(out, f"partner_vx__kp{k}"), _values(out, f"partner_vy__kp{k}")
        )
        world = speed[f"speed__kp{k}"].to_numpy(dtype=np.float64)
        assert np.isfinite(own).all()
        np.testing.assert_allclose(own, world, rtol=1e-4)
    # The frames whose derivative windows cover the focal's missing frames.
    assert {15, 19, 30, 34} <= set(out["frame"].tolist())


# --- Pairs and perspectives ---


def test_both_perspectives_are_written_and_the_ids_swap() -> None:
    focal = [_walk(40, s) for s in (1, 2, 3)]
    partner = [_walk(40, s) for s in (4, 5, 6)]
    out = _apply(
        _scene(_individual(focal, 0), _individual(partner, 1)),
        fps=FPS,
        heading=(0, 1),
    )

    assert set(out["perspective"]) == {0, 1}
    first = _perspective(out, 0)
    second = _perspective(out, 1)
    assert set(first["id1"]) == {0} and set(first["id2"]) == {1}
    assert set(second["id1"]) == {1} and set(second["id2"]) == {0}
    assert not out.duplicated(subset=["frame", "id1", "id2"]).any()
    assert list(out.columns[:6]) == [
        "frame",
        "perspective",
        "id1",
        "id2",
        "group",
        "sequence",
    ]
    assert out[["frame", "id1", "id2"]].equals(
        out[["frame", "id1", "id2"]].sort_values(["frame", "id1", "id2"])
    )
    assert pd.api.types.is_integer_dtype(out["perspective"])
    assert all(out[c].dtype == np.float32 for c in _measures(out))


def test_an_inter_distance_runs_from_the_focal_to_the_partner() -> None:
    a = [_walk(40, s) for s in (1, 2, 3)]
    b = [_walk(40, s) for s in (4, 5, 6)]
    table = _scene(_individual(a, 0), _individual(b, 1))
    out = _apply(table, fps=FPS, heading=(0, 1), inter_distances=[(0, 1), (1, 0)])

    first = _perspective(out, 0)
    second = _perspective(out, 1)
    np.testing.assert_allclose(
        first["inter_dist__kp0__kp1"], np.linalg.norm(a[0] - b[1], axis=1), rtol=1e-5
    )
    np.testing.assert_allclose(
        second["inter_dist__kp0__kp1"], np.linalg.norm(b[0] - a[1], axis=1), rtol=1e-5
    )
    np.testing.assert_allclose(
        first["inter_dist__kp1__kp0"], np.linalg.norm(a[1] - b[0], axis=1), rtol=1e-5
    )
    assert [c for c in out.columns if c.startswith("inter_dist__")] == [
        "inter_dist__kp0__kp1",
        "inter_dist__kp1__kp0",
    ]


def test_the_columns_follow_the_selected_keypoints_and_distances() -> None:
    focal = [_walk(40, s) for s in (1, 2, 3)]
    partner = [_walk(40, s) for s in (4, 5, 6)]
    table = _scene(_individual(focal, 0), _individual(partner, 1))
    names = ["neck", "tail_base", "paw"]

    out = _apply(
        table,
        fps=FPS,
        keypoint_names=names,
        keypoints=["paw", "neck"],
        heading=("neck", "tail_base"),
    )
    assert _measures(out) == [
        "partner_x__paw",
        "partner_x__neck",
        "partner_y__paw",
        "partner_y__neck",
        "partner_vx__paw",
        "partner_vx__neck",
        "partner_vy__paw",
        "partner_vy__neck",
    ]

    only = _apply(
        table, fps=FPS, keypoints=[], heading=(0, 1), inter_distances=[(2, 0)]
    )
    assert _measures(only) == ["inter_dist__kp2__kp0"]


def test_each_output_prefix_is_declared() -> None:
    declared = {
        (conditional.column.prefix, conditional.when_param)
        for conditional in PairKeypoints.provides.conditional
        if isinstance(conditional.column, Prefix)
    }
    for prefix in PARTNER_PREFIXES:
        assert (prefix, "") in declared
    assert ("inter_dist__", "inter_distances") in declared


def test_individuals_of_two_arenas_are_not_paired() -> None:
    bodies = [[_walk(30, 3 * i + s) for s in (1, 2, 3)] for i in range(3)]
    table = _scene(
        _individual(bodies[0], 0, arena="a"),
        _individual(bodies[1], 1, arena="b"),
        _individual(bodies[2], 2, arena="a"),
    )
    out = _apply(table, fps=FPS, heading=(0, 1))

    pairs = set(map(tuple, out[["id1", "id2"]].drop_duplicates().to_numpy().tolist()))
    assert pairs == {(0, 2), (2, 0)}


def test_fewer_than_two_individuals_give_an_empty_frame() -> None:
    out = _apply(
        _individual([_walk(30, s) for s in (1, 2)], 0), fps=FPS, heading=(0, 1)
    )

    assert out.empty
    assert {"frame", "perspective", "id1", "id2"} <= set(out.columns)


# --- What is NaN ---


def test_a_missing_heading_keypoint_or_partner_keypoint_is_nan() -> None:
    focal = [_walk(60, s) for s in (1, 2, 3)]
    partner = [_walk(60, s) for s in (4, 5, 6)]
    focal[0][20] = np.nan  # the focal's neck
    partner[2][40] = np.nan  # one partner keypoint
    table = _scene(_individual(focal, 0), _individual(partner, 1))
    out = _perspective(
        _apply(table, fps=FPS, heading=(0, 1), inter_distances=[(2, 0)]), 0
    )

    frame_columns = [c for c in _measures(out) if c.startswith(PARTNER_PREFIXES)]

    def row(index: int, columns: list[str]) -> FloatArray:
        return out.loc[index, columns].to_numpy(dtype=np.float64)

    assert np.isnan(row(20, frame_columns)).all()
    assert np.isfinite(row(20, ["inter_dist__kp2__kp0"])).all()
    assert np.isnan(row(40, [f"{prefix}kp2" for prefix in PARTNER_PREFIXES])).all()
    assert np.isfinite(row(40, [f"{prefix}kp0" for prefix in PARTNER_PREFIXES])).all()
    others = out.drop(index=[20, 40])
    assert others[frame_columns].notna().all().all()


# --- Frame rate and refusals ---


def test_the_frame_rate_is_read_off_the_time_column() -> None:
    table = _scene(
        _individual([_walk(40, s) for s in (1, 2)], 0),
        _individual([_walk(40, s) for s in (3, 4)], 1),
    )
    pd.testing.assert_frame_equal(
        _apply(table, heading=(0, 1)), _apply(table, fps=FPS, heading=(0, 1))
    )


def test_without_a_frame_rate_or_a_time_column_it_refuses() -> None:
    table = _scene(
        _individual([_walk(40, s) for s in (1, 2)], 0),
        _individual([_walk(40, s) for s in (3, 4)], 1),
    ).drop(columns="time")
    with pytest.raises(
        ValueError, match="pair-keypoints reads its rate off the 'time' column"
    ):
        _ = _apply(table, heading=(0, 1))


def test_a_repeated_frame_refuses() -> None:
    table = _scene(
        _individual([_walk(40, s) for s in (1, 2)], 0),
        _individual([_walk(40, s) for s in (3, 4)], 1),
    )
    table = pd.concat([table, table.iloc[[3]]], ignore_index=True)
    with pytest.raises(ValueError, match="repeated frames"):
        _ = _apply(table, fps=FPS, heading=(0, 1))


def test_a_3d_table_refuses_and_points_at_keypoint_kinematics() -> None:
    rng = np.random.default_rng(0)
    table = _scene(
        _individual([rng.normal(size=(30, 3)) for _ in range(2)], 0),
        _individual([rng.normal(size=(30, 3)) for _ in range(2)], 1),
    )
    with pytest.raises(ValueError, match="planar") as refused:
        _ = _apply(table, fps=FPS, heading=(0, 1))
    message = str(refused.value)
    assert "keypoint-kinematics" in message
    assert "speed" not in message


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"heading": (0, 1), "window": 8}, "must be odd"),
        ({"heading": (0, 1), "window": 3, "polyorder": 3}, "must exceed"),
        ({"heading": (0, 1), "polyorder": 0}, "greater than or equal to 1"),
        ({"heading": (0, 1), "keypoint_names": ["a", "a"]}, "repeats a name"),
        ({"heading": (0, 1), "keypoints": ["paw"]}, "not given keypoint names"),
        ({"heading": (0, -1)}, "negative"),
        ({"heading": (0, 1), "inter_distances": [(0, -2)]}, "negative"),
        (
            {"heading": ("neck", "tail"), "keypoint_names": ["neck", "paw"]},
            "no keypoint named",
        ),
        ({"heading": (1, 1)}, "heading .* a keypoint and itself"),
        ({"heading": ("b", 1), "keypoint_names": ["a", "b"]}, "itself"),
        ({"heading": (0, 1), "keypoints": [2, 2]}, "names a keypoint twice"),
        (
            {"heading": (0, 1), "inter_distances": [(0, 1), (0, 1)]},
            "requested twice",
        ),
        (
            {
                "heading": (0, 1),
                "keypoint_names": ["a", "b"],
                "inter_distances": [(0, "b"), ("a", 1)],
            },
            "requested twice",
        ),
        ({"heading": (0, 1), "keypoints": []}, "nothing to compute"),
        ({}, "heading"),
    ],
)
def test_a_parameter_no_table_could_satisfy_is_refused_at_construction(
    params: dict[str, object], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        _ = PairKeypoints(params=params)


def test_a_self_distance_and_both_orders_are_accepted() -> None:
    params = PairKeypoints.Params(
        heading=(0, 1), inter_distances=[(0, 0), (0, 1), (1, 0)]
    )
    assert len(params.inter_distances) == 3


def test_a_keypoint_past_the_tables_keypoints_is_refused_at_apply() -> None:
    table = _scene(
        _individual([_walk(30, s) for s in (1, 2)], 0),
        _individual([_walk(30, s) for s in (3, 4)], 1),
    )
    with pytest.raises(ValueError, match="outside the table's 2 keypoints"):
        _ = _apply(table, fps=FPS, heading=(0, 2))
