"""``visual-network``: who each individual sees, by ray casting from its eyes.

The reference networks were computed by the ray caster of Davidson et al. (2021)
on two golden-shiner frames, and ``apply`` must reproduce every edge and every
ray count. The synthetic cases each place a few fish so that one rule decides
the answer: occlusion, the blind wedge, angular size, arenas, missing values.
"""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter, ValidationError

from mosaic.behavior.feature_library.visual_network import VisualNetwork
from tests.helpers import TESTS_ROOT

REFERENCE = TESTS_ROOT / "data" / "visual_network_reference.json"

EYES: dict[str, object] = {"tail_index": 1, "left_eye_index": 2, "right_eye_index": 3}
"""The keypoint layout of :func:`_fish`: head 0, tail 1, left eye 2, right eye 3."""

HEAD_ONLY: dict[str, object] = {"tail_index": 1}
"""One viewpoint at the head, the outline the head-to-tail segment."""


def _row(
    identity: int,
    head: tuple[float, float],
    heading: float,
    tail: tuple[float, float],
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
    frame: int = 0,
) -> dict[str, object]:
    return {
        "frame": frame,
        "id": identity,
        "group": "g",
        "sequence": "s",
        "X": (head[0] + tail[0]) / 2,
        "Y": (head[1] + tail[1]) / 2,
        "ANGLE": heading,
        "poseX0": head[0],
        "poseY0": head[1],
        "poseX1": tail[0],
        "poseY1": tail[1],
        "poseX2": left_eye[0],
        "poseY2": left_eye[1],
        "poseX3": right_eye[0],
        "poseY3": right_eye[1],
    }


def _fish(
    identity: int,
    x: float,
    y: float,
    heading: float,
    length: float = 10.0,
    frame: int = 0,
) -> dict[str, object]:
    """A fish centered at (x, y), in image coordinates (y down).

    The right side is a positive turn from the heading, so the right eye sits at
    the heading plus 90 degrees.
    """
    forward = np.array([math.cos(heading), math.sin(heading)])
    right = np.array([-math.sin(heading), math.cos(heading)])
    center = np.array([x, y])
    head = center + forward * length / 2
    tail = center - forward * length / 2
    eye_base = head - forward * length * 0.1
    left_eye = eye_base - right * length * 0.1
    right_eye = eye_base + right * length * 0.1
    return _row(
        identity,
        (float(head[0]), float(head[1])),
        heading,
        (float(tail[0]), float(tail[1])),
        (float(left_eye[0]), float(left_eye[1])),
        (float(right_eye[0]), float(right_eye[1])),
        frame,
    )


def _edges(out: pd.DataFrame) -> dict[tuple[int, int], int]:
    return {
        (int(a), int(b)): int(n)
        for a, b, n in zip(out["id1"], out["id2"], out["n_rays"], strict=True)
    }


class _ReferenceFish(TypedDict):
    id: int
    head: tuple[float, float]
    heading: float
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    tail: tuple[float, float]


class _ReferenceCase(TypedDict):
    name: str
    fish: list[_ReferenceFish]
    edges: list[tuple[int, int, int]]


class _Reference(TypedDict):
    cases: list[_ReferenceCase]


def _reference_cases() -> list[_ReferenceCase]:
    reference = TypeAdapter(_Reference).validate_json(REFERENCE.read_bytes())
    return reference["cases"]


@pytest.mark.parametrize("case", _reference_cases(), ids=lambda case: case["name"])
def test_the_reference_networks_are_reproduced_ray_for_ray(
    case: _ReferenceCase,
) -> None:
    rows = [
        _row(f["id"], f["head"], f["heading"], f["tail"], f["left_eye"], f["right_eye"])
        for f in case["fish"]
    ]
    out = VisualNetwork(params=EYES).apply(pd.DataFrame(rows))

    expected = {(a, b): n for a, b, n in case["edges"]}
    assert _edges(out) == expected


def _line_of_three() -> list[dict[str, object]]:
    """An observer facing +x, a broadside fish ahead, and a smaller one behind it."""
    return [
        _fish(0, 0.0, 0.0, 0.0),
        _fish(1, 20.0, 0.0, math.pi / 2, length=10.0),
        _fish(2, 40.0, 0.0, math.pi / 2, length=4.0),
    ]


def test_a_nearer_fish_hides_the_one_behind_it() -> None:
    feature = VisualNetwork(params=EYES)
    hidden = _edges(feature.apply(pd.DataFrame(_line_of_three())))
    assert (0, 1) in hidden
    assert (0, 2) not in hidden

    without_middle = [row for row in _line_of_three() if row["id"] != 1]
    shown = _edges(feature.apply(pd.DataFrame(without_middle)))
    assert (0, 2) in shown


def test_a_fish_in_the_blind_wedge_is_not_seen() -> None:
    behind = (-20.0, 0.0)
    off_heading = math.radians(150.0)
    beside = (20.0 * math.cos(off_heading), 20.0 * math.sin(off_heading))
    rows = [
        _fish(0, 0.0, 0.0, 0.0),
        _fish(1, *behind, math.pi / 2, length=4.0),
        _fish(2, *beside, off_heading + math.pi / 2, length=4.0),
    ]
    out = _edges(VisualNetwork(params=HEAD_ONLY).apply(pd.DataFrame(rows)))
    assert (0, 1) not in out
    assert (0, 2) in out

    no_wedge = VisualNetwork(params={**HEAD_ONLY, "blind_angle_deg": 0.0})
    assert (0, 1) in _edges(no_wedge.apply(pd.DataFrame(rows)))


def test_angular_size_matches_the_subtended_angle_within_one_ray() -> None:
    length, distance, n_rays = 10.0, 50.0, 1000
    rows = [
        _fish(0, 0.0, 0.0, 0.0, length=2.0),
        _fish(1, distance + 1.0, 0.0, math.pi / 2, length=length),
    ]
    feature = VisualNetwork(params={**HEAD_ONLY, "n_rays": n_rays})
    out = feature.apply(pd.DataFrame(rows))
    row = out[(out["id1"] == 0) & (out["id2"] == 1)]
    assert len(row) == 1

    # The head sits 1.0 ahead of the observer's center, so the target is at
    # `distance` from the viewpoint.
    expected = math.degrees(2 * math.atan(length / (2 * distance)))
    assert abs(float(row["angular_size_deg"].iloc[0]) - expected) <= 360 / n_rays
    assert float(row["distance"].iloc[0]) == pytest.approx(distance + 1.0)


def test_a_fish_of_another_arena_neither_is_seen_nor_hides() -> None:
    rows = _line_of_three()
    for row, arena in zip(rows, ("a", "b", "a"), strict=True):
        row["arena"] = arena
    out = _edges(VisualNetwork(params=EYES).apply(pd.DataFrame(rows)))

    assert (0, 2) in out
    assert not any(1 in pair for pair in out)


def test_an_individual_missing_in_a_frame_is_skipped_there_alone() -> None:
    first = _line_of_three()
    second = [{**row, "frame": 1} for row in _line_of_three()]
    second[1]["poseX0"] = float("nan")
    out = VisualNetwork(params=EYES).apply(pd.DataFrame(first + second))

    at_first = _edges(out[out["frame"] == 0])
    at_second = _edges(out[out["frame"] == 1])
    assert (0, 1) in at_first and (0, 2) not in at_first
    assert not any(1 in pair for pair in at_second)
    assert (0, 2) in at_second

    alone = [row for row in _line_of_three() if row["id"] != 1]
    assert at_second == _edges(VisualNetwork(params=EYES).apply(pd.DataFrame(alone)))


def _four() -> list[dict[str, object]]:
    return [
        _fish(0, 0.0, 0.0, 0.0),
        _fish(1, 20.0, 0.0, math.pi / 2),
        _fish(2, 40.0, 0.0, math.pi / 2, length=4.0),
        _fish(3, 0.0, 30.0, math.pi),
    ]


def test_include_hidden_writes_every_ordered_pair_with_its_flag() -> None:
    rows = _four() + [{**row, "frame": 1} for row in _four()]
    sparse = VisualNetwork(params={**EYES, "min_rays": 3}).apply(pd.DataFrame(rows))
    dense = VisualNetwork(params={**EYES, "min_rays": 3, "include_hidden": True}).apply(
        pd.DataFrame(rows)
    )

    assert "visible" not in sparse.columns
    assert dense.groupby("frame").size().tolist() == [12, 12]
    assert dense["visible"].dtype == bool
    assert (dense["visible"] == (dense["n_rays"] >= 3)).all()
    assert not dense["visible"].all()

    shown = dense[dense["visible"]].drop(columns="visible").reset_index(drop=True)
    pd.testing.assert_frame_equal(shown, sparse.reset_index(drop=True))


def test_perspective_says_which_ordering_a_row_is() -> None:
    out = VisualNetwork(params=EYES).apply(pd.DataFrame(_four()))
    assert out["perspective"].dtype == np.int64
    assert ((out["id1"] < out["id2"]) == (out["perspective"] == 0)).all()


def test_one_eye_without_the_other_is_refused() -> None:
    with pytest.raises(ValidationError, match="right_eye_index"):
        VisualNetwork(params={"left_eye_index": 2})


def test_the_tail_defaults_to_the_last_keypoint() -> None:
    rows = pd.DataFrame(_four()).drop(columns=["poseX2", "poseY2", "poseX3", "poseY3"])
    implicit = VisualNetwork().apply(rows)
    explicit = VisualNetwork(params=HEAD_ONLY).apply(rows)
    pd.testing.assert_frame_equal(implicit, explicit)
    assert not implicit.empty


def test_heading_from_keypoints_matches_the_angle_column() -> None:
    rows = pd.DataFrame(_four())
    from_angle = VisualNetwork(params=EYES).apply(rows)
    from_keypoints = VisualNetwork(
        params={**EYES, "heading_source": "keypoints", "heading_back_index": 1}
    ).apply(rows.drop(columns="ANGLE"))
    pd.testing.assert_frame_equal(from_angle, from_keypoints)


def test_an_empty_table_gives_typed_empty_output() -> None:
    rows = pd.DataFrame(_four()).iloc[:0]
    out = VisualNetwork(params=EYES).apply(rows)
    assert out.empty
    assert {"frame", "id1", "id2", "perspective", "n_rays"} <= set(out.columns)
    assert out["n_rays"].dtype == np.int64
    assert out["perspective"].dtype == np.int64
