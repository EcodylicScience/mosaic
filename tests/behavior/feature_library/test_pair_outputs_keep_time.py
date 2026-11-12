"""A per-frame feature's output carries the ``time`` of each frame it writes.

``temporal-stack`` and the wavelets measure their frame rate from the ``time`` of the
table they read, and a pair output is joined to nothing that could supply it. So each
pair feature keeps the input's ``time`` beside its ``frame``, on a table that dropped
frames as on an even one.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.approach_avoidance import ApproachAvoidance
from mosaic.behavior.feature_library.attention_target import AttentionTarget
from mosaic.behavior.feature_library.body_scale import BodyScaleFeature
from mosaic.behavior.feature_library.orientation_relative import (
    OrientationRelativeFeature,
)
from mosaic.behavior.feature_library.pair_facing import PairFacing
from mosaic.behavior.feature_library.pair_keypoints import PairKeypoints
from mosaic.behavior.feature_library.pair_world import PairWorld
from mosaic.behavior.feature_library.pair_interaction_filter import (
    PairInteractionFilter,
)
from mosaic.behavior.feature_library.temporal_stacking import TemporalStackingFeature
from mosaic.behavior.feature_library.visual_network import VisualNetwork
from mosaic.core.pipeline.types import Result

from tests.helpers import dropped_frame_times

KEYPOINTS = 7
ROWS = 200


def _animal(frame: int, time: float, animal: int, x: float, y: float, angle: float):
    row: dict[str, object] = {
        "frame": frame,
        "time": time,
        "id": animal,
        "X": x,
        "Y": y,
        "ANGLE": angle,
        "sequence": "s1",
        "group": "g1",
    }
    for keypoint in range(KEYPOINTS):
        offset = keypoint - KEYPOINTS // 2
        row[f"poseX{keypoint}"] = x - offset * math.cos(angle)
        row[f"poseY{keypoint}"] = y - offset * math.sin(angle)
    return row


def _facing() -> pd.DataFrame:
    """Animals 0 and 1 face each other at rest on a clock that dropped frames."""
    times = dropped_frame_times(ROWS)
    rows: list[dict[str, object]] = []
    for frame in range(ROWS):
        time = float(times[frame])
        rows.append(_animal(frame, time, 0, 0.0, 0.0, 0.0))
        rows.append(_animal(frame, time, 1, 20.0, 0.0, math.pi))
        rows.append(_animal(frame, time, 2, 500.0, 500.0, 0.0))
    return pd.DataFrame(rows)


def _facing_world() -> pd.DataFrame:
    """:func:`_facing` lifted into 3D, each animal at its own height."""
    table = _facing()
    height = 0.5 * table["id"].to_numpy(dtype=np.float64)
    table["Z"] = height
    for keypoint in range(KEYPOINTS):
        table[f"poseZ{keypoint}"] = height
    return table


def _fish(frame: int, time: float, animal: int, x: float, heading: float):
    """A fish of length 10 in image coordinates, with two eyes."""
    forward = np.array([math.cos(heading), math.sin(heading)])
    right = np.array([-math.sin(heading), math.cos(heading)])
    center = np.array([x, 0.0])
    head = center + forward * 5.0
    tail = center - forward * 5.0
    eye_base = head - forward
    left_eye = eye_base - right
    right_eye = eye_base + right
    row: dict[str, object] = {
        "frame": frame,
        "time": time,
        "id": animal,
        "group": "g",
        "sequence": "s",
        "X": x,
        "Y": 0.0,
        "ANGLE": heading,
    }
    for keypoint, point in enumerate((head, tail, left_eye, right_eye)):
        row[f"poseX{keypoint}"] = float(point[0])
        row[f"poseY{keypoint}"] = float(point[1])
    return row


def _school() -> pd.DataFrame:
    """Three fish in a row, each facing the next, on a clock that dropped frames."""
    times = dropped_frame_times(ROWS)
    rows: list[dict[str, object]] = []
    for frame in range(ROWS):
        time = float(times[frame])
        rows.append(_fish(frame, time, 0, 0.0, 0.0))
        rows.append(_fish(frame, time, 1, 20.0, math.pi / 2))
        rows.append(_fish(frame, time, 2, 40.0, math.pi / 2))
    return pd.DataFrame(rows)


EYES: dict[str, object] = {"tail_index": 1, "left_eye_index": 2, "right_eye_index": 3}

_RUNS: dict[str, Callable[[], pd.DataFrame]] = {
    "pair-facing": lambda: PairFacing().apply(_facing()),
    "orientation-rel": lambda: OrientationRelativeFeature().apply(_facing()),
    "pair-interaction-filter": lambda: PairInteractionFilter(
        params={
            "use_pixel_coords": False,
            "min_run_frames": 5,
            "morphological_structure_size": 0,
            "frame_padding": 0,
        }
    ).apply(_facing()),
    "approach-avoidance": lambda: ApproachAvoidance(
        params={"orientation_gate_cos": None, "min_event_length": 3}
    ).apply(_facing()),
    "attention-target": lambda: AttentionTarget().apply(PairFacing().apply(_facing())),
    "body-scale": lambda: BodyScaleFeature().apply(_facing()),
    "pair-keypoints": lambda: PairKeypoints(params={"heading": [3, 6]}).apply(
        _facing()
    ),
    "pair-world": lambda: PairWorld(params={"front": 0, "rear": 6}).apply(
        _facing_world()
    ),
    "visual-network": lambda: VisualNetwork(params={**EYES, "min_rays": 1}).apply(
        _school()
    ),
}


@pytest.mark.parametrize("slug", sorted(_RUNS))
def test_the_output_keeps_the_time_of_each_frame(slug: str) -> None:
    out = _RUNS[slug]()

    assert not out.empty
    times = dropped_frame_times(ROWS)
    np.testing.assert_allclose(
        out["time"].to_numpy(dtype=np.float64),
        times[out["frame"].to_numpy(dtype=np.int64)],
    )


def test_temporal_stack_pools_over_visual_network_at_its_defaults(
    caplog: pytest.LogCaptureFixture,
) -> None:
    network = VisualNetwork(params={**EYES, "min_rays": 1, "include_hidden": True})
    pairs = network.apply(_school())
    stack = TemporalStackingFeature(
        TemporalStackingFeature.Inputs((Result(feature="visual-network"),))
    )

    with caplog.at_level(logging.WARNING, logger="mosaic.core.timebase"):
        out = stack.apply(pairs)

    assert not out.empty
    assert any(column.endswith("__pool_mean") for column in out.columns)
    assert any("[temporal-stack]" in record.getMessage() for record in caplog.records)
