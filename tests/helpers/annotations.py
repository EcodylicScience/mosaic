"""Keypoint annotation sets, in the shape the annotator's saved state holds them.

The label series, preparation and library suites all save sets and claim their
revisions. One builder keeps the saved shape, and the revision's file name, in
one place.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Final

from mosaic.core.annotations.bbox import BboxPolicy
from mosaic.core.annotations.model import Bbox, Keypoint, KeypointSchema
from mosaic.core.annotations.pose_annotations import (
    ObjectOrigin,
    PoseAnnotationSet,
    PoseDefinition,
    PoseFrame,
    PoseObject,
)
from mosaic.core.pipeline.label_series import series_spec

__all__ = [
    "KEYPOINTS_PAYLOAD",
    "MOUSE",
    "pose_frame",
    "pose_object",
    "pose_set",
    "revision_file",
]

KEYPOINTS_PAYLOAD: Final = series_spec("keypoints").payload_filename
"""The file a keypoints revision's saved state is in."""

MOUSE: Final = PoseDefinition(
    id=3,
    name="mouse",
    schema=KeypointSchema(names=("nose", "tail"), skeleton=((0, 1),)),
)
"""A two-keypoint pose, the default every builder here uses."""


def revision_file(revision: int) -> str:
    """A revision's saved state, relative to its set's directory: what a claim names."""
    return f"rev{revision}/{KEYPOINTS_PAYLOAD}"


def pose_object(
    *points: tuple[float, float],
    pose_id: int = MOUSE.id,
    alias_id: int | None = None,
    bbox: Bbox | None = None,
    origin: ObjectOrigin | None = "human",
) -> PoseObject:
    """An object whose keypoints are all placed and visible, at *points*."""
    return PoseObject(
        pose_id=pose_id,
        keypoints=tuple(Keypoint(x=x, y=y, visibility=2) for x, y in points),
        alias_id=alias_id,
        origin=origin,
        bbox=bbox,
    )


def pose_frame(
    path: str | Path,
    *objects: PoseObject,
    sequence: str = "mouse003",
    frame_index: int = 0,
    usable: bool = True,
) -> PoseFrame:
    """A 64x48 frame holding *objects*, finished unless *usable* says otherwise."""
    return PoseFrame(
        image_path=Path(path),
        width=64,
        height=48,
        usable=usable,
        objects=objects,
        sequence=sequence,
        frame_index=frame_index,
    )


def pose_set(
    frames: Iterable[PoseFrame],
    *,
    poses: tuple[PoseDefinition, ...] = (MOUSE,),
    image_root: Path | None = None,
    bbox_policy: BboxPolicy | None = None,
) -> PoseAnnotationSet:
    """A saved state over *frames*, of *poses*."""
    return PoseAnnotationSet(
        poses=poses,
        frames=tuple(frames),
        bbox_policy=bbox_policy or BboxPolicy(),
        image_root=image_root,
    )
