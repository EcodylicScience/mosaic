"""A label row carries geometry only, whoever placed the object."""

from __future__ import annotations

from dataclasses import replace

from mosaic.core.annotations.model import AnnotationObject, Keypoint
from mosaic.tracking.pose_training.converters.emit import yolo_pose_line


def test_an_emitted_label_row_ignores_origin_and_source_reference() -> None:
    plain = AnnotationObject(
        keypoints=(
            Keypoint(x=10.0, y=10.0, visibility=2),
            Keypoint(x=20.0, y=30.0, visibility=2),
        )
    )
    tagged = replace(plain, origin="model", source_ref="run-7")

    assert yolo_pose_line(tagged, 64, 48) == yolo_pose_line(plain, 64, 48)
