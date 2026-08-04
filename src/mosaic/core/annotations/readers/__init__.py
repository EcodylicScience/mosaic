"""Turning an external annotation format into the canonical representation.

One module per source. A reader's whole job is to answer, for this format, the
questions the representation asks: where the image is, how big it is, which
instances it holds, and for each keypoint whether it was placed and how
confidently. Nothing here writes anything.

``yolo_pose`` reads a tree mosaic itself emitted, which is not a contradiction:
the emitted dataset is sometimes the only copy left, and reading it back is what
lets its boxes be recomputed without relabelling.
"""

from mosaic.core.annotations.readers.coco import read_coco_keypoints
from mosaic.core.annotations.readers.cvat import read_cvat_points
from mosaic.core.annotations.readers.yolo_pose import read_yolo_pose

__all__ = ["read_coco_keypoints", "read_cvat_points", "read_yolo_pose"]
