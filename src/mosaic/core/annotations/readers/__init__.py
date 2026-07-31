"""Turning an external annotation format into the canonical representation.

One module per source. A reader's whole job is to answer, for this format, the
questions the representation asks: where the image is, how big it is, which
instances it holds, and for each keypoint whether it was placed and how
confidently. Nothing here writes anything.
"""

from mosaic.core.annotations.readers.coco import read_coco_keypoints

__all__ = ["read_coco_keypoints"]
