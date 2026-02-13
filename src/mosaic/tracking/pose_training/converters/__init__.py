"""Format converters for annotation data → YOLO pose / POLO point / localizer labels."""
from .base import (
    KeypointSchema,
    LocalizerSchema,
    PointDetectionSchema,
    keypoints_to_bbox,
    normalize_coords,
    format_yolo_pose_line,
    format_polo_label_line,
    write_yolo_label,
)
