"""Format converters for pose annotation data → YOLO pose labels."""
from .base import KeypointSchema, keypoints_to_bbox, normalize_coords, format_yolo_pose_line, write_yolo_label
