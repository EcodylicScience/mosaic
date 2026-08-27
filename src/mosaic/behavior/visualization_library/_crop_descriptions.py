"""Field prose shared by the two egocentric-crop features.

:class:`~mosaic.behavior.visualization_library.egocentric_crop.EgocentricCrop`
and
:class:`~mosaic.behavior.visualization_library.interaction_crop.InteractionCropPipeline`
extract the same rotated, optionally masked crop around a focal individual.
Every field they hold in common means the same thing in both, declared once
here instead of twice.
"""

from __future__ import annotations

CENTER_MODE_DESCRIPTION = (
    "How to compute the crop center. Known values are default, xy and "
    "pose0. default averages the pose points present on each row, and uses "
    "the body center where the table has no pose columns. xy uses the body "
    "center alone, even where pose points exist. pose0 uses the first pose "
    "point, and an integer names a specific pose point index. Reading the "
    "body center needs pixel coordinates. A run refuses an entry recorded "
    "on the centimeter-era trex_v1 schema under xy, and under any mode "
    "where the table has no pose columns. An unrecorded schema is read as "
    "trex_v1."
)

POSE_DESCRIPTION = "Pose keypoint column naming and selection."

CROP_SIZE_DESCRIPTION = "Width and height of the output crop."

ROTATE_TO_HEADING_DESCRIPTION = (
    "Rotate the crop so the animal's heading aligns with the +x axis."
)

HEADING_POINTS_DESCRIPTION = (
    "Pose point indices used for heading, as (neck index, tail index). "
    "The heading direction runs from the tail to the neck, the direction "
    "the animal faces."
)

MARGIN_FACTOR_DESCRIPTION = (
    "Extra margin for the pre-rotation crop, as a multiple of the final crop size."
)

CENTER_OFFSET_PX_DESCRIPTION = (
    "Offset from the computed center along the heading direction, "
    "positive toward the head. Useful for centering on a specific body "
    "part instead of the detected center."
)

BODY_MASK_DESCRIPTION = "Apply an elliptical mask isolating the focal individual."

BODY_MASK_LENGTH_PX_DESCRIPTION = (
    "Full length of the body mask ellipse along its major axis."
)

BODY_MASK_WIDTH_PX_DESCRIPTION = (
    "Full length of the body mask ellipse along its minor axis."
)

USE_CLAHE_DESCRIPTION = (
    "Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the crop."
)

CLAHE_CLIP_LIMIT_DESCRIPTION = "Clip limit for CLAHE contrast enhancement."

CLAHE_TILE_GRID_SIZE_DESCRIPTION = (
    "CLAHE tile grid size, applied to both width and height."
)

GRAYSCALE_DESCRIPTION = "Convert the crop to single-channel grayscale."

OUTPUT_FPS_DESCRIPTION = (
    "Output video frame rate. Unset, uses the source video frame rate."
)

INTERPOLATION_FLAG_DESCRIPTION = (
    "OpenCV interpolation flag used when rotating the crop. Known values "
    "are cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, "
    "cv2.INTER_AREA and cv2.INTER_LANCZOS4."
)

BACKGROUND_COLOR_DESCRIPTION = (
    "Fill value for pixels with no source content, for example 0 for black "
    "or 255 for white. It pads where the crop window extends past the "
    "frame, fills the border rotation introduces, and fills the whole crop "
    "where the center is not finite."
)

ANGLE_COL_DESCRIPTION = (
    "Name of a track column recording a pre-computed heading angle, in "
    "degrees or radians (auto-detected). Unset, heading is derived from "
    "heading_points."
)
