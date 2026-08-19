"""Custom model training pipeline for pose estimation, point detection, and localization.

Converters: transform annotation formats to YOLO pose / POLO point / localizer labels.
Training:   train YOLO pose, POLO point-detection, or localizer heatmap models.
Inference:  the heatmap localizer runs here; YOLO pose and POLO point inference
            run in environments of their own, driven from `ultralytics_infer`.

Requires optional dependency:
    Pose:       pip install mosaic-behavior[pose]      (training only)
    POLO:       pip install mosaic-behavior[polo]      (training only)
    Localizer:  pip install mosaic-behavior[deep-learning]
"""

from mosaic.core.annotations import KeypointSchema
from mosaic.core.annotations.bbox import (
    BboxPolicy,
    keypoints_to_bbox,
    keypoints_to_bbox_isotropic,
    keypoints_to_bbox_oriented,
)

from . import converters
from .converters import (
    lightning_pose,
    coco_keypoints,
    coco_points,
    coco_localizer,
    cvat_points,
    cvat_localizer,
)
from .converters.base import (
    PointDetectionSchema,
    LocalizerSchema,
)
from .repad import repad_yolo_pose
from .prep import (
    prepare_yolo_dataset,
    make_data_yaml,
    make_polo_data_yaml,
    check_dataset,
    tracks_to_yolo_pose,
)
from .train import (
    train_pose_model,
    train_point_model,
    find_best_model,
    find_last_checkpoint,
    validate_model,
    validate_point_model,
    load_training_curves,
)
from .inference import (
    visualize_keypoints,
    visualize_detections,
    visualize_inference,
)
from .localizer_train import train_localizer, TrainingResult
from .localizer_inference import (
    detect_locations,
    run_localizer_inference,
    localizer_detections_to_dataframe,
)
from .localizer_weights import convert_keras_weights, load_localizer_weights
from .augmentation import (
    YOLO_AUGMENTATION_PRESETS,
    LOCALIZER_AUGMENT_PRESETS,
    LocalizerAugmentConfig,
    resolve_augmentation,
    resolve_localizer_augment,
    augment_localizer_batch,
)

__all__ = [
    "LOCALIZER_AUGMENT_PRESETS",
    "YOLO_AUGMENTATION_PRESETS",
    "KeypointSchema",
    "LocalizerAugmentConfig",
    "LocalizerSchema",
    "PointDetectionSchema",
    "TrainingResult",
    "augment_localizer_batch",
    "check_dataset",
    "coco_keypoints",
    "coco_localizer",
    "coco_points",
    "converters",
    "convert_keras_weights",
    "cvat_localizer",
    "cvat_points",
    "detect_locations",
    "find_best_model",
    "find_last_checkpoint",
    "BboxPolicy",
    "keypoints_to_bbox",
    "keypoints_to_bbox_isotropic",
    "keypoints_to_bbox_oriented",
    "lightning_pose",
    "load_localizer_weights",
    "load_training_curves",
    "localizer_detections_to_dataframe",
    "make_data_yaml",
    "make_polo_data_yaml",
    "prepare_yolo_dataset",
    "resolve_augmentation",
    "resolve_localizer_augment",
    "repad_yolo_pose",
    "run_localizer_inference",
    "tracks_to_yolo_pose",
    "train_localizer",
    "train_point_model",
    "train_pose_model",
    "validate_model",
    "validate_point_model",
    "visualize_detections",
    "visualize_inference",
    "visualize_keypoints",
]
