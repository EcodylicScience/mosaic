"""Custom model training pipeline for pose estimation, point detection, and localization.

Converters: transform annotation formats to YOLO pose / POLO point / localizer labels.
Training:   train YOLO pose, POLO point-detection, or localizer heatmap models.
Inference:  test trained models on video (non-production; production uses TRex).

Requires optional dependency:
    Pose:       pip install mosaic-behavior[pose]
    POLO:       pip install mosaic-behavior[polo]
    Localizer:  pip install mosaic-behavior[localizer]
"""
from . import converters
from .converters import lightning_pose, coco_keypoints, coco_points, coco_localizer, cvat_points, cvat_localizer
from .converters.base import KeypointSchema, PointDetectionSchema, LocalizerSchema
from .prep import prepare_yolo_dataset, make_data_yaml, make_polo_data_yaml, check_dataset
from .train import (
    train_pose_model,
    train_point_model,
    find_best_model,
    validate_model,
    validate_point_model,
    load_training_curves,
)
from .inference import (
    run_inference,
    run_point_inference,
    visualize_keypoints,
    inference_to_dataframe,
    locations_to_dataframe,
)
from .localizer_train import train_localizer, TrainingResult
from .localizer_inference import (
    detect_locations,
    run_localizer_inference,
    localizer_detections_to_dataframe,
)
from .localizer_weights import convert_keras_weights, load_localizer_weights
