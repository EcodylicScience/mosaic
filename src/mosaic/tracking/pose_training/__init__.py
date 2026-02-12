"""Custom pose model training pipeline.

Converters: transform annotation formats (Lightning Pose, etc.) to YOLO pose labels.
Training:   train YOLO pose models via ultralytics.
Inference:  test trained models on video (non-production; production uses TRex).

Requires optional dependency: pip install behavior[pose]
"""
from . import converters
from .prep import prepare_yolo_dataset, make_data_yaml, check_dataset
from .train import train_pose_model, find_best_model, validate_model
from .inference import run_inference, inference_to_dataframe
