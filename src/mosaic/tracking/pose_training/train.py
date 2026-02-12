"""YOLO pose model training wrapper.

Wraps the ultralytics Python API for training custom pose estimation models.
Requires: pip install ultralytics
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any


def _require_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for pose model training. "
            "Install it with: pip install ultralytics"
        )


def train_pose_model(
    data_yaml: str | Path,
    *,
    model: str = "yolo11n-pose.pt",
    epochs: int = 300,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str | Path | None = None,
    name: str | None = None,
    patience: int = 50,
    augmentation: dict[str, Any] | None = None,
    **extra_args: Any,
) -> Any:
    """Train a YOLO pose estimation model.

    Parameters
    ----------
    data_yaml : path
        Path to data.yaml (must include kpt_shape for pose).
    model : str
        Base model to start from (e.g. "yolo11n-pose.pt", "yolo11s-pose.pt").
    epochs : int
        Number of training epochs.
    imgsz : int
        Training image size (pixels).
    batch : int
        Batch size (-1 for auto).
    device : str
        Device string ("0" for first GPU, "cpu" for CPU, "0,1" for multi-GPU).
    project : path, optional
        Project directory for saving runs.  Defaults to "./runs/pose".
    name : str, optional
        Run name.  Defaults to a timestamp.
    patience : int
        Early stopping patience (0 to disable).
    augmentation : dict, optional
        YOLO augmentation overrides. Common keys:
        degrees, translate, scale, flipud, fliplr, mosaic, mixup,
        copy_paste, hsv_h, hsv_s, hsv_v, close_mosaic, amp.
    **extra_args
        Additional keyword arguments passed to YOLO.train().

    Returns
    -------
    results
        Ultralytics training results object.
    """
    YOLO = _require_ultralytics()

    if project is None:
        project = "./runs/pose"
    if name is None:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")

    yolo = YOLO(model)

    train_kwargs: dict[str, Any] = dict(
        data=str(data_yaml),
        task="pose",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        patience=patience,
    )
    if augmentation:
        train_kwargs.update(augmentation)
    train_kwargs.update(extra_args)

    results = yolo.train(**train_kwargs)
    return results


def find_best_model(project_dir: str | Path) -> Path | None:
    """Find the best.pt model from the most recent training run.

    Searches *project_dir* for subdirectories containing weights/best.pt,
    returns the most recently modified one.
    """
    project_path = Path(project_dir)
    candidates = sorted(
        project_path.glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def validate_model(
    model_path: str | Path,
    data_yaml: str | Path,
    *,
    device: str = "0",
    imgsz: int = 640,
) -> Any:
    """Run validation on a trained pose model.

    Returns the ultralytics validation results with metrics.
    """
    YOLO = _require_ultralytics()
    yolo = YOLO(str(model_path))
    results = yolo.val(data=str(data_yaml), device=device, imgsz=imgsz)
    return results
