"""YOLO pose and POLO point-detection model training wrappers.

Wraps the ultralytics Python API for training custom pose estimation and
point-detection (POLO) models.

Requires:
    Pose:  pip install ultralytics
    POLO:  pip install git+https://github.com/mooch443/POLO.git
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


def load_training_curves(run_dir: str | Path) -> "pd.DataFrame":
    """Load per-epoch training metrics from a YOLO training run.

    Parameters
    ----------
    run_dir : path
        Path to the training run directory (contains ``results.csv``).
        Also accepts a path to ``weights/best.pt`` — will resolve to
        the parent run directory automatically.

    Returns
    -------
    DataFrame
        One row per epoch with columns for train losses and val metrics.
    """
    import pandas as pd

    p = Path(run_dir)
    # Accept path to best.pt or weights/ dir
    if p.name == "best.pt" or p.name == "last.pt":
        p = p.parent.parent
    elif p.name == "weights":
        p = p.parent

    csv_path = p / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No results.csv in {p}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


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
    split: str = "val",
) -> Any:
    """Run validation on a trained pose model.

    Parameters
    ----------
    model_path : path
        Path to trained .pt model.
    data_yaml : path
        Path to data.yaml.
    device : str
        Device for validation.
    imgsz : int
        Image size.
    split : str
        Dataset split to evaluate on: "val", "test", or "train".

    Returns the ultralytics validation results with metrics.
    """
    YOLO = _require_ultralytics()
    yolo = YOLO(str(model_path))
    results = yolo.val(data=str(data_yaml), device=device, imgsz=imgsz, split=split)
    return results


# --------------------------------------------------------------------------- #
# POLO point-detection training
# --------------------------------------------------------------------------- #

def _require_polo():
    """Import YOLO from a POLO fork and verify the 'locate' task is available.

    Returns the YOLO class from the POLO-extended ultralytics package.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "A POLO-compatible ultralytics fork is required for point model "
            "training.  Install with:\n"
            "  pip install git+https://github.com/mooch443/POLO.git"
        )

    try:
        from ultralytics.nn.tasks import LocalizationModel  # noqa: F401
    except ImportError:
        raise ImportError(
            "Your ultralytics installation does not support the 'locate' task. "
            "Install the POLO fork:\n"
            "  pip install git+https://github.com/mooch443/POLO.git"
        )
    return YOLO


def train_point_model(
    data_yaml: str | Path,
    *,
    model: str = "polov8n.yaml",
    epochs: int = 300,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str | Path | None = None,
    name: str | None = None,
    patience: int = 50,
    loc: float = 5.0,
    loc_loss: str = "mse",
    dor: float = 0.8,
    augmentation: dict[str, Any] | None = None,
    backend: str = "polo",
    **extra_args: Any,
) -> Any:
    """Train a POLO point-detection model.

    Parameters
    ----------
    data_yaml : path
        Path to data.yaml with ``radii`` and class ``names``.
    model : str
        Model config or pretrained weights.  For POLO:
        ``"locate/polov8n.yaml"`` (nano), ``"locate/polov8s.yaml"`` (small),
        etc.  Or a path to a previously trained ``.pt`` file.
    epochs : int
        Number of training epochs.
    imgsz : int
        Training image size (pixels).
    batch : int
        Batch size (-1 for auto).
    device : str
        Device string (``"0"`` for first GPU, ``"cpu"`` for CPU).
    project : path, optional
        Project directory for saving runs.  Defaults to ``"./runs/locate"``.
    name : str, optional
        Run name.  Defaults to a timestamp.
    patience : int
        Early stopping patience (0 to disable).
    loc : float
        Localization loss weight (POLO-specific).
    loc_loss : str
        Localization loss type: ``"mse"``, ``"hausdorff"``, etc.
    dor : float
        Distance of Reference threshold for evaluation (POLO-specific).
    augmentation : dict, optional
        YOLO augmentation overrides (degrees, translate, scale, etc.).
    backend : str
        Point-detection backend.  Currently only ``"polo"`` is supported.
    **extra_args
        Additional keyword arguments passed to ``YOLO.train()``.

    Returns
    -------
    results
        Training results object from the POLO ultralytics fork.
    """
    if backend != "polo":
        raise ValueError(
            f"Unsupported point-detection backend: {backend!r}. "
            f"Currently only 'polo' is supported."
        )

    YOLO = _require_polo()

    if project is None:
        project = "./runs/locate"
    if name is None:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")

    yolo = YOLO(model)

    train_kwargs: dict[str, Any] = dict(
        data=str(data_yaml),
        task="locate",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        patience=patience,
        loc=loc,
        loc_loss=loc_loss,
        dor=dor,
    )
    if augmentation:
        train_kwargs.update(augmentation)
    train_kwargs.update(extra_args)

    results = yolo.train(**train_kwargs)
    return results


def validate_point_model(
    model_path: str | Path,
    data_yaml: str | Path,
    *,
    device: str = "0",
    imgsz: int = 640,
    split: str = "val",
    dor: float = 0.8,
    radii: dict[int, float] | None = None,
) -> Any:
    """Run validation on a trained POLO point-detection model.

    Parameters
    ----------
    model_path : path
        Path to trained ``.pt`` model.
    data_yaml : path
        Path to data.yaml with ``radii``.
    device : str
        Device for validation.
    imgsz : int
        Image size.
    split : str
        Dataset split: ``"val"``, ``"test"``, or ``"train"``.
    dor : float
        Distance of Reference threshold.
    radii : dict, optional
        Override radii from data.yaml.  ``{class_id: radius_px}``.

    Returns
    -------
    results
        Validation results with DoR-based metrics (mAP100, mAP100-10).
    """
    YOLO = _require_polo()
    yolo = YOLO(str(model_path))

    val_kwargs: dict[str, Any] = dict(
        data=str(data_yaml),
        device=device,
        imgsz=imgsz,
        split=split,
        dor=dor,
    )
    if radii is not None:
        val_kwargs["radii"] = radii

    return yolo.val(**val_kwargs)
