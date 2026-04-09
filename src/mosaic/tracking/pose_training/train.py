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
    resume: bool | str | Path = False,
    augmentation: str | dict[str, Any] | None = None,
    callback: Any = None,
    **extra_args: Any,
) -> Any:
    """Train a YOLO pose estimation model.

    Parameters
    ----------
    data_yaml : path
        Path to data.yaml (must include kpt_shape for pose).
    model : str
        Base model to start from (e.g. "yolo11n-pose.pt", "yolo11s-pose.pt").
        Ignored when *resume* is used.
    epochs : int
        Number of training epochs.  When resuming, this is the new total
        epoch target (training continues from the last checkpoint).
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
    resume : bool, str, or Path
        Resume training from a checkpoint.

        - ``False`` (default): train from scratch.
        - ``True``: find the most recent ``last.pt`` under
          *project*/*name* (or the latest run in *project*).
        - A path (str or Path): explicit path to a ``last.pt`` checkpoint.

        When resuming, ultralytics restores all training state (optimizer,
        scheduler, augmentation) from the checkpoint.  Only *epochs* and
        *device* are meaningful overrides.
    augmentation : str, dict, or None
        Augmentation configuration.  Can be:

        - A preset name: ``"none"``, ``"light"``, ``"medium"``, ``"heavy"``.
        - A dict with a ``"preset"`` key plus overrides, e.g.
          ``{"preset": "light", "flipud": 0.5}``.
        - A raw dict of ultralytics augmentation kwargs (degrees, translate,
          scale, flipud, fliplr, mosaic, mixup, copy_paste, hsv_h, hsv_s,
          hsv_v, close_mosaic, etc.).
        - ``None`` to use ultralytics defaults.
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
    if name is None and not resume:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve resume checkpoint
    if resume and resume is not True:
        checkpoint = Path(resume)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        yolo = YOLO(str(checkpoint))
    elif resume is True:
        checkpoint = find_last_checkpoint(project, name)
        yolo = YOLO(str(checkpoint))
    else:
        yolo = YOLO(model)

    train_kwargs: dict[str, Any] = dict(
        data=str(data_yaml),
        task="pose",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        patience=patience,
    )
    if name is not None:
        train_kwargs["name"] = name
    if resume:
        train_kwargs["resume"] = True

    if not resume:
        from .augmentation import resolve_augmentation
        aug_dict = resolve_augmentation(augmentation)
        if aug_dict:
            train_kwargs.update(aug_dict)
    train_kwargs.update(extra_args)

    if callback is not None and hasattr(callback, "on_epoch_end"):
        _register_ultralytics_callback(yolo, callback, epochs)

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


def find_last_checkpoint(
    project_dir: str | Path, name: str | None = None
) -> Path:
    """Find the last.pt checkpoint for resuming training.

    Parameters
    ----------
    project_dir : path
        Project directory containing training runs.
    name : str, optional
        Specific run name.  If given, looks for
        ``project_dir/name/weights/last.pt``.  Otherwise searches all
        subdirectories and returns the most recently modified checkpoint.

    Returns
    -------
    Path
        Path to the ``last.pt`` checkpoint.

    Raises
    ------
    FileNotFoundError
        If no checkpoint is found.
    """
    project_path = Path(project_dir)
    if name is not None:
        checkpoint = project_path / name / "weights" / "last.pt"
        if checkpoint.exists():
            return checkpoint
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint}. "
            f"Has a training run with name={name!r} completed at least one epoch?"
        )
    candidates = sorted(
        project_path.glob("*/weights/last.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No last.pt checkpoint found under {project_path}. "
        f"Has a training run completed at least one epoch?"
    )


def validate_model(
    model_path: str | Path,
    data_yaml: str | Path,
    *,
    device: str = "0",
    imgsz: int = 640,
    split: str = "val",
    project: str | Path | None = None,
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
    project : path, optional
        Directory where validation results are saved.  Defaults to the
        training run directory (two levels up from the model weights).

    Returns the ultralytics validation results with metrics.
    """
    YOLO = _require_ultralytics()
    yolo = YOLO(str(model_path))

    if project is None:
        project = str(Path(model_path).resolve().parent.parent)

    results = yolo.val(
        data=str(data_yaml), device=device, imgsz=imgsz, split=split,
        project=str(project),
    )
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
    resume: bool | str | Path = False,
    augmentation: str | dict[str, Any] | None = None,
    backend: str = "polo",
    callback: Any = None,
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
        Ignored when *resume* is used.
    epochs : int
        Number of training epochs.  When resuming, this is the new total
        epoch target.
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
    resume : bool, str, or Path
        Resume training from a checkpoint.

        - ``False`` (default): train from scratch.
        - ``True``: find the most recent ``last.pt`` under
          *project*/*name* (or the latest run in *project*).
        - A path (str or Path): explicit path to a ``last.pt`` checkpoint.

        When resuming, ultralytics restores all training state from the
        checkpoint.  Only *epochs* and *device* are meaningful overrides.
    augmentation : str, dict, or None
        Augmentation configuration.  Can be:

        - A preset name: ``"none"``, ``"light"``, ``"medium"``, ``"heavy"``.
        - A dict with a ``"preset"`` key plus overrides, e.g.
          ``{"preset": "light", "flipud": 0.5}``.
        - A raw dict of ultralytics augmentation kwargs.
        - ``None`` to use ultralytics defaults.
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
    if name is None and not resume:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve resume checkpoint
    if resume and resume is not True:
        checkpoint = Path(resume)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        yolo = YOLO(str(checkpoint))
    elif resume is True:
        checkpoint = find_last_checkpoint(project, name)
        yolo = YOLO(str(checkpoint))
    else:
        yolo = YOLO(model)

    train_kwargs: dict[str, Any] = dict(
        data=str(data_yaml),
        task="locate",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        patience=patience,
        loc=loc,
        loc_loss=loc_loss,
        dor=dor,
    )
    if name is not None:
        train_kwargs["name"] = name
    if resume:
        train_kwargs["resume"] = True

    if not resume:
        from .augmentation import resolve_augmentation
        aug_dict = resolve_augmentation(augmentation)
        if aug_dict:
            train_kwargs.update(aug_dict)
    train_kwargs.update(extra_args)

    if callback is not None and hasattr(callback, "on_epoch_end"):
        _register_ultralytics_callback(yolo, callback, epochs)

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
    project: str | Path | None = None,
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
    project : path, optional
        Directory where validation results are saved.  Defaults to the
        training run directory (two levels up from the model weights).

    Returns
    -------
    results
        Validation results with DoR-based metrics (mAP100, mAP100-10).
    """
    YOLO = _require_polo()
    yolo = YOLO(str(model_path))

    if project is None:
        project = str(Path(model_path).resolve().parent.parent)

    val_kwargs: dict[str, Any] = dict(
        data=str(data_yaml),
        device=device,
        imgsz=imgsz,
        split=split,
        dor=dor,
        project=str(project),
    )
    if radii is not None:
        val_kwargs["radii"] = radii

    return yolo.val(**val_kwargs)


# ---------------------------------------------------------------------------
# Progress callback bridge for Ultralytics trainers
# ---------------------------------------------------------------------------


def _register_ultralytics_callback(
    yolo: Any, callback: Any, total_epochs: int
) -> None:
    """Register an ``on_train_epoch_end`` hook on a YOLO model instance.

    Ultralytics trainers expose ``add_callback(event, func)`` where *func*
    receives the trainer instance.  We bridge that to the mosaic
    :class:`~mosaic.core.pipeline.progress.TrainingProgressCallback` protocol.
    """

    def _on_epoch_end(trainer: Any) -> None:
        epoch = getattr(trainer, "epoch", 0)
        metrics: dict[str, float] = {}
        # Trainer exposes loss as a tensor and metrics as a dict
        loss = getattr(trainer, "loss", None)
        if loss is not None:
            try:
                metrics["loss"] = float(loss.item() if hasattr(loss, "item") else loss)
            except Exception:
                pass
        trainer_metrics = getattr(trainer, "metrics", {})
        if isinstance(trainer_metrics, dict):
            for k, v in trainer_metrics.items():
                try:
                    metrics[k] = float(v)
                except (TypeError, ValueError):
                    pass
        callback.on_epoch_end(epoch, total_epochs, metrics)

    yolo.add_callback("on_train_epoch_end", _on_epoch_end)
