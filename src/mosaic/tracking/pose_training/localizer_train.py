"""Custom PyTorch training loop for the localizer heatmap model.

Trains on grayscale patches with binary cross-entropy loss.
Supports fine-tuning from existing weights (PyTorch ``.pt`` or Keras ``.h5``).

Requires: ``torch >= 2.0``
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for localizer training. "
            "Install with: pip install mosaic-behavior[localizer]"
        )


@dataclass
class TrainingResult:
    """Result of a localizer training run."""

    best_model_path: Path
    last_model_path: Path
    run_dir: Path
    best_epoch: int
    best_val_loss: float
    history: dict[str, list[float]] = field(default_factory=dict)


def _make_dataloader(dataset_dir: Path, subset: str, batch_size: int, shuffle: bool):
    """Create a DataLoader from saved ``.npy`` patch files."""
    torch = _require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    patches_path = dataset_dir / subset / "patches.npy"
    labels_path = dataset_dir / subset / "labels.npy"

    if not patches_path.exists():
        raise FileNotFoundError(f"No patches found at {patches_path}")

    patches = np.load(str(patches_path))  # (N, H, W) float32
    labels = np.load(str(labels_path))    # (N, num_classes) float32

    # Add channel dimension: (N, H, W) → (N, 1, H, W)
    patches = patches[:, np.newaxis, :, :]

    dataset = TensorDataset(
        torch.from_numpy(patches),
        torch.from_numpy(labels),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )


def _augment_batch(patches, rng: np.random.RandomState):
    """Apply random augmentations to a batch of (B, 1, H, W) tensors."""
    # Random horizontal flip (per-batch — fast)
    if rng.random() > 0.5:
        patches = patches.flip(-1)
    # Random vertical flip
    if rng.random() > 0.5:
        patches = patches.flip(-2)
    # Random 90° rotation
    k = rng.randint(0, 4)
    if k > 0:
        patches = patches.rot90(k, dims=(-2, -1))
    return patches


def train_localizer(
    dataset_dir: str | Path,
    *,
    num_classes: int = 4,
    initial_channels: int = 32,
    weights: str | Path | None = None,
    freeze_encoder: bool = False,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    min_lr: float = 1e-5,
    lr_patience: int = 15,
    lr_factor: float = 0.25,
    early_stopping_patience: int = 20,
    device: str = "0",
    project: str | Path | None = None,
    name: str | None = None,
    augment: bool = True,
    seed: int = 42,
) -> TrainingResult:
    """Train a localizer heatmap model.

    Parameters
    ----------
    dataset_dir : path
        Directory with ``train/`` and ``valid/`` subdirs containing
        ``patches.npy`` and ``labels.npy`` (from :func:`convert_coco_localizer`).
    num_classes : int
        Number of output heatmap channels.
    initial_channels : int
        Base channel width.
    weights : path, optional
        Path to pretrained weights (``.pt`` or ``.h5``) for fine-tuning.
    freeze_encoder : bool
        Freeze all layers except the 1×1 output head.  Useful for
        fine-tuning on small datasets.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Training batch size.
    lr : float
        Initial learning rate (Adam).
    min_lr : float
        Minimum learning rate for scheduler.
    lr_patience : int
        ``ReduceLROnPlateau`` patience (epochs).
    lr_factor : float
        ``ReduceLROnPlateau`` multiplicative factor.
    early_stopping_patience : int
        Stop if val loss doesn't improve for this many epochs.
    device : str
        ``"0"`` for first GPU, ``"cpu"`` for CPU.
    project : path, optional
        Project directory.  Defaults to ``"./runs/localizer"``.
    name : str, optional
        Run name.  Defaults to a timestamp.
    augment : bool
        Apply random flips/rotations during training.
    seed : int
        Random seed.

    Returns
    -------
    TrainingResult
        Paths to best/last checkpoints, run directory, and training history.
    """
    torch = _require_torch()
    import torch.nn as nn
    from .localizer_model import LocalizerEncoder, LocalizerTrainWrapper
    from .localizer_weights import load_localizer_weights

    dataset_dir = Path(dataset_dir)

    if project is None:
        project = "./runs/localizer"
    if name is None:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = Path(project) / name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if device == "cpu":
        dev = torch.device("cpu")
    else:
        dev = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    # Build model
    encoder = LocalizerEncoder(num_classes=num_classes, initial_channels=initial_channels)

    if weights is not None:
        load_localizer_weights(encoder, weights, strict=True)
        print(f"[localizer] Loaded weights from {weights}")

    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        for param in encoder.conv_reduce.parameters():
            param.requires_grad = True
        for param in encoder.conv_out.parameters():
            param.requires_grad = True
        print("[localizer] Encoder frozen — only output head is trainable")

    wrapper = LocalizerTrainWrapper(encoder)
    wrapper.to(dev)

    # Data loaders
    train_loader = _make_dataloader(dataset_dir, "train", batch_size, shuffle=True)
    valid_loader = _make_dataloader(dataset_dir, "valid", batch_size, shuffle=False)

    # Optimizer / scheduler / loss
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, wrapper.parameters()),
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr,
    )

    # Training state
    history: dict[str, list[float]] = {
        "train/loss": [],
        "val/loss": [],
        "lr": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    np_rng = np.random.RandomState(seed)

    print(f"[localizer] Training on {dev} for up to {epochs} epochs")
    print(
        f"  train: {len(train_loader.dataset)} patches, "
        f"valid: {len(valid_loader.dataset)} patches"
    )

    for epoch in range(epochs):
        t0 = time.time()

        # ---- train ----
        wrapper.train()
        train_loss_sum = 0.0
        n_train = 0

        for patches, labels in train_loader:
            patches, labels = patches.to(dev), labels.to(dev)
            if augment:
                patches = _augment_batch(patches, np_rng)

            preds = wrapper(patches)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(patches)
            n_train += len(patches)

        train_loss = train_loss_sum / n_train

        # ---- validate ----
        wrapper.eval()
        val_loss_sum = 0.0
        n_valid = 0

        with torch.no_grad():
            for patches, labels in valid_loader:
                patches, labels = patches.to(dev), labels.to(dev)
                preds = wrapper(patches)
                loss = criterion(preds, labels)
                val_loss_sum += loss.item() * len(patches)
                n_valid += len(patches)

        val_loss = val_loss_sum / n_valid

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history["train/loss"].append(train_loss)
        history["val/loss"].append(val_loss)
        history["lr"].append(current_lr)

        dt = time.time() - t0
        print(
            f"  epoch {epoch + 1}/{epochs} — "
            f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
            f"lr: {current_lr:.1e}, time: {dt:.1f}s"
        )

        # Save last checkpoint (encoder state_dict only)
        torch.save(encoder.state_dict(), str(weights_dir / "last.pt"))

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(encoder.state_dict(), str(weights_dir / "best.pt"))
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(
                f"[localizer] Early stopping at epoch {epoch + 1} "
                f"(patience={early_stopping_patience})"
            )
            break

    # Save results.csv (compatible with load_training_curves)
    csv_path = run_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        cols = list(history.keys())
        writer.writerow(cols)
        for i in range(len(history["train/loss"])):
            writer.writerow([history[c][i] for c in cols])

    # Save run config
    config = {
        "num_classes": num_classes,
        "initial_channels": initial_channels,
        "epochs_run": len(history["train/loss"]),
        "best_epoch": best_epoch + 1,
        "best_val_loss": best_val_loss,
        "lr": lr,
        "batch_size": batch_size,
        "freeze_encoder": freeze_encoder,
        "weights": str(weights) if weights else None,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(
        f"[localizer] Training complete. "
        f"Best val_loss: {best_val_loss:.4f} at epoch {best_epoch + 1}"
    )
    print(f"  Best model: {weights_dir / 'best.pt'}")

    return TrainingResult(
        best_model_path=weights_dir / "best.pt",
        last_model_path=weights_dir / "last.pt",
        run_dir=run_dir,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        history=history,
    )
