"""Augmentation presets and utilities for pose, point, and localizer training.

Provides curated augmentation configurations for the three model types:

- **YOLO pose / POLO point**: Named presets that resolve to ultralytics
  parameter dicts (``"none"``, ``"light"``, ``"medium"``, ``"heavy"``).
- **Localizer**: Typed configuration with geometric (flip, rot90) and
  photometric (brightness, contrast, noise) transforms using pure torch ops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# YOLO / POLO augmentation presets
# --------------------------------------------------------------------------- #

YOLO_AUGMENTATION_PRESETS: dict[str, dict[str, Any]] = {
    "none": {
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
    },
    "light": {
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.2,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.5,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.01,
        "hsv_s": 0.3,
        "hsv_v": 0.2,
    },
    "medium": {
        "degrees": 15.0,
        "translate": 0.15,
        "scale": 0.4,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "close_mosaic": 10,
    },
    "heavy": {
        "degrees": 30.0,
        "translate": 0.2,
        "scale": 0.5,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.2,
        "copy_paste": 0.1,
        "hsv_h": 0.02,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "close_mosaic": 10,
    },
}


def resolve_augmentation(
    augmentation: str | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Resolve an augmentation specification to an ultralytics-compatible dict.

    Parameters
    ----------
    augmentation : str, dict, or None
        - ``str``: preset name (``"none"``, ``"light"``, ``"medium"``, ``"heavy"``).
        - ``dict``: if it contains a ``"preset"`` key, start from that preset
          and override with the remaining keys; otherwise pass through as-is.
        - ``None``: return ``None`` (use ultralytics defaults).

    Returns
    -------
    dict or None
        Augmentation kwargs to unpack into ``YOLO.train()``, or ``None``.

    Examples
    --------
    >>> resolve_augmentation("light")
    {'degrees': 5.0, 'translate': 0.1, ...}

    >>> resolve_augmentation({"preset": "light", "flipud": 0.5})
    {'degrees': 5.0, ..., 'flipud': 0.5}

    >>> resolve_augmentation({"degrees": 10, "scale": 0.3})
    {'degrees': 10, 'scale': 0.3}
    """
    if augmentation is None:
        return None

    if isinstance(augmentation, str):
        if augmentation not in YOLO_AUGMENTATION_PRESETS:
            raise ValueError(
                f"Unknown augmentation preset: {augmentation!r}. "
                f"Choose from: {list(YOLO_AUGMENTATION_PRESETS)}"
            )
        return dict(YOLO_AUGMENTATION_PRESETS[augmentation])

    if isinstance(augmentation, dict):
        if "preset" in augmentation:
            preset_name = augmentation["preset"]
            if preset_name not in YOLO_AUGMENTATION_PRESETS:
                raise ValueError(
                    f"Unknown augmentation preset: {preset_name!r}. "
                    f"Choose from: {list(YOLO_AUGMENTATION_PRESETS)}"
                )
            merged = dict(YOLO_AUGMENTATION_PRESETS[preset_name])
            merged.update({k: v for k, v in augmentation.items() if k != "preset"})
            return merged
        return dict(augmentation)

    raise TypeError(
        f"augmentation must be str, dict, or None, got {type(augmentation).__name__}"
    )


# --------------------------------------------------------------------------- #
# Localizer augmentation
# --------------------------------------------------------------------------- #

@dataclass
class LocalizerAugmentConfig:
    """Configuration for localizer patch augmentation.

    All transforms preserve center-pixel label semantics.

    Geometric transforms (applied per-batch for speed):
        flip_h, flip_v, rotate_90

    Photometric transforms (applied per-sample for variety):
        brightness, contrast, gaussian_noise_std
    """
    # Geometric
    flip_h: bool = True
    flip_v: bool = True
    rotate_90: bool = True

    # Photometric
    brightness: float = 0.0
    """Max additive brightness shift (uniform in [-brightness, +brightness])."""

    contrast: tuple[float, float] = (1.0, 1.0)
    """Multiplicative contrast range (uniform in [lo, hi])."""

    gaussian_noise_std: float = 0.0
    """Standard deviation of additive Gaussian noise."""


LOCALIZER_AUGMENT_PRESETS: dict[str, LocalizerAugmentConfig] = {
    "none": LocalizerAugmentConfig(
        flip_h=False, flip_v=False, rotate_90=False,
    ),
    "light": LocalizerAugmentConfig(
        flip_h=True, flip_v=True, rotate_90=True,
    ),
    "medium": LocalizerAugmentConfig(
        flip_h=True, flip_v=True, rotate_90=True,
        brightness=0.05, contrast=(0.9, 1.1),
    ),
    "heavy": LocalizerAugmentConfig(
        flip_h=True, flip_v=True, rotate_90=True,
        brightness=0.1, contrast=(0.8, 1.2),
        gaussian_noise_std=0.02,
    ),
}


def resolve_localizer_augment(
    augment: bool | str | LocalizerAugmentConfig,
) -> LocalizerAugmentConfig | None:
    """Resolve a localizer augment parameter to a config object.

    Parameters
    ----------
    augment : bool, str, or LocalizerAugmentConfig
        - ``True``: maps to ``"light"`` preset (backwards compatible —
          closest to the original flip+rotate behaviour).
        - ``False``: no augmentation.
        - ``str``: preset name (``"none"``, ``"light"``, ``"medium"``, ``"heavy"``).
        - ``LocalizerAugmentConfig``: used directly.

    Returns
    -------
    LocalizerAugmentConfig or None
        ``None`` when augmentation is disabled.
    """
    if isinstance(augment, LocalizerAugmentConfig):
        return augment

    if augment is True:
        return LOCALIZER_AUGMENT_PRESETS["light"]

    if augment is False or augment == "none":
        return None

    if isinstance(augment, str):
        if augment not in LOCALIZER_AUGMENT_PRESETS:
            raise ValueError(
                f"Unknown localizer augment preset: {augment!r}. "
                f"Choose from: {list(LOCALIZER_AUGMENT_PRESETS)}"
            )
        return LOCALIZER_AUGMENT_PRESETS[augment]

    raise TypeError(
        f"augment must be bool, str, or LocalizerAugmentConfig, "
        f"got {type(augment).__name__}"
    )


def augment_localizer_batch(
    patches: "torch.Tensor",
    config: LocalizerAugmentConfig,
    rng: np.random.RandomState,
) -> "torch.Tensor":
    """Apply augmentations to a batch of localizer patches.

    All transforms preserve center-pixel semantics (no spatial shift).

    Parameters
    ----------
    patches : Tensor
        Shape ``(B, 1, H, W)``, float32 in [0, 1].
    config : LocalizerAugmentConfig
        Which transforms to apply and their strength.
    rng : RandomState
        Numpy random state for reproducibility.

    Returns
    -------
    Tensor
        Augmented patches, same shape, clamped to [0, 1].
    """
    import torch

    # --- Geometric (per-batch — fast, same as original behaviour) ---
    if config.flip_h and rng.random() > 0.5:
        patches = patches.flip(-1)
    if config.flip_v and rng.random() > 0.5:
        patches = patches.flip(-2)
    if config.rotate_90:
        k = rng.randint(0, 4)
        if k > 0:
            patches = patches.rot90(k, dims=(-2, -1))

    # --- Photometric (per-sample for variety) ---
    B = patches.shape[0]
    dev = patches.device

    if config.brightness > 0:
        shift = torch.from_numpy(
            rng.uniform(-config.brightness, config.brightness, size=(B, 1, 1, 1)).astype(np.float32)
        ).to(dev)
        patches = patches + shift

    lo, hi = config.contrast
    if lo != 1.0 or hi != 1.0:
        factor = torch.from_numpy(
            rng.uniform(lo, hi, size=(B, 1, 1, 1)).astype(np.float32)
        ).to(dev)
        mean = patches.mean(dim=(-2, -1), keepdim=True)
        patches = (patches - mean) * factor + mean

    if config.gaussian_noise_std > 0:
        noise = torch.from_numpy(
            rng.normal(0, config.gaussian_noise_std, size=patches.shape).astype(np.float32)
        ).to(dev)
        patches = patches + noise

    # Clamp if any photometric transform was applied
    if config.brightness > 0 or lo != 1.0 or hi != 1.0 or config.gaussian_noise_std > 0:
        patches = patches.clamp(0.0, 1.0)

    return patches
