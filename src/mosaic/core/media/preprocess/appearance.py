"""Convert images to gray and equalize them with CLAHE.

The ``grayscale`` and ``clahe`` steps, ``egocentric-crop`` and
``interaction-crop-pipeline`` all convert and equalize through these functions.

Each result keeps its input's dtype. A step's frames are 8-bit, and a crop feature
reading a raw imgstore of a 16-bit camera passes 16-bit crops, which stay 16-bit.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

__all__ = ["apply_clahe", "make_clahe", "to_gray"]


def make_clahe(clip_limit: float, tile_grid_size: int) -> cv2.CLAHE:
    """Build a CLAHE over a square ``tile_grid_size`` grid, clipped at *clip_limit*.

    Building one initializes a lookup table per tile. Build it once and apply it to
    every frame.
    """
    return cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)
    )


def apply_clahe[D: (np.uint8, np.uint16)](
    image: npt.NDArray[D], clahe: cv2.CLAHE
) -> npt.NDArray[D]:
    """Apply *clahe* to a gray *image* directly, and to L of LAB for a BGR one.

    A color image is equalized on lightness alone. Its hues are kept.

    Raises:
        cv2.error: If *image* is a 16-bit color image. OpenCV converts only an
            8-bit or float image to LAB.
    """
    if image.ndim == 2:
        return np.asarray(clahe.apply(image), dtype=image.dtype)
    lab = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2LAB), dtype=image.dtype)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return np.asarray(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), dtype=image.dtype)


def to_gray[D: (np.uint8, np.uint16)](image: npt.NDArray[D]) -> npt.NDArray[D]:
    """Return *image* as a gray ``H x W`` image. A gray image is returned as it is."""
    if image.ndim == 2:
        return image
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=image.dtype)
