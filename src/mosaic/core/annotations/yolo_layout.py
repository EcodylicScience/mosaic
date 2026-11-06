"""The file layout rules a YOLO dataset follows, as Ultralytics applies them.

A leaf module: it imports only the standard library, so every reader, emitter and
fingerprint can share one statement of the rules. The rules mirror Ultralytics
release 8.4.121 and the POLO fork's 8.4.84, neither importable in the base
environment.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["YOLO_IMAGE_SUFFIXES", "label_path_for_image"]

YOLO_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {
        ".avif",
        ".bmp",
        ".dng",
        ".heic",
        ".heif",
        ".jp2",
        ".jpeg",
        ".jpg",
        ".mpo",
        ".pfm",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
)
"""The dotted, lowercase image suffixes either trainer accepts (``IMG_FORMATS``).

The union of two sets: Ultralytics 8.4.121, which ``train-pose`` runs, accepts
``bmp dng jpeg jpg mpo png tif tiff webp pfm heic``, and the POLO fork at
Ultralytics 8.4.84, which ``train-points`` runs, accepts
``avif bmp dng heic heif jp2 jpeg jpg mpo png tif tiff webp``. A union is safe
because an extra suffix only adds files to a training fingerprint and caches to
remove.

Ultralytics compares a file's suffix lowercased, so test
``path.suffix.lower() in YOLO_IMAGE_SUFFIXES``.
"""


def label_path_for_image(image: Path) -> Path:
    """The label file Ultralytics reads for an image.

    Applies ``img2label_paths`` of Ultralytics 8.4.121 to the absolute path: the
    last ``images`` directory is replaced by ``labels`` and the suffix by ``.txt``.
    A path with no ``images`` directory keeps its directories and swaps only the
    suffix. The stem is kept as written.

    Links are not followed. Ultralytics resolves the split directory and lists the
    files under it without resolving them, so the label of a linked image sits
    beside the link's ``images`` directory, not beside the link's target.

    Args:
        image: Path of an image file.

    Returns:
        The path of the corresponding label file.
    """
    parts = list(Path(os.path.abspath(image)).parts)
    for index in range(len(parts) - 2, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    return Path(*parts).with_suffix(".txt")
