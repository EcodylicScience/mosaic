"""Writing a mosaic annotation set out as a Lightning Pose training dataset.

Lightning Pose reads DeepLabCut's layout: a ``CollectedData.csv`` with a
three-row header, and the images beside it under ``labeled-data/<video>/``. It is
plain text, so unlike the SLEAP export mosaic writes it directly -- there is no
library on the other side that has to be run.

**Single instance, and it refuses rather than truncates.** ``CollectedData`` has
one row per image and no instance axis, so a frame with two animals cannot be
expressed. Taking the first would silently discard labelling work, so a
multi-instance set is an error naming the frames that could not fit.

**An unplaced keypoint is an empty cell.** DeepLabCut's convention, and the one
Lightning Pose's loader expects; a zero there would be a keypoint at the origin.
"""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Final

import yaml

from mosaic.core.annotations.model import AnnotationSet

__all__ = ["DEFAULT_RESIZE", "RESIZE_MULTIPLE", "write_litpose_dataset"]

_COLLECTED = "CollectedData.csv"


def _video_of(annotations: AnnotationSet, index: int) -> str:
    """Which ``labeled-data`` subdirectory a frame belongs in."""
    video = annotations.frames[index].video
    return video or "images"


RESIZE_MULTIPLE: Final[int] = 128
"""What Lightning Pose requires ``image_resize_dims`` to be a multiple of.

Asserted in its own ``ModelConfig._validate_data``, because the backbone's
downsampling stages need the input to divide evenly.
"""

DEFAULT_RESIZE: Final[int] = 256
"""The network input size written when the caller names none.

Lightning Pose resizes every image to ``data.image_resize_dims`` before the
backbone sees it, so this is a real training decision rather than bookkeeping --
it sets what the model learns from and what inference reproduces. 256 is the
smallest square Lightning Pose's own examples use and a multiple of
:data:`RESIZE_MULTIPLE`; a project whose subjects are small in frame will want a
larger one, which is what the parameter is for.
"""


def write_litpose_dataset(
    annotations: AnnotationSet,
    out_dir: str | Path,
    *,
    scorer: str = "mosaic",
    train_prob: float = 0.8,
    copy_images: bool = True,
    resize_height: int = DEFAULT_RESIZE,
    resize_width: int = DEFAULT_RESIZE,
) -> Path:
    """Write *annotations* as a Lightning Pose project directory.

    Args:
        annotations: What to write. Every frame must hold at most one instance.
        out_dir: The project root. ``CollectedData.csv``, ``config.yaml`` and
            ``labeled-data/`` are written under it.
        scorer: The name in the header's first row, which DeepLabCut uses to
            identify who labelled the data.
        train_prob: Written into the config as the training fraction.
        copy_images: Copy the images in. Off leaves them where they are, which
            only works when *out_dir* already sees them.
        resize_height: Network input height. See :data:`DEFAULT_RESIZE`.
        resize_width: Network input width. See :data:`DEFAULT_RESIZE`.

    Returns:
        The project root.

    Raises:
        ValueError: The set is empty, or any frame holds more than one instance.
    """
    import shutil

    out_dir = Path(out_dir)
    if not annotations.frames:
        raise ValueError("cannot write a Lightning Pose dataset from no frames")

    crowded = [
        frame.image_path.name for frame in annotations.frames if len(frame.objects) > 1
    ]
    if crowded:
        raise ValueError(
            f"Lightning Pose is single-animal and CollectedData has no instance "
            f"axis, so {len(crowded)} frame(s) cannot be written: "
            f"{crowded[:5]}{' ...' if len(crowded) > 5 else ''}"
        )

    names = annotations.schema.names
    rows: list[list[str]] = []
    header_scorer = ["scorer"] + [scorer] * (len(names) * 2)
    header_parts = ["bodyparts"]
    header_coords = ["coords"]
    for name in names:
        header_parts.extend([name, name])
        header_coords.extend(["x", "y"])

    for index, frame in enumerate(annotations.frames):
        video = _video_of(annotations, index)
        relative = Path("labeled-data") / video / frame.image_path.name
        cells: list[str] = [relative.as_posix()]
        placed = frame.objects[0].keypoints if frame.objects else ()
        for position in range(len(names)):
            point = placed[position] if position < len(placed) else None
            if point is None or not point.is_placed:
                # DeepLabCut writes an unplaced point as an empty cell; a zero
                # would be a keypoint at the image origin.
                cells.extend(["", ""])
            else:
                cells.extend([f"{point.x:.6f}", f"{point.y:.6f}"])
        rows.append(cells)

        if copy_images:
            destination = out_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = annotations.resolve(frame)
            if source.exists():
                _ = shutil.copy2(source, destination)

    out_dir.mkdir(parents=True, exist_ok=True)
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header_scorer)
    writer.writerow(header_parts)
    writer.writerow(header_coords)
    writer.writerows(rows)
    _ = (out_dir / _COLLECTED).write_text(buffer.getvalue())

    config = {
        "data": {
            "image_orig_dims": {
                "height": annotations.frames[0].height,
                "width": annotations.frames[0].width,
            },
            # Required, not optional. Lightning Pose reads
            # ``data.image_resize_dims`` in struct mode -- its own
            # ``ModelConfig._validate_data`` indexes the key before checking
            # whether the values are set -- so a config that omits it fails
            # validation before training starts. The values are then handed
            # straight to ``cv2.resize`` on every frame, at training and at
            # inference alike, so they have to be real numbers rather than nulls.
            "image_resize_dims": {"height": resize_height, "width": resize_width},
            "data_dir": str(out_dir),
            "video_dir": str(out_dir / "videos"),
            "csv_file": _COLLECTED,
            "num_keypoints": len(names),
            "keypoint_names": list(names),
        },
        "training": {"train_prob": train_prob, "val_prob": round(1.0 - train_prob, 6)},
    }
    # Created, not merely named. Lightning Pose asserts ``video_dir`` is a
    # directory before training begins, whether or not the run uses unlabelled
    # video, so a project that names one it did not make fails on a bare
    # ``AssertionError`` with no message. Empty is a legitimate state -- it means
    # supervised-only training.
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    _ = (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return out_dir
