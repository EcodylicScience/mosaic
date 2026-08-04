"""Writing a canonical annotation set out as a YOLO-style training dataset.

The layout every trainer here expects::

    <root>/<split>/images/<stem>.<ext>
    <root>/<split>/labels/<stem>.txt

Four converters wrote that loop separately -- find the usable images, split
them, write a label, place the image beside it, drop the test split when nothing
landed in it -- differing only in how they built the line. This is the loop; the
line is a callback.

It lives in ``tracking`` rather than beside the representation on purpose. An
annotation set is what was labelled; this directory tree is what one family of
trainers reads, which is a tracking-domain concern that a reader must not need
to know about.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from mosaic.core.annotations.bbox import BboxPolicy, keypoints_to_bbox
from mosaic.core.annotations.model import (
    AnnotationFrame,
    AnnotationObject,
    AnnotationSet,
)

from .base import format_yolo_pose_line, normalize_coords, write_yolo_label

__all__ = [
    "SPLITS",
    "usable_frames",
    "write_split_tree",
    "yolo_pose_line",
]

SPLITS: tuple[str, str, str] = ("train", "valid", "test")


def usable_frames(
    annotations: AnnotationSet,
) -> list[tuple[AnnotationFrame, Path]]:
    """Annotated frames whose image is actually on disk, with that path.

    Written five times across the converter package, each over a slightly
    different record shape, which is how three of them ended up with subtly
    different definitions of "usable".
    """
    found: list[tuple[AnnotationFrame, Path]] = []
    for frame in annotations.frames:
        if not frame.is_annotated:
            continue
        path = annotations.resolve(frame)
        if path.exists():
            found.append((frame, path))
    return found


def yolo_pose_line(
    obj: AnnotationObject,
    width: int,
    height: int,
    *,
    class_id: int = 0,
    policy: BboxPolicy | None = None,
) -> str | None:
    """One instance as a YOLO-pose label row, or ``None`` when untrainable.

    ``None`` covers the two cases a trainer cannot use: no keypoint was placed,
    and the derived box has no area. Both are counted by the caller rather than
    raised, because one bad instance in a large set is a statistic and not a
    failure.

    The source's own box wins when it has one. That is what ``bbox_source`` used
    to select, and it matters because a hand-drawn COCO box encloses the animal
    while a box derived from midline keypoints encloses only the midline.
    """
    policy = policy or BboxPolicy()

    if obj.bbox is not None:
        box = obj.bbox
        bbox = (
            float(np.clip((box.x + box.width / 2.0) / width, 0, 1)),
            float(np.clip((box.y + box.height / 2.0) / height, 0, 1)),
            float(np.clip(box.width / width, 0, 1)),
            float(np.clip(box.height / height, 0, 1)),
        )
    else:
        # NaN for an unplaced point, which is what keeps a box off the origin:
        # the sources write (0, 0) there and the geometry filters on isfinite.
        points = np.array(
            [[point.x, point.y] for point in obj.keypoints], dtype=np.float64
        )
        bbox = keypoints_to_bbox(
            points,
            width,
            height,
            margin=policy.margin,
            method=policy.method,
            head_idx=policy.head_index,
            tail_idx=policy.tail_index,
            pad_frac_of_body=policy.pad_frac_of_body,
            min_pad_px=policy.min_pad_px,
            length_pad_frac=policy.length_pad_frac,
            side_pad_frac=policy.side_pad_frac,
        )

    if bbox[2] <= 0 or bbox[3] <= 0:
        return None

    triplets: list[tuple[float, float, int]] = []
    placed = 0
    for point in obj.keypoints:
        if point.visibility == 0:
            # YOLO has no NaN; an unlabelled point is written as the origin
            # with the flag that says to ignore it.
            triplets.append((0.0, 0.0, 0))
            continue
        x, y = normalize_coords(point.x, point.y, width, height)
        triplets.append((x, y, int(point.visibility)))
        placed += 1

    if placed == 0:
        return None
    return format_yolo_pose_line(class_id, bbox, triplets)


def write_split_tree(
    frames: Iterable[tuple[AnnotationFrame, Path]],
    output_dir: Path,
    split_of: dict[str, str],
    line_of: Callable[[AnnotationFrame], list[str]],
    *,
    symlink_images: bool = True,
) -> tuple[int, int]:
    """Write labels and place images under ``<output_dir>/<split>/``.

    Returns ``(written, skipped)``. A frame whose *line_of* yields nothing is
    skipped rather than written empty: an empty label file means "this image
    contains no instances", which is a different claim from "this image had
    instances mosaic could not express".
    """
    for subset in SPLITS:
        (output_dir / subset / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / subset / "labels").mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for frame, source in frames:
        lines = line_of(frame)
        if not lines:
            skipped += 1
            continue
        filename = frame.image_path.name
        subset = split_of.get(filename, "train")
        write_yolo_label(
            output_dir / subset / "labels" / f"{Path(filename).stem}.txt", lines
        )

        destination = output_dir / subset / "images" / filename
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        if symlink_images:
            destination.symlink_to(source.resolve())
        else:
            shutil.copy2(source, destination)
        written += 1

    # An empty test split is noise a trainer will try to validate against.
    test_images = output_dir / "test" / "images"
    if test_images.exists() and not any(test_images.iterdir()):
        shutil.rmtree(output_dir / "test")

    return written, skipped
