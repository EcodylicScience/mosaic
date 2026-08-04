"""Reading an emitted YOLO-pose dataset back into the canonical representation.

The other readers exist because a source format arrived from outside. This one
exists because the emitted tree is sometimes the only copy left: a dataset gets
built, the CVAT export moves or the notebook that made it is gone, and the
question "can the boxes be padded differently without relabelling" still has to
have an answer.

With this, it does, and a better one than a bespoke rewriter. Reading an emitted
dataset gives an ordinary annotation set, so the same set can be re-emitted with
a different :class:`~mosaic.core.annotations.bbox.BboxPolicy`, re-split,
converted to COCO, or exported as a ``.slp``. Recomputing the four box columns in
place was one of those four things.

**What survives the round trip, and what does not.** Keypoints, visibility, the
instance axis, the split each frame was in, and the image are all recoverable.
Two things are not: coordinates were written to six decimal places, so they come
back rounded to roughly a hundredth of a pixel on a 1000-pixel image; and a box
the source supplied is indistinguishable from one a policy derived, because YOLO
stores only the result. The second matters -- a re-emission derives every box
unless told otherwise, which is what re-padding wants and what a faithful
round trip would not do.
"""

from __future__ import annotations

import math
from pathlib import Path

from mosaic.core.annotations.model import (
    AnnotationFrame,
    AnnotationObject,
    AnnotationSet,
    Bbox,
    Keypoint,
    KeypointSchema,
    Split,
    Visibility,
)

__all__ = ["read_yolo_pose"]

_SPLITS: tuple[Split, Split, Split] = ("train", "valid", "test")
_IMAGE_SUFFIXES: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


def _visibility(flag: float) -> Visibility:
    value = int(round(flag))
    if value <= 0:
        return 0
    return 1 if value == 1 else 2


def read_yolo_pose(
    dataset_dir: str | Path,
    schema: KeypointSchema,
    *,
    categories: tuple[str, ...] = ("animal",),
    keep_boxes: bool = False,
) -> AnnotationSet:
    """Read ``<dataset_dir>/<split>/{images,labels}`` as one annotation set.

    Args:
        dataset_dir: The dataset root.
        schema: The keypoint layout the labels were written against. YOLO does
            not record it -- the ``data.yaml`` beside the tree carries only a
            count -- so the caller supplies it, and a mismatch is an error
            rather than a silent misread.
        categories: Class names, indexed by the label's leading class id.
        keep_boxes: Carry each written box through as a source box. Off by
            default because the common reason to read an emitted dataset is to
            derive the boxes again; on, the round trip is faithful.

    Returns:
        Every labelled frame found, each carrying the split it was read from.

    Raises:
        FileNotFoundError: *dataset_dir* holds none of the expected splits.
        ValueError: A label row does not match *schema*.
    """
    dataset_dir = Path(dataset_dir)
    present: list[Split] = [s for s in _SPLITS if (dataset_dir / s / "labels").is_dir()]
    if not present:
        raise FileNotFoundError(
            f"{dataset_dir} holds no <split>/labels directory; expected any of "
            f"{list(_SPLITS)}"
        )

    frames: list[AnnotationFrame] = []
    for split in present:
        labels_dir = dataset_dir / split / "labels"
        images_dir = dataset_dir / split / "images"
        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = _find_image(images_dir, label_path.stem)
            if image_path is None:
                continue
            width, height = _image_size(image_path)
            frames.append(
                AnnotationFrame(
                    image_path=image_path,
                    width=width,
                    height=height,
                    objects=tuple(
                        _read_row(row, schema, width, height, categories, keep_boxes)
                        for row in label_path.read_text().splitlines()
                        if row.strip()
                    ),
                    split=split,
                )
            )

    return AnnotationSet(
        schema=schema,
        frames=tuple(frames),
        categories=categories,
        source_format="yolo-pose",
    )


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in _IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _image_size(path: Path) -> tuple[int, int]:
    """The image's pixel size, which YOLO's normalized coordinates need.

    Imported here rather than at module scope: ``core`` should not require an
    imaging library to be installed for the rest of this package to import, and
    only this reader needs one.
    """
    from PIL import Image

    with Image.open(path) as image:
        return image.width, image.height


def _read_row(
    row: str,
    schema: KeypointSchema,
    width: int,
    height: int,
    categories: tuple[str, ...],
    keep_boxes: bool,
) -> AnnotationObject:
    """One ``cls cx cy w h (x y v)*`` row as one instance, in native pixels."""
    tokens = row.split()
    expected = 5 + schema.num_keypoints * 3
    if len(tokens) != expected:
        raise ValueError(
            f"label row has {len(tokens)} values but the schema declares "
            f"{schema.num_keypoints} keypoints, so {expected} were expected: {row!r}"
        )

    class_id = int(tokens[0])
    cx, cy, box_w, box_h = (float(v) for v in tokens[1:5])

    keypoints: list[Keypoint] = []
    for index in range(schema.num_keypoints):
        x, y, flag = (float(v) for v in tokens[5 + index * 3 : 8 + index * 3])
        visibility = _visibility(flag)
        keypoints.append(
            Keypoint.absent()
            if visibility == 0
            else Keypoint(x=x * width, y=y * height, visibility=visibility)
        )

    bbox = None
    if keep_boxes and not (math.isclose(box_w, 0.0) and math.isclose(box_h, 0.0)):
        bbox = Bbox(
            x=(cx - box_w / 2.0) * width,
            y=(cy - box_h / 2.0) * height,
            width=box_w * width,
            height=box_h * height,
        )

    return AnnotationObject(
        keypoints=tuple(keypoints),
        category=categories[class_id] if class_id < len(categories) else "animal",
        bbox=bbox,
    )
