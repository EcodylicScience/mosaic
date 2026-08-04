"""Writing the canonical representation back out as COCO Keypoints.

The inverse of :mod:`mosaic.core.annotations.readers.coco`, and the reason it
exists is interchange rather than training. COCO is what other tools read: given
this, ``sleap-io`` will build a ``.slp``, and several annotation tools will
import it. A training-dataset *layout* -- the ``<split>/images`` and
``<split>/labels`` tree a YOLO trainer walks -- is a different thing, belongs to
whichever trainer wants it, and lives in ``tracking``.

**Round-tripping is the contract.** Reading what this writes must give back the
set that was written, and a test asserts it. That is what makes the pair usable
as a boundary: mosaic can hand an annotation set to any tool that speaks COCO
without either side losing the instance axis or the difference between an
unplaced keypoint and one at the origin.

The one asymmetry is deliberate. COCO has no NaN, so an unplaced keypoint is
written as ``(0, 0, 0)`` -- the convention every COCO producer uses and the
reader knows to undo. Nothing else is lossy.
"""

from __future__ import annotations

import json
from pathlib import Path

from mosaic.core.annotations.model import AnnotationObject, AnnotationSet

__all__ = ["write_coco_keypoints"]


def write_coco_keypoints(
    annotations: AnnotationSet,
    json_path: str | Path,
    *,
    indent: int | None = 2,
) -> Path:
    """Write *annotations* as a COCO Keypoints file.

    Image paths are written relative to the set's ``image_root`` when it has
    one, because a COCO file names images relative to a dataset root and an
    absolute path in that field is what makes a dataset unmovable.

    Args:
        annotations: What to write.
        json_path: Where to write it.
        indent: JSON indentation. ``None`` writes it compact.

    Returns:
        The path written.
    """
    json_path = Path(json_path)

    category = {
        "id": 1,
        "name": annotations.categories[0] if annotations.categories else "animal",
        "supercategory": "animal",
        "keypoints": list(annotations.schema.names),
        # Back to COCO's one-based endpoints, which is what every other
        # reader of this file expects.
        "skeleton": [[a + 1, b + 1] for a, b in annotations.schema.skeleton],
    }

    images: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for image_id, frame in enumerate(annotations.frames, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": frame.image_path.as_posix(),
                "width": frame.width,
                "height": frame.height,
            }
        )
        for obj in frame.objects:
            records.append(_write_object(obj, image_id, len(records) + 1))

    document = {
        "images": images,
        "annotations": records,
        "categories": [category],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _ = json_path.write_text(json.dumps(document, indent=indent))
    return json_path


def _write_object(
    obj: AnnotationObject, image_id: int, annotation_id: int
) -> dict[str, object]:
    """One instance as a COCO annotation record."""
    flat: list[float] = []
    placed = 0
    for point in obj.keypoints:
        if point.visibility == 0:
            # COCO has no NaN. Every producer writes the origin here, and the
            # reader knows not to believe it.
            flat.extend((0.0, 0.0, 0.0))
            continue
        flat.extend((point.x, point.y, float(point.visibility)))
        placed += 1

    if obj.bbox is not None:
        box = [obj.bbox.x, obj.bbox.y, obj.bbox.width, obj.bbox.height]
    else:
        xs = [point.x for point in obj.placed_keypoints]
        ys = [point.y for point in obj.placed_keypoints]
        box = (
            [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            if xs
            else [0.0] * 4
        )

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "iscrowd": 0,
        "num_keypoints": placed,
        "area": float(box[2] * box[3]),
        "bbox": box,
        "keypoints": flat,
    }
