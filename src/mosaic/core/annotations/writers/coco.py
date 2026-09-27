"""Writing annotations out as COCO Keypoints, for the tools that read COCO.

COCO is what other tools read: CVAT imports it, pycocotools and FiftyOne load it,
and given it ``sleap-io`` builds a ``.slp``. mosaic's own record of an annotation
set is :mod:`mosaic.core.annotations.pose_annotations`; this is the export written
beside every saved revision of one, and the file a person downloads. A training
dataset *layout* -- the ``<split>/images`` and ``<split>/labels`` tree a YOLO
trainer walks -- is a different thing and lives in ``tracking``.

**Correct for a reader that knows nothing about mosaic.** Only finished frames are
written, so every listed image is exhaustively annotated and one with no
annotations is a true negative, which is what COCO means by it. Every annotation
carries a ``bbox``: the drawn box, or the one the set's padding derives, which is
the box the annotator saw.

**Extended only where other tools already look.** Each pose is a category.
Everything else mosaic knows rides conventions other tools share:

- an annotation's ``attributes`` object -- the extension CVAT writes and reads
  back, and Datumaro, FiftyOne and sleap-io read -- holds ``alias``, ``alias_id``,
  ``origin``, ``source_ref`` and ``bbox_source`` (``drawn`` or ``derived``), all
  scalars, as CVAT's attributes are;
- a category's ``attributes`` holds ``pose_id``, the pose's stable id. The COCO
  ids themselves stay 1..N, because common converters compute a class as
  ``category_id - 1``;
- an image's ``seq_id`` and ``frame_num`` are the recording and the frame within
  it, the COCO Camera Traps names for exactly that pair.

Keypoint mirror pairs have no COCO convention and are not written; the formats
that carry them (``.slp``, a YOLO ``data.yaml``) get them from the saved state.

An unplaced keypoint is written as ``(0, 0, 0)``, the convention every COCO
producer uses and :mod:`mosaic.core.annotations.readers.coco` undoes.
"""

from __future__ import annotations

import json
from pathlib import Path

from mosaic.core.annotations.bbox import derived_bbox
from mosaic.core.annotations.model import AnnotationSet, Bbox
from mosaic.core.annotations.pose_annotations import (
    Alias,
    PoseAnnotationSet,
    PoseDefinition,
    PoseObject,
    widen,
)
from mosaic.core.json_value import JsonValue

__all__ = ["coco_keypoints_document", "coco_keypoints_payload", "write_coco_keypoints"]


def write_coco_keypoints(
    annotations: AnnotationSet,
    json_path: str | Path,
    *,
    indent: int | None = 2,
) -> Path:
    """Write a single-schema set as a COCO Keypoints file.

    Image paths are written relative to the set's ``image_root`` when it has
    one, because a COCO file names images relative to a dataset root and an
    absolute path in that field is what makes a dataset unmovable.

    Args:
        annotations: What to write. A box the set does not carry is written as
            the bare hull of the placed keypoints.
        json_path: Where to write it.
        indent: JSON indentation. ``None`` writes it compact.

    Returns:
        The path written.
    """
    json_path = Path(json_path)
    document = coco_keypoints_document(widen(annotations))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _ = json_path.write_text(json.dumps(document, indent=indent))
    return json_path


def coco_keypoints_payload(annotations: PoseAnnotationSet) -> bytes:
    """The export of *annotations* as bytes: compact, keys sorted."""
    document = coco_keypoints_document(annotations)
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def coco_keypoints_document(annotations: PoseAnnotationSet) -> dict[str, JsonValue]:
    """*annotations* as the COCO Keypoints mapping, not yet serialized.

    The usable frames only, in the order the set holds them. Ids are assigned by
    position: categories by ascending pose id, images and annotations in order.
    """
    poses = sorted(annotations.poses, key=lambda pose: pose.id)
    category_of = {pose.id: position for position, pose in enumerate(poses, start=1)}
    aliases = {alias.id: alias for pose in poses for alias in pose.aliases}

    images: list[JsonValue] = []
    records: list[JsonValue] = []
    usable = (frame for frame in annotations.frames if frame.usable)
    for image_id, frame in enumerate(usable, start=1):
        image: dict[str, JsonValue] = {
            "id": image_id,
            "file_name": frame.image_path.as_posix(),
            "width": frame.width,
            "height": frame.height,
        }
        if frame.sequence:
            image["seq_id"] = frame.sequence
        if frame.frame_index >= 0:
            image["frame_num"] = frame.frame_index
        images.append(image)
        for obj in frame.objects:
            drawn = obj.bbox is not None
            box = obj.bbox or derived_bbox(
                obj.keypoints, frame.width, frame.height, annotations.bbox_policy
            )
            records.append(
                _record(
                    obj,
                    box,
                    drawn=drawn,
                    alias=None if obj.alias_id is None else aliases[obj.alias_id],
                    ids=(len(records) + 1, image_id, category_of[obj.pose_id]),
                )
            )

    return {
        "images": images,
        "annotations": records,
        "categories": [
            _category(pose, category_id)
            for category_id, pose in enumerate(poses, start=1)
        ],
    }


def _category(pose: PoseDefinition, category_id: int) -> dict[str, JsonValue]:
    return {
        "id": category_id,
        "name": pose.name,
        "supercategory": "animal",
        "keypoints": list(pose.schema.names),
        # COCO's endpoints count from one, which is what every other reader of
        # this file expects.
        "skeleton": [[a + 1, b + 1] for a, b in sorted(pose.schema.skeleton)],
        "attributes": {"pose_id": pose.id},
    }


def _record(
    obj: PoseObject,
    box: Bbox,
    *,
    drawn: bool,
    alias: Alias | None,
    ids: tuple[int, int, int],
) -> dict[str, JsonValue]:
    """One instance as a COCO annotation record."""
    annotation_id, image_id, category_id = ids
    flat: list[JsonValue] = []
    placed = 0
    for point in obj.keypoints:
        if point.visibility == 0:
            # COCO has no NaN. Every producer writes the origin here, and the
            # reader knows not to believe it.
            flat.extend((0.0, 0.0, 0.0))
            continue
        flat.extend((float(point.x), float(point.y), float(point.visibility)))
        placed += 1

    attributes: dict[str, JsonValue] = {"bbox_source": "drawn" if drawn else "derived"}
    if alias is not None:
        attributes["alias"] = alias.name
        attributes["alias_id"] = alias.id
    if obj.origin is not None:
        attributes["origin"] = obj.origin
    if obj.source_ref is not None:
        attributes["source_ref"] = obj.source_ref

    record: dict[str, JsonValue] = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": 0,
        "num_keypoints": placed,
        "area": float(box.width * box.height),
        "bbox": [float(box.x), float(box.y), float(box.width), float(box.height)],
        "keypoints": flat,
        "attributes": attributes,
    }
    # Emitted only when there is one. The readers on the other side chain their
    # candidate keys with ``or``, so a falsy value reads the same as an absent
    # one and an instance carrying it loses its identity silently rather than
    # failing. ``track_id`` is a ``str`` where ``""`` means "no track", so
    # truthiness is exactly the right test: a caller numbering animals from zero
    # passes ``"0"``, which is truthy. Widening the field to an int is what would
    # reintroduce the silent case.
    if obj.track_id:
        record["track_id"] = obj.track_id
    return record
