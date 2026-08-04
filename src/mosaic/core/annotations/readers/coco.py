"""Reading COCO Keypoints into the canonical representation.

COCO is the closest of the three sources to what mosaic wants: it already has an
instance axis, per-keypoint visibility on the same nought-to-two scale, and an
optional per-instance box. So this reader mostly renames things, which is why
COCO's visibility scale was chosen as the canonical one.

**Parsed into models rather than indexed as dictionaries.** A JSON file is
untyped at the boundary and something has to narrow it; doing that with casts
moves the risk rather than removing it, and a malformed file then fails somewhere
downstream with a ``KeyError`` naming a field the user never heard of. Declaring
the shape means the failure happens here, says which field, and the rest of the
module is ordinary typed code.

Three things it does not pass through unchanged:

**An unlabelled keypoint arrives as literal ``(0, 0, 0)``.** Keeping those
coordinates would drag an instance's box corner to the image origin, so
visibility zero becomes NaN here rather than a position at the top left. That
distinction is the reason the representation has one.

**Category selection is a filter, not a merge.** A COCO file may describe several
categories with different keypoint layouts, and one set has one schema.

**A keypoint subset renumbers the skeleton**, because dropping names without
remapping edges leaves them naming positions that no longer exist.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from mosaic.core.annotations.model import (
    AnnotationFrame,
    AnnotationObject,
    AnnotationSet,
    Bbox,
    Keypoint,
    KeypointSchema,
    Visibility,
)

__all__ = ["read_coco_keypoints"]


class _Category(BaseModel):
    """The subset of a COCO category this reader needs."""

    id: int = 0
    name: str = "animal"
    keypoints: list[str] = Field(default_factory=list)
    skeleton: list[list[int]] = Field(default_factory=list)


class _Image(BaseModel):
    id: int = 0
    file_name: str = ""
    width: int = 0
    height: int = 0


class _Annotation(BaseModel):
    id: int = 0
    image_id: int = 0
    category_id: int = 0
    keypoints: list[float] = Field(default_factory=list)
    bbox: list[float] = Field(default_factory=list)


class _CocoFile(BaseModel):
    """Only what a keypoint reader reads; COCO's other sections are ignored."""

    images: list[_Image] = Field(default_factory=list)
    annotations: list[_Annotation] = Field(default_factory=list)
    categories: list[_Category] = Field(default_factory=list)


def _visibility(flag: float) -> Visibility:
    """COCO's flag, narrowed to the three values the representation allows."""
    value = int(flag)
    if value <= 0:
        return 0
    return 1 if value == 1 else 2


def read_coco_keypoints(
    coco_json_path: str | Path,
    images_dir: str | Path,
    *,
    category_name: str | None = None,
    keypoint_indices: Sequence[int] | None = None,
) -> AnnotationSet:
    """Read a COCO Keypoints file as one annotation set.

    Args:
        coco_json_path: The COCO JSON.
        images_dir: What ``file_name`` values are relative to.
        category_name: Which category to read. Defaults to the first declared,
            which is right for the single-category files this mostly meets and
            wrong for anything else, so a file with several should say.
        keypoint_indices: Read only these keypoints, renumbering the skeleton.

    Returns:
        Every image in the file, annotated or not. An image with no instances is
        a real state -- looked at, nothing present -- and the caller decides what
        it means.

    Raises:
        ValueError: The file declares no categories, or names one that is absent.
    """
    coco_json_path = Path(coco_json_path)
    coco = _CocoFile.model_validate(json.loads(coco_json_path.read_text()))

    if not coco.categories:
        raise ValueError(f"{coco_json_path.name} declares no categories")
    if category_name is None:
        category = coco.categories[0]
    else:
        matches = [c for c in coco.categories if c.name == category_name]
        if not matches:
            available = sorted(c.name for c in coco.categories)
            raise ValueError(
                f"Category {category_name!r} not found in {coco_json_path.name}. "
                f"Available: {available}"
            )
        category = matches[0]

    # COCO indexes skeleton endpoints from one; this representation indexes
    # from zero. Read straight through, a two-edge skeleton on three keypoints
    # names node 3, which does not exist -- and nothing downstream checks, so
    # the dangling edge simply travelled.
    schema = KeypointSchema(
        names=tuple(category.keypoints),
        skeleton=tuple(
            (edge[0] - 1, edge[1] - 1)
            for edge in category.skeleton
            if len(edge) >= 2 and edge[0] >= 1 and edge[1] >= 1
        ),
    )
    selected = (
        list(keypoint_indices)
        if keypoint_indices is not None
        else list(range(len(category.keypoints)))
    )
    if keypoint_indices is not None:
        schema = schema.subset(selected)

    by_image: dict[int, list[_Annotation]] = defaultdict(list)
    for annotation in coco.annotations:
        if annotation.category_id == category.id:
            by_image[annotation.image_id].append(annotation)

    frames = tuple(
        AnnotationFrame(
            image_path=Path(image.file_name),
            width=image.width,
            height=image.height,
            objects=tuple(
                _read_object(annotation, selected, category.name)
                for annotation in by_image.get(image.id, [])
            ),
        )
        for image in coco.images
    )

    return AnnotationSet(
        schema=schema,
        frames=frames,
        categories=(category.name,),
        image_root=Path(images_dir),
        source_format="coco",
    )


def _read_object(
    annotation: _Annotation, selected: Sequence[int], category: str
) -> AnnotationObject:
    """One COCO annotation as one instance."""
    flat = annotation.keypoints
    triplets = [flat[i : i + 3] for i in range(0, len(flat), 3)]

    keypoints: list[Keypoint] = []
    for index in selected:
        if index >= len(triplets) or len(triplets[index]) < 3:
            keypoints.append(Keypoint.absent())
            continue
        x, y, flag = triplets[index]
        visibility = _visibility(flag)
        # An unlabelled point arrives as (0, 0, 0). Those coordinates are
        # filler, not a position, and must not reach a box.
        keypoints.append(
            Keypoint.absent()
            if visibility == 0
            else Keypoint(x=x, y=y, visibility=visibility)
        )

    box = annotation.bbox
    return AnnotationObject(
        keypoints=tuple(keypoints),
        category=category,
        source_id=str(annotation.id),
        bbox=(
            Bbox(x=box[0], y=box[1], width=box[2], height=box[3])
            if len(box) >= 4
            else None
        ),
    )
