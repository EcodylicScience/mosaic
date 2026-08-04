"""Reading CVAT "for Images" XML into the canonical representation.

CVAT point annotations are the sparsest of the three sources: a click, a class
attribute, and nothing else. So this reader has to invent more than the others,
and every invention is a decision worth naming.

**One point is one instance, carrying one keypoint.** A CVAT ``<points>`` element
may hold several coordinates, and the existing converter flattens them all into
separate label lines. That is right for point detection -- each click is a
separate animal -- and it means the schema has exactly one keypoint, named by
the caller.

**Occlusion is not expressible, so it is never claimed.** CVAT carries
``occluded`` and ``outside`` attributes on a shape, but the point converters
never read them and the labels they wrote were all fully visible. Reading them
now would change what every existing CVAT dataset means, so a placed point is
visibility 2 and the middle value simply does not occur. That is a limitation
recorded rather than a gap papered over: a reader that cannot distinguish
occlusion should say so once, here, instead of leaving each emitter to guess.

**A class attribute is the category.** Which attribute holds it is a parameter,
because CVAT lets a project call it anything; the classes are collected in the
order they are first seen, which is what fixes their numbering downstream.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from mosaic.core.annotations.model import (
    AnnotationFrame,
    AnnotationObject,
    AnnotationSet,
    Keypoint,
    KeypointSchema,
)

__all__ = ["read_cvat_points"]


def read_cvat_points(
    xml_path: str | Path,
    images_dir: str | Path,
    *,
    keypoint_name: str = "point",
    class_attribute: str | None = "class",
    class_names: tuple[str, ...] | None = None,
) -> AnnotationSet:
    """Read CVAT point annotations as one annotation set.

    Args:
        xml_path: A CVAT "for Images 1.1" XML export.
        images_dir: What each image's ``name`` is relative to.
        keypoint_name: What to call the single keypoint each instance carries.
        class_attribute: Which shape attribute holds the class. ``None`` puts
            every point in one class.
        class_names: Fix the class order, and therefore the ids they are
            numbered with. Defaults to first-seen order, which is stable for one
            file and not across two.

    Returns:
        Every image in the export, annotated or not.

    Raises:
        OSError: The XML cannot be read.
        ET.ParseError: The XML is malformed.
    """
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()

    frames: list[AnnotationFrame] = []
    seen: dict[str, None] = {}

    for image in root.iter("image"):
        objects: list[AnnotationObject] = []
        for shape in image.iter("points"):
            coordinates = shape.get("points", "")
            if not coordinates:
                continue

            category = ""
            if class_attribute:
                for attribute in shape.iter("attribute"):
                    if attribute.get("name") == class_attribute:
                        category = (attribute.text or "").strip()
                        break
            if category:
                seen[category] = None

            # "x1,y1;x2,y2;..." -- each coordinate is its own instance, because
            # each click is its own animal.
            for pair in coordinates.split(";"):
                parts = pair.split(",")
                if len(parts) < 2:
                    continue
                objects.append(
                    AnnotationObject(
                        keypoints=(
                            Keypoint(
                                x=float(parts[0]), y=float(parts[1]), visibility=2
                            ),
                        ),
                        category=category or "animal",
                    )
                )

        frames.append(
            AnnotationFrame(
                image_path=Path(image.get("name", "")),
                width=int(image.get("width", 0)),
                height=int(image.get("height", 0)),
                objects=tuple(objects),
            )
        )

    categories = class_names if class_names is not None else tuple(seen)
    return AnnotationSet(
        schema=KeypointSchema(names=(keypoint_name,)),
        frames=tuple(frames),
        categories=categories or ("animal",),
        image_root=Path(images_dir),
        source_format="cvat-points",
    )
