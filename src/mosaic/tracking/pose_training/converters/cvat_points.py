"""Convert CVAT XML point annotations to YOLO pose or POLO point format.

CVAT for Images 1.1 XML structure::

    <annotations>
      <image name="filename.png" width="1920" height="646">
        <points label="thorax" points="416.51,173.86" ...>
          <attribute name="class">UnmarkedBee</attribute>
        </points>
      </image>
    </annotations>

Two converters:
- ``convert_cvat_points`` → YOLO pose format (bbox + single keypoint)
- ``convert_cvat_points_polo`` → POLO point format (class, radius, x, y)

The class is determined by an optional attribute (e.g. ``class``) on each
point; when absent, all annotations receive class ID 0.
"""
from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import numpy as np

from typing import Mapping

from .base import (
    KeypointSchema,
    PointDetectionSchema,
    format_polo_label_line,
    format_yolo_pose_line,
    normalize_coords,
    write_yolo_label,
)


def _parse_cvat_xml(
    xml_path: Path,
    class_attr: str | None,
) -> tuple[list[dict], list[str]]:
    """Parse CVAT XML and return per-image annotation records.

    Returns
    -------
    images : list[dict]
        Each dict has keys: ``name``, ``width``, ``height``, ``annotations``.
        Each annotation is ``(x, y, class_name)``.
    class_names_seen : list[str]
        Ordered unique class names encountered (insertion order).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    images: list[dict] = []
    seen_classes: dict[str, None] = {}  # ordered set via dict

    for image_el in root.iter("image"):
        name = image_el.get("name", "")
        width = int(image_el.get("width", 0))
        height = int(image_el.get("height", 0))

        annotations: list[tuple[float, float, str]] = []
        for pts_el in image_el.iter("points"):
            coords_str = pts_el.get("points", "")
            if not coords_str:
                continue

            # Determine class from attribute (shared by all points in element)
            class_name = ""
            if class_attr:
                for attr_el in pts_el.iter("attribute"):
                    if attr_el.get("name") == class_attr:
                        class_name = (attr_el.text or "").strip()
                        break
            if class_name:
                seen_classes[class_name] = None

            # CVAT format: "x1,y1;x2,y2;..." (semicolon-separated points)
            for point_str in coords_str.split(";"):
                xy = point_str.split(",")
                if len(xy) < 2:
                    continue
                x, y = float(xy[0]), float(xy[1])
                annotations.append((x, y, class_name))

        images.append({
            "name": name,
            "width": width,
            "height": height,
            "annotations": annotations,
        })

    return images, list(seen_classes.keys())


def convert_cvat_points(
    cvat_xml_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    *,
    class_attribute: str | None = "class",
    class_names: Sequence[str] | None = None,
    keypoint_name: str = "thorax",
    bbox_size: int = 40,
    split: tuple[float, float, float] = (0.8, 0.15, 0.05),
    symlink_images: bool = True,
    seed: int = 42,
) -> KeypointSchema:
    """Convert CVAT XML point annotations to YOLO pose labels.

    Parameters
    ----------
    cvat_xml_path : path
        Path to the CVAT for Images 1.1 XML export file.
    images_dir : path
        Directory containing the source images.  Filenames must match
        the ``name`` attribute in the XML ``<image>`` elements.
    output_dir : path
        Root directory for the YOLO dataset output.
    class_attribute : str or None
        Name of the ``<attribute>`` element on each ``<points>`` that
        holds the class name (e.g. ``"class"``).  Set to ``None`` if
        there is no class attribute (all annotations get class 0).
    class_names : sequence of str, optional
        Explicit ordered list of class names.  Determines the mapping
        from class name to YOLO class ID (index in list).  If ``None``,
        class names are auto-detected from the XML in insertion order.
    keypoint_name : str
        Name for the single keypoint (used in the returned schema).
    bbox_size : int
        Side length in pixels for the bounding box centred on each point.
    split : (train, valid, test) floats
        Fraction of *annotated* images per split.
    symlink_images : bool
        If True, create symlinks to source images; if False, copy them.
    seed : int
        Random seed for train/valid/test assignment.

    Returns
    -------
    KeypointSchema
        Keypoint schema with a single keypoint and no skeleton.
    """
    cvat_xml_path = Path(cvat_xml_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    # Parse XML
    all_images, seen_classes = _parse_cvat_xml(cvat_xml_path, class_attribute)

    # Resolve class name → class ID mapping
    if class_names is not None:
        name_to_id = {name: idx for idx, name in enumerate(class_names)}
    elif seen_classes:
        name_to_id = {name: idx for idx, name in enumerate(seen_classes)}
    else:
        name_to_id = {}

    resolved_names = list(name_to_id.keys()) if name_to_id else []
    schema = KeypointSchema(names=[keypoint_name], skeleton=[])

    # Filter to annotated images that exist on disk
    usable: list[tuple[dict, Path]] = []
    for img_rec in all_images:
        if not img_rec["annotations"]:
            continue
        img_path = images_dir / img_rec["name"]
        if not img_path.exists():
            continue
        usable.append((img_rec, img_path))

    if not usable:
        n_annotated = sum(1 for r in all_images if r["annotations"])
        print(
            f"[cvat_points] WARNING: no usable images found. "
            f"{len(all_images)} images in XML, {n_annotated} annotated. "
            f"Check that images_dir contains matching filenames."
        )
        return schema

    # Assign to splits
    rng = random.Random(seed)
    shuffled = list(usable)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * split[0])
    n_valid = int(n * split[1])

    split_assignment: dict[str, str] = {}
    for rec, _ in shuffled[:n_train]:
        split_assignment[rec["name"]] = "train"
    for rec, _ in shuffled[n_train: n_train + n_valid]:
        split_assignment[rec["name"]] = "valid"
    for rec, _ in shuffled[n_train + n_valid:]:
        split_assignment[rec["name"]] = "test"

    # Create output directories
    for subset in ("train", "valid", "test"):
        (output_dir / subset / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / subset / "labels").mkdir(parents=True, exist_ok=True)

    half = bbox_size / 2.0
    written = 0
    skipped = 0
    class_counts: dict[str, int] = {}

    for img_rec, img_path in usable:
        img_w = img_rec["width"]
        img_h = img_rec["height"]
        filename = img_rec["name"]

        lines: list[str] = []
        for x, y, class_name in img_rec["annotations"]:
            # Determine class ID
            if name_to_id:
                if class_name not in name_to_id:
                    continue  # skip unknown classes
                class_id = name_to_id[class_name]
            else:
                class_id = 0

            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # Bbox centred on point, clamped to image bounds
            bx = max(0.0, x - half)
            by = max(0.0, y - half)
            bw = min(bbox_size, img_w - bx)
            bh = min(bbox_size, img_h - by)
            cx = float(np.clip((bx + bw / 2.0) / img_w, 0, 1))
            cy = float(np.clip((by + bh / 2.0) / img_h, 0, 1))
            nw = float(np.clip(bw / img_w, 0, 1))
            nh = float(np.clip(bh / img_h, 0, 1))

            # Single keypoint (visibility = 2 = labeled and visible)
            kp_x, kp_y = normalize_coords(x, y, img_w, img_h)
            line = format_yolo_pose_line(
                class_id, (cx, cy, nw, nh), [(kp_x, kp_y, 2)]
            )
            lines.append(line)

        if not lines:
            skipped += 1
            continue

        subset = split_assignment.get(filename, "train")
        stem = Path(filename).stem

        write_yolo_label(output_dir / subset / "labels" / f"{stem}.txt", lines)

        dest_image = output_dir / subset / "images" / filename
        if dest_image.exists() or dest_image.is_symlink():
            dest_image.unlink()
        if symlink_images:
            dest_image.symlink_to(img_path.resolve())
        else:
            shutil.copy2(img_path, dest_image)

        written += 1

    # Remove empty test split
    test_imgs = output_dir / "test" / "images"
    if test_imgs.exists() and not any(test_imgs.iterdir()):
        shutil.rmtree(output_dir / "test")

    print(f"[cvat_points] Wrote {written} labels to {output_dir}")
    if skipped:
        print(f"  Skipped {skipped} images with no valid annotations")
    if resolved_names:
        print(f"  Classes: {resolved_names}")
    if class_counts:
        print(f"  Counts: {class_counts}")
    print(f"  Splits: train={n_train}, valid={n_valid}, test={n - n_train - n_valid}")

    return schema


# ----------------------------------------------------------------------- #
# Shared helpers
# ----------------------------------------------------------------------- #

def _resolve_classes(
    class_names: Sequence[str] | None,
    seen_classes: list[str],
) -> dict[str, int]:
    """Build class name → ID mapping."""
    if class_names is not None:
        return {name: idx for idx, name in enumerate(class_names)}
    if seen_classes:
        return {name: idx for idx, name in enumerate(seen_classes)}
    return {}


def _find_usable_images(
    all_images: list[dict],
    images_dir: Path,
) -> list[tuple[dict, Path]]:
    """Filter to annotated images that exist on disk."""
    usable: list[tuple[dict, Path]] = []
    for img_rec in all_images:
        if not img_rec["annotations"]:
            continue
        img_path = images_dir / img_rec["name"]
        if img_path.exists():
            usable.append((img_rec, img_path))
    return usable


def _assign_splits(
    usable: list[tuple[dict, Path]],
    split: tuple[float, float, float],
    seed: int,
) -> tuple[dict[str, str], int, int]:
    """Shuffle and assign images to train/valid/test splits."""
    rng = random.Random(seed)
    shuffled = list(usable)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * split[0])
    n_valid = int(n * split[1])

    assignment: dict[str, str] = {}
    for rec, _ in shuffled[:n_train]:
        assignment[rec["name"]] = "train"
    for rec, _ in shuffled[n_train: n_train + n_valid]:
        assignment[rec["name"]] = "valid"
    for rec, _ in shuffled[n_train + n_valid:]:
        assignment[rec["name"]] = "test"

    return assignment, n_train, n_valid


def _write_images(
    usable: list[tuple[dict, Path]],
    output_dir: Path,
    split_assignment: dict[str, str],
    line_fn,
    symlink_images: bool,
    tag: str,
) -> tuple[int, int, dict[str, int]]:
    """Write label files and link/copy images for each usable image.

    Parameters
    ----------
    line_fn : callable(img_rec, x, y, class_name, name_to_id) -> str | None
        Produces a label line for one annotation, or None to skip it.
    """
    for subset in ("train", "valid", "test"):
        (output_dir / subset / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / subset / "labels").mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    class_counts: dict[str, int] = {}

    for img_rec, img_path in usable:
        filename = img_rec["name"]
        lines: list[str] = []
        for x, y, class_name in img_rec["annotations"]:
            line = line_fn(img_rec, x, y, class_name)
            if line is not None:
                lines.append(line)
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        if not lines:
            skipped += 1
            continue

        subset = split_assignment.get(filename, "train")
        stem = Path(filename).stem

        write_yolo_label(output_dir / subset / "labels" / f"{stem}.txt", lines)

        dest_image = output_dir / subset / "images" / filename
        if dest_image.exists() or dest_image.is_symlink():
            dest_image.unlink()
        if symlink_images:
            dest_image.symlink_to(img_path.resolve())
        else:
            shutil.copy2(img_path, dest_image)

        written += 1

    # Remove empty test split
    test_imgs = output_dir / "test" / "images"
    if test_imgs.exists() and not any(test_imgs.iterdir()):
        shutil.rmtree(output_dir / "test")

    return written, skipped, class_counts


# ----------------------------------------------------------------------- #
# POLO (point-detection) converter
# ----------------------------------------------------------------------- #

def convert_cvat_points_polo(
    cvat_xml_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    *,
    radii: Mapping[str, float],
    class_attribute: str | None = "class",
    class_names: Sequence[str] | None = None,
    split: tuple[float, float, float] = (0.8, 0.15, 0.05),
    symlink_images: bool = True,
    seed: int = 42,
) -> PointDetectionSchema:
    """Convert CVAT XML point annotations to POLO point-detection labels.

    Label format per line: ``<class_id> <radius> <x_rel> <y_rel>``

    Parameters
    ----------
    cvat_xml_path : path
        Path to the CVAT for Images 1.1 XML export file.
    images_dir : path
        Directory containing the source images.
    output_dir : path
        Root directory for the POLO dataset output.
    radii : dict
        Mapping from **class name** to detection radius in pixels.
        Every class must have a radius entry.
    class_attribute : str or None
        Name of the ``<attribute>`` on each ``<points>`` that holds the
        class name.  ``None`` = single-class (all get class 0).
    class_names : sequence of str, optional
        Explicit ordered class names (index = class ID).  If ``None``,
        auto-detected from the XML.
    split : (train, valid, test) floats
        Fraction of annotated images per split.
    symlink_images : bool
        If True, create symlinks to source images; if False, copy them.
    seed : int
        Random seed for split assignment.

    Returns
    -------
    PointDetectionSchema
        Schema with class names and per-class radii.
    """
    cvat_xml_path = Path(cvat_xml_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    all_images, seen_classes = _parse_cvat_xml(cvat_xml_path, class_attribute)
    name_to_id = _resolve_classes(class_names, seen_classes)
    resolved_names = list(name_to_id.keys())

    # Build per-class-id radii
    radii_by_id: dict[int, float] = {}
    for name, cid in name_to_id.items():
        if name not in radii:
            raise ValueError(
                f"Missing radius for class '{name}'. Provide it in the radii dict."
            )
        radii_by_id[cid] = radii[name]

    schema = PointDetectionSchema(names=resolved_names, radii=radii_by_id)

    usable = _find_usable_images(all_images, images_dir)

    if not usable:
        n_annotated = sum(1 for r in all_images if r["annotations"])
        print(
            f"[cvat_points_polo] WARNING: no usable images found. "
            f"{len(all_images)} images in XML, {n_annotated} annotated. "
            f"Check that images_dir contains matching filenames."
        )
        return schema

    split_assignment, n_train, n_valid = _assign_splits(usable, split, seed)
    n = len(usable)

    def _make_line(img_rec, x, y, class_name):
        if name_to_id and class_name not in name_to_id:
            return None
        cid = name_to_id[class_name] if name_to_id else 0
        radius = radii_by_id.get(cid, 100.0)
        x_rel, y_rel = normalize_coords(x, y, img_rec["width"], img_rec["height"])
        return format_polo_label_line(cid, radius, x_rel, y_rel)

    written, skipped, class_counts = _write_images(
        usable, output_dir, split_assignment, _make_line, symlink_images,
        tag="cvat_points_polo",
    )

    print(f"[cvat_points_polo] Wrote {written} labels to {output_dir}")
    if skipped:
        print(f"  Skipped {skipped} images with no valid annotations")
    if resolved_names:
        print(f"  Classes: {resolved_names}")
    if class_counts:
        print(f"  Counts: {class_counts}")
    print(f"  Radii: {radii_by_id}")
    print(f"  Splits: train={n_train}, valid={n_valid}, test={n - n_train - n_valid}")

    return schema
