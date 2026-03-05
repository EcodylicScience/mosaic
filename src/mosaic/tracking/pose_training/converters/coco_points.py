"""Convert COCO JSON annotations to POLO point-detection labels.

Supports two annotation sources from CVAT (or other COCO-compatible tools):
1. COCO Keypoints with a 1-node skeleton (``point_source="keypoint"``).
2. COCO Instances with bounding boxes (``point_source="bbox_center"``).

POLO label format (per line):
    <class_id> <radius> <x_rel> <y_rel>

All coordinates normalized to [0, 1].  Radius is in pixels.
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping

from .base import (
    PointDetectionSchema,
    format_polo_label_line,
    normalize_coords,
    write_yolo_label,
)
from .cvat_points import _default_group_key, _print_split_summary, split_filenames


def _load_coco_multi(
    coco_json_path: Path,
    category_names: list[str] | None,
) -> tuple[dict[int, dict], dict[int, list[dict]], list[dict], dict[int, int]]:
    """Load COCO JSON and return lookup structures for one or more categories.

    Returns
    -------
    images_by_id : dict[int, dict]
        Mapping from image ID to image record.
    anns_by_image_id : dict[int, list[dict]]
        Annotations grouped by image ID, filtered to selected categories.
    categories : list[dict]
        Selected COCO categories, in order.
    cat_id_to_class_id : dict[int, int]
        Mapping from COCO category_id to contiguous 0-indexed class ID.
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    all_categories = coco.get("categories", [])
    if not all_categories:
        raise ValueError("COCO JSON has no categories")

    # Select categories
    if category_names is not None:
        selected = []
        for name in category_names:
            matches = [c for c in all_categories if c["name"] == name]
            if not matches:
                available = [c["name"] for c in all_categories]
                raise ValueError(
                    f"Category '{name}' not found. Available: {available}"
                )
            selected.append(matches[0])
    else:
        selected = list(all_categories)

    # Map COCO category IDs to contiguous 0-indexed class IDs
    cat_id_to_class_id = {cat["id"]: idx for idx, cat in enumerate(selected)}
    selected_cat_ids = set(cat_id_to_class_id)

    images_by_id = {img["id"]: img for img in coco.get("images", [])}

    anns_by_image_id: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("category_id") in selected_cat_ids:
            anns_by_image_id[ann["image_id"]].append(ann)

    return images_by_id, dict(anns_by_image_id), selected, cat_id_to_class_id


def _extract_point(
    ann: dict[str, Any],
    img_w: int,
    img_h: int,
    point_source: str,
) -> tuple[float, float] | None:
    """Extract a single (x_norm, y_norm) point from a COCO annotation.

    Parameters
    ----------
    ann : dict
        COCO annotation dict.
    img_w, img_h : int
        Image dimensions for normalization.
    point_source : str
        ``"keypoint"`` -- use first visible keypoint.
        ``"bbox_center"`` -- compute center of annotation bbox.

    Returns
    -------
    (x_norm, y_norm) or None if no valid point can be extracted.
    """
    if point_source == "keypoint":
        raw_kps = ann.get("keypoints", [])
        n_kps = len(raw_kps) // 3
        for i in range(n_kps):
            x, y, vis = raw_kps[i * 3], raw_kps[i * 3 + 1], raw_kps[i * 3 + 2]
            if vis > 0:
                return normalize_coords(x, y, img_w, img_h)
        return None

    elif point_source == "bbox_center":
        bbox = ann.get("bbox")
        if bbox is None or len(bbox) < 4:
            return None
        bx, by, bw, bh = bbox  # COCO: top-left x, y, width, height (pixels)
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
        return normalize_coords(cx, cy, img_w, img_h)

    else:
        raise ValueError(f"Unknown point_source: {point_source!r}")


def convert_coco_points(
    coco_json_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    *,
    radii: Mapping[str, float],
    category_names: list[str] | None = None,
    point_source: str = "keypoint",
    split: tuple[float, float, float] = (0.8, 0.15, 0.05),
    symlink_images: bool = True,
    seed: int = 42,
    split_by: str = "image",
    group_key: Callable[[str], str] | None = None,
) -> PointDetectionSchema:
    """Convert COCO JSON to POLO point-detection labels.

    Parameters
    ----------
    coco_json_path : path
        Path to the COCO JSON file (e.g. CVAT COCO Keypoints 1.0 export).
    images_dir : path
        Directory containing the source images.
    output_dir : path
        Root directory for the POLO dataset output.
    radii : dict
        Mapping from **class name** to radius in pixels.  Every category
        being converted must have a radius.  Example: ``{"bee": 30.0}``.
    category_names : list of str, optional
        Which COCO categories to include.  ``None`` = all categories.
    point_source : str
        ``"keypoint"`` -- extract from COCO keypoints (1-node skeleton).
        ``"bbox_center"`` -- compute center of annotation bounding box.
    split : (train, valid, test) floats
        Fraction of images per split.  Must sum to ~1.0.
    symlink_images : bool
        If True, create symlinks to source images; if False, copy them.
    seed : int
        Random seed for train/valid/test assignment.
    split_by : ``"image"`` or ``"group"``
        When ``"group"``, all images sharing the same group key (e.g.
        frames from the same video) are kept together in one split.
    group_key : callable, optional
        Function ``filename -> group_name``.  Only used when
        ``split_by="group"``.  Defaults to splitting on ``__frame``.

    Returns
    -------
    PointDetectionSchema
        Schema with class names and per-class radii (keyed by class ID).
    """
    coco_json_path = Path(coco_json_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    # Load COCO JSON
    images_by_id, anns_by_image_id, categories, cat_id_to_class_id = (
        _load_coco_multi(coco_json_path, category_names)
    )

    # Build class name list and per-class-id radii
    class_names = [cat["name"] for cat in categories]
    radii_by_id: dict[int, float] = {}
    for cat in categories:
        name = cat["name"]
        if name not in radii:
            raise ValueError(
                f"Missing radius for category '{name}'. "
                f"Provide it in the radii dict."
            )
        class_id = cat_id_to_class_id[cat["id"]]
        radii_by_id[class_id] = radii[name]

    schema = PointDetectionSchema(names=class_names, radii=radii_by_id)

    # Find usable images (exist on disk AND have annotations)
    usable: list[tuple[dict, Path, list[dict]]] = []
    for img in images_by_id.values():
        img_path = images_dir / img["file_name"]
        if not img_path.exists():
            continue
        img_id = img["id"]
        if img_id not in anns_by_image_id:
            continue
        usable.append((img, img_path, anns_by_image_id[img_id]))

    if not usable:
        print(
            f"[coco_points] WARNING: no usable images found. "
            f"{len(images_by_id)} images in JSON, "
            f"{len(anns_by_image_id)} annotated. "
            f"Check that images_dir contains matching filenames."
        )
        return schema

    # Assign to splits
    filenames = [img_rec["file_name"] for img_rec, _, _ in usable]
    split_assignment, n_train, n_valid = split_filenames(
        filenames, split, seed, split_by=split_by, group_key=group_key,
    )
    n = len(usable)

    # Create output directories
    for subset in ("train", "valid", "test"):
        (output_dir / subset / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / subset / "labels").mkdir(parents=True, exist_ok=True)

    # Process each image
    written = 0
    skipped = 0
    for img_record, img_path, annotations in usable:
        img_w = int(img_record["width"])
        img_h = int(img_record["height"])
        filename = img_record["file_name"]

        lines = []
        for ann in annotations:
            coco_cat_id = ann.get("category_id")
            if coco_cat_id not in cat_id_to_class_id:
                continue
            class_id = cat_id_to_class_id[coco_cat_id]
            radius = radii_by_id[class_id]

            point = _extract_point(ann, img_w, img_h, point_source)
            if point is None:
                continue

            x_rel, y_rel = point
            lines.append(format_polo_label_line(class_id, radius, x_rel, y_rel))

        if not lines:
            skipped += 1
            continue

        subset = split_assignment.get(filename, "train")
        stem = Path(filename).stem

        # Write POLO label
        write_yolo_label(output_dir / subset / "labels" / f"{stem}.txt", lines)

        # Link or copy image
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

    print(
        f"[coco_points] Wrote {written} labels to {output_dir}"
        + (f"  (skipped {skipped} with no valid points)" if skipped else "")
    )
    print(f"  Categories: {class_names}, radii: {radii_by_id}")
    _print_split_summary(
        split_assignment, n_train, n_valid, n, split_by,
        group_key or _default_group_key,
    )

    return schema
