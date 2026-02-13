"""Convert COCO Keypoints JSON (e.g. from CVAT export) to YOLO pose format.

COCO Keypoints 1.0 JSON structure:
    images: [{id, file_name, width, height}, ...]
    annotations: [{image_id, category_id, bbox, keypoints, num_keypoints}, ...]
    categories: [{id, name, keypoints, skeleton}, ...]

COCO bbox format: [x_topleft, y_topleft, width, height] in pixels.
COCO keypoints: flat list [x1, y1, v1, x2, y2, v2, ...] in pixels.
COCO visibility: 0 = not labeled, 1 = labeled not visible, 2 = labeled visible.
  (same semantics as YOLO visibility flags)
"""
from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .base import (
    KeypointSchema,
    format_yolo_pose_line,
    keypoints_to_bbox,
    normalize_coords,
    write_yolo_label,
)


def _load_coco(
    coco_json_path: Path,
    category_name: str | None,
) -> tuple[dict[int, dict], dict[int, list[dict]], dict]:
    """Load COCO JSON and return lookup structures.

    Returns
    -------
    images_by_id : dict[int, dict]
        Mapping from image ID to image record.
    anns_by_image_id : dict[int, list[dict]]
        Annotations grouped by image ID, filtered to selected category.
    category : dict
        The selected COCO category (with keypoints and skeleton).
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    categories = coco.get("categories", [])
    if not categories:
        raise ValueError("COCO JSON has no categories")

    # Resolve category
    if category_name is not None:
        matches = [c for c in categories if c["name"] == category_name]
        if not matches:
            available = [c["name"] for c in categories]
            raise ValueError(
                f"Category '{category_name}' not found. Available: {available}"
            )
        category = matches[0]
    else:
        category = categories[0]

    images_by_id = {img["id"]: img for img in coco.get("images", [])}

    anns_by_image_id: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("category_id") == category["id"]:
            anns_by_image_id[ann["image_id"]].append(ann)

    return images_by_id, dict(anns_by_image_id), category


def _coco_ann_to_yolo_line(
    ann: dict[str, Any],
    img_w: int,
    img_h: int,
    selected_indices: list[int],
    class_id: int,
    bbox_source: str,
    bbox_margin: float,
) -> str | None:
    """Convert one COCO annotation to a YOLO pose label line.

    Returns None if the annotation has no valid keypoints.
    """
    raw_kps = ann.get("keypoints", [])
    n_total = len(raw_kps) // 3
    if n_total == 0:
        return None

    all_kps = np.array(raw_kps, dtype=np.float64).reshape(n_total, 3)
    selected = all_kps[selected_indices]

    # Determine bounding box
    if bbox_source == "annotation" and "bbox" in ann:
        bx, by, bw, bh = ann["bbox"]  # COCO: top-left x, y, w, h in pixels
        cx = (bx + bw / 2.0) / img_w
        cy = (by + bh / 2.0) / img_h
        nw = bw / img_w
        nh = bh / img_h
        bbox = (
            float(np.clip(cx, 0, 1)),
            float(np.clip(cy, 0, 1)),
            float(np.clip(nw, 0, 1)),
            float(np.clip(nh, 0, 1)),
        )
    else:
        kps_xy = selected[:, :2]
        bbox = keypoints_to_bbox(kps_xy, img_w, img_h, margin=bbox_margin)

    if bbox[2] <= 0 or bbox[3] <= 0:
        return None

    # Build normalized keypoint triplets
    kps_xyv = []
    n_visible = 0
    for i in range(len(selected)):
        vis = int(selected[i, 2])
        if vis == 0:
            kps_xyv.append((0.0, 0.0, 0))
        else:
            nx, ny = normalize_coords(selected[i, 0], selected[i, 1], img_w, img_h)
            kps_xyv.append((nx, ny, vis))
            n_visible += 1

    if n_visible == 0:
        return None

    return format_yolo_pose_line(class_id, bbox, kps_xyv)


def convert_coco_keypoints(
    coco_json_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    *,
    category_name: str | None = None,
    keypoint_indices: Sequence[int] | None = None,
    split: tuple[float, float, float] = (0.8, 0.15, 0.05),
    class_id: int = 0,
    bbox_source: str = "annotation",
    bbox_margin: float = 0.1,
    symlink_images: bool = True,
    seed: int = 42,
) -> KeypointSchema:
    """Convert COCO Keypoints JSON to YOLO pose labels.

    Reads a COCO Keypoints 1.0 JSON file (e.g. exported from CVAT) and
    produces a YOLO pose dataset with train/valid/test splits.

    Parameters
    ----------
    coco_json_path : path
        Path to the COCO Keypoints JSON file.
    images_dir : path
        Directory containing the source images.  Image filenames must match
        the ``file_name`` field in the COCO JSON ``images`` array.
    output_dir : path
        Root directory for the YOLO dataset output.
    category_name : str, optional
        Which COCO category to convert.  None = use the first category.
    keypoint_indices : sequence of int, optional
        Subset of keypoint indices to include.  None = all keypoints.
    split : (train, valid, test) floats
        Fraction of images per split.  Must sum to ~1.0.
    class_id : int
        YOLO class ID (typically 0 for single-class pose).
    bbox_source : str
        ``"annotation"`` uses the COCO annotation bbox directly.
        ``"keypoints"`` recomputes the bbox from keypoint coordinates.
    bbox_margin : float
        Margin for keypoint-derived bbox (only used when
        ``bbox_source="keypoints"``).
    symlink_images : bool
        If True, create symlinks to source images.  If False, copy them.
    seed : int
        Random seed for train/valid/test assignment.

    Returns
    -------
    KeypointSchema
        Keypoint schema with names and skeleton from the COCO categories.
    """
    coco_json_path = Path(coco_json_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    # Load and parse COCO JSON
    images_by_id, anns_by_image_id, category = _load_coco(
        coco_json_path, category_name
    )

    # Extract keypoint names and skeleton from category
    all_kp_names = category.get("keypoints", [])
    coco_skeleton = category.get("skeleton", [])

    if keypoint_indices is not None:
        selected_indices = list(keypoint_indices)
        selected_names = [all_kp_names[i] for i in selected_indices]
        # Remap skeleton to selected indices
        index_map = {old: new for new, old in enumerate(selected_indices)}
        skeleton = []
        for edge in coco_skeleton:
            # COCO skeleton uses 0-indexed pairs
            a, b = edge[0], edge[1]
            if a in index_map and b in index_map:
                skeleton.append((index_map[a], index_map[b]))
    else:
        selected_indices = list(range(len(all_kp_names)))
        selected_names = list(all_kp_names)
        # COCO skeleton edges are already 0-indexed pairs
        skeleton = [(a, b) for a, b in coco_skeleton]

    schema = KeypointSchema(names=selected_names, skeleton=skeleton)

    # Find annotated images that have corresponding files on disk
    images_by_filename: dict[str, dict] = {}
    for img in images_by_id.values():
        images_by_filename[img["file_name"]] = img

    usable = []
    for filename, img_record in images_by_filename.items():
        img_path = images_dir / filename
        if not img_path.exists():
            continue
        img_id = img_record["id"]
        if img_id not in anns_by_image_id:
            continue
        usable.append((img_record, img_path, anns_by_image_id[img_id]))

    if not usable:
        print(
            f"[coco_keypoints] WARNING: no usable images found. "
            f"{len(images_by_id)} images in JSON, "
            f"{len(anns_by_image_id)} annotated. "
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
    for img_rec, _, _ in shuffled[:n_train]:
        split_assignment[img_rec["file_name"]] = "train"
    for img_rec, _, _ in shuffled[n_train : n_train + n_valid]:
        split_assignment[img_rec["file_name"]] = "valid"
    for img_rec, _, _ in shuffled[n_train + n_valid :]:
        split_assignment[img_rec["file_name"]] = "test"

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

        # Convert all annotations for this image
        lines = []
        for ann in annotations:
            line = _coco_ann_to_yolo_line(
                ann, img_w, img_h, selected_indices,
                class_id, bbox_source, bbox_margin,
            )
            if line is not None:
                lines.append(line)

        if not lines:
            skipped += 1
            continue

        subset = split_assignment.get(filename, "train")
        stem = Path(filename).stem

        # Write YOLO label
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

    # Remove empty test split if no frames assigned
    test_imgs = output_dir / "test" / "images"
    if test_imgs.exists() and not any(test_imgs.iterdir()):
        shutil.rmtree(output_dir / "test")

    print(
        f"[coco_keypoints] Wrote {written} labels to {output_dir}"
        + (f"  (skipped {skipped} with no valid keypoints)" if skipped else "")
    )
    print(f"  Category: '{category['name']}', keypoints: {len(selected_names)}")
    print(f"  Splits: train={n_train}, valid={n_valid}, test={n - n_train - n_valid}")

    return schema
