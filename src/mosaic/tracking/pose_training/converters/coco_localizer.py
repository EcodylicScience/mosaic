"""Convert COCO JSON annotations to localizer patch dataset.

Extracts grayscale patches centered on annotation points (positive samples)
and random background locations (negative samples) for training the localizer
heatmap model.

Supports COCO keypoints (1-node skeleton from CVAT) or bbox centers.
"""
from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .base import LocalizerSchema


def _load_coco(
    coco_json_path: Path,
    category_names: list[str] | None,
) -> tuple[dict[int, dict], dict[int, list[dict]], list[dict], dict[int, int]]:
    """Load COCO JSON and return lookup structures."""
    with open(coco_json_path) as f:
        coco = json.load(f)

    all_categories = coco.get("categories", [])
    if not all_categories:
        raise ValueError("COCO JSON has no categories")

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

    cat_id_to_class_id = {cat["id"]: idx for idx, cat in enumerate(selected)}
    selected_cat_ids = set(cat_id_to_class_id)

    images_by_id = {img["id"]: img for img in coco.get("images", [])}

    anns_by_image_id: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("category_id") in selected_cat_ids:
            anns_by_image_id[ann["image_id"]].append(ann)

    return images_by_id, dict(anns_by_image_id), selected, cat_id_to_class_id


def _extract_point_px(
    ann: dict[str, Any],
    point_source: str,
) -> tuple[float, float] | None:
    """Extract point coordinates in pixels from a COCO annotation."""
    if point_source == "keypoint":
        raw_kps = ann.get("keypoints", [])
        n_kps = len(raw_kps) // 3
        for i in range(n_kps):
            x, y, vis = raw_kps[i * 3], raw_kps[i * 3 + 1], raw_kps[i * 3 + 2]
            if vis > 0:
                return (float(x), float(y))
        return None

    elif point_source == "bbox_center":
        bbox = ann.get("bbox")
        if bbox is None or len(bbox) < 4:
            return None
        bx, by, bw, bh = bbox
        return (float(bx + bw / 2.0), float(by + bh / 2.0))

    else:
        raise ValueError(f"Unknown point_source: {point_source!r}")


def _extract_patch(
    image: np.ndarray,
    cx: float,
    cy: float,
    patch_size: int,
) -> np.ndarray:
    """Extract a grayscale patch centered at (cx, cy), with border padding.

    Returns
    -------
    ndarray, shape (patch_size, patch_size), dtype float32 in [0, 1]
    """
    h, w = image.shape[:2]
    half = patch_size // 2

    ix, iy = int(round(cx)), int(round(cy))

    # Source region in image coords
    x0, y0 = ix - half, iy - half
    x1, y1 = x0 + patch_size, y0 + patch_size

    # Clamp to image bounds
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)

    # Destination region in patch coords
    dx0, dy0 = sx0 - x0, sy0 - y0
    dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)

    patch = np.zeros((patch_size, patch_size), dtype=np.float32)

    if sx1 > sx0 and sy1 > sy0:
        region = image[sy0:sy1, sx0:sx1]
        if region.ndim == 3:
            region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        patch[dy0:dy1, dx0:dx1] = region.astype(np.float32) / 255.0

    return patch


def convert_coco_localizer(
    coco_json_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    *,
    category_names: list[str] | None = None,
    thresholds: Mapping[str, float] | None = None,
    point_source: str = "keypoint",
    patch_size: int = 128,
    neg_ratio: float = 1.0,
    min_negative_dist: float = 64.0,
    split: tuple[float, float, float] = (0.8, 0.15, 0.05),
    seed: int = 42,
) -> LocalizerSchema:
    """Convert COCO JSON to patch dataset for localizer training.

    For each annotated point, a *patch_size* × *patch_size* grayscale patch
    is extracted centered on the annotation.  Negative (background) patches
    are sampled randomly from locations ≥ *min_negative_dist* pixels from
    any annotation.

    Output structure::

        output_dir/
            manifest.json
            train/
                patches.npy   (N, patch_size, patch_size) float32 [0, 1]
                labels.npy    (N, num_classes) float32 binary
            valid/
                patches.npy
                labels.npy

    Parameters
    ----------
    coco_json_path : path
        Path to COCO JSON file (e.g. CVAT COCO Keypoints 1.0 export).
    images_dir : path
        Directory containing the source images.
    output_dir : path
        Root directory for the output patch dataset.
    category_names : list of str, optional
        Which COCO categories to include.  ``None`` = all categories.
    thresholds : dict, optional
        Per-class detection thresholds (by name).  Stored in the manifest
        for later use in inference.
    point_source : str
        ``"keypoint"`` — extract from COCO keypoints (1-node skeleton).
        ``"bbox_center"`` — compute center of annotation bounding box.
    patch_size : int
        Size of extracted patches (square, in pixels).
    neg_ratio : float
        Number of negative patches per positive patch.
    min_negative_dist : float
        Minimum pixel distance from any annotation for negative samples.
    split : (train, valid, test) floats
        Fraction of *images* per split.  Must sum to ~1.0.
    seed : int
        Random seed for reproducible splits and sampling.

    Returns
    -------
    LocalizerSchema
        Schema with class names and optional thresholds.
    """
    coco_json_path = Path(coco_json_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    images_by_id, anns_by_image_id, categories, cat_id_to_class_id = _load_coco(
        coco_json_path, category_names
    )

    class_names = [cat["name"] for cat in categories]
    num_classes = len(class_names)

    thresholds_by_id: dict[int, float] = {}
    if thresholds:
        for cat in categories:
            if cat["name"] in thresholds:
                thresholds_by_id[cat_id_to_class_id[cat["id"]]] = thresholds[cat["name"]]

    schema = LocalizerSchema(names=class_names, thresholds=thresholds_by_id)

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
            f"[coco_localizer] WARNING: no usable images found. "
            f"{len(images_by_id)} images in JSON, "
            f"{len(anns_by_image_id)} annotated. "
            f"Check that images_dir contains matching filenames."
        )
        return schema

    # Assign images to splits
    rng = random.Random(seed)
    shuffled = list(usable)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * split[0])
    n_valid = int(n * split[1])

    split_map: dict[int, str] = {}
    for img_rec, _, _ in shuffled[:n_train]:
        split_map[img_rec["id"]] = "train"
    for img_rec, _, _ in shuffled[n_train : n_train + n_valid]:
        split_map[img_rec["id"]] = "valid"
    for img_rec, _, _ in shuffled[n_train + n_valid :]:
        split_map[img_rec["id"]] = "test"

    # Collect patches per split
    half = patch_size // 2
    split_patches: dict[str, list[np.ndarray]] = {
        "train": [], "valid": [], "test": [],
    }
    split_labels: dict[str, list[np.ndarray]] = {
        "train": [], "valid": [], "test": [],
    }
    np_rng = np.random.RandomState(seed)

    for img_record, img_path, annotations in usable:
        img_id = img_record["id"]
        subset = split_map[img_id]
        img_w = int(img_record["width"])
        img_h = int(img_record["height"])

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # ---- positive patches ----
        ann_points: list[tuple[float, float, int]] = []
        for ann in annotations:
            coco_cat_id = ann.get("category_id")
            if coco_cat_id not in cat_id_to_class_id:
                continue
            class_id = cat_id_to_class_id[coco_cat_id]
            pt = _extract_point_px(ann, point_source)
            if pt is None:
                continue
            ann_points.append((pt[0], pt[1], class_id))

        for x_px, y_px, class_id in ann_points:
            patch = _extract_patch(image, x_px, y_px, patch_size)
            label = np.zeros(num_classes, dtype=np.float32)
            label[class_id] = 1.0
            split_patches[subset].append(patch)
            split_labels[subset].append(label)

        # ---- negative patches ----
        n_neg = max(1, int(len(ann_points) * neg_ratio))
        ann_coords = (
            np.array([(x, y) for x, y, _ in ann_points])
            if ann_points
            else np.empty((0, 2))
        )

        neg_sampled = 0
        max_attempts = n_neg * 20
        attempts = 0
        while neg_sampled < n_neg and attempts < max_attempts:
            nx = np_rng.uniform(half, max(half + 1, img_w - half))
            ny = np_rng.uniform(half, max(half + 1, img_h - half))

            if len(ann_coords) > 0:
                dists = np.sqrt(((ann_coords - [nx, ny]) ** 2).sum(axis=1))
                if dists.min() < min_negative_dist:
                    attempts += 1
                    continue

            patch = _extract_patch(image, nx, ny, patch_size)
            label = np.zeros(num_classes, dtype=np.float32)
            split_patches[subset].append(patch)
            split_labels[subset].append(label)
            neg_sampled += 1
            attempts += 1

    # Save patches and labels per split
    output_dir.mkdir(parents=True, exist_ok=True)
    total_saved = 0

    for subset in ("train", "valid", "test"):
        patches = split_patches[subset]
        labels = split_labels[subset]
        if not patches:
            continue

        subset_dir = output_dir / subset
        subset_dir.mkdir(parents=True, exist_ok=True)

        patches_arr = np.stack(patches)
        labels_arr = np.stack(labels)

        # Shuffle within split
        idx = np.arange(len(patches_arr))
        np_rng.shuffle(idx)
        patches_arr = patches_arr[idx]
        labels_arr = labels_arr[idx]

        np.save(str(subset_dir / "patches.npy"), patches_arr)
        np.save(str(subset_dir / "labels.npy"), labels_arr)
        total_saved += len(patches_arr)

    # Write manifest
    manifest = {
        "class_names": class_names,
        "num_classes": num_classes,
        "patch_size": patch_size,
        "neg_ratio": neg_ratio,
        "min_negative_dist": min_negative_dist,
        "thresholds": {str(k): v for k, v in thresholds_by_id.items()},
        "splits": {s: len(split_patches[s]) for s in ("train", "valid", "test")},
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Remove empty test split directory
    test_dir = output_dir / "test"
    if test_dir.exists() and not (test_dir / "patches.npy").exists():
        shutil.rmtree(test_dir)

    print(f"[coco_localizer] Saved {total_saved} patches to {output_dir}")
    print(f"  Classes: {class_names}")
    print(
        f"  Splits: "
        + ", ".join(
            f"{s}={len(split_patches[s])}" for s in ("train", "valid", "test")
        )
    )

    return schema
