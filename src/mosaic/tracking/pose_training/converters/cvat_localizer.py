"""Convert CVAT XML point annotations to localizer patch dataset.

Extracts grayscale patches centered on annotation points (positive samples)
and random background locations (negative samples) for training the localizer
heatmap model.

Uses the same CVAT for Images 1.1 XML format as :mod:`cvat_points`, producing
the same ``patches.npy`` / ``labels.npy`` output as :mod:`coco_localizer`.
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from .base import LocalizerSchema
from .coco_localizer import _extract_patch
from .cvat_points import _parse_cvat_xml


def convert_cvat_localizer(
    cvat_xml_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    *,
    class_attribute: str | None = "class",
    class_names: Sequence[str] | None = None,
    thresholds: Mapping[str, float] | None = None,
    patch_size: int = 128,
    neg_ratio: float = 1.0,
    min_negative_dist: float = 64.0,
    split: tuple[float, float, float] = (0.8, 0.15, 0.05),
    seed: int = 42,
) -> LocalizerSchema:
    """Convert CVAT XML to patch dataset for localizer training.

    For each annotated point, a *patch_size* x *patch_size* grayscale patch
    is extracted centered on the annotation.  Negative (background) patches
    are sampled randomly from locations >= *min_negative_dist* pixels from
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
    cvat_xml_path : path
        Path to CVAT for Images 1.1 XML export file.
    images_dir : path
        Directory containing the source images.  Filenames must match
        the ``name`` attribute in the XML ``<image>`` elements.
    output_dir : path
        Root directory for the output patch dataset.
    class_attribute : str or None
        Name of the ``<attribute>`` element on each ``<points>`` that
        holds the class name (e.g. ``"class"``).  ``None`` = single-class.
    class_names : sequence of str, optional
        Explicit ordered list of class names (index = class ID).
        ``None`` = auto-detect from the XML in insertion order.
    thresholds : dict, optional
        Per-class detection thresholds (by name).  Stored in the manifest
        for later use in inference.
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
    cvat_xml_path = Path(cvat_xml_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    # ── Parse CVAT XML ──
    all_images, seen_classes = _parse_cvat_xml(cvat_xml_path, class_attribute)

    # Resolve class names
    if class_names is not None:
        name_to_id = {name: idx for idx, name in enumerate(class_names)}
    elif seen_classes:
        name_to_id = {name: idx for idx, name in enumerate(seen_classes)}
    else:
        name_to_id = {}

    resolved_names = list(name_to_id.keys())
    num_classes = len(resolved_names) if resolved_names else 1

    # Build thresholds
    thresholds_by_id: dict[int, float] = {}
    if thresholds:
        for name, thresh in thresholds.items():
            if name in name_to_id:
                thresholds_by_id[name_to_id[name]] = thresh

    schema = LocalizerSchema(names=resolved_names, thresholds=thresholds_by_id)

    # ── Filter to annotated images that exist on disk ──
    usable: list[tuple[dict, Path]] = []
    for img_rec in all_images:
        if not img_rec["annotations"]:
            continue
        img_path = images_dir / img_rec["name"]
        if img_path.exists():
            usable.append((img_rec, img_path))

    if not usable:
        n_annotated = sum(1 for r in all_images if r["annotations"])
        print(
            f"[cvat_localizer] WARNING: no usable images found. "
            f"{len(all_images)} images in XML, {n_annotated} annotated. "
            f"Check that images_dir contains matching filenames."
        )
        return schema

    # ── Assign images to splits ──
    rng = random.Random(seed)
    shuffled = list(usable)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * split[0])
    n_valid = int(n * split[1])

    split_map: dict[str, str] = {}
    for rec, _ in shuffled[:n_train]:
        split_map[rec["name"]] = "train"
    for rec, _ in shuffled[n_train : n_train + n_valid]:
        split_map[rec["name"]] = "valid"
    for rec, _ in shuffled[n_train + n_valid :]:
        split_map[rec["name"]] = "test"

    # ── Collect patches per split ──
    half = patch_size // 2
    split_patches: dict[str, list[np.ndarray]] = {
        "train": [], "valid": [], "test": [],
    }
    split_labels: dict[str, list[np.ndarray]] = {
        "train": [], "valid": [], "test": [],
    }
    np_rng = np.random.RandomState(seed)

    for img_rec, img_path in usable:
        subset = split_map[img_rec["name"]]
        img_w = img_rec["width"]
        img_h = img_rec["height"]

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # ---- positive patches ----
        ann_points: list[tuple[float, float, int]] = []
        for x, y, class_name in img_rec["annotations"]:
            if name_to_id and class_name not in name_to_id:
                continue
            class_id = name_to_id[class_name] if name_to_id else 0
            ann_points.append((x, y, class_id))

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

    # ── Save patches and labels per split ──
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

    # ── Write manifest ──
    manifest = {
        "class_names": resolved_names,
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

    print(f"[cvat_localizer] Saved {total_saved} patches to {output_dir}")
    print(f"  Classes: {resolved_names}")
    print(
        f"  Splits: "
        + ", ".join(
            f"{s}={len(split_patches[s])}" for s in ("train", "valid", "test")
        )
    )

    return schema
