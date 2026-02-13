"""YOLO dataset preparation utilities.

Adapted from train-custom-YOLO-Colab/utils/prep.py.  Provides dataset
filtering, label simplification, train/val/test splitting, and data.yaml
generation — with extensions for pose (kpt_shape) and POLO (radii).

Tiling and Colab-specific UI code has been removed.
"""
from __future__ import annotations

import math
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp
from typing import Mapping, Sequence

import yaml


# --------------------------------------------------------------------------- #
# Label helpers
# --------------------------------------------------------------------------- #

def build_collapse_map(allowed_ids: set[int]) -> dict[int, int]:
    """Map original class IDs to contiguous indices (smallest → 0, etc.)."""
    return {old: new for new, old in enumerate(sorted(set(allowed_ids)))}


def filter_labels(pool_dir: str | Path, allowed_ids: set[int]) -> None:
    """Keep only annotations whose class ID is in *allowed_ids*."""
    pool_path = Path(pool_dir)
    lbl_dir = pool_path / "labels"
    if not lbl_dir.exists():
        return

    kept_files = dropped_files = 0
    for lp in lbl_dir.glob("*.txt"):
        lines = [ln.strip() for ln in lp.read_text().splitlines() if ln.strip()]
        new_lines = []
        for ln in lines:
            parts = ln.split()
            if not parts:
                continue
            try:
                cid = int(float(parts[0]))
            except Exception:
                continue
            if cid in allowed_ids:
                new_lines.append(ln)
        if new_lines:
            lp.write_text("\n".join(new_lines) + "\n")
            kept_files += 1
        else:
            lp.write_text("")
            dropped_files += 1
    print(f"[filter_labels] kept={kept_files}, emptied={dropped_files}")


def simplify_labels(
    dataset_root: str | Path,
    collapse_map: Mapping[int, str],
    new_class_ids: Mapping[str, int],
    drop_others: bool = False,
) -> None:
    """Collapse a YOLO dataset taxonomy to coarser groupings."""
    for subset in ("train", "valid", "test"):
        lbl_dir = Path(dataset_root) / subset / "labels"
        if not lbl_dir.is_dir():
            continue
        for path in lbl_dir.glob("*.txt"):
            out_lines: list[str] = []
            with path.open("r") as src:
                for ln in src:
                    if not ln.strip():
                        continue
                    tok0, *rest = ln.split()
                    if not tok0.replace(".", "", 1).isdigit():
                        if not drop_others:
                            out_lines.append(ln.rstrip())
                        continue
                    try:
                        orig_id = int(float(tok0))
                    except ValueError:
                        if not drop_others:
                            out_lines.append(ln.rstrip())
                        continue
                    if orig_id in collapse_map:
                        new_name = collapse_map[orig_id]
                        if new_name not in new_class_ids:
                            if not drop_others:
                                out_lines.append(ln.rstrip())
                            continue
                        new_id = new_class_ids[new_name]
                        out_lines.append(" ".join([str(int(new_id)), *rest]).strip())
                    elif not drop_others:
                        out_lines.append(ln.rstrip())
            path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))


# --------------------------------------------------------------------------- #
# Dataset preparation pipeline
# --------------------------------------------------------------------------- #

def prepare_yolo_dataset(
    dataset_path: str,
    out_dir: str,
    *,
    do_change_labels: bool = True,
    allowed_ids: set[int] | None = None,
    collapse_map: dict[int, str] | None = None,
    new_class_ids: dict[str, int] | None = None,
    drop_others: bool = False,
    prune_empty_fraction: float = 1.0,
    do_rebalance: bool = False,
    split: tuple[float, float, float] = (0.7, 0.2, 0.1),
    remove_test: bool = False,
    seed: int = 0,
    clear_output: bool = True,
) -> None:
    """End-to-end pipeline to prepare a YOLO dataset.

    Parameters
    ----------
    dataset_path : str
        Input dataset root (must contain images/ and labels/ or split dirs).
    out_dir : str
        Destination for the prepared dataset.
    collapse_map : dict, optional
        Mapping from original class id -> group name.
    new_class_ids : dict, optional
        Reverse mapping from group name -> new contiguous id.
    allowed_ids : set, optional
        Class IDs to keep before collapsing.
    prune_empty_fraction : float
        Fraction of empty-label pairs to remove (1.0 = remove all).
    do_rebalance : bool
        If True, redistribute into train/valid/test splits.
    split : tuple
        (train, valid, test) fractions.
    """
    # Canonicalise label-mapping parameters
    collapse_keys = set(collapse_map.keys()) if collapse_map else set()
    effective_allowed_ids: set[int] | None = set(allowed_ids) if allowed_ids is not None else None
    if collapse_keys:
        if effective_allowed_ids is None:
            effective_allowed_ids = set(collapse_keys)
        else:
            missing = collapse_keys - effective_allowed_ids
            if missing:
                print(f"[warn] collapse_map references IDs not in allowed_ids; adding: {sorted(missing)}")
                effective_allowed_ids |= missing
    if effective_allowed_ids is not None:
        effective_allowed_ids = set(sorted(int(x) for x in effective_allowed_ids))

    def _yolo_label_files(labels_dir: str):
        if not os.path.isdir(labels_dir):
            return []
        return [os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith(".txt")]

    def _find_image_for_label(label_path: str, images_dir: str):
        stem = os.path.splitext(os.path.basename(label_path))[0]
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
            candidate = os.path.join(images_dir, stem + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def _filter_labels(pool_dir: str, allowed: set[int]) -> None:
        labels_dir = os.path.join(pool_dir, "labels")
        kept, dropped, emptied = 0, 0, 0
        for lp in _yolo_label_files(labels_dir) or []:
            with open(lp, "r") as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
            new_lines = []
            for ln in lines:
                parts = ln.split()
                try:
                    cid = int(float(parts[0]))
                except Exception:
                    continue
                if cid in allowed:
                    new_lines.append(ln)
                else:
                    dropped += 1
            kept += len(new_lines)
            with open(lp, "w") as fh:
                fh.write("\n".join(new_lines) + ("\n" if new_lines else ""))
            if not new_lines:
                emptied += 1
        print(f"  filter_labels: kept={kept}, dropped={dropped}, emptied_files={emptied}")

    def _simplify_labels(pool_dir: str,
                         collapse_mapping: dict[int, str],
                         new_ids: dict[str, int],
                         drop: bool) -> None:
        labels_dir = os.path.join(pool_dir, "labels")
        remapped, kept_as_is, dropped = 0, 0, 0
        missing_names: set[str] = set()
        for lp in _yolo_label_files(labels_dir) or []:
            with open(lp, "r") as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
            out_lines: list[str] = []
            for ln in lines:
                parts = ln.split()
                try:
                    old_id = int(float(parts[0]))
                except Exception:
                    continue
                if old_id in collapse_mapping:
                    new_name = collapse_mapping[old_id]
                    if new_name not in new_ids:
                        missing_names.add(new_name)
                        if not drop:
                            out_lines.append(ln)
                            kept_as_is += 1
                        else:
                            dropped += 1
                        continue
                    new_id = new_ids[new_name]
                    parts[0] = str(int(new_id))
                    out_lines.append(" ".join(parts))
                    remapped += 1
                else:
                    if drop:
                        dropped += 1
                    else:
                        out_lines.append(ln)
                        kept_as_is += 1
            with open(lp, "w") as fh:
                fh.write("\n".join(out_lines) + ("\n" if out_lines else ""))
        if missing_names:
            print(f"[warn] Missing names in new_class_ids: {sorted(missing_names)}")
        print(f"  simplify_labels: remapped={remapped} kept_as_is={kept_as_is} dropped={dropped}")

    def prune_empty_labels(pool_dir: str, fraction: float = 1.0, seed_: int = 0) -> None:
        rng = random.Random(seed_)
        labels_dir = os.path.join(pool_dir, "labels")
        images_dir = os.path.join(pool_dir, "images")
        empties = []
        for lp in _yolo_label_files(labels_dir) or []:
            try:
                size = os.path.getsize(lp)
            except FileNotFoundError:
                size = 0
            if size == 0:
                empties.append(lp)
        n_drop = int(len(empties) * max(0.0, min(1.0, fraction)))
        rng.shuffle(empties)
        for lp in empties[:n_drop]:
            try:
                os.remove(lp)
            except FileNotFoundError:
                pass
            img = _find_image_for_label(lp, images_dir)
            if img:
                try:
                    os.remove(img)
                except FileNotFoundError:
                    pass
        print(f"  prune_empty_labels: removed {n_drop} empty label/image pairs")

    def rebalance_dataset(
        dataset_root: str,
        output_path: str,
        split_: tuple[float, float, float],
        remove_test_: bool,
        seed_: int,
    ) -> None:
        assert math.isclose(sum(split_), 1.0, abs_tol=1e-6), "split must sum to 1.0"
        rng = random.Random(seed_)
        all_images = []
        for subset in ("train", "valid", "test"):
            img_dir = os.path.join(dataset_root, subset, "images")
            lbl_dir = os.path.join(dataset_root, subset, "labels")
            if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
                continue
            for fname in os.listdir(img_dir):
                stem, ext = os.path.splitext(fname)
                label = os.path.join(lbl_dir, stem + ".txt")
                if os.path.exists(label):
                    all_images.append((os.path.join(img_dir, fname), label))

        if remove_test_:
            split_ = (split_[0], split_[1], 0.0)

        rng.shuffle(all_images)
        total = len(all_images)
        if total == 0:
            raise RuntimeError(f"No paired image/label files found under {dataset_root}")

        n_train = int(total * split_[0])
        n_valid = int(total * split_[1])

        train_pairs = all_images[:n_train]
        valid_pairs = all_images[n_train:n_train + n_valid]
        test_pairs = all_images[n_train + n_valid:]

        def _copy(batch, subset: str):
            img_out = os.path.join(output_path, subset, "images")
            lbl_out = os.path.join(output_path, subset, "labels")
            os.makedirs(img_out, exist_ok=True)
            os.makedirs(lbl_out, exist_ok=True)
            for src_img, src_lbl in batch:
                shutil.copy2(src_img, os.path.join(img_out, os.path.basename(src_img)))
                shutil.copy2(src_lbl, os.path.join(lbl_out, os.path.basename(src_lbl)))

        _copy(train_pairs, "train")
        _copy(valid_pairs, "valid")
        if split_[2] > 0:
            _copy(test_pairs, "test")

        print(
            f"  split counts: train={len(train_pairs)} valid={len(valid_pairs)}"
            + (f" test={len(test_pairs)}" if split_[2] > 0 else "")
        )

    # --- Main pipeline ---
    dataset_abs = os.path.abspath(dataset_path)
    if not os.path.isdir(dataset_abs):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    working_root = mkdtemp(prefix="yolo_prep_")
    shutil.copytree(dataset_abs, working_root, dirs_exist_ok=True)
    working = working_root
    print(f"Working directory: {working}")

    # Merge all splits into a single pool under train/
    merged_images = os.path.join(working, "train", "images")
    merged_labels = os.path.join(working, "train", "labels")
    os.makedirs(merged_images, exist_ok=True)
    os.makedirs(merged_labels, exist_ok=True)

    for subset in ("train", "valid", "test"):
        src_img = os.path.join(working, subset, "images")
        src_lbl = os.path.join(working, subset, "labels")
        if not os.path.isdir(src_img):
            continue
        for f in os.listdir(src_img):
            shutil.move(os.path.join(src_img, f), os.path.join(merged_images, f))
        for f in os.listdir(src_lbl):
            shutil.move(os.path.join(src_lbl, f), os.path.join(merged_labels, f))
        if subset != "train":
            shutil.rmtree(os.path.join(working, subset), ignore_errors=True)

    pool_dir = os.path.join(working, "train")

    # Label filtering and simplification
    if do_change_labels:
        if effective_allowed_ids is not None:
            print("Filtering labels...")
            _filter_labels(pool_dir, effective_allowed_ids)
        if collapse_map and new_class_ids:
            print("Collapsing class taxonomy...")
            _simplify_labels(pool_dir, collapse_map, new_class_ids, drop_others)
    elif any([allowed_ids, collapse_map, new_class_ids]):
        print("[info] do_change_labels=False; ignoring allowed_ids/collapse_map/new_class_ids")

    prune_empty_labels(pool_dir, fraction=prune_empty_fraction, seed_=seed)

    # Output
    if clear_output:
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    if do_rebalance:
        print("Rebalancing into splits...")
        rebalance_dataset(
            dataset_root=os.path.dirname(pool_dir),
            output_path=out_dir,
            split_=split,
            remove_test_=remove_test,
            seed_=seed,
        )
    else:
        print("Exporting single unsplit pool...")
        shutil.copytree(os.path.join(pool_dir, "images"), os.path.join(out_dir, "images"))
        shutil.copytree(os.path.join(pool_dir, "labels"), os.path.join(out_dir, "labels"))

    print(f"Final dataset: {out_dir}")


# --------------------------------------------------------------------------- #
# Dataset inspection
# --------------------------------------------------------------------------- #

def _iter_label_files(root: str, splits: Sequence[str] = ("train", "valid", "test")):
    """Yield all label file paths from split or single-pool layouts."""
    root_path = Path(root)
    found_any = False
    for s in splits:
        lbl_dir = root_path / s / "labels"
        if lbl_dir.is_dir():
            found_any = True
            yield from (str(p) for p in lbl_dir.glob("*.txt"))
    if not found_any:
        single_pool = root_path / "labels"
        if single_pool.is_dir():
            yield from (str(p) for p in single_pool.glob("*.txt"))


def summarize_classes(dataset_root: str):
    """Return (instance_counts, file_counts, total_files, empty_files)."""
    instance_counts: Counter[int] = Counter()
    file_counts: Counter[int] = Counter()
    total_label_files = 0
    empty_files = 0

    for label_path in _iter_label_files(dataset_root):
        total_label_files += 1
        seen_in_file: set[int] = set()
        try:
            with open(label_path, "r") as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
        except Exception:
            continue
        if not lines:
            empty_files += 1
            continue
        for ln in lines:
            parts = ln.split()
            if not parts:
                continue
            try:
                cid = int(float(parts[0]))
            except Exception:
                continue
            instance_counts[cid] += 1
            seen_in_file.add(cid)
        for cid in seen_in_file:
            file_counts[cid] += 1

    return instance_counts, file_counts, total_label_files, empty_files


def auto_select_allowed_ids(
    dataset_root: str,
    *,
    min_instances: int | None = 50,
    min_files: int | None = None,
    ensure_ids: set[int] | None = None,
) -> tuple[set[int], Counter, Counter]:
    """Suggest allowed_ids based on occurrence thresholds."""
    inst, files, total_files, empty = summarize_classes(dataset_root)

    print(f"Dataset: {dataset_root}")
    print(f"Label files: {total_files}  (empty: {empty})")
    if not inst:
        print("No annotations found.")
        return set(), inst, files

    print("\nPer-class summary:")
    print("class_id | instances | files_present_in")
    for cid in sorted(inst):
        print(f"{cid:7d} | {inst[cid]:9d} | {files.get(cid, 0):15d}")

    min_inst_threshold = min_instances if min_instances is not None else 0
    allowed = {
        cid
        for cid in inst
        if inst[cid] >= min_inst_threshold and (min_files is None or files.get(cid, 0) >= min_files)
    }

    if ensure_ids:
        missing = ensure_ids - allowed
        if missing:
            print(f"Including {sorted(missing)} via ensure_ids.")
        allowed |= ensure_ids

    print(f"\nallowed_ids = {sorted(allowed)}")
    return allowed, inst, files


def count_labels(label_dir: str) -> tuple[dict[int, int], int, int]:
    """Count labels per class inside *label_dir*."""
    class_counts: Counter[int] = Counter()
    empty_count = 0
    total_files = 0
    for path in Path(label_dir).glob("*.txt"):
        total_files += 1
        with path.open("r") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        if not lines:
            empty_count += 1
            continue
        for ln in lines:
            parts = ln.split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue
            class_counts[class_id] += 1
    return dict(sorted(class_counts.items())), empty_count, total_files


def check_dataset(out_dir: str, splits: Sequence[str] = ("train", "valid", "test")) -> None:
    """Print a summary of class counts for a prepared dataset."""
    out_path = Path(out_dir).resolve()
    print(f"Dataset: {out_path}")

    found_any_split = False
    for s in splits:
        label_path = out_path / s / "labels"
        if label_path.is_dir():
            found_any_split = True
            counts, empty_count, total_files = count_labels(str(label_path))
            print(f"\n[{s}] labels: {label_path}")
            print(f"  Class counts: {counts}")
            print(f"  Empty: {empty_count} / {total_files}")
        else:
            print(f"[warn] Missing: {label_path}")

    if not found_any_split:
        single_labels = out_path / "labels"
        if single_labels.is_dir():
            print("\n[info] No split folders; single pool:")
            counts, empty_count, total_files = count_labels(str(single_labels))
            print(f"  Class counts: {counts}")
            print(f"  Empty: {empty_count} / {total_files}")
        else:
            print("\nNo labels found.")


# --------------------------------------------------------------------------- #
# YAML helpers
# --------------------------------------------------------------------------- #

def _load_yaml_names(src_yaml: str) -> list[str] | None:
    """Extract names array from a YOLO data.yaml file."""
    if not os.path.exists(src_yaml):
        return None
    with open(src_yaml, "r") as stream:
        data = yaml.safe_load(stream) or {}
    names = data.get("names")
    if names is None:
        return None
    if isinstance(names, dict):
        max_id = max(int(k) for k in names)
        out: list[str | None] = [None] * (max_id + 1)
        for k, v in names.items():
            out[int(k)] = str(v)
        for idx, val in enumerate(out):
            if val is None:
                out[idx] = f"class_{idx}"
        return [str(name) for name in out]
    if isinstance(names, list):
        return [str(n) for n in names]
    return None


def build_new_class_ids_from_yaml(
    src_yaml: str,
    allowed_ids: set[int],
    collapse_map: Mapping[int, int],
) -> dict[str, int]:
    """Create name -> new_id mapping from an existing data.yaml file."""
    sorted_ids = sorted(set(int(a) for a in allowed_ids))
    orig_names = _load_yaml_names(src_yaml)
    new_map: dict[str, int] = {}
    if orig_names is None:
        for old in sorted_ids:
            new_id = collapse_map[old]
            new_map[f"class_{new_id}"] = new_id
        return new_map

    for old in sorted_ids:
        new_id = collapse_map[old]
        cname = orig_names[old] if old < len(orig_names) else f"class_{old}"
        new_map[cname] = new_id
    return new_map


def make_data_yaml(
    dataset_root: str | Path,
    new_class_ids: Mapping[str, int],
    *,
    kpt_shape: list[int] | None = None,
    yaml_name: str = "data.yaml",
    has_test: bool | None = None,
) -> str:
    """Write a YOLO-style data.yaml, optionally with pose kpt_shape.

    Parameters
    ----------
    dataset_root : path
        Root directory of the YOLO dataset.
    new_class_ids : dict
        Mapping from class name -> contiguous integer ID.
    kpt_shape : [num_keypoints, dims], optional
        For pose models, e.g. [27, 3] for 27 keypoints with (x, y, vis).
    has_test : bool, optional
        Whether a test split exists.  Auto-detected if None.
    """
    max_id = max(new_class_ids.values())
    names: list[str | None] = [None] * (max_id + 1)
    for name, idx in new_class_ids.items():
        names[idx] = name
    if any(n is None for n in names):
        raise ValueError("new_class_ids must cover a contiguous 0..N range without gaps.")

    if has_test is None:
        has_test = (Path(dataset_root) / "test" / "images").exists()

    data: dict = {
        "path": os.path.abspath(dataset_root),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(names),
        "names": names,
    }
    if has_test:
        data["test"] = "test/images"
    if kpt_shape is not None:
        data["kpt_shape"] = kpt_shape

    out_path = Path(dataset_root) / yaml_name
    with out_path.open("w") as stream:
        yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)
    return str(out_path)


def make_polo_data_yaml(
    dataset_root: str | Path,
    class_names: Mapping[str, int] | list[str],
    radii: Mapping[int, float],
    *,
    yaml_name: str = "data.yaml",
    has_test: bool | None = None,
) -> str:
    """Write a POLO-style data.yaml with radii instead of kpt_shape.

    Parameters
    ----------
    dataset_root : path
        Root directory of the POLO dataset.
    class_names : dict or list
        If dict: mapping from class name -> contiguous integer ID.
        If list: names in order (index = class ID).
    radii : dict
        Mapping from class_id (int) -> radius (float) in pixels.
    yaml_name : str
        Output filename.
    has_test : bool, optional
        Whether a test split exists.  Auto-detected if None.

    Returns
    -------
    str
        Path to the written YAML file.
    """
    # Build ordered name list
    if isinstance(class_names, dict):
        max_id = max(class_names.values())
        names_list: list[str | None] = [None] * (max_id + 1)
        for name, idx in class_names.items():
            names_list[idx] = name
        if any(n is None for n in names_list):
            raise ValueError("class_names must cover a contiguous 0..N range.")
    else:
        names_list = list(class_names)  # type: ignore[assignment]

    if has_test is None:
        has_test = (Path(dataset_root) / "test" / "images").exists()

    # Validate radii coverage
    for i in range(len(names_list)):
        if i not in radii:
            raise ValueError(f"Missing radius for class {i} ('{names_list[i]}')")

    data: dict = {
        "path": os.path.abspath(dataset_root),
        "train": "train/images",
        "val": "valid/images",
        "names": {i: name for i, name in enumerate(names_list)},
        "radii": {int(k): float(v) for k, v in radii.items()},
    }
    if has_test:
        data["test"] = "test/images"

    out_path = Path(dataset_root) / yaml_name
    with out_path.open("w") as stream:
        yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)
    return str(out_path)
