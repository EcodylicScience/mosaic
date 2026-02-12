"""Overlay preparation and frame drawing.

This module contains functions for building and rendering overlays:
- prepare_overlay: Precompute per-frame overlay structures
- draw_frame: Render pose, bbox, labels onto a single frame
"""
from __future__ import annotations
from typing import Dict, Tuple, Any, Optional, Iterable
import pandas as pd
import numpy as np
import cv2

from .helpers import (
    _pose_column_pairs,
    _extract_pose_points,
    _compute_bbox,
    _extract_centroid,
    _color_for_id,
    _color_for_label,
    _lookup_label_series,
    _scalar_from_series,
    _format_label_text,
)


def _pick_gt_entry(entries: list[dict]) -> Optional[dict]:
    """
    Choose one GT entry from possibly multiple candidates.

    Preference: highest numeric label_id; otherwise first non-null entry.
    """
    if not entries:
        return None
    scored: list[tuple[float, dict]] = []
    for ent in entries:
        lbl = ent.get("label_id")
        score = -1.0
        try:
            if lbl is not None and not pd.isna(lbl):
                score = float(lbl)
        except Exception:
            score = -1.0
        scored.append((score, ent))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _build_gt_maps(gt_df: pd.DataFrame) -> tuple[dict[int, dict], dict[int, dict[tuple[int, int], dict]]]:
    """
    Build GT lookup maps:
      - global_map[frame] -> {"label_id", "label_name"}
      - pair_map[frame][(id_lo,id_hi)] -> {"label_id", "label_name", "id1", "id2"}
    """
    global_map: dict[int, dict] = {}
    pair_map: dict[int, dict[tuple[int, int], dict]] = {}
    if gt_df is None or gt_df.empty or "frame" not in gt_df.columns:
        return global_map, pair_map

    df = gt_df.copy()
    if "label_name" not in df.columns and "label_id" in df.columns:
        df["label_name"] = df["label_id"]

    has_pairs = {"id1", "id2"}.issubset(df.columns)
    for _, row in df.iterrows():
        try:
            frame = int(row["frame"])
        except Exception:
            continue

        entry = {
            "label_id": row.get("label_id"),
            "label_name": row.get("label_name"),
        }

        if has_pairs:
            id1 = row.get("id1")
            id2 = row.get("id2")
            if id1 is not None and id2 is not None and not pd.isna(id1) and not pd.isna(id2):
                try:
                    a = int(id1)
                    b = int(id2)
                except Exception:
                    continue
                pair = tuple(sorted((a, b)))
                frame_pairs = pair_map.setdefault(frame, {})
                prev = frame_pairs.get(pair)
                frame_pairs[pair] = _pick_gt_entry([prev, {**entry, "id1": a, "id2": b}] if prev else [{**entry, "id1": a, "id2": b}])
                continue

        prev_g = global_map.get(frame)
        global_map[frame] = _pick_gt_entry([prev_g, entry] if prev_g else [entry])

    return global_map, pair_map


def _collect_gt_for_id(frame_pair_map: dict[tuple[int, int], dict], id_val: Any) -> Optional[dict]:
    """
    Resolve GT for one individual at a frame from pair-aware GT rows.
    """
    try:
        id_norm = int(id_val)
    except Exception:
        id_norm = id_val
    matches = []
    for pair, ent in frame_pair_map.items():
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            continue
        if id_norm in pair:
            matches.append(ent)
    return _pick_gt_entry(matches)


def _resolve_id_key(ids_map: dict, raw_id: Any) -> Optional[Any]:
    """
    Resolve an ID key in ids_map with tolerant numeric matching.
    """
    if raw_id in ids_map:
        return raw_id
    try:
        tgt = int(raw_id)
    except Exception:
        tgt = raw_id
    for key in ids_map.keys():
        try:
            if int(key) == tgt:
                return key
        except Exception:
            if key == raw_id:
                return key
    return None


def _collect_pair_label_for_id(per_pair_map: dict, id_val: Any, frame_int: int) -> Any:
    """
    Aggregate pair labels touching id_val at a frame into one display value.
    """
    try:
        id_norm = int(id_val)
    except Exception:
        id_norm = id_val

    vals = []
    for pair, series in per_pair_map.items():
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            continue
        try:
            p0 = int(pair[0])
            p1 = int(pair[1])
        except Exception:
            p0, p1 = pair[0], pair[1]
        if id_norm not in (p0, p1):
            continue
        val = _scalar_from_series(series.get(frame_int))
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        vals.append(val)

    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]

    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
        return max(vals)

    deduped = list(dict.fromkeys(vals))
    return deduped[0] if len(deduped) == 1 else tuple(deduped)


def _coerce_color(value: Any, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return tuple(int(v) for v in value)
        except Exception:
            return fallback
    return fallback


def _anchor_for_id_info(info: dict[str, Any]) -> Optional[tuple[float, float]]:
    bbox = info.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(np.isfinite(bbox)):
        x1, y1, x2, y2 = bbox
        return (float((x1 + x2) * 0.5), float((y1 + y2) * 0.5))
    c = info.get("centroid")
    if isinstance(c, (list, tuple)) and len(c) == 2 and all(np.isfinite(c)):
        return (float(c[0]), float(c[1]))
    return None


def prepare_overlay(tracks_df: pd.DataFrame,
                    labels: dict,
                    gt_df: Optional[pd.DataFrame] = None,
                    kinds: Iterable[str] = ("pose", "bbox"),
                    color_by: Optional[str] = None,
                    hide_unlabeled: bool = False) -> dict:
    """
    Precompute lightweight per-frame overlay structures (pose keypoints, bounding boxes, labels).

    Parameters
    ----------
    tracks_df : DataFrame
        Output of load_tracks_and_labels()[0].
    labels : dict
        Output of load_tracks_and_labels()[1].
    gt_df : DataFrame, optional
        Output of load_ground_truth_labels (used as global per-frame labels).
    kinds : Iterable[str]
        Overlay primitives to compute ("pose", "bbox").

    Returns
    -------
    dict with keys:
        frames: sorted list of frame numbers
        per_frame: {frame -> {"ids": {id -> info}, "global_labels": {...}}}
        id_colors: {id -> (B,G,R)}
    """
    if tracks_df.empty:
        raise ValueError("tracks_df is empty; cannot build overlay.")
    kinds = tuple(kinds)
    pose_pairs = _pose_column_pairs(tracks_df.columns)

    # Precompute label sources for quick lookup
    per_id_labels = labels.get("per_id", {})
    per_pair_labels = labels.get("per_pair", {})

    gt_global_map, gt_pair_map = _build_gt_maps(gt_df if gt_df is not None else pd.DataFrame())

    per_frame: dict[int, dict[str, Any]] = {}
    id_colors: dict[Any, Tuple[int, int, int]] = {}
    label_colors: dict[str, Tuple[int, int, int]] = {}
    color_mode = (color_by or "").strip().lower()
    color_feature = None
    if color_mode and color_mode != "gt":
        feature_names = list(dict.fromkeys([*per_id_labels.keys(), *per_pair_labels.keys()]))
        for feat in feature_names:
            if feat.lower() == color_mode:
                color_feature = feat
                break

    centroid_cols = [("X#wcentroid", "Y#wcentroid"), ("X", "Y")]

    grouped = tracks_df.groupby("frame", sort=True)
    for frame_val, frame_df in grouped:
        frame_int = int(frame_val)
        id_infos: dict[Any, dict[str, Any]] = {}
        global_labels = gt_global_map.get(frame_int, {})
        frame_pair_labels: dict[str, dict[tuple[int, int], Any]] = {}

        # Pair labels from model/feature outputs
        for feat_name, per_pair_map in per_pair_labels.items():
            frame_pairs: dict[tuple[int, int], Any] = {}
            for pair, series in per_pair_map.items():
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    continue
                val = _scalar_from_series(series.get(frame_int))
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                try:
                    a = int(pair[0])
                    b = int(pair[1])
                except Exception:
                    continue
                frame_pairs[tuple(sorted((a, b)))] = val
            if frame_pairs:
                frame_pair_labels[feat_name] = frame_pairs

        # Pair labels from GT rows
        gt_pairs_here = gt_pair_map.get(frame_int, {})
        if gt_pairs_here:
            gt_pairs_out: dict[tuple[int, int], Any] = {}
            for pair, ent in gt_pairs_here.items():
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    continue
                val = ent.get("label_name") or ent.get("label_id")
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                try:
                    a = int(pair[0])
                    b = int(pair[1])
                except Exception:
                    continue
                gt_pairs_out[tuple(sorted((a, b)))] = val
            if gt_pairs_out:
                frame_pair_labels["gt"] = gt_pairs_out

        frame_color = None
        if color_mode == "gt" and global_labels:
            label_val = global_labels.get("label_name") or global_labels.get("label_id")
            if label_val is not None:
                color_key = f"gt:{label_val}"
                if color_key not in label_colors:
                    label_colors[color_key] = _color_for_label(label_val)
                frame_color = label_colors[color_key]
        for _, row in frame_df.iterrows():
            id_val = row.get("id")
            if pd.isna(id_val):
                continue
            info: dict[str, Any] = {}
            centroid = _extract_centroid(row, centroid_cols)
            if "pose" in kinds and pose_pairs:
                pose_pts = _extract_pose_points(row, pose_pairs)
                if pose_pts:
                    info["pose"] = pose_pts
            if "bbox" in kinds and info.get("pose"):
                info["bbox"] = _compute_bbox(info["pose"])
            if centroid:
                info["centroid"] = centroid

            labels_for_id: dict[str, Any] = {}
            if color_mode == "gt":
                gt_entry = _collect_gt_for_id(gt_pair_map.get(frame_int, {}), id_val)
                if gt_entry is None and global_labels:
                    gt_entry = global_labels
                if gt_entry is not None:
                    gt_val = gt_entry.get("label_name") or gt_entry.get("label_id")
                    if gt_val is not None and not (isinstance(gt_val, float) and np.isnan(gt_val)):
                        labels_for_id["gt"] = gt_val
            feature_names = list(dict.fromkeys([*per_id_labels.keys(), *per_pair_labels.keys()]))
            for feat_name in feature_names:
                per_id_map = per_id_labels.get(feat_name, {})
                series = _lookup_label_series(per_id_map, id_val) if per_id_map else None
                if series is not None:
                    val = _scalar_from_series(series.get(frame_int))
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        labels_for_id[feat_name] = val
                        continue
                per_pair_map = per_pair_labels.get(feat_name, {})
                if per_pair_map:
                    pair_val = _collect_pair_label_for_id(per_pair_map, id_val, frame_int)
                    if pair_val is not None:
                        labels_for_id[feat_name] = pair_val
            if labels_for_id:
                info["labels"] = labels_for_id

            if hide_unlabeled and not labels_for_id:
                continue
            if not info:
                continue
            color = None
            if color_feature:
                label_val = labels_for_id.get(color_feature)
                if label_val is not None:
                    color_key = f"{color_feature}:{label_val}"
                    if color_key not in label_colors:
                        label_colors[color_key] = _color_for_label(label_val)
                    color = label_colors[color_key]
            elif color_mode == "gt":
                gt_val = labels_for_id.get("gt")
                if gt_val is not None:
                    color_key = f"gt:{gt_val}"
                    if color_key not in label_colors:
                        label_colors[color_key] = _color_for_label(gt_val)
                    color = label_colors[color_key]
                elif frame_color is not None:
                    color = frame_color
            if color is None:
                color = id_colors.get(id_val)
                if color is None:
                    color = _color_for_id(id_val)
                    id_colors[id_val] = color
            info["color"] = color
            id_infos[id_val] = info

        if not id_infos and not global_labels and not frame_pair_labels:
            continue
        per_frame[frame_int] = {
            "ids": id_infos,
            "global_labels": global_labels,
            "frame_color": frame_color,
            "pair_labels": frame_pair_labels,
        }

    frames = sorted(per_frame.keys())
    return {
        "frames": frames,
        "per_frame": per_frame,
        "id_colors": id_colors,
        "color_mode": color_mode,
        "color_feature": color_feature,
    }


def draw_frame(image: np.ndarray,
               frame_overlay: dict,
               id_colors: dict,
               show_labels: bool = True,
               point_radius: int = 8,
               bbox_thickness: int = 2,
               show_individual_bboxes: bool = True,
               scale: Tuple[float, float] = (1.0, 1.0),
               color_feature: Optional[str] = None,
               color_mode: Optional[str] = None,
               pair_box_feature: Optional[str] = None,
               pair_box_behaviors: Optional[Iterable[Any]] = None,
               hide_individual_bboxes_for_pair: bool = False) -> np.ndarray:
    """
    Draw pose points, bounding boxes, and labels for a single frame.

    Parameters
    ----------
    image : np.ndarray (H,W,3)
        Video frame in BGR order.
    frame_overlay : dict
        Entry from overlay_data["per_frame"][frame].
    id_colors : dict
        Mapping produced by prepare_overlay.
    """
    canvas = image.copy()
    sx, sy = scale
    ids = frame_overlay.get("ids", {})
    frame_color = frame_overlay.get("frame_color")
    render_layers = frame_overlay.get("render_layers") or {}

    # Layer-driven style overrides applied before drawing.
    for style in render_layers.get("id_styles", []) or []:
        if not isinstance(style, dict):
            continue
        raw_id = style.get("id")
        key = _resolve_id_key(ids, raw_id)
        if key is None:
            continue
        info = ids.get(key) or {}
        if "color" in style:
            info["color"] = _coerce_color(style.get("color"), info.get("color") or id_colors.get(key, (0, 255, 0)))
        lbl = style.get("label")
        if lbl is not None:
            label_key = style.get("label_key", "overlay")
            labels_map = info.setdefault("labels", {})
            labels_map[str(label_key)] = lbl
        ids[key] = info

    pair_boxes = []
    ids_in_pair_boxes = set()
    pair_labels_all = frame_overlay.get("pair_labels") or {}
    behavior_set = {str(v).strip().lower() for v in (pair_box_behaviors or []) if str(v).strip()}

    selected_pair_feature = pair_box_feature
    if selected_pair_feature is None and color_mode == "gt" and "gt" in pair_labels_all:
        selected_pair_feature = "gt"
    if selected_pair_feature:
        for feat_name in pair_labels_all.keys():
            if str(feat_name).strip().lower() == str(selected_pair_feature).strip().lower():
                selected_pair_feature = feat_name
                break

    if selected_pair_feature and behavior_set:
        pair_map = pair_labels_all.get(selected_pair_feature, {})
        grouped = {}

        def _canon_pair(a, b):
            try:
                return tuple(sorted((a, b)))
            except TypeError:
                return tuple(sorted((a, b), key=lambda v: str(v)))

        for pair, val in pair_map.items():
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                continue
            lbl_norm = str(val).strip().lower()
            if lbl_norm not in behavior_set:
                continue

            key_a = _resolve_id_key(ids, pair[0])
            key_b = _resolve_id_key(ids, pair[1])
            if key_a is None or key_b is None:
                continue
            info_a = ids.get(key_a) or {}
            info_b = ids.get(key_b) or {}
            if "bbox" not in info_a or "bbox" not in info_b:
                continue
            xa1, ya1, xa2, ya2 = info_a["bbox"]
            xb1, yb1, xb2, yb2 = info_b["bbox"]
            if not all(np.isfinite([xa1, ya1, xa2, ya2, xb1, yb1, xb2, yb2])):
                continue

            union = (
                min(float(xa1), float(xb1)),
                min(float(ya1), float(yb1)),
                max(float(xa2), float(xb2)),
                max(float(ya2), float(yb2)),
            )

            src = key_a
            dst = key_b
            canon = _canon_pair(src, dst)
            grouped.setdefault(canon, []).append({
                "src": src,
                "dst": dst,
                "label": val,
                "label_norm": lbl_norm,
                "bbox": union,
                "color": _color_for_label(val),
            })
            ids_in_pair_boxes.update({key_a, key_b})

        for _, entries in grouped.items():
            # Remove exact duplicates (same direction + same normalized label).
            unique_entries = {}
            for e in entries:
                k = (e["src"], e["dst"], e["label_norm"])
                if k not in unique_entries:
                    unique_entries[k] = e
            collapsed = list(unique_entries.values())
            if not collapsed:
                continue

            # If both directions carry the same behavior label, draw only once.
            label_norms = {e["label_norm"] for e in collapsed}
            if len(label_norms) == 1:
                e = collapsed[0]
                pair_boxes.append({
                    "pair": (e["src"], e["dst"]),
                    "label": e["label"],
                    "bbox": e["bbox"],
                    "color": e["color"],
                    "offset_px": 0,
                })
                continue

            # Asymmetric case: keep directional entries and offset them for visibility.
            collapsed.sort(key=lambda e: (str(e["src"]), str(e["dst"]), str(e["label"])))
            for idx, e in enumerate(collapsed):
                pair_boxes.append({
                    "pair": (e["src"], e["dst"]),
                    "label": f"{e['src']}->{e['dst']}:{_format_label_text(e['label'])}",
                    "bbox": e["bbox"],
                    "color": e["color"],
                    "offset_px": idx * 3,
                })

    for id_val, info in ids.items():
        base_color = info.get("color")
        if base_color is None:
            base_color = id_colors.get(id_val, (0, 255, 0))
        color = tuple(int(c) for c in base_color)
        if show_individual_bboxes and "bbox" in info:
            x1, y1, x2, y2 = info["bbox"]
            if all(np.isfinite([x1, y1, x2, y2])) and not (
                hide_individual_bboxes_for_pair and id_val in ids_in_pair_boxes
            ):
                pt1 = (int(x1 * sx), int(y1 * sy))
                pt2 = (int(x2 * sx), int(y2 * sy))
                cv2.rectangle(canvas, pt1, pt2, color, bbox_thickness)
        if "pose" in info:
            for x, y in info["pose"]:
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                pt = (int(x * sx), int(y * sy))
                cv2.circle(canvas, pt, point_radius, color, -1, lineType=cv2.LINE_AA)
        if show_labels and (info.get("labels") or color_mode == "gt"):
            labels_map = info.get("labels") or {}
            dominant = None
            if color_feature and color_feature in labels_map:
                dominant = labels_map[color_feature]
            elif color_mode == "gt":
                dominant = labels_map.get("gt")
                if dominant is None:
                    global_label = frame_overlay.get("global_labels", {})
                    dominant = global_label.get("label_name") or global_label.get("label_id")
            label_text = None
            if dominant is not None:
                label_text = _format_label_text(dominant)
            elif labels_map:
                label_text = " | ".join(f"{k}:{_format_label_text(v)}" for k, v in labels_map.items())
            if not label_text:
                continue
            anchor = None
            if "bbox" in info:
                x1, y1, _, _ = info["bbox"]
                anchor = (x1, y1)
            if anchor is None:
                anchor = info.get("centroid")
            if anchor and all(np.isfinite(anchor)):
                pos = (int(anchor[0] * sx), int(anchor[1] * sy) - 4)
                cv2.putText(canvas, str(label_text), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Draw pair-level boxes after per-id overlays so they stay visible.
    for pb in pair_boxes:
        x1, y1, x2, y2 = pb["bbox"]
        color = tuple(int(c) for c in pb["color"])
        off = int(pb.get("offset_px", 0))
        pt1 = (int(x1 * sx) - off, int(y1 * sy) - off)
        pt2 = (int(x2 * sx) + off, int(y2 * sy) + off)
        cv2.rectangle(canvas, pt1, pt2, color, max(1, bbox_thickness + 1))
        if show_labels:
            lbl = _format_label_text(pb["label"])
            if lbl:
                pos = (pt1[0], max(12, pt1[1] - 6))
                cv2.putText(canvas, str(lbl), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # Draw group-level outlines (union over member ids).
    for item in render_layers.get("group_outlines", []) or []:
        if not isinstance(item, dict):
            continue
        members = item.get("ids") or []
        rects = []
        for rid in members:
            key = _resolve_id_key(ids, rid)
            if key is None:
                continue
            info = ids.get(key) or {}
            bbox = info.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(np.isfinite(bbox)):
                rects.append(tuple(float(v) for v in bbox))
                continue
            anchor = _anchor_for_id_info(info)
            if anchor is not None:
                x, y = anchor
                r = float(item.get("fallback_radius", 20.0))
                rects.append((x - r, y - r, x + r, y + r))
        if not rects:
            continue
        x1 = min(r[0] for r in rects)
        y1 = min(r[1] for r in rects)
        x2 = max(r[2] for r in rects)
        y2 = max(r[3] for r in rects)

        color = _coerce_color(item.get("color"), _color_for_label(item.get("group_size", len(members))))
        thickness = int(item.get("thickness", max(1, bbox_thickness + 1)))
        pt1 = (int(x1 * sx), int(y1 * sy))
        pt2 = (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(canvas, pt1, pt2, color, thickness)
        if show_labels:
            lbl = _format_label_text(item.get("label"))
            if lbl:
                pos = (pt1[0], max(12, pt1[1] - 6))
                cv2.putText(canvas, str(lbl), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # Draw optional line edges between ids.
    for edge in render_layers.get("edges", []) or []:
        if not isinstance(edge, dict):
            continue
        p1 = None
        p2 = None
        if "p1" in edge and "p2" in edge:
            p1 = edge.get("p1")
            p2 = edge.get("p2")
        else:
            id_a = edge.get("id1")
            id_b = edge.get("id2")
            key_a = _resolve_id_key(ids, id_a)
            key_b = _resolve_id_key(ids, id_b)
            if key_a is not None and key_b is not None:
                p1 = _anchor_for_id_info(ids.get(key_a) or {})
                p2 = _anchor_for_id_info(ids.get(key_b) or {})
        if p1 is None or p2 is None:
            continue
        if not all(np.isfinite([p1[0], p1[1], p2[0], p2[1]])):
            continue
        color = _coerce_color(edge.get("color"), (255, 255, 255))
        thickness = int(edge.get("thickness", 1))
        cv2.line(
            canvas,
            (int(p1[0] * sx), int(p1[1] * sy)),
            (int(p2[0] * sx), int(p2[1] * sy)),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    # Draw optional vectors (velocity arrows or any directional primitive).
    for vec in render_layers.get("vectors", []) or []:
        if not isinstance(vec, dict):
            continue
        origin = vec.get("origin")
        if origin is None and "id" in vec:
            key = _resolve_id_key(ids, vec.get("id"))
            if key is not None:
                origin = _anchor_for_id_info(ids.get(key) or {})
        if origin is None:
            continue
        dx = float(vec.get("dx", 0.0))
        dy = float(vec.get("dy", 0.0))
        scale_v = float(vec.get("scale", 1.0))
        ox, oy = float(origin[0]), float(origin[1])
        ex, ey = ox + dx * scale_v, oy + dy * scale_v
        if not all(np.isfinite([ox, oy, ex, ey])):
            continue
        color = _coerce_color(vec.get("color"), (255, 255, 255))
        thickness = int(vec.get("thickness", 1))
        tip_len = float(vec.get("tip_length", 0.2))
        cv2.arrowedLine(
            canvas,
            (int(ox * sx), int(oy * sy)),
            (int(ex * sx), int(ey * sy)),
            color,
            thickness,
            line_type=cv2.LINE_AA,
            tipLength=tip_len,
        )

    # Draw optional polygons (e.g., ROIs).
    for poly in render_layers.get("polygons", []) or []:
        if not isinstance(poly, dict):
            continue
        pts = poly.get("points") or []
        if len(pts) < 2:
            continue
        out_pts = []
        for p in pts:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                continue
            if not all(np.isfinite(p)):
                continue
            out_pts.append([int(float(p[0]) * sx), int(float(p[1]) * sy)])
        if len(out_pts) < 2:
            continue
        arr = np.asarray(out_pts, dtype=np.int32).reshape((-1, 1, 2))
        color = _coerce_color(poly.get("color"), (255, 255, 255))
        thickness = int(poly.get("thickness", 1))
        if poly.get("fill", False):
            alpha = float(poly.get("alpha", 0.2))
            tmp = canvas.copy()
            cv2.fillPoly(tmp, [arr], color)
            canvas = cv2.addWeighted(tmp, alpha, canvas, 1.0 - alpha, 0.0)
        cv2.polylines(canvas, [arr], bool(poly.get("closed", True)), color, thickness, lineType=cv2.LINE_AA)

    # Draw optional free text labels.
    for txt in render_layers.get("texts", []) or []:
        if not isinstance(txt, dict):
            continue
        pos = txt.get("pos")
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            continue
        if not all(np.isfinite(pos)):
            continue
        text = txt.get("text")
        if text is None:
            continue
        color = _coerce_color(txt.get("color"), (255, 255, 255))
        scale_txt = float(txt.get("font_scale", 0.5))
        thickness = int(txt.get("thickness", 1))
        cv2.putText(
            canvas,
            str(text),
            (int(float(pos[0]) * sx), int(float(pos[1]) * sy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale_txt,
            color,
            thickness,
            cv2.LINE_AA,
        )

    global_labels = frame_overlay.get("global_labels")
    if global_labels and color_mode != "gt":
        text = ", ".join(f"{k}:{v}" for k, v in global_labels.items() if v is not None)
        if text:
            cv2.putText(canvas, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _remap_overlay_labels(overlay: dict, label_maps: Dict[str, dict]) -> None:
    """In-place remap of labels in an already-built overlay."""
    if not overlay:
        return
    per_frame = overlay.get("per_frame", {})
    if not per_frame:
        return
    for frame_val, fdata in per_frame.items():
        ids_map = fdata.get("ids", {})
        for _id, info in ids_map.items():
            labels_map = info.get("labels") or {}
            for feat, mapping in label_maps.items():
                if feat in labels_map and labels_map[feat] in mapping:
                    labels_map[feat] = mapping[labels_map[feat]]
