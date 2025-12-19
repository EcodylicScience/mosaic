# visualization.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, Iterable
import pandas as pd
import numpy as np
import cv2

from .helpers import to_safe_name

try:
    from .dataset import (
        _yield_sequences,
        _feature_index_path,
        _latest_feature_run_root,
    )
except Exception as exc:  # pragma: no cover - allows import-time clarity in notebooks
    raise ImportError("visualization requires dataset module to be importable.") from exc


def _pick_label_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristic to pick a label-like column from a feature/model output frame.
    Prioritizes common names, falls back to the first non-index column.
    """
    preferred = ["label_id", "label", "prediction", "cluster", "behavior", "state"]
    for col in preferred:
        if col in df.columns:
            return col
    skip = {"frame", "sequence", "group", "id", "id_a", "id_b"}
    for col in df.columns:
        if col not in skip:
            return col
    return None


def load_tracks_and_labels(
    ds,
    group: str,
    sequence: str,
    feature_runs: Dict[str, Optional[str]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load a single sequence's tracks plus per-frame labels from feature/model runs.

    Parameters
    ----------
    ds : Dataset
        Loaded Dataset instance.
    group, sequence : str
        The scope to load.
    feature_runs : dict[str, str | None]
        Mapping of feature/model storage names -> run_id.
        If run_id is None, the latest finished run is used.

    Returns
    -------
    tracks_df : pd.DataFrame
        Standard tracks for the requested (group, sequence).
    labels : dict
        {
          "per_id": {feature_name: {id_value: Series}},
          "per_pair": {feature_name: {(id_a, id_b): Series}},
          "raw": {feature_name: DataFrame}  # full frame per feature for bespoke use
        }
        Series are indexed by frame and hold the chosen label column.
    """
    tracks_df = None
    for _, _, df in _yield_sequences(ds, groups=[group], sequences=[sequence]):
        tracks_df = df
        break
    if tracks_df is None:
        raise FileNotFoundError(f"No tracks found for group='{group}', sequence='{sequence}'.")

    per_id: dict[str, dict[Any, pd.Series]] = {}
    per_pair: dict[str, dict[Tuple[Any, Any], pd.Series]] = {}
    raw: dict[str, pd.DataFrame] = {}

    safe_seq = to_safe_name(sequence)
    safe_group = to_safe_name(group)

    for feature_name, run_id in feature_runs.items():
        # Resolve run_id if not provided
        resolved_run_id = run_id
        if not resolved_run_id:
            resolved_run_id, _ = _latest_feature_run_root(ds, feature_name)

        idx_path = _feature_index_path(ds, feature_name)
        if not idx_path.exists():
            raise FileNotFoundError(f"Missing feature index for '{feature_name}': {idx_path}")
        df_idx = pd.read_csv(idx_path)

        # Normalize NaNs/None to empty strings so blank/absent groups still match
        for col in ("sequence", "sequence_safe", "group", "group_safe"):
            if col in df_idx.columns:
                df_idx[col] = df_idx[col].fillna("").astype(str)

        df_idx = df_idx[df_idx["run_id"].astype(str) == str(resolved_run_id)]
        if "sequence_safe" in df_idx.columns:
            df_idx = df_idx[df_idx["sequence_safe"] == safe_seq]
        else:
            df_idx = df_idx[df_idx["sequence"].astype(str) == str(sequence)]
        if "group_safe" in df_idx.columns:
            df_idx = df_idx[df_idx["group_safe"] == safe_group]
        elif "group" in df_idx.columns:
            df_idx = df_idx[df_idx["group"].astype(str) == str(group)]

        if df_idx.empty:
            raise FileNotFoundError(
                f"No rows in feature index for '{feature_name}' run_id='{resolved_run_id}' "
                f"group='{group}' sequence='{sequence}'."
            )

        abs_path_raw = df_idx.iloc[0]["abs_path"]
        path = Path(abs_path_raw)
        if hasattr(ds, "remap_path"):
            path = ds.remap_path(path)
        df_feat = pd.read_parquet(path)
        raw[feature_name] = df_feat

        label_col = _pick_label_column(df_feat)
        if not label_col or "frame" not in df_feat.columns:
            continue  # nothing label-like to index

        if "id" in df_feat.columns:
            for id_val, sub in df_feat.groupby("id"):
                series = sub.set_index("frame")[label_col].sort_index()
                per_id.setdefault(feature_name, {})[id_val] = series
        elif {"id_a", "id_b"}.issubset(df_feat.columns):
            # Normalize pairs to sorted tuples for stable lookup
            pairs = df_feat[["id_a", "id_b"]].apply(
                lambda row: tuple(sorted([row["id_a"], row["id_b"]])), axis=1
            )
            df_feat = df_feat.assign(_pair=pairs)
            for pair, sub in df_feat.groupby("_pair"):
                series = sub.set_index("frame")[label_col].sort_index()
                per_pair.setdefault(feature_name, {})[pair] = series
        else:
            # Global series (no ids) stays under per_id with a None id key
            series = df_feat.set_index("frame")[label_col].sort_index()
            per_id.setdefault(feature_name, {})[None] = series

    labels = {"per_id": per_id, "per_pair": per_pair, "raw": raw}
    return tracks_df, labels


def load_ground_truth_labels(
    ds,
    label_kind: str,
    group: str,
    sequence: str,
) -> pd.DataFrame:
    """
    Load per-frame ground-truth labels for a given kind/group/sequence.

    Returns a DataFrame with columns:
        frame, label_id, label_name (if mapping provided in the npz).
    """
    labels_root = Path(ds.get_root("labels")) / label_kind
    idx_path = labels_root / "index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(f"Label index not found for kind='{label_kind}': {idx_path}")
    df_idx = pd.read_csv(idx_path)
    if df_idx.empty:
        raise FileNotFoundError(f"No labels indexed for kind='{label_kind}'.")

    safe_group = to_safe_name(group)
    safe_seq = to_safe_name(sequence)

    hits = df_idx[
        (df_idx["group"].astype(str) == str(group)) &
        (df_idx["sequence"].astype(str) == str(sequence))
    ]
    if hits.empty and {"group_safe", "sequence_safe"}.issubset(df_idx.columns):
        hits = df_idx[
            (df_idx["group_safe"] == safe_group) &
            (df_idx["sequence_safe"] == safe_seq)
        ]
    if hits.empty:
        raise FileNotFoundError(
            f"No GT labels for kind='{label_kind}' group='{group}' sequence='{sequence}'."
        )

    path = Path(hits.iloc[0]["abs_path"])
    if hasattr(ds, "remap_path"):
        path = ds.remap_path(path)
    payload = np.load(path, allow_pickle=True)
    frames = payload["frames"]
    label_ids = payload["labels"]
    label_id_list = payload.get("label_ids")
    label_name_list = payload.get("label_names")
    id_to_name: dict[int, str] = {}
    if label_id_list is not None and label_name_list is not None:
        for lid, name in zip(label_id_list, label_name_list):
            id_to_name[int(lid)] = str(name)
    label_names = [id_to_name.get(int(val), str(val)) for val in label_ids]
    return pd.DataFrame({
        "frame": frames.astype(int, copy=False),
        "label_id": label_ids.astype(int, copy=False),
        "label_name": label_names,
    })


# Convenience example for notebooks:
def demo_load_visual_inputs(ds, group: str, sequence: str, features: Dict[str, Optional[str]]):
    """
    Small wrapper to quickly inspect what load_tracks_and_labels returns.
    Usage (notebook):
        tracks, labels = demo_load_visual_inputs(dataset, "G1", "S1",
                                                 {"temporal-stack": None,
                                                  "behavior-xgb-pred": "<run_id>"})
    """
    tracks, labels = load_tracks_and_labels(ds, group, sequence, features)
    print(f"Tracks shape: {tracks.shape}")
    for kind in ("per_id", "per_pair"):
        print(f"{kind}:")
        for feat, mapping in labels[kind].items():
            print(f"  {feat}: {len(mapping)} series")
    return tracks, labels


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

    gt_map: dict[int, dict[str, Any]] = {}
    if gt_df is not None and not gt_df.empty and "frame" in gt_df.columns:
        if "label_name" not in gt_df.columns and "label_id" in gt_df.columns:
            gt_df = gt_df.assign(label_name=gt_df["label_id"])
        gt_map = gt_df.set_index("frame").to_dict(orient="index")

    per_frame: dict[int, dict[str, Any]] = {}
    id_colors: dict[Any, Tuple[int, int, int]] = {}
    label_colors: dict[str, Tuple[int, int, int]] = {}
    color_mode = (color_by or "").strip().lower()
    color_feature = None
    if color_mode and color_mode != "gt":
        for feat in per_id_labels.keys():
            if feat.lower() == color_mode:
                color_feature = feat
                break

    centroid_cols = [("X#wcentroid", "Y#wcentroid"), ("X", "Y")]

    grouped = tracks_df.groupby("frame", sort=True)
    for frame_val, frame_df in grouped:
        frame_int = int(frame_val)
        id_infos: dict[Any, dict[str, Any]] = {}
        global_labels = gt_map.get(frame_int, {})
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
            for feat_name, per_id_map in per_id_labels.items():
                series = _lookup_label_series(per_id_map, id_val)
                if series is not None:
                    val = _scalar_from_series(series.get(frame_int))
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        labels_for_id[feat_name] = val
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
            elif frame_color is not None:
                color = frame_color
            if color is None:
                color = id_colors.get(id_val)
                if color is None:
                    color = _color_for_id(id_val)
                    id_colors[id_val] = color
            info["color"] = color
            id_infos[id_val] = info

        if not id_infos and not global_labels:
            continue
        per_frame[frame_int] = {
            "ids": id_infos,
            "global_labels": global_labels,
            "frame_color": frame_color,
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
               point_radius: int = 4,
               bbox_thickness: int = 2,
               scale: Tuple[float, float] = (1.0, 1.0),
               color_feature: Optional[str] = None,
               color_mode: Optional[str] = None) -> np.ndarray:
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
    for id_val, info in ids.items():
        base_color = info.get("color")
        if base_color is None:
            base_color = id_colors.get(id_val, (0, 255, 0))
        color = tuple(int(c) for c in (frame_color if color_mode == "gt" and frame_color is not None else base_color))
        if "bbox" in info:
            x1, y1, x2, y2 = info["bbox"]
            if all(np.isfinite([x1, y1, x2, y2])):
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

    global_labels = frame_overlay.get("global_labels")
    if global_labels and color_mode != "gt":
        text = ", ".join(f"{k}:{v}" for k, v in global_labels.items() if v is not None)
        if text:
            cv2.putText(canvas, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def render_stream(video_path: Path | str,
                  overlay_data: dict,
                  start: int = 0,
                  end: Optional[int] = None,
                  downscale: float = 1.0):
    """
    Return an iterable that yields (frame_index, frame_bgr_with_overlay).
    """
    cap, fps, base_size = _open_video_capture(video_path)
    scaled_size = _scaled_size(base_size, downscale)
    per_frame = overlay_data.get("per_frame", {})
    id_colors = overlay_data.get("id_colors", {})
    color_feature = overlay_data.get("color_feature")
    color_mode = overlay_data.get("color_mode")
    return _FrameStream(
        cap, fps, base_size, scaled_size, per_frame, id_colors,
        start, end, color_feature=color_feature, color_mode=color_mode)


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


def play_video(ds,
               group: str,
               sequence: str,
               feature_runs: Dict[str, Optional[str]],
               label_kind: Optional[str] = "behavior",
               color_by: Optional[str] = None,
               label_maps: Optional[Dict[str, dict]] = None,
               hide_unlabeled: bool = False,
               overlay_data: Optional[dict] = None,
                start: int = 0,
                end: Optional[int] = None,
                downscale: float = 1.0,
                output_path: Optional[Path | str] = None,
                show_window: bool = True,
                window_name: Optional[str] = None) -> Optional[Path]:
    """
    Stream a video with overlays; optionally save to disk.

    Parameters
    ----------
    label_maps : dict[str, dict], optional
        Optional mapping per feature to replace numeric labels with names, e.g.
        {"behavior-xgb-pred__from__...": {0: "attack", 1: "investigation", ...}}.
    hide_unlabeled : bool
        If True, skip drawing ids that lack labels (after any filtering/mapping).
    overlay_data : dict, optional
        Precomputed overlay from prepare_overlay(). If provided, skips rebuilding overlay
        (useful when you want to pre-filter labels before playback).
    """
    overlay = overlay_data
    if overlay is None:
        tracks_df, labels = load_tracks_and_labels(ds, group, sequence, feature_runs)

        if label_maps:
            # Remap predicted labels to human-friendly names if provided
            for feat, mapping in label_maps.items():
                per_id = labels.get("per_id", {}).get(feat, {})
                for key, series in list(per_id.items()):
                    per_id[key] = series.map(mapping).fillna(series)

        gt_df = None
        if label_kind:
            try:
                gt_df = load_ground_truth_labels(ds, label_kind, group, sequence)
            except FileNotFoundError as exc:
                print(f"[play_video] warning: {exc}")
        overlay = prepare_overlay(tracks_df, labels, gt_df=gt_df, color_by=color_by, hide_unlabeled=hide_unlabeled)
    elif label_maps:
        _remap_overlay_labels(overlay, label_maps)

    video_path = ds.resolve_media_path(group, sequence)

    stream = render_stream(video_path, overlay, start=start, end=end, downscale=downscale)
    writer = None
    out_path = None
    if output_path:
        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_size = getattr(stream, "frame_size", (0, 0))
        writer = cv2.VideoWriter(str(out_path), fourcc, float(getattr(stream, "fps", 30.0)), frame_size)
        if not writer.isOpened():
            writer = None
            out_path = None
            print("[play_video] warning: failed to open VideoWriter; skipping output file.")

    win = window_name or f"{group}:{sequence}"
    try:
        stream_iter = iter(stream)
        paused = False
        step_once = False
        current = None
        frame_idx = None
        while True:
            if not paused or step_once or current is None:
                try:
                    frame_idx, current = next(stream_iter)
                except StopIteration:
                    break
            if writer:
                writer.write(current)
            if show_window:
                cv2.imshow(win, current)
                delay = 1 if not paused else 50
                key = cv2.waitKey(delay) & 0xFF
            else:
                key = -1

            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                paused = not paused
                step_once = False
            elif key == ord("s") and frame_idx is not None:
                snap_path = Path(f"frame_{frame_idx}.png")
                cv2.imwrite(str(snap_path), current)
                print(f"[play_video] saved frame -> {snap_path}")
            elif key == ord("d"):
                paused = True
                step_once = True
            else:
                step_once = False
    finally:
        if hasattr(stream, "close"):
            stream.close()
        if writer:
            writer.release()
        if show_window:
            try:
                cv2.destroyWindow(win)
                cv2.waitKey(1)
            except cv2.error:
                pass
    return out_path


# ---- Internal helpers ----

def _pose_column_pairs(columns: Iterable[str]) -> list[Tuple[str, str]]:
    pose_pairs = []
    xs = [c for c in columns if c.startswith("poseX")]
    ys = [c for c in columns if c.startswith("poseY")]
    for x_col in sorted(xs):
        idx = x_col[5:]
        y_col = f"poseY{idx}"
        if y_col in columns:
            pose_pairs.append((x_col, y_col))
    return pose_pairs


def _extract_pose_points(row: pd.Series, pose_pairs: list[Tuple[str, str]]) -> list[Tuple[float, float]]:
    pts = []
    for x_col, y_col in pose_pairs:
        x = row.get(x_col)
        y = row.get(y_col)
        if x is None or y is None:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        pts.append((float(x), float(y)))
    return pts


def _compute_bbox(points: list[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points if np.isfinite(p[0])]
    ys = [p[1] for p in points if np.isfinite(p[1])]
    if not xs or not ys:
        return (np.nan, np.nan, np.nan, np.nan)
    return (min(xs), min(ys), max(xs), max(ys))


def _extract_centroid(row: pd.Series, candidates: list[Tuple[str, str]]) -> Optional[Tuple[float, float]]:
    for x_col, y_col in candidates:
        x = row.get(x_col)
        y = row.get(y_col)
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
            return (float(x), float(y))
    return None


def _color_for_id(id_val: Any) -> Tuple[int, int, int]:
    palette = [
        (230, 57, 70),
        (29, 53, 87),
        (69, 123, 157),
        (168, 218, 220),
        (240, 128, 128),
        (255, 195, 0),
        (88, 24, 69),
        (0, 109, 119),
        (144, 190, 109),
        (67, 170, 139),
    ]
    idx = hash(str(id_val)) % len(palette)
    return palette[idx]


def _color_for_label(label_val: Any) -> Tuple[int, int, int]:
    base_palette = [
        (220, 20, 60),
        (25, 130, 196),
        (60, 179, 113),
        (255, 140, 0),
        (147, 112, 219),
        (70, 130, 180),
        (255, 182, 193),
        (0, 206, 209),
        (255, 215, 0),
        (244, 164, 96),
    ]
    idx = hash(str(label_val)) % len(base_palette)
    return base_palette[idx]


def _lookup_label_series(per_id_map: dict, id_val: Any) -> Optional[pd.Series]:
    if id_val in per_id_map:
        return per_id_map[id_val]
    candidates = []
    if id_val is not None:
        candidates.append(str(id_val))
        try:
            candidates.append(int(id_val))
        except Exception:
            pass
    for cand in candidates:
        if cand in per_id_map:
            return per_id_map[cand]
    for fallback in (None, "", "none"):
        if fallback in per_id_map:
            return per_id_map[fallback]
    return None


def _scalar_from_series(value: Any) -> Any:
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        return value.iloc[-1]
    return value


def _open_video_capture(path: Path | str):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, float(fps), (width, height)


def _scaled_size(base_size: Tuple[int, int], downscale: float) -> Tuple[int, int]:
    w, h = base_size
    if downscale and downscale > 0 and downscale != 1.0:
        return (max(1, int(w * downscale)), max(1, int(h * downscale)))
    return base_size


class _FrameStream:
    def __init__(self, cap, fps, base_size, scaled_size, per_frame, id_colors,
                 start, end, color_feature=None, color_mode=None):
        self._cap = cap
        self._fps = fps or 30.0
        self._base_size = base_size
        self._scaled_size = scaled_size
        sx = scaled_size[0] / base_size[0] if base_size[0] else 1.0
        sy = scaled_size[1] / base_size[1] if base_size[1] else 1.0
        self._scale = (sx, sy)
        self._per_frame = per_frame
        self._id_colors = id_colors
        self._start = max(0, int(start))
        self._end = int(end) if end is not None else None
        self._started = False
        self._frame_idx = 0
        self._released = False
        self._color_feature = color_feature
        self._color_mode = color_mode

    @property
    def fps(self) -> float:
        return float(self._fps)

    @property
    def frame_size(self) -> Tuple[int, int]:
        return self._scaled_size

    def __iter__(self):
        return self

    def __next__(self):
        if self._released:
            raise StopIteration
        cap = self._cap
        if not self._started:
            if self._start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self._start)
            self._frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            self._started = True
        if self._end is not None and self._frame_idx > self._end:
            self.close()
            raise StopIteration
        ret, frame = cap.read()
        if not ret:
            self.close()
            raise StopIteration
        if self._scaled_size != self._base_size:
            frame = cv2.resize(frame, self._scaled_size, interpolation=cv2.INTER_AREA)
        idx = self._frame_idx
        self._frame_idx += 1
        frame_overlay = self._per_frame.get(idx)
        if frame_overlay:
            frame = draw_frame(
                frame, frame_overlay, self._id_colors,
                scale=self._scale, color_feature=self._color_feature,
                color_mode=self._color_mode)
        return idx, frame

    def close(self):
        if not self._released:
            self._cap.release()
            self._released = True

    def __del__(self):
        self.close()


def _format_label_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " | ".join(_format_label_text(v) for v in value)
    try:
        return f"{value}"
    except Exception:
        return str(value)
