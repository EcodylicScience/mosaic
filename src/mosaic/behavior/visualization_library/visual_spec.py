"""Visualization spec + adapter registry for overlay layers.

This module provides a lightweight, phase-1 foundation for spec-driven overlays:
- normalize_visualization_spec: validate/normalize user specs
- register_visual_adapter: plugin point for layer builders
- apply_visualization_spec: apply adapters to an overlay's per-frame payload
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import pandas as pd

from .helpers import _color_for_label


VisualAdapter = Callable[[dict[str, Any], dict[str, Any]], None]
_VISUAL_ADAPTERS: dict[str, VisualAdapter] = {}


def register_visual_adapter(name: str) -> Callable[[VisualAdapter], VisualAdapter]:
    """Register a visualization adapter by layer type name."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Adapter name must be non-empty.")

    def _decorator(fn: VisualAdapter) -> VisualAdapter:
        _VISUAL_ADAPTERS[key] = fn
        return fn

    return _decorator


def list_visual_adapters() -> list[str]:
    """List registered adapter names."""
    return sorted(_VISUAL_ADAPTERS.keys())


def normalize_visualization_spec(spec: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Normalize user-provided visualization spec."""
    if spec is None:
        return {"layers": [], "playback": {}}
    if not isinstance(spec, dict):
        raise TypeError("visualization_spec must be a dict or None.")

    layers = spec.get("layers") or []
    if not isinstance(layers, list):
        raise TypeError("visualization_spec['layers'] must be a list.")
    playback = spec.get("playback") or {}
    if not isinstance(playback, dict):
        raise TypeError("visualization_spec['playback'] must be a dict.")
    return {"layers": layers, "playback": playback}


def playback_kwargs_from_spec(spec: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Extract playback kwargs from a visualization spec."""
    norm = normalize_visualization_spec(spec)
    return dict(norm.get("playback") or {})


def _coerce_color(value: Any, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return tuple(int(v) for v in value)
        except Exception:
            return fallback
    return fallback


def apply_visualization_spec(
    overlay: dict[str, Any],
    tracks_df: pd.DataFrame,
    labels: dict[str, Any],
    spec: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Apply all enabled layers from a visualization spec into overlay in-place."""
    norm = normalize_visualization_spec(spec)
    if not norm["layers"]:
        return overlay

    ctx = {
        "overlay": overlay,
        "tracks_df": tracks_df,
        "labels": labels,
    }

    for idx, layer in enumerate(norm["layers"]):
        if not isinstance(layer, dict):
            raise TypeError(f"Layer #{idx} must be a dict.")
        if layer.get("enabled", True) is False:
            continue

        layer_type = str(layer.get("type") or "").strip().lower()
        if not layer_type:
            raise ValueError(f"Layer #{idx} is missing 'type'.")
        adapter = _VISUAL_ADAPTERS.get(layer_type)
        if adapter is None:
            available = ", ".join(list_visual_adapters()) or "<none>"
            raise KeyError(
                f"Unknown visualization layer type '{layer_type}'. "
                f"Registered adapters: {available}"
            )
        adapter(ctx, layer)

    return overlay


@register_visual_adapter("group_outlines")
def _adapter_group_outlines(ctx: dict[str, Any], layer: dict[str, Any]) -> None:
    """Create one outline per frame/group from membership rows (e.g., ffgroups)."""
    overlay = ctx["overlay"]
    labels = ctx["labels"]
    per_frame = overlay.setdefault("per_frame", {})

    feature = layer.get("feature")
    if not feature:
        raise ValueError("group_outlines layer requires 'feature'.")

    raw_df = (labels.get("raw") or {}).get(feature)
    if raw_df is None:
        raise KeyError(f"group_outlines: raw labels missing for feature '{feature}'.")

    frame_col = str(layer.get("frame_col", "frame"))
    id_col = str(layer.get("id_col", "id"))
    group_col = str(layer.get("group_col", "group_membership"))
    size_col = str(layer.get("size_col", "group_size"))
    min_group_size = int(layer.get("min_group_size", 1))
    include_singletons = bool(layer.get("include_singletons", True))
    label_mode = str(layer.get("label_mode", "group_size")).strip().lower()
    color_by = str(layer.get("color_by", "group_size")).strip().lower()

    for col in (frame_col, id_col, group_col):
        if col not in raw_df.columns:
            raise ValueError(f"group_outlines: required column '{col}' missing in '{feature}'.")

    keep_cols = [frame_col, id_col, group_col]
    if size_col in raw_df.columns:
        keep_cols.append(size_col)
    work = raw_df[keep_cols].copy()
    work[frame_col] = pd.to_numeric(work[frame_col], errors="coerce")
    work[id_col] = pd.to_numeric(work[id_col], errors="coerce")
    if size_col in work.columns:
        work[size_col] = pd.to_numeric(work[size_col], errors="coerce")
    work = work.dropna(subset=[frame_col, id_col, group_col])
    if work.empty:
        return

    work[frame_col] = work[frame_col].astype(int)
    work[id_col] = work[id_col].astype(int)

    color_map_raw = layer.get("color_map") or {}
    color_map: dict[int, tuple[int, int, int]] = {}
    if isinstance(color_map_raw, dict):
        for k, v in color_map_raw.items():
            try:
                color_map[int(k)] = _coerce_color(v, (0, 255, 255))
            except Exception:
                continue
    default_color = _coerce_color(layer.get("default_color"), (0, 255, 255))

    for (frame, group_id), sub in work.groupby([frame_col, group_col], sort=False):
        frame_data = per_frame.get(int(frame))
        if frame_data is None:
            continue

        ids = sorted(sub[id_col].astype(int).unique().tolist())
        if not include_singletons and len(ids) == 1:
            continue
        if len(ids) < min_group_size:
            continue

        group_size = len(ids)
        if size_col in sub.columns:
            sizes = sub[size_col].dropna()
            if not sizes.empty:
                try:
                    group_size = int(sizes.iloc[0])
                except Exception:
                    group_size = len(ids)

        if color_by == "group_size":
            color = color_map.get(group_size)
            if color is None:
                color = _coerce_color(_color_for_label(f"group_size:{group_size}"), default_color)
        else:
            color = default_color

        label = None
        if label_mode == "group_size":
            label = f"gsize={group_size}"
        elif label_mode == "group_id":
            label = f"group={group_id}"
        elif label_mode == "both":
            label = f"group={group_id} size={group_size}"

        render_layers = frame_data.setdefault("render_layers", {})
        outlines = render_layers.setdefault("group_outlines", [])
        outlines.append(
            {
                "ids": ids,
                "group_id": group_id,
                "group_size": int(group_size),
                "color": color,
                "label": label,
            }
        )


@register_visual_adapter("pair_roles")
def _adapter_pair_roles(ctx: dict[str, Any], layer: dict[str, Any]) -> None:
    """Apply role-based per-id color/label styles from pair-label rows."""
    overlay = ctx["overlay"]
    labels = ctx["labels"]
    per_frame = overlay.setdefault("per_frame", {})

    feature = layer.get("feature")
    if not feature:
        raise ValueError("pair_roles layer requires 'feature'.")
    raw_df = (labels.get("raw") or {}).get(feature)
    if raw_df is None:
        raise KeyError(f"pair_roles: raw labels missing for feature '{feature}'.")

    frame_col = str(layer.get("frame_col", "frame"))
    id1_col = str(layer.get("id1_col", "id1"))
    id2_col = str(layer.get("id2_col", "id2"))
    forward_col = str(layer.get("forward_col", "aa_event_12"))
    reverse_col = str(layer.get("reverse_col", "aa_event_21"))
    active_col = str(layer.get("active_col", "label_id"))
    label_key = str(layer.get("label_key", "role"))

    for col in (frame_col, id1_col, id2_col):
        if col not in raw_df.columns:
            raise ValueError(f"pair_roles: required column '{col}' missing in '{feature}'.")

    keep = [frame_col, id1_col, id2_col]
    for c in (forward_col, reverse_col, active_col):
        if c in raw_df.columns:
            keep.append(c)
    work = raw_df[keep].copy()
    for c in keep:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=[frame_col, id1_col, id2_col])
    if work.empty:
        return

    work[frame_col] = work[frame_col].astype(int)
    work[id1_col] = work[id1_col].astype(int)
    work[id2_col] = work[id2_col].astype(int)

    approach_color = _coerce_color(layer.get("approach_color"), (0, 0, 255))
    avoid_color = _coerce_color(layer.get("avoid_color"), (255, 0, 0))
    other_color = layer.get("other_color")
    other_color = _coerce_color(other_color, (140, 140, 140)) if other_color is not None else None
    approach_label = str(layer.get("approach_label", "approach"))
    avoid_label = str(layer.get("avoid_label", "avoid"))

    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in work.itertuples(index=False):
        frame = int(getattr(row, frame_col))
        id1 = int(getattr(row, id1_col))
        id2 = int(getattr(row, id2_col))

        fwd = int(getattr(row, forward_col, 0)) if forward_col in keep else 0
        rev = int(getattr(row, reverse_col, 0)) if reverse_col in keep else 0
        active = int(getattr(row, active_col, 0)) if active_col in keep else int(fwd == 1 or rev == 1)

        if active != 1 and fwd != 1 and rev != 1:
            continue

        if fwd == 1:
            by_frame.setdefault(frame, []).extend(
                [
                    {"id": id1, "color": approach_color, "label_key": label_key, "label": approach_label},
                    {"id": id2, "color": avoid_color, "label_key": label_key, "label": avoid_label},
                ]
            )
        elif rev == 1:
            by_frame.setdefault(frame, []).extend(
                [
                    {"id": id2, "color": approach_color, "label_key": label_key, "label": approach_label},
                    {"id": id1, "color": avoid_color, "label_key": label_key, "label": avoid_label},
                ]
            )

    for frame, frame_data in per_frame.items():
        render_layers = frame_data.setdefault("render_layers", {})
        styles = render_layers.setdefault("id_styles", [])
        if other_color is not None:
            for raw_id in (frame_data.get("ids") or {}).keys():
                styles.append({"id": raw_id, "color": other_color})
        styles.extend(by_frame.get(int(frame), []))


@register_visual_adapter("id_styles_from_feature")
def _adapter_id_styles_from_feature(ctx: dict[str, Any], layer: dict[str, Any]) -> None:
    """Apply per-id style overrides from a raw feature column (frame,id,value)."""
    overlay = ctx["overlay"]
    labels = ctx["labels"]
    per_frame = overlay.setdefault("per_frame", {})

    feature = layer.get("feature")
    if not feature:
        raise ValueError("id_styles_from_feature layer requires 'feature'.")
    raw_df = (labels.get("raw") or {}).get(feature)
    if raw_df is None:
        raise KeyError(f"id_styles_from_feature: raw labels missing for feature '{feature}'.")

    frame_col = str(layer.get("frame_col", "frame"))
    id_col = str(layer.get("id_col", "id"))
    value_col = str(layer.get("value_col", "label_id"))
    label_key = str(layer.get("label_key", value_col))
    label_prefix = str(layer.get("label_prefix", ""))
    color_mode = str(layer.get("color_mode", "value")).strip().lower()
    default_color = _coerce_color(layer.get("default_color"), (255, 255, 255))

    for col in (frame_col, id_col, value_col):
        if col not in raw_df.columns:
            raise ValueError(f"id_styles_from_feature: required column '{col}' missing in '{feature}'.")

    work = raw_df[[frame_col, id_col, value_col]].copy()
    work[frame_col] = pd.to_numeric(work[frame_col], errors="coerce")
    work[id_col] = pd.to_numeric(work[id_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[frame_col, id_col, value_col])
    if work.empty:
        return

    work[frame_col] = work[frame_col].astype(int)
    work[id_col] = work[id_col].astype(int)

    color_map_raw = layer.get("color_map") or {}
    color_map: dict[Any, tuple[int, int, int]] = {}
    if isinstance(color_map_raw, dict):
        for k, v in color_map_raw.items():
            color_map[k] = _coerce_color(v, default_color)
            try:
                color_map[int(k)] = _coerce_color(v, default_color)
            except Exception:
                pass

    for row in work.itertuples(index=False):
        frame = int(getattr(row, frame_col))
        rid = int(getattr(row, id_col))
        val = getattr(row, value_col)
        frame_data = per_frame.get(frame)
        if frame_data is None:
            continue

        style: dict[str, Any] = {"id": rid}
        if color_mode == "value":
            color = color_map.get(val)
            if color is None:
                try:
                    color = color_map.get(int(val))
                except Exception:
                    color = None
            if color is None:
                color = _coerce_color(_color_for_label(f"{value_col}:{val}"), default_color)
            style["color"] = color
        elif color_mode == "constant":
            style["color"] = default_color

        label_val = f"{label_prefix}{int(val) if float(val).is_integer() else val}"
        style["label_key"] = label_key
        style["label"] = label_val

        render_layers = frame_data.setdefault("render_layers", {})
        styles = render_layers.setdefault("id_styles", [])
        styles.append(style)


@register_visual_adapter("pair_focus")
def _adapter_pair_focus(ctx: dict[str, Any], layer: dict[str, Any]) -> None:
    """Keep only one pair in frame pair_labels for a feature (helpful for event clips)."""
    overlay = ctx["overlay"]
    per_frame = overlay.setdefault("per_frame", {})

    feature = layer.get("feature")
    if not feature:
        raise ValueError("pair_focus layer requires 'feature'.")

    pair_val = layer.get("pair")
    if pair_val is None:
        id1 = layer.get("id1")
        id2 = layer.get("id2")
        if id1 is None or id2 is None:
            raise ValueError("pair_focus requires either 'pair' or both 'id1' and 'id2'.")
        pair_val = (id1, id2)

    try:
        a, b = int(pair_val[0]), int(pair_val[1])
    except Exception as exc:
        raise ValueError(f"pair_focus: invalid pair value {pair_val!r}") from exc
    target = tuple(sorted((a, b)))

    for frame_data in per_frame.values():
        pair_labels = frame_data.get("pair_labels")
        if not isinstance(pair_labels, dict):
            continue
        pair_map = pair_labels.get(feature, {})
        if not isinstance(pair_map, dict):
            pair_labels[feature] = {}
            continue
        if target in pair_map:
            pair_labels[feature] = {target: pair_map[target]}
        else:
            pair_labels[feature] = {}
