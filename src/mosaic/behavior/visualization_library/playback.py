"""Video playback with interactive controls.

This module contains the high-level playback orchestrator:
- play_video: Full pipeline with window display + optional file output
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Iterable, Any, Tuple
import cv2

from .data_loading import load_tracks_and_labels, load_ground_truth_labels
from .overlay import prepare_overlay, _remap_overlay_labels
from .video_stream import render_stream
from .visual_spec import apply_visualization_spec, playback_kwargs_from_spec


def build_overlay(
    ds,
    group: str,
    sequence: str,
    feature_runs: Dict[str, Optional[str]],
    label_kind: Optional[str] = "behavior",
    color_by: Optional[str] = None,
    label_maps: Optional[Dict[str, dict]] = None,
    hide_unlabeled: bool = False,
    visualization_spec: Optional[dict] = None,
) -> Tuple[dict, Any, Dict[str, Any]]:
    """Build a base overlay (and optional spec layers), returning overlay/tracks/labels."""
    tracks_df, labels = load_tracks_and_labels(ds, group, sequence, feature_runs)

    if label_maps:
        for feat, mapping in label_maps.items():
            per_id = labels.get("per_id", {}).get(feat, {})
            for key, series in list(per_id.items()):
                per_id[key] = series.map(mapping).fillna(series)

    gt_df = None
    if label_kind:
        try:
            gt_df = load_ground_truth_labels(ds, label_kind, group, sequence)
        except FileNotFoundError as exc:
            print(f"[build_overlay] warning: {exc}")

    overlay = prepare_overlay(tracks_df, labels, gt_df=gt_df, color_by=color_by, hide_unlabeled=hide_unlabeled)
    if visualization_spec:
        apply_visualization_spec(overlay, tracks_df, labels, visualization_spec)
    return overlay, tracks_df, labels


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
               draw_options: Optional[Dict[str, Any]] = None,
               show_individual_bboxes: bool = True,
               pair_box_feature: Optional[str] = None,
               pair_box_behaviors: Optional[Iterable[Any]] = None,
               hide_individual_bboxes_for_pair: bool = False,
               output_path: Optional[Path | str] = None,
               show_window: bool = True,
               window_name: Optional[str] = None,
               visualization_spec: Optional[dict] = None) -> Optional[Path]:
    """
    Stream a video with overlays; optionally save to disk.

    Parameters
    ----------
    ds : Dataset
        Loaded Dataset instance.
    group : str
        Group name.
    sequence : str
        Sequence name.
    feature_runs : dict[str, str | None]
        Mapping of feature/model storage names -> run_id.
    label_kind : str, optional
        Kind of labels to load (default "behavior").
    color_by : str, optional
        Feature name to color by, or "gt" for ground-truth.
    label_maps : dict[str, dict], optional
        Optional mapping per feature to replace numeric labels with names, e.g.
        {"behavior-xgb-pred__from__...": {0: "attack", 1: "investigation", ...}}.
    hide_unlabeled : bool
        If True, skip drawing ids that lack labels (after any filtering/mapping).
    overlay_data : dict, optional
        Precomputed overlay from prepare_overlay(). If provided, skips rebuilding overlay
        (useful when you want to pre-filter labels before playback).
    start : int
        Starting frame index.
    end : int, optional
        Ending frame index.
    downscale : float
        Downscale factor (1.0 = no scaling).
    draw_options : dict, optional
        Optional frame-drawing options. Allowed keys: "show_labels", "point_radius", "bbox_thickness".
        You can also store defaults in overlay_data["draw_options"].
    show_individual_bboxes : bool
        If False, skip drawing per-id bounding boxes while keeping pose points/labels.
    pair_box_feature : str, optional
        Pair-label feature to inspect when drawing union boxes.
    pair_box_behaviors : iterable, optional
        Behavior values that should trigger pair-level boxes.
    hide_individual_bboxes_for_pair : bool
        If True, do not draw per-id boxes for ids participating in selected pair boxes.
    output_path : Path or str, optional
        If provided, saves video to this path.
    show_window : bool
        If True, displays video in a window.
    window_name : str, optional
        Name for the display window.
    visualization_spec : dict, optional
        Optional spec with extra render layers and playback overrides.

    Returns
    -------
    Path or None
        Path to the saved video file if output_path was provided.

    Keyboard Controls
    -----------------
    - q or Esc: Quit
    - Space: Pause/resume
    - d: Step one frame (while paused)
    - s: Save current frame as PNG
    """
    overlay = overlay_data
    spec_playback = playback_kwargs_from_spec(visualization_spec) if visualization_spec else {}
    if overlay is None:
        overlay, _, _ = build_overlay(
            ds=ds,
            group=group,
            sequence=sequence,
            feature_runs=feature_runs,
            label_kind=label_kind,
            color_by=color_by,
            label_maps=label_maps,
            hide_unlabeled=hide_unlabeled,
            visualization_spec=visualization_spec,
        )
    elif label_maps:
        _remap_overlay_labels(overlay, label_maps)

    # Spec can provide defaults, direct args still win.
    if pair_box_feature is None:
        pair_box_feature = spec_playback.get("pair_box_feature")
    if pair_box_behaviors is None:
        pair_box_behaviors = spec_playback.get("pair_box_behaviors")
    show_individual_bboxes = bool(spec_playback.get("show_individual_bboxes", show_individual_bboxes))
    hide_individual_bboxes_for_pair = bool(
        spec_playback.get("hide_individual_bboxes_for_pair", hide_individual_bboxes_for_pair)
    )

    video_paths = ds.resolve_media_paths(group, sequence)

    stream = render_stream(
        video_paths,
        overlay,
        start=start,
        end=end,
        downscale=downscale,
        draw_options=draw_options,
        show_individual_bboxes=show_individual_bboxes,
        pair_box_feature=pair_box_feature,
        pair_box_behaviors=pair_box_behaviors,
        hide_individual_bboxes_for_pair=hide_individual_bboxes_for_pair,
    )
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


def play_video_with_spec(
    ds,
    group: str,
    sequence: str,
    feature_runs: Dict[str, Optional[str]],
    visualization_spec: dict,
    **kwargs: Any,
) -> Optional[Path]:
    """
    Convenience wrapper: build overlay from tracks/labels + visualization_spec, then play/save.

    Any explicit kwargs are forwarded to play_video and override spec playback defaults.
    """
    overlay, _, _ = build_overlay(
        ds=ds,
        group=group,
        sequence=sequence,
        feature_runs=feature_runs,
        label_kind=kwargs.pop("label_kind", "behavior"),
        color_by=kwargs.pop("color_by", None),
        label_maps=kwargs.pop("label_maps", None),
        hide_unlabeled=kwargs.pop("hide_unlabeled", False),
        visualization_spec=visualization_spec,
    )
    return play_video(
        ds=ds,
        group=group,
        sequence=sequence,
        feature_runs=feature_runs,
        overlay_data=overlay,
        visualization_spec=visualization_spec,
        **kwargs,
    )
