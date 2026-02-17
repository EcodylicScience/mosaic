"""Video streaming with overlay rendering.

This module contains the video streaming infrastructure:
- _FrameStream: Iterator yielding (frame_idx, frame_bgr) with overlay
- render_stream: Factory function returning a _FrameStream
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Iterable, Any, Dict, Union
import cv2

from .helpers import _scaled_size
from .overlay import draw_frame

_ALLOWED_DRAW_OPTIONS = {"show_labels", "point_radius", "bbox_thickness"}


class _FrameStream:
    """Iterator that yields (frame_index, frame_bgr_with_overlay) tuples."""

    def __init__(self, reader, fps, base_size, scaled_size, per_frame, id_colors,
                 start, end, color_feature=None, color_mode=None,
                 show_individual_bboxes: bool = True,
                 pair_box_feature: Optional[str] = None,
                 pair_box_behaviors: Optional[Iterable[Any]] = None,
                 hide_individual_bboxes_for_pair: bool = False,
                 draw_options: Optional[Dict[str, Any]] = None):
        self._reader = reader
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
        self._show_individual_bboxes = show_individual_bboxes
        self._pair_box_feature = pair_box_feature
        self._pair_box_behaviors = pair_box_behaviors
        self._hide_individual_bboxes_for_pair = hide_individual_bboxes_for_pair
        self._draw_options = draw_options or {}

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
        if not self._started:
            if self._start > 0:
                self._reader.seek(self._start)
            self._frame_idx = self._start
            self._started = True
        if self._end is not None and self._frame_idx > self._end:
            self.close()
            raise StopIteration
        ret, frame = self._reader.read()
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
                color_mode=self._color_mode,
                show_individual_bboxes=self._show_individual_bboxes,
                pair_box_feature=self._pair_box_feature,
                pair_box_behaviors=self._pair_box_behaviors,
                hide_individual_bboxes_for_pair=self._hide_individual_bboxes_for_pair,
                **self._draw_options)
        return idx, frame

    def close(self):
        if not self._released:
            self._reader.close()
            self._released = True

    def __del__(self):
        self.close()


def render_stream(video_paths: Union[list[Path], Path, str],
                  overlay_data: dict,
                  start: int = 0,
                  end: Optional[int] = None,
                  downscale: float = 1.0,
                  show_individual_bboxes: bool = True,
                  pair_box_feature: Optional[str] = None,
                  pair_box_behaviors: Optional[Iterable[Any]] = None,
                  hide_individual_bboxes_for_pair: bool = False,
                  draw_options: Optional[Dict[str, Any]] = None) -> _FrameStream:
    """
    Return an iterable that yields (frame_index, frame_bgr_with_overlay).

    Parameters
    ----------
    video_paths : list[Path], Path, or str
        Path(s) to the video file(s). For multi-video sequences, pass an
        ordered list of Paths. A single Path/str is also accepted.
    overlay_data : dict
        Output from prepare_overlay()
    start : int
        Starting frame index
    end : int, optional
        Ending frame index (inclusive). If None, streams to end of video.
    downscale : float
        Downscale factor (1.0 = no scaling, 0.5 = half size)
    show_individual_bboxes : bool
        If False, skip drawing per-id bounding boxes while keeping pose points/labels.
    pair_box_feature : str, optional
        Pair-label feature to inspect when drawing union boxes.
    pair_box_behaviors : iterable, optional
        Behavior values that should trigger pair-level boxes.
    hide_individual_bboxes_for_pair : bool
        If True, do not draw per-id boxes for ids participating in selected pair boxes.
    draw_options : dict, optional
        Optional frame-drawing options. Allowed keys: "show_labels", "point_radius", "bbox_thickness".

    Returns
    -------
    _FrameStream
        Iterator yielding (frame_index, frame_bgr) tuples
    """
    from mosaic.media.video_io import MultiVideoReader

    reader = MultiVideoReader(video_paths)
    base_size = (reader.width, reader.height)
    fps = reader.fps
    scaled_size = _scaled_size(base_size, downscale)
    per_frame = overlay_data.get("per_frame", {})
    id_colors = overlay_data.get("id_colors", {})
    color_feature = overlay_data.get("color_feature")
    color_mode = overlay_data.get("color_mode")
    merged_draw_options = {}
    overlay_draw_options = overlay_data.get("draw_options")
    if isinstance(overlay_draw_options, dict):
        merged_draw_options.update({k: v for k, v in overlay_draw_options.items() if k in _ALLOWED_DRAW_OPTIONS})
    if isinstance(draw_options, dict):
        merged_draw_options.update({k: v for k, v in draw_options.items() if k in _ALLOWED_DRAW_OPTIONS})

    return _FrameStream(
        reader, fps, base_size, scaled_size, per_frame, id_colors,
        start, end, color_feature=color_feature, color_mode=color_mode,
        show_individual_bboxes=show_individual_bboxes,
        pair_box_feature=pair_box_feature,
        pair_box_behaviors=pair_box_behaviors,
        hide_individual_bboxes_for_pair=hide_individual_bboxes_for_pair,
        draw_options=merged_draw_options)
