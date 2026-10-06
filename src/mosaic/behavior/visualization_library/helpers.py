"""Shared visualization utilities.

This module contains helper functions used across visualization modules:
- Geometry utilities (pose extraction, bounding boxes, centroids)
- Color utilities (palettes, color assignment)
- Video I/O utilities (capture, writer, scaling)
- Cropping utilities (safe crop with padding, rotation)
"""

from __future__ import annotations
import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Tuple, Any, Optional, Dict
import numpy as np
import pandas as pd
import cv2

from mosaic.core.pipeline.tracks_index import read_tracks_index, select_variant_rows
from mosaic.core.pipeline.types import COLUMNS
from mosaic.core.schema import schema_space


# =============================================================================
# Units
# =============================================================================


def require_pixel_positions(
    ds: Any,
    group: str,
    sequence: str,
    who: str,
    *,
    schema: str | None = None,
    tracks_run_id: str | None = None,
) -> None:
    """Refuse to place a table's coordinates on video unless they are pixels.

    Called before anything is drawn or cropped, whether the caller places the
    body centre or the keypoints: in every schema family both are in the same
    space, so a keypoint is no safer than ``X``. Only the ``image_px`` space --
    ``mosaic_v1`` and everything extending it -- is video pixels.

    - ``plane_cm`` (``trex_v1``, ``mosaic_cm_v1``) is centimetres, and a marker
      placed at those coordinates lands somewhere else in the frame entirely.
    - ``world_m`` (``mosaic_world_v1``) is metres in a world frame, which reaches
      a video only through a projection with the recording's calibration.
    - A schema not registered in this process cannot be vouched for, and is
      refused rather than guessed at.

    Which table is judged matters once an entry holds several variants. A caller
    that knows the schema of the table it was handed passes it as *schema* (a
    feature reads it off ``ResolvedScope.tracks_schemas``). Otherwise the entry's
    index row is chosen the way the loader chose it, by ``select_variant_rows``
    under *tracks_run_id*, rather than by taking whichever row comes first. An
    entry with no row, or a blank cell, is the legacy ``trex_v1``.

    Args:
        ds: The dataset holding the tracks index.
        group: Entry group.
        sequence: Entry sequence.
        who: Name of the caller, for the message.
        schema: The schema of the table the caller holds, when it knows it.
        tracks_run_id: The tracks variant the caller read, when it does not.

    Raises:
        ValueError: If the table's space is not ``image_px``, or if the entry
            holds two variants and neither *schema* nor *tracks_run_id* says
            which one was read.
    """
    recorded = (
        schema
        if schema is not None
        else _recorded_schema(ds, group, sequence, tracks_run_id)
    )
    space = schema_space(recorded)
    if space == "image_px":
        return
    head = (
        f"{who} is placing coordinates for {group}/{sequence} on video, and this "
        f"table records schema {recorded or '(none)'!r}"
    )
    if space == "plane_cm":
        raise ValueError(
            f"{head}, whose spatial columns are centimetres rather than video "
            "pixels. A trex_v1 table is converted to pixels by `mosaic "
            "upgrade-tracks`; a mosaic_cm_v1 table records no pixel factor and "
            "cannot be drawn."
        )
    if space == "world_m":
        raise ValueError(
            f"{head}, whose spatial columns are metres in a world frame. They reach "
            "a video only through a projection with the recording's calibration, "
            f"which {who} does not perform."
        )
    raise ValueError(
        f"{head}, which is not registered in this process, so nothing says what "
        "its coordinates are. Register the schema, or extend 'mosaic_v1' if its "
        "coordinates are video pixels."
    )


def _recorded_schema(
    ds: Any, group: str, sequence: str, tracks_run_id: str | None
) -> str:
    """The schema an entry's tracks row records, chosen as the loader chooses."""
    index = read_tracks_index(ds)
    if index.empty:
        return ""
    hit = index[
        (index[COLUMNS.group_col] == group) & (index[COLUMNS.seq_col] == sequence)
    ]
    chosen = select_variant_rows(hit, tracks_run_id)
    if chosen.empty:
        return ""
    return str(chosen["std_format"].iloc[0])


# =============================================================================
# Parameter Merging (follows feature_library pattern)
# =============================================================================


def _merge_params(
    overrides: Optional[Dict[str, Any]], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge user overrides with defaults. None values in overrides use defaults."""
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


# =============================================================================
# Color Palettes
# =============================================================================

ID_PALETTE = [
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

LABEL_PALETTE = [
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


def _palette_index(value: Any, palette_size: int) -> int:
    """Stable palette slot for *value*, identical in every process.

    ``hash()`` on a ``str`` is salted by ``PYTHONHASHSEED``, so the obvious
    spelling of this gave one id a different color on every run -- two renders of
    the same tracks disagreeing about which animal is which. A digest has no
    per-process salt, so a sequence renders the same way today, tomorrow, and on
    someone else's machine.
    """
    digest = hashlib.blake2b(str(value).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % palette_size


def color_for_id(id_val: Any) -> Tuple[int, int, int]:
    """Stable per-ID color."""
    return ID_PALETTE[_palette_index(id_val, len(ID_PALETTE))]


def color_for_label(label_val: Any) -> Tuple[int, int, int]:
    """Stable per-label color."""
    return LABEL_PALETTE[_palette_index(label_val, len(LABEL_PALETTE))]


# =============================================================================
# Geometry Utilities
# =============================================================================


def _extract_pose_points(
    row: pd.Series, pose_pairs: list[Tuple[str, str]]
) -> list[Tuple[float, float]]:
    """Extract list of (x, y) coordinates from a DataFrame row."""
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


def _compute_bbox(
    points: list[Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    """Compute axis-aligned bounding box from point list."""
    xs = [p[0] for p in points if np.isfinite(p[0])]
    ys = [p[1] for p in points if np.isfinite(p[1])]
    if not xs or not ys:
        return (np.nan, np.nan, np.nan, np.nan)
    return (min(xs), min(ys), max(xs), max(ys))


def _extract_centroid(
    row: pd.Series, candidates: list[Tuple[str, str]]
) -> Optional[Tuple[float, float]]:
    """Find first valid centroid from candidate (x, y) column pairs."""
    for x_col, y_col in candidates:
        x = row.get(x_col)
        y = row.get(y_col)
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
            return (float(x), float(y))
    return None


# =============================================================================
# Video I/O Utilities
# =============================================================================


def _scaled_size(base_size: Tuple[int, int], downscale: float) -> Tuple[int, int]:
    """Compute resized dimensions with downscale factor."""
    w, h = base_size
    if downscale and downscale > 0 and downscale != 1.0:
        return (max(1, int(w * downscale)), max(1, int(h * downscale)))
    return base_size


def create_video_writer(
    path: Path | str, fps: float, frame_size: Tuple[int, int], fourcc: str = "mp4v"
) -> cv2.VideoWriter:
    """Create a VideoWriter with standard settings.

    Parameters
    ----------
    path : Path or str
        Output file path
    fps : float
        Frames per second
    frame_size : tuple of (width, height)
        Frame dimensions
    fourcc : str
        Four-character codec code (default "mp4v")

    Returns
    -------
    cv2.VideoWriter
        Opened video writer. Caller is responsible for calling .release()
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    codec = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(path), codec, float(fps), frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {path}")
    return writer


# =============================================================================
# Label/Data Lookup Utilities
# =============================================================================


def _lookup_label_series(per_id_map: dict, id_val: Any) -> Optional[pd.Series]:
    """Flexible lookup with fallback: exact match -> str(id_val) -> int(id_val) -> None."""
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


def remap_label_value(value: object, mapping: Mapping[object, object]) -> object:
    """Look a label value up in a rename map, by the value and by its text.

    A map declared on ``Params`` is JSON-native and so keyed by strings, while a
    label read out of a parquet column is whatever that column holds -- an int
    under ``label_id``, ``prediction``, ``cluster`` and ``state``. Matching both
    spellings lets one map serve a params run and a direct call. A value that
    matches neither is returned unchanged.
    """
    try:
        if value in mapping:
            return mapping[value]
    except TypeError:
        # An unhashable label (the list a pair feature collects) keys nothing.
        return value
    text = str(value)
    return mapping[text] if text in mapping else value


def label_renamer(mapping: Mapping[object, object]) -> Callable[[object], object]:
    """A one-argument rename over *mapping*, for ``Series.map``.

    Binding the map here rather than in a lambda at the call site keeps the
    rename usable inside a loop over several feature maps: each renamer closes
    over the map it was built from.
    """

    def rename(value: object) -> object:
        return remap_label_value(value, mapping)

    return rename


def _scalar_from_series(value: Any) -> Any:
    """Extract scalar from pd.Series or return as-is."""
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        return value.iloc[-1]
    return value


def _format_label_text(value: Any) -> str:
    """Format value(s) for text overlay."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " | ".join(_format_label_text(v) for v in value)
    try:
        return f"{value}"
    except Exception:
        return str(value)


# =============================================================================
# Cropping Utilities (NEW for egocentric crop)
# =============================================================================


def safe_crop_with_padding(
    image: np.ndarray,
    center: Tuple[int, int],
    crop_size: Tuple[int, int],
    pad_value: int = 0,
) -> np.ndarray:
    """Extract a crop centered at (cx, cy), padding if out of bounds.

    Parameters
    ----------
    image : np.ndarray
        Source image (H, W, C) or (H, W)
    center : tuple of (cx, cy)
        Center point in pixel coordinates
    crop_size : tuple of (width, height)
        Size of output crop
    pad_value : int
        Padding color (0=black, 255=white)

    Returns
    -------
    np.ndarray
        Cropped image of size (height, width, C) or (height, width)
    """
    cx, cy = int(center[0]), int(center[1])
    crop_w, crop_h = crop_size
    h, w = image.shape[:2]

    # Compute source region
    x0 = cx - crop_w // 2
    y0 = cy - crop_h // 2
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    # Compute valid source region (clamped to image bounds)
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)

    # Compute destination region in output crop
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    # Create output with padding
    if image.ndim == 3:
        crop = np.full((crop_h, crop_w, image.shape[2]), pad_value, dtype=image.dtype)
    else:
        crop = np.full((crop_h, crop_w), pad_value, dtype=image.dtype)

    # Copy valid region
    if src_x1 > src_x0 and src_y1 > src_y0:
        crop[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]

    return crop


def compute_heading_angle(
    neck_xy: Tuple[float, float], tail_xy: Tuple[float, float]
) -> float:
    """Compute heading angle from anatomical landmarks (neck -> tail direction).

    Parameters
    ----------
    neck_xy : tuple of (x, y)
        Neck/head position
    tail_xy : tuple of (x, y)
        Tail position

    Returns
    -------
    float
        Heading angle in radians. 0 = facing right (+x), pi/2 = facing down (+y)
        The heading points FROM tail TO neck (i.e., where the animal is facing).
    """
    dx = neck_xy[0] - tail_xy[0]
    dy = neck_xy[1] - tail_xy[1]
    return float(np.arctan2(dy, dx))


def infer_angle_degrees(angle_series: pd.Series) -> bool:
    """Infer whether angle values are in degrees or radians.

    Parameters
    ----------
    angle_series : pd.Series
        Series of angle values

    Returns
    -------
    bool
        True if angles appear to be in degrees, False if radians
    """
    # Filter to finite values
    valid = angle_series.dropna()
    valid = valid[np.isfinite(valid)]
    if valid.empty:
        return False  # Default to radians if no data

    abs_max = np.abs(valid).max()
    # If max absolute value > 2*pi, likely degrees
    return abs_max > 2 * np.pi
