"""
PairInteractionFilter -- detect pairwise interaction segments from trajectories.

Identifies frames where pairs of individuals meet configurable distance and
angular thresholds.  Applies morphological filtering to remove noise and
enforces a minimum interaction duration.

Typical use cases:
  - Detecting face-to-face interactions (distance + facing criterion)
  - Proximity-based pair detection (distance only, ``require_facing=False``)
  - Pre-filtering for expensive downstream processing (e.g. interaction crops)

All thresholds are parameterized and should be tuned per application.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
from pydantic import Field
from scipy.ndimage import binary_closing, binary_opening

from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    Inputs,
    InputStream,
    Params,
    Result,
    TrackInput,
    resolve_order_col,
)

from .helpers import ensure_columns
from .registry import register_feature


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_interaction_conditions(
    x_a: np.ndarray,
    y_a: np.ndarray,
    orient_a: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    orient_b: np.ndarray,
    shift_dist: float,
    max_dist: float,
    max_inv_orientation_diff_deg: float,
    require_facing: bool,
) -> np.ndarray:
    """Per-frame boolean: do two individuals meet distance + angle criteria?

    Both positions are shifted forward in their respective orientation
    directions before computing distance.  The angular test checks whether
    the individuals are roughly facing each other (inverse orientation
    difference < threshold).
    """
    # Shift positions forward along each individual's heading
    x_a_shifted = x_a + np.cos(orient_a) * shift_dist
    y_a_shifted = y_a + np.sin(orient_a) * shift_dist
    x_b_shifted = x_b + np.cos(orient_b) * shift_dist
    y_b_shifted = y_b + np.sin(orient_b) * shift_dist

    dist = np.hypot(x_a_shifted - x_b_shifted, y_a_shifted - y_b_shifted)
    meets = dist < max_dist

    if require_facing:
        # Circular orientation difference (shortest arc)
        orient_diff = np.abs(orient_a - orient_b)
        orient_diff = np.minimum(orient_diff, 2 * np.pi - orient_diff)

        # Check if individuals face roughly *opposite* directions
        inv_orient_diff = np.pi - orient_diff
        angle_ok = inv_orient_diff < np.deg2rad(max_inv_orientation_diff_deg)
        meets = meets & angle_ok

    return meets


def _remove_bool_islands(
    arr: np.ndarray,
    structure_length: int = 25,
) -> np.ndarray:
    """Remove short boolean islands via morphological close then open."""
    structure = np.ones(structure_length, dtype=bool)
    # Closing first: fills small False gaps inside True regions
    return binary_opening(binary_closing(arr, structure=structure), structure=structure)


def _find_long_true_runs(
    arr: np.ndarray,
    min_run: int = 250,
    n_frame_padding: int = 10,
) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for True runs >= *min_run* frames.

    Each run is padded by *n_frame_padding* on both sides (clamped to array
    bounds).  Indices are relative to *arr*.
    """
    if not arr.any():
        return []

    edges = np.flatnonzero(np.diff(arr.astype(np.int8)))
    starts = np.r_[0, edges + 1]
    ends = np.r_[edges, len(arr) - 1]

    true_mask = arr[starts]
    true_starts = starts[true_mask]
    true_ends = ends[true_mask]
    lengths = true_ends - true_starts + 1

    keep = lengths >= min_run
    true_starts = true_starts[keep]
    true_ends = true_ends[keep]

    runs = []
    for s, e in zip(true_starts, true_ends):
        padded_start = max(0, s - n_frame_padding)
        padded_end = min(len(arr), e + 1 + n_frame_padding)
        runs.append((padded_start, padded_end))
    return runs


# ---------------------------------------------------------------------------
# Feature
# ---------------------------------------------------------------------------


@final
@register_feature
class PairInteractionFilter:
    """Detect pairwise interaction segments from trajectory data.

    For every unique pair of individuals in a sequence, tests per-frame
    distance and (optionally) angular criteria, applies morphological
    filtering, and extracts continuous interaction segments that meet a
    minimum duration.

    Output columns (one row per frame per interaction segment):
      - frame: frame number
      - id_a, id_b: individual IDs (id_a < id_b by convention)
      - interaction_id: integer label for the segment within this pair
      - interaction_start: first frame of this segment
      - interaction_end: last frame (exclusive) of this segment

    Params
    ------
    shift_dist : float
        Pixel shift along heading before distance check (default 15).
        Set to 0 to use raw positions without forward shift.
    max_dist : float
        Maximum shifted-position distance in pixels (default 40).
    require_facing : bool
        If True (default), require individuals to face each other
        (inverse orientation difference < ``max_inv_orientation_diff_deg``).
        Set to False for distance-only filtering.
    max_inv_orientation_diff_deg : float
        Max angle (degrees) between inverse orientations (default 80).
        Only used when ``require_facing=True``.
    min_run_frames : int
        Minimum continuous frames for a valid interaction (default 250).
    frame_padding : int
        Frames to pad before/after each segment (default 10).
    morphological_structure_size : int
        Structure element length for binary close/open (default 25).
        Set to 0 to disable morphological filtering.
    px_scale : float
        Scale factor applied to shift_dist and max_dist (default 1.0).
        Use to adjust for videos with different pixel resolutions.
    use_pixel_coords : bool
        If True, use poseX/poseY columns (pixel coordinates) for
        distance calculations instead of X/Y (world coordinates).
        Default True since thresholds are in pixel units.
    pose_head_index : int | None
        If set and use_pixel_coords is True, use this pose index
        as the position for distance calculations.
    """

    category = "per-frame"
    name = "pair-interaction-filter"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        shift_dist: float = 15.0
        max_dist: float = 40.0
        require_facing: bool = True
        max_inv_orientation_diff_deg: float = 80.0
        min_run_frames: int = 250
        frame_padding: int = 10
        morphological_structure_size: int = 25
        px_scale: float = 1.0
        use_pixel_coords: bool = True
        pose_head_index: int | None = None

    def __init__(
        self,
        inputs: PairInteractionFilter.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

    # --- State (stateless feature) ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    # --- Apply ---

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        order_col = resolve_order_col(df)
        ensure_columns(df, [C.id_col, C.orientation_col])

        p = self.params
        shift_dist = p.shift_dist * p.px_scale
        max_dist = p.max_dist * p.px_scale

        # Determine which columns to use for position
        if p.use_pixel_coords and p.pose_head_index is not None:
            x_col = f"poseX{p.pose_head_index}"
            y_col = f"poseY{p.pose_head_index}"
            ensure_columns(df, [x_col, y_col])
        elif p.use_pixel_coords:
            # Try pixel columns, fall back to X/Y
            x_col = "x_pixels" if "x_pixels" in df.columns else C.x_col
            y_col = "y_pixels" if "y_pixels" in df.columns else C.y_col
        else:
            x_col = C.x_col
            y_col = C.y_col

        ensure_columns(df, [x_col, y_col])

        # Sort by frame
        df = df.sort_values([C.id_col, order_col]).reset_index(drop=True)

        all_segments: list[pd.DataFrame] = []

        for _, gseq in df.groupby(C.seq_col) if C.seq_col in df.columns else [(None, df)]:
            ids = sorted(gseq[C.id_col].unique())
            if len(ids) < 2:
                continue

            for id_a, id_b in combinations(ids, 2):
                segments = self._detect_pair_interactions(
                    gseq, id_a, id_b, order_col, x_col, y_col,
                    shift_dist, max_dist,
                )
                if segments is not None and not segments.empty:
                    all_segments.append(segments)

        if not all_segments:
            return pd.DataFrame(
                columns=[
                    C.frame_col, "id_a", "id_b",
                    "interaction_id", "interaction_start", "interaction_end",
                ]
            )

        out = pd.concat(all_segments, ignore_index=True)
        # Attach group/sequence metadata if present
        for col in (C.group_col, C.seq_col):
            if col in df.columns:
                out[col] = df[col].iloc[0]
        return out

    def _detect_pair_interactions(
        self,
        gseq: pd.DataFrame,
        id_a,
        id_b,
        order_col: str,
        x_col: str,
        y_col: str,
        shift_dist: float,
        max_dist: float,
    ) -> pd.DataFrame | None:
        """Detect interaction segments for a single pair."""
        p = self.params

        df_a = gseq[gseq[C.id_col] == id_a][[order_col, x_col, y_col, C.orientation_col]]
        df_b = gseq[gseq[C.id_col] == id_b][[order_col, x_col, y_col, C.orientation_col]]

        if df_a.empty or df_b.empty:
            return None

        # Inner join on frame to align the two trajectories
        merged = df_a.merge(
            df_b,
            on=order_col,
            suffixes=("_a", "_b"),
            how="inner",
        )
        if merged.empty:
            return None

        frames = merged[order_col].to_numpy()
        x_a = merged[f"{x_col}_a"].to_numpy(dtype=float)
        y_a = merged[f"{y_col}_a"].to_numpy(dtype=float)
        orient_a = merged[f"{C.orientation_col}_a"].to_numpy(dtype=float)
        x_b = merged[f"{x_col}_b"].to_numpy(dtype=float)
        y_b = merged[f"{y_col}_b"].to_numpy(dtype=float)
        orient_b = merged[f"{C.orientation_col}_b"].to_numpy(dtype=float)

        # Per-frame interaction condition
        meets_cond = _check_interaction_conditions(
            x_a, y_a, orient_a,
            x_b, y_b, orient_b,
            shift_dist=shift_dist,
            max_dist=max_dist,
            max_inv_orientation_diff_deg=p.max_inv_orientation_diff_deg,
            require_facing=p.require_facing,
        )

        # Morphological filtering
        if p.morphological_structure_size > 0:
            meets_cond = _remove_bool_islands(meets_cond, p.morphological_structure_size)

        # Find long runs
        runs = _find_long_true_runs(
            meets_cond,
            min_run=p.min_run_frames,
            n_frame_padding=p.frame_padding,
        )

        if not runs:
            return None

        # Build output DataFrame — one row per frame per interaction segment
        rows = []
        for seg_id, (start_idx, end_idx) in enumerate(runs):
            seg_frames = frames[start_idx:end_idx]
            for f in seg_frames:
                rows.append({
                    C.frame_col: int(f),
                    "id_a": id_a,
                    "id_b": id_b,
                    "interaction_id": seg_id,
                    "interaction_start": int(frames[start_idx]),
                    "interaction_end": int(frames[min(end_idx - 1, len(frames) - 1)]),
                })

        return pd.DataFrame(rows)
