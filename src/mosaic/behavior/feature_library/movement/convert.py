"""Bidirectional conversion between mosaic DataFrames and movement xarray Datasets."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from mosaic.core.pipeline.types import COLUMNS


def _pose_column_pairs(columns) -> list[tuple[str, str]]:
    """Extract (poseX*, poseY*) column pairs from column names."""
    pose_pairs = []
    xs = [c for c in columns if c.startswith("poseX")]
    for x_col in sorted(xs):
        idx = x_col[5:]
        y_col = f"poseY{idx}"
        if y_col in columns:
            pose_pairs.append((x_col, y_col))
    return pose_pairs


def _ensure_movement():
    """Lazily import movement, raising a clear error if not installed."""
    try:
        import movement  # noqa: F401

        return movement
    except ImportError:
        raise ImportError(
            "The 'movement' package is required for this feature. "
            "Install with: pip install 'mosaic[movement]'"
        ) from None


def to_movement_dataset(
    df: pd.DataFrame,
    fps: float | None = None,
    keypoint_names: list[str] | None = None,
    include_centroid: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Convert a mosaic tracks DataFrame to a movement xarray Dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Mosaic tracks DataFrame with columns like X, Y, poseX0..N, poseY0..N,
        id, frame, etc.
    fps : float, optional
        Frames per second. If None, the time dimension uses frame numbers.
    keypoint_names : list[str], optional
        Names for the pose keypoints. If None, defaults to "keypoint_0", etc.
    include_centroid : bool
        Whether to include the centroid (X, Y) as an additional keypoint
        named "centroid". Default True.

    Returns
    -------
    ds : xarray.Dataset
        movement poses Dataset with dimensions (time, space, keypoints, individuals).
    metadata : dict
        Metadata needed by ``from_movement_dataset`` to convert back:
        ``individual_ids``, ``frame_index``, ``include_centroid``, ``pose_pairs``.
    """
    _ensure_movement()
    from movement.io.load_poses import from_numpy

    id_col = COLUMNS.id_col
    frame_col = COLUMNS.frame_col
    x_col = COLUMNS.x_col
    y_col = COLUMNS.y_col

    # --- Detect pose columns ---
    pose_pairs = _pose_column_pairs(df.columns)
    n_pose_kp = len(pose_pairs)

    # Detect confidence columns (poseP0, poseP1, ...)
    conf_cols = []
    for i in range(n_pose_kp):
        pcol = f"poseP{i}"
        if pcol in df.columns:
            conf_cols.append(pcol)
        else:
            conf_cols.append(None)
    has_confidence = any(c is not None for c in conf_cols)

    # --- Build keypoint list ---
    kp_names: list[str] = []
    if include_centroid and x_col in df.columns and y_col in df.columns:
        kp_names.append("centroid")
    if keypoint_names is not None:
        if len(keypoint_names) != n_pose_kp:
            raise ValueError(
                f"keypoint_names has {len(keypoint_names)} entries but "
                f"found {n_pose_kp} pose column pairs"
            )
        kp_names.extend(keypoint_names)
    else:
        kp_names.extend([f"keypoint_{i}" for i in range(n_pose_kp)])

    n_keypoints = len(kp_names)
    if n_keypoints == 0:
        raise ValueError(
            "No keypoints to convert: no pose columns found and "
            "include_centroid=False or X/Y columns missing."
        )

    # --- Individuals and frames ---
    individual_ids = sorted(df[id_col].unique(), key=str)
    n_individuals = len(individual_ids)
    id_to_idx = {v: i for i, v in enumerate(individual_ids)}

    frames = sorted(df[frame_col].unique())
    n_frames = len(frames)
    frame_to_idx = {f: i for i, f in enumerate(frames)}

    # --- Allocate arrays ---
    position = np.full(
        (n_frames, 2, n_keypoints, n_individuals), np.nan, dtype=np.float64
    )
    confidence = np.full(
        (n_frames, n_keypoints, n_individuals), np.nan, dtype=np.float64
    )

    # --- Fill arrays per individual ---
    has_centroid = "centroid" in kp_names
    centroid_offset = 1 if has_centroid else 0

    for ind_id, sub in df.groupby(id_col, sort=False):
        ind_idx = id_to_idx[ind_id]
        fidxs = sub[frame_col].map(frame_to_idx).to_numpy()

        # Centroid
        if has_centroid:
            if x_col in sub.columns:
                position[fidxs, 0, 0, ind_idx] = sub[x_col].to_numpy(dtype=float)
            if y_col in sub.columns:
                position[fidxs, 1, 0, ind_idx] = sub[y_col].to_numpy(dtype=float)
            # Centroid confidence: 1.0 where position is not NaN
            cx = position[fidxs, 0, 0, ind_idx]
            cy = position[fidxs, 1, 0, ind_idx]
            centroid_valid = np.isfinite(cx) & np.isfinite(cy)
            confidence[fidxs, 0, ind_idx] = np.where(centroid_valid, 1.0, np.nan)

        # Pose keypoints
        for kp_i, (xcol, ycol) in enumerate(pose_pairs):
            ki = kp_i + centroid_offset
            position[fidxs, 0, ki, ind_idx] = sub[xcol].to_numpy(dtype=float)
            position[fidxs, 1, ki, ind_idx] = sub[ycol].to_numpy(dtype=float)

            # Confidence
            if has_confidence and conf_cols[kp_i] is not None:
                confidence[fidxs, ki, ind_idx] = sub[conf_cols[kp_i]].to_numpy(
                    dtype=float
                )
            else:
                # Default confidence = 1.0 where position is valid
                px = position[fidxs, 0, ki, ind_idx]
                py = position[fidxs, 1, ki, ind_idx]
                valid = np.isfinite(px) & np.isfinite(py)
                confidence[fidxs, ki, ind_idx] = np.where(valid, 1.0, np.nan)

    # --- Build movement Dataset ---
    ds = from_numpy(
        position_array=position.astype(np.float32),
        confidence_array=confidence.astype(np.float32),
        individual_names=[str(i) for i in individual_ids],
        keypoint_names=kp_names,
        fps=fps,
        source_software="mosaic",
    )

    metadata = {
        "individual_ids": individual_ids,
        "frame_index": frames,
        "include_centroid": has_centroid,
        "pose_pairs": pose_pairs,
        "has_confidence": has_confidence,
        "conf_cols": conf_cols,
    }
    return ds, metadata


def from_movement_dataset(
    ds: Any,
    original_df: pd.DataFrame,
    metadata: dict[str, Any],
    update_confidence: bool = False,
) -> pd.DataFrame:
    """Merge a movement xarray Dataset back into a mosaic DataFrame.

    Overwrites X/Y and poseX/poseY columns in a copy of ``original_df``
    with the (smoothed/filtered) values from the Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        movement Dataset with ``position`` and ``confidence`` data variables.
    original_df : pd.DataFrame
        The original mosaic DataFrame to merge into.
    metadata : dict
        Metadata returned by ``to_movement_dataset``.
    update_confidence : bool
        Whether to also overwrite poseP columns from the Dataset's
        confidence values. Default False.

    Returns
    -------
    pd.DataFrame
        Copy of ``original_df`` with position columns replaced.
    """
    id_col = COLUMNS.id_col
    frame_col = COLUMNS.frame_col
    x_col = COLUMNS.x_col
    y_col = COLUMNS.y_col

    individual_ids = metadata["individual_ids"]
    frame_index = metadata["frame_index"]
    has_centroid = metadata["include_centroid"]
    pose_pairs = metadata["pose_pairs"]
    conf_cols = metadata["conf_cols"]

    id_to_idx = {v: i for i, v in enumerate(individual_ids)}
    frame_to_idx = {f: i for i, f in enumerate(frame_index)}

    # Extract numpy arrays from xarray
    pos_arr = ds["position"].values  # (n_frames, 2, n_keypoints, n_individuals)
    conf_arr = ds["confidence"].values  # (n_frames, n_keypoints, n_individuals)

    centroid_offset = 1 if has_centroid else 0

    result = original_df.copy()

    for ind_id, sub_idx in result.groupby(id_col, sort=False).groups.items():
        ind_idx = id_to_idx[ind_id]
        rows = result.loc[sub_idx]
        fidxs = rows[frame_col].map(frame_to_idx).to_numpy()

        # Centroid
        if has_centroid:
            if x_col in result.columns:
                result.loc[sub_idx, x_col] = pos_arr[fidxs, 0, 0, ind_idx]
            if y_col in result.columns:
                result.loc[sub_idx, y_col] = pos_arr[fidxs, 1, 0, ind_idx]

        # Pose keypoints
        for kp_i, (xcol, ycol) in enumerate(pose_pairs):
            ki = kp_i + centroid_offset
            result.loc[sub_idx, xcol] = pos_arr[fidxs, 0, ki, ind_idx]
            result.loc[sub_idx, ycol] = pos_arr[fidxs, 1, ki, ind_idx]

            # Confidence
            if update_confidence and conf_cols[kp_i] is not None:
                result.loc[sub_idx, conf_cols[kp_i]] = conf_arr[fidxs, ki, ind_idx]

    return result
