"""Bidirectional conversion between mosaic DataFrames and movement xarray Datasets.

Pose columns are read here and not through
:func:`~mosaic.core.pose_columns.keypoint_arrays`, because a movement dataset is a
different computation from the arrays that function returns for one individual.
It places every individual on one frame axis, the union of their frames. It scores
an absent point NaN, and a point with a finite position and no score column 1,
keypoint by keypoint. It can add the body center as a ``centroid`` keypoint.
"""

from __future__ import annotations

from typing import Any, Final, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mosaic.core.pose_columns import pose_column_pairs
from mosaic.core.pipeline.column_contract import (
    POSE_X,
    AnyOf,
    Column,
    Provides,
    Requirement,
)
from mosaic.core.pipeline.types import COLUMNS
from mosaic.optional_dependency import OptionalModule

MOVEMENT: Final = OptionalModule(
    "movement", "movement", "the movement-library filtering and smoothing features"
)
"""The movement library, a prerequisite of every feature in this package."""

MOVEMENT_REQUIRES: Final[tuple[Requirement, ...]] = (
    Column(COLUMNS.id_col),
    Column(COLUMNS.frame_col),
    AnyOf((Column(COLUMNS.x_col), POSE_X)),
)
"""What :func:`movement_arrays` reads: the individual, the frame and a keypoint.

A keypoint is the centroid ``X``/``Y`` or a pose pair. With ``include_centroid``
false only a pose pair counts, and :func:`movement_arrays` raises without one.
"""

MOVEMENT_PROVIDES: Final = Provides(passthrough=True)
"""What :func:`write_positions` returns: its input, with positions replaced."""


class MovementMetadata(TypedDict):
    """What :func:`write_positions` needs of the arrays it writes back.

    Attributes:
        individual_ids: The ids in the order of the arrays' last axis.
        frame_index: The frames in the order of the arrays' first axis.
        include_centroid: Whether keypoint 0 is the ``centroid``.
        pose_pairs: The ``(poseX, poseY)`` column pairs, after the centroid.
        has_confidence: Whether any ``poseP`` column is present.
        conf_cols: The ``poseP`` column of each pose pair, ``None`` where absent.
        keypoint_names: The names of the keypoints, the centroid first.
    """

    individual_ids: list[Any]
    frame_index: list[Any]
    include_centroid: bool
    pose_pairs: list[tuple[str, str]]
    has_confidence: bool
    conf_cols: list[str | None]
    keypoint_names: list[str]


def movement_arrays(
    df: pd.DataFrame,
    keypoint_names: list[str] | None,
    include_centroid: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], MovementMetadata]:
    """Place a mosaic tracks DataFrame on the arrays of a movement dataset.

    This reads no movement library. Each individual is placed on one frame axis,
    the sorted union of the table's frames, and a frame an individual lacks is NaN.
    An infinite position is read as NaN.

    Args:
        df: Mosaic tracks with ``id``, ``frame``, ``X``/``Y`` and ``poseX<k>``,
            ``poseY<k>`` columns, and ``poseP<k>`` scores where present.
        keypoint_names: Names for the pose keypoints, one per pose pair. ``None``
            names them ``keypoint_0``, ``keypoint_1`` and so on.
        include_centroid: Whether to add ``X``/``Y`` as the keypoint ``centroid``,
            before the pose keypoints.

    Returns:
        The position array of shape ``(frames, 2, keypoints, individuals)``, the
        confidence array of shape ``(frames, keypoints, individuals)``, and the
        metadata :func:`write_positions` reads. Both arrays are float64. A
        confidence is the ``poseP`` column where there is one, and otherwise 1
        where the point's position is finite and NaN where it is not.

    Raises:
        ValueError: If ``keypoint_names`` does not have one entry per pose pair, or
            there is no keypoint at all.
    """
    id_col = COLUMNS.id_col
    frame_col = COLUMNS.frame_col
    x_col = COLUMNS.x_col
    y_col = COLUMNS.y_col

    # --- Detect pose columns ---
    pose_pairs = pose_column_pairs(df.columns)
    n_pose_kp = len(pose_pairs)

    # Detect confidence columns (poseP0, poseP1, ...)
    conf_cols: list[str | None] = []
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
            conf_col = conf_cols[kp_i]
            if has_confidence and conf_col is not None:
                confidence[fidxs, ki, ind_idx] = sub[conf_col].to_numpy(dtype=float)
            else:
                # Default confidence = 1.0 where position is valid
                px = position[fidxs, 0, ki, ind_idx]
                py = position[fidxs, 1, ki, ind_idx]
                valid = np.isfinite(px) & np.isfinite(py)
                confidence[fidxs, ki, ind_idx] = np.where(valid, 1.0, np.nan)

    # An infinite position is a missing one
    position = np.where(np.isinf(position), np.nan, position)

    metadata: MovementMetadata = {
        "individual_ids": individual_ids,
        "frame_index": frames,
        "include_centroid": has_centroid,
        "pose_pairs": pose_pairs,
        "has_confidence": has_confidence,
        "conf_cols": conf_cols,
        "keypoint_names": kp_names,
    }
    return position, confidence, metadata


def write_positions(
    original_df: pd.DataFrame,
    position: NDArray[np.floating[Any]],
    metadata: MovementMetadata,
    confidence: NDArray[np.floating[Any]] | None = None,
) -> pd.DataFrame:
    """Write position arrays back into a mosaic DataFrame.

    This reads no movement library. Each row takes the value at its own frame and
    individual, so rows in any order, and an individual that lacks a frame, are
    written where they belong.

    Args:
        original_df: The mosaic DataFrame the arrays came from.
        position: The positions, of shape ``(frames, 2, keypoints, individuals)``
            in the layout :func:`movement_arrays` returned.
        metadata: The metadata :func:`movement_arrays` returned.
        confidence: The scores, of shape ``(frames, keypoints, individuals)``. Given,
            they overwrite the ``poseP`` columns that ``metadata["conf_cols"]``
            names. Unset, those columns are as they were.

    Returns:
        A copy of ``original_df`` with the position columns replaced.
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

    centroid_offset = 1 if has_centroid else 0

    result = original_df.copy()

    for ind_id, sub_idx in result.groupby(id_col, sort=False).groups.items():
        ind_idx = id_to_idx[ind_id]
        rows = result.loc[sub_idx]
        fidxs = rows[frame_col].map(frame_to_idx).to_numpy()

        # Centroid
        if has_centroid:
            if x_col in result.columns:
                result.loc[sub_idx, x_col] = position[fidxs, 0, 0, ind_idx]
            if y_col in result.columns:
                result.loc[sub_idx, y_col] = position[fidxs, 1, 0, ind_idx]

        # Pose keypoints
        for kp_i, (xcol, ycol) in enumerate(pose_pairs):
            ki = kp_i + centroid_offset
            result.loc[sub_idx, xcol] = position[fidxs, 0, ki, ind_idx]
            result.loc[sub_idx, ycol] = position[fidxs, 1, ki, ind_idx]

            # Confidence
            conf_col = conf_cols[kp_i]
            if confidence is not None and conf_col is not None:
                result.loc[sub_idx, conf_col] = confidence[fidxs, ki, ind_idx]

    return result


def to_movement_dataset(
    df: pd.DataFrame,
    fps: float | None = None,
    keypoint_names: list[str] | None = None,
    include_centroid: bool = True,
) -> tuple[Any, MovementMetadata]:
    """Convert a mosaic tracks DataFrame to a movement xarray Dataset.

    Args:
        df: Mosaic tracks DataFrame with columns like X, Y, poseX0..N, poseY0..N,
            id, frame, etc.
        fps: Frames per second. If None, the time dimension uses frame numbers.
        keypoint_names: Names for the pose keypoints. If None, defaults to
            "keypoint_0", etc.
        include_centroid: Whether to include the centroid (X, Y) as an additional
            keypoint named "centroid".

    Returns:
        The movement poses Dataset with dimensions (time, space, keypoints,
        individuals), built from float32 copies of :func:`movement_arrays`, and the
        metadata that :func:`from_movement_dataset` converts back with.
    """
    _ = MOVEMENT.load()
    from movement.io.load_poses import from_numpy

    position, confidence, metadata = movement_arrays(
        df, keypoint_names, include_centroid
    )
    ds = from_numpy(
        position_array=position.astype(np.float32),
        confidence_array=confidence.astype(np.float32),
        individual_names=[str(i) for i in metadata["individual_ids"]],
        keypoint_names=metadata["keypoint_names"],
        fps=fps,
        source_software="mosaic",
    )
    return ds, metadata


def from_movement_dataset(
    ds: Any,
    original_df: pd.DataFrame,
    metadata: MovementMetadata,
    update_confidence: bool = False,
) -> pd.DataFrame:
    """Merge a movement xarray Dataset back into a mosaic DataFrame.

    Overwrites X/Y and poseX/poseY columns in a copy of ``original_df``
    with the (smoothed/filtered) values from the Dataset.

    Args:
        ds: Movement Dataset with ``position`` and ``confidence`` data variables.
        original_df: The original mosaic DataFrame to merge into.
        metadata: Metadata returned by ``to_movement_dataset``.
        update_confidence: Whether to also overwrite poseP columns from the
            Dataset's confidence values.

    Returns:
        Copy of ``original_df`` with position columns replaced.
    """
    position = ds["position"].values  # (n_frames, 2, n_keypoints, n_individuals)
    confidence = ds["confidence"].values if update_confidence else None
    return write_positions(original_df, position, metadata, confidence)
