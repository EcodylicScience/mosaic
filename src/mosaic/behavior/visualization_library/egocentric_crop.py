"""Egocentric video crop generation.

This module provides the EgocentricCrop feature class for generating
animal-centered video crops, optionally rotated to align heading with +x axis.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import cv2

from mosaic.core.dataset import register_feature
from .helpers import (
    _merge_params,
    safe_crop_with_padding,
    compute_heading_angle,
    create_video_writer,
    infer_angle_degrees,
)


@register_feature
class EgocentricCrop:
    """
    Generate egocentric (animal-centered) video crops.

    Centers the view on a target individual (or all individuals if target_id=None),
    optionally rotating to align the animal's heading with the +x axis.
    Can output as video file or individual frame images.

    Parameters
    ----------
    target_id : Any, optional
        ID of the individual to center on. If None, processes ALL individuals
        found in the tracks data, creating separate outputs for each.
    center_mode : str or int
        How to compute the center point:
        - "default": mean of all pose points poseX0..N/poseY0..N (pixel coords)
        - "pose0" or 0: use poseX0/poseY0 (typically head/nose)
        - int: use specific pose point index
    crop_size : tuple of (int, int)
        Output crop dimensions as (width, height) in pixels
    rotate_to_heading : bool
        If True, rotate crop so animal's heading aligns with +x axis
    heading_points : tuple of (int, int)
        (neck_idx, tail_idx) pose point indices for heading computation.
        Heading points FROM tail TO neck (direction animal is facing).
    margin_factor : float
        Extra margin for rotation (1.5 = 50% larger pre-crop before rotation)
    output_mode : str
        Output format:
        - "video": single video file per individual
        - "frames": individual frame images per individual
        - "both": video + frames
    output_fps : float, optional
        Output video FPS. If None, uses source video FPS.
    frame_format : str
        Format for frame images ("png" or "jpg")
    background_color : int
        Padding color for out-of-bounds regions (0=black, 255=white)

    Examples
    --------
    Process a single individual:
    >>> crop = EgocentricCrop(params={"target_id": 0, "crop_size": (256, 256)})
    >>> dataset.run_feature(crop, sequences=["hex_3"])

    Process all individuals:
    >>> crop = EgocentricCrop(params={"target_id": None, "output_mode": "video"})
    >>> dataset.run_feature(crop, sequences=["hex_3"])
    """

    name = "egocentric-crop"
    version = "0.1"
    parallelizable = False  # Video I/O is sequential
    output_type = "viz"

    _defaults = dict(
        # Target individual
        target_id=None,  # None = process all IDs

        # Centering
        center_mode="default",  # "default", "pose0", or int
        pose_x_prefix="poseX",
        pose_y_prefix="poseY",
        pose_n=7,

        # Crop geometry
        crop_size=(256, 256),  # (width, height)
        rotate_to_heading=True,
        heading_points=(3, 6),  # (neck_idx, tail_idx) for heading
        margin_factor=1.5,  # Extra margin for rotation

        # Output
        output_mode="video",  # "video", "frames", "both"
        output_fps=None,
        frame_format="png",

        # Processing
        interpolation=cv2.INTER_LINEAR,
        background_color=0,  # Padding color (0=black, 255=white)

        # Column names
        id_col="id",
        frame_col="frame",
        seq_col="sequence",
        group_col="group",

        # Optional: if tracks have angle column, use it instead of computing
        angle_col=None,  # e.g., "ANGLE" if available
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = _merge_params(params, self._defaults)
        self._ds = None
        self._scope_filter = {}

        # Storage settings (for feature pipeline integration)
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = False
        self.skip_existing_outputs = False

    # ----------------------- Dataset hooks -----------------------

    def bind_dataset(self, ds):
        """Called by Dataset.run_feature before any fit/transform."""
        self._ds = ds

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        """Receive scope constraints from run_feature."""
        self._scope_filter = scope or {}

    # ----------------------- Fit protocol ------------------------

    def needs_fit(self) -> bool:
        return False

    def supports_partial_fit(self) -> bool:
        return False

    def fit(self, X_iter) -> None:
        pass

    def partial_fit(self, df: pd.DataFrame) -> None:
        pass

    def finalize_fit(self) -> None:
        pass

    def save_model(self, path: Path) -> None:
        pass

    def load_model(self, path: Path) -> None:
        pass

    # ----------------------- Core logic --------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single sequence's tracks to generate egocentric crop video/frames.

        Parameters
        ----------
        df : pd.DataFrame
            Tracks DataFrame for a single sequence

        Returns
        -------
        pd.DataFrame
            Metadata DataFrame with crop info per frame/id
        """
        if df is None or df.empty:
            return pd.DataFrame()

        p = self.params
        group = str(df[p["group_col"]].iloc[0]) if p["group_col"] in df.columns else ""
        sequence = str(df[p["seq_col"]].iloc[0]) if p["seq_col"] in df.columns else ""

        # Resolve video paths (supports multi-video sequences)
        video_paths = self._ds.resolve_media_paths(group, sequence)

        # Determine which IDs to process
        target_id = p["target_id"]
        if target_id is None:
            # Process all unique IDs
            unique_ids = df[p["id_col"]].dropna().unique()
            all_metadata = []
            for uid in unique_ids:
                df_target = df[df[p["id_col"]] == uid].copy()
                if df_target.empty:
                    continue
                metadata = self._process_single_id(video_paths, df_target, group, sequence, uid)
                all_metadata.append(metadata)
            if all_metadata:
                return pd.concat(all_metadata, ignore_index=True)
            return pd.DataFrame()
        else:
            # Process single ID
            df_target = df[df[p["id_col"]] == target_id].copy()
            if df_target.empty:
                raise ValueError(f"No data for target_id={target_id}")
            return self._process_single_id(video_paths, df_target, group, sequence, target_id)

    # ----------------------- Internal methods --------------------

    def _get_center(self, row: pd.Series) -> Tuple[float, float]:
        """Extract center point (in pixel coords) from a tracks row."""
        p = self.params
        mode = p["center_mode"]

        if mode == "default":
            # Average all available pose points (these are in pixel coordinates).
            # X/Y and X#wcentroid/Y#wcentroid are in real-world units (e.g. cm)
            # and cannot be used directly for video cropping.
            xs, ys = [], []
            for i in range(p["pose_n"]):
                px = row.get(f"{p['pose_x_prefix']}{i}")
                py = row.get(f"{p['pose_y_prefix']}{i}")
                if px is not None and py is not None and np.isfinite(px) and np.isfinite(py):
                    xs.append(px)
                    ys.append(py)
            if not xs:
                return (np.nan, np.nan)
            return (float(np.mean(xs)), float(np.mean(ys)))
        elif mode == "pose0" or isinstance(mode, int):
            idx = 0 if mode == "pose0" else int(mode)
            x = row.get(f"{p['pose_x_prefix']}{idx}")
            y = row.get(f"{p['pose_y_prefix']}{idx}")
            if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
                return (np.nan, np.nan)
            return (float(x), float(y))
        else:
            raise ValueError(f"Unknown center_mode: {mode}. Use 'default', 'pose0', or an int pose index.")

    def _get_heading_angle(self, row: pd.Series) -> float:
        """Compute heading angle from anatomical landmarks or angle column."""
        p = self.params

        # If angle column is specified, use it
        if p["angle_col"] and p["angle_col"] in row.index:
            angle = row.get(p["angle_col"])
            if angle is not None and np.isfinite(angle):
                return float(angle)

        # Otherwise compute from neck/tail pose points
        neck_idx, tail_idx = p["heading_points"]

        neck_x = row.get(f"{p['pose_x_prefix']}{neck_idx}")
        neck_y = row.get(f"{p['pose_y_prefix']}{neck_idx}")
        tail_x = row.get(f"{p['pose_x_prefix']}{tail_idx}")
        tail_y = row.get(f"{p['pose_y_prefix']}{tail_idx}")

        if any(v is None or not np.isfinite(v) for v in [neck_x, neck_y, tail_x, tail_y]):
            return 0.0

        return compute_heading_angle((neck_x, neck_y), (tail_x, tail_y))

    def _extract_egocentric_crop(
        self,
        frame: np.ndarray,
        center: Tuple[float, float],
        angle: float,
    ) -> np.ndarray:
        """
        Extract an egocentric crop from a video frame.

        Parameters
        ----------
        frame : np.ndarray
            Source video frame (H, W, C)
        center : Tuple[float, float]
            (cx, cy) center point in pixel coordinates
        angle : float
            Heading angle in radians (0 = facing right/+x)

        Returns
        -------
        np.ndarray
            Cropped (and optionally rotated) frame
        """
        p = self.params
        crop_w, crop_h = p["crop_size"]
        cx, cy = center

        if not np.isfinite(cx) or not np.isfinite(cy):
            # Return blank frame if center is invalid
            return np.full((crop_h, crop_w, 3), p["background_color"], dtype=np.uint8)

        if p["rotate_to_heading"]:
            # Rotation approach: rotate around animal center, then crop
            # atan2 in image coords (Y-down) already flips sign vs math convention,
            # and getRotationMatrix2D uses math convention (CCW-positive, Y-up),
            # so the two negations cancel — just convert to degrees directly.
            angle_deg = np.degrees(angle)

            # Compute larger pre-crop to account for rotation
            margin = p["margin_factor"]
            pre_crop_size = int(max(crop_w, crop_h) * margin)

            # First extract a larger centered crop
            pre_crop = safe_crop_with_padding(
                frame,
                (int(cx), int(cy)),
                (pre_crop_size, pre_crop_size),
                pad_value=p["background_color"]
            )

            # Rotate around center of pre-crop
            pre_center = (pre_crop_size / 2.0, pre_crop_size / 2.0)
            M = cv2.getRotationMatrix2D(pre_center, angle_deg, 1.0)
            rotated = cv2.warpAffine(
                pre_crop, M,
                (pre_crop_size, pre_crop_size),
                flags=p["interpolation"],
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(p["background_color"],) * 3
            )

            # Final center crop from rotated image
            return safe_crop_with_padding(
                rotated,
                (pre_crop_size // 2, pre_crop_size // 2),
                (crop_w, crop_h),
                pad_value=p["background_color"]
            )
        else:
            # Simple centered crop without rotation
            return safe_crop_with_padding(
                frame,
                (int(cx), int(cy)),
                (crop_w, crop_h),
                pad_value=p["background_color"]
            )

    def _process_single_id(
        self,
        video_paths: list[Path],
        df_target: pd.DataFrame,
        group: str,
        sequence: str,
        target_id: Any,
    ) -> pd.DataFrame:
        """
        Process video(s) to generate egocentric crops for a single individual.

        Parameters
        ----------
        video_paths : list[Path]
            Ordered list of video paths for this sequence.

        Returns metadata DataFrame with crop info per frame.
        """
        from mosaic.media.video_io import MultiVideoReader

        p = self.params

        # Sort by frame
        df_target = df_target.sort_values(p["frame_col"]).reset_index(drop=True)

        # Open video(s) via MultiVideoReader
        reader = MultiVideoReader(video_paths)
        output_fps = p["output_fps"] or reader.fps

        # Build frame -> row lookup
        frame_to_row = {int(row[p["frame_col"]]): row for _, row in df_target.iterrows()}

        # Prepare output paths
        run_root = self._get_run_root(group, sequence)
        run_root.mkdir(parents=True, exist_ok=True)

        # Determine output crop size
        crop_w, crop_h = p["crop_size"]

        # Initialize outputs
        writer = None
        if p["output_mode"] in ("video", "both"):
            video_out_path = run_root / f"egocentric_id{target_id}.mp4"
            writer = create_video_writer(video_out_path, output_fps, (crop_w, crop_h))

        frames_dir = None
        if p["output_mode"] in ("frames", "both"):
            frames_dir = run_root / f"frames_id{target_id}"
            frames_dir.mkdir(exist_ok=True)

        # Check if angle column exists and infer if degrees
        angle_is_degrees = False
        if p["angle_col"] and p["angle_col"] in df_target.columns:
            angle_is_degrees = infer_angle_degrees(df_target[p["angle_col"]])

        # Process frames
        metadata_rows = []
        frame_idx = 0
        total_frames = reader.total_frames

        try:
            while True:
                ret, frame = reader.read()
                if not ret:
                    break

                row = frame_to_row.get(frame_idx)
                if row is not None:
                    center = self._get_center(row)
                    if p["rotate_to_heading"]:
                        angle = self._get_heading_angle(row)
                        # Convert to radians if needed
                        if angle_is_degrees:
                            angle = np.radians(angle)
                    else:
                        angle = 0.0

                    crop = self._extract_egocentric_crop(frame, center, angle)

                    if writer is not None:
                        writer.write(crop)

                    if frames_dir is not None:
                        frame_path = frames_dir / f"frame_{frame_idx:06d}.{p['frame_format']}"
                        cv2.imwrite(str(frame_path), crop)

                    metadata_rows.append({
                        "frame": frame_idx,
                        "center_x": center[0],
                        "center_y": center[1],
                        "heading_angle": angle,
                        "target_id": target_id,
                        "group": group,
                        "sequence": sequence,
                    })

                frame_idx += 1

                # Progress logging for long videos
                if frame_idx % 1000 == 0:
                    print(f"  [egocentric-crop] id={target_id}: {frame_idx}/{total_frames} frames")

        finally:
            reader.close()
            if writer is not None:
                writer.release()

        print(f"  [egocentric-crop] id={target_id}: processed {len(metadata_rows)} frames -> {run_root}")

        return pd.DataFrame(metadata_rows)

    def _get_run_root(self, group: str, sequence: str) -> Path:
        """Get the output directory for this run."""
        # Use the dataset's feature output structure
        features_root = Path(self._ds.get_root("features"))
        # This will be set properly by run_feature, but provide a fallback
        return features_root / self.name / f"{group}__{sequence}"
