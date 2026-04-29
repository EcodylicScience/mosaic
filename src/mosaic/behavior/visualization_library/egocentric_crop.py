"""Egocentric video crop generation.

This module provides the EgocentricCrop feature class for generating
animal-centered video crops, optionally rotated to align heading with +x axis.

Supports optional body masking (elliptical mask to isolate focal individual),
CLAHE contrast enhancement, grayscale conversion, center offset along heading,
and keypoint coordinate transformation into crop space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.pipeline._utils import Scope
from mosaic.core.pipeline.types import COLUMNS, Inputs, Params, PoseConfig, TrackInput
from mosaic.core.pipeline.types.feature import DependencyLookup

from ..feature_library.registry import register_feature
from .helpers import (
    compute_heading_angle,
    create_video_writer,
    infer_angle_degrees,
    safe_crop_with_padding,
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
    center_offset_px : float
        Pixel offset along heading direction from computed center (default 0).
        Positive shifts forward (toward head). Useful for centering on
        specific body parts, e.g. 35 for body center in bees.
    body_mask : bool
        If True, apply elliptical body mask to isolate the focal individual.
    body_mask_length_px : int
        Length (semi-major axis) of the body mask ellipse in pixels.
    body_mask_width_px : int
        Width (semi-minor axis) of the body mask ellipse in pixels.
    use_clahe : bool
        If True, apply CLAHE (Contrast Limited Adaptive Histogram
        Equalization) to improve contrast in crops.
    clahe_clip_limit : float
        CLAHE clip limit parameter.
    clahe_tile_grid_size : int
        CLAHE tile grid size (both dimensions).
    grayscale : bool
        If True, convert output to single-channel grayscale.
    transform_keypoints : bool
        If True, transform pose keypoint coordinates into crop space and
        include them in the metadata output as poseX<i>_crop, poseY<i>_crop.
    output_mode : str
        Output format:
        - "video": single video file per individual
        - "frames": individual frame images per individual
        - "both": video + frames
    output_fps : float, optional
        Output video FPS. If None, uses source video FPS.
    output_root : str, optional
        Override output directory. If None, outputs to media/egocentric_crops/.
    frame_format : str
        Format for frame images ("png" or "jpg")
    background_color : int
        Padding color for out-of-bounds regions (0=black, 255=white)

    Examples
    --------
    Process a single individual:
    >>> crop = EgocentricCrop(params={"target_id": 0, "crop_size": (256, 256)})
    >>> dataset.run_feature(crop, sequences=["hex_3"])

    Bee-style crop with body masking and CLAHE:
    >>> crop = EgocentricCrop(params={
    ...     "crop_size": (192, 192),
    ...     "center_offset_px": 35.0,
    ...     "body_mask": True,
    ...     "use_clahe": True,
    ...     "grayscale": True,
    ...     "angle_col": "ANGLE",
    ... })
    >>> dataset.run_feature(crop, sequences=["hex_3"])
    """

    category = "viz"
    name = "egocentric-crop"
    version = "0.2"
    parallelizable = False  # Video I/O is sequential
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        target_id: int | None = None
        center_mode: str | int = "default"
        pose: PoseConfig = Field(default_factory=PoseConfig)
        crop_size: tuple[int, int] = (256, 256)
        rotate_to_heading: bool = True
        heading_points: tuple[int, int] = (3, 6)
        margin_factor: float = 1.5
        # Center offset along heading direction (pixels)
        center_offset_px: float = 0.0
        # Body masking
        body_mask: bool = False
        body_mask_length_px: int = 96
        body_mask_width_px: int = 64
        # CLAHE contrast enhancement
        use_clahe: bool = False
        clahe_clip_limit: float = 2.0
        clahe_tile_grid_size: int = 25
        # Grayscale output
        grayscale: bool = False
        # Keypoint transformation
        transform_keypoints: bool = False
        # Output settings
        output_mode: str = "video"
        output_fps: float | None = None
        output_root: str | None = None
        frame_format: str = "png"
        interpolation: int = 1  # cv2.INTER_LINEAR
        background_color: int = 0
        angle_col: str | None = None

    def __init__(
        self,
        inputs: EgocentricCrop.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._ds = None
        self._scope: Scope = Scope()
        self._run_root: Path | None = None
        self._clahe = None  # lazily constructed; reused across frames

        # Storage settings (for feature pipeline integration)
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = False
        self.skip_existing_outputs = False

    def _get_clahe(self):
        """Return a single CLAHE instance, built once from params.

        Building the CLAHE object costs O(tileGridSize^2) for LUT init, so
        we construct it once per feature-instance rather than per frame.
        """
        if self._clahe is None:
            p = self.params
            self._clahe = cv2.createCLAHE(
                clipLimit=p.clahe_clip_limit,
                tileGridSize=(p.clahe_tile_grid_size, p.clahe_tile_grid_size),
            )
        return self._clahe

    # ----------------------- Dataset hooks -----------------------

    def bind_dataset(self, ds):
        """Called by Dataset.run_feature before any fit/transform."""
        self._ds = ds

    def set_scope(self, scope: Scope) -> None:
        """Receive scope constraints from run_feature."""
        self._scope = scope

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

    # ----------------------- Feature protocol ---------------------

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._run_root = run_root
        return True

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)

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
        group = (
            str(df[COLUMNS.group_col].iloc[0])
            if COLUMNS.group_col in df.columns
            else ""
        )
        sequence = (
            str(df[COLUMNS.seq_col].iloc[0]) if COLUMNS.seq_col in df.columns else ""
        )

        # Resolve video paths (supports multi-video sequences)
        video_paths = self._ds.resolve_media_paths(group, sequence)

        # Determine which IDs to process
        if p.target_id is None:
            # Process all unique IDs
            unique_ids = df[COLUMNS.id_col].dropna().unique()
            all_metadata = []
            for uid in unique_ids:
                df_target = df[df[COLUMNS.id_col] == uid].copy()
                if df_target.empty:
                    continue
                metadata = self._process_single_id(
                    video_paths, df_target, group, sequence, uid
                )
                all_metadata.append(metadata)
            if all_metadata:
                return pd.concat(all_metadata, ignore_index=True)
            return pd.DataFrame()
        else:
            # Process single ID
            df_target = df[df[COLUMNS.id_col] == p.target_id].copy()
            if df_target.empty:
                raise ValueError(f"No data for target_id={p.target_id}")
            return self._process_single_id(
                video_paths, df_target, group, sequence, p.target_id
            )

    # ----------------------- Internal methods --------------------

    def _get_center(
        self, row: pd.Series, angle: float | None = None
    ) -> tuple[float, float]:
        """Extract center point (in pixel coords) from a tracks row.

        If ``center_offset_px`` is non-zero, the center is shifted along the
        heading direction by that amount.  Requires *angle* (radians) when
        the offset is active.
        """
        p = self.params
        mode = p.center_mode

        if mode == "default":
            # Average all available pose points (these are in pixel coordinates).
            # X/Y and X#wcentroid/Y#wcentroid are in real-world units (e.g. cm)
            # and cannot be used directly for video cropping.
            xs, ys = [], []
            for i in range(p.pose.pose_n):
                px = row.get(f"{p.pose.x_prefix}{i}")
                py = row.get(f"{p.pose.y_prefix}{i}")
                if (
                    px is not None
                    and py is not None
                    and np.isfinite(px)
                    and np.isfinite(py)
                ):
                    xs.append(px)
                    ys.append(py)
            if not xs:
                return (np.nan, np.nan)
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
        elif mode == "pose0" or isinstance(mode, int):
            idx = 0 if mode == "pose0" else int(mode)
            x = row.get(f"{p.pose.x_prefix}{idx}")
            y = row.get(f"{p.pose.y_prefix}{idx}")
            if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
                return (np.nan, np.nan)
            cx, cy = float(x), float(y)
        else:
            raise ValueError(
                f"Unknown center_mode: {mode}. Use 'default', 'pose0', or an int pose index."
            )

        # Apply offset along heading direction (positive = forward / toward head)
        if p.center_offset_px != 0.0 and angle is not None and np.isfinite(angle):
            cx += np.cos(angle) * p.center_offset_px
            cy += np.sin(angle) * p.center_offset_px

        return (cx, cy)

    def _precompute_geometry(
        self, df_target: pd.DataFrame, angle_is_degrees: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute heading angles and centers for all frames as numpy arrays.

        Vectorized equivalent of the per-row ``_get_heading_angle`` /
        ``_get_center`` path. Returns three arrays indexed by the row
        order of *df_target*:

        - ``angles``: heading in radians (0 = facing +x). 0.0 where invalid
          or not applicable (no pose points, no angle column).
        - ``cx``, ``cy``: center in pixel coords, with ``center_offset_px``
          shift applied along the heading direction. NaN where the center
          cannot be determined.
        """
        p = self.params
        n = len(df_target)

        # --- Angles ---
        angles = np.zeros(n, dtype=np.float64)
        if p.angle_col and p.angle_col in df_target.columns:
            raw = df_target[p.angle_col].to_numpy(dtype=np.float64)
            valid = np.isfinite(raw)
            if angle_is_degrees:
                angles[valid] = np.radians(raw[valid])
            else:
                angles[valid] = raw[valid]
        else:
            neck_idx, tail_idx = p.heading_points
            nx_col = f"{p.pose.x_prefix}{neck_idx}"
            ny_col = f"{p.pose.y_prefix}{neck_idx}"
            tx_col = f"{p.pose.x_prefix}{tail_idx}"
            ty_col = f"{p.pose.y_prefix}{tail_idx}"
            if all(c in df_target.columns for c in (nx_col, ny_col, tx_col, ty_col)):
                nx = df_target[nx_col].to_numpy(dtype=np.float64)
                ny = df_target[ny_col].to_numpy(dtype=np.float64)
                tx = df_target[tx_col].to_numpy(dtype=np.float64)
                ty = df_target[ty_col].to_numpy(dtype=np.float64)
                valid = (
                    np.isfinite(nx) & np.isfinite(ny)
                    & np.isfinite(tx) & np.isfinite(ty)
                )
                angles[valid] = np.arctan2(ny[valid] - ty[valid], nx[valid] - tx[valid])

        # --- Centers ---
        mode = p.center_mode
        if mode == "default":
            xs_list, ys_list = [], []
            for i in range(p.pose.pose_n):
                xc = f"{p.pose.x_prefix}{i}"
                yc = f"{p.pose.y_prefix}{i}"
                if xc in df_target.columns and yc in df_target.columns:
                    xs_list.append(df_target[xc].to_numpy(dtype=np.float64))
                    ys_list.append(df_target[yc].to_numpy(dtype=np.float64))
            if xs_list:
                xs = np.column_stack(xs_list)
                ys = np.column_stack(ys_list)
                with np.errstate(invalid="ignore"):
                    cx = np.nanmean(xs, axis=1)
                    cy = np.nanmean(ys, axis=1)
            else:
                cx = np.full(n, np.nan, dtype=np.float64)
                cy = np.full(n, np.nan, dtype=np.float64)
        elif mode == "pose0" or isinstance(mode, int):
            idx = 0 if mode == "pose0" else int(mode)
            xc = f"{p.pose.x_prefix}{idx}"
            yc = f"{p.pose.y_prefix}{idx}"
            cx = df_target[xc].to_numpy(dtype=np.float64).copy()
            cy = df_target[yc].to_numpy(dtype=np.float64).copy()
        else:
            raise ValueError(
                f"Unknown center_mode: {mode}. Use 'default', 'pose0', or an int pose index."
            )

        # Apply offset along heading direction (positive = forward / toward head)
        if p.center_offset_px != 0.0:
            finite_a = np.isfinite(angles)
            cx[finite_a] += np.cos(angles[finite_a]) * p.center_offset_px
            cy[finite_a] += np.sin(angles[finite_a]) * p.center_offset_px

        return angles, cx, cy

    def _get_heading_angle(self, row: pd.Series) -> float:
        """Compute heading angle from anatomical landmarks or angle column."""
        p = self.params

        # If angle column is specified, use it
        if p.angle_col and p.angle_col in row.index:
            angle = row.get(p.angle_col)
            if angle is not None and np.isfinite(angle):
                return float(angle)

        # Otherwise compute from neck/tail pose points
        neck_idx, tail_idx = p.heading_points

        neck_x = row.get(f"{p.pose.x_prefix}{neck_idx}")
        neck_y = row.get(f"{p.pose.y_prefix}{neck_idx}")
        tail_x = row.get(f"{p.pose.x_prefix}{tail_idx}")
        tail_y = row.get(f"{p.pose.y_prefix}{tail_idx}")

        if any(
            v is None or not np.isfinite(v) for v in [neck_x, neck_y, tail_x, tail_y]
        ):
            return 0.0

        return compute_heading_angle((neck_x, neck_y), (tail_x, tail_y))

    def _extract_egocentric_crop(
        self,
        frame: np.ndarray,
        center: tuple[float, float],
        angle: float,
    ) -> np.ndarray:
        """
        Extract an egocentric crop from a video frame.

        Parameters
        ----------
        frame : np.ndarray
            Source video frame (H, W, C) or (H, W)
        center : tuple[float, float]
            (cx, cy) center point in pixel coordinates
        angle : float
            Heading angle in radians (0 = facing right/+x)

        Returns
        -------
        np.ndarray
            Cropped (and optionally rotated) frame with body mask / CLAHE /
            grayscale applied if configured.
        """
        p = self.params
        crop_w, crop_h = p.crop_size
        cx, cy = center
        n_channels = frame.shape[2] if frame.ndim == 3 else 1

        if not np.isfinite(cx) or not np.isfinite(cy):
            if p.grayscale or n_channels == 1:
                return np.full((crop_h, crop_w), p.background_color, dtype=np.uint8)
            return np.full((crop_h, crop_w, 3), p.background_color, dtype=np.uint8)

        if p.rotate_to_heading:
            angle_deg = np.degrees(angle)

            margin = p.margin_factor
            pre_crop_size = int(max(crop_w, crop_h) * margin)

            pre_crop = safe_crop_with_padding(
                frame,
                (int(cx), int(cy)),
                (pre_crop_size, pre_crop_size),
                pad_value=p.background_color,
            )

            pre_center = (pre_crop_size / 2.0, pre_crop_size / 2.0)
            M = cv2.getRotationMatrix2D(pre_center, angle_deg, 1.0)
            bv = (p.background_color,) * (n_channels if n_channels > 1 else 1)
            rotated = cv2.warpAffine(
                pre_crop,
                M,
                (pre_crop_size, pre_crop_size),
                flags=p.interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=bv,
            )

            crop = safe_crop_with_padding(
                rotated,
                (pre_crop_size // 2, pre_crop_size // 2),
                (crop_w, crop_h),
                pad_value=p.background_color,
            )
        else:
            crop = safe_crop_with_padding(
                frame,
                (int(cx), int(cy)),
                (crop_w, crop_h),
                pad_value=p.background_color,
            )

        # --- Post-processing pipeline ---

        # Body mask: elliptical mask centered on the crop, aligned to heading
        if p.body_mask:
            crop = self._apply_body_mask(crop, crop_w, crop_h)

        # Grayscale conversion
        if p.grayscale and crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # CLAHE contrast enhancement
        if p.use_clahe:
            crop = self._apply_clahe(crop)

        return crop

    def _apply_body_mask(
        self, crop: np.ndarray, crop_w: int, crop_h: int
    ) -> np.ndarray:
        """Apply an elliptical body mask centered on the crop.

        After egocentric rotation the animal faces right (+x), so the
        ellipse major axis is horizontal.
        """
        p = self.params
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        center = (crop_w // 2, crop_h // 2)
        axes = (p.body_mask_length_px // 2, p.body_mask_width_px // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        if crop.ndim == 3:
            crop = cv2.bitwise_and(crop, crop, mask=mask)
        else:
            crop = np.where(mask > 0, crop, 0).astype(crop.dtype)
        return crop

    def _apply_clahe(self, crop: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement (object reused across frames)."""
        clahe = self._get_clahe()
        if crop.ndim == 2:
            return clahe.apply(crop)
        # For color images, convert to LAB, apply CLAHE to L channel
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _transform_keypoints(
        self,
        row: pd.Series,
        cx: float,
        cy: float,
        angle: float,
    ) -> dict[str, float]:
        """Transform pose keypoints into crop coordinate space.

        Returns a dict with keys ``poseX<i>_crop``, ``poseY<i>_crop``.
        """
        p = self.params
        crop_w, crop_h = p.crop_size
        crop_cx, crop_cy = crop_w / 2.0, crop_h / 2.0
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)

        kp = {}
        for i in range(p.pose.pose_n):
            px = row.get(f"{p.pose.x_prefix}{i}")
            py = row.get(f"{p.pose.y_prefix}{i}")
            if px is None or py is None or not np.isfinite(px) or not np.isfinite(py):
                kp[f"{p.pose.x_prefix}{i}_crop"] = np.nan
                kp[f"{p.pose.y_prefix}{i}_crop"] = np.nan
                continue
            # Translate to center-relative coordinates
            dx = px - cx
            dy = py - cy
            if p.rotate_to_heading:
                # Rotate into crop space
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
            else:
                rx, ry = dx, dy
            kp[f"{p.pose.x_prefix}{i}_crop"] = crop_cx + rx
            kp[f"{p.pose.y_prefix}{i}_crop"] = crop_cy + ry
        return kp

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
        df_target = df_target.sort_values(COLUMNS.frame_col).reset_index(drop=True)

        # Open video(s) via MultiVideoReader
        reader = MultiVideoReader(video_paths)
        output_fps = p.output_fps or reader.fps

        # Build frame -> row position lookup (for precomputed geometry arrays
        # and for optional per-frame lookups like transform_keypoints)
        frame_array = df_target[COLUMNS.frame_col].to_numpy(dtype=int)
        frame_to_pos = {int(f): i for i, f in enumerate(frame_array)}

        # Prepare output paths
        run_root = self._get_run_root(group, sequence)
        run_root.mkdir(parents=True, exist_ok=True)

        # Determine output crop size
        crop_w, crop_h = p.crop_size

        # For grayscale video output, VideoWriter needs isColor=False
        writer = None
        if p.output_mode in ("video", "both"):
            video_out_path = run_root / f"egocentric_id{target_id}.mp4"
            if p.grayscale:
                path = Path(video_out_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                codec = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(path), codec, float(output_fps), (crop_w, crop_h), isColor=False
                )
            else:
                writer = create_video_writer(video_out_path, output_fps, (crop_w, crop_h))

        frames_dir = None
        if p.output_mode in ("frames", "both"):
            frames_dir = run_root / f"frames_id{target_id}"
            frames_dir.mkdir(exist_ok=True)

        # Check if angle column exists and infer if degrees
        angle_is_degrees = False
        if p.angle_col and p.angle_col in df_target.columns:
            angle_is_degrees = infer_angle_degrees(df_target[p.angle_col])

        # Pre-compute heading + center for every labeled frame (vectorized).
        # Arrays are indexed by position in df_target (see frame_to_pos above).
        angles, centers_x, centers_y = self._precompute_geometry(
            df_target, angle_is_degrees
        )

        # Keep per-row access for the (rare) transform_keypoints path
        rows_by_pos = df_target if p.transform_keypoints else None

        # Process frames
        metadata_rows = []
        frame_idx = 0
        total_frames = reader.total_frames

        try:
            while True:
                ret, frame = reader.read()
                if not ret:
                    break

                pos = frame_to_pos.get(frame_idx)
                if pos is not None:
                    angle = float(angles[pos]) if (p.rotate_to_heading or p.center_offset_px != 0.0) else 0.0
                    center = (float(centers_x[pos]), float(centers_y[pos]))
                    crop = self._extract_egocentric_crop(frame, center, angle)

                    if writer is not None:
                        writer.write(crop)

                    if frames_dir is not None:
                        frame_path = (
                            frames_dir / f"frame_{frame_idx:06d}.{p.frame_format}"
                        )
                        cv2.imwrite(str(frame_path), crop)

                    meta = {
                        "frame": frame_idx,
                        "center_x": center[0],
                        "center_y": center[1],
                        "heading_angle": angle,
                        "target_id": target_id,
                        "group": group,
                        "sequence": sequence,
                    }

                    if p.transform_keypoints:
                        row = rows_by_pos.iloc[pos]
                        meta.update(
                            self._transform_keypoints(row, center[0], center[1], angle)
                        )

                    metadata_rows.append(meta)

                frame_idx += 1

                # Progress logging for long videos
                if frame_idx % 1000 == 0:
                    print(
                        f"  [egocentric-crop] id={target_id}: {frame_idx}/{total_frames} frames"
                    )

        finally:
            reader.close()
            if writer is not None:
                writer.release()

        print(
            f"  [egocentric-crop] id={target_id}: processed {len(metadata_rows)} frames -> {run_root}"
        )

        return pd.DataFrame(metadata_rows)

    def _get_run_root(self, group: str, sequence: str) -> Path:
        """Get the output directory for this run.

        Uses ``run_root`` from the pipeline (run_id-tagged) when available,
        falls back to ``media/egocentric_crops/<group>__<sequence>``.
        Can be overridden via ``output_root`` param.
        """
        if self.params.output_root:
            return Path(self.params.output_root) / f"{group}__{sequence}"
        if self._run_root is not None:
            return self._run_root / f"{group}__{sequence}"
        media_root = Path(self._ds.get_root("media"))
        return media_root / "egocentric_crops" / f"{group}__{sequence}"
