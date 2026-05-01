"""Subsample tracks rows via k-means on canonical pose, or uniform stride.

Used to thin dense per-frame tracks (e.g. ~22k rows for a 15 min video at
25 fps) down to a small set of postural representatives suitable as input
to downstream features like ``EgocentricCrop``, where one tracks row maps
to one output crop. Output preserves the ``trex_v1`` schema, so any
downstream feature that accepts ``TrackInput`` can also accept this
feature's ``Result``.

Example::

    from mosaic.behavior.feature_library import TrackSubsample
    from mosaic.behavior.visualization_library.egocentric_crop import EgocentricCrop
    from mosaic.core.pipeline.types import Inputs, Result

    sub = TrackSubsample(params={"method": "kmeans", "target_frames": 300})
    sub_result = ds.run_feature(sub)

    crop = EgocentricCrop(
        inputs=EgocentricCrop.Inputs((sub_result,)),
        params={"crop_size": (224, 224), "grayscale": True, "output_mode": "frames"},
    )
    ds.run_feature(crop)
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, final

import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.pipeline.types import (
    COLUMNS,
    DependencyLookup,
    Inputs,
    InputStream,
    Params,
    PoseConfig,
    TrackInput,
)

from .registry import register_feature


@final
@register_feature
class TrackSubsample:
    """Subsample tracks rows for downstream per-row features.

    Two methods:

    - ``"kmeans"`` (default): k-means cluster frames in body-canonical
      pose space (translate to ``(X, Y)`` origin, rotate by ``-ANGLE``),
      then keep the frame closest to each cluster centroid. Maximizes
      postural diversity for downstream identity/visual training, where
      adjacent frames in dense recordings are highly redundant.
    - ``"uniform"``: pick frames at constant stride. Simple and
      predictable; faster but redundant on slow-moving subjects.

    Frames with NaN ``ANGLE`` / ``X`` / ``Y`` / required keypoints are
    excluded from k-means clustering (they cannot be canonicalized). When
    ``drop_nan=True`` (default) those frames are also removed from the
    output. When ``len(df) <= target_frames`` the input is returned
    unchanged.

    Output preserves all input columns -- only row count changes.

    Params:
        method: ``"kmeans"`` or ``"uniform"``. Default ``"kmeans"``.
        target_frames: Target output row count per sequence. Default 300.
        pose: Pose-column naming and count config. Default
            ``PoseConfig()`` (``pose_n=7``, ``poseX*`` / ``poseY*``).
        seed: Random seed for k-means. Default 42.
        drop_nan: If True, drop frames with non-finite required columns
            from the output. Default True.
    """

    category = "per-frame"
    name = "track-subsample"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        method: Literal["kmeans", "uniform"] = "kmeans"
        target_frames: int = Field(default=300, ge=1)
        pose: PoseConfig = Field(default_factory=PoseConfig)
        seed: int = 42
        drop_nan: bool = True

    def __init__(
        self,
        inputs: TrackSubsample.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ) -> None:
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
        if df is None or df.empty:
            return df

        p = self.params
        if len(df) <= p.target_frames:
            return df.reset_index(drop=True)

        if p.method == "uniform":
            return self._uniform(df, p.target_frames)

        # Discover available pose columns (allows pose-subset tracks).
        n_kp = p.pose.pose_n
        x_cols = [
            f"{p.pose.x_prefix}{i}"
            for i in range(n_kp)
            if f"{p.pose.x_prefix}{i}" in df.columns
        ]
        y_cols = [
            f"{p.pose.y_prefix}{i}"
            for i in range(n_kp)
            if f"{p.pose.y_prefix}{i}" in df.columns
        ]
        if not x_cols or not y_cols or len(x_cols) != len(y_cols):
            # Pose columns missing or inconsistent — fall back to uniform.
            return self._uniform(df, p.target_frames)

        # Required centroid + heading columns
        if (
            COLUMNS.x_col not in df.columns
            or COLUMNS.y_col not in df.columns
            or COLUMNS.orientation_col not in df.columns
        ):
            return self._uniform(df, p.target_frames)

        pose_x = df[x_cols].to_numpy(dtype=np.float64)
        pose_y = df[y_cols].to_numpy(dtype=np.float64)
        cx = df[COLUMNS.x_col].to_numpy(dtype=np.float64)
        cy = df[COLUMNS.y_col].to_numpy(dtype=np.float64)
        ang = df[COLUMNS.orientation_col].to_numpy(dtype=np.float64)

        valid = (
            np.isfinite(pose_x).all(axis=1)
            & np.isfinite(pose_y).all(axis=1)
            & np.isfinite(cx)
            & np.isfinite(cy)
            & np.isfinite(ang)
        )

        if int(valid.sum()) <= p.target_frames:
            sub = df[valid] if p.drop_nan else df
            return sub.reset_index(drop=True)

        # Translate to (X, Y) and rotate by -ANGLE → body-canonical pose.
        rel_x = pose_x[valid] - cx[valid, None]
        rel_y = pose_y[valid] - cy[valid, None]
        a = -ang[valid]
        cos_a = np.cos(a)[:, None]
        sin_a = np.sin(a)[:, None]
        can_x = rel_x * cos_a - rel_y * sin_a
        can_y = rel_x * sin_a + rel_y * cos_a
        feats = np.hstack([can_x, can_y])  # (N_valid, 2 * n_kp_avail)

        # K-means clustering. Imported lazily to avoid a hard sklearn
        # dependency for the 'uniform' path.
        from sklearn.cluster import KMeans

        km = KMeans(
            n_clusters=p.target_frames,
            random_state=p.seed,
            n_init="auto",
        ).fit(feats)

        # For each cluster, pick the member closest to the centroid.
        chosen_pos: list[int] = []
        for c in range(p.target_frames):
            members = np.where(km.labels_ == c)[0]
            if len(members) == 0:
                continue
            d = np.linalg.norm(feats[members] - km.cluster_centers_[c], axis=1)
            chosen_pos.append(int(members[np.argmin(d)]))

        valid_idx = np.where(valid)[0]
        chosen_idx = sorted(valid_idx[chosen_pos].tolist())
        return df.iloc[chosen_idx].reset_index(drop=True)

    @staticmethod
    def _uniform(df: pd.DataFrame, target_frames: int) -> pd.DataFrame:
        stride = max(1, len(df) // target_frames)
        return df.iloc[::stride].head(target_frames).reset_index(drop=True)
