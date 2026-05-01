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

    Three methods:

    - ``"kmeans"`` (default): k-means cluster frames in body-canonical
      pose space (translate to ``(X, Y)`` origin, rotate by ``-ANGLE``),
      then keep the frame closest to each cluster centroid. Maximizes
      postural diversity for downstream identity/visual training, where
      adjacent frames in dense recordings are highly redundant.
    - ``"uniform"``: pick frames at constant stride. Simple and
      predictable; faster but redundant on slow-moving subjects.
    - ``"clips"``: pick K seeds via k-means on canonical pose, then
      take ``clip_len`` consecutive frames starting at each seed. Output
      is organized as K contiguous spans of ``clip_len`` frames each, for
      downstream features that consume short clips (e.g. temporal
      identity heads). With ``method="clips"`` we set
      ``K = target_frames // clip_len``.

    Frames with NaN ``ANGLE`` / ``X`` / ``Y`` / required keypoints are
    excluded from k-means clustering (they cannot be canonicalized). When
    ``drop_nan=True`` (default) those frames are also removed from the
    output. When ``len(df) <= target_frames`` the input is returned
    unchanged.

    Output preserves all input columns -- only row count changes.

    .. note::
       For ``method="clips"``, downstream temporal features (e.g.
       ``GlobalIdentityDinoV2Temporal``) sort frames alphabetically and
       slide a window of size ``clip_len`` over them. Each contiguous
       span of ``clip_len`` frames becomes one clip when the downstream
       ``clip_stride`` equals ``clip_len`` (the default). Setting
       ``clip_stride < clip_len`` downstream will produce windows that
       straddle the gap between two spans -- meaningless temporal
       samples. Use the default non-overlapping stride.

    Params:
        method: ``"kmeans"`` (default), ``"uniform"``, or ``"clips"``.
        target_frames: Target output row count per sequence. Default 300.
            For ``method="clips"`` this is interpreted as
            ``n_clips * clip_len``; the number of clips is
            ``target_frames // clip_len``.
        clip_len: Frames per clip when ``method="clips"``. Default 8.
            Ignored otherwise.
        pose: Pose-column naming and count config. Default
            ``PoseConfig()`` (``pose_n=7``, ``poseX*`` / ``poseY*``).
        seed: Random seed for k-means. Default 42.
        drop_nan: If True, drop frames with non-finite required columns
            from the output. Default True.
    """

    category = "per-frame"
    name = "track-subsample"
    version = "0.2"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        method: Literal["kmeans", "uniform", "clips"] = "kmeans"
        target_frames: int = Field(default=300, ge=1)
        clip_len: int = Field(default=8, ge=2)
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

        if p.method == "uniform":
            if len(df) <= p.target_frames:
                return df.reset_index(drop=True)
            return self._uniform(df, p.target_frames)

        if p.method == "kmeans":
            if len(df) <= p.target_frames:
                return df.reset_index(drop=True)
            seeds = self._kmeans_seeds(df, p.target_frames)
            if seeds is None:
                return self._uniform(df, p.target_frames)
            return df.iloc[sorted(seeds)].reset_index(drop=True)

        # method == "clips": pick K seeds, expand each to clip_len consecutive frames.
        n_clips = max(1, p.target_frames // p.clip_len)
        if len(df) < p.clip_len:
            return df.reset_index(drop=True)

        # If video is just barely long enough, fall back to as-many-clips-as-fit.
        n_clips = min(n_clips, len(df) // p.clip_len)
        if n_clips < 1:
            return df.reset_index(drop=True)

        seeds = self._kmeans_seeds(df, n_clips)
        if seeds is None:
            # Fallback: K evenly-spaced clip starts.
            stride = max(p.clip_len, len(df) // n_clips)
            seeds = [min(i * stride, len(df) - p.clip_len) for i in range(n_clips)]

        # Snap each seed into [0, len(df)-clip_len].
        max_start = len(df) - p.clip_len
        snapped = sorted(min(max(int(s), 0), max_start) for s in seeds)

        # Greedy disjoint: when a seed's [s, s+clip_len) overlaps the previous
        # span, snap it forward to start exactly at the previous span's end.
        # If snapping forward exceeds max_start, drop the seed. The resulting
        # spans are strictly non-overlapping, so the downstream sorted-glob +
        # non-overlapping sliding window aligns each span 1:1 with one clip.
        disjoint: list[int] = []
        last_end = -1
        for s in snapped:
            if s < last_end:
                s = last_end
            if s + p.clip_len > len(df):
                break
            disjoint.append(s)
            last_end = s + p.clip_len

        # If everything collapsed (extreme overlap), at least emit one clip
        # at row 0 so EgocentricCrop has something to work with.
        if not disjoint:
            disjoint = [0] if max_start >= 0 else []

        keep_rows: list[int] = []
        for s in disjoint:
            keep_rows.extend(range(s, s + p.clip_len))
        return df.iloc[keep_rows].reset_index(drop=True)

    @staticmethod
    def _uniform(df: pd.DataFrame, target_frames: int) -> pd.DataFrame:
        stride = max(1, len(df) // target_frames)
        return df.iloc[::stride].head(target_frames).reset_index(drop=True)

    def _kmeans_seeds(
        self, df: pd.DataFrame, n_clusters: int
    ) -> list[int] | None:
        """Pick `n_clusters` seed row indices via k-means on canonical pose.

        Returns row positions in `df` (not `frame` values), or None when the
        canonical-pose feature space cannot be computed (missing columns,
        too few valid frames, etc.) so the caller can fall back to uniform.
        """
        p = self.params

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
            return None

        if (
            COLUMNS.x_col not in df.columns
            or COLUMNS.y_col not in df.columns
            or COLUMNS.orientation_col not in df.columns
        ):
            return None

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

        if int(valid.sum()) <= n_clusters:
            return list(np.where(valid)[0])

        rel_x = pose_x[valid] - cx[valid, None]
        rel_y = pose_y[valid] - cy[valid, None]
        a = -ang[valid]
        cos_a = np.cos(a)[:, None]
        sin_a = np.sin(a)[:, None]
        can_x = rel_x * cos_a - rel_y * sin_a
        can_y = rel_x * sin_a + rel_y * cos_a
        feats = np.hstack([can_x, can_y])

        from sklearn.cluster import KMeans

        km = KMeans(
            n_clusters=n_clusters, random_state=p.seed, n_init="auto"
        ).fit(feats)

        chosen_pos: list[int] = []
        for c in range(n_clusters):
            members = np.where(km.labels_ == c)[0]
            if len(members) == 0:
                continue
            d = np.linalg.norm(feats[members] - km.cluster_centers_[c], axis=1)
            chosen_pos.append(int(members[np.argmin(d)]))

        valid_idx = np.where(valid)[0]
        return [int(valid_idx[i]) for i in chosen_pos]
