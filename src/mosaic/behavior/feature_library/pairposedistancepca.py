from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TypedDict, final

import joblib
import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.decomposition import IncrementalPCA

from .helpers import clean_animal_track, ensure_columns
from .spec import COLUMNS as C
from .spec import (
    Inputs,
    InterpolationConfig,
    OutputType,
    Params,
    PoseConfig,
    TrackInput,
    register_feature,
    resolve_order_col,
)


class PairPoseDistancePCABundle(TypedDict):
    ipca: IncrementalPCA
    tri_i: np.ndarray | None
    tri_j: np.ndarray | None
    feat_len: int | None
    version: str


@dataclass(frozen=True, slots=True)
class _BatchMeta:
    """Metadata yielded alongside each feature batch."""

    frames: np.ndarray | None
    id1: int
    id2: int


@final
@register_feature
class PairPoseDistancePCA:
    """
    'pair-posedistance-pca' — builds per-frame pairwise pose-distance features and
    fits an IncrementalPCA globally; outputs PC scores per sequence (and perspective).
    """

    name = "pair-posedistance-pca"
    version = "0.1"
    parallelizable = True
    scope_dependent = True
    output_type: OutputType = "per_frame"

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        interpolation: InterpolationConfig = Field(default_factory=InterpolationConfig)
        pose: PoseConfig = Field(default_factory=PoseConfig)
        include_intra_A: bool = True
        include_intra_B: bool = True
        include_inter: bool = True
        duplicate_perspective: bool = True
        n_components: int = 6
        batch_size: int = Field(default=5000, gt=0)

    def __init__(
        self,
        inputs: PairPoseDistancePCA.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._ipca: IncrementalPCA = IncrementalPCA(
            n_components=self.params.n_components,
            batch_size=self.params.batch_size,
        )
        self._fitted = False
        self._tri_i: np.ndarray | None = None
        self._tri_j: np.ndarray | None = None
        self._feat_len: int | None = None

    # ---------- Feature protocol ----------
    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_indices: dict[str, pd.DataFrame],
    ) -> bool:
        path = run_root / "model.joblib"
        if path.exists():
            bundle: PairPoseDistancePCABundle = joblib.load(path)
            self._ipca = bundle["ipca"]
            self._tri_i = bundle["tri_i"]
            self._tri_j = bundle["tri_j"]
            self._feat_len = bundle["feat_len"]
            self._fitted = True
            return True
        return False

    def fit(self, inputs: Callable[[], Iterator[tuple[str, pd.DataFrame]]]) -> None:
        for _entry_key, df in inputs():
            for batch, _, _ in self._feature_batches(df, for_fit=True):
                if batch.size == 0:
                    continue
                self._ipca.partial_fit(batch)
                self._fitted = True

    def save_state(self, run_root: Path) -> None:
        if not self._fitted:
            return
        run_root.mkdir(parents=True, exist_ok=True)
        bundle: PairPoseDistancePCABundle = {
            "ipca": self._ipca,
            "tri_i": self._tri_i,
            "tri_j": self._tri_j,
            "feat_len": self._feat_len,
            "version": self.version,
        }
        joblib.dump(bundle, run_root / "model.joblib")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(
                "pair-posedistance-pca: not fitted yet; run fit/partial_fit first."
            )

        pcs: list[pd.DataFrame] = []
        for features, meta, persp in self._feature_batches(df, for_fit=False):
            if features.size == 0:
                continue
            scores = self._ipca.transform(features)
            out = pd.DataFrame(
                scores, columns=[f"PC{i}" for i in range(scores.shape[1])]
            )
            if meta.frames is not None:
                out[C.frame_col] = meta.frames
            out["perspective"] = persp
            out["id1"] = meta.id1
            out["id2"] = meta.id2
            for col in (C.seq_col, C.group_col):
                if col in df.columns:
                    out[col] = df[col].iloc[0]
            pcs.append(out)

        if not pcs:
            return pd.DataFrame(
                columns=["perspective"]
                + [f"PC{i}" for i in range(self.params.n_components)]
            )

        out_df = pd.concat(pcs, ignore_index=True)
        sort_keys: list[str] = ["id1", "id2", "perspective"]
        if C.frame_col in out_df.columns:
            sort_keys.append(C.frame_col)
        elif C.time_col in out_df.columns:
            sort_keys.append(C.time_col)
        out_df = out_df.sort_values(sort_keys).reset_index(drop=True)
        return out_df

    # ---------- Internals ----------
    def _get_pose_indices(self) -> list[int]:
        """Return the list of pose point indices to use."""
        indices = self.params.pose.pose_indices
        if indices is None:
            return list(range(self.params.pose.pose_n))
        return list(indices)

    def _effective_pose_n(self) -> int:
        """Return the number of pose points being used."""
        return len(self._get_pose_indices())

    def _column_names(self) -> tuple[list[str], list[str]]:
        indices = self._get_pose_indices()
        xs = [f"{self.params.pose.x_prefix}{i}" for i in indices]
        ys = [f"{self.params.pose.y_prefix}{i}" for i in indices]
        return xs, ys

    def _prep_pairs(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[tuple[str, int, int]]]:
        x_cols, y_cols = self._column_names()
        pose_cols = x_cols + y_cols
        order_col = resolve_order_col(df)

        ensure_columns(df, [C.id_col, C.seq_col, order_col] + pose_cols)

        need = [C.id_col, C.seq_col, order_col] + pose_cols
        df_small = df[need]

        group_cols = [C.seq_col, C.id_col]

        df_small = df_small.groupby(group_cols, group_keys=False).apply(
            lambda g: clean_animal_track(
                g, pose_cols, order_col, self.params.interpolation
            )
        )

        pairs: list[tuple[str, int, int]] = []
        for seq, gseq in df_small.groupby(C.seq_col):
            ids: list[int] = sorted(gseq[C.id_col].unique())
            if len(ids) < 2:
                continue
            for id_a, id_b in combinations(ids, 2):
                pairs.append((str(seq), int(id_a), int(id_b)))

        if not pairs:
            raise ValueError(
                "[pair-posedistance-pca] No sequence with at least two IDs found."
            )

        if self._tri_i is None or self._tri_j is None or self._feat_len is None:
            N = self._effective_pose_n()
            tri_i, tri_j = np.tril_indices(N, k=-1)
            n_intra = len(tri_i)
            n_cross = N * N
            feat_len = 0
            if self.params.include_intra_A:
                feat_len += n_intra
            if self.params.include_intra_B:
                feat_len += n_intra
            if self.params.include_inter:
                feat_len += n_cross
            self._tri_i, self._tri_j, self._feat_len = tri_i, tri_j, feat_len
        return df_small, pairs

    def _build_pair_feat(self, row_a: np.ndarray, row_b: np.ndarray) -> np.ndarray:
        parts: list[np.ndarray] = []
        pts_a = self._pose_to_points(row_a)
        pts_b = self._pose_to_points(row_b)
        if self.params.include_intra_A:
            parts.append(self._intra_lower_tri(pts_a))
        if self.params.include_intra_B:
            parts.append(self._intra_lower_tri(pts_b))
        if self.params.include_inter:
            parts.append(self._inter_all(pts_a, pts_b))
        return (
            np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=np.float32)
        )

    def _feature_batches(
        self, df: pd.DataFrame, for_fit: bool
    ) -> Iterator[tuple[np.ndarray, _BatchMeta, np.ndarray]]:
        x_cols, y_cols = self._column_names()
        pose_cols = x_cols + y_cols
        order_col = resolve_order_col(df)

        df_small, pairs = self._prep_pairs(df)
        bs = self.params.batch_size
        dup = self.params.duplicate_perspective
        has_frames = C.frame_col in df.columns

        for seq_key, id_a, id_b in pairs:
            gseq = df_small[df_small[C.seq_col] == seq_key]
            df_a = gseq[gseq[C.id_col] == id_a][[order_col] + pose_cols]
            df_b = gseq[gseq[C.id_col] == id_b][[order_col] + pose_cols]
            df_a = df_a.sort_values(order_col)
            df_b = df_b.sort_values(order_col)
            merged = df_a.merge(df_b, on=order_col, suffixes=("_A", "_B"))
            if merged.empty:
                continue

            n = len(merged)
            for i in range(0, n, bs):
                chunk = merged.iloc[i : min(i + bs, n)]
                vals_a = chunk[[c + "_A" for c in pose_cols]].to_numpy(dtype=float)
                vals_b = chunk[[c + "_B" for c in pose_cols]].to_numpy(dtype=float)
                feat_rows = [
                    self._build_pair_feat(a, b) for a, b in zip(vals_a, vals_b)
                ]
                feat_mat = np.vstack(feat_rows).astype(np.float32, copy=False)

                persp = np.zeros(feat_mat.shape[0], dtype=np.int8)
                frame_arr = chunk[order_col].to_numpy() if has_frames else None

                if dup:
                    feat_rows2 = [
                        self._build_pair_feat(b, a) for a, b in zip(vals_a, vals_b)
                    ]
                    feat_dup = np.vstack(feat_rows2).astype(np.float32, copy=False)
                    feat_mat = np.vstack([feat_mat, feat_dup])
                    persp = np.concatenate(
                        [persp, np.ones(feat_dup.shape[0], dtype=np.int8)], axis=0
                    )
                    if frame_arr is not None:
                        frame_arr = np.concatenate([frame_arr, frame_arr], axis=0)

                if self._feat_len is not None and feat_mat.shape[1] != self._feat_len:
                    msg = f"Feature length mismatch: got {feat_mat.shape[1]}, expected {self._feat_len}"
                    raise ValueError(msg)

                yield feat_mat, _BatchMeta(frame_arr, id_a, id_b), persp

    def _pose_to_points(self, row_vals: np.ndarray) -> np.ndarray:
        N = self._effective_pose_n()
        xs = row_vals[:N]
        ys = row_vals[N:]
        return np.stack([xs, ys], axis=1)

    def _intra_lower_tri(self, pts: np.ndarray) -> np.ndarray:
        dif = pts[self._tri_i] - pts[self._tri_j]
        return np.sqrt((dif**2).sum(axis=1))

    def _inter_all(self, pts_a: np.ndarray, pts_b: np.ndarray) -> np.ndarray:
        dif = pts_a[:, None, :] - pts_b[None, :, :]
        d = np.sqrt((dif**2).sum(axis=2))
        return d.ravel()
