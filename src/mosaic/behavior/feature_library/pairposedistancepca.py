from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Annotated, ClassVar, TypedDict, TypeGuard, final, override

import joblib
import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.decomposition import IncrementalPCA

from mosaic.core.pipeline.column_contract import (
    ORDER_COLUMN,
    Column,
    Keypoints,
    Prefix,
    Provides,
    Requirement,
    kept,
)
from mosaic.core.pipeline.types import (
    COLUMNS as C,
    EmitsLevel,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    InputStream,
    JoblibArtifact,
    JoblibLoadSpec,
    POSE_CONFIG_DESCRIPTION,
    PoseConfig,
    TrackInputs,
    resolve_order_col,
)
from mosaic.core.params import Declared, Params

from .helpers import clean_tracks_grouped, ensure_columns
from .registry import register_feature
from .types import INTERPOLATION_CONFIG_DESCRIPTION, InterpolationConfig

_INCLUDE_INTRA_A_DESCRIPTION = (
    "Include the pairwise distances between the focal individual's own keypoints."
)

_INCLUDE_INTRA_B_DESCRIPTION = (
    "Include the pairwise distances between the other individual's own keypoints."
)

_INCLUDE_INTER_DESCRIPTION = (
    "Include the pairwise distances between the focal individual's "
    "keypoints and the other individual's keypoints."
)

_DUPLICATE_PERSPECTIVE_DESCRIPTION = (
    "Emit each pair twice, once with each individual as id1, the focal "
    "one. False emits only the row set with the lower-id individual as "
    "id1."
)

_N_COMPONENTS_DESCRIPTION = "The number of principal components the PCA retains."

_BATCH_SIZE_DESCRIPTION = (
    "How many pair-frames are read per batch. duplicate_perspective "
    "doubles the rows each IncrementalPCA partial_fit or transform call "
    "receives."
)


_MODEL_DESCRIPTION = (
    "The model.joblib of a pair-posedistance-pca run to apply, such as a fit "
    "step scoped to the training groups. Unset, the feature fits a basis on "
    "the entries in its scope. Set, nothing is fitted, the entries in scope "
    "leave the run id, and n_components, pose, include_intra_A, include_intra_B "
    "and include_inter must equal the fit's."
)


class PairPoseDistancePCABundle(TypedDict):
    ipca: IncrementalPCA
    tri_i: np.ndarray | None
    tri_j: np.ndarray | None
    feat_len: int | None
    version: str


def _is_bundle(value: object) -> TypeGuard[PairPoseDistancePCABundle]:
    """Whether *value* is a pair-posedistance-pca bundle.

    ``global-kmeans`` and others also write ``model.joblib``, so a ``model``
    naming the wrong run resolves to a file that loads.
    """
    return isinstance(value, dict) and all(
        key in value for key in ("ipca", "tri_i", "tri_j", "feat_len")
    )


class PairPoseDistancePCAModelArtifact(JoblibArtifact[PairPoseDistancePCABundle]):
    """Fitted PCA basis (model.joblib)."""

    feature: str = "pair-posedistance-pca"
    pattern: str = "model.joblib"
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


@dataclass(frozen=True, slots=True)
class _BatchMeta:
    """Row-aligned identity for one feature batch.

    ``id1`` is the focal and ``id2`` the other individual, so the pair **swaps**
    between the two perspectives: the perspective-1 rows are built as ``(b, a)``,
    and labelling them ``(a, b)`` made one ``(frame, id1, id2)`` name two different
    rows. Arrays rather than scalars for exactly that reason -- a scalar stamped
    across a batch cannot express the swap.
    """

    frames: np.ndarray | None
    id1: np.ndarray
    id2: np.ndarray


@final
@register_feature
class PairPoseDistancePCA:
    """Principal components of the pose distances between every pair, per frame.

    Builds per-frame pairwise pose-distance features, fits an IncrementalPCA over
    every sequence, and writes the PC scores per sequence and perspective. This is
    the postural stage of a behavioral map that follows the approach of Berman et
    al. (2014), MotionMapper, with keypoint distances in place of image eigenmodes.

    One row per **ordered** pair per frame: ``id1`` is the focal, ``id2`` the other,
    and ``perspective`` says which ordering, so the key is
    ``(frame, id1, id2, perspective)``.

    The basis is fitted on the entries in scope, so a different scope is a
    different fit. ``model`` names a fitted run's ``model.joblib`` instead: the
    run then applies that basis and fits nothing, and its run id does not depend
    on the entries it covers. A fit step scoped to the training groups and an
    apply step over every group, with ``model`` naming the fit step, give the test
    groups the training basis. ``scope_dependent`` is therefore True on the class,
    which describes a fit, and False on an instance that was given a model.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.pairposedistancepca.PairPoseDistancePCA.Params`.
    """

    category = "per-frame"
    name = "pair-posedistance-pca"
    # 0.3: `model` applies a basis fitted by another run. Without one the output
    # is that of 0.2, and the digest of the run id is unchanged.
    version = "0.3"
    parallelizable = True
    scope_dependent = True
    fits = True  # learns its basis; ``__init__`` clears this for a run given a model
    accepts_overlap = False  # computes within a frame, so gains nothing
    temporal_reach = 0
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "pair"
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.id_col),
        Column(C.seq_col),
        ORDER_COLUMN,
        Keypoints(),
    )
    # One PC column per component, PC0 to PC<n_components - 1>. The frame column
    # contains the order column's values, and is written when the input has frame.
    provides: ClassVar[Provides] = Provides(
        adds=(
            Prefix("PC"),
            Column("perspective"),
            Column("id1"),
            Column("id2"),
            Column(C.seq_col),
        ),
        conditional=kept(C.frame_col, C.group_col),
    )

    ModelArtifact = PairPoseDistancePCAModelArtifact

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        interpolation: Annotated[
            InterpolationConfig, Declared(INTERPOLATION_CONFIG_DESCRIPTION)
        ] = Field(default_factory=InterpolationConfig)
        pose: Annotated[PoseConfig, Declared(POSE_CONFIG_DESCRIPTION)] = Field(
            default_factory=PoseConfig
        )
        include_intra_A: Annotated[bool, Declared(_INCLUDE_INTRA_A_DESCRIPTION)] = True
        include_intra_B: Annotated[bool, Declared(_INCLUDE_INTRA_B_DESCRIPTION)] = True
        include_inter: Annotated[bool, Declared(_INCLUDE_INTER_DESCRIPTION)] = True
        duplicate_perspective: Annotated[
            bool, Declared(_DUPLICATE_PERSPECTIVE_DESCRIPTION)
        ] = True
        n_components: Annotated[int, Declared(_N_COMPONENTS_DESCRIPTION)] = 6
        batch_size: Annotated[int, Declared(_BATCH_SIZE_DESCRIPTION)] = Field(
            default=5000, gt=0
        )
        model: Annotated[
            PairPoseDistancePCAModelArtifact | None, Declared(_MODEL_DESCRIPTION)
        ] = None

        @override
        def identity_dump(self) -> dict[str, object]:
            """The hashed params, with ``model`` left out while it is unset.

            A run that fits has the identity it had before the field existed, so
            only the version segment of its run id moves.
            """
            dumped = super().identity_dump()
            if self.model is None:
                del dumped["model"]
            return dumped

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
        # A fit derives its basis from the entries in scope, so they are part of its
        # identity. A run given a model reads none of them for a fit.
        self.scope_dependent = self.params.model is None
        self.fits = self.params.model is None

    # ---------- Feature protocol ----------
    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        path = run_root / "model.joblib"
        if path.exists():
            self._adopt(joblib.load(path))
            return True
        model = self.params.model
        if model is None:
            return False
        # A named basis that is not there must not become a fit over this scope.
        if "model" not in artifact_paths:
            raise FileNotFoundError(
                f"[pair-posedistance-pca] The model {model.feature!r} run "
                f"{model.run_id!r} holds no {model.pattern!r}. Run the fit that "
                f"writes it first."
            )
        loaded: object = model.from_path(artifact_paths["model"])
        if not _is_bundle(loaded):
            raise ValueError(
                f"[pair-posedistance-pca] {artifact_paths['model']} is not a "
                f"pair-posedistance-pca model. Name the run of a "
                f"pair-posedistance-pca fit."
            )
        self._refuse_mismatch(loaded)
        self._adopt(loaded)
        return True

    def fit(self, inputs: InputStream) -> None:
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
            # The identity columns are named even when there is nothing to identify:
            # an empty frame missing them reads as unidentified rather than pair.
            return pd.DataFrame(
                columns=[C.frame_col, "id1", "id2", "perspective"]
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
    def _adopt(self, bundle: PairPoseDistancePCABundle) -> None:
        """Take a fitted basis as this feature's state."""
        self._ipca = bundle["ipca"]
        self._tri_i = bundle["tri_i"]
        self._tri_j = bundle["tri_j"]
        self._feat_len = bundle["feat_len"]
        self._fitted = True

    def _refuse_mismatch(self, bundle: PairPoseDistancePCABundle) -> None:
        """Refuse params that would not rebuild the fit's vectors or its scores."""
        _, _, feat_len = self._vector_layout()
        found: list[str] = []
        n_fitted = int(
            bundle["ipca"].n_components_  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        )
        if n_fitted != self.params.n_components:
            found.append(
                f"n_components is {self.params.n_components} and the model "
                f"holds {n_fitted}"
            )
        if bundle["feat_len"] != feat_len:
            found.append(
                f"pose, include_intra_A, include_intra_B and include_inter make "
                f"vectors of {feat_len} distances and the model was fitted on "
                f"{bundle['feat_len']}"
            )
        if found:
            raise ValueError(
                "[pair-posedistance-pca] The params do not match the model: "
                + "; ".join(found)
                + ". Give the apply step the params of the fit step."
            )

    def _vector_layout(self) -> tuple[np.ndarray, np.ndarray, int]:
        """The lower-triangle indices of one animal's keypoints and the vector length."""
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
        return tri_i, tri_j, feat_len

    def _effective_pose_n(self) -> int:
        """Return the number of pose points being used."""
        return len(self.params.pose.selected_indices())

    def _prep_pairs(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[tuple[str, int, int]]]:
        x_cols, y_cols = self.params.pose.keypoint_columns()
        pose_cols = x_cols + y_cols
        order_col = resolve_order_col(df)

        ensure_columns(df, [C.id_col, C.seq_col, order_col] + pose_cols)

        need = [C.id_col, C.seq_col, order_col] + pose_cols
        df_small = df[need]

        group_cols = [C.seq_col, C.id_col]

        df_small = clean_tracks_grouped(
            df_small, group_cols, pose_cols, order_col, self.params.interpolation
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
            self._tri_i, self._tri_j, self._feat_len = self._vector_layout()
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
        x_cols, y_cols = self.params.pose.keypoint_columns()
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

                n_base = feat_mat.shape[0]
                persp = np.zeros(n_base, dtype=np.int64)
                id1 = np.full(n_base, id_a, dtype=np.int64)
                id2 = np.full(n_base, id_b, dtype=np.int64)
                frame_arr = chunk[order_col].to_numpy() if has_frames else None

                if dup:
                    feat_rows2 = [
                        self._build_pair_feat(b, a) for a, b in zip(vals_a, vals_b)
                    ]
                    feat_dup = np.vstack(feat_rows2).astype(np.float32, copy=False)
                    n_dup = feat_dup.shape[0]
                    feat_mat = np.vstack([feat_mat, feat_dup])
                    persp = np.concatenate(
                        [persp, np.ones(n_dup, dtype=np.int64)], axis=0
                    )
                    # B is the focal on the mirrored block, so the ids swap with it.
                    id1 = np.concatenate([id1, np.full(n_dup, id_b, dtype=np.int64)])
                    id2 = np.concatenate([id2, np.full(n_dup, id_a, dtype=np.int64)])
                    if frame_arr is not None:
                        frame_arr = np.concatenate([frame_arr, frame_arr], axis=0)

                if self._feat_len is not None and feat_mat.shape[1] != self._feat_len:
                    msg = f"Feature length mismatch: got {feat_mat.shape[1]}, expected {self._feat_len}"
                    raise ValueError(msg)

                yield feat_mat, _BatchMeta(frame_arr, id1, id2), persp

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
