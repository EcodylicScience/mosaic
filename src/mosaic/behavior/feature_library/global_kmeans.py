"""
GlobalKMeansClustering feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, final

import joblib
import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.cluster import KMeans as _SklearnKMeans

from mosaic.core.dataset import _dataset_base_dir

from .spec import register_feature
from mosaic.core.helpers import to_safe_name

from .global_tsne import GlobalTSNE
from .helpers import (
    PartialIndexRow,
    StreamingFeatureHelper,
    _build_index_row,
    _build_sequence_lookup,
    _get_feature_run_root,
    _load_joblib_artifact,
    _parse_scope_filter,
    _resolve_sequence_identity,
)
from .spec import (
    ArtifactSpec,
    FeatureLabelsSource,
    InputRequire,
    Inputs,
    JoblibLoadSpec,
    NNResult,
    NpzLoadSpec,
    OutputType,
    Params,
    ParquetLoadSpec,
    Result,
)


def _get_kmeans_class(device: str) -> type:
    """Return the appropriate KMeans class for the requested device.

    When ``device="cuda"``, tries to import cuML's GPU-accelerated KMeans.
    Falls back to scikit-learn KMeans (with a warning) if cuML is unavailable.
    """
    if device == "cuda":
        try:
            from cuml.cluster import KMeans as _CumlKMeans

            return _CumlKMeans
        except ImportError:
            print(
                "[global-kmeans] cuML not available; falling back to sklearn KMeans. "
                "Install cuml for GPU acceleration.",
                file=sys.stderr,
            )
    return _SklearnKMeans


@final
@register_feature
class GlobalKMeansClustering:
    """
    Global K-Means that fits on an artifact matrix resolved from the dataset's
    feature index. Optional assign phase uses Result-based inputs to label
    per-frame features with the fitted cluster IDs.
    """

    name: str = "global-kmeans"
    version: str = "0.4"
    output_type: OutputType = "global"
    parallelizable = False
    skip_transform_phase: bool = True

    class ModelArtifact(ArtifactSpec[JoblibLoadSpec]):
        """KMeans model (model.joblib)."""

        feature: str = "global-kmeans"
        pattern: str = "model.joblib"
        load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)

    class ClusterCentersArtifact(ArtifactSpec[NpzLoadSpec]):
        """Cluster center vectors (cluster_centers.npz)."""

        feature: str = "global-kmeans"
        pattern: str = "cluster_centers.npz"
        load: NpzLoadSpec = Field(default_factory=lambda: NpzLoadSpec(key="centers"))

    class ClusterSizesArtifact(ArtifactSpec[ParquetLoadSpec]):
        """Per-cluster sample counts (cluster_sizes.parquet)."""

        feature: str = "global-kmeans"
        pattern: str = "cluster_sizes.parquet"
        load: ParquetLoadSpec = Field(default_factory=ParquetLoadSpec)

    class ArtifactLabelsArtifact(ArtifactSpec[NpzLoadSpec]):
        """Labels for the artifact points used in fitting (artifact_labels.npz)."""

        feature: str = "global-kmeans"
        pattern: str = "artifact_labels.npz"
        load: NpzLoadSpec = Field(default_factory=lambda: NpzLoadSpec(key="labels"))

    class SeqLabelsArtifact(FeatureLabelsSource):
        """Per-sequence cluster labels (global_kmeans_labels_seq=*.npz)."""

        feature: str = "global-kmeans"
        pattern: str = "global_kmeans_labels_seq=*.npz"

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "any"

    class Params(Params):
        """Global K-means clustering parameters.

        Attributes:
            k: Number of clusters. Default 100.
            random_state: Random seed. Default 42.
            n_init: KMeans initializations. Default "auto".
            max_iter: Max iterations per run. Default 300.
            device: Compute device. Default "cpu".
            templates: Templates artifact to fit on.
            label_artifact_points: Label points used for fitting. Default True.
            scaler: Optional scaler for assign phase.
            pair_filter: Nearest-neighbor pair filter. Default None.
        """

        k: int = Field(default=100, ge=1)
        random_state: int = 42
        n_init: str | int = "auto"
        max_iter: int = Field(default=300, ge=1)
        device: str = "cpu"
        templates: GlobalTSNE.TemplatesArtifact
        label_artifact_points: bool = True
        scaler: GlobalTSNE.ScalerArtifact | None = None
        pair_filter: NNResult | None = None

    def __init__(
        self,
        inputs: GlobalKMeansClustering.Inputs = Inputs(()),
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True

        self._ds: object = None
        self._kmeans: object = None
        self._fit_dim: int | None = None
        self._fit_columns: list[str] | None = None
        self._fit_artifact_info: dict[str, object] | None = None
        self._artifact_labels: np.ndarray | None = None
        self._assign_labels: dict[str, np.ndarray] = {}
        self._assign_frames: dict[str, np.ndarray] = {}
        self._assign_id1: dict[str, np.ndarray] = {}
        self._assign_id2: dict[str, np.ndarray] = {}
        self._assign_entity_level: dict[str, str] = {}
        self._pair_map: dict[str, tuple[str, str]] = {}
        self._sequence_lookup_cache: dict[str, tuple[str, str]] | None = None
        self._additional_index_rows: list[PartialIndexRow] = []
        self._allowed_safe_sequences: set[str] | None = None
        self._scope_filter: dict[str, object] = {}

    def set_scope_filter(self, scope: dict[str, object] | None) -> None:
        self._scope_filter = scope or {}

    def set_scope_constraints(self, scope: dict[str, object] | None) -> None:
        """Capture dataset-level sequence filters so assignment can respect them."""
        self._allowed_safe_sequences, self._pair_map = _parse_scope_filter(scope)
        self._sequence_lookup_cache = None

    def _get_sequence_lookup(self) -> dict[str, tuple[str, str]]:
        if self._sequence_lookup_cache is not None:
            return self._sequence_lookup_cache
        self._sequence_lookup_cache = _build_sequence_lookup(
            self._ds, self._allowed_safe_sequences
        )
        return self._sequence_lookup_cache

    # Dataset binding
    def needs_fit(self) -> bool:
        return True

    def supports_partial_fit(self) -> bool:
        return False

    def loads_own_data(self) -> bool:
        # NOTE: time/frame scope filters (filter_start_frame, etc.) are not
        # applied when loading own data. run_feature() raises RuntimeError
        # if these filters are set. Future work: apply them during loading.
        return True

    def bind_dataset(self, ds: object) -> None:
        self._ds = ds

    # --- Fit helpers ---

    def _load_npz_matrix(
        self, files: list[Path], key: str, transpose: bool
    ) -> tuple[np.ndarray, dict[str, object]]:
        mats = []
        for p in files:
            npz = np.load(p, allow_pickle=True)
            if key not in npz.files:
                continue
            A = np.asarray(npz[key])
            if A.ndim == 1:
                A = A[None, :]
            A = A.astype(np.float32, copy=False)
            A = A.T if transpose else A
            mats.append(A)
        if not mats:
            raise FileNotFoundError(
                f"No NPZ containing key '{key}' among: {[p.name for p in files]}"
            )
        X = np.vstack(mats)
        meta: dict[str, object] = {
            "loader_kind": "npz",
            "key": key,
            "transpose": bool(transpose),
        }
        return X, meta

    def _load_parquet_matrix(
        self, files: list[Path], spec: ParquetLoadSpec
    ) -> tuple[np.ndarray, dict[str, object], list[str]]:
        cols = spec.columns
        drop_cols = set(spec.drop_columns or [])
        numeric_only = spec.numeric_only

        def load_df(p: Path) -> pd.DataFrame:
            df = pd.read_parquet(p)
            if drop_cols:
                df = df.drop(columns=list(drop_cols), errors="ignore")
            if cols is not None:
                use = [c for c in cols if c in df.columns]
                if not use:
                    return pd.DataFrame()
                df = df[use]
            else:
                if numeric_only:
                    df = df.select_dtypes(include=["number"])
                    for mc in (
                        "frame",
                        "time",
                        "id",
                        "id1",
                        "id2",
                        "id_a",
                        "id_b",
                        "id_A",
                        "id_B",
                    ):
                        if mc in df.columns:
                            df = df.drop(columns=[mc])
                else:
                    df = df.apply(pd.to_numeric, errors="coerce")
            return df

        dfs = []
        first_cols: list[str] | None = None
        for p in files:
            df = load_df(p)
            if df.empty:
                continue
            if first_cols is None:
                first_cols = df.columns.tolist()
            else:
                df = df.reindex(columns=first_cols)
            dfs.append(df)

        if not dfs:
            raise FileNotFoundError(
                f"No Parquet files with usable numeric columns among: {[p.name for p in files]}"
            )

        D = pd.concat(dfs, ignore_index=True)
        A = D.to_numpy(dtype=np.float32, copy=False)
        assert first_cols is not None
        meta: dict[str, object] = {
            "loader_kind": "parquet",
            "columns": first_cols,
            "drop_columns": list(drop_cols) if drop_cols else [],
            "numeric_only": numeric_only,
        }
        return A, meta, first_cols

    def _load_artifact_matrix(self) -> np.ndarray:
        art = self.params.templates
        feature = art.feature
        resolved_run_id, run_root = _get_feature_run_root(self._ds, feature, art.run_id)
        pattern = art.pattern
        loader = art.load
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching '{pattern}' in {run_root}")

        self._fit_columns = None
        X, meta = self._load_npz_matrix(files, loader.key, loader.transpose)

        self._fit_artifact_info = {
            "feature": feature,
            "run_id": resolved_run_id,
            "run_root": str(run_root),
            "pattern": pattern,
            "loader": meta,
        }
        return X

    # --- Fit / Transform / Save ---

    def partial_fit(self, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        self._assign_labels = {}
        self._assign_frames = {}
        self._assign_id1 = {}
        self._assign_id2 = {}
        self._assign_entity_level = {}
        self._additional_index_rows = []

        X = self._load_artifact_matrix()
        self._fit_dim = X.shape[1]

        if X.shape[0] < self.params.k:
            raise ValueError(
                f"Not enough samples to fit KMeans: n={X.shape[0]} < k={self.params.k}"
            )

        KMeansCls = _get_kmeans_class(self.params.device)
        self._kmeans = KMeansCls(
            n_clusters=self.params.k,
            n_init=self.params.n_init,
            random_state=self.params.random_state,
            max_iter=self.params.max_iter,
        ).fit(X)

        if self.params.label_artifact_points:
            self._artifact_labels = self._kmeans.predict(X)

        # Assign phase: label per-frame features if inputs provided
        feature_inputs = self.inputs.feature_inputs
        if feature_inputs:
            import gc

            import pyarrow as pa

            scaler = None
            scaler_spec = self.params.scaler
            if scaler_spec is not None:
                scaler = _load_joblib_artifact(self._ds, scaler_spec)

            helper = StreamingFeatureHelper(self._ds, "global-kmeans")
            if self.params.pair_filter:
                helper.set_pair_filter(self.params.pair_filter)
            scope_filter = (
                {"safe_sequences": self._allowed_safe_sequences}
                if self._allowed_safe_sequences
                else None
            )
            manifest = helper.build_manifest_from_results(
                feature_inputs, scope_filter=scope_filter
            )

            keys = list(manifest.keys())
            n_keys = len(keys)
            for i, key in enumerate(keys):
                X_full, frames, id1_vals, id2_vals, entity_level = (
                    helper.load_key_data_with_identity(
                        manifest[key],
                        extract_frames=True,
                        key=key,
                    )
                )
                if X_full is None:
                    continue
                D_total = X_full.shape[1]

                if scaler is not None:
                    if not hasattr(scaler, "n_features_in_"):
                        raise ValueError(
                            "Scaler object must have n_features_in_ (e.g. sklearn StandardScaler)"
                        )
                    if scaler.n_features_in_ != D_total:
                        raise ValueError(
                            f"Scaler expects n_features_in_={getattr(scaler, 'n_features_in_', None)}, "
                            f"got {D_total} columns for key={key}"
                        )
                    X_use = scaler.transform(X_full)
                    del X_full
                else:
                    if self._fit_dim is None:
                        raise RuntimeError(
                            "GlobalKMeansClustering internal error: fit_dim is None before assignment."
                        )
                    if D_total != self._fit_dim:
                        raise ValueError(
                            f"Assign-without-scaler requires feature dim {self._fit_dim}, "
                            f"but got {D_total} for key={key}. Provide a scaler or align inputs."
                        )
                    X_use = X_full

                labels = self._kmeans.predict(X_use)
                self._assign_labels[key] = labels.astype(np.int32)
                self._assign_frames[key] = (
                    None
                    if frames is None
                    else np.asarray(frames, dtype=np.int64).ravel()
                )
                if id1_vals is not None:
                    self._assign_id1[key] = np.asarray(
                        id1_vals, dtype=np.float64
                    ).ravel()
                if id2_vals is not None:
                    self._assign_id2[key] = np.asarray(
                        id2_vals, dtype=np.float64
                    ).ravel()
                self._assign_entity_level[key] = str(entity_level or "global")

                del X_use, labels
                gc.collect()
                pa.default_memory_pool().release_unused()

                if (i + 1) % 10 == 0 or i == n_keys - 1:
                    print(
                        f"[global-kmeans] Processed {i + 1}/{n_keys} sequences",
                        file=sys.stderr,
                    )

    def finalize_fit(self) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign clusters to an input DataFrame.

        Only succeeds if columns can be aligned to the fitted feature space:
          - If we fitted on parquet with explicit/numeric columns, require those columns.
          - Otherwise, require numeric matrix with matching dimensionality.
        If alignment fails, return an empty DataFrame (no error).
        """
        if self._kmeans is None or self._fit_dim is None:
            raise RuntimeError("GlobalKMeansClustering not fitted yet.")

        if self._fit_columns:
            missing = [c for c in self._fit_columns if c not in df.columns]
            if missing:
                return pd.DataFrame(columns=["frame", "cluster"])
            A = df[self._fit_columns].to_numpy(dtype=np.float32, copy=False)
        else:
            num = df.select_dtypes(include=["number"])
            for mc in (
                "frame",
                "time",
                "id",
                "id1",
                "id2",
                "id_a",
                "id_b",
                "id_A",
                "id_B",
            ):
                if mc in num.columns:
                    num = num.drop(columns=[mc])
            if num.shape[1] != self._fit_dim:
                return pd.DataFrame(columns=["frame", "cluster"])
            A = num.to_numpy(dtype=np.float32, copy=False)

        mask = np.isfinite(A).all(axis=1)
        labels = np.full(A.shape[0], -1, dtype=np.int32)
        if mask.any():
            labels[mask] = self._kmeans.predict(A[mask])

        out = pd.DataFrame(
            {
                "frame": df["frame"].astype(int, errors="ignore")
                if "frame" in df.columns
                else np.arange(len(df), dtype=int),
                "cluster": labels,
            }
        )
        return out

    def save_model(self, path: Path) -> None:
        run_root = path.parent
        run_root.mkdir(parents=True, exist_ok=True)
        self._additional_index_rows = []

        joblib.dump(
            {
                "kmeans": self._kmeans,
                "fit_dim": int(self._fit_dim or 0),
                "fit_columns": self._fit_columns,
                "artifact_info": self._fit_artifact_info,
                "version": self.version,
                "params": self.params.model_dump(),
            },
            path,
        )

        centers = np.asarray(self._kmeans.cluster_centers_, dtype=np.float32)
        np.savez_compressed(run_root / "cluster_centers.npz", centers=centers)

        if self._artifact_labels is not None:
            np.savez_compressed(
                run_root / "artifact_labels.npz", labels=self._artifact_labels
            )
            uniq, cnt = np.unique(self._artifact_labels, return_counts=True)
            pd.DataFrame(
                {"cluster": uniq.astype(int), "count": cnt.astype(int)}
            ).to_parquet(run_root / "cluster_sizes.parquet", index=False)

        for safe_seq, labels in self._assign_labels.items():
            labels_arr = np.asarray(labels, dtype=np.int32).ravel()
            frames = self._assign_frames.get(safe_seq)
            if frames is not None and len(frames) != len(labels_arr):
                frames = None
            if frames is None:
                frames = np.arange(len(labels_arr), dtype=np.int64)
            id1_vals = self._assign_id1.get(safe_seq)
            id2_vals = self._assign_id2.get(safe_seq)
            if id1_vals is None or len(id1_vals) != len(labels_arr):
                id1_vals = np.full(len(labels_arr), np.nan, dtype=np.float64)
            if id2_vals is None or len(id2_vals) != len(labels_arr):
                id2_vals = np.full(len(labels_arr), np.nan, dtype=np.float64)
            entity_level = self._assign_entity_level.get(safe_seq, "global")
            group, sequence = _resolve_sequence_identity(
                safe_seq, self._pair_map, self._get_sequence_lookup()
            )
            safe_group = to_safe_name(group) if group else ""
            out_name = f"{safe_group + '__' if safe_group else ''}{safe_seq}.parquet"
            out_path = run_root / out_name
            df_out = pd.DataFrame(
                {
                    "frame": frames.astype(np.int64, copy=False),
                    "cluster": labels_arr,
                    "id1": pd.array(id1_vals, dtype="Int64"),
                    "id2": pd.array(id2_vals, dtype="Int64"),
                    "entity_level": np.full(
                        len(labels_arr), entity_level, dtype=object
                    ),
                    "sequence": sequence,
                    "group": group,
                }
            )
            df_out.to_parquet(out_path, index=False)
            self._additional_index_rows.append(
                _build_index_row(
                    group,
                    sequence,
                    out_path,
                    int(len(df_out)),
                    dataset_root=_dataset_base_dir(self._ds) if self._ds else None,
                )
            )
            fname = f"global_kmeans_labels_seq={safe_seq}.npz"
            np.savez_compressed(
                run_root / fname,
                labels=labels_arr,
                frames=frames,
                id1=id1_vals,
                id2=id2_vals,
                entity_level=np.array([entity_level], dtype=object),
            )

        marker_seq = "__global__"
        safe_marker_seq = to_safe_name(marker_seq)
        marker_path = run_root / f"{safe_marker_seq}.parquet"
        marker_df = pd.DataFrame({"run_marker": [True]})
        marker_df.to_parquet(marker_path, index=False)
        self._additional_index_rows.append(
            _build_index_row(
                "",
                marker_seq,
                marker_path,
                int(len(marker_df)),
                dataset_root=_dataset_base_dir(self._ds) if self._ds else None,
            )
        )

    def load_model(self, path: Path) -> None:
        bundle = joblib.load(path)
        self._kmeans = bundle["kmeans"]
        self._fit_dim = int(bundle.get("fit_dim") or 0)
        self._fit_columns = bundle.get("fit_columns")
        self._fit_artifact_info = bundle.get("artifact_info", {})
        saved = bundle.get("params", {})
        if isinstance(saved, dict):
            self.params = self.Params.model_validate(saved)

    def get_additional_index_rows(self) -> list[PartialIndexRow]:
        return list(self._additional_index_rows)
