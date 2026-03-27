"""
GlobalKMeansClustering feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar, Literal, TypedDict, final

import joblib
import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.cluster import KMeans as _SklearnKMeans

from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    GlobalModelParams,
    InputRequire,
    Inputs,
    InputStream,
    JoblibArtifact,
    JoblibLoadSpec,
    NNResult,
    NpzArtifact,
    NpzLoadSpec,
    ParquetArtifact,
    ParquetLoadSpec,
    Result,
)

from .helpers import ensure_columns
from .registry import register_feature


class KMeansModelBundle(TypedDict):
    kmeans: _SklearnKMeans
    feature_columns: list[str]
    version: str


def _get_kmeans_class(device: str) -> type[_SklearnKMeans]:
    """Return the appropriate KMeans class for the requested device.

    When ``device="cuda"``, tries to import cuML's GPU-accelerated KMeans.
    Falls back to scikit-learn KMeans (with a warning) if cuML is unavailable.
    """
    if device == "cuda":
        try:
            from cuml.cluster import (  # pyright: ignore[reportMissingImports]
                KMeans as _CumlKMeans,  # pyright: ignore[reportUnknownVariableType]
            )

            return _CumlKMeans  # pyright: ignore[reportUnknownVariableType]
        except ImportError:
            print(
                "[global-kmeans] cuML not available; falling back to sklearn KMeans. "
                "Install cuml for GPU acceleration.",
                file=sys.stderr,
            )
    return _SklearnKMeans


class KMeansModelArtifact(JoblibArtifact[KMeansModelBundle]):
    """KMeans model (model.joblib)."""

    feature: str = "global-kmeans"
    pattern: str = "model.joblib"
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


class KMeansClusterCentersArtifact(NpzArtifact):
    """Cluster center vectors (cluster_centers.npz)."""

    feature: str = "global-kmeans"
    pattern: str = "cluster_centers.npz"
    load: NpzLoadSpec = Field(default_factory=lambda: NpzLoadSpec(key="centers"))


class KMeansClusterSizesArtifact(ParquetArtifact):
    """Per-cluster sample counts (cluster_sizes.parquet)."""

    feature: str = "global-kmeans"
    pattern: str = "cluster_sizes.parquet"
    load: ParquetLoadSpec = Field(default_factory=ParquetLoadSpec)


class KMeansArtifactLabelsArtifact(NpzArtifact):
    """Labels for the artifact points used in fitting (artifact_labels.npz)."""

    feature: str = "global-kmeans"
    pattern: str = "artifact_labels.npz"
    load: NpzLoadSpec = Field(default_factory=lambda: NpzLoadSpec(key="labels"))


@final
@register_feature
class GlobalKMeansClustering:
    """
    Global K-Means clustering on templates loaded via load_state.
    Per-sequence cluster assignment is done in apply().

    Params:
        templates: Templates artifact to fit on (inherited from
            GlobalModelParams).
        model: Pre-fitted KMeansModelArtifact to load (skip fit).
            Default: KMeansModelArtifact().
        k: Number of clusters. Default: 100.
        random_state: Random seed for KMeans initialization.
            Default: 42.
        n_init: Number of KMeans initializations to run. Default: "auto".
        max_iter: Maximum iterations per KMeans run. Default: 300.
        device: Compute device — "cpu" or "cuda" (requires cuML).
            Default: "cpu".
        label_artifact_points: If True, assign cluster labels to the
            template points used for fitting. Default: True.
        pair_filter: Optional NNResult for nearest-neighbor pair
            filtering during dependency resolution. Default: None.
    """

    name: str = "global-kmeans"
    version: str = "0.4"
    parallelizable = False
    scope_dependent = False

    ModelArtifact = KMeansModelArtifact
    ClusterCentersArtifact = KMeansClusterCentersArtifact
    ClusterSizesArtifact = KMeansClusterSizesArtifact
    ArtifactLabelsArtifact = KMeansArtifactLabelsArtifact

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "any"

    class Params(GlobalModelParams[KMeansModelArtifact]):
        """Global K-means clustering parameters.

        Attributes:
            templates: Templates artifact to fit on (inherited).
            model: Pre-fitted KMeans model artifact (skip fit).
            k: Number of clusters. Default 100.
            random_state: Random seed. Default 42.
            n_init: KMeans initializations. Default "auto".
            max_iter: Max iterations per run. Default 300.
            device: Compute device. Default "cpu".
            label_artifact_points: Label points used for fitting. Default True.
            pair_filter: Nearest-neighbor pair filter for dependency resolution. Default None.
        """

        model: KMeansModelArtifact | None = Field(default_factory=KMeansModelArtifact)
        k: int = Field(default=100, ge=1)
        random_state: int = 42
        n_init: Literal["auto"] | int = "auto"
        max_iter: int = Field(default=300, ge=1)
        device: str = "cpu"
        label_artifact_points: bool = True
        pair_filter: NNResult | None = None

    def __init__(
        self,
        inputs: GlobalKMeansClustering.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

        self._kmeans: _SklearnKMeans | None = None
        self._feature_columns: list[str] | None = None
        self._artifact_labels: np.ndarray | None = None
        self._templates: np.ndarray | None = None

    # --- Feature protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._kmeans = None
        self._feature_columns = None
        self._artifact_labels = None
        self._templates = None

        # Check for cached model
        cached_path = run_root / "model.joblib"
        if cached_path.exists():
            bundle: KMeansModelBundle = KMeansModelArtifact().from_path(cached_path)
            self._kmeans = bundle["kmeans"]
            self._feature_columns = bundle["feature_columns"]
            return True

        # Load pre-fitted model from artifact_paths
        if self.params.model is not None and "model" in artifact_paths:
            bundle = self.params.model.from_path(artifact_paths["model"])
            self._kmeans = bundle["kmeans"]
            self._feature_columns = bundle["feature_columns"]
            return True

        # Load templates from artifact_paths
        if self.params.templates is not None and "templates" in artifact_paths:
            df = self.params.templates.from_path(artifact_paths["templates"])
            self._feature_columns = list(df.columns)
            self._templates = df.to_numpy(dtype=np.float64)
            return False

        return False

    def fit(self, inputs: InputStream) -> None:
        if self._templates is None:
            msg = "[global-kmeans] No templates loaded. Check load_state."
            raise RuntimeError(msg)

        self._templates = self._templates.astype(np.float32, copy=False)

        if self._templates.shape[0] < self.params.k:
            msg = (
                f"Not enough samples to fit KMeans: "
                f"n={self._templates.shape[0]} < k={self.params.k}"
            )
            raise ValueError(msg)

        KMeansCls = _get_kmeans_class(self.params.device)
        self._kmeans = KMeansCls(
            n_clusters=self.params.k,
            n_init=self.params.n_init,
            random_state=self.params.random_state,
            max_iter=self.params.max_iter,
        ).fit(self._templates)

        if self.params.label_artifact_points:
            self._artifact_labels = self._kmeans.predict(self._templates)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._kmeans is None or self._feature_columns is None:
            msg = "GlobalKMeansClustering not fitted yet."
            raise RuntimeError(msg)

        ensure_columns(df, [C.frame_col] + self._feature_columns)

        arr = df[self._feature_columns].to_numpy(dtype=np.float32, copy=False)

        mask = np.isfinite(arr).all(axis=1)
        idx = np.flatnonzero(mask).tolist()
        valid = df.iloc[idx].reset_index(drop=True)
        meta_cols = sorted(set(valid.columns) - set(self._feature_columns))
        out = valid[meta_cols].copy()
        out["cluster"] = self._kmeans.predict(arr[mask])
        return out

    def save_state(self, run_root: Path) -> None:
        if self._kmeans is None:
            return
        run_root.mkdir(parents=True, exist_ok=True)

        bundle: KMeansModelBundle = {
            "kmeans": self._kmeans,
            "feature_columns": self._feature_columns or [],
            "version": self.version,
        }
        joblib.dump(bundle, run_root / "model.joblib")

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
