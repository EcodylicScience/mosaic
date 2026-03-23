"""GlobalTSNE feature."""

# openTSNE and faiss are untyped; suppress cascading Unknown errors from those libs.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportMissingImports=false

from __future__ import annotations

import gc
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import ClassVar, TypedDict, final

import joblib
import numpy as np
import pandas as pd
from openTSNE import TSNEEmbedding, affinity, initialization
from pydantic import Field

from mosaic.core.pipeline._loaders import StrictModel
from mosaic.core.pipeline.types import (
    GlobalModelParams,
    InputRequire,
    Inputs,
    JoblibArtifact,
    JoblibLoadSpec,
    NpzArtifact,
    NpzLoadSpec,
    Result,
)

from .helpers import ensure_columns
from .registry import register_feature


class _FaissKNNIndex:
    """FAISS-backed kNN index conforming to the openTSNE KNNIndex protocol.

    Conforms to ``openTSNE.nearest_neighbors.KNNIndex``:
    - ``__init__(data, k, ...)`` stores training data and k
    - ``build()`` builds the FAISS index and returns (indices, distances) for the training data
    - ``query(query, k)`` finds nearest neighbors of new points (used by ``prepare_partial``)
    - ``.k`` attribute is read by ``PerplexityBasedNN``

    Parameters
    ----------
    data : np.ndarray, shape (n, d)
        Training data points.
    k : int
        Number of nearest neighbors.
    use_gpu : bool
        If True, use a FAISS GPU index (requires faiss-gpu).
    """

    VALID_METRICS = ["euclidean"]

    def __init__(
        self, data: np.ndarray, k: int, use_gpu: bool = False, **kwargs: object
    ) -> None:
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.k = k
        self.n_samples = data.shape[0]
        self._use_gpu = use_gpu
        self._index: object = None

    @staticmethod
    def check_metric(metric: str) -> str:
        if metric != "euclidean":
            raise ValueError(
                f"_FaissKNNIndex only supports euclidean metric, got {metric!r}"
            )
        return metric

    def build(self) -> tuple[np.ndarray, np.ndarray]:
        """Build FAISS index and return kNN for the training data."""
        import faiss

        d = self.data.shape[1]
        index = faiss.IndexFlatL2(d)
        if self._use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(self.data)
        self._index = index

        # Query k+1 neighbors (first result is the point itself)
        sq_dist, idx = index.search(self.data, self.k + 1)

        # Remove self-match (first column)
        indices = idx[:, 1:].astype(np.int64)
        distances = np.sqrt(np.maximum(sq_dist[:, 1:], 0)).astype(np.float64)

        return indices, distances

    def query(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Query nearest neighbors for new points against the built index."""
        query_f32 = np.ascontiguousarray(query, dtype=np.float32)
        sq_dist, idx = self._index.search(query_f32, k)  # pyright: ignore[reportAttributeAccessIssue]
        return idx.astype(np.int64), np.sqrt(np.maximum(sq_dist, 0)).astype(np.float64)


class TSNECoordsArtifact(NpzArtifact):
    """t-SNE coordinates of templates (global_tsne_templates.npz)."""

    feature: str = "global-tsne"
    pattern: str = "global_tsne_templates.npz"
    load: NpzLoadSpec = Field(default_factory=lambda: NpzLoadSpec(key="Y"))


class TSNEFitConfig(StrictModel):
    """openTSNE fitting parameters.

    Attributes:
        learning_rate: Learning rate ("auto" lets openTSNE compute). Default "auto".
        exaggeration_iters: Early exaggeration phase iterations. Default 250.
        exaggeration: Early exaggeration factor. Default 12.
        exaggeration_momentum: Momentum during early exaggeration. Default 0.5.
        iters: Refinement phase iterations. Default 750.
        momentum: Momentum during refinement. Default 0.8.
    """

    learning_rate: float | str = "auto"
    exaggeration_iters: int = Field(default=250, ge=1)
    exaggeration: float = Field(default=12, gt=0)
    exaggeration_momentum: float = Field(default=0.5, ge=0)
    iters: int = Field(default=750, ge=1)
    momentum: float = Field(default=0.8, ge=0)


class TSNEMapConfig(StrictModel):
    """Parameters for mapping new points into the fitted embedding.

    Attributes:
        k: Neighbors for partial embedding. Default 25.
        iters: Optimization iterations. Default 100.
        learning_rate: Learning rate. Default 1.0.
        exaggeration: Exaggeration factor. Default 2.0.
        momentum: Momentum. Default 0.0.
        chunk_size: Chunk size for large sequences. Default 50000.
    """

    k: int = Field(default=25, ge=1)
    iters: int = Field(default=100, ge=1)
    learning_rate: float = Field(default=1.0, gt=0)
    exaggeration: float = Field(default=2.0, gt=0)
    momentum: float = Field(default=0.0, ge=0)
    chunk_size: int = Field(default=50_000, ge=1)


class TSNEModelBundle(TypedDict):
    embedding: TSNEEmbedding
    feature_columns: list[str]
    version: str


class TSNEModelArtifact(JoblibArtifact[TSNEModelBundle]):
    """Fitted t-SNE embedding model (embedding.joblib)."""

    feature: str = "global-tsne"
    pattern: str = "embedding.joblib"
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


@final
@register_feature
class GlobalTSNE:
    """Fit an openTSNE embedding on templates and map per-sequence data.

    Consumes a templates artifact (from ExtractTemplates, GlobalScaler, or
    any feature producing templates). Produces an embedding model bundle
    and template coordinates.
    """

    name: str = "global-tsne"
    version: str = "0.4"
    parallelizable = False
    scope_dependent = False

    ModelArtifact = TSNEModelArtifact
    TSNECoordsArtifact = TSNECoordsArtifact

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "any"

    class Params(GlobalModelParams[TSNEModelArtifact]):
        """Global t-SNE parameters.

        Attributes:
            templates: Templates artifact to fit embedding on.
            model: Pre-fitted embedding model artifact (skip fit).
            random_state: Random seed. Default 42.
            perplexity: t-SNE perplexity. Default 50.
            knn_method: kNN method ("annoy", "faiss", "faiss-gpu"). Default "annoy".
            n_jobs: Parallel jobs for openTSNE. Default 8.
            fit: Embedding fitting parameters.
            mapping: Partial embedding mapping parameters.
        """

        model: TSNEModelArtifact | None = Field(default_factory=TSNEModelArtifact)
        random_state: int = 42
        perplexity: int = Field(default=50, ge=1)
        knn_method: str = "annoy"
        n_jobs: int = Field(default=8, ge=1)
        fit: TSNEFitConfig = Field(default_factory=TSNEFitConfig)
        mapping: TSNEMapConfig = Field(default_factory=TSNEMapConfig)

    def __init__(
        self,
        inputs: GlobalTSNE.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

        self._feature_columns: list[str] | None = None
        self._embedding: TSNEEmbedding | None = None
        self._templates: np.ndarray | None = None

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_indices: dict[str, pd.DataFrame],
    ) -> bool:
        self._feature_columns = None
        self._embedding = None
        self._templates = None

        # Check for cached model
        cached_path = run_root / "embedding.joblib"
        if cached_path.exists():
            bundle: TSNEModelBundle = TSNEModelArtifact().from_path(cached_path)
            self._embedding = bundle["embedding"]
            self._feature_columns = bundle["feature_columns"]
            return True

        # Load pre-fitted model from artifact_paths
        if self.params.model is not None and "model" in artifact_paths:
            bundle = self.params.model.from_path(artifact_paths["model"])
            self._embedding = bundle["embedding"]
            self._feature_columns = bundle["feature_columns"]
            return True

        # Load templates from artifact_paths
        if self.params.templates is not None and "templates" in artifact_paths:
            df = self.params.templates.from_path(artifact_paths["templates"])
            self._feature_columns = list(df.columns)
            self._templates = df.to_numpy(dtype=np.float64)
            return False

        return False

    def fit(
        self,
        inputs: Callable[[], Iterator[tuple[str, pd.DataFrame]]],
    ) -> None:
        if self._templates is None:
            msg = "[global-tsne] No templates loaded. Check load_state."
            raise RuntimeError(msg)

        templates = self._templates.astype(np.float32, copy=False)

        perplexity = self.params.perplexity
        knn_method = self.params.knn_method.lower()
        if knn_method in ("faiss", "faiss-gpu"):
            use_gpu = knn_method == "faiss-gpu"
            k_neighbors = min(3 * perplexity, templates.shape[0] - 1)
            faiss_knn = _FaissKNNIndex(templates, k_neighbors, use_gpu=use_gpu)
            aff = affinity.PerplexityBasedNN(
                knn_index=faiss_knn,
                perplexity=perplexity,
                n_jobs=self.params.n_jobs,
            )
        else:
            aff = affinity.PerplexityBasedNN(
                templates,
                perplexity=perplexity,
                metric="euclidean",
                method="annoy",
                n_jobs=self.params.n_jobs,
                random_state=self.params.random_state,
            )

        init = initialization.pca(templates, random_state=self.params.random_state)
        embedding = TSNEEmbedding(
            init,
            aff,
            learning_rate=self.params.fit.learning_rate,
            negative_gradient_method="fft",
            n_jobs=self.params.n_jobs,
            random_state=self.params.random_state,
        )
        embedding.optimize(
            n_iter=self.params.fit.exaggeration_iters,
            exaggeration=self.params.fit.exaggeration,
            momentum=self.params.fit.exaggeration_momentum,
            inplace=True,
            verbose=False,
        )
        embedding.optimize(
            n_iter=self.params.fit.iters,
            momentum=self.params.fit.momentum,
            inplace=True,
            verbose=False,
        )

        self._embedding = embedding
        self._templates = None  # free memory

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._embedding is None or self._feature_columns is None:
            msg = "[global-tsne] Not fitted. Call fit() or load_state() first."
            raise RuntimeError(msg)

        ensure_columns(df, self._feature_columns)
        features = df[self._feature_columns].to_numpy(dtype=np.float32, copy=False)

        valid_mask = np.isfinite(features).all(axis=1)
        valid_features = features[valid_mask]

        coords = np.full((features.shape[0], 2), np.nan, dtype=np.float32)
        if valid_features.shape[0] > 0:
            chunk_size = self.params.mapping.chunk_size
            valid_coords = np.empty((valid_features.shape[0], 2), dtype=np.float32)
            for start in range(0, valid_features.shape[0], chunk_size):
                end = min(start + chunk_size, valid_features.shape[0])
                valid_coords[start:end] = self._map_chunk(valid_features[start:end])
            coords[valid_mask] = valid_coords

        meta_cols = sorted(set(df.columns) - set(self._feature_columns))
        out = df[meta_cols].copy()
        out["tsne_x"] = coords[:, 0]
        out["tsne_y"] = coords[:, 1]
        return out

    def _map_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Map a chunk of feature vectors to t-SNE coordinates."""
        assert self._embedding is not None
        partial_embedding = self._embedding.prepare_partial(
            chunk,
            initialization="median",
            k=self.params.mapping.k,
            perplexity=self.params.perplexity,
        )
        partial_embedding.optimize(
            n_iter=self.params.mapping.iters,
            learning_rate=self.params.mapping.learning_rate,
            exaggeration=self.params.mapping.exaggeration,
            momentum=self.params.mapping.momentum,
            inplace=True,
            verbose=False,
        )
        coords = np.asarray(partial_embedding, dtype=np.float32).copy()
        del partial_embedding
        gc.collect()
        return coords

    def save_state(self, run_root: Path) -> None:
        if self._embedding is None or self._feature_columns is None:
            return
        run_root.mkdir(parents=True, exist_ok=True)

        bundle: TSNEModelBundle = {
            "embedding": self._embedding,
            "feature_columns": self._feature_columns,
            "version": self.version,
        }
        joblib.dump(bundle, run_root / "embedding.joblib")

        # Save template coordinates for visualization
        coords = np.asarray(self._embedding)
        np.savez_compressed(run_root / "global_tsne_templates.npz", Y=coords)
