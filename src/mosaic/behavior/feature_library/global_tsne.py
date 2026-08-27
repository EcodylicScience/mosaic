"""GlobalTSNE feature."""

# openTSNE and faiss are untyped; suppress cascading Unknown errors from those libs.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportMissingImports=false

from __future__ import annotations

import gc
from pathlib import Path
from typing import Annotated, ClassVar, TypedDict, final

import joblib
import numpy as np
import pandas as pd
from openTSNE import TSNEEmbedding, affinity, initialization
from pydantic import Field

from mosaic.core.strict_model import StrictModel
from mosaic.core.pipeline.types import (
    EmitsLevel,
    DependencyLookup,
    GlobalModelParams,
    InputRequire,
    Inputs,
    InputStream,
    JoblibArtifact,
    JoblibLoadSpec,
    NpzArtifact,
    NpzLoadSpec,
    Result,
)
from mosaic.core.params import Declared
from mosaic.optional_dependency import require

from .helpers import ensure_columns
from .registry import register_feature
from mosaic.core.pipeline._utils import atomic_savez


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
        # Guarded rather than a bare import: `knn_method` defaults to "annoy",
        # so this is reached only by a run that asked for faiss by name, and an
        # unguarded ModuleNotFoundError names a package no install instruction
        # here ever mentions.
        faiss = require("faiss", "faiss", 'the "faiss" kNN backend of global-tsne')

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


_FIT_LEARNING_RATE_DESCRIPTION = (
    "The learning rate for the t-SNE embedding, forwarded to both "
    "optimization phases. The value auto makes openTSNE compute one from "
    "the template count and the exaggeration factor."
)

_FIT_EXAGGERATION_ITERS_DESCRIPTION = (
    "The number of iterations in the early exaggeration phase."
)

_FIT_EXAGGERATION_DESCRIPTION = (
    "The exaggeration factor during the early exaggeration phase, "
    "increasing the attractive force between nearby points to form more "
    "compact clusters."
)

_FIT_EXAGGERATION_MOMENTUM_DESCRIPTION = (
    "The momentum during the early exaggeration phase."
)

_FIT_ITERS_DESCRIPTION = (
    "The number of iterations in the refinement phase that follows early exaggeration."
)

_FIT_MOMENTUM_DESCRIPTION = "The momentum during the refinement phase."


class TSNEFitConfig(StrictModel):
    """openTSNE fitting parameters.

    Attributes:
        learning_rate: The learning rate for the t-SNE embedding, forwarded
            to both optimization phases. The value auto makes openTSNE
            compute one from the template count and the exaggeration
            factor.
        exaggeration_iters: The number of iterations in the early
            exaggeration phase.
        exaggeration: The exaggeration factor during the early exaggeration
            phase, increasing the attractive force between nearby points to
            form more compact clusters.
        exaggeration_momentum: The momentum during the early exaggeration
            phase.
        iters: The number of iterations in the refinement phase that
            follows early exaggeration.
        momentum: The momentum during the refinement phase.
    """

    learning_rate: Annotated[float | str, Declared(_FIT_LEARNING_RATE_DESCRIPTION)] = (
        "auto"
    )
    exaggeration_iters: Annotated[
        int, Declared(_FIT_EXAGGERATION_ITERS_DESCRIPTION)
    ] = Field(default=250, ge=1)
    exaggeration: Annotated[float, Declared(_FIT_EXAGGERATION_DESCRIPTION)] = Field(
        default=12, gt=0
    )
    exaggeration_momentum: Annotated[
        float, Declared(_FIT_EXAGGERATION_MOMENTUM_DESCRIPTION)
    ] = Field(default=0.5, ge=0)
    iters: Annotated[int, Declared(_FIT_ITERS_DESCRIPTION)] = Field(default=750, ge=1)
    momentum: Annotated[float, Declared(_FIT_MOMENTUM_DESCRIPTION)] = Field(
        default=0.8, ge=0
    )


_MAP_K_DESCRIPTION = (
    "The number of nearest neighbors used to place a new point's initial "
    "position in the fitted embedding."
)

_MAP_ITERS_DESCRIPTION = (
    "The number of optimization iterations when mapping a new point into "
    "the fitted embedding."
)

_MAP_LEARNING_RATE_DESCRIPTION = (
    "The learning rate used when mapping a new point into the fitted embedding."
)

_MAP_EXAGGERATION_DESCRIPTION = (
    "The exaggeration factor used when mapping a new point into the fitted embedding."
)

_MAP_MOMENTUM_DESCRIPTION = (
    "The momentum used when mapping a new point into the fitted embedding."
)

_MAP_CHUNK_SIZE_DESCRIPTION = (
    "How many rows apply() maps into the fitted embedding in one prepare_partial call."
)


class TSNEMapConfig(StrictModel):
    """Parameters for mapping new points into the fitted embedding.

    Attributes:
        k: The number of nearest neighbors used to place a new point's
            initial position in the fitted embedding.
        iters: The number of optimization iterations when mapping a new
            point into the fitted embedding.
        learning_rate: The learning rate used when mapping a new point into
            the fitted embedding.
        exaggeration: The exaggeration factor used when mapping a new point
            into the fitted embedding.
        momentum: The momentum used when mapping a new point into the
            fitted embedding.
        chunk_size: How many rows apply() maps into the fitted embedding in
            one prepare_partial call.
    """

    k: Annotated[int, Declared(_MAP_K_DESCRIPTION)] = Field(default=25, ge=1)
    iters: Annotated[int, Declared(_MAP_ITERS_DESCRIPTION)] = Field(default=100, ge=1)
    learning_rate: Annotated[float, Declared(_MAP_LEARNING_RATE_DESCRIPTION)] = Field(
        default=1.0, gt=0
    )
    exaggeration: Annotated[float, Declared(_MAP_EXAGGERATION_DESCRIPTION)] = Field(
        default=2.0, gt=0
    )
    momentum: Annotated[float, Declared(_MAP_MOMENTUM_DESCRIPTION)] = Field(
        default=0.0, ge=0
    )
    chunk_size: Annotated[int, Declared(_MAP_CHUNK_SIZE_DESCRIPTION)] = Field(
        default=50_000, ge=1
    )


class TSNEModelBundle(TypedDict):
    embedding: TSNEEmbedding
    feature_columns: list[str]
    version: str


class TSNEModelArtifact(JoblibArtifact[TSNEModelBundle]):
    """Fitted t-SNE embedding model (embedding.joblib)."""

    feature: str = "global-tsne"
    pattern: str = "embedding.joblib"
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


_MODEL_DESCRIPTION = (
    "A pre-fitted t-SNE embedding artifact to load, skipping the fit. "
    "Exactly one of templates and model must be given."
)

_RANDOM_STATE_DESCRIPTION = (
    "The random seed for PCA initialization and embedding optimization, "
    "and for neighbor search when knn_method is annoy."
)

_PERPLEXITY_DESCRIPTION = (
    "The t-SNE perplexity, controlling the effective number of nearest "
    "neighbors each point is compared against."
)

_KNN_METHOD_DESCRIPTION = (
    "The nearest-neighbor backend. Known values are annoy, faiss and "
    "faiss-gpu. Any other value falls back to annoy."
)

_N_JOBS_DESCRIPTION = (
    "How many parallel jobs openTSNE uses for the neighbor search and optimization."
)

_FIT_DESCRIPTION = (
    "The embedding-fit parameters: learning rate, exaggeration schedule and momentum."
)

_MAPPING_DESCRIPTION = (
    "The parameters for mapping new points into the fitted embedding: "
    "neighbor count, iterations, learning rate, exaggeration, momentum "
    "and chunk size."
)


@final
@register_feature
class GlobalTSNE:
    """Fit an openTSNE embedding on templates and map per-sequence data.

    Consumes a templates artifact (from ExtractTemplates, GlobalScaler, or
    any feature producing templates). Produces an embedding model bundle
    and template coordinates.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.global_tsne.GlobalTSNE.Params`.
    """

    category = "global"
    name: str = "global-tsne"
    version: str = "0.4"
    parallelizable = False
    scope_dependent = False
    accepts_overlap = False  # computes within a frame, so gains nothing
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "as-input"
    ModelArtifact = TSNEModelArtifact
    TSNECoordsArtifact = TSNECoordsArtifact

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "any"

    class Params(GlobalModelParams[TSNEModelArtifact]):
        """Global t-SNE parameters."""

        model: Annotated[TSNEModelArtifact | None, Declared(_MODEL_DESCRIPTION)] = (
            Field(default_factory=TSNEModelArtifact)
        )
        random_state: Annotated[int, Declared(_RANDOM_STATE_DESCRIPTION)] = 42
        perplexity: Annotated[int, Declared(_PERPLEXITY_DESCRIPTION)] = Field(
            default=50, ge=1
        )
        knn_method: Annotated[
            str,
            Field(examples=["annoy", "faiss", "faiss-gpu"]),
            Declared(_KNN_METHOD_DESCRIPTION),
        ] = "annoy"
        n_jobs: Annotated[int, Declared(_N_JOBS_DESCRIPTION)] = Field(default=8, ge=1)
        fit: Annotated[TSNEFitConfig, Declared(_FIT_DESCRIPTION)] = Field(
            default_factory=TSNEFitConfig
        )
        mapping: Annotated[TSNEMapConfig, Declared(_MAPPING_DESCRIPTION)] = Field(
            default_factory=TSNEMapConfig
        )

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
        dependency_lookups: dict[str, DependencyLookup],
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

    def fit(self, inputs: InputStream) -> None:
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
        atomic_savez(run_root / "global_tsne_templates.npz", Y=coords)
