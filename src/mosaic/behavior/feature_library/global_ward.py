"""
GlobalWardClustering feature.

Fits Ward hierarchical linkage on templates, cuts at n_clusters,
builds centroids, and assigns per-sequence rows via 1-NN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, TypedDict, final

import joblib
import numpy as np
import pandas as pd
from pydantic import Field
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as _sch_linkage
from sklearn.neighbors import NearestNeighbors

from mosaic.core.pipeline.types import (
    COLUMNS as C,
    EmitsLevel,
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
    Result,
    TemplatesRef,
)
from mosaic.core.params import Declared

from .helpers import ensure_columns
from ._pair_filter_descriptions import PAIR_FILTER_DESCRIPTION
from .registry import register_feature

_TEMPLATES_DESCRIPTION = (
    "The templates artifact to cluster. Mutually exclusive with model."
)

_MODEL_DESCRIPTION = (
    "A pre-fitted Ward clustering model artifact to load, skipping the fit. "
    "Mutually exclusive with templates."
)

_N_CLUSTERS_DESCRIPTION = (
    "The largest number of clusters cut from the Ward linkage tree."
)

_METHOD_DESCRIPTION = (
    "The linkage method passed to scipy.cluster.hierarchy.linkage. ward is "
    "the only value the fit accepts, case-insensitively. Any other value "
    "raises."
)


class WardModelBundle(TypedDict):
    linkage_matrix: np.ndarray
    cluster_ids: list[int]
    assign_nn: NearestNeighbors
    feature_columns: list[str]
    version: str


class WardModelArtifact(JoblibArtifact[WardModelBundle]):
    """Ward linkage model (model.joblib)."""

    feature: str = "global-ward"
    pattern: str = "model.joblib"
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


@final
@register_feature
class GlobalWardClustering:
    """
    Ward hierarchical clustering on templates with per-sequence 1-NN assignment.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.global_ward.GlobalWardClustering.Params`.
    """

    category = "global"
    name = "global-ward"
    version = "0.3"
    parallelizable = False
    scope_dependent = False
    accepts_overlap = False  # computes within a frame, so gains nothing
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "as-input"
    ModelArtifact = WardModelArtifact

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "any"

    class Params(GlobalModelParams[WardModelArtifact]):
        templates: Annotated[TemplatesRef | None, Declared(_TEMPLATES_DESCRIPTION)] = (
            None
        )
        model: Annotated[WardModelArtifact | None, Declared(_MODEL_DESCRIPTION)] = (
            Field(default_factory=WardModelArtifact)
        )
        n_clusters: Annotated[int, Declared(_N_CLUSTERS_DESCRIPTION)] = Field(
            default=20, ge=1
        )
        method: Annotated[
            str, Field(examples=["ward"]), Declared(_METHOD_DESCRIPTION)
        ] = "ward"
        pair_filter: Annotated[NNResult | None, Declared(PAIR_FILTER_DESCRIPTION)] = (
            None
        )

    def __init__(
        self,
        inputs: GlobalWardClustering.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

        self._linkage: np.ndarray | None = None
        self._cluster_ids: np.ndarray | None = None
        self._assign_nn: NearestNeighbors | None = None
        self._feature_columns: list[str] | None = None
        self._templates: np.ndarray | None = None

    # --- Feature protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._linkage = None
        self._cluster_ids = None
        self._assign_nn = None
        self._feature_columns = None
        self._templates = None

        # Branch 1: cached model in run_root
        cached_path = run_root / "model.joblib"
        if cached_path.exists():
            bundle: WardModelBundle = WardModelArtifact().from_path(cached_path)
            self._linkage = bundle["linkage_matrix"]
            self._cluster_ids = np.asarray(bundle["cluster_ids"], dtype=np.int32)
            self._assign_nn = bundle["assign_nn"]
            self._feature_columns = bundle["feature_columns"]
            return True

        # Branch 2: pre-fitted model from artifact_paths
        if self.params.model is not None and "model" in artifact_paths:
            bundle = self.params.model.from_path(artifact_paths["model"])
            self._linkage = bundle["linkage_matrix"]
            self._cluster_ids = np.asarray(bundle["cluster_ids"], dtype=np.int32)
            self._assign_nn = bundle["assign_nn"]
            self._feature_columns = bundle["feature_columns"]
            return True

        # Branch 3: templates from artifact_paths
        if self.params.templates is not None and "templates" in artifact_paths:
            df = self.params.templates.from_path(artifact_paths["templates"])
            self._feature_columns = list(df.columns)
            self._templates = df.to_numpy(dtype=np.float64)
            return False

        return False

    def fit(self, inputs: InputStream) -> None:
        if self._templates is None:
            msg = "[global-ward] No templates loaded. Check load_state."
            raise RuntimeError(msg)

        templates = self._templates.astype(np.float64, copy=False)

        if templates.ndim != 2 or templates.shape[0] < 2:
            msg = (
                f"[global-ward] Need a 2D matrix with >=2 samples; "
                f"got shape={templates.shape}"
            )
            raise ValueError(msg)

        method = self.params.method.lower()
        if method != "ward":
            msg = f"[global-ward] Only 'ward' is supported here, got '{method}'."
            raise ValueError(msg)

        self._linkage = _sch_linkage(templates, method=method)

        labels = fcluster(self._linkage, self.params.n_clusters, criterion="maxclust")
        unique_ids = np.unique(labels)
        centroids = np.vstack(
            [templates[labels == cid].mean(axis=0) for cid in unique_ids]
        )
        self._cluster_ids = unique_ids.astype(np.int32)
        self._assign_nn = NearestNeighbors(n_neighbors=1).fit(centroids)

        # Free templates memory
        self._templates = None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if (
            self._assign_nn is None
            or self._cluster_ids is None
            or self._feature_columns is None
        ):
            msg = "GlobalWardClustering not fitted yet."
            raise RuntimeError(msg)

        ensure_columns(df, [C.frame_col] + self._feature_columns)

        arr = df[self._feature_columns].to_numpy(dtype=np.float32, copy=False)

        mask = np.isfinite(arr).all(axis=1)
        idx = np.flatnonzero(mask).tolist()
        valid = df.iloc[idx].reset_index(drop=True)
        meta_cols = sorted(set(valid.columns) - set(self._feature_columns))
        out = valid[meta_cols].copy()

        _, indices = self._assign_nn.kneighbors(arr[mask])
        out["cluster"] = self._cluster_ids[indices.ravel()]
        return out

    def save_state(self, run_root: Path) -> None:
        if (
            self._assign_nn is None
            or self._cluster_ids is None
            or self._linkage is None
        ):
            return
        run_root.mkdir(parents=True, exist_ok=True)

        bundle: WardModelBundle = {
            "linkage_matrix": self._linkage,
            "cluster_ids": self._cluster_ids.tolist(),
            "assign_nn": self._assign_nn,
            "feature_columns": self._feature_columns or [],
            "version": self.version,
        }
        joblib.dump(bundle, run_root / "model.joblib")
