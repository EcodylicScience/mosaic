from __future__ import annotations

from pathlib import Path
from typing import ClassVar, TypedDict, final

import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.preprocessing import StandardScaler

from mosaic.core.pipeline.types import (
    DependencyLookup,
    GlobalModelParams,
    InputRequire,
    Inputs,
    InputStream,
    JoblibArtifact,
    JoblibLoadSpec,
    ParquetArtifact,
    ParquetLoadSpec,
    Result,
)

from .helpers import ensure_columns
from .registry import register_feature


class ScalerModelBundle(TypedDict):
    scaler: StandardScaler
    feature_columns: list[str]
    version: str


class ScalerModelArtifact(JoblibArtifact[ScalerModelBundle]):
    """Fitted scaler model bundle (scaler.joblib)."""

    feature: str = "global-scaler"
    pattern: str = "scaler.joblib"
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


class ScaledTemplatesArtifact(ParquetArtifact):
    """Scaled template vectors (scaled_templates.parquet)."""

    feature: str = "global-scaler"
    pattern: str = "scaled_templates.parquet"
    load: ParquetLoadSpec = Field(default_factory=ParquetLoadSpec)


@final
@register_feature
class GlobalScaler:
    """Fit a StandardScaler on templates and scale per-sequence data.

    Consumes a templates artifact (from ExtractTemplates or any feature
    producing templates.parquet). Produces a scaler model bundle and
    scaled templates.

    Params:
        templates: Templates artifact to fit the scaler on (inherited
            from GlobalModelParams).
        model: Pre-fitted ScalerModelArtifact to load (skip fit).
            Default: ScalerModelArtifact().
    """

    name = "global-scaler"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    ScalerModelArtifact = ScalerModelArtifact
    ScaledTemplatesArtifact = ScaledTemplatesArtifact

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "any"

    class Params(GlobalModelParams[ScalerModelArtifact]):
        """GlobalScaler parameters.

        Attributes:
            templates: Templates artifact to fit scaler on.
            model: Pre-fitted scaler model artifact (skip fit).
        """

        model: ScalerModelArtifact | None = Field(default_factory=ScalerModelArtifact)

    def __init__(
        self,
        inputs: GlobalScaler.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

        self._feature_columns: list[str] | None = None
        self._scaler: StandardScaler | None = None
        self._scaled_templates: np.ndarray | None = None

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._feature_columns = None
        self._scaler = None
        self._scaled_templates = None

        # Check for cached model
        cached_path = run_root / "scaler.joblib"
        if cached_path.exists():
            bundle: ScalerModelBundle = ScalerModelArtifact().from_path(cached_path)
            self._scaler = bundle["scaler"]
            self._feature_columns = bundle["feature_columns"]

            scaled_path = run_root / "scaled_templates.parquet"
            if scaled_path.exists():
                df = pd.read_parquet(scaled_path)
                self._scaled_templates = df.to_numpy(dtype=np.float64)

            return True

        # Load pre-fitted model from artifact_paths
        if self.params.model is not None and "model" in artifact_paths:
            bundle = self.params.model.from_path(artifact_paths["model"])
            self._scaler = bundle["scaler"]
            self._feature_columns = bundle["feature_columns"]
            return True

        # Load templates from artifact_paths
        if self.params.templates is not None and "templates" in artifact_paths:
            df = self.params.templates.from_path(artifact_paths["templates"])
            self._feature_columns = list(df.columns)
            self._scaled_templates = df.to_numpy(dtype=np.float64)
            # _scaled_templates temporarily holds raw templates until fit()
            return False

        return False

    def fit(self, inputs: InputStream) -> None:
        if self._scaled_templates is None:
            msg = "[global-scaler] No templates loaded. Check load_state."
            raise RuntimeError(msg)

        # _scaled_templates holds raw templates at this point
        raw_templates = self._scaled_templates
        scaler = StandardScaler()
        scaler.fit(raw_templates)
        self._scaler = scaler
        self._scaled_templates = scaler.transform(raw_templates)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._scaler is None or self._feature_columns is None:
            msg = "[global-scaler] Not fitted. Call fit() or load_state() first."
            raise RuntimeError(msg)

        ensure_columns(df, self._feature_columns)
        result = df.copy()
        feature_data = df[self._feature_columns].to_numpy(dtype=np.float64)
        scaled = self._scaler.transform(feature_data)
        result[self._feature_columns] = scaled
        return result

    def save_state(self, run_root: Path) -> None:
        if self._scaler is None or self._feature_columns is None:
            return
        run_root.mkdir(parents=True, exist_ok=True)

        import joblib

        bundle: ScalerModelBundle = {
            "scaler": self._scaler,
            "feature_columns": self._feature_columns,
            "version": self.version,
        }
        joblib.dump(bundle, run_root / "scaler.joblib")

        if self._scaled_templates is not None:
            df = pd.DataFrame(self._scaled_templates, columns=self._feature_columns)
            df.to_parquet(run_root / "scaled_templates.parquet", index=False)
