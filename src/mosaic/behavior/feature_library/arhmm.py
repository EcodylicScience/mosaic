"""AR-HMM global feature.

Fits an autoregressive Hidden Markov Model on arbitrary upstream feature
inputs and produces per-frame syllable (state) labels.  This is a native
mosaic implementation — no KPMS or JAX dependency.

The feature accepts any combination of upstream ``Result`` inputs.  Mosaic's
manifest system merges them via inner join on alignment columns, so the
feature receives a single merged DataFrame whose numeric columns are the
union of all input features.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TypedDict, final

import joblib
import numpy as np
from pydantic import Field
from sklearn.decomposition import PCA

from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.types import (
    COLUMNS as C,
    DependencyLookup,
    InputRequire,
    Inputs,
    InputStream,
    JoblibArtifact,
    JoblibLoadSpec,
    Params,
    Result,
)

from .arhmm_model import ARHMM
from .helpers import feature_columns
from .registry import register_feature

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)


# --- Model artifact ---


class ArHmmModelBundle(TypedDict):
    model: ARHMM
    pca: PCA | None
    scaler_mean: np.ndarray | None
    scaler_std: np.ndarray | None
    feature_columns: list[str]
    downsample_rate: int | None
    version: str


class ArHmmModelArtifact(JoblibArtifact[ArHmmModelBundle]):
    """Fitted AR-HMM model bundle (arhmm_model.joblib)."""

    feature: str = "arhmm"
    pattern: str = "arhmm_model.joblib"
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


# --- Feature class ---


@final
@register_feature
class ArHmmFeature:
    """AR-HMM behavioral syllable discovery as a pipeline feature.

    Fits an autoregressive Hidden Markov Model across all input sequences
    and assigns per-frame syllable labels via Viterbi decoding.

    Params:
        model: Pre-fitted ArHmmModelArtifact to load (skip fit).
            Default: None (fit from scratch).
        pca_dim: Number of PCA components for dimensionality reduction
            before fitting.  None skips PCA.  Default: None.
        n_states: Maximum number of HMM states (pruned after fit).
            Default: 50.
        n_lags: AR order (number of lagged frames as regressors).
            Default: 1.
        sticky_weight: Extra pseudo-count on the diagonal of the
            transition matrix (encourages state persistence).
            Default: 100.0.
        n_iter: Maximum EM iterations per restart.  Default: 200.
        tol: Convergence tolerance on relative LL change.
            Default: 1e-4.
        n_restarts: Number of random restarts (best LL kept).
            Default: 1.
        standardize: If True, z-score features before fitting.
            Default: True.
        downsample_rate: Temporal downsampling factor.  None disables.
            Default: None.
        prune_threshold: Drop states with posterior mass below this
            fraction.  Default: 0.01.
        random_state: Random seed.  Default: 42.
    """

    name = "arhmm"
    version = "0.1"
    parallelizable = False
    scope_dependent = True

    ModelArtifact = ArHmmModelArtifact

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "nonempty"

    class Params(Params):
        model: ArHmmModelArtifact | None = None
        pca_dim: int | None = Field(default=None, ge=1)
        n_states: int = Field(default=50, ge=2)
        n_lags: int = Field(default=1, ge=1)
        sticky_weight: float = Field(default=100.0, ge=0)
        n_iter: int = Field(default=200, ge=1)
        tol: float = Field(default=1e-4, gt=0)
        n_restarts: int = Field(default=1, ge=1)
        standardize: bool = True
        downsample_rate: int | None = Field(default=None, ge=1)
        prune_threshold: float = Field(default=0.01, ge=0, le=1)
        random_state: int = 42

    def __init__(
        self,
        inputs: ArHmmFeature.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

        self._model: ARHMM | None = None
        self._pca: PCA | None = None
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None
        self._feature_columns: list[str] | None = None

    # --- Feature protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._model = None
        self._pca = None
        self._scaler_mean = None
        self._scaler_std = None
        self._feature_columns = None

        # Branch 1: cached model in run_root
        cached_path = run_root / "arhmm_model.joblib"
        if cached_path.exists():
            self._load_bundle(ArHmmModelArtifact().from_path(cached_path))
            return True

        # Branch 2: pre-fitted model from artifact_paths
        if self.params.model is not None and "model" in artifact_paths:
            self._load_bundle(self.params.model.from_path(artifact_paths["model"]))
            return True

        return False

    def fit(self, inputs: InputStream) -> None:
        sequences: list[np.ndarray] = []
        p = self.params

        for entry_key, df in inputs():
            if self._feature_columns is None:
                self._feature_columns = feature_columns(df)
                if not self._feature_columns:
                    msg = "[arhmm] No numeric feature columns found in input."
                    raise RuntimeError(msg)

            cols = self._feature_columns
            if "id" in df.columns:
                id_col = df["id"]
                for ind_id, sub in df.groupby(id_col, sort=False):
                    sub = sub.sort_values("frame").reset_index(drop=True)
                    arr = self._df_to_array(sub, cols)
                    if arr is not None:
                        sequences.append(arr)
            else:
                arr = self._df_to_array(df, cols)
                if arr is not None:
                    sequences.append(arr)

        if not sequences:
            msg = "[arhmm] No valid sequences found for fitting."
            raise RuntimeError(msg)

        D = sequences[0].shape[1]
        print(
            f"[arhmm] Fitting on {len(sequences)} sequences, "
            f"{D} features, n_states={p.n_states}, n_lags={p.n_lags}",
            file=sys.stderr,
        )

        # Standardize
        if p.standardize:
            all_data = np.vstack(sequences)
            self._scaler_mean = all_data.mean(axis=0)
            self._scaler_std = all_data.std(axis=0)
            self._scaler_std[self._scaler_std < 1e-10] = 1.0
            sequences = [
                (s - self._scaler_mean) / self._scaler_std for s in sequences
            ]

        # PCA
        if p.pca_dim is not None and p.pca_dim < D:
            all_data = np.vstack(sequences)
            self._pca = PCA(n_components=p.pca_dim, random_state=p.random_state)
            self._pca.fit(all_data)
            sequences = [self._pca.transform(s) for s in sequences]
            print(
                f"[arhmm] PCA: {D} → {p.pca_dim} components "
                f"({self._pca.explained_variance_ratio_.sum():.1%} variance)",
                file=sys.stderr,
            )

        # Fit AR-HMM
        self._model = ARHMM(
            n_states=p.n_states,
            n_lags=p.n_lags,
            sticky_weight=p.sticky_weight,
            n_iter=p.n_iter,
            tol=p.tol,
            n_restarts=p.n_restarts,
            random_state=p.random_state,
        )
        self._model.fit(sequences)

        # Prune unused states
        if p.prune_threshold > 0:
            self._model.prune_states(sequences, threshold=p.prune_threshold)

        n_states = self._model.n_states
        print(f"[arhmm] Fit complete: {n_states} active states.", file=sys.stderr)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd

        if self._model is None or self._feature_columns is None:
            msg = "AR-HMM not fitted — call fit() or load_state() first."
            raise RuntimeError(msg)

        cols = self._feature_columns
        results: list[pd.DataFrame] = []

        if "id" in df.columns:
            id_col = df["id"]
            for ind_id, sub in df.groupby(id_col, sort=False):
                sub = sub.sort_values("frame").reset_index(drop=True)
                syllables = self._apply_one(sub, cols)
                result = pd.DataFrame(
                    {"frame": sub["frame"].values, "syllable": syllables}
                )
                result["id"] = ind_id
                results.append(result)
        else:
            syllables = self._apply_one(df, cols)
            results.append(
                pd.DataFrame(
                    {"frame": df["frame"].values, "syllable": syllables}
                )
            )

        return pd.concat(results, ignore_index=True)

    def save_state(self, run_root: Path) -> None:
        if self._model is None or self._feature_columns is None:
            return
        run_root.mkdir(parents=True, exist_ok=True)

        bundle: ArHmmModelBundle = {
            "model": self._model,
            "pca": self._pca,
            "scaler_mean": self._scaler_mean,
            "scaler_std": self._scaler_std,
            "feature_columns": self._feature_columns,
            "downsample_rate": self.params.downsample_rate,
            "version": self.version,
        }
        joblib.dump(bundle, run_root / "arhmm_model.joblib")

    # --- Helpers ---

    def _load_bundle(self, bundle: ArHmmModelBundle) -> None:
        self._model = bundle["model"]
        self._pca = bundle["pca"]
        self._scaler_mean = bundle["scaler_mean"]
        self._scaler_std = bundle["scaler_std"]
        self._feature_columns = bundle["feature_columns"]

    def _df_to_array(
        self, df: pd.DataFrame, cols: list[str]
    ) -> np.ndarray | None:
        """Convert a DataFrame to a numpy array for AR-HMM, with
        downsampling and NaN handling."""
        arr = df[cols].to_numpy(dtype=np.float64)

        # Replace non-finite values with 0
        bad_mask = ~np.isfinite(arr)
        if bad_mask.any():
            n_bad = int(bad_mask.sum())
            pct = 100 * n_bad / arr.size
            print(
                f"[arhmm] Replaced {n_bad} non-finite values ({pct:.1f}%) with 0",
                file=sys.stderr,
            )
            arr = np.where(bad_mask, 0.0, arr)

        # Downsample
        if self.params.downsample_rate is not None and self.params.downsample_rate > 1:
            arr = arr[:: self.params.downsample_rate]

        # Skip sequences too short for AR
        if arr.shape[0] <= self.params.n_lags:
            return None

        return arr

    def _apply_one(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        """Apply the fitted model to a single sub-sequence."""
        assert self._model is not None
        original_len = len(df)
        arr = df[cols].to_numpy(dtype=np.float64)
        arr = np.where(~np.isfinite(arr), 0.0, arr)

        ds_rate = self.params.downsample_rate
        if ds_rate is not None and ds_rate > 1:
            arr = arr[::ds_rate]

        # Standardize
        if self._scaler_mean is not None and self._scaler_std is not None:
            arr = (arr - self._scaler_mean) / self._scaler_std

        # PCA
        if self._pca is not None:
            arr = self._pca.transform(arr)

        # Viterbi decode
        syllables = self._model.predict(arr)

        # Upsample if downsampled
        if ds_rate is not None and ds_rate > 1:
            syllables = np.repeat(syllables, ds_rate)[:original_len]

        return syllables
