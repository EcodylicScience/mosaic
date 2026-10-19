"""AR-HMM global feature.

Fits an autoregressive Hidden Markov Model on arbitrary upstream feature
inputs and produces per-frame syllable (state) labels.  This is a native
mosaic implementation — no KPMS or JAX dependency.

The model follows the AR-HMM behavioral syllable approach of MoSeq: Wiltschko
et al. (2015), Neuron 88(6):1121-1135, doi:10.1016/j.neuron.2015.11.031. See
NOTICE.

The feature accepts any combination of upstream ``Result`` inputs.  Mosaic's
manifest system merges them via inner join on alignment columns, so the
feature receives a single merged DataFrame whose numeric columns are the
union of all input features.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, TypedDict, final

import joblib
import numpy as np
from pydantic import Field
from sklearn.decomposition import PCA

from mosaic.core.pipeline.column_contract import (
    Column,
    Payload,
    Provides,
    Requirement,
    kept,
)
from mosaic.core.pipeline.types import (
    EmitsLevel,
    DependencyLookup,
    InputRequire,
    Inputs,
    InputStream,
    JoblibArtifact,
    JoblibLoadSpec,
    Result,
)
from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
    Params,
)

from .arhmm_model import ARHMM, fit_numba, predict_numba
from .helpers import (
    chosen_payload_columns,
    iter_series,
    label_series,
    refuse_repeated_series_rows,
)
from .registry import register_feature
from .types import PAYLOAD_COLUMNS_DESCRIPTION, PayloadColumnNames

if TYPE_CHECKING:
    import pandas as pd
    from pandas._typing import Scalar

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


_MODEL_DESCRIPTION = (
    "A pre-fitted model artifact to load, skipping the fit. Unset, the "
    "model is fit from scratch."
)

_PCA_DIM_DESCRIPTION = (
    "How many components PCA reduces the feature dimensionality to "
    "before fitting. Unset, no PCA is applied. Ignored when it is not "
    "smaller than the number of input feature columns."
)

_N_STATES_DESCRIPTION = (
    "The maximum number of HMM states. Pruned down after fitting when "
    "prune_threshold is greater than zero."
)

_N_LAGS_DESCRIPTION = (
    "How many preceding steps enter as autoregressive regressors. A "
    "step is one row of the sequence being fit, which is a video frame "
    "unless downsample_rate has subsampled it."
)

_STICKY_WEIGHT_DESCRIPTION = (
    "The extra pseudo-count added to the diagonal of the transition "
    "matrix, encouraging state persistence across frames."
)

_N_ITER_DESCRIPTION = "The ceiling on EM passes per restart."

_TOL_DESCRIPTION = (
    "The convergence tolerance on the relative change in "
    "log-likelihood between EM iterations."
)

_N_RESTARTS_DESCRIPTION = (
    "How many random restarts run, keeping the one with the best log-likelihood."
)

_STANDARDIZE_DESCRIPTION = (
    "Z-score every feature column before fitting. When False, raw "
    "values are used directly."
)

_DOWNSAMPLE_RATE_DESCRIPTION = (
    "The factor every sequence is subsampled by before fitting and "
    "inference, upsampled back afterward. Unset, no downsampling is "
    "applied."
)

_PRUNE_THRESHOLD_DESCRIPTION = (
    "The posterior mass fraction below which a fitted state is dropped."
)

_RANDOM_STATE_DESCRIPTION = (
    "The random seed for PCA and model fitting, for reproducible restarts."
)

_BACKEND_DESCRIPTION = (
    "The compute backend for EM and Viterbi decoding. auto uses the "
    "numba-compiled path when available and falls back to numpy. numba "
    "requests the compiled path but still falls back to numpy if it "
    "raises. numpy forces the plain path. jax is reserved for a future "
    "GPU path and is not implemented yet."
)


# --- Missing values ---


def _finite_moments(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean and standard deviation of each column over its finite values.

    Two passes over the sequences, so that no copy of all of them is made. A
    column without a finite value has mean 0, and a standard deviation of
    almost 0 is 1.
    """
    n_columns = sequences[0].shape[1]
    count = np.zeros(n_columns)
    total = np.zeros(n_columns)
    for sequence in sequences:
        finite = np.isfinite(sequence)
        count += finite.sum(axis=0)
        total += np.where(finite, sequence, 0.0).sum(axis=0)
    mean = np.divide(total, count, out=np.zeros(n_columns), where=count > 0)
    squares = np.zeros(n_columns)
    for sequence in sequences:
        finite = np.isfinite(sequence)
        squares += (np.where(finite, sequence - mean, 0.0) ** 2).sum(axis=0)
    std = np.sqrt(np.divide(squares, count, out=np.zeros(n_columns), where=count > 0))
    std[std < 1e-10] = 1.0
    return mean, std


def _fill_missing(arr: np.ndarray) -> np.ndarray:
    """Return *arr* with each non-finite value set to 0."""
    return np.where(np.isfinite(arr), arr, 0.0)


# --- Feature class ---


@final
@register_feature
class ArHmmFeature:
    """AR-HMM behavioral syllable discovery as a pipeline feature.

    Fits an autoregressive Hidden Markov Model across all input sequences
    and assigns per-frame syllable labels via Viterbi decoding. A sequence is
    one individual's rows, or for a pair table the rows of one ordered pair
    and perspective, so the model never reads one animal's frame after
    another's. The output carries the columns that identify its rows.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.arhmm.ArHmmFeature.Params`.
    """

    category = "global"
    costly_fit = True  # a superseded-run cleanup keeps its runs
    name = "arhmm"
    # NOTE: bump only when the *model math* changes, not for a backend swap.
    # Backends are numerically-equivalent implementations of the same algorithm,
    # so a stable version lets numpy- and numba-computed runs share one cache
    # entry (run_id = f"{version}-{params_hash}", and `backend` is HASH_EXCLUDE).
    # 0.2: ``nn_id`` and ``attention_target_id``, track ids, are no longer
    # fitted (``BOOKKEEPING_COLS``).
    # 0.3: a pair table is read as one series per ``(id1, id2, perspective)``
    # and the output carries those columns. Under 0.2 its two perspectives were
    # one sequence of interleaved rows.
    version = "0.3"
    parallelizable = False
    scope_dependent = True
    accepts_overlap = (
        False  # decodes a whole sequence at once, so context is not a window
    )
    consumed_roots: tuple[str, ...] = ()
    # The output is keyed as the input is: by id for individuals, by the ordered
    # pair for a pair table.
    emits: EmitsLevel = "as-input"
    # Apply decodes each series' payload columns in frame order, every one or
    # those that ``columns`` names. A series is an individual or an ordered
    # pair. Its code names the frame column literally, and so does this
    # declaration. ``columns`` is not declared, because a requirement on a list
    # param is met by any one of its names. The fit refuses each named column
    # that the input lacks.
    requires: ClassVar[tuple[Requirement, ...]] = (Column("frame"), Payload())
    provides: ClassVar[Provides] = Provides(
        adds=(Column("frame"), Column("syllable")),
        conditional=kept("id", "id1", "id2", "perspective", "group", "sequence"),
    )
    ModelArtifact = ArHmmModelArtifact

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "nonempty"

    class Params(Params):
        model: Annotated[ArHmmModelArtifact | None, Declared(_MODEL_DESCRIPTION)] = None
        pca_dim: Annotated[int | None, Declared(_PCA_DIM_DESCRIPTION)] = Field(
            default=None, ge=1
        )
        n_states: Annotated[int, Declared(_N_STATES_DESCRIPTION)] = Field(
            default=50, ge=2
        )
        n_lags: Annotated[int, Declared(_N_LAGS_DESCRIPTION)] = Field(default=1, ge=1)
        sticky_weight: Annotated[float, Declared(_STICKY_WEIGHT_DESCRIPTION)] = Field(
            default=100.0, ge=0
        )
        n_iter: Annotated[int, Declared(_N_ITER_DESCRIPTION, unit="iterations")] = (
            Field(default=200, ge=1)
        )
        tol: Annotated[float, Declared(_TOL_DESCRIPTION)] = Field(default=1e-4, gt=0)
        n_restarts: Annotated[int, Declared(_N_RESTARTS_DESCRIPTION)] = Field(
            default=1, ge=1
        )
        standardize: Annotated[bool, Declared(_STANDARDIZE_DESCRIPTION)] = True
        downsample_rate: Annotated[
            int | None, Declared(_DOWNSAMPLE_RATE_DESCRIPTION)
        ] = Field(default=None, ge=1)
        prune_threshold: Annotated[float, Declared(_PRUNE_THRESHOLD_DESCRIPTION)] = (
            Field(default=0.01, ge=0, le=1)
        )
        random_state: Annotated[int, Declared(_RANDOM_STATE_DESCRIPTION)] = 42
        columns: Annotated[
            PayloadColumnNames | None, Declared(PAYLOAD_COLUMNS_DESCRIPTION)
        ] = None
        # HASH_EXCLUDE: backends are numerically equivalent, so this is dropped
        # from the run_id hash (a numpy run and a numba run share one cache entry).
        backend: Annotated[
            Literal["auto", "numpy", "numba", "jax"],
            HASH_EXCLUDE,
            Declared(_BACKEND_DESCRIPTION),
        ] = "auto"

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
                self._feature_columns = chosen_payload_columns(df, p.columns, self.name)
                if not self._feature_columns:
                    msg = "[arhmm] No numeric feature columns found in input."
                    raise RuntimeError(msg)

            cols = self._feature_columns
            refuse_repeated_series_rows(df, self.name)
            for _, rows in iter_series(df):
                arr = self._df_to_array(rows, cols)
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
        n_missing = sum(int((~np.isfinite(s)).sum()) for s in sequences)
        if n_missing:
            total = sum(s.size for s in sequences)
            print(
                f"[arhmm] {n_missing} non-finite values ({100 * n_missing / total:.1f}%) "
                "read as missing",
                file=sys.stderr,
            )

        # Standardize over the finite values, then read a missing value as 0,
        # which is the column mean once standardized.
        if p.standardize:
            self._scaler_mean, self._scaler_std = _finite_moments(sequences)
            sequences = [(s - self._scaler_mean) / self._scaler_std for s in sequences]
        sequences = [_fill_missing(s) for s in sequences]

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

        # Fit AR-HMM (backend-dispatched; numpy is the always-available fallback)
        backend = self._select_backend()
        if backend == "numba":
            try:
                self._model = fit_numba(
                    sequences,
                    n_states=p.n_states,
                    n_lags=p.n_lags,
                    sticky_weight=p.sticky_weight,
                    n_iter=p.n_iter,
                    tol=p.tol,
                    n_restarts=p.n_restarts,
                    random_state=p.random_state,
                )
            except Exception as exc:  # noqa: BLE001 - any numba failure → numpy
                log.warning(
                    "[arhmm] numba backend failed (%s); falling back to numpy.",
                    exc,
                )
                backend = "numpy"
        if backend == "numpy":
            self._model = ARHMM(
                n_states=p.n_states,
                n_lags=p.n_lags,
                sticky_weight=p.sticky_weight,
                n_iter=p.n_iter,
                tol=p.tol,
                n_restarts=p.n_restarts,
                random_state=p.random_state,
            ).fit(sequences)

        # Prune unused states
        if p.prune_threshold > 0:
            self._model.prune_states(sequences, threshold=p.prune_threshold)

        n_states = self._model.n_states
        print(f"[arhmm] Fit complete: {n_states} active states.", file=sys.stderr)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._model is None or self._feature_columns is None:
            msg = "AR-HMM not fitted — call fit() or load_state() first."
            raise RuntimeError(msg)

        cols = self._feature_columns
        refuse_repeated_series_rows(df, self.name)
        left_out: list[int] = []

        def decode(_: tuple[Scalar, ...], rows: pd.DataFrame) -> np.ndarray | None:
            states = self._apply_one(rows, cols)
            if states is None:
                left_out.append(len(rows))
            return states

        result = label_series(df, decode)
        if left_out:
            print(
                f"[arhmm] {len(left_out)} series of {sum(left_out)} rows are too "
                f"short to decode and have no states",
                file=sys.stderr,
            )
        return result

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

    def _select_backend(self) -> str:
        """Resolve the compute backend.

        ``"auto"`` prefers the numba path (a core dependency) and falls back to
        numpy if numba can't be imported.  ``"jax"`` is reserved for a future GPU
        path and raises until implemented.
        """
        b = self.params.backend
        if b == "jax":
            msg = (
                "backend='jax' (GPU) is not yet implemented for arhmm; use "
                "'numba' (default) for the accelerated CPU path, or 'numpy'."
            )
            raise NotImplementedError(msg)
        if b in ("numpy", "numba"):
            return b
        try:  # auto
            import numba  # noqa: F401
        except Exception:  # pragma: no cover - numba is a core dependency
            return "numpy"
        return "numba"

    def _load_bundle(self, bundle: ArHmmModelBundle) -> None:
        self._model = bundle["model"]
        self._pca = bundle["pca"]
        self._scaler_mean = bundle["scaler_mean"]
        self._scaler_std = bundle["scaler_std"]
        self._feature_columns = bundle["feature_columns"]

    def _df_to_array(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray | None:
        """Convert a series to a numpy array for AR-HMM, with downsampling.

        A non-finite value is NaN in the result: standardizing reads the finite
        values only, and :func:`_fill_missing` fills the rest.
        """
        arr = df[cols].to_numpy(dtype=np.float64)
        arr = np.where(np.isfinite(arr), arr, np.nan)

        # Downsample
        if self.params.downsample_rate is not None and self.params.downsample_rate > 1:
            arr = arr[:: self.params.downsample_rate]

        # Skip sequences too short for AR
        if arr.shape[0] <= self.params.n_lags:
            return None

        return arr

    def _apply_one(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray | None:
        """Apply the fitted model to a single sub-sequence.

        Returns:
            The state of each row, or ``None`` for a sequence too short for the
            model to decode, which the fit leaves out as well.
        """
        assert self._model is not None
        original_len = len(df)
        arr = df[cols].to_numpy(dtype=np.float64)

        ds_rate = self.params.downsample_rate
        if ds_rate is not None and ds_rate > 1:
            arr = arr[::ds_rate]
        if arr.shape[0] <= self.params.n_lags:
            return None

        # Standardize, then read a missing value as the column mean
        if self._scaler_mean is not None and self._scaler_std is not None:
            arr = (arr - self._scaler_mean) / self._scaler_std
        arr = _fill_missing(arr)

        # PCA
        if self._pca is not None:
            arr = self._pca.transform(arr)

        # Viterbi decode (numba when available; labels identical to numpy)
        if self._select_backend() == "numba":
            try:
                syllables = predict_numba(self._model, arr)
            except Exception as exc:  # noqa: BLE001 - any numba failure → numpy
                log.warning("[arhmm] numba predict failed (%s); using numpy.", exc)
                syllables = self._model.predict(arr)
        else:
            syllables = self._model.predict(arr)

        # Upsample if downsampled
        if ds_rate is not None and ds_rate > 1:
            syllables = np.repeat(syllables, ds_rate)[:original_len]

        return syllables
