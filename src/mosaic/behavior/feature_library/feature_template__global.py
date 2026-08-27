"""
Template for a global feature (clustering, embedding, dimensionality reduction).

Copy this file, rename the class and `name`, and fill in your logic.

Protocol (7 attributes + 4 methods):
  - name, version, parallelizable, scope_dependent, accepts_overlap, emits,
    consumed_roots
  - load_state(run_root, artifact_paths, dependency_lookups) -> bool
  - fit(inputs: factory returning iterator of (entry_key, DataFrame)) -> None
  - save_state(run_root) -> None
  - apply(df: DataFrame) -> DataFrame

Global features are stateful: fit() iterates over all sequences to build a
model, save_state() persists it, and load_state() restores it to skip re-fitting.
apply() then maps per-sequence data using the fitted model.

Set scope_dependent = False unless outputs change depending on which sequences
are in scope (most global features are scope-independent once fitted).

See GlobalTSNE and GlobalWardClustering for real examples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, final

import joblib
import numpy as np
import pandas as pd

from mosaic.core.pipeline.types import (
    DependencyLookup,
    EmitsLevel,
    InputRequire,
    Inputs,
    InputStream,
    Result,
)
from mosaic.core.params import Declared, Params

from .helpers import feature_columns, meta_columns

# from .registry import register_feature  # <-- uncomment when ready

_RANDOM_STATE_DESCRIPTION = "Random seed for the fit."


@final
# @register_feature   # <-- uncomment when your feature is ready
class MyGlobalFeature:
    """
    Template for a global feature.

    Global features load data from prior feature outputs (via Result-based
    inputs), run a cross-sequence algorithm in fit(), and persist the model
    via save_state(). The apply() method maps per-sequence data using the
    fitted model.

    Typical workflow:
      1. load_state() checks for a cached model on disk
      2. fit() iterates over all sequences, accumulates data, runs algorithm
      3. save_state() persists the model to run_root
      4. apply() maps per-sequence data using the fitted model
    """

    category = (
        "global"  # diagram color: "per-frame" / "summary" / "tag" / "global" / custom
    )
    name = "my-global-feature"
    version = "0.1"
    parallelizable = False
    scope_dependent = False
    accepts_overlap = False  # the template declares the conservative answer
    # "individual" / "pair" / "unidentified" / "as-input". A global fitter
    # augments the frame it was handed rather than re-keying it, so it passes
    # its input's level through. Change this if yours does re-key.
    emits: EmitsLevel = "as-input"
    consumed_roots: tuple[str, ...] = ()

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "any"

    class Params(Params):
        """Global feature template parameters."""

        random_state: Annotated[int, Declared(_RANDOM_STATE_DESCRIPTION)] = 42

    def __init__(
        self,
        inputs: MyGlobalFeature.Inputs,
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._model: np.ndarray | None = None

    # --- State ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        # Check for cached model from a previous run
        cached_path = run_root / "model.joblib"
        if cached_path.exists():
            bundle = joblib.load(cached_path)
            self._model = bundle["model"]
            return True
        return False

    def fit(self, inputs: InputStream) -> None:
        # Iterate over all sequences to accumulate data
        all_data: list[np.ndarray] = []
        for _entry_key, df in inputs():
            if df.empty:
                continue
            # Select the measurements. Every identity column here is a number
            # too, so a bare numeric select fits the model on `frame` and the ids.
            cols = feature_columns(df)
            if not cols:
                continue
            all_data.append(df[cols].to_numpy(dtype=np.float32))

        if not all_data:
            msg = f"{self.name}: no usable inputs found."
            raise RuntimeError(msg)

        stacked = np.concatenate(all_data, axis=0)

        # --- YOUR GLOBAL ALGORITHM HERE ---
        # EXAMPLE: project to first 2 dimensions
        if stacked.shape[1] >= 2:
            self._model = stacked[:, :2]
        else:
            self._model = stacked

    def save_state(self, run_root: Path) -> None:
        if self._model is None:
            return
        run_root.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "params": self.params.model_dump(),
            },
            run_root / "model.joblib",
        )

    # --- Apply ---

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            msg = f"{self.name}: not fitted. Call fit() or load_state() first."
            raise RuntimeError(msg)

        # --- YOUR PER-SEQUENCE MAPPING HERE ---
        # EXAMPLE: project to first 2 dimensions using fitted model
        cols = feature_columns(df)
        width = min(len(cols), 2)
        projected = df[cols[:width]].to_numpy(dtype=np.float32)

        out_cols = [f"feat_{i}" for i in range(projected.shape[1])]
        out = pd.DataFrame(projected, columns=out_cols, index=df.index)

        # Carry the identity through. Splitting on numeric-vs-not instead would
        # drop the numeric identity columns from both halves at once -- `id1`,
        # `id2`, `perspective` and `frame` are numbers, and they are not data.
        for col in meta_columns(df):
            out[col] = df[col].values

        return out
