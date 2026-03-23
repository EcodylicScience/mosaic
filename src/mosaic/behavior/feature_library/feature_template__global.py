"""
Template for a global feature (clustering, embedding, dimensionality reduction).

Copy this file, rename the class and `name`, and fill in your logic.

Current patterns this template follows (as of 2026-03):
  - skip_transform_phase = True  (all work in fit/save_model, no per-seq transform)
  - Uses Pydantic Params for typed, validated parameters
  - Uses StreamingFeatureHelper with build_manifest_from_results for data loading
  - set_run_root() for streaming writes during fit
  - get_additional_index_rows() to register artifacts in the feature index
  - Shared helpers from helpers.py for scope parsing, sequence identity, index rows
  - Data inputs via constructor `inputs` (Result-based), not Params
  - output_type = "global"

See GlobalTSNE and WardAssignClustering for real examples.
"""

from __future__ import annotations

import gc
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import final

import joblib
import numpy as np
import pandas as pd

# from .spec import register_feature  # <-- uncomment when ready
from mosaic.core.helpers import entry_key
from mosaic.core.pipeline._utils import Scope

from .helpers import (
    PartialIndexRow,
    StreamingFeatureHelper,
    _build_index_row,
    _resolve_sequence_identity,
)
from .spec import Inputs, NNResult, OutputType, Params, Result


@final
# @register_feature   # <-- uncomment when your feature is ready
class MyGlobalFeature:
    """
    Template for a global feature.

    Global features load data from prior feature outputs (via Result-based
    inputs), run a cross-sequence algorithm in fit(), and write artifacts
    directly to disk.  The transform phase is skipped entirely.

    Typical workflow:
      1. fit() loads matrices via StreamingFeatureHelper.build_manifest_from_results()
      2. Runs global computation (clustering, embedding, etc.)
      3. Writes per-sequence outputs + global artifacts via _run_root
      4. save_model() persists any model state (linkage, embedding, etc.)
    """

    name = "my-global-feature"
    version = "0.1"
    parallelizable = False
    output_type: OutputType = "global"
    skip_transform_phase = True  # all work in fit/save_model

    class Inputs(Inputs[Result]):
        pass

    class Params(Params):
        """Global feature template parameters.

        Attributes:
            random_state: Random seed. Default 42.
            pair_filter: Nearest-neighbor pair filter. Default None.
        """

        random_state: int = 42
        pair_filter: NNResult | None = None

    def __init__(
        self,
        inputs: MyGlobalFeature.Inputs,
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True

        self._ds = None
        self._run_root: Path | None = None
        self._scope: Scope = Scope()
        self._additional_index_rows: list[PartialIndexRow] = []
        self._artifacts: dict[str, object] = {}

    # ----------------------- Dataset hooks -----------------------

    def bind_dataset(self, ds):
        self._ds = ds

    def set_scope(self, scope: Scope) -> None:
        self._scope = scope

    def set_run_root(self, run_root: Path) -> None:
        """Set by run_feature before fit(); use for streaming writes."""
        self._run_root = Path(run_root)

    def get_additional_index_rows(self) -> list[PartialIndexRow]:
        """Return index rows for artifacts written during fit/save_model."""
        return list(self._additional_index_rows)

    # ----------------------- Feature protocol --------------------

    def needs_fit(self) -> bool:
        return True

    def supports_partial_fit(self) -> bool:
        return False

    def partial_fit(self, X: pd.DataFrame) -> None:
        raise NotImplementedError

    def finalize_fit(self) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=[])

    # ----------------------- Fit ---------------------------------

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        if self._ds is None:
            raise RuntimeError(f"{self.name}: dataset not bound.")
        self._additional_index_rows = []

        # Build manifest from Result-based inputs
        helper = StreamingFeatureHelper(self._ds, self.name)
        if self.params.pair_filter:
            helper.set_pair_filter(self.params.pair_filter)
        manifest = helper.build_manifest_from_results(
            self.inputs.feature_inputs, scope=self._scope
        )
        if not manifest:
            raise RuntimeError(f"{self.name}: no usable inputs found.")

        # Stream through data
        keys = list(manifest.keys())
        for i, key in enumerate(keys):
            X, frames = helper.load_key_data(
                manifest[key],
                extract_frames=True,
                key=key,
            )
            if X is None or X.shape[0] == 0:
                continue

            # --- YOUR GLOBAL LOGIC HERE ---
            result = self._process_sequence(key, X, frames)

            # Write per-sequence output
            if result is not None and self._run_root is not None:
                self._write_sequence_output(key, result)

            del X, frames, result
            gc.collect()

            if (i + 1) % 10 == 0 or i == len(keys) - 1:
                print(
                    f"[{self.name}] Processed {i + 1}/{len(keys)} sequences",
                    file=sys.stderr,
                )

    # ----------------------- Save / Load -------------------------

    def save_model(self, path: Path) -> None:
        run_root = path.parent
        run_root.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "params": self.params.model_dump(),
            },
            path,
        )

    def load_model(self, path: Path) -> None:
        bundle = joblib.load(path)
        saved = bundle.get("params", {})
        if isinstance(saved, dict):
            self.params = self.Params.model_validate(saved)

    # ----------------------- Internal helpers --------------------
    # These use shared functions from helpers.py -- the same ones used
    # by GlobalTSNE, WardAssign, GlobalKMeans, etc.

    def _process_sequence(
        self, key: str, X: np.ndarray, frames: np.ndarray | None
    ) -> np.ndarray | None:
        """
        Pure computation for one sequence.  X is (T, D) float32.

        Return an array to write, or None to skip this sequence.
        Replace this with your own algorithm.
        """
        # EXAMPLE: project to first 2 dims
        if X.shape[1] >= 2:
            return X[:, :2].astype(np.float32, copy=False)
        return None

    def _write_sequence_output(self, safe_seq: str, data: np.ndarray) -> None:
        """Write per-sequence output and register an index row."""
        if self._run_root is None:
            return
        group, sequence = _resolve_sequence_identity(safe_seq, self._scope.entry_map)
        out_name = f"{entry_key(group, sequence)}.parquet"
        out_path = self._run_root / out_name

        cols = [f"feat_{i}" for i in range(data.shape[1])]
        df_out = pd.DataFrame(data, columns=cols)
        df_out["sequence"] = sequence
        df_out["group"] = group
        df_out.to_parquet(out_path, index=False)

        self._additional_index_rows.append(
            _build_index_row(
                group,
                sequence,
                out_path,
                int(len(df_out)),
            )
        )
