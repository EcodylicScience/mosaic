"""
Template for a global feature (clustering, embedding, dimensionality reduction).

Copy this file, rename the class and `name`, and fill in your logic.

Current patterns this template follows (as of 2026-02):
  - skip_transform_phase = True  (all work in fit/save_model, no per-seq transform)
  - loads_own_data() = True      (skips run_feature pre-loading)
  - Uses StreamingFeatureHelper for manifest building + data loading
  - set_run_root() for streaming writes during fit
  - get_additional_index_rows() to register artifacts in the feature index
  - Shared helpers from helpers.py for scope parsing, sequence identity, index rows
  - output_type = "global"

See GlobalTSNE and WardAssignClustering for real examples.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Iterable
import gc
import sys

import numpy as np
import pandas as pd
import joblib

from mosaic.core.dataset import (
    register_feature,
    _resolve_inputs,
)
from .helpers import (
    StreamingFeatureHelper,
    _parse_scope_filter, _build_sequence_lookup, _resolve_sequence_identity,
    _build_index_row,
)
from mosaic.core.helpers import to_safe_name


# @register_feature   # <-- uncomment when your feature is ready
class MyGlobalFeature:
    """
    Template for a global feature.

    Global features load data from prior feature outputs (via an inputset or
    explicit inputs), run a cross-sequence algorithm in fit(), and write
    artifacts directly to disk.  The transform phase is skipped entirely.

    Typical workflow:
      1. fit() loads matrices via StreamingFeatureHelper
      2. Runs global computation (clustering, embedding, etc.)
      3. Writes per-sequence outputs + global artifacts via _run_root
      4. save_model() persists any model state (linkage, embedding, etc.)
    """

    name = "my-global-feature"
    version = "0.1"
    parallelizable = False
    output_type = "global"
    skip_transform_phase = True     # all work in fit/save_model

    def __init__(self, params: Optional[dict] = None):
        defaults = dict(
            inputs=[],
            inputset=None,
            random_state=42,
            # Add algorithm-specific params here
        )
        self.params = {**defaults, **(params or {})}
        self._inputs_overridden = bool(params and "inputs" in params)

        self._ds = None
        self._run_root: Optional[Path] = None
        self._scope_filter: Optional[dict] = None
        self._allowed_safe_sequences: Optional[set[str]] = None
        self._pair_map: dict[str, tuple[str, str]] = {}
        self._sequence_lookup_cache: Optional[dict[str, tuple[str, str]]] = None
        self._additional_index_rows: list[dict] = []
        self._artifacts: Dict[str, Any] = {}

    # ----------------------- Dataset hooks -----------------------

    def bind_dataset(self, ds):
        self._ds = ds

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        self._scope_filter = scope or {}
        self._allowed_safe_sequences, self._pair_map = _parse_scope_filter(scope)
        self._sequence_lookup_cache = None

    def set_run_root(self, run_root: Path) -> None:
        """Set by run_feature before fit(); use for streaming writes."""
        self._run_root = Path(run_root)

    def get_additional_index_rows(self) -> list[dict]:
        """Return index rows for artifacts written during fit/save_model."""
        return list(self._additional_index_rows)

    # ----------------------- Feature protocol --------------------

    def needs_fit(self) -> bool: return True
    def supports_partial_fit(self) -> bool: return False
    def loads_own_data(self) -> bool: return True
    def partial_fit(self, X: pd.DataFrame) -> None: raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=[])

    # ----------------------- Fit ---------------------------------

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        if self._ds is None:
            raise RuntimeError(f"{self.name}: dataset not bound.")
        self._additional_index_rows = []

        # 1) Resolve inputs from inputset or explicit list
        inputset_name = self.params.get("inputset")
        explicit_inputs = (
            self.params["inputs"]
            if (self._inputs_overridden or not inputset_name)
            else None
        )
        inputs, inputs_meta = _resolve_inputs(
            self._ds, explicit_inputs, inputset_name,
            explicit_override=self._inputs_overridden,
        )

        # 2) Build manifest (paths only, no data loaded yet)
        helper = StreamingFeatureHelper(self._ds, self.name)
        if inputs_meta.get("pair_filter"):
            helper.set_pair_filter(inputs_meta["pair_filter"])
        scope_filter = (
            {"safe_sequences": self._allowed_safe_sequences}
            if self._allowed_safe_sequences else None
        )
        manifest = helper.build_manifest(inputs, scope_filter=scope_filter)
        if not manifest:
            raise RuntimeError(f"{self.name}: no usable inputs found.")

        # 3) Stream through data
        keys = list(manifest.keys())
        for i, key in enumerate(keys):
            X, frames = helper.load_key_data(
                manifest[key], extract_frames=True, key=key,
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
                print(f"[{self.name}] Processed {i + 1}/{len(keys)} sequences",
                      file=sys.stderr)

        self._artifacts["inputs_meta"] = inputs_meta

    # ----------------------- Save / Load -------------------------

    def save_model(self, path: Path) -> None:
        run_root = path.parent
        run_root.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"params": self.params, "meta": self._artifacts.get("inputs_meta", {})},
            path,
        )

    def load_model(self, path: Path) -> None:
        bundle = joblib.load(path)
        self.params.update(bundle.get("params", {}))

    # ----------------------- Internal helpers --------------------
    # These use shared functions from helpers.py — the same ones used
    # by GlobalTSNE, WardAssign, GlobalKMeans, etc.

    def _get_sequence_lookup(self) -> dict[str, tuple[str, str]]:
        """Cached wrapper around shared _build_sequence_lookup."""
        if self._sequence_lookup_cache is not None:
            return self._sequence_lookup_cache
        self._sequence_lookup_cache = _build_sequence_lookup(
            self._ds, self._allowed_safe_sequences
        )
        return self._sequence_lookup_cache

    def _process_sequence(
        self, key: str, X: np.ndarray, frames: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
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
        group, sequence = _resolve_sequence_identity(
            safe_seq, self._pair_map, self._get_sequence_lookup()
        )
        safe_group = to_safe_name(group) if group else ""
        out_name = f"{safe_group + '__' if safe_group else ''}{safe_seq}.parquet"
        out_path = self._run_root / out_name

        cols = [f"feat_{i}" for i in range(data.shape[1])]
        df_out = pd.DataFrame(data, columns=cols)
        df_out["sequence"] = sequence
        df_out["group"] = group
        df_out.to_parquet(out_path, index=False)

        self._additional_index_rows.append(
            _build_index_row(safe_seq, group, sequence, out_path, int(len(df_out)))
        )
