"""
Template for a per-sequence feature.

Copy this file, rename the class and `name`, and fill in your logic.

Current patterns this template follows (as of 2026-02):
  - Uses _merge_params from helpers (not a local copy)
  - Declares output_type for feature registry
  - Handles pair-aware (id1/id2) and single-individual inputs
  - Includes id1/id2 columns in output when present
  - Uses from __future__ import annotations
  - finalize_fit() present for protocol completeness
  - Stateless by default (needs_fit=False, save_model raises)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List

import numpy as np
import pandas as pd

from mosaic.core.dataset import register_feature
from .helpers import _merge_params
from mosaic.core.helpers import to_safe_name


# @register_feature   # <-- uncomment when your feature is ready
class MyPerSequenceFeature:
    """
    Template for a per-sequence feature.

    Input:
      A DataFrame for a single (group, sequence) from either:
        * tracks (input_kind="tracks")
        * another feature (input_kind="feature")
        * an inputset (input_kind="inputset")

    Output:
      A DataFrame with one row per frame (or per frame x pair), with:
        * frame (or time)
        * group, sequence
        * id1, id2 (when pair-aware)
        * your feature columns
    """

    # Stored under dataset_root/features/<name>/
    name = "my-new-feature"
    version = "0.1"
    parallelizable = True       # safe if transform(df) only depends on df
    output_type = "per_frame"   # "per_frame" | "summary" | "global"

    _defaults = dict(
        # Column conventions
        seq_col="sequence",
        group_col="group",
        order_pref=("frame", "time"),

        # Algorithm-specific parameters:
        window_size=15,
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = _merge_params(params, self._defaults)
        self._ds = None

        # Storage overrides (set before run_feature processes the feature)
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True    # appends "__from__<input>" to run dir
        self.skip_existing_outputs = False      # set True if idempotent + expensive

    # ----------------------- Dataset hooks -----------------------

    def bind_dataset(self, ds):
        """Called by Dataset.run_feature before any fit/transform."""
        self._ds = ds

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        """Restrict which sequences are processed (used by inputset path)."""
        self._scope_filter = scope or {}

    # ----------------------- Fit protocol ------------------------

    def needs_fit(self) -> bool:
        return False

    def supports_partial_fit(self) -> bool:
        return False

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        """Global fit over all sequences. Only called if needs_fit() == True."""
        return

    def partial_fit(self, df: pd.DataFrame) -> None:
        """Streaming fit per sequence. Used when supports_partial_fit() == True."""
        return

    def finalize_fit(self) -> None:
        """Called after all fit/partial_fit calls complete."""
        return

    def save_model(self, path: Path) -> None:
        raise NotImplementedError("Stateless feature; no model to save.")

    def load_model(self, path: Path) -> None:
        raise NotImplementedError("Stateless feature; no model to load.")

    # ----------------------- Core logic --------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for a single (group, sequence).

        For pair-aware inputs the df may contain multiple (id1, id2) pairs;
        process each pair independently to avoid mixing contexts.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        p = self.params
        order_col = self._order_col(df)

        # Detect pair structure
        has_pairs = "id1" in df.columns and "id2" in df.columns
        if has_pairs:
            group_keys = ["id1", "id2"]
        else:
            group_keys = None

        # --- Select numeric input columns (exclude metadata) ---
        meta_like = {
            p["seq_col"], p["group_col"],
            "frame", "time", "id", "perspective", "fps",
            "id1", "id2",
        }
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in meta_like
        ]
        if not numeric_cols:
            return pd.DataFrame()

        # --- Process per-pair (or whole df if no pairs) ---
        if group_keys:
            blocks: List[pd.DataFrame] = []
            for group_vals, g in df.groupby(group_keys, sort=False):
                cur_id1, cur_id2 = group_vals
                block_out = self._process_block(g, numeric_cols, order_col, p)
                if block_out is not None and len(block_out):
                    block_out["id1"] = int(cur_id1)
                    block_out["id2"] = int(cur_id2)
                    blocks.append(block_out)
            return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
        else:
            return self._process_block(df, numeric_cols, order_col, p)

    def _process_block(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        order_col: str,
        params: dict,
    ) -> pd.DataFrame:
        """Process a single block (one pair or one sequence)."""
        p = params
        seq_col = p["seq_col"]
        group_col = p["group_col"]

        df = df.sort_values(order_col).reset_index(drop=True)
        X = df[numeric_cols].to_numpy(dtype=np.float32, copy=False)

        # --- YOUR LOGIC HERE ---
        features = self._compute(X, params)

        if features.ndim == 1:
            features = features[:, None]
        T_out = features.shape[0]

        # Align output to input frames
        if T_out != len(df):
            base = df.iloc[-T_out:].reset_index(drop=True)
        else:
            base = df.reset_index(drop=True)

        # Build output DataFrame
        out_cols = [f"myfeat_{i}" for i in range(features.shape[1])]
        out = pd.DataFrame(features, columns=out_cols)

        # Attach standard metadata columns
        if "frame" in base.columns:
            out["frame"] = base["frame"].to_numpy()
        if "time" in base.columns:
            out["time"] = base["time"].to_numpy()
        if seq_col in base.columns:
            out[seq_col] = base[seq_col].iloc[0]
        if group_col in base.columns:
            out[group_col] = base[group_col].iloc[0]

        return out

    # ----------------------- Internal helpers --------------------

    def _order_col(self, df: pd.DataFrame) -> str:
        for c in self.params["order_pref"]:
            if c in df.columns:
                return c
        raise ValueError("Need either 'frame' or 'time' column to order rows.")

    def _compute(self, X: np.ndarray, params: dict) -> np.ndarray:
        """
        Pure computational logic.  X is (T, D) float32 for one block.

        Replace this with your own algorithm.  This function should NOT
        touch DataFrames, dataset, or file I/O.
        """
        # EXAMPLE: sliding-window mean
        win = int(params["window_size"])
        if win <= 1:
            return X
        T, D = X.shape
        out = np.zeros_like(X)
        for t in range(T):
            lo = max(0, t - win // 2)
            hi = min(T, t + win // 2 + 1)
            out[t] = X[lo:hi].mean(axis=0)
        return out
