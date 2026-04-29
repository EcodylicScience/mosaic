"""
Template for a per-sequence feature.

Copy this file, rename the class and `name`, and fill in your logic.

Protocol (4 attributes + 4 methods):
  - name, version, parallelizable, scope_dependent
  - load_state(run_root, artifact_paths, dependency_lookups) -> bool
  - fit(inputs: factory returning iterator of (entry_key, DataFrame)) -> None
  - save_state(run_root) -> None
  - apply(df: DataFrame) -> DataFrame

Per-sequence features are stateless by default: load_state returns True
(nothing to restore), fit/save_state are no-ops, and apply does all the work.
Set scope_dependent = False unless outputs depend on which sequences are in scope.

See SpeedAngvel for a real per-sequence feature.
"""

from __future__ import annotations

from pathlib import Path
from typing import final

import numpy as np
import pandas as pd

# from .registry import register_feature  # <-- uncomment when ready
from mosaic.core.pipeline.types import (
    COLUMNS,
    DependencyLookup,
    Inputs,
    InputStream,
    Params,
    TrackInput,
    resolve_order_col,
)


@final
# @register_feature   # <-- uncomment when your feature is ready
class MyPerSequenceFeature:
    """
    Template for a per-sequence feature.

    Input:
      A DataFrame for a single (group, sequence) from either:
        * tracks (input_kind="tracks")
        * another feature (input_kind="feature")
        * a multi-input Inputs() tuple

    Output:
      A DataFrame with one row per frame (or per frame x pair), with:
        * frame (or time)
        * group, sequence
        * id1, id2 (when pair-aware)
        * your feature columns
    """

    category = "per-frame"  # diagram color: "per-frame" / "summary" / "tag" / "global" / custom
    name = "my-new-feature"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        """Per-sequence feature template parameters.

        Attributes:
            window_size: Sliding window size. Default 15.
        """

        window_size: int = 15

    def __init__(
        self,
        inputs: MyPerSequenceFeature.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params: MyPerSequenceFeature.Params = self.Params.from_overrides(params)

    # --- State ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        return True  # stateless -- nothing to restore

    def fit(self, inputs: InputStream) -> None:
        pass  # stateless -- no fitting required

    def save_state(self, run_root: Path) -> None:
        pass  # stateless -- nothing to persist

    # --- Apply ---

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for a single (group, sequence).

        For pair-aware inputs the df may contain multiple (id1, id2) pairs;
        process each pair independently to avoid mixing contexts.
        """
        if df.empty:
            return pd.DataFrame()

        order_col = resolve_order_col(df)

        # Detect pair structure
        has_pairs = "id1" in df.columns and "id2" in df.columns
        if has_pairs:
            group_keys = ["id1", "id2"]
        else:
            group_keys = None

        # Select numeric input columns (exclude metadata)
        meta_like = {
            COLUMNS.seq_col,
            COLUMNS.group_col,
            "frame",
            "time",
            "id",
            "perspective",
            "fps",
            "id1",
            "id2",
        }
        numeric_cols = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c not in meta_like
        ]
        if not numeric_cols:
            return pd.DataFrame()

        # Process per-pair (or whole df if no pairs)
        if group_keys:
            blocks: list[pd.DataFrame] = []
            for group_vals, g in df.groupby(group_keys, sort=False):
                assert isinstance(group_vals, tuple)
                cur_id1, cur_id2 = group_vals
                block_out = self._process_block(g, numeric_cols, order_col)
                if len(block_out):
                    block_out["id1"] = int(cur_id1)  # type: ignore[arg-type]
                    block_out["id2"] = int(cur_id2)  # type: ignore[arg-type]
                    blocks.append(block_out)
            return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
        else:
            return self._process_block(df, numeric_cols, order_col)

    def _process_block(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        order_col: str,
    ) -> pd.DataFrame:
        """Process a single block (one pair or one sequence)."""
        seq_col = COLUMNS.seq_col
        group_col = COLUMNS.group_col

        df = df.sort_values(order_col).reset_index(drop=True)
        X = df[numeric_cols].to_numpy(dtype=np.float32, copy=False)

        # --- YOUR LOGIC HERE ---
        features = self._compute(X)

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

    # --- Internal helpers ---

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """
        Pure computational logic.  X is (T, D) float32 for one block.

        Replace this with your own algorithm.  This function should NOT
        touch DataFrames, dataset, or file I/O.
        """
        # EXAMPLE: sliding-window mean
        win = self.params.window_size
        if win <= 1:
            return X
        T = X.shape[0]
        out = np.zeros_like(X)
        for t in range(T):
            lo = max(0, t - win // 2)
            hi = min(T, t + win // 2 + 1)
            out[t] = X[lo:hi].mean(axis=0)
        return out
