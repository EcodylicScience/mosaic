"""Temporal stacking feature.

Builds temporal context windows over per-sequence feature data by stacking
Gaussian-smoothed frames at time offsets and optional pooled statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from scipy.ndimage import gaussian_filter1d

from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import DependencyLookup, Inputs, InputStream, NNResult, Params, Result

from .helpers import feature_columns
from .registry import register_feature

# --- Computation helpers ---


def _estimate_output_bytes(
    num_rows: int,
    num_input_cols: int,
    num_offsets: int,
    num_pool_stats: int,
) -> int:
    """Estimate output matrix size in bytes (float32)."""
    stacked_cols = num_input_cols * num_offsets
    pool_cols = num_input_cols * num_pool_stats
    return num_rows * (stacked_cols + pool_cols) * 4


def _check_memory(estimated_bytes: int) -> None:
    """Raise if estimated allocation would leave less than 20% free memory."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        available = mem.available
    except (ImportError, AttributeError):
        return
    headroom = 0.20
    if estimated_bytes > available * (1.0 - headroom):
        est_gb = estimated_bytes / (1024**3)
        avail_gb = available / (1024**3)
        msg = (
            f"[temporal-stack] Output would require ~{est_gb:.1f} GB "
            f"but only {avail_gb:.1f} GB available"
        )
        raise MemoryError(msg)


def _temporal_stack(
    base: np.ndarray,
    base_names: list[str],
    half: int,
    skip: int,
    sigma_stack: float,
    use_stack: bool,
    add_pool: bool,
    pool_stats: tuple[str, ...],
    sigma_pool: float,
    fps: float,
    win_sec: float,
) -> tuple[np.ndarray, list[str]]:
    """Compute temporal stacking and pooled stats on a feature matrix.

    Parameters
    ----------
    base : ndarray, shape (T, D)
        Input feature matrix.
    base_names : list[str]
        Column names for each of the D features.

    Returns
    -------
    result : ndarray, shape (T, D_out), float32
    result_names : list[str]
    """
    num_rows, num_cols = base.shape
    offsets = list(range(-half, half + 1, skip))

    # Memory check
    num_pool = (
        sum(1 for s in pool_stats if s in ("mean", "std", "variance"))
        if add_pool
        else 0
    )
    estimated = _estimate_output_bytes(
        num_rows, num_cols, len(offsets) if use_stack else 1, num_pool
    )
    _check_memory(estimated)

    parts: list[np.ndarray] = []
    names: list[str] = []

    # Temporal stack: smoothed copies at each time offset
    if use_stack:
        smoothed = base
        if sigma_stack > 0:
            smoothed = gaussian_filter1d(
                base, sigma=sigma_stack, axis=0, mode="nearest"
            )
        if half > 0:
            padded = np.pad(smoothed, ((half, half), (0, 0)), mode="edge")
            idx = np.arange(num_rows) + half
            stack = np.concatenate([padded[idx + off] for off in offsets], axis=1)
        else:
            stack = np.tile(smoothed, (1, len(offsets)))
        parts.append(stack)
        for off in offsets:
            names.extend([f"{name}__t{off:+03d}" for name in base_names])
    else:
        parts.append(base)
        names.extend(base_names)

    # Pooled statistics
    if add_pool and pool_stats:
        pool_parts, pool_names = _pooled_stats(
            base,
            base_names,
            pool_stats,
            sigma_pool,
            fps,
            win_sec,
        )
        parts.extend(pool_parts)
        names.extend(pool_names)

    result = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    return result, names


def _pooled_stats(
    base: np.ndarray,
    base_names: list[str],
    stats: tuple[str, ...],
    sigma: float,
    fps: float,
    win_sec: float,
) -> tuple[list[np.ndarray], list[str]]:
    """Compute pooled statistics over a sliding Gaussian window."""
    if not stats:
        return [], []
    if sigma <= 0:
        sigma = max(1.0, win_sec * fps / 6.0)
    win_frames = max(1, int(round(win_sec * fps)))
    truncate = max(1.0, win_frames / (2.0 * sigma)) if sigma > 0 else 4.0
    mean_vals = gaussian_filter1d(
        base,
        sigma=sigma,
        axis=0,
        mode="nearest",
        truncate=truncate,
    )
    outputs: list[np.ndarray] = []
    names: list[str] = []
    if "mean" in stats:
        outputs.append(mean_vals)
        names.extend([f"{name}__pool_mean" for name in base_names])
    if "std" in stats or "variance" in stats:
        second = gaussian_filter1d(
            base**2,
            sigma=sigma,
            axis=0,
            mode="nearest",
            truncate=truncate,
        )
        var = np.clip(second - mean_vals**2, 0.0, None)
        if "variance" in stats:
            outputs.append(var)
            names.extend([f"{name}__pool_var" for name in base_names])
        if "std" in stats:
            outputs.append(np.sqrt(var))
            names.extend([f"{name}__pool_std" for name in base_names])
    return outputs, names


# --- Feature class ---


@final
@register_feature
class TemporalStackingFeature:
    """Build temporal context windows over per-sequence feature data."""

    name = "temporal-stack"
    version = "0.3"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[Result]):
        pass

    class Params(Params):
        half: int = Field(default=60, ge=0)
        skip: int = Field(default=5, ge=1)
        use_temporal_stack: bool = True
        sigma_stack: float = 30.0
        add_pool: bool = True
        pool_stats: tuple[str, ...] = ("mean",)
        sigma_pool: float = 30.0
        fps: float = Field(default=30.0, gt=0)
        win_sec: float = Field(default=0.5, gt=0)
        pair_filter: NNResult | None = None

        @model_validator(mode="before")
        @classmethod
        def _normalize_pool_stats(cls, data: dict[str, object]) -> dict[str, object]:
            if isinstance(data, dict):
                ps = data.get("pool_stats")
                if isinstance(ps, str):
                    data["pool_stats"] = (ps.lower(),)
                elif ps is not None:
                    data["pool_stats"] = tuple(str(s).lower() for s in ps)
            return data

    def __init__(
        self,
        inputs: TemporalStackingFeature.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

    # --- Protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        cols = feature_columns(df)
        if not cols:
            msg = "[temporal-stack] No feature columns found in input"
            raise ValueError(msg)

        has_pairs = "id1" in df.columns and "id2" in df.columns
        if has_pairs:
            return self._apply_pairs(df, cols)
        return self._apply_single(df, cols)

    # --- Internals ---

    def _apply_single(
        self,
        df: pd.DataFrame,
        cols: list[str],
    ) -> pd.DataFrame:
        meta = sorted(((C.meta_set() | {"id1", "id2"}) & set(df.columns)) - set(cols))
        base = df[cols].to_numpy(dtype=np.float32)
        stacked, stacked_names = self._stack(base, cols)
        result = pd.DataFrame(stacked, columns=stacked_names, index=df.index)
        for col in meta:
            result[col] = df[col].values
        return result

    def _apply_pairs(
        self,
        df: pd.DataFrame,
        cols: list[str],
    ) -> pd.DataFrame:
        meta = sorted(((C.meta_set() | {"id1", "id2"}) & set(df.columns)) - set(cols))
        parts: list[pd.DataFrame] = []
        for _, sub in df.groupby(["id1", "id2"], sort=False):
            sub = sub.sort_values("frame").reset_index(drop=True)
            base = sub[cols].to_numpy(dtype=np.float32)
            stacked, stacked_names = self._stack(base, cols)
            part = pd.DataFrame(stacked, columns=stacked_names, index=sub.index)
            for col in meta:
                part[col] = sub[col].values
            parts.append(part)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    def _stack(
        self,
        base: np.ndarray,
        base_names: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        params = self.params
        return _temporal_stack(
            base,
            base_names,
            half=params.half,
            skip=params.skip,
            sigma_stack=params.sigma_stack,
            use_stack=params.use_temporal_stack,
            add_pool=params.add_pool,
            pool_stats=params.pool_stats,
            sigma_pool=params.sigma_pool,
            fps=params.fps,
            win_sec=params.win_sec,
        )
