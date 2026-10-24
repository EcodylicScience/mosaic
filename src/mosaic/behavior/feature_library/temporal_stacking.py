"""Temporal stacking feature.

Builds temporal context windows over per-sequence feature data by stacking
Gaussian-smoothed frames at time offsets and optional pooled statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, final

import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from scipy.ndimage import gaussian_filter1d

from mosaic.core.pipeline.column_contract import (
    META_COLUMNS_KEPT,
    Payload,
    Provides,
    Requirement,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    EmitsLevel,
    Inputs,
    InputStream,
    NNResult,
    Result,
)
from mosaic.core.params import Declared, Params

from .helpers import chosen_payload_columns, meta_columns
from ._pair_filter_descriptions import PAIR_FILTER_DESCRIPTION
from .registry import register_feature
from .types import PAYLOAD_COLUMNS_DESCRIPTION, PayloadColumnNames

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


def _pool_window(sigma: float, fps: float, win_sec: float) -> tuple[float, float]:
    """The sigma and truncation of the pooling Gaussian, in frames."""
    if sigma <= 0:
        sigma = max(1.0, win_sec * fps / 6.0)
    win_frames = max(1, int(round(win_sec * fps)))
    truncate = max(1.0, win_frames / (2.0 * sigma)) if sigma > 0 else 4.0
    return sigma, truncate


def _gaussian_radius(sigma: float, truncate: float) -> int:
    """Frames on each side that ``gaussian_filter1d`` reads: ``int(truncate * sigma + 0.5)``."""
    return int(truncate * sigma + 0.5)


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
    sigma, truncate = _pool_window(sigma, fps, win_sec)
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

_HALF_DESCRIPTION = (
    "Half-width of the temporal window. The full window spans -half to "
    "+half around each row."
)

_SKIP_DESCRIPTION = "The step between time offsets in the stacking window."

_USE_TEMPORAL_STACK_DESCRIPTION = (
    "Concatenate Gaussian-smoothed copies of each feature column at "
    "every offset in the window. False passes through the input "
    "feature columns unchanged."
)

_SIGMA_STACK_DESCRIPTION = (
    "The Gaussian smoothing sigma applied before stacking. 0 disables smoothing."
)

_ADD_POOL_DESCRIPTION = (
    "Append pooled statistics computed over a sliding Gaussian window."
)

_POOL_STATS_DESCRIPTION = (
    "Which pooled statistics to compute over the sliding Gaussian "
    "window. Known values are mean, std and variance, matched "
    "case-insensitively. Any other value is ignored."
)

_SIGMA_POOL_DESCRIPTION = (
    "The Gaussian sigma for the pooling window. Non-positive computes "
    "it from win_sec and fps."
)

_FPS_DESCRIPTION = "The frame rate used to convert win_sec into a frame count."

_WIN_SEC_DESCRIPTION = "The width of the pooling window."


@final
@register_feature
class TemporalStackingFeature:
    """Build temporal context windows over per-sequence feature data.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.temporal_stacking.TemporalStackingFeature.Params`.
    """

    category = "per-frame"
    name = "temporal-stack"
    # 0.5: ``nn_id`` and ``attention_target_id``, track ids, are neither
    # stacked nor carried (``BOOKKEEPING_COLS``).
    version = "0.5"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "as-input"
    # ``apply`` also sorts a pair input, one with ``id1`` and ``id2``, by
    # ``frame``. A requirement cannot depend on the input's other columns.
    # ``columns`` is not declared, because a requirement on a list param is met
    # by any one of its names. ``apply`` refuses each named column that the
    # input lacks.
    requires: ClassVar[tuple[Requirement, ...]] = (Payload(),)
    # ``apply`` reads every payload column, or those that ``columns`` names. Each
    # is written as ``<column>__t<offset>`` per offset, or unchanged without the
    # stack, and as ``<column>__pool_mean``, ``__pool_var`` or ``__pool_std``
    # per pooled statistic. The other payload columns and the bookkeeping
    # columns are dropped. A declared family cannot depend on ``columns``, and
    # validation expects one per payload column.
    provides: ClassVar[Provides] = Provides(
        conditional=META_COLUMNS_KEPT, payload_families=True
    )

    class Inputs(Inputs[Result]):
        pass

    class Params(Params):
        half: Annotated[int, Declared(_HALF_DESCRIPTION, unit="frames")] = Field(
            default=60, ge=0
        )
        skip: Annotated[int, Declared(_SKIP_DESCRIPTION, unit="frames")] = Field(
            default=5, ge=1
        )
        use_temporal_stack: Annotated[
            bool, Declared(_USE_TEMPORAL_STACK_DESCRIPTION)
        ] = True
        sigma_stack: Annotated[
            float, Declared(_SIGMA_STACK_DESCRIPTION, unit="frames")
        ] = 30.0
        add_pool: Annotated[bool, Declared(_ADD_POOL_DESCRIPTION)] = True
        pool_stats: Annotated[
            tuple[str, ...],
            Field(examples=[("mean",), ("mean", "std", "variance")]),
            Declared(_POOL_STATS_DESCRIPTION),
        ] = ("mean",)
        sigma_pool: Annotated[
            float, Declared(_SIGMA_POOL_DESCRIPTION, unit="frames")
        ] = 30.0
        fps: Annotated[float, Declared(_FPS_DESCRIPTION, unit="fps")] = Field(
            default=30.0, gt=0
        )
        win_sec: Annotated[float, Declared(_WIN_SEC_DESCRIPTION, unit="s")] = Field(
            default=0.5, gt=0
        )
        pair_filter: Annotated[NNResult | None, Declared(PAIR_FILTER_DESCRIPTION)] = (
            None
        )
        columns: Annotated[
            PayloadColumnNames | None, Declared(PAYLOAD_COLUMNS_DESCRIPTION)
        ] = None

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

    @staticmethod
    def temporal_reach(params: TemporalStackingFeature.Params) -> int:
        """Frames on each side that one row reads.

        The stack reaches ``half`` frames, and the Gaussian smoothing of each copy
        reads ``int(4 * sigma_stack + 0.5)`` more (scipy's default truncation). The
        pooled statistics read their own Gaussian window. The row reads the larger
        of the two. Rows are taken to be consecutive frames.
        """
        reach = 0
        if params.use_temporal_stack:
            smoothing = (
                _gaussian_radius(params.sigma_stack, 4.0)
                if params.sigma_stack > 0
                else 0
            )
            reach = params.half + smoothing
        if params.add_pool and any(
            stat in ("mean", "std", "variance") for stat in params.pool_stats
        ):
            sigma, truncate = _pool_window(
                params.sigma_pool, params.fps, params.win_sec
            )
            reach = max(reach, _gaussian_radius(sigma, truncate))
        return reach

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

        cols = chosen_payload_columns(df, self.params.columns, self.name)
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
        meta = meta_columns(df)
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
        meta = meta_columns(df)
        parts: list[pd.DataFrame] = []
        for _, sub in df.groupby(self._pair_keys(df), sort=False):
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

    @staticmethod
    def _pair_keys(df: pd.DataFrame) -> list[str]:
        """The columns that separate one pair's time series from another's.

        ``perspective`` is one of them wherever it is present. Grouping on
        ``id1``/``id2`` alone puts both perspectives of a pair in one block, where
        sorting by frame interleaves them -- so the smoothing and the offset stack
        read across the two, and a row's own value and both of its neighbours come
        from the wrong perspective.
        """
        keys = ["id1", "id2"]
        if "perspective" in df.columns:
            keys.append("perspective")
        return keys

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
