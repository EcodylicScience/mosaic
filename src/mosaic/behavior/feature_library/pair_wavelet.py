"""PairWavelet feature -- CWT spectrograms on PairPoseDistancePCA outputs."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, final

import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import (
    Inputs,
    Params,
    TrackInput,
    resolve_order_col,
)

from .helpers import ensure_columns
from .registry import register_feature
from .types import SamplingConfig

if TYPE_CHECKING:
    import pywt

try:
    import pywt  # pyright: ignore[reportUnknownVariableType]

    _has_pywt = True
except ImportError:
    _has_pywt = False


@final
@register_feature
class PairWavelet:
    """
    CWT spectrograms on PairPoseDistancePCA outputs.

    Expects input df to contain columns:
      - 'perspective' (0 = A->B, 1 = B->A)
      - 'frame' (preferred) or 'time' (if used as order column)
      - PC0..PC{k-1} (k = number of PCA components)

    Returns a DataFrame with columns:
      - frame (or time if that was the order col)
      - perspective
      - W_{col}_f{fi} (log-power, clamped, for each component x frequency)
      and (optionally) passthrough group/sequence if present in df.

    Stateless (no fitting). FPS is inferred from constant df['fps'] if
    present, otherwise from fps_default. Frequencies are dyadically spaced
    in [f_min, f_max].
    """

    name = "pair-wavelet"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        sampling: SamplingConfig = Field(default_factory=SamplingConfig)
        f_min: float = Field(default=0.2, gt=0)
        f_max: float = Field(default=5.0, gt=0)
        n_freq: int = Field(default=25, gt=0)
        wavelet: str = "cmor1.5-1.0"
        log_floor: float = -3.0
        pc_prefix: str = "PC"
        cols: list[str] | None = None

    def __init__(
        self,
        inputs: PairWavelet.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        if not _has_pywt:
            raise ImportError(
                "PyWavelets (pywt) not available. Install with `pip install PyWavelets`."
            )
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._cache_key: tuple[str, float, float, int, float] | None = None
        self._frequencies: np.ndarray | None = None
        self._scales: np.ndarray | None = None
        self._central_f: float | None = None

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_indices: dict[str, pd.DataFrame],
    ) -> bool:
        return True

    def fit(self, inputs: Callable[[], Iterator[tuple[str, pd.DataFrame]]]) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        order_col = resolve_order_col(df)
        fps = self._infer_fps(df, p.sampling.fps_default)
        in_cols = self._select_input_columns(df)
        ensure_columns(df, ["perspective"])

        self._prepare_band(fps)
        assert self._frequencies is not None and self._scales is not None

        n_freq = len(self._frequencies)
        has_pair_ids = "id1" in df.columns and "id2" in df.columns
        group_keys = ["id1", "id2", "perspective"] if has_pair_ids else ["perspective"]

        out_blocks: list[pd.DataFrame] = []
        for _, g in df.groupby(group_keys):
            g = g.sort_values(order_col)
            persp = int(g["perspective"].iloc[0])
            signal = g[in_cols].to_numpy(dtype=float)  # (T, k)
            n_time, n_components = signal.shape

            # CWT power spectrogram per component
            power = np.empty((n_components, n_freq, n_time), dtype=np.float32)
            for comp in range(n_components):
                coeffs, _ = pywt.cwt(  # pyright: ignore[reportUnknownMemberType]
                    signal[:, comp],
                    self._scales,
                    self._wavelet_obj(),
                    sampling_period=1.0 / float(fps),
                )
                power[comp] = (np.abs(coeffs) ** 2).astype(np.float32)

            # log + clamp
            eps = np.finfo(np.float32).tiny
            log_power = np.maximum(np.log(power + eps), p.log_floor)

            # flatten to (T, k*n_freq)
            flat = log_power.reshape(n_components * n_freq, n_time).T
            colnames = [
                f"W_{in_cols[comp]}_f{fi}"
                for comp in range(n_components)
                for fi in range(n_freq)
            ]
            block = pd.DataFrame(flat, columns=colnames)
            block[order_col] = g[order_col].to_numpy()
            block["perspective"] = persp

            if has_pair_ids:
                block["id1"] = g["id1"].iloc[0]
                block["id2"] = g["id2"].iloc[0]

            for col in (C.seq_col, C.group_col):
                if col in df.columns:
                    block[col] = df[col].iloc[0]

            out_blocks.append(block)

        if not out_blocks:
            return pd.DataFrame(columns=[order_col, "perspective"])

        out = pd.concat(out_blocks, ignore_index=True)
        sort_keys: list[str] = []
        if "id1" in out.columns:
            sort_keys += ["id1", "id2"]
        sort_keys += ["perspective", order_col]
        out = out.sort_values(sort_keys).reset_index(drop=True)

        out.attrs["frequencies_hz"] = self._frequencies.tolist()
        out.attrs["scales"] = self._scales.tolist()
        out.attrs["wavelet"] = str(p.wavelet)
        out.attrs["fps"] = float(fps)
        out.attrs["pc_cols"] = [c for c in in_cols if c.startswith(p.pc_prefix)]
        out.attrs["input_columns"] = list(map(str, in_cols))
        return out

    def _select_input_columns(self, df: pd.DataFrame) -> list[str]:
        # 1) explicit columns override
        cols_param = self.params.cols
        if cols_param:
            cols = [c for c in cols_param if c in df.columns]
            if not cols:
                raise ValueError(
                    "[pair-wavelet] None of the requested 'cols' are present in df."
                )
            return cols
        # 2) PC-prefixed columns
        pc_cols = self._pc_columns(df, self.params.pc_prefix)
        if pc_cols:
            return pc_cols
        # 3) Auto-detect: all numeric columns except known meta
        meta_like = C.meta_set() | {"perspective", "fps", "id1", "id2"}
        num_cols = sorted(
            set(df.select_dtypes(include=[np.number]).columns) - meta_like
        )
        if not num_cols:
            raise ValueError(
                "[pair-wavelet] Could not auto-detect numeric feature columns."
            )
        return num_cols

    def _infer_fps(self, df: pd.DataFrame, default: float) -> float:
        if "fps" in df.columns:
            vals = pd.Series(df["fps"]).dropna().unique()
            if len(vals) == 1:
                return float(vals[0])
        return float(default)

    def _pc_columns(self, df: pd.DataFrame, prefix: str) -> list[str]:
        pc_cols: list[str] = []
        i = 0
        while True:
            col = f"{prefix}{i}"
            if col in df.columns:
                pc_cols.append(col)
                i += 1
            else:
                break
        return pc_cols

    def _prepare_band(self, fps: float) -> None:
        key = (
            self.params.wavelet,
            self.params.f_min,
            self.params.f_max,
            self.params.n_freq,
            float(fps),
        )
        if self._cache_key == key and self._frequencies is not None:
            return
        f_min = self.params.f_min
        f_max = self.params.f_max
        n_freq = self.params.n_freq
        freqs = np.logspace(math.log2(f_min), math.log2(f_max), n_freq, base=2.0)
        w = self._wavelet_obj()
        central_f: float = pywt.central_frequency(w)  # pyright: ignore[reportUnknownMemberType]
        scales = float(fps) / (freqs * central_f)
        self._frequencies = freqs.astype(np.float32)
        self._scales = scales.astype(np.float32)
        self._central_f = float(central_f)
        self._cache_key = key

    def _wavelet_obj(self) -> object:
        return pywt.ContinuousWavelet(self.params.wavelet)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
