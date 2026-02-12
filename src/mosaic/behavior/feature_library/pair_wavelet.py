"""
PairWavelet feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    import pywt
    _PYWT_OK = True
except Exception:
    _PYWT_OK = False

from mosaic.core.dataset import register_feature
from .helpers import _merge_params


@register_feature
class PairWavelet:
    """
    'pair-wavelet' — CWT spectrograms on PairPoseDistancePCA outputs.
    Expects input df to contain columns:
        - 'perspective' (0 = A→B, 1 = B→A)
        - 'frame' (preferred) or 'time' (if used as order column)
        - PC0..PC{k-1} (k = number of PCA components)
    Returns a DataFrame with columns:
        - frame (or time if that was the order col)
        - perspective
        - W_c{comp}_f{fi}  (log-power, clamped, for each component×frequency)
      and (optionally) passthrough group/sequence if present in df.

    Notes:
      • Stateless (no fitting).
      • FPS is inferred from constant df['fps'] if present; else fps_default.
      • Frequencies are dyadically spaced in [f_min, f_max].
    """

    name = "pair-wavelet"
    version = "0.1"
    parallelizable = True
    output_type = "per_frame"

    _defaults = dict(
        # sampling
        fps_default=30.0,

        # wavelet band and resolution
        f_min=0.2,
        f_max=5.0,
        n_freq=25,

        # wavelet family string (PyWavelets)
        wavelet="cmor1.5-1.0",

        # log power clamp
        log_floor=-3.0,

        # naming / passthrough
        pc_prefix="PC",                 # columns like PC0, PC1, ...
        order_pref=("frame", "time"),   # which column to use as the time base
        seq_col="sequence",
        group_col="group",
        cols=None,  # explicit list of columns to transform; if None, fallback to PC prefix or auto-detect numeric columns
    )
    def _select_input_columns(self, df: pd.DataFrame) -> List[str]:
        # 1) explicit columns override
        cols_param = self.params.get("cols", None)
        if cols_param:
            cols = [c for c in cols_param if c in df.columns]
            if not cols:
                raise ValueError("[pair-wavelet] None of the requested 'cols' are present in df.")
            return cols
        # 2) PC-prefixed columns
        pc_cols = self._pc_columns(df, self.params["pc_prefix"])
        if pc_cols:
            return pc_cols
        # 3) Auto-detect: all numeric columns except known meta
        meta_like = {self.params.get("seq_col", "sequence"),
                     self.params.get("group_col", "group"),
                     "frame", "time", "perspective", "id", "fps",
                     "id1", "id2"}
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in meta_like]
        if not num_cols:
            raise ValueError("[pair-wavelet] Could not auto-detect numeric feature columns.")
        return num_cols

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        if not _PYWT_OK:
            raise ImportError(
                "PyWavelets (pywt) not available. Install with `pip install PyWavelets`."
            )
        self.params = _merge_params(params, self._defaults)
        # pre-build frequency vector & scales for speed; will recompute if params change
        self._cache_key = None
        self._frequencies = None
        self._scales = None
        self._central_f = None

    # ---- feature protocol ----
    def needs_fit(self) -> bool: return False
    def supports_partial_fit(self) -> bool: return False
    def finalize_fit(self) -> None: pass
    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None: return
    def partial_fit(self, df: pd.DataFrame) -> None: return

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        order_col = self._order_col(df)
        fps = self._infer_fps(df, p["fps_default"])
        in_cols = self._select_input_columns(df)
        if "perspective" not in df.columns:
            raise ValueError("[pair-wavelet] Missing 'perspective' column.")

        # prepare wavelet frequencies/scales
        self._prepare_band(fps)

        # compute per (id1, id2, perspective) block — or just perspective if no pair IDs
        has_pair_ids = "id1" in df.columns and "id2" in df.columns
        group_keys = ["id1", "id2", "perspective"] if has_pair_ids else ["perspective"]

        out_blocks: List[pd.DataFrame] = []
        for group_vals, g in df.groupby(group_keys):
            if has_pair_ids:
                cur_id1, cur_id2, persp = group_vals
            else:
                persp = group_vals if not isinstance(group_vals, tuple) else group_vals[0]
                cur_id1 = cur_id2 = None

            g = g.sort_values(order_col)
            Z = g[in_cols].to_numpy(dtype=float)  # shape (T, k)
            T, k = Z.shape

            # compute power spectrogram (k components × n_freq × T)
            power = np.empty((k, len(self._frequencies), T), dtype=np.float32)
            # each component independently
            for comp in range(k):
                coeffs, _ = pywt.cwt(
                    Z[:, comp],
                    self._scales,
                    self._wavelet_obj(),
                    sampling_period=1.0 / float(fps),
                )
                power[comp] = (np.abs(coeffs) ** 2).astype(np.float32)

            # log + clamp
            eps = np.finfo(np.float32).tiny
            log_power = np.log(power + eps)
            log_power = np.maximum(log_power, float(p["log_floor"]))

            # flatten to (T, k*n_freq)
            flat = log_power.reshape(k * len(self._frequencies), T).T  # (T, F_flat)

            # column names: W_{in_cols[comp]}_f{fi}
            colnames = [
                f"W_{in_cols[comp]}_f{fi}"
                for comp in range(k)
                for fi in range(len(self._frequencies))
            ]
            block = pd.DataFrame(flat, columns=colnames)
            block[order_col] = g[order_col].to_numpy()
            block["perspective"] = int(persp)

            # passthrough pair IDs
            if cur_id1 is not None:
                block["id1"] = cur_id1
                block["id2"] = cur_id2

            # optional passthrough
            for col in (p["seq_col"], p["group_col"]):
                if col in df.columns:
                    block[col] = df[col].iloc[0]

            out_blocks.append(block)

        if not out_blocks:
            return pd.DataFrame(columns=[order_col, "perspective"])

        out = pd.concat(out_blocks, ignore_index=True)
        sort_keys = []
        if "id1" in out.columns:
            sort_keys += ["id1", "id2"]
        sort_keys += ["perspective", order_col]
        out = out.sort_values(sort_keys).reset_index(drop=True)

        # Attach JSON-serializable metadata only (so parquet writers won't error)
        try:
            out.attrs["frequencies_hz"] = self._frequencies.tolist() if self._frequencies is not None else []
            out.attrs["scales"] = self._scales.tolist() if self._scales is not None else []
            out.attrs["wavelet"] = str(self.params.get("wavelet", ""))
            out.attrs["fps"] = float(fps)
            out.attrs["pc_cols"] = [c for c in in_cols if c.startswith(self.params.get("pc_prefix","PC"))]
            out.attrs["input_columns"] = list(map(str, in_cols))
        except Exception:
            # As a safety net, drop attrs if anything is not serializable
            out.attrs.clear()
        return out

    # ---- internals ----
    def _order_col(self, df: pd.DataFrame) -> str:
        for c in self.params["order_pref"]:
            if c in df.columns:
                return c
        raise ValueError("Need either 'frame' or 'time' column in df.")

    def _infer_fps(self, df: pd.DataFrame, default: float) -> float:
        if "fps" in df.columns:
            vals = pd.Series(df["fps"]).dropna().unique()
            if len(vals) == 1:
                try:
                    return float(vals[0])
                except Exception:
                    pass
        return float(default)

    def _pc_columns(self, df: pd.DataFrame, prefix: str) -> List[str]:
        # accept PC0, PC1, ... contiguous from 0 until missing
        pc_cols = []
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
        key = (self.params["wavelet"], float(self.params["f_min"]),
               float(self.params["f_max"]), int(self.params["n_freq"]), float(fps))
        if self._cache_key == key and self._frequencies is not None:
            return
        f_min = float(self.params["f_min"])
        f_max = float(self.params["f_max"])
        n_freq = int(self.params["n_freq"])
        # dyadic spacing
        freqs = 2.0 ** np.linspace(np.log2(f_min), np.log2(f_max), n_freq)
        w = self._wavelet_obj()
        central_f = pywt.central_frequency(w)
        scales = float(fps) / (freqs * central_f)
        self._frequencies = freqs.astype(np.float32)
        self._scales = scales.astype(np.float32)
        self._central_f = float(central_f)
        self._cache_key = key

    def _wavelet_obj(self):
        return pywt.ContinuousWavelet(self.params["wavelet"])
