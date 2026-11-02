"""Continuous wavelet transform of numeric columns, shared by the wavelet features.

The transform is a continuous wavelet transform with PyWavelets, by default the
complex Morlet wavelet ``cmor1.5-1.0`` at 25 frequencies. Each column gives the
natural log of the squared coefficient magnitude, clamped from below at
``log_floor``. Frequencies are geometrically spaced in ``[f_min, f_max]``, with
equal steps in log2 frequency.

**Dtypes.** The scales are float32 and each column is one ``pywt.cwt`` call. The
power is cast to float32 before the float32 ``tiny`` is added and the log taken,
then the clamp applies. A float64 computation of ``log(abs(coeffs) ** 2 + tiny)``
differs from this by up to 2.6e-4 relative, so a caller that matches the output
bit for bit repeats this sequence.

``who`` in the functions below is the name of the feature that opens each error
message.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pandas as pd
import pywt
from pydantic import Field

from mosaic.core.params import Declared, Params

from ._sampling_descriptions import SAMPLING_FPS_ONLY_DESCRIPTION
from .helpers import feature_columns
from .types import SamplingConfig

__all__ = [
    "WaveletBand",
    "WaveletParams",
    "constant_fps",
    "log_power",
    "wavelet_attrs",
    "wavelet_band",
    "wavelet_columns",
    "wavelet_input_columns",
]

_F_MIN_DESCRIPTION = "Minimum frequency of the CWT frequency band."

_F_MAX_DESCRIPTION = "Maximum frequency of the CWT frequency band."

_N_FREQ_DESCRIPTION = (
    "Number of frequency bins, dyadically spaced between f_min and f_max."
)

_WAVELET_DESCRIPTION = "Name of the PyWavelets continuous wavelet used for the CWT."

_LOG_FLOOR_DESCRIPTION = "Minimum value the log-power spectrogram is clamped to."

_PC_PREFIX_DESCRIPTION = (
    "Column-name prefix used to auto-detect principal-component input "
    "columns, for example PC0 and PC1."
)

_COLS_DESCRIPTION = (
    "Explicit input column names for the CWT. Unset, columns are "
    "auto-detected via pc_prefix, falling back to every numeric feature "
    "column when none match."
)

type Float32Array = npt.NDArray[np.float32]


class WaveletParams(Params):
    """The band, wavelet, floor and input-column params of a wavelet feature."""

    sampling: Annotated[SamplingConfig, Declared(SAMPLING_FPS_ONLY_DESCRIPTION)] = (
        Field(default_factory=SamplingConfig)
    )
    f_min: Annotated[float, Field(gt=0), Declared(_F_MIN_DESCRIPTION, unit="Hz")] = 0.2
    f_max: Annotated[float, Field(gt=0), Declared(_F_MAX_DESCRIPTION, unit="Hz")] = 5.0
    n_freq: Annotated[int, Field(gt=0), Declared(_N_FREQ_DESCRIPTION)] = 25
    wavelet: Annotated[str, Declared(_WAVELET_DESCRIPTION)] = "cmor1.5-1.0"
    log_floor: Annotated[float, Declared(_LOG_FLOOR_DESCRIPTION)] = -3.0
    pc_prefix: Annotated[str, Declared(_PC_PREFIX_DESCRIPTION)] = "PC"
    cols: Annotated[list[str] | None, Declared(_COLS_DESCRIPTION)] = None


@dataclass(frozen=True)
class WaveletBand:
    """The frequencies of a band in Hz and the wavelet scales that realize them."""

    frequencies: Float32Array
    scales: Float32Array


@lru_cache(maxsize=32)
def wavelet_band(
    wavelet: str, f_min: float, f_max: float, n_freq: int, fps: float
) -> WaveletBand:
    """Return the band of *n_freq* frequencies from *f_min* to *f_max* at *fps*.

    The result is cached on the arguments, and its arrays are read-only.

    Args:
        wavelet: The PyWavelets continuous wavelet name.
        f_min: The lowest frequency in Hz.
        f_max: The highest frequency in Hz.
        n_freq: The number of frequencies, spaced equally in log2 frequency.
        fps: The frame rate in frames per second.

    Returns:
        The float32 frequencies and scales.
    """
    freqs = np.logspace(math.log2(f_min), math.log2(f_max), n_freq, base=2.0)
    central_f: float = pywt.central_frequency(pywt.ContinuousWavelet(wavelet))
    scales = float(fps) / (freqs * central_f)
    frequencies = freqs.astype(np.float32)
    scales_f32 = scales.astype(np.float32)
    frequencies.setflags(write=False)
    scales_f32.setflags(write=False)
    return WaveletBand(frequencies=frequencies, scales=scales_f32)


def wavelet_attrs(
    band: WaveletBand, params: WaveletParams, fps: float, in_cols: list[str]
) -> dict[str, object]:
    """The attributes a wavelet feature attaches to its output frame.

    Args:
        band: The frequencies and scales of the transform.
        params: The feature's wavelet params.
        fps: The frame rate the band was built at.
        in_cols: The input columns, in the order transformed.

    Returns:
        The band, wavelet, frame rate and input columns, keyed by name.
    """
    return {
        "frequencies_hz": band.frequencies.tolist(),
        "scales": band.scales.tolist(),
        "wavelet": str(params.wavelet),
        "fps": float(fps),
        "pc_cols": [c for c in in_cols if c.startswith(params.pc_prefix)],
        "input_columns": list(map(str, in_cols)),
    }


def log_power(
    signal: npt.NDArray[np.float64],
    band: WaveletBand,
    wavelet: str,
    fps: float,
    log_floor: float,
) -> Float32Array:
    """Return the clamped log power of each column of *signal* at each frequency.

    Args:
        signal: The series, one column each, shape ``(n_time, n_columns)``.
        band: The band from :func:`wavelet_band`.
        wavelet: The PyWavelets continuous wavelet name.
        fps: The frame rate in frames per second.
        log_floor: The value the log power is clamped to from below.

    Returns:
        Float32 array of shape ``(n_time, n_columns * n_freq)``. The columns run
        input column major, then frequency, as :func:`wavelet_columns` names them.
    """
    n_time, n_columns = signal.shape
    n_freq = len(band.scales)
    continuous = pywt.ContinuousWavelet(wavelet)
    power = np.empty((n_columns, n_freq, n_time), dtype=np.float32)
    for column in range(n_columns):
        coeffs, _ = pywt.cwt(
            signal[:, column],
            band.scales,
            continuous,
            sampling_period=1.0 / float(fps),
        )
        power[column] = (np.abs(coeffs) ** 2).astype(np.float32)
    tiny = np.finfo(np.float32).tiny
    clamped = np.maximum(np.log(power + tiny), log_floor)
    return clamped.reshape(n_columns * n_freq, n_time).T


def wavelet_columns(in_cols: list[str], n_freq: int) -> list[str]:
    """Return the ``W_{col}_f{fi}`` names, input column major, then frequency."""
    return [f"W_{col}_f{fi}" for col in in_cols for fi in range(n_freq)]


def wavelet_input_columns(
    df: pd.DataFrame, params: WaveletParams, who: str
) -> list[str]:
    """Return the columns of *df* to transform.

    The columns that ``params.cols`` names win. Unset, the ``pc_prefix`` columns
    ``PC0``, ``PC1`` and so on follow, and when none exist every numeric feature
    column.

    Args:
        df: The input table.
        params: The feature's params.
        who: The feature name that opens each error message.

    Returns:
        The column names, in transform order.

    Raises:
        ValueError: If none of ``cols`` is in *df*, or no numeric feature column
            exists.
    """
    if params.cols:
        cols = [c for c in params.cols if c in df.columns]
        if not cols:
            raise ValueError(f"{who} None of the requested 'cols' are present in df.")
        return cols
    pc_cols = _pc_columns(df, params.pc_prefix)
    if pc_cols:
        return pc_cols
    num_cols = feature_columns(df)
    if not num_cols:
        raise ValueError(f"{who} Could not auto-detect numeric feature columns.")
    return num_cols


def constant_fps(df: pd.DataFrame, default: float) -> float:
    """Return the constant ``fps`` column of *df*, or *default* when there is none."""
    if "fps" in df.columns:
        vals = pd.Series(df["fps"]).dropna().unique()
        if len(vals) == 1:
            return float(vals[0])
    return float(default)


def _pc_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    pc_cols: list[str] = []
    i = 0
    while f"{prefix}{i}" in df.columns:
        pc_cols.append(f"{prefix}{i}")
        i += 1
    return pc_cols
