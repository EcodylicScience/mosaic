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

A second transform, :func:`morlet_log_amplitude`, follows Berman et al. 2014
("Mapping the stereotyped behaviour of freely moving fruit flies", J. R. Soc.
Interface 11:20140672). A Morlet wavelet of scale ``s`` seconds, with
``s = (omega0 + sqrt(2 + omega0**2)) / (4 * pi * f)`` at frequency ``f``, has the
Fourier factor ``H(w) = pi**-0.25 * sqrt(s) * exp(-(s * w + omega0) ** 2 / 2)`` at
angular frequency ``w``. Each column is padded with zeros and convolved by FFT,
and the log amplitude is
``log(|q| * pi**-0.25 * exp((omega0 - sqrt(omega0**2 + 2)) ** 2 / 4) / sqrt(2 * s))``,
which gives a unit sinusoid at any band frequency one value. It runs in float64
throughout, and its output is float64.

**Methods.** ``method`` chooses between them. ``pywt``, the default, is the
transform above. ``motionmapper`` is the second transform. :func:`wavelet_spectra`,
:func:`wavelet_dtype` and :func:`wavelet_attrs` are the one place the choice is
made, and both wavelet features call them and no transform function directly.

``who`` in the functions below is the name of the feature that opens each error
message.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Literal, Self, override

import numpy as np
import numpy.typing as npt
import pandas as pd
import pywt
from pydantic import Field, model_validator

from mosaic.core.params import Declared, Params, has_fields

from .helpers import feature_columns

__all__ = [
    "MorletBand",
    "WaveletBand",
    "WaveletMethod",
    "WaveletParams",
    "log_power",
    "morlet_band",
    "morlet_log_amplitude",
    "wavelet_attrs",
    "wavelet_band",
    "wavelet_columns",
    "wavelet_dtype",
    "wavelet_input_columns",
    "wavelet_spectra",
]

_F_MIN_DESCRIPTION = "Minimum frequency of the CWT frequency band."

_F_MAX_DESCRIPTION = "Maximum frequency of the CWT frequency band."

_N_FREQ_DESCRIPTION = (
    "Number of frequency bins, dyadically spaced between f_min and f_max."
)

_METHOD_DESCRIPTION = (
    "The transform. pywt is the PyWavelets wavelet named by wavelet, with its "
    "float32 log power. motionmapper is the Morlet wavelet of Berman et al. "
    "2014 with omega0, its normalized amplitude by zero-padded FFT convolution, "
    "and the float64 log amplitude."
)

_OMEGA0_DESCRIPTION = (
    "Central angular frequency of the Morlet wavelet. Required with method "
    "motionmapper and only then. Berman et al. 2014 use 5."
)

_DEFAULT_WAVELET = "cmor1.5-1.0"

_WAVELET_DESCRIPTION = (
    "Name of the PyWavelets continuous wavelet used for the CWT. It applies "
    "under method pywt only."
)

_LOG_FLOOR_DESCRIPTION = (
    "Minimum value the log-power spectrogram is clamped to, or the log "
    "amplitude under method motionmapper."
)

_PC_PREFIX_DESCRIPTION = (
    "Column-name prefix used to auto-detect principal-component input "
    "columns, for example PC0 and PC1."
)

_COLS_DESCRIPTION = (
    "Explicit input column names for the CWT. Unset, columns are "
    "auto-detected via pc_prefix, falling back to every numeric feature "
    "column when none match."
)

type WaveletMethod = Literal["pywt", "motionmapper"]
type Float32Array = npt.NDArray[np.float32]
type Float64Array = npt.NDArray[np.float64]


class WaveletParams(Params):
    """The band, wavelet, floor and input-column params of a wavelet feature."""

    f_min: Annotated[float, Field(gt=0), Declared(_F_MIN_DESCRIPTION, unit="Hz")] = 0.2
    f_max: Annotated[float, Field(gt=0), Declared(_F_MAX_DESCRIPTION, unit="Hz")] = 5.0
    n_freq: Annotated[int, Field(gt=0), Declared(_N_FREQ_DESCRIPTION)] = 25
    wavelet: Annotated[str, Declared(_WAVELET_DESCRIPTION)] = _DEFAULT_WAVELET
    log_floor: Annotated[float, Declared(_LOG_FLOOR_DESCRIPTION)] = -3.0
    pc_prefix: Annotated[str, Declared(_PC_PREFIX_DESCRIPTION)] = "PC"
    cols: Annotated[list[str] | None, Declared(_COLS_DESCRIPTION)] = None
    method: Annotated[WaveletMethod, Declared(_METHOD_DESCRIPTION)] = "pywt"
    omega0: Annotated[
        Annotated[float, Field(gt=0)] | None, Declared(_OMEGA0_DESCRIPTION)
    ] = None

    @model_validator(mode="after")
    def _method_fits_its_params(self) -> Self:
        if has_fields(self, "method", "omega0"):
            if self.method == "motionmapper" and self.omega0 is None:
                msg = (
                    "omega0 is required with method: motionmapper; "
                    "Berman et al. 2014 use 5"
                )
                raise ValueError(msg)
            if self.method == "pywt" and self.omega0 is not None:
                msg = "omega0 applies only with method: motionmapper"
                raise ValueError(msg)
        if (
            has_fields(self, "method", "wavelet")
            and self.method == "motionmapper"
            and self.wavelet != _DEFAULT_WAVELET
        ):
            msg = (
                "wavelet names a PyWavelets wavelet, and method: motionmapper "
                "uses the Morlet wavelet of omega0"
            )
            raise ValueError(msg)
        return self

    @override
    def identity_dump(self) -> dict[str, object]:
        """The hashed params, less what leaves the result as it was.

        ``method`` is left out while it is ``pywt``, ``omega0`` while it is unset,
        and ``wavelet`` under ``motionmapper``, which does not read it. A run that
        sets none of them keeps the identity of a params set without them.
        """
        dumped = super().identity_dump()
        if getattr(self, "method", "pywt") == "pywt":
            _ = dumped.pop("method", None)
        if getattr(self, "omega0", None) is None:
            _ = dumped.pop("omega0", None)
        if getattr(self, "method", "pywt") == "motionmapper":
            _ = dumped.pop("wavelet", None)
        return dumped


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


@dataclass(frozen=True)
class MorletBand:
    """The frequencies of a band in Hz and the Morlet scales that realize them."""

    frequencies: Float64Array
    scales: Float64Array
    omega0: float


@lru_cache(maxsize=32)
def morlet_band(f_min: float, f_max: float, n_freq: int, omega0: float) -> MorletBand:
    """Return the band of *n_freq* frequencies from *f_min* to *f_max*.

    The scale at frequency ``f`` is
    ``(omega0 + sqrt(2 + omega0**2)) / (4 * pi * f)`` seconds, so the band does
    not depend on the frame rate. The result is cached on the arguments, and its
    arrays are float64 and read-only.

    Args:
        f_min: The lowest frequency in Hz.
        f_max: The highest frequency in Hz.
        n_freq: The number of frequencies, spaced equally in log2 frequency.
        omega0: The dimensionless Morlet center frequency.

    Returns:
        The ascending frequencies and their scales.
    """
    frequencies = np.logspace(math.log2(f_min), math.log2(f_max), n_freq, base=2.0)
    scales = (omega0 + math.sqrt(2.0 + omega0**2)) / (4.0 * math.pi * frequencies)
    frequencies.setflags(write=False)
    scales.setflags(write=False)
    return MorletBand(frequencies=frequencies, scales=scales, omega0=omega0)


def wavelet_attrs(
    params: WaveletParams, fps: float, in_cols: list[str]
) -> dict[str, object]:
    """The attributes a wavelet feature attaches to its output frame.

    Args:
        params: The feature's wavelet params.
        fps: The frame rate of the input.
        in_cols: The input columns, in the order transformed.

    Returns:
        The band, wavelet, frame rate and input columns, keyed by name. Under
        ``motionmapper`` the wavelet is ``"morlet"``, the scales are in seconds,
        and ``method`` and ``omega0`` follow the input columns.
    """
    head: dict[str, object]
    tail: dict[str, object] = {}
    if params.method == "motionmapper":
        band = _morlet_band_of(params)
        head = {
            "frequencies_hz": band.frequencies.tolist(),
            "scales": band.scales.tolist(),
            "wavelet": "morlet",
        }
        tail = {"method": params.method, "omega0": band.omega0}
    else:
        pywt_band = _pywt_band_of(params, fps)
        head = {
            "frequencies_hz": pywt_band.frequencies.tolist(),
            "scales": pywt_band.scales.tolist(),
            "wavelet": str(params.wavelet),
        }
    return {
        **head,
        "fps": float(fps),
        "pc_cols": [c for c in in_cols if c.startswith(params.pc_prefix)],
        "input_columns": list(map(str, in_cols)),
        **tail,
    }


def wavelet_dtype(params: WaveletParams) -> type[np.float32] | type[np.float64]:
    """The dtype of the spectra: float32 under ``pywt``, float64 under ``motionmapper``."""
    return np.float64 if params.method == "motionmapper" else np.float32


def wavelet_spectra(
    signal: Float64Array, params: WaveletParams, fps: float
) -> Float32Array | Float64Array:
    """Return the log spectra of each column of *signal* under ``params.method``.

    Args:
        signal: The series, one column each, shape ``(n_time, n_columns)``.
        params: The feature's wavelet params.
        fps: The frame rate in frames per second.

    Returns:
        Array of shape ``(n_time, n_columns * n_freq)`` of dtype
        :func:`wavelet_dtype`, columns as :func:`wavelet_columns` names them.

    Raises:
        ValueError: If *signal* has no rows, under either method.
    """
    if params.method == "motionmapper":
        return morlet_log_amplitude(
            signal, _morlet_band_of(params), fps, params.log_floor
        )
    return log_power(
        signal, _pywt_band_of(params, fps), params.wavelet, fps, params.log_floor
    )


def _pywt_band_of(params: WaveletParams, fps: float) -> WaveletBand:
    return wavelet_band(params.wavelet, params.f_min, params.f_max, params.n_freq, fps)


def _morlet_band_of(params: WaveletParams) -> MorletBand:
    if params.omega0 is None:
        msg = "method: motionmapper needs omega0"
        raise ValueError(msg)
    return morlet_band(params.f_min, params.f_max, params.n_freq, params.omega0)


def log_power(
    signal: Float64Array,
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


def morlet_log_amplitude(
    signal: Float64Array, band: MorletBand, fps: float, log_floor: float
) -> Float64Array:
    """Return the clamped log amplitude of each column of *signal* at each frequency.

    Per column, a NaN sample reads as 0, an odd length gets one zero appended to
    reach an even length ``M``, and ``M // 2`` zeros go before and after, so the
    padded length is ``2 * M``. The padded series is multiplied in the Fourier
    domain by ``H(w) = pi**-0.25 * sqrt(s) * exp(-(s * w + omega0) ** 2 / 2)``
    with ``w = 2 * pi * fftfreq`` in rad/s, one scale at a time, and the middle
    samples are kept. The amplitude is
    ``|q| * pi**-0.25 * exp((omega0 - sqrt(omega0**2 + 2)) ** 2 / 4) / sqrt(2 * s)``
    and the output its natural log, clamped from below at *log_floor*. An
    amplitude of exactly 0 gives *log_floor*. A row within
    ``ceil(8 * s * fps)`` samples of a NaN of its column is NaN, since the
    wavelet envelope is below 1.3e-14 of its peak at 8 scales.

    Args:
        signal: The series, one column each, shape ``(n_time, n_columns)``.
        band: The band from :func:`morlet_band`.
        fps: The frame rate in frames per second.
        log_floor: The value the log amplitude is clamped to from below.

    Returns:
        Float64 array of shape ``(n_time, n_columns * n_freq)``. The columns run
        input column major, then ascending frequency, as :func:`wavelet_columns`
        names them.

    Raises:
        ValueError: If *signal* has no rows, as ``pywt.cwt`` refuses an empty
            series.
    """
    n_time = int(signal.shape[0])
    if n_time == 0:
        msg = "the series is empty"
        raise ValueError(msg)
    n_columns = int(signal.shape[1])
    n_freq = len(band.scales)
    omega0 = band.omega0
    even = n_time + n_time % 2
    start = even // 2
    angular = 2.0 * np.pi * np.fft.fftfreq(2 * even, d=1.0 / float(fps))
    normalization = np.pi**-0.25 * np.exp(
        0.25 * (omega0 - np.sqrt(omega0**2 + 2.0)) ** 2
    )
    out = np.empty((n_time, n_columns * n_freq), dtype=np.float64)
    for column in range(n_columns):
        missing = np.isnan(signal[:, column])
        padded = np.zeros(2 * even, dtype=np.float64)
        padded[start : start + n_time] = np.where(missing, 0.0, signal[:, column])
        spectrum = np.fft.fft(padded)
        nan_count = np.concatenate(([0], np.cumsum(missing)))
        rows = np.arange(n_time, dtype=np.intp)
        for j, scale in enumerate(band.scales):
            response = (
                np.pi**-0.25
                * np.sqrt(scale)
                * np.exp(-0.5 * (scale * angular + omega0) ** 2)
            )
            coefficients = np.fft.ifft(spectrum * response)[start : start + n_time]
            amplitude = np.abs(coefficients) * normalization / np.sqrt(2.0 * scale)
            with np.errstate(divide="ignore"):
                log_amplitude = np.maximum(np.log(amplitude), log_floor)
            if nan_count[-1]:
                reach = math.ceil(8.0 * float(scale) * float(fps))
                near = (
                    nan_count[np.minimum(rows + reach + 1, n_time)]
                    - nan_count[np.maximum(rows - reach, 0)]
                )
                log_amplitude[near > 0] = np.nan
            out[:, column * n_freq + j] = log_amplitude
    return out


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


def _pc_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    pc_cols: list[str] = []
    i = 0
    while f"{prefix}{i}" in df.columns:
        pc_cols.append(f"{prefix}{i}")
        i += 1
    return pc_cols
