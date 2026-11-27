"""The attributes of the wavelet features and the Morlet log-amplitude transform."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.wavelet import (
    WaveletParams,
    log_power,
    morlet_band,
    morlet_log_amplitude,
    wavelet_attrs,
    wavelet_band,
    wavelet_dtype,
    wavelet_spectra,
)
from tests.helpers import narrowed_params


def test_the_attributes_describe_the_band_the_wavelet_and_the_inputs() -> None:
    params = WaveletParams(pc_prefix="PC")
    band = wavelet_band(params.wavelet, params.f_min, params.f_max, params.n_freq, 30.0)

    attrs = wavelet_attrs(params, 30.0, ["PC0", "speed", "PC1"])

    assert list(attrs) == [
        "frequencies_hz",
        "scales",
        "wavelet",
        "fps",
        "pc_cols",
        "input_columns",
    ]
    assert attrs["frequencies_hz"] == band.frequencies.tolist()
    assert attrs["scales"] == band.scales.tolist()
    assert attrs["wavelet"] == params.wavelet
    assert attrs["fps"] == 30.0
    assert attrs["pc_cols"] == ["PC0", "PC1"]
    assert attrs["input_columns"] == ["PC0", "speed", "PC1"]


def test_the_motionmapper_attributes_hold_the_morlet_band_and_name_the_method() -> None:
    params = WaveletParams(method="motionmapper", omega0=5.0, pc_prefix="PC")
    band = morlet_band(params.f_min, params.f_max, params.n_freq, 5.0)

    attrs = wavelet_attrs(params, 30.0, ["PC0", "speed", "PC1"])

    assert list(attrs) == [
        "frequencies_hz",
        "scales",
        "wavelet",
        "fps",
        "pc_cols",
        "input_columns",
        "method",
        "omega0",
    ]
    assert attrs["frequencies_hz"] == band.frequencies.tolist()
    assert attrs["scales"] == band.scales.tolist()
    assert attrs["wavelet"] == "morlet"
    assert attrs["fps"] == 30.0
    assert attrs["pc_cols"] == ["PC0", "PC1"]
    assert attrs["input_columns"] == ["PC0", "speed", "PC1"]
    assert attrs["method"] == "motionmapper"
    assert attrs["omega0"] == 5.0


# The method and omega0 params.


@pytest.mark.parametrize("params", [WaveletParams(), WaveletParams(method="pywt")])
def test_the_pywt_method_dumps_without_method_and_omega0(params: WaveletParams) -> None:
    dumped = params.identity_dump()

    assert "method" not in dumped
    assert "omega0" not in dumped
    full = WaveletParams().model_dump()
    assert dumped == {k: v for k, v in full.items() if k not in ("method", "omega0")}


def test_the_motionmapper_method_dumps_method_and_omega0_and_no_wavelet() -> None:
    dumped = WaveletParams(method="motionmapper", omega0=5.0).identity_dump()

    assert dumped["method"] == "motionmapper"
    assert dumped["omega0"] == 5.0
    assert "wavelet" not in dumped
    assert dumped["f_min"] == 0.2


def test_the_default_wavelet_beside_motionmapper_is_accepted_and_left_out_of_identity() -> (
    None
):
    plain = WaveletParams(method="motionmapper", omega0=5.0)
    explicit = WaveletParams(method="motionmapper", omega0=5.0, wavelet="cmor1.5-1.0")

    assert explicit.identity_dump() == plain.identity_dump()


def test_a_full_model_dump_rebuilds_a_motionmapper_params() -> None:
    plain = WaveletParams(method="motionmapper", omega0=5.0)

    rebuilt = WaveletParams(**plain.model_dump())

    assert rebuilt.identity_dump() == plain.identity_dump()


def test_motionmapper_without_omega0_is_refused_naming_omega0_and_berman() -> None:
    with pytest.raises(ValueError, match=r"omega0.*motionmapper.*5"):
        _ = WaveletParams(method="motionmapper")


def test_omega0_under_pywt_is_refused_naming_the_method() -> None:
    with pytest.raises(ValueError, match=r"omega0.*method: motionmapper"):
        _ = WaveletParams(omega0=5.0)


def test_another_wavelet_beside_motionmapper_is_refused() -> None:
    with pytest.raises(
        ValueError, match=r"wavelet names a PyWavelets wavelet.*Morlet.*omega0"
    ):
        _ = WaveletParams(method="motionmapper", omega0=5.0, wavelet="cmor2.5-1.0")


def test_an_omega0_of_zero_is_refused() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        _ = WaveletParams(method="motionmapper", omega0=0.0)


@pytest.mark.parametrize(
    ("without", "payload", "dumped"),
    [
        ("omega0", {"method": "motionmapper"}, {"method": "motionmapper"}),
        ("method", {"omega0": 5.0}, {"omega0": 5.0}),
        ("wavelet", {"method": "motionmapper", "omega0": 5.0}, {"omega0": 5.0}),
    ],
)
def test_a_narrowed_model_validates_without_a_verdict_and_dumps_an_identity(
    without: str, payload: dict[str, object], dumped: dict[str, object]
) -> None:
    narrowed = narrowed_params(WaveletParams, {without})

    identity = narrowed.model_validate(payload).identity_dump()

    assert without not in identity
    assert {k: identity[k] for k in dumped} == dumped


def test_a_narrowed_model_dumps_as_the_full_model_does() -> None:
    narrowed = narrowed_params(WaveletParams, {"method", "omega0"}).model_validate({})

    assert narrowed.identity_dump() == WaveletParams().identity_dump()


# The dispatch.


def test_the_pywt_method_dispatches_to_the_float32_log_power() -> None:
    params = WaveletParams(f_min=1.0, f_max=4.0, n_freq=3, log_floor=-2.0)
    signal = np.random.default_rng(1).standard_normal((200, 2))
    band = wavelet_band(params.wavelet, 1.0, 4.0, 3, 30.0)

    out = wavelet_spectra(signal, params, 30.0)

    np.testing.assert_array_equal(
        out, log_power(signal, band, params.wavelet, 30.0, -2.0)
    )
    assert out.dtype == np.float32
    assert wavelet_dtype(params) == np.float32


def test_the_motionmapper_method_dispatches_to_the_float64_log_amplitude() -> None:
    params = WaveletParams(
        method="motionmapper",
        omega0=5.0,
        f_min=1.0,
        f_max=4.0,
        n_freq=3,
        log_floor=-2.0,
    )
    signal = np.random.default_rng(1).standard_normal((200, 2))

    out = wavelet_spectra(signal, params, 30.0)

    np.testing.assert_array_equal(
        out, morlet_log_amplitude(signal, morlet_band(1.0, 4.0, 3, 5.0), 30.0, -2.0)
    )
    assert out.dtype == np.float64
    assert wavelet_dtype(params) == np.float64


@pytest.mark.parametrize(
    "params",
    [WaveletParams(), WaveletParams(method="motionmapper", omega0=5.0)],
    ids=["pywt", "motionmapper"],
)
def test_an_empty_series_is_refused_the_same_way_under_both_methods(
    params: WaveletParams,
) -> None:
    with pytest.raises(ValueError, match="empty"):
        _ = wavelet_spectra(np.empty((0, 2)), params, 30.0)


# The Morlet log-amplitude transform.

FPS = 50.0
OMEGA0 = 5.0


def _flat_log_amplitude(omega0: float) -> float:
    """The log amplitude of a unit cosine at any band frequency."""
    d = omega0 - math.sqrt(omega0**2 + 2)
    return d**2 / 8 - math.log(2 * math.sqrt(2 * math.pi))


def test_the_morlet_band_holds_float64_frequencies_and_scales_from_the_formula() -> (
    None
):
    band = morlet_band(0.5, 20.0, 25, OMEGA0)

    frequencies = np.logspace(math.log2(0.5), math.log2(20.0), 25, base=2.0)
    scales = (OMEGA0 + math.sqrt(2.0 + OMEGA0**2)) / (4.0 * math.pi * frequencies)
    assert band.omega0 == OMEGA0
    assert band.frequencies.dtype == np.float64
    assert band.scales.dtype == np.float64
    assert np.array_equal(band.frequencies, frequencies)
    assert np.array_equal(band.scales, scales)
    assert bool(np.all(np.diff(band.frequencies) > 0))
    assert not band.frequencies.flags.writeable
    assert not band.scales.flags.writeable
    assert morlet_band(0.5, 20.0, 25, OMEGA0) is band


@pytest.mark.parametrize("omega0", [5.0, 6.0])
def test_a_unit_cosine_at_a_band_frequency_has_one_log_amplitude_at_every_frequency(
    omega0: float,
) -> None:
    band = morlet_band(0.5, 5.0, 5, omega0)
    n_time = 4001
    time = np.arange(n_time) / FPS
    signal = np.stack([np.cos(2 * np.pi * f * time) for f in band.frequencies], axis=1)

    out = morlet_log_amplitude(signal, band, FPS, -np.inf)

    margin = math.ceil(8 * band.scales[0] * FPS)
    interior = slice(margin, n_time - margin)
    for j in range(len(band.frequencies)):
        np.testing.assert_allclose(
            out[interior, j * len(band.frequencies) + j],
            _flat_log_amplitude(omega0),
            atol=1e-9,
            rtol=0,
        )


def test_the_fft_transform_equals_a_direct_convolution_with_the_morlet_kernel() -> None:
    """Every row matches, the edges included, at 1e-9.

    The band keeps ``s * fps`` at least 8 samples and the series is longer than
    ``8 s fps``, so neither the aliasing of the sampled kernel nor the wrap of the
    padded transform reaches that tolerance.
    """
    band = morlet_band(2.0, 5.0, 4, OMEGA0)
    n_time = 201
    series = np.random.default_rng(7).standard_normal(n_time)
    assert float(np.min(band.scales)) * FPS >= 8
    assert n_time > 8 * float(np.max(band.scales)) * FPS
    dt = 1.0 / FPS
    lags = (np.arange(n_time)[:, None] - np.arange(n_time)[None, :]) * dt

    out = morlet_log_amplitude(series[:, None], band, FPS, -np.inf)

    for j, s in enumerate(band.scales):
        kernel = (
            np.pi**-0.25
            / np.sqrt(2 * np.pi * s)
            * np.exp(-1j * OMEGA0 * lags / s)
            * np.exp(-(lags**2) / (2 * s**2))
        )
        q = dt * (kernel @ series)
        amplitude = (
            np.abs(q)
            * np.pi**-0.25
            * np.exp(0.25 * (OMEGA0 - np.sqrt(OMEGA0**2 + 2)) ** 2)
            / np.sqrt(2 * s)
        )
        np.testing.assert_allclose(out[:, j], np.log(amplitude), atol=1e-9, rtol=0)


def test_the_output_is_float64() -> None:
    band = morlet_band(0.5, 5.0, 3, OMEGA0)
    signal = np.random.default_rng(0).standard_normal((300, 1))

    assert morlet_log_amplitude(signal, band, FPS, -3.0).dtype == np.float64


def test_a_zero_column_gives_the_floor_and_the_floor_clamps_from_below_only() -> None:
    band = morlet_band(0.5, 5.0, 5, OMEGA0)
    time = np.arange(2001) / FPS
    cosine = np.cos(2 * np.pi * band.frequencies[2] * time)[:, None]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        zeros = morlet_log_amplitude(np.zeros((300, 1)), band, FPS, -4.5)
        unfloored = morlet_log_amplitude(cosine, band, FPS, -np.inf)
        above = morlet_log_amplitude(cosine, band, FPS, -2.0)
        below = morlet_log_amplitude(cosine, band, FPS, -1.0)

    assert np.all(zeros == -4.5)
    flat = _flat_log_amplitude(OMEGA0)
    assert -2.0 < flat < -1.0
    np.testing.assert_array_equal(above, np.maximum(unfloored, -2.0))
    np.testing.assert_array_equal(below, np.maximum(unfloored, -1.0))
    assert above[1000, 2] == pytest.approx(flat, abs=1e-9)
    assert below[1000, 2] == -1.0


def test_a_nan_blanks_eight_scales_on_each_side_in_its_own_column_only() -> None:
    band = morlet_band(0.5, 5.0, 5, OMEGA0)
    rng = np.random.default_rng(3)
    signal = rng.standard_normal((2001, 2))
    center = 1000
    signal[center, 0] = np.nan

    out = morlet_log_amplitude(signal, band, FPS, -5.0)

    assert math.ceil(8 * band.scales[0] * FPS) == 650
    rows = np.arange(2001)
    for j, s in enumerate(band.scales):
        reach = math.ceil(8 * s * FPS)
        blank = (rows >= center - reach) & (rows <= center + reach)
        np.testing.assert_array_equal(np.isnan(out[:, j]), blank)
    assert bool(np.all(np.isfinite(out[:, 5:])))
    assert np.flatnonzero(np.isnan(out[:, 0]))[[0, -1]].tolist() == [350, 1650]


def test_two_columns_give_the_first_columns_frequencies_first() -> None:
    band = morlet_band(1.0, 4.0, 3, OMEGA0)
    signal = np.random.default_rng(5).standard_normal((400, 2))

    both = morlet_log_amplitude(signal, band, FPS, -5.0)
    first = morlet_log_amplitude(signal[:, :1], band, FPS, -5.0)
    second = morlet_log_amplitude(signal[:, 1:], band, FPS, -5.0)

    assert both.shape == (400, 6)
    np.testing.assert_array_equal(both[:, :3], first)
    np.testing.assert_array_equal(both[:, 3:], second)
