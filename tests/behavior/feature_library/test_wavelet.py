"""The attributes that every wavelet feature attaches to its output."""

from __future__ import annotations

from mosaic.behavior.feature_library.wavelet import (
    WaveletParams,
    wavelet_attrs,
    wavelet_band,
)


def test_the_attributes_describe_the_band_the_wavelet_and_the_inputs() -> None:
    params = WaveletParams(pc_prefix="PC")
    band = wavelet_band(params.wavelet, params.f_min, params.f_max, params.n_freq, 30.0)

    attrs = wavelet_attrs(band, params, 30.0, ["PC0", "speed", "PC1"])

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
