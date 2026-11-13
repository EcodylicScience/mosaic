"""Tests for the PairSeriesCorrelation feature: one row per pair, correlating a column."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.pair_series_correlation import (
    PairSeriesCorrelation,
)

GROUP = "G"
SEQUENCE = "S1"
RATE = 20.0
N_FRAMES = 200


def _tracks(
    series: Mapping[int, np.ndarray],
    *,
    arenas: Mapping[int, str] | None = None,
    frames: np.ndarray | None = None,
    times: np.ndarray | None = None,
) -> pd.DataFrame:
    """One block of rows per id, with ``ANGLE`` taken from *series*, in radians."""
    blocks: list[pd.DataFrame] = []
    for identity, values in series.items():
        frame = np.arange(len(values)) if frames is None else frames
        block = pd.DataFrame(
            {
                "group": GROUP,
                "sequence": SEQUENCE,
                "id": identity,
                "frame": frame,
                "time": frame / RATE if times is None else times,
                "ANGLE": values,
            }
        )
        if arenas is not None:
            block["arena"] = arenas[identity]
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def _angles(seed: int, count: int = N_FRAMES) -> np.ndarray:
    """Angles in radians, in [-pi, pi)."""
    return np.random.default_rng(seed).uniform(-np.pi, np.pi, count)


def _apply(df: pd.DataFrame, **overrides: object) -> pd.DataFrame:
    return PairSeriesCorrelation(params=overrides).apply(df)


def _row(out: pd.DataFrame, id1: int, id2: int) -> pd.Series[float]:
    chosen = out[(out["id1"] == id1) & (out["id2"] == id2)]
    assert len(chosen) == 1
    return chosen.iloc[0]


def _fisher_lee_brute_force(first: np.ndarray, second: np.ndarray) -> float:
    """The Fisher-Lee correlation as the sum over every pair of samples."""
    numerator = 0.0
    spread_first = 0.0
    spread_second = 0.0
    for i in range(len(first)):
        for j in range(i + 1, len(first)):
            sine_first = math.sin(first[i] - first[j])
            sine_second = math.sin(second[i] - second[j])
            numerator += sine_first * sine_second
            spread_first += sine_first**2
            spread_second += sine_second**2
    return numerator / math.sqrt(spread_first * spread_second)


def test_pearson_equals_corrcoef_of_wrapped_degrees() -> None:
    first, second = _angles(1), 0.5 * _angles(1) + 0.5 * _angles(2)
    out = _apply(_tracks({0: first, 1: second}))
    degrees = [
        (np.degrees(series) + 180.0) % 360.0 - 180.0 for series in (first, second)
    ]
    expected = float(np.corrcoef(degrees[0], degrees[1])[0, 1])
    assert _row(out, 0, 1)["correlation"] == pytest.approx(expected, abs=1e-12)


def test_circular_equals_the_sum_over_pairs_of_samples() -> None:
    first = _angles(3, 40)
    second = first + 0.3 * _angles(4, 40)
    out = _apply(_tracks({0: first, 1: second}), method="circular")
    expected = _fisher_lee_brute_force(first, second)
    assert _row(out, 0, 1)["correlation"] == pytest.approx(expected, abs=1e-12)
    assert abs(expected) < 1.0
    assert expected > 0.5


def test_circular_ignores_a_turn_added_to_one_sample() -> None:
    first = _angles(5, 40)
    second = first + 0.3 * _angles(6, 40)
    turned = second.copy()
    turned[7] += 2.0 * np.pi
    plain = _apply(_tracks({0: first, 1: second}), method="circular")
    shifted = _apply(_tracks({0: first, 1: turned}), method="circular")
    assert _row(shifted, 0, 1)["correlation"] == pytest.approx(
        _row(plain, 0, 1)["correlation"], abs=1e-12
    )


def test_a_copy_delayed_by_half_a_second_peaks_at_that_lag() -> None:
    delay_frames = 10
    first = _angles(7)
    second = np.concatenate([_angles(8, delay_frames), first[:-delay_frames]])
    out = _apply(_tracks({0: first, 1: second}), max_lag_s=1.0)
    forward = _row(out, 0, 1)
    backward = _row(out, 1, 0)
    assert forward["perspective"] == 0
    assert backward["perspective"] == 1
    assert forward["peak_lag_s"] == pytest.approx(0.5)
    assert backward["peak_lag_s"] == pytest.approx(-0.5)
    assert forward["peak_correlation"] == pytest.approx(1.0, abs=1e-9)
    assert abs(forward["correlation"]) < 0.5


def test_a_lag_longer_than_the_series_leaves_its_peak_to_the_lags_that_remain() -> None:
    first = _angles(9, 30)
    out = _apply(_tracks({0: first, 1: first}), max_lag_s=10.0, min_samples=10)
    row = _row(out, 0, 1)
    assert row["peak_lag_s"] == 0.0
    assert row["peak_correlation"] == pytest.approx(1.0)
    assert not np.signbit(out["peak_lag_s"].to_numpy(dtype=float)).any()


def test_without_max_lag_the_peak_columns_are_nan() -> None:
    out = _apply(_tracks({0: _angles(10), 1: _angles(11)}))
    assert out["peak_correlation"].isna().all()
    assert out["peak_lag_s"].isna().all()


def test_equal_peaks_take_the_lag_nearest_zero_then_the_negative_one() -> None:
    # Two ids with a period of 2 frames correlate equally at every even lag.
    alternating = np.tile([0.0, 1.0], 50)
    out = _apply(_tracks({0: alternating, 1: alternating}), max_lag_s=0.5)
    assert _row(out, 0, 1)["peak_lag_s"] == 0.0
    # At odd lags the series anti-correlate, so the peak is at even lags only.
    shifted = _apply(_tracks({0: alternating, 1: 1.0 - alternating}), max_lag_s=0.5)
    assert _row(shifted, 0, 1)["peak_lag_s"] == pytest.approx(-1.0 / RATE)


def test_two_arenas_of_two_ids_give_only_pairs_within_an_arena() -> None:
    series = {identity: _angles(20 + identity) for identity in range(4)}
    arenas = {0: "A1", 1: "A1", 2: "B1", 3: "B1"}
    out = _apply(_tracks(series, arenas=arenas))
    assert {(a, b) for a, b in zip(out["id1"], out["id2"], strict=True)} == {
        (0, 1),
        (1, 0),
        (2, 3),
        (3, 2),
    }
    assert dict(zip(out["id1"], out["arena"], strict=True)) == {
        0: "A1",
        1: "A1",
        2: "B1",
        3: "B1",
    }


def test_frames_without_a_value_for_both_ids_are_left_out_of_n_samples() -> None:
    first = _angles(30, 60)
    second = _angles(31, 60)
    second[10] = np.nan
    table = _tracks({0: first, 1: second})
    table = table[~((table["id"] == 1) & (table["frame"] < 5))]
    out = _apply(table)
    assert _row(out, 0, 1)["n_samples"] == 60 - 5 - 1
    assert _row(out, 1, 0)["n_samples"] == 60 - 5 - 1


def test_fewer_than_min_samples_gives_nan() -> None:
    table = _tracks({0: _angles(32, 12), 1: _angles(33, 12)})
    enough = _apply(table, min_samples=12)
    too_few = _apply(table, min_samples=13)
    assert np.isfinite(_row(enough, 0, 1)["correlation"])
    assert math.isnan(_row(too_few, 0, 1)["correlation"])
    assert _row(too_few, 0, 1)["n_samples"] == 12


@pytest.mark.parametrize("method", ["pearson", "circular"])
def test_a_constant_series_gives_nan(method: str) -> None:
    out = _apply(_tracks({0: np.full(50, 0.1), 1: _angles(34, 50)}), method=method)
    assert math.isnan(_row(out, 0, 1)["correlation"])
    assert _row(out, 0, 1)["n_samples"] == 50


def _irregular_tracks() -> pd.DataFrame:
    frames = np.arange(60)
    times = frames / RATE
    times[30:] += 2.0
    return _tracks({0: _angles(35, 60), 1: _angles(36, 60)}, times=times)


def test_an_irregular_time_axis_under_max_lag_logs_the_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    _apply(_irregular_tracks(), max_lag_s=0.5)
    assert "steps unevenly" in caplog.text
    assert "pair-series-correlation" in caplog.text


def test_an_irregular_time_axis_without_max_lag_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    _apply(_irregular_tracks())
    assert "steps unevenly" not in caplog.text


def test_each_pair_writes_both_perspectives_with_the_ids_swapped() -> None:
    series = {identity: _angles(40 + identity) for identity in range(3)}
    out = _apply(_tracks(series), max_lag_s=0.5)
    assert len(out) == 6
    assert list(zip(out["id1"], out["id2"], strict=True)) == sorted(
        (a, b) for a in range(3) for b in range(3) if a != b
    )
    for low, high in ((0, 1), (0, 2), (1, 2)):
        forward, backward = _row(out, low, high), _row(out, high, low)
        assert (forward["perspective"], backward["perspective"]) == (0, 1)
        for column in ("correlation", "n_samples", "peak_correlation"):
            assert forward[column] == backward[column]
        assert forward["peak_lag_s"] == -backward["peak_lag_s"]
    assert (out["group"] == GROUP).all()
    assert (out["sequence"] == SEQUENCE).all()
    assert "arena" not in out.columns


def test_one_individual_gives_an_empty_frame() -> None:
    assert _apply(_tracks({0: _angles(50)})).empty


def test_an_empty_input_gives_an_empty_frame() -> None:
    assert _apply(_tracks({0: _angles(51)}).iloc[0:0]).empty


def test_a_repeated_frame_is_refused() -> None:
    table = _tracks({0: _angles(52, 20), 1: _angles(53, 20)})
    with pytest.raises(ValueError, match="pair-series-correlation"):
        _apply(pd.concat([table, table.iloc[:1]], ignore_index=True))


def test_a_missing_value_column_names_the_column() -> None:
    with pytest.raises(ValueError, match="heading_unwrapped"):
        _apply(
            _tracks({0: _angles(54), 1: _angles(55)}),
            value_column="heading_unwrapped",
        )


@pytest.mark.parametrize(
    "overrides", [{"min_samples": 2}, {"max_lag_s": 0.0}, {"method": "spearman"}]
)
def test_params_refuse_values_out_of_range(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PairSeriesCorrelation.Params.from_overrides(overrides)
