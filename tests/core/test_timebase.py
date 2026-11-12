"""One reading of a track table's time axis, on a table that dropped frames."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mosaic.core.timebase import (
    last_frames_at_or_before,
    local_periods,
    measure_time_axis,
    step_seconds,
    uniform_grid_rate,
)

from tests.helpers import (
    GAP_SECONDS,
    STEP_SECONDS,
    dropped_frame_times,
    dropped_frame_tracks,
    mean_rate,
)


def test_steps_are_the_seconds_between_rows() -> None:
    steps = step_seconds(dropped_frame_times())
    assert np.isnan(steps[0])
    assert steps[1] == pytest.approx(STEP_SECONDS)
    for row, seconds in GAP_SECONDS.items():
        assert steps[row] == pytest.approx(seconds)


def test_a_step_across_two_individuals_is_undefined() -> None:
    time = np.array([0.0, 0.1, 0.2, 0.0, 0.1])
    same = np.array([False, True, True, False, True])
    np.testing.assert_allclose(
        step_seconds(time, same=same), [np.nan, 0.1, 0.1, np.nan, 0.1]
    )


def test_a_repeated_time_has_no_step() -> None:
    assert np.isnan(step_seconds(np.array([0.0, 0.0, 0.1]))[1])


def test_a_longer_lag_reaches_back_that_many_rows() -> None:
    np.testing.assert_allclose(
        step_seconds(np.array([0.0, 0.1, 0.3, 0.6]), lag=2), [np.nan, np.nan, 0.3, 0.5]
    )


def test_the_rate_is_the_frame_span_over_the_time_span() -> None:
    axis = measure_time_axis(dropped_frame_tracks(), "test")
    assert axis.rate == pytest.approx(mean_rate())
    assert axis.rate < 1.0 / STEP_SECONDS


def test_the_largest_step_is_the_longest_gap() -> None:
    axis = measure_time_axis(dropped_frame_tracks(), "test")
    assert axis.largest_step == pytest.approx(1.45)
    assert axis.largest_step_frames == (89, 90)
    assert axis.median_step == pytest.approx(STEP_SECONDS)
    assert axis.irregular


def test_many_individuals_are_one_axis() -> None:
    one = measure_time_axis(dropped_frame_tracks(ids=(0,)), "test")
    many = measure_time_axis(dropped_frame_tracks(ids=(0, 1, 2)), "test")
    assert many.rate == pytest.approx(one.rate)


def test_an_even_table_is_not_irregular(caplog: pytest.LogCaptureFixture) -> None:
    even = pd.DataFrame({"frame": np.arange(100), "time": np.arange(100) / 30.0})
    with caplog.at_level(logging.WARNING, logger="mosaic.core.timebase"):
        assert uniform_grid_rate(even, "test") == pytest.approx(30.0)
    assert caplog.records == []


def test_a_table_that_steps_unevenly_warns_with_the_gap_and_the_remedy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="mosaic.core.timebase"):
        rate = uniform_grid_rate(dropped_frame_tracks(), "individual-wavelet")
    assert rate == pytest.approx(mean_rate())
    [record] = caplog.records
    message = record.getMessage()
    assert "[individual-wavelet]" in message
    assert "1.45 s" in message
    assert "frame 89 to 90" in message
    assert "resample-tracks" in message


def test_a_table_without_time_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match=r"test .*'time'"):
        _ = uniform_grid_rate(pd.DataFrame({"frame": [0, 1]}), "test")


def test_a_time_that_does_not_advance_is_refused() -> None:
    flat = pd.DataFrame({"frame": [0, 1, 2], "time": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="does not advance"):
        _ = uniform_grid_rate(flat, "test")


def test_an_event_maps_to_the_last_frame_at_or_before_its_second() -> None:
    times = dropped_frame_times()
    frames = np.arange(times.size, dtype=float)
    inside_gap = float(times[89] + 0.7)
    at = last_frames_at_or_before(
        frames,
        times,
        np.array([times[10], inside_gap, times[-1] + 1.0, times[0] - 1.0]),
    )
    assert at[0] == 10.0
    assert at[1] == 89.0
    assert at[2] == float(times.size - 1)
    assert np.isnan(at[3])


def test_a_float32_time_does_not_bias_the_measured_rate() -> None:
    """Three hours in, float32 resolution is a millisecond, and the span is exact."""
    frames = np.arange(0, 324_001, 7)
    long_table = pd.DataFrame(
        {"frame": frames, "time": (frames / 30.0).astype(np.float32)}
    )
    assert uniform_grid_rate(long_table, "test") == pytest.approx(30.0, rel=1e-6)


def test_a_table_without_frames_reads_one_row_per_time() -> None:
    table = pd.DataFrame({"time": np.arange(50) * 0.5})
    assert uniform_grid_rate(table, "test") == pytest.approx(2.0)


def test_a_rows_period_is_its_step_to_the_next_row_per_frame() -> None:
    frames = np.array([0.0, 1.0, 2.0, 4.0, 5.0])
    times = np.array([0.0, 0.1, 1.6, 1.8, 1.9])
    np.testing.assert_allclose(
        local_periods(frames, times, fallback=9.0), [0.1, 1.5, 0.1, 0.1, 0.1]
    )


def test_a_row_with_no_step_takes_the_fallback_period() -> None:
    np.testing.assert_allclose(
        local_periods(np.array([3.0]), np.array([1.0]), fallback=0.25), [0.25]
    )
    np.testing.assert_allclose(
        local_periods(np.array([0.0, 1.0]), np.array([1.0, 1.0]), fallback=0.25),
        [0.25, 0.25],
    )
