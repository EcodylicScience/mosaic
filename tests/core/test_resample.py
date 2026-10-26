"""The resampling leaf behind the sequence bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.core import resample
from mosaic.core.resample import (
    check_bootstrap_settings,
    percentile_interval,
    resampled_sums,
)
from tests.helpers import modules_loaded_by


def _counts() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 20, size=(7, 4)).astype(np.int64)


def test_resampled_sums_returns_n_boot_rows_of_the_counts_width() -> None:
    sums = resampled_sums(_counts(), n_boot=25, random_state=1)

    assert sums.shape == (25, 4)


def test_each_resample_draws_as_many_units_as_there_are_rows() -> None:
    counts = np.ones((7, 1), dtype=np.int64)

    sums = resampled_sums(counts, n_boot=10, random_state=1)

    assert (sums == 7).all()


def test_the_same_random_state_draws_the_same_rows() -> None:
    first = resampled_sums(_counts(), n_boot=40, random_state=3)
    again = resampled_sums(_counts(), n_boot=40, random_state=3)
    other = resampled_sums(_counts(), n_boot=40, random_state=4)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_a_draw_of_several_chunks_is_the_chunks_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A draw of several chunks is the chunks' rows in order."""
    counts = _counts()
    monkeypatch.setattr(resample, "_DRAW_CELLS", 7 * 5)

    chunked = resampled_sums(counts, n_boot=12, random_state=2)

    rng = np.random.default_rng(2)
    chances = np.full(7, 1.0 / 7)
    expected = np.concatenate(
        [rng.multinomial(7, chances, size=size) @ counts for size in (5, 5, 2)]
    )
    assert np.array_equal(chunked, expected)


def test_a_percentile_interval_reads_the_tails_of_the_replicates() -> None:
    replicates = np.arange(101, dtype=np.float64)

    interval = percentile_interval(50.5, replicates, 0.9)

    assert interval["estimate"] == 50.5
    assert interval["lower"] == pytest.approx(5.0)
    assert interval["upper"] == pytest.approx(95.0)


@pytest.mark.parametrize(
    ("n_boot", "confidence", "message"),
    [
        (0, 0.95, "n_boot must be at least 1, not 0"),
        (1, 0.0, "confidence must lie between 0 and 1, not 0.0"),
        (1, 1.0, "confidence must lie between 0 and 1, not 1.0"),
    ],
)
def test_bootstrap_settings_outside_their_range_are_refused(
    n_boot: int, confidence: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        check_bootstrap_settings(n_boot, confidence)


def test_one_resample_at_an_open_confidence_is_accepted() -> None:
    check_bootstrap_settings(1, 0.5)


def test_the_leaf_loads_no_pandas() -> None:
    assert modules_loaded_by("import mosaic.core.resample", ["pandas"]) == []
