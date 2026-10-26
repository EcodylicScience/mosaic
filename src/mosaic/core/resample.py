"""Bootstrap resampling of additive counts, with numpy alone."""

from __future__ import annotations

from typing import TypedDict

import numpy as np

__all__ = [
    "Interval",
    "check_bootstrap_settings",
    "percentile_interval",
    "resampled_sums",
]

_DRAW_CELLS = 1 << 22
"""How many resample weights :func:`resampled_sums` holds at once."""


class Interval(TypedDict):
    """A score and the percentile interval of its bootstrap."""

    estimate: float
    lower: float
    upper: float


def check_bootstrap_settings(n_boot: int, confidence: float) -> None:
    """Refuse bootstrap settings outside their ranges, before any draw.

    Args:
        n_boot: How many resamples, at least 1.
        confidence: The width of the interval, between 0 and 1 exclusive.

    Raises:
        ValueError: *n_boot* is below 1, or *confidence* is outside (0, 1).
    """
    if n_boot < 1:
        msg = f"n_boot must be at least 1, not {n_boot}"
        raise ValueError(msg)
    if not 0.0 < confidence < 1.0:
        msg = f"confidence must lie between 0 and 1, not {confidence}"
        raise ValueError(msg)


def resampled_sums(counts: np.ndarray, *, n_boot: int, random_state: int) -> np.ndarray:
    """Sum the rows of *counts* over *n_boot* resamples of its rows.

    Each resample draws as many rows as *counts* has, with replacement and equal
    chance, and adds them. The weights are drawn in chunks of at most
    ``_DRAW_CELLS`` cells, so a large *n_boot* does not hold them all.

    Args:
        counts: One row of additive counts per unit, shape ``(units, width)``.
        n_boot: How many resamples.
        random_state: The seed. The same value draws the same rows.

    Returns:
        An array of shape ``(n_boot, width)``, one row of sums per resample.
    """
    n_units = counts.shape[0]
    rng = np.random.default_rng(random_state)
    chances = np.full(n_units, 1.0 / n_units)
    step = max(1, _DRAW_CELLS // n_units)
    return np.concatenate(
        [
            rng.multinomial(n_units, chances, size=min(step, n_boot - start)) @ counts
            for start in range(0, n_boot, step)
        ]
    )


def percentile_interval(
    estimate: float, replicates: np.ndarray, confidence: float
) -> Interval:
    """The percentile interval of *replicates* around *estimate*.

    Args:
        estimate: The score of the unresampled data.
        replicates: The score of each resample.
        confidence: The width of the interval, between 0 and 1 exclusive.

    Returns:
        *estimate* with the ``(1 - confidence) / 2`` and ``(1 + confidence) / 2``
        quantiles of *replicates*.
    """
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(replicates, [tail, 1.0 - tail])
    return {"estimate": estimate, "lower": float(lower), "upper": float(upper)}
