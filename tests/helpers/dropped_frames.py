"""A tracks table that dropped frames, with an analytic trajectory.

The rows step by ``1 / 53`` s, except at three rows that follow a gap of 0.4 s,
1.45 s and 0.9 s. ``frame`` counts rows, so a frame-based rate reads the table as
evenly sampled, which it is not. Each individual moves at a constant velocity and
turns at a constant rate, so every velocity a feature reports has one right
answer, whatever the gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STEP_SECONDS = 1.0 / 53.0
"""The seconds between two rows that no gap separates."""

GAP_SECONDS: dict[int, float] = {40: 0.4, 90: 1.45, 140: 0.9}
"""Row index -> seconds from the row before it, where a gap sits."""

VELOCITY = (3.0, 4.0)
"""The constant velocity of every individual, in position units per second."""

SPEED = 5.0
"""The speed that velocity has."""

ANGULAR_VELOCITY = 0.5
"""The constant turn rate of every individual, in radians per second."""

N_ROWS = 200


def dropped_frame_times(n_rows: int = N_ROWS) -> np.ndarray:
    """The time of each row: even steps, with the gaps of :data:`GAP_SECONDS`."""
    steps = np.full(n_rows, STEP_SECONDS)
    steps[0] = 0.0
    for row, seconds in GAP_SECONDS.items():
        if row < n_rows:
            steps[row] = seconds
    return np.cumsum(steps)


def mean_rate(n_rows: int = N_ROWS) -> float:
    """The frames per second over the whole span, which is what a feature measures."""
    times = dropped_frame_times(n_rows)
    return (n_rows - 1) / float(times[-1] - times[0])


def dropped_frame_tracks(
    ids: tuple[int, ...] = (0, 1), n_rows: int = N_ROWS, sequence: str = "s1"
) -> pd.DataFrame:
    """The table: ``frame``, ``time``, ``id``, ``X``, ``Y``, ``ANGLE`` and entry columns.

    Individual ``k`` starts ``10 * k`` units to the right of the origin.
    """
    times = dropped_frame_times(n_rows)
    blocks: list[pd.DataFrame] = []
    for individual in ids:
        blocks.append(
            pd.DataFrame(
                {
                    "frame": np.arange(n_rows, dtype=np.int64),
                    "time": times,
                    "id": np.full(n_rows, individual, dtype=np.int64),
                    "group": [""] * n_rows,
                    "sequence": [sequence] * n_rows,
                    "X": VELOCITY[0] * times + 10.0 * individual,
                    "Y": VELOCITY[1] * times,
                    "ANGLE": ANGULAR_VELOCITY * times,
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)
