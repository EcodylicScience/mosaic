"""The frames that receive VAME latents, and the check of a vame output table.

The feature library is imported inside :func:`assert_latents_at_centers`.
Every test module imports ``tests.helpers``, and a module-level import here
would load the feature library into each of them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


def window_centers(sessions: Iterable[range], time_window: int) -> list[int]:
    """Return the center frame of each window of each session, in order.

    A window of *time_window* frames starts at each frame of a session that
    leaves room for it. Its center is ``time_window // 2`` frames after its
    start.
    """
    return [
        first + time_window // 2
        for session in sessions
        for first in range(session.start, session.stop - time_window + 1)
    ]


def assert_latents_at_centers(
    table: pd.DataFrame, zdims: int, centers_by_id: Mapping[int, list[int]]
) -> None:
    """Assert that *table* places a finite float32 latent on each center frame.

    Args:
        table: An output table of the vame feature.
        zdims: The latent dimensions of the model, whose columns end the table.
        centers_by_id: The frames of each individual's rows, in order. Its keys
            are the individuals of the table.
    """
    from mosaic.behavior.feature_library.vame_sessions import latent_columns

    columns = latent_columns(zdims)
    assert list(table.columns[-zdims:]) == columns
    assert (table[columns].dtypes == np.float32).all()
    assert np.isfinite(table[columns].to_numpy()).all()
    assert sorted(table["id"].unique()) == sorted(centers_by_id)
    for individual, centers in centers_by_id.items():
        rows = table[table["id"] == individual]
        assert rows["frame"].tolist() == centers, individual
