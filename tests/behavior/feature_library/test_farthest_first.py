"""Greedy farthest-point selection over rows that may contain non-finite values."""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.behavior.feature_library.helpers import farthest_first

_NAN = np.nan


def test_the_first_row_has_a_finite_value() -> None:
    """The first row drawn has a finite value whenever a row of the pool has one.

    A row with no finite value is at distance 0 from every row. Drawn first, it
    left every distance at 0, and argmax then chose row 0 for each later pick.
    Seed 1 draws row 2 from all six rows.
    """
    pool = np.array(
        [
            [_NAN, _NAN],
            [0.0, 0.0],
            [_NAN, _NAN],
            [3.0, 4.0],
            [_NAN, _NAN],
            [-5.0, 1.0],
        ]
    )
    seed = 1
    unrestricted_first = int(np.random.default_rng(seed).integers(0, len(pool)))
    assert not np.isfinite(pool[unrestricted_first]).any()

    _, indices = farthest_first(pool, 3, np.random.default_rng(seed))

    assert sorted(indices.tolist()) == [1, 3, 5]


@pytest.mark.parametrize(
    "pool",
    [
        pytest.param(np.ones((3, 2)), id="rows-equal-in-every-column"),
        pytest.param(
            np.array([[1.0, _NAN, _NAN], [_NAN, 1.0, _NAN], [_NAN, _NAN, 1.0]]),
            id="rows-sharing-no-finite-column",
        ),
    ],
)
def test_no_row_is_selected_twice(pool: np.ndarray) -> None:
    """Every row is at distance 0 from every other, and the selection takes each once.

    argmax over distances that are all 0 returned row 0, already selected or
    not, for each pick after the first.
    """
    selected, indices = farthest_first(pool, 3, np.random.default_rng(0))

    assert sorted(indices.tolist()) == [0, 1, 2]
    np.testing.assert_array_equal(selected, pool[indices])
