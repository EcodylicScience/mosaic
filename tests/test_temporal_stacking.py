"""`temporal-stack` must not lose or mix a pair's perspective.

Two defects, both silent, both reached only through the pair branch of `apply`.

The first is what the output *carries*. `feature_columns` excludes `perspective`
from the measurements, and the passthrough set was spelled inline as
`COLUMNS.meta_set() | {"id1", "id2"}` -- so the column fell between the two and
was dropped. Everything downstream then emitted two rows per
`(group, sequence, frame, id1, id2)` with nothing to tell them apart, and
`load_values` correctly refused to join a classifier's predictions back to the
embedding they were computed from.

The second is worse and was not in the report. `_apply_pairs` grouped on
`["id1", "id2"]` alone. Where both perspectives of a pair share an id -- which is
what `pair-posedistance-pca` wrote before it swapped them -- that put two rows per
frame in one block, and sorting by frame interleaved them. The Gaussian smoothing
and the offset stack then read *across* the two perspectives, so a row's own value
and both of its neighbours came from the wrong one. Nothing raised; the numbers
were simply somebody else's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.temporal_stacking import TemporalStackingFeature
from mosaic.core.pipeline.types import Result
from tests.helpers import make_pair_df

_KEY = ["frame", "id1", "id2", "perspective"]


def _feature(**overrides: object) -> TemporalStackingFeature:
    """A stack with no smoothing and no pooling, so offsets are exact copies."""
    params: dict[str, object] = {
        "half": 1,
        "skip": 1,
        "sigma_stack": 0.0,
        "add_pool": False,
    }
    params.update(overrides)
    return TemporalStackingFeature(
        TemporalStackingFeature.Inputs((Result(feature="pair-wavelet"),)),
        params=params,
    )


def test_apply_keeps_pair_identity() -> None:
    df = make_pair_df(6, 2)

    out = _feature().apply(df)

    assert set(_KEY) <= set(out.columns)
    assert not out.duplicated(subset=_KEY).any()
    assert len(out) == len(df)


def test_a_perspective_stacks_only_its_own_rows() -> None:
    """The neighbours at t-1 and t+1 come from the same perspective.

    `separable=True` puts perspective 0 below 1 and perspective 1 above 100, so a
    value crossing over is unambiguous rather than a numerical near-miss.
    """
    df = make_pair_df(6, 1, separable=True)

    out = _feature().apply(df)

    stacked = [c for c in out.columns if c.startswith("feat_0__t")]
    assert stacked, "the stack wrote no offset columns"
    assert (out.loc[out["perspective"] == 0, stacked].to_numpy() < 1.0).all()
    assert (out.loc[out["perspective"] == 1, stacked].to_numpy() >= 100.0).all()


def test_a_pair_sharing_ids_across_perspectives_still_separates() -> None:
    """`perspective` alone is enough to tell the two time series apart.

    A table written before the ids were swapped carries the same `id1`/`id2` on
    both rows. The grouping must still separate them -- otherwise a stack over an
    archived run reads across the two and says nothing about it.
    """
    df = make_pair_df(6, 1, separable=True)
    df["id1"] = 0
    df["id2"] = 1

    out = _feature().apply(df)

    stacked = [c for c in out.columns if c.startswith("feat_0__t")]
    assert (out.loc[out["perspective"] == 0, stacked].to_numpy() < 1.0).all()
    assert (out.loc[out["perspective"] == 1, stacked].to_numpy() >= 100.0).all()


def test_an_individual_frame_keeps_its_metadata() -> None:
    """The non-pair branch carries the same identity set, minus the pair columns."""
    df = pd.DataFrame(
        {
            "frame": np.arange(5),
            "time": np.arange(5, dtype=float) / 30.0,
            "id": np.zeros(5, dtype=int),
            "group": ["g"] * 5,
            "sequence": ["s"] * 5,
            "feat_0": np.arange(5, dtype=float),
        }
    )

    out = _feature().apply(df)

    assert {"frame", "time", "id", "group", "sequence"} <= set(out.columns)
    assert len(out) == 5


def test_an_empty_frame_returns_empty() -> None:
    assert _feature().apply(pd.DataFrame()).empty


def test_a_frame_with_no_measurements_refuses() -> None:
    df = make_pair_df(4, 0)

    with pytest.raises(ValueError, match="No feature columns"):
        _ = _feature().apply(df)
