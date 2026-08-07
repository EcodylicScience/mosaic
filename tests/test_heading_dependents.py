"""Features that need a heading say so, and say where to get one.

``ANGLE`` used to arrive in the track table because four converters computed it
inline. It does not any more -- a heading is derived from keypoints, so it
belongs to the ``heading`` feature -- which means every feature that reads it
now meets tables that do not have it.

That has to be a refusal rather than a fallback, and the refusal has to name the
producer. The failure mode it replaces is specific: a per-entry exception becomes
an ``entry_error``, which becomes ``status: "partial"``, which **exits zero**. So
a feature that quietly produced nothing for every entry would look like a
successful run, and the only signal would be an empty output directory.

Three of these four features had no behavior test at all before this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.approach_avoidance import ApproachAvoidance
from mosaic.behavior.feature_library.orientation_relative import (
    OrientationRelativeFeature,
)
from mosaic.behavior.feature_library.pair_interaction_filter import (
    PairInteractionFilter,
)
from mosaic.behavior.feature_library.pair_position import PairPositionFeatures


def _table_without_heading(n: int = 6, n_ids: int = 2) -> pd.DataFrame:
    """A schema-valid ``mosaic_v1`` table: pixels, keypoints, and no ANGLE."""
    total = n * n_ids
    frame = np.tile(np.arange(n, dtype=np.int64), n_ids)
    identity = np.repeat(np.arange(n_ids, dtype=np.int64), n)
    position = np.linspace(0.0, 10.0, total) + identity
    return pd.DataFrame(
        {
            "frame": frame,
            "time": frame / 30.0,
            "id": identity,
            "group": [""] * total,
            "sequence": ["seq"] * total,
            "X": position,
            "Y": position,
            "poseX0": position,
            "poseY0": position + 1.0,
            "poseX1": position + 2.0,
            "poseY1": position + 3.0,
        }
    )


_FEATURES = [
    pytest.param(PairPositionFeatures, id="pair-position"),
    pytest.param(ApproachAvoidance, id="approach-avoidance"),
    pytest.param(OrientationRelativeFeature, id="orientation-rel"),
    pytest.param(PairInteractionFilter, id="pair-interaction-filter"),
]


@pytest.mark.parametrize("feature_class", _FEATURES)
def test_a_feature_needing_a_heading_refuses_without_one(
    feature_class: type,
) -> None:
    with pytest.raises(ValueError, match="ANGLE"):
        _ = feature_class().apply(_table_without_heading())


@pytest.mark.parametrize("feature_class", _FEATURES)
def test_the_refusal_names_the_feature_that_produces_a_heading(
    feature_class: type,
) -> None:
    """Written once, in ``ensure_columns``, so all four say the same thing."""
    with pytest.raises(ValueError, match="'heading' feature"):
        _ = feature_class().apply(_table_without_heading())


def test_the_hint_explains_why_no_tracker_supplies_it() -> None:
    """The distinction that makes the refusal make sense rather than annoy."""
    with pytest.raises(ValueError, match="derived from keypoints"):
        _ = OrientationRelativeFeature().apply(_table_without_heading())


@pytest.mark.parametrize("feature_class", _FEATURES)
def test_the_same_feature_runs_once_a_heading_is_present(
    feature_class: type,
) -> None:
    """The counter-test: the refusal is about the missing column, nothing else.

    Without this, a feature broken for an unrelated reason would still pass the
    refusal tests above and look correctly guarded.
    """
    table = _table_without_heading()
    table["ANGLE"] = np.zeros(len(table))
    result = feature_class().apply(table)
    assert isinstance(result, pd.DataFrame)
