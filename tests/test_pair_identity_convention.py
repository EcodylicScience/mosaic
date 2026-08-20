"""One rule for what identifies a pair row, asserted against the producers.

A pair-level feature emits one row per **ordered** pair per frame: ``id1`` is the
focal, ``id2`` the other, and ``perspective`` says which ordering. So the key is
``(frame, id1, id2, perspective)``, and ``(frame, id1, id2)`` alone is one too.

Two producers followed that and one did not. ``pair-position`` and
``pair-egocentric`` swap the ids between perspectives; ``pair-posedistance-pca``
built the mirrored block as ``(b, a)`` but labelled it ``(a, b)``, so ``id1`` meant
the focal on two features and the lower id on the third. Merging any two of them
then keyed on ``{frame, id1, id2}``, found one side unique on it, and bound every
row of the other to the wrong perspective while half its rows matched nothing --
which is what the shipped CalMS21 notebook did to its own embedding.

The features are exercised through ``apply`` rather than checked by inspection:
what the convention is about is the values in the rows, and only running them says
what those are.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.pair_egocentric import PairEgocentricFeatures
from mosaic.behavior.feature_library.pair_position import PairPositionFeatures
from mosaic.behavior.feature_library.pairposedistancepca import PairPoseDistancePCA
from mosaic.core.pipeline.types import InputStream

N_FRAMES = 12
N_KEYPOINTS = 3
IDS = (0, 1)


def _tracks() -> pd.DataFrame:
    """A two-individual pose track table, moving so nothing is degenerate."""
    rows: list[dict[str, object]] = []
    for individual in IDS:
        for frame in range(N_FRAMES):
            row: dict[str, object] = {
                "frame": frame,
                "time": frame / 30.0,
                "id": individual,
                "group": "g",
                "sequence": "s",
                "X": 10.0 * individual + frame,
                "Y": 5.0 * individual + 0.5 * frame,
                "ANGLE": 0.1 * frame + individual,
            }
            for k in range(N_KEYPOINTS):
                row[f"poseX{k}"] = 10.0 * individual + frame + k
                row[f"poseY{k}"] = 5.0 * individual + 0.5 * frame + 2.0 * k
            rows.append(row)
    return pd.DataFrame(rows)


def _pair_position() -> pd.DataFrame:
    # Reads X / Y / ANGLE rather than keypoints, so it takes no pose config.
    return PairPositionFeatures().apply(_tracks())


def _pair_egocentric() -> pd.DataFrame:
    feature = PairEgocentricFeatures(
        params={"neck_idx": 0, "tail_base_idx": 2, "pose": {"pose_n": N_KEYPOINTS}}
    )
    return feature.apply(_tracks())


def _pair_posedistance_pca() -> pd.DataFrame:
    tracks = _tracks()
    feature = PairPoseDistancePCA(
        params={"n_components": 2, "pose": {"pose_n": N_KEYPOINTS}}
    )
    feature.fit(InputStream(lambda: iter([(("g", "s"), tracks)]), n_entries=1))
    return feature.apply(tracks)


PRODUCERS = [
    pytest.param(_pair_position, id="pair-position"),
    pytest.param(_pair_egocentric, id="pair-egocentric"),
    pytest.param(_pair_posedistance_pca, id="pair-posedistance-pca"),
]


@pytest.mark.parametrize("produce", PRODUCERS)
def test_a_pair_row_names_its_perspective(produce) -> None:  # noqa: ANN001
    out = produce()

    assert {"id1", "id2", "perspective"} <= set(out.columns)
    assert set(out["perspective"]) == {0, 1}


@pytest.mark.parametrize("produce", PRODUCERS)
def test_id1_is_the_focal_so_the_ids_swap_with_the_perspective(produce) -> None:  # noqa: ANN001
    """The rule `entity_level_of` states: `id1` the focal, `id2` the target.

    Stated as the ids differing between the perspectives rather than as a literal
    ordering, so it holds however the producer enumerates its pairs.
    """
    out = produce()

    first = out.loc[out["perspective"] == 0, ["id1", "id2"]].drop_duplicates()
    second = out.loc[out["perspective"] == 1, ["id1", "id2"]].drop_duplicates()

    assert len(first) == 1 and len(second) == 1
    assert list(first.iloc[0]) == list(second.iloc[0])[::-1]


@pytest.mark.parametrize("produce", PRODUCERS)
def test_the_ordered_pair_alone_is_a_key(produce) -> None:  # noqa: ANN001
    """`(frame, id1, id2)` names one row, which is what makes a merge one-to-one.

    The multiplicity guard in `_merge_parquet_inputs` allows a join where one side
    is unique on the keys, reading it as a broadcast. A producer duplicating this
    triple therefore does not raise -- it is silently fanned out against whichever
    input got the ordering right.
    """
    out = produce()

    assert not out.duplicated(subset=["frame", "id1", "id2"]).any()


@pytest.mark.parametrize("produce", PRODUCERS)
def test_perspective_is_an_integer(produce) -> None:  # noqa: ANN001
    """It is a join key, so the four producers must agree on its dtype."""
    out = produce()

    assert np.issubdtype(out["perspective"].dtype, np.integer)


def _approach_avoidance() -> pd.DataFrame:
    """`approach-avoidance` mirrors an unordered detection to two ordered rows."""
    from mosaic.behavior.feature_library.approach_avoidance import ApproachAvoidance

    return ApproachAvoidance(
        params={"distance_threshold": 1e6, "min_event_length": 1, "min_event_count": 1}
    ).apply(_tracks())


def _pair_facing() -> pd.DataFrame:
    from mosaic.behavior.feature_library.pair_facing import PairFacing

    return PairFacing(params={"pose_head_index": 0, "pose_abdomen_index": 2}).apply(
        _tracks()
    )


def _orientation_rel() -> pd.DataFrame:
    from mosaic.behavior.feature_library.orientation_relative import (
        OrientationRelativeFeature,
    )

    return OrientationRelativeFeature().apply(_tracks())


def _pair_interaction_filter() -> pd.DataFrame:
    """The symmetric one: its criteria do not distinguish the two individuals.

    It writes both orderings anyway. A lone unordered row matches only half the
    rows of any ordered partner, and the merge reads as a broadcast rather than
    refusing -- so one shape with no special case is what makes any two pair
    results joinable.
    """
    from mosaic.behavior.feature_library.pair_interaction_filter import (
        PairInteractionFilter,
    )

    return PairInteractionFilter(
        params={
            "max_dist": 1e6,
            "require_facing": False,
            "min_run_frames": 1,
            "frame_padding": 0,
            "morphological_structure_size": 0,
            "shift_dist": 0.0,
            "use_pixel_coords": False,
        }
    ).apply(_tracks())


LATE_PRODUCERS = [
    pytest.param(_approach_avoidance, id="approach-avoidance"),
    pytest.param(_pair_facing, id="pair-facing"),
    pytest.param(_orientation_rel, id="orientation-rel"),
    pytest.param(_pair_interaction_filter, id="pair-interaction-filter"),
]


@pytest.mark.parametrize("produce", LATE_PRODUCERS)
def test_every_pair_producer_writes_the_same_shape(produce) -> None:  # noqa: ANN001
    """The four that used four different spellings between them.

    `orientation-rel` and `pair-interaction-filter` wrote `id_a`/`id_b`,
    `pair-facing` wrote `focal_id`/`target_id`, and `approach-avoidance` wrote one
    unordered row with the direction in two payload columns. `entity_level_of`
    reads identity by name, so the `focal_id` pair read as carrying none at all.
    """
    out = produce()

    assert not out.empty, "the fixture produced no rows to assert on"
    assert {"id1", "id2", "perspective"} <= set(out.columns)
    assert set(out["perspective"]) == {0, 1}
    assert not out.duplicated(subset=["frame", "id1", "id2"]).any()

    first = out.loc[out["perspective"] == 0, ["id1", "id2"]].drop_duplicates()
    second = out.loc[out["perspective"] == 1, ["id1", "id2"]].drop_duplicates()
    assert list(first.iloc[0]) == list(second.iloc[0])[::-1]
