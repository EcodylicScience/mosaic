"""Pair features relate only individuals of one arena, and read a table without arenas as before.

Four fish in two tanks: ids 0 and 1 in ``A1``, ids 2 and 3 in ``B1``. Every
feature that forms pairs itself is run on the table with ``arena`` and without
it. With arenas it pairs only within a tank. A table whose individuals share one
arena gives the output of the table without ``arena``.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from mosaic.core.entry import make_entry_key
from mosaic.core.pipeline.types import InputStream

N_FRAMES = 12
N_KEYPOINTS = 3
IDS = (0, 1, 2, 3)
ARENA_OF = {0: "A1", 1: "A1", 2: "B1", 3: "B1"}


def _tracks(arenas: dict[int, str] | None) -> pd.DataFrame:
    """Four moving fish, two to a tank, with ``arena`` when *arenas* is given."""
    rows: list[dict[str, object]] = []
    for individual in IDS:
        for frame in range(N_FRAMES):
            x = 10.0 * individual + frame
            y = 5.0 * individual + 0.5 * frame
            row: dict[str, object] = {
                "frame": frame,
                "time": frame / 30.0,
                "id": individual,
                "group": "g",
                "sequence": "s",
                "X": x,
                "Y": y,
                "ANGLE": 0.1 * frame + individual,
                "speed": 1.0 + 0.1 * individual,
            }
            for k in range(N_KEYPOINTS):
                row[f"poseX{k}"] = x + k
                row[f"poseY{k}"] = y + 2.0 * k
            if arenas is not None:
                row["arena"] = arenas[individual]
            rows.append(row)
    return pd.DataFrame(rows)


def _pair_position(df: pd.DataFrame) -> pd.DataFrame:
    from mosaic.behavior.feature_library.pair_position import PairPositionFeatures

    return PairPositionFeatures().apply(df)


def _pair_egocentric(df: pd.DataFrame) -> pd.DataFrame:
    from mosaic.behavior.feature_library.pair_egocentric import PairEgocentricFeatures

    return PairEgocentricFeatures(
        params={"neck_idx": 0, "tail_base_idx": 2, "pose": {"pose_n": N_KEYPOINTS}}
    ).apply(df)


def _pair_posedistance_pca(df: pd.DataFrame) -> pd.DataFrame:
    from mosaic.behavior.feature_library.pairposedistancepca import PairPoseDistancePCA

    feature = PairPoseDistancePCA(
        params={"n_components": 2, "pose": {"pose_n": N_KEYPOINTS}}
    )
    feature.fit(
        InputStream(lambda: iter([(make_entry_key("g", "s"), df)]), n_entries=1)
    )
    return feature.apply(df)


def _approach_avoidance(df: pd.DataFrame) -> pd.DataFrame:
    from mosaic.behavior.feature_library.approach_avoidance import ApproachAvoidance

    return ApproachAvoidance(
        params={"distance_threshold": 1e6, "min_event_length": 1, "min_event_count": 1}
    ).apply(df)


def _pair_facing(df: pd.DataFrame) -> pd.DataFrame:
    from mosaic.behavior.feature_library.pair_facing import PairFacing

    return PairFacing(params={"pose_head_index": 0, "pose_abdomen_index": 2}).apply(df)


def _orientation_rel(df: pd.DataFrame) -> pd.DataFrame:
    from mosaic.behavior.feature_library.orientation_relative import (
        OrientationRelativeFeature,
    )

    return OrientationRelativeFeature().apply(df)


def _pair_interaction_filter(df: pd.DataFrame) -> pd.DataFrame:
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
    ).apply(df)


def _neighbor_delta_response(df: pd.DataFrame) -> pd.DataFrame:
    from mosaic.behavior.feature_library.neighbor_delta_response import (
        NeighborDeltaResponse,
    )

    return NeighborDeltaResponse(params={"speed_col": "speed"}).apply(df)


PRODUCERS = [
    pytest.param(_pair_position, id="pair-position"),
    pytest.param(_pair_egocentric, id="pair-egocentric"),
    pytest.param(_pair_posedistance_pca, id="pair-posedistance-pca"),
    pytest.param(_approach_avoidance, id="approach-avoidance"),
    pytest.param(_pair_facing, id="pair-facing"),
    pytest.param(_orientation_rel, id="orientation-rel"),
    pytest.param(_pair_interaction_filter, id="pair-interaction-filter"),
    pytest.param(_neighbor_delta_response, id="neighbor-delta-response"),
]


def _pairs(out: pd.DataFrame) -> set[frozenset[int]]:
    return {
        frozenset((int(a), int(b))) for a, b in zip(out["id1"], out["id2"], strict=True)
    }


@pytest.mark.parametrize("produce", PRODUCERS)
def test_a_pair_is_formed_only_within_one_arena(
    produce: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    every = _pairs(produce(_tracks(None)))
    within = _pairs(produce(_tracks(ARENA_OF)))

    assert len(every) == 6
    assert within == {frozenset((0, 1)), frozenset((2, 3))}


@pytest.mark.parametrize("produce", PRODUCERS)
def test_one_arena_gives_the_output_of_a_table_without_arenas(
    produce: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    shared = produce(_tracks(dict.fromkeys(IDS, "A1")))
    bare = produce(_tracks(None))

    pd.testing.assert_frame_equal(
        shared.drop(columns="arena", errors="ignore"), bare, check_like=True
    )
