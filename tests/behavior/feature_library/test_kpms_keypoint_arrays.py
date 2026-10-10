"""The recordings kpms sends keypoint-MoSeq for one entry, read without the server.

kpms reads every keypoint in a table, in suffix order, whatever ``pose_n`` and
``pose_indices`` say: keypoint-MoSeq narrows to ``pose_indices`` itself, as
positions among those keypoints. Each individual is one recording, named by its
id and ordered by frame.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.kpms import kpms_keypoint_arrays, recording_key
from mosaic.core.individuals import iter_individuals
from mosaic.core.pipeline.types import PoseConfig
from mosaic.core.pose_columns import KeypointArrays
from tests.helpers import keypoint_rows

_ENTRY = "g__s"


def _table(
    n_keypoints: int, *, ids: Sequence[int] | None = None, n_frames: int = 4
) -> pd.DataFrame:
    """Frame-major rows of keypoints ``0..n_keypoints-1``, scored.

    Keypoint ``k`` of individual ``i`` at frame ``f`` is at
    ``(100 i + f + k / 10, -(100 i + f + k / 10))`` and scores ``0.5 + k / 100``.
    """
    scores = [0.5 + k / 100 for k in range(n_keypoints)]
    per_individual = [
        keypoint_rows(
            range(n_frames),
            individual,
            n_keypoints=n_keypoints,
            confidence=scores,
            offset=100.0 * (individual or 0),
        )
        for individual in (ids if ids is not None else [None])
    ]
    return pd.concat(per_individual).sort_values(
        "frame", kind="stable", ignore_index=True
    )


def _expected_position(
    frames: Sequence[int], n_keypoints: int, individual: int | None
) -> np.ndarray:
    base = np.asarray(frames, dtype=np.float64)[:, None] + 100.0 * (individual or 0)
    x = base + np.arange(n_keypoints) / 10
    return np.stack([x, -x], axis=2).astype(np.float32)


def _recordings(df: pd.DataFrame, pose: PoseConfig) -> list[tuple[str, KeypointArrays]]:
    return [
        (recording_key(_ENTRY, individual), kpms_keypoint_arrays(rows, pose))
        for individual, rows in iter_individuals(df)
    ]


def _partial_scores() -> pd.DataFrame:
    return _table(3).drop(columns="poseP1")


def _nan_score() -> pd.DataFrame:
    table = _table(3)
    table.loc[2, "poseP1"] = np.nan
    return table


def _infinite_positions() -> pd.DataFrame:
    table = _table(3)
    table.loc[1, "poseX0"] = np.inf
    table.loc[3, "poseY2"] = -np.inf
    return table


def _ids_out_of_order() -> pd.DataFrame:
    table = _table(3, ids=[5, 2, 9])
    return table.sample(frac=1.0, random_state=0).reset_index(drop=True)


CASES: dict[str, tuple[pd.DataFrame, PoseConfig]] = {
    "more keypoints than pose_n": (_table(9), PoseConfig(pose_n=3)),
    "pose_indices set": (_table(9), PoseConfig(pose_n=9, pose_indices=[1, 4])),
    "poseX10 beside poseX2": (_table(11), PoseConfig(pose_n=11)),
    "partial poseP": (_partial_scores(), PoseConfig(pose_n=3)),
    "NaN poseP": (_nan_score(), PoseConfig(pose_n=3)),
    "infinite positions": (_infinite_positions(), PoseConfig(pose_n=3)),
    "ids out of order": (_ids_out_of_order(), PoseConfig(pose_n=3)),
    "no id column": (_table(3), PoseConfig(pose_n=3)),
}


@pytest.mark.parametrize(
    "case", ["more keypoints than pose_n", "pose_indices set", "poseX10 beside poseX2"]
)
def test_every_keypoint_of_the_table_is_read_in_suffix_order(case: str) -> None:
    df, pose = CASES[case]
    n_keypoints = int(df.columns.str.startswith("poseX").sum())

    [(key, arrays)] = _recordings(df, pose)

    assert key == _ENTRY
    np.testing.assert_array_equal(
        arrays.position, _expected_position(range(4), n_keypoints, None)
    )
    np.testing.assert_allclose(
        arrays.confidence[0], 0.5 + np.arange(n_keypoints) / 100, rtol=1e-6
    )


def test_a_keypoint_without_a_score_leaves_every_point_scored_one() -> None:
    [(_, arrays)] = _recordings(*CASES["partial poseP"])

    assert (arrays.confidence == 1.0).all()


def test_a_nan_score_at_a_finite_point_is_passed_on() -> None:
    [(_, arrays)] = _recordings(*CASES["NaN poseP"])

    assert np.isnan(arrays.confidence[2, 1])
    assert np.isfinite(np.delete(arrays.confidence.ravel(), 2 * 3 + 1)).all()


def test_an_infinite_coordinate_makes_its_point_missing() -> None:
    [(_, arrays)] = _recordings(*CASES["infinite positions"])

    missing = np.zeros((4, 3), dtype=bool)
    missing[1, 0] = missing[3, 2] = True
    assert (np.isnan(arrays.position).all(axis=2) == missing).all()
    assert np.isfinite(arrays.position[~missing]).all()
    assert (arrays.confidence[missing] == 0.0).all()
    assert (arrays.confidence[~missing] > 0.0).all()


def test_each_id_is_a_recording_in_order_of_first_appearance() -> None:
    df, pose = CASES["ids out of order"]
    first_seen = list(dict.fromkeys(df["id"].tolist()))

    recordings = _recordings(df, pose)

    assert [key for key, _ in recordings] == [f"{_ENTRY}__id{i}" for i in first_seen]
    for individual, (_, arrays) in zip(first_seen, recordings):
        np.testing.assert_array_equal(
            arrays.position, _expected_position(range(4), 3, individual)
        )


def test_a_table_without_ids_is_one_recording_named_by_its_entry() -> None:
    [(key, arrays)] = _recordings(*CASES["no id column"])

    assert key == _ENTRY
    np.testing.assert_array_equal(
        arrays.position, _expected_position(range(4), 3, None)
    )


def test_a_table_without_keypoints_is_refused() -> None:
    with pytest.raises(ValueError, match="No keypoint columns found"):
        _ = kpms_keypoint_arrays(pd.DataFrame({"frame": [0, 1]}), PoseConfig())
