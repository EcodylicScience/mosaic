"""The symmetric pair columns that features declare hold on the rows they write.

A ``sum`` or ``count`` that a plan resolves to ``once`` reads the perspective-0
row of each pair. That is the whole pair only when the feature writes both
perspective rows of every pair on every frame (``Provides.mirrored``) and the
column holds one value on the two (``Provides.symmetric``). Each declaration is
checked here on a small synthetic table. One directional column of each feature
differs between the two rows, so the check would see a swap.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.approach_avoidance import ApproachAvoidance
from mosaic.behavior.feature_library.orientation_relative import (
    OrientationRelativeFeature,
)
from mosaic.behavior.feature_library.pair_egocentric import PairEgocentricFeatures
from mosaic.behavior.feature_library.pair_facing import PairFacing
from mosaic.behavior.feature_library.pair_interaction_filter import (
    PairInteractionFilter,
)
from mosaic.behavior.feature_library.pair_keypoints import PairKeypoints
from mosaic.behavior.feature_library.pair_position import PairPositionFeatures
from mosaic.behavior.feature_library.pair_world import PairWorld
from mosaic.core.pipeline.column_contract import Column, declared_provides
from tests.helpers import mirrored_rows

_FRAMES = 40
_KEYPOINTS = 7


def _row(
    frame: int, animal: int, x: float, y: float, angle: float
) -> dict[str, object]:
    """One track row, its keypoints laid along the heading through the centre."""
    row: dict[str, object] = {
        "frame": frame,
        "time": frame / 30.0,
        "id": animal,
        "X": x,
        "Y": y,
        "ANGLE": angle,
        "sequence": "s1",
        "group": "g1",
        "fps": 30.0,
    }
    for keypoint in range(_KEYPOINTS):
        offset = keypoint - _KEYPOINTS // 2
        row[f"poseX{keypoint}"] = x - offset * np.cos(angle)
        row[f"poseY{keypoint}"] = y - offset * np.sin(angle)
    return row


def _moving() -> pd.DataFrame:
    """Two animals moving along x one behind the other, and a third circling.

    Animal 0 approaches animal 1 while 1 moves away from it, so the approach is
    directed. Animal 2 circles far from both at another speed, so a focal's
    speed differs from its partner's.
    """
    rows: list[dict[str, object]] = []
    for frame in range(_FRAMES):
        rows.append(_row(frame, 0, 6.0 * frame, 0.0, 0.0))
        rows.append(_row(frame, 1, 50.0 + 6.0 * frame, 0.0, 0.0))
        phase = frame / 5.0
        rows.append(
            _row(
                frame,
                2,
                300.0 + 20.0 * np.cos(phase),
                20.0 * np.sin(phase),
                phase + np.pi / 2,
            )
        )
    return pd.DataFrame(rows)


def _facing() -> pd.DataFrame:
    """Animals 0 and 1 face each other at rest, and animal 2 is far from both."""
    rows: list[dict[str, object]] = []
    for frame in range(_FRAMES):
        rows.append(_row(frame, 0, 0.0, 0.0, 0.0))
        rows.append(_row(frame, 1, 20.0, 0.0, np.pi))
        rows.append(_row(frame, 2, 500.0, 500.0, 0.0))
    return pd.DataFrame(rows)


def _moving_world() -> pd.DataFrame:
    """:func:`_moving` lifted into 3D, each animal at its own rising height.

    Every keypoint of a row sits at the row's height, so the heading stays
    level, and the animals' heights differ, so ``dz`` changes sign between
    the two rows of a pair.
    """
    table = _moving()
    height = 0.5 * table["id"].to_numpy(dtype=np.float64) + 0.01 * table["frame"]
    table["Z"] = height
    for keypoint in range(_KEYPOINTS):
        table[f"poseZ{keypoint}"] = height
    return table


class _PairFeature(Protocol):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(frozen=True)
class _Case:
    """One feature that declares mirrored rows, the table it reads, and its params.

    Attributes:
        feature: The feature class.
        table: Builds the tracks the feature reads.
        directional: A column whose two rows of a pair differ on the table,
            empty for a feature that declares no symmetric column.
        params: The params the feature is built with.
    """

    feature: Callable[..., _PairFeature]
    table: Callable[[], pd.DataFrame]
    directional: str = ""
    params: dict[str, object] = field(default_factory=dict[str, object])


_INTERACTION_PARAMS: dict[str, object] = {
    "use_pixel_coords": False,
    "min_run_frames": 5,
    "morphological_structure_size": 0,
    "frame_padding": 0,
}

_CASES: dict[str, _Case] = {
    "approach-avoidance": _Case(
        ApproachAvoidance,
        _moving,
        "aa_event_directed",
        {"min_event_length": 3, "min_event_count": 2},
    ),
    "pair-position": _Case(PairPositionFeatures, _moving, "A_speed"),
    "pair-egocentric": _Case(PairEgocentricFeatures, _moving, "A_speed"),
    "pair-facing": _Case(PairFacing, _moving, "bearing_deg"),
    "pair-interaction-filter": _Case(
        PairInteractionFilter, _facing, "id1", _INTERACTION_PARAMS
    ),
    "pair-keypoints": _Case(PairKeypoints, _moving, params={"heading": [3, 6]}),
    "pair-world": _Case(PairWorld, _moving_world, "dz", {"front": 0, "rear": 6}),
}
"""Every feature declaring ``mirrored`` that runs here. ``hidra`` runs an external
tool, and ``test_hidra_feature.py`` checks its rows against a fake of it.
``pair-series-correlation`` writes one row per pair per sequence, with no ``frame``
for ``mirrored_rows`` to key on, and ``test_pair_series_correlation.py`` checks its
two perspectives."""

_SYMMETRIC = sorted(slug for slug, case in _CASES.items() if case.directional)


def _with_gaps(table: pd.DataFrame) -> pd.DataFrame:
    """*table* with animal 1 missing frames 10 to 14 and animal 0 frame 30."""
    missing = ((table["id"] == 1) & table["frame"].between(10, 14)) | (
        (table["id"] == 0) & (table["frame"] == 30)
    )
    return table[~missing].reset_index(drop=True)


def _apply(slug: str, *, gaps: bool = False) -> pd.DataFrame:
    case = _CASES[slug]
    table = case.table()
    out = case.feature(params=case.params).apply(_with_gaps(table) if gaps else table)
    assert not out.empty
    return out


def _symmetric_columns(slug: str) -> list[str]:
    case = _CASES[slug]
    provides = declared_provides(type(case.feature(params=case.params)))
    assert provides is not None
    return [ref.name for ref in provides.symmetric if isinstance(ref, Column)]


@pytest.mark.parametrize("gaps", [False, True], ids=["every-frame", "missed-frames"])
@pytest.mark.parametrize("slug", sorted(_CASES))
def test_each_pair_holds_both_perspective_rows(slug: str, gaps: bool) -> None:
    """Both rows of a pair are written, also when an animal misses frames."""
    _ = mirrored_rows(_apply(slug, gaps=gaps))


@pytest.mark.parametrize("slug", _SYMMETRIC)
def test_a_symmetric_column_holds_one_value_on_both_rows(slug: str) -> None:
    first, second = mirrored_rows(_apply(slug))

    columns = _symmetric_columns(slug)
    assert columns
    for column in columns:
        np.testing.assert_array_equal(
            first[column].to_numpy(), second[column].to_numpy(), err_msg=column
        )


@pytest.mark.parametrize("slug", _SYMMETRIC)
def test_a_directional_column_differs_between_the_rows(slug: str) -> None:
    first, second = mirrored_rows(_apply(slug))

    directional = _CASES[slug].directional
    assert directional not in _symmetric_columns(slug)
    assert (first[directional].to_numpy() != second[directional].to_numpy()).any()


def test_orientation_rel_drops_a_mirror_row_without_a_scale() -> None:
    """A focal whose keypoints coincide has no scale, and measures no one.

    Without a body-scale run each focal is scaled by its median keypoint
    distance, which is 0 here. Its rows as the focal are dropped, and its
    partners' rows toward it stay, so the pair keeps one perspective row. (A
    focal with fewer than two keypoints is dropped as the other too.)
    """
    table = _moving()
    collapsed = table["id"] == 2
    for keypoint in range(_KEYPOINTS):
        table.loc[collapsed, f"poseX{keypoint}"] = table.loc[collapsed, "X"]
        table.loc[collapsed, f"poseY{keypoint}"] = table.loc[collapsed, "Y"]

    out = OrientationRelativeFeature().apply(table)

    assert not (out["id1"] == 2).any()
    assert (out["id2"] == 2).any()
