"""A declared ``temporal_reach`` is the distance at which an input frame stops mattering.

The gap check of a block cross-validated classifier trusts these numbers, so each is
measured here: one input frame is moved a long way, and the farthest output row that
changes is compared with what the feature declares. Where the declaration is exact the
two are equal, and a feature that reached further than it declared would fail.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.pair_wavelet import PairWavelet
from mosaic.behavior.feature_library.sequence_heads import (
    TcnConfig,
    get_head,
)
from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
from mosaic.behavior.feature_library.temporal_stacking import TemporalStackingFeature
from mosaic.behavior.feature_library.trajectory_smooth import TrajectorySmooth
from mosaic.core.pipeline.types import Result

N = 800
AT = 400
BUMP = 1000.0


def _farthest_change(
    apply: Callable[[pd.DataFrame], pd.DataFrame], frames: pd.DataFrame, column: str
) -> int:
    """How far from frame ``AT`` the output changes when that frame's input moves."""
    moved = frames.copy()
    values = moved[column].to_numpy(dtype=float).copy()
    values[AT] += BUMP
    moved[column] = values
    before = apply(frames).reset_index(drop=True).select_dtypes("number").to_numpy()
    after = apply(moved).reset_index(drop=True).select_dtypes("number").to_numpy()
    changed = np.flatnonzero((np.abs(after - before) > 0).any(axis=1))
    return int(np.abs(changed - AT).max()) if len(changed) else 0


def _payload() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"frame": np.arange(N), "group": "g", "sequence": "s", "f0": rng.normal(size=N)}
    )


def _track() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "frame": np.arange(N),
            "group": "g",
            "sequence": "s",
            "id": 0,
            "X": rng.normal(size=N).cumsum(),
            "Y": rng.normal(size=N).cumsum(),
        }
    )


@pytest.mark.parametrize(
    "params",
    [
        {"half": 10, "sigma_stack": 0, "add_pool": False},
        {"half": 10, "sigma_stack": 2, "add_pool": False},
        {"half": 5, "sigma_stack": 0, "sigma_pool": 10, "win_sec": 2, "fps": 30},
        {},
    ],
)
def test_a_stack_declares_the_distance_it_reads(params: dict[str, object]) -> None:
    feature = TemporalStackingFeature(
        TemporalStackingFeature.Inputs((Result(feature="x"),)), params=params
    )

    measured = _farthest_change(feature.apply, _payload(), "f0")

    assert measured == feature.temporal_reach(feature.params)


@pytest.mark.parametrize(
    "params",
    [{}, {"step_size": 4}, {"smooth_window": 9}, {"step_size": 2, "smooth_window": 21}],
)
def test_a_speed_declares_the_distance_it_reads(params: dict[str, object]) -> None:
    feature = SpeedAngvel(params=params)

    measured = _farthest_change(feature.apply, _track(), "X")

    assert measured == feature.temporal_reach(feature.params)


@pytest.mark.parametrize(
    "params",
    [
        {"interpolate_centroid": False, "savgol_window": 11},
        {
            "interpolate_centroid": False,
            "savgol_window": 3,
            "speed_threshold": 5.0,
            "expand_frames": 3,
        },
    ],
)
def test_a_trajectory_smooth_declares_the_distance_it_reads(
    params: dict[str, object],
) -> None:
    feature = TrajectorySmooth(params=params)

    reach = feature.temporal_reach(feature.params)
    measured = _farthest_change(feature.apply, _track(), "X")

    assert reach is not None
    # A moved frame that is flagged and filled is not covered by a window alone, so
    # the declaration may be wider than what one frame reaches, never narrower.
    assert measured <= reach


def test_a_wavelet_declares_the_distance_it_reads() -> None:
    wavelet = PairWavelet(
        PairWavelet.Inputs((Result(feature="pair-posedistance-pca"),)),
        params={"f_min": 2.0, "f_max": 5.0, "n_freq": 4},
    )
    rng = np.random.default_rng(2)
    pairs = pd.DataFrame(
        {
            "frame": np.arange(N),
            "perspective": 0,
            "id1": 0,
            "id2": 1,
            "group": "g",
            "sequence": "s",
            "PC0": rng.normal(size=N),
            "PC1": rng.normal(size=N),
        }
    )

    measured = _farthest_change(wavelet.apply, pairs, "PC0")

    assert measured == wavelet.temporal_reach(wavelet.params) == 120


def test_a_tcn_reaches_half_its_receptive_field() -> None:
    config = TcnConfig(blocks=6, kernel_size=3)

    # The Declared text of ``blocks``: (kernel_size - 1) * (2**blocks - 1) + 1 frames.
    field = (config.kernel_size - 1) * (2**config.blocks - 1) + 1
    assert field == 127
    assert get_head("tcn").temporal_reach(config) == (field - 1) // 2 == 63
