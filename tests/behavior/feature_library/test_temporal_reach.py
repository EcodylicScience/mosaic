"""A declared ``temporal_reach`` is the distance at which an input frame stops mattering.

The gap check of a block cross-validated classifier trusts these numbers, so each is
measured here: one input frame is moved a long way, and the farthest output row that
changes is compared with what the feature declares. Where the declaration is exact the
two are equal, and a feature that reached further than it declared would fail.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.individual_wavelet import IndividualWavelet
from mosaic.behavior.feature_library.label_predictions import LabelPredictions
from mosaic.behavior.feature_library.pair_keypoints import PairKeypoints
from mosaic.behavior.feature_library.pair_wavelet import PairWavelet
from mosaic.behavior.feature_library.sequence_heads import (
    TcnConfig,
    TransformerConfig,
    UnetConfig,
    assemble_network,
    get_head,
    network_reach,
)
from mosaic.behavior.feature_library.smooth_probabilities import (
    SmoothProbabilitiesFeature,
)
from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
from mosaic.behavior.feature_library.temporal_stacking import TemporalStackingFeature
from mosaic.behavior.feature_library.trajectory_smooth import TrajectorySmooth
from mosaic.core.pipeline.types import Result
from tests.helpers import write_predictions

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


def _farthest_pair_change(
    apply: Callable[[pd.DataFrame], pd.DataFrame],
    tracks: pd.DataFrame,
    partner_id: int,
    column: str,
    at: int,
) -> int:
    """How far from frame *at* a pair output changes when the partner's input moves.

    The input holds two individuals and the output one row per ordered pair, so
    rows are matched on their key rather than by position.
    """
    moved = tracks.copy()
    row = (moved["id"] == partner_id) & (moved["frame"] == at)
    moved.loc[row, column] = moved.loc[row, column] + BUMP
    key = ["frame", "id1", "id2", "perspective"]
    joined = apply(tracks).merge(apply(moved), on=key, suffixes=("", "__moved"))
    values = [c for c in joined.columns if c not in key and not c.endswith("__moved")]
    numeric = [c for c in values if pd.api.types.is_numeric_dtype(joined[c])]
    before = joined[numeric].to_numpy(dtype=float)
    after = joined[[f"{c}__moved" for c in numeric]].to_numpy(dtype=float)
    changed = (np.abs(after - before) > 0).any(axis=1)
    frames = joined["frame"].to_numpy()[changed]
    return int(np.abs(frames - at).max()) if frames.size else 0


def _two_animals(first: int, partner_gap: range = range(0)) -> pd.DataFrame:
    """Two random walkers with three keypoints from frame *first* to ``N - 1``.

    The partner, id 1, is missing its third keypoint on the frames of *partner_gap*.
    """
    rng = np.random.default_rng(3)
    frames = np.arange(first, N)
    tables: list[pd.DataFrame] = []
    for individual in (0, 1):
        columns: dict[str, object] = {
            "frame": frames,
            "group": "g",
            "sequence": "s",
            "id": individual,
        }
        for k in range(3):
            columns[f"poseX{k}"] = rng.normal(size=frames.size).cumsum()
            columns[f"poseY{k}"] = rng.normal(size=frames.size).cumsum()
        table = pd.DataFrame(columns)
        if individual == 1:
            gap = table["frame"].isin(list(partner_gap))
            table.loc[gap, ["poseX2", "poseY2"]] = np.nan
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


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
            "time": np.arange(N) / 30.0,
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
    ],
)
def test_a_stack_declares_the_distance_it_reads(params: dict[str, object]) -> None:
    feature = TemporalStackingFeature(
        TemporalStackingFeature.Inputs((Result(feature="x"),)), params=params
    )

    measured = _farthest_change(feature.apply, _payload(), "f0")

    assert measured == feature.temporal_reach(feature.params)


def test_a_stack_with_pooled_statistics_declares_no_reach() -> None:
    """The pooling window is in seconds, so its frames follow the table's rate."""
    pooled = TemporalStackingFeature.Params(sigma_pool=10, win_sec=2)

    assert TemporalStackingFeature.temporal_reach(pooled) is None
    assert (
        TemporalStackingFeature.temporal_reach(TemporalStackingFeature.Params()) is None
    )


def _probabilities() -> pd.DataFrame:
    """Three class probabilities per frame, the moved one the lowest of its column.

    A median moves with an input only where the input crosses it, so the frame
    that is moved starts below every other value of its column.
    """
    probs = np.random.default_rng(2).dirichlet(np.ones(3), size=N) + 0.01
    probs[AT, 1] = 0.0
    table = pd.DataFrame({"frame": np.arange(N), "group": "g", "sequence": "s"})
    for column in range(3):
        table[f"prob_{column}"] = probs[:, column]
    return table


@pytest.mark.parametrize(
    "params",
    [
        {"sigma": 2.0},
        {"sigma": 5.5},
        {"method": "median", "window": 5},
        {"method": "median", "window": 11},
    ],
)
def test_a_probability_smoothing_declares_the_distance_it_reads(
    params: dict[str, object],
) -> None:
    feature = SmoothProbabilitiesFeature(
        SmoothProbabilitiesFeature.Inputs((Result(feature="x"),)),
        params={"default_class": 0, **params},
    )
    _ = feature.load_state(Path("unused"), {}, {})

    measured = _farthest_change(feature.apply, _probabilities(), "prob_1")

    assert measured == feature.temporal_reach(feature.params)


def test_a_probability_smoothing_that_merges_bouts_declares_no_reach() -> None:
    params = SmoothProbabilitiesFeature.Params(default_class=0, min_bout_frames=3)

    assert SmoothProbabilitiesFeature.temporal_reach(params) is None


def test_a_bridged_prediction_reaches_only_its_own_frame(tmp_path: Path) -> None:
    """Moving the prediction of frame ``AT`` changes the row of that frame alone."""
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 2, size=N)
    first = rng.uniform(size=N)
    rows = _payload()[["frame", "group", "sequence"]]

    def applied(moved: bool) -> np.ndarray:
        predicted = labels.copy()
        probability = first.copy()
        if moved:
            predicted[AT] = 1 - predicted[AT]
            probability[AT] = 1.0 - probability[AT]
        path = write_predictions(
            tmp_path / f"{moved}.npz",
            [(frame, -1, -1, int(predicted[frame])) for frame in range(N)],
            probabilities=np.column_stack([probability, 1.0 - probability]).tolist(),
        )
        feature = LabelPredictions()
        _ = feature.load_state(
            tmp_path / "run", {}, {"predictions": {("g", "s"): path}}
        )
        out = feature.apply(rows)
        return out[["predicted_label", "prob_0", "prob_1"]].to_numpy(dtype=float)

    changed = np.flatnonzero((applied(False) != applied(True)).any(axis=1))

    assert changed.tolist() == [AT]
    assert LabelPredictions.temporal_reach == 0


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


def test_a_wavelet_declares_no_reach_because_the_time_column_sets_the_frame_rate() -> (
    None
):
    wavelet = PairWavelet(
        PairWavelet.Inputs((Result(feature="pair-posedistance-pca"),)),
        params={"f_min": 2.0, "f_max": 5.0, "n_freq": 4},
    )
    rng = np.random.default_rng(2)
    pairs = pd.DataFrame(
        {
            "frame": np.arange(N),
            "time": np.arange(N) / 30.0,
            "perspective": 0,
            "id1": 0,
            "id2": 1,
            "group": "g",
            "sequence": "s",
            "PC0": rng.normal(size=N),
            "PC1": rng.normal(size=N),
        }
    )

    at_30 = _farthest_change(wavelet.apply, pairs, "PC0")
    at_60 = _farthest_change(
        wavelet.apply, pairs.assign(time=np.arange(N) / 60.0), "PC0"
    )

    # The reach follows the frame rate that ``time`` gives, so no declaration from
    # one rate holds for another.
    assert at_30 == 120
    assert at_60 > at_30
    assert wavelet.temporal_reach(wavelet.params) is None


def test_an_individual_wavelet_declares_no_reach_as_a_pair_wavelet_does() -> None:
    wavelet = IndividualWavelet(
        IndividualWavelet.Inputs((Result(feature="keypoint-kinematics"),)),
        params={"f_min": 2.0, "f_max": 5.0, "n_freq": 4},
    )
    rng = np.random.default_rng(2)
    tracks = pd.DataFrame(
        {
            "frame": np.arange(N),
            "time": np.arange(N) / 30.0,
            "id": 0,
            "group": "g",
            "sequence": "s",
            "speed": rng.normal(size=N),
        }
    )

    at_30 = _farthest_change(wavelet.apply, tracks, "speed")
    at_60 = _farthest_change(
        wavelet.apply, tracks.assign(time=np.arange(N) / 60.0), "speed"
    )

    assert at_30 == 120
    assert at_60 > at_30
    assert wavelet.temporal_reach(wavelet.params) is None


def test_a_tcn_reaches_half_its_receptive_field() -> None:
    config = TcnConfig(blocks=6, kernel_size=3)

    # The Declared text of ``blocks``: (kernel_size - 1) * (2**blocks - 1) + 1 frames.
    field = (config.kernel_size - 1) * (2**config.blocks - 1) + 1
    assert field == 127
    assert get_head("tcn").temporal_reach(config) == (field - 1) // 2 == 63


def test_a_pair_keypoint_velocity_reaches_half_a_window_on_a_dense_track() -> None:
    w = 7
    feature = PairKeypoints(params={"heading": [0, 1], "fps": 30.0, "window": w})

    measured = _farthest_pair_change(feature.apply, _two_animals(0), 1, "poseX2", AT)

    assert measured == w // 2
    reach = feature.temporal_reach(feature.params)
    assert reach is not None and measured <= reach


def test_a_pair_keypoint_velocity_reaches_its_declared_bound_across_a_gap() -> None:
    """An edge row fits a window that holds a gap bridged to the moved frame."""
    w = 7
    feature = PairKeypoints(params={"heading": [0, 1], "fps": 30.0, "window": w})
    tracks = _two_animals(AT - (2 * w - 1), partner_gap=range(AT - w, AT))

    measured = _farthest_pair_change(feature.apply, tracks, 1, "poseX2", AT)

    assert measured == 2 * w - 1
    assert measured == feature.temporal_reach(feature.params)


@pytest.mark.identity
def test_a_two_stage_tcn_reaches_as_far_as_its_network_reach() -> None:
    torch = pytest.importorskip("torch")
    config = TcnConfig(hidden=8, blocks=2, stages=2)
    _ = torch.manual_seed(0)
    network = assemble_network(
        "tcn", config, input_dropout=0.1, n_features=1, n_classes=3
    ).eval()
    frames = _payload()

    def apply(table: pd.DataFrame) -> pd.DataFrame:
        values = torch.as_tensor(table["f0"].to_numpy(dtype=np.float32))[None, :, None]
        with torch.no_grad():
            logits = network(values)[0].numpy()
        return pd.DataFrame(logits)

    measured = _farthest_change(apply, frames, "f0")

    assert measured == network_reach("tcn", config) == 3 + 63


@pytest.mark.identity
@pytest.mark.parametrize(
    "config",
    [
        TransformerConfig(
            hidden=8, layers=2, attention_heads=2, window=2, kernel_size=1
        ),
        TransformerConfig(
            hidden=8, layers=2, attention_heads=2, window=4, kernel_size=3
        ),
        TransformerConfig(hidden=8, layers=1, attention_heads=2, window=5, stages=2),
    ],
)
def test_a_transformer_reaches_as_far_as_its_network_reach(
    config: TransformerConfig,
) -> None:
    _ = pytest.importorskip("torch")
    import torch

    from mosaic.behavior.feature_library.sequence_training import seed_torch

    seed_torch(0, device="cpu")
    network = assemble_network(
        "transformer", config, input_dropout=0.1, n_features=1, n_classes=3
    ).eval()
    length = 200
    series = torch.as_tensor(
        np.random.default_rng(3).normal(size=length).astype(np.float32)
    )

    def logits(values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return network(values[None, :, None])[0]

    # The block that a frame falls in sets how far it reaches, so the frame moved
    # takes every position of the series.
    before = logits(series)
    measured = 0
    for at in range(length):
        moved = series.clone()
        moved[at] += BUMP
        changed = torch.nonzero((logits(moved) - before).abs().amax(dim=-1) > 0)
        measured = max(measured, int((changed - at).abs().max()))

    reach = network_reach("transformer", config)
    assert reach is not None
    assert reach < length // 2
    assert measured == reach


@pytest.mark.identity
@pytest.mark.parametrize(
    "config",
    [
        UnetConfig(hidden=8, depth=2, kernel_size=3),
        UnetConfig(hidden=8, depth=3, kernel_size=1),
        UnetConfig(hidden=8),
        UnetConfig(hidden=8, depth=2, kernel_size=3, stages=2),
    ],
)
def test_a_unet_reaches_no_further_than_its_network_reach(config: UnetConfig) -> None:
    _ = pytest.importorskip("torch")
    import torch

    from mosaic.behavior.feature_library.sequence_training import seed_torch

    seed_torch(0, device="cpu")
    network = assemble_network(
        "unet", config, input_dropout=0.1, n_features=1, n_classes=3
    ).eval()
    # Not a multiple of 2**depth, so the last frame is repeated into the padding.
    length = 301
    series = torch.as_tensor(
        np.random.default_rng(4).normal(size=length).astype(np.float32)
    )

    def logits(values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return network(values[None, :, None])[0]

    # Where a frame falls among the pooled pairs of each level sets how far it
    # reaches, so the frame moved takes every position of the series.
    before = logits(series)
    measured = 0
    for at in range(length):
        moved = series.clone()
        moved[at] += BUMP
        changed = torch.nonzero((logits(moved) - before).abs().amax(dim=-1) > 0)
        measured = max(measured, int((changed - at).abs().max()))

    reach = network_reach("unet", config)
    assert reach is not None
    assert reach < length // 2
    assert reach - 2**config.depth <= measured <= reach
