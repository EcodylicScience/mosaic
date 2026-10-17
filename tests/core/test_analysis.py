"""``compute_cluster_label_agreement``, the notebook entry point for cluster scoring.

It pooled both perspectives of a pair, read unannotated frames as class 0, cut a
sequence's rows to the shorter of two arrays and guessed a frame source by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from mosaic.core.analysis import compute_cluster_label_agreement
from mosaic.core.dataset import Dataset
from tests.helpers import (
    PairClusterFeature,
    add_labels,
    add_track_sequences,
    make_dataset,
)

STORAGE = "prov-pair-cluster__from__tracks"
N_FRAMES = 4


def _clustered(tmp_path: Path) -> Dataset:
    """One entry of pair rows, labeled resident -> intruder on every frame."""
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, ("g", "s"), n_rows=N_FRAMES, n_ids=2)
    _ = add_labels(ds, "g", "s", [(f, 0, 1, f % 2) for f in range(N_FRAMES)])
    _ = ds.run_feature(PairClusterFeature())
    return ds


def test_compute_cluster_label_agreement_filters_a_perspective(
    tmp_path: Path,
) -> None:
    """The intruder-focal rows are unlabeled, so pooling only drops them."""
    ds = _clustered(tmp_path)

    resident = compute_cluster_label_agreement(ds, STORAGE, perspective=0)
    pooled = compute_cluster_label_agreement(ds, STORAGE)

    assert resident["n_unlabeled_dropped"] == 0
    assert resident["n_frames_total"] == N_FRAMES
    assert pooled["n_unlabeled_dropped"] == N_FRAMES
    metrics = resident["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["purity"] == 1.0
    assert resident["aligned_sequences"] == [
        {
            "group": "g",
            "sequence": "s",
            "sequence_safe": "s",
            "entry_key": "g__s",
            "n_frames": N_FRAMES,
        }
    ]


def test_each_perspective_is_reported(tmp_path: Path) -> None:
    ds = _clustered(tmp_path)

    result = compute_cluster_label_agreement(ds, STORAGE, perspective="each")

    by_perspective = cast(
        dict[int, dict[str, object]], result["metrics_by_perspective"]
    )
    assert sorted(by_perspective) == [0, 1]
    assert by_perspective[0]["n_samples"] == N_FRAMES
    assert by_perspective[1]["n_samples"] == 0
    assert result["perspective_consistency"] == 1.0
    assert result["n_perspective_pairs"] == N_FRAMES
    assert result["n_perspectives"] == 2


def test_frame_source_feature_is_deprecated(tmp_path: Path) -> None:
    ds = _clustered(tmp_path)

    with pytest.deprecated_call():
        _ = compute_cluster_label_agreement(
            ds, STORAGE, perspective=0, frame_source_feature="pair-egocentric"
        )
