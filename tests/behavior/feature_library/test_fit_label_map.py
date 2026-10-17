"""``fit-label-map``: each cluster's majority class, fitted on its own entries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.fit_label_map import (
    LABEL_MAP_FILE,
    MAPPED_COLUMN,
    FitLabelMap,
    read_label_map,
)
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.types import InputStream, Result
from tests.helpers import write_sparse_labels

FRAMES = 10


def _rows(clusters: list[int]) -> pd.DataFrame:
    """Rows of individual 0 focal on 1, one per frame, in cluster *clusters*."""
    return pd.DataFrame(
        {
            "group": "g",
            "sequence": "s",
            "frame": np.arange(FRAMES),
            "id1": 0,
            "id2": 1,
            "cluster": clusters,
        }
    )


def test_writes_the_majority_map(tmp_path: Path) -> None:
    """Cluster 5 is mostly class 1, cluster 6 class 0, and cluster 8 is unlabeled."""
    events = [(frame, 0, 1, 1 if frame < 6 else 0) for frame in range(8)]
    labels = write_sparse_labels(tmp_path / "labels.npz", events)
    rows = _rows([5, 5, 5, 5, 5, 6, 6, 6, 8, 8])
    feature = FitLabelMap(FitLabelMap.Inputs((Result(feature="global-kmeans"),)), {})
    run_root = tmp_path / "run"

    assert not feature.load_state(run_root, {}, {"labels": {("g", "s"): labels}})
    feature.fit(
        InputStream(lambda: iter([(make_entry_key("g", "s"), rows)]), n_entries=1)
    )
    feature.save_state(run_root)
    applied = feature.apply(rows)

    assert read_label_map(run_root / LABEL_MAP_FILE) == {5: 1, 6: 0}
    saved = pd.read_parquet(run_root / LABEL_MAP_FILE)
    assert saved["n_rows"].tolist() == [5, 3]
    assert applied[MAPPED_COLUMN].tolist() == [1] * 5 + [0] * 3 + [-1] * 2
    reloaded = FitLabelMap(FitLabelMap.Inputs((Result(feature="global-kmeans"),)), {})
    assert reloaded.load_state(run_root, {}, {"labels": {}})


def test_no_labeled_row_is_refused(tmp_path: Path) -> None:
    feature = FitLabelMap(FitLabelMap.Inputs((Result(feature="global-kmeans"),)), {})
    _ = feature.load_state(tmp_path / "run", {}, {"labels": {}})

    with pytest.raises(RuntimeError, match="no labeled row"):
        feature.fit(
            InputStream(
                lambda: iter([(make_entry_key("g", "s"), _rows([1] * FRAMES))]),
                n_entries=1,
            )
        )
