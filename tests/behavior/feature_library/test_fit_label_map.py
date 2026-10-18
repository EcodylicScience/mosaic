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


def test_a_row_without_a_cluster_is_left_out_of_the_fit(tmp_path: Path) -> None:
    """A gap in the cluster column is skipped, not cast to an arbitrary cluster."""
    events = [(frame, 0, 1, 1) for frame in range(FRAMES)]
    labels = write_sparse_labels(tmp_path / "labels.npz", events)
    rows = _rows([5] * FRAMES).astype({"cluster": float})
    rows.loc[[2, 3], "cluster"] = np.nan
    feature = FitLabelMap(FitLabelMap.Inputs((Result(feature="global-kmeans"),)), {})
    run_root = tmp_path / "run"
    _ = feature.load_state(run_root, {}, {"labels": {("g", "s"): labels}})

    feature.fit(
        InputStream(lambda: iter([(make_entry_key("g", "s"), rows)]), n_entries=1)
    )
    feature.save_state(run_root)

    saved = pd.read_parquet(run_root / LABEL_MAP_FILE)
    assert saved[["cluster", "label", "n_rows"]].values.tolist() == [[5, 1, 8]]


def _rare_class_rows(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    """Class 0 on frames 0-7 and the rare class 1 on frames 8-9.

    Cluster 5 holds three of class 0's eight rows and both of class 1's two, so
    class 0 has more rows in it and class 1 has more of its own.
    """
    events = [(frame, 0, 1, 0 if frame < 8 else 1) for frame in range(FRAMES)]
    labels = write_sparse_labels(tmp_path / "labels.npz", events)
    return _rows([5, 5, 5, 6, 6, 6, 6, 6, 5, 5]), labels


def _fit(
    rows: pd.DataFrame, labels: Path, run_root: Path, **params: object
) -> FitLabelMap:
    feature = FitLabelMap(
        FitLabelMap.Inputs((Result(feature="global-kmeans"),)), dict(params)
    )
    _ = feature.load_state(run_root, {}, {"labels": {("g", "s"): labels}})
    feature.fit(
        InputStream(lambda: iter([(make_entry_key("g", "s"), rows)]), n_entries=1)
    )
    feature.save_state(run_root)
    return feature


def test_the_default_weighting_follows_the_count_of_rows(tmp_path: Path) -> None:
    rows, labels = _rare_class_rows(tmp_path)

    _ = _fit(rows, labels, tmp_path / "default")
    _ = _fit(rows, labels, tmp_path / "count", weighting="count")

    assert FitLabelMap.Params().weighting == "count"
    assert read_label_map(tmp_path / "default" / LABEL_MAP_FILE) == {5: 0, 6: 0}
    assert read_label_map(tmp_path / "count" / LABEL_MAP_FILE) == {5: 0, 6: 0}


def test_class_balanced_weighting_maps_a_cluster_to_the_class_it_holds_most_of(
    tmp_path: Path,
) -> None:
    rows, labels = _rare_class_rows(tmp_path)

    feature = _fit(rows, labels, tmp_path / "run", weighting="class_balanced")

    assert read_label_map(tmp_path / "run" / LABEL_MAP_FILE) == {5: 1, 6: 0}
    saved = pd.read_parquet(tmp_path / "run" / LABEL_MAP_FILE)
    assert saved["n_rows"].tolist() == [5, 5], "n_rows still counts the labeled rows"
    assert feature.apply(rows)[MAPPED_COLUMN].tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]


def test_the_weighting_enters_the_run_identity_and_the_version() -> None:
    count = FitLabelMap.Params(weighting="count").identity_dump()
    balanced = FitLabelMap.Params(weighting="class_balanced").identity_dump()

    assert count != balanced
    assert FitLabelMap.version == "0.2"


def test_an_unknown_weighting_is_refused() -> None:
    with pytest.raises(ValueError, match="class_balanced"):
        _ = FitLabelMap.Params.model_validate({"weighting": "prevalence"})
