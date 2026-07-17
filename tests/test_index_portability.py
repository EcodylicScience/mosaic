"""Path-portability + reindex tests for the feature index.

Guards the fix that makes feature indexes store dataset-root-*relative* paths
(so a moved / synced dataset resolves on any machine) and the resolve-then-skip
behavior in ``manifest._resolve_feature``:

- relocated-but-present outputs resolve under a different root (no false-fail),
- an all-missing run raises a loud, actionable error (dataset moved),
- a partially-missing run skips the gone entries (they recompute upstream),
- ``Dataset.reindex_features`` prunes only genuinely-missing rows and reconciles
  the SQLite registry mirror,
- a real ``run_feature`` writes relative paths (regression guard).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library import SpeedAngvel
from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline import FeatureStep, Pipeline
from mosaic.core.pipeline.index import (
    FeatureIndexRow,
    feature_index,
    feature_index_path,
)
from mosaic.core.pipeline.manifest import _resolve_feature
from mosaic.core.pipeline.registry import open_registry


# --- Mock dataset (resolves relative paths against its root) ---


class _MockDataset:
    def __init__(self, root: Path):
        self._root = root
        (root / "features").mkdir(parents=True, exist_ok=True)

    def get_root(self, key: str) -> Path:
        return self._root / key

    def resolve_path(self, stored_path: object, anchor: object = None) -> Path:
        path = Path(str(stored_path))
        return path if path.is_absolute() else self._root / path


# --- Helpers ---


def _make_feat_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "frame": range(5),
            "time": [f / 30.0 for f in range(5)],
            "id": [0] * 5,
            "feat": np.arange(5.0),
        }
    ).to_parquet(path)


def _write_relative_feature_index(
    root: Path,
    feat: str,
    run_id: str,
    pairs: list[tuple[str, str]],
) -> None:
    """Write a feature index under *root* whose abs_path values are RELATIVE."""
    idx = feature_index(feature_index_path(_MockDataset(root), feat))
    idx.ensure()
    rows: list[FeatureIndexRow] = []
    for g, s in pairs:
        rel = f"features/{feat}/{run_id}/{g}__{s}.parquet"
        _make_feat_parquet(root / rel)
        rows.append(
            FeatureIndexRow(
                run_id=run_id,
                feature=feat,
                version="0.1",
                group=g,
                sequence=s,
                abs_path=rel,
                n_rows=5,
                params_hash="h",
            )
        )
    idx.append(rows)
    idx.mark_finished(run_id)


# --- _resolve_feature portability behavior ---


def test_relative_index_resolves_under_a_different_root(tmp_path: Path) -> None:
    """A relative index built under root A resolves after a copy to root B."""
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    _write_relative_feature_index(root_a, "feat", "0.1-abc", [("g", "s1"), ("g", "s2")])
    shutil.copytree(root_a, root_b)

    ds_b = _MockDataset(root_b)
    scoped, path_map, full_order, path_map_all = _resolve_feature(
        ds_b, "feat", "0.1-abc", None, None, None
    )
    assert scoped == {("g", "s1"), ("g", "s2")}
    assert full_order == [("g", "s1"), ("g", "s2")]
    # Every resolved path lives under root_b and exists.
    for resolved, _spec in path_map_all.values():
        assert resolved.exists()
        assert root_b in resolved.parents


def test_all_missing_run_raises_actionable_error(tmp_path: Path) -> None:
    root = tmp_path / "a"
    root.mkdir()
    _write_relative_feature_index(root, "feat", "0.1-abc", [("g", "s1"), ("g", "s2")])
    # Remove every output for the run -> "dataset moved" signal.
    shutil.rmtree(root / "features" / "feat" / "0.1-abc")

    ds = _MockDataset(root)
    with pytest.raises(FileNotFoundError, match="output file"):
        _resolve_feature(ds, "feat", "0.1-abc", None, None, None)


def test_partial_missing_run_skips(tmp_path: Path) -> None:
    root = tmp_path / "a"
    root.mkdir()
    _write_relative_feature_index(root, "feat", "0.1-abc", [("g", "s1"), ("g", "s2")])
    (root / "features" / "feat" / "0.1-abc" / "g__s1.parquet").unlink()

    ds = _MockDataset(root)
    scoped, _path_map, full_order, _path_map_all = _resolve_feature(
        ds, "feat", "0.1-abc", None, None, None
    )
    # The surviving entry is kept; the missing one is dropped (recomputed upstream).
    assert scoped == {("g", "s2")}
    assert full_order == [("g", "s2")]


# --- Dataset.reindex_features ---


def _dataset_with_manual_feature(tmp_path: Path) -> tuple[Dataset, str, str]:
    manifest = new_dataset_manifest("t", base_dir=tmp_path)
    ds = Dataset(manifest_path=manifest).load()
    feat, run_id = "feat", "0.1-abc"
    _write_relative_feature_index(
        ds.get_root("features").parent, feat, run_id, [("g", "s1"), ("g", "s2")]
    )
    return ds, feat, run_id


def test_reindex_dry_run_reports_without_writing(tmp_path: Path) -> None:
    ds, feat, run_id = _dataset_with_manual_feature(tmp_path)
    (ds.get_root("features") / feat / run_id / "g__s1.parquet").unlink()

    report = ds.reindex_features(dry_run=True)
    assert sum(report.values()) == 1
    # Nothing rewritten on disk.
    idx = feature_index(feature_index_path(ds, feat))
    assert len(idx.read(run_id=run_id)) == 2


def test_reindex_drops_missing_keeps_present_and_reconciles_registry(
    tmp_path: Path,
) -> None:
    ds, feat, run_id = _dataset_with_manual_feature(tmp_path)
    # Seed the SQLite mirror from the CSV (2 entries).
    reg = open_registry(ds.get_root("features"))
    assert len(reg.list_entries(feat, run_id)) == 2
    reg.close()

    (ds.get_root("features") / feat / run_id / "g__s1.parquet").unlink()

    report = ds.reindex_features(dry_run=False)
    assert sum(report.values()) == 1

    idx = feature_index(feature_index_path(ds, feat))
    remaining = idx.read(run_id=run_id)
    assert len(remaining) == 1
    assert (remaining.iloc[0]["group"], remaining.iloc[0]["sequence"]) == ("g", "s2")

    reg = open_registry(ds.get_root("features"), migrate_csv=False)
    entries = reg.list_entries(feat, run_id)
    reg.close()
    assert len(entries) == 1
    assert entries.iloc[0]["sequence"] == "s2"


def test_reindex_keeps_relocated_present_rows(tmp_path: Path) -> None:
    """Relative paths that resolve to existing files are never pruned."""
    ds, feat, run_id = _dataset_with_manual_feature(tmp_path)
    report = ds.reindex_features(dry_run=False)
    assert report == {}  # all present -> nothing dropped
    idx = feature_index(feature_index_path(ds, feat))
    assert len(idx.read(run_id=run_id)) == 2


# --- run_feature writes relative paths (regression guard for the root cause) ---


def test_run_feature_writes_relative_paths(tmp_path: Path) -> None:
    manifest = new_dataset_manifest("t", base_dir=tmp_path)
    ds = Dataset(manifest_path=manifest).load()
    tracks_root = ds.get_root("tracks")
    rows = []
    for group, sequence in [("g", "s1"), ("g", "s2")]:
        n = 12
        df = pd.DataFrame(
            {
                "frame": range(n),
                "time": [f / 30.0 for f in range(n)],
                "id": [0] * n,
                "X": np.linspace(0.0, 5.0, n),
                "Y": np.linspace(0.0, 2.0, n),
            }
        )
        path = tracks_root / f"{group}__{sequence}.parquet"
        df.to_parquet(path)
        rows.append({"group": group, "sequence": sequence, "abs_path": str(path)})
    pd.DataFrame(rows).to_csv(tracks_root / "index.csv", index=False)

    pipe = Pipeline()
    pipe.add(FeatureStep("speed", SpeedAngvel, {"step_size": 1}))
    results = pipe.run(ds)
    feat, run_id = results["speed"].feature, results["speed"].run_id

    df_idx = feature_index(feature_index_path(ds, feat)).read(run_id=run_id)
    assert len(df_idx) == 2
    for stored in df_idx["abs_path"]:
        assert not Path(stored).is_absolute(), f"index path is absolute: {stored}"
        # And it still resolves to a real file under the dataset root.
        assert ds.resolve_path(stored).exists()
