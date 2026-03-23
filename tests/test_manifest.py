"""Tests for the unified manifest builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mosaic.core.pipeline._utils import Scope
from mosaic.core.pipeline.index import FeatureIndexRow, feature_index
from mosaic.core.pipeline.loading import EntryData
from mosaic.core.pipeline.manifest import build_manifest, iter_manifest
from mosaic.core.pipeline.types import Inputs, Result


class _MockDataset:
    def __init__(self, root: Path):
        self._root = root
        for d in ("tracks", "features"):
            (root / d).mkdir(parents=True, exist_ok=True)

    def get_root(self, key: str) -> Path:
        return self._root / key

    def resolve_path(self, stored_path, anchor=None) -> Path:
        p = Path(stored_path)
        return p if p.is_absolute() else self._root / p


def _make_parquet(path: Path, n_rows: int = 10) -> None:
    df = pd.DataFrame(
        {
            "frame": range(n_rows),
            "time": [f / 30.0 for f in range(n_rows)],
            "id": [0] * n_rows,
            "feat_a": np.random.randn(n_rows),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _write_tracks_index(ds, entries):
    idx_path = ds.get_root("tracks") / "index.csv"
    rows = [{"group": g, "sequence": s, "abs_path": str(p)} for g, s, p in entries]
    pd.DataFrame(rows).to_csv(idx_path, index=False)


def _setup_feature(ds, feat_name, pairs, run_id="v1-abc"):
    feat_dir = ds.get_root("features") / feat_name
    run_dir = feat_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    idx = feature_index(feat_dir / "index.csv")
    idx.ensure()
    rows = []
    for g, s in pairs:
        p = run_dir / f"{g}__{s}.parquet"
        _make_parquet(p)
        rows.append(
            FeatureIndexRow(
                run_id=run_id,
                feature=feat_name,
                version="0.1",
                group=g,
                sequence=s,
                abs_path=str(p),
                n_rows=10,
                params_hash="abc",
            )
        )
    idx.append(rows)
    idx.mark_finished(run_id)
    return run_id


def test_build_manifest_tracks_only(tmp_path):
    ds = _MockDataset(tmp_path)
    entries = []
    for g, s in [("g1", "s1"), ("g1", "s2")]:
        p = tmp_path / "tracks" / f"{g}__{s}.parquet"
        _make_parquet(p)
        entries.append((g, s, p))
    _write_tracks_index(ds, entries)

    from mosaic.core.pipeline.manifest import build_manifest

    inputs = Inputs(("tracks",))
    manifest, scope = build_manifest(ds, inputs)
    assert len(manifest) == 2
    assert scope.entries == {("g1", "s1"), ("g1", "s2")}
    # Each entry has one (path, ParquetLoadSpec) tuple
    for key, specs in manifest.items():
        assert len(specs) == 1
        path, load_spec = specs[0]
        assert path.exists()


def test_build_manifest_feature_result(tmp_path):
    ds = _MockDataset(tmp_path)
    run_id = _setup_feature(ds, "speed-angvel", [("g1", "s1"), ("g1", "s2")])
    inputs = Inputs((Result(feature="speed-angvel", run_id=run_id),))

    from mosaic.core.pipeline.manifest import build_manifest

    manifest, scope = build_manifest(ds, inputs)
    assert len(manifest) == 2
    assert scope.entries == {("g1", "s1"), ("g1", "s2")}


def test_build_manifest_mixed_intersects(tmp_path):
    ds = _MockDataset(tmp_path)
    # Tracks have s1, s2, s3
    entries = []
    for s in ("s1", "s2", "s3"):
        p = tmp_path / "tracks" / f"g1__{s}.parquet"
        _make_parquet(p)
        entries.append(("g1", s, p))
    _write_tracks_index(ds, entries)
    # Feature has only s1, s2 -- no run_id, exercises latest-run resolution
    _setup_feature(ds, "nn", [("g1", "s1"), ("g1", "s2")])

    from mosaic.core.pipeline.manifest import build_manifest

    inputs = Inputs(("tracks", Result(feature="nn")))
    manifest, scope = build_manifest(ds, inputs)
    # Intersection: only s1 and s2
    assert scope.entries == {("g1", "s1"), ("g1", "s2")}
    assert len(manifest) == 2
    # Each entry has 2 file specs (track + feature)
    for key, specs in manifest.items():
        assert len(specs) == 2


def test_build_manifest_group_filter(tmp_path):
    ds = _MockDataset(tmp_path)
    entries = []
    for g, s in [("g1", "s1"), ("g2", "s2")]:
        p = tmp_path / "tracks" / f"{g}__{s}.parquet"
        _make_parquet(p)
        entries.append((g, s, p))
    _write_tracks_index(ds, entries)

    from mosaic.core.pipeline.manifest import build_manifest

    inputs = Inputs(("tracks",))
    manifest, scope = build_manifest(ds, inputs, groups={"g1"})
    assert scope.entries == {("g1", "s1")}


def test_iter_manifest_yields_keydata(tmp_path):
    ds = _MockDataset(tmp_path)
    entries = []
    for g, s in [("g1", "s1"), ("g1", "s2")]:
        p = tmp_path / "tracks" / f"{g}__{s}.parquet"
        _make_parquet(p, n_rows=10)
        entries.append((g, s, p))
    _write_tracks_index(ds, entries)

    inputs = Inputs(("tracks",))
    manifest, _ = build_manifest(ds, inputs)

    results = list(iter_manifest(manifest))
    assert len(results) == 2
    for entry_key, entry_data in results:
        assert isinstance(entry_key, str)
        assert isinstance(entry_data, EntryData)
        assert entry_data.features.shape[0] == 10
        assert entry_data.frames.shape == (10,)


def test_iter_manifest_mixed_inner_join(tmp_path):
    ds = _MockDataset(tmp_path)

    # Tracks: frames 0-9
    track_path = tmp_path / "tracks" / "g1__s1.parquet"
    track_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "frame": range(10),
            "time": [f / 30.0 for f in range(10)],
            "id": [0] * 10,
            "feat_a": np.random.randn(10),
        }
    ).to_parquet(track_path)
    _write_tracks_index(ds, [("g1", "s1", track_path)])

    # Feature: frames 2-7 only
    run_id = "v1-abc"
    feat_dir = ds.get_root("features") / "narrowfeat"
    run_dir = feat_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    feat_path = run_dir / "g1__s1.parquet"
    pd.DataFrame(
        {
            "frame": range(2, 8),
            "time": [f / 30.0 for f in range(2, 8)],
            "id": [0] * 6,
            "feat_b": np.random.randn(6),
        }
    ).to_parquet(feat_path)

    idx = feature_index(feat_dir / "index.csv")
    idx.ensure()
    idx.append(
        [
            FeatureIndexRow(
                run_id=run_id,
                feature="narrowfeat",
                version="0.1",
                group="g1",
                sequence="s1",
                abs_path=str(feat_path),
                n_rows=6,
                params_hash="abc",
            )
        ]
    )
    idx.mark_finished(run_id)

    inputs = Inputs(("tracks", Result(feature="narrowfeat", run_id=run_id)))
    manifest, _ = build_manifest(ds, inputs)

    results = list(iter_manifest(manifest))
    assert len(results) == 1
    entry_key, entry_data = results[0]
    # Inner join on frames 2-7 -> 6 rows
    assert entry_data.features.shape[0] == 6
    assert entry_data.frames.shape == (6,)
