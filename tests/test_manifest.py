"""Tests for the unified manifest builder."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from mosaic.core.pipeline.index import FeatureIndexRow, feature_index
from mosaic.core.pipeline.manifest import (
    ManifestEntry,
    build_manifest,
    iter_manifest,
)
from mosaic.core.pipeline.types import Inputs, ParquetLoadSpec, Result


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

    inputs = Inputs(("tracks",))
    manifest, scope = build_manifest(ds, inputs)
    assert len(manifest) == 2
    assert scope.entries == {("g1", "s1"), ("g1", "s2")}
    # Each entry has one (path, ParquetLoadSpec) tuple
    for key, entry in manifest.items():
        assert isinstance(entry, ManifestEntry)
        assert len(entry.file_specs) == 1
        path, load_spec = entry.file_specs[0]
        assert path.exists()


def test_build_manifest_feature_result(tmp_path):
    ds = _MockDataset(tmp_path)
    run_id = _setup_feature(ds, "speed-angvel", [("g1", "s1"), ("g1", "s2")])
    inputs = Inputs((Result(feature="speed-angvel", run_id=run_id),))

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

    inputs = Inputs(("tracks", Result(feature="nn")))
    manifest, scope = build_manifest(ds, inputs)
    # Intersection: only s1 and s2
    assert scope.entries == {("g1", "s1"), ("g1", "s2")}
    assert len(manifest) == 2
    # Each entry has 2 file specs (track + feature)
    for key, entry in manifest.items():
        assert len(entry.file_specs) == 2


def test_build_manifest_group_filter(tmp_path):
    ds = _MockDataset(tmp_path)
    entries = []
    for g, s in [("g1", "s1"), ("g2", "s2")]:
        p = tmp_path / "tracks" / f"{g}__{s}.parquet"
        _make_parquet(p)
        entries.append((g, s, p))
    _write_tracks_index(ds, entries)

    inputs = Inputs(("tracks",))
    manifest, scope = build_manifest(ds, inputs, groups={"g1"})
    assert scope.entries == {("g1", "s1")}


def test_build_manifest_adjacency(tmp_path):
    """Verify prev/next adjacency pointers are set correctly."""
    ds = _MockDataset(tmp_path)
    entries = []
    for s in ("s1", "s2", "s3"):
        p = tmp_path / "tracks" / f"g1__{s}.parquet"
        _make_parquet(p)
        entries.append(("g1", s, p))
    _write_tracks_index(ds, entries)

    inputs = Inputs(("tracks",))
    manifest, scope = build_manifest(ds, inputs)
    assert len(manifest) == 3

    entry_s1 = manifest["g1__s1"]
    assert entry_s1.prev_entry_key is None
    assert entry_s1.prev_file_specs is None
    assert entry_s1.next_entry_key == "g1__s2"
    assert entry_s1.next_file_specs is not None

    entry_s2 = manifest["g1__s2"]
    assert entry_s2.prev_entry_key == "g1__s1"
    assert entry_s2.prev_file_specs is not None
    assert entry_s2.next_entry_key == "g1__s3"
    assert entry_s2.next_file_specs is not None

    entry_s3 = manifest["g1__s3"]
    assert entry_s3.prev_entry_key == "g1__s2"
    assert entry_s3.prev_file_specs is not None
    assert entry_s3.next_entry_key is None
    assert entry_s3.next_file_specs is None


def test_build_manifest_adjacency_cross_group(tmp_path):
    """Adjacency does not cross group boundaries."""
    ds = _MockDataset(tmp_path)
    entries = []
    for g, s in [("g1", "s1"), ("g1", "s2"), ("g2", "s3")]:
        p = tmp_path / "tracks" / f"{g}__{s}.parquet"
        _make_parquet(p)
        entries.append((g, s, p))
    _write_tracks_index(ds, entries)

    inputs = Inputs(("tracks",))
    manifest, _ = build_manifest(ds, inputs)

    # g1__s2 next should be None (s3 is in g2)
    assert manifest["g1__s2"].next_entry_key is None
    assert manifest["g1__s2"].next_file_specs is None

    # g2__s3 prev should be None (s2 is in g1)
    assert manifest["g2__s3"].prev_entry_key is None
    assert manifest["g2__s3"].prev_file_specs is None


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
    for entry_key, df in results:
        assert isinstance(entry_key, str)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10


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
    entry_key, df = results[0]
    # Inner join on frames 2-7 -> 6 rows
    assert len(df) == 6


# --- Helpers for overlap / filter_factory tests ---


def _make_simple_parquet(path: Path, n_rows: int, start_frame: int = 0) -> None:
    """Write a simple parquet with frame, time, id, feat_a columns."""
    df = pd.DataFrame(
        {
            "frame": range(start_frame, start_frame + n_rows),
            "time": [f / 30.0 for f in range(start_frame, start_frame + n_rows)],
            "id": [0] * n_rows,
            "feat_a": np.arange(n_rows, dtype=float),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _build_three_seq_manifest(tmp_path: Path) -> dict[str, ManifestEntry]:
    """Build a 3-sequence manifest (s1, s2, s3) with adjacency, each 10 rows."""
    paths = {}
    for name in ("s1", "s2", "s3"):
        path = tmp_path / f"g1__{name}.parquet"
        _make_simple_parquet(path, n_rows=10, start_frame=0)
        paths[name] = path

    def specs(name: str) -> list[tuple[Path, ParquetLoadSpec]]:
        return [(paths[name], ParquetLoadSpec())]

    return {
        "g1__s1": ManifestEntry(
            file_specs=specs("s1"),
            prev_file_specs=None,
            prev_entry_key=None,
            next_file_specs=specs("s2"),
            next_entry_key="g1__s2",
        ),
        "g1__s2": ManifestEntry(
            file_specs=specs("s2"),
            prev_file_specs=specs("s1"),
            prev_entry_key="g1__s1",
            next_file_specs=specs("s3"),
            next_entry_key="g1__s3",
        ),
        "g1__s3": ManifestEntry(
            file_specs=specs("s3"),
            prev_file_specs=specs("s2"),
            prev_entry_key="g1__s2",
            next_file_specs=None,
            next_entry_key=None,
        ),
    }


# --- overlap_frames tests ---


def test_iter_manifest_overlap_frames_zero(tmp_path):
    """overlap_frames=0 yields 4-tuples with core_start=0, core_end=len(df)."""
    manifest = _build_three_seq_manifest(tmp_path)
    results = list(iter_manifest(manifest, overlap_frames=0))
    assert len(results) == 3
    for entry_key, df, core_start, core_end in results:
        assert core_start == 0
        assert core_end == 10
        assert len(df) == 10


def test_iter_manifest_overlap_frames_positive(tmp_path):
    """overlap_frames > 0 loads and concatenates neighbor data."""
    manifest = _build_three_seq_manifest(tmp_path)
    results = list(iter_manifest(manifest, overlap_frames=3))
    assert len(results) == 3

    # s1: no prev, has next -> 10 core + 3 next = 13
    key_s1, df_s1, start_s1, end_s1 = results[0]
    assert key_s1 == "g1__s1"
    assert start_s1 == 0
    assert end_s1 == 10
    assert len(df_s1) == 13

    # s2: has prev and next -> 3 prev + 10 core + 3 next = 16
    key_s2, df_s2, start_s2, end_s2 = results[1]
    assert key_s2 == "g1__s2"
    assert start_s2 == 3
    assert end_s2 == 13
    assert len(df_s2) == 16

    # s3: has prev, no next -> 3 prev + 10 core = 13
    key_s3, df_s3, start_s3, end_s3 = results[2]
    assert key_s3 == "g1__s3"
    assert start_s3 == 3
    assert end_s3 == 13
    assert len(df_s3) == 13


def test_iter_manifest_overlap_frames_exceeds_neighbor(tmp_path):
    """overlap_frames larger than neighbor length trims to available rows."""
    manifest = _build_three_seq_manifest(tmp_path)
    # Each sequence has 10 rows; request 50 overlap
    results = list(iter_manifest(manifest, overlap_frames=50))
    assert len(results) == 3

    # s2: prev has 10 rows (all used), next has 10 rows (all used) -> 30 total
    key_s2, df_s2, start_s2, end_s2 = results[1]
    assert key_s2 == "g1__s2"
    assert start_s2 == 10
    assert end_s2 == 20
    assert len(df_s2) == 30


def test_iter_manifest_no_overlap_yields_two_tuples(tmp_path):
    """overlap_frames=None (default) yields 2-tuples."""
    manifest = _build_three_seq_manifest(tmp_path)
    results = list(iter_manifest(manifest))
    assert len(results) == 3
    for item in results:
        assert len(item) == 2


# --- filter_factory tests ---


def test_iter_manifest_filter_factory(tmp_path):
    """filter_factory filters are applied to the loaded data."""
    manifest = _build_three_seq_manifest(tmp_path)

    def factory(entry_key: str) -> Iterable[Callable[[pd.DataFrame], pd.DataFrame]]:
        # Keep only first 5 rows
        return [lambda df: df.iloc[:5]]

    results = list(iter_manifest(manifest, filter_factory=factory))
    assert len(results) == 3
    for entry_key, df in results:
        assert len(df) == 5


def test_iter_manifest_filter_factory_with_overlap(tmp_path):
    """filter_factory is applied to both core and neighbor segments."""
    manifest = _build_three_seq_manifest(tmp_path)

    def factory(entry_key: str) -> Iterable[Callable[[pd.DataFrame], pd.DataFrame]]:
        # Keep only first 5 rows of each segment
        return [lambda df: df.iloc[:5]]

    results = list(iter_manifest(manifest, filter_factory=factory, overlap_frames=3))
    assert len(results) == 3

    # s2: prev filtered to 5 rows, trimmed to 3; core filtered to 5; next filtered to 5, trimmed to 3
    key_s2, df_s2, start_s2, end_s2 = results[1]
    assert key_s2 == "g1__s2"
    assert start_s2 == 3
    assert end_s2 == 8
    assert len(df_s2) == 11  # 3 + 5 + 3


def test_iter_manifest_filter_factory_empty_skips(tmp_path):
    """Entries filtered to empty are skipped."""
    manifest = _build_three_seq_manifest(tmp_path)

    def factory(entry_key: str) -> Iterable[Callable[[pd.DataFrame], pd.DataFrame]]:
        # Return empty DataFrame
        return [lambda df: df.iloc[0:0]]

    results = list(iter_manifest(manifest, filter_factory=factory))
    assert len(results) == 0
