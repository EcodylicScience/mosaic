"""Tests for mosaic.core.pipeline.iteration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.pipeline._utils import InputScope, ResolvedInput
from mosaic.core.pipeline.index import FeatureIndexRow, feature_index
from mosaic.core.pipeline.iteration import (
    _read_tracks_index,
    resolve_feature_pairs,
    resolve_tracks_pairs,
    yield_feature_frames,
    yield_inputset_frames,
    yield_sequences,
    yield_sequences_with_overlap,
)

# --- Helpers ---


class _MockDataset:
    """Minimal Dataset stand-in for iteration tests."""

    def __init__(self, root: Path):
        self._root = root
        self._roots = {
            "tracks": root / "tracks",
            "features": root / "features",
        }
        for d in self._roots.values():
            d.mkdir(parents=True, exist_ok=True)

    def get_root(self, key: str) -> Path:
        if key not in self._roots:
            raise KeyError(f"Root not configured: {key}")
        return self._roots[key]

    def resolve_path(self, stored_path, anchor=None) -> Path:
        p = Path(stored_path)
        if p.is_absolute():
            return p
        return self._root / p


def _make_parquet(path: Path, n_rows: int = 10, n_ids: int = 2) -> pd.DataFrame:
    """Create a simple tracks-like parquet."""
    rows = []
    for fid in range(n_ids):
        for f in range(n_rows):
            rows.append({"frame": f, "time": f / 30.0, "id": fid, "X": f, "Y": f + 1})
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def _write_tracks_index(ds, entries: list[tuple[str, str, Path]]) -> Path:
    """Write tracks/index.csv with (group, sequence, abs_path) entries."""
    idx_path = ds.get_root("tracks") / "index.csv"
    rows = [{"group": g, "sequence": s, "abs_path": str(p)} for g, s, p in entries]
    pd.DataFrame(rows).to_csv(idx_path, index=False)
    return idx_path


def _setup_feature(ds, tmp_path, feat_name, pairs, run_id="v1-abc"):
    """Create a feature with index CSV and parquet files for each (group, seq) pair."""
    feat_dir = tmp_path / "features" / feat_name
    run_dir = feat_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    idx = feature_index(feat_dir / "index.csv")
    idx.ensure()

    for g, s in pairs:
        p = run_dir / f"{g}__{s}.parquet"
        _make_parquet(p, n_rows=10, n_ids=2)
        idx.append(
            [
                FeatureIndexRow(
                    run_id=run_id,
                    feature=feat_name,
                    version="v1",
                    group=g,
                    sequence=s,
                    abs_path=str(p),
                    params_hash="abc",
                    n_rows=20,
                )
            ]
        )
    idx.mark_finished(run_id)
    return idx


# --- Fixtures ---


@pytest.fixture
def ds(tmp_path):
    return _MockDataset(tmp_path)


@pytest.fixture
def populated_ds(ds, tmp_path):
    """Dataset with 3 sequences across 2 groups, each with a parquet file."""
    entries = []
    for g, s in [("arena", "s1"), ("arena", "s2"), ("field", "s3")]:
        p = tmp_path / "tracks" / f"{g}__{s}.parquet"
        _make_parquet(p)
        entries.append((g, s, p))
    _write_tracks_index(ds, entries)
    return ds


# --- _read_tracks_index ---


class TestReadTracksIndex:
    def test_reads_index(self, populated_ds):
        df = _read_tracks_index(populated_ds)
        assert len(df) == 3
        assert set(df.columns) == {"group", "sequence", "abs_path"}

    def test_missing_index_raises(self, ds):
        with pytest.raises(FileNotFoundError, match="tracks/index.csv not found"):
            _read_tracks_index(ds)

    def test_empty_strings_preserved(self, ds, tmp_path):
        """keep_default_na=False: empty strings stay as empty strings, not NaN."""
        p = tmp_path / "tracks" / "dummy.parquet"
        _make_parquet(p)
        _write_tracks_index(ds, [("", "s1", p)])
        df = _read_tracks_index(ds)
        assert df.iloc[0]["group"] == ""
        assert not pd.isna(df.iloc[0]["group"])


# --- yield_sequences ---


class TestYieldSequences:
    def test_yields_all(self, populated_ds):
        results = list(yield_sequences(populated_ds))
        assert len(results) == 3
        pairs = {(g, s) for g, s, _ in results}
        assert pairs == {("arena", "s1"), ("arena", "s2"), ("field", "s3")}

    def test_filter_groups(self, populated_ds):
        results = list(yield_sequences(populated_ds, groups=["arena"]))
        assert len(results) == 2
        assert all(g == "arena" for g, _, _ in results)

    def test_filter_sequences(self, populated_ds):
        results = list(yield_sequences(populated_ds, sequences=["s1"]))
        assert len(results) == 1
        assert results[0][:2] == ("arena", "s1")

    def test_filter_allowed_pairs(self, populated_ds):
        pairs = {("arena", "s2"), ("field", "s3")}
        results = list(yield_sequences(populated_ds, allowed_pairs=pairs))
        result_pairs = {(g, s) for g, s, _ in results}
        assert result_pairs == pairs

    def test_yields_dataframes(self, populated_ds):
        for _, _, df in yield_sequences(populated_ds):
            assert isinstance(df, pd.DataFrame)
            assert "frame" in df.columns

    def test_missing_parquet_raises(self, ds, tmp_path):
        _write_tracks_index(ds, [("g", "s", tmp_path / "missing.parquet")])
        with pytest.raises(FileNotFoundError, match="Stale tracks index"):
            list(yield_sequences(ds))

    def test_missing_index_raises(self, ds):
        with pytest.raises(FileNotFoundError):
            list(yield_sequences(ds))

    def test_combined_filters(self, populated_ds):
        results = list(
            yield_sequences(populated_ds, groups=["arena"], sequences=["s2"])
        )
        assert len(results) == 1
        assert results[0][:2] == ("arena", "s2")


# --- yield_sequences_with_overlap ---


class TestYieldSequencesWithOverlap:
    @pytest.fixture
    def three_seq_ds(self, ds, tmp_path):
        """3 sequences in same group, 100 rows each (1 id)."""
        entries = []
        for s in ["s1", "s2", "s3"]:
            p = tmp_path / "tracks" / f"arena__{s}.parquet"
            _make_parquet(p, n_rows=100, n_ids=1)
            entries.append(("arena", s, p))
        _write_tracks_index(ds, entries)
        return ds

    def test_zero_overlap_delegates(self, populated_ds):
        results = list(yield_sequences_with_overlap(populated_ds, overlap_frames=0))
        assert len(results) == 3
        for _, _, df, start, end in results:
            assert start == 0
            assert end == len(df)

    def test_overlap_sizes(self, three_seq_ds):
        results = {
            s: (df, start, end)
            for _, s, df, start, end in yield_sequences_with_overlap(
                three_seq_ds, overlap_frames=10
            )
        }
        # First: no prefix, 10-frame suffix
        df, start, end = results["s1"]
        assert start == 0
        assert end - start == 100
        assert len(df) == 110

        # Middle: 10-frame prefix, 10-frame suffix
        df, start, end = results["s2"]
        assert start == 10
        assert end - start == 100
        assert len(df) == 120

        # Last: 10-frame prefix, no suffix
        df, start, end = results["s3"]
        assert start == 10
        assert end - start == 100
        assert len(df) == 110

    def test_overlap_clamps_to_available(self, ds, tmp_path):
        """Overlap larger than adjacent sequence clamps to that sequence's length."""
        entries = []
        for s, n in [("s1", 5), ("s2", 100)]:
            p = tmp_path / "tracks" / f"arena__{s}.parquet"
            _make_parquet(p, n_rows=n, n_ids=1)
            entries.append(("arena", s, p))
        _write_tracks_index(ds, entries)

        results = {
            s: (df, start, end)
            for _, s, df, start, end in yield_sequences_with_overlap(
                ds, overlap_frames=50
            )
        }
        # s2 should have prefix of 5 (all of s1), not 50
        df, start, end = results["s2"]
        assert start == 5
        assert end - start == 100

    def test_overlap_data_content(self, ds, tmp_path):
        """Verify overlap frames contain actual data from adjacent sequences."""
        entries = []
        for i, s in enumerate(["s1", "s2", "s3"]):
            p = tmp_path / "tracks" / f"arena__{s}.parquet"
            # Each sequence has distinct X values: s1=[100..], s2=[200..], s3=[300..]
            base = (i + 1) * 100
            rows = [
                {"frame": f, "time": f / 30.0, "id": 0, "X": base + f}
                for f in range(10)
            ]
            df = pd.DataFrame(rows)
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p)
            entries.append(("arena", s, p))
        _write_tracks_index(ds, entries)

        results = {
            s: (df, start, end)
            for _, s, df, start, end in yield_sequences_with_overlap(
                ds, overlap_frames=3
            )
        }
        # s2: prefix should be last 3 frames of s1 (X=107,108,109), suffix first 3 of s3 (X=300,301,302)
        df, start, end = results["s2"]
        prefix_x = df["X"].iloc[:start].tolist()
        assert prefix_x == [107, 108, 109]
        suffix_x = df["X"].iloc[end:].tolist()
        assert suffix_x == [300, 301, 302]
        # Core data is s2
        core_x = df["X"].iloc[start:end].tolist()
        assert core_x == list(range(200, 210))

    def test_no_overlap_across_groups(self, populated_ds):
        """Adjacent sequences are only within the same group."""
        results = {
            (g, s): (df, start, end)
            for g, s, df, start, end in yield_sequences_with_overlap(
                populated_ds, overlap_frames=5
            )
        }
        # field/s3 is alone in its group -> no overlap
        df, start, end = results[("field", "s3")]
        assert start == 0
        assert end == len(df)


# --- yield_feature_frames ---


class TestYieldFeatureFrames:
    @pytest.fixture
    def feature_ds(self, ds, tmp_path):
        _setup_feature(ds, tmp_path, "speed", [("arena", "s1"), ("arena", "s2")])
        return ds

    def test_yields_all_for_run(self, feature_ds):
        results = list(yield_feature_frames(feature_ds, "speed", run_id="v1-abc"))
        assert len(results) == 2

    def test_auto_resolves_latest_run(self, feature_ds):
        results = list(yield_feature_frames(feature_ds, "speed"))
        assert len(results) == 2

    def test_filter_by_group(self, feature_ds):
        results = list(yield_feature_frames(feature_ds, "speed", groups=["arena"]))
        assert len(results) == 2

    def test_filter_by_sequence(self, feature_ds):
        results = list(yield_feature_frames(feature_ds, "speed", sequences=["s1"]))
        assert len(results) == 1
        assert results[0][1] == "s1"

    def test_filter_by_allowed_pairs(self, feature_ds):
        results = list(
            yield_feature_frames(feature_ds, "speed", allowed_pairs={("arena", "s2")})
        )
        assert len(results) == 1
        assert results[0][1] == "s2"

    def test_no_such_feature_raises(self, ds):
        with pytest.raises(FileNotFoundError):
            list(yield_feature_frames(ds, "nonexistent"))

    def test_no_such_run_raises(self, feature_ds):
        with pytest.raises(ValueError, match="No entries"):
            list(yield_feature_frames(feature_ds, "speed", run_id="nonexistent"))

    def test_skips_tiny_tables(self, ds, tmp_path):
        """Tables with <= 1 row are skipped."""
        feat_dir = tmp_path / "features" / "tiny"
        run_dir = feat_dir / "v1-abc"
        run_dir.mkdir(parents=True)

        idx = feature_index(feat_dir / "index.csv")
        idx.ensure()

        p = run_dir / "arena__s1.parquet"
        pd.DataFrame({"frame": [0], "X": [1.0], "Y": [2.0]}).to_parquet(p)
        idx.append(
            [
                FeatureIndexRow(
                    run_id="v1-abc",
                    feature="tiny",
                    version="v1",
                    group="arena",
                    sequence="s1",
                    abs_path=str(p),
                    params_hash="abc",
                    n_rows=1,
                )
            ]
        )
        idx.mark_finished("v1-abc")

        results = list(yield_feature_frames(ds, "tiny", run_id="v1-abc"))
        assert len(results) == 0

    def test_skips_non_parquet_entries(self, ds, tmp_path):
        feat_dir = tmp_path / "features" / "npz_feat"
        run_dir = feat_dir / "v1-abc"
        run_dir.mkdir(parents=True)

        npz_path = run_dir / "data.npz"
        npz_path.touch()

        idx = feature_index(feat_dir / "index.csv")
        idx.ensure()
        idx.append(
            [
                FeatureIndexRow(
                    run_id="v1-abc",
                    feature="npz_feat",
                    version="v1",
                    group="arena",
                    sequence="s1",
                    abs_path=str(npz_path),
                    params_hash="abc",
                    n_rows=100,
                )
            ]
        )
        idx.mark_finished("v1-abc")

        results = list(yield_feature_frames(ds, "npz_feat", run_id="v1-abc"))
        assert len(results) == 0

    def test_skips_few_numeric_columns(self, ds, tmp_path):
        """Tables with < 2 numeric columns are skipped."""
        feat_dir = tmp_path / "features" / "sparse"
        run_dir = feat_dir / "v1-abc"
        run_dir.mkdir(parents=True)

        idx = feature_index(feat_dir / "index.csv")
        idx.ensure()

        p = run_dir / "arena__s1.parquet"
        # Only 1 numeric column
        pd.DataFrame({"label": ["a", "b", "c"], "val": [1, 2, 3]}).to_parquet(p)
        idx.append(
            [
                FeatureIndexRow(
                    run_id="v1-abc",
                    feature="sparse",
                    version="v1",
                    group="arena",
                    sequence="s1",
                    abs_path=str(p),
                    params_hash="abc",
                    n_rows=3,
                )
            ]
        )
        idx.mark_finished("v1-abc")

        results = list(yield_feature_frames(ds, "sparse", run_id="v1-abc"))
        assert len(results) == 0


# --- resolve_tracks_pairs ---


class TestResolveTracksPairs:
    def test_returns_all_pairs(self, populated_ds):
        pairs = resolve_tracks_pairs(populated_ds, None, None)
        assert pairs == {("arena", "s1"), ("arena", "s2"), ("field", "s3")}

    def test_filter_by_groups(self, populated_ds):
        pairs = resolve_tracks_pairs(populated_ds, {"arena"}, None)
        assert len(pairs) == 2
        assert all(g == "arena" for g, _ in pairs)

    def test_filter_by_sequences(self, populated_ds):
        pairs = resolve_tracks_pairs(populated_ds, None, {"s1"})
        assert pairs == {("arena", "s1")}

    def test_missing_index_raises(self, ds):
        with pytest.raises(FileNotFoundError):
            resolve_tracks_pairs(ds, None, None)


# --- resolve_feature_pairs ---


class TestResolveFeaturePairs:
    @pytest.fixture
    def feature_ds(self, ds, tmp_path):
        _setup_feature(ds, tmp_path, "speed", [("arena", "s1"), ("arena", "s2")])
        return ds

    def test_returns_pairs_and_resolved_input(self, feature_ds):
        pairs, resolved = resolve_feature_pairs(
            feature_ds, "speed", "v1-abc", None, None
        )
        assert pairs == {("arena", "s1"), ("arena", "s2")}
        assert resolved.kind == "feature"
        assert resolved.feature == "speed"
        assert resolved.run_id == "v1-abc"
        assert len(resolved.path_map) == 2

    def test_auto_resolves_latest_run(self, feature_ds):
        pairs, resolved = resolve_feature_pairs(feature_ds, "speed", None, None, None)
        assert resolved.run_id == "v1-abc"
        assert len(pairs) == 2

    def test_filter_by_groups(self, feature_ds):
        pairs, _ = resolve_feature_pairs(feature_ds, "speed", "v1-abc", {"arena"}, None)
        assert len(pairs) == 2

    def test_filter_by_sequences(self, feature_ds):
        pairs, _ = resolve_feature_pairs(feature_ds, "speed", "v1-abc", None, {"s1"})
        assert pairs == {("arena", "s1")}

    def test_drops_global_rows(self, ds, tmp_path):
        """Rows with sequence='__global__' or empty sequence are excluded."""
        feat_dir = tmp_path / "features" / "gfeat"
        run_dir = feat_dir / "v1-abc"
        run_dir.mkdir(parents=True)

        idx = feature_index(feat_dir / "index.csv")
        idx.ensure()

        # Normal row
        p = run_dir / "arena__s1.parquet"
        _make_parquet(p)
        idx.append(
            [
                FeatureIndexRow(
                    run_id="v1-abc",
                    feature="gfeat",
                    version="v1",
                    group="arena",
                    sequence="s1",
                    abs_path=str(p),
                    params_hash="abc",
                    n_rows=10,
                )
            ]
        )
        # Global row
        global_path = run_dir / "g.npz"
        global_path.touch()
        idx.append(
            [
                FeatureIndexRow(
                    run_id="v1-abc",
                    feature="gfeat",
                    version="v1",
                    group="arena",
                    sequence="__global__",
                    abs_path=str(global_path),
                    params_hash="abc",
                    n_rows=0,
                )
            ]
        )
        # Empty sequence row
        empty_path = run_dir / "e.npz"
        empty_path.touch()
        idx.append(
            [
                FeatureIndexRow(
                    run_id="v1-abc",
                    feature="gfeat",
                    version="v1",
                    group="arena",
                    sequence="",
                    abs_path=str(empty_path),
                    params_hash="abc",
                    n_rows=0,
                )
            ]
        )
        idx.mark_finished("v1-abc")

        pairs, _ = resolve_feature_pairs(ds, "gfeat", "v1-abc", None, None)
        assert pairs == {("arena", "s1")}

    def test_nonexistent_run_raises(self, feature_ds):
        with pytest.raises(FileNotFoundError):
            resolve_feature_pairs(feature_ds, "speed", "nonexistent", None, None)

    def test_no_feature_raises(self, ds):
        with pytest.raises(FileNotFoundError):
            resolve_feature_pairs(ds, "nonexistent", "v1-abc", None, None)


# --- yield_inputset_frames ---


class TestYieldInputsetFrames:
    @pytest.fixture
    def inputset_ds(self, ds, tmp_path):
        """Dataset with tracks and a feature, both covering the same 2 sequences."""
        pairs = [("arena", "s1"), ("arena", "s2")]

        # Tracks
        entries = []
        for g, s in pairs:
            p = tmp_path / "tracks" / f"{g}__{s}.parquet"
            _make_parquet(p)
            entries.append((g, s, p))
        _write_tracks_index(ds, entries)

        # Feature
        _setup_feature(ds, tmp_path, "speed", pairs)
        return ds

    def _make_scope(self, ds, tmp_path, pairs, include_feature=True):
        """Build a minimal InputScope for testing."""
        resolved_inputs = [ResolvedInput(kind="tracks")]
        if include_feature:
            feat_dir = tmp_path / "features" / "speed" / "v1-abc"
            path_map = {}
            for g, s in pairs:
                path_map[(g, s)] = feat_dir / f"{g}__{s}.parquet"
            resolved_inputs.append(
                ResolvedInput(
                    kind="feature",
                    feature="speed",
                    run_id="v1-abc",
                    path_map=path_map,
                )
            )
        return InputScope(
            pairs=set(pairs),
            safe_sequences={p[1] for p in pairs},
            resolved_inputs=resolved_inputs,
        )

    def test_yields_merged_frames(self, inputset_ds, tmp_path):
        pairs = [("arena", "s1"), ("arena", "s2")]
        scope = self._make_scope(inputset_ds, tmp_path, pairs)
        results = list(yield_inputset_frames(inputset_ds, scope=scope))
        assert len(results) == 2
        for _, _, df in results:
            assert isinstance(df, pd.DataFrame)

    def test_metadata_only(self, inputset_ds, tmp_path):
        pairs = [("arena", "s1"), ("arena", "s2")]
        scope = self._make_scope(inputset_ds, tmp_path, pairs)
        results = list(
            yield_inputset_frames(inputset_ds, scope=scope, metadata_only=True)
        )
        assert len(results) == 2
        for _, _, df in results:
            # Should only have routing columns
            assert all(
                c in ("frame", "time", "group", "sequence", "id") for c in df.columns
            )

    def test_empty_scope_yields_nothing(self, inputset_ds, tmp_path):
        scope = InputScope()
        results = list(yield_inputset_frames(inputset_ds, scope=scope))
        assert len(results) == 0

    def test_tracks_only_scope(self, inputset_ds, tmp_path):
        pairs = [("arena", "s1")]
        scope = self._make_scope(inputset_ds, tmp_path, pairs, include_feature=False)
        results = list(yield_inputset_frames(inputset_ds, scope=scope))
        assert len(results) == 1
