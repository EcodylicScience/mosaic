"""Tests for mosaic.core.pipeline.iteration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.pipeline.iteration import (
    _read_tracks_index,
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
