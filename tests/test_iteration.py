"""Tests for mosaic.core.pipeline.iteration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.pipeline.tracks_index import TRACKS_INDEX_COLUMNS
from mosaic.core.scope import Scope
from mosaic.core.pipeline.iteration import (
    read_tracks_index,
    yield_sequences,
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


# --- read_tracks_index ---


class TestReadTracksIndex:
    def test_reads_index(self, populated_ds):
        """The reader projects a legacy three-column file onto the full schema.

        Kept as an exact equality rather than a subset: it is the only guard on
        the projection, so a column silently going missing must fail here.
        """
        df = read_tracks_index(populated_ds)
        assert len(df) == 3
        assert list(df.columns) == TRACKS_INDEX_COLUMNS

    def test_missing_index_reads_as_an_empty_one(self, ds):
        """Absent and empty are one dataset state, and now answer alike.

        The full column set matters: callers filter on group/sequence straight
        away, and a column-less empty frame turns "no tracks yet" into KeyError.
        """
        df = read_tracks_index(ds)
        assert len(df) == 0
        assert list(df.columns) == TRACKS_INDEX_COLUMNS
        assert df[df["group"] == "g"].empty

    def test_empty_strings_preserved(self, ds, tmp_path):
        """An empty cell stays an empty string rather than becoming NaN."""
        p = tmp_path / "tracks" / "dummy.parquet"
        _make_parquet(p)
        _write_tracks_index(ds, [("", "s1", p)])
        df = read_tracks_index(ds)
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
        results = list(yield_sequences(populated_ds, Scope(groups=["arena"])))
        assert len(results) == 2
        assert all(g == "arena" for g, _, _ in results)

    def test_filter_sequences(self, populated_ds):
        results = list(yield_sequences(populated_ds, Scope(sequences=["s1"])))
        assert len(results) == 1
        assert results[0][:2] == ("arena", "s1")

    def test_filter_entries(self, populated_ds):
        pairs = [("arena", "s2"), ("field", "s3")]
        results = list(yield_sequences(populated_ds, Scope(entries=pairs)))
        result_pairs = {(g, s) for g, s, _ in results}
        assert result_pairs == set(pairs)

    def test_yields_dataframes(self, populated_ds):
        for _, _, df in yield_sequences(populated_ds):
            assert isinstance(df, pd.DataFrame)
            assert "frame" in df.columns

    def test_missing_parquet_raises(self, ds, tmp_path):
        _write_tracks_index(ds, [("g", "s", tmp_path / "missing.parquet")])
        with pytest.raises(FileNotFoundError, match="Stale tracks index"):
            list(yield_sequences(ds))

    def test_missing_index_yields_nothing(self, ds):
        """A dataset with no tracks has no sequences to iterate, not an error.

        The stale-parquet raise above is unaffected: a row pointing at a file
        that is gone is a broken index, not an absent one.
        """
        assert list(yield_sequences(ds)) == []

    def test_combined_filters(self, populated_ds):
        results = list(
            yield_sequences(populated_ds, Scope(groups=["arena"], sequences=["s2"]))
        )
        assert len(results) == 1
        assert results[0][:2] == ("arena", "s2")
