"""Tests for the raw-tracks index writer.

Covers the pure serialization helpers in ``mosaic.core.pipeline.tracks_index``,
the root-relative-in-tree / absolute-out-of-tree ``abs_path`` contract and
atomic write that ``Dataset.index_tracks_raw`` now enforces (the scan/identity
logic -- TREx suffix strip, ``multi_sequences_per_file`` grouping -- is asserted
unchanged), and the propagation of that relative form into the merged
``tracks/index.csv`` ``source_abs_path`` written by ``convert_all_tracks``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import mosaic.core.track_library  # noqa: F401  -- registers the trex_npz converter
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.tracks_index import (
    TRACKS_RAW_INDEX_COLUMNS,
    build_tracks_raw_row,
    frame_from_rows,
    load_tracks_index_frame,
    read_tracks_index,
    write_tracks_index_rows,
)


def _make_dataset(base: Path) -> Dataset:
    base.mkdir(parents=True, exist_ok=True)
    ds = Dataset(
        manifest_path=base / "dataset.yaml",
        roots={
            "tracks_raw": str(base / "tracks_raw"),
            "tracks": str(base / "tracks"),
        },
    )
    ds.ensure_roots()
    # Seed the manifest so base_dir resolves to the manifest's parent (the same
    # thing index_tracks_raw's relative_to_root measures against).
    ds.save()
    return ds


def _trex_npz(path: Path, *, n: int, seed: int) -> None:
    """A minimal per-id TRex NPZ: a time axis plus one pose keypoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    np.savez(
        path,
        time=np.arange(n, dtype=float),
        poseX0=rng.random(n),
        poseY0=rng.random(n),
    )


# --- pure module helpers ---------------------------------------------------


def test_build_tracks_raw_row_assembles_via_to_store_path(tmp_path: Path) -> None:
    f = tmp_path / "seq_fish0.npz"
    f.write_bytes(b"x" * 7)
    row = build_tracks_raw_row(
        path=f,
        stat=f.stat(),
        to_store_path=lambda p: f"STORED/{p.name}",
        group="g",
        sequence="seq",
        src_format="trex_npz",
        md5="abc",
    )
    assert set(row) == set(TRACKS_RAW_INDEX_COLUMNS)
    assert row["abs_path"] == "STORED/seq_fish0.npz"  # produced by to_store_path
    assert row["group"] == "g"
    assert row["sequence"] == "seq"
    assert row["src_format"] == "trex_npz"
    assert row["size_bytes"] == 7
    assert row["md5"] == "abc"
    assert row["mtime_iso"].endswith("+00:00")  # UTC ISO-8601


def test_write_read_round_trip_keeps_columns_and_cells(tmp_path: Path) -> None:
    a = tmp_path / "a.npy"
    a.write_bytes(b"aaaa")
    b = tmp_path / "b.npy"
    b.write_bytes(b"bb")
    rows = [
        build_tracks_raw_row(
            path=p,
            stat=p.stat(),
            to_store_path=lambda q: f"raw/{q.name}",
            group="",
            sequence=p.stem,
            src_format="calms21_npy",
        )
        for p in (a, b)
    ]
    index_path = tmp_path / "index.csv"
    write_tracks_index_rows(index_path, frame_from_rows(rows))

    # Header order is exactly the canonical schema.
    assert list(pd.read_csv(index_path, nrows=0).columns) == TRACKS_RAW_INDEX_COLUMNS
    # Records read back as string cells (drop-in for a csv.DictReader caller).
    records = read_tracks_index(index_path)
    by_seq = {r["sequence"]: r for r in records}
    assert by_seq["a"]["abs_path"] == "raw/a.npy"
    assert by_seq["a"]["size_bytes"] == "4"
    assert by_seq["b"]["size_bytes"] == "2"
    assert by_seq["a"]["md5"] == ""


def test_load_frame_coerces_missing_and_nan_text_cells_to_empty(tmp_path: Path) -> None:
    # A legacy CSV missing the md5 column and with a blank group cell.
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [{"group": "", "sequence": "s", "abs_path": "raw/s.npy", "src_format": "x"}]
    ).to_csv(index_path, index=False)

    frame = load_tracks_index_frame(index_path)
    assert list(frame.columns) == TRACKS_RAW_INDEX_COLUMNS  # missing columns added
    # Text cells are object "" (not float NaN), so later cell writes are clean.
    assert frame.loc[0, "md5"] == ""
    assert frame.loc[0, "group"] == ""


def test_write_is_atomic_leaves_no_temp_orphans_and_overwrites(tmp_path: Path) -> None:
    f = tmp_path / "a.npy"
    f.write_bytes(b"a")
    index_path = tmp_path / "sub" / "index.csv"  # nested: write must mkdir
    row = build_tracks_raw_row(
        path=f,
        stat=f.stat(),
        to_store_path=lambda p: f"raw/{p.name}",
        group="",
        sequence="a",
        src_format="calms21_npy",
    )
    write_tracks_index_rows(index_path, frame_from_rows([row]))
    # Only the final file remains -- no leftover ".<stem>-*.tmp" temp.
    assert [p.name for p in index_path.parent.iterdir()] == ["index.csv"]
    assert len(read_tracks_index(index_path)) == 1

    # A second write fully replaces the file (and still leaves no orphan).
    row2 = {**row, "sequence": "b", "abs_path": "raw/b.npy"}
    write_tracks_index_rows(index_path, frame_from_rows([row2]))
    assert [p.name for p in index_path.parent.iterdir()] == ["index.csv"]
    records = read_tracks_index(index_path)
    assert [r["sequence"] for r in records] == ["b"]


# --- Dataset.index_tracks_raw: the abs_path contract -----------------------


def test_index_tracks_raw_stores_relative_for_in_tree_files(tmp_path: Path) -> None:
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    src = base / "raw_src"
    src.mkdir(parents=True)
    (src / "a.npy").write_bytes(b"aa")
    (src / "b.npy").write_bytes(b"bbb")

    ds.index_tracks_raw([src], patterns=["*.npy"], src_format="calms21_npy")

    rows = read_tracks_index(ds.get_root("tracks_raw") / "index.csv")
    assert len(rows) == 2
    for row in rows:
        assert not Path(row["abs_path"]).is_absolute()  # in-tree -> relative
        assert row["abs_path"] == f"raw_src/{row['sequence']}.npy"
        assert ds.resolve_path(row["abs_path"]).exists()


def test_index_tracks_raw_keeps_external_files_absolute(tmp_path: Path) -> None:
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    external = (tmp_path / "outside").resolve()
    external.mkdir(parents=True)
    (external / "c.npy").write_bytes(b"cccc")

    ds.index_tracks_raw([external], patterns=["*.npy"], src_format="calms21_npy")

    rows = read_tracks_index(ds.get_root("tracks_raw") / "index.csv")
    assert len(rows) == 1
    # Out-of-tree file -> abs_path stays absolute (relative_to_root's fallback).
    assert Path(rows[0]["abs_path"]).is_absolute()
    assert rows[0]["abs_path"] == str(external / "c.npy")
    assert ds.resolve_path(rows[0]["abs_path"]).exists()


def test_index_tracks_raw_trex_suffix_strip_and_group_pattern(tmp_path: Path) -> None:
    # The scan/identity logic is unchanged: per-id TREx files collapse to one
    # sequence and the group pattern still applies -- only the path form changed.
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    src = base / "raw_src"
    src.mkdir(parents=True)
    (src / "hex_7_fish0.npz").write_bytes(b"x")
    (src / "hex_7_fish1.npz").write_bytes(b"x")

    ds.index_tracks_raw(
        [src],
        patterns=["*.npz"],
        src_format="trex_npz",
        group_pattern=r"^(hex)_",
    )

    rows = read_tracks_index(ds.get_root("tracks_raw") / "index.csv")
    assert {r["sequence"] for r in rows} == {"hex_7"}  # _fish0/_fish1 stripped
    assert {r["group"] for r in rows} == {"hex"}
    for row in rows:
        assert not Path(row["abs_path"]).is_absolute()


def test_index_tracks_raw_multi_sequences_per_file_grouping(tmp_path: Path) -> None:
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    bundle_dir = base / "raw_src" / "sessionX"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "bundle.npy").write_bytes(b"x")

    ds.index_tracks_raw(
        [base / "raw_src"],
        patterns=["*.npy"],
        src_format="calms21_npy",
        multi_sequences_per_file=True,
        group_from="parent",
    )

    rows = read_tracks_index(ds.get_root("tracks_raw") / "index.csv")
    assert len(rows) == 1
    assert rows[0]["group"] == "sessionX"  # from parent dir
    assert rows[0]["sequence"] == ""  # blank -- many sequences live in the file
    assert not Path(rows[0]["abs_path"]).is_absolute()


# --- propagation into convert_all_tracks' merged index ---------------------


def test_convert_all_tracks_merge_source_abs_path_is_relative(tmp_path: Path) -> None:
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    src = base / "raw_src"
    _trex_npz(src / "myseq_fish0.npz", n=5, seed=0)
    _trex_npz(src / "myseq_fish1.npz", n=5, seed=1)

    ds.index_tracks_raw([src], patterns=["*.npz"], src_format="trex_npz")
    ds.convert_all_tracks()

    # The two per-id files merged into one standardized parquet.
    out_parquet = ds.get_root("tracks") / "myseq.parquet"
    assert out_parquet.exists()

    tracks_index = pd.read_csv(ds.get_root("tracks") / "index.csv")
    assert len(tracks_index) == 1
    source_abs_path = str(tracks_index.loc[0, "source_abs_path"])
    # Now root-relative and portable (was absolute before) -- matches the
    # non-merge convert_one_track path and resolves against the dataset root.
    assert not Path(source_abs_path).is_absolute()
    assert source_abs_path.startswith("raw_src/")
    assert ds.resolve_path(source_abs_path).exists()
