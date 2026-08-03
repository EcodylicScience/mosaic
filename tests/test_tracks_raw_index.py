"""Tests for the raw-tracks index writer.

Covers the pure serialization helpers in ``mosaic.core.pipeline.tracks_raw_index``,
the root-relative-in-tree / absolute-out-of-tree ``abs_path`` contract and
atomic write that ``Dataset.index_tracks_raw`` now enforces (the scan/identity
logic -- TREx suffix strip, ``multi_sequences_per_file`` grouping -- is asserted
unchanged), and the propagation of that relative form into the merged
``tracks/index.csv`` ``source_abs_path`` written by ``convert_all_tracks``.
"""

import os
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mosaic.core.track_library  # noqa: F401  -- registers the trex_npz converter
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.tracks_raw_index import (
    TRACKS_RAW_INDEX_COLUMNS,
    TracksRawIndexScope,
    build_tracks_raw_row,
    frame_from_rows,
    iter_track_files,
    load_tracks_raw_index_frame,
    read_tracks_raw_index,
    write_tracks_raw_index_rows,
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


def _trex_npz(
    path: Path,
    *,
    n: int,
    seed: int,
    extra: Mapping[str, np.ndarray] | None = None,
) -> None:
    """A minimal per-id TRex NPZ: a time axis plus one pose keypoint.

    *extra* adds fields this individual carries and another need not. TRex writes
    what its ``output_fields`` asked for, per individual, so the several files of
    one sequence are not obliged to agree on their columns.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    np.savez(
        path,
        time=np.arange(n, dtype=float),
        poseX0=rng.random(n),
        poseY0=rng.random(n),
        **dict(extra or {}),
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
    write_tracks_raw_index_rows(index_path, frame_from_rows(rows))

    # Header order is exactly the canonical schema.
    assert list(pd.read_csv(index_path, nrows=0).columns) == TRACKS_RAW_INDEX_COLUMNS
    # Records read back as string cells (drop-in for a csv.DictReader caller).
    records = read_tracks_raw_index(index_path)
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

    frame = load_tracks_raw_index_frame(index_path)
    assert list(frame.columns) == TRACKS_RAW_INDEX_COLUMNS  # missing columns added
    # Text cells are "" (not float NaN), whose str() is the word "nan" -- which
    # every consumer of this frame would otherwise carry into an entry name.
    assert frame.loc[0, "md5"] == ""
    assert frame.loc[0, "group"] == ""


def test_a_numeric_sequence_name_keeps_its_spelling(tmp_path: Path) -> None:
    """Pinning the dtype, not just repairing the blank.

    Inference reads ``001`` as the integer ``1``, and then ``== "001"`` is
    always False -- the failure ``IndexCSV._read_frame`` describes and every
    other index reader avoids. The numeric names are the CalMS21 and MABe
    convention, so this is reachable rather than theoretical.
    """
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "group": "007",
                "sequence": "001",
                "abs_path": "raw/001.npy",
                "src_format": "calms21_npy",
                "size_bytes": 12,
                "mtime_iso": "",
                "md5": "",
            }
        ]
    ).to_csv(index_path, index=False)

    frame = load_tracks_raw_index_frame(index_path)

    assert frame.loc[0, "sequence"] == "001"
    assert frame.loc[0, "group"] == "007"
    # The one numeric column stays numeric.
    assert int(frame.loc[0, "size_bytes"]) == 12


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
    write_tracks_raw_index_rows(index_path, frame_from_rows([row]))
    # Only the final file remains -- no leftover ".<stem>-*.tmp" temp.
    assert [p.name for p in index_path.parent.iterdir()] == ["index.csv"]
    assert len(read_tracks_raw_index(index_path)) == 1

    # A second write fully replaces the file (and still leaves no orphan).
    row2 = {**row, "sequence": "b", "abs_path": "raw/b.npy"}
    write_tracks_raw_index_rows(index_path, frame_from_rows([row2]))
    assert [p.name for p in index_path.parent.iterdir()] == ["index.csv"]
    records = read_tracks_raw_index(index_path)
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

    rows = read_tracks_raw_index(ds.get_root("tracks_raw") / "index.csv")
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

    rows = read_tracks_raw_index(ds.get_root("tracks_raw") / "index.csv")
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

    rows = read_tracks_raw_index(ds.get_root("tracks_raw") / "index.csv")
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

    rows = read_tracks_raw_index(ds.get_root("tracks_raw") / "index.csv")
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

    tracks_index = pd.read_csv(ds.get_root("tracks") / "index.csv")
    assert len(tracks_index) == 1

    # The two per-id files merged into one standardized parquet, under the
    # variant directory this conversion minted.
    out_parquet = ds.resolve_path(str(tracks_index.loc[0, "abs_path"]))
    assert out_parquet.name == "myseq.parquet"
    assert out_parquet.parent.name.startswith("convert-trex_npz.")
    assert out_parquet.exists()
    source_abs_path = str(tracks_index.loc[0, "source_abs_path"])
    # Now root-relative and portable (was absolute before) -- matches the
    # non-merge convert_one_track path and resolves against the dataset root.
    assert not Path(source_abs_path).is_absolute()
    assert source_abs_path.startswith("raw_src/")
    assert ds.resolve_path(source_abs_path).exists()


def test_a_merge_takes_the_union_of_the_columns_its_files_carry(
    tmp_path: Path,
) -> None:
    """Per-individual files need not agree on their columns.

    TRex exports the fields its ``output_fields`` asked for, per individual, so
    one ``.npz`` of a sequence can carry a field another does not. The merge
    aligns on the union and leaves the absent cells NaN, rather than dropping
    the column or refusing the sequence.

    Characterization: this records what the merge does today, so that a change
    to the branch reaching it is either intended and re-blessed here, or a
    defect. The other two merge tests write identical column sets, so nothing
    covered the alignment before.
    """
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    src = base / "raw_src"
    _trex_npz(
        src / "myseq_fish0.npz",
        n=5,
        seed=0,
        extra={"tracklet_id": np.arange(5, dtype=float)},
    )
    _trex_npz(
        src / "myseq_fish1.npz",
        n=5,
        seed=1,
        extra={"blobid": np.arange(100.0, 105.0)},
    )

    ds.index_tracks_raw([src], patterns=["*.npz"], src_format="trex_npz")
    ds.convert_all_tracks()

    tracks_index = pd.read_csv(ds.get_root("tracks") / "index.csv")
    assert len(tracks_index) == 1, "two per-id files are one sequence, one table"
    merged = pd.read_parquet(ds.resolve_path(str(tracks_index.loc[0, "abs_path"])))

    # Every row of both files, under one entry, with both fields present.
    assert len(merged) == 10
    assert {"tracklet_id", "blobid"} <= set(merged.columns)
    assert set(merged["sequence"]) == {"myseq"}
    assert set(merged["id"]) == {0, 1}

    # And NaN exactly where the individual's own file did not carry the field:
    # the field each one exported is filled for it and empty for the other.
    carried = {
        individual: set(rows.dropna(axis="columns", how="all").columns)
        for individual, rows in merged.groupby("id")
    }
    assert "tracklet_id" in carried[0] and "blobid" not in carried[0]
    assert "blobid" in carried[1] and "tracklet_id" not in carried[1]


def test_one_row_of_another_format_does_not_turn_merging_off(
    tmp_path: Path,
) -> None:
    """Merging is each format's own answer, not a property of the whole index.

    It used to be ``(src_format == "trex_npz").all()``, so a single row of any
    other format switched merging off for the TRex rows too -- and the per-id
    files it then converted one at a time all named the same (group, sequence)
    output, so the first landed and the rest were skipped as already written.
    Two individuals silently became one.
    """
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    src = base / "raw_src"
    _trex_npz(src / "myseq_fish0.npz", n=5, seed=0)
    _trex_npz(src / "myseq_fish1.npz", n=5, seed=1)
    ds.index_tracks_raw([src], patterns=["*.npz"], src_format="trex_npz")

    # A row of a non-merging format, as a second index of another source would
    # leave it. Its own conversion is not what is under test -- its presence is.
    index_csv = ds.get_root("tracks_raw") / "index.csv"
    rows = pd.read_csv(index_csv)
    intruder = rows.iloc[[0]].copy()
    intruder["src_format"] = "deeplabcut"
    intruder["sequence"] = "elsewhere"
    pd.concat([rows, intruder], ignore_index=True).to_csv(index_csv, index=False)

    ds.convert_all_tracks()

    tracks_index = pd.read_csv(ds.get_root("tracks") / "index.csv")
    merged_row = tracks_index[tracks_index["sequence"] == "myseq"]
    assert len(merged_row) == 1
    merged = pd.read_parquet(ds.resolve_path(str(merged_row.iloc[0]["abs_path"])))
    assert len(merged) == 10, "both individuals, not just whichever landed first"
    assert set(merged["id"]) == {0, 1}


# --- iter_track_files: the shared deterministic scanner --------------------


def test_iter_track_files_dedups_skips_resource_forks_excludes_and_sorts(
    tmp_path: Path,
) -> None:
    d = tmp_path / "scan"
    d.mkdir()
    (d / "b.npy").write_bytes(b"b")
    (d / "a.npy").write_bytes(b"a")
    (d / "._hidden.npy").write_bytes(b"x")  # macOS resource fork -> skipped
    (d / "skipme.npy").write_bytes(b"s")  # excluded by pattern

    results = iter_track_files(
        [d],
        ["*.npy", "*.np*"],  # overlapping globs -> each file yielded once
        exclude_patterns=["skipme.*"],
    )

    names = [p.name for p, _ in results]
    assert names == ["a.npy", "b.npy"]  # deduped, ._* skipped, excluded, sorted
    assert all(isinstance(st, os.stat_result) for _, st in results)


# --- Dataset.write_tracks_raw_index: the assignment-driven projection ----------


def test_write_tracks_raw_index_assigns_scope_identity_and_stores_relative(
    tmp_path: Path,
) -> None:
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    seq_dir = base / "tracks_raw" / "seqA"
    seq_dir.mkdir(parents=True)
    (seq_dir / "a.npy").write_bytes(b"aa")
    (seq_dir / "b.npy").write_bytes(b"bbb")

    ds.write_tracks_raw_index(
        [
            TracksRawIndexScope(
                directory=seq_dir, group="g", sequence="seqA", src_format="calms21_npy"
            )
        ],
        patterns=["*.npy"],
    )

    rows = ds.read_tracks_raw_index()
    assert {r["sequence"] for r in rows} == {"seqA"}  # every file gets the scope id
    assert {r["group"] for r in rows} == {"g"}
    assert {r["src_format"] for r in rows} == {"calms21_npy"}
    assert {Path(r["abs_path"]).name for r in rows} == {"a.npy", "b.npy"}
    for r in rows:
        assert not Path(r["abs_path"]).is_absolute()  # in-tree -> relative
        assert r["abs_path"] == f"tracks_raw/seqA/{Path(r['abs_path']).name}"
        assert ds.resolve_path(r["abs_path"]).exists()


def test_write_tracks_raw_index_multi_file_sequence_takes_scope_id_no_strip(
    tmp_path: Path,
) -> None:
    # Assignment analog of the _fishN-strip test: per-id files in one scope dir
    # all take the scope's sequence VERBATIM -- no _fishN strip, not the stems.
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    seq_dir = base / "tracks_raw" / "myseq"
    seq_dir.mkdir(parents=True)
    (seq_dir / "myseq_fish0.npz").write_bytes(b"x")
    (seq_dir / "myseq_fish1.npz").write_bytes(b"x")

    ds.write_tracks_raw_index(
        [
            TracksRawIndexScope(
                directory=seq_dir, group="", sequence="myseq", src_format="trex_npz"
            )
        ],
        patterns=["*.npz"],
    )

    rows = ds.read_tracks_raw_index()
    assert len(rows) == 2
    assert {r["sequence"] for r in rows} == {"myseq"}  # not "myseq_fish0"/"_fish1"


def test_write_tracks_raw_index_preserves_other_and_external_rows(
    tmp_path: Path,
) -> None:
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    index_path = ds.get_root("tracks_raw") / "index.csv"

    # Seed: one already-indexed in-tree sequence (seqB) and one external ref.
    seeded: dict[str, object] = {column: "" for column in TRACKS_RAW_INDEX_COLUMNS}
    other = {
        **seeded,
        "sequence": "seqB",
        "abs_path": "tracks_raw/seqB/x.npy",
        "src_format": "calms21_npy",
        "size_bytes": 3,
    }
    external = {
        **seeded,
        "sequence": "remote",
        "abs_path": "/mnt/nas/clip.npy",
        "src_format": "calms21_npy",
        "size_bytes": 9,
    }
    write_tracks_raw_index_rows(index_path, frame_from_rows([other, external]))

    seq_dir = base / "tracks_raw" / "seqA"
    seq_dir.mkdir(parents=True)
    (seq_dir / "a.npy").write_bytes(b"aa")
    ds.write_tracks_raw_index(
        [
            TracksRawIndexScope(
                directory=seq_dir, group="", sequence="seqA", src_format="calms21_npy"
            )
        ],
        patterns=["*.npy"],
    )

    abs_paths = {r["abs_path"] for r in ds.read_tracks_raw_index()}
    assert "tracks_raw/seqA/a.npy" in abs_paths  # freshly stamped, relative
    assert "tracks_raw/seqB/x.npy" in abs_paths  # other sequence preserved
    assert "/mnt/nas/clip.npy" in abs_paths  # external preserved, still absolute


def test_write_tracks_raw_index_reimport_replaces_a_scopes_rows(tmp_path: Path) -> None:
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    seq_dir = base / "tracks_raw" / "seqA"
    seq_dir.mkdir(parents=True)
    (seq_dir / "a.npy").write_bytes(b"a")
    scope = TracksRawIndexScope(
        directory=seq_dir, group="", sequence="seqA", src_format="calms21_npy"
    )
    ds.write_tracks_raw_index([scope], patterns=["*.npy"])

    # Add a second file and re-import the same scope: both present, no stale/dup.
    (seq_dir / "b.npy").write_bytes(b"bb")
    ds.write_tracks_raw_index([scope], patterns=["*.npy"])

    rows = ds.read_tracks_raw_index()
    assert sorted(Path(r["abs_path"]).name for r in rows) == ["a.npy", "b.npy"]
    assert len(rows) == 2


def test_write_tracks_raw_index_external_scope_dir_stays_absolute(
    tmp_path: Path,
) -> None:
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    external = (tmp_path / "outside").resolve()
    external.mkdir(parents=True)
    (external / "c.npy").write_bytes(b"cccc")

    ds.write_tracks_raw_index(
        [
            TracksRawIndexScope(
                directory=external, group="", sequence="ext", src_format="calms21_npy"
            )
        ],
        patterns=["*.npy"],
    )

    rows = ds.read_tracks_raw_index()
    assert len(rows) == 1
    assert Path(rows[0]["abs_path"]).is_absolute()  # out-of-tree -> absolute
    assert rows[0]["abs_path"] == str(external / "c.npy")


def test_write_tracks_raw_index_hashes_by_default_and_can_be_turned_off(
    tmp_path: Path,
) -> None:
    """On by default, because the ``tracks_raw`` composition is over these.

    The old default was False and nothing in the toolkit ever passed True, so
    the column was empty in every real dataset -- which would leave every
    sequence's composition unestablishable. ``--no-md5`` stays for a corpus too
    slow to hash, and what it buys is an honest empty rather than a wrong value.
    """
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    seq_dir = base / "tracks_raw" / "seqA"
    seq_dir.mkdir(parents=True)
    (seq_dir / "a.npy").write_bytes(b"payload")
    scope = TracksRawIndexScope(
        directory=seq_dir, group="", sequence="seqA", src_format="calms21_npy"
    )

    ds.write_tracks_raw_index([scope], patterns=["*.npy"])
    assert ds.read_tracks_raw_index()[0]["md5"] != ""

    ds.write_tracks_raw_index([scope], patterns=["*.npy"], compute_md5=False)
    assert ds.read_tracks_raw_index()[0]["md5"] == ""


def test_write_tracks_raw_index_carries_a_digest_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unchanged file is not re-hashed, which is what makes the default affordable.

    Every scope directory is rescanned on every write, so without the
    carry-forward a re-finalize would re-read every byte of a sequence to
    reproduce digests it already had.
    """
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    seq_dir = base / "tracks_raw" / "seqA"
    seq_dir.mkdir(parents=True)
    (seq_dir / "a.npy").write_bytes(b"payload")
    scope = TracksRawIndexScope(
        directory=seq_dir, group="", sequence="seqA", src_format="calms21_npy"
    )

    ds.write_tracks_raw_index([scope], patterns=["*.npy"])
    first = ds.read_tracks_raw_index()[0]["md5"]

    def _refuse(path: Path, chunk: int = 1 << 20) -> str:
        raise AssertionError(f"re-hashed an unchanged file: {path}")

    monkeypatch.setattr("mosaic.core.dataset._md5", _refuse)
    ds.write_tracks_raw_index([scope], patterns=["*.npy"])
    assert ds.read_tracks_raw_index()[0]["md5"] == first

    # A changed file has a new size, so the carried digest is not reused.
    monkeypatch.undo()
    (seq_dir / "a.npy").write_bytes(b"a longer payload than before")
    ds.write_tracks_raw_index([scope], patterns=["*.npy"])
    assert ds.read_tracks_raw_index()[0]["md5"] != first


def test_write_tracks_raw_index_empty_scopes_rewrites_existing_verbatim(
    tmp_path: Path,
) -> None:
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    index_path = ds.get_root("tracks_raw") / "index.csv"
    seeded: dict[str, object] = {column: "" for column in TRACKS_RAW_INDEX_COLUMNS}
    row = {
        **seeded,
        "sequence": "seqB",
        "abs_path": "tracks_raw/seqB/x.npy",
        "src_format": "calms21_npy",
        "size_bytes": 3,
    }
    write_tracks_raw_index_rows(index_path, frame_from_rows([row]))

    ds.write_tracks_raw_index([], patterns=["*.npy"])  # no scopes -> idempotent

    rows = ds.read_tracks_raw_index()
    assert len(rows) == 1
    assert rows[0]["sequence"] == "seqB"


# --- index_tracks_raw refactor regression ----------------------------------


def test_index_tracks_raw_skips_resource_forks_and_sorts(tmp_path: Path) -> None:
    # The shared scanner now skips macOS ._* files and yields deterministically
    # sorted output -- both benign improvements over the prior inline scan.
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    src = base / "raw_src"
    src.mkdir(parents=True)
    (src / "b.npy").write_bytes(b"bb")
    (src / "a.npy").write_bytes(b"a")
    (src / "._a.npy").write_bytes(b"x")  # resource fork -> skipped

    ds.index_tracks_raw([src], patterns=["*.npy"], src_format="calms21_npy")

    rows = ds.read_tracks_raw_index()
    names = [Path(r["abs_path"]).name for r in rows]
    assert "._a.npy" not in names  # resource fork skipped
    assert names == ["a.npy", "b.npy"]  # deterministically sorted


# --- group_from_path: a grouping that is a rule, not a substring -------------


def _guppy_group(path: Path) -> str:
    """The guppies rule: condition from the name, phase from the day number."""
    day = int(re.findall(r"_d(\d+)_", path.stem)[0])
    condition = "control" if "_control_" in path.stem else "exp"
    return f"{condition}_{'baseline' if day == 1 else 'treatment'}"


def test_group_from_path_expresses_a_rule_a_regex_cannot(tmp_path: Path) -> None:
    # `day == 1 -> baseline, else treatment` is a conditional over a captured
    # value, which `group_pattern` (one capturing group, lifted verbatim) cannot
    # express. Before this seam such a dataset had to patch index.csv after the
    # fact -- which conversion could not see and the next re-index undid.
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    src = base / "raw_src"
    src.mkdir(parents=True)
    for name in (
        "guppy_8_t10_d10_control_20191002_115117_fish0.npz",
        "guppy_8_t10_d10_control_20191002_115117_fish1.npz",
        "guppy_8_t13_d1_20190929_111944_fish0.npz",
        "guppy_8_t2_d1_control_20190923_104858_fish0.npz",
        "guppy_8_t5_d10_20191003_092208_fish0.npz",
    ):
        (src / name).write_bytes(b"x")

    ds.index_tracks_raw(
        [src],
        patterns=["*.npz"],
        src_format="trex_npz",
        group_from_path=_guppy_group,
    )

    rows = read_tracks_raw_index(ds.get_root("tracks_raw") / "index.csv")
    by_sequence = {r["sequence"]: r["group"] for r in rows}
    assert by_sequence == {
        "guppy_8_t10_d10_control_20191002_115117": "control_treatment",
        "guppy_8_t13_d1_20190929_111944": "exp_baseline",
        "guppy_8_t2_d1_control_20190923_104858": "control_baseline",
        "guppy_8_t5_d10_20191003_092208": "exp_treatment",
    }
    # The per-id files still collapse to one sequence: the rule sets the group
    # and leaves the sequence derivation alone.
    assert len(rows) == 5


def test_group_from_path_reaches_the_composition_projection(tmp_path: Path) -> None:
    # The reason patching index.csv by hand was never enough: sequences.csv is
    # keyed by (group, sequence), and it is what the composition hash reads.
    from mosaic.core.pipeline.sequence_index import read_sequence_index

    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    src = base / "raw_src"
    src.mkdir(parents=True)
    (src / "guppy_1_t1_d1_control_20190101_000000_fish0.npz").write_bytes(b"x")

    ds.index_tracks_raw(
        [src], patterns=["*.npz"], src_format="trex_npz", group_from_path=_guppy_group
    )

    compositions = read_sequence_index(ds, "tracks_raw")
    assert set(compositions["group"]) == {"control_baseline"}


def test_group_from_path_supersedes_group_from(tmp_path: Path) -> None:
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    bundle = base / "raw_src" / "sessionX"
    bundle.mkdir(parents=True)
    (bundle / "bundle.npy").write_bytes(b"x")

    ds.index_tracks_raw(
        [base / "raw_src"],
        patterns=["*.npy"],
        src_format="calms21_npy",
        multi_sequences_per_file=True,
        group_from="parent",
        group_from_path=lambda p: f"cohort_{p.parent.name}",
    )

    rows = read_tracks_raw_index(ds.get_root("tracks_raw") / "index.csv")
    assert [r["group"] for r in rows] == ["cohort_sessionX"]
    assert [r["sequence"] for r in rows] == [""]


def test_group_from_path_and_group_pattern_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    src = base / "raw_src"
    src.mkdir(parents=True)
    (src / "hex_7_fish0.npz").write_bytes(b"x")

    with pytest.raises(ValueError, match="not both"):
        ds.index_tracks_raw(
            [src],
            patterns=["*.npz"],
            src_format="trex_npz",
            group_pattern=r"^(hex)_",
            group_from_path=lambda p: "hex",
        )


def test_group_from_path_error_propagates(tmp_path: Path) -> None:
    # A file the rule cannot classify is an error worth seeing, not a silent "".
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    src = base / "raw_src"
    src.mkdir(parents=True)
    (src / "unparseable.npz").write_bytes(b"x")

    def strict(path: Path) -> str:
        raise ValueError(f"cannot classify {path.name}")

    with pytest.raises(ValueError, match="cannot classify unparseable.npz"):
        ds.index_tracks_raw(
            [src], patterns=["*.npz"], src_format="trex_npz", group_from_path=strict
        )


def test_group_from_path_rejects_a_group_with_a_separator(tmp_path: Path) -> None:
    # An entry name is one path component; the write boundary still enforces it.
    base = (tmp_path / "ds").resolve()
    ds = _make_dataset(base)
    src = base / "raw_src"
    src.mkdir(parents=True)
    (src / "a_fish0.npz").write_bytes(b"x")

    with pytest.raises(ValueError):
        ds.index_tracks_raw(
            [src],
            patterns=["*.npz"],
            src_format="trex_npz",
            group_from_path=lambda p: "control/baseline",
        )
