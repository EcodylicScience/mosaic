"""Tests for the raw-tracks index writer.

Covers the pure serialization helpers in ``mosaic.core.pipeline.tracks_raw_index``,
the root-relative-in-tree / absolute-out-of-tree ``abs_path`` contract and
atomic write that ``Dataset.index_tracks_raw`` now enforces (the scan/identity
logic -- TREx suffix strip, ``multi_sequences_per_file`` grouping -- is asserted
unchanged), and the propagation of that relative form into the merged
``tracks/index.csv`` ``source_abs_path`` written by ``convert_all_tracks``.

Every dataset here is saved before it is used, because ``index_tracks_raw``
measures its root-relative ``abs_path`` against ``base_dir`` -- which an unsaved
manifest resolves to the manifest path rather than its parent.
"""

import fnmatch
import hashlib
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mosaic.core.track_library  # noqa: F401  -- registers the trex_npz converter
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.composition import SourceMember, tracks_raw_composition
from mosaic.core.pipeline.definitions.store import is_under_definitions_root
from mosaic.core.pipeline.label_series import is_under_label_series
from mosaic.core.pipeline.scan_claim import ScanClaim
from mosaic.core.pipeline.sequence_index import read_sequence_index
from mosaic.core.pipeline.tracking_roots import is_under_tracking_root
from mosaic.core.pipeline.tracks_raw_index import (
    TRACKS_RAW_INDEX_COLUMNS,
    CompanionRecord,
    RawEntryNamedTwice,
    TracksRawIndexScope,
    build_tracks_raw_row,
    companions_cell,
    companions_of,
    entries_named_twice,
    frame_from_rows,
    group_for_path,
    iter_track_files,
    load_tracks_raw_index_frame,
    no_companions,
    read_tracks_raw_index,
    sequence_rule,
    source_file_filter,
    source_members_from_rows,
    stem_as_sequence,
    write_tracks_raw_index_rows,
)

from tests.helpers import (
    TRACKTOR_STEM,
    make_dataset,
    raw_index_files,
    set_companions_cell,
    tracktor_dataset,
    write_calms21_npy,
    write_tracktor_recording,
    write_tracktor_recordings,
    write_trex_npz,
)

# The raw uploads and the tables converted from them, and nothing else: no test
# here reads media, and a root a test never touches would only obscure which
# ``abs_path`` is being measured against what.
RAW_AND_CONVERTED = ("tracks_raw", "tracks")


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
    write_trex_npz(
        path,
        n=n,
        columns={
            "time": np.arange(n, dtype=float),
            "poseX0": rng.random(n),
            "poseY0": rng.random(n),
            **(extra or {}),
        },
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
    # "No leftover '.<stem>-*.tmp' orphan" is what this pins -- not the exact
    # directory contents. ``index_lock`` legitimately puts an ``index.csv.lock``
    # sidecar beside an index; this writer takes no lock (its callers do), so
    # none appears here, and pinning the temp's absence is what stays true
    # either way.
    assert [p.name for p in index_path.parent.iterdir() if p.suffix == ".tmp"] == []
    assert len(read_tracks_raw_index(index_path)) == 1

    # A second write fully replaces the file (and still leaves no orphan).
    row2 = {**row, "sequence": "b", "abs_path": "raw/b.npy"}
    write_tracks_raw_index_rows(index_path, frame_from_rows([row2]))
    assert [p.name for p in index_path.parent.iterdir() if p.suffix == ".tmp"] == []
    records = read_tracks_raw_index(index_path)
    assert [r["sequence"] for r in records] == ["b"]


# --- Dataset.index_tracks_raw: the abs_path contract -----------------------


def test_index_tracks_raw_stores_relative_for_in_tree_files(tmp_path: Path) -> None:
    base = (tmp_path / "ds").resolve()
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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


# --- iter_track_files: the walk and the per-file predicate agree ------------


def _reference_walk(
    search_dirs: list[Path],
    patterns: list[str],
    *,
    recursive: bool,
    exclude_patterns: list[str],
) -> list[tuple[Path, os.stat_result]]:
    """The walk as it was: one ``rglob(pattern)`` or ``glob(pattern)`` per pattern."""
    by_resolved: dict[Path, tuple[Path, os.stat_result]] = {}
    for directory in search_dirs:
        for pattern in patterns:
            matches = directory.rglob(pattern) if recursive else directory.glob(pattern)
            for path in matches:
                if not path.is_file():
                    continue
                if path.name.startswith("._"):
                    continue
                if any(fnmatch.fnmatch(path.name, ex) for ex in exclude_patterns):
                    continue
                if is_under_tracking_root(path.parts):
                    continue
                if is_under_definitions_root(path.parts):
                    continue
                if is_under_label_series(path):
                    continue
                by_resolved.setdefault(path.resolve(), (path, path.stat()))
    return [by_resolved[key] for key in sorted(by_resolved)]


def _fixture_tree(root: Path) -> list[Path]:
    """A tree that exercises every rule of the walk; returns every file in it."""
    files = [
        "top.csv",
        "top.npz",
        "top.h5",
        "xray.h5",
        "xray.csv",
        "._top.csv",
        "skip_me.csv",
        ".hidden.csv",
        "a/one.csv",
        "a/two.npz",
        "a/._fork.csv",
        "a/skip_me.csv",
        "a/b/three.csv",
        "a/b/three.npz",
        "a/b/four.h5",
        "b/five.npz",
        "b/five.csv",
        "b/deep/six.npz",
        "b/deep/b/seven.npz",
        "b/deep/b/seven.csv",
        "x1/eight.h5",
        "x1/eight.csv",
        "x2/nine/ten.h5",
        "xx/eleven.h5",
        "docs/readme.txt",
        "docs/.dot/twelve.csv",
        "_tracking/trex/run/data/thirteen.npz",
        "_tracking/fourteen.csv",
        "a/_tracking/fifteen.csv",
        "feature_definitions/sixteen.csv",
        "a/feature_definitions/seventeen.csv",
        "series/.mosaic-series",
        "series/index.csv",
        "series/key/rev1/manifest.json",
        "series/key/rev1/labels.csv",
        "series/key/rev1/b/deep.npz",
        "series/other.csv",
        "_trackingx/eighteen.csv",
        "keypoints/mine.csv",
    ]
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(name.encode())
    return [root / name for name in files]


_WALK_PATTERNS: list[list[str]] = [
    ["*.csv"],
    ["*.csv", "*.npz"],
    ["a/*.csv"],
    ["**/b/*.npz", "*.csv"],
    ["x*/*.h5"],
    ["*/*.h5"],
    ["b/*/*.npz"],
    ["*.h5", "*.H5"],
    ["**/*.npz"],
    ["x?/*/*.h5"],
]
_WALK_EXCLUDES: list[list[str]] = [[], ["skip_me.*"], ["skip_me.*", "*.h5"]]


@pytest.mark.parametrize("exclude_patterns", _WALK_EXCLUDES, ids=str)
@pytest.mark.parametrize("recursive", [True, False], ids=["recursive", "flat"])
@pytest.mark.parametrize("patterns", _WALK_PATTERNS, ids=str)
def test_the_walk_returns_what_the_rglob_walk_returned(
    tmp_path: Path,
    patterns: list[str],
    recursive: bool,
    exclude_patterns: list[str],
) -> None:
    base = (tmp_path / "ds").resolve()
    _ = _fixture_tree(base)

    expected = _reference_walk(
        [base], patterns, recursive=recursive, exclude_patterns=exclude_patterns
    )
    found = iter_track_files(
        [base], patterns, recursive=recursive, exclude_patterns=exclude_patterns
    )

    assert [(p, s.st_ino, s.st_size, s.st_mtime_ns) for p, s in found] == [
        (p, s.st_ino, s.st_size, s.st_mtime_ns) for p, s in expected
    ]


def test_the_reference_walk_has_files_to_compare(tmp_path: Path) -> None:
    base = (tmp_path / "ds").resolve()
    _ = _fixture_tree(base)

    names = [
        path.relative_to(base).as_posix()
        for path, _stat in iter_track_files([base], ["*.csv"])
    ]

    assert {
        ".hidden.csv",
        "a/b/three.csv",
        "docs/.dot/twelve.csv",
        "keypoints/mine.csv",
        "_trackingx/eighteen.csv",
    } <= set(names)
    assert "top.csv" in names
    assert "._top.csv" not in names
    assert "_tracking/fourteen.csv" not in names
    assert "series/other.csv" not in names
    assert "feature_definitions/sixteen.csv" not in names


@pytest.mark.parametrize("exclude_patterns", _WALK_EXCLUDES, ids=str)
@pytest.mark.parametrize("recursive", [True, False], ids=["recursive", "flat"])
@pytest.mark.parametrize("patterns", _WALK_PATTERNS, ids=str)
def test_the_filter_accepts_what_the_walk_returns(
    tmp_path: Path,
    patterns: list[str],
    recursive: bool,
    exclude_patterns: list[str],
) -> None:
    base = (tmp_path / "ds").resolve()
    files = _fixture_tree(base)
    walked = {
        path.resolve()
        for path, _stat in _reference_walk(
            [base], patterns, recursive=recursive, exclude_patterns=exclude_patterns
        )
    }

    accepts = source_file_filter(
        base, patterns, recursive=recursive, exclude_patterns=exclude_patterns
    )

    assert {path.resolve() for path in files if accepts.accepts(path.resolve())} == (
        walked
    )


@pytest.mark.parametrize("recursive", [True, False], ids=["recursive", "flat"])
@pytest.mark.parametrize(
    "patterns", [["*.csv"], ["a/*.csv"], ["link/*.csv"], ["*/*.csv"]], ids=str
)
def test_the_walk_follows_a_symlinked_folder_as_the_rglob_walk_did(
    tmp_path: Path, patterns: list[str], recursive: bool
) -> None:
    real = tmp_path / "real"
    (real / "a").mkdir(parents=True)
    _ = (real / "a" / "one.csv").write_bytes(b"one")
    _ = (real / "top.csv").write_bytes(b"top")
    base = (tmp_path / "ds").resolve()
    base.mkdir()
    (base / "link").symlink_to(real, target_is_directory=True)
    (base / "a").symlink_to(real / "a", target_is_directory=True)

    expected = _reference_walk(
        [base], patterns, recursive=recursive, exclude_patterns=[]
    )
    found = iter_track_files([base], patterns, recursive=recursive)

    assert [path for path, _stat in found] == [path for path, _stat in expected]
    if patterns != ["*.csv"]:
        assert found, "the symlinked folder must give the comparison something"


@pytest.mark.parametrize("recursive", [True, False], ids=["recursive", "flat"])
def test_the_walk_lists_only_what_a_pattern_matches(
    tmp_path: Path, recursive: bool
) -> None:
    base = (tmp_path / "ds").resolve()
    for folder in ("", "a/", "a/b/"):
        for number in range(30):
            path = base / f"{folder}noise{number}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_bytes(b"")
    matching = [base / "one.npz", base / "a" / "two.npz", base / "a" / "b" / "3.npz"]
    for path in matching:
        _ = path.write_bytes(b"")
    accepts = source_file_filter(
        base, ["*.npz"], recursive=recursive, exclude_patterns=[]
    )

    listed = [path for path in accepts.candidates(base) if path.is_file()]

    expected = matching if recursive else matching[:1]
    assert sorted(listed) == sorted(expected)
    assert len(list(accepts.candidates(base))) == len(expected)


def test_the_walk_of_a_symlinked_root_agrees_with_the_filter_on_resolved_paths(
    tmp_path: Path,
) -> None:
    real = (tmp_path / "real").resolve()
    _ = _fixture_tree(real)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    found = iter_track_files([link], ["**/b/*.npz", "*.csv"])
    accepts = source_file_filter(link, ["**/b/*.npz", "*.csv"], recursive=True)

    assert found
    assert all(path.is_relative_to(link) for path, _stat in found)
    assert [path.resolve() for path, _stat in found] == [
        path.resolve()
        for path, _stat in _reference_walk(
            [link], ["**/b/*.npz", "*.csv"], recursive=True, exclude_patterns=[]
        )
    ]
    assert all(accepts.accepts(path.resolve()) for path, _stat in found)


def test_the_filter_rejects_a_path_outside_its_base(tmp_path: Path) -> None:
    base = (tmp_path / "ds").resolve()
    _ = _fixture_tree(base)
    sibling = tmp_path / "ds2" / "top.csv"
    sibling.parent.mkdir()
    _ = sibling.write_bytes(b"")

    accepts = source_file_filter(base, ["*.csv"], recursive=True, exclude_patterns=[])

    assert accepts.accepts(base / "top.csv")
    assert not accepts.accepts(sibling.resolve())
    assert not accepts.accepts(base.parent / "top.csv")


def test_the_walk_keeps_a_listed_file_whatever_the_patterns_say(
    tmp_path: Path,
) -> None:
    listed = tmp_path / "calls.json"
    _ = listed.write_bytes(b"{}")
    forked = tmp_path / "._calls.json"
    _ = forked.write_bytes(b"{}")
    skipped: list[Path] = []

    found = iter_track_files(
        [listed, forked], ["*.csv"], recursive=False, listed_skipped=skipped
    )

    assert [path for path, _stat in found] == [listed]
    assert skipped == [forked]


# --- Dataset.write_tracks_raw_index: the assignment-driven projection ----------


def test_write_tracks_raw_index_assigns_scope_identity_and_stores_relative(
    tmp_path: Path,
) -> None:
    base = (tmp_path / "ds").resolve()
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
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


class TestGroupForPath:
    """The one rule that decides a file's group, and whether the pattern missed it."""

    @staticmethod
    def decide(
        path: str,
        *,
        group_re: str | None = None,
        path_re: str | None = None,
        multi: bool = False,
        group_from: str | None = None,
        group_from_path: Callable[[Path], str] | None = None,
        sequence_from_stem: Callable[[str], str] = stem_as_sequence,
        sequence_path_pattern: str | None = None,
    ) -> str | None:
        return group_for_path(
            Path("/src") / path,
            sequence_rule=sequence_rule(
                sequence_from_stem=sequence_from_stem,
                sequence_path_pattern=sequence_path_pattern,
                path_root=Path("/src"),
                source="s",
            ),
            multi_sequences_per_file=multi,
            group_from=group_from,
            group_re=None if group_re is None else re.compile(group_re),
            path_re=None if path_re is None else re.compile(path_re),
            path_root=Path("/src"),
            group_from_path=group_from_path,
        )

    def test_no_pattern_is_the_empty_group_and_not_a_miss(self) -> None:
        assert self.decide("a/x.npy") == ""

    def test_a_stem_pattern_lifts_its_group_or_misses(self) -> None:
        assert self.decide("hex_1.npy", group_re=r"^(hex)_") == "hex"
        assert self.decide("plain.npy", group_re=r"^(hex)_") is None

    def test_a_stem_pattern_searches_the_converters_sequence(self) -> None:
        def strip(stem: str) -> str:
            return stem.removesuffix("_fish0")

        assert (
            self.decide("hex_fish0.npz", group_re=r"^(hex)$", sequence_from_stem=strip)
            == "hex"
        )

    def test_a_path_pattern_searches_the_path_below_the_root(self) -> None:
        assert self.decide("g1/x.npy", path_re=r"^([^/]+)/") == "g1"
        assert self.decide("x.npy", path_re=r"^([^/]+)/") is None

    def test_a_multi_sequence_file_takes_the_group_from_its_name_or_folder(
        self,
    ) -> None:
        assert self.decide("s1/x.npy", multi=True, group_from="parent") == "s1"
        assert self.decide("s1/x.npy", multi=True, group_from="filename") == "x"
        assert self.decide("s1/x.npy", multi=True) == ""

    def test_a_group_rule_never_misses_a_file_whose_sequence_it_names(self) -> None:
        assert self.decide("x.npy", group_from_path=lambda path: "g") == "g"

    def test_a_path_the_sequence_rule_misses_has_no_group(self) -> None:
        pattern = r"^[^/]+/(run\d+)/"

        assert self.decide("g1/run1/x.npy", sequence_path_pattern=pattern) == ""
        assert self.decide("g1/other/x.npy", sequence_path_pattern=pattern) is None
        assert (
            self.decide(
                "g1/other/x.npy",
                sequence_path_pattern=pattern,
                group_from_path=lambda path: "g",
            )
            is None
        )

    def test_a_stem_pattern_searches_the_sequence_the_path_pattern_names(
        self,
    ) -> None:
        pattern = r"^[^/]+/(run\d+)/([^/]+)/"

        found = self.decide(
            "g1/run1/trial/x.npy", sequence_path_pattern=pattern, group_re=r"^(run1)__"
        )

        assert found == "run1"
        assert (
            self.decide(
                "g1/run2/trial/x.npy",
                sequence_path_pattern=pattern,
                group_re=r"^(run1)__",
            )
            is None
        )


class TestSequenceRule:
    """One function of a path names the sequence, from a pattern or from the stem."""

    ROOT = Path("/src")

    def rule(self, pattern: str | None = None) -> Callable[[Path], str | None]:
        return sequence_rule(
            sequence_from_stem=lambda stem: stem.removesuffix("_fish0"),
            sequence_path_pattern=pattern,
            path_root=self.ROOT,
            source="s",
        )

    def test_without_a_pattern_it_is_the_converters_stem_rule(self) -> None:
        rule = self.rule()

        for name in ("hex_7_fish0.npz", "plain.npy", "a/b/c_fish0.csv"):
            path = self.ROOT / name
            assert rule(path) == path.stem.removesuffix("_fish0")

    def test_without_a_pattern_the_stem_rule_is_ignored_by_the_folders(self) -> None:
        assert self.rule()(self.ROOT / "deep" / "er" / "x.npy") == "x"

    def test_the_capture_groups_join_with_two_underscores_in_order(self) -> None:
        rule = self.rule(r"^[^/]+/([^/]+)/(day[^/]+)/([^/]+)/segm/t\.mat$")

        found = rule(self.ROOT / "1/batchA/day2/trial7/segm/t.mat")

        assert found == "batchA__day2__trial7"

    def test_a_path_the_pattern_misses_has_no_sequence(self) -> None:
        assert self.rule(r"^([^/]+)/x\.npy$")(self.ROOT / "a/b/x.npy") is None

    def test_the_pattern_searches_the_path_below_the_root_only(self) -> None:
        rule = self.rule(r"^([^/]+)/x\.npy$")

        assert rule(self.ROOT / "g1" / "x.npy") == "g1"
        assert rule(Path("/elsewhere/g1/x.npy")) is None

    @pytest.mark.parametrize(
        ("pattern", "path"),
        [(r"(a)?(b)", "b/x.npy"), (r"(x*)", "g/x.npy")],
        ids=["optional group not taking part", "group matching empty"],
    )
    def test_a_capture_that_names_nothing_is_refused_naming_file_source_pattern(
        self, pattern: str, path: str
    ) -> None:
        with pytest.raises(ValueError) as caught:
            _ = self.rule(pattern)(self.ROOT / path)

        message = str(caught.value)
        assert str(self.ROOT / path) in message
        assert "'s'" in message
        assert pattern in message

    def test_a_pattern_without_a_capture_group_is_refused(self) -> None:
        with pytest.raises(ValueError, match="capture group"):
            _ = self.rule(r"x\.npy$")

    def test_a_pattern_that_is_not_a_regular_expression_is_refused_by_name(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="not a regular expression") as caught:
            _ = self.rule("(unclosed")

        assert "'s'" in str(caught.value)
        assert "(unclosed" in str(caught.value)
        assert isinstance(caught.value.__cause__, re.error)

    def test_a_pattern_needs_a_root(self) -> None:
        with pytest.raises(ValueError, match="group_path_root"):
            _ = sequence_rule(
                sequence_from_stem=stem_as_sequence,
                sequence_path_pattern=r"(a)",
                path_root=None,
                source="s",
            )


# --- the companions column -------------------------------------------------

STAMP = "2026-01-02T03:04:05.000006+00:00"
"""A modification time as the index spells one."""

RAW_NAME = f"{TRACKTOR_STEM}_tracked.csv"
CORRECTED_NAME = f"{TRACKTOR_STEM}_tracked_fixed_final.tsv"


def _md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _companions_in(
    ds: Dataset, root: str = "tracks_raw"
) -> dict[str, tuple[CompanionRecord, ...]]:
    """Each indexed file's companion records, keyed by the file's basename."""
    return {
        Path(row["abs_path"]).name: companions_of(row)
        for row in read_tracks_raw_index(ds.get_root(root) / "index.csv")
    }


def test_a_companions_cell_is_compact_json_sorted_by_name() -> None:
    records = (
        CompanionRecord(name="b.tsv", size_bytes=7, mtime_iso=STAMP, md5="d2"),
        CompanionRecord(name="a.tsv", size_bytes=3, mtime_iso=STAMP, md5="d1"),
    )

    cell = companions_cell(records)

    assert cell == (
        f'[{{"name":"a.tsv","size_bytes":3,"mtime_iso":"{STAMP}","md5":"d1"}},'
        f'{{"name":"b.tsv","size_bytes":7,"mtime_iso":"{STAMP}","md5":"d2"}}]'
    )
    assert companions_cell(()) == ""


def test_a_companions_cell_round_trips_through_a_written_index(tmp_path: Path) -> None:
    source = tmp_path / "a.csv"
    source.write_bytes(b"x")
    records = (
        CompanionRecord(name="b.tsv", size_bytes=7, mtime_iso=STAMP, md5="d2"),
        CompanionRecord(name="a.tsv", size_bytes=3, mtime_iso=STAMP, md5=""),
    )
    row = build_tracks_raw_row(
        path=source,
        stat=source.stat(),
        to_store_path=lambda path: path.name,
        group="",
        sequence="a",
        src_format="tracktor_raw_csv",
        companions=records,
    )
    index_path = tmp_path / "index.csv"

    write_tracks_raw_index_rows(index_path, frame_from_rows([row]))

    assert row["companions"] == companions_cell(records)
    (record,) = read_tracks_raw_index(index_path)
    assert companions_of(record) == tuple(sorted(records, key=lambda r: r.name))
    assert list(load_tracks_raw_index_frame(index_path)["companions"]) == [
        companions_cell(records)
    ]


@pytest.mark.parametrize("cell", ["", None, float("nan")], ids=["blank", "none", "nan"])
def test_a_blank_absent_or_missing_cell_lists_no_companions(
    cell: str | float | None,
) -> None:
    assert companions_of({"abs_path": "a.csv", "companions": cell}) == ()
    assert companions_of({"abs_path": "a.csv"}) == ()


@pytest.mark.parametrize(
    "cell",
    [
        "not json",
        "{}",
        "[1]",
        '[{"name":"a.tsv"}]',
        '[{"name":"a.tsv","size_bytes":"3","mtime_iso":"x","md5":"y"}]',
        '[{"name":"","size_bytes":3,"mtime_iso":"x","md5":"y"}]',
        '[{"name":"a.tsv","size_bytes":3,"mtime_iso":"x","md5":"y","extra":1}]',
        7,
    ],
)
def test_a_malformed_cell_raises_and_names_the_file(cell: str | int) -> None:
    with pytest.raises(ValueError, match=re.escape("raw/some_file.csv")):
        _ = companions_of({"abs_path": "raw/some_file.csv", "companions": cell})


def test_an_index_written_without_the_column_reads_a_blank_cell(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [{"group": "", "sequence": "s", "abs_path": "raw/s.npy", "src_format": "x"}]
    ).to_csv(index_path, index=False)

    frame = load_tracks_raw_index_frame(index_path)

    assert frame.loc[0, "companions"] == ""


def test_rows_missing_a_text_cell_carry_a_blank_and_keep_a_numeric_size() -> None:
    frame = frame_from_rows(
        [{"group": "g", "sequence": "s", "abs_path": "p", "size_bytes": 3}]
    )

    assert frame.loc[0, "companions"] == ""
    assert frame.loc[0, "md5"] == ""
    assert frame.loc[0, "size_bytes"] == 3
    assert pd.api.types.is_numeric_dtype(frame["size_bytes"])


def _two_source_dataset(tmp_path: Path) -> tuple[Dataset, Path, Path]:
    """A dataset holding a calms21 source B and a TREx source A, both unscanned."""
    base = (tmp_path / "ds").resolve()
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
    source_a = base / "a_src"
    source_b = base / "b_src"
    write_trex_npz(source_a / "alpha_fish0.npz", n=6)
    write_trex_npz(source_a / "alpha_fish1.npz", n=6, individual=1)
    (source_b / "one.npy").parent.mkdir(parents=True)
    (source_b / "one.npy").write_bytes(b"one")
    (source_b / "two.npy").write_bytes(b"two")
    return ds, source_a, source_b


def _scan_a(ds: Dataset, source_a: Path, *, compute_md5: bool = True) -> Path:
    return ds.index_tracks_raw(
        [source_a],
        patterns=["*.npz"],
        src_format="trex_npz",
        compute_md5=compute_md5,
    )


def _scan_b(ds: Dataset, source_b: Path) -> Path:
    return ds.index_tracks_raw([source_b], patterns=["*.npy"], src_format="calms21_npy")


@pytest.mark.parametrize("compute_md5", [True, False])
def test_a_malformed_cell_in_a_kept_row_refuses_the_scan_before_any_write(
    tmp_path: Path, compute_md5: bool
) -> None:
    ds, source_a, source_b = _two_source_dataset(tmp_path)
    index_path = _scan_b(ds, source_b)
    set_companions_cell(index_path, "b_src/one.npy", "[broken")
    sequences = ds.get_root("tracks_raw") / "sequences.csv"
    index_before = index_path.read_bytes()
    sequences_before = sequences.read_bytes()

    with pytest.raises(ValueError, match=re.escape("b_src/one.npy")):
        _ = _scan_a(ds, source_a, compute_md5=compute_md5)

    assert index_path.read_bytes() == index_before
    assert sequences.read_bytes() == sequences_before


def test_dropping_rows_refuses_a_malformed_cell_in_a_kept_row_before_any_write(
    tmp_path: Path,
) -> None:
    ds, source_a, source_b = _two_source_dataset(tmp_path)
    _ = _scan_a(ds, source_a)
    index_path = _scan_b(ds, source_b)
    set_companions_cell(index_path, "b_src/one.npy", "[broken")
    sequences = ds.get_root("tracks_raw") / "sequences.csv"
    index_before = index_path.read_bytes()
    sequences_before = sequences.read_bytes()

    with pytest.raises(ValueError, match=re.escape("b_src/one.npy")):
        _ = ds.drop_claimed_rows("tracks", ScanClaim.over_directories([source_a]))

    assert index_path.read_bytes() == index_before
    assert sequences.read_bytes() == sequences_before


def test_a_malformed_cell_in_a_row_the_scan_replaces_is_replaced(
    tmp_path: Path,
) -> None:
    ds, _, source_b = _two_source_dataset(tmp_path)
    index_path = _scan_b(ds, source_b)
    set_companions_cell(index_path, "b_src/one.npy", "[broken")

    _ = _scan_b(ds, source_b)

    assert set(_companions_in(ds).values()) == {()}


def test_write_tracks_raw_index_refuses_a_malformed_cell_in_a_kept_row(
    tmp_path: Path,
) -> None:
    ds, _, source_b = _two_source_dataset(tmp_path)
    index_path = _scan_b(ds, source_b)
    set_companions_cell(index_path, "b_src/one.npy", "[broken")
    scope_dir = ds.base_dir / "tracks_raw" / "seqA"
    scope_dir.mkdir(parents=True)
    (scope_dir / "a.npy").write_bytes(b"payload")
    scope = TracksRawIndexScope(
        directory=scope_dir, group="", sequence="seqA", src_format="calms21_npy"
    )
    index_before = index_path.read_bytes()

    with pytest.raises(ValueError, match=re.escape("b_src/one.npy")):
        _ = ds.write_tracks_raw_index([scope], patterns=["*.npy"])

    assert index_path.read_bytes() == index_before


def test_a_seven_column_index_of_another_source_is_kept_with_its_composition(
    tmp_path: Path,
) -> None:
    ds, source_a, source_b = _two_source_dataset(tmp_path)
    index_path = _scan_b(ds, source_b)
    old = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    old.drop(columns=["companions"]).to_csv(index_path, index=False)
    assert len(pd.read_csv(index_path).columns) == 7
    composed_before = read_sequence_index(ds, "tracks_raw")
    assert set(composed_before["sequence"]) == {"one", "two"}

    _ = _scan_a(ds, source_a)

    rewritten = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    assert list(rewritten.columns) == TRACKS_RAW_INDEX_COLUMNS
    assert sorted(rewritten["sequence"]) == ["alpha", "alpha", "one", "two"]
    assert set(rewritten["companions"]) == {""}
    composed = read_sequence_index(ds, "tracks_raw")
    assert set(composed["sequence"]) == {"alpha", "one", "two"}
    compared = ["group", "sequence", "composition", "member_count"]
    kept = composed[composed["sequence"].isin(["one", "two"])]
    pd.testing.assert_frame_equal(
        kept[compared].sort_values("sequence").reset_index(drop=True),
        composed_before[compared].sort_values("sequence").reset_index(drop=True),
    )


def test_a_labels_scan_writes_blank_companion_cells(tmp_path: Path) -> None:
    base = (tmp_path / "ds").resolve()
    ds = make_dataset(base)
    write_calms21_npy(base / "uploads" / "file.npy", {"annotator": {"seq": 4}})

    index_path = ds.index_labels_raw(
        [base / "uploads"], patterns=["*.npy"], src_format="calms21_npy"
    )

    rows = read_tracks_raw_index(index_path)
    assert len(rows) == 1
    assert rows[0]["companions"] == ""
    assert list(pd.read_csv(index_path).columns) == TRACKS_RAW_INDEX_COLUMNS


def test_a_scan_records_each_companion_of_a_tracktor_table(tmp_path: Path) -> None:
    ds, recording = tracktor_dataset(
        tmp_path / "ds", "tracktor_raw_csv", "*_tracked.csv"
    )

    (record,) = _companions_in(ds)[RAW_NAME]

    assert record.name == CORRECTED_NAME
    assert record.size_bytes == recording.tsv.stat().st_size
    assert record.md5 == _md5_of(recording.tsv)
    members = [
        SourceMember(name=RAW_NAME, digest=_md5_of(recording.csv), algo="md5"),
        SourceMember(name=CORRECTED_NAME, digest=_md5_of(recording.tsv), algo="md5"),
    ]
    (composition,) = read_sequence_index(ds, "tracks_raw")["composition"]
    assert composition == tracks_raw_composition(members).digest


def test_the_companion_digest_is_left_blank_and_the_composition_incomplete_without_md5(
    tmp_path: Path,
) -> None:
    ds, _ = tracktor_dataset(tmp_path / "ds", "tracktor_raw_csv", "*_tracked.csv")

    _ = ds.index_tracks_raw(
        [ds.base_dir / "raw"],
        patterns=["*_tracked.csv"],
        src_format="tracktor_raw_csv",
        compute_md5=False,
    )

    (record,) = _companions_in(ds)[RAW_NAME]
    assert record.name == CORRECTED_NAME
    assert record.md5 == ""
    composed = read_sequence_index(ds, "tracks_raw")
    assert list(composed["composition"]) == [""]
    assert list(composed["member_count"]) == ["2"]


def _hashed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the name of each file the scan reads for a digest."""
    names: list[str] = []

    def counted(path: Path, chunk: int = 1 << 20) -> str:
        names.append(path.name)
        return _md5_of(path)

    monkeypatch.setattr("mosaic.core.dataset._md5", counted)
    return names


def test_a_companions_digest_is_carried_while_its_size_and_mtime_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, _ = tracktor_dataset(tmp_path / "ds", "tracktor_raw_csv", "*_tracked.csv")
    names = _hashed(monkeypatch)

    _ = ds.index_tracks_raw(
        [ds.base_dir / "raw"],
        patterns=["*_tracked.csv"],
        src_format="tracktor_raw_csv",
    )

    assert names == []


@pytest.mark.parametrize("moved", ["mtime", "size"])
def test_a_companions_digest_is_recomputed_when_its_size_or_mtime_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, moved: str
) -> None:
    ds, recording = tracktor_dataset(
        tmp_path / "ds", "tracktor_raw_csv", "*_tracked.csv"
    )
    stat = recording.tsv.stat()
    if moved == "size":
        with recording.tsv.open("ab") as handle:
            _ = handle.write(b"\n")
        os.utime(recording.tsv, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    else:
        os.utime(recording.tsv, (stat.st_mtime + 10, stat.st_mtime + 10))
    names = _hashed(monkeypatch)

    _ = ds.index_tracks_raw(
        [ds.base_dir / "raw"],
        patterns=["*_tracked.csv"],
        src_format="tracktor_raw_csv",
    )

    assert names == [CORRECTED_NAME]
    (record,) = _companions_in(ds)[RAW_NAME]
    assert record.md5 == _md5_of(recording.tsv)


def test_a_companion_is_one_member_per_name_within_a_merged_sequence() -> None:
    def row(path: str, md5: str, *records: CompanionRecord) -> dict[str, object]:
        return {
            "group": "g",
            "sequence": "s",
            "abs_path": path,
            "md5": md5,
            "companions": companions_cell(records),
        }

    shared = CompanionRecord(name="beside.tsv", size_bytes=1, mtime_iso=STAMP, md5="c1")
    other = CompanionRecord(name="other.tsv", size_bytes=1, mtime_iso=STAMP, md5="c2")

    members = source_members_from_rows(
        [
            row("a/x_fish0.csv", "d0", shared),
            row("a/x_fish1.csv", "d1", shared, other),
            row("a/x_fish2.csv", "d2"),
        ]
    )

    assert sorted(m.name for m in members[("g", "s")]) == [
        "beside.tsv",
        "other.tsv",
        "x_fish0.csv",
        "x_fish1.csv",
        "x_fish2.csv",
    ]
    assert {m.name: m.digest for m in members[("g", "s")]}["beside.tsv"] == "c1"
    assert all(m.algo == "md5" for m in members[("g", "s")])


def test_no_companions_lists_nothing(tmp_path: Path) -> None:
    assert no_companions(tmp_path / "a.csv") == ()


def test_calms21_and_trex_compositions_are_those_of_their_own_files(
    tmp_path: Path,
) -> None:
    """A source whose converter lists no companion composes from its files alone."""
    ds, source_a, source_b = _two_source_dataset(tmp_path)
    _ = _scan_a(ds, source_a)
    _ = _scan_b(ds, source_b)

    expected = {
        ("", "alpha"): tracks_raw_composition(
            [
                SourceMember(
                    name=f"alpha_fish{n}.npz",
                    digest=_md5_of(source_a / f"alpha_fish{n}.npz"),
                    algo="md5",
                )
                for n in (0, 1)
            ]
        ).digest,
        **{
            ("", name): tracks_raw_composition(
                [
                    SourceMember(
                        name=f"{name}.npy",
                        digest=_md5_of(source_b / f"{name}.npy"),
                        algo="md5",
                    )
                ]
            ).digest
            for name in ("one", "two")
        },
    }
    composed = read_sequence_index(ds, "tracks_raw")
    assert {
        (row["group"], row["sequence"]): row["composition"]
        for _, row in composed.iterrows()
    } == expected
    assert set(_companions_in(ds).values()) == {()}


def test_write_tracks_raw_index_records_the_companions_of_its_scopes_format(
    tmp_path: Path,
) -> None:
    base = (tmp_path / "ds").resolve()
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
    recording = write_tracktor_recording(base / "tracks_raw" / "arena")
    other = base / "tracks_raw" / "other"
    other.mkdir(parents=True)
    (other / "x.npz").write_bytes(b"x")
    scopes = [
        TracksRawIndexScope(
            directory=recording.csv.parent,
            group="",
            sequence="arena",
            src_format="tracktor_raw_csv",
        ),
        TracksRawIndexScope(
            directory=other, group="", sequence="other", src_format="trex"
        ),
    ]

    _ = ds.write_tracks_raw_index(scopes, patterns=["*_tracked.csv", "*.npz"])

    found = _companions_in(ds)
    assert [record.name for record in found[RAW_NAME]] == [CORRECTED_NAME]
    assert found[RAW_NAME][0].md5 == _md5_of(recording.tsv)
    assert found["x.npz"] == ()


# --- checksum workers ------------------------------------------------------

SCANNED_RECORDINGS = 6
"""Six recordings, each a source and a companion: twelve files to hash."""


def _scan_recordings(
    ds: Dataset, *, checksum_workers: int = 1, compute_md5: bool = True
) -> Path:
    return ds.index_tracks_raw(
        [ds.base_dir / "raw"],
        patterns=["*_tracked.csv"],
        src_format="tracktor_raw_csv",
        compute_md5=compute_md5,
        checksum_workers=checksum_workers,
    )


def _recordings_dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset((tmp_path / "ds").resolve(), roots=RAW_AND_CONVERTED)
    _ = write_tracktor_recordings(ds.base_dir / "raw", SCANNED_RECORDINGS)
    return ds


def _forget_the_scan(ds: Dataset) -> None:
    """Delete what a scan wrote, so the next one carries no digest."""
    for name in ("index.csv", "sequences.csv"):
        (ds.get_root("tracks_raw") / name).unlink()


def test_four_workers_write_the_index_and_compositions_of_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _recordings_dataset(tmp_path)
    index_path = _scan_recordings(ds, checksum_workers=1)
    root = ds.get_root("tracks_raw")
    one_worker = raw_index_files(root)
    _forget_the_scan(ds)
    names = _hashed(monkeypatch)

    _ = _scan_recordings(ds, checksum_workers=4)

    assert len(names) == 2 * SCANNED_RECORDINGS
    assert raw_index_files(root) == one_worker
    rows = read_tracks_raw_index(index_path)
    assert all(row["md5"] and companions_of(row)[0].md5 for row in rows)


def test_the_digests_do_not_depend_on_the_order_the_workers_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _recordings_dataset(tmp_path)
    index_path = _scan_recordings(ds, checksum_workers=1)
    expected = index_path.read_bytes()
    _forget_the_scan(ds)
    finished: list[str] = []
    ranked = sorted(path.name for path in (ds.base_dir / "raw").iterdir())

    def slower_for_earlier_names(path: Path, chunk: int = 1 << 20) -> str:
        time.sleep(0.01 * (len(ranked) - ranked.index(path.name)))
        finished.append(path.name)
        return _md5_of(path)

    monkeypatch.setattr("mosaic.core.dataset._md5", slower_for_earlier_names)

    _ = _scan_recordings(ds, checksum_workers=4)

    assert finished != sorted(finished)
    assert index_path.read_bytes() == expected


def test_workers_hash_in_threads_of_the_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _recordings_dataset(tmp_path)
    threads: set[int] = set()

    def noting(path: Path, chunk: int = 1 << 20) -> str:
        threads.add(threading.get_ident())
        return _md5_of(path)

    monkeypatch.setattr("mosaic.core.dataset._md5", noting)

    _ = _scan_recordings(ds, checksum_workers=1)
    assert threads == {threading.get_ident()}

    threads.clear()
    _forget_the_scan(ds)
    _ = _scan_recordings(ds, checksum_workers=4)
    assert threads
    assert threading.get_ident() not in threads


def test_a_carried_digest_is_not_recomputed_under_four_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _recordings_dataset(tmp_path)
    _ = _scan_recordings(ds)
    touched = sorted((ds.base_dir / "raw").glob("*_tracked.csv"))[0]
    stat = touched.stat()
    os.utime(touched, (stat.st_mtime + 10, stat.st_mtime + 10))
    names = _hashed(monkeypatch)

    _ = _scan_recordings(ds, checksum_workers=4)

    assert names == [touched.name]


def test_a_file_that_is_a_source_and_a_companion_is_read_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The corrected table lists the raw one beside it, which is also a source."""
    ds = make_dataset((tmp_path / "ds").resolve(), roots=RAW_AND_CONVERTED)
    recording = write_tracktor_recording(ds.base_dir / "raw")
    names = _hashed(monkeypatch)

    # Each file is named by its whole stem, so the two files are two entries.
    _ = ds.index_tracks_raw(
        [ds.base_dir / "raw"],
        patterns=["*_tracked.csv", "*.tsv"],
        src_format="tracktor_corrected_tsv",
        group_path_root=ds.base_dir / "raw",
        sequence_path_pattern=r"^(.*)\.(?:csv|tsv)$",
        checksum_workers=4,
    )

    assert sorted(names) == sorted([recording.csv.name, recording.tsv.name])


def test_write_tracks_raw_index_hashes_each_file_once_and_carries_it_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = (tmp_path / "ds").resolve()
    ds = make_dataset(base, roots=RAW_AND_CONVERTED)
    recording = write_tracktor_recording(base / "tracks_raw" / "arena")
    scope = TracksRawIndexScope(
        directory=recording.csv.parent,
        group="",
        sequence="arena",
        src_format="tracktor_raw_csv",
    )
    names = _hashed(monkeypatch)

    _ = ds.write_tracks_raw_index([scope], patterns=["*_tracked.csv"])

    assert sorted(names) == sorted([RAW_NAME, CORRECTED_NAME])
    (record,) = _companions_in(ds)[RAW_NAME]
    assert record.md5 == _md5_of(recording.tsv)
    names.clear()

    _ = ds.write_tracks_raw_index([scope], patterns=["*_tracked.csv"])

    assert names == []


def _symlink_the_corrected_table(
    recording_directory: Path, target_directory: Path
) -> Path:
    """Move the corrected table to *target_directory* and leave a symlink in its place."""
    link = recording_directory / CORRECTED_NAME
    target_directory.mkdir(parents=True)
    target = target_directory / CORRECTED_NAME
    _ = link.replace(target)
    link.symlink_to(target)
    return target


@pytest.mark.parametrize("workers", [1, 4])
def test_a_companion_that_is_a_symlink_is_hashed_through_to_its_target(
    tmp_path: Path, workers: int
) -> None:
    ds = make_dataset((tmp_path / "ds").resolve(), roots=RAW_AND_CONVERTED)
    recording = write_tracktor_recording(ds.base_dir / "raw")
    target = _symlink_the_corrected_table(recording.csv.parent, tmp_path / "elsewhere")

    _ = ds.index_tracks_raw(
        [ds.base_dir / "raw"],
        patterns=["*_tracked.csv"],
        src_format="tracktor_raw_csv",
        checksum_workers=workers,
    )

    (record,) = _companions_in(ds)[RAW_NAME]
    assert record.name == CORRECTED_NAME
    assert record.md5 == _md5_of(target)


def test_write_tracks_raw_index_hashes_a_companion_that_is_a_symlink(
    tmp_path: Path,
) -> None:
    ds = make_dataset((tmp_path / "ds").resolve(), roots=RAW_AND_CONVERTED)
    recording = write_tracktor_recording(ds.base_dir / "tracks_raw" / "arena")
    target = _symlink_the_corrected_table(recording.csv.parent, tmp_path / "elsewhere")
    scope = TracksRawIndexScope(
        directory=recording.csv.parent,
        group="",
        sequence="arena",
        src_format="tracktor_raw_csv",
    )

    _ = ds.write_tracks_raw_index([scope], patterns=["*_tracked.csv"])

    (record,) = _companions_in(ds)[RAW_NAME]
    assert record.md5 == _md5_of(target)


@pytest.mark.parametrize("method", ["index_tracks_raw", "index_labels_raw"])
def test_zero_checksum_workers_is_refused_before_any_write(
    tmp_path: Path, method: str
) -> None:
    ds = make_dataset(
        (tmp_path / "ds").resolve(), roots=(*RAW_AND_CONVERTED, "labels_raw")
    )
    _ = write_tracktor_recordings(ds.base_dir / "raw", 2)
    scan = getattr(ds, method)
    kwargs = (
        {"src_format": "tracktor_raw_csv"}
        if method == "index_tracks_raw"
        else {"src_format": "calms21_npy"}
    )

    with pytest.raises(ValueError, match="checksum_workers"):
        _ = scan(
            [ds.base_dir / "raw"],
            patterns=["*_tracked.csv"],
            checksum_workers=0,
            **kwargs,
        )

    assert not (ds.get_root("tracks_raw") / "index.csv").exists()
    assert not (ds.get_root("labels_raw") / "index.csv").exists()


@pytest.mark.parametrize("method", ["scan_tracks", "scan_labels"])
def test_zero_checksum_workers_is_refused_by_a_source_scan(
    tmp_path: Path, method: str
) -> None:
    ds = _recordings_dataset(tmp_path)

    with pytest.raises(ValueError, match="checksum_workers"):
        _ = getattr(ds, method)(checksum_workers=0)


# --- two files naming one entry -------------------------------------------


def _named_row(group: str, sequence: str, path: Path) -> dict[str, object]:
    return {"group": group, "sequence": sequence, "abs_path": str(path)}


def _resolve_stored(stored: str) -> Path:
    return Path(stored).resolve()


def test_two_files_naming_one_entry_are_found_with_their_resolved_paths(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "a" / "t1.h5", tmp_path / "b" / "t1.h5"
    rows = [_named_row("g", "t1", second), _named_row("g", "t1", first)]

    found = entries_named_twice(
        rows, one_file_per_sequence=True, resolve=_resolve_stored
    )

    assert [(each.entry, each.files) for each in found] == [
        (("g", "t1"), (first.resolve(), second.resolve()))
    ]


def test_a_format_that_merges_or_enumerates_is_not_checked(tmp_path: Path) -> None:
    rows = [
        _named_row("g", "t1", tmp_path / "a" / "t1.h5"),
        _named_row("g", "t1", tmp_path / "b" / "t1.h5"),
    ]

    assert (
        entries_named_twice(rows, one_file_per_sequence=False, resolve=_resolve_stored)
        == []
    )


def test_rows_with_a_blank_sequence_name_no_entry(tmp_path: Path) -> None:
    rows = [
        _named_row("g", "", tmp_path / "a.npy"),
        _named_row("g", "", tmp_path / "b.npy"),
    ]

    assert (
        entries_named_twice(rows, one_file_per_sequence=True, resolve=_resolve_stored)
        == []
    )


def test_a_promoted_correction_is_not_a_second_file_of_its_entry(
    tmp_path: Path,
) -> None:
    rows = [
        _named_row("g", "t1", tmp_path / "raw" / "t1.h5"),
        _named_row("g", "t1", tmp_path / "t1" / "corrected.rev1.h5"),
        _named_row("g", "t1", tmp_path / "t1" / "corrected.rev2.h5"),
    ]

    assert (
        entries_named_twice(rows, one_file_per_sequence=True, resolve=_resolve_stored)
        == []
    )


def test_a_file_never_collides_with_itself(tmp_path: Path) -> None:
    target = tmp_path / "real" / "t1.h5"
    target.parent.mkdir()
    _ = target.write_bytes(b"\x00")
    link = tmp_path / "link.h5"
    link.symlink_to(target)
    rows = [
        _named_row("g", "t1", target),
        _named_row("g", "t1", link),
        _named_row("g", "t1", target),
    ]

    assert (
        entries_named_twice(rows, one_file_per_sequence=True, resolve=_resolve_stored)
        == []
    )


def test_the_files_found_do_not_depend_on_the_row_order(tmp_path: Path) -> None:
    files = [tmp_path / name / "t1.h5" for name in ("c", "a", "b")]
    rows = [_named_row("g", "t1", path) for path in files]

    forward = entries_named_twice(
        rows, one_file_per_sequence=True, resolve=_resolve_stored
    )
    backward = entries_named_twice(
        rows[::-1], one_file_per_sequence=True, resolve=_resolve_stored
    )

    assert forward == backward
    assert [each.files for each in forward] == [
        tuple(sorted(path.resolve() for path in files))
    ]


def test_the_refusal_names_the_entry_both_files_and_each_source(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "a" / "t1.h5", tmp_path / "b" / "t1.h5"

    refusal = RawEntryNamedTwice(
        ("g", "t1"), (first, second), ("source 'one'", "source 'two'")
    )

    message = str(refusal)
    assert "g__t1" in message
    assert str(first) in message and str(second) in message
    assert "source 'one'" in message and "source 'two'" in message
    assert "keep one" in message
    assert (refusal.entry, refusal.files) == (("g", "t1"), (first, second))
