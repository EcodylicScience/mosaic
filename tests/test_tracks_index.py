"""The typed index of the standardized tracks under ``tracks/``.

Covers three things the hand-written writer it replaces had no coverage for at
all: that the schema on disk matches the row class, that an older on-disk shape
is adopted without losing a row, and that an absent index reads as an empty one
rather than raising.

The module name was freed by the ``tracks_index`` -> ``tracks_raw_index`` rename;
this file tests the ``tracks/`` index, ``test_tracks_raw_index`` the
``tracks_raw/`` one.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.tracks_index import (
    DROPPED_LEGACY_COLUMNS,
    TRACKS_INDEX_COLUMNS,
    TRACKS_INDEX_PATH_COLUMNS,
    TracksIndexRow,
    adopt_legacy_columns,
    consumed_roots_for,
    encode_source_roots,
    legacy_view,
    read_tracks_index,
    tracks_index,
    tracks_index_path,
    write_tracks_row,
)

# The exact header the hand-written writer emitted, in its order. Pinned as a
# literal rather than imported, because the point of the adoption tests is that
# a file written by code that no longer exists still reads.
LEGACY_COLUMNS = [
    "group",
    "sequence",
    "group_safe",
    "sequence_safe",
    "collection",
    "collection_safe",
    "abs_path",
    "std_format",
    "source_abs_path",
    "source_md5",
    "n_rows",
]


def _dataset(base: Path, roots: dict[str, str] | None = None) -> Dataset:
    base.mkdir(parents=True, exist_ok=True)
    ds = Dataset(
        manifest_path=base / "dataset.yaml",
        roots=roots
        or {"tracks": str(base / "tracks"), "tracks_raw": str(base / "tracks_raw")},
    )
    ds.ensure_roots()
    # Seed the manifest so base_dir resolves to its parent -- the same thing
    # relative_to_root measures against.
    ds.save()
    return ds


def _track_parquet(ds: Dataset, sequence: str, n_rows: int = 40) -> Path:
    """Write a real parquet under tracks/ and return it."""
    root = ds.get_root("tracks")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{sequence}.parquet"
    pd.DataFrame({"frame": range(n_rows), "id": [0] * n_rows}).to_parquet(path)
    return path


def _write_legacy_index(ds: Dataset, rows: list[dict[str, object]]) -> Path:
    """Write an index in the pre-Stage-2 eleven-column shape."""
    path = tracks_index_path(ds)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=LEGACY_COLUMNS).to_csv(path, index=False)
    return path


# --- the schema ------------------------------------------------------------


def test_row_field_order_is_the_csv_column_order(tmp_path: Path) -> None:
    """The contract every other assertion here rests on."""
    ds = _dataset(tmp_path)
    write_tracks_row(
        ds,
        run_id="convert-x.0.1-aaaaaaaaaa",
        group="",
        sequence="s",
        out_path=_track_parquet(ds, "s"),
        producer="convert-x",
        std_format="trex_v1",
        n_rows=40,
    )
    header = tracks_index_path(ds).read_text().splitlines()[0]
    assert header.split(",") == TRACKS_INDEX_COLUMNS
    assert TRACKS_INDEX_COLUMNS == [f.name for f in dataclasses.fields(TracksIndexRow)]
    assert TRACKS_INDEX_COLUMNS[:4] == [
        "abs_path",
        "run_id",
        "started_at",
        "finished_at",
    ]


def test_the_registered_path_column_is_the_one_holding_a_path(tmp_path: Path) -> None:
    """A new path column must join this tuple or stop being portable."""
    assert TRACKS_INDEX_PATH_COLUMNS == ("source_abs_path",)
    assert all(column in TRACKS_INDEX_COLUMNS for column in TRACKS_INDEX_PATH_COLUMNS)


def test_a_row_count_stays_an_integer_on_disk(tmp_path: Path) -> None:
    """``40`` not ``40.0`` -- the int column widened by a concat is the classic."""
    ds = _dataset(tmp_path)
    write_tracks_row(
        ds,
        run_id="convert-x.0.1-aaaaaaaaaa",
        group="",
        sequence="s",
        out_path=_track_parquet(ds, "s"),
        producer="convert-x",
        std_format="trex_v1",
        n_rows=40,
    )
    text = tracks_index_path(ds).read_text()
    assert text.rstrip().endswith(",40")
    assert "40.0" not in text


# --- writing ---------------------------------------------------------------


def test_paths_are_stored_root_relative(tmp_path: Path) -> None:
    """Both path columns, or the index does not survive a move."""
    ds = _dataset(tmp_path)
    source = ds.get_root("tracks_raw") / "raw.npy"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"x")

    write_tracks_row(
        ds,
        run_id="convert-x.0.1-aaaaaaaaaa",
        group="",
        sequence="s",
        out_path=_track_parquet(ds, "s"),
        producer="convert-x",
        std_format="trex_v1",
        n_rows=40,
        source=source,
    )
    row = read_tracks_index(ds).iloc[0]
    assert not Path(str(row["abs_path"])).is_absolute()
    assert str(row["source_abs_path"]) == "tracks_raw/raw.npy"
    assert ds.resolve_path(str(row["source_abs_path"])).exists()


def test_an_absent_source_stays_empty_not_a_dot(tmp_path: Path) -> None:
    """Why source_abs_path is a str: ``Path("")`` renders as ``"."``."""
    ds = _dataset(tmp_path)
    write_tracks_row(
        ds,
        run_id="infer-points.0.1-aaaaaaaaaa",
        group="",
        sequence="s",
        out_path=_track_parquet(ds, "s"),
        producer="infer-points",
        std_format="trex_v1",
        n_rows=40,
    )
    assert str(read_tracks_index(ds).iloc[0]["source_abs_path"]) == ""


def test_a_numpy_scalar_identity_is_stringified(tmp_path: Path) -> None:
    """One caller reads group/sequence off a pandas Series.

    Left as an ``np.int64`` the cell would be an integer column, and the dedup
    that holds this index to one row per entry compares against a ``str``.
    """
    ds = _dataset(tmp_path)
    series = pd.DataFrame({"group": [""], "sequence": [1]}).iloc[0]
    for _ in range(3):
        write_tracks_row(
            ds,
            run_id="convert-x.0.1-aaaaaaaaaa",
            group=series["group"],
            sequence=series["sequence"],
            out_path=_track_parquet(ds, "1"),
            producer="convert-x",
            std_format="trex_v1",
            n_rows=40,
        )
    df = read_tracks_index(ds)
    assert len(df) == 1
    assert str(df.iloc[0]["sequence"]) == "1"


def test_one_row_per_entry_across_all_three_producers(tmp_path: Path) -> None:
    """The M1 invariant. Stage 3.4 is what makes a second row legal."""
    ds = _dataset(tmp_path)
    path = _track_parquet(ds, "s")
    for run_id, producer in [
        ("convert-trex_npz.0.1-aaaaaaaaaa", "convert-trex_npz"),
        ("trex.0.1-bbbbbbbbbb", "trex"),
        ("infer-points.0.1-cccccccccc", "infer-points"),
    ]:
        write_tracks_row(
            ds,
            run_id=run_id,
            group="",
            sequence="s",
            out_path=path,
            producer=producer,
            std_format="trex_v1",
            n_rows=40,
        )
    df = read_tracks_index(ds)
    assert len(df) == 1
    assert str(df.iloc[0]["producer"]) == "infer-points"


def test_the_write_is_atomic_and_leaves_no_orphan(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    write_tracks_row(
        ds,
        run_id="convert-x.0.1-aaaaaaaaaa",
        group="",
        sequence="s",
        out_path=_track_parquet(ds, "s"),
        producer="convert-x",
        std_format="trex_v1",
        n_rows=40,
    )
    names = sorted(p.name for p in ds.get_root("tracks").iterdir())
    assert names == ["index.csv", "s.parquet"]


# --- consumed_source_roots -------------------------------------------------


def test_source_roots_encode_sorted_and_deduplicated() -> None:
    assert encode_source_roots(["tracks_raw", "media", "tracks_raw"]) == (
        "media,tracks_raw"
    )
    assert encode_source_roots([]) == ""
    assert encode_source_roots(["", "media"]) == "media"


def test_consumed_roots_prefers_the_deepest_containing_root(tmp_path: Path) -> None:
    """Roots nest -- trex lives under tracks_raw -- so shortest-match lies."""
    ds = _dataset(
        tmp_path,
        roots={
            "tracks_raw": str(tmp_path / "tracks_raw"),
            "trex": str(tmp_path / "tracks_raw" / "trex"),
        },
    )
    npz = ds.get_root("trex") / "seq" / "data.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    npz.write_bytes(b"x")

    assert consumed_roots_for(ds, [npz]) == ("trex",)


def test_a_path_under_no_declared_root_contributes_nothing(tmp_path: Path) -> None:
    """Honest omission over a guess -- the M3 rule."""
    ds = _dataset(tmp_path)
    outside = tmp_path.parent / "elsewhere.npy"
    outside.write_bytes(b"x")
    assert consumed_roots_for(ds, [outside]) == ()


# --- adoption --------------------------------------------------------------


def test_adoption_widens_a_legacy_index_without_losing_a_row(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    _ = _track_parquet(ds, "legacy")
    _write_legacy_index(
        ds,
        [
            {
                "group": "",
                "sequence": "legacy",
                "group_safe": "",
                "sequence_safe": "legacy",
                "collection": "",
                "collection_safe": "",
                "abs_path": "tracks/legacy.parquet",
                "std_format": "trex_v1",
                "source_abs_path": "tracks_raw/legacy.npy",
                "source_md5": "",
                "n_rows": 40,
            }
        ],
    )

    write_tracks_row(
        ds,
        run_id="convert-x.0.1-aaaaaaaaaa",
        group="",
        sequence="fresh",
        out_path=_track_parquet(ds, "fresh"),
        producer="convert-x",
        std_format="trex_v1",
        n_rows=40,
    )

    header = tracks_index_path(ds).read_text().splitlines()[0]
    assert header.split(",") == TRACKS_INDEX_COLUMNS
    df = read_tracks_index(ds)
    assert sorted(str(v) for v in df["sequence"]) == ["fresh", "legacy"]
    legacy = df[df["sequence"] == "legacy"].iloc[0]
    # Honestly empty: this row predates run identity, and its source pointer
    # and row count survive the widening.
    assert str(legacy["run_id"]) == ""
    assert str(legacy["producer"]) == ""
    assert str(legacy["source_abs_path"]) == "tracks_raw/legacy.npy"
    assert str(legacy["n_rows"]) == "40"


def test_adoption_drops_the_four_dead_legacy_columns(tmp_path: Path) -> None:
    """group_safe/sequence_safe are derivable; collection* had no reader."""
    ds = _dataset(tmp_path)
    _write_legacy_index(
        ds,
        [
            {
                "group": "g",
                "sequence": "s",
                "group_safe": "g",
                "sequence_safe": "s",
                "collection": "hint",
                "collection_safe": "hint",
                "abs_path": "tracks/g__s.parquet",
                "std_format": "trex_v1",
                "source_abs_path": "",
                "source_md5": "",
                "n_rows": 40,
            }
        ],
    )
    df = read_tracks_index(ds)
    assert DROPPED_LEGACY_COLUMNS == (
        "group_safe",
        "sequence_safe",
        "collection",
        "collection_safe",
    )
    for dropped in DROPPED_LEGACY_COLUMNS:
        assert dropped not in df.columns
        assert dropped in LEGACY_COLUMNS


def test_adoption_is_idempotent(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    for sequence in ("a", "b"):
        write_tracks_row(
            ds,
            run_id="convert-x.0.1-aaaaaaaaaa",
            group="",
            sequence=sequence,
            out_path=_track_parquet(ds, sequence),
            producer="convert-x",
            std_format="trex_v1",
            n_rows=40,
        )
    first = tracks_index_path(ds).read_text()
    adopted = adopt_legacy_columns(adopt_legacy_columns(read_tracks_index(ds)))
    assert list(adopted.columns) == TRACKS_INDEX_COLUMNS
    assert len(adopted) == 2
    assert first.splitlines()[0].split(",") == TRACKS_INDEX_COLUMNS


def test_adoption_collapses_preexisting_duplicate_entries_keeping_last(
    tmp_path: Path,
) -> None:
    """An index written before string columns were read as strings can hold these."""
    ds = _dataset(tmp_path)
    _write_legacy_index(
        ds,
        [
            {"group": "", "sequence": "s", "abs_path": "tracks/a.parquet", "n_rows": 1},
            {"group": "", "sequence": "s", "abs_path": "tracks/b.parquet", "n_rows": 2},
        ],
    )
    df = read_tracks_index(ds)
    assert len(df) == 1
    assert str(df.iloc[0]["abs_path"]) == "tracks/b.parquet"


def test_adoption_coerces_nan_already_on_disk(tmp_path: Path) -> None:
    """NaN is not hypothetical: the old writer concatenated it into widened columns."""
    ds = _dataset(tmp_path)
    path = tracks_index_path(ds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("group,sequence,abs_path,std_format\n,s,tracks/s.parquet,\n")
    df = read_tracks_index(ds)
    assert str(df.iloc[0]["std_format"]) == ""
    assert str(df.iloc[0]["producer"]) == ""


# --- reading ---------------------------------------------------------------


def test_an_absent_index_reads_as_an_empty_one_with_the_full_schema(
    tmp_path: Path,
) -> None:
    """Absence and emptiness are one dataset state and must answer alike.

    The column set matters: callers filter on group/sequence immediately, and a
    column-less empty frame turns "no tracks yet" into KeyError.
    """
    ds = _dataset(tmp_path)
    df = read_tracks_index(ds)
    assert len(df) == 0
    assert list(df.columns) == TRACKS_INDEX_COLUMNS
    assert df[df["group"] == "g"].empty


def test_a_header_only_index_reads_the_same_as_an_absent_one(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    tracks_index(tracks_index_path(ds)).ensure()
    df = read_tracks_index(ds)
    assert len(df) == 0
    assert list(df.columns) == TRACKS_INDEX_COLUMNS


def test_reading_never_writes(tmp_path: Path) -> None:
    """Adoption is on write. A read-only mount must still list its sequences."""
    ds = _dataset(tmp_path)
    path = _write_legacy_index(
        ds, [{"group": "", "sequence": "s", "abs_path": "tracks/s.parquet"}]
    )
    before = path.read_bytes()
    _ = read_tracks_index(ds)
    _ = read_tracks_index(ds)
    assert path.read_bytes() == before


def test_the_run_id_selector_filters_and_reports_an_unknown_run(
    tmp_path: Path,
) -> None:
    """The surface Stage 3.3 drives; nothing passes a non-None value in M1."""
    ds = _dataset(tmp_path)
    write_tracks_row(
        ds,
        run_id="convert-x.0.1-aaaaaaaaaa",
        group="",
        sequence="s",
        out_path=_track_parquet(ds, "s"),
        producer="convert-x",
        std_format="trex_v1",
        n_rows=40,
    )
    idx = tracks_index(tracks_index_path(ds))
    assert len(idx.read(run_id="convert-x.0.1-aaaaaaaaaa")) == 1
    with pytest.raises(FileNotFoundError):
        _ = idx.read(run_id="convert-x.0.1-zzzzzzzzzz")


# --- legacy_view -----------------------------------------------------------


def test_legacy_view_rederives_the_safe_names(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    write_tracks_row(
        ds,
        run_id="convert-calms21_npy.0.1-aaaaaaaaaa",
        group="calms21_task1",
        sequence="task1__test__mouse075",
        out_path=_track_parquet(ds, "seq"),
        producer="convert-calms21_npy",
        std_format="trex_v1",
        n_rows=40,
    )
    view = legacy_view(read_tracks_index(ds))
    assert str(view.iloc[0]["group_safe"]) == "calms21_task1"
    assert str(view.iloc[0]["sequence_safe"]) == "task1__test__mouse075"


def test_legacy_view_derives_over_a_blank_stored_cell(tmp_path: Path) -> None:
    """The old ``row.get(...) or ...`` fallback did not fire on a blank cell.

    A NaN read from a widened column is truthy, so the fallback returned the NaN
    and the next ``.lower()`` raised AttributeError. Deriving unconditionally is
    the fix, and a migrated index is exactly where blanks come from.
    """
    ds = _dataset(tmp_path)
    _write_legacy_index(
        ds,
        [
            {
                "group": "g",
                "sequence": "s",
                "group_safe": "",
                "sequence_safe": "",
                "abs_path": "tracks/g__s.parquet",
            }
        ],
    )
    view = legacy_view(read_tracks_index(ds))
    assert str(view.iloc[0]["group_safe"]) == "g"
    assert str(view.iloc[0]["sequence_safe"]) == "s"


# --- entry names -----------------------------------------------------------


def test_the_index_writer_refuses_a_path_separator(tmp_path: Path) -> None:
    """The choke point every producer goes through."""
    ds = _dataset(tmp_path)
    with pytest.raises(ValueError, match="forward slash"):
        write_tracks_row(
            ds,
            run_id="convert-x.0.1-aaaaaaaaaa",
            group="",
            sequence="task1/test/m075",
            out_path=_track_parquet(ds, "s"),
            producer="convert-x",
            std_format="trex_v1",
            n_rows=40,
        )


def test_an_index_that_already_holds_a_slash_name_still_reads(tmp_path: Path) -> None:
    """Validation is on write only -- this is what keeps the change additive.

    A dataset converted before the rule existed keeps resolving; nothing
    rewrites it, and nothing refuses to read it.
    """
    ds = _dataset(tmp_path)
    _write_legacy_index(
        ds,
        [
            {
                "group": "",
                "sequence": "task1/test/m075",
                "abs_path": "tracks/task1%2Ftest%2Fm075.parquet",
                "std_format": "trex_v1",
                "n_rows": 40,
            }
        ],
    )
    df = read_tracks_index(ds)
    assert [str(s) for s in df["sequence"]] == ["task1/test/m075"]
