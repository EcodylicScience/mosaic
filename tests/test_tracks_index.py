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
    backfill_frame_extents,
    frame_extent,
    read_frame_extent,
    read_frame_extents,
    TRACKS_INDEX_COLUMNS,
    TRACKS_INDEX_PATH_COLUMNS,
    TracksIndexRow,
    adopt_legacy_columns,
    consumed_roots_for,
    encode_source_roots,
    legacy_view,
    read_tracks_index,
    select_variant_rows,
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


def _write_two_variant_rows(ds: Dataset, sequence: str, *variants: str) -> Path:
    """One entry under several variants: real tables, and rows naming them.

    The rows are written past the typed writer and positionally, so adding a
    column to ``TRACKS_INDEX_COLUMNS`` breaks this loudly rather than shifting
    every cell one place along.
    """
    for variant in variants:
        table = ds.get_root("tracks") / variant / f"{sequence}.parquet"
        table.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"frame": range(40), "id": [0] * 40}).to_parquet(table)

    path = tracks_index_path(ds)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(TRACKS_INDEX_COLUMNS)
    rows = "\n".join(
        f"tracks/{variant}/{sequence}.parquet,{variant},,,,"
        f"{sequence},convert-x,trex_v1,,,,,40"
        for variant in variants
    )
    path.write_text(f"{header}\n{rows}\n")
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
    """``40`` not ``40.0`` -- the int column widened by a concat is the classic.

    Asserted on the ``n_rows`` cell rather than on the file text. Searching the
    whole file for ``"40.0"`` also searches ``started_at``, which is an ISO
    timestamp: a row written at forty-point-zero-something seconds past the
    minute matched it, so the test failed roughly one run in six hundred for a
    reason that had nothing to do with dtypes.
    """
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
    header, row = tracks_index_path(ds).read_text().splitlines()[:2]
    cells = dict(zip(header.split(","), row.split(","), strict=True))
    assert cells["n_rows"] == "40"


def _posed_track_parquet(ds: Dataset, sequence: str, n_keypoints: int = 7) -> Path:
    """A real parquet under tracks/ carrying *n_keypoints* keypoint pairs."""
    root = ds.get_root("tracks")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{sequence}.parquet"
    frame = pd.DataFrame({"frame": range(40), "id": [0] * 40})
    for k in range(n_keypoints):
        frame[f"poseX{k}"] = 1.0
        frame[f"poseY{k}"] = 2.0
    frame.to_parquet(path)
    return path


def test_the_keypoint_count_is_measured_from_the_written_table(
    tmp_path: Path,
) -> None:
    """Measured, not passed in.

    Every caller holds the frame and could have passed a count, but six of them
    would each have to remember to, and one that forgot would record "no
    keypoints" about a table full of them -- a claim, not an absence.
    """
    ds = _dataset(tmp_path)
    write_tracks_row(
        ds,
        run_id="convert-x.0.1-aaaaaaaaaa",
        group="",
        sequence="posed",
        out_path=_posed_track_parquet(ds, "posed"),
        producer="convert-x",
        std_format="mosaic_v1",
        n_rows=40,
    )
    row = read_tracks_index(ds).iloc[0]
    assert str(row["n_keypoints"]) == "7"


def test_a_centroid_only_table_records_no_keypoints(tmp_path: Path) -> None:
    """Zero is the honest answer, and now a legal one."""
    ds = _dataset(tmp_path)
    write_tracks_row(
        ds,
        run_id="convert-x.0.1-aaaaaaaaaa",
        group="",
        sequence="s",
        out_path=_track_parquet(ds, "s"),
        producer="convert-x",
        std_format="mosaic_v1",
        n_rows=40,
    )
    row = read_tracks_index(ds).iloc[0]
    assert str(row["n_keypoints"]) == "0"


def test_a_keypoint_count_stays_an_integer_on_disk(tmp_path: Path) -> None:
    """``7`` not ``7.0`` -- the same concat widening ``n_rows`` guards against."""
    ds = _dataset(tmp_path)
    write_tracks_row(
        ds,
        run_id="convert-x.0.1-aaaaaaaaaa",
        group="",
        sequence="posed",
        out_path=_posed_track_parquet(ds, "posed"),
        producer="convert-x",
        std_format="mosaic_v1",
        n_rows=40,
    )
    header, row = tracks_index_path(ds).read_text().splitlines()[:2]
    cells = dict(zip(header.split(","), row.split(","), strict=True))
    assert cells["n_keypoints"] == "7"


def test_a_legacy_row_reads_as_unknown_not_as_no_keypoints(tmp_path: Path) -> None:
    """The distinction a reader has to respect.

    A row written before this column existed projects to ``""``. Reading that as
    zero would say "this entry has no keypoints" of a table that may be full of
    them -- and something branching on it would then skip every pose feature on
    every dataset converted before today.
    """
    ds = _dataset(tmp_path)
    _write_legacy_index(
        ds,
        [
            {
                "abs_path": "tracks/legacy.parquet",
                "group": "",
                "sequence": "legacy",
                "n_rows": 40,
            }
        ],
    )
    write_tracks_row(
        ds,
        run_id="convert-x.0.1-aaaaaaaaaa",
        group="",
        sequence="posed",
        out_path=_posed_track_parquet(ds, "posed"),
        producer="convert-x",
        std_format="mosaic_v1",
        n_rows=40,
    )
    df = read_tracks_index(ds)
    legacy = df[df["sequence"] == "legacy"].iloc[0]
    fresh = df[df["sequence"] == "posed"].iloc[0]
    assert str(legacy["n_keypoints"]) == ""
    assert str(fresh["n_keypoints"]) == "7"


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


def test_all_three_producers_keep_their_row_for_one_entry(tmp_path: Path) -> None:
    """The inversion of the M1 invariant, and the point of the stage.

    Three recipes for one sequence used to leave one row, the last writer's --
    which was right while they all overwrote one flat parquet, and is a lost
    record now that each writes into its own directory.
    """
    ds = _dataset(tmp_path)
    producers = [
        ("convert-trex_npz.0.1-aaaaaaaaaa", "convert-trex_npz"),
        ("trex.0.1-bbbbbbbbbb", "trex"),
        ("infer-points.0.1-cccccccccc", "infer-points"),
    ]
    for run_id, producer in producers:
        write_tracks_row(
            ds,
            run_id=run_id,
            group="",
            sequence="s",
            out_path=_track_parquet(ds, "s"),
            producer=producer,
            std_format="trex_v1",
            n_rows=40,
        )
    df = read_tracks_index(ds)
    assert len(df) == 3
    assert list(df["run_id"]) == [run_id for run_id, _ in producers]

    # Re-writing one of them still replaces its own row rather than adding a
    # fourth: the dedup key is the triple, not the run_id alone.
    write_tracks_row(
        ds,
        run_id="trex.0.1-bbbbbbbbbb",
        group="",
        sequence="s",
        out_path=_track_parquet(ds, "s"),
        producer="trex",
        std_format="trex_v1",
        n_rows=99,
    )
    df = read_tracks_index(ds)
    assert len(df) == 3
    assert set(df["run_id"]) == {run_id for run_id, _ in producers}


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
    # index.csv.lock is index_lock's sidecar: one zero-byte file per index,
    # created on the first locked write and never removed. What must *not* be
    # here is a ".<stem>-*.tmp" orphan from a torn atomic_write.
    assert names == ["index.csv", "index.csv.lock", "s.parquet"]


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


def test_adoption_keeps_preexisting_duplicate_entries(tmp_path: Path) -> None:
    """Adoption no longer collapses; choosing is the reader's job now.

    An index written before string columns were read as strings can already hold
    two rows for one entry. Adoption used to drop one keep-last -- but it is
    ``IndexCSV``'s ``adopt`` hook, so it runs on the way *in* as well, and would
    now discard a legitimate second variant at append time. Both rows survive
    here, and :func:`select_variant_rows` decides which one an entry means.
    """
    ds = _dataset(tmp_path)
    _write_legacy_index(
        ds,
        [
            {"group": "", "sequence": "s", "abs_path": "tracks/a.parquet", "n_rows": 1},
            {"group": "", "sequence": "s", "abs_path": "tracks/b.parquet", "n_rows": 2},
        ],
    )
    df = read_tracks_index(ds)
    assert len(df) == 2

    # Both are unlabelled, so neither supersedes the other and there is no
    # ambiguity to refuse: last wins, exactly as adoption used to do.
    selected = select_variant_rows(df)
    assert len(selected) == 1
    assert str(selected.iloc[0]["abs_path"]) == "tracks/b.parquet"


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


# --- select_variant_rows ---------------------------------------------------


def _rows(*specs: tuple[str, str, str]) -> pd.DataFrame:
    """A minimal tracks frame from ``(run_id, group, sequence)`` triples."""
    frame = pd.DataFrame(
        {column: pd.Series(dtype="object") for column in TRACKS_INDEX_COLUMNS}
    )
    built = [
        {
            **{column: "" for column in TRACKS_INDEX_COLUMNS},
            "run_id": run_id,
            "group": group,
            "sequence": sequence,
            "abs_path": f"tracks/{run_id or 'flat'}/{sequence}.parquet",
        }
        for run_id, group, sequence in specs
    ]
    return pd.concat([frame, pd.DataFrame(built)], ignore_index=True)


def test_an_unlabelled_row_is_left_alone_when_it_is_all_there_is() -> None:
    """Every pre-Stage-3 dataset is entirely unlabelled; nothing may change."""
    rows = select_variant_rows(_rows(("", "", "a"), ("", "", "b")))
    assert list(rows["sequence"]) == ["a", "b"]
    assert list(rows["run_id"]) == ["", ""]


def test_a_labelled_row_supersedes_an_unlabelled_one_for_the_same_entry() -> None:
    """The state one ordinary re-conversion of an existing dataset produces.

    An empty ``run_id`` means "predates the scheme", not "a recipe called
    nothing", so it is an ancestor rather than a competitor. Were it treated as a
    peer, the ambiguity below would fire on every dataset in existence the first
    time someone re-ran their conversion cell.
    """
    rows = select_variant_rows(_rows(("", "", "a"), ("convert-x.0.1-aa", "", "a")))
    assert list(rows["run_id"]) == ["convert-x.0.1-aa"]


def test_two_real_variants_of_one_entry_refuse_to_choose() -> None:
    """Two recipes for one table has no defensible default."""
    rows = _rows(("convert-x.0.1-aa", "", "a"), ("trex.0.1-bb", "", "a"))
    with pytest.raises(ValueError) as excinfo:
        _ = select_variant_rows(rows)
    message = str(excinfo.value)
    assert "convert-x.0.1-aa" in message
    assert "trex.0.1-bb" in message
    assert "sequence='a'" in message
    assert "tracks_run_id" in message, "the error must name the way out"


def test_the_ambiguity_message_never_offers_an_unlabelled_candidate() -> None:
    """An empty run_id is not something a caller could pass to disambiguate.

    A non-empty group here so that the only ``''`` the message could contain is
    a candidate rather than an empty group name.
    """
    rows = _rows(
        ("", "g", "a"), ("convert-x.0.1-aa", "g", "a"), ("trex.0.1-bb", "g", "a")
    )
    with pytest.raises(ValueError) as excinfo:
        _ = select_variant_rows(rows)
    message = str(excinfo.value)
    assert "2 variants" in message
    assert "''" not in message


def test_different_entries_may_carry_different_variants() -> None:
    """A mixed dataset -- some converted, some tracked -- stays resolvable."""
    rows = select_variant_rows(
        _rows(("convert-x.0.1-aa", "", "a"), ("trex.0.1-bb", "", "b"))
    )
    assert list(rows["sequence"]) == ["a", "b"]


def test_selecting_a_variant_by_name_takes_exactly_it() -> None:
    rows = select_variant_rows(
        _rows(("convert-x.0.1-aa", "", "a"), ("trex.0.1-bb", "", "a")),
        "trex.0.1-bb",
    )
    assert list(rows["run_id"]) == ["trex.0.1-bb"]


def test_the_unlabelled_tables_stay_addressable_by_name() -> None:
    """``run_id=""`` names the pre-Stage-3 flat layout, which is the way back."""
    rows = select_variant_rows(_rows(("", "", "a"), ("convert-x.0.1-aa", "", "a")), "")
    assert len(rows) == 1
    assert str(rows.iloc[0]["abs_path"]) == "tracks/flat/a.parquet"


def test_an_empty_frame_selects_to_an_empty_frame() -> None:
    """A cold dataset must not raise on the way to reporting it has no tracks."""
    assert select_variant_rows(_rows()).empty


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


def test_an_absent_group_is_recorded_as_empty(tmp_path: Path) -> None:
    """The writer takes ``object`` because one caller reads off a pandas Series.

    A blank cell arrives there as a float NaN, which ``str()`` spells as the
    word -- and the ``is not None`` guard this used to carry never saw it. The
    table beside this row is named ``s.parquet``, so a row saying ``nan`` would
    be the index contradicting the layout it indexes.
    """
    ds = _dataset(tmp_path)
    write_tracks_row(
        ds,
        run_id="convert-x.0.1-aaaaaaaaaa",
        group=float("nan"),
        sequence="s",
        out_path=_track_parquet(ds, "s"),
        producer="convert-x",
        std_format="trex_v1",
        n_rows=40,
    )
    row = read_tracks_index(ds).iloc[0]
    assert str(row["group"]) == ""
    assert str(row["sequence"]) == "s"


def test_an_absent_group_still_finds_its_composition(tmp_path: Path) -> None:
    """Why the spelling matters beyond the filename.

    The row's composition is looked up by the same key the row is written
    under, and ``index_tracks_raw`` records it under ``("", sequence)``. A group
    spelled ``nan`` misses that entry and records nothing -- silently, because
    an unestablished composition is a legitimate state.
    """
    from mosaic.core.pipeline.composition import SourceMember, tracks_raw_composition
    from mosaic.core.pipeline.sequence_index import write_sequence_compositions

    ds = _dataset(tmp_path)
    comp = tracks_raw_composition([SourceMember(name="s.npz", digest="d", algo="md5")])
    _ = write_sequence_compositions(ds, "tracks_raw", compositions={("", "s"): comp})

    write_tracks_row(
        ds,
        run_id="convert-x.0.1-aaaaaaaaaa",
        group=float("nan"),
        sequence="s",
        out_path=_track_parquet(ds, "s"),
        producer="convert-x",
        std_format="trex_v1",
        n_rows=40,
        consumed_source_roots=("tracks_raw",),
    )
    row = read_tracks_index(ds).iloc[0]
    assert row["consumed_composition"] == comp.digest != ""


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


# --- the manifest run selector ---------------------------------------------


def _feature_inputs():
    from mosaic.core.pipeline.types import Inputs

    return Inputs(("tracks",))


def test_build_manifest_defaults_to_every_variant(tmp_path: Path) -> None:
    """``None`` is every row, not "the latest run".

    A mixed dataset carries different variants on different entries, so "latest"
    would collapse the universe to whichever recipe was written last.
    """
    from mosaic.core.pipeline.manifest import build_manifest

    ds = _dataset(tmp_path)
    for sequence, variant in [("a", "convert-x.0.1-aaaaaaaaaa"), ("b", "trex.0.1-bb")]:
        write_tracks_row(
            ds,
            run_id=variant,
            group="",
            sequence=sequence,
            out_path=_track_parquet(ds, sequence),
            producer="convert-x",
            std_format="trex_v1",
            n_rows=40,
        )

    _, scope = build_manifest(ds, _feature_inputs())
    assert scope.entries == {("", "a"), ("", "b")}


def test_build_manifest_can_select_one_variant(tmp_path: Path) -> None:
    """The surface Stage 3.3 drives. Nothing passes a value in M1."""
    from mosaic.core.pipeline.manifest import build_manifest

    ds = _dataset(tmp_path)
    for sequence, variant in [("a", "convert-x.0.1-aaaaaaaaaa"), ("b", "trex.0.1-bb")]:
        write_tracks_row(
            ds,
            run_id=variant,
            group="",
            sequence=sequence,
            out_path=_track_parquet(ds, sequence),
            producer="convert-x",
            std_format="trex_v1",
            n_rows=40,
        )

    _, scope = build_manifest(ds, _feature_inputs(), tracks_run_id="trex.0.1-bb")
    assert scope.entries == {("", "b")}


def test_an_unknown_variant_raises_but_predicts_empty(tmp_path: Path) -> None:
    """Mirrors ``_resolve_feature``: loud when executing, quiet when predicting."""
    from mosaic.core.pipeline.manifest import build_manifest

    ds = _dataset(tmp_path)
    write_tracks_row(
        ds,
        run_id="convert-x.0.1-aaaaaaaaaa",
        group="",
        sequence="a",
        out_path=_track_parquet(ds, "a"),
        producer="convert-x",
        std_format="trex_v1",
        n_rows=40,
    )

    with pytest.raises(FileNotFoundError, match="no-such-variant"):
        _ = build_manifest(ds, _feature_inputs(), tracks_run_id="no-such-variant")

    _, scope = build_manifest(
        ds,
        _feature_inputs(),
        tracks_run_id="no-such-variant",
        on_missing_run="empty",
    )
    assert scope.entries == set()


def test_an_empty_tracks_index_resolves_empty_rather_than_raising(
    tmp_path: Path,
) -> None:
    """The cold-start state two other code paths depend on.

    ``_read_track_universe``'s glob fallback and ``load_tracks``'s auto-convert
    both rely on this, which is why ``_resolve_tracks`` does not adopt
    ``_resolve_feature``'s all-missing raise.
    """
    from mosaic.core.pipeline.manifest import build_manifest

    ds = _dataset(tmp_path)
    _, scope = build_manifest(ds, _feature_inputs())
    assert scope.entries == set()


def test_two_real_variants_of_an_entry_refuse_to_resolve_unasked(
    tmp_path: Path,
) -> None:
    """The inversion, at the resolver.

    Both rows now survive the read, so ``_resolve_tracks`` is the thing that has
    to choose -- and between two genuinely different recipes it declines, rather
    than taking the last one silently as the collapse used to.
    """
    from mosaic.core.pipeline.manifest import build_manifest

    ds = _dataset(tmp_path)
    _write_two_variant_rows(ds, "a", "convert-x.0.1-aaaaaaaaaa", "trex.0.1-bbbbbbbbbb")

    assert len(read_tracks_index(ds)) == 2
    with pytest.raises(ValueError, match="tracks_run_id"):
        _ = build_manifest(ds, _feature_inputs())

    # Naming one resolves it, and the two identify differently.
    _, first = build_manifest(
        ds, _feature_inputs(), tracks_run_id="convert-x.0.1-aaaaaaaaaa"
    )
    _, second = build_manifest(
        ds, _feature_inputs(), tracks_run_id="trex.0.1-bbbbbbbbbb"
    )
    assert first.entries == second.entries == {("", "a")}
    assert first.tracks_variants != second.tracks_variants


def test_an_empty_tracks_scope_is_not_recorded_as_a_completed_run(
    tmp_path: Path,
) -> None:
    """A run over nothing must not claim it finished something.

    ``_resolve_step_cache`` -- the only caller that predicts rather than
    executes -- is reached from ``Pipeline.run`` and ``.clean`` as well as
    ``.status``, so this is not merely a display problem.
    """
    from mosaic.core.pipeline.index import feature_index, feature_index_path
    from mosaic.core.pipeline.run import run_feature
    from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel

    ds = _dataset(tmp_path)
    ds.set_root("features", str(tmp_path / "features"))
    ds.ensure_roots()

    _ = run_feature(ds, SpeedAngvel())

    index_path = feature_index_path(ds, "speed-angvel__from__tracks")
    if not index_path.exists():
        return  # nothing recorded at all is the strongest form of the guarantee
    rows = feature_index(index_path).read()
    assert "__global__" not in {str(s) for s in rows["sequence"]}, (
        "an empty tracks scope was recorded as a finished per-frame run"
    )


# --- the measured frame extent ---


def _write_table(path: Path, start: int, n_frames: int, n_ids: int = 1) -> None:
    frames = [start + f for f in range(n_frames) for _ in range(n_ids)]
    ids = [i for _ in range(n_frames) for i in range(n_ids)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": frames, "id": ids, "X": 0.0, "Y": 0.0}).to_parquet(path)


def test_frame_extent_is_measured_from_the_table(tmp_path: Path) -> None:
    """Measured, not passed: no call site can record an extent the table denies."""
    path = tmp_path / "t.parquet"
    _write_table(path, start=10, n_frames=5, n_ids=3)
    assert frame_extent(path) == (10, 14)


def test_frame_extent_is_unknown_rather_than_zero_when_it_cannot_be_read(
    tmp_path: Path,
) -> None:
    """An unanswerable question must not answer ``(0, 0)``.

    Frame 0 is where the first sequence of every recording begins, so a
    substituted zero would read as a measurement rather than as its absence.
    """
    assert frame_extent(tmp_path / "missing.parquet") is None
    no_frames = tmp_path / "n.parquet"
    pd.DataFrame({"id": [1]}).to_parquet(no_frames)
    assert frame_extent(no_frames) is None


def test_a_blank_extent_cell_reads_as_unknown_not_as_frame_zero() -> None:
    """The trap this column has that ``n_keypoints`` does not."""
    assert read_frame_extent(pd.Series({"frame_min": "", "frame_max": ""})) is None
    assert read_frame_extent(pd.Series({"frame_min": "0", "frame_max": "9"})) == (0, 9)
    # A half-written pair is unknown, not a range starting at zero.
    assert read_frame_extent(pd.Series({"frame_min": "0", "frame_max": ""})) is None


def test_write_tracks_row_records_the_extent(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    out = ds.get_root("tracks") / "g__s1.parquet"
    _write_table(out, start=100, n_frames=4)
    write_tracks_row(
        ds,
        run_id="v1",
        group="g",
        sequence="s1",
        out_path=out,
        producer="convert-x",
        std_format="mosaic_v1",
        n_rows=4,
    )
    assert read_frame_extents(ds) == {("g", "s1"): (100, 103)}


def test_backfill_measures_only_the_rows_that_lack_an_extent(tmp_path: Path) -> None:
    """The migration for a dataset converted before the columns existed.

    Nothing else in the toolkit re-measures an index column: ``reindex`` prunes
    rows without opening a file, and ``reconcile`` re-addresses runs from their
    sidecars.
    """
    ds = _dataset(tmp_path)
    rows = []
    for index, name in enumerate(("s1", "s2")):
        path = ds.get_root("tracks") / f"g__{name}.parquet"
        _write_table(path, start=index * 10, n_frames=10)
        rows.append({"group": "g", "sequence": name, "abs_path": str(path)})
    pd.DataFrame(rows).to_csv(tracks_index_path(ds), index=False)

    assert read_frame_extents(ds) == {}

    would = backfill_frame_extents(ds, dry_run=True)
    assert len(would) == 2
    assert read_frame_extents(ds) == {}, "a dry run must not write"

    filled = backfill_frame_extents(ds, dry_run=False)
    assert len(filled) == 2
    assert read_frame_extents(ds) == {("g", "s1"): (0, 9), ("g", "s2"): (10, 19)}

    assert len(backfill_frame_extents(ds, dry_run=False)) == 0, "not idempotent"
