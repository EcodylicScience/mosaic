"""What each of the five tracks writers records about where a table came from.

Three producers in two packages write into ``tracks/``, and before Stage 2 the
index said nothing about which: no run identity, no producer, a ``source_abs_path``
that one bridge left empty and another filled with a bare absolute directory, and
nothing anywhere reading either.

One test per writer, asserting the four provenance columns and that both stored
paths are root-relative. The TREx bridge is called directly rather than through
``run_trex`` -- the tracker needs a real binary -- which is also the only way to
observe its source pointer becoming portable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.op_identity import parse_op_run_id
from mosaic.core.pipeline.tracks_index import read_tracks_index, tracks_index_path

from tests.helpers import write_trex_npz


def _dataset(base: Path) -> Dataset:
    base.mkdir(parents=True, exist_ok=True)
    ds = Dataset(
        manifest_path=base / "dataset.yaml",
        roots={
            "tracks_raw": str(base / "tracks_raw"),
            "tracks": str(base / "tracks"),
            "_tracking": str(base / "_tracking"),
            "trex": str(base / "_tracking" / "trex"),
            "sleap": str(base / "_tracking" / "sleap"),
            "litpose": str(base / "_tracking" / "litpose"),
            "media_raw": str(base / "media_raw"),
            "models": str(base / "models"),
        },
    )
    ds.ensure_roots()
    ds.save()
    return ds


def _trex_npz(path: Path, *, n: int = 8, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    write_trex_npz(
        path,
        n=n,
        time=np.arange(n, dtype=float),
        poseX0=rng.random(n),
        poseY0=rng.random(n),
    )


def _one_row(ds: Dataset) -> pd.Series:
    df = read_tracks_index(ds)
    assert len(df) == 1, f"expected exactly one row, got {len(df)}"
    return df.iloc[0]


def _track_parquet(ds: Dataset, sequence: str, n_rows: int = 40) -> Path:
    """A real parquet under tracks/, for tests that only need a row to point at."""
    root = ds.get_root("tracks")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{sequence}.parquet"
    pd.DataFrame({"frame": range(n_rows), "id": [0] * n_rows}).to_parquet(path)
    return path


def _tables_by_sequence(ds: Dataset) -> dict[str, Path]:
    """``sequence -> resolved parquet path``, read out of the index.

    Tables live under ``tracks/<variant>/``, so a test that wants one asks the
    index where it is rather than assembling the path -- which is what every
    production reader does, and what keeps these assertions layout-agnostic.
    """
    return {
        str(row["sequence"]): ds.resolve_path(str(row["abs_path"]))
        for _, row in read_tracks_index(ds).iterrows()
    }


def _assert_portable(ds: Dataset, row: pd.Series) -> None:
    """Neither stored path may be absolute, or the index dies on a move."""
    for column in ("abs_path", "source_abs_path"):
        stored = str(row[column])
        if not stored:
            continue
        assert not Path(stored).is_absolute(), f"{column} is not portable: {stored}"
        assert ds.resolve_path(stored).exists(), f"{column} does not resolve: {stored}"


# --- writers 1 and 2: convert_one_track ------------------------------------


def test_a_converted_table_names_its_converter_and_its_upload(tmp_path: Path) -> None:
    """The single-sequence branch: one raw file, one entry."""
    ds = _dataset(tmp_path)
    _trex_npz(ds.get_root("tracks_raw") / "seq_a.npz")
    _ = ds.index_tracks_raw(
        [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
    )
    ds.convert_all_tracks()

    row = _one_row(ds)
    assert str(row["producer"]) == "convert-trex_npz"
    assert str(row["run_id"]).startswith("convert-trex_npz.")
    # A conversion has no op run behind it, so this is honestly empty rather
    # than a copy of run_id.
    assert str(row["producer_run_id"]) == ""
    assert str(row["consumed_source_roots"]) == "tracks_raw"
    assert str(row["source_abs_path"]).startswith("tracks_raw/")
    assert int(row["n_rows"]) == 8
    _assert_portable(ds, row)


def test_the_producer_column_is_the_run_ids_own_kind(tmp_path: Path) -> None:
    """Redundant on purpose, and the redundancy must not drift.

    ``producer`` is the segment Stage 3.2 turns into a directory name, so it has
    to agree with what ``parse_op_run_id`` reads back out of ``run_id``.
    """
    ds = _dataset(tmp_path)
    _trex_npz(ds.get_root("tracks_raw") / "seq_a.npz")
    _ = ds.index_tracks_raw(
        [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
    )
    ds.convert_all_tracks()

    row = _one_row(ds)
    parsed = parse_op_run_id(str(row["run_id"]))
    assert parsed is not None
    assert parsed.kind == str(row["producer"])


def test_one_recipe_is_one_variant_across_every_sequence(tmp_path: Path) -> None:
    """Params-only and scope-free: three sequences, one run_id."""
    ds = _dataset(tmp_path)
    for name in ("seq_a", "seq_b", "seq_c"):
        _trex_npz(ds.get_root("tracks_raw") / f"{name}.npz")
    _ = ds.index_tracks_raw(
        [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
    )
    ds.convert_all_tracks()

    df = read_tracks_index(ds)
    assert len(df) == 3
    assert len(set(df["run_id"])) == 1


# --- writer 3: convert_all_tracks, the trex_npz merge ----------------------


def test_a_merged_multi_id_table_records_the_first_of_its_sources(
    tmp_path: Path,
) -> None:
    """Two per-individual NPZ merge into one table.

    ``source_abs_path`` names only the first, as it always has -- the full set
    stays recoverable from ``tracks_raw/index.csv``. What is new is that it is
    root-relative and that the row says which recipe merged them.
    """
    ds = _dataset(tmp_path)
    _trex_npz(ds.get_root("tracks_raw") / "seq_a_fish0.npz", seed=1)
    _trex_npz(ds.get_root("tracks_raw") / "seq_a_fish1.npz", seed=2)
    _ = ds.index_tracks_raw(
        [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
    )
    ds.convert_all_tracks()

    row = _one_row(ds)
    assert str(row["producer"]) == "convert-trex_npz"
    assert str(row["run_id"]).startswith("convert-trex_npz.")
    assert str(row["consumed_source_roots"]) == "tracks_raw"
    assert int(row["n_rows"]) == 16
    _assert_portable(ds, row)


# --- writer 4: the TREx bridge ---------------------------------------------


def test_the_tracker_bridge_records_the_run_and_a_portable_source(
    tmp_path: Path,
) -> None:
    """The one non-portable value in any index, now stored like the rest.

    It used to be ``str(npz_paths[0].parent)`` -- a bare absolute directory that
    did not survive a move or a sync between machines.
    """
    from mosaic.tracking.trex.dataset_runs import _bridge_npz_to_tracks

    ds = _dataset(tmp_path)
    seq_dir = ds.get_root("trex") / "trex.0.1-aaaaaaaaaa" / "vid1" / "data"
    npz = seq_dir / "vid1_fish0.npz"
    _trex_npz(npz)
    video = ds.get_root("media_raw") / "vid1.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"v")

    written = _bridge_npz_to_tracks(
        ds,
        "",
        "vid1",
        [npz],
        tracks_variant="trex.0.1-bbbbbbbbbb",
        producer_run_id="trex.0.1-cccccccccc",
        video_paths=[video],
        timeline=None,
        overwrite=True,
    )

    assert written is not None
    assert written.n_rows == 8
    row = _one_row(ds)
    assert str(row["producer"]) == "trex"
    assert str(row["run_id"]) == "trex.0.1-bbbbbbbbbb"
    # The tracker run and the tracks variant are separate columns, so an
    # op-version bump does not relocate the table.
    assert str(row["producer_run_id"]) == "trex.0.1-cccccccccc"
    assert str(row["source_abs_path"]).startswith("_tracking/trex/")
    assert set(str(row["consumed_source_roots"]).split(",")) == {"trex", "media_raw"}
    _assert_portable(ds, row)


def test_the_tracker_bridge_prefers_the_deepest_root_it_read(tmp_path: Path) -> None:
    """``trex`` nests under ``_tracking``; naming the parent would lose which."""
    from mosaic.tracking.trex.dataset_runs import _bridge_npz_to_tracks

    ds = _dataset(tmp_path)
    npz = ds.get_root("trex") / "run" / "vid1" / "data" / "vid1_fish0.npz"
    _trex_npz(npz)
    video = ds.get_root("media_raw") / "vid1.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"v")

    _ = _bridge_npz_to_tracks(
        ds,
        "",
        "vid1",
        [npz],
        tracks_variant="trex.0.1-bbbbbbbbbb",
        producer_run_id="trex.0.1-cccccccccc",
        video_paths=[video],
        timeline=None,
        overwrite=True,
    )

    roots = set(str(_one_row(ds)["consumed_source_roots"]).split(","))
    assert "trex" in roots
    assert "_tracking" not in roots
    assert "tracks_raw" not in roots


# --- writer 5: the inference bridge ----------------------------------------


def test_the_inference_bridge_points_back_at_its_predictions(tmp_path: Path) -> None:
    """``source_abs_path`` was empty here; it now names the prediction directory.

    That pointer is what item 8.7 needs in order to retire the ``predictions``
    root -- the tracks row records which model run produced the table.
    """
    from mosaic.tracking.ops.infer import _bridge_df_to_tracks

    ds = _dataset(tmp_path)
    ds.set_root("infer-points", str(tmp_path / "_tracking" / "infer-points"))
    seq_dir = ds.get_root("infer-points") / "run" / "vid1"
    seq_dir.mkdir(parents=True, exist_ok=True)
    video = ds.get_root("media_raw") / "vid1.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"v")
    model = ds.get_root("models") / "points" / "best.pt"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"w")

    frame = pd.DataFrame({"frame": range(5), "poseX0": [1.0] * 5, "poseY0": [2.0] * 5})
    written = _bridge_df_to_tracks(
        ds,
        frame,
        "",
        "vid1",
        tracks_variant="infer-points.0.1-bbbbbbbbbb",
        producer_run_id="infer-points.0.1-bbbbbbbbbb",
        kind="infer-points",
        seq_dir=seq_dir,
        video_path=video,
        model_pt=model,
        overwrite=True,
    )

    assert written == 5
    row = _one_row(ds)
    assert str(row["producer"]) == "infer-points"
    assert str(row["run_id"]) == "infer-points.0.1-bbbbbbbbbb"
    assert str(row["producer_run_id"]) == "infer-points.0.1-bbbbbbbbbb"
    assert str(row["source_abs_path"]).startswith("_tracking/infer-points/")
    assert set(str(row["consumed_source_roots"]).split(",")) == {"media_raw", "models"}
    _assert_portable(ds, row)


# --- the invariant across producers ----------------------------------------


def test_a_second_producer_adds_a_row_rather_than_replacing_the_first(
    tmp_path: Path,
) -> None:
    """Two real producers for one sequence, both surviving -- the point of M2.

    Until Stage 3.2 both targeted the same flat parquet, so the inference bridge
    silently overwrote the conversion's table and the index recorded only the
    last writer. Each now writes into its own variant directory, both tables
    exist, and both rows are kept.
    """
    from mosaic.core.pipeline.tracks_index import select_variant_rows
    from mosaic.tracking.ops.infer import _bridge_df_to_tracks

    ds = _dataset(tmp_path)
    ds.set_root("infer-points", str(tmp_path / "_tracking" / "infer-points"))
    _trex_npz(ds.get_root("tracks_raw") / "vid1.npz")
    _ = ds.index_tracks_raw(
        [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
    )
    ds.convert_all_tracks()
    assert str(_one_row(ds)["producer"]) == "convert-trex_npz"

    seq_dir = ds.get_root("infer-points") / "run" / "vid1"
    seq_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"frame": range(5), "poseX0": [1.0] * 5, "poseY0": [2.0] * 5})
    _ = _bridge_df_to_tracks(
        ds,
        frame,
        "",
        "vid1",
        tracks_variant="infer-points.0.1-bbbbbbbbbb",
        producer_run_id="infer-points.0.1-bbbbbbbbbb",
        kind="infer-points",
        seq_dir=seq_dir,
        video_path=ds.get_root("media_raw") / "vid1.mp4",
        model_pt=ds.get_root("models") / "best.pt",
        overwrite=True,
    )

    rows = read_tracks_index(ds)
    assert len(rows) == 2
    assert set(rows["producer"]) == {"convert-trex_npz", "infer-points"}
    # Two rows, two tables, two directories -- neither overwrote the other.
    tables = {ds.resolve_path(str(path)) for path in rows["abs_path"]}
    assert len({table.parent for table in tables}) == 2
    assert all(table.exists() for table in tables)

    # And an unasked resolution now declines rather than taking the last writer.
    with pytest.raises(ValueError, match="tracks_run_id"):
        _ = select_variant_rows(rows)


# --- item 5.1's tracks half: the composition a table consumed ---------------
#
# ``consumed_source_roots`` says which root a change would have to be under;
# only ``consumed_composition`` says whether it has changed. Until this cell
# existed, a ``tracks_raw`` change moved nothing downstream -- the one dated gap
# in the Stage 4 design, named at items 4.4 and 5.1.


def test_a_converted_table_records_the_composition_it_consumed(
    tmp_path: Path,
) -> None:
    """The recorded cell is the value ``tracks_raw/sequences.csv`` holds."""
    from mosaic.core.pipeline.sequence_index import read_sequence_index

    ds = _dataset(tmp_path)
    _trex_npz(ds.get_root("tracks_raw") / "seq_a.npz")
    _ = ds.index_tracks_raw(
        [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
    )
    ds.convert_all_tracks()

    recorded = str(_one_row(ds)["consumed_composition"])
    assert recorded, "a converted table recorded no composition"

    projected = read_sequence_index(ds, "tracks_raw")
    expected = {
        (str(row["group"]), str(row["sequence"])): str(row["composition"])
        for _, row in projected.iterrows()
    }
    assert recorded == expected[("", "seq_a")]


def test_a_changed_source_moves_the_composition_a_reconversion_records(
    tmp_path: Path,
) -> None:
    """The dated gap, closed: a ``tracks_raw`` change is now visible downstream.

    Nothing about the *recipe* changes here -- same converter, same version, same
    params -- so the variant identity is deliberately unmoved. What moves is the
    recorded edge, which is exactly the split item 3.1 fixes: the name says how a
    table was produced, the row says what it was produced from.
    """
    ds = _dataset(tmp_path)
    source = ds.get_root("tracks_raw") / "seq_a.npz"
    _trex_npz(source, seed=0)
    _ = ds.index_tracks_raw(
        [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
    )
    ds.convert_all_tracks()
    before = _one_row(ds)
    variant_before = str(before["run_id"])
    composition_before = str(before["consumed_composition"])
    assert composition_before

    # Same sequence, different bytes -- a corrected upload.
    _trex_npz(source, seed=1)
    _ = ds.index_tracks_raw(
        [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
    )
    ds.convert_all_tracks(overwrite=True)

    after = _one_row(ds)
    assert str(after["consumed_composition"]) != composition_before, (
        "a changed source left the recorded composition alone"
    )
    assert str(after["run_id"]) == variant_before, (
        "the recipe did not change, so the variant identity must not move"
    )


def test_a_derived_root_contributes_nothing_to_the_composition(
    tmp_path: Path,
) -> None:
    """Only source roots can answer, and the bridges legitimately name others.

    The TREx bridge records ``{trex, media_raw}`` and the inference bridge
    ``{media_raw, models}``. Neither ``trex`` nor ``models`` holds anything that
    cannot be recomputed, so neither has a composition -- and asking for one must
    yield nothing rather than an empty-string member that would compare equal to
    a genuinely different sequence.
    """
    from mosaic.core.pipeline.tracks_index import consumed_composition_for

    ds = _dataset(tmp_path)
    _trex_npz(ds.get_root("tracks_raw") / "seq_a.npz")
    _ = ds.index_tracks_raw(
        [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
    )

    assert consumed_composition_for(ds, "", "seq_a", ["trex", "models"]) == ""
    # ... while the source root among them still answers.
    assert consumed_composition_for(ds, "", "seq_a", ["trex", "tracks_raw"]) != ""


def test_an_unindexed_source_records_nothing_rather_than_guessing(
    tmp_path: Path,
) -> None:
    """No projection yet is *absent*, not "composed of nothing"."""
    from mosaic.core.pipeline.tracks_index import consumed_composition_for

    ds = _dataset(tmp_path)
    assert consumed_composition_for(ds, "", "never_indexed", ["tracks_raw"]) == ""


# --- superseded entries ----------------------------------------------------
#
# A converter that changes how it spells an entry writes rows under the new
# names while the old ones stay. Both then resolve, and every feature runs over
# each sequence twice. Reported, never repaired automatically.


def test_a_conversion_that_rewrites_the_same_names_says_nothing(
    tmp_path: Path, capsys
) -> None:
    """A normal re-conversion must stay quiet, or the warning is noise."""
    ds = _dataset(tmp_path)
    _trex_npz(ds.get_root("tracks_raw") / "seq_a.npz")
    _ = ds.index_tracks_raw(
        [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
    )
    ds.convert_all_tracks()
    _ = capsys.readouterr()

    ds.convert_all_tracks(overwrite=True)

    assert "were not rewritten" not in capsys.readouterr().err


def test_an_entrys_stamp_is_its_latest_whatever_the_row_order(tmp_path: Path) -> None:
    """One stamp per entry, taken as the newest rather than the last row.

    The supersede warning asks "did this call touch this entry", which an entry
    carrying several variants can only answer by looking across its rows. The
    dict comprehension this replaces also produced one stamp per entry -- last
    row wins -- and was right in practice only because ``IndexCSV`` re-appends a
    rewritten row at the end, so the newest stamp happened to be last. Here the
    newest row is written first, which is the arrangement that told them apart.
    """
    from mosaic.core.pipeline.tracks_index import write_tracks_row

    ds = _dataset(tmp_path)
    path = _track_parquet(ds, "seq_a")
    for run_id, stamp in [
        ("convert-x.0.1-newer0000", "2026-07-28T12:00:00+00:00"),
        ("convert-x.0.1-older0000", "2026-07-01T12:00:00+00:00"),
    ]:
        write_tracks_row(
            ds,
            run_id=run_id,
            group="",
            sequence="seq_a",
            out_path=path,
            producer="convert-x",
            std_format="trex_v1",
            n_rows=40,
        )
        frame = read_tracks_index(ds)
        frame.loc[frame["run_id"] == run_id, "started_at"] = stamp
        frame.to_csv(tracks_index_path(ds), index=False)

    assert list(read_tracks_index(ds)["run_id"]) == [
        "convert-x.0.1-newer0000",
        "convert-x.0.1-older0000",
    ]
    assert ds._entry_stamps()[("", "seq_a")] == "2026-07-28T12:00:00+00:00"


def test_drop_entries_can_retire_one_variant_and_keep_the_rest(
    tmp_path: Path,
) -> None:
    """Retiring a recipe is also how an ambiguous entry stops being ambiguous."""
    from mosaic.core.pipeline.tracks_index import select_variant_rows, write_tracks_row

    ds = _dataset(tmp_path)
    for run_id in ("convert-x.0.1-aaaaaaaaaa", "trex.0.1-bbbbbbbbbb"):
        write_tracks_row(
            ds,
            run_id=run_id,
            group="",
            sequence="seq_a",
            out_path=_track_parquet(ds, "seq_a"),
            producer=run_id.split(".")[0],
            std_format="trex_v1",
            n_rows=40,
        )
    with pytest.raises(ValueError, match="tracks_run_id"):
        _ = select_variant_rows(read_tracks_index(ds))

    assert ds.drop_entries([("", "seq_a")], run_id="trex.0.1-bbbbbbbbbb") == 1

    remaining = read_tracks_index(ds)
    assert list(remaining["run_id"]) == ["convert-x.0.1-aaaaaaaaaa"]
    assert len(select_variant_rows(remaining)) == 1


def test_drop_entries_without_a_variant_takes_every_row(tmp_path: Path) -> None:
    """The default stays what a rename cleanup wants: the entry, entirely."""
    from mosaic.core.pipeline.tracks_index import write_tracks_row

    ds = _dataset(tmp_path)
    for run_id in ("convert-x.0.1-aaaaaaaaaa", "trex.0.1-bbbbbbbbbb"):
        write_tracks_row(
            ds,
            run_id=run_id,
            group="",
            sequence="seq_a",
            out_path=_track_parquet(ds, "seq_a"),
            producer=run_id.split(".")[0],
            std_format="trex_v1",
            n_rows=40,
        )
    assert ds.drop_entries([("", "seq_a")]) == 2
    assert len(read_tracks_index(ds)) == 0


def test_an_entry_left_behind_by_a_conversion_is_reported(
    tmp_path: Path, capsys
) -> None:
    """The CalMS21 0.2 rename is what this exists for."""
    ds = _dataset(tmp_path)
    _trex_npz(ds.get_root("tracks_raw") / "old_name.npz")
    _ = ds.index_tracks_raw(
        [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
    )
    ds.convert_all_tracks()

    # Stand in for a converter that started spelling its entries differently.
    (ds.get_root("tracks_raw") / "old_name.npz").rename(
        ds.get_root("tracks_raw") / "new_name.npz"
    )
    _ = ds.index_tracks_raw(
        [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
    )
    _ = capsys.readouterr()
    ds.convert_all_tracks()

    err = capsys.readouterr().err
    assert "claimed by no current raw source" in err
    assert "old_name" in err
    assert "drop_entries" in err
    # The remedy has to name run_id="": the default drops every variant of the
    # entry, which with delete_files=True deletes the conversion just made.
    assert 'run_id=""' in err
    # Both resolve until the user acts -- which is the problem being reported.
    assert len(read_tracks_index(ds)) == 2


def test_drop_entries_removes_the_row_and_optionally_the_table(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path)
    for name in ("keep", "drop"):
        _trex_npz(ds.get_root("tracks_raw") / f"{name}.npz")
    _ = ds.index_tracks_raw(
        [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
    )
    ds.convert_all_tracks()
    # Resolved through the index rather than assembled here: the tables live
    # under tracks/<variant>/, and the index is what every reader consults.
    tables = _tables_by_sequence(ds)
    assert tables["drop"].exists()
    assert tables["keep"].exists()

    dropped = ds.drop_entries([("", "drop")], delete_files=True)

    assert dropped == 1
    assert not tables["drop"].exists()
    assert [str(s) for s in read_tracks_index(ds)["sequence"]] == ["keep"]
    # The kept row's table is untouched.
    assert tables["keep"].exists()


def test_drop_entries_keeps_the_table_by_default(tmp_path: Path) -> None:
    """An orphaned table is recoverable; a deleted one is not."""
    ds = _dataset(tmp_path)
    _trex_npz(ds.get_root("tracks_raw") / "seq_a.npz")
    _ = ds.index_tracks_raw(
        [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
    )
    ds.convert_all_tracks()
    table = _tables_by_sequence(ds)["seq_a"]

    assert ds.drop_entries([("", "seq_a")]) == 1
    assert table.exists()
    assert len(read_tracks_index(ds)) == 0


def test_dropping_an_absent_entry_is_a_no_op(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    assert ds.drop_entries([("", "never-existed")]) == 0
    assert ds.drop_entries([]) == 0


# --- media matching --------------------------------------------------------


def test_compound_names_do_not_collide_on_a_tail_key(tmp_path: Path) -> None:
    """The keymap registers ``Path(sequence).name`` as a shorthand.

    With slash-path names that shorthand is the last level, so two CalMS21
    entries under different splits both claimed ``m010``. Compound names have no
    slash, so the shorthand is the whole name and the collision does not arise --
    the second reason the previous commit's flattening was worth doing.
    """
    from mosaic.core.dataset import Dataset as _Dataset

    ds = _dataset(tmp_path)
    for split in ("train", "test"):
        _trex_npz(ds.get_root("tracks_raw") / f"task1__{split}__m010.npz")
    _ = ds.index_tracks_raw(
        [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
    )
    ds.convert_all_tracks()

    keymap = ds._build_media_sequence_keymap()
    assert "m010" not in keymap
    for split in ("train", "test"):
        hit = _Dataset._match_media_sequence(keymap, f"task1__{split}__m010")
        assert hit is not None and hit["sequence"] == f"task1__{split}__m010"


def test_a_shared_tail_key_is_refused_rather_than_guessed(capsys) -> None:
    from mosaic.core.dataset import Dataset as _Dataset

    # The collision built directly: two entries whose Path(...).name is "m010".
    # A dataset written before the compound-name change looks exactly like this.
    keymap = {
        "m010": [
            {
                "group": "",
                "sequence": "task1/train/m010",
                "group_safe": "",
                "sequence_safe": "task1%2Ftrain%2Fm010",
            },
            {
                "group": "",
                "sequence": "task1/test/m010",
                "group_safe": "",
                "sequence_safe": "task1%2Ftest%2Fm010",
            },
        ]
    }
    _ = capsys.readouterr()

    assert _Dataset._match_media_sequence(keymap, "m010") is None
    err = capsys.readouterr().err
    assert "matches 2 track entries" in err


def test_one_entry_registering_a_key_twice_still_matches(tmp_path: Path) -> None:
    """Only *distinct* entries are ambiguous; one entry hitting a key is fine."""
    from mosaic.core.dataset import Dataset as _Dataset

    meta = {
        "group": "g",
        "sequence": "s",
        "group_safe": "g",
        "sequence_safe": "s",
    }
    keymap = {"s": [meta, dict(meta)]}

    hit = _Dataset._match_media_sequence(keymap, "s")
    assert hit is not None and hit["sequence"] == "s"


# --- writer 5: the SLEAP bridge --------------------------------------------


def _sleap_analysis_h5(path: Path, *, n: int = 6) -> None:
    """A tiny matlab-layout SLEAP analysis HDF5: 1 track, 1 node, *n* frames."""
    import json as _json

    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    # canonical (frame, track, node, xy) -> matlab (track, xy, node, frame)
    tracks = rng.random((n, 1, 1, 2))
    arr = np.transpose(tracks, (1, 3, 2, 0))
    with h5py.File(str(path), "w") as f:
        d = f.create_dataset("tracks", data=arr)
        d.attrs["dims"] = _json.dumps(["track", "xy", "node", "frame"])


def test_the_sleap_bridge_records_the_run_and_a_portable_source(
    tmp_path: Path,
) -> None:
    """The SLEAP bridge writes the fifth tracks producer path with full provenance."""
    pytest.importorskip("h5py")
    from mosaic.tracking.sleap.dataset_runs import _bridge_analysis_h5_to_tracks

    ds = _dataset(tmp_path)
    seq_dir = ds.get_root("sleap") / "sleap.1.6-aaaaaaaaaa" / "vid1"
    h5 = seq_dir / "vid1.analysis.h5"
    _sleap_analysis_h5(h5)
    video = ds.get_root("media_raw") / "vid1.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"v")

    written = _bridge_analysis_h5_to_tracks(
        ds,
        "",
        "vid1",
        h5,
        tracks_variant="sleap.1.6-bbbbbbbbbb",
        producer_run_id="sleap.1.6-cccccccccc",
        video_path=video,
        model_checkpoints=[],
        fps=30.0,
        overwrite=True,
    )

    assert written is not None
    assert written.n_rows == 6 and written.n_ids == 1
    row = _one_row(ds)
    assert str(row["producer"]) == "sleap"
    assert str(row["run_id"]) == "sleap.1.6-bbbbbbbbbb"
    # the tracker run and the tracks variant are separate columns
    assert str(row["producer_run_id"]) == "sleap.1.6-cccccccccc"
    assert str(row["source_abs_path"]).startswith("_tracking/sleap/")
    # the video (media_raw) and the predictions (sleap); the external model
    # directory sits under no dataset root, so it contributes nothing
    assert set(str(row["consumed_source_roots"]).split(",")) == {"sleap", "media_raw"}
    _assert_portable(ds, row)


# --- writer 6: the Lightning Pose bridge -----------------------------------


def _litpose_csv(path: Path, *, n: int = 6) -> None:
    """A tiny single-animal DeepLabCut / Lightning Pose CSV: 1 bodypart, *n* frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    lines = [
        "scorer,heatmap_tracker,heatmap_tracker,heatmap_tracker",
        "bodyparts,nose,nose,nose",
        "coords,x,y,likelihood",
    ]
    for i in range(n):
        x, y = rng.uniform(0, 100, 2)
        lines.append(f"{i},{x:.6f},{y:.6f},0.9")
    path.write_text("\n".join(lines))


def test_the_litpose_bridge_records_the_run_and_a_portable_source(
    tmp_path: Path,
) -> None:
    """The Lightning Pose bridge writes a tracks producer path with full provenance."""
    from mosaic.tracking.litpose.dataset_runs import _bridge_csv_to_tracks

    ds = _dataset(tmp_path)
    seq_dir = ds.get_root("litpose") / "litpose.2.3-aaaaaaaaaa" / "vid1"
    csv = seq_dir / "vid1.predictions.csv"
    _litpose_csv(csv)
    video = ds.get_root("media_raw") / "vid1.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"v")

    written = _bridge_csv_to_tracks(
        ds,
        "",
        "vid1",
        csv,
        tracks_variant="litpose.2.3-bbbbbbbbbb",
        producer_run_id="litpose.2.3-cccccccccc",
        video_path=video,
        model_files=[],
        fps=30.0,
        overwrite=True,
    )

    assert written is not None
    assert written.n_rows == 6 and written.n_ids == 1
    row = _one_row(ds)
    assert str(row["producer"]) == "litpose"
    assert str(row["run_id"]) == "litpose.2.3-bbbbbbbbbb"
    # the tracker run and the tracks variant are separate columns
    assert str(row["producer_run_id"]) == "litpose.2.3-cccccccccc"
    assert str(row["source_abs_path"]).startswith("_tracking/litpose/")
    # the video (media_raw) and the predictions (litpose); the external model
    # directory sits under no dataset root, so it contributes nothing
    assert set(str(row["consumed_source_roots"]).split(",")) == {"litpose", "media_raw"}
    _assert_portable(ds, row)
