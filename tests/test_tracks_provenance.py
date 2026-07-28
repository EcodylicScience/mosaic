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
from mosaic.core.pipeline.tracks_index import read_tracks_index


def _dataset(base: Path) -> Dataset:
    base.mkdir(parents=True, exist_ok=True)
    ds = Dataset(
        manifest_path=base / "dataset.yaml",
        roots={
            "tracks_raw": str(base / "tracks_raw"),
            "tracks": str(base / "tracks"),
            "trex": str(base / "tracks_raw" / "trex"),
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
    np.savez(
        path,
        time=np.arange(n, dtype=float),
        poseX0=rng.random(n),
        poseY0=rng.random(n),
    )


def _one_row(ds: Dataset) -> pd.Series:
    df = read_tracks_index(ds)
    assert len(df) == 1, f"expected exactly one row, got {len(df)}"
    return df.iloc[0]


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
        video_path=video,
        overwrite=True,
    )

    assert written == 8
    row = _one_row(ds)
    assert str(row["producer"]) == "trex"
    assert str(row["run_id"]) == "trex.0.1-bbbbbbbbbb"
    # The tracker run and the tracks variant are separate columns, so an
    # op-version bump does not relocate the table.
    assert str(row["producer_run_id"]) == "trex.0.1-cccccccccc"
    assert str(row["source_abs_path"]).startswith("tracks_raw/trex/")
    assert set(str(row["consumed_source_roots"]).split(",")) == {"trex", "media_raw"}
    _assert_portable(ds, row)


def test_the_tracker_bridge_prefers_the_deepest_root_it_read(tmp_path: Path) -> None:
    """``trex`` nests under ``tracks_raw``; naming the parent would lose which."""
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
        video_path=video,
        overwrite=True,
    )

    roots = set(str(_one_row(ds)["consumed_source_roots"]).split(","))
    assert "trex" in roots
    assert "tracks_raw" not in roots


# --- writer 5: the inference bridge ----------------------------------------


def test_the_inference_bridge_points_back_at_its_predictions(tmp_path: Path) -> None:
    """``source_abs_path`` was empty here; it now names the prediction directory.

    That pointer is what item 8.7 needs in order to retire the ``predictions``
    root -- the tracks row records which model run produced the table.
    """
    from mosaic.tracking.ops.infer import _bridge_df_to_tracks

    ds = _dataset(tmp_path)
    ds.set_root("predictions", str(tmp_path / "predictions"))
    seq_dir = ds.get_root("predictions") / "infer-points" / "run" / "vid1"
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
    assert str(row["source_abs_path"]).startswith("predictions/infer-points/")
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
    ds.set_root("predictions", str(tmp_path / "predictions"))
    _trex_npz(ds.get_root("tracks_raw") / "vid1.npz")
    _ = ds.index_tracks_raw(
        [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
    )
    ds.convert_all_tracks()
    assert str(_one_row(ds)["producer"]) == "convert-trex_npz"

    seq_dir = ds.get_root("predictions") / "infer-points" / "run" / "vid1"
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
    assert "were not rewritten" in err
    assert "old_name" in err
    assert "drop_entries" in err
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
