"""Promoting a manual correction.

A second correction of the same sequence is the next revision of a source file,
not a conflict that needs a force every time. These tests pin that, and pin
that the block is about *derivatives* rather than about history.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.entry import EntryKeyCollisionError
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.promotion import (
    correction_revision,
    next_revision,
    promote_correction,
)
from mosaic.core.pipeline.provenance import reached_by
from mosaic.core.pipeline.sequence_index import (
    read_sequence_index,
    sequence_label_path,
    sequence_labels,
)
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.pipeline.tracks_raw_index import read_tracks_raw_index

from tests.helpers import (
    CONVERTED_ENTRIES,
    MediaClip,
    converted_source,
    copy_of,
    latest_snapshot,
    make_dataset,
    write_media_index,
    write_sleap_analysis_h5,
    write_tracktor_recording,
    write_trex_npz,
)


def _corrected(
    tmp_path: Path, name: str = "vid1_fish0.npz", value: float = 1.0
) -> Path:
    """A corrected tracker output, as it would sit in a working directory."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    np.savez(path, X=np.array([value, value]), Y=np.array([value, value]))
    return path.with_suffix(".npz")


# --- O1: an append-only revision series --------------------------------------


def test_a_first_promotion_lands_as_revision_one(tmp_path: Path) -> None:
    """Source, in a source root, under this sequence's own directory."""
    ds = make_dataset(tmp_path / "ds", name="promote")
    correction = _corrected(tmp_path)

    report = promote_correction(
        ds, "", "vid1", correction, src_format="trex_npz", apply=True
    )

    assert report.applied
    assert report.revision == 1
    landed = ds.get_root("tracks_raw") / "vid1" / "corrected.rev1.npz"
    assert landed.exists()
    assert report.promoted == (landed,)


def test_a_second_correction_is_a_revision_not_a_conflict(tmp_path: Path) -> None:
    """O1's resolution, and the assertion that distinguishes it from the other.

    Under the rejected answer this would need a force and would overwrite. Under
    this one both revisions are on disk and addressable, which is what makes an
    earlier correction recoverable after a later one turns out worse.
    """
    ds = make_dataset(tmp_path / "ds", name="promote")

    first = promote_correction(
        ds, "", "vid1", _corrected(tmp_path), src_format="trex_npz", apply=True
    )
    second = promote_correction(
        ds,
        "",
        "vid1",
        _corrected(tmp_path / "b", value=2.0),
        src_format="trex_npz",
        apply=True,
    )

    assert (first.revision, second.revision) == (1, 2)
    series = sorted(p.name for p in (ds.get_root("tracks_raw") / "vid1").glob("*.npz"))
    assert series == ["corrected.rev1.npz", "corrected.rev2.npz"]


def test_the_revision_series_is_read_from_disk(tmp_path: Path) -> None:
    """The files are the series; a stored counter could disagree with them."""
    destination = tmp_path / "seq"
    destination.mkdir()
    assert next_revision(destination) == 1
    (destination / "corrected.rev1.npz").write_bytes(b"")
    (destination / "corrected.rev7.npz").write_bytes(b"")
    assert next_revision(destination) == 8


# --- What the promotion sets in motion ---------------------------------------


def test_a_promotion_moves_the_sequence_composition(tmp_path: Path) -> None:
    """The whole point: no new identity machinery, the existing hash moves.

    Checksums are on by default and the sequence composition already covers
    this root, so a corrected file arriving is enough to invalidate every
    artifact built from it -- which is what makes promotion a *source* change
    rather than a new kind of thing.
    """
    ds = make_dataset(tmp_path / "ds", name="promote")
    _ = promote_correction(
        ds, "", "vid1", _corrected(tmp_path), src_format="trex_npz", apply=True
    )
    before = read_sequence_index(ds, "tracks_raw")

    _ = promote_correction(
        ds,
        "",
        "vid1",
        _corrected(tmp_path / "b", value=2.0),
        src_format="trex_npz",
        apply=True,
    )
    after = read_sequence_index(ds, "tracks_raw")

    first = before[before["sequence"] == "vid1"].iloc[0]["composition"]
    second = after[after["sequence"] == "vid1"].iloc[0]["composition"]
    assert first and second and first != second


def test_the_promoted_file_is_indexed_and_checksummed(tmp_path: Path) -> None:
    """It is source now, so it is scanned and hashed like any other source.

    And the format column is what the caller named, not what promotion assumed.
    """
    ds = make_dataset(tmp_path / "ds", name="promote")

    _ = promote_correction(
        ds, "", "vid1", _corrected(tmp_path), src_format="trex_npz", apply=True
    )

    rows = read_tracks_raw_index(ds.get_root("tracks_raw") / "index.csv")
    assert len(rows) == 1
    assert rows[0]["sequence"] == "vid1"
    assert rows[0]["src_format"] == "trex_npz"
    assert rows[0]["md5"], "a promoted correction must be checksummed"


def test_the_promoted_format_is_the_callers_to_name(tmp_path: Path) -> None:
    """A correction from a SLEAP run is not a TREx one.

    ``_tracking`` holds three trackers' working directories and promotion serves
    all of them, so the format travels from the caller into the index rather
    than being assumed. Only the column is under test here -- the promoted bytes
    are copied unread, so they need not be a real SLEAP export.
    """
    ds = make_dataset(tmp_path / "ds", name="promote")

    _ = promote_correction(
        ds,
        "",
        "vid1",
        _corrected(tmp_path),
        src_format="sleap_analysis_h5",
        apply=True,
    )

    rows = read_tracks_raw_index(ds.get_root("tracks_raw") / "index.csv")
    assert rows[0]["src_format"] == "sleap_analysis_h5"


def test_the_lineage_is_recorded_without_touching_the_label(tmp_path: Path) -> None:
    """``derived_from`` gets its own writer, and must not un-name a sequence.

    Folding this into ``set_display_name`` would make one gesture mean two
    things: a promotion that clears a display name, a rename that claims a
    lineage.
    """
    ds = make_dataset(tmp_path / "ds", name="promote")
    ds.set_display_name("", "vid1", "Trial 1 (north tank)")

    _ = promote_correction(
        ds,
        "",
        "vid1",
        _corrected(tmp_path),
        src_format="trex_npz",
        derived_from="trex.1.0-abcdef0123",
        apply=True,
    )

    labels = sequence_labels(sequence_label_path(ds)).read()
    row = labels[labels["sequence"] == "vid1"].iloc[0]
    assert row["derived_from"] == "trex.1.0-abcdef0123"
    assert row["display_name"] == "Trial 1 (north tank)"


# --- The block, and what forcing does and does not do ------------------------


def test_a_dry_run_promotes_nothing(tmp_path: Path) -> None:
    """Preview is the default, and it must not be one in name only."""
    ds = make_dataset(tmp_path / "ds", name="promote")

    report = promote_correction(
        ds, "", "vid1", _corrected(tmp_path), src_format="trex_npz"
    )

    assert not report.applied
    assert report.promoted == ()
    assert not (ds.get_root("tracks_raw") / "vid1").exists()


def test_existing_derivatives_block_the_promotion(tmp_path: Path) -> None:
    """P4's rule at the gesture: a source change blocks while derivatives exist.

    ``reached_by`` is documented as answering *membership* when run before the
    change, which is what makes one function serve the preview and the audit.
    """
    ds = make_dataset(tmp_path / "ds", name="promote")
    _prior_derivative(ds)

    report = promote_correction(
        ds, "", "vid1", _corrected(tmp_path), src_format="trex_npz", apply=True
    )

    assert report.blocked
    assert not report.would_proceed
    assert not report.applied
    assert not (ds.get_root("tracks_raw") / "vid1" / "corrected.rev1.npz").exists()


def test_forcing_promotes_but_deletes_nothing(tmp_path: Path) -> None:
    """Promotion refuses to be two destructive operations wearing one name.

    Forcing past the block promotes; removing what became stale stays
    ``delete_set``'s gesture, behind its own force. A promote that also deleted
    would make the safer-sounding flag the more destructive one.
    """
    ds = make_dataset(tmp_path / "ds", name="promote")
    derivative = _prior_derivative(ds)

    report = promote_correction(
        ds,
        "",
        "vid1",
        _corrected(tmp_path),
        src_format="trex_npz",
        apply=True,
        force=True,
    )

    assert report.applied
    assert not report.blocked
    assert derivative.exists(), "promotion deleted a derivative"


def test_a_missing_source_raises_rather_than_reporting_success(tmp_path: Path) -> None:
    """Nothing to promote is a caller error, not an empty promotion."""
    ds = make_dataset(tmp_path / "ds", name="promote")

    with pytest.raises(FileNotFoundError, match="nothing to promote"):
        _ = promote_correction(
            ds, "", "vid1", tmp_path / "absent.npz", src_format="trex_npz", apply=True
        )


def _prior_derivative(ds: Dataset) -> Path:
    """A tracks table recorded as built from this sequence's ``tracks_raw``."""
    from mosaic.core.pipeline.tracks_index import write_tracks_row

    tracks = ds.get_root("tracks") / "convert-trex_npz.0.1-aaaaaaaaaa"
    tracks.mkdir(parents=True, exist_ok=True)
    table = tracks / "vid1.parquet"
    pd.DataFrame({"frame": [0, 1]}).to_parquet(table)
    write_tracks_row(
        ds,
        group="",
        sequence="vid1",
        out_path=table,
        run_id="convert-trex_npz.0.1-aaaaaaaaaa",
        producer="convert-trex_npz",
        producer_run_id="",
        consumed_source_roots=["tracks_raw"],
        std_format="trex_v1",
        n_rows=2,
        carries_tracklets=None,
    )
    return table


def test_a_multi_file_promotion_keeps_every_file(tmp_path: Path) -> None:
    """One revision, several members -- the shape TRex produces.

    The revision is one event: a correction of a sequence whose tracker wrote a
    file per individual. Naming every member ``corrected.rev<N>`` gave them all
    one destination, so each copy overwrote the last and only the final file
    survived -- with the index then recording that survivor's checksum, so
    nothing downstream could tell the others were gone.
    """
    ds = make_dataset(tmp_path / "ds", name="promote")
    first = _corrected(tmp_path, name="vid1_fish0.npz", value=1.0)
    second = _corrected(tmp_path, name="vid1_fish1.npz", value=2.0)

    report = promote_correction(
        ds, "", "vid1", [first, second], src_format="trex_npz", apply=True
    )

    assert report.applied
    assert len(set(report.promoted)) == 2, report.promoted
    for landed in report.promoted:
        assert landed.exists()
    # Both individuals survive, distinguishably.
    values = sorted(float(np.load(landed)["X"][0]) for landed in set(report.promoted))
    assert values == [1.0, 2.0]
    # Still one revision: the members belong to the same correction.
    assert report.revision == 1
    assert all(".rev1" in landed.name for landed in report.promoted)


def _corrected_table(tmp_path: Path, name: str) -> Path:
    """A corrected table of a format that reads one file per sequence."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    _ = path.write_text("frame\tx\ty\n0\t1.0\t1.0\n", encoding="utf-8")
    return path


def _files_under(root: Path) -> list[Path]:
    """Every file below *root*, which is none when *root* does not exist."""
    return sorted(path for path in root.rglob("*") if path.is_file())


def test_several_files_of_a_one_file_per_sequence_format_are_refused_in_a_dry_run(
    tmp_path: Path,
) -> None:
    """A conversion keeps one file per sequence, so the others would be lost."""
    ds = make_dataset(tmp_path / "ds", name="promote")
    first = _corrected_table(tmp_path, "a.tsv")
    second = _corrected_table(tmp_path, "b.tsv")

    with pytest.raises(ValueError) as refusal:
        _ = promote_correction(
            ds,
            "day1",
            "trial01",
            [first, second],
            src_format="tracktor_corrected_tsv",
        )

    message = str(refusal.value)
    assert "tracktor_corrected_tsv" in message
    assert "day1__trial01" in message
    assert str(first) in message
    assert str(second) in message
    assert _files_under(ds.get_root("tracks_raw")) == []


def test_several_files_of_a_one_file_per_sequence_format_are_refused_on_apply(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds", name="promote")
    first = _corrected_table(tmp_path, "vid1_a.tsv")
    second = _corrected_table(tmp_path, "vid1_b.tsv")

    with pytest.raises(ValueError, match="tracktor_corrected_tsv"):
        _ = promote_correction(
            ds,
            "",
            "vid1",
            [first, second],
            src_format="tracktor_corrected_tsv",
            apply=True,
        )

    assert _files_under(ds.get_root("tracks_raw")) == []


def test_several_files_of_an_enumerable_format_are_refused(tmp_path: Path) -> None:
    """A promoted file names its entry, so an enumerable format reads it as one file."""
    ds = make_dataset(tmp_path / "ds", name="promote")
    tmp_path.mkdir(parents=True, exist_ok=True)
    files = [tmp_path / "vid1_a.npy", tmp_path / "vid1_b.npy"]
    for each in files:
        _ = each.write_bytes(b"")

    with pytest.raises(ValueError, match="calms21_npy"):
        _ = promote_correction(
            ds, "", "vid1", files, src_format="calms21_npy", apply=True
        )

    assert _files_under(ds.get_root("tracks_raw")) == []


def test_the_refusal_is_not_lifted_by_force(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds", name="promote")
    files = [
        _corrected_table(tmp_path, "vid1_a.tsv"),
        _corrected_table(tmp_path, "vid1_b.tsv"),
    ]

    with pytest.raises(ValueError, match="tracktor_corrected_tsv"):
        _ = promote_correction(
            ds,
            "",
            "vid1",
            files,
            src_format="tracktor_corrected_tsv",
            apply=True,
            force=True,
        )

    assert _files_under(ds.get_root("tracks_raw")) == []


def test_one_file_of_a_one_file_per_sequence_format_still_promotes(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds", name="promote")

    report = promote_correction(
        ds,
        "",
        "vid1",
        [_corrected_table(tmp_path, "vid1.tsv")],
        src_format="tracktor_corrected_tsv",
        apply=True,
    )

    assert report.applied
    assert len(report.promoted) == 1
    assert report.promoted[0].exists()


def test_several_files_of_a_merging_format_promote_in_a_dry_run(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds", name="promote")
    files = [
        _corrected(tmp_path, name="vid1_fish0.npz"),
        _corrected(tmp_path, name="vid1_fish1.npz"),
    ]

    report = promote_correction(ds, "", "vid1", files, src_format="trex_npz")

    assert report.would_proceed
    assert not report.applied


# --- a correction converts as its own variant ---------------------------------


def _trex_npz(path: Path, value: float) -> None:
    write_trex_npz(
        path,
        n=5,
        columns={
            "X": np.full(5, value),
            "Y": np.full(5, value),
            "X#wcentroid": np.full(5, value),
            "Y#wcentroid": np.full(5, value),
            "poseX": np.stack([np.full(5, value)] * 2, axis=1),
            "poseY": np.stack([np.full(5, value)] * 2, axis=1),
        },
    )


def test_a_correction_converts_as_its_own_variant(tmp_path: Path) -> None:
    """Both readings survive, distinguishable, rather than merging into one table.

    A promoted correction and the upload it corrects are the same format for the
    same entry, so the conversion's merge key put them in one group: the table
    then held *both* readings of every frame under one identifier, and every
    per-frame feature saw each frame twice with contradictory coordinates.

    They are two variants, which is what ``tracks/`` being a contract root means
    -- both legitimate, comparable, and selectable side by side.
    """
    import mosaic.core.track_library  # noqa: F401  -- registers trex_npz

    ds = make_dataset(tmp_path / "ds", name="promote")
    raw = ds.get_root("tracks_raw")
    _trex_npz(raw / "vidA_fish0.npz", 1.0)
    _ = ds.index_tracks_raw([raw], patterns="*.npz", src_format="trex_npz")
    ds.convert_all_tracks()
    uncorrected = sorted(p.name for p in ds.get_root("tracks").iterdir() if p.is_dir())
    assert len(uncorrected) == 1

    corrections = tmp_path / "corr"
    corrections.mkdir()
    _trex_npz(corrections / "vidA_fish0.npz", 99.0)
    _ = promote_correction(
        ds,
        "",
        "vidA",
        corrections / "vidA_fish0.npz",
        src_format="trex_npz",
        apply=True,
        force=True,
    )
    ds.convert_all_tracks(overwrite=True)

    variants = sorted(p.name for p in ds.get_root("tracks").iterdir() if p.is_dir())
    assert len(variants) == 2, variants
    # The uncorrected variant keeps the identifier it already had.
    assert uncorrected[0] in variants
    tables = {
        name: pd.read_parquet(ds.get_root("tracks") / name / "vidA.parquet")
        for name in variants
    }
    for name, table in tables.items():
        assert len(table) == 5, f"{name} holds both readings"
    assert {float(t["X"].iloc[0]) for t in tables.values()} == {1.0, 99.0}


def _upload_with_a_stale_correction(tmp_path: Path) -> Dataset:
    """A dataset whose upload and correction of one entry are converted, the
    correction's table then gone."""
    ds = make_dataset(tmp_path / "ds", name="promote")
    raw = ds.get_root("tracks_raw")
    _trex_npz(raw / "vidA_fish0.npz", 1.0)
    _ = ds.index_tracks_raw([raw], patterns="*.npz", src_format="trex_npz")
    ds.convert_all_tracks()
    (upload,) = sorted(p.name for p in ds.get_root("tracks").iterdir() if p.is_dir())
    corrections = tmp_path / "corr"
    corrections.mkdir()
    _trex_npz(corrections / "vidA_fish0.npz", 99.0)
    _ = promote_correction(
        ds,
        "",
        "vidA",
        corrections / "vidA_fish0.npz",
        src_format="trex_npz",
        apply=True,
        force=True,
    )
    ds.convert_all_tracks()
    (correction,) = (
        p for p in ds.get_root("tracks").iterdir() if p.is_dir() and p.name != upload
    )
    (correction / "vidA.parquet").unlink()
    return ds


def test_an_entry_written_under_one_variant_is_not_current_under_another(
    tmp_path: Path,
) -> None:
    """The upload's table is current and the correction's is written, for one entry."""
    ds = _upload_with_a_stale_correction(tmp_path)

    outcome = ds.convert_all_tracks()

    assert outcome.converted == 1
    assert outcome.current == 0
    assert outcome.converted - outcome.current == 1


def test_a_conversion_that_writes_a_correction_is_not_a_cache_hit(
    tmp_path: Path,
) -> None:
    ds = _upload_with_a_stale_correction(tmp_path)

    _ = run_op(ds, "convert-tracks", {"source_format": "trex_npz"})

    snapshot = latest_snapshot(ds)
    assert snapshot["entries_written"] == 1
    assert not snapshot["cache_hit"]


def test_only_the_newest_correction_converts(tmp_path: Path) -> None:
    """The series is append-only history, not one variant per revision."""
    import mosaic.core.track_library  # noqa: F401

    ds = make_dataset(tmp_path / "ds", name="promote")
    raw = ds.get_root("tracks_raw")
    _trex_npz(raw / "vidA_fish0.npz", 1.0)
    _ = ds.index_tracks_raw([raw], patterns="*.npz", src_format="trex_npz")
    corrections = tmp_path / "corr"
    corrections.mkdir()
    for revision, value in ((1, 50.0), (2, 99.0)):
        _trex_npz(corrections / f"r{revision}.npz", value)
        _ = promote_correction(
            ds,
            "",
            "vidA",
            corrections / f"r{revision}.npz",
            src_format="trex_npz",
            apply=True,
            force=True,
        )
    ds.convert_all_tracks(overwrite=True)

    tables = {
        p.name: pd.read_parquet(p / "vidA.parquet")
        for p in ds.get_root("tracks").iterdir()
        if p.is_dir() and (p / "vidA.parquet").exists()
    }
    values = {float(t["X"].iloc[0]) for t in tables.values()}
    assert values == {1.0, 99.0}, "rev1 should not have converted; rev2 should"


def _tracktor_file(directory: Path, value: float) -> Path:
    """A corrected Tracktor table whose positions follow *value*."""
    return write_tracktor_recording(
        directory, frames=6, flies=2, first_frame=int(value), write_csv=False
    ).tsv


def _sleap_file(directory: Path, value: float) -> Path:
    """A SLEAP analysis file whose every point sits at *value*."""
    path = directory / "myseq.h5"
    write_sleap_analysis_h5(path, np.full((5, 1, 2, 2), value), np.full((5, 1, 2), 0.9))
    return path


ONE_FILE_AT_A_TIME = {
    "tracktor_corrected_tsv": (_tracktor_file, "*.tsv"),
    "sleap_analysis_h5": (_sleap_file, "*.h5"),
}
"""Formats that convert each raw file into a table of its own, by source format."""


def _upload_dataset(tmp_path: Path, src_format: str) -> Dataset:
    """A dataset whose one upload of *src_format* is indexed and converted."""
    write, pattern = ONE_FILE_AT_A_TIME[src_format]
    ds = make_dataset(tmp_path / "ds", name="promote")
    ds.meta["fps_default"] = 30.0
    raw = ds.get_root("tracks_raw")
    _ = write(raw, 3.0)
    _ = ds.index_tracks_raw([raw], patterns=pattern, src_format=src_format)
    _ = ds.convert_all_tracks()
    return ds


def _variant_tables(ds: Dataset) -> dict[str, pd.DataFrame]:
    """Each tracks variant's one table, by variant."""
    root = ds.get_root("tracks")
    return {
        variant.name: pd.read_parquet(table)
        for variant in sorted(root.iterdir())
        if variant.is_dir()
        for table in variant.glob("*.parquet")
    }


def _converted_alone(tmp_path: Path, src_format: str, value: float) -> pd.DataFrame:
    """The table that a file of *value* converts to as an upload."""
    write, pattern = ONE_FILE_AT_A_TIME[src_format]
    ds = make_dataset(tmp_path / "alone", name="alone")
    ds.meta["fps_default"] = 30.0
    raw = ds.get_root("tracks_raw")
    _ = write(raw, value)
    _ = ds.index_tracks_raw([raw], patterns=pattern, src_format=src_format)
    _ = ds.convert_all_tracks()
    ((_, table),) = _variant_tables(ds).items()
    return table


def _promote_values(
    ds: Dataset, tmp_path: Path, src_format: str, values: tuple[float, ...]
) -> None:
    """Promote one correction of the dataset's entry for each of *values*."""
    write, _ = ONE_FILE_AT_A_TIME[src_format]
    ((group, sequence),) = {
        (row["group"], row["sequence"]) for row in ds.read_tracks_raw_index()
    }
    for number, value in enumerate(values):
        _ = promote_correction(
            ds,
            group,
            sequence,
            write(tmp_path / f"corr{number}", value),
            src_format=src_format,
            apply=True,
            force=True,
        )


@pytest.mark.parametrize("src_format", sorted(ONE_FILE_AT_A_TIME))
def test_a_correction_of_a_one_file_format_converts_as_its_own_variant(
    tmp_path: Path, src_format: str
) -> None:
    """A format that converts a file at a time separates a correction too.

    The correction and its upload name one entry. Written into one variant, one
    table would replace the other without a word.
    """
    ds = _upload_dataset(tmp_path, src_format)
    ((upload, before),) = _variant_tables(ds).items()

    _promote_values(ds, tmp_path, src_format, (10.0,))
    _ = ds.convert_all_tracks()

    tables = _variant_tables(ds)
    assert len(tables) == 2, sorted(tables)
    pd.testing.assert_frame_equal(tables[upload], before)
    (correction,) = (name for name in tables if name != upload)
    pd.testing.assert_frame_equal(
        tables[correction], _converted_alone(tmp_path, src_format, 10.0)
    )
    assert len(read_tracks_index(ds)) == 2


@pytest.mark.parametrize("src_format", sorted(ONE_FILE_AT_A_TIME))
def test_a_second_promotion_of_a_one_file_format_converts_the_newest_only(
    tmp_path: Path, src_format: str
) -> None:
    ds = _upload_dataset(tmp_path, src_format)
    ((upload, before),) = _variant_tables(ds).items()

    _promote_values(ds, tmp_path, src_format, (10.0, 20.0))
    _ = ds.convert_all_tracks()

    tables = _variant_tables(ds)
    assert len(tables) == 2, sorted(tables)
    pd.testing.assert_frame_equal(tables[upload], before)
    (correction,) = (name for name in tables if name != upload)
    pd.testing.assert_frame_equal(
        tables[correction], _converted_alone(tmp_path, src_format, 20.0)
    )


@pytest.mark.parametrize("src_format", sorted(ONE_FILE_AT_A_TIME))
def test_progress_counts_only_the_files_that_convert(
    tmp_path: Path, src_format: str
) -> None:
    """A superseded revision is no file of the run, so progress reaches its total."""
    ds = _upload_dataset(tmp_path, src_format)
    _promote_values(ds, tmp_path, src_format, (10.0, 20.0))
    reports: list[tuple[int, int]] = []

    _ = ds.convert_all_tracks(
        progress=lambda done, total: reports.append((done, total))
    )

    assert reports
    done, total = reports[-1]
    assert (done, total) == (2, 2)


def test_a_correction_revision_is_read_from_the_name() -> None:
    assert correction_revision(Path("corrected.rev1.npz")) == 1
    assert correction_revision(Path("corrected.rev12.vidA_fish0.npz")) == 12
    assert correction_revision(Path("vidA_fish0.npz")) == 0
    assert correction_revision(Path("corrected.notarev.npz")) == 0


def test_a_second_promotion_keeps_the_first_correction_s_lineage(
    tmp_path: Path,
) -> None:
    """Every superseded producer run stays named, not just the newest.

    ``derived_from`` is read by ``sweep_tracking`` as its *primary* eviction
    signal: a tracker's working directory whose output has been corrected has
    served its purpose and need not wait out its retention window. There is one
    label row per sequence, so writing only the newest un-superseded the first
    correction's tracker output and quietly put it back on age-based retention.
    """
    ds = make_dataset(tmp_path / "ds", name="promote")
    correction = _corrected(tmp_path)

    for producer in ("trex.1.0-aaaaaaaaaa", "trex.1.0-bbbbbbbbbb"):
        _ = promote_correction(
            ds,
            "",
            "vid1",
            correction,
            src_format="trex_npz",
            derived_from=producer,
            apply=True,
            force=True,
        )

    assert ds._promoted_from() == {"trex.1.0-aaaaaaaaaa", "trex.1.0-bbbbbbbbbb"}


# --- an entry key another entry names ----------------------------------------


def test_a_dry_run_reports_a_key_another_entry_names(tmp_path: Path) -> None:
    """The collision is a blocked reason, which ``force`` does not lift."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds, [MediaClip(group="R", sequence="0900", filename="R-0900.mp4")]
    )

    report = promote_correction(
        ds, "", "R__0900", _corrected(tmp_path), src_format="trex_npz", force=True
    )

    assert not report.applied
    assert len(report.blocked) == 1
    assert "'R__0900'" in report.blocked[0]


def test_an_applied_promotion_refuses_a_key_another_entry_names(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds, [MediaClip(group="R", sequence="0900", filename="R-0900.mp4")]
    )

    with pytest.raises(EntryKeyCollisionError, match="'R__0900'"):
        _ = promote_correction(
            ds,
            "",
            "R__0900",
            _corrected(tmp_path),
            src_format="trex_npz",
            apply=True,
            force=True,
        )

    assert not (ds.get_root("tracks_raw") / "R__0900").exists()


def test_a_promotion_is_not_blocked_by_derivatives_a_link_holds(
    tmp_path: Path,
) -> None:
    source = converted_source(tmp_path / "source")
    copy = copy_of(source, tmp_path / "copy")
    entry = CONVERTED_ENTRIES[0]
    assert not reached_by(copy, [entry], "tracks_raw").empty, "no linked row reached"

    report = promote_correction(
        copy, *entry, _corrected(tmp_path), src_format="trex_npz", apply=True
    )

    assert not report.blocked
    assert report.applied
    assert report.revision == 1
