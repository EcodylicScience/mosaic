"""Promoting a manual correction -- item 8.6, with open item O1 resolved.

O1 asked whether a second correction of the same sequence is a conflict needing
a force every time, or the next revision of a source file. These pin the second
answer, and pin that the block is about *derivatives* rather than about history.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.promotion import (
    correction_revision,
    next_revision,
    promote_correction,
)
from mosaic.core.pipeline.sequence_index import (
    read_sequence_index,
    sequence_label_path,
    sequence_labels,
)
from mosaic.core.pipeline.tracks_raw_index import read_tracks_raw_index

from .conftest import write_trex_npz


def _dataset(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest(name="promote", base_dir=tmp_path / "ds")
    return Dataset(manifest_path=manifest).load(ensure_roots=True)


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
    ds = _dataset(tmp_path)
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
    ds = _dataset(tmp_path)

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

    Item 4.5's checksums are on by default and item 4.4's composition already
    covers this root, so a corrected file arriving is enough to invalidate every
    artifact built from it -- which is what makes promotion a *source* change
    rather than a new kind of thing.
    """
    ds = _dataset(tmp_path)
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
    ds = _dataset(tmp_path)

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
    ds = _dataset(tmp_path)

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
    ds = _dataset(tmp_path)
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
    ds = _dataset(tmp_path)

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
    ds = _dataset(tmp_path)
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
    ds = _dataset(tmp_path)
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
    ds = _dataset(tmp_path)

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
    ds = _dataset(tmp_path)
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


# --- a correction converts as its own variant ---------------------------------


def _trex_npz(path: Path, value: float) -> None:
    write_trex_npz(
        path,
        n=5,
        X=np.full(5, value),
        Y=np.full(5, value),
        **{"X#wcentroid": np.full(5, value), "Y#wcentroid": np.full(5, value)},
        poseX=np.stack([np.full(5, value)] * 2, axis=1),
        poseY=np.stack([np.full(5, value)] * 2, axis=1),
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

    ds = _dataset(tmp_path)
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


def test_only_the_newest_correction_converts(tmp_path: Path) -> None:
    """The series is append-only history, not one variant per revision."""
    import mosaic.core.track_library  # noqa: F401

    ds = _dataset(tmp_path)
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
    ds = _dataset(tmp_path)
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
