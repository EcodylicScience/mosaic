"""``mosaic measure-tracks``: measure the frame axis of tables already published.

The only way to ask an already-published table whether its frame axis is its
media's. A run records the comparison as it publishes, but a table on disk
cannot be re-bridged without re-tracking, so a session tracked before anyone
was recording the second number can only be measured afterwards.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from tests.helpers import invoke_json

runner = CliRunner()


def test_measure_tracks_is_a_dry_run_by_default(dataset: tuple[Path, Dataset]) -> None:
    manifest, ds = dataset
    payload = invoke_json(["measure-tracks", "-m", str(manifest), "--json"])

    assert payload["applied"] is False
    assert payload["frame_extents_measured"] == 2
    # The fixture's rows name no producer, so no tool's read is rebuilt.
    assert (payload["media_frames_measured"], payload["frames_read_measured"]) == (0, 0)
    from mosaic.core.pipeline.tracks_index import read_frame_extents

    assert read_frame_extents(ds) == {}, "a dry run must not write"


def test_measure_tracks_counts_each_pass_s_rewrites_by_what_it_did(
    dataset: tuple[Path, Dataset],
) -> None:
    """Every count cell's pass reports what it cleared, kept and could not read.

    A converted table read no media, so its stale cells are cleared. A TREx row
    that names no record and no working directory cannot be established, so its
    cells are kept. The command registers the tracking ops, so nothing is
    unregistered.
    """
    from mosaic.core.pipeline.tracks_index import (
        read_tracks_index,
        tracks_index_path,
        write_tracks_row,
    )

    manifest, ds = dataset
    for sequence, producer, run_id in (
        ("s1", "convert-trex_npz", "convert-trex_npz.0.1-0123456789"),
        ("s2", "trex", "v1"),
    ):
        write_tracks_row(
            ds,
            run_id=run_id,
            group="g",
            sequence=sequence,
            out_path=ds.get_root("tracks") / f"g__{sequence}.parquet",
            producer=producer,
            std_format="trex_v2",
            n_rows=12,
            media_frames=20,
            frames_read=18,
            known_tail_loss=2,
        )
    # The fixture's hand-written rows have no run_id. Drop them, as
    # `_one_trex_row` does.
    frame = read_tracks_index(ds)
    frame[frame["run_id"] != ""].to_csv(tracks_index_path(ds), index=False)

    payload = invoke_json(["measure-tracks", "-m", str(manifest), "--json"])

    cells = ("media_frames", "frames_read", "known_tail_loss")
    outcomes = {"measured": 0, "cleared": 1, "not_established": 1, "unregistered": 0}
    reported = {f"{c}_{o}": payload[f"{c}_{o}"] for c in cells for o in outcomes}
    assert reported == {f"{c}_{o}": n for c in cells for o, n in outcomes.items()}


def test_measure_tracks_apply_writes_each_count_cell(
    dataset: tuple[Path, Dataset],
) -> None:
    """A converted table read no media, so each pass clears its stale cell.

    A dry run leaves the cells, and ``--apply`` clears them.
    """
    from mosaic.core.pipeline.tracks_index import (
        read_frames_read,
        read_known_tail_loss,
        read_media_frames,
        read_tracks_index,
        tracks_index_path,
        write_tracks_row,
    )

    manifest, ds = dataset
    write_tracks_row(
        ds,
        run_id="convert-trex_npz.0.1-0123456789",
        group="g",
        sequence="s1",
        out_path=ds.get_root("tracks") / "g__s1.parquet",
        producer="convert-trex_npz",
        std_format="trex_v2",
        n_rows=12,
        media_frames=20,
        frames_read=18,
        known_tail_loss=2,
    )
    frame = read_tracks_index(ds)
    frame[frame["run_id"] != ""].to_csv(tracks_index_path(ds), index=False)

    def cells() -> tuple[int | None, int | None, int | None]:
        (row,) = (row for _, row in read_tracks_index(ds).iterrows())
        return read_media_frames(row), read_frames_read(row), read_known_tail_loss(row)

    _ = invoke_json(["measure-tracks", "-m", str(manifest), "--json"])
    assert cells() == (20, 18, 2)

    _ = invoke_json(["measure-tracks", "-m", str(manifest), "--apply", "--json"])
    assert cells() == (None, None, None)


def test_measure_tracks_apply_records_the_extents(
    dataset: tuple[Path, Dataset],
) -> None:
    manifest, ds = dataset
    payload = invoke_json(["measure-tracks", "-m", str(manifest), "--apply", "--json"])

    assert payload["applied"] is True
    from mosaic.core.pipeline.tracks_index import read_frame_extents

    assert read_frame_extents(ds) == {("g", "s1"): (0, 11), ("g", "s2"): (0, 11)}


def _one_trex_row(ds: Dataset, *, read: int, media: int) -> None:
    """Replace the fixture's rows with one TREx row that read *read* of *media*."""
    from mosaic.core.pipeline.tracks_index import (
        read_tracks_index,
        tracks_index_path,
        write_tracks_row,
    )

    out = ds.get_root("tracks") / "g__s1.parquet"
    write_tracks_row(
        ds,
        run_id="v1",
        group="g",
        sequence="s1",
        out_path=out,
        producer="trex",
        std_format="trex_v2",
        n_rows=12,
        media_frames=media,
        frames_read=read,
    )
    # The fixture's hand-written rows carry no run_id, so drop them: an
    # unlabelled row and a labelled one for the same entry is a resolution
    # question this test is not about.
    frame = read_tracks_index(ds)
    frame[frame["run_id"] == "v1"].to_csv(tracks_index_path(ds), index=False)


def test_measure_tracks_names_a_frame_axis_that_is_not_its_media(
    dataset: tuple[Path, Dataset],
) -> None:
    """Both numbers, because the gap is the content of the report."""
    manifest, ds = dataset
    _one_trex_row(ds, read=16, media=20)

    payload = invoke_json(["measure-tracks", "-m", str(manifest), "--apply", "--json"])

    assert payload["frame_axis_mismatch"] == [
        {
            "run_id": "v1",
            "group": "g",
            "sequence": "s1",
            "frames_read": 16,
            "media_frames": 20,
        }
    ]
    assert payload["frame_tail_short"] == []


def test_measure_tracks_names_a_known_tail_loss_apart(
    dataset: tuple[Path, Dataset],
) -> None:
    """TREx's shortfall at the end of a file, with what a count cannot say.

    The row names no file its tool read, so the most TREx loses on any file is
    allowed, and the report says the file's own loss is unknown.
    """
    manifest, ds = dataset
    _one_trex_row(ds, read=18, media=20)

    payload = invoke_json(["measure-tracks", "-m", str(manifest), "--apply", "--json"])

    assert payload["frame_axis_mismatch"] == []
    assert payload["frame_tail_short"] == [
        {
            "run_id": "v1",
            "group": "g",
            "sequence": "s1",
            "frames_read": 18,
            "media_frames": 20,
            "allowed": 3,
            "allowance_known": False,
        }
    ]
    result = runner.invoke(app, ["measure-tracks", "-m", str(manifest), "--apply"])
    assert result.exit_code == 0, result.stderr
    assert "known tail loss g/s1 [v1]: trex read 18 of 20 frames" in result.stderr
    assert "the file it read is unknown or could not be probed" in result.stderr
    assert "cannot show that the missing frames are at the end" in result.stderr
