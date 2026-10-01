"""TREx's known tail loss is exact per file: the rule of the file it read.

TREx counts a file's frames from the container's count less its decoder's reorder
delay, and never drains the decoder at the end. So it stops short of the end of a
file by that file's reorder depth, and by one more when the container carries no
frame count. Measured over 26 files: 2 for H.264 or HEVC with B-frames, 1 for one
B-frame, 0 for AV1 and without B-frames, and 3 or 1 in Matroska. A shortfall is a
known tail loss only within what the file TREx read gives. A conversion of several
files loses frames at each boundary, so it is allowed none.

The file is the one TREx records as its source in the ``.pv`` header. The run tests
write real clips, hand them to the fake TREx, and let it record them there.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NoReturn

import pandas as pd
import pytest

import mosaic.tracking.trex.dataset_runs as trex_runs
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS, TailLoss
from mosaic.core.pipeline.tracks_axis import tail_loss_of_files
from mosaic.core.pipeline.tracks_index import (
    TailAllowance,
    backfill_known_tail_loss,
    read_known_tail_loss,
    read_tracks_index,
    tracks_index_path,
)
from mosaic.tracking.trex.params import TrexParams
from tests.helpers import (
    MediaClip,
    clip_facts,
    latest_snapshot,
    make_dataset,
    stub_join,
    write_media_index,
)
from tests.helpers.media import write_h264_mp4
from tests.helpers.trex import install_fake_trex, write_pv_header

_FRAMES = 30
"""How many frames each clip holds."""


def _trex_loss() -> TailLoss:
    loss = TRACKING_ROOTS["trex"].tail_loss
    assert loss is not None
    return loss


# --- the rule ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("depth", "declared", "loss"),
    [(2, 600, 2), (1, 300, 1), (0, 600, 0), (2, 0, 3), (0, 0, 1)],
    ids=["b-frames", "one-b-frame", "none", "matroska-b-frames", "matroska"],
)
def test_trex_loses_its_reorder_depth_and_one_without_a_frame_count(
    depth: int, declared: int, loss: int
) -> None:
    header = dataclasses.replace(
        clip_facts(), coded_reordering_depth=depth, declared_frame_count=declared
    )

    assert _trex_loss().of_file(header) == loss


def test_several_files_are_allowed_no_tail_loss() -> None:
    """TREx loses frames at each boundary, and those move every frame after them."""
    assert tail_loss_of_files("trex", ["/data/a.mp4", "/data/b.mp4"]) == 0


def test_no_file_or_an_unreadable_one_gives_no_known_loss(tmp_path: Path) -> None:
    stub = tmp_path / "stub.mp4"
    _ = stub.write_bytes(b"fake")

    assert tail_loss_of_files("trex", []) is None
    assert tail_loss_of_files("trex", [str(stub)]) is None
    assert tail_loss_of_files("trex", [str(tmp_path / "absent.mp4")]) is None


def test_a_producer_declaring_no_tail_loss_has_no_known_loss() -> None:
    assert tail_loss_of_files("ultralytics", ["/data/a.mp4"]) is None


# --- a run over a real clip ------------------------------------------------------


type WriteClip = Callable[[Path], None]


def _h264(path: Path) -> None:
    write_h264_mp4(path, frames=_FRAMES)


def _av1(write_cfr_mp4: Callable[..., None]) -> WriteClip:
    def write(path: Path) -> None:
        write_cfr_mp4(path, frames=_FRAMES)

    return write


def _tracked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    filename: str,
    write: WriteClip,
    read: int,
) -> Dataset:
    """Run the fake TREx over one real clip, recording that it read *read* frames.

    The entry's media composition is projected as a scan projects it, so the row
    records what its run read.
    """
    ds = make_dataset(tmp_path / "ds")
    write(ds.get_root(ds.resolve_media_root()) / filename)
    _index_clip(ds, filename, video_uuid="uid-vid1")
    trex = install_fake_trex(monkeypatch)
    trex.npz_frames, trex.pv_frames = 20, read
    _ = trex_runs.run_trex(ds, TrexParams())
    return ds


def _index_clip(ds: Dataset, filename: str, *, video_uuid: str) -> None:
    """Index the entry's one clip, and project its composition as a scan does."""
    write_media_index(
        ds,
        [
            MediaClip(
                sequence="vid1",
                filename=filename,
                video_uuid=video_uuid,
                frame_count=_FRAMES,
            )
        ],
    )
    _ = ds.rebuild_sequence_index("media_raw")


def _logged(ds: Dataset) -> tuple[int, int]:
    """The run's counts of tail losses and of mismatches, as its bridge judged them.

    The bridge judges by the loss it records, so they agree with the index's.
    """
    snapshot = latest_snapshot(ds)
    return (
        snapshot["entries_frame_tail_short"],
        snapshot["entries_frame_axis_mismatch"],
    )


def _recorded_loss(ds: Dataset) -> int | None:
    rows = read_tracks_index(ds)
    assert len(rows) == 1
    return read_known_tail_loss(rows.iloc[0])


@pytest.mark.media
def test_an_av1_run_two_short_is_a_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_cfr_mp4: Callable[..., None],
) -> None:
    """AV1 reorders nothing, so TREx reads it to the end, and two short is real."""
    ds = _tracked(
        tmp_path,
        monkeypatch,
        filename="vid1.mp4",
        write=_av1(write_cfr_mp4),
        read=_FRAMES - 2,
    )

    assert _recorded_loss(ds) == 0
    assert ds.frame_tail_shortfalls() == ()
    (found,) = ds.frame_axis_mismatches()
    assert (found.read, found.media) == (_FRAMES - 2, _FRAMES)
    assert _logged(ds) == (0, 1)


@pytest.mark.media
@pytest.mark.parametrize(
    ("filename", "short"),
    [("vid1.mp4", 2), ("vid1.mkv", 3)],
    ids=["h264-b-frames", "matroska"],
)
def test_a_run_short_by_its_file_s_loss_is_a_tail_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    requires_ffmpeg: None,
    filename: str,
    short: int,
) -> None:
    """H.264 with B-frames holds back two, and Matroska records no frame count.

    The run's line names the loss as this file's own.
    """
    ds = _tracked(
        tmp_path, monkeypatch, filename=filename, write=_h264, read=_FRAMES - short
    )

    assert _recorded_loss(ds) == short
    assert ds.frame_axis_mismatches() == ()
    (found,) = ds.frame_tail_shortfalls()
    assert (found.read, found.media) == (_FRAMES - short, _FRAMES)
    assert found.allowance == TailAllowance(frames=short, known=True)
    assert _logged(ds) == (1, 0)
    line = (
        f"the tool read {_FRAMES - short} of {_FRAMES} frames, within the {short} "
        "that trex leaves unread at the end of this file."
    )
    assert line in capsys.readouterr().err


@pytest.mark.media
def test_an_h264_run_short_by_more_than_its_file_s_loss_is_a_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requires_ffmpeg: None
) -> None:
    """Three short is within the most TREx loses, and past what this mp4 gives."""
    ds = _tracked(
        tmp_path, monkeypatch, filename="vid1.mp4", write=_h264, read=_FRAMES - 3
    )

    assert ds.frame_tail_shortfalls() == ()
    (found,) = ds.frame_axis_mismatches()
    assert (found.read, found.media) == (_FRAMES - 3, _FRAMES)
    assert _logged(ds) == (0, 1)


# --- the backfill ------------------------------------------------------------------


def _set_cell(ds: Dataset, column: str, value: str) -> None:
    path = tracks_index_path(ds)
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows.assign(**{column: value}).to_csv(path, index=False)


def _the_pv(ds: Dataset) -> Path:
    (pv,) = ds.get_root("trex-convert").rglob("*.pv")
    return pv


@pytest.mark.media
def test_the_backfill_gives_a_row_the_loss_the_bridge_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requires_ffmpeg: None
) -> None:
    ds = _tracked(
        tmp_path, monkeypatch, filename="vid1.mp4", write=_h264, read=_FRAMES - 2
    )
    _set_cell(ds, "known_tail_loss", "")

    done = backfill_known_tail_loss(ds)

    assert len(done.written) == 1
    assert _recorded_loss(ds) == 2


@pytest.mark.media
def test_a_legacy_row_of_several_files_two_short_is_a_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requires_ffmpeg: None
) -> None:
    """A conversion of a clip list, as TREx made before mosaic joined clips.

    Each clip here would be allowed two, but a list loses frames at every
    boundary, so its row is allowed none once its ``.pv`` is read.
    """
    ds = _tracked(
        tmp_path, monkeypatch, filename="vid1.mp4", write=_h264, read=_FRAMES - 2
    )
    media_root = ds.get_root(ds.resolve_media_root())
    second = media_root / "vid1-b.mp4"
    _h264(second)
    write_pv_header(_the_pv(ds), _FRAMES - 2, sources=[media_root / "vid1.mp4", second])
    _set_cell(ds, "known_tail_loss", "")
    assert ds.frame_axis_mismatches() == (), "an unread file is allowed the most"

    done = backfill_known_tail_loss(ds)

    assert len(done.written) == 1
    assert _recorded_loss(ds) == 0
    assert ds.frame_tail_shortfalls() == ()
    (found,) = ds.frame_axis_mismatches()
    assert (found.read, found.media) == (_FRAMES - 2, _FRAMES)


@pytest.mark.media
def test_a_row_whose_clip_was_replaced_since_its_run_keeps_its_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requires_ffmpeg: None,
    write_cfr_mp4: Callable[..., None],
) -> None:
    """The file at the source path now is not the file TREx read.

    TREx read an H.264 clip two short, which its B-frames explain. That clip was
    then replaced in place by an AV1 clip, which loses nothing, and re-indexed.
    The header there now says nothing about the run, so the row keeps its loss.
    """
    ds = _tracked(
        tmp_path, monkeypatch, filename="vid1.mp4", write=_h264, read=_FRAMES - 2
    )
    clip = ds.get_root(ds.resolve_media_root()) / "vid1.mp4"
    clip.unlink()
    _av1(write_cfr_mp4)(clip)
    _index_clip(ds, "vid1.mp4", video_uuid="uid-replaced")

    done = backfill_known_tail_loss(ds)

    assert _recorded_loss(ds) == 2
    assert (len(done.written), len(done.cleared)) == (0, 0)
    assert done.not_established["known_tail_loss"].tolist() == ["2"]
    assert ds.frame_axis_mismatches() == ()
    (found,) = ds.frame_tail_shortfalls()
    assert (found.read, found.media) == (_FRAMES - 2, _FRAMES)


def _tracked_clips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: str,
    clips: Mapping[str, int],
) -> Dataset:
    """Run the fake TREx at integration *version* over entries of *clips* clips.

    *clips* maps each entry's sequence to its number of clips. Every conversion
    records two frames short of the longest entry. Then every conversion's
    ``.pv`` is swept. The bridge could not read the stub clips' headers, so each
    row's known loss is blank.
    """
    ds = make_dataset(tmp_path / "ds")
    names = {
        sequence: [f"{sequence}-c{order}.mp4" for order in range(count)]
        for sequence, count in clips.items()
    }
    write_media_index(
        ds,
        [
            MediaClip(
                sequence=sequence,
                filename=name,
                video_order=order,
                video_uuid=f"uid-{name}",
                frame_count=_FRAMES,
            )
            for sequence, named in names.items()
            for order, name in enumerate(named)
        ],
    )
    for named in names.values():
        if len(named) > 1:
            _ = stub_join(ds, [f"uid-{name}" for name in named])
    monkeypatch.setattr(trex_runs, "TREX_VERSION", version)
    trex = install_fake_trex(monkeypatch)
    trex.npz_frames, trex.pv_frames = 20, max(clips.values()) * _FRAMES - 2
    _ = trex_runs.run_trex(ds, TrexParams())
    for pv in ds.get_root("trex-convert").rglob("*.pv"):
        pv.unlink()
    return ds


@pytest.mark.parametrize(
    ("version", "clips", "loss"),
    [("0.1", 2, 0), ("0.2", 2, None), ("0.1", 1, None)],
    ids=["clip-list", "joined", "one-clip"],
)
def test_a_legacy_row_of_several_clips_whose_pv_is_gone_is_allowed_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    clips: int,
    loss: int | None,
) -> None:
    """The run index proves a clip list when the swept ``.pv`` cannot.

    Integration version 0.1 handed TREx a multi-clip entry's clips as a list, and
    its index row counts and lists them. From 0.2 TREx reads one joined file,
    whose loss only that file's header can tell, and one clip is no list.
    """
    ds = _tracked_clips(tmp_path, monkeypatch, version=version, clips={"sess": clips})
    media = clips * _FRAMES

    done = backfill_known_tail_loss(ds)

    assert _recorded_loss(ds) == loss
    if loss == 0:
        assert len(done.written) == 1
        assert ds.frame_tail_shortfalls() == ()
        (mismatch,) = ds.frame_axis_mismatches()
        assert (mismatch.read, mismatch.media) == (media - 2, media)
    else:
        assert len(done.written) == 0
        assert ds.frame_axis_mismatches() == ()
        (shortfall,) = ds.frame_tail_shortfalls()
        assert (shortfall.read, shortfall.media) == (media - 2, media)
        assert shortfall.allowance.known is False


def _older_index(rows: pd.DataFrame) -> pd.DataFrame:
    """The index as TREx runs wrote it before they recorded their clips."""
    return rows.drop(columns=["n_source_videos", "video_sources"])


def _counts_three(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.assign(n_source_videos="3")


def _counts_one(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.assign(n_source_videos="1")


@pytest.mark.parametrize(
    "edit",
    [_older_index, _counts_three, _counts_one],
    ids=["no-clip-columns", "counts-more-than-listed", "counts-fewer-than-listed"],
)
def test_a_run_index_row_that_does_not_prove_a_clip_list_leaves_the_loss_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    edit: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    """A row without its clip cells, or whose count and list disagree, proves nothing.

    An index that no run has appended to since those cells existed has no such
    columns, and a read does not add them.
    """
    ds = _tracked_clips(tmp_path, monkeypatch, version="0.1", clips={"sess": 2})
    path = trex_runs.trex_index_path(ds)
    edit(pd.read_csv(path, dtype=str, keep_default_na=False)).to_csv(path, index=False)

    done = backfill_known_tail_loss(ds)

    assert len(done.written) == 0
    assert _recorded_loss(ds) is None


def test_a_run_after_the_clip_list_version_answers_without_reading_the_run_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backfill asks once for each row, and a later run cannot have read a list.

    Its working directory names its run, so no row of the run index is read.
    """
    ds = _tracked_clips(tmp_path, monkeypatch, version="0.2", clips={"sess": 2})

    def unread(path: Path) -> NoReturn:
        raise AssertionError(f"read {path}")

    monkeypatch.setattr(trex_runs, "trex_index", unread)

    done = backfill_known_tail_loss(ds)

    assert len(done.written) == 0
    assert _recorded_loss(ds) is None


def test_only_the_table_s_own_working_directory_proves_a_clip_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One run over an entry of two clips and an entry of one.

    Both entries' rows are the run's, and only the row naming a table's own
    working directory says what TREx read for it.
    """
    ds = _tracked_clips(
        tmp_path, monkeypatch, version="0.1", clips={"sess": 2, "solo": 1}
    )

    done = backfill_known_tail_loss(ds)

    assert done.written["sequence"].tolist() == ["sess"]
    losses = {
        str(row["sequence"]): read_known_tail_loss(row)
        for _, row in read_tracks_index(ds).iterrows()
    }
    assert losses == {"sess": 0, "solo": None}
