"""What a TREx conversion read, measured against its media and reported.

TREx's ``FFmpegVideoCapture`` under-counts every file it opens and then reads
only as many frames as it counted, so a session's clips converted into a ``.pv``
that had dropped the tail of each one. Measured on a six-clip fixture carrying
its own frame numbers: 1,800 media frames converted to 1,788, the offset constant
inside each clip and stepping by two at every boundary. The published table is
then numbered on the tracker's axis while every consumer that reads pixels is on
the media's -- right at the start of a sequence and progressively wrong through
it.

mosaic cannot correct that (it holds no map from one axis to the other, and TREx
records none), so what these pin is that it is *measured and reported*, and that
reporting it costs the table nothing.
"""

from __future__ import annotations

import pandas as pd
import pytest

import mosaic.tracking.trex.dataset_runs as dr
from mosaic.core.dataset import Dataset
from mosaic.tracking.trex.params import TrexParams
from tests.helpers import FakeTrex, index_session, latest_snapshot, scope_over


def _tracks_row(ds: Dataset) -> "pd.Series[object]":
    from mosaic.core.pipeline.tracks_index import read_tracks_index

    frame = read_tracks_index(ds)
    assert len(frame) == 1
    return frame.iloc[0]


def test_a_joined_session_records_the_length_of_its_media(
    ds: Dataset, trex: FakeTrex
) -> None:
    """The comparison needs both numbers, and this is where the second is taken."""
    from mosaic.core.pipeline.tracks_index import read_media_frames

    trex.npz_frames = 600
    index_session(ds, "c0.mp4", "c1.mp4", frame_count=300)

    _ = dr.run_trex(ds, TrexParams(), scope_over(("", "sess")))

    assert read_media_frames(_tracks_row(ds)) == 600
    assert latest_snapshot(ds)["entries_frame_axis_mismatch"] == 0


def test_a_short_joined_conversion_records_both_numbers(
    ds: Dataset, trex: FakeTrex
) -> None:
    """The defect itself: 596 frames published against 600 frames of media."""
    from mosaic.core.pipeline.tracks_index import read_frame_extent, read_media_frames

    trex.npz_frames = 596
    index_session(ds, "c0.mp4", "c1.mp4", frame_count=300)

    _ = dr.run_trex(ds, TrexParams(), scope_over(("", "sess")))

    row = _tracks_row(ds)
    assert read_media_frames(row) == 600
    assert read_frame_extent(row) == (0, 595)
    (found,) = ds.frame_axis_mismatches()
    assert (found.group, found.sequence) == ("", "sess")
    assert (found.read, found.media) == (596, 600)


def test_a_short_joined_conversion_reports_itself_on_the_run_log(
    ds: Dataset, trex: FakeTrex
) -> None:
    """The record that survives a queue sending the child's stderr to DEVNULL."""
    trex.npz_frames = 596
    index_session(ds, "c0.mp4", "c1.mp4", frame_count=300)

    _ = dr.run_trex(ds, TrexParams(), scope_over(("", "sess")))

    assert latest_snapshot(ds)["entries_frame_axis_mismatch"] == 1


def test_a_short_joined_conversion_still_publishes_a_usable_table(
    ds: Dataset, trex: FakeTrex
) -> None:
    """Recorded, never refused -- and this is the assertion that holds that line.

    Raising instead would be permanent. The condition is deterministic, so every
    re-run fails the same entry, and a published table cannot be re-bridged
    without re-tracking. Everything computed inside the table is unaffected by
    the axis being short, so throwing it away would cost the analyses that never
    depended on registration in order to flag the one thing that does.
    """
    trex.npz_frames = 596
    index_session(ds, "c0.mp4", "c1.mp4", frame_count=300)

    _ = dr.run_trex(ds, TrexParams(), scope_over(("", "sess")))

    row = _tracks_row(ds)
    table = ds.resolve_path(str(row["abs_path"]))
    assert pd.read_parquet(table).shape[0] == 596
    snapshot = latest_snapshot(ds)
    assert snapshot["entries_failed"] == 0
    assert snapshot["entries_written"] == 1
    assert snapshot["status"] == "finished"


@pytest.mark.parametrize(
    "window",
    [
        {"analysis_range": (0, 100)},
        {"track_extra_settings": {"analysis_range": [0, 100]}},
        {"convert_extra_settings": {"video_conversion_range": [0, 100]}},
    ],
    ids=["field", "track-setting", "convert-setting"],
)
def test_an_analysis_range_run_asks_no_question(
    ds: Dataset, trex: FakeTrex, window: dict[str, object]
) -> None:
    """A run told to cover part of the video is not a run that lost the rest.

    The range may be the field or a frame setting passed through to TREx.
    """
    from mosaic.core.pipeline.tracks_index import read_media_frames

    trex.npz_frames = 100
    index_session(ds, "c0.mp4", "c1.mp4", frame_count=300)

    _ = dr.run_trex(ds, TrexParams.model_validate(window), scope_over(("", "sess")))

    assert read_media_frames(_tracks_row(ds)) is None
    assert ds.frame_axis_mismatches() == ()
    assert latest_snapshot(ds)["entries_frame_axis_mismatch"] == 0


def test_a_table_that_ends_early_is_not_a_short_conversion(
    ds: Dataset, trex: FakeTrex
) -> None:
    """Nobody is tracked in the last twenty frames, and the ``.pv`` holds them all.

    ``frame_max`` is the last frame carrying a row, not the last frame the tracker
    saw: TREx exports each individual from its first tracked frame to its last.
    Compared with the media, it would report an animal that leaves before the end
    as a broken frame axis on every ordinary run. What TREx read is the ``.pv``'s
    frame count, and that is what the media is compared with.
    """
    from mosaic.core.pipeline.tracks_index import read_frame_extent, read_media_frames

    trex.npz_frames, trex.pv_frames = 280, 300
    index_session(ds, "c0.mp4", frame_count=300)

    _ = dr.run_trex(ds, TrexParams(), scope_over(("", "sess")))

    row = _tracks_row(ds)
    assert read_frame_extent(row) == (0, 279)
    assert read_media_frames(row) == 300
    assert ds.frame_axis_mismatches() == ()
    assert latest_snapshot(ds)["entries_frame_axis_mismatch"] == 0
