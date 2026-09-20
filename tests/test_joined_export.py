"""Joining an entry's clips into the one video an external tool opens.

Two properties carry the design and are asserted hardest here: the join holds
every frame of every clip in ``video_order`` -- so joined frame *i* is global
frame *i* -- and it is addressed by the clip set it holds, so adding, removing
or reordering a clip addresses a different file rather than silently reusing
the old one.

The op exists because handing a tool several files was wrong in both
directions. SLEAP, Lightning Pose and Ultralytics were truncated to clip 0 and
tracked none of the rest of a recording; TREx took the whole list and its
``FFmpegVideoCapture`` under-counts every file it opens, so it lost the tail of
each clip and the published table's ``frame`` column stopped addressing the
video -- measured as a staircase stepping two frames at every boundary.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from mosaic_media.transcode import TranscodeError

from mosaic.core.dataset import Dataset
from mosaic.core.media.video_io import open_frame_reader
from mosaic.core.pipeline.joined_export import (
    JoinedExportParams,
    joined_export_path,
    joined_recipe_hash,
    joined_source_uid,
    write_joined_export,
)
from mosaic.core.pipeline.ops import run_op
from mosaic.core.scope import Scope
from mosaic.tracking.common.scope import build_work_items
from mosaic.tracking.common.tool_input import (
    JoinedExportMissingError,
    resolve_tool_inputs,
)
from tests.helpers import add_media_sequence, make_dataset

pytestmark = pytest.mark.media

CLIP_FRAMES = 6


@pytest.fixture
def ds(tmp_path: Path, requires_ffmpeg: None) -> Dataset:
    """A dataset holding one two-clip sequence of real, differing videos."""
    dataset = make_dataset(tmp_path, roots=["media_raw", "media", "tracks"])
    add_media_sequence(dataset, "sess", videos=("a.mp4", "b.mp4"), frames=CLIP_FRAMES)
    return dataset


def _join(ds: Dataset, *, overwrite: bool = False) -> Path:
    _ = run_op(
        ds,
        "export-joined",
        {},
        scope=Scope(entries=[("", "sess")]),
        overwrite=overwrite,
    )
    resolved = ds.resolve_media("", "sess")
    return joined_export_path(
        ds,
        joined_source_uid(list(resolved.facts)),
        joined_recipe_hash(JoinedExportParams()),
    )


def test_the_join_holds_every_frame_of_every_clip(ds: Dataset) -> None:
    """The property the whole op exists for: joined frame i is global frame i."""
    joined = _join(ds)
    assert joined.is_file()

    reader = open_frame_reader(joined, target="raw")
    try:
        frames = [frame for _, frame in reader]
    finally:
        reader.close()

    assert len(frames) == 2 * CLIP_FRAMES
    # add_media_sequence shades each clip by its filename, so the two halves are
    # distinguishable -- which is what makes this an ordering assertion and not
    # merely a count.
    means = [float(np.mean(f)) for f in frames]
    first, second = means[0], means[CLIP_FRAMES]
    assert abs(first - second) > 1.0, "the clips must be distinguishable"
    assert all(m == pytest.approx(first, abs=0.5) for m in means[:CLIP_FRAMES])
    assert all(m == pytest.approx(second, abs=0.5) for m in means[CLIP_FRAMES:])


def test_the_name_carries_the_clip_set(ds: Dataset) -> None:
    """Reordering the clips addresses a different file, never reuses this one."""
    joined = _join(ds)
    resolved = ds.resolve_media("", "sess")
    forwards = joined_source_uid(list(resolved.facts))
    backwards = joined_source_uid(list(reversed(list(resolved.facts))))

    assert forwards != backwards
    assert joined.name.startswith(forwards)
    other = joined_export_path(ds, backwards, joined_recipe_hash(JoinedExportParams()))
    assert not other.exists()


def test_a_second_run_reuses_the_file(ds: Dataset) -> None:
    joined = _join(ds)
    stamp = joined.stat().st_mtime_ns
    assert _join(ds) == joined
    assert joined.stat().st_mtime_ns == stamp, "reused, not rewritten"


def test_overwrite_rewrites_it(ds: Dataset) -> None:
    """How a file left by a build no longer trusted is replaced."""
    joined = _join(ds)
    joined.write_bytes(b"not a video")
    _ = _join(ds, overwrite=True)
    assert joined.stat().st_size > len(b"not a video")


def test_a_single_clip_entry_is_a_no_op(tmp_path: Path, requires_ffmpeg: None) -> None:
    """A correct answer to "join it", not a failure.

    A pipeline running this over every entry of a dataset must not fail on the
    single-clip ones, which are the overwhelming majority.
    """
    dataset = make_dataset(tmp_path, roots=["media_raw", "media", "tracks"])
    add_media_sequence(dataset, "one", videos=("a.mp4",), frames=CLIP_FRAMES)

    _ = run_op(dataset, "export-joined", {}, scope=Scope(entries=[("", "one")]))

    root = dataset.get_root("media") / "joined"
    assert not root.exists() or not list(root.glob("*.mp4"))


def test_clips_that_cannot_be_copied_are_refused_naming_the_remedy(
    ds: Dataset, tmp_path: Path
) -> None:
    """A copy cannot reconcile two streams, and re-encoding here would hide that.

    The remedy is real and already exists, so the refusal names it rather than
    starting a long CPU-bound pass of its own.
    """
    resolved = ds.resolve_media("", "sess")
    paths = list(resolved.paths)
    facts = list(resolved.facts)
    wider = dataclasses.replace(facts[1], width=facts[1].width * 2)

    with pytest.raises(TranscodeError, match="transcode"):
        _ = write_joined_export(paths, [facts[0], wider], tmp_path / "j.mp4")


def test_a_join_that_does_not_line_up_is_refused_and_not_published(
    ds: Dataset, tmp_path: Path
) -> None:
    """The check that is the whole value of the op, because TREx does not do it.

    ffmpeg's concat demuxer is frame-exact for matching streams and silently is
    not for mismatched ones, so the result is counted rather than trusted. Here
    the clips' recorded counts are made to disagree with what the files hold,
    which is the same inequality a bad concatenation produces.
    """
    resolved = ds.resolve_media("", "sess")
    paths = list(resolved.paths)
    facts = [
        dataclasses.replace(clip, frame_count=clip.frame_count + 5)
        for clip in resolved.facts
    ]
    dest = tmp_path / "j.mp4"

    with pytest.raises(TranscodeError, match="would not line up"):
        _ = write_joined_export(paths, facts, dest)

    assert not dest.exists(), "a short join must not reach the recipe address"
    assert not list(tmp_path.glob("*.partial*")), "nor leave its partial behind"
    assert not list(tmp_path.glob("*.concat.txt")), "nor its listing"


def _item(ds: Dataset):
    (item,) = build_work_items(ds, ds.resolve_media_scope(None), kind="trex")
    return item


def test_a_tracker_is_handed_the_join_and_not_the_clips(ds: Dataset) -> None:
    joined = _join(ds)
    assert resolve_tool_inputs(ds, _item(ds), kind="trex") == (joined,)


def test_a_tracker_is_refused_when_the_join_is_missing(ds: Dataset) -> None:
    """Refused, not silently truncated and not silently joined by the tool."""
    with pytest.raises(JoinedExportMissingError, match="export-joined"):
        _ = resolve_tool_inputs(ds, _item(ds), kind="trex")
