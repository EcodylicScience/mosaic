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
import subprocess
from pathlib import Path

import numpy as np
import pytest
from mosaic_media import MediaFacts, probe_media
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


def test_clips_that_disagree_on_geometry_are_refused_outright(
    ds: Dataset, tmp_path: Path
) -> None:
    """The one mismatch no flag unlocks, because normalising it would lie.

    Making the clips agree would mean rescaling or rotating one, and every
    coordinate a tracker reported for those frames would then be in a different
    space from the rest of the session.
    """
    resolved = ds.resolve_media("", "sess")
    paths = list(resolved.paths)
    facts = list(resolved.facts)
    wider = dataclasses.replace(facts[1], width=facts[1].width * 2)

    with pytest.raises(TranscodeError, match="frame geometry"):
        _ = write_joined_export(paths, [facts[0], wider], tmp_path / "j.mp4")

    with pytest.raises(TranscodeError, match="frame geometry"):
        _ = write_joined_export(
            paths, [facts[0], wider], tmp_path / "j.mp4", reencode=True
        )


def _as_av1(source: Path, dest: Path) -> "MediaFacts":
    """The same frames as an AV1 derivative, so a clip set is genuinely mixed.

    This direction and not the other, because it is the one that occurs: the
    clips are H.264 originals as a camera wrote them, and `transcode` gives one
    defective sibling an AV1 analysis derivative. That is the ESI corpus, and it
    is also what made a join need re-encoding at all.
    """
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libsvtav1",
            "-crf",
            "30",
            "-preset",
            "8",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "passthrough",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )
    return probe_media(dest)


def test_a_mixed_clip_set_is_refused_by_default_naming_both_remedies(
    ds: Dataset, tmp_path: Path
) -> None:
    """A copy cannot reconcile two streams, and re-encoding is opted into.

    This is the ESI corpus: `transcode` gives a defective clip an analysis
    derivative and leaves its clean siblings alone, so the entry resolves to a
    mix no routing change can make uniform.
    """
    resolved = ds.resolve_media("", "sess")
    paths = list(resolved.paths)
    facts = list(resolved.facts)
    other = tmp_path / "b_av1.mp4"
    mixed_facts = [facts[0], _as_av1(paths[1], other)]
    mixed_paths = [paths[0], other]

    with pytest.raises(TranscodeError) as excinfo:
        _ = write_joined_export(mixed_paths, mixed_facts, tmp_path / "j.mp4")
    assert "transcode" in str(excinfo.value)
    assert "reencode=true" in str(excinfo.value)


def test_reencode_normalises_the_odd_clip_and_joins_every_frame(
    ds: Dataset, tmp_path: Path
) -> None:
    """The opt-in path, and it must still hold every frame in order."""
    resolved = ds.resolve_media("", "sess")
    paths = list(resolved.paths)
    facts = list(resolved.facts)
    other = tmp_path / "b_av1.mp4"
    mixed_facts = [facts[0], _as_av1(paths[1], other)]
    dest = tmp_path / "j.mp4"

    written = write_joined_export([paths[0], other], mixed_facts, dest, reencode=True)

    assert written == 2 * CLIP_FRAMES
    assert int(probe_media(dest).frame_count) == 2 * CLIP_FRAMES
    assert not list(tmp_path.glob("*.normalised*")), "the temp clip is swept"


def test_the_majority_profile_is_normalised_to_not_from(tmp_path: Path) -> None:
    """One derivative among sixteen originals must cost one re-encode, not sixteen."""
    from mosaic.core.pipeline.joined_export import _outliers
    from tests.test_media_timeline import _facts

    av1 = dataclasses.replace(_facts(), codec_name="av1", pixel_format="yuv420p")
    h264 = dataclasses.replace(_facts(), codec_name="h264", pixel_format="yuv420p")
    majority, odd = _outliers([h264] * 16 + [av1])

    assert majority == ("h264", "yuv420p")
    assert odd == [16], "the single AV1 clip is the one re-encoded"


def test_a_codec_mosaic_declares_no_encoder_for_is_refused_by_name(
    ds: Dataset, tmp_path: Path
) -> None:
    resolved = ds.resolve_media("", "sess")
    paths = list(resolved.paths)
    facts = list(resolved.facts)
    exotic = dataclasses.replace(facts[0], codec_name="theora")
    # Two clips claiming theora outvote the real one, so theora is the target.
    with pytest.raises(TranscodeError, match="no encoder"):
        _ = write_joined_export(
            [paths[0], paths[1], paths[0]],
            [exotic, exotic, facts[1]],
            tmp_path / "j.mp4",
            reencode=True,
        )


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


# --- a consumer asks for the clips, not for one recipe ---------------------
#
# The first version re-derived the filename from DEFAULT parameters, so a join
# built with any non-default setting was invisible: `export-joined` with
# reencode=true wrote one file and the tracker looked for another, then reported
# the join missing on a session that had just been joined successfully. It cost
# a canary run on a 30-clip session to find.


def test_a_join_built_with_non_default_params_is_still_found(ds: Dataset) -> None:
    """The regression. Any complete join of these clips answers the question."""
    _ = run_op(
        ds,
        "export-joined",
        {"reencode": True},
        scope=Scope(entries=[("", "sess")]),
    )
    resolved = ds.resolve_media("", "sess")
    uid = joined_source_uid(list(resolved.facts))
    built = (
        ds.get_root("media")
        / "joined"
        / f"{uid}.{joined_recipe_hash(JoinedExportParams(reencode=True))}.joined.mp4"
    )
    assert built.is_file(), "the op wrote the reencode-recipe file"
    assert not joined_export_path(
        ds, uid, joined_recipe_hash(JoinedExportParams())
    ).exists(), "and deliberately not the default-recipe one"

    assert resolve_tool_inputs(ds, _item(ds), kind="trex") == (built,)


def test_two_joins_of_one_clip_set_are_refused_not_chosen_between(
    ds: Dataset,
) -> None:
    """Different inputs, so picking by sort order would be picking by accident.

    The same refusal `select_variant_rows` makes for two recipes of one entry,
    and for the same reason: what a run read must not depend on which filename
    happens to sort first.
    """
    _ = run_op(ds, "export-joined", {}, scope=Scope(entries=[("", "sess")]))
    _ = run_op(
        ds,
        "export-joined",
        {"reencode": True},
        scope=Scope(entries=[("", "sess")]),
    )

    with pytest.raises(JoinedExportMissingError, match="different recipes"):
        _ = resolve_tool_inputs(ds, _item(ds), kind="trex")


def test_the_op_reports_that_it_joined_something(ds: Dataset) -> None:
    """ "finished" alone could not tell a joined session from a skipped one.

    Which is how the addressing bug above hid behind a clean exit: the run said
    finished, entries_written 0, and 0 means "not reported".
    """
    from mosaic.runlog import reduce_run_log, run_log_dir

    _ = run_op(ds, "export-joined", {}, scope=Scope(entries=[("", "sess")]))

    logs = sorted(run_log_dir(ds.base_dir).glob("*.jsonl"))
    snapshot = reduce_run_log(max(logs, key=lambda p: p.stat().st_mtime))
    assert snapshot is not None
    assert snapshot["entries_written"] == 1


def test_a_join_is_counted_in_frames_not_in_distinct_timestamps(
    ds: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression that refused a complete 390,986-frame session.

    ``probe_media(...).frame_count`` is the number of *distinct presentation
    timestamps*, so two frames sharing one contribute a single count. Joining
    clips recorded at different rates does exactly that at the boundaries --
    ffmpeg re-times each segment by the previous one's duration and the rounding
    collides -- while losing nothing. Counting that value made the op refuse a
    video that held every frame and stop the session being tracked.

    Simulated by making the probe under-report, because a file whose timestamps
    genuinely collide cannot be built reliably across ffmpeg versions. The test
    still bites: restore the old measurement and this refuses again.
    """
    import mosaic.core.pipeline.joined_export as module

    real = module.probe_media

    def under_reporting(path: Path) -> MediaFacts:
        facts = real(path)
        return dataclasses.replace(facts, frame_count=facts.frame_count - 2)

    resolved = ds.resolve_media("", "sess")
    paths, facts = list(resolved.paths), list(resolved.facts)
    expected = sum(int(clip.frame_count) for clip in facts)
    monkeypatch.setattr(module, "probe_media", under_reporting)
    dest = tmp_path / "j.mp4"

    written = write_joined_export(paths, facts, dest)

    assert written == expected, "every frame is present, so every frame counts"
    assert dest.exists(), "a complete join must reach the recipe address"


def test_a_shortfall_names_the_clip_that_is_wrong(ds: Dataset, tmp_path: Path) -> None:
    """A refusal that cannot say which side is short sends the reader nowhere.

    The old message offered "the clips may disagree on frame rate or carry stale
    frame counts" for every failure alike. The clips are on disk and can simply
    be measured, so the message says whether they hold what the index claims --
    and when they do, that the copy itself is at fault and where the evidence is.
    """
    resolved = ds.resolve_media("", "sess")
    paths = list(resolved.paths)
    facts = list(resolved.facts)
    facts[1] = dataclasses.replace(facts[1], frame_count=facts[1].frame_count + 5)
    dest = tmp_path / "j.mp4"

    with pytest.raises(TranscodeError) as caught:
        _ = write_joined_export(paths, facts, dest)

    message = str(caught.value)
    assert paths[1].name in message, "the message names the clip that disagrees"
    assert "reprobe-media" in message, "and the command that re-measures it"
    assert paths[0].name not in message, "and does not accuse the sound clips"


def _clip_at_rate(dest: Path, *, fps: int, timescale: int, frames: int) -> MediaFacts:
    """A real clip recorded at *fps* and counting time in *timescale* ticks.

    Both halves matter and they are independent. A session's clips genuinely
    differ in rate -- 30, 29.948 and 31 fps is a measured example -- and each
    rate brings its own tick count, which is the quantity the concat demuxer
    does not reconcile.
    """
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=64x48:rate={fps}:duration={frames / fps}",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-video_track_timescale",
            str(timescale),
            str(dest),
        ],
        check=True,
    )
    return probe_media(dest)


def test_a_mixed_rate_join_gets_a_uniform_timeline(
    tmp_path: Path, requires_ffmpeg: None
) -> None:
    """The defect that made TREx crawl for thirty hours and read wrong pixels.

    Clips recorded at different rates count time in different ticks. The concat
    demuxer offsets each segment by the previous one's duration in the *first*
    clip's ticks and does not rescale, so the second clip lands at the wrong
    timestamp and every frame after it is addressed wrongly. A tool that seeks
    by timestamp -- TREx does -- then lands on the wrong frame.

    Measured on this fixture before the restamp: neighbouring timestamps stepped
    4.92 frame periods apart. Nothing was lost, which is exactly why the frame
    count could not catch it.
    """
    a = _clip_at_rate(tmp_path / "a.mp4", fps=30, timescale=15360, frames=120)
    b = _clip_at_rate(tmp_path / "b.mp4", fps=31, timescale=15872, frames=124)
    dest = tmp_path / "j.mp4"

    written = write_joined_export(
        [tmp_path / "a.mp4", tmp_path / "b.mp4"], [a, b], dest
    )

    measured = probe_media(dest)
    assert written == 244, "every frame of both clips survives the copy"
    assert measured.max_timestamp_gap_frame_periods <= 1.5, (
        "frame i has to be findable at i periods by a tool that seeks"
    )
    assert measured.constant_frame_rate, "the imposed grid is uniform"


def test_a_normalised_clip_is_written_in_its_neighbours_ticks(
    tmp_path: Path, requires_ffmpeg: None
) -> None:
    """An encoder picks its own tick rate, and the copy will not rescale it.

    libx264 chose 1/953497 for the clip this was found on, which the concat
    demuxer then wrote into a 1/15360 track verbatim -- timestamps 62x too
    large. The re-encode has to count in the ticks its neighbours count in.
    """
    from mosaic.core.pipeline.joined_export import _normalise_clip
    from mosaic.core.pipeline.stream_copy import stream_timescale

    source = tmp_path / "odd.mp4"
    clip = _clip_at_rate(source, fps=30, timescale=953497, frames=30)
    assert stream_timescale(source) == 953497, "the fixture carries the odd rate"
    dest = tmp_path / "normalised.mp4"

    _normalise_clip(source, clip, ("h264", "yuv420p"), dest, timescale=15360)

    assert stream_timescale(dest) == 15360


def test_a_join_whose_timeline_is_not_uniform_is_refused_and_kept(
    ds: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refused before publishing, and the evidence is kept.

    Simulated through the probe, because the restamp makes a genuinely
    non-uniform join unbuildable -- which is the point. What is asserted is that
    the measurement is acted on: it was already taken, already carried on
    ``MediaFacts``, and read by nothing.
    """
    import mosaic.core.pipeline.joined_export as module

    real = module.probe_media

    def wide_gap(path: Path) -> MediaFacts:
        return dataclasses.replace(
            real(path), max_timestamp_gap_frame_periods=106546.65
        )

    resolved = ds.resolve_media("", "sess")
    monkeypatch.setattr(module, "probe_media", wide_gap)
    dest = tmp_path / "j.mp4"

    with pytest.raises(TranscodeError, match="seeks by timestamp"):
        _ = write_joined_export(list(resolved.paths), list(resolved.facts), dest)

    assert not dest.exists(), "a join that cannot be seeked must not be published"
    assert (tmp_path / "j.partial.mp4").is_file(), "the evidence is kept"


def test_the_restamped_recipe_addresses_a_different_file(ds: Dataset) -> None:
    """The fix changes the bytes, so it must change the address.

    A join written by the old recipe is not this one, and reusing it by name
    would hand a tracker the broken timeline the restamp exists to remove.
    """
    from mosaic.core.pipeline._utils import hash_params
    from mosaic.core.pipeline.joined_export import JoinedExportOp

    params = JoinedExportParams()
    previous = hash_params(
        {"op_version": "0.1", "params": params.identity_dump()},
    )

    assert JoinedExportOp.version == "0.2"
    assert joined_recipe_hash(params) != previous
