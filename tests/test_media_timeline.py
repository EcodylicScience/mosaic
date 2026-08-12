"""A run of clips as one timeline: where each frame lands, and at what rate.

The case that motivates every test here is real: one recording session whose
clips measure 30 fps, 29.95 fps and 31 fps. Dividing the global frame index by
one rate is wrong for most of it, and wrong cumulatively, so the timeline has to
place each clip at its own rate and still hand back a monotone axis.
"""

from __future__ import annotations

import numpy as np
import pytest
from mosaic_media import MediaFacts

from mosaic.core.media.timeline import concatenated_timeline
from mosaic.core.media.uniformity import geometry_mismatch, rate_uniform


def _facts(
    *,
    fps: float = 30.0,
    frame_count: int = 300,
    width: int = 64,
    height: int = 48,
    rotation: int = 0,
    duration: float | None = None,
    start_time: float = 0.0,
) -> MediaFacts:
    """One clip's probed facts, carrying just what the timeline reads."""
    return MediaFacts(
        container="mp4",
        codec_name="h264",
        pixel_format="yuv420p",
        color_range="",
        color_primaries="",
        color_transfer="",
        width=width,
        height=height,
        rotation_degrees=rotation,
        square_pixels=True,
        progressive=True,
        has_audio=False,
        video_stream_count=1,
        duration=(frame_count / fps if fps else 0.0) if duration is None else duration,
        fps=fps,
        frame_count=frame_count,
        start_time=start_time,
        constant_frame_rate=True,
        max_instantaneous_fps=None,
        declared_duration=frame_count / fps if fps else 0.0,
        declared_fps=fps,
        declared_frame_count=frame_count,
        moov_at_start=True,
        max_keyframe_interval_frames=1,
        max_gop_bytes=1,
        discard_flagged_packets=0,
        leading_non_keyframe_frames=0,
        coded_reordering_depth=0,
        max_timestamp_gap_frame_periods=1.0,
        timing_source="presentation",
        video_uuid="uuid",
        content_digest="digest",
        identity_scheme="1",
        prober_version="test",
    )


# The measured shape of session 20250922, shortened: 30, then 29.95, then 31.
SESSION = [
    _facts(fps=30.0, frame_count=300),
    _facts(fps=29.95, frame_count=300),
    _facts(fps=31.0, frame_count=300),
]


class TestPlacement:
    def test_a_single_clip_is_frame_over_fps(self) -> None:
        timeline = concatenated_timeline([_facts(fps=25.0, frame_count=10)])
        frames = np.arange(10)
        assert timeline.times(frames) == pytest.approx(frames / 25.0)

    def test_every_clip_is_placed_at_its_own_rate(self) -> None:
        timeline = concatenated_timeline(SESSION)
        assert timeline.total_frames == 900
        # Each segment starts where the previous one's frames ran out.
        assert [s.start_frame for s in timeline.segments] == [0, 300, 600]
        assert timeline.segments[1].start_time == pytest.approx(300 / 30.0)
        assert timeline.segments[2].start_time == pytest.approx(
            300 / 30.0 + 300 / 29.95
        )
        assert timeline.total_duration == pytest.approx(
            300 / 30.0 + 300 / 29.95 + 300 / 31.0
        )

    def test_a_boundary_costs_exactly_one_frame_period(self) -> None:
        """No jump and no overlap where one clip hands over to the next."""
        timeline = concatenated_timeline(SESSION)
        for segment in timeline.segments[1:]:
            previous = timeline.segments[segment.index - 1]
            before = timeline.times([segment.start_frame - 1])[0]
            after = timeline.times([segment.start_frame])[0]
            assert after - before == pytest.approx(1 / previous.fps)

    def test_frame_count_over_fps_places_a_clip_not_its_measured_duration(self) -> None:
        """A container duration absorbs padding; the axis must not.

        The first clip reports a duration a whole second longer than its frames
        account for. Placing the second clip by that duration would put a
        one-second hole at the boundary.
        """
        clips = [
            _facts(fps=30.0, frame_count=300, duration=11.0),
            _facts(fps=30.0, frame_count=300),
        ]
        timeline = concatenated_timeline(clips)
        assert timeline.segments[0].measured_duration == pytest.approx(11.0)
        assert timeline.segments[1].start_time == pytest.approx(10.0)
        before = timeline.times([299])[0]
        after = timeline.times([300])[0]
        assert after - before == pytest.approx(1 / 30.0)

    def test_the_axis_is_monotone_across_the_whole_session(self) -> None:
        timeline = concatenated_timeline(SESSION)
        times = timeline.times(np.arange(timeline.total_frames))
        assert np.all(np.diff(times) > 0)

    def test_start_time_moves_nothing(self) -> None:
        """It is a container presentation offset, not a wall clock."""
        shifted = [_facts(start_time=4.5), _facts(start_time=4.5)]
        timeline = concatenated_timeline(shifted)
        assert timeline.times([0])[0] == pytest.approx(0.0)
        assert timeline.segments[1].start_time == pytest.approx(10.0)


class TestRates:
    def test_the_rate_in_force_follows_the_segment(self) -> None:
        timeline = concatenated_timeline(SESSION)
        rates = timeline.rates([0, 299, 300, 599, 600, 899])
        assert list(rates) == [30.0, 30.0, 29.95, 29.95, 31.0, 31.0]

    def test_a_session_of_one_rate_is_uniform(self) -> None:
        timeline = concatenated_timeline([_facts(), _facts()])
        assert timeline.uniform_rate is True

    def test_the_measured_session_is_not_uniform(self) -> None:
        assert concatenated_timeline(SESSION).uniform_rate is False

    def test_a_same_rig_pair_stays_uniform(self) -> None:
        """The tolerance exists so measurement noise is not a disagreement."""
        pair = [_facts(fps=30.0), _facts(fps=30.000000040)]
        assert concatenated_timeline(pair).uniform_rate is True


class TestEdges:
    def test_a_frame_past_the_end_extrapolates_rather_than_raising(self) -> None:
        """A container frame count off by one against a decoder is mundane."""
        timeline = concatenated_timeline(SESSION)
        beyond = timeline.times([timeline.total_frames + 2])[0]
        assert beyond > timeline.times([timeline.total_frames - 1])[0]
        assert np.isfinite(beyond)

    def test_no_clips_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one clip"):
            _ = concatenated_timeline([])

    def test_an_unknown_rate_raises_rather_than_being_defaulted(self) -> None:
        with pytest.raises(ValueError, match="frame rate"):
            _ = concatenated_timeline([_facts(), _facts(fps=0.0)])

    def test_segment_for_frame_finds_the_holder(self) -> None:
        timeline = concatenated_timeline(SESSION)
        assert timeline.segment_for_frame(0).index == 0
        assert timeline.segment_for_frame(299).index == 0
        assert timeline.segment_for_frame(300).index == 1
        assert timeline.segment_for_frame(10_000).index == 2


class TestTheTwoHalvesAreAskedSeparately:
    def test_a_rate_disagreement_is_not_a_geometry_one(self) -> None:
        """The measured session must pass the refusal it would be gated on."""
        assert geometry_mismatch(SESSION) is None
        assert rate_uniform(SESSION) is False

    def test_a_rotation_difference_is_caught_at_equal_coded_size(self) -> None:
        """Coded-only would wave this through; the reader then transposes it."""
        mismatch = geometry_mismatch([_facts(), _facts(rotation=90)])
        assert mismatch is not None
        assert mismatch.field == "rotation_degrees"
        assert mismatch.index == 1

    def test_a_width_difference_names_the_clip_that_disagrees(self) -> None:
        mismatch = geometry_mismatch([_facts(), _facts(), _facts(width=1280)])
        assert mismatch is not None
        assert (mismatch.field, mismatch.index) == ("width", 2)
        assert (mismatch.first, mismatch.other) == (64, 1280)

    def test_one_clip_agrees_with_itself(self) -> None:
        assert geometry_mismatch([_facts()]) is None
        assert rate_uniform([_facts()]) is True
