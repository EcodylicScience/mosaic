"""What a run of clips looks like as one continuous timeline.

A recorder that chops one observation into clips leaves a sequence whose frame
index is global and whose *time* is not: each clip carries its own frame rate,
and a session's clips routinely disagree -- 30 fps for the first, 29.95 for the
second, 31 for the rest is a real, measured example. Anything that divides one
global frame index by one frame rate is then wrong for every clip but the first,
and wrong cumulatively.

**There are no per-frame timestamps to fall back on.** :class:`MediaFacts`
carries aggregate timing only, and its ``start_time`` is a container property
rather than a wall clock. An imgstore records a ``frame_time`` per frame and is
read that way; a plain video file has no equivalent, and neither does the ``.pv``
a tracker converts one into. So the honest per-frame time is *reconstructed* from
what was measured per clip, which is what this module does.

**Segments are placed by ``frame_count / fps``, never by the measured
``duration``.** Within a segment the slope is ``1 / fps``, so the last frame of
segment *i* sits at ``start_time_i + (n_i - 1) / fps_i``. Offsetting segment
*i+1* by anything else -- and a container ``duration`` absorbs the final frame's
display period, edit lists and audio padding -- puts a jump or an overlap at the
boundary. A discontinuity in a monotone time column is worse than a small
absolute error, and the model's unit is frames. The measured duration is recorded
on the segment anyway, unused, so the discrepancy stays visible instead of being
silently resolved.

**This is a "segments played back-to-back" model, and it cannot be anything
else.** mosaic has no way to detect a real recording gap between two clips: the
facts carry no creation timestamp, so a recorder that stopped for five minutes
between clips produces a timeline that reads as continuous. A session with a gap
is timed as if there were none. Where that matters, the gap has to come from
outside the probe.

Pure: no pandas, no ``Dataset``, no I/O. It turns a sequence of already-probed
facts into a model and answers questions about it, which is what lets it be
tested without a dataset on disk and reused by any caller holding facts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from mosaic_media import MediaFacts

__all__ = [
    "ConcatenatedTimeline",
    "FrameIndices",
    "TimelineSegment",
    "concatenated_timeline",
]

type FrameIndices = npt.NDArray[np.int64] | Sequence[int]
"""Global frame indices to place: a track table's ``frame`` column, or a list."""


@dataclass(frozen=True, slots=True)
class TimelineSegment:
    """One clip's place in the concatenation.

    ``start_frame`` and ``start_time`` are where this clip begins in the global
    frame index and the reconstructed time axis. ``measured_duration`` is what
    the probe reported for the file and is deliberately **not** what placed the
    segment -- it is carried so a caller can see the two disagree.
    """

    index: int
    start_frame: int
    frame_count: int
    fps: float
    start_time: float
    measured_duration: float

    @property
    def end_frame(self) -> int:
        """One past this segment's last global frame."""
        return self.start_frame + self.frame_count

    @property
    def duration(self) -> float:
        """The length this segment was placed by: ``frame_count / fps``."""
        return self.frame_count / self.fps


@dataclass(frozen=True, slots=True)
class ConcatenatedTimeline:
    """A sequence's clips as one frame index and one time axis.

    ``uniform_rate`` records whether one frame rate indexes every clip. It is a
    *classification*, not a gate: a heterogeneous session is a legitimate thing
    to track, and this is how a caller knows that a per-second quantity computed
    against a single rate cannot be trusted.
    """

    segments: tuple[TimelineSegment, ...]
    uniform_rate: bool

    @property
    def total_frames(self) -> int:
        """Every clip's frames, summed -- the length a tracker sees."""
        return sum(segment.frame_count for segment in self.segments)

    @property
    def total_duration(self) -> float:
        """Where the last frame's period ends on the reconstructed axis."""
        last = self.segments[-1]
        return last.start_time + last.duration

    def segment_for_frame(self, frame: int) -> TimelineSegment:
        """Which clip a global frame index falls in.

        A frame past the end resolves to the **last** segment rather than
        raising: a container frame count that disagrees with what a tool actually
        decoded is mundane, and refusing the whole table over the last frame or
        two would lose a run that is otherwise entirely usable.
        """
        starts = [segment.start_frame for segment in self.segments]
        position = int(np.searchsorted(starts, frame, side="right")) - 1
        return self.segments[max(position, 0)]

    def times(self, frames: FrameIndices) -> npt.NDArray[np.float64]:
        """The reconstructed time of each global frame index, in seconds.

        Vectorized because it is applied to a whole track table. Frames past the
        last segment extrapolate at that segment's rate, for the reason
        :meth:`segment_for_frame` gives.
        """
        index = self._segment_index(frames)
        starts = np.array([s.start_frame for s in self.segments], dtype=np.int64)
        offsets = np.array([s.start_time for s in self.segments], dtype=np.float64)
        rates = np.array([s.fps for s in self.segments], dtype=np.float64)
        asked = np.asarray(frames, dtype=np.int64)
        return offsets[index] + (asked - starts[index]) / rates[index]

    def rates(self, frames: FrameIndices) -> npt.NDArray[np.float64]:
        """The frame rate in force at each global frame index."""
        index = self._segment_index(frames)
        rates = np.array([s.fps for s in self.segments], dtype=np.float64)
        return rates[index]

    def _segment_index(self, frames: FrameIndices) -> npt.NDArray[np.int64]:
        """Which segment each frame falls in, clamped into range at both ends."""
        starts = np.array([s.start_frame for s in self.segments], dtype=np.int64)
        asked = np.asarray(frames, dtype=np.int64)
        position = np.searchsorted(starts, asked, side="right") - 1
        return np.clip(position, 0, len(self.segments) - 1).astype(np.int64)


def concatenated_timeline(facts: Sequence[MediaFacts]) -> ConcatenatedTimeline:
    """Build the timeline *facts* describe, in the order they are given.

    The order is the caller's and is never sorted here: it is ``video_order``, it
    is semantic, and re-sorting it would silently re-time the sequence.

    Args:
        facts: One clip's probed facts per element, already in playback order.

    Returns:
        The concatenation, with ``uniform_rate`` classifying the frame rates.

    Raises:
        ValueError: If *facts* is empty, or a clip reports a non-positive frame
            rate. An unknown rate makes the segment unplaceable, and substituting
            a default would put a wrong slope on one segment of an otherwise
            measured timeline -- a plausible number recorded nowhere.
    """
    if not facts:
        raise ValueError("a timeline needs at least one clip")

    segments: list[TimelineSegment] = []
    start_frame = 0
    start_time = 0.0
    for index, clip in enumerate(facts):
        if clip.fps <= 0:
            raise ValueError(
                f"clip {index} reports a frame rate of {clip.fps}, so its frames "
                "cannot be placed on a time axis"
            )
        segments.append(
            TimelineSegment(
                index=index,
                start_frame=start_frame,
                frame_count=int(clip.frame_count),
                fps=float(clip.fps),
                start_time=start_time,
                measured_duration=float(clip.duration),
            )
        )
        start_frame += int(clip.frame_count)
        start_time += int(clip.frame_count) / float(clip.fps)

    # Local: `uniformity` reaches `core.pipeline.media_index` for the ordering
    # ranker, and this module is imported from `core.media.__init__`. A
    # module-level import would pull the pipeline package into the media
    # package's own import, which is the cycle this deferral exists to avoid.
    from mosaic.core.media.uniformity import rate_uniform

    return ConcatenatedTimeline(
        segments=tuple(segments), uniform_rate=rate_uniform(facts)
    )
