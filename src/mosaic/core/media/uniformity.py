"""Whether a reader would accept a sequence's clips -- item 6.5's precheck.

``MultiVideoReader`` refuses a sequence whose clips disagree on displayed
dimensions or frame rate, and the frame-rate arm compares every clip against
*the first one*: the drift allowance is ``|other - reference| / reference``
scaled by the shorter clip's frame count. So which clip sits at position 0 can
decide whether a marginal sequence is readable, and a reorder can flip that with
no artifact deleted and no warning anywhere.

This answers the question **before** the arrangement is committed, from the
media index, with no probe.

**Only the frame-rate arm can flip.** The dimension arm is exact equality
against the reference, which is a total agreement test: if any clip disagrees
then some clip disagrees with the first, in every ordering. Saying so keeps the
precheck from reporting a sequence a reorder cannot affect.

**Displayed, never coded.** The reader autorotates, so a quarter-turn clip is
emitted with its width and height swapped, and the index's flat ``width`` /
``height`` cells are the *coded* numbers by design. The comparison therefore goes
through :func:`~mosaic.core.media.video_io.facts_to_video_metadata`, which is
where that swap lives for every other caller -- an upright clip and a
quarter-turned one of equal coded size are uniform on the coded numbers and
transposed on the displayed ones, so a coded comparison would pass a sequence the
reader then refuses.

**Per camera.** ``video_order`` is a dense counter per ``(group, sequence,
camera)`` and the media resolver emits one entry per camera precisely so parallel
cameras are never concatenated into one timeline. A per-sequence answer would be
about a reader nobody constructs.

Dataset-agnostic, like ``drift`` and ``prune`` beside it: it takes rows and
returns a verdict, so the comparison is testable without a dataset on disk and
the root resolution has one home in :meth:`Dataset.sequence_uniformity`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mosaic_media import MeasuredVideoProperties, PropertyMismatch, uniform_properties

from mosaic.core.media.facts_columns import row_facts_or_none
from mosaic.core.media.video_io import facts_to_video_metadata
from mosaic.core.pipeline.media_index import VideoOrderKey, assign_video_order

__all__ = [
    "UniformityVerdict",
    "camera_uniformity",
]


@dataclass(frozen=True, slots=True)
class UniformityVerdict:
    """What a camera's clips say about whether a reader would open them.

    ``mismatch`` is the first disagreement :func:`uniform_properties` finds, or
    ``None`` when the clips agree. ``unmeasured`` names the clips whose stored
    facts could not be rebuilt, and it is load-bearing rather than diagnostic: a
    ``None`` mismatch over rows that were skipped is *unknown*, not agreement,
    and a caller must not read the two alike.
    """

    mismatch: PropertyMismatch | None
    unmeasured: tuple[str, ...]

    @property
    def established(self) -> bool:
        """Whether every clip contributed a measurement."""
        return not self.unmeasured

    @property
    def readable(self) -> bool:
        """Whether a reader would accept this arrangement, as far as can be told.

        ``True`` on an unestablished verdict is a statement about evidence, not
        about the file -- check :attr:`established` when that distinction matters.
        """
        return self.mismatch is None


def camera_uniformity(
    rows: Sequence[Mapping[str, object]],
    *,
    order_by_name: Mapping[str, int] | None = None,
) -> UniformityVerdict:
    """Whether one camera's *rows*, in the proposed order, would open as a sequence.

    *rows* are media-index rows for a single ``(group, sequence, camera)``.
    *order_by_name* is the arrangement to test, mapping a clip's basename to its
    position, exactly as :class:`~mosaic.core.pipeline.media_index.MediaIndexScope`
    carries it; ``None`` tests the order the index already holds.

    The arrangement is produced by :func:`assign_video_order`, the same function
    the media writer uses, so the order checked here is the order a write would
    commit rather than a second implementation of the ranking that could disagree
    with it.
    """
    arranged = _arranged(rows, order_by_name)
    measured: list[MeasuredVideoProperties] = []
    unmeasured: list[str] = []
    for row in arranged:
        name = Path(str(row.get("abs_path", ""))).name or str(row.get("name", ""))
        facts = row_facts_or_none(row)
        if facts is None:
            unmeasured.append(name)
            continue
        # Through the shared swap rather than facts.width/height: the reader
        # emits display-oriented frames and this must compare what it emits.
        displayed = facts_to_video_metadata(Path(name), facts)
        measured.append(
            MeasuredVideoProperties(
                fps=displayed.fps,
                width=displayed.width,
                height=displayed.height,
                frame_count=displayed.frame_count,
                duration=facts.duration,
            )
        )
    return UniformityVerdict(
        mismatch=uniform_properties(measured),
        unmeasured=tuple(unmeasured),
    )


def _arranged(
    rows: Sequence[Mapping[str, object]],
    order_by_name: Mapping[str, int] | None,
) -> list[Mapping[str, object]]:
    """*rows* in the order a write would commit them."""
    positions = order_by_name or {}

    def key_of(row: Mapping[str, object]) -> VideoOrderKey:
        name = Path(str(row.get("abs_path", ""))).name
        return VideoOrderKey(
            group=str(row.get("group", "")),
            sequence=str(row.get("sequence", "")),
            camera=str(row.get("camera", "") or ""),
            name=str(row.get("name", "")),
            prior_order=_order(row),
            session_position=positions.get(name),
        )

    return [row for row, _position in assign_video_order(list(rows), key_of)]


def _order(row: Mapping[str, object]) -> int | None:
    """A row's committed ``video_order``, or ``None`` when it carries none."""
    raw = str(row.get("video_order", "") or "").strip()
    if not raw or raw.lower() == "nan":
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None
