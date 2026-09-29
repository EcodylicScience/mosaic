"""Map a media variant's pixels and frames into its entry's source media.

A media variant is a video re-encoded from an entry's media by pre-processing
steps: a crop moves its pixels, and a trim or a decimation selects its frames. A
tracker reads the variant, while its tracks are published in source space, the
pixel grid and frame axis of the entry media. :class:`Placement` is the map back.
It records the source rectangle that the variant image covers, and the source
frames of its frames as a :class:`FrameMap`. Each step derives the placement after
it from the placement before it. A chain of steps, or a variant read by another
variant, still maps back through one placement.

The module does not perform I/O, and numpy is its heaviest import.
"""

from __future__ import annotations

import json
import math
import operator
from dataclasses import dataclass
from typing import Final, Self

import numpy as np
import numpy.typing as npt

from mosaic.core.json_value import JsonValue

__all__ = ["FrameMap", "Placement"]


@dataclass(frozen=True, slots=True)
class FrameMap:
    """Map file frame ``i`` to source frame ``start + step * i``.

    The file has ``count`` frames, ``i`` in ``range(count)``. A trim narrows the
    map with :meth:`within` and a decimation thins it with :meth:`every`. Neither
    leaves the grid that it started on. The map stays three integers however many
    steps built it.

    Attributes:
        start: The source frame of file frame 0.
        step: The source frames between consecutive file frames.
        count: The number of frames in the file.
    """

    start: int
    step: int
    count: int

    def __post_init__(self) -> None:
        """Refuse an impossible grid.

        Raises:
            TypeError: If a field is not an integer. A numpy integer is one.
            ValueError: If a field is out of range.
        """
        _require_integers(start=self.start, step=self.step, count=self.count)
        if self.start < 0:
            raise ValueError(
                f"a frame map starts at source frame {self.start}, before the "
                f"first source frame, 0"
            )
        if self.step < 1:
            raise ValueError(
                f"a frame map steps by {self.step}. The step is at least 1"
            )
        if self.count < 0:
            raise ValueError(
                f"a frame map has {self.count} frames. The count is at least 0"
            )

    @property
    def end(self) -> int:
        """The exclusive end of the source span covered: ``start + step * count``."""
        return self.start + self.step * self.count

    def source_frames(self, frames: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """Return the source frame of each file frame in *frames*."""
        return self.start + self.step * frames

    def within(self, start: int, stop: int) -> FrameMap:
        """Return this map's frames that lie in the source range ``[start, stop)``.

        A ``trim`` applies this. A *start* between two kept frames begins at the
        next kept frame. A trim after a decimation stays on the decimated grid.

        The range may extend to :attr:`end`, and the span to :attr:`end`
        includes the last kept frame's full step. After a decimation it runs up to
        ``step - 1`` source frames past the last kept frame, which may lie past the
        source's last frame.

        Raises:
            ValueError: If *start* is not before *stop*, if the range extends
                outside ``[self.start, self.end)``, or if it contains none of this
                map's frames.
        """
        requested = f"[{start}, {stop})"
        if start >= stop:
            raise ValueError(
                f"source frames {requested} are an empty range. The start must be "
                f"before the stop. The frame map spans {_span(self)}."
            )
        if start < self.start or stop > self.end:
            raise ValueError(
                f"source frames {requested} extend outside the frame map, which "
                f"spans {_span(self)}."
            )
        first = _ceil_div(start - self.start, self.step)
        stop_index = _ceil_div(stop - self.start, self.step)
        if stop_index <= first:
            raise ValueError(
                f"source frames {requested} do not contain a frame of the frame map, "
                f"which spans {_span(self)} in steps of {self.step}."
            )
        return FrameMap(self.start + self.step * first, self.step, stop_index - first)

    def within_segment(self, start: int, stop: int) -> tuple[int, int] | None:
        """Return the first and last of this map's frames in ``[start, stop)``.

        A reader of one clip of a multi-clip entry calls this. The clip has source
        frames ``[start, stop)``, and the reader reads this map's frames in it
        from the first to the last, :attr:`step` frames apart. Unlike with
        :meth:`within`, the range may contain none of this map's frames or extend
        past its ends.

        Returns:
            ``(first, last)`` as source frames, or ``None`` when the range contains
            no frame of this map.
        """
        if self.count == 0:
            return None
        low = max(self.start, start)
        high = min(self.end - self.step, stop - 1)
        first = self.start + self.step * _ceil_div(low - self.start, self.step)
        if first > high:
            return None
        last = self.start + self.step * ((high - self.start) // self.step)
        return first, last

    def every(self, n: int) -> FrameMap:
        """Return every *n*-th frame of this map, starting with its first.

        A ``decimate`` applies this. A remainder keeps its first frame. Ten frames
        thinned by three keep four.

        Raises:
            ValueError: If *n* is below 2, which is not a decimation.
        """
        if n < 2:
            raise ValueError(
                f"a decimation factor of {n} is not a decimation. The factor is at "
                f"least 2"
            )
        return FrameMap(self.start, self.step * n, _ceil_div(self.count, n))

    def file_indices(self, upstream: FrameMap) -> tuple[int, int, int]:
        """Locate this map's frames in a file written under *upstream*.

        A variant built on another variant reads the upstream file, whose frame
        ``j`` is source frame ``upstream.start + upstream.step * j``. This map's
        frame ``i`` is then the upstream file's frame ``first + stride * i``.

        Returns:
            ``(first, stride, count)``: the upstream file frame with this map's
            first frame, the upstream file frames between consecutive frames of
            this map, and how many frames this map reads.

        Raises:
            ValueError: If this map is not a subset of the upstream grid: its start
                is not on the grid, its step is not a multiple of the upstream
                step, or it runs past the upstream file's last frame.
        """
        offset = self.start - upstream.start
        if offset < 0 or offset % upstream.step != 0:
            raise ValueError(
                f"source frame {self.start} is not on the upstream grid, which "
                f"starts at source frame {upstream.start} in steps of "
                f"{upstream.step}"
            )
        if self.step % upstream.step != 0:
            raise ValueError(
                f"a step of {self.step} source frames is not a multiple of the "
                f"upstream step of {upstream.step}"
            )
        first = offset // upstream.step
        stride = self.step // upstream.step
        if self.count > 0 and first + stride * (self.count - 1) >= upstream.count:
            raise ValueError(
                f"the frame map spanning {_span(self)} runs past the upstream "
                f"file, which spans {_span(upstream)}"
            )
        return first, stride, self.count

    def is_identity_over(self, frame_count: int) -> bool:
        """Whether this map selects each of *frame_count* source frames, in order."""
        return self.start == 0 and self.step == 1 and self.count == frame_count


@dataclass(frozen=True, slots=True)
class Placement:
    """Map a variant file to its entry's source media.

    The variant image is the source rectangle ``(offset_x, offset_y, width,
    height)``: variant pixel ``(x, y)`` is source pixel
    ``(x + offset_x, y + offset_y)``. Variant frame ``i`` is source frame
    ``frames.start + frames.step * i``. The source's size and frame count are
    stored beside them. A placement reports whether it is the identity without
    reading anything else.

    ``fps`` is the rate that the variant file is labeled at, which sets the file's
    timestamp grid. It is not a statement about real time. The identity
    predicates leave it out. The meaning of a relabeled rate is decided where
    tracks are mapped back.

    Attributes:
        offset_x: The source column of the variant image's left edge.
        offset_y: The source row of the variant image's top edge.
        width: The variant image's width, in pixels.
        height: The variant image's height, in pixels.
        source_width: The entry media's frame width, in pixels.
        source_height: The entry media's frame height, in pixels.
        source_frame_count: The number of frames in the entry media, across all
            of its clips.
        frames: The map from the variant's frames to source frames.
        fps: The variant file's labeled frame rate.
    """

    offset_x: int
    offset_y: int
    width: int
    height: int
    source_width: int
    source_height: int
    source_frame_count: int
    frames: FrameMap
    fps: float

    def __post_init__(self) -> None:
        """Refuse a rectangle or a frame map outside the source, and a non-rate.

        Raises:
            TypeError: If an integer field is not an integer. A numpy integer is
                one.
            ValueError: If the rectangle or the frame map is not inside the
                source, or ``fps`` is not positive and finite.
        """
        _require_integers(
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            width=self.width,
            height=self.height,
            source_width=self.source_width,
            source_height=self.source_height,
            source_frame_count=self.source_frame_count,
        )
        rectangle = f"({self.offset_x}, {self.offset_y}, {self.width}, {self.height})"
        source = f"{self.source_width}x{self.source_height}"
        if self.offset_x < 0 or self.offset_y < 0:
            raise ValueError(
                f"the rectangle {rectangle} starts outside the {source} source. "
                f"Offsets are at least 0"
            )
        if self.width < 1 or self.height < 1:
            raise ValueError(f"the rectangle {rectangle} does not cover a pixel")
        if (
            self.offset_x + self.width > self.source_width
            or self.offset_y + self.height > self.source_height
        ):
            raise ValueError(
                f"the rectangle {rectangle} is not inside the {source} source"
            )
        if self.source_frame_count < 0:
            raise ValueError(
                f"a source of {self.source_frame_count} frames is not a source. "
                f"The count is at least 0"
            )
        last = self.frames.start + self.frames.step * (self.frames.count - 1)
        if self.frames.count > 0 and last >= self.source_frame_count:
            raise ValueError(
                f"the frame map spanning {_span(self.frames)} selects source frame "
                f"{last}, past the source's {self.source_frame_count} frames"
            )
        if not (math.isfinite(self.fps) and self.fps > 0):
            raise ValueError(
                f"a labeled rate of {self.fps} fps is not a frame rate. It must be "
                f"positive and finite"
            )

    @classmethod
    def identity(cls, width: int, height: int, frame_count: int, fps: float) -> Self:
        """Return the identity placement of a source file, labeled at *fps*."""
        return cls(
            offset_x=0,
            offset_y=0,
            width=width,
            height=height,
            source_width=width,
            source_height=height,
            source_frame_count=frame_count,
            frames=FrameMap(0, 1, frame_count),
            fps=fps,
        )

    @property
    def is_spatial_identity(self) -> bool:
        """Whether the variant image is the whole source image, unmoved."""
        return (
            self.offset_x == 0
            and self.offset_y == 0
            and self.width == self.source_width
            and self.height == self.source_height
        )

    @property
    def is_frame_identity(self) -> bool:
        """Whether the variant contains every source frame, in order."""
        return self.frames.is_identity_over(self.source_frame_count)

    @property
    def is_identity(self) -> bool:
        """Whether the variant's pixels and frames are the source's."""
        return self.is_spatial_identity and self.is_frame_identity

    def to_json(self) -> str:
        """Return the placement as canonical JSON: sorted keys, compact separators.

        Equal placements give equal strings, and a stored placement compares as
        text. Numbers are written as plain ``int`` and ``float``. A numpy integer or
        an integral ``fps`` writes the same bytes as the plain value. No value is
        truncated here, because the constructor has already refused a fractional
        integer field.
        """
        document: dict[str, JsonValue] = {
            "offset_x": operator.index(self.offset_x),
            "offset_y": operator.index(self.offset_y),
            "width": operator.index(self.width),
            "height": operator.index(self.height),
            "source_width": operator.index(self.source_width),
            "source_height": operator.index(self.source_height),
            "source_frame_count": operator.index(self.source_frame_count),
            "frames": {
                "start": operator.index(self.frames.start),
                "step": operator.index(self.frames.step),
                "count": operator.index(self.frames.count),
            },
            "fps": float(self.fps),
        }
        return json.dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @classmethod
    def from_json(cls, text: str) -> Self:
        """Read a placement written by :meth:`to_json`, refusing any other shape.

        Raises:
            ValueError: If *text* is not JSON, is not an object with exactly a
                placement's fields, has a field of a disallowed type, or describes
                a placement that the constructor refuses.
        """
        document = _json_object(json.loads(text), "a placement", _PLACEMENT_FIELDS)
        frames = _json_object(
            document["frames"], "a placement's frames", _FRAME_MAP_FIELDS
        )
        return cls(
            offset_x=_json_integer(document, "offset_x"),
            offset_y=_json_integer(document, "offset_y"),
            width=_json_integer(document, "width"),
            height=_json_integer(document, "height"),
            source_width=_json_integer(document, "source_width"),
            source_height=_json_integer(document, "source_height"),
            source_frame_count=_json_integer(document, "source_frame_count"),
            frames=FrameMap(
                start=_json_integer(frames, "start"),
                step=_json_integer(frames, "step"),
                count=_json_integer(frames, "count"),
            ),
            fps=_json_number(document, "fps"),
        )


_FRAME_MAP_FIELDS: Final = frozenset({"start", "step", "count"})
_PLACEMENT_FIELDS: Final = frozenset(
    {
        "offset_x",
        "offset_y",
        "width",
        "height",
        "source_width",
        "source_height",
        "source_frame_count",
        "frames",
        "fps",
    }
)


def _ceil_div(numerator: int, denominator: int) -> int:
    """``ceil(numerator / denominator)`` in integers, for a positive denominator."""
    return -(-numerator // denominator)


def _require_integers(**fields: int) -> None:
    """Raise ``TypeError`` naming the first field that is not an integer.

    :func:`operator.index` accepts ``int`` and numpy integers and refuses a float
    such as ``1.5``, which ``int()`` truncates.
    """
    for name, value in fields.items():
        try:
            _ = operator.index(value)
        except TypeError:
            raise TypeError(f"{name} must be an integer, not {value!r}") from None


def _span(frames: FrameMap) -> str:
    """Return the source span that a frame map covers, as a half-open range."""
    return f"[{frames.start}, {frames.end})"


def _json_object(
    value: JsonValue, what: str, fields: frozenset[str]
) -> dict[str, JsonValue]:
    """Return *value* as a JSON object with exactly *fields*, or raise naming *what*."""
    if not isinstance(value, dict):
        raise ValueError(f"{what} must be a JSON object, not {type(value).__name__}")
    missing = sorted(fields - value.keys())
    unknown = sorted(value.keys() - fields)
    if missing or unknown:
        raise ValueError(
            f"{what} must have exactly the fields {sorted(fields)}; missing "
            f"{missing}, unknown {unknown}"
        )
    return value


def _json_integer(document: dict[str, JsonValue], key: str) -> int:
    """Return the integer at *key*, refusing a boolean, a float or a string."""
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer, not {value!r}")
    return value


def _json_number(document: dict[str, JsonValue], key: str) -> float:
    """Return the number at *key* as a float, refusing a boolean or a string."""
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number, not {value!r}")
    return float(value)
