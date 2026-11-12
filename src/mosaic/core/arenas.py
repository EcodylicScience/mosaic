"""Arena layouts: the tanks of a camera's view, keyed by recording and camera.

A layout names each arena of one camera's frame by a rectangle in source pixels.
An arena may also carry a boundary, its wall when it is not a rectangle, and
named regions of interest (:class:`~mosaic.core.regions.Roi`). A layout is
declared once per recording, keyed by ``(RecordingKey, camera)``, and a lookup
for one entry falls back from its own key to its recording's and from its camera
to every camera (:func:`layout_lookup_order`). This module imports pydantic, the
standard library, :mod:`mosaic.core.entry`, :mod:`mosaic.core.recording` and
:mod:`mosaic.core.regions` only, as a pure model that the agent server imports
lazily and whose schema the API routes publish.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, pairwise
from typing import ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from .entry import validate_entry_name
from .recording import RecordingKey
from .regions import CircleShape, PolygonShape, RectShape, Roi, bounding_box

__all__ = [
    "Arena",
    "ArenaKey",
    "ArenaLayout",
    "ArenaSpace",
    "Rect",
    "layout_lookup_order",
    "region_boundary",
    "validate_arena_name",
]

type ArenaSpace = Literal["image_px"]
"""The space of a layout's rectangles: pixels of the source media's frame."""


class _FrozenModel(BaseModel):
    """A frozen model refusing unknown keys, so a layout is a value."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class Rect(_FrozenModel):
    """A half-open pixel rectangle: ``x0 <= x < x1`` and ``y0 <= y < y1``.

    Attributes:
        x0: The left edge, inside the rectangle.
        y0: The top edge, inside the rectangle.
        x1: The right edge, outside the rectangle.
        y1: The bottom edge, outside the rectangle.
    """

    x0: NonNegativeInt
    y0: NonNegativeInt
    x1: PositiveInt
    y1: PositiveInt

    @model_validator(mode="after")
    def _check_extent(self) -> Self:
        """Refuse a rectangle without area."""
        if self.x0 >= self.x1 or self.y0 >= self.y1:
            message = (
                f"a rect needs x0 < x1 and y0 < y1, got ({self.x0}, {self.y0}) to "
                f"({self.x1}, {self.y1})"
            )
            raise ValueError(message)
        return self

    def overlaps(self, other: Rect) -> bool:
        """Return whether *self* and *other* share a pixel."""
        return (
            self.x0 < other.x1
            and other.x0 < self.x1
            and self.y0 < other.y1
            and other.y0 < self.y1
        )

    def within(self, other: Rect) -> bool:
        """Return whether every pixel of *self* lies in *other*."""
        return (
            other.x0 <= self.x0
            and self.x1 <= other.x1
            and other.y0 <= self.y0
            and self.y1 <= other.y1
        )


def region_boundary(rect: Rect) -> RectShape:
    """Return *rect* as a rect boundary, its float box.

    An arena without a boundary uses its region as one wherever a boundary is
    read.
    """
    return RectShape(
        x0=float(rect.x0), y0=float(rect.y0), x1=float(rect.x1), y1=float(rect.y1)
    )


def _is_none(value: object) -> bool:
    return value is None


def _is_empty(value: tuple[Roi, ...]) -> bool:
    return not value


def _clipped_box(
    shape: CircleShape | PolygonShape, width: float | None, height: float | None
) -> tuple[float, float, float, float] | None:
    """Return the bounding box of *shape* clipped to the frame, ``None`` if empty.

    The frame's near edges are ``x = 0`` and ``y = 0``. *width* and *height* clip
    the far edges when given.
    """
    x_min, y_min, x_max, y_max = bounding_box(shape)
    x_min, y_min = max(x_min, 0.0), max(y_min, 0.0)
    if width is not None:
        x_max = min(x_max, width)
    if height is not None:
        y_max = min(y_max, height)
    if x_min >= x_max or y_min >= y_max:
        return None
    return (x_min, y_min, x_max, y_max)


def validate_arena_name(name: str) -> str:
    """Return *name* when it can be an arena's name, an entry-name token.

    An arena's name joins a recording's key in an individual's identity, so it
    is one path component without the ``__`` and ``:`` that join entry tokens.

    Raises:
        ValueError: If *name* is empty, padded, or holds a forbidden character.
    """
    if not name or name != name.strip():
        message = f"an arena name must be non-empty and unpadded, got {name!r}"
        raise ValueError(message)
    _ = validate_entry_name(name, "arena")
    for joiner in ("__", ":"):
        if joiner in name:
            message = f"an arena name may not contain {joiner!r}: {name!r}"
            raise ValueError(message)
    return name


class Arena(_FrozenModel):
    """One arena of a layout.

    Attributes:
        name: The arena's name, an entry-name token unique in its layout.
        region: The pixels of the arena.
        exclude: Rectangles inside the region whose rows are dropped, such as a
            reflection of a neighboring tank.
        capacity: The most individuals the arena holds in one frame, ``None``
            for no limit.
        boundary: The arena's wall in source pixels, ``None`` for a rectangle:
            the region is then the boundary. Its bounding box, clipped to the
            frame, lies inside the region. Left out of a dump when ``None``, so
            a layout without one keeps its digest.
        rois: The arena's named regions of interest, with unique names. Left
            out of a dump when empty.
    """

    name: str
    region: Rect
    exclude: tuple[Rect, ...] = ()
    capacity: PositiveInt | None = None
    boundary: CircleShape | PolygonShape | None = Field(
        default=None, discriminator="kind", exclude_if=_is_none
    )
    rois: tuple[Roi, ...] = Field(default=(), exclude_if=_is_empty)

    @field_validator("name")
    @classmethod
    def _check_name(cls, name: str) -> str:
        return validate_arena_name(name)

    @model_validator(mode="after")
    def _check_exclude(self) -> Self:
        """Refuse an excluded rectangle outside the region."""
        for rect in self.exclude:
            if not rect.within(self.region):
                message = f"arena {self.name!r} excludes {rect!r} outside its region"
                raise ValueError(message)
        return self

    @model_validator(mode="after")
    def _check_roi_names(self) -> Self:
        """Refuse two regions of interest of one name."""
        names = [roi.name for roi in self.rois]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            message = f"arena {self.name!r} repeats ROI names in rois: {repeated}"
            raise ValueError(message)
        return self


def _digest_of(layout: ArenaLayout) -> str:
    """Return the SHA-256 of *layout*'s canonical JSON, in hex."""
    canonical = json.dumps(
        layout.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class ArenaLayout(_FrozenModel):
    """The arenas of one camera's frame.

    Attributes:
        space: The space of the rectangles, the source media's pixels.
        width: The frame's width in pixels.
        height: The frame's height in pixels.
        arenas: The arenas, in ordinal order. Their regions are disjoint and
            lie inside the frame, and each boundary lies inside its region where
            it is inside the frame.
    """

    space: ArenaSpace = "image_px"
    width: PositiveInt
    height: PositiveInt
    arenas: tuple[Arena, ...] = Field(json_schema_extra={"minItems": 1})

    @field_validator("arenas")
    @classmethod
    def _check_not_empty(cls, arenas: tuple[Arena, ...]) -> tuple[Arena, ...]:
        """Refuse a layout without an arena.

        A length constraint on the field would also count the arenas that failed
        their own checks, and report a bad arena as an empty layout besides.
        """
        if not arenas:
            message = "a layout names at least one arena"
            raise ValueError(message)
        return arenas

    @model_validator(mode="after")
    def _check_arenas(self) -> Self:
        """Refuse a repeated name, a region outside the frame or two overlapping.

        A boundary is refused when its bounding box, clipped to the frame, leaves
        the arena's region. A boundary may reach past the frame, as an arena the
        camera crops does.
        """
        frame = Rect(x0=0, y0=0, x1=self.width, y1=self.height)
        names = [arena.name for arena in self.arenas]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            message = f"arena names repeat: {repeated}"
            raise ValueError(message)
        for arena in self.arenas:
            if not arena.region.within(frame):
                message = (
                    f"arena {arena.name!r} leaves the {self.width}x{self.height} frame"
                )
                raise ValueError(message)
            if arena.boundary is not None:
                self._check_boundary(arena, arena.boundary)
        for first, second in combinations(self.arenas, 2):
            if first.region.overlaps(second.region):
                message = f"arenas {first.name!r} and {second.name!r} overlap"
                raise ValueError(message)
        return self

    def _check_boundary(
        self, arena: Arena, boundary: CircleShape | PolygonShape
    ) -> None:
        """Refuse a boundary whose clipped box is empty or leaves the region."""
        box = _clipped_box(boundary, self.width, self.height)
        if box is None:
            message = (
                f"arena {arena.name!r} has a boundary with no part inside the "
                f"{self.width}x{self.height} frame"
            )
            raise ValueError(message)
        region = arena.region
        x_min, y_min, x_max, y_max = box
        if not (
            region.x0 <= x_min
            and x_max <= region.x1
            and region.y0 <= y_min
            and y_max <= region.y1
        ):
            message = (
                f"arena {arena.name!r} has a boundary reaching outside its region "
                f"{region!r}: its box clipped to the frame is ({x_min}, {y_min}) to "
                f"({x_max}, {y_max})"
            )
            raise ValueError(message)

    @property
    def names(self) -> tuple[str, ...]:
        """The arena names in ordinal order."""
        return tuple(arena.name for arena in self.arenas)

    def digest(self) -> str:
        """Return the SHA-256 of the layout's canonical JSON, in hex.

        Equal layouts have one digest, whatever order their keys were given in.
        """
        return _digest_of(self)

    def assignment_digest(self) -> str:
        """Return the digest of the layout without its boundaries and ROIs.

        It equals :meth:`digest` for a layout that has none, and it stays put
        when a boundary or a region of interest is added, because the arena a
        row falls in depends on the regions alone.
        """
        bare = tuple(
            arena.model_copy(update={"boundary": None, "rois": ()})
            for arena in self.arenas
        )
        return _digest_of(self.model_copy(update={"arenas": bare}))

    @classmethod
    def single(
        cls,
        boundary: CircleShape | PolygonShape,
        *,
        name: str = "A1",
        capacity: int | None = 1,
        rois: Sequence[Roi] = (),
        width: int | None = None,
        height: int | None = None,
    ) -> ArenaLayout:
        """Return a layout of one arena bounded by *boundary*.

        The region is the boundary's bounding box rounded outward to whole
        pixels and clipped to ``x, y >= 0`` and, when given, to *width* and
        *height*, so a circle that the camera crops still has a valid region.
        The frame defaults to the region's far edges, as :meth:`grid` defaults
        to the last edges.

        Args:
            boundary: The arena's wall.
            name: The arena's name.
            capacity: The arena's capacity.
            rois: The arena's regions of interest.
            width: The frame's width, the region's right edge when ``None``.
            height: The frame's height, the region's bottom edge when ``None``.

        Raises:
            ValueError: If no part of *boundary* lies inside the frame.
        """
        box = _clipped_box(boundary, width, height)
        if box is None:
            message = "the boundary has no part inside the frame"
            raise ValueError(message)
        region = Rect(
            x0=math.floor(box[0]),
            y0=math.floor(box[1]),
            x1=math.ceil(box[2]),
            y1=math.ceil(box[3]),
        )
        return cls(
            width=region.x1 if width is None else width,
            height=region.y1 if height is None else height,
            arenas=(
                Arena(
                    name=name,
                    region=region,
                    capacity=capacity,
                    boundary=boundary,
                    rois=tuple(rois),
                ),
            ),
        )

    @classmethod
    def grid(
        cls,
        xs: Sequence[int],
        ys: Sequence[int],
        *,
        width: int | None = None,
        height: int | None = None,
        capacity: int | None = 1,
        exclude: Mapping[str, Sequence[Rect]] | None = None,
    ) -> ArenaLayout:
        """Return a layout of one arena per grid cell.

        Columns are lettered from ``A`` and rows numbered from 1, so the cell in
        the first column and second row is ``A2``. Arenas are ordered column by
        column.

        Args:
            xs: The column edges, increasing.
            ys: The row edges, increasing.
            width: The frame's width, the last column edge when ``None``.
            height: The frame's height, the last row edge when ``None``.
            capacity: Every arena's capacity.
            exclude: The rectangles each named arena excludes.

        Raises:
            ValueError: If the edges do not make a grid of at most 26 columns,
                or *exclude* names an arena the grid lacks.
        """
        if len(xs) < 2 or len(ys) < 2 or len(xs) > 27:
            message = "a grid needs 2 to 27 column edges and at least 2 row edges"
            raise ValueError(message)
        excluded = dict(exclude or {})
        arenas: list[Arena] = []
        for column, (x0, x1) in enumerate(pairwise(xs)):
            for row, (y0, y1) in enumerate(pairwise(ys), start=1):
                name = f"{chr(ord('A') + column)}{row}"
                arenas.append(
                    Arena(
                        name=name,
                        region=Rect(x0=x0, y0=y0, x1=x1, y1=y1),
                        exclude=tuple(excluded.pop(name, ())),
                        capacity=capacity,
                    )
                )
        if excluded:
            message = f"exclude names arenas the grid lacks: {sorted(excluded)}"
            raise ValueError(message)
        return cls(
            width=xs[-1] if width is None else width,
            height=ys[-1] if height is None else height,
            arenas=tuple(arenas),
        )


@dataclass(frozen=True, slots=True, order=True)
class ArenaKey:
    """The key a layout is declared under.

    Attributes:
        group: The recording key's group.
        sequence: The recording key's sequence, ``""`` for a divided recording.
        camera: The camera the layout applies to, ``""`` for every camera.
    """

    group: str
    sequence: str
    camera: str

    @property
    def recording(self) -> RecordingKey:
        """The recording key without the camera."""
        return RecordingKey(self.group, self.sequence)


def layout_lookup_order(
    entry: RecordingKey, recording: RecordingKey, camera: str
) -> tuple[ArenaKey, ...]:
    """Return the keys a lookup for one entry and camera tries, in order.

    The entry's own key, then its recording's, under *camera*; then the same
    two under camera ``""``. Each key is tried once.

    Args:
        entry: The entry's own ``(group, sequence)``.
        recording: The key of the entry's recording, ``(G, "")`` for a divided
            one and the entry's own pair otherwise.
        camera: The camera whose pixels are looked up, ``""`` for none.
    """
    cameras = (camera, "") if camera else ("",)
    keys = (
        ArenaKey(level.group, level.sequence, name)
        for name in cameras
        for level in (entry, recording)
    )
    return tuple(dict.fromkeys(keys))
