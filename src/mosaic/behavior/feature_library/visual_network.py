"""VisualNetwork -- who each individual can see, by ray casting from its eyes.

The method of Davidson, Sosna, Twomey, Sridhar, Leblanc and Couzin (2021),
"Collective detection based on visual information in animal groups", Journal of
the Royal Society Interface 18: 20210142. Each individual casts rays in the
plane from each eye against the body outlines of every individual, and a ray
links its caster to the individual whose outline it meets first.

Geometry, in image coordinates (y down, so a positive turn is clockwise on
screen):

- Each eye casts ``n_rays`` rays at the absolute angles
  ``2 * pi * k / n_rays - pi``, ``k = 0 .. n_rays - 1``, independent of the
  heading, and keeps those inside its field.
- The right eye sits on the positive side of the heading. Its field is the
  heading plus ``[-overlap, pi - blind / 2]``, and the left eye's is the
  heading plus ``[-(pi - blind / 2), overlap]``, with ``overlap`` the binocular
  overlap and ``blind`` the blind angle behind the individual. A ray's angle
  relative to the heading is wrapped to ``[-pi, pi]`` and compared inclusively.
- The default outline is the closed polygon head, left eye, tail, right eye.
  An eye ignores the segments of its own outline that end at that eye. The
  individual's other segments still stop its rays, and a ray stopped by its own
  body links to nobody.
- A pair's ``n_rays`` counts the ray indices ``k`` at which either eye's ray
  ``k`` meets the target first, so a ray seen by both eyes counts once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Final, Literal, Self, final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, model_validator

from mosaic.core.individuals import arena_codes, id_arenas, same_arena_codes
from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.column_contract import (
    ORDER_COLUMN,
    Column,
    Conditional,
    ParamIs,
    Provides,
    Requirement,
    Template,
    When,
    kept,
)
from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pipeline.types import (
    DependencyLookup,
    EmitsLevel,
    Inputs,
    InputStream,
    Result,
    TrackInput,
    resolve_order_col,
)
from mosaic.core.pipeline.types.feature import RelatesLevel

from .helpers import ensure_columns
from .registry import register_feature

__all__ = ["VisualNetwork"]

_NAME = "visual-network"

HeadingSource = Literal["ANGLE", "keypoints"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

_PARALLEL_TOLERANCE: Final = 1e-12
"""The cross product below which a ray and a segment count as parallel."""

_RAY_MARGIN: Final = 1
"""Rays tested past each end of a segment's angular extent.

The extent comes from ``arctan2`` of its ends, and a ray at the very edge of
it could fall either side of the computed boundary. One ray of margin keeps
every ray that meets the segment among those tested.
"""

_WIDE_EXTENT: Final = 1e-6
"""How close to pi a segment's angular extent may come before every ray is tested.

A segment the viewpoint nearly lies on subtends nearly pi, and the side of its
extent is then uncertain.
"""

_OUTPUT_COLUMNS: Final = (
    C.frame_col,
    "id1",
    "id2",
    "perspective",
    "angular_size_deg",
    "n_rays",
    "distance",
)

_HEAD_INDEX_DESCRIPTION = (
    "The pose keypoint index of the head. Without eyes, every ray is cast from it."
)

_TAIL_INDEX_DESCRIPTION = (
    "The pose keypoint index of the tail tip, a corner of the default outline. "
    "-1 means the highest keypoint index the table carries."
)

_LEFT_EYE_INDEX_DESCRIPTION = (
    "The pose keypoint index of the left eye, on the negative side of the "
    "heading in image coordinates. Set with right_eye_index or not at all. "
    "Unset, one viewpoint at the head sees the union of both eyes' fields, and "
    "the default outline is the head-to-tail segment."
)

_RIGHT_EYE_INDEX_DESCRIPTION = (
    "The pose keypoint index of the right eye, on the positive side of the "
    "heading in image coordinates. Set with left_eye_index or not at all."
)

_OUTLINE_INDICES_DESCRIPTION = (
    "The pose keypoint indices of a closed body outline, in order, replacing "
    "the default outline. A viewpoint ignores its own outline's segments that "
    "end at it. Two indices make one segment."
)

_HEADING_SOURCE_DESCRIPTION = (
    "Where the heading comes from: the ANGLE column, in radians in image "
    "coordinates, or keypoints, the direction from heading_back_index to the "
    "head."
)

_HEADING_BACK_INDEX_DESCRIPTION = (
    "The pose keypoint index the heading points away from, when heading_source "
    "is keypoints."
)

_N_RAYS_DESCRIPTION = (
    "The rays each eye casts around the full circle. One ray spans "
    "360 / n_rays degrees, the resolution of angular_size_deg."
)

_BLIND_ANGLE_DESCRIPTION = "The width of the blind wedge behind the individual."

_BINOCULAR_OVERLAP_DESCRIPTION = (
    "How far past the heading each eye's field reaches into the other side."
)

_MIN_RAYS_DESCRIPTION = "The fewest rays that make a target visible."

_INCLUDE_HIDDEN_DESCRIPTION = (
    "Whether to write every ordered pair of one arena present in a frame, with a "
    "visible column, instead of the visible pairs alone. Set it for a consumer "
    "that expects every pair, such as temporal-stack or frame-aggregate."
)

_X_PREFIX_DESCRIPTION = "The column name prefix for pose X coordinates."

_Y_PREFIX_DESCRIPTION = "The column name prefix for pose Y coordinates."


def _wrap(angle: FloatArray) -> FloatArray:
    """Wrap angles in radians to ``[-pi, pi]``."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def _cross(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _candidates(
    first: FloatArray, second: FloatArray, n_rays: int
) -> tuple[IntArray, IntArray]:
    """Return the ``(ray, segment)`` pairs whose ray may meet the segment.

    A segment subtends less than pi from a point off its line, and only the rays
    between the directions of its two ends can meet it. Each segment is paired
    with those rays and :data:`_RAY_MARGIN` more at each side, wrapping past the
    ray at ``-pi``. A segment whose extent nears pi, or whose end is the
    viewpoint itself, is paired with every ray.

    Args:
        first: The ``(S, 2)`` first ends of the segments, relative to the viewpoint.
        second: The ``(S, 2)`` second ends, relative to the viewpoint.
        n_rays: The rays around the full circle, ray ``k`` at
            ``2 * pi * k / n_rays - pi``.
    """
    start_angle = np.arctan2(first[:, 1], first[:, 0])
    end_angle = np.arctan2(second[:, 1], second[:, 0])
    turn = _wrap(end_angle - start_angle)
    low = np.where(turn >= 0, start_angle, end_angle)
    extent = np.abs(turn)
    spacing = 2 * np.pi / n_rays
    lowest = np.floor((low + np.pi) / spacing).astype(np.int64) - _RAY_MARGIN
    counts = np.ceil(extent / spacing).astype(np.int64) + 1 + 2 * _RAY_MARGIN
    whole = (
        (extent > np.pi - _WIDE_EXTENT)
        | ~np.any(first != 0, axis=1)
        | ~np.any(second != 0, axis=1)
    )
    lowest[whole] = 0
    counts = np.where(whole, n_rays, np.minimum(counts, n_rays))
    segments = np.repeat(np.arange(len(first), dtype=np.int64), counts)
    offsets = np.arange(len(segments), dtype=np.int64) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    rays = (np.repeat(lowest, counts) + offsets) % n_rays
    return rays, segments


def _first_hits(
    origin: FloatArray,
    directions: FloatArray,
    inside: BoolArray,
    starts: FloatArray,
    ends: FloatArray,
) -> IntArray:
    """Return, per ray, the index of the segment it meets first, ``-1`` for none.

    A ray outside *inside* meets nothing. A ray is tested only against the
    segments whose angular extent it falls in (:func:`_candidates`), and among
    equally near segments the lowest index wins.

    Args:
        origin: The ``(2,)`` point every ray leaves from.
        directions: The ``(R, 2)`` unit directions of all rays.
        inside: The ``(R,)`` rays the viewpoint keeps.
        starts: The ``(S, 2)`` first ends of the segments.
        ends: The ``(S, 2)`` second ends of the segments.
    """
    hits = np.full(len(directions), -1, dtype=np.int64)
    if len(starts) == 0:
        return hits
    first = starts - origin
    second = ends - origin
    along = second - first
    rays, segments = _candidates(first, second, len(directions))
    kept = inside[rays]
    rays, segments = rays[kept], segments[kept]
    ray = directions[rays]
    offset = first[segments]
    span = along[segments]
    denominator = _cross(ray, span)
    with np.errstate(divide="ignore", invalid="ignore"):
        reach = _cross(offset, span) / denominator
        position = _cross(offset, ray) / denominator
    meets = (
        (np.abs(denominator) > _PARALLEL_TOLERANCE)
        & (reach > 0)
        & (position >= 0)
        & (position <= 1)
    )
    rays, segments, reach = rays[meets], segments[meets], reach[meets]
    if len(rays) == 0:
        return hits
    order = np.lexsort((segments, reach, rays))
    rays, segments = rays[order], segments[order]
    nearest = np.r_[True, rays[1:] != rays[:-1]]
    hits[rays[nearest]] = segments[nearest]
    return hits


@dataclass(frozen=True)
class _Viewpoint:
    """One place rays are cast from, and the field it keeps.

    Attributes:
        keypoint: The column position, in the plan's keypoints, of the viewpoint.
        fields: The ``(low, high)`` ranges of ray angle relative to the heading
            that it keeps, in radians, each inclusive.
        own_skipped: Per outline segment, whether the viewpoint ignores it on
            its own outline.
    """

    keypoint: int
    fields: tuple[tuple[float, float], ...]
    own_skipped: BoolArray


@dataclass(frozen=True)
class _Plan:
    """The keypoints, outline and viewpoints that one table's casting reads.

    Attributes:
        keypoints: The pose indices read, in column order of the coordinates.
        segment_starts: Per outline segment, the position of its first end.
        segment_ends: Per outline segment, the position of its second end.
        viewpoints: The viewpoints each individual casts from.
        heading_back: The position of the heading's back keypoint, or ``None``
            when the heading comes from ``ANGLE``.
    """

    keypoints: tuple[int, ...]
    segment_starts: IntArray
    segment_ends: IntArray
    viewpoints: tuple[_Viewpoint, ...]
    heading_back: int | None


@dataclass(frozen=True)
class _Frame:
    """The individuals of one frame that can observe, occlude and be seen.

    Attributes:
        codes: Each individual's code among the table's sorted ids.
        points: The ``(n, K, 2)`` keypoint coordinates.
        headings: The ``(n,)`` headings, in radians.
        centers: The ``(n, 2)`` body centers.
    """

    codes: IntArray
    points: FloatArray
    headings: FloatArray
    centers: FloatArray


@dataclass(frozen=True)
class _Links:
    """The ordered pairs one frame writes, as positions in the frame.

    Attributes:
        observers: The observer of each pair.
        targets: The target of each pair.
        counts: The rays that link them.
    """

    observers: IntArray
    targets: IntArray
    counts: IntArray


@final
@register_feature
class VisualNetwork:
    """Per-frame visibility between individuals, by 2-D ray casting.

    One row per ordered ``(id1, id2)`` pair per frame in which at least
    ``min_rays`` rays from ``id1``'s eyes meet ``id2`` before any other body:
    ``id1`` sees ``id2``. The output is sparse, and an absent row means not
    visible. ``temporal-stack`` and ``frame-aggregate`` expect every pair in
    every frame, so run this feature with ``include_hidden=True`` before them.

    Output columns, keyed by ``(frame, id1, id2, perspective)``:

    - ``frame``
    - ``id1`` (the observer), ``id2`` (the target), ``perspective``: 0 when
      ``id1 < id2``, else 1
    - ``angular_size_deg``: ``n_rays * 360 / n_rays_param``, the angle the
      target subtends in the observer's view
    - ``n_rays``: the ray indices at which either eye meets the target first
    - ``distance``: between the two body centers ``X``/``Y``, in their unit
    - ``visible`` (only with ``include_hidden``): ``n_rays >= min_rays``

    Coordinates are image coordinates, with y down, and ``ANGLE`` is the
    heading in radians in them. The right eye is on the positive side of the
    heading. An individual with a missing value among the coordinates and the
    heading the casting reads is left out of that frame: it observes nothing,
    hides nothing and is seen by nobody. Individuals of different arenas neither
    see nor hide each other. The geometry is in the module docstring, after
    Davidson et al. (2021), "Collective detection based on visual information in
    animal groups", J. R. Soc. Interface 18: 20210142.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.visual_network.VisualNetwork.Params`.
    """

    category = "per-frame"
    name = _NAME
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    temporal_reach = 0  # each frame is cast on its own
    relates: RelatesLevel = "arena-aware"
    accepts_overlap = False  # computes within a frame, so gains nothing
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "pair"
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.id_col),
        ORDER_COLUMN,
        Column(C.x_col),
        Column(C.y_col),
        Template("{x_prefix}{head_index}"),
        Template("{y_prefix}{head_index}"),
        Template("{x_prefix}", family=True),
        Template("{y_prefix}", family=True),
        When(Template("{x_prefix}{left_eye_index}"), (ParamIs("left_eye_index"),)),
        When(Template("{y_prefix}{left_eye_index}"), (ParamIs("left_eye_index"),)),
        When(Template("{x_prefix}{right_eye_index}"), (ParamIs("right_eye_index"),)),
        When(Template("{y_prefix}{right_eye_index}"), (ParamIs("right_eye_index"),)),
        When(Column(C.orientation_col), (ParamIs("heading_source", equals="ANGLE"),)),
        When(
            Template("{x_prefix}{heading_back_index}"),
            (ParamIs("heading_source", equals="keypoints"),),
        ),
        When(
            Template("{y_prefix}{heading_back_index}"),
            (ParamIs("heading_source", equals="keypoints"),),
        ),
    )
    provides: ClassVar[Provides] = Provides(
        adds=tuple(Column(name) for name in _OUTPUT_COLUMNS),
        conditional=(
            Conditional(Column("visible"), when_param="include_hidden"),
            *kept(C.group_col, C.seq_col),
        ),
    )

    # The tracks, joined with heading's ANGLE where they do not record it.
    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        head_index: Annotated[int, Field(ge=0), Declared(_HEAD_INDEX_DESCRIPTION)] = 0
        tail_index: Annotated[int, Field(ge=-1), Declared(_TAIL_INDEX_DESCRIPTION)] = -1
        left_eye_index: Annotated[
            int | None, Field(ge=0), Declared(_LEFT_EYE_INDEX_DESCRIPTION)
        ] = None
        right_eye_index: Annotated[
            int | None, Field(ge=0), Declared(_RIGHT_EYE_INDEX_DESCRIPTION)
        ] = None
        outline_indices: Annotated[
            list[Annotated[int, Field(ge=0)]] | None,
            Field(min_length=2),
            Declared(_OUTLINE_INDICES_DESCRIPTION),
        ] = None
        heading_source: Annotated[
            HeadingSource, Declared(_HEADING_SOURCE_DESCRIPTION)
        ] = "ANGLE"
        heading_back_index: Annotated[
            int, Field(ge=0), Declared(_HEADING_BACK_INDEX_DESCRIPTION)
        ] = 2
        n_rays: Annotated[int, Field(ge=8), Declared(_N_RAYS_DESCRIPTION)] = 1000
        blind_angle_deg: Annotated[
            float,
            Field(ge=0.0, le=360.0),
            Declared(_BLIND_ANGLE_DESCRIPTION, unit="deg"),
        ] = 25.0
        binocular_overlap_deg: Annotated[
            float,
            Field(ge=0.0, le=180.0),
            Declared(_BINOCULAR_OVERLAP_DESCRIPTION, unit="deg"),
        ] = 12.5
        min_rays: Annotated[int, Field(ge=1), Declared(_MIN_RAYS_DESCRIPTION)] = 1
        include_hidden: Annotated[bool, Declared(_INCLUDE_HIDDEN_DESCRIPTION)] = False
        x_prefix: Annotated[str, Declared(_X_PREFIX_DESCRIPTION)] = "poseX"
        y_prefix: Annotated[str, Declared(_Y_PREFIX_DESCRIPTION)] = "poseY"

        @model_validator(mode="after")
        def _check_eyes(self) -> Self:
            """Refuse one eye without the other, which no table could make valid."""
            if (self.left_eye_index is None) != (self.right_eye_index is None):
                message = (
                    f"{_NAME}: set left_eye_index and right_eye_index together, "
                    "or neither for one viewpoint at the head."
                )
                raise ValueError(message)
            return self

    def __init__(
        self,
        inputs: VisualNetwork.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        order_col = resolve_order_col(df)
        plan = self._plan(df)
        needed = [C.id_col, C.x_col, C.y_col]
        for index in plan.keypoints:
            needed.extend(self._coordinate_columns(index))
        if plan.heading_back is None:
            needed.append(C.orientation_col)
        ensure_columns(df, needed)
        if df.empty:
            return self._empty_output(df, order_col)

        keys = [C.seq_col, order_col] if C.seq_col in df.columns else [order_col]
        table = df.sort_values([*keys, C.id_col], kind="stable")
        codes_all, uniques = pd.factorize(table[C.id_col], sort=True)
        arenas = id_arenas(df)
        arena_of = None if arenas is None else arena_codes(list(uniques), arenas)

        points = np.stack(
            [
                table[list(self._coordinate_columns(index))].to_numpy(dtype=np.float64)
                for index in plan.keypoints
            ],
            axis=1,
        )
        centers = table[[C.x_col, C.y_col]].to_numpy(dtype=np.float64)
        if plan.heading_back is None:
            headings = table[C.orientation_col].to_numpy(dtype=np.float64)
        else:
            head = plan.keypoints.index(p.head_index)
            offset = points[:, head] - points[:, plan.heading_back]
            headings = np.arctan2(offset[:, 1], offset[:, 0])
        usable = (
            np.isfinite(points).all(axis=(1, 2))
            & np.isfinite(centers).all(axis=1)
            & np.isfinite(headings)
            & (codes_all >= 0)
        )

        frame_keys = table[keys].reset_index(drop=True)
        starts = np.flatnonzero(
            np.r_[True, (frame_keys.shift() != frame_keys).any(axis=1).to_numpy()[1:]]
        )
        bounds = [int(start) for start in starts] + [len(table)]

        angles = 2 * np.pi * np.arange(p.n_rays) / p.n_rays - np.pi
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

        pair_parts: list[_Links] = []
        observer_parts: list[IntArray] = []
        target_parts: list[IntArray] = []
        for first, stop in zip(bounds[:-1], bounds[1:], strict=True):
            kept_rows: IntArray = np.arange(first, stop)[usable[first:stop]]
            if len(kept_rows) < 2:
                continue
            frame = _Frame(
                codes=codes_all[kept_rows].astype(np.int64),
                points=points[kept_rows],
                headings=headings[kept_rows],
                centers=centers[kept_rows],
            )
            links = self._links(frame, plan, arena_of, angles, directions)
            if len(links.observers) == 0:
                continue
            pair_parts.append(links)
            observer_parts.append(kept_rows[links.observers])
            target_parts.append(kept_rows[links.targets])

        if not pair_parts:
            return self._empty_output(df, order_col)

        observer_rows = np.concatenate(observer_parts)
        target_rows = np.concatenate(target_parts)
        counts = np.concatenate([links.counts for links in pair_parts])
        observer_codes = codes_all[observer_rows]
        target_codes = codes_all[target_rows]
        separation = centers[observer_rows] - centers[target_rows]

        out = pd.DataFrame(
            {
                C.frame_col: table[order_col].to_numpy()[observer_rows],
                "id1": uniques.take(observer_codes),
                "id2": uniques.take(target_codes),
                "perspective": (observer_codes > target_codes).astype(np.int64),
                "angular_size_deg": counts * 360.0 / p.n_rays,
                "n_rays": counts.astype(np.int64),
                "distance": np.hypot(separation[:, 0], separation[:, 1]),
            }
        )
        if p.include_hidden:
            out["visible"] = counts >= p.min_rays
        for col in (C.group_col, C.seq_col):
            if col in table.columns:
                out[col] = table[col].to_numpy()[observer_rows]
        return out

    def _links(
        self,
        frame: _Frame,
        plan: _Plan,
        arena_of: IntArray | None,
        angles: FloatArray,
        directions: FloatArray,
    ) -> _Links:
        """Cast every individual's rays in one frame and count its links."""
        p = self.params
        n = len(frame.codes)
        n_segments = len(plan.segment_starts)
        owners = np.repeat(np.arange(n), n_segments)
        segment_starts = frame.points[:, plan.segment_starts].reshape(-1, 2)
        segment_ends = frame.points[:, plan.segment_ends].reshape(-1, 2)
        positions = np.arange(n)
        arenas = None if arena_of is None else arena_of[frame.codes]

        observers: list[IntArray] = []
        targets: list[IntArray] = []
        counts: list[IntArray] = []
        for i in range(n):
            peers = (
                np.ones(n, dtype=bool)
                if arenas is None
                else same_arena_codes(arenas[i : i + 1], arenas)
            )
            peers[i] = True
            relative = _wrap(angles - _wrap(frame.headings[i : i + 1]))
            seen: list[IntArray] = []
            for viewpoint in plan.viewpoints:
                inside = np.zeros(len(angles), dtype=bool)
                for low, high in viewpoint.fields:
                    inside |= (relative >= low) & (relative <= high)
                casting = peers[owners]
                casting[i * n_segments : (i + 1) * n_segments] &= ~viewpoint.own_skipped
                segments = np.flatnonzero(casting)
                hit = _first_hits(
                    frame.points[i, viewpoint.keypoint],
                    directions,
                    inside,
                    segment_starts[segments],
                    segment_ends[segments],
                )
                rays = np.flatnonzero(hit >= 0)
                hit_owner = owners[segments[hit[rays]]]
                other = hit_owner != i
                seen.append(rays[other] * n + hit_owner[other])
            union = np.unique(np.concatenate(seen))
            per_target = np.bincount(union % n, minlength=n).astype(np.int64)
            if p.include_hidden:
                chosen = positions[(positions != i) & peers]
            else:
                chosen = np.flatnonzero(per_target >= p.min_rays)
            observers.append(np.full(len(chosen), i, dtype=np.int64))
            targets.append(chosen.astype(np.int64))
            counts.append(per_target[chosen])
        return _Links(
            observers=np.concatenate(observers),
            targets=np.concatenate(targets),
            counts=np.concatenate(counts),
        )

    def _plan(self, df: pd.DataFrame) -> _Plan:
        """Resolve the keypoints, the outline and the viewpoints against *df*."""
        p = self.params
        tail = p.tail_index if p.tail_index >= 0 else self._last_keypoint(df)
        eyes = (
            None
            if p.left_eye_index is None or p.right_eye_index is None
            else (p.left_eye_index, p.right_eye_index)
        )
        if p.outline_indices is not None:
            outline = list(p.outline_indices)
        elif eyes is not None:
            outline = [p.head_index, eyes[0], tail, eyes[1]]
        else:
            outline = [p.head_index, tail]
        if len(outline) == 2:
            pairs = [(outline[0], outline[1])]
        else:
            pairs = [
                (outline[k], outline[(k + 1) % len(outline)])
                for k in range(len(outline))
            ]

        back = p.heading_back_index if p.heading_source == "keypoints" else None
        wanted = [
            p.head_index,
            *outline,
            *(eyes or ()),
            *([back] if back is not None else []),
        ]
        keypoints = tuple(dict.fromkeys(wanted))
        position = {index: k for k, index in enumerate(keypoints)}

        blind_edge = np.pi - np.deg2rad(p.blind_angle_deg) / 2
        overlap = np.deg2rad(p.binocular_overlap_deg)
        right_field = (-overlap, blind_edge)
        left_field = (-blind_edge, overlap)

        def viewpoint(
            index: int, fields: tuple[tuple[float, float], ...]
        ) -> _Viewpoint:
            skipped = np.array([index in pair for pair in pairs], dtype=bool)
            return _Viewpoint(position[index], fields, skipped)

        if eyes is None:
            viewpoints = (viewpoint(p.head_index, (left_field, right_field)),)
        else:
            viewpoints = (
                viewpoint(eyes[0], (left_field,)),
                viewpoint(eyes[1], (right_field,)),
            )
        return _Plan(
            keypoints=keypoints,
            segment_starts=np.array([position[a] for a, _ in pairs], dtype=np.int64),
            segment_ends=np.array([position[b] for _, b in pairs], dtype=np.int64),
            viewpoints=viewpoints,
            heading_back=None if back is None else position[back],
        )

    def _last_keypoint(self, df: pd.DataFrame) -> int:
        """Return the highest pose index with both coordinate columns in *df*."""
        p = self.params
        pattern = re.compile(rf"{re.escape(p.x_prefix)}(\d+)")
        indices = [
            int(found.group(1))
            for column in map(str, df.columns)
            if (found := pattern.fullmatch(column)) is not None
            and f"{p.y_prefix}{found.group(1)}" in df.columns
        ]
        if not indices:
            message = (
                f"{_NAME}: tail_index -1 names the last keypoint, and the table "
                f"has no {p.x_prefix}<i>/{p.y_prefix}<i> columns."
            )
            raise ValueError(message)
        return max(indices)

    def _coordinate_columns(self, index: int) -> tuple[str, str]:
        p = self.params
        return f"{p.x_prefix}{index}", f"{p.y_prefix}{index}"

    def _empty_output(self, df: pd.DataFrame, order_col: str) -> pd.DataFrame:
        columns: dict[str, pd.Series] = {
            C.frame_col: pd.Series([], dtype=df[order_col].dtype),
            "id1": pd.Series([], dtype=df[C.id_col].dtype),
            "id2": pd.Series([], dtype=df[C.id_col].dtype),
            "perspective": pd.Series([], dtype=np.int64),
            "angular_size_deg": pd.Series([], dtype=np.float64),
            "n_rays": pd.Series([], dtype=np.int64),
            "distance": pd.Series([], dtype=np.float64),
        }
        if self.params.include_hidden:
            columns["visible"] = pd.Series([], dtype=bool)
        for col in (C.group_col, C.seq_col):
            if col in df.columns:
                columns[col] = pd.Series([], dtype=df[col].dtype)
        return pd.DataFrame(columns)
