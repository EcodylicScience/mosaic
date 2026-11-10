"""Each individual's 3D keypoints, expressed in its own body frame.

A world-frame table says where every keypoint is in the room. Posture is not
that. Two animals holding one pose at opposite ends of an arena, facing opposite
ways, carry entirely different world coordinates, and a clustering of world
coordinates separates places and headings before it separates anything an animal
does with its body. This feature removes position and heading, the two
quantities a body frame is defined to absorb, and passes on what is left.

The frame is built from keypoints the caller names, because which landmarks span
a body is a fact about the skeleton rather than something to infer:

- **Origin**: the mean of the ``origin`` keypoints, both hips for instance.
- **+Z**: world up, ``(0, 0, 1)``.
- **+X and +Y**, forward and the subject's left, from one keypoint pair:
  - ``lateral`` (left, right): +Y is the horizontal part (``z`` set to zero) of
    ``left - right``, normalized, and +X is ``Y x Z``.
  - ``heading`` (front, rear): +X is the horizontal part of ``front - rear``,
    normalized, and +Y is ``Z x X``. This serves a skeleton whose keypoints lie
    on its midline and form no left-right pair.

Either way the frame is right-handed like the world frame and turns with the
body about the vertical axis only. Pitch and roll stay in the egocentric
coordinates, which is where they belong: a rearing animal is in a posture, and a
frame tilted to follow the spine would rotate the rearing out of the numbers
that are supposed to show it. For the same reason ``ego_z`` is height above the
origin along world up.

The clustering and sequence features (``global-scaler``, ``global-tsne``,
``global-kmeans``, ``arhmm``) take ``Result`` inputs only, so this is also the
step through which a world-frame table's posture reaches them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Self, final

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from mosaic.core.keypoint_refs import (
    check_heading_pair,
    check_keypoint_refs,
    check_keypoints_distinct,
    keypoint_index,
    resolve_keypoint_refs,
)
from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.column_contract import (
    META_COLUMNS_KEPT,
    POSE_X,
    POSE_Y,
    Column,
    Conditional,
    Prefix,
    Provides,
    Requirement,
)
from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pipeline.types import (
    DependencyLookup,
    EmitsLevel,
    InputStream,
    TrackInputs,
)
from mosaic.core.pipeline.types.feature import RelatesLevel
from mosaic.core.schema import TrackSpace

from .derivatives import (
    BoolArray,
    FloatArray,
    IntArray,
    KeypointColumns,
    check_keypoint_names,
    keypoint_columns,
)
from .helpers import meta_columns, wrap_angle
from .registry import register_feature
from .world_frame import FrameKind, KeypointRef, body_axes, world_points

__all__ = ["PoseEgocentric"]

_NAME = "pose-egocentric"

_KeypointGroup = Annotated[list[KeypointRef], Field(min_length=1)]

_ORIGIN_DESCRIPTION = (
    "The keypoints whose mean is the body frame's origin, by index or by name, "
    "such as both hips. Every one must be present on a row for the origin to be "
    "defined there: a mean of whichever happen to be detected moves the origin "
    "from row to row, and that movement would read as a change of posture."
)

_LATERAL_DESCRIPTION = (
    "The (left, right) keypoint pair, by index or by name. The horizontal part of "
    "left minus right is the subject's left, the body frame's +Y, and forward is "
    "+Y cross up. Left and right are the subject's own, not the camera's. Set "
    "exactly one of lateral and heading."
)

_HEADING_DESCRIPTION = (
    "The (front, rear) keypoint pair, by index or by name, such as head and "
    "tail base on a midline skeleton. The horizontal part of front minus rear is "
    "forward, the body frame's +X, and the subject's left is up cross +X. Set "
    "exactly one of lateral and heading."
)

_KEYPOINT_NAMES_DESCRIPTION = (
    "A name for each keypoint, in the table's keypoint order, so that origin, "
    "lateral, heading, keypoints and scale_by may name keypoints rather than "
    "number them. Its length must equal the table's keypoint count. Output "
    "columns are numbered by position whether or not names are given."
)

_KEYPOINTS_DESCRIPTION = (
    "Which keypoints to express in the body frame and emit, by index or by name. "
    "An index is a keypoint's position in the table's keypoint order, and keypoint "
    "k is emitted as ego_x{k}, ego_y{k} and ego_z{k}. Unset, every keypoint the "
    "table carries."
)

_SCALE_BY_DESCRIPTION = (
    "Two keypoint groups, such as the hips and the shoulders. Set, the egocentric "
    "coordinates are divided by the median distance between the two groups' "
    "centroids, taken per individual and per sequence, so that posture compares "
    "across body sizes. Unset, they stay in the table's length unit."
)

_CONTEXT_DESCRIPTION = (
    "Also emit body_yaw and origin_x/y/z: where the body frame sits and which way "
    "it faces. They are context, not posture, and a model fitted on every numeric "
    "column (arhmm, the template features) would fit them as posture; body_yaw "
    "jumps by 2 pi where it wraps. Off by default for that reason."
)

_MAX_GAP_DESCRIPTION = (
    "The longest run of undefined values filled by linear interpolation, per "
    "individual and per output column, after the body frame is computed. A longer "
    "run stays undefined throughout, and a run at either end of an individual's "
    "track is never filled. 0 fills nothing. arhmm replaces an undefined value "
    "with zero, which it then fits as a pose, so a run meant for it usually wants "
    "this set."
)


@dataclass(frozen=True, slots=True)
class _Plan:
    """The parameters' keypoint references, resolved against one table.

    Attributes:
        keypoints: The coordinate triplet of every keypoint the table carries,
            in keypoint order.
        origin: The keypoints whose mean is the origin.
        kind: Whether ``pair`` is a lateral or a heading pair.
        pair: The (left, right) or (front, rear) keypoints.
        emit: The keypoints expressed in the body frame.
        scale: The two keypoint groups whose separation is the body size.
    """

    keypoints: KeypointColumns
    origin: tuple[int, ...]
    kind: FrameKind
    pair: tuple[int, int]
    emit: tuple[int, ...]
    scale: tuple[tuple[int, ...], tuple[int, ...]] | None


def _centroid(frame: pd.DataFrame, triplets: Sequence[tuple[str, ...]]) -> FloatArray:
    """The mean of *triplets*, NaN on any row missing one of them.

    A plain mean on purpose, not ``nanmean``. A centroid of whichever keypoints
    were detected is a different point on every row that detects a different
    set, and the difference would arrive downstream as movement of the body.
    """
    total = np.zeros((len(frame), 3), dtype=np.float64)
    for triplet in triplets:
        total += world_points(frame, triplet)
    return total / len(triplets)


def _segment_bounds(segment: IntArray) -> tuple[IntArray, IntArray]:
    """The first and last row of each row's segment, for rows sorted by segment."""
    n = segment.shape[0]
    is_start = np.ones(n, dtype=bool)
    is_start[1:] = segment[1:] != segment[:-1]
    starts = np.flatnonzero(is_start).astype(np.int64)
    ends = np.append(starts[1:] - 1, n - 1).astype(np.int64)
    which = np.cumsum(is_start) - 1
    return starts[which], ends[which]


@dataclass(frozen=True, slots=True)
class _GapFill:
    """Where a column's short interior gaps are, and how to fill them.

    Planned from a column's definedness alone, so the columns that share one
    mask -- the three axes of a keypoint, the three of the origin -- share one
    plan, and the search for gaps runs once for each rather than once per
    column.

    Attributes:
        targets: The rows to fill.
        before: Each target's nearest defined row before it, in its segment.
        after: The same, after it.
        weight: How far each target sits from *before* toward *after*, in
            ``(0, 1)``, measured in frame numbers.
    """

    targets: IntArray
    before: IntArray
    after: IntArray
    weight: FloatArray

    @classmethod
    def plan(
        cls,
        defined: BoolArray,
        frames: FloatArray,
        first_row: IntArray,
        last_row: IntArray,
        max_gap: int,
    ) -> _GapFill | None:
        """Plan the fill of each interior run no longer than *max_gap*.

        A run's length is counted in frame numbers between the two defined
        samples that bound it, so frames with no row at all count toward it: a
        gap of missing rows is as much a gap as a run of NaN rows. A run longer
        than *max_gap* is left whole rather than filled at its ends, since a
        partly filled gap is an invented trajectory with nothing to say where
        the invention starts. A run touching either end of a segment has one
        bound only, and filling it would be extrapolation.

        Args:
            defined: Which rows of the column are finite, sorted by segment and
                then frame.
            frames: The rows' frame numbers, float64.
            first_row: Each row's segment's first row.
            last_row: Each row's segment's last row.
            max_gap: The longest run to fill, in frames.

        Returns:
            The plan, or ``None`` when there is nothing to fill.
        """
        n = len(defined)
        if max_gap < 1 or defined.all() or not defined.any():
            return None
        rows = np.arange(n, dtype=np.int64)
        prev = np.maximum.accumulate(np.where(defined, rows, -1))
        nxt = np.minimum.accumulate(np.where(defined, rows, n)[::-1])[::-1]
        targets = np.flatnonzero(
            ~defined & (prev >= first_row) & (nxt <= last_row)
        ).astype(np.int64)
        before = prev[targets]
        after = nxt[targets]
        span = frames[after] - frames[before]
        short = (span > 0) & (span - 1 <= max_gap)
        if not short.any():
            return None
        targets, before, after, span = (
            targets[short],
            before[short],
            after[short],
            span[short],
        )
        weight = (frames[targets] - frames[before]) / span
        return cls(targets=targets, before=before, after=after, weight=weight)

    def fill(self, values: FloatArray, *, angular: bool = False) -> FloatArray:
        """A copy of *values* with the planned rows linearly interpolated.

        Args:
            values: The column the plan was made from, or one sharing its mask.
            angular: Interpolate along the shorter arc and wrap to
                ``[-pi, pi)``, for an angle column, where a straight line from
                ``3.1`` to ``-3.1`` would sweep through zero.
        """
        delta = values[self.after] - values[self.before]
        if angular:
            delta = wrap_angle(delta)
        filled = values[self.before] + self.weight * delta
        out = values.copy()
        out[self.targets] = wrap_angle(filled) if angular else filled
        return out


@final
@register_feature
class PoseEgocentric:
    """Per-frame 3D keypoints in each individual's own body frame.

    One row per input row, keyed by ``frame`` and ``id`` with the metadata
    columns carried through, and these measurements as float32:

    - ``ego_x{k}``, ``ego_y{k}``, ``ego_z{k}`` for each emitted keypoint, ``k``
      its position in the table's keypoint order: forward, leftward and upward
      of the origin, in the table's length unit (meters for a world table), or
      as a fraction of the body size under ``scale_by``. This is the posture.

    With ``context`` set, also:

    - ``body_yaw``: the direction of +X in the world XY plane, radians
      counterclockwise from world +X seen from above, in ``[-pi, pi)``.
    - ``origin_x``, ``origin_y``, ``origin_z``: where the origin is in the world.

    ``body_yaw`` and ``origin_*`` are context -- where the body frame sits and
    which way it faces -- and not posture, so a model fitted on every numeric
    column (``arhmm``, the template features) does not read them by default.

    Each value is NaN exactly where something it is computed from is undefined,
    never zero. The egocentric coordinates need the whole frame: the origin,
    the frame's keypoint pair, a pair vector that is neither zero nor vertical,
    the keypoint itself and, under ``scale_by``, a measurable body size.
    ``body_yaw`` needs only the pair, and ``origin_*`` only the origin
    keypoints. A keypoint missing any one axis is missing. ``max_gap`` then
    fills short interior runs.

    Runs on world-frame tables (``mosaic_world_v1``) only. Field documentation
    is on
    :class:`~mosaic.behavior.feature_library.pose_egocentric.PoseEgocentric.Params`.
    """

    category = "per-frame"
    name = _NAME
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "none"
    # scale_by takes a median over each individual's rows in one sequence. Handed
    # a neighbor's overlap rows, that median would be partly the neighbor's.
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    # Gaps are filled and the body size is a median along each individual's rows.
    follows_ids: bool = True
    track_spaces: ClassVar[frozenset[TrackSpace]] = frozenset({"world_m"})
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.frame_col),
        Column(C.id_col),
        POSE_X,
        POSE_Y,
        Prefix("poseZ"),
    )
    provides: ClassVar[Provides] = Provides(
        conditional=(
            Conditional(Prefix("ego_x")),
            Conditional(Prefix("ego_y")),
            Conditional(Prefix("ego_z")),
            Conditional(Column("body_yaw"), when_param="context"),
            Conditional(Column("origin_x"), when_param="context"),
            Conditional(Column("origin_y"), when_param="context"),
            Conditional(Column("origin_z"), when_param="context"),
            *META_COLUMNS_KEPT,
        ),
    )

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        """Pose-egocentric parameters."""

        origin: Annotated[
            list[KeypointRef],
            Field(min_length=1),
            Declared(_ORIGIN_DESCRIPTION, keypoint="either"),
        ]
        lateral: Annotated[
            tuple[KeypointRef, KeypointRef] | None,
            Declared(_LATERAL_DESCRIPTION, keypoint="either"),
        ] = None
        heading: Annotated[
            tuple[KeypointRef, KeypointRef] | None,
            Declared(_HEADING_DESCRIPTION, keypoint="either"),
        ] = None
        keypoint_names: Annotated[
            list[str] | None, Declared(_KEYPOINT_NAMES_DESCRIPTION, keypoint="name")
        ] = None
        keypoints: Annotated[
            list[KeypointRef] | None,
            Field(min_length=1),
            Declared(_KEYPOINTS_DESCRIPTION, keypoint="either"),
        ] = None
        scale_by: Annotated[
            tuple[_KeypointGroup, _KeypointGroup] | None,
            Declared(_SCALE_BY_DESCRIPTION, keypoint="either"),
        ] = None
        max_gap: Annotated[
            int, Field(ge=0), Declared(_MAX_GAP_DESCRIPTION, unit="frames")
        ] = 0
        context: Annotated[bool, Declared(_CONTEXT_DESCRIPTION)] = False

        @model_validator(mode="after")
        def _check_references(self) -> Self:
            """Reject at construction what no table could make valid.

            A raise from ``apply`` is recorded per entry and the run carries on,
            so a mistake every entry would repeat belongs here. What depends on
            the table -- an index past its last keypoint, a name list of the
            wrong length -- is checked in ``apply``.
            """
            if (self.lateral is None) == (self.heading is None):
                raise ValueError(
                    f"{_NAME}: set exactly one of lateral and heading; each "
                    "defines the body frame's horizontal axes on its own"
                )
            names = self.keypoint_names
            check_keypoint_names(names, _NAME)
            groups = self.scale_by or ([], [])
            refs: list[KeypointRef] = [
                *self.origin,
                *(self.lateral or ()),
                *(self.heading or ()),
                *(self.keypoints or []),
                *groups[0],
                *groups[1],
            ]
            check_keypoint_refs(refs, names, _NAME)
            check_keypoints_distinct(self.lateral, names, _NAME, param="lateral")
            if self.heading is not None:
                check_heading_pair(self.heading, names, _NAME)
            check_keypoints_distinct(self.origin, names, _NAME, param="origin")
            check_keypoints_distinct(self.keypoints, names, _NAME)
            if self.scale_by is not None:
                for group in self.scale_by:
                    check_keypoints_distinct(group, names, _NAME, param="scale_by")
                first, second = (
                    {keypoint_index(ref, names) for ref in group}
                    for group in self.scale_by
                )
                if first == second:
                    raise ValueError(
                        f"{_NAME}: both scale_by groups are keypoints "
                        f"{sorted(first)}, so the distance between their "
                        "centroids is zero on every row"
                    )
            return self

    def __init__(
        self,
        inputs: PoseEgocentric.Inputs = Inputs(("tracks",)),
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
        if df.empty:
            return pd.DataFrame()
        missing = [c for c in (C.frame_col, C.id_col) if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.name} needs the columns {missing}: a body frame is built "
                "per individual and frame, and gaps are measured along the frame "
                "axis"
            )
        plan = self._plan(df)

        keys = [c for c in (C.group_col, C.seq_col, C.id_col) if c in df.columns]
        df = df.sort_values([*keys, C.frame_col]).reset_index(drop=True)
        groups = df.groupby(keys, sort=False, dropna=False).ngroup()
        segment: IntArray = groups.to_numpy().astype(np.int64)
        columns = plan.keypoints.columns

        origin = _centroid(df, [columns[k] for k in plan.origin])
        first, second = plan.pair
        axes = body_axes(
            world_points(df, columns[first]) - world_points(df, columns[second]),
            plan.kind,
        )
        scale = (
            self._body_size(df, columns, plan.scale, segment)
            if plan.scale is not None
            else None
        )

        measured: dict[str, FloatArray] = {}
        for k in plan.emit:
            offset = world_points(df, columns[k]) - origin
            ego = (
                offset[:, 0] * axes.forward_x + offset[:, 1] * axes.forward_y,
                offset[:, 0] * axes.left_x + offset[:, 1] * axes.left_y,
                np.where(axes.oriented, offset[:, 2], np.nan),
            )
            for axis, values in zip("xyz", ego, strict=True):
                measured[f"ego_{axis}{k}"] = values if scale is None else values / scale
        if self.params.context:
            # arctan2 returns +pi or -pi for a backward heading depending on the
            # sign of a zero; wrapping gives one spelling for one direction.
            measured["body_yaw"] = wrap_angle(
                np.arctan2(axes.forward_y, axes.forward_x)
            )
            for axis, values in zip("xyz", origin.T, strict=True):
                measured[f"origin_{axis}"] = values

        frames = df[C.frame_col].to_numpy(dtype=np.float64)
        out = self._fill_gaps(measured, frames, segment)
        return pd.concat(
            [df[meta_columns(df)], pd.DataFrame(out, index=df.index)], axis=1
        )

    def _fill_gaps(
        self,
        measured: Mapping[str, FloatArray],
        frames: FloatArray,
        segment: IntArray,
    ) -> dict[str, np.ndarray[tuple[int], np.dtype[np.float32]]]:
        """Fill each column's short gaps, then narrow it to float32.

        A plan is reused for as long as consecutive columns share a mask,
        which the axes of one point do.
        """
        if self.params.max_gap < 1:
            return {c: v.astype(np.float32) for c, v in measured.items()}
        first_row, last_row = _segment_bounds(segment)
        out: dict[str, np.ndarray[tuple[int], np.dtype[np.float32]]] = {}
        mask: BoolArray | None = None
        plan: _GapFill | None = None
        for column, values in measured.items():
            defined = np.isfinite(values)
            if mask is None or not np.array_equal(defined, mask):
                mask = defined
                plan = _GapFill.plan(
                    defined, frames, first_row, last_row, self.params.max_gap
                )
            filled = (
                values
                if plan is None
                else plan.fill(values, angular=column == "body_yaw")
            )
            out[column] = filled.astype(np.float32)
        return out

    def _plan(self, df: pd.DataFrame) -> _Plan:
        """Resolve every keypoint reference in the parameters against *df*."""
        p = self.params
        keypoints = keypoint_columns(
            df,
            p.keypoint_names,
            self.name,
            mixed_axes_hint="A body frame needs every keypoint's height.",
            no_keypoints_hint=(
                "A body frame is spanned by keypoints; a table with only a "
                "centroid has one point per individual and no posture to express "
                "in it."
            ),
        )
        if not keypoints.three_d:
            raise ValueError(
                f"{self.name} needs 3D keypoint columns (poseX*/poseY*/poseZ*) and "
                "this table carries 2D ones only. A body frame is built about "
                "world up, which a planar table does not have; it runs on a "
                "world-frame table (mosaic_world_v1)."
            )
        n_keypoints = len(keypoints.columns)

        def indices(param: str, refs: Sequence[KeypointRef]) -> tuple[int, ...]:
            return tuple(
                resolve_keypoint_refs(
                    refs, p.keypoint_names, n_keypoints, who=f"{self.name} {param}"
                )
            )

        kind: FrameKind = "lateral" if p.lateral is not None else "heading"
        pair_refs = p.lateral if p.lateral is not None else p.heading
        assert pair_refs is not None  # the validator requires one of the two
        first, second = indices(kind, pair_refs)
        scale: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        if p.scale_by is not None:
            scale = (
                indices("scale_by", p.scale_by[0]),
                indices("scale_by", p.scale_by[1]),
            )
        return _Plan(
            keypoints=keypoints,
            origin=indices("origin", p.origin),
            kind=kind,
            pair=(first, second),
            emit=(
                tuple(range(n_keypoints))
                if p.keypoints is None
                else indices("keypoints", p.keypoints)
            ),
            scale=scale,
        )

    @staticmethod
    def _body_size(
        df: pd.DataFrame,
        columns: Sequence[tuple[str, ...]],
        groups: tuple[tuple[int, ...], tuple[int, ...]],
        segment: IntArray,
    ) -> FloatArray:
        """Each row's body size: its segment's median distance between the groups.

        A median rather than each row's own distance, because a body does not
        change size between frames while the measured distance does whenever the
        animal bends; dividing by the per-row distance would cancel exactly the
        posture this feature exists to keep. Zero or undefined is NaN, so a
        segment with no measurable size has undefined coordinates rather than
        infinite ones.
        """
        first = _centroid(df, [columns[k] for k in groups[0]])
        second = _centroid(df, [columns[k] for k in groups[1]])
        distance = pd.Series(np.linalg.norm(first - second, axis=1))
        size = distance.groupby(segment).transform("median").to_numpy(dtype=np.float64)
        return np.where(np.isfinite(size) & (size > 0), size, np.nan)
