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
- **+Y**: the subject's left. The horizontal part (``z`` set to zero) of
  ``left - right`` for the ``lateral`` pair, normalised.
- **+Z**: world up, ``(0, 0, 1)``.
- **+X**: forward, ``Y x Z``, so the frame is right-handed like the world frame.

The frame turns with the body about the vertical axis only. Pitch and roll stay
in the egocentric coordinates, which is where they belong: a rearing animal is
in a posture, and a frame tilted to follow the spine would rotate the rearing
out of the numbers that are supposed to show it. For the same reason ``ego_z``
is height above the origin along world up.

The clustering and sequence features (``global-scaler``, ``global-tsne``,
``global-kmeans``, ``arhmm``) take ``Result`` inputs only, so this is also the
step through which a world-frame table's posture reaches them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Final, Self, final

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pipeline.types import (
    DependencyLookup,
    EmitsLevel,
    InputStream,
    TrackInputs,
)
from mosaic.core.pose_columns import (
    pose_column_pairs,
    pose_column_triplets,
    resolve_keypoint_refs,
)

from .helpers import meta_columns, wrap_angle
from .registry import register_feature

KeypointRef = int | str
"""A keypoint named by its index or, given ``keypoint_names``, by its name."""

_KeypointGroup = Annotated[list[KeypointRef], Field(min_length=1)]

_VERTICAL_TOLERANCE: Final = 1e-6
"""The smallest horizontal share of the lateral vector that still fixes a yaw.

Below this the left-right vector points straight up or down, its horizontal part
is rounding error, and a direction normalised from it would be noise reported as
a heading. Relative to the vector's own length, so it does not depend on the
table's length unit.
"""

_ORIGIN_DESCRIPTION = (
    "The keypoints whose mean is the body frame's origin, by index or by name, "
    "such as both hips. Every one must be present on a row for the origin to be "
    "defined there: a mean of whichever happen to be detected moves the origin "
    "from row to row, and that movement would read as a change of posture."
)

_LATERAL_DESCRIPTION = (
    "The (left, right) keypoint pair, by index or by name. The horizontal part of "
    "left minus right is the subject's left, the body frame's +Y. Left and right "
    "are the subject's own, not the camera's."
)

_KEYPOINT_NAMES_DESCRIPTION = (
    "The keypoint names in keypoint index order, so that origin, lateral, "
    "keypoints and scale_by may name keypoints rather than number them. It must "
    "name every keypoint the table numbers, from 0 to the highest."
)

_KEYPOINTS_DESCRIPTION = (
    "Which keypoints to express in the body frame and emit, by index or by name. "
    "Unset, every keypoint the table carries on all three axes."
)

_SCALE_BY_DESCRIPTION = (
    "Two keypoint groups, such as the hips and the shoulders. Set, the egocentric "
    "coordinates are divided by the median distance between the two groups' "
    "centroids, taken per individual and per sequence, so that posture compares "
    "across body sizes. Unset, they stay in the table's length unit."
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
    """The parameters' keypoint references, resolved against one table."""

    origin: tuple[int, ...]
    left: int
    right: int
    emit: tuple[int, ...]
    scale: tuple[tuple[int, ...], tuple[int, ...]] | None


def _xyz(frame: pd.DataFrame, triplet: tuple[str, str, str]) -> np.ndarray:
    """One keypoint's ``(n_rows, 3)`` world coordinates, as float64.

    A row missing any of the three axes is missing all of them. A point with no
    height is not a point on the floor, and keeping its other two axes would
    give it a forward and a leftward coordinate with a NaN height beside them.
    """
    points = np.column_stack(
        [frame[column].to_numpy(dtype=np.float64) for column in triplet]
    )
    points[~np.isfinite(points).all(axis=1)] = np.nan
    return points


def _centroid(
    frame: pd.DataFrame, triplets: Sequence[tuple[str, str, str]]
) -> np.ndarray:
    """The mean of *triplets*, NaN on any row missing one of them.

    A plain mean on purpose, not ``nanmean``. A centroid of whichever keypoints
    were detected is a different point on every row that detects a different
    set, and the difference would arrive downstream as movement of the body.
    """
    total = np.zeros((len(frame), 3), dtype=np.float64)
    for triplet in triplets:
        total += _xyz(frame, triplet)
    return total / len(triplets)


def _segment_bounds(segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The first and last row of each row's segment, for rows sorted by segment."""
    n = segment.shape[0]
    is_start = np.ones(n, dtype=bool)
    is_start[1:] = segment[1:] != segment[:-1]
    starts = np.flatnonzero(is_start)
    ends = np.append(starts[1:] - 1, n - 1)
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

    targets: np.ndarray
    before: np.ndarray
    after: np.ndarray
    weight: np.ndarray

    @classmethod
    def plan(
        cls,
        defined: np.ndarray,
        frames: np.ndarray,
        first_row: np.ndarray,
        last_row: np.ndarray,
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
        rows: np.ndarray = np.arange(n, dtype=np.intp)
        prev = np.maximum.accumulate(np.where(defined, rows, -1))
        nxt = np.minimum.accumulate(np.where(defined, rows, n)[::-1])[::-1]
        targets = np.flatnonzero(~defined & (prev >= first_row) & (nxt <= last_row))
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

    def fill(self, values: np.ndarray, *, angular: bool = False) -> np.ndarray:
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

    - ``ego_x{k}``, ``ego_y{k}``, ``ego_z{k}`` for each emitted keypoint ``k``:
      forward, leftward and upward of the origin, in the table's length unit
      (metres for a world table), or as a fraction of the body size under
      ``scale_by``. This is the posture.
    - ``body_yaw``: the direction of +X in the world XY plane, radians
      counterclockwise from world +X seen from above, in ``[-pi, pi)``.
    - ``origin_x``, ``origin_y``, ``origin_z``: where the origin is in the world.

    ``body_yaw`` and ``origin_*`` are context -- where the body frame sits and
    which way it faces -- and not posture. A model meant to see posture alone
    is not fitted on them.

    Each value is NaN exactly where something it is computed from is undefined,
    never zero. The egocentric coordinates need the whole frame: the origin,
    the lateral pair, a lateral vector that is neither zero nor vertical, the
    keypoint itself and, under ``scale_by``, a measurable body size.
    ``body_yaw`` needs only the lateral pair, and ``origin_*`` only the origin
    keypoints. A keypoint missing any one axis is missing. ``max_gap`` then
    fills short interior runs.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.pose_egocentric.PoseEgocentric.Params`.
    """

    category = "per-frame"
    name = "pose-egocentric"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    # scale_by takes a median over each individual's rows in one sequence. Handed
    # a neighbour's overlap rows, that median would be partly the neighbour's.
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    track_spaces: ClassVar[frozenset[str]] = frozenset({"world_m"})

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        origin: Annotated[
            list[KeypointRef], Field(min_length=1), Declared(_ORIGIN_DESCRIPTION)
        ]
        lateral: Annotated[
            tuple[KeypointRef, KeypointRef], Declared(_LATERAL_DESCRIPTION)
        ]
        keypoint_names: Annotated[
            list[str] | None, Declared(_KEYPOINT_NAMES_DESCRIPTION)
        ] = None
        keypoints: Annotated[
            list[KeypointRef] | None,
            Field(min_length=1),
            Declared(_KEYPOINTS_DESCRIPTION),
        ] = None
        scale_by: Annotated[
            tuple[_KeypointGroup, _KeypointGroup] | None,
            Declared(_SCALE_BY_DESCRIPTION),
        ] = None
        max_gap: Annotated[
            int, Field(ge=0), Declared(_MAX_GAP_DESCRIPTION, unit="frames")
        ] = 0

        @model_validator(mode="after")
        def _check_references(self) -> Self:
            left, right = self.lateral
            if left == right:
                raise ValueError(
                    f"lateral names {left!r} twice; it is a (left, right) pair of "
                    "two keypoints"
                )
            names = self.keypoint_names or []
            repeated = sorted({name for name in names if names.count(name) > 1})
            if repeated:
                raise ValueError(
                    f"keypoint_names lists {repeated} more than once, so a "
                    "reference by name could mean either keypoint"
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
        present = self._keypoint_triplets(df)
        plan = self._resolve(present)

        keys = [c for c in (C.group_col, C.seq_col, C.id_col) if c in df.columns]
        df = df.sort_values([*keys, C.frame_col]).reset_index(drop=True)
        segment = df.groupby(keys, sort=False, dropna=False).ngroup().to_numpy()

        origin = _centroid(df, [present[k] for k in plan.origin])
        lateral = _xyz(df, present[plan.left]) - _xyz(df, present[plan.right])
        horizontal = np.hypot(lateral[:, 0], lateral[:, 1])
        length = np.linalg.norm(lateral, axis=1)
        # NaN compares false, so a row missing either lateral keypoint is
        # undefined here too.
        oriented = horizontal > _VERTICAL_TOLERANCE * length
        with np.errstate(invalid="ignore", divide="ignore"):
            left_x = np.where(oriented, lateral[:, 0] / horizontal, np.nan)
            left_y = np.where(oriented, lateral[:, 1] / horizontal, np.nan)
        # Forward is Y x Z = (left_y, -left_x, 0).
        forward_x, forward_y = left_y, -left_x

        scale = (
            self._body_size(df, present, plan.scale, segment)
            if plan.scale is not None
            else None
        )

        measured: dict[str, np.ndarray] = {}
        for k in plan.emit:
            offset = _xyz(df, present[k]) - origin
            ego = (
                offset[:, 0] * forward_x + offset[:, 1] * forward_y,
                offset[:, 0] * left_x + offset[:, 1] * left_y,
                np.where(oriented, offset[:, 2], np.nan),
            )
            for axis, values in zip("xyz", ego, strict=True):
                measured[f"ego_{axis}{k}"] = values if scale is None else values / scale
        # arctan2 returns +pi or -pi for a backward heading depending on the sign
        # of a zero; wrapping gives one spelling for one direction.
        measured["body_yaw"] = wrap_angle(np.arctan2(forward_y, forward_x))
        for axis, values in zip("xyz", origin.T, strict=True):
            measured[f"origin_{axis}"] = values

        frames = df[C.frame_col].to_numpy(dtype=np.float64)
        out = self._fill_gaps(measured, frames, segment)
        return pd.concat(
            [df[meta_columns(df)], pd.DataFrame(out, index=df.index)], axis=1
        )

    def _fill_gaps(
        self,
        measured: Mapping[str, np.ndarray],
        frames: np.ndarray,
        segment: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Fill each column's short gaps, then narrow it to float32.

        A plan is reused for as long as consecutive columns share a mask,
        which the axes of one point do.
        """
        if self.params.max_gap < 1:
            return {c: v.astype(np.float32) for c, v in measured.items()}
        first_row, last_row = _segment_bounds(segment)
        out: dict[str, np.ndarray] = {}
        mask: np.ndarray | None = None
        plan: _GapFill | None = None
        for column, values in measured.items():
            defined = np.isfinite(values)
            if mask is None or not np.array_equal(defined, mask):
                mask = defined
                plan = _GapFill.plan(
                    defined, frames, first_row, last_row, self.params.max_gap
                )
            if plan is not None:
                values = plan.fill(values, angular=column == "body_yaw")
            out[column] = values.astype(np.float32)
        return out

    def _keypoint_triplets(self, df: pd.DataFrame) -> dict[int, tuple[str, str, str]]:
        """The table's numbered 3D keypoints, by keypoint index.

        Indexed by the number in the column name, so keypoint ``k`` is
        ``poseX{k}`` on input and ``ego_x{k}`` on output whatever else the
        table carries. A positional index would rename every keypoint after a
        gap in the numbering.
        """
        columns = [str(c) for c in df.columns]
        every = pose_column_triplets(columns)
        triplets = {
            int(x[len("poseX") :]): (x, y, z)
            for x, y, z in every
            if x[len("poseX") :].isdigit()
        }
        if triplets:
            return triplets
        if every:
            raise ValueError(
                f"{self.name} needs numbered 3D keypoint columns (poseX0, poseY0, "
                f"poseZ0, ...) and this table names its keypoints instead "
                f"({[x for x, _, _ in every]}). Keypoints are referred to and "
                "emitted by number, so a named set has nothing to resolve against."
            )
        if pose_column_pairs(columns):
            raise ValueError(
                f"{self.name} needs 3D keypoint columns (poseX*/poseY*/poseZ*) and "
                "this table carries 2D ones only. A body frame is built about "
                "world up, which a planar table does not have; it runs on a "
                "world-frame table (mosaic_world_v1)."
            )
        raise ValueError(
            f"{self.name} needs 3D keypoint columns (poseX*/poseY*/poseZ*) and this "
            "table carries none. A body frame is spanned by keypoints; a table "
            "with only a centroid has one point per individual and no posture "
            "to express in it."
        )

    def _resolve(self, present: Mapping[int, tuple[str, str, str]]) -> _Plan:
        """Resolve every keypoint reference in the parameters against *present*."""
        p = self.params
        n_keypoints = max(present) + 1
        names = p.keypoint_names
        if names is not None and len(names) != n_keypoints:
            raise ValueError(
                f"{self.name}: keypoint_names lists {len(names)} names and this "
                f"table numbers its keypoints 0..{n_keypoints - 1}. The names are "
                "read in keypoint index order, so a list of another length names "
                "keypoints of a different skeleton."
            )

        def indices(param: str, refs: Sequence[KeypointRef]) -> tuple[int, ...]:
            who = f"{self.name} {param}"
            resolved = resolve_keypoint_refs(refs, names, n_keypoints, who=who)
            absent = sorted({k for k in resolved if k not in present})
            if absent:
                raise ValueError(
                    f"{who}: the table carries no poseX/poseY/poseZ triplet for "
                    f"keypoint(s) {absent}; it carries {sorted(present)}"
                )
            repeated = sorted({k for k in resolved if resolved.count(k) > 1})
            if repeated:
                raise ValueError(f"{who} names keypoint(s) {repeated} more than once")
            return tuple(resolved)

        left, right = indices("lateral", p.lateral)
        scale: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        if p.scale_by is not None:
            scale = (
                indices("scale_by", p.scale_by[0]),
                indices("scale_by", p.scale_by[1]),
            )
            if set(scale[0]) == set(scale[1]):
                raise ValueError(
                    f"{self.name} scale_by: both groups are keypoints "
                    f"{sorted(scale[0])}, so the distance between their centroids "
                    "is zero on every row"
                )
        return _Plan(
            origin=indices("origin", p.origin),
            left=left,
            right=right,
            emit=(
                tuple(sorted(present))
                if p.keypoints is None
                else indices("keypoints", p.keypoints)
            ),
            scale=scale,
        )

    @staticmethod
    def _body_size(
        df: pd.DataFrame,
        present: Mapping[int, tuple[str, str, str]],
        groups: tuple[tuple[int, ...], tuple[int, ...]],
        segment: np.ndarray,
    ) -> np.ndarray:
        """Each row's body size: its segment's median distance between the groups.

        A median rather than each row's own distance, because a body does not
        change size between frames while the measured distance does whenever the
        animal bends; dividing by the per-row distance would cancel exactly the
        posture this feature exists to keep. Zero or undefined is NaN, so a
        segment with no measurable size has undefined coordinates rather than
        infinite ones.
        """
        first = _centroid(df, [present[k] for k in groups[0]])
        second = _centroid(df, [present[k] for k in groups[1]])
        distance = pd.Series(np.linalg.norm(first - second, axis=1))
        size = distance.groupby(segment).transform("median").to_numpy(dtype=np.float64)
        return np.where(np.isfinite(size) & (size > 0), size, np.nan)
