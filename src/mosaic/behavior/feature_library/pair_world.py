"""Pair geometry in the world frame: where each partner is, seen from each focal.

``pair-egocentric`` and ``pair-position`` measure a pair from one planar heading
angle. In a world-frame table the partner may be above or below, and the focal
may be pitched, so this feature measures in three dimensions.

**The focal frame** is yaw-only, as the ``heading`` frame of ``pose-egocentric``
is: a rotation about world up that turns the horizontal part of the focal's
heading (``front`` minus ``rear``) onto +X, with +Y the focal's left. Pitch stays
measurable in it. The world frame is right-handed with +Z up, so a positive
``bearing`` is a partner on the focal's left.

**Undefined geometry is NaN, never 0.** A heading's yaw is undefined where its
horizontal part is no more than
:data:`~mosaic.behavior.feature_library.world_frame.VERTICAL_TOLERANCE` of its
length, which makes ``bearing`` and the relative yaw NaN while ``pitch`` stays
finite at plus or minus pi/2. A partner straight above or below has no bearing,
and a partner at the focal's own position has no bearing, elevation or approach
direction. A zero-length heading has no pitch and no angle to another heading.

**Gaps.** Nothing is interpolated. A point missing any axis is missing. The
derivatives are central differences against the ``time`` column between
consecutive frames only, so a derivative at either end of a track or beside a
missing frame is NaN.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Final, Self, final

import numpy as np
import pandas as pd
from pydantic import model_validator

from mosaic.core.individuals import id_arenas, individual_pairs, refuse_repeated_rows
from mosaic.core.keypoint_refs import (
    check_heading_pair,
    check_keypoint_refs,
    resolve_keypoint_refs,
)
from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.column_contract import (
    POSE_X,
    POSE_Y,
    Column,
    Prefix,
    Provides,
    Requirement,
    kept,
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
    check_keypoint_names,
    keypoint_columns,
    row_norm,
)
from .helpers import wrap_angle
from .registry import register_feature
from .world_frame import KeypointRef, body_axes, world_points

__all__ = ["PAIR_WORLD_COLUMNS", "PAIR_WORLD_PROVIDES", "PairWorld"]

_NAME = "pair-world"

PAIR_WORLD_COLUMNS: Final[tuple[str, ...]] = (
    "dist",
    "dist_xy",
    "dz",
    "bearing",
    "elevation",
    "rel_yaw_cos",
    "rel_yaw_sin",
    "rel_pitch",
    "heading_angle",
    "pitch",
    "speed",
    "approach_speed",
    "closing_speed",
)
"""The measurements of one pair row, in output order."""

PAIR_WORLD_PROVIDES: Final = Provides(
    adds=(
        Column(C.frame_col),
        Column("perspective"),
        Column("id1"),
        Column("id2"),
        Column(C.seq_col),
        *(Column(name) for name in PAIR_WORLD_COLUMNS),
    ),
    conditional=kept(C.group_col),
    # One distance, one angle between the headings and cos(-a) = cos(a) on both
    # rows. The others swap roles or flip sign.
    symmetric=(
        Column("closing_speed"),
        Column("dist"),
        Column("dist_xy"),
        Column("heading_angle"),
        Column("rel_yaw_cos"),
    ),
)
"""The output of ``pair-world``: one row per frame, pair and perspective."""

_FRONT_DESCRIPTION = (
    "The keypoint at the front of the heading, by index or by name, such as the "
    "head. The focal's heading is front minus rear."
)

_REAR_DESCRIPTION = (
    "The keypoint at the rear of the heading, by index or by name, such as the "
    "pectoral point. It must differ from front."
)

_POSITION_DESCRIPTION = (
    "The keypoint each individual's position is read from, by index or by name. "
    "Unset, the body center X/Y/Z."
)

_KEYPOINT_NAMES_DESCRIPTION = (
    "A name for each keypoint, in the table's keypoint order, so that front, rear "
    "and position may name keypoints rather than number them. Its length must "
    "equal the table's keypoint count."
)


@dataclass(frozen=True, slots=True)
class _Plan:
    """The parameters' keypoint references, resolved against one table.

    Attributes:
        front: The coordinate columns of the heading's front keypoint.
        rear: Those of its rear keypoint.
        position: Those of the point each individual's position is read from.
    """

    front: tuple[str, ...]
    rear: tuple[str, ...]
    position: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Track:
    """One individual's rows of one entry, in frame order, as float64.

    Attributes:
        frames: The frame numbers.
        times: The ``time`` of each row.
        position: ``(n, 3)`` positions.
        heading: ``(n, 3)`` front-minus-rear vectors.
        velocity: ``(n, 3)`` velocities, NaN where no central difference spans
            the row.
    """

    frames: FloatArray
    times: FloatArray
    position: FloatArray
    heading: FloatArray
    velocity: FloatArray

    def take(self, rows: IntArray) -> _Track:
        """The track at *rows*."""
        return _Track(
            frames=self.frames[rows],
            times=self.times[rows],
            position=self.position[rows],
            heading=self.heading[rows],
            velocity=self.velocity[rows],
        )


def _central_difference(
    values: FloatArray, frames: FloatArray, times: FloatArray
) -> FloatArray:
    """The derivative of *values* against *times*, by central difference.

    Defined on a row whose frames before and after are both present, and NaN
    elsewhere: at either end and beside a missing frame.

    Args:
        values: ``(n,)`` or ``(n, k)`` values, in frame order.
        frames: The frame of each row.
        times: The time of each row.
    """
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if len(frames) < 3:
        return out
    spanned: BoolArray = (frames[1:-1] - frames[:-2] == 1) & (
        frames[2:] - frames[1:-1] == 1
    )
    step = times[2:] - times[:-2]
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = (values[2:] - values[:-2]) / (
            step[:, None] if values.ndim == 2 else step
        )
    mask = spanned[:, None] if values.ndim == 2 else spanned
    out[1:-1] = np.where(mask, rate, np.nan)
    return out


def _pitch(heading: FloatArray, length: FloatArray) -> FloatArray:
    """Each heading's angle above the horizontal, NaN where it has no length."""
    angle = np.arctan2(heading[:, 2], np.hypot(heading[:, 0], heading[:, 1]))
    return np.where(length > 0, angle, np.nan)


def _measure(
    focal: _Track, partner: _Track, closing: FloatArray
) -> dict[str, FloatArray]:
    """The measurements of *partner* from *focal*, on their joined frames.

    Args:
        focal: The focal's rows on the joined frames.
        partner: The partner's rows on the same frames.
        closing: The pair's closing speed on those frames.
    """
    offset = partner.position - focal.position
    dist = row_norm(offset)
    dist_xy = np.hypot(offset[:, 0], offset[:, 1])
    axes = body_axes(focal.heading, "heading")
    partner_axes = body_axes(partner.heading, "heading")
    ahead = offset[:, 0] * axes.forward_x + offset[:, 1] * axes.forward_y
    leftward = offset[:, 0] * axes.left_x + offset[:, 1] * axes.left_y
    focal_length = row_norm(focal.heading)
    partner_length = row_norm(partner.heading)
    focal_pitch = _pitch(focal.heading, focal_length)
    turn = np.cross(focal.heading, partner.heading)
    alignment = np.einsum("ij,ij->i", focal.heading, partner.heading)
    with np.errstate(invalid="ignore", divide="ignore"):
        approach = np.einsum("ij,ij->i", focal.velocity, offset) / dist
    # arctan2 returns +pi or -pi for a partner straight behind depending on the
    # sign of a zero; wrapping gives one spelling for one direction.
    bearing = wrap_angle(np.arctan2(leftward, ahead))
    return {
        "dist": dist,
        "dist_xy": dist_xy,
        "dz": offset[:, 2],
        "bearing": np.where(dist_xy > 0, bearing, np.nan),
        "elevation": np.where(dist > 0, np.arctan2(offset[:, 2], dist_xy), np.nan),
        "rel_yaw_cos": axes.forward_x * partner_axes.forward_x
        + axes.forward_y * partner_axes.forward_y,
        "rel_yaw_sin": axes.forward_x * partner_axes.forward_y
        - axes.forward_y * partner_axes.forward_x,
        "rel_pitch": _pitch(partner.heading, partner_length) - focal_pitch,
        "heading_angle": np.where(
            (focal_length > 0) & (partner_length > 0),
            np.arctan2(row_norm(turn), alignment),
            np.nan,
        ),
        "pitch": focal_pitch,
        "speed": row_norm(focal.velocity),
        "approach_speed": np.where(dist > 0, approach, np.nan),
        "closing_speed": closing,
    }


@final
@register_feature
class PairWorld:
    """Per-frame geometry of every pair of individuals, in the world frame.

    One row per frame, pair and perspective: perspective 0 measures ``id2``
    from ``id1`` and perspective 1 the reverse, ``id1`` being the focal. A pair
    has rows on each frame where both individuals have one. Pairs are formed
    within one arena. The columns, float32, lengths in meters and angles in
    radians:

    - ``dist``, ``dist_xy``: the 3D and the horizontal distance to the partner.
    - ``dz``: the partner's height minus the focal's.
    - ``bearing``: the partner's azimuth in the focal frame, in ``[-pi, pi)``,
      positive to the focal's left.
    - ``elevation``: the partner's angle above the focal's horizontal plane.
    - ``rel_yaw_cos``, ``rel_yaw_sin``: the partner's heading yaw relative to
      the focal's.
    - ``rel_pitch``: the partner's pitch minus the focal's.
    - ``heading_angle``: the 3D angle between the two headings, in ``[0, pi]``.
    - ``pitch``: the focal's heading pitch.
    - ``speed``: the focal's 3D speed, in m/s.
    - ``approach_speed``: the focal's velocity along the direction to the
      partner, in m/s.
    - ``closing_speed``: the rate at which ``dist`` shrinks, in m/s.

    ``dist``, ``dist_xy``, ``heading_angle``, ``rel_yaw_cos`` and
    ``closing_speed`` hold one value on both rows of a pair.

    Runs on world-frame tables (``mosaic_world_v1``) only. Field documentation
    is on :class:`~mosaic.behavior.feature_library.pair_world.PairWorld.Params`.
    """

    category = "per-frame"
    name = _NAME
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "arena-aware"
    # Rows are paired within one sequence, so a neighbor's overlap rows would
    # form pairs of their own rather than extend this entry's derivatives.
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "pair"
    # Velocities and the closing speed difference each individual's rows.
    follows_ids: bool = True
    track_spaces: ClassVar[frozenset[TrackSpace]] = frozenset({"world_m"})
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.frame_col),
        Column(C.time_col),
        Column(C.id_col),
        Column(C.seq_col),
        Column(C.x_col),
        Column(C.y_col),
        Column("Z"),
        POSE_X,
        POSE_Y,
        Prefix("poseZ"),
    )
    provides: ClassVar[Provides] = PAIR_WORLD_PROVIDES

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        """Pair-world parameters."""

        front: Annotated[KeypointRef, Declared(_FRONT_DESCRIPTION, keypoint="either")]
        rear: Annotated[KeypointRef, Declared(_REAR_DESCRIPTION, keypoint="either")]
        position: Annotated[
            KeypointRef | None, Declared(_POSITION_DESCRIPTION, keypoint="either")
        ] = None
        keypoint_names: Annotated[
            list[str] | None, Declared(_KEYPOINT_NAMES_DESCRIPTION, keypoint="name")
        ] = None

        @model_validator(mode="after")
        def _check_references(self) -> Self:
            """Reject at construction what no table could make valid.

            What depends on the table, an index past its last keypoint or a
            name list of the wrong length, is checked in ``apply``.
            """
            names = self.keypoint_names
            check_keypoint_names(names, _NAME)
            refs: list[KeypointRef] = [self.front, self.rear]
            if self.position is not None:
                refs.append(self.position)
            check_keypoint_refs(refs, names, _NAME)
            check_heading_pair((self.front, self.rear), names, _NAME)
            return self

    def __init__(
        self,
        inputs: PairWorld.Inputs = Inputs(("tracks",)),
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
        needed = (C.frame_col, C.time_col, C.id_col, C.seq_col)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.name} needs the columns {missing}: pairs are joined on the "
                "frame within one sequence, and speeds are taken against time"
            )
        if df.empty:
            return _empty()
        plan = self._plan(df)
        keys = [c for c in (C.group_col, C.seq_col) if c in df.columns]
        arenas = id_arenas(df)
        out: list[pd.DataFrame] = []
        for key, entry in df.groupby(keys, sort=True, dropna=False):
            refuse_repeated_rows(entry, self.name)
            tracks = {
                identity: _track(rows, plan)
                for identity, rows in entry.groupby(C.id_col, sort=True)
            }
            meta = dict(zip(keys, key, strict=True))
            for first, second in individual_pairs(list(tracks), arenas):
                out.extend(_pair_rows(first, second, tracks, meta))
        if not out:
            return _empty()
        rows = pd.concat(out, ignore_index=True)
        rows = rows.sort_values(
            ["id1", "id2", "perspective", C.frame_col], kind="stable"
        )
        return rows.reset_index(drop=True)

    def _plan(self, df: pd.DataFrame) -> _Plan:
        """Resolve every keypoint reference in the parameters against *df*."""
        p = self.params
        keypoints = keypoint_columns(
            df,
            p.keypoint_names,
            self.name,
            mixed_axes_hint="A heading in the world frame needs its keypoints' heights.",
        )
        if not keypoints.three_d:
            raise ValueError(
                f"{self.name} needs 3D keypoint columns (poseX*/poseY*/poseZ*) and "
                "this table carries 2D ones only. It measures pairs in the world "
                "frame about world up, which a planar table does not have; it runs "
                "on a world-frame table (mosaic_world_v1)."
            )
        n_keypoints = len(keypoints.columns)

        def column(param: str, ref: KeypointRef) -> tuple[str, ...]:
            [index] = resolve_keypoint_refs(
                [ref], p.keypoint_names, n_keypoints, who=f"{self.name} {param}"
            )
            return keypoints.columns[index]

        return _Plan(
            front=column("front", p.front),
            rear=column("rear", p.rear),
            position=(
                (C.x_col, C.y_col, "Z")
                if p.position is None
                else column("position", p.position)
            ),
        )


def _empty() -> pd.DataFrame:
    """No pair rows, with the declared columns."""
    columns = [ref.name for ref in PAIR_WORLD_PROVIDES.adds if isinstance(ref, Column)]
    return pd.DataFrame(columns=columns)


def _track(rows: pd.DataFrame, plan: _Plan) -> _Track:
    """One individual's rows as a :class:`_Track`, its velocity included."""
    rows = rows.sort_values(C.frame_col, kind="stable")
    frames = rows[C.frame_col].to_numpy(dtype=np.float64)
    times = rows[C.time_col].to_numpy(dtype=np.float64)
    position = world_points(rows, plan.position)
    return _Track(
        frames=frames,
        times=times,
        position=position,
        heading=world_points(rows, plan.front) - world_points(rows, plan.rear),
        velocity=_central_difference(position, frames, times),
    )


def _pair_rows[T: Hashable](
    first: T,
    second: T,
    tracks: Mapping[T, _Track],
    meta: Mapping[str, object],
) -> Sequence[pd.DataFrame]:
    """Both perspectives' rows of one pair, on the frames both individuals have."""
    a, b = tracks[first], tracks[second]
    _, rows_a, rows_b = np.intersect1d(
        a.frames, b.frames, assume_unique=True, return_indices=True
    )
    if rows_a.size == 0:
        return ()
    a = a.take(rows_a.astype(np.int64))
    b = b.take(rows_b.astype(np.int64))
    dist = row_norm(b.position - a.position)
    # The time of one individual for both rows keeps the column symmetric.
    closing = -_central_difference(dist, a.frames, a.times)
    frames = a.frames.astype(np.int64)
    return tuple(
        pd.DataFrame(
            {
                C.frame_col: frames,
                "perspective": perspective,
                "id1": [focal] * len(frames),
                "id2": [partner] * len(frames),
                C.seq_col: meta.get(C.seq_col),
                **{
                    name: values.astype(np.float32)
                    for name, values in _measure(
                        focal_track, partner_track, closing
                    ).items()
                },
                **({C.group_col: meta[C.group_col]} if C.group_col in meta else {}),
            }
        )
        for perspective, focal, partner, focal_track, partner_track in (
            (0, first, second, a, b),
            (1, second, first, b, a),
        )
    )
