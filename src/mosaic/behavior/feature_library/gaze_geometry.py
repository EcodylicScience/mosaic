"""Gaze and head geometry on world-frame tracks: ``ray-target`` and ``head-eye``.

Both features read what a ``mosaic_world_v1`` table may carry beside its body
centre -- a gaze ray (``gaze*``), a head pose (``head*``) and 3D keypoints -- and
compute in metres and radians in the table's own world frame. Neither has a
planar meaning: an angle between a ray and a target is only an angle once both
are in one three-dimensional frame, and a head yaw read off two axes of three is
the projection of a rotation rather than the rotation. So both declare
``track_spaces = {"world_m"}`` and nothing else, and ``run_feature`` refuses
them a pixel or centimetre table before any work starts.

**What a column means is the schema's, not this module's.** The gaze ray is an
origin ``gazeX/Y/Z`` and a direction ``gazeDX/DY/DZ``; the head pose is a
head-to-world unit quaternion ``headQW/QX/QY/QZ``, scalar first, with the head
frame +X forward, +Y to the subject's left and +Z up through the crown (see
:data:`~mosaic.core.schema.GAZE_COLUMNS` and
:data:`~mosaic.core.schema.HEAD_COLUMNS`). Both are normalised here regardless,
because a producer's unit vector is unit to within its own float precision and
a resampled one is not unit at all; a direction or quaternion of zero or
non-finite length is NaN, never a default orientation.

**An angular speed is a one-frame difference or nothing.** Each is the angle
between this frame's direction (or rotation) and the previous frame's, times the
frame rate, and it is NaN on the first row of an individual and on any row whose
previous row is not the frame immediately before. Bridging a gap would divide a
multi-frame turn by one frame's time and report it as a fast one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, ClassVar, Final, Self, final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, field_validator, model_validator

from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pipeline.types import (
    DependencyLookup,
    EmitsLevel,
    InputStream,
    TrackInputs,
)
from mosaic.core.pose_columns import pose_column_triplets, resolve_keypoint_refs

from .helpers import ensure_columns, meta_columns
from .registry import register_feature

__all__ = ["HeadEye", "RayTarget"]

type _Floats = npt.NDArray[np.float64]
type _Mask = npt.NDArray[np.bool_]

_IDENTIFIER: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WORLD_ONLY: Final[frozenset[str]] = frozenset({"world_m"})
_QUATERNION_COLUMNS: Final = ("headQW", "headQX", "headQY", "headQZ")
_GAZE_DIRECTION_COLUMNS: Final = ("gazeDX", "gazeDY", "gazeDZ")


# --- shared geometry ----------------------------------------------------------


def _sorted_by_id_and_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """The identity columns and *columns* of *df*, by individual, then frame.

    Only what the feature reads goes through the sort, so a table carrying many
    keypoints is not copied whole to measure a few of them. The index is fresh
    and positional.
    """
    keep = list(dict.fromkeys([C.id_col, C.frame_col, *meta_columns(df), *columns]))
    return df.loc[:, keep].sort_values([C.id_col, C.frame_col]).reset_index(drop=True)


def _stack(df: pd.DataFrame, columns: Sequence[str]) -> _Floats:
    """The named columns of *df* as an ``(n, len(columns))`` float64 array."""
    return df.loc[:, list(columns)].to_numpy(dtype=np.float64)


def _normalised(vectors: _Floats) -> _Floats:
    """Each row scaled to unit length; a zero or non-finite row becomes NaN."""
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    usable = np.isfinite(norm) & (norm > 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(usable, vectors / norm, np.nan)


def _angle_between(a: _Floats, b: _Floats) -> _Floats:
    """The unsigned angle between paired rows of *a* and *b*, in ``[0, pi]``.

    ``atan2(|a x b|, a . b)`` rather than ``arccos`` of the dot product: the
    latter loses every digit near 0 and pi, which is exactly where a fixation and
    a frame-to-frame direction change sit. Neither input needs to be unit length.
    """
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    dot = np.einsum("ij,ij->i", a, b)
    return np.arctan2(cross, dot)


def _same_individual_as_previous(df: pd.DataFrame) -> _Mask:
    """Whether each row after the first has the previous row's id, ``(n-1,)``.

    A row without an id is nobody's, so it continues nothing and nothing
    continues it: two unidentified detections in consecutive frames are not one
    individual turning. *df* must already be sorted by individual, then frame.
    """
    ids = pd.factorize(df[C.id_col])[0]  # -1 for a missing id
    return (ids[1:] == ids[:-1]) & (ids[1:] >= 0)


def _follows_previous_frame(df: pd.DataFrame) -> _Mask:
    """Whether each row's previous row is the same individual, one frame earlier.

    *df* must already be sorted by individual, then frame.
    """
    frames = df[C.frame_col].to_numpy(dtype=np.float64)
    follows = np.zeros(len(df), dtype=bool)
    follows[1:] = _same_individual_as_previous(df) & (frames[1:] - frames[:-1] == 1.0)
    return follows


def _frames_per_second(df: pd.DataFrame, fps: float | None, who: str) -> float:
    """The frame rate: *fps* when given, else measured off the ``time`` column.

    Measured as the reciprocal of the median of ``dtime / dframe`` over
    consecutive rows of each individual, which is the median time step on a
    contiguous table and stays right where frames are skipped. The median rather
    than the mean so that one jittered or paused timestamp does not move it.
    *df* must already be sorted by individual, then frame.
    """
    if fps is not None:
        return fps
    if C.time_col not in df.columns:
        raise ValueError(
            f"{who} needs a frame rate for its angular speeds and this table has no "
            f"{C.time_col!r} column to measure one from. Pass fps=."
        )
    frames = df[C.frame_col].to_numpy(dtype=np.float64)
    times = df[C.time_col].to_numpy(dtype=np.float64)
    d_frame = frames[1:] - frames[:-1]
    d_time = times[1:] - times[:-1]
    usable = (
        _same_individual_as_previous(df)
        & (d_frame > 0)
        & np.isfinite(d_time)
        & (d_time > 0)
    )
    if not usable.any():
        raise ValueError(
            f"{who} needs a frame rate for its angular speeds and the "
            f"{C.time_col!r} column gives none: no two consecutive rows of one "
            f"individual have a finite, increasing time. Pass fps=."
        )
    seconds_per_frame = float(np.median(d_time[usable] / d_frame[usable]))
    return 1.0 / seconds_per_frame


def _step_rates(
    df: pd.DataFrame, steps: Sequence[_Floats], fps: float | None, who: str
) -> list[_Floats]:
    """Per-row rates from per-step changes, in the changes' unit per second.

    Each of *steps* holds ``n - 1`` changes, the ``i``-th from row ``i`` to row
    ``i + 1``. A rate is that change times the frame rate on a row whose previous
    row is the same individual one frame earlier, and NaN on every other row. The
    frame rate is asked for only when some row has such a predecessor, since
    without one every rate is NaN whatever it is. *df* must already be sorted by
    individual, then frame.
    """
    follows = _follows_previous_frame(df)
    rates = [np.full(len(df), np.nan) for _ in steps]
    if not follows.any():
        return rates
    rate = _frames_per_second(df, fps, who)
    for out, step in zip(rates, steps, strict=True):
        out[1:] = np.where(follows[1:], step * rate, np.nan)
    return rates


def _with_meta(df: pd.DataFrame, measurements: dict[str, _Floats]) -> pd.DataFrame:
    """The identity columns of *df* beside *measurements*, cast to float32."""
    meta = df.loc[:, meta_columns(df)].reset_index(drop=True)
    values = pd.DataFrame(
        {name: column.astype(np.float32) for name, column in measurements.items()}
    )
    return pd.concat([meta, values], axis=1)


def _quaternion_matrices(quaternions: _Floats) -> _Floats:
    """Rotation matrices ``(n, 3, 3)`` from unit scalar-first quaternions ``(n, 4)``.

    A NaN quaternion -- what :func:`_normalised` makes of a zero or non-finite
    one -- gives an all-NaN matrix. The matrix maps a head-frame vector to the
    world frame, so its columns are the head's +X, +Y and +Z axes expressed in
    world coordinates.
    """
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    return np.stack(
        [
            np.stack(
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - w * z),
                    2.0 * (x * z + w * y),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - w * x),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    2.0 * (x * z - w * y),
                    2.0 * (y * z + w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )


def _quaternion_step_angles(quaternions: _Floats) -> _Floats:
    """The rotation angle from each row's quaternion to the next row's, ``(n-1,)``.

    The angle of ``conj(q0) * q1``, as ``2 * atan2(|vector|, |scalar|)``. The
    absolute value of the scalar part is what makes ``q`` and ``-q`` -- one
    rotation, two spellings, and producers flip between them freely -- read as no
    rotation at all rather than as a full turn. *quaternions* must be unit length
    or NaN.
    """
    w0, w1 = quaternions[:-1, 0], quaternions[1:, 0]
    v0, v1 = quaternions[:-1, 1:], quaternions[1:, 1:]
    scalar = w0 * w1 + np.einsum("ij,ij->i", v0, v1)
    vector = w0[:, None] * v1 - w1[:, None] * v0 - np.cross(v0, v1)
    return 2.0 * np.arctan2(np.linalg.norm(vector, axis=-1), np.abs(scalar))


def _refuse_missing_family(
    df: pd.DataFrame, columns: Sequence[str], family: str, who: str
) -> None:
    """Raise naming *who* when *df* lacks any of a column family's *columns*."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{who} needs the {family} columns {list(columns)} and this table lacks "
            f"{missing}. A world-frame table carries them only where its producer "
            f"measured them."
        )


# --- ray-target ---------------------------------------------------------------


_RAY_DESCRIPTION = (
    "The column prefix of the ray to measure: its origin is read from "
    "<ray>X, <ray>Y and <ray>Z and its direction from <ray>DX, <ray>DY and "
    "<ray>DZ. The default is the gaze ray every mosaic_world_v1 table may carry. "
    "The direction is renormalised; a zero or non-finite one gives NaN."
)

_KEYPOINT_NAMES_DESCRIPTION = (
    "The keypoint names, one per keypoint of the table and in its keypoint order, "
    "so that targets can refer to keypoints by name. A list of another length is "
    "refused. Unset, targets refer to keypoints by index only."
)

_TARGETS_DESCRIPTION = (
    "Each target's name mapped to the keypoints that define it, by index or by "
    "name, each keypoint once. A target is the mean of its keypoints that are "
    "finite on the row. "
    "Names become output column suffixes, so each must be a Python identifier. "
    "The keypoints are the measured individual's own."
)

_RAY_FPS_DESCRIPTION = (
    "The frame rate ray_angular_speed is scaled by. Unset, it is measured from "
    "the time column as the median time step between consecutive frames."
)


@final
@register_feature
class RayTarget:
    """Where a ray points relative to targets on the body, in a world frame.

    For every ``(frame, id)`` of a ``mosaic_world_v1`` table, and for every
    target named in ``targets``:

    - ``angle__<target>`` -- the angle, in radians in ``[0, pi]``, between the
      ray's direction and the line from the ray's origin to the target. Zero is
      the ray pointing straight at it.
    - ``dist__<target>`` -- the distance from the ray's origin to the target, in
      the table's unit (metres).

    and once per row:

    - ``ray_angular_speed`` -- how fast the ray's direction turns, in radians per
      second: the angle between this frame's direction and the previous frame's,
      times the frame rate. NaN on an individual's first row and across a
      missing frame.

    A target is the mean of the finite keypoints of its group on that row, so a
    head target over four landmarks survives one of them dropping out. With none
    finite, or the target exactly at the ray's origin, the angle is NaN. Targets
    are emitted in name order, so a recipe's key order changes nothing.

    Refuses a table without the ray's six columns or without 3D keypoints, naming
    itself. A keypoint index outside the table's keypoints, and a
    ``keypoint_names`` list of another length than the table's keypoints, are
    refused per table.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.gaze_geometry.RayTarget.Params`.
    """

    category = "per-frame"
    name = "ray-target"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    track_spaces: ClassVar[frozenset[str]] = _WORLD_ONLY

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        ray: Annotated[str, Declared(_RAY_DESCRIPTION)] = "gaze"
        keypoint_names: Annotated[
            list[str] | None, Declared(_KEYPOINT_NAMES_DESCRIPTION)
        ] = None
        targets: Annotated[
            dict[str, list[int | str]],
            Field(min_length=1),
            Declared(_TARGETS_DESCRIPTION),
        ]
        fps: Annotated[float | None, Declared(_RAY_FPS_DESCRIPTION, unit="fps")] = (
            Field(default=None, gt=0)
        )

        @field_validator("ray")
        @classmethod
        def _ray_is_a_prefix(cls, ray: str) -> str:
            if not _IDENTIFIER.match(ray):
                raise ValueError(
                    f"ray must be a column prefix such as 'gaze', got {ray!r}"
                )
            return ray

        @field_validator("targets")
        @classmethod
        def _targets_are_named_groups(
            cls, targets: dict[str, list[int | str]]
        ) -> dict[str, list[int | str]]:
            for target, refs in targets.items():
                if not _IDENTIFIER.match(target):
                    raise ValueError(
                        f"target name {target!r} becomes the column suffix of "
                        f"angle__{target} and dist__{target}, so it must be an "
                        "identifier: letters, digits and underscores, not "
                        "starting with a digit"
                    )
                if not refs:
                    raise ValueError(f"target {target!r} names no keypoints")
            return targets

        @model_validator(mode="after")
        def _named_keypoints_resolve(self) -> Self:
            """Refuse an unresolvable name or a repeated keypoint, before any table.

            An index can only be checked against a table's keypoint count, so
            that half waits for ``apply``; a name, and a target listing one
            keypoint twice, can be checked now.
            """
            names = self.keypoint_names
            if names is not None and len(set(names)) != len(names):
                raise ValueError(f"keypoint_names repeats a name: {names}")
            for target, refs in self.targets.items():
                indices: list[int] = []
                for ref in refs:
                    if not isinstance(ref, str):
                        indices.append(ref)
                        continue
                    if names is None:
                        raise ValueError(
                            f"target {target!r} names keypoint {ref!r} but no "
                            "keypoint_names were given; pass them or refer to "
                            "keypoints by index"
                        )
                    if ref not in names:
                        raise ValueError(
                            f"target {target!r}: no keypoint named {ref!r}; the "
                            f"names are {names}"
                        )
                    indices.append(names.index(ref))
                # A target is a mean, so a keypoint listed twice -- or once by
                # name and once by index -- would silently count double.
                if len(set(indices)) != len(indices):
                    raise ValueError(
                        f"target {target!r} lists a keypoint more than once, "
                        f"{refs}; it is the mean of its keypoints, so a repeated "
                        "one would count double"
                    )
            return self

    def __init__(
        self,
        inputs: RayTarget.Inputs = Inputs(("tracks",)),
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
        ensure_columns(df, [C.frame_col, C.id_col])
        ray = self.params.ray
        origin_columns = (f"{ray}X", f"{ray}Y", f"{ray}Z")
        direction_columns = (f"{ray}DX", f"{ray}DY", f"{ray}DZ")
        _refuse_missing_family(
            df, origin_columns + direction_columns, f"{ray!r} ray", self.name
        )
        triplets = pose_column_triplets(df.columns)
        if not triplets:
            raise ValueError(
                f"{self.name} needs 3D keypoint columns (poseX*/poseY*/poseZ*) to "
                "place its targets and this table carries none. A table with "
                "poseX*/poseY* only is planar, and a target there has no height "
                "to aim a ray at."
            )
        names = self.params.keypoint_names
        if names is not None and len(names) != len(triplets):
            raise ValueError(
                f"{self.name}: keypoint_names lists {len(names)} names and this "
                f"table carries {len(triplets)} keypoints. A name is matched to a "
                "keypoint by position, so a list of another length names the "
                "keypoints of a different skeleton."
            )
        targets = {
            target: resolve_keypoint_refs(
                refs, names, len(triplets), who=f"{self.name} target {target!r}"
            )
            for target, refs in sorted(self.params.targets.items())
        }
        used = sorted({index for indices in targets.values() for index in indices})

        table = _sorted_by_id_and_frame(
            df,
            [
                *origin_columns,
                *direction_columns,
                *(column for index in used for column in triplets[index]),
            ],
        )
        origin = _stack(table, origin_columns)
        direction = _normalised(_stack(table, direction_columns))

        measurements: dict[str, _Floats] = {}
        for target, indices in targets.items():
            group = np.stack([_stack(table, triplets[i]) for i in indices], axis=1)
            point = _finite_mean(group)  # (n, 3)
            offset = point - origin
            distance = np.linalg.norm(offset, axis=-1)
            angle = _angle_between(direction, offset)
            # A target at the origin has no direction to be off from.
            angle[~(distance > 0.0)] = np.nan
            measurements[f"angle__{target}"] = angle
            measurements[f"dist__{target}"] = distance
        (measurements["ray_angular_speed"],) = _step_rates(
            table,
            [_angle_between(direction[1:], direction[:-1])],
            self.params.fps,
            self.name,
        )
        return _with_meta(table, measurements)


def _finite_mean(points: _Floats) -> _Floats:
    """The mean over axis 1 of the points finite on all three axes, ``(n, 3)``.

    NaN for a row with none. Written out rather than ``np.nanmean`` so that a
    keypoint missing one axis is dropped whole rather than contributing two of
    its three coordinates, and so that an empty row warns nothing.
    """
    finite = np.isfinite(points).all(axis=-1)  # (n, g)
    count = finite.sum(axis=1)
    total = np.where(finite[..., None], points, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count[:, None]
    mean[count == 0] = np.nan
    return mean


# --- head-eye -----------------------------------------------------------------


_HEAD_FPS_DESCRIPTION = (
    "The frame rate head_angular_speed and gaze_angular_speed are scaled by. "
    "Unset, it is measured from the time column as the median time step between "
    "consecutive frames."
)


@final
@register_feature
class HeadEye:
    """Head orientation, and where the eyes point within the head, in a world frame.

    Reads the head pose (``headQW/QX/QY/QZ``, the head-to-world rotation) and the
    gaze direction (``gazeDX/DY/DZ``) of a ``mosaic_world_v1`` table. The head
    frame is the schema's: +X forward out of the face, +Y to the subject's left,
    +Z up through the crown. The world frame is right-handed with +Z up. All
    angles are in radians and all rates in radians per second.

    Per ``(frame, id)``:

    - ``head_yaw``, ``head_pitch``, ``head_roll`` -- the head-to-world rotation as
      intrinsic Z-Y-X Tait-Bryan angles, ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``:
      read off the matrix as ``yaw = atan2(R10, R00)``,
      ``pitch = atan2(-R20, hypot(R00, R10))`` and ``roll = atan2(R21, R22)``.
      Each follows the right-hand rule about its axis, and with +Z up and +Y to
      the left that makes the physical readings:

      - **positive yaw** turns the nose from world +X toward world +Y,
        counter-clockwise seen from above; range ``(-pi, pi]``.
      - **positive pitch tips the nose DOWN** -- a right-handed turn about the
        head's leftward +Y axis carries +X toward -Z. This is the opposite sign
        to ``gaze_elevation`` below, which is positive up; range
        ``[-pi/2, pi/2]``.
      - **positive roll lifts the left ear**, tilting the crown toward the
        subject's right shoulder; range ``(-pi, pi]``.

      At a pitch of exactly ``+-pi/2`` yaw and roll share one axis and only
      their combination is defined; the split reported there is arbitrary.

    - ``gaze_azimuth`` and ``gaze_elevation`` -- the gaze direction expressed in
      the head frame, ``d = R^T g``: ``azimuth = atan2(d_y, d_x)``, positive to
      the subject's left, and ``elevation = atan2(d_z, hypot(d_x, d_y))``,
      positive up. Zero and zero is looking straight ahead of the face.
    - ``eye_eccentricity`` -- the angle between the gaze and the head's +X axis,
      in ``[0, pi]``: how far off head-forward the eyes point, in any direction.
    - ``head_angular_speed`` -- the angle of the rotation taking the previous
      frame's head pose to this frame's, times the frame rate. A quaternion and
      its negation are one rotation, and read as no movement.
    - ``gaze_angular_speed`` -- the angle between the previous frame's gaze
      direction and this frame's, in the world frame, times the frame rate. Gaze
      in space, so a head turn with the eyes fixed in the head moves it.

    Both speeds are NaN on an individual's first row and across a missing
    frame. A zero or non-finite quaternion or gaze direction makes every
    quantity that reads it NaN.

    Refuses a table without the ``head*`` quaternion or the ``gaze*`` direction,
    naming itself.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.gaze_geometry.HeadEye.Params`.
    """

    category = "per-frame"
    name = "head-eye"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    track_spaces: ClassVar[frozenset[str]] = _WORLD_ONLY

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        fps: Annotated[float | None, Declared(_HEAD_FPS_DESCRIPTION, unit="fps")] = (
            Field(default=None, gt=0)
        )

    def __init__(
        self,
        inputs: HeadEye.Inputs = Inputs(("tracks",)),
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
        ensure_columns(df, [C.frame_col, C.id_col])
        _refuse_missing_family(df, _QUATERNION_COLUMNS, "head-pose", self.name)
        _refuse_missing_family(df, _GAZE_DIRECTION_COLUMNS, "gaze", self.name)

        table = _sorted_by_id_and_frame(
            df, _QUATERNION_COLUMNS + _GAZE_DIRECTION_COLUMNS
        )
        quaternions = _normalised(_stack(table, _QUATERNION_COLUMNS))
        rotation = _quaternion_matrices(quaternions)  # head -> world
        gaze = _normalised(_stack(table, _GAZE_DIRECTION_COLUMNS))

        yaw = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
        pitch = np.arctan2(
            -rotation[:, 2, 0], np.hypot(rotation[:, 0, 0], rotation[:, 1, 0])
        )
        roll = np.arctan2(rotation[:, 2, 1], rotation[:, 2, 2])

        in_head = np.einsum("nji,nj->ni", rotation, gaze)  # R^T g
        azimuth = np.arctan2(in_head[:, 1], in_head[:, 0])
        elevation = np.arctan2(in_head[:, 2], np.hypot(in_head[:, 0], in_head[:, 1]))
        eccentricity = np.arctan2(np.hypot(in_head[:, 1], in_head[:, 2]), in_head[:, 0])
        head_speed, gaze_speed = _step_rates(
            table,
            [
                _quaternion_step_angles(quaternions),
                _angle_between(gaze[1:], gaze[:-1]),
            ],
            self.params.fps,
            self.name,
        )

        return _with_meta(
            table,
            {
                "head_yaw": yaw,
                "head_pitch": pitch,
                "head_roll": roll,
                "gaze_azimuth": azimuth,
                "gaze_elevation": elevation,
                "eye_eccentricity": eccentricity,
                "head_angular_speed": head_speed,
                "gaze_angular_speed": gaze_speed,
            },
        )
