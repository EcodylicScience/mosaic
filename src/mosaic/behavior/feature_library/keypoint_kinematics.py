"""KeypointKinematics: per-keypoint speed and acceleration, and keypoint distances.

The movement-quality measures a phenotype is built from -- how fast a hand
moves, how sharply it accelerates, the rhythm of a cutting motion in a wrist's
vertical velocity, how far apart the two hands are -- are per-keypoint
quantities, where ``speed-angvel`` measures the body centre alone. This feature
measures them on every keypoint a table carries, in 2D or 3D, in the table's own
units.

Derivatives come from a Savitzky-Golay fit rather than a frame difference; the
method is in :mod:`mosaic.behavior.feature_library.derivatives`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Self, final, override

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, model_validator

from mosaic.core.pipeline.types.feature import RelatesLevel
from mosaic.core.schema import TRACK_SPACES, TrackSpace
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
from mosaic.core.keypoint_refs import (
    check_heading_pair,
    check_keypoint_refs,
    check_keypoints_distinct,
    keypoint_index,
    resolve_keypoint_refs,
)

from .derivatives import (
    FPS_DESCRIPTION,
    KEYPOINT_NAMES_DESCRIPTION,
    WINDOW_DESCRIPTION,
    FloatArray,
    FrameAxis,
    KeypointColumns,
    RowLayout,
    SavgolKernel,
    check_keypoint_names,
    check_savgol_window,
    frame_axis,
    frame_rate,
    keypoint_columns,
    keypoint_positions,
    row_norm,
    savgol_derivatives,
)
from .helpers import body_frame, ego_rotate, ensure_columns, meta_columns
from .registry import register_feature

__all__ = ["KeypointKinematics"]

_NAME = "keypoint-kinematics"

_KEYPOINTS_DESCRIPTION = (
    "The keypoints to report speed and acceleration for, by index or by name. "
    "With heading set, they also get the four own-frame columns. "
    "An index is a keypoint's position in the table's keypoint order, which is "
    "its column number when the keypoints are numbered from 0 without a gap. "
    "Unset, every keypoint the table carries. An empty list reports none, for a "
    "run that wants only distances."
)

_POLYORDER_DESCRIPTION = (
    "The order of the polynomial fitted in each window. At least 2, because "
    "acceleration is the fit's second derivative and a lower order would report "
    "it as zero everywhere."
)

_DISTANCES_DESCRIPTION = (
    "Keypoint pairs, by index or by name, whose separation is reported on every "
    "row as dist__{a}__{b}. Euclidean over the table's two or three axes, and "
    "measured on the unsmoothed positions."
)

_NORMALIZE_BY_DESCRIPTION = (
    "A keypoint pair whose median separation over each individual's rows in the "
    "sequence divides every distance, speed, vertical velocity, acceleration, "
    "own-frame velocity and own-frame position, making them independent of body "
    "size. Unset, they stay in the table's units. An individual on which the pair "
    "is never seen reports NaN throughout."
)

_HEADING_DESCRIPTION = (
    "A keypoint pair (neck, tail base), by index or by name, that sets each "
    "row's body frame: the heading points from tail base to neck, and the body "
    "center is their midpoint. Set, every reported keypoint also gets vx__{kp} "
    "and vy__{kp}, its velocity along the heading and along the heading's "
    "left-hand normal, and ego_x__{kp} and ego_y__{kp}, its position from the "
    "body center along the same two axes. Positive y is 90 degrees "
    "counterclockwise from the heading in a y-up frame. On a 3D table the heading "
    "is projected on the XY plane and the rotation is about Z; vz__{kp} stays "
    "the vertical velocity. The four columns are NaN on a row where either "
    "keypoint of the pair is missing or the two coincide in the plane. Unset, "
    "none of them is written."
)


@dataclass(frozen=True)
class _Plan:
    """What one table asks for, resolved against its keypoint columns.

    Attributes:
        keypoints: The coordinate columns and labels of every keypoint the
            table carries.
        selected: The keypoints whose speed and acceleration are reported.
        distances: The keypoint pairs whose separation is reported.
        normalize_by: The pair whose median separation divides every length.
        heading: The (neck, tail base) pair that sets the body frame.
    """

    keypoints: KeypointColumns
    selected: tuple[int, ...]
    distances: tuple[tuple[int, int], ...]
    normalize_by: tuple[int, int] | None
    heading: tuple[int, int] | None


@final
@register_feature
class KeypointKinematics:
    """Per-keypoint smoothed speed and acceleration, and inter-keypoint distances.

    For each keypoint, per individual, on every input row:

    - ``speed__{kp}`` -- the magnitude of the velocity, in the table's units per
      second.
    - ``accel__{kp}`` -- the magnitude of the acceleration vector, in the table's
      units per second squared. The magnitude of the second derivative, not the
      rate of change of speed: a keypoint circling at constant speed accelerates.
    - ``vz__{kp}`` -- on a 3D table only, the signed velocity along +Z, which
      is up in a table whose Z axis is vertical, so a rising keypoint is positive.

    With ``heading`` set to a (neck, tail base) pair, also, for each keypoint:

    - ``vx__{kp}`` and ``vy__{kp}`` -- the velocity along the animal's heading
      and along its left-hand normal, in the table's units per second.
    - ``ego_x__{kp}`` and ``ego_y__{kp}`` -- the position from the body center,
      the midpoint of neck and tail base, along the same two axes.

    The heading points from tail base to neck on the row's unsmoothed positions,
    and positive y is 90 degrees counterclockwise from it in a y-up frame. The
    velocities are the world velocities rotated into that frame. On a 3D table
    the heading is projected on the XY plane, the rotation is about Z,
    ``vz__{kp}`` is unchanged and no ``ego_z__`` column is written. The four
    columns are NaN on a row where either heading keypoint is missing or the two
    coincide in the plane, an animal pointing straight up included.

    For each requested pair, ``dist__{a}__{b}``, the Euclidean distance
    between two keypoints on that row. ``{kp}`` is the keypoint's name when
    ``keypoint_names`` is given, else ``kp{k}`` with ``k`` its position in the
    table's keypoint order. Every measure is ``float32``.

    **2D or 3D, decided by the table.** A table carrying ``Z`` or ``poseZ*``
    columns is measured on its keypoint triplets, and one carrying neither on its
    ``poseX*``/``poseY*`` pairs. Lengths are in the table's own unit: video
    pixels, centimetres, or the metres of a 3D producer. A table with no keypoints is
    refused rather than measured on its body centre, which ``speed-angvel``
    already does.

    **Method and NaN.** The module docstring of
    :mod:`~mosaic.behavior.feature_library.derivatives` gives the Savitzky-Golay
    method, the gap handling with ``window``, and which rows are NaN.

    **Normalization.** With ``normalize_by``, every distance, speed, vertical
    velocity, acceleration and own-frame column is divided by the median length
    of that keypoint segment over the individual's rows in the sequence, which
    makes them comparable across body sizes. A distance is then in segment
    lengths, a speed in segment lengths per second, an acceleration per second
    squared.

    One row per ``(frame, id)`` is required, so a table holding one row per
    camera is refused. Field documentation is on
    :class:`~mosaic.behavior.feature_library.keypoint_kinematics.KeypointKinematics.Params`.
    """

    category = "per-frame"
    name = _NAME
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "none"
    accepts_overlap = False  # one individual's frame axis is one entry's
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    follows_ids: bool = True
    track_spaces: ClassVar[frozenset[TrackSpace]] = frozenset(TRACK_SPACES)
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.frame_col),
        Column(C.id_col),
        POSE_X,
        POSE_Y,
    )
    provides: ClassVar[Provides] = Provides(
        conditional=(
            Conditional(Prefix("speed__")),
            Conditional(Prefix("accel__")),
            Conditional(Prefix("vz__"), when_present=Prefix("poseZ")),
            Conditional(Prefix("vx__"), when_param="heading"),
            Conditional(Prefix("vy__"), when_param="heading"),
            Conditional(Prefix("ego_x__"), when_param="heading"),
            Conditional(Prefix("ego_y__"), when_param="heading"),
            Conditional(Prefix("dist__"), when_param="distances"),
            *META_COLUMNS_KEPT,
        ),
    )

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        """Keypoint-kinematics parameters."""

        keypoint_names: Annotated[
            list[str] | None, Declared(KEYPOINT_NAMES_DESCRIPTION, keypoint="name")
        ] = None
        keypoints: Annotated[
            list[int | str] | None, Declared(_KEYPOINTS_DESCRIPTION, keypoint="either")
        ] = None
        fps: Annotated[float | None, Declared(FPS_DESCRIPTION, unit="fps")] = Field(
            default=None, gt=0
        )
        window: Annotated[int, Declared(WINDOW_DESCRIPTION, unit="frames")] = Field(
            default=7, ge=3
        )
        polyorder: Annotated[int, Declared(_POLYORDER_DESCRIPTION)] = Field(
            default=2, ge=2
        )
        distances: Annotated[
            list[tuple[int | str, int | str]],
            Declared(_DISTANCES_DESCRIPTION, keypoint="either"),
        ] = Field(default_factory=list)
        normalize_by: Annotated[
            tuple[int | str, int | str] | None,
            Declared(_NORMALIZE_BY_DESCRIPTION, keypoint="either"),
        ] = None
        heading: Annotated[
            tuple[int | str, int | str] | None,
            Declared(_HEADING_DESCRIPTION, keypoint="either"),
        ] = None

        @model_validator(mode="after")
        def _check(self) -> Self:
            """Reject at construction what no table could make valid.

            A raise from ``apply`` is recorded per entry and the run carries on,
            so a mistake every entry would repeat belongs here, where it stops
            the run before anything is read. What depends on the table -- an
            index past its last keypoint, a name list of the wrong length -- is
            checked in ``apply``.
            """
            check_savgol_window(self.window, self.polyorder, _NAME)
            names = self.keypoint_names
            check_keypoint_names(names, _NAME)
            refs: list[int | str] = [
                *(self.keypoints or []),
                *(ref for pair in self.distances for ref in pair),
                *(self.normalize_by or ()),
                *(self.heading or ()),
            ]
            check_keypoint_refs(refs, names, _NAME)
            if self.keypoints == [] and not self.distances:
                raise ValueError(
                    f"{_NAME}: keypoints=[] reports no keypoint and distances is "
                    "empty, so there is nothing to compute. Leave keypoints unset "
                    "for every keypoint."
                )
            if self.keypoints == [] and self.heading is not None:
                raise ValueError(
                    f"{_NAME}: heading is set but keypoints=[] reports no keypoint, "
                    "so none of its columns would be written. Leave keypoints unset "
                    "for every keypoint, or name the ones to place in the body frame."
                )
            self._check_repeats(names)
            return self

        @override
        def identity_dump(self) -> dict[str, object]:
            """The hashed params, with ``heading`` left out while it is unset.

            A run without a heading has the identity it had before the field
            existed.
            """
            dumped = super().identity_dump()
            if self.heading is None:
                del dumped["heading"]
            return dumped

        def _check_repeats(self, names: list[str] | None) -> None:
            """Refuse a keypoint, or a pair, that is named twice or paired with itself.

            Index and name are one keypoint wherever they agree, so ``0`` and
            the first name are compared as the same reference. Nothing here
            depends on the table.
            """

            check_keypoints_distinct(self.keypoints, names, _NAME)
            seen: set[frozenset[int]] = set()
            for a_ref, b_ref in self.distances:
                pair = frozenset(
                    (keypoint_index(a_ref, names), keypoint_index(b_ref, names))
                )
                if len(pair) == 1:
                    raise ValueError(
                        f"{_NAME}: distance ({a_ref!r}, {b_ref!r}) is between a "
                        "keypoint and itself, which is zero by construction."
                    )
                if pair in seen:
                    raise ValueError(
                        f"{_NAME}: distance ({a_ref!r}, {b_ref!r}) is requested "
                        "twice, in one order or the other."
                    )
                seen.add(pair)
            if self.normalize_by is not None:
                a_ref, b_ref = self.normalize_by
                if keypoint_index(a_ref, names) == keypoint_index(b_ref, names):
                    raise ValueError(
                        f"{_NAME}: normalize_by {self.normalize_by} is a keypoint "
                        "and itself, whose length is zero."
                    )
            if self.heading is not None:
                check_heading_pair(self.heading, names, _NAME)

    def __init__(
        self,
        inputs: KeypointKinematics.Inputs = Inputs(("tracks",)),
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

    # --- Apply ---

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        ensure_columns(df, [C.frame_col, C.id_col])
        plan = self._plan(df)
        axis = frame_axis(df, self.name)
        fps = frame_rate(df, axis, self.params.fps, self.name)
        measures = self._measure(df, plan, axis, fps)
        meta = df[meta_columns(df)].iloc[axis.order].reset_index(drop=True)
        return pd.concat([meta, pd.DataFrame(measures)], axis=1)

    # --- Internal helpers ---

    def _plan(self, df: pd.DataFrame) -> _Plan:
        """Resolve the parameters against *df*'s keypoint columns."""
        p = self.params
        keypoints = keypoint_columns(
            df,
            p.keypoint_names,
            self.name,
            mixed_axes_hint=(
                "Measuring them in the plane would mix 2D and 3D speeds in one output."
            ),
            no_keypoints_hint=(
                "The speed of a body center is what 'speed-angvel' measures."
            ),
        )
        n_keypoints = len(keypoints.columns)

        def resolve(refs: list[int | str], what: str) -> list[int]:
            return resolve_keypoint_refs(
                refs, p.keypoint_names, n_keypoints, who=f"{self.name} {what}"
            )

        # Repeats and self-pairs were refused at construction; what is left to
        # check against the table is that every index is one it carries.
        selected = (
            list(range(n_keypoints))
            if p.keypoints is None
            else resolve(list(p.keypoints), "keypoints")
        )
        distances: list[tuple[int, int]] = []
        for a_ref, b_ref in p.distances:
            a, b = resolve([a_ref, b_ref], "distances")
            distances.append((a, b))
        normalize_by: tuple[int, int] | None = None
        if p.normalize_by is not None:
            a, b = resolve(list(p.normalize_by), "normalize_by")
            normalize_by = (a, b)
        heading: tuple[int, int] | None = None
        if p.heading is not None:
            neck, tail = resolve(list(p.heading), "heading")
            heading = (neck, tail)
        return _Plan(
            keypoints=keypoints,
            selected=tuple(selected),
            distances=tuple(distances),
            normalize_by=normalize_by,
            heading=heading,
        )

    def _measure(
        self,
        df: pd.DataFrame,
        plan: _Plan,
        axis: FrameAxis,
        fps: float,
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Every output column, in output row order, keyed by its name."""
        p = self.params
        order = axis.order

        def positions(k: int) -> FloatArray:
            return keypoint_positions(df, plan.keypoints.columns[k], order)

        # One divisor per row: the individual's median segment length, or 1.
        scale = np.ones(order.size)
        if plan.normalize_by is not None:
            a, b = plan.normalize_by
            segment = pd.Series(row_norm(positions(a) - positions(b)))
            # NaN for an individual on which the segment is never seen.
            medians = segment.groupby(axis.codes).median()
            per_individual = medians.to_numpy(dtype=np.float64, copy=True)
            per_individual[~(per_individual > 0)] = np.nan
            scale = per_individual[axis.codes]

        kernels = (
            SavgolKernel.build(p.window, p.polyorder, 1, 1.0 / fps),
            SavgolKernel.build(p.window, p.polyorder, 2, 1.0 / fps),
        )
        layout = RowLayout.build(axis, p.window)
        labels = plan.keypoints.labels
        speed: dict[str, FloatArray] = {}
        accel: dict[str, FloatArray] = {}
        vertical: dict[str, FloatArray] = {}
        own: dict[str, dict[str, FloatArray]] = {
            prefix: {} for prefix in ("vx", "vy", "ego_x", "ego_y")
        }
        frame = (
            None
            if plan.heading is None
            else body_frame(positions(plan.heading[0]), positions(plan.heading[1]))
        )
        for k in plan.selected:
            observed = positions(k)
            (velocity, acceleration), absent = savgol_derivatives(
                observed, layout, p.window, kernels
            )
            label = labels[k]
            speed[f"speed__{label}"] = (
                np.where(absent, np.nan, row_norm(velocity)) / scale
            )
            accel[f"accel__{label}"] = (
                np.where(absent, np.nan, row_norm(acceleration)) / scale
            )
            if plan.keypoints.three_d:
                vertical[f"vz__{label}"] = (
                    np.where(absent, np.nan, velocity[:, 2]) / scale
                )
            if frame is not None:
                vx, vy = ego_rotate(velocity[:, 0], velocity[:, 1], frame.heading)
                ego_x, ego_y = ego_rotate(
                    observed[:, 0] - frame.center_x,
                    observed[:, 1] - frame.center_y,
                    frame.heading,
                )
                for prefix, values in (
                    ("vx", vx),
                    ("vy", vy),
                    ("ego_x", ego_x),
                    ("ego_y", ego_y),
                ):
                    own[prefix][f"{prefix}__{label}"] = (
                        np.where(absent, np.nan, values) / scale
                    )
        dist = {
            f"dist__{labels[a]}__{labels[b]}": row_norm(positions(a) - positions(b))
            / scale
            for a, b in plan.distances
        }
        return {
            name: values.astype(np.float32)
            for name, values in {
                **speed,
                **accel,
                **vertical,
                **own["vx"],
                **own["vy"],
                **own["ego_x"],
                **own["ego_y"],
                **dist,
            }.items()
        }
