"""PairKeypoints: the partner's keypoints and their velocities in the focal's body frame.

``pair-egocentric`` relates two animals by their keypoint means. This feature
keeps every keypoint: where each of the partner's keypoints sits relative to the
focal animal's body, how it moves in that frame, and how far chosen keypoints of
the two animals are apart, which is what a sniff of the partner's head or tail
base is read from.

Velocities come from the Savitzky-Golay fit of
:mod:`mosaic.behavior.feature_library.derivatives`, on each animal's own frame
axis, and the body frame is :func:`~mosaic.behavior.feature_library.helpers.body_frame`.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Self, final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, model_validator

from mosaic.core.individuals import id_arenas, individual_pairs
from mosaic.core.keypoint_refs import (
    check_heading_pair,
    check_keypoint_refs,
    check_keypoints_distinct,
    keypoint_index,
    resolve_keypoint_refs,
)
from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.column_contract import (
    POSE_X,
    POSE_Y,
    Column,
    Conditional,
    ParamIs,
    Prefix,
    Provides,
    Requirement,
    When,
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

from .derivatives import (
    FPS_DESCRIPTION,
    KEYPOINT_NAMES_DESCRIPTION,
    WINDOW_DESCRIPTION,
    FloatArray,
    FrameAxis,
    IntArray,
    RowLayout,
    SavgolKernel,
    check_keypoint_names,
    check_savgol_window,
    frame_axis,
    frame_rate,
    keypoint_columns,
    keypoint_positions,
    savgol_derivatives,
)
from .helpers import BodyFrame, body_frame, ego_rotate, ensure_columns, point_distances
from .registry import register_feature

__all__ = ["PairKeypoints"]

_NAME = "pair-keypoints"

_PARTNER_PREFIXES = ("partner_x", "partner_y", "partner_vx", "partner_vy")

_KEYPOINTS_DESCRIPTION = (
    "The partner keypoints to place in the focal's body frame, by index or by "
    "name. An index is a keypoint's position in the table's keypoint order, which "
    "is its column number when the keypoints are numbered from 0 without a gap. "
    "Unset, every keypoint the table carries. An empty list places none, for a run "
    "that wants only inter_distances."
)

_HEADING_DESCRIPTION = (
    "The focal's keypoint pair (neck, tail base), by index or by name, that sets "
    "each row's body frame: the heading points from tail base to neck, and the "
    "body center is their midpoint. Every partner_ column is measured along the "
    "heading (x) and along its left-hand normal (y); positive y is 90 degrees "
    "counterclockwise from the heading in a y-up frame. A row where either "
    "keypoint of the focal's pair is missing, or the two coincide, has no body "
    "frame, and its partner_ columns are NaN."
)

_INTER_DISTANCES_DESCRIPTION = (
    "Keypoint pairs (a, b), by index or by name, whose separation is reported "
    "on every row as inter_dist__{a}__{b}: from the focal's keypoint a to the "
    "partner's keypoint b, on the unsmoothed positions. The order matters, so "
    "(a, b) and (b, a) are two distances, and (a, a) is the distance between the "
    "two animals' keypoint a."
)

_POLYORDER_DESCRIPTION = (
    "The order of the polynomial fitted in each window. At least 1, because only "
    "the velocity, the fit's first derivative, is taken."
)


@dataclass(frozen=True)
class _Plan:
    """What one table asks for, resolved against its keypoint columns.

    Attributes:
        columns: The coordinate columns of every keypoint the table carries.
        labels: Each keypoint's label in an output column name.
        selected: The partner keypoints placed in the focal's body frame.
        heading: The (neck, tail base) pair that sets the body frame.
        inter_distances: The (focal keypoint, partner keypoint) pairs measured.
    """

    columns: tuple[tuple[str, ...], ...]
    labels: tuple[str, ...]
    selected: tuple[int, ...]
    heading: tuple[int, int]
    inter_distances: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _Rows:
    """Every input row's measures, in the order of the table's frame axis.

    Attributes:
        positions: Each keypoint's ``(n_rows, 2)`` positions, NaN where missing.
        velocities: Each selected keypoint's ``(n_rows, 2)`` velocities, NaN
            where the keypoint is missing.
        frame: Each row's body frame.
    """

    positions: dict[int, FloatArray]
    velocities: dict[int, FloatArray]
    frame: BodyFrame


@final
@register_feature
class PairKeypoints:
    """Each partner keypoint's position and velocity in the focal's body frame.

    For every pair of individuals in one arena, on every frame both are present,
    two rows: perspective 0 with ``id1`` the focal and ``id2`` the partner, and
    perspective 1 with the roles swapped. On each row, for each selected partner
    keypoint:

    - ``partner_x__{kp}`` and ``partner_y__{kp}`` -- the partner keypoint's
      position from the focal's body center, along the focal's heading and along
      its left-hand normal, in the table's units.
    - ``partner_vx__{kp}`` and ``partner_vy__{kp}`` -- the partner keypoint's
      velocity along the same two axes, in the table's units per second.

    For each requested ``(a, b)``, ``inter_dist__{a}__{b}``, the distance from
    the focal's keypoint ``a`` to the partner's keypoint ``b``. ``{kp}`` is the
    keypoint's name when ``keypoint_names`` is given, else ``kp{k}`` with ``k``
    its position in the table's keypoint order. Every measure is ``float32``.

    **Body frame.** The focal's heading points from its tail base to its neck on
    the row's unsmoothed positions, and its body center is their midpoint.
    Positive y is 90 degrees counterclockwise from the heading in a y-up frame,
    as ``AB_dy_egoA`` of ``pair-egocentric``. The velocity is the partner's world
    velocity rotated into the focal's frame, not the derivative of the rotated
    position.

    **Method and NaN.** The velocity is the Savitzky-Golay first derivative on the
    partner's own frame axis, so a frame the focal lacks does not shorten it. The
    module docstring of :mod:`~mosaic.behavior.feature_library.derivatives` gives
    the method, the gap handling with ``window``, and which rows are NaN. The
    ``partner_`` columns are also NaN where the focal has no body frame.

    A 3D table is refused: the body frame is planar, and
    ``keypoint-kinematics`` with ``heading`` measures an animal's own keypoints
    in 3D. One row per ``(frame, id)`` is required. Field documentation is on
    :class:`~mosaic.behavior.feature_library.pair_keypoints.PairKeypoints.Params`.
    """

    category = "per-frame"
    name = _NAME
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "arena-aware"
    accepts_overlap = False  # one individual's frame axis is one entry's
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "pair"
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.frame_col),
        Column(C.id_col),
        POSE_X,
        POSE_Y,
        When(Column(C.time_col), (ParamIs("fps", negated=True),)),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(
            Column(C.frame_col),
            Column("perspective"),
            Column("id1"),
            Column("id2"),
        ),
        conditional=(
            *(Conditional(Prefix(f"{prefix}__")) for prefix in _PARTNER_PREFIXES),
            Conditional(Prefix("inter_dist__"), when_param="inter_distances"),
            *kept(C.seq_col, C.group_col),
        ),
    )

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        """Pair-keypoints parameters."""

        keypoint_names: Annotated[
            list[str] | None, Declared(KEYPOINT_NAMES_DESCRIPTION)
        ] = None
        keypoints: Annotated[
            list[int | str] | None, Declared(_KEYPOINTS_DESCRIPTION)
        ] = None
        heading: Annotated[tuple[int | str, int | str], Declared(_HEADING_DESCRIPTION)]
        inter_distances: Annotated[
            list[tuple[int | str, int | str]], Declared(_INTER_DISTANCES_DESCRIPTION)
        ] = Field(default_factory=list)
        fps: Annotated[float | None, Declared(FPS_DESCRIPTION, unit="fps")] = Field(
            default=None, gt=0
        )
        window: Annotated[int, Declared(WINDOW_DESCRIPTION, unit="frames")] = Field(
            default=7, ge=3
        )
        polyorder: Annotated[int, Declared(_POLYORDER_DESCRIPTION)] = Field(
            default=2, ge=1
        )

        @model_validator(mode="after")
        def _check(self) -> Self:
            """Reject at construction what no table could make valid.

            A raise from ``apply`` is recorded per entry and the run carries on,
            so a mistake every entry would repeat belongs here. What depends on
            the table -- an index past its last keypoint, a name list of the
            wrong length -- is checked in ``apply``.
            """
            check_savgol_window(self.window, self.polyorder, _NAME)
            names = self.keypoint_names
            check_keypoint_names(names, _NAME)
            check_keypoint_refs(
                [
                    *(self.keypoints or []),
                    *self.heading,
                    *(ref for pair in self.inter_distances for ref in pair),
                ],
                names,
                _NAME,
            )
            if self.keypoints == [] and not self.inter_distances:
                raise ValueError(
                    f"{_NAME}: keypoints=[] places no keypoint and inter_distances "
                    "is empty, so there is nothing to compute. Leave keypoints unset "
                    "for every keypoint."
                )
            self._check_repeats(names)
            return self

        def _check_repeats(self, names: list[str] | None) -> None:
            """Refuse a keypoint or a distance named twice, and a one-keypoint heading.

            Index and name are one keypoint wherever they agree, so ``0`` and
            the first name are compared as the same reference.
            """

            check_keypoints_distinct(self.keypoints, names, _NAME)
            check_heading_pair(self.heading, names, _NAME)
            seen: set[tuple[int, int]] = set()
            for a_ref, b_ref in self.inter_distances:
                pair = (keypoint_index(a_ref, names), keypoint_index(b_ref, names))
                if pair in seen:
                    raise ValueError(
                        f"{_NAME}: inter-distance ({a_ref!r}, {b_ref!r}) is "
                        "requested twice."
                    )
                seen.add(pair)

    @staticmethod
    def temporal_reach(params: PairKeypoints.Params) -> int | None:
        """Return ``2 * window - 1``, the farthest frame a velocity reads.

        An edge row of a stretch is evaluated on the polynomial fitted to the
        stretch's first ``window`` rows, up to ``window - 1`` frames away. One
        of those rows can be interpolated across a gap of up to ``window``
        frames, from the known row past the gap, up to ``window`` frames
        further. Positions and the heading are read on the row itself, and the
        frame rate scales values without shifting a frame.
        """
        return 2 * params.window - 1

    def __init__(
        self,
        inputs: PairKeypoints.Inputs = Inputs(("tracks",)),
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
        kept_columns = [c for c in (C.group_col, C.seq_col) if c in df.columns]
        empty = pd.DataFrame(
            columns=[C.frame_col, "perspective", "id1", "id2", *kept_columns]
        )
        if df.empty:
            return empty
        ensure_columns(df, [C.frame_col, C.id_col])
        plan = self._plan(df)
        axis = frame_axis(df, self.name)
        fps = frame_rate(df, axis, self.params.fps, self.name)
        order = axis.order

        def column(name: str) -> npt.NDArray[np.object_]:
            return df[name].to_numpy(dtype=object)[order]

        rows = self._measure_rows(
            df, plan, order, RowLayout.build(axis, self.params.window), fps
        )
        ids = column(C.id_col)
        blocks = [column(c) for c in kept_columns]
        pairs = _pair_rows(axis, ids, blocks, id_arenas(df))
        if pairs is None:
            return empty
        focal, partner, perspective = pairs
        identity = pd.DataFrame(
            {
                C.frame_col: axis.frames[focal],
                "perspective": perspective,
                "id1": pd.Series(ids[focal]).infer_objects(),
                "id2": pd.Series(ids[partner]).infer_objects(),
                **{
                    name: pd.Series(values[focal]).infer_objects()
                    for name, values in zip(kept_columns, blocks, strict=True)
                },
            }
        )
        measures = pd.DataFrame(self._measure_pairs(plan, rows, focal, partner))
        out = pd.concat([identity, measures], axis=1)
        return out.sort_values([C.frame_col, "id1", "id2"], kind="stable").reset_index(
            drop=True
        )

    # --- Internal helpers ---

    def _plan(self, df: pd.DataFrame) -> _Plan:
        """Resolve the parameters against *df*'s keypoint columns."""
        p = self.params
        keypoints = keypoint_columns(
            df,
            p.keypoint_names,
            self.name,
            mixed_axes_hint="Drop the Z column to measure the table in the plane.",
            no_keypoints_hint=(
                "'pair-position' relates two individuals by their body centers."
            ),
        )
        if keypoints.three_d:
            raise ValueError(
                f"{self.name}: this table is 3D, and the body frame is planar. "
                "'keypoint-kinematics' with heading measures an animal's own "
                "keypoints in its body frame in 3D."
            )
        n_keypoints = len(keypoints.columns)

        def resolve(refs: list[int | str], what: str) -> list[int]:
            return resolve_keypoint_refs(
                refs, p.keypoint_names, n_keypoints, who=f"{self.name} {what}"
            )

        selected = (
            list(range(n_keypoints))
            if p.keypoints is None
            else resolve(list(p.keypoints), "keypoints")
        )
        neck, tail = resolve(list(p.heading), "heading")
        inter_distances: list[tuple[int, int]] = []
        for a_ref, b_ref in p.inter_distances:
            a, b = resolve([a_ref, b_ref], "inter_distances")
            inter_distances.append((a, b))
        return _Plan(
            columns=keypoints.columns,
            labels=keypoints.labels,
            selected=tuple(selected),
            heading=(neck, tail),
            inter_distances=tuple(inter_distances),
        )

    def _measure_rows(
        self,
        df: pd.DataFrame,
        plan: _Plan,
        order: IntArray,
        layout: RowLayout,
        fps: float,
    ) -> _Rows:
        """Positions, velocities and the body frame of every row, in axis order."""
        p = self.params
        needed = {
            *plan.selected,
            *plan.heading,
            *(k for pair in plan.inter_distances for k in pair),
        }
        positions: dict[int, FloatArray] = {}
        for k in sorted(needed):
            positions[k] = keypoint_positions(df, plan.columns[k], order)
        kernel = SavgolKernel.build(p.window, p.polyorder, 1, 1.0 / fps)
        velocities: dict[int, FloatArray] = {}
        for k in plan.selected:
            (velocity,), absent = savgol_derivatives(
                positions[k], layout, p.window, (kernel,)
            )
            velocity[absent] = np.nan
            velocities[k] = velocity
        neck, tail = plan.heading
        return _Rows(
            positions=positions,
            velocities=velocities,
            frame=body_frame(positions[neck], positions[tail]),
        )

    @staticmethod
    def _measure_pairs(
        plan: _Plan, rows: _Rows, focal: IntArray, partner: IntArray
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Every measure column of the pair rows, keyed by its name, in output order."""
        heading = rows.frame.heading[focal]
        center_x = rows.frame.center_x[focal]
        center_y = rows.frame.center_y[focal]
        columns: dict[str, dict[str, FloatArray]] = {
            prefix: {} for prefix in _PARTNER_PREFIXES
        }
        for k in plan.selected:
            label = plan.labels[k]
            position = rows.positions[k][partner]
            velocity = rows.velocities[k][partner]
            x, y = ego_rotate(
                position[:, 0] - center_x, position[:, 1] - center_y, heading
            )
            vx, vy = ego_rotate(velocity[:, 0], velocity[:, 1], heading)
            for prefix, values in zip(_PARTNER_PREFIXES, (x, y, vx, vy), strict=True):
                columns[prefix][f"{prefix}__{label}"] = values
        distances = {
            f"inter_dist__{plan.labels[a]}__{plan.labels[b]}": point_distances(
                rows.positions[a][focal], rows.positions[b][partner]
            )
            for a, b in plan.inter_distances
        }
        return {
            name: values.astype(np.float32)
            for name, values in {
                **{
                    name: values
                    for prefix in _PARTNER_PREFIXES
                    for name, values in columns[prefix].items()
                },
                **distances,
            }.items()
        }


def _pair_rows(
    axis: FrameAxis,
    ids: npt.NDArray[np.object_],
    blocks: list[npt.NDArray[np.object_]],
    arenas: dict[Hashable, str] | None,
) -> tuple[IntArray, IntArray, IntArray] | None:
    """The focal and partner row of every pair row, and its perspective.

    The individuals of one ``(group, sequence)`` block are paired by
    :func:`~mosaic.core.individuals.individual_pairs`, and each pair is joined on
    the frames both have. Each joined frame gives two rows: perspective 0 with
    the pair's first individual the focal, and perspective 1 with the second.

    Args:
        axis: The table's frame axis.
        ids: Each row's id, in axis order.
        blocks: Each row's group and sequence values, in axis order.
        arenas: Each id's arena, from :func:`~mosaic.core.individuals.id_arenas`.

    Returns:
        Positions on the axis of each pair row's focal and partner, and its
        perspective; ``None`` when no two individuals are paired.
    """
    starts, ends = axis.starts, axis.ends
    first_ids: list[Hashable] = ids[starts].tolist()
    block_values: list[list[object]] = [values[starts].tolist() for values in blocks]
    by_block: dict[tuple[object, ...], dict[Hashable, int]] = {}
    for code, identity in enumerate(first_ids):
        block = tuple(column[code] for column in block_values)
        by_block.setdefault(block, {})[identity] = code
    focal: list[IntArray] = []
    partner: list[IntArray] = []
    perspective: list[IntArray] = []
    for codes in by_block.values():
        for id_a, id_b in individual_pairs(list(codes), arenas):
            a, b = codes[id_a], codes[id_b]
            _, in_a, in_b = np.intersect1d(
                axis.frames[starts[a] : ends[a]],
                axis.frames[starts[b] : ends[b]],
                assume_unique=True,
                return_indices=True,
            )
            rows_a = (starts[a] + in_a).astype(np.int64)
            rows_b = (starts[b] + in_b).astype(np.int64)
            focal.extend((rows_a, rows_b))
            partner.extend((rows_b, rows_a))
            perspective.extend(
                (np.zeros(rows_a.size, np.int64), np.ones(rows_a.size, np.int64))
            )
    if not focal:
        return None
    return np.concatenate(focal), np.concatenate(partner), np.concatenate(perspective)
