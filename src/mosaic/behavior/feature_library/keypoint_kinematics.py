"""KeypointKinematics: per-keypoint speed and acceleration, and keypoint distances.

The movement-quality measures a phenotype is built from -- how fast a hand
moves, how sharply it accelerates, the rhythm of a cutting motion in a wrist's
vertical velocity, how far apart the two hands are -- are per-keypoint
quantities, where ``speed-angvel`` measures the body centre alone. This feature
measures them on every keypoint a table carries, in 2D or 3D, in the table's own
units.

Derivatives come from a Savitzky-Golay fit rather than a frame difference. A
difference of noisy positions amplifies the noise by the frame rate, and a
second difference by its square, so an acceleration from differences is mostly
detector jitter. The fit estimates the derivative of a local polynomial instead,
which is one smoothing, applied the same way to every keypoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Self, final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, model_validator
from scipy.ndimage import correlate1d
from scipy.signal import savgol_coeffs

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
from mosaic.core.pose_columns import (
    pose_column_pairs,
    pose_column_triplets,
    resolve_keypoint_refs,
)

from .helpers import ensure_columns, meta_columns
from .registry import register_feature

__all__ = ["KeypointKinematics"]

type _FloatArray = npt.NDArray[np.float64]
type _BoolArray = npt.NDArray[np.bool_]
type _IntArray = npt.NDArray[np.int64]

_NAME = "keypoint-kinematics"

_KEYPOINT_NAMES_DESCRIPTION = (
    "A name for each keypoint, in the table's keypoint order. Used in the output "
    "column names, and accepted wherever a parameter refers to a keypoint. Its "
    "length must equal the table's keypoint count. Unset, keypoint k is labelled "
    "kp{k} and is referred to by its index."
)

_KEYPOINTS_DESCRIPTION = (
    "The keypoints to report speed and acceleration for, by index or by name. "
    "An index is a keypoint's position in the table's keypoint order, which is "
    "its column number when the keypoints are numbered from 0 without a gap. "
    "Unset, every keypoint the table carries. An empty list reports none, for a "
    "run that wants only distances."
)

_FPS_DESCRIPTION = (
    "The recording's frame rate, which sets the time step of every derivative. "
    "Unset, it is the reciprocal of the median time step per frame read off the "
    "time column, and a table whose time column cannot give one is refused."
)

_WINDOW_DESCRIPTION = (
    "The Savitzky-Golay window length: odd, and larger than polyorder. It is also "
    "the longest gap in a keypoint's track that is linearly interpolated before "
    "filtering. A longer gap ends the stretch the filter runs over."
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
    "sequence divides every distance, speed, vertical velocity and acceleration, "
    "making them independent of body size. Unset, they stay in the table's units. "
    "An individual on which the pair is never seen reports NaN throughout."
)


@dataclass(frozen=True)
class _Plan:
    """What one table asks for, resolved against its keypoint columns.

    Attributes:
        columns: The coordinate columns of every keypoint the table carries, in
            keypoint order: pairs on a 2D table, triplets on a 3D one.
        labels: Each keypoint's label in an output column name.
        selected: The keypoints whose speed and acceleration are reported.
        distances: The keypoint pairs whose separation is reported.
        normalize_by: The pair whose median separation divides every length.
    """

    columns: tuple[tuple[str, ...], ...]
    labels: tuple[str, ...]
    selected: tuple[int, ...]
    distances: tuple[tuple[int, int], ...]
    normalize_by: tuple[int, int] | None

    @property
    def three_d(self) -> bool:
        return len(self.columns[0]) == 3


@dataclass(frozen=True)
class _Individuals:
    """The table's rows in output order, and where each individual's rows lie.

    Attributes:
        order: The permutation of the input rows into output order: by group,
            sequence and id, then by frame.
        frames: The frame of each row, in that order.
        starts: The first row of each individual.
        ends: One past its last row.
        codes: The individual each row belongs to, ``0..n_individuals-1``.
    """

    order: _IntArray
    frames: _IntArray
    starts: _IntArray
    ends: _IntArray
    codes: _IntArray


@dataclass(frozen=True)
class _Layout:
    """Every individual's rows placed on one axis, which the filter runs along.

    Each individual is reindexed onto its contiguous frame range, and the ranges
    are laid end to end, so that one filter pass serves every individual: a
    table of tracklets holds tens of thousands of them, and a pass per
    individual and keypoint cost far more than the arithmetic. Where a gap
    between two rows is too long to bridge -- and between two individuals --
    a single *break* position stands in for it. A break is missing, so it ends
    the stretch on either side, and no gap holding one is interpolated across.
    That keeps the axis no longer than the rows plus the gaps short enough to
    bridge, however sparse the frame numbers are.

    Attributes:
        rows: The axis position of each row, in output order.
        size: The axis length.
        breaks: ``True`` at each break position.
    """

    rows: _IntArray
    size: int
    breaks: _BoolArray

    @classmethod
    def build(cls, individuals: _Individuals, longest_gap: int) -> _Layout:
        frames = individuals.frames
        steps = np.diff(frames)
        unbridged = (individuals.codes[1:] != individuals.codes[:-1]) | (
            steps - 1 > longest_gap
        )
        steps = np.where(unbridged, 2, steps)
        rows = np.concatenate(([0], np.cumsum(steps))).astype(np.int64)
        size = int(rows[-1]) + 1
        breaks = np.zeros(size, dtype=bool)
        breaks[rows[1:][unbridged] - 1] = True
        return cls(rows=rows, size=size, breaks=breaks)


@dataclass(frozen=True)
class _Kernel:
    """The Savitzky-Golay coefficients for one derivative, computed once per table.

    Attributes:
        centre: Dotted with a window to estimate the derivative at its centre.
        lead: One row per position among a run's first ``window // 2`` rows,
            dotted with the run's first ``window`` rows.
        trail: The same for its last ``window // 2`` rows and last window.
    """

    centre: _FloatArray
    lead: _FloatArray
    trail: _FloatArray

    @classmethod
    def build(cls, window: int, polyorder: int, deriv: int, delta: float) -> _Kernel:
        half = window // 2

        def at(pos: int | None) -> _FloatArray:
            return np.asarray(
                savgol_coeffs(
                    window, polyorder, deriv=deriv, delta=delta, pos=pos, use="dot"
                ),
                dtype=np.float64,
            )

        return cls(
            centre=at(None),
            lead=np.stack([at(j) for j in range(half)]),
            trail=np.stack([at(window - half + j) for j in range(half)]),
        )


def _runs(mask: _BoolArray) -> tuple[_IntArray, _IntArray]:
    """The start and exclusive end of each run of ``True`` in *mask*."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1]).astype(np.int64)
    return edges[0::2], edges[1::2]


def _cover(starts: _IntArray, ends: _IntArray, n: int) -> _BoolArray:
    """A length-*n* mask that is ``True`` on every ``[start, end)`` interval."""
    steps = np.zeros(n + 1, dtype=np.int64)
    np.add.at(steps, starts, 1)
    np.add.at(steps, ends, -1)
    return np.cumsum(steps[:n]) > 0


def _fill_short_gaps(
    positions: _FloatArray, missing: _BoolArray, longest: int, breaks: _BoolArray
) -> tuple[_FloatArray, _BoolArray]:
    """Linearly interpolate every interior gap of at most *longest* rows.

    A gap touching either end of the axis, or holding a break, is left alone:
    at an end there is only one side to interpolate from, and a break stands for
    a gap too long to bridge or for the boundary between two individuals.

    Returns:
        The positions with those gaps filled, and what is still missing.
    """
    n = missing.size
    starts, ends = _runs(missing)
    broken = np.concatenate(([0], np.cumsum(breaks, dtype=np.int64)))
    short = (
        (starts > 0)
        & (ends < n)
        & (ends - starts <= longest)
        & (broken[ends] == broken[starts])
    )
    if not short.any():
        return positions, missing
    starts, ends = starts[short], ends[short]
    lengths = ends - starts
    # The rows of every gap, flattened, each paired with the known rows either
    # side of its gap: one interpolation for all gaps rather than a search.
    first_of_gap = np.repeat(np.cumsum(lengths) - lengths, lengths)
    rows = np.repeat(starts, lengths) + np.arange(int(lengths.sum())) - first_of_gap
    before = np.repeat(starts - 1, lengths)
    after = np.repeat(ends, lengths)
    fraction = ((rows - before) / (after - before))[:, None]
    filled = positions.copy()
    filled[rows] = positions[before] + fraction * (positions[after] - positions[before])
    still_missing = missing.copy()
    still_missing[rows] = False
    return filled, still_missing


def _savgol_on_runs(
    positions: _FloatArray, valid: _BoolArray, kernels: tuple[_Kernel, ...]
) -> list[_FloatArray]:
    """``savgol_filter(mode="interp")`` applied to each run of valid rows.

    A run is filtered on its own, so no window reaches across a gap. The
    interior of a run is the centred filter; the first and last ``window // 2``
    rows of it are evaluated on the polynomial fitted to its first or last
    ``window`` rows, which is what ``mode="interp"`` does at an array's ends. A
    run shorter than the window has none to fit and stays NaN, as does every
    invalid row.

    Vectorised over runs, since a poorly tracked keypoint can break into tens of
    thousands of them, and one filter call per run would cost far more than the
    arithmetic does.

    Returns:
        One ``(n_rows, n_axes)`` array per kernel, in the kernels' order.
    """
    n, n_axes = positions.shape
    window = kernels[0].centre.size
    half = window // 2
    starts, ends = _runs(valid)
    keep = ends - starts >= window
    starts, ends = starts[keep], ends[keep]
    if starts.size == 0:
        return [np.full((n, n_axes), np.nan) for _ in kernels]
    # Zero rather than NaN outside the runs: every window kept below lies wholly
    # inside one run, so the zeros never reach a result, while a NaN would make
    # the result depend on whether the arithmetic multiplies it by a zero
    # coefficient (the centre coefficient of a first derivative is zero).
    filled = np.where(valid[:, None], positions, 0.0)
    outside = np.flatnonzero(~_cover(starts + half, ends - half, n))
    offsets = np.arange(window)
    edge = np.arange(half)
    head = filled[starts[:, None] + offsets]
    tail = filled[(ends - window)[:, None] + offsets]
    head_rows = starts[:, None] + edge
    tail_rows = (ends - half)[:, None] + edge
    outputs: list[_FloatArray] = []
    for kernel in kernels:
        out = correlate1d(filled, kernel.centre, axis=0, mode="constant", cval=0.0)
        out[outside] = np.nan
        out[head_rows] = np.einsum("jw,rwd->rjd", kernel.lead, head)
        out[tail_rows] = np.einsum("jw,rwd->rjd", kernel.trail, tail)
        outputs.append(out)
    return outputs


def _derivatives(
    observed: _FloatArray, layout: _Layout, window: int, kernels: tuple[_Kernel, ...]
) -> tuple[list[_FloatArray], _BoolArray]:
    """One keypoint of every individual, differentiated along *layout*.

    Args:
        observed: ``(n_rows, n_axes)`` positions in output order, NaN where not
            finite.
        layout: Where each row sits on the axis the filter runs along.
        window: The longest gap to interpolate across.
        kernels: The derivatives to take.

    Returns:
        One ``(n_rows, n_axes)`` array per kernel on the input rows, and which
        input rows were missing the keypoint. Those rows are not yet masked.
    """
    absent = ~np.isfinite(observed).all(axis=1)
    contiguous = layout.size == observed.shape[0]
    if contiguous:
        dense, missing = observed, absent
    else:
        dense = np.full((layout.size, observed.shape[1]), np.nan)
        dense[layout.rows] = observed
        missing = np.ones(layout.size, dtype=bool)
        missing[layout.rows] = absent
    dense, missing = _fill_short_gaps(dense, missing, window, layout.breaks)
    derived = _savgol_on_runs(dense, ~missing, kernels)
    if not contiguous:
        derived = [values[layout.rows] for values in derived]
    return derived, absent


def _row_norm(vectors: _FloatArray) -> _FloatArray:
    """The Euclidean length of each row."""
    return np.sqrt(np.einsum("ij,ij->i", vectors, vectors))


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

    and for each requested pair, ``dist__{a}__{b}``, the Euclidean distance
    between two keypoints on that row. ``{kp}`` is the keypoint's name when
    ``keypoint_names`` is given, else ``kp{k}`` with ``k`` its position in the
    table's keypoint order. Every measure is ``float32``.

    **2D or 3D, decided by the table.** A table carrying ``Z`` or ``poseZ*``
    columns is measured on its keypoint triplets, and one carrying neither on its
    ``poseX*``/``poseY*`` pairs. Lengths are in the table's own unit: video
    pixels, centimetres, or the metres of a 3D producer. A table with no keypoints is
    refused rather than measured on its body centre, which ``speed-angvel``
    already does.

    **Method.** Each individual's rows are sorted by frame and placed on its
    contiguous frame range, frames it lacks reading as missing. A keypoint is
    missing on a row where any of its coordinates is not finite. Gaps of at most
    ``window`` frames are linearly interpolated, so a brief dropout does not cut
    the filter short; a longer gap splits the track into stretches that are
    filtered separately. Each coordinate is then differentiated once and twice by
    a Savitzky-Golay fit with the time step ``1 / fps``, which near the ends of a
    stretch evaluates the polynomial fitted to its first or last window, as
    ``scipy.signal.savgol_filter(mode="interp")`` does.

    **What is NaN.** A keypoint's measures are NaN on every row where it was
    missing in the input, interpolated or not: an interpolated position is there
    to keep the filter running, and is not a measurement. They are also NaN on a
    stretch shorter than ``window``, which has no window to fit. Rows are emitted
    for exactly the input's ``(frame, id)`` rows; the frames the reindexing added
    are dropped again.

    **Normalization.** With ``normalize_by``, every distance, speed, vertical
    velocity and acceleration is divided by the median length of that keypoint
    segment over the individual's rows in the sequence, which makes them
    comparable across body sizes. A distance is then in segment lengths, a speed
    in segment lengths per second, an acceleration per second squared.

    One row per ``(frame, id)`` is required, so a table holding one row per
    camera is refused. Field documentation is on
    :class:`~mosaic.behavior.feature_library.keypoint_kinematics.KeypointKinematics.Params`.
    """

    category = "per-frame"
    name = _NAME
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False  # one individual's frame axis is one entry's
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
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
            Conditional(Prefix("dist__"), when_param="distances"),
            *META_COLUMNS_KEPT,
        ),
    )

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        """Keypoint-kinematics parameters."""

        keypoint_names: Annotated[
            list[str] | None, Declared(_KEYPOINT_NAMES_DESCRIPTION)
        ] = None
        keypoints: Annotated[
            list[int | str] | None, Declared(_KEYPOINTS_DESCRIPTION)
        ] = None
        fps: Annotated[float | None, Declared(_FPS_DESCRIPTION, unit="fps")] = Field(
            default=None, gt=0
        )
        window: Annotated[int, Declared(_WINDOW_DESCRIPTION, unit="frames")] = Field(
            default=7, ge=3
        )
        polyorder: Annotated[int, Declared(_POLYORDER_DESCRIPTION)] = Field(
            default=2, ge=2
        )
        distances: Annotated[
            list[tuple[int | str, int | str]], Declared(_DISTANCES_DESCRIPTION)
        ] = Field(default_factory=list)
        normalize_by: Annotated[
            tuple[int | str, int | str] | None, Declared(_NORMALIZE_BY_DESCRIPTION)
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
            if self.window % 2 == 0:
                raise ValueError(
                    f"{_NAME}: window={self.window} must be odd, so that each "
                    "window is centred on the frame it estimates."
                )
            if self.window <= self.polyorder:
                raise ValueError(
                    f"{_NAME}: window={self.window} must exceed "
                    f"polyorder={self.polyorder}; a window no longer than the "
                    "polynomial's order fits it exactly and smooths nothing."
                )
            names = self.keypoint_names
            if names is not None:
                if len(set(names)) != len(names):
                    raise ValueError(
                        f"{_NAME}: keypoint_names repeats a name, {names}; each "
                        "name labels output columns of its own."
                    )
                bad = [name for name in names if not name or "__" in name]
                if bad:
                    raise ValueError(
                        f"{_NAME}: keypoint names {bad} are empty or contain "
                        "'__', which separates the parts of an output column name."
                    )
            refs: list[int | str] = [
                *(self.keypoints or []),
                *(ref for pair in self.distances for ref in pair),
                *(self.normalize_by or ()),
            ]
            negative = [ref for ref in refs if isinstance(ref, int) and ref < 0]
            if negative:
                raise ValueError(
                    f"{_NAME}: keypoint indices {negative} are negative; a keypoint "
                    "is referred to by its position from 0, or by name."
                )
            if names is not None:
                # The table must carry one keypoint per name, so the names
                # bound every index as well.
                _ = resolve_keypoint_refs(refs, names, len(names), who=_NAME)
            else:
                named = [ref for ref in refs if isinstance(ref, str)]
                _ = resolve_keypoint_refs(named, None, 0, who=_NAME)
            if self.keypoints == [] and not self.distances:
                raise ValueError(
                    f"{_NAME}: keypoints=[] reports no keypoint and distances is "
                    "empty, so there is nothing to compute. Leave keypoints unset "
                    "for every keypoint."
                )
            self._check_repeats(names or [])
            return self

        def _check_repeats(self, names: list[str]) -> None:
            """Refuse a keypoint, or a pair, that is named twice or paired with itself.

            Index and name are one keypoint wherever they agree, so ``0`` and
            the first name are compared as the same reference. Nothing here
            depends on the table.
            """

            def index(ref: int | str) -> int:
                return names.index(ref) if isinstance(ref, str) else ref

            selected = [index(ref) for ref in self.keypoints or []]
            if len(set(selected)) != len(selected):
                raise ValueError(
                    f"{_NAME}: keypoints names a keypoint twice, {self.keypoints}."
                )
            seen: set[frozenset[int]] = set()
            for a_ref, b_ref in self.distances:
                pair = frozenset((index(a_ref), index(b_ref)))
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
                if index(a_ref) == index(b_ref):
                    raise ValueError(
                        f"{_NAME}: normalize_by {self.normalize_by} is a keypoint "
                        "and itself, whose length is zero."
                    )

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
        individuals = self._individuals(df)
        fps = self._fps(df, individuals)
        measures = self._measure(df, plan, individuals, fps)
        meta = df[meta_columns(df)].iloc[individuals.order].reset_index(drop=True)
        return pd.concat([meta, pd.DataFrame(measures)], axis=1)

    # --- Internal helpers ---

    def _plan(self, df: pd.DataFrame) -> _Plan:
        """Resolve the parameters against *df*'s keypoint columns."""
        p = self.params
        names = [str(column) for column in df.columns]
        pairs = pose_column_pairs(names)
        triplets = pose_column_triplets(names)
        three_d = "Z" in names or any(name.startswith("poseZ") for name in names)
        if three_d and len(triplets) != len(pairs):
            raise ValueError(
                f"{self.name}: this table carries a Z axis, but "
                f"{len(pairs) - len(triplets)} of its {len(pairs)} keypoints have "
                "no poseZ column. Measuring them in the plane would mix 2D and 3D "
                "speeds in one output."
            )
        columns: list[tuple[str, ...]] = [*triplets] if three_d else [*pairs]
        if not columns:
            raise ValueError(
                f"{self.name} needs pose keypoint columns (poseX*/poseY*, and "
                "poseZ* on a 3D table) and this table carries none. The speed of "
                "a body centre is what 'speed-angvel' measures."
            )
        n_keypoints = len(columns)
        if p.keypoint_names is not None and len(p.keypoint_names) != n_keypoints:
            raise ValueError(
                f"{self.name}: keypoint_names has {len(p.keypoint_names)} names "
                f"and the table carries {n_keypoints} keypoints; a name is matched "
                "to a keypoint by position, so the two must agree."
            )
        labels = (
            tuple(p.keypoint_names)
            if p.keypoint_names is not None
            else tuple(f"kp{k}" for k in range(n_keypoints))
        )

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
        return _Plan(
            columns=tuple(columns),
            labels=labels,
            selected=tuple(selected),
            distances=tuple(distances),
            normalize_by=normalize_by,
        )

    def _individuals(self, df: pd.DataFrame) -> _Individuals:
        """Order the rows by individual, then frame, and find each one's rows.

        Ordered through a permutation rather than a sort of the frame, which
        would copy every column of a table whose keypoints are read one at a
        time anyway.
        """
        keys = [c for c in (C.group_col, C.seq_col) if c in df.columns] + [C.id_col]
        raw: _FloatArray = df[C.frame_col].to_numpy(dtype=np.float64)
        if not np.isfinite(raw).all() or (raw != np.floor(raw)).any():
            raise ValueError(
                f"{self.name}: the '{C.frame_col}' column holds values that are "
                "not whole frame numbers, so there is no frame axis to place rows "
                "on."
            )
        frames = raw.astype(np.int64)
        codes = (
            df.groupby(keys, sort=True, dropna=False).ngroup().to_numpy(dtype=np.int64)
        )
        order = np.lexsort((frames, codes)).astype(np.int64)
        frames, codes = frames[order], codes[order]
        same = np.diff(codes) == 0
        repeated = np.flatnonzero(same & (np.diff(frames) == 0))
        if repeated.size:
            first = int(order[repeated[0]])
            hint = (
                " This table has a camera column; a table holding one row per "
                "camera measures each camera's view apart, and this feature "
                "does not."
                if "camera" in df.columns
                else ""
            )
            raise ValueError(
                f"{self.name}: id {df[C.id_col].iloc[first]!r} has "
                f"{repeated.size} repeated frames, the first at frame "
                f"{df[C.frame_col].iloc[first]}. A derivative needs one position "
                f"per frame, so one row per (frame, id).{hint}"
            )
        boundaries = np.flatnonzero(~same) + 1
        starts = np.concatenate(([0], boundaries)).astype(np.int64)
        ends = np.concatenate((boundaries, [frames.size])).astype(np.int64)
        return _Individuals(
            order=order,
            frames=frames,
            starts=starts,
            ends=ends,
            codes=np.repeat(np.arange(starts.size, dtype=np.int64), ends - starts),
        )

    def _fps(self, df: pd.DataFrame, individuals: _Individuals) -> float:
        """The frame rate: the parameter, or else the time column's median step."""
        if self.params.fps is not None:
            return self.params.fps
        if C.time_col not in df.columns:
            raise ValueError(
                f"{self.name} needs a frame rate: pass fps, or give the table a "
                f"'{C.time_col}' column to read it from."
            )
        times = df[C.time_col].to_numpy(dtype=np.float64)[individuals.order]
        # A step across two individuals' rows is not a time step.
        within = np.ones(max(times.size - 1, 0), dtype=bool)
        within[individuals.ends[:-1] - 1] = False
        frame_steps = np.diff(individuals.frames)
        time_steps = np.diff(times)
        usable = within & (frame_steps > 0) & np.isfinite(time_steps)
        if usable.any():
            period = float(np.median(time_steps[usable] / frame_steps[usable]))
            if np.isfinite(period) and period > 0:
                return 1.0 / period
        raise ValueError(
            f"{self.name} could not read a frame rate off the '{C.time_col}' "
            "column: no two consecutive rows of an individual give a positive "
            "time step per frame. Pass fps."
        )

    def _measure(
        self,
        df: pd.DataFrame,
        plan: _Plan,
        individuals: _Individuals,
        fps: float,
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Every output column, in output row order, keyed by its name."""
        p = self.params
        order = individuals.order

        def positions(k: int) -> _FloatArray:
            values = df[list(plan.columns[k])].to_numpy(dtype=np.float64)[order]
            values[~np.isfinite(values)] = np.nan
            return values

        # One divisor per row: the individual's median segment length, or 1.
        scale = np.ones(order.size)
        if plan.normalize_by is not None:
            a, b = plan.normalize_by
            segment = pd.Series(_row_norm(positions(a) - positions(b)))
            # NaN for an individual on which the segment is never seen.
            medians = segment.groupby(individuals.codes).median()
            per_individual = medians.to_numpy(dtype=np.float64, copy=True)
            per_individual[~(per_individual > 0)] = np.nan
            scale = per_individual[individuals.codes]

        kernels = (
            _Kernel.build(p.window, p.polyorder, 1, 1.0 / fps),
            _Kernel.build(p.window, p.polyorder, 2, 1.0 / fps),
        )
        layout = _Layout.build(individuals, p.window)
        labels = plan.labels
        speed: dict[str, _FloatArray] = {}
        accel: dict[str, _FloatArray] = {}
        vertical: dict[str, _FloatArray] = {}
        for k in plan.selected:
            (velocity, acceleration), absent = _derivatives(
                positions(k), layout, p.window, kernels
            )
            label = labels[k]
            speed[f"speed__{label}"] = (
                np.where(absent, np.nan, _row_norm(velocity)) / scale
            )
            accel[f"accel__{label}"] = (
                np.where(absent, np.nan, _row_norm(acceleration)) / scale
            )
            if plan.three_d:
                vertical[f"vz__{label}"] = (
                    np.where(absent, np.nan, velocity[:, 2]) / scale
                )
        dist = {
            f"dist__{labels[a]}__{labels[b]}": _row_norm(positions(a) - positions(b))
            / scale
            for a, b in plan.distances
        }
        return {
            name: values.astype(np.float32)
            for name, values in {**speed, **accel, **vertical, **dist}.items()
        }
