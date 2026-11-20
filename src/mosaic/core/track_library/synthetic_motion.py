"""Simulates the motion a synthetic track file describes.

Each individual is a correlated random walk. Its speed is an Ornstein-Uhlenbeck
process around ``motion.mean_speed``, capped at ``motion.max_speed``. Its heading
is a unit vector that turns by a Gaussian draw in the plane orthogonal to it, and
the walk reflects at the arena's walls.

With ``spec.zones`` set, each frame first turns every heading toward a direction
its neighbors set, by the zones of repulsion, orientation and attraction of
Couzin, Krause, James, Ruxton and Franks (2002), "Collective memory and spatial
sorting in animal groups", J. Theor. Biol. 218:1-11. :func:`zone_step` gives the
rules. The step draws nothing, so the streams are those of independent walkers.
Neighbors come from dense pairwise distances, one ``individuals`` by
``individuals`` array per frame.

Random draws follow one rule: one stream per kind of draw, and a fixed count of
values per frame. :data:`STREAM_NAMES` names the eight streams, spawned from
``SeedSequence(seed)`` in that order. A chunk of ``n`` frames then consumes the
values that ``n`` single frames would, so the output does not depend on the chunk
size, and a draw added to one stream later leaves every other stream, and so every
trajectory, unchanged.

The simulation advances :data:`CHUNK_FRAMES` frames at a time and writes each
chunk's rows into column arrays allocated once at ``frames * individuals`` rows.
Besides the table, the generator holds one chunk of draws.

With ``spec.dropouts`` set, two-state chains decide which rows are written. Each
individual's chain, from the ``id_chain`` stream, takes a present id to missing
with probability ``id_rate`` per frame and back with probability
``1 / id_mean_frames``. One more chain over frames, from ``gap_chain`` with
``gap_rate`` and ``gap_mean_frames``, removes every row of a frame in a gap. The
streams draw a fixed count of uniforms per frame, burn-in included, so the chunk
size and the burn-in shift no draw. The chains start with every id present and run
through the burn-in, so the state entering frame 0 is their stationary draw once
the burn-in is long against the chains' mean runs. Only written rows are dropped:
positions, keypoints and confidence are computed for every individual and frame,
and a dropout changes none of them. With ``relabel_after_gap`` an individual
returning after a dropout or a frame gap takes a new id (see :func:`simulate`).

With ``spec.body.keypoints = K > 0`` each individual carries K keypoints on the
body axis. The axis is the heading ``u`` in force at the written position, after
any reflection at a wall, and the turn and step of the frame after it come later.
Keypoint ``k`` lies at the position plus ``(0.5 - k / (K - 1)) * length * u``, head
first and tail last, or at the position for ``K = 1``, plus Gaussian noise of
``keypoint_sd`` per axis. The position is then the mean of the noise-free
keypoints. A keypoint's confidence is ``clip(1 - |N(0, confidence_sd^2)|, 0, 1)``.
The ``keypoint`` and ``confidence`` streams draw for every individual in every
simulated frame, burn-in included, so a frame's draws do not depend on the rows
that are kept.

Imports numpy, the standard library and :mod:`mosaic.core.track_library.synthetic_spec`
at module level.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mosaic.core.track_library.synthetic_spec import (
    ArenaSpec,
    BodySpec,
    DropoutSpec,
    SyntheticSpec,
    ZoneSpec,
)

__all__ = [
    "CHUNK_FRAMES",
    "STREAM_NAMES",
    "SyntheticColumns",
    "simulate",
    "turn_headings",
    "zone_step",
]

CHUNK_FRAMES = 4_096
"""The frames the simulation advances between writes into the column arrays."""

STREAM_NAMES = (
    "start_position",
    "start_heading",
    "speed",
    "heading",
    "keypoint",
    "confidence",
    "id_chain",
    "gap_chain",
)
"""The random streams, in the order ``SeedSequence(seed).spawn`` hands them out."""

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
Bools = NDArray[np.bool_]


@dataclass(frozen=True)
class SyntheticColumns:
    """The generated rows as contiguous columns, ordered by frame and then id.

    Attributes:
        frame: Frame index, starting at 0 after the burn-in.
        id: Individual index.
        X: Position along the first axis.
        Y: Position along the second axis.
        Z: Position along the third axis, ``None`` for a 2D file.
        pose: The keypoint columns by name, empty without a body. Per keypoint
            ``k`` in order: ``poseX{k}``, ``poseY{k}``, ``poseZ{k}`` in 3D, and
            the confidence ``poseP{k}``.
    """

    frame: Ints
    id: Ints
    X: Floats
    Y: Floats
    Z: Floats | None
    pose: Mapping[str, Floats]

    def named_columns(self) -> Iterator[tuple[str, Ints | Floats]]:
        """Yield every column with its table name, in the table's column order.

        The order is ``frame``, ``id``, ``X``, ``Y``, ``Z`` in 3D, and then each
        keypoint's columns as :attr:`pose` lists them.
        """
        yield "frame", self.frame
        yield "id", self.id
        yield "X", self.X
        yield "Y", self.Y
        if self.Z is not None:
            yield "Z", self.Z
        yield from self.pose.items()


@dataclass(frozen=True)
class _Walls:
    """Where the walk reflects.

    Axes in ``box_axes`` reflect at ``0`` and at their length. ``disk_radius`` is
    set when the first two axes form a disk centered at ``(radius, radius)``.
    """

    box_axes: slice
    box_lengths: Floats
    disk_radius: float | None

    @classmethod
    def of(cls, arena: ArenaSpec, dimensions: int) -> _Walls:
        if arena.shape == "disk":
            assert arena.radius is not None
            lengths: list[float] = [] if dimensions == 2 else [_required(arena.depth)]
            return cls(slice(2, dimensions), np.array(lengths), arena.radius)
        lengths = [_required(arena.width), _required(arena.height)]
        if dimensions == 3:
            lengths.append(_required(arena.depth))
        return cls(slice(0, dimensions), np.array(lengths), None)


def _required(value: float | None) -> float:
    assert value is not None
    return value


def turn_headings(headings: Floats, noise: Floats) -> Floats:
    """Rotate each unit heading by a Gaussian step in its tangent plane.

    The part of ``noise`` along a heading is dropped, and the rest is the tangent
    step. The heading moves along the great circle through that step by its
    length, so in 2D this is the rotation by the signed angle the noise has
    orthogonal to the heading. The result is renormalized, which holds its length
    at 1 however many turns are chained.

    Args:
        headings: Unit vectors, shape ``(individuals, dimensions)``.
        noise: Step draws already scaled to the turning standard deviation, the
            same shape.

    Returns:
        The turned unit vectors.
    """
    tangent = noise - np.sum(noise * headings, axis=1, keepdims=True) * headings
    angle = np.linalg.norm(tangent, axis=1, keepdims=True)
    turned = np.cos(angle) * headings + np.sinc(angle / np.pi) * tangent
    return turned / np.linalg.norm(turned, axis=1, keepdims=True)


_COLLINEAR = 1e-12
"""The relative length below which a zone step's vector counts as zero."""


def zone_step(
    positions: Floats, headings: Floats, zones: ZoneSpec, dt: float
) -> Floats:
    """Turn each heading toward the direction its perceived neighbors set.

    Individual ``i`` perceives ``j != i`` when the angle between ``u_i`` and the
    offset ``r_ij = x_j - x_i`` is at most half of ``zones.field_of_view_deg``.
    A neighbor at ``i``'s own position has no direction and is not perceived.
    Over the perceived neighbors at distance ``d_ij``:

    - Any within ``repulsion`` set the desired direction ``-sum r_ij / d_ij``,
      and the other zones are ignored.
    - Otherwise ``d_o = u_i + sum u_j`` over those in ``(repulsion, orientation]``
      and ``d_a = sum r_ij / d_ij`` over those in ``(orientation, attraction]``.
      The desired direction is ``d_o + d_a``, or the nonempty zone's term alone.
    - With no neighbor in any zone, the heading is kept.

    The heading then rotates toward the desired direction along the great circle
    through both, by their angle or ``zones.max_turn_rate * dt``, whichever is
    smaller, through :func:`turn_headings`.

    Two cases are classified with the relative tolerance :data:`_COLLINEAR`, so
    rounding does not choose a turn. The desired direction is a sum of unit
    vectors, so its scale is their count. A desired direction no longer than
    the tolerance times that count cancels, and the heading is kept. Otherwise
    the part of the desired direction orthogonal to the heading is compared
    with the desired direction's length. Within the tolerance, a desired
    direction along the heading keeps it, and one opposite the heading leaves
    the plane of the turn undefined: the heading then turns toward the
    coordinate axis least aligned with it, the first such axis on a tie.

    Args:
        positions: Positions, shape ``(individuals, dimensions)``.
        headings: Unit headings, the same shape.
        zones: The zone radii, field of view and turn rate.
        dt: The frame interval in seconds.

    Returns:
        The turned unit headings.
    """
    offsets = positions[None, :, :] - positions[:, None, :]
    distance = np.sqrt(np.einsum("ijk,ijk->ij", offsets, offsets))
    nonzero = distance > 0
    directions = np.divide(
        offsets,
        distance[:, :, None],
        out=np.zeros_like(offsets),
        where=nonzero[:, :, None],
    )
    perceived = nonzero
    if zones.field_of_view_deg < 360:
        half_view = math.cos(math.radians(zones.field_of_view_deg) / 2)
        perceived = perceived & (
            np.einsum("ijk,ik->ij", directions, headings) >= half_view
        )
    repelled = perceived & (distance <= zones.repulsion)
    beyond = perceived & ~repelled
    oriented = beyond & (distance <= zones.orientation)
    attracted = beyond & (distance > zones.orientation) & (distance <= zones.attraction)
    has_orientation = oriented.any(axis=1, keepdims=True)
    has_attraction = attracted.any(axis=1, keepdims=True)
    repelled_count = repelled.sum(axis=1, keepdims=True)
    oriented_count = oriented.sum(axis=1, keepdims=True)
    attracted_count = attracted.sum(axis=1, keepdims=True)

    orientation_term = np.where(
        has_orientation, headings + oriented.astype(np.float64) @ headings, 0.0
    )
    attraction_term = np.einsum("ij,ijk->ik", attracted.astype(np.float64), directions)
    desired = np.where(
        has_orientation | has_attraction,
        orientation_term + attraction_term,
        headings,
    )
    terms = np.where(
        has_orientation | has_attraction,
        np.where(has_orientation, oriented_count + 1, 0) + attracted_count,
        1,
    )
    repelling = repelled_count > 0
    desired = np.where(
        repelling,
        -np.einsum("ij,ijk->ik", repelled.astype(np.float64), directions),
        desired,
    )
    terms = np.where(repelling, repelled_count, terms)

    desired_length = np.linalg.norm(desired, axis=1, keepdims=True)
    along = np.sum(desired * headings, axis=1, keepdims=True)
    tangent = desired - along * headings
    tangent_length = np.linalg.norm(tangent, axis=1, keepdims=True)
    angle = np.arctan2(tangent_length, along)
    cancelled = desired_length <= _COLLINEAR * terms
    collinear = cancelled | (tangent_length <= _COLLINEAR * desired_length)
    opposite = (collinear & ~cancelled & (along < 0))[:, 0]
    tangent = np.where(collinear, 0.0, tangent)
    tangent_length = np.where(collinear, 0.0, tangent_length)
    if opposite.any():
        reversed_headings = headings[opposite]
        axis = np.argmin(np.abs(reversed_headings), axis=1)
        basis = np.eye(headings.shape[1])[axis]
        normal = basis - np.sum(basis * reversed_headings, axis=1, keepdims=True) * (
            reversed_headings
        )
        tangent[opposite] = normal
        tangent_length[opposite] = np.linalg.norm(normal, axis=1, keepdims=True)
        angle[opposite] = np.pi
    turn = np.minimum(angle, zones.max_turn_rate * dt)
    step = np.divide(
        tangent * turn,
        tangent_length,
        out=np.zeros_like(tangent),
        where=tangent_length > 0,
    )
    return turn_headings(headings, step)


def _start_positions(spec: SyntheticSpec, uniform: Floats) -> Floats:
    """Place the individuals from uniform draws of shape ``(individuals, dimensions)``.

    ``arena.start_radius`` bounds the footprint only: x and y fall uniformly
    within it of the footprint's center, and in 3D z is uniform over
    ``[0, depth]`` whatever its value. Without it the start spans the whole
    footprint. A box footprint's center is ``(width / 2, height / 2)`` and a
    disk's is ``(radius, radius)``.
    """
    arena = spec.arena
    positions = np.empty_like(uniform)
    if arena.shape == "box":
        width, height = _required(arena.width), _required(arena.height)
        center = (width / 2, height / 2)
        start_radius = arena.start_radius
        if start_radius is None:
            positions[:, 0] = uniform[:, 0] * width
            positions[:, 1] = uniform[:, 1] * height
    else:
        radius = _required(arena.radius)
        center = (radius, radius)
        start_radius = radius if arena.start_radius is None else arena.start_radius
    if start_radius is not None:
        distance = start_radius * np.sqrt(uniform[:, 0])
        angle = 2 * np.pi * uniform[:, 1]
        positions[:, 0] = center[0] + distance * np.cos(angle)
        positions[:, 1] = center[1] + distance * np.sin(angle)
    if spec.dimensions == 3:
        positions[:, 2] = uniform[:, 2] * _required(arena.depth)
    return positions


def _reflect(positions: Floats, headings: Floats, walls: _Walls) -> None:
    """Reflect positions at the walls in place, flipping the heading to match.

    A step never exceeds the arena's smallest extent, so one reflection per axis
    returns every individual to the arena.
    """
    axes = walls.box_axes
    lengths = walls.box_lengths
    coordinate = positions[:, axes]
    if coordinate.shape[1]:
        below = coordinate < 0
        above = coordinate > lengths
        crossed = below | above
        if crossed.any():
            mirrored = np.where(
                below,
                -coordinate,
                np.where(above, 2 * lengths - coordinate, coordinate),
            )
            positions[:, axes] = np.clip(mirrored, 0.0, lengths)
            direction = headings[:, axes]
            headings[:, axes] = np.where(crossed, -direction, direction)
    radius = walls.disk_radius
    if radius is not None:
        relative = positions[:, :2] - radius
        distance = np.hypot(relative[:, 0], relative[:, 1])
        outside = distance > radius
        if outside.any():
            normal = relative[outside] / distance[outside, None]
            positions[outside, :2] = (
                radius + normal * (2 * radius - distance[outside])[:, None]
            )
            direction = headings[outside, :2]
            radial = np.sum(direction * normal, axis=1, keepdims=True)
            headings[outside, :2] = direction - 2 * radial * normal


_AXIS_NAMES = ("X", "Y", "Z")


@dataclass(frozen=True)
class _PoseColumns:
    """The preallocated keypoint columns of one sequence.

    ``axes[k]`` holds keypoint ``k``'s coordinate columns in axis order and
    ``confidence[k]`` its confidence column.
    """

    body: BodySpec
    axes: list[list[Floats]]
    confidence: list[Floats]

    @classmethod
    def allocate(cls, body: BodySpec, dimensions: int, rows: int) -> _PoseColumns:
        return cls(
            body,
            [
                [np.empty(rows) for _ in range(dimensions)]
                for _ in range(body.keypoints)
            ],
            [np.empty(rows) for _ in range(body.keypoints)],
        )

    def by_name(self, written: int) -> dict[str, Floats]:
        """The first ``written`` rows of each column, keyed by its table name."""
        named: dict[str, Floats] = {}
        for k, (coordinates, confidence) in enumerate(
            zip(self.axes, self.confidence, strict=True)
        ):
            for name, column in zip(_AXIS_NAMES, coordinates, strict=False):
                named[f"pose{name}{k}"] = column[:written]
            named[f"poseP{k}"] = confidence[:written]
        return named

    def write(
        self,
        rows: slice,
        cells: tuple[Ints, Ints],
        centers: Floats,
        headings: Floats,
        position_noise: Floats,
        confidence_noise: Floats,
    ) -> None:
        """Write the keypoints of the written rows of the given frames into ``rows``.

        Args:
            rows: The rows of the column arrays the written rows fill.
            cells: The frame index and the individual index of each written
                row, in row order.
            centers: Positions, shape ``(frames, individuals, dimensions)``.
            headings: The unit headings at those positions, the same shape.
            position_noise: Standard normal draws, shape
                ``(frames, individuals, keypoints, dimensions)``. The array is
                overwritten.
            confidence_noise: Standard normal draws, shape
                ``(frames, individuals, keypoints)``.
        """
        keypoints = self.body.keypoints
        length = _required(self.body.length)
        along = (
            0.5 - np.arange(keypoints) / (keypoints - 1)
            if keypoints > 1
            else np.zeros(1)
        ) * length
        position_noise *= self.body.keypoint_sd
        position_noise += centers[:, :, None, :]
        position_noise += along[None, None, :, None] * headings[:, :, None, :]
        confidence = np.clip(
            1.0 - np.abs(confidence_noise * self.body.confidence_sd), 0.0, 1.0
        )
        for k in range(keypoints):
            for axis, column in enumerate(self.axes[k]):
                column[rows] = position_noise[:, :, k, axis][cells]
            self.confidence[k][rows] = confidence[:, :, k][cells]


@dataclass
class _Chains:
    """The two-state chains that decide which individuals and frames are written.

    Every id starts present and no gap is open. ``advance`` takes a chunk's
    uniforms and returns the cells written.
    """

    id_leave: float
    id_return: float
    gap_start: float
    gap_end: float
    id_present: Bools
    in_gap: bool = False

    @classmethod
    def start(cls, dropouts: DropoutSpec, individuals: int) -> _Chains:
        return cls(
            dropouts.id_rate,
            1.0 / dropouts.id_mean_frames,
            dropouts.gap_rate,
            1.0 / dropouts.gap_mean_frames,
            np.ones(individuals, dtype=np.bool_),
        )

    def advance(self, id_uniform: Floats, gap_uniform: Floats) -> Bools:
        """Step the chains through ``frames`` frames.

        Args:
            id_uniform: Uniform draws, shape ``(frames, individuals)``.
            gap_uniform: Uniform draws, shape ``(frames,)``.

        Returns:
            Which cells of ``(frames, individuals)`` are written.
        """
        written = np.empty(id_uniform.shape, dtype=np.bool_)
        for k, (id_draw, gap_draw) in enumerate(
            zip(id_uniform, gap_uniform, strict=True)
        ):
            threshold: Floats = np.where(self.id_present, self.id_leave, self.id_return)
            self.id_present = self.id_present ^ (id_draw < threshold)
            self.in_gap = bool(
                gap_draw >= self.gap_end if self.in_gap else gap_draw < self.gap_start
            )
            written[k] = self.id_present & (not self.in_gap)
        return written


@dataclass
class _Relabeler:
    """The ids of the written rows when a returning individual takes a new one.

    The counter of unused ids starts at ``individuals`` and persists across
    chunks.
    """

    current: Ints
    absent: Bools
    next_id: int

    @classmethod
    def start(cls, individuals: int) -> _Relabeler:
        return cls(
            np.arange(individuals, dtype=np.int64),
            np.zeros(individuals, dtype=np.bool_),
            individuals,
        )

    def assign(self, written: Bools) -> Ints:
        """The id of every cell of ``written``, shape ``(frames, individuals)``.

        An individual absent from the previous frame (its own dropout, or a frame
        gap) and written now takes the next unused id, individuals in index order
        within a frame. The first frame follows no frame, so an individual
        written there keeps its index, and one missing there takes a new id at
        its first written row.
        """
        ids = np.empty(written.shape, dtype=np.int64)
        for k, live in enumerate(written):
            returning = live & self.absent
            returned = int(np.count_nonzero(returning))
            if returned:
                self.current[returning] = self.next_id + np.arange(returned)
                self.next_id += returned
            ids[k] = self.current
            self.absent = ~live
        return ids


def _rows_by_id(written: Bools, ids: Ints) -> tuple[Ints, Ints]:
    """The written cells ordered by frame and then id.

    Args:
        written: Which cells of ``(frames, individuals)`` are written.
        ids: The id of every cell, unique within a frame.

    Returns:
        The frame index and the individual index of each written cell.
    """
    per_frame: Ints = written.sum(axis=1)
    unwritten_last: Ints = np.where(written, ids, np.iinfo(np.int64).max)
    individual_by_id: Ints = np.argsort(unwritten_last, axis=1)
    frames: int = written.shape[0]
    individuals: int = written.shape[1]
    frame_index: Ints = np.repeat(np.arange(frames), per_frame)
    leading: Bools = np.arange(individuals) < per_frame[:, None]
    return frame_index, individual_by_id[leading]


def simulate(
    spec: SyntheticSpec,
    *,
    relabel_after_gap: bool = False,
    chunk_frames: int = CHUNK_FRAMES,
) -> SyntheticColumns:
    """Generate the rows of one synthetic sequence.

    The result depends on ``spec`` and ``relabel_after_gap`` alone, and not on
    ``chunk_frames``. The ``burn_in_frames`` first frames are simulated and not
    written, and the dropout chains run through them. A keypoint lies on the body
    axis of the heading that goes with its frame's position (see the module
    docstring).

    With ``relabel_after_gap``, an individual whose previous frame was missing for
    it, by its own dropout or because the whole frame was in a gap, takes the next
    unused id at its next written row. The ids count up from ``individuals``, and
    when several individuals return in one frame they take them in index order.
    Frame 0 follows no frame, so an individual missing at frame 0 returns at its
    first written row, and one written there keeps its own index. A burn-in frame
    is not written and relabels no one. Ids are then ``0`` to ``individuals - 1``
    without it.

    Args:
        spec: The validated synthetic track file.
        relabel_after_gap: Whether a returning individual takes a new id.
        chunk_frames: Frames advanced between writes into the column arrays.

    Returns:
        The columns, ordered by frame and then id, without the rows the dropouts
        remove. Each is a slice of an array allocated at ``frames * individuals``
        rows.
    """
    if chunk_frames < 1:
        msg = f"chunk_frames must be at least 1, got {chunk_frames}"
        raise ValueError(msg)
    individuals = spec.individuals
    dimensions = spec.dimensions
    motion = spec.motion
    walls = _Walls.of(spec.arena, dimensions)
    streams = dict(
        zip(
            STREAM_NAMES,
            (
                np.random.default_rng(child)
                for child in np.random.SeedSequence(spec.seed).spawn(len(STREAM_NAMES))
            ),
            strict=True,
        )
    )

    positions = _start_positions(
        spec, streams["start_position"].random((individuals, dimensions))
    )
    headings = streams["start_heading"].standard_normal((individuals, dimensions))
    headings /= np.linalg.norm(headings, axis=1, keepdims=True)
    speeds = np.full(individuals, motion.mean_speed)

    step_s = 1.0 / spec.fps
    decay = math.exp(-step_s / motion.speed_tau_s)
    speed_noise_scale = motion.speed_sd * math.sqrt(1.0 - decay * decay)
    turn_scale = motion.turning_sd * math.sqrt(step_s)
    zones = spec.zones

    total_rows = spec.frames * individuals
    frame_column = np.empty(total_rows, dtype=np.int64)
    id_column = np.empty(total_rows, dtype=np.int64)
    axis_columns = [np.empty(total_rows) for _ in range(dimensions)]
    chains = (
        None if spec.dropouts is None else _Chains.start(spec.dropouts, individuals)
    )
    relabeler = _Relabeler.start(individuals) if relabel_after_gap else None
    body = spec.body
    pose = (
        _PoseColumns.allocate(body, dimensions, total_rows)
        if body is not None and body.keypoints > 0
        else None
    )
    written = 0

    simulated_frames = spec.burn_in_frames + spec.frames
    for chunk_start in range(0, simulated_frames, chunk_frames):
        count = min(chunk_frames, simulated_frames - chunk_start)
        speed_draws = streams["speed"].standard_normal((count, individuals))
        heading_draws = streams["heading"].standard_normal(
            (count, individuals, dimensions)
        )
        heading_draws *= turn_scale
        pose_draws = None
        if pose is not None:
            pose_draws = (
                streams["keypoint"].standard_normal(
                    (count, individuals, pose.body.keypoints, dimensions)
                ),
                streams["confidence"].standard_normal(
                    (count, individuals, pose.body.keypoints)
                ),
            )
        present = (
            None
            if chains is None
            else chains.advance(
                streams["id_chain"].random((count, individuals)),
                streams["gap_chain"].random(count),
            )
        )
        track = np.empty((count, individuals, dimensions))
        track_headings = np.empty_like(track) if pose is not None else track
        for k in range(count):
            track[k] = positions
            if pose is not None:
                track_headings[k] = headings
            speeds = np.minimum(
                np.abs(
                    motion.mean_speed
                    + (speeds - motion.mean_speed) * decay
                    + speed_noise_scale * speed_draws[k]
                ),
                motion.max_speed,
            )
            if zones is not None:
                headings = zone_step(positions, headings, zones, step_s)
            headings = turn_headings(headings, heading_draws[k])
            positions += (speeds * step_s)[:, None] * headings
            _reflect(positions, headings, walls)

        first = min(max(spec.burn_in_frames - chunk_start, 0), count)
        kept_frames = count - first
        if kept_frames == 0:
            continue
        keep = (
            np.ones((kept_frames, individuals), dtype=np.bool_)
            if present is None
            else present[first:]
        )
        if relabeler is None:
            frame_index, individual = np.nonzero(keep)
            row_ids = individual
        else:
            labelled = relabeler.assign(keep)
            frame_index, individual = _rows_by_id(keep, labelled)
            row_ids = labelled[frame_index, individual]
        cells = (frame_index, individual)
        stop = written + frame_index.size
        first_frame = chunk_start + first - spec.burn_in_frames
        frame_column[written:stop] = first_frame + frame_index
        id_column[written:stop] = row_ids
        for axis, column in enumerate(axis_columns):
            column[written:stop] = track[first:, :, axis][cells]
        if pose is not None and pose_draws is not None:
            pose.write(
                slice(written, stop),
                cells,
                track[first:],
                track_headings[first:],
                pose_draws[0][first:],
                pose_draws[1][first:],
            )
        written = stop

    return SyntheticColumns(
        frame=frame_column[:written],
        id=id_column[:written],
        X=axis_columns[0][:written],
        Y=axis_columns[1][:written],
        Z=axis_columns[2][:written] if dimensions == 3 else None,
        pose={} if pose is None else pose.by_name(written),
    )
