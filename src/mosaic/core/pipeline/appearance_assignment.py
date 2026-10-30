"""Assign tracklets of an entry to identity prototypes by appearance.

A prototype is the normalized mean of an identity's normalized reference vectors.
A source track splits into tracklets at gaps longer than ``max_gap`` frames, and
each tracklet is scored by its mean cosine to every prototype. :func:`assign`
gives each tracklet its best identity unless a threshold fails or a co-present
tracklet already holds that identity. Sorted keys fix every order, and the result
does not depend on row order. The module uses numpy alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatMatrix = npt.NDArray[np.floating]
"""Vectors in rows, any float dtype."""

IntArray = npt.NDArray[np.integer]
"""Integer frame numbers or track ids."""


@dataclass(frozen=True)
class Prototypes:
    """The identity prototypes, one unit row per identity.

    Attributes:
        names: The identity names, sorted. Row ``i`` of ``matrix`` belongs to
            ``names[i]``, and an identity's index is its position here.
        matrix: The float32 ``(k, D)`` prototype matrix.
    """

    names: tuple[str, ...]
    matrix: npt.NDArray[np.float32]


@dataclass(frozen=True)
class Tracklet:
    """A run of one source track's rows without a gap longer than ``max_gap``.

    Attributes:
        source_id: The source track id.
        first_frame: The first frame of the run.
        last_frame: The last frame of the run.
        rows: The positions of the run's rows in the input arrays, in frame order.
        frames: The frames of those rows, ascending and distinct.
    """

    source_id: int
    first_frame: int
    last_frame: int
    rows: npt.NDArray[np.intp]
    frames: npt.NDArray[np.int64]

    @property
    def length(self) -> int:
        """The number of rows."""
        return len(self.rows)


@dataclass(frozen=True)
class Assignment:
    """The outcome of one tracklet.

    Attributes:
        tracklet: The tracklet.
        identity: The index of the assigned identity, or ``None`` when the
            tracklet abstains.
        best: The index of the best-scoring identity.
        score: The best score.
        margin: The best score minus the second best.
    """

    tracklet: Tracklet
    identity: int | None
    best: int
    score: float
    margin: float


def _unit_rows(vectors: FloatMatrix, what: str) -> npt.NDArray[np.float64]:
    """Return *vectors* as float64 rows of unit length.

    Raises:
        ValueError: When *vectors* is not a matrix or a row is not finite or zero.
    """
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{what}: expected a (n, D) matrix, got shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{what}: a vector is not finite")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms == 0.0).any():
        raise ValueError(f"{what}: a vector has zero length")
    return matrix / norms


def prototypes(vectors_by_identity: Mapping[str, FloatMatrix]) -> Prototypes:
    """Build one prototype per identity from its reference vectors.

    Each vector is normalized, the identity's mean is taken in float64 and the
    mean is normalized. The rows are summed in lexicographic order of their
    values. The prototype then does not depend on the reference row order, and
    a caller need not sort the rows by ``(group, sequence, frame, id)``.

    Args:
        vectors_by_identity: Each identity's reference vectors, ``(n, D)``.

    Returns:
        The prototypes, in sorted name order.

    Raises:
        ValueError: When there is no identity, a name is empty, an identity has
            no vector, a vector is zero or not finite, or the dimensions differ.
    """
    if not vectors_by_identity:
        raise ValueError("prototypes need at least one identity")
    names = tuple(sorted(vectors_by_identity))
    rows: list[npt.NDArray[np.float64]] = []
    for name in names:
        if not name:
            raise ValueError("an identity name is empty")
        unit = _unit_rows(vectors_by_identity[name], f"identity {name!r}")
        if len(unit) == 0:
            raise ValueError(f"identity {name!r} has no reference vector")
        if rows and unit.shape[1] != rows[0].shape[0]:
            raise ValueError(
                f"identity {name!r} has vectors of dimension {unit.shape[1]}, "
                f"expected {rows[0].shape[0]}"
            )
        ordered = unit[np.lexsort(unit.T[::-1])]
        mean = ordered.sum(axis=0) / len(ordered)
        rows.append(mean / np.linalg.norm(mean))
    return Prototypes(names=names, matrix=np.stack(rows).astype(np.float32))


def tracklets(frame: IntArray, id: IntArray, *, max_gap: int) -> list[Tracklet]:
    """Split each source track into tracklets at gaps longer than *max_gap*.

    Args:
        frame: The frame of each row.
        id: The source track id of each row.
        max_gap: The largest frame difference between consecutive rows of one
            tracklet.

    Returns:
        The tracklets, sorted by ``(source_id, first_frame)``.

    Raises:
        ValueError: When *max_gap* is negative, the arrays differ in length, or
            one id has two rows in one frame.
    """
    if max_gap < 0:
        raise ValueError(f"max_gap must be at least 0, got {max_gap}")
    frames = np.asarray(frame, dtype=np.int64)
    ids = np.asarray(id, dtype=np.int64)
    if frames.shape != ids.shape or frames.ndim != 1:
        raise ValueError("frame and id must be one-dimensional and of equal length")
    order = np.lexsort((frames, ids))
    sorted_frames = frames[order]
    sorted_ids = ids[order]
    same_id = sorted_ids[1:] == sorted_ids[:-1]
    steps = np.diff(sorted_frames)
    repeated = np.flatnonzero(same_id & (steps == 0))
    if len(repeated):
        position = repeated[0]
        raise ValueError(
            f"id {sorted_ids[position]} has two rows in frame {sorted_frames[position]}"
        )
    breaks = np.flatnonzero(~same_id | (steps > max_gap)) + 1
    result: list[Tracklet] = []
    for run in np.split(np.arange(len(order)), breaks):
        if len(run) == 0:
            continue
        run_frames = sorted_frames[run]
        result.append(
            Tracklet(
                source_id=int(sorted_ids[run[0]]),
                first_frame=int(run_frames[0]),
                last_frame=int(run_frames[-1]),
                rows=order[run].astype(np.intp),
                frames=run_frames,
            )
        )
    return result


def score(
    tracklet_vectors: FloatMatrix, prototypes: Prototypes
) -> npt.NDArray[np.float64]:
    """Return the mean cosine of a tracklet's vectors to each prototype.

    Each detection counts once, accumulated in float64.

    Args:
        tracklet_vectors: The tracklet's vectors, ``(n, D)`` with ``n >= 1``.
        prototypes: The prototypes to score against.

    Returns:
        The float64 scores, one per identity in ``prototypes.names`` order.

    Raises:
        ValueError: When there is no vector, a vector is zero or not finite, or
            the dimension differs from the prototypes'.
    """
    unit = _unit_rows(tracklet_vectors, "tracklet")
    dimension = prototypes.matrix.shape[1]
    if len(unit) == 0 or unit.shape[1] != dimension:
        raise ValueError(
            f"tracklet vectors of shape {unit.shape} do not match prototypes of "
            f"dimension {dimension}"
        )
    return (unit @ prototypes.matrix.astype(np.float64).T).mean(axis=0)


def _shared_frame(frames: npt.NDArray[np.int64], taken: set[int]) -> int | None:
    """Return the first of *frames* in *taken*, or ``None``."""
    return next((value for value in frames.tolist() if value in taken), None)


def assign(
    entry_tracklets: Sequence[Tracklet],
    scores: FloatMatrix,
    *,
    min_similarity: float,
    min_margin: float,
    declared: Mapping[int, int] | None = None,
) -> list[Assignment]:
    """Assign the tracklets of one entry to identities.

    A tracklet's best identity is its highest score, the first in name order on
    a tie, and its margin is that score minus the second best. Declared
    tracklets take their identity first. The others are visited by descending
    margin, then descending length, then ascending ``(source_id, first_frame)``.
    A visited tracklet takes its best identity when its score is at least
    *min_similarity*, its margin is at least *min_margin*, and no tracklet that
    holds that identity shares a frame with it. Otherwise it abstains. A
    blocked tracklet does not fall back to its second identity, because its
    margin was measured against that identity. A tracklet whose score row is
    not finite, such as one without a vector, abstains unless it is declared,
    and its score and margin are NaN.

    Args:
        entry_tracklets: The tracklets of one entry.
        scores: The ``(n, k)`` scores, row ``i`` for ``entry_tracklets[i]`` and
            columns in sorted identity order. A row of NaN marks a tracklet
            without a score.
        min_similarity: The lowest best score that is assigned.
        min_margin: The lowest margin that is assigned.
        declared: Identity indices by tracklet position, for tracklets that a
            reference names.

    Returns:
        One assignment per tracklet, in the order of *entry_tracklets*.

    Raises:
        ValueError: When the score matrix does not have one row per tracklet and
            at least two columns, a declared position or identity is out of
            range, or two declared tracklets of one identity share a frame.
    """
    matrix = np.asarray(scores, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(entry_tracklets)
        or matrix.shape[1] < 2
    ):
        raise ValueError(
            f"scores of shape {matrix.shape} need one row per tracklet "
            f"({len(entry_tracklets)}) and at least two identities"
        )
    scored = np.isfinite(matrix).all(axis=1)
    best = np.argmax(np.where(scored[:, None], matrix, 0.0), axis=1)
    best_scores = np.where(scored, matrix[np.arange(len(matrix)), best], np.nan)
    margins = best_scores - np.sort(matrix, axis=1)[:, -2]

    identities: list[int | None] = [None] * len(entry_tracklets)
    taken: dict[int, set[int]] = {}
    for position, identity in sorted((declared or {}).items()):
        if not 0 <= position < len(entry_tracklets):
            raise ValueError(f"declared tracklet position {position} is out of range")
        if not 0 <= identity < matrix.shape[1]:
            raise ValueError(f"declared identity {identity} is out of range")
        tracklet = entry_tracklets[position]
        frames = taken.setdefault(identity, set())
        shared = _shared_frame(tracklet.frames, frames)
        if shared is not None:
            raise ValueError(
                f"two tracklets declared identity {identity} share frame {shared}"
            )
        frames.update(tracklet.frames.tolist())
        identities[position] = identity

    def visit_key(position: int) -> tuple[float, int, int, int]:
        tracklet = entry_tracklets[position]
        return (
            -float(margins[position]),
            -tracklet.length,
            tracklet.source_id,
            tracklet.first_frame,
        )

    undeclared = [
        p for p in range(len(entry_tracklets)) if identities[p] is None and scored[p]
    ]
    for position in sorted(undeclared, key=visit_key):
        if best_scores[position] < min_similarity or margins[position] < min_margin:
            continue
        identity = int(best[position])
        frames = taken.setdefault(identity, set())
        tracklet = entry_tracklets[position]
        if _shared_frame(tracklet.frames, frames) is not None:
            continue
        frames.update(tracklet.frames.tolist())
        identities[position] = identity

    return [
        Assignment(
            tracklet=tracklet,
            identity=identities[position],
            best=int(best[position]),
            score=float(best_scores[position]),
            margin=float(margins[position]),
        )
        for position, tracklet in enumerate(entry_tracklets)
    ]
