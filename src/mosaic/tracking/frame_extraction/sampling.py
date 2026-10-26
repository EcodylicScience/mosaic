"""Frame selection algorithms for media extraction."""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from mosaic.core.json_value import JsonValue

SamplingMethod = Literal["uniform", "kmeans"]
"""A method that chooses the frames to keep: evenly spaced or by k-means clustering."""

ExtractionMethod = Literal[SamplingMethod, "list"]
"""How the frames to keep are chosen: by a sampling method, or named by the caller."""


class CandidateFeatures(Protocol):
    """The candidate frames of one video or sequence, read with their features."""

    def candidates(
        self, start: int, end: int, step: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return every *step*-th frame from *start* to *end* inclusive, with features.

        Args:
            start: The first frame of the range.
            end: The last frame of the range.
            step: The stride between candidates.

        Returns:
            The candidate indices as an ``(N,)`` array, and their features as an
            ``(N, D)`` array with one row per index.
        """
        ...

    def at(self, indices: np.ndarray) -> np.ndarray:
        """Return the features of the frames at *indices*.

        Args:
            indices: One or more frame indices, in any order.

        Returns:
            The ``(M, D)`` features, one row per index in the order of *indices*,
            computed as :meth:`candidates` computes them.
        """
        ...


@dataclass(frozen=True, slots=True)
class KmeansSettings:
    """The settings of k-means sampling, recorded in the ``sampling`` record.

    Attributes:
        resize: The (width, height) that each candidate frame is resized to.
        grayscale: Whether candidate frames are converted to grayscale.
        max_candidates: The candidate count above which the candidate step widens,
            or ``None`` (or a value of 0 or less) to keep the run's step.
        batch_size: The ``MiniBatchKMeans`` batch size.
        max_iter: The ``MiniBatchKMeans`` iteration limit.
        n_init: The ``MiniBatchKMeans`` initialization count, or ``"auto"``.
    """

    resize: tuple[int, int]
    grayscale: bool
    max_candidates: int | None
    batch_size: int
    max_iter: int
    n_init: int | str


@dataclass(frozen=True, slots=True)
class Exclusion:
    """The frames of one camera that a run does not write and samples around.

    Attributes:
        indices: The excluded frame indices of the camera, possibly empty.
        spacing: The largest distance in frames from an excluded frame at which a
            candidate is removed too.
        entries_ignored: The number of the run's exclude entries that name no
            entry of the run, recorded in the ``sampling`` record.
    """

    indices: tuple[int, ...]
    spacing: int
    entries_ignored: int


def excluded_mask(
    candidates: np.ndarray, excluded: np.ndarray, spacing: int
) -> np.ndarray:
    """Mark the candidates within *spacing* frames of an excluded frame.

    Args:
        candidates: The candidate frame indices, ``(N,)``.
        excluded: The excluded frame indices, ``(M,)``, sorted ascending.
        spacing: The largest distance at which a candidate is marked.

    Returns:
        A boolean ``(N,)`` array, True where the nearest excluded frame lies at
        most *spacing* frames from the candidate.
    """
    frames = np.asarray(candidates, dtype=np.int64)
    if excluded.size == 0:
        return np.zeros(frames.shape, dtype=bool)
    after = np.searchsorted(excluded, frames)
    last = excluded.size - 1
    below = np.asarray(excluded[np.clip(after - 1, 0, last)], dtype=np.int64)
    above = np.asarray(excluded[np.clip(after, 0, last)], dtype=np.int64)
    nearest = np.minimum(np.abs(frames - below), np.abs(above - frames))
    return nearest <= spacing


def _nearest_to_midpoint(inside: np.ndarray, low: int, high: int) -> int:
    """Return the frame of *inside* nearest ``(low + high) / 2``, ties to the lower."""
    offsets = np.abs(2 * inside - (low + high))
    return int(inside[int(np.argmin(offsets))])


def fill_largest_gaps(
    remaining: np.ndarray, anchors: np.ndarray, n: int, *, first: int, last: int
) -> np.ndarray:
    """Pick up to *n* candidates, each at the middle of the largest gap left.

    The boundaries are the anchors and the virtual frames ``first - 1`` and
    ``last + 1``, which measure a gap at either end of the range as an interior gap
    is measured. Each step takes the pair of adjacent boundaries ``(a, b)`` of
    largest ``b - a`` with a candidate strictly between them, ties to the smaller
    ``a``. It picks that gap's candidate nearest ``(a + b) / 2``, ties to the
    smaller frame, and adds the pick to the boundaries. The steps stop after *n*
    picks or when no gap contains a candidate.

    Args:
        remaining: The candidate frame indices that may be picked.
        anchors: The excluded frame indices from *first* to *last*.
        n: The number of frames to pick.
        first: The first candidate of the range, before any was removed.
        last: The last candidate of the range, before any was removed.

    Returns:
        The picked frame indices, sorted ascending.
    """
    pool = np.sort(np.asarray(remaining, dtype=np.int64))
    boundaries = sorted({int(anchor) for anchor in anchors} | {first - 1, last + 1})
    # Each gap is (low - high, low, high, start, stop), so the heap pops the largest
    # gap first and the smaller low on a tie. pool[start:stop] contains the
    # candidates strictly inside the gap.
    gaps: list[tuple[int, int, int, int, int]] = []

    def push_gap(low: int, high: int) -> None:
        start = int(np.searchsorted(pool, low, side="right"))
        stop = int(np.searchsorted(pool, high, side="left"))
        if start < stop:
            heapq.heappush(gaps, (low - high, low, high, start, stop))

    for low, high in itertools.pairwise(boundaries):
        push_gap(low, high)
    picks: list[int] = []
    while gaps and len(picks) < n:
        _, low, high, start, stop = heapq.heappop(gaps)
        pick = _nearest_to_midpoint(pool[start:stop], low, high)
        picks.append(pick)
        push_gap(low, pick)
        push_gap(pick, high)
    return np.asarray(sorted(picks), dtype=np.int32)


def _require_candidates(candidates: np.ndarray) -> None:
    """Raise ``ValueError`` if *candidates* holds no frame."""
    if candidates.size == 0:
        raise ValueError("No candidate frames to sample from.")


def _ordered_unique(values: np.ndarray) -> np.ndarray:
    out = []
    seen = set()
    for v in values.tolist():
        iv = int(v)
        if iv in seen:
            continue
        seen.add(iv)
        out.append(iv)
    return np.asarray(out, dtype=np.int32)


def select_uniform_frames(candidate_indices: np.ndarray, n_frames: int) -> np.ndarray:
    """Evenly sample frame indices from candidate indices."""
    candidates = np.asarray(candidate_indices, dtype=np.int32).ravel()
    _require_candidates(candidates)
    if n_frames <= 0:
        raise ValueError("n_frames must be > 0.")
    if n_frames >= candidates.size:
        return candidates.copy()

    raw_positions = np.linspace(0, candidates.size - 1, num=n_frames)
    positions = np.round(raw_positions).astype(np.int32)
    positions = _ordered_unique(positions)

    if positions.size < n_frames:
        missing = n_frames - positions.size
        all_pos = np.arange(candidates.size, dtype=np.int32)
        keep = np.setdiff1d(all_pos, positions, assume_unique=False)
        positions = np.concatenate([positions, keep[:missing]])

    return candidates[np.sort(positions[:n_frames])]


def _squared_distances(features: np.ndarray, point: np.ndarray) -> np.ndarray:
    return ((features - point) ** 2).sum(axis=1)


def fill_by_farthest_point(
    features: np.ndarray,
    selected: list[int],
    target_count: int,
    random_state: int,
    anchors: np.ndarray | None = None,
) -> list[int]:
    """Return *selected* extended to *target_count* rows by farthest-point picks.

    Each pick is the remaining row whose squared distance to its nearest anchor or
    selected row is largest, with ties to the lowest row. A running vector of those
    nearest distances is updated with each pick. When neither *anchors* nor
    *selected* has a row, the first pick is a uniform draw seeded by *random_state*.

    Args:
        features: The ``(N, D)`` features, one row per candidate.
        selected: The rows already selected, in pick order.
        target_count: The number of rows to return.
        random_state: The seed of the first pick when no row seeds the distances.
        anchors: The ``(M, D)`` features of other frames, which seed the distances
            as selected rows do and are not returned, or ``None``. An array without
            elements seeds no distance, whatever its shape.

    Returns:
        The rows of *selected* followed by the picks, at most *target_count* rows.
        The picks stop early when every row is selected.
    """
    picks = list(selected)
    n_rows = int(features.shape[0])
    final_count = min(target_count, len(picks) + n_rows - len(set(picks)))
    if len(picks) >= final_count:
        return picks[:target_count]

    seeds = features[picks]
    if anchors is not None and anchors.size > 0:
        seeds = np.concatenate([np.asarray(anchors, dtype=features.dtype), seeds])
    if seeds.shape[0] == 0:
        rng = np.random.default_rng(random_state)
        picks.append(int(rng.choice(n_rows)))
        seeds = features[picks]

    nearest = np.full(n_rows, np.inf)
    for seed in seeds:
        np.minimum(nearest, _squared_distances(features, seed), out=nearest)
    nearest[picks] = -np.inf
    while len(picks) < final_count:
        best = int(np.argmax(nearest))
        picks.append(best)
        np.minimum(nearest, _squared_distances(features, features[best]), out=nearest)
        nearest[best] = -np.inf
    return picks


class _Clustering(Protocol):
    """The part of a ``MiniBatchKMeans`` estimator that k-means sampling reads."""

    def fit_predict(self, X: np.ndarray) -> object:
        """Fit the clusters to the rows of *X* and return each row's cluster."""
        ...

    @property
    def cluster_centers_(self) -> object:
        """Return the center of each cluster."""
        ...


def _mini_batch_kmeans(
    n_clusters: int,
    *,
    random_state: int,
    batch_size: int,
    max_iter: int,
    n_init: int | str,
) -> _Clustering:
    """Return an unfitted ``MiniBatchKMeans`` with these settings."""
    # Imported lazily so `import mosaic.tracking` stays light (sklearn is only
    # needed when k-means sampling runs).
    from sklearn.cluster import MiniBatchKMeans

    return MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init=n_init,
    )


def _fit_kmeans(
    features: np.ndarray,
    n_clusters: int,
    *,
    random_state: int,
    batch_size: int,
    max_iter: int,
    n_init: int | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster the rows of *features* with ``MiniBatchKMeans``.

    The batch size is clipped to the row count.

    Returns:
        The cluster of each row, ``(N,)``, and the cluster centers, ``(K, D)``.
    """
    kmeans = _mini_batch_kmeans(
        n_clusters,
        random_state=random_state,
        batch_size=max(1, min(batch_size, int(features.shape[0]))),
        max_iter=max_iter,
        n_init=n_init,
    )
    labels = np.asarray(kmeans.fit_predict(features), dtype=np.intp)
    centers = np.asarray(kmeans.cluster_centers_)
    return labels, centers


def _nearest_member(
    features: np.ndarray, members: np.ndarray, center: np.ndarray
) -> int:
    """Return the row of *members* nearest *center*, ties to the first."""
    distances = _squared_distances(features[members], center)
    return int(members[int(np.argmin(distances))])


def select_kmeans_frames(
    candidate_indices: np.ndarray,
    features: np.ndarray,
    n_frames: int,
    random_state: int = 42,
    batch_size: int = 1024,
    max_iter: int = 100,
    n_init: int | str = "auto",
) -> np.ndarray:
    """
    Select representative frames using k-means over candidate frame features.
    """
    candidates = np.asarray(candidate_indices, dtype=np.int32).ravel()
    X = np.asarray(features, dtype=np.float32)

    _require_candidates(candidates)
    if X.ndim != 2 or X.shape[0] != candidates.size:
        raise ValueError(
            "features must be shape (N, D) and aligned with candidate_indices."
        )
    if n_frames <= 0:
        raise ValueError("n_frames must be > 0.")
    if n_frames >= candidates.size:
        return candidates.copy()

    k = min(int(n_frames), int(candidates.size))
    labels, centers = _fit_kmeans(
        X,
        k,
        random_state=int(random_state),
        batch_size=int(batch_size),
        max_iter=int(max_iter),
        n_init=n_init,
    )

    selected_local: list[int] = []
    for cid in range(k):
        members = np.flatnonzero(labels == cid)
        if members.size > 0:
            selected_local.append(_nearest_member(X, members, centers[cid]))

    selected_local = _ordered_unique(
        np.asarray(selected_local, dtype=np.int32)
    ).tolist()

    if len(selected_local) < n_frames:
        selected_local = fill_by_farthest_point(
            X,
            selected_local,
            target_count=int(n_frames),
            random_state=int(random_state),
        )

    selected_local = selected_local[: int(n_frames)]
    selected_local = sorted(selected_local)
    return candidates[np.asarray(selected_local, dtype=np.int32)]


def _select_uncovered_clusters(
    candidates: np.ndarray,
    features: np.ndarray,
    anchor_features: np.ndarray,
    n_frames: int,
    *,
    kmeans: KmeansSettings,
    random_state: int,
) -> tuple[np.ndarray, int]:
    """Select *n_frames* candidates from the clusters that contain no anchor.

    The ``R`` candidates are clustered with the ``M`` anchors into
    ``min(n_frames + M, R + M)`` clusters. A cluster with an anchor is covered. The
    uncovered clusters are taken largest first, ties to the lower cluster, and each
    gives its candidate nearest its center, ties to the lower candidate. A shortfall
    is filled by farthest-point picks measured from the anchors and the picks.

    Args:
        candidates: The candidate frame indices, ``(R,)``, ascending.
        features: The candidate features, ``(R, D)``.
        anchor_features: The features of the excluded frames, ``(M, D)``.
        n_frames: The number of frames to select.
        kmeans: The k-means settings.
        random_state: The k-means seed.

    Returns:
        The selected frame indices, sorted, and the number of covered clusters.
    """
    if anchor_features.ndim != 2 or anchor_features.shape[1:] != features.shape[1:]:
        message = (
            f"anchor features of shape {anchor_features.shape} do not match "
            f"candidate features of shape {features.shape}"
        )
        raise ValueError(message)
    rows = int(candidates.size)
    stacked = np.concatenate([features, anchor_features]).astype(np.float32)
    members, anchors = stacked[:rows], stacked[rows:]
    n_clusters = min(n_frames, rows) + int(anchors.shape[0])
    labels, centers = _fit_kmeans(
        stacked,
        n_clusters,
        random_state=random_state,
        batch_size=kmeans.batch_size,
        max_iter=kmeans.max_iter,
        n_init=kmeans.n_init,
    )
    member_labels = labels[:rows]
    covered = {int(label) for label in labels[rows:]}
    sizes = np.bincount(member_labels, minlength=n_clusters)
    uncovered = sorted(
        (
            cluster
            for cluster in range(n_clusters)
            if cluster not in covered and sizes[cluster] > 0
        ),
        key=lambda cluster: (-int(sizes[cluster]), cluster),
    )
    chosen = [
        _nearest_member(
            members, np.flatnonzero(member_labels == cluster), centers[cluster]
        )
        for cluster in uncovered[:n_frames]
    ]
    if len(chosen) < n_frames:
        chosen = fill_by_farthest_point(
            members, chosen, n_frames, random_state, anchors=anchors
        )
    return candidates[np.sort(np.asarray(chosen, dtype=np.intp))], len(covered)


@dataclass(frozen=True, slots=True)
class _MaskedCandidates:
    """The candidates that an exclusion keeps, and the excluded frames among them.

    Attributes:
        keep: A boolean ``(N,)`` array, True for each candidate kept.
        anchors: The excluded frame indices from the first candidate to the last,
            sorted ascending.
        removed: The number of candidates removed.
    """

    keep: np.ndarray
    anchors: np.ndarray
    removed: int


def _mask_candidates(candidates: np.ndarray, excluded: Exclusion) -> _MaskedCandidates:
    """Return the candidates that *excluded* keeps, and the anchors among them.

    Raises:
        ValueError: If there is no candidate, or if every candidate is removed.
    """
    _require_candidates(candidates)
    indices = np.unique(np.asarray(excluded.indices, dtype=np.int64))
    keep = ~excluded_mask(candidates, indices, excluded.spacing)
    if not keep.any():
        message = f"all {candidates.size} candidate frames are excluded"
        raise ValueError(message)
    inside = (indices >= candidates[0]) & (indices <= candidates[-1])
    removed = int(candidates.size - np.count_nonzero(keep))
    return _MaskedCandidates(keep=keep, anchors=indices[inside], removed=removed)


def _exclusion_record(
    excluded: Exclusion, masked: _MaskedCandidates
) -> dict[str, JsonValue]:
    """Return the exclude keys of a ``sampling`` record, in order."""
    return {
        "exclude_count": int(masked.anchors.size),
        "exclude_spacing": int(excluded.spacing),
        "candidates_removed": masked.removed,
        "exclude_entries_ignored": int(excluded.entries_ignored),
    }


def _kmeans_candidate_step(
    start: int, end: int, candidate_step: int, max_candidates: int | None
) -> int:
    """Return the k-means candidate step for the frames from *start* to *end*.

    The range contains N candidates at *candidate_step*. When N exceeds a positive
    *max_candidates*, the step is *candidate_step* times ``ceil(N / max_candidates)``.
    Otherwise it is *candidate_step*.
    """
    if max_candidates is None or max_candidates <= 0:
        return candidate_step
    approx_candidates = ((end - start) // candidate_step) + 1
    if approx_candidates <= max_candidates:
        return candidate_step
    stride_mult = math.ceil(approx_candidates / float(max_candidates))
    return candidate_step * max(1, stride_mult)


def _kmeans_picks(
    candidates: np.ndarray,
    features: np.ndarray,
    n_frames: int,
    kmeans: KmeansSettings,
    random_state: int,
) -> np.ndarray:
    """Return :func:`select_kmeans_frames` over *candidates* with *kmeans* settings."""
    return select_kmeans_frames(
        candidate_indices=candidates,
        features=features,
        n_frames=n_frames,
        random_state=random_state,
        batch_size=kmeans.batch_size,
        max_iter=kmeans.max_iter,
        n_init=kmeans.n_init,
    )


def select_frame_indices(
    method: SamplingMethod,
    *,
    start: int,
    end: int,
    candidate_step: int,
    n_frames: int,
    source: CandidateFeatures,
    kmeans: KmeansSettings,
    random_state: int,
    excluded: Exclusion | None,
) -> tuple[np.ndarray, dict[str, JsonValue]]:
    """Select *n_frames* frames from *start* to *end* inclusive, with their record.

    Uniform sampling spaces its picks over every *candidate_step*-th frame and
    returns an empty record. K-means sampling widens the step to keep the candidates
    within ``kmeans.max_candidates``, reads the candidates from *source*, and records
    its settings and the candidate count.

    With *excluded*, the candidates within ``excluded.spacing`` of an excluded frame
    are removed, and the record gains the exclude keys. The excluded frames from the
    first candidate to the last are anchors. Without an anchor, the method runs over
    the remaining candidates. With anchors, uniform sampling splits the largest gaps
    between them (:func:`fill_largest_gaps`), and k-means sampling reads the
    anchors' features from *source* and takes the clusters that contain no anchor.

    Args:
        method: The sampling method.
        start: The first frame of the range.
        end: The last frame of the range.
        candidate_step: The stride between candidates before any k-means widening.
        n_frames: The number of frames to select.
        source: The reader of candidate features, read under k-means only.
        kmeans: The k-means settings, read under k-means only.
        random_state: The k-means seed.
        excluded: The frames that the selection avoids, or ``None``.

    Returns:
        The selected frame indices, sorted, and the run's ``sampling`` record.

    Raises:
        ValueError: If the range has no candidate, or if *excluded* removes every
            candidate.
    """
    if method == "uniform":
        candidates = np.arange(start, end + 1, candidate_step, dtype=np.int32)
        if excluded is None:
            return select_uniform_frames(candidates, n_frames), {}
        masked = _mask_candidates(candidates, excluded)
        remaining = candidates[masked.keep]
        if masked.anchors.size == 0:
            selected = select_uniform_frames(remaining, n_frames)
        else:
            selected = fill_largest_gaps(
                remaining,
                masked.anchors,
                n_frames,
                first=int(candidates[0]),
                last=int(candidates[-1]),
            )
        return selected, _exclusion_record(excluded, masked)

    effective_step = _kmeans_candidate_step(
        start, end, candidate_step, kmeans.max_candidates
    )
    candidates, features = source.candidates(start, end, effective_step)
    details: dict[str, JsonValue] = {
        "kmeans_resize": [int(kmeans.resize[0]), int(kmeans.resize[1])],
        "kmeans_grayscale": bool(kmeans.grayscale),
        "kmeans_max_candidates": None
        if kmeans.max_candidates is None
        else int(kmeans.max_candidates),
        "kmeans_effective_candidate_step": int(effective_step),
        "kmeans_batch_size": int(kmeans.batch_size),
        "kmeans_max_iter": int(kmeans.max_iter),
        "kmeans_n_init": kmeans.n_init,
        "candidate_count": int(candidates.size),
    }
    if excluded is None:
        selected = _kmeans_picks(candidates, features, n_frames, kmeans, random_state)
        return selected, details

    masked = _mask_candidates(candidates, excluded)
    remaining, remaining_features = candidates[masked.keep], features[masked.keep]
    if masked.anchors.size == 0:
        selected = _kmeans_picks(
            remaining, remaining_features, n_frames, kmeans, random_state
        )
        covered = 0
    else:
        selected, covered = _select_uncovered_clusters(
            remaining,
            remaining_features,
            source.at(masked.anchors),
            n_frames,
            kmeans=kmeans,
            random_state=random_state,
        )
    details.update(_exclusion_record(excluded, masked))
    details["clusters_covered"] = covered
    return selected, details
