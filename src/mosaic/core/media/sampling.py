"""Frame selection algorithms for media extraction."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import MiniBatchKMeans


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
    if candidates.size == 0:
        raise ValueError("No candidate frames to sample from.")
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


def _fill_by_farthest_point(
    features: np.ndarray,
    selected: list[int],
    target_count: int,
    random_state: int,
) -> list[int]:
    """Fill shortfall using greedy farthest-point sampling."""
    n_total = int(features.shape[0])
    if len(selected) >= target_count:
        return selected[:target_count]

    rng = np.random.default_rng(random_state)
    selected_set = set(selected)
    remaining = [i for i in range(n_total) if i not in selected_set]
    if not remaining:
        return selected

    if not selected:
        first = int(rng.choice(remaining))
        selected.append(first)
        selected_set.add(first)
        remaining.remove(first)

    while len(selected) < target_count and remaining:
        chosen = features[np.asarray(selected, dtype=np.int32)]
        rem_idx = np.asarray(remaining, dtype=np.int32)
        rem = features[rem_idx]
        d2 = ((rem[:, None, :] - chosen[None, :, :]) ** 2).sum(axis=2)
        min_d2 = d2.min(axis=1)
        best_pos = int(np.argmax(min_d2))
        best_idx = int(rem_idx[best_pos])
        selected.append(best_idx)
        selected_set.add(best_idx)
        remaining = [i for i in remaining if i != best_idx]

    return selected[:target_count]


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

    if candidates.size == 0:
        raise ValueError("No candidate frames to sample from.")
    if X.ndim != 2 or X.shape[0] != candidates.size:
        raise ValueError("features must be shape (N, D) and aligned with candidate_indices.")
    if n_frames <= 0:
        raise ValueError("n_frames must be > 0.")
    if n_frames >= candidates.size:
        return candidates.copy()

    k = min(int(n_frames), int(candidates.size))
    mb_size = max(1, min(int(batch_size), int(candidates.size)))
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=int(random_state),
        batch_size=mb_size,
        max_iter=int(max_iter),
        n_init=n_init,
    )
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    selected_local: list[int] = []
    for cid in range(k):
        members = np.where(labels == cid)[0]
        if members.size == 0:
            continue
        sub = X[members]
        center = centers[cid]
        d2 = ((sub - center[None, :]) ** 2).sum(axis=1)
        best = int(members[int(np.argmin(d2))])
        selected_local.append(best)

    selected_local = _ordered_unique(np.asarray(selected_local, dtype=np.int32)).tolist()

    if len(selected_local) < n_frames:
        selected_local = _fill_by_farthest_point(
            X,
            selected_local,
            target_count=int(n_frames),
            random_state=int(random_state),
        )

    selected_local = selected_local[: int(n_frames)]
    selected_local = sorted(selected_local)
    return candidates[np.asarray(selected_local, dtype=np.int32)]
