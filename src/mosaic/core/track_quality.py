"""Measures of a track's quality that other modules share.

numpy is imported at module scope and pandas only inside functions and under
``TYPE_CHECKING``, so a module that loads no pandas can import this one and
still load none. Nothing here imports from ``core.pipeline``.

**The jump rule here differs from ``trajectory_smooth``'s.** The smoother
differences consecutive rows, so a step across a frame gap counts, and its
threshold is a speed in units per frame or per second. The rules here measure
consecutive frames only, in body lengths. The two are not logically
equivalent, so neither is written in terms of the other.
"""

from __future__ import annotations

import numpy as np

__all__ = ["instance_body_lengths"]


def instance_body_lengths(
    xy: np.ndarray, head: int | None, tail: int | None
) -> np.ndarray:
    """Return one body length per instance.

    The length is the head-to-tail distance when both keypoints are in range
    and finite, and otherwise the diagonal of the tight box around the finite
    keypoints.

    Args:
        xy: Keypoints of shape ``(N, K, 2)``.
        head: Index of the head keypoint, or None.
        tail: Index of the tail keypoint, or None.

    Returns:
        Float array of shape ``(N,)``, NaN for an instance with no finite
        keypoint.
    """
    points = np.asarray(xy, dtype=float)
    finite = np.isfinite(points).all(axis=-1)
    low = np.where(finite[..., None], points, np.inf).min(axis=1)
    high = np.where(finite[..., None], points, -np.inf).max(axis=1)
    lengths = np.hypot(*(high - low).T)
    lengths = np.where(finite.any(axis=1), lengths, np.nan)
    count = points.shape[1]
    if (
        head is not None
        and tail is not None
        and 0 <= head < count
        and 0 <= tail < count
    ):
        both = finite[:, head] & finite[:, tail]
        direct = np.linalg.norm(points[:, head] - points[:, tail], axis=-1)
        lengths = np.where(both, direct, lengths)
    return lengths
