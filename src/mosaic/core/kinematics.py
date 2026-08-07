"""Deriving heading from keypoints.

These used to live in ``core/track_library/helpers.py`` and be called by four
converters, which is what made a heading look like something a tracker reports.
It is not: a pose model returns keypoints, and turning those into a direction is
an inference with choices in it. The choices matter enough to be visible --
``angle_from_pca`` in particular returns an *axis*, whose sign is arbitrary and
can flip between consecutive frames of a perfectly clean track.

They are here rather than in ``track_library`` because no track converter calls
them any more. Their caller is the ``heading`` feature, where the method is a
parameter, the result is addressed by a run identifier, and a reader can tell
which rule produced the numbers they are looking at.

**Front and rear, not neck and tail.** The two-point form needs an ordered pair
of landmarks along the body axis and nothing more; ``neck``/``tail`` named a fish
and a mouse and read as nonsense for an insect or a bird.
"""

from __future__ import annotations

import numpy as np

__all__ = ["angle_from_pca", "angle_from_two_points"]


def angle_from_two_points(front_xy: np.ndarray, rear_xy: np.ndarray) -> np.ndarray:
    """Heading from *rear* toward *front*, in radians about the +x axis.

    Args:
        front_xy: ``(T, 2)`` positions of the forward landmark.
        rear_xy: ``(T, 2)`` positions of the rearward landmark.

    Returns:
        ``(T,)`` angles in radians.

    A true heading rather than an axis: the pair is ordered, so the direction is
    determined and stable frame to frame. Prefer it whenever two landmarks along
    the body can be named.

    Measured in image coordinates, where ``y`` increases *downward*, so a
    positive angle turns clockwise on screen. That is the frame every tracker
    mosaic reads reports in, and flipping it here would put the pose keypoints
    and the heading in different conventions.
    """
    vector = front_xy - rear_xy
    return np.arctan2(vector[:, 1], vector[:, 0])


def angle_from_pca(xy: np.ndarray) -> np.ndarray:
    """Body-axis angle per frame, from the first principal component.

    Args:
        xy: ``(T, L, 2)`` landmark positions for one individual.

    Returns:
        ``(T,)`` angles in radians.

    **This returns an axis, not a heading, and its sign is arbitrary.** A
    principal component is defined up to negation, so nothing here distinguishes
    facing forward from facing backward, and two consecutive frames of an
    otherwise clean track can come back pi apart. Anything that differences
    successive angles -- an angular velocity, a turn rate -- will read those
    flips as real turns.

    Use :func:`angle_from_two_points` where two landmarks can be named. This is
    the fallback for a keypoint set with no orientable pair, and a caller
    choosing it is choosing that trade rather than inheriting it.

    Vectorized: builds every frame's 2x2 covariance at once and solves the
    eigenproblem in closed form.
    """
    # Centre each frame.
    mu = xy.mean(axis=1, keepdims=True)  # (T, 1, 2)
    centred = xy - mu  # (T, L, 2)

    # Covariance per frame: cov[t] = centred[t].T @ centred[t].
    cov = np.einsum("tli,tlj->tij", centred, centred)  # (T, 2, 2)

    # For symmetric [[a, b], [b, d]] the larger eigenvalue's eigenvector is
    # available analytically.
    a = cov[:, 0, 0]
    b = cov[:, 0, 1]
    d = cov[:, 1, 1]

    diff = a - d
    disc = np.sqrt(diff * diff + 4.0 * b * b)
    lam_max = 0.5 * ((a + d) + disc)

    # Eigenvector (b, lam_max - a), normalized implicitly by arctan2.
    vx = b
    vy = lam_max - a

    # b ~ 0 means the matrix is already diagonal, so the axis is x or y.
    diagonal = np.abs(b) < 1e-12
    vx = np.where(diagonal, np.where(a >= d, 1.0, 0.0), vx)
    vy = np.where(diagonal, np.where(a >= d, 0.0, 1.0), vy)

    return np.arctan2(vy, vx)
