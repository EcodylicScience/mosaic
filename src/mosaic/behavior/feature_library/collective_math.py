"""
Shared numerics for the collective-motion order parameters.

The primitives behind ``collective-motion-metrics`` (group-level, one row per
frame) and ``local-order-metrics`` (per individual, one row per frame and id).
Both features compute the same two order parameters over different populations
-- the whole shoal versus a disc around a focal individual -- so the arithmetic
lives here once and neither feature owns a copy.

Definitions follow Tunstrom K, Katz Y, Ioannou CC, Huepe C, Lutz MJ, Couzin ID
(2013) "Collective States, Multistability and Transitional Behavior in Schooling
Fish", PLoS Comput Biol 9(2): e1002915.

  polarization  O_p = (1/N) |sum_i u_i|
  rotation      O_r = (1/N) |sum_i u_i x r_i|

where ``u_i`` is the unit heading of individual *i* and ``r_i`` the unit vector
from the group's center of mass towards it. Both lie in [0, 1].

This module holds no feature and registers nothing.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull, Delaunay, QhullError

__all__ = [
    "STATE_HIGH",
    "STATE_LOW",
    "AreaMethod",
    "GroupState",
    "HeadingSource",
    "ResolvedHeadingSource",
    "alpha_shape_area",
    "backward_dt",
    "classify_state",
    "cross2",
    "hull_area",
    "polarization",
    "principal_axes",
    "resolve_heading_source",
    "scrub_positions",
    "step_masks",
    "unit_headings",
    "unit_radial",
]

FloatArray: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
ObjectArray: TypeAlias = npt.NDArray[np.object_]

HeadingSource: TypeAlias = Literal["auto", "orientation", "velocity"]
ResolvedHeadingSource: TypeAlias = Literal["orientation", "velocity"]
AreaMethod: TypeAlias = Literal["convex_hull", "alpha_shape", "none"]
GroupState: TypeAlias = Literal[
    "Polarized", "Milling", "Swarm", "Transitional", "Undefined"
]

# Tunstrom 2013, Methods, "Defining states". Deliberately module constants
# rather than feature params: a column named ``state`` should mean the paper's
# state and nothing else, and parameterizing the cuts would let two datasets
# carry the same column name under different rules.
STATE_LOW: float = 0.35
STATE_HIGH: float = 0.65


def scrub_positions(
    x: FloatArray, y: FloatArray
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Map every row with a non-finite X *or* Y to (NaN, NaN); return the joint mask.

    The mask is **joint** on purpose. Masking X and Y independently -- as a
    per-column ``nanmean`` does -- assembles a centroid from two different
    populations of individuals: an animal whose X is lost still contributes its
    Y. ``np.isfinite`` is also the only predicate that rejects NaN *and* both
    infinities; ``np.isnan(inf)`` is False and ``dropna()`` keeps inf, and one
    infinite coordinate makes the whole centroid infinite.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    return np.where(finite, x, np.nan), np.where(finite, y, np.nan), finite


def step_masks(
    ids: npt.NDArray[np.generic], order: FloatArray
) -> tuple[BoolArray, FloatArray]:
    """Backward-difference validity mask and order step, for rows sorted by (id, order).

    Returns ``(same_id, dstep)`` where ``same_id[k]`` is True when row *k* and
    row *k-1* belong to one individual, and ``dstep`` is the order-column
    difference. Both are NaN/False at the first row.
    """
    n = ids.shape[0]
    same_id = np.zeros(n, dtype=bool)
    dstep = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return same_id, dstep
    same_id[1:] = ids[1:] == ids[:-1]
    dstep[1:] = order[1:] - order[:-1]
    return same_id, dstep


def resolve_heading_source(
    source: HeadingSource, angle: FloatArray | None
) -> ResolvedHeadingSource:
    """Resolve ``"auto"`` against the orientation column actually present.

    ``"auto"`` picks ``"orientation"`` only when the column exists **and** holds
    at least one finite value; a column of all-NaN angles is not an orientation
    signal. ``"orientation"`` with no usable column raises rather than falling
    back, because a silent fallback changes every emitted number.
    """
    if source == "velocity":
        return "velocity"
    usable = angle is not None and bool(np.isfinite(angle).any())
    if source == "orientation":
        if not usable:
            msg = (
                "heading_source='orientation' but no usable orientation column: "
                "it is absent or entirely non-finite. Pass "
                "heading_source='velocity' to derive headings from motion."
            )
            raise ValueError(msg)
        return "orientation"
    return "orientation" if usable else "velocity"


def unit_headings(
    resolved: ResolvedHeadingSource,
    angle: FloatArray | None,
    x: FloatArray,
    y: FloatArray,
    same_id: BoolArray,
) -> tuple[FloatArray, FloatArray]:
    """Per-row unit heading ``u_i``, NaN where undefined.

    Under ``"orientation"`` this is ``(cos, sin)`` of the angle column. Under
    ``"velocity"`` it is the normalized backward position difference within one
    individual -- ``dt`` cancels in a unit vector, so no timing information is
    needed. Positions must already be scrubbed.

    The first row of each individual is NaN, never a copy of the second: padding
    it injects a zero-lag repeat that autocorrelation and state-space models read
    as signal. An individual that did not move has a zero-length difference and
    is likewise NaN rather than an arbitrary direction.
    """
    if resolved == "orientation":
        if angle is None:
            msg = "unit_headings: resolved to 'orientation' with no angle array"
            raise ValueError(msg)
        return np.cos(angle), np.sin(angle)

    n = x.shape[0]
    ux = np.full(n, np.nan, dtype=np.float64)
    uy = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return ux, uy
    dx = np.full(n, np.nan, dtype=np.float64)
    dy = np.full(n, np.nan, dtype=np.float64)
    dx[1:] = x[1:] - x[:-1]
    dy[1:] = y[1:] - y[:-1]
    norm = np.hypot(dx, dy)
    ok = same_id & (norm > 0)
    ux[ok] = dx[ok] / norm[ok]
    uy[ok] = dy[ok] / norm[ok]
    return ux, uy


def backward_dt(
    order: FloatArray,
    order_is_frame: bool,
    time: FloatArray | None,
    fps: float | None,
    same_id: BoolArray,
    dstep: FloatArray,
    max_frame_gap: int | None,
) -> FloatArray:
    """Per-row backward-difference ``dt``, NaN where a difference is not defined.

    Follows the ``speed-angvel`` ladder: ``frame`` plus ``fps`` first (immune to
    the jittery wall-clock timestamps some trackers embed), then a ``time``
    column, then one order step as a last resort. Steps that are absent,
    non-positive, or larger than *max_frame_gap* become NaN -- a gap wider than
    the caller's tolerance means the shoal reorganized, and averaging across it
    is a lower-resolution number dressed as a measurement.
    """
    n = order.shape[0]
    if order_is_frame and fps is not None:
        dt = dstep / fps
    elif time is not None:
        dt = np.full(n, np.nan, dtype=np.float64)
        if n >= 2:
            dt[1:] = time[1:] - time[:-1]
    else:
        dt = dstep.copy()

    ok = same_id & (dstep > 0) & np.isfinite(dt) & (dt > 0)
    if max_frame_gap is not None:
        ok &= dstep <= float(max_frame_gap)
    return np.where(ok, dt, np.nan)


def cross2(
    ax: FloatArray, ay: FloatArray, bx: FloatArray, by: FloatArray
) -> FloatArray:
    """z-component of ``a x b`` for 2-D vectors: ``ax*by - ay*bx``.

    Written out rather than delegated: ``np.cross`` on 2-vectors is deprecated in
    numpy 2 and slated for removal.
    """
    return ax * by - ay * bx


def unit_radial(
    rx: FloatArray, ry: FloatArray
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Unit centroid-to-individual vectors; return ``(rhx, rhy, at_centre)``.

    ``at_centre`` is ``~(|r| > 0)``, so it flags an individual sitting exactly at
    the centroid *and* one whose position is not finite; a caller that needs only
    the former conjoins its own finite mask. Those rows get NaN rather than a
    substituted zero: a zero radial vector contributes nothing to the rotation
    numerator but would silently pass for a real measurement.
    """
    rnorm = np.hypot(rx, ry)
    at_centre = ~(rnorm > 0)
    safe = np.where(at_centre, np.nan, rnorm)
    return rx / safe, ry / safe, at_centre


def polarization(sum_ux: FloatArray, sum_uy: FloatArray, n: FloatArray) -> FloatArray:
    """Tunstrom ``O_p`` from summed unit headings and their count; NaN where n <= 0.

    Sum-over-count rather than a mean, so the divisor is explicit: it is the
    number of individuals with a *usable* heading, not the number tracked. A
    shoal of thirty with three usable headings would otherwise report the
    polarization of three as if it were the polarization of thirty.
    """
    denom = np.where(n > 0, n, np.nan)
    return np.hypot(sum_ux, sum_uy) / denom


def principal_axes(
    sxx: FloatArray, syy: FloatArray, sxy: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Closed-form 2x2 symmetric eigen-solve of the position covariance.

    Takes the second moments about the centroid and returns
    ``(sd_major, sd_minor, elongation, axis_angle)``: the standard deviations
    along the two principal axes, their ratio, and the major axis orientation in
    (-pi/2, pi/2]. Unlike a heading-frame extent this is defined for a stationary
    group, where the centroid velocity is not.

    Two clamps are load-bearing. The discriminant ``tr^2/4 - det`` is
    analytically ``((Sxx-Syy)/2)^2 + Sxy^2`` and so non-negative, but rounds
    negative for a near-isotropic group; and the smaller eigenvalue of a
    near-collinear covariance rounds negative roughly half the time. Both would
    otherwise yield NaN from ``sqrt``.

    Closed form rather than ``np.linalg.eigh`` because this is vectorized over
    every output row at once, and because the major axis is undirected -- there
    is no eigenvector sign to canonicalize.
    """
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    sd_major = np.sqrt(np.maximum(tr / 2.0 + disc, 0.0))
    sd_minor = np.sqrt(np.maximum(tr / 2.0 - disc, 0.0))
    elongation = np.where(
        sd_minor > 0,
        sd_major / np.where(sd_minor > 0, sd_minor, np.nan),
        np.where(sd_major > 0, np.inf, np.nan),
    )
    axis_angle = 0.5 * np.arctan2(2.0 * sxy, sxx - syy)
    return sd_major, sd_minor, elongation, axis_angle


def hull_area(pts: FloatArray) -> float:
    """Convex-hull area of 2-D points; NaN when degenerate.

    *pts* must already be finite and deduplicated. Returns ``ConvexHull.volume``,
    which for 2-D input is the enclosed **area** -- ``.area`` is the perimeter.

    Both ``QhullError`` and ``ValueError`` are caught: fewer than three points and
    collinear points raise the first, a non-finite coordinate the second. The
    ``QJ`` joggle option is deliberately not used -- it returns a perturbation
    artifact on the order of 1e-10 for collinear input instead of failing, which
    reaches a density as ~1e10.
    """
    if pts.shape[0] < 3:
        return float("nan")
    try:
        return float(ConvexHull(pts).volume)
    except (QhullError, ValueError):
        return float("nan")


def alpha_shape_area(pts: FloatArray, alpha: float) -> tuple[float, int]:
    """Alpha-shape area of 2-D points and the triangle count, as ``(area, n)``.

    The area spanned by a *concave* group: the summed area of the Delaunay
    triangles whose circumradius is below *alpha*, which is the standard
    alpha-shape construction and needs no dependency beyond scipy. Tunstrom's
    packing fraction is built on an alpha shape for exactly this reason -- a
    convex hull bridges the concavities of a real school and overstates its area.

    *alpha* is a circumradius cutoff in position units, so it scales with the
    data and composes with a body length. Some of the literature writes the same
    parameter as ``1/R``; a value carried in from elsewhere may need inverting.

    Returns ``(nan, 0)`` when nothing is admitted -- never ``(0.0, 0)``, which
    would make a density infinite rather than absent.
    """
    if pts.shape[0] < 3:
        return float("nan"), 0
    try:
        tri = Delaunay(pts)
    except (QhullError, ValueError):
        return float("nan"), 0
    simplices = tri.simplices
    if simplices.shape[0] == 0:
        return float("nan"), 0

    pa = pts[simplices[:, 0]]
    pb = pts[simplices[:, 1]]
    pc = pts[simplices[:, 2]]
    side_a = np.hypot(pb[:, 0] - pc[:, 0], pb[:, 1] - pc[:, 1])
    side_b = np.hypot(pa[:, 0] - pc[:, 0], pa[:, 1] - pc[:, 1])
    side_c = np.hypot(pa[:, 0] - pb[:, 0], pa[:, 1] - pb[:, 1])
    semi = (side_a + side_b + side_c) / 2.0
    area = np.sqrt(
        np.maximum(semi * (semi - side_a) * (semi - side_b) * (semi - side_c), 0.0)
    )
    circumradius = np.where(
        area > 0,
        side_a * side_b * side_c / (4.0 * np.where(area > 0, area, np.nan)),
        np.inf,
    )
    keep = circumradius < alpha
    n_kept = int(keep.sum())
    if n_kept == 0:
        return float("nan"), 0
    return float(area[keep].sum()), n_kept


def classify_state(
    rotation_abs: FloatArray, polarization_values: FloatArray
) -> ObjectArray:
    """The Tunstrom 2013 collective state from ``|O_r|`` and ``O_p``.

    Polarized when ``O_p > 0.65`` and ``|O_r| < 0.35``; milling when
    ``O_p < 0.35`` and ``|O_r| > 0.65``; swarm when both are below 0.35;
    transitional otherwise.

    ``"Undefined"`` is written last, and is not one of the paper's states. A
    frame whose order parameters are NaN fails every threshold comparison and so
    falls through to the transitional class, which makes every tracking failure
    read as a behavioral transition -- precisely the frames a transition analysis
    is about.

    Exported so the paper's own procedure can be reproduced: it smooths the
    order-parameter series with a 30-frame (1 s) moving average *before*
    classifying. Neither feature does that internally, so a faithful comparison
    means rolling this feature's ``polarization`` and ``rotation`` columns and
    re-applying this function to the result.
    """
    out: ObjectArray = np.full(rotation_abs.shape, "Transitional", dtype=object)
    low_r = rotation_abs < STATE_LOW
    low_p = polarization_values < STATE_LOW
    out[low_r & (polarization_values > STATE_HIGH)] = "Polarized"
    out[low_r & low_p] = "Swarm"
    out[(rotation_abs > STATE_HIGH) & low_p] = "Milling"
    out[~(np.isfinite(rotation_abs) & np.isfinite(polarization_values))] = "Undefined"
    return out
