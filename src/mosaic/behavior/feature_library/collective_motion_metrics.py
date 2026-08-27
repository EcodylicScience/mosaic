"""
CollectiveMotionMetrics -- group-level collective-motion order parameters.

Reduces multi-individual tracks to one row per frame carrying the two order
parameters of Tunstrom et al. 2013 -- polarization ``O_p`` and rotation ``O_r``
-- the collective state they define, and the group's centroid kinematics, shape,
area and density. With ``subgroup_col`` set it emits one row per frame per
subgroup instead, so a fissioned shoal is described subgroup by subgroup.

The per-individual counterpart is ``local-order-metrics``, which restricts the
same order parameters to a neighborhood around each focal individual.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self, final

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from mosaic.core.pipeline.types import (
    COLUMNS as C,
    EmitsLevel,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    Inputs,
    InputStream,
    Result,
    TrackInput,
    resolve_order_col,
)
from mosaic.core.params import Params

from .collective_math import (
    AreaMethod,
    HeadingSource,
    alpha_shape_area,
    backward_dt,
    classify_state,
    cross2,
    hull_area,
    polarization,
    principal_axes,
    resolve_heading_source,
    scrub_positions,
    step_masks,
    unit_headings,
    unit_radial,
)
from .helpers import apply_exclude_cols, ensure_columns
from .registry import register_feature

# Every column this feature emits, for the subgroup_col collision check. A
# subgroup column sharing a name with an output would give the assembled frame
# two identically-labeled columns.
EMITTED_COLUMNS: frozenset[str] = frozenset(
    {
        "n_ids",
        "n_ids_heading",
        "n_at_centroid",
        "n_centroid_common",
        "polarization",
        "mean_heading",
        "rotation",
        "rotation_signed",
        "group_angvel",
        "state",
        "centroid_x",
        "centroid_y",
        "centroid_speed",
        "centroid_heading",
        "sd_major",
        "sd_minor",
        "elongation",
        "principal_axis_angle",
        "area",
        "density",
        "alpha_n_triangles",
        "mean_speed",
        "heading_source",
    }
)


@final
@register_feature
class CollectiveMotionMetrics:
    """
    Per-frame group-level collective-motion metrics (Tunstrom et al. 2013).

    Reference: Tunstrom K, Katz Y, Ioannou CC, Huepe C, Lutz MJ, Couzin ID (2013)
    "Collective States, Multistability and Transitional Behavior in Schooling
    Fish", PLoS Comput Biol 9(2): e1002915.

    Output columns (one row per frame, or per frame and subgroup):

    Keys and counts
      - ``frame`` (or ``time``), ``time``, and the subgroup column when set
      - ``n_ids``: individuals with a jointly finite position
      - ``n_ids_heading``: individuals with a usable unit heading -- the divisor
        of ``polarization``, and the number to check before trusting it
      - ``n_at_centroid``: individuals sitting exactly at the centroid, where the
        radial direction is undefined
      - ``n_centroid_common``: individuals present in both this frame and the
        previous one -- the divisor of the centroid velocity

    Order parameters and state
      - ``polarization``: ``O_p = (1/N)|sum u_i|``, in [0, 1]
      - ``mean_heading``: direction of that mean vector, radians. Meaningless
        when ``polarization`` is near zero, where the direction is undefined
      - ``rotation``: ``O_r = (1/N)|sum u_i x r_i|``, in [0, 1] -- the paper's
        quantity, on which its state thresholds are stated
      - ``rotation_signed``: the same mean without the absolute value, so
        chirality survives. Positive is counter-clockwise **in the stored
        coordinate frame**; with image coordinates (+Y down, which every
        pixel-space converter produces) that reads as clockwise in the video.
        Averaging the signed column over time cancels clockwise against
        counter-clockwise milling towards zero -- average ``rotation`` instead
      - ``group_angvel``: mean angular velocity of individuals about the
        centroid, rad/s (rad per frame when neither ``fps`` nor ``time`` is
        available). Unlike ``rotation`` this carries magnitude, not just direction
      - ``state``: ``Polarized`` | ``Milling`` | ``Swarm`` | ``Transitional`` |
        ``Undefined``

    Centroid, shape and density
      - ``centroid_x`` / ``centroid_y``, ``centroid_speed``, ``centroid_heading``
      - ``sd_major`` / ``sd_minor``: standard deviations along the principal axes
      - ``elongation``: their ratio, >= 1; ``inf`` for collinear individuals
      - ``principal_axis_angle``: major-axis orientation in (-pi/2, pi/2]
      - ``area``, ``density`` = ``n_ids / area``, and ``alpha_n_triangles`` under
        ``area_method="alpha_shape"``
      - ``mean_speed`` when ``speed_col`` is set

    Provenance
      - ``heading_source``: ``"orientation"`` or ``"velocity"``, the runtime
        resolution of the ``heading_source`` param
      - ``sequence``, ``group``

    Params:
        heading_source: Where each unit heading comes from. ``"auto"`` (default)
            uses the ``ANGLE`` column when it is present and holds at least one
            finite value, otherwise the direction of travel -- which is what
            simulation output without a body orientation needs.
            ``"orientation"`` raises rather than falling back.
        subgroup_col: Column partitioning each frame into subgroups, from
            ``ffgroups``. Prefer ``"event"``: ``"group_membership"`` is a
            per-frame connected-component label with no identity across frames,
            so ``centroid_speed`` and ``group_angvel`` would difference two
            possibly disjoint sets of animals. Default None (whole group).
        area_method: ``"convex_hull"`` (default), ``"alpha_shape"``, or
            ``"none"`` to skip ``area`` and ``density``.
        alpha: Circumradius cutoff in position units; required under
            ``area_method="alpha_shape"``.
        min_individuals: Below this count the relational metrics are NaN and
            ``state`` is ``"Undefined"``. Default 3 rather than 2 because at
            N=2 the two radial vectors are antiparallel by construction, so
            rotation carries no information yet still feeds the classifier.
        fps: Frames per second. With a ``frame`` order column this makes
            ``dt = frame_diff / fps``, which is immune to the jittery wall-clock
            timestamps some trackers embed. Default None.
        max_frame_gap: Frame steps wider than this yield NaN velocity rather
            than an average across a gap the shoal reorganized over.
        min_group_speed: ``centroid_heading`` is NaN at or below this speed. A
            mill's centroid is stationary by definition and ``arctan2(0, 0)``
            silently returns 0 -- due east. Default 0.0.
        speed_col: Per-individual speed column to average as ``mean_speed``.
            Worth setting under ``subgroup_col``, where ``frame-aggregate``
            cannot express a per-subgroup mean. Default None.
        filter_expr: ``DataFrame.query`` filter applied first. The way to drop
            ``event == -1``, the pooled non-event pseudo-group: that sentinel is
            an integer, so ``exclude_cols`` cannot express it.
        exclude_cols: Boolean columns whose truthy rows are dropped first. One
            bad position moves the centroid, and with it every radial vector,
            the covariance and the hull.

    Notes:
        **Smoothing is deliberately not a parameter.** The paper smooths the
        order-parameter *series* with a 30-frame (1 s) moving average before
        classifying states, and its 0.35 / 0.65 thresholds were calibrated on
        that smoothed signal. Applied per frame they classify many more frames
        as ``Transitional``, because per-frame ``O_p`` and ``O_r`` have variance
        of order 1/N. To reproduce the paper, roll this feature's output and
        re-apply ``classify_state``::

            out["polarization"] = out["polarization"].rolling(30, center=True).mean()
            out["rotation"] = out["rotation"].rolling(30, center=True).mean()
            out["state"] = classify_state(
                out["rotation"].to_numpy(), out["polarization"].to_numpy()
            )

        Note that ``trajectory-smooth`` and ``movement-smooth`` smooth the
        *tracks*, which is a different operation with a different effect.

        **Wiring.** To compute per subgroup, pass the ``Result`` object returned
        by ``run_feature`` rather than a hand-written one, and list ``"tracks"``
        first::

            ff = ds.run_feature(FFGroups(params={"distance_cutoff": 120.0}))
            ds.run_feature(CollectiveMotionMetrics(
                Inputs(("tracks", ff)),
                {"fps": 30.0, "subgroup_col": "event", "filter_expr": "event >= 0"},
            ))

        A hand-written reference must carry the *storage* name, not the slug --
        ``Result(feature="ffgroups__from__tracks")``. A slug names a directory
        that does not exist, which resolves to an empty manifest and a run that
        writes nothing and reports success. Listing ``"tracks"`` first keeps
        ``group`` and ``sequence`` unsuffixed, since the merge suffixes the later
        input's duplicate columns.

        ``centroid_heading`` is named to match ``FFGroupsMetrics``'s
        ``centroid_heading_col`` default, so the two features compose. ``state``
        is the per-frame collective state, in the column name the overlay reads
        as a label with no further wiring.

        **Every row reduced together must share one coordinate frame.** That
        holds by construction today -- no track converter emits a ``camera``
        column, so a track table describes one view. When multi-camera tracks
        arrive, ``subgroup_col="camera"`` is the answer and needs no change
        here: it is a generic per-(frame, id) partition key, so each camera gets
        its own centroid, order parameters, hull and state, exactly as each
        ffgroup does. Pooling two views into one centroid would otherwise be
        silent, since nothing in the numbers says a coordinate system changed.

        **``overlap_frames`` is supported, and worth using on a continuous
        group.** ``centroid_speed`` and ``group_angvel`` are backward differences,
        so without context they are NaN on the first frame of every sequence --
        the artifact overlap exists to remove. It needs the group's sequences to
        be declared continuous and numbered on one frame axis; a run that asks
        for it otherwise is refused rather than approximated.
    """

    category = "summary"
    name = "collective-motion-metrics"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "unidentified"

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        """Collective-motion-metrics parameters. See the class docstring."""

        heading_source: HeadingSource = "auto"
        subgroup_col: str | None = None
        area_method: AreaMethod = "convex_hull"
        alpha: float | None = Field(default=None, gt=0)
        min_individuals: int = Field(default=3, ge=1)
        fps: float | None = Field(default=None, gt=0)
        max_frame_gap: int | None = Field(default=None, ge=1)
        min_group_speed: float = Field(default=0.0, ge=0)
        speed_col: str | None = None
        filter_expr: str | None = None
        exclude_cols: list[str] = Field(default_factory=list)

        @model_validator(mode="after")
        def _check(self) -> Self:
            """Reject at construction, where a raise is visible.

            ``run_feature`` catches every exception out of ``apply``, prints one
            line and carries on, so a raise there is a silently dropped entry and
            an exit code of zero.
            """
            if self.area_method == "alpha_shape" and self.alpha is None:
                msg = (
                    "area_method='alpha_shape' requires 'alpha', a circumradius "
                    "cutoff in position units. There is no scale-free default: a "
                    "value derived per sequence would make areas incomparable "
                    "between frames and between subgroups of one frame."
                )
                raise ValueError(msg)
            if self.subgroup_col is not None:
                reserved = C.meta_set() | EMITTED_COLUMNS
                if self.subgroup_col in reserved:
                    msg = (
                        f"subgroup_col={self.subgroup_col!r} collides with a "
                        f"metadata or emitted column. Use 'event' from ffgroups."
                    )
                    raise ValueError(msg)
            return self

    def __init__(
        self,
        inputs: CollectiveMotionMetrics.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params: CollectiveMotionMetrics.Params = self.Params.from_overrides(params)

    # --- State protocol (stateless per-sequence feature) ---

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

    def _areas(
        self, work: pd.DataFrame, keys: list[str], n_rows: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-key spanned area and admitted-triangle count, in sorted key order."""
        p = self.params
        area = np.full(n_rows, np.nan, dtype=float)
        n_tri = np.zeros(n_rows, dtype=np.int64)
        if p.area_method == "none":
            return area, n_tri
        for pos, (_, g) in enumerate(work.groupby(keys, sort=True)):
            pts = np.column_stack(
                (g["_x"].to_numpy(dtype=float), g["_y"].to_numpy(dtype=float))
            )
            pts = pts[np.isfinite(pts).all(axis=1)]
            if pts.shape[0] == 0:
                continue
            pts = np.unique(pts, axis=0)
            if p.area_method == "alpha_shape":
                # alpha is not None here: the Params validator enforces it.
                value, count = alpha_shape_area(pts, float(p.alpha or 0.0))
                area[pos] = value
                n_tri[pos] = count
            else:
                area[pos] = hull_area(pts)
        return area, n_tri

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        p = self.params
        order_col = resolve_order_col(df)
        required = [C.id_col, C.x_col, C.y_col]
        if p.subgroup_col is not None:
            required.append(p.subgroup_col)
        if p.speed_col is not None:
            required.append(p.speed_col)
        ensure_columns(df, required)

        if p.filter_expr:
            df = df.query(p.filter_expr)
            if df.empty:
                return pd.DataFrame()
        df = apply_exclude_cols(df, p.exclude_cols)
        if df.empty:
            return pd.DataFrame()

        # Sorting by (id, order) is what makes the backward differences below
        # per-individual. The per-frame reductions are groupby-based and do not
        # depend on row order.
        df = df.sort_values([C.id_col, order_col], kind="stable").reset_index(drop=True)
        n = len(df)

        x, y, finite = scrub_positions(
            df[C.x_col].to_numpy(dtype=float), df[C.y_col].to_numpy(dtype=float)
        )
        order = df[order_col].to_numpy(dtype=float)
        same_id, dstep = step_masks(df[C.id_col].to_numpy(), order)

        angle = (
            df[C.orientation_col].to_numpy(dtype=float)
            if C.orientation_col in df.columns
            else None
        )
        resolved = resolve_heading_source(p.heading_source, angle)
        ux, uy = unit_headings(resolved, angle, x, y, same_id)
        # An individual whose position is unknown is not a confirmed member of
        # the group, whatever its recorded angle. Masking here keeps
        # n_ids_heading <= n_ids, the invariant a reader audits the row with.
        ux = np.where(finite, ux, np.nan)
        uy = np.where(finite, uy, np.nan)

        keys = [order_col] if p.subgroup_col is None else [order_col, p.subgroup_col]

        work = pd.DataFrame(index=df.index)
        work[order_col] = df[order_col].to_numpy()
        if p.subgroup_col is not None:
            work[p.subgroup_col] = df[p.subgroup_col].to_numpy()
        work["_x"] = x
        work["_y"] = y
        work["_ux"] = ux
        work["_uy"] = uy
        work["_finite"] = finite
        if C.time_col in df.columns:
            work[C.time_col] = df[C.time_col].to_numpy()
        for meta_col in (C.seq_col, C.group_col):
            if meta_col in df.columns:
                work[meta_col] = df[meta_col].to_numpy()
        if p.speed_col is not None:
            work[p.speed_col] = df[p.speed_col].to_numpy()

        # Centroid broadcast back onto every row. groupby-mean skips NaN, which
        # after scrub_positions is exactly a finite-only mean.
        cx = (
            work.groupby(keys, sort=False)["_x"].transform("mean").to_numpy(dtype=float)
        )
        cy = (
            work.groupby(keys, sort=False)["_y"].transform("mean").to_numpy(dtype=float)
        )

        rx = x - cx
        ry = y - cy
        rhx, rhy, at_centre = unit_radial(rx, ry)
        work["_cross"] = cross2(rhx, rhy, ux, uy)
        work["_at_centroid"] = at_centre & finite
        work["_rxx"] = rx * rx
        work["_ryy"] = ry * ry
        work["_rxy"] = rx * ry

        # Per-individual velocity, over the id set common to both frames. Taking
        # the mean of these is the derivative of the common-set centroid;
        # differencing the all-ids centroid instead makes a track that merely
        # appears look like the whole group lurched.
        time = (
            df[C.time_col].to_numpy(dtype=float) if C.time_col in df.columns else None
        )
        dt = backward_dt(
            order,
            order_col == C.frame_col,
            time,
            p.fps,
            same_id,
            dstep,
            p.max_frame_gap,
        )
        prev_finite = np.zeros(n, dtype=bool)
        prev_finite[1:] = finite[:-1]
        ok_v = same_id & finite & prev_finite & np.isfinite(dt)
        if p.subgroup_col is not None:
            sub = df[p.subgroup_col].to_numpy()
            same_sub = np.zeros(n, dtype=bool)
            if n >= 2:
                same_sub[1:] = sub[1:] == sub[:-1]
            # An individual that changed subgroup between frames contributes two
            # positions measured against two different origins -- a large
            # spurious signal at exactly every fission and fusion.
            ok_v &= same_sub
        dxp = np.full(n, np.nan, dtype=float)
        dyp = np.full(n, np.nan, dtype=float)
        if n >= 2:
            dxp[1:] = x[1:] - x[:-1]
            dyp[1:] = y[1:] - y[:-1]
        work["_vx"] = np.where(ok_v, dxp / dt, np.nan)
        work["_vy"] = np.where(ok_v, dyp / dt, np.nan)

        cvx = work.groupby(keys, sort=False)["_vx"].transform("mean").to_numpy(float)
        cvy = work.groupby(keys, sort=False)["_vy"].transform("mean").to_numpy(float)
        r2 = rx * rx + ry * ry
        r2 = np.where(r2 > 0, r2, np.nan)
        work["_angvel"] = (
            cross2(
                rx,
                ry,
                work["_vx"].to_numpy(float) - cvx,
                work["_vy"].to_numpy(float) - cvy,
            )
            / r2
        )

        named: dict[str, pd.NamedAgg] = {
            "centroid_x": pd.NamedAgg(column="_x", aggfunc="mean"),
            "centroid_y": pd.NamedAgg(column="_y", aggfunc="mean"),
            "_sum_ux": pd.NamedAgg(column="_ux", aggfunc="sum"),
            "_sum_uy": pd.NamedAgg(column="_uy", aggfunc="sum"),
            "n_ids_heading": pd.NamedAgg(column="_ux", aggfunc="count"),
            "_sum_cross": pd.NamedAgg(column="_cross", aggfunc="sum"),
            "_n_cross": pd.NamedAgg(column="_cross", aggfunc="count"),
            "_cvx": pd.NamedAgg(column="_vx", aggfunc="mean"),
            "_cvy": pd.NamedAgg(column="_vy", aggfunc="mean"),
            "n_centroid_common": pd.NamedAgg(column="_vx", aggfunc="count"),
            "_sxx": pd.NamedAgg(column="_rxx", aggfunc="mean"),
            "_syy": pd.NamedAgg(column="_ryy", aggfunc="mean"),
            "_sxy": pd.NamedAgg(column="_rxy", aggfunc="mean"),
            "n_ids": pd.NamedAgg(column="_finite", aggfunc="sum"),
            "n_at_centroid": pd.NamedAgg(column="_at_centroid", aggfunc="sum"),
            "group_angvel": pd.NamedAgg(column="_angvel", aggfunc="mean"),
        }
        if C.time_col in work.columns:
            named[C.time_col] = pd.NamedAgg(column=C.time_col, aggfunc="first")
        # Identity travels with the frame, not from row 0 of the whole input.
        # With overlap the input spans the neighbouring sequences and row 0 is
        # the previous one's; a frame belongs to exactly one sequence, so taking
        # it per group is exact whether or not overlap is in play.
        for meta_col in (C.seq_col, C.group_col):
            if meta_col in work.columns:
                named[meta_col] = pd.NamedAgg(column=meta_col, aggfunc="first")
        if p.speed_col is not None:
            named["mean_speed"] = pd.NamedAgg(column=p.speed_col, aggfunc="mean")

        agg = work.groupby(keys, sort=True).agg(**named).reset_index()
        m = len(agg)

        n_heading = agg["n_ids_heading"].to_numpy(dtype=float)
        sum_ux = agg["_sum_ux"].to_numpy(dtype=float)
        sum_uy = agg["_sum_uy"].to_numpy(dtype=float)
        pol = polarization(sum_ux, sum_uy, n_heading)
        mean_heading = np.where(n_heading > 0, np.arctan2(sum_uy, sum_ux), np.nan)

        # The rotation numerator drops individuals sitting on the centroid (their
        # radial direction is undefined) while the divisor keeps them, so O_r
        # shares O_p's divisor and stays in [0, 1]. The resulting bias is towards
        # zero, i.e. away from Milling; n_at_centroid makes it auditable.
        #
        # When *nobody* contributed a radial term the parameter is unmeasured,
        # not zero -- and a pandas sum over an all-NaN group is 0.0, not NaN, so
        # this has to be said explicitly. A lone individual sits on its own
        # centroid, and without the guard it would report (O_p, O_r) = (1, 0):
        # perfectly polarized, with maximum confidence, from one animal.
        #
        # np.where evaluates both branches, so the divisor is masked rather than
        # the result -- otherwise an empty frame raises a divide warning that a
        # caller running under -W error sees as a crash.
        n_cross = agg["_n_cross"].to_numpy(dtype=float)
        rotation_divisor = np.where((n_heading > 0) & (n_cross > 0), n_heading, np.nan)
        rotation_signed = agg["_sum_cross"].to_numpy(dtype=float) / rotation_divisor
        rotation = np.abs(rotation_signed)

        cvx_agg = agg["_cvx"].to_numpy(dtype=float)
        cvy_agg = agg["_cvy"].to_numpy(dtype=float)
        centroid_speed = np.hypot(cvx_agg, cvy_agg)
        centroid_heading = np.where(
            centroid_speed > p.min_group_speed, np.arctan2(cvy_agg, cvx_agg), np.nan
        )

        sd_major, sd_minor, elongation, axis_angle = principal_axes(
            agg["_sxx"].to_numpy(dtype=float),
            agg["_syy"].to_numpy(dtype=float),
            agg["_sxy"].to_numpy(dtype=float),
        )
        area, n_tri = self._areas(work, keys, m)
        n_ids = agg["n_ids"].to_numpy(dtype=float)
        area_divisor = np.where(np.isfinite(area) & (area > 0), area, np.nan)
        density = n_ids / area_divisor

        group_angvel = agg["group_angvel"].to_numpy(dtype=float)

        # Below min_individuals the relational metrics are not measurements. The
        # centroid and the counts are never nulled: the centroid is exactly what
        # it claims even for one individual, and the counts are the evidence.
        weak = n_ids < p.min_individuals
        weak_h = n_heading < p.min_individuals
        pol = np.where(weak_h, np.nan, pol)
        mean_heading = np.where(weak_h, np.nan, mean_heading)
        # O_r depends on positions through r_hat as much as on headings, so it
        # needs both counts.
        rot_weak = weak | weak_h
        rotation = np.where(rot_weak, np.nan, rotation)
        rotation_signed = np.where(rot_weak, np.nan, rotation_signed)
        group_angvel = np.where(weak, np.nan, group_angvel)
        sd_major = np.where(weak, np.nan, sd_major)
        sd_minor = np.where(weak, np.nan, sd_minor)
        elongation = np.where(weak, np.nan, elongation)
        axis_angle = np.where(weak, np.nan, axis_angle)
        area = np.where(weak, np.nan, area)
        density = np.where(weak, np.nan, density)

        out = pd.DataFrame({order_col: agg[order_col].to_numpy()})
        if C.time_col in agg.columns:
            out[C.time_col] = agg[C.time_col].to_numpy()
        if p.subgroup_col is not None:
            out[p.subgroup_col] = agg[p.subgroup_col].to_numpy()
        out["n_ids"] = agg["n_ids"].to_numpy()
        out["n_ids_heading"] = agg["n_ids_heading"].to_numpy()
        out["n_at_centroid"] = agg["n_at_centroid"].to_numpy()
        out["n_centroid_common"] = agg["n_centroid_common"].to_numpy()
        out["polarization"] = pol
        out["mean_heading"] = mean_heading
        out["rotation"] = rotation
        out["rotation_signed"] = rotation_signed
        out["group_angvel"] = group_angvel
        out["state"] = classify_state(rotation, pol)
        out["centroid_x"] = agg["centroid_x"].to_numpy()
        out["centroid_y"] = agg["centroid_y"].to_numpy()
        out["centroid_speed"] = centroid_speed
        out["centroid_heading"] = centroid_heading
        out["sd_major"] = sd_major
        out["sd_minor"] = sd_minor
        out["elongation"] = elongation
        out["principal_axis_angle"] = axis_angle
        out["area"] = area
        out["density"] = density
        if p.area_method == "alpha_shape":
            out["alpha_n_triangles"] = np.where(weak, 0, n_tri)
        if p.speed_col is not None:
            out["mean_speed"] = agg["mean_speed"].to_numpy()
        out["heading_source"] = resolved
        for meta_col in (C.seq_col, C.group_col):
            if meta_col in agg.columns:
                out[meta_col] = agg[meta_col].to_numpy()

        return out.reset_index(drop=True)
