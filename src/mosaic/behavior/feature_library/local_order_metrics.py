"""
LocalOrderMetrics -- per-individual local collective-motion order parameters.

The per-individual counterpart of ``collective-motion-metrics``: the same two
order parameters of Tunstrom et al. 2013, restricted to a disc around each focal
individual (Fig 7B / S10) and decomposed into concentric shells about the group's
center of mass (Fig 7D / S11). One row per (frame, id).

Where the group-level feature answers "what is the shoal doing", this answers
"what does the shoal look like from where this individual is standing" -- which
is how the paper shows that local structure is nearly identical across group
sizes even as the global state changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self, TypeAlias, final

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import (
    BodyScaleResult,
    DependencyLookup,
    Inputs,
    InputStream,
    Params,
    Result,
    TrackInput,
    resolve_order_col,
)

from .collective_math import (
    HeadingSource,
    cross2,
    resolve_heading_source,
    scrub_positions,
    step_masks,
    unit_headings,
    unit_radial,
)
from .helpers import apply_exclude_cols, ensure_columns
from .registry import register_feature

RadiusUnits: TypeAlias = Literal["position", "body_scale"]

# Columns this feature emits, for the subgroup_col collision check.
EMITTED_COLUMNS: frozenset[str] = frozenset(
    {
        "local_heading_source",
        "local_radius",
        "heading_x",
        "heading_y",
        "n_local_neighbors",
        "n_local_headings",
        "local_polarization",
        "local_heading_x",
        "local_heading_y",
        "dist_to_group_center",
        "radial_rotation_self",
        "local_rotation",
        "group_outer_radius",
        "shell_index",
        "shell_n",
        "shell_radial_rotation",
    }
)

# Columns whose presence means each individual appears more than once per frame.
_PAIR_MARKERS = ("id1", "id2", "perspective")


@final
@register_feature
class LocalOrderMetrics:
    """
    Per-(frame, id) local order parameters (Tunstrom et al. 2013, Fig 7B/D).

    Reference: Tunstrom K, Katz Y, Ioannou CC, Huepe C, Lutz MJ, Couzin ID (2013)
    "Collective States, Multistability and Transitional Behavior in Schooling
    Fish", PLoS Comput Biol 9(2): e1002915.

    Output columns (one row per surviving input row):

    Provenance
      - ``local_heading_source``: ``"orientation"`` or ``"velocity"``, the
        runtime resolution of the ``heading_source`` param. Named disjointly
        from the group feature's ``heading_source`` so merging the two outputs
        does not silently suffix one of them
      - ``local_radius``: the radius actually used, in position units, which is
        what makes a ``body_scale`` resolution auditable

    The focal individual
      - ``heading_x`` / ``heading_y``: its unit heading, so every column below is
        re-derivable from the output alone
      - ``n_local_neighbors``: position-valid *other* individuals inside the
        disc. ``-1`` when the focal's own position is not finite; ``0`` is a
        genuinely isolated but valid focal
      - ``n_local_headings``: heading-valid *other* disc members -- the count
        ``min_neighbors`` gates on

    Local order (Fig 7B / S10)
      - ``local_polarization``: ``O_p`` over the disc, focal included -- it sits
        at distance zero in its own circle, and excluding it would leave an
        isolated individual undefined rather than trivially polarized
      - ``local_heading_x`` / ``local_heading_y``: the mean unit heading whose
        magnitude is ``local_polarization``. Kept because the magnitude alone
        discards which way the neighborhood is pointing
      - ``local_rotation``: mean of ``radial_rotation_self`` over the disc

    Radial structure (Fig 7D / S11)
      - ``dist_to_group_center``: distance from the frame's center of mass
      - ``radial_rotation_self``: this individual's own
        ``cross(r_hat, u)`` about the center of mass, signed
      - ``group_outer_radius``: ``R_out``, the median distance-to-center of the
        ``n_peripheral`` most peripheral individuals
      - ``shell_index`` in ``0 .. n_shells-1``, ``-1`` when undefined; ``shell_n``
      - ``shell_radial_rotation``: mean ``radial_rotation_self`` over the shell,
        broadcast to each of its members

    Params:
        radius: Neighborhood radius. **Required, deliberately.** The paper's is
            three body lengths (15.6 cm at BL = 5.2 cm), but mosaic tracks are in
            pixels for every pixel-space converter and the schema records no
            units. A defaulted unit-bearing radius would produce a fully
            populated, entirely wrong table on any dataset at a different scale,
            with no error to notice.
        radius_units: ``"position"`` (default) reads ``radius`` in X/Y units.
            ``"body_scale"`` multiplies it by the sequence's mean ``body-scale``
            output -- which is the median pairwise distance among an individual's
            *finite* keypoints, not a body length, and the keypoint set varies
            per row so there is no fixed ratio to one. Calibrate by running
            ``body-scale``, reading its mean, and dividing the physical radius
            you want by that.
        body_scale: Reference to the upstream ``body-scale`` run; required under
            ``radius_units="body_scale"``. Pass the whole model
            (``{"feature": ..., "run_id": ...}``), not a partial dict: this field
            has no default to merge onto, deliberately, so a ``"position"`` run
            does not resolve a dependency it never uses.
        heading_source: As ``collective-motion-metrics``. ``ANGLE`` is radians.
            A head/tail identity flip inverts that individual's heading and
            destroys its local polarization; running once with ``"orientation"``
            and once with ``"velocity"`` and comparing is a cheap diagnostic.
        n_shells: Concentric shells about the center of mass. Default 6, the
            paper's.
        n_peripheral: Individuals whose median distance-to-center defines
            ``R_out``. Default 5, the paper's.
        min_neighbors: Minimum heading-valid *other* disc members for the local
            metrics to be emitted. Default 1. Gating on the heading-valid count
            rather than the position-valid one matters: a focal with three
            neighbors none of whom has a usable heading would otherwise report
            a local polarization of 1.0 computed over itself alone.
        subgroup_col: Confines every disc, centroid, ``R_out`` and shell to one
            subgroup. Default None.
        filter_expr: ``DataFrame.query`` filter. Applied *after* headings are
            computed, so excluding a row cannot change its successor's heading.
        exclude_cols: Boolean columns whose truthy rows are dropped, same timing.

    Notes:
        ``local_rotation`` averages each disc member's rotation about the
        **group** center, not about a per-focal local center of mass. Inside a
        mill of radius R much larger than the disc radius every disc member moves
        nearly parallel, so a local-center version averages to zero in the one
        regime rotation exists to detect. The paper defines ``O_r`` only
        group-wide; this is the natural local extension of it.

        The shell columns are broadcast onto every member row rather than emitted
        as a per-shell table, because the pipeline aligns inputs on
        ``{frame, time, id, id1, id2}`` only -- a table keyed on
        ``(frame, shell_index)`` would fan every shell across every individual in
        the frame. Nothing is lost: ``out.drop_duplicates(["frame",
        "shell_index"])`` reconstructs the per-shell table, and the reverse is
        impossible.

        With small groups (N of about 8 or fewer, which is mosaic's common case
        rather than the paper's 30 to 300) most of the six shells are empty and
        ``shell_radial_rotation`` equals ``radial_rotation_self`` on most rows.
        Lower ``n_shells`` accordingly.

        Cost is O(N^2) per frame, of the same order as ``nearest-neighbor``.

        Input must carry one row per (frame, id); pair-shaped output is refused
        rather than silently double-counted.

        Every row in one neighborhood must share a coordinate frame, which holds
        by construction today -- no track converter emits a ``camera`` column.
        When multi-camera tracks arrive, ``subgroup_col="camera"`` confines every
        disc, group center, ``R_out`` and shell to one view with no change here.

        ``overlap_frames`` is supported, including with a filter set: the trim
        selects on the frame axis rather than on row offsets, so rows the filter
        dropped cost nothing. The velocity heading is a backward difference, so
        context removes its NaN on the first frame of every sequence. Under
        ``radius_units="body_scale"`` each neighbourhood resolves its own entry's
        radius, so a neighbour's body scale never scales the core's discs.
    """

    category = "per-frame"
    name = "local-order-metrics"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        """Local-order-metrics parameters. See the class docstring."""

        radius: float = Field(gt=0)
        radius_units: RadiusUnits = "position"
        body_scale: BodyScaleResult | None = None
        heading_source: HeadingSource = "auto"
        n_shells: int = Field(default=6, ge=1)
        n_peripheral: int = Field(default=5, ge=1)
        min_neighbors: int = Field(default=1, ge=0)
        subgroup_col: str | None = None
        filter_expr: str | None = None
        exclude_cols: list[str] = Field(default_factory=list)

        @model_validator(mode="after")
        def _check(self) -> Self:
            """Reject at construction; a raise inside apply is a silent drop."""
            if self.radius_units == "body_scale" and self.body_scale is None:
                msg = (
                    "radius_units='body_scale' requires 'body_scale', a reference "
                    "to an upstream body-scale run, e.g. "
                    '{"feature": "body-scale__from__tracks"}.'
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
        inputs: LocalOrderMetrics.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params: LocalOrderMetrics.Params = self.Params.from_overrides(params)
        self._scale_lookup: DependencyLookup | None = None

    # --- State protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._scale_lookup = dependency_lookups.get("body_scale")
        if self.params.radius_units == "body_scale" and not self._scale_lookup:
            # Raise rather than return False: load_state runs once in the parent
            # process, so a raise here is fatal and visible, whereas False means
            # "not fitted" and sends the feature into fit(). The same check
            # inside apply() would be caught per entry and the run would report
            # success having written nothing.
            msg = (
                "radius_units='body_scale' but the body-scale dependency "
                "resolved to no entries. Run body-scale first and reference it "
                'as {"body_scale": {"feature": "body-scale__from__tracks"}}.'
            )
            raise ValueError(msg)
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    # --- Apply ---

    def _resolve_radius(self, group: str, sequence: str) -> float:
        """Radius in position units, raising rather than substituting a default.

        A silently substituted radius changes every number this feature emits
        with nothing in the output to signal it.
        """
        p = self.params
        if p.radius_units == "position":
            return p.radius
        if self._scale_lookup is None:
            msg = "radius_units='body_scale' but no body-scale lookup was loaded"
            raise ValueError(msg)
        path = self._scale_lookup.get((group, sequence))
        if path is None:
            msg = (
                f"radius_units='body_scale' but body-scale has no output for "
                f"({group!r}, {sequence!r})."
            )
            raise ValueError(msg)
        mean_scale = float(pd.read_parquet(path)["scale"].dropna().mean())
        if not np.isfinite(mean_scale) or mean_scale <= 0:
            msg = (
                f"body-scale for ({group!r}, {sequence!r}) has a non-positive "
                f"mean of {mean_scale!r}; cannot scale the radius by it."
            )
            raise ValueError(msg)
        return p.radius * mean_scale

    @staticmethod
    def _entries_present(df: pd.DataFrame) -> list[tuple[str, str]]:
        """Every ``(group, sequence)`` the frame holds, deduplicated and sorted.

        One entry without overlap; the core plus its neighbours with it. Sorted so
        the radius lookups happen in a fixed order, which matters because
        resolving one can raise.
        """
        if C.group_col not in df.columns and C.seq_col not in df.columns:
            return [("", "")]
        groups = df[C.group_col].astype(str) if C.group_col in df.columns else ""
        sequences = df[C.seq_col].astype(str) if C.seq_col in df.columns else ""
        pairs = pd.DataFrame({"group": groups, "sequence": sequences})
        unique = pairs.drop_duplicates().sort_values(["group", "sequence"])
        return [(str(row.group), str(row.sequence)) for row in unique.itertuples()]

    def _refuse_duplicated_ids(self, df: pd.DataFrame, order_col: str) -> None:
        """Refuse input carrying an individual more than once per frame."""
        present = [c for c in _PAIR_MARKERS if c in df.columns]
        if present:
            msg = (
                f"local-order-metrics needs one row per (frame, id) but the input "
                f"carries {present}, which marks pair-shaped output. Every "
                f"neighborhood would double-count. Consume tracks or a per-id "
                f"feature instead."
            )
            raise ValueError(msg)
        if df.duplicated(subset=[order_col, C.id_col]).any():
            msg = (
                f"local-order-metrics needs one row per ({order_col}, id) but the "
                f"input has duplicates."
            )
            raise ValueError(msg)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        p = self.params
        order_col = resolve_order_col(df)
        required = [C.id_col, C.x_col, C.y_col]
        if p.subgroup_col is not None:
            required.append(p.subgroup_col)
        ensure_columns(df, required)
        self._refuse_duplicated_ids(df, order_col)

        # A radius per entry present in the frame, rather than one resolved from
        # row 0. Under overlap the frame spans the neighbouring sequences, and
        # ``radius_units="body_scale"`` reads a per-entry body-scale run -- so row
        # 0 would scale every disc, the core's included, by the *previous*
        # sequence's mean. Identity is constant within a frame, so each
        # neighbourhood below is measured against its own entry's radius.
        radius_by_entry = {
            entry: self._resolve_radius(*entry) for entry in self._entries_present(df)
        }

        df = df.sort_values([C.id_col, order_col], kind="stable").reset_index(drop=True)

        # Headings first, exclusion second. The velocity heading is a backward
        # difference within one individual, so dropping a row before computing it
        # would silently change the heading of the next surviving row -- a row
        # the caller did not exclude.
        x, y, finite = scrub_positions(
            df[C.x_col].to_numpy(dtype=float), df[C.y_col].to_numpy(dtype=float)
        )
        same_id, _ = step_masks(
            df[C.id_col].to_numpy(), df[order_col].to_numpy(dtype=float)
        )
        angle = (
            df[C.orientation_col].to_numpy(dtype=float)
            if C.orientation_col in df.columns
            else None
        )
        resolved = resolve_heading_source(p.heading_source, angle)
        ux, uy = unit_headings(resolved, angle, x, y, same_id)
        ux = np.where(finite, ux, np.nan)
        uy = np.where(finite, uy, np.nan)

        work = df.copy()
        work["_x"] = x
        work["_y"] = y
        work["_ux"] = ux
        work["_uy"] = uy
        work["_finite"] = finite
        if p.filter_expr:
            work = work.query(p.filter_expr)
            if work.empty:
                return pd.DataFrame()
        work = apply_exclude_cols(work, p.exclude_cols)
        if work.empty:
            return pd.DataFrame()
        work = work.reset_index(drop=True)

        n = len(work)
        n_neighbors = np.full(n, -1, dtype=np.int32)
        n_headings = np.full(n, -1, dtype=np.int32)
        local_pol = np.full(n, np.nan, dtype=float)
        local_hx = np.full(n, np.nan, dtype=float)
        local_hy = np.full(n, np.nan, dtype=float)
        dist_center = np.full(n, np.nan, dtype=float)
        self_rr = np.full(n, np.nan, dtype=float)
        local_rot = np.full(n, np.nan, dtype=float)
        outer_radius = np.full(n, np.nan, dtype=float)
        shell_index = np.full(n, -1, dtype=np.int16)
        shell_pop = np.full(n, -1, dtype=np.int32)
        shell_rr = np.full(n, np.nan, dtype=float)

        keys = [order_col] if p.subgroup_col is None else [p.subgroup_col, order_col]
        # Filled per neighbourhood from that neighbourhood's own entry, and
        # emitted, so a body_scale resolution stays auditable row by row.
        radius_of_row = np.full(n, np.nan, dtype=float)

        def _entry_of(rows: pd.DataFrame) -> tuple[str, str]:
            """The entry a frame's rows belong to. Constant within a frame."""
            group_name = (
                str(rows[C.group_col].iloc[0]) if C.group_col in rows.columns else ""
            )
            sequence_name = (
                str(rows[C.seq_col].iloc[0]) if C.seq_col in rows.columns else ""
            )
            return group_name, sequence_name

        # ffgroups marks a non-event row with -1. That is a pooled pseudo-group,
        # not a subgroup: treating it as one would invent neighborhoods out of
        # animals that were never grouped. Those rows keep the sentinels.
        skip = np.zeros(n, dtype=bool)
        if p.subgroup_col is not None and pd.api.types.is_numeric_dtype(
            work[p.subgroup_col]
        ):
            skip = work[p.subgroup_col].to_numpy(dtype=float) < 0
        work["_skip"] = skip

        for _, g in work.groupby(keys, sort=False):
            if bool(g["_skip"].iloc[0]):
                continue
            idx = g.index.to_numpy()
            radius = radius_by_entry[_entry_of(g)]
            radius_of_row[idx] = radius
            r2 = radius * radius
            gx = g["_x"].to_numpy(dtype=float)
            gy = g["_y"].to_numpy(dtype=float)
            gux = g["_ux"].to_numpy(dtype=float)
            guy = g["_uy"].to_numpy(dtype=float)
            ok_p = g["_finite"].to_numpy(dtype=bool)
            if not ok_p.any():
                continue
            ok_u = ok_p & np.isfinite(gux) & np.isfinite(guy)

            with np.errstate(invalid="ignore"):
                d2 = (gx[None, :] - gx[:, None]) ** 2 + (gy[None, :] - gy[:, None]) ** 2
                # A NaN comparison is False, so an invalid row leaves no trace;
                # the explicit conjunction also drops an invalid focal's own
                # diagonal, which would otherwise be 0 <= r2.
                disc = (d2 <= r2) & ok_p[None, :] & ok_p[:, None]

            n_neighbors[idx] = np.where(ok_p, disc.sum(axis=1) - 1, -1)

            # Elementwise multiply then sum, never a matmul: (n,n) @ (n,) hands
            # the reduction to a threaded BLAS whose summation order varies with
            # the thread count, which would forfeit determinism.
            ux0 = np.where(ok_u, gux, 0.0)
            uy0 = np.where(ok_u, guy, 0.0)
            du = (disc & ok_u[None, :]).astype(float)
            cnt_u = du.sum(axis=1)
            n_headings[idx] = np.where(ok_p, cnt_u - ok_u.astype(np.int32), -1)
            sx = (du * ux0[None, :]).sum(axis=1)
            sy = (du * uy0[None, :]).sum(axis=1)
            denom_u = np.where(cnt_u > 0, cnt_u, np.nan)
            local_hx[idx] = sx / denom_u
            local_hy[idx] = sy / denom_u
            local_pol[idx] = np.hypot(sx, sy) / denom_u

            gcx = float(np.nanmean(np.where(ok_p, gx, np.nan)))
            gcy = float(np.nanmean(np.where(ok_p, gy, np.nan)))
            rgx = gx - gcx
            rgy = gy - gcy
            rad = np.where(ok_p, np.hypot(rgx, rgy), np.nan)
            dist_center[idx] = rad
            rhx, rhy, _ = unit_radial(rgx, rgy)
            rr = np.where(ok_u, cross2(rhx, rhy, gux, guy), np.nan)
            self_rr[idx] = rr

            # Sum of zero-filled over count, not nanmean: an all-NaN row makes
            # nanmean emit a RuntimeWarning that np.errstate does not suppress.
            rr_ok = np.isfinite(rr)
            du_rr = (disc & rr_ok[None, :]).astype(float)
            cnt_rr = du_rr.sum(axis=1)
            sum_rr = (du_rr * np.where(rr_ok, rr, 0.0)[None, :]).sum(axis=1)
            local_rot[idx] = sum_rr / np.where(cnt_rr > 0, cnt_rr, np.nan)

            thin = n_headings[idx] < p.min_neighbors
            local_pol[idx[thin]] = np.nan
            local_hx[idx[thin]] = np.nan
            local_hy[idx[thin]] = np.nan
            local_rot[idx[thin]] = np.nan

            # R_out: the median distance-to-center of the n_peripheral most
            # peripheral individuals (Fig S11). np.partition is O(n) and a median
            # is order-independent, so ties cannot break determinism.
            rv = rad[np.isfinite(rad)]
            if rv.size == 0:
                continue
            k = min(p.n_peripheral, rv.size)
            peripheral = np.partition(rv, rv.size - k)[rv.size - k :]
            outer = float(np.median(peripheral))
            outer_radius[idx] = outer
            if not np.isfinite(outer) or outer <= 0:
                continue

            width = outer / p.n_shells
            with np.errstate(invalid="ignore"):
                raw_shell = np.floor(rad / width)
            # The clip is what implements "the outermost shell includes
            # peripheral fish": R_out is a median, so about half the peripheral
            # set lies beyond it and would otherwise fall off the end.
            clipped = np.clip(raw_shell, 0, p.n_shells - 1)
            s = np.where(np.isfinite(rad), clipped, -1).astype(np.int16)
            shell_index[idx] = s

            pop = np.zeros(p.n_shells, dtype=np.int32)
            acc = np.zeros(p.n_shells, dtype=float)
            cnt = np.zeros(p.n_shells, dtype=float)
            np.add.at(pop, s[s >= 0], 1)
            has_val = (s >= 0) & rr_ok
            np.add.at(acc, s[has_val], rr[has_val])
            np.add.at(cnt, s[has_val], 1.0)
            per_shell = acc / np.where(cnt > 0, cnt, np.nan)
            safe = np.where(s >= 0, s, 0)
            shell_pop[idx] = np.where(s >= 0, pop[safe], -1)
            shell_rr[idx] = np.where(s >= 0, per_shell[safe], np.nan)

        out = pd.DataFrame(
            {
                "local_heading_source": resolved,
                "local_radius": radius_of_row,
                "heading_x": work["_ux"].to_numpy(dtype=float),
                "heading_y": work["_uy"].to_numpy(dtype=float),
                "n_local_neighbors": n_neighbors,
                "n_local_headings": n_headings,
                "local_polarization": local_pol,
                "local_heading_x": local_hx,
                "local_heading_y": local_hy,
                "dist_to_group_center": dist_center,
                "radial_rotation_self": self_rr,
                "local_rotation": local_rot,
                "group_outer_radius": outer_radius,
                "shell_index": shell_index,
                "shell_n": shell_pop,
                "shell_radial_rotation": shell_rr,
            },
            index=work.index,
        )
        carry = sorted(C.meta_set() & set(work.columns))
        if p.subgroup_col is not None and p.subgroup_col not in carry:
            carry.append(p.subgroup_col)
        return out.join(work[carry])
