from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self, final

import numpy as np
import pandas as pd
from pydantic import model_validator

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

from .helpers import ensure_columns
from .registry import register_feature

# Every column this feature emits, for the subgroup_col collision check. A
# subgroup column sharing a name with an output would give the assembled frame
# two identically-labeled columns. The conditional ones are listed too: the set
# is what the feature *may* emit, not what a particular run does.
EMITTED_COLUMNS: frozenset[str] = frozenset(
    {
        "nn_align",
        "frac_nn_ahead",
        "nn_bearing_x",
        "nn_bearing_y",
        "speed_match_nn",
        "speed_match_group",
        "speed_cv",
        "speed_rcv",
        "accel_mean",
        "accel_med",
        "accel_cv",
        "accel_rcv",
        "jerk_mean",
        "jerk_med",
        "jerk_cv",
        "jerk_rcv",
        "kick_rate",
        "burst_coast_ratio",
        "n_frames",
        "n_social_frames",
        "n_accel",
        "n_jerk",
    }
)


def _safe_ratio(num: float, den: float) -> float:
    """``num / den`` guarded against a zero / non-finite denominator."""
    if not np.isfinite(den) or den == 0.0:
        return float("nan")
    return float(num / den)


def _iqr(vals: np.ndarray) -> float:
    """Interquartile range (75th - 25th percentile), NaN-safe."""
    if vals.size == 0 or not np.any(np.isfinite(vals)):
        return float("nan")
    q75, q25 = np.nanpercentile(vals, [75.0, 25.0])
    return float(q75 - q25)


def _dispersion(vals: np.ndarray) -> tuple[float, float]:
    """Return (CV = std/mean, robust-CV = IQR/median) for a value series.

    The robust form is more stable on noisy tracking data where a few large
    outliers inflate the mean/std.
    """
    if vals.size == 0 or not np.any(np.isfinite(vals)):
        return float("nan"), float("nan")
    mean = float(np.nanmean(vals))
    std = float(np.nanstd(vals))
    med = float(np.nanmedian(vals))
    cv = _safe_ratio(std, mean)
    rcv = _safe_ratio(_iqr(vals), med)
    return cv, rcv


def _magnitude_stats(prefix: str, mag: np.ndarray) -> dict[str, float]:
    """Mean/median central tendency plus both dispersion forms for ``|mag|``.

    ``mag`` is already a magnitude array (e.g. ``|accel|``). Emits
    ``{prefix}_mean``, ``{prefix}_med``, ``{prefix}_cv`` (std/mean) and
    ``{prefix}_rcv`` (IQR/median).
    """
    if mag.size == 0 or not np.any(np.isfinite(mag)):
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_med": float("nan"),
            f"{prefix}_cv": float("nan"),
            f"{prefix}_rcv": float("nan"),
        }
    mean = float(np.nanmean(mag))
    med = float(np.nanmedian(mag))
    std = float(np.nanstd(mag))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_med": med,
        f"{prefix}_cv": _safe_ratio(std, mean),
        f"{prefix}_rcv": _safe_ratio(_iqr(mag), med),
    }


def _derivative(values: np.ndarray, frames: np.ndarray, fps: float) -> np.ndarray:
    """Discrete time-derivative ``d(values)/dt`` with ``dt = d(frame) / fps``.

    Returns an array of length ``len(values) - 1`` aligned to the intervals
    between consecutive samples. Intervals with a non-positive frame gap are
    set to NaN so gaps in the track do not create spurious spikes.
    """
    if values.size < 2:
        return np.empty(0, dtype=float)
    dv = np.diff(values)
    dframe = np.diff(frames)
    dt = dframe / fps
    dt = np.where(dt > 0, dt, np.nan)
    return dv / dt


def _subgroups(g: pd.DataFrame, col: str | None) -> list[tuple[object, np.ndarray]]:
    """``(value, sample-mask)`` pairs for one fish, in a deterministic order.

    ``col=None`` yields exactly one pair whose mask selects every sample, so the
    split and the unsplit path are one code path and the unsplit numbers stay
    the numbers they were.

    Values are taken off the original column, so an integer ``group_size`` emits
    an integer column that merges against ``ffgroups-metrics`` without a dtype
    mismatch. Rows whose value is NaN belong to no subgroup: ``== NaN`` is False
    everywhere, so they fall out of every mask rather than forming a key nothing
    downstream could match.
    """
    if col is None:
        return [(None, np.ones(len(g), dtype=bool))]
    series = g[col]
    return [
        (value, (series == value).to_numpy(dtype=bool))
        for value in sorted(series.dropna().unique())
    ]


@dataclass(frozen=True)
class _FishSeries:
    """One fish's full, time-ordered series plus the intervals derived from it.

    Built once per fish and then reduced under one or more sample masks. The
    derivatives live on the **full** series deliberately. Differentiating a
    subgroup's rows alone would put two moments minutes apart side by side and
    call the difference an acceleration, and -- the worse failure -- the time
    base would then drift with the subgroup: fission/fusion boundaries flicker
    at the distance cutoff, so one subgroup's samples might be 2 frames apart
    and another's 3, making their derivatives differently smoothed quantities
    that cannot be compared. Computed once, a mask can only *remove* intervals,
    never invent one, so every retained interval carries the value it would have
    had with no split at all.

    ``neighbor_speed`` and ``dev_group`` arrive already sliced to this fish.
    """

    speed: np.ndarray  # (n,)
    dframe: np.ndarray  # (n-1,)  frame gaps
    accel: np.ndarray  # (n-1,)  interval k spans samples k, k+1
    jerk: np.ndarray  # (n-2,)  interval j spans samples j, j+1, j+2
    social: np.ndarray  # (n,) bool
    neighbor_speed: np.ndarray  # (n,)
    dev_group: np.ndarray  # (n,)
    nn_angle: np.ndarray | None
    ego_x: np.ndarray | None
    ego_y: np.ndarray | None
    has_nn: bool
    has_gm: bool


@final
@register_feature
class SocialMotionSummary:
    """Per-fish summary of social-interaction and locomotor-style metrics.

    A ``summary`` feature built to provide mechanism / interaction metrics that
    are **not** mechanically tied to how often an individual is isolated --
    unlike isolation-event duration. It consumes already-computed per-frame
    features and reduces them per fish.

    **Output shape.** One row per ``id`` by default; one row per
    ``(id, subgroup value)`` when ``subgroup_col`` is set. A fish emits rows only
    for the values it was actually observed in, never the full cross product, so
    the output is ragged: a fish never seen in a group of five has no such row
    rather than an all-NaN one asserting it was measured and found empty.

    Consumes (merged on ``frame``/``id`` by the pipeline):
      - ``nearest-neighbor``: ``nn_id``, ``nn_delta_angle``,
        ``nn_delta_x_ego``, ``nn_delta_y_ego``
      - ``speed-angvel``: ``speed`` (or ``speed_smooth`` via ``speed_col``)
      - ``ffgroups`` (optional): ``group_membership``, ``group_size``

    Social metrics (over frames with a valid nearest neighbor):
      - ``nn_align``:        mean cos(nn_delta_angle) -- local heading alignment
      - ``frac_nn_ahead``:   fraction of social time the neighbor is ahead
      - ``nn_bearing_x/y``:  mean unit-vector bearing of the neighbor (ego frame)
      - ``speed_match_nn``:    mean |own speed - nearest-neighbor speed|
        (needs only ``nn_id`` + ``speed`` -- independent of group definitions).
        Note this is a mean absolute *difference*: larger means worse matched.
      - ``speed_match_group``: mean |own speed - group-mean speed| (needs
        ``group_membership``)

    Motion metrics (over the frames the row is about):
      - ``speed_cv`` / ``speed_rcv``: speed dispersion (std/mean and IQR/median)
      - ``accel_{mean,med,cv,rcv}``: acceleration magnitude |d speed / dt|
      - ``jerk_{mean,med,cv,rcv}``:  jerk magnitude |d accel / dt|
      - ``kick_rate`` / ``burst_coast_ratio`` (only if ``compute_burst_coast``)

    Sample sizes (emitted only when ``subgroup_col`` is set):
      - ``n_frames``, ``n_social_frames``, ``n_accel``, ``n_jerk``. Once rows are
        keyed by subgroup, a row built from 40 frames sits beside one built from
        30,000 and looks identical. ``n_accel`` is not derivable from
        ``n_frames``: the gap between them counts contiguity breaks, so it
        distinguishes one 6-frame episode from six 1-frame flickers.

    **How a split divides the frames.** A sample belongs to the subgroup whose
    value its row carries. An *interval* -- an acceleration or a jerk -- belongs
    to a subgroup only if every sample it touches does, so an interval straddling
    a change of subgroup counts for neither. That is what makes a row's
    acceleration a statement about behaviour at that subgroup value rather than
    about the transition into it.

    **``social_min_group_size`` still gates the social mask under a split.** With
    ``subgroup_col="group_size"`` and the default of 2, the ``group_size == 1``
    row therefore carries NaN for every social metric, which is the correct
    reading -- there is no neighbour to align with. Raising the threshold NaNs
    every subgroup row below it, which is what the parameter says but is
    surprising once rows are keyed by size.

    Two things to know before publishing a per-group-size row:

      - ``group_size == 1`` **conflates genuine isolation with non-detection.**
        ``ffgroups`` builds a full (frames x ids) grid; a fish whose position is
        NaN has NaN distances to everyone, joins no component, and reads as a
        group of one. Pooled into a per-fish row that is diluted; as its own row
        it looks like isolation.
      - At ``social_min_group_size=1`` a lone fish reports
        ``speed_match_group == 0.0`` exactly -- it is its own group, so it
        deviates from its own mean by zero. Not a bug; it reads as a perfect
        social match.

    ``subgroup_col="event"`` is not recommended: ``ffgroups`` fills non-event
    rows with ``-1`` and this feature has no ``filter_expr`` to drop them, so
    every non-event frame would pool into a ``-1`` pseudo-subgroup.

    A split output duplicates on ``id``, so feeding it to another feature
    alongside a second table that also duplicates on ``id`` (``ffgroups-metrics``
    is keyed on the very same ``(id, group_size)``) is refused by the pipeline's
    merge rather than fanned out silently. Read it from parquet and merge by hand
    on ``[id, subgroup_col]``.

    Params:
        fps: Frames per second, used to convert frame steps to seconds when
            differentiating speed. Default 30.0.
        speed_col: Column holding per-frame speed. Default "speed". Pass
            "speed_smooth" to reduce the Savitzky-Golay-filtered speed instead,
            which is the better input for the derivative-based metrics -- it
            exists only when ``speed-angvel`` ran with ``smooth_window`` set,
            which is why it is not the default.
        social_min_group_size: Minimum ``group_size`` for a frame to count as
            "social" (when ``group_size`` is available). Default 2.
        subgroup_col: Column whose values split the output into one row each,
            typically "group_size" from ``ffgroups``. Default None (one row per
            fish). Setting it also stops the motion metrics pooling isolated and
            social frames, which is usually the point: pooled, ``speed_rcv``
            partly tracks how *often* a fish is alone rather than how it swims.
        compute_burst_coast: If True, also emit a simple burst-and-coast
            gait summary (``kick_rate``, ``burst_coast_ratio``). Default False.
    """

    category = "summary"
    name = "social-motion-summary"
    version = "0.2"
    parallelizable = True
    scope_dependent = False
    # A summary carries no frame axis for the overlap trim to work on, and the
    # sequence/group metadata below is read from row 0 -- which under overlap
    # belongs to the neighbouring sequence.
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        fps: float = 30.0
        speed_col: str = "speed"
        social_min_group_size: int = 2
        subgroup_col: str | None = None
        compute_burst_coast: bool = False

        @model_validator(mode="after")
        def _check(self) -> Self:
            """Reject at construction, where a raise is visible.

            ``run_feature`` catches every exception out of ``apply``, prints one
            line and carries on, so a raise there is a silently dropped entry
            and an exit code of zero.
            """
            if self.subgroup_col is not None:
                reserved = C.meta_set() | EMITTED_COLUMNS
                if self.subgroup_col in reserved:
                    msg = (
                        f"subgroup_col={self.subgroup_col!r} collides with a "
                        f"metadata or emitted column. Use 'group_size' from "
                        f"ffgroups."
                    )
                    raise ValueError(msg)
            return self

    def __init__(
        self,
        inputs: SocialMotionSummary.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

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

    def _neighbor_speed(self, df: pd.DataFrame, order_col: str) -> np.ndarray:
        """Speed of each row's nearest neighbor, via a self-join on (frame, nn_id)."""
        speed_col = self.params.speed_col
        lut = df[[order_col, C.id_col, speed_col]].copy()
        lut[C.id_col] = lut[C.id_col].astype(float)
        lut = lut.rename(columns={C.id_col: "_nn_key", speed_col: "_nbr_speed"})
        left = pd.DataFrame(
            {
                order_col: df[order_col].to_numpy(),
                "_nn_key": df["nn_id"].astype(float).to_numpy(),
            }
        )
        merged = left.merge(lut, on=[order_col, "_nn_key"], how="left")
        return merged["_nbr_speed"].to_numpy(dtype=float)

    def _summarize(self, s: _FishSeries, sample: np.ndarray) -> dict[str, object]:
        """Reduce one fish's series under one sample mask into a metric row.

        ``sample`` selects the samples this row is about: every sample when
        ``subgroup_col`` is unset, one subgroup's when it is set. The interval
        masks are *derived* from it rather than recomputed from a subset, which
        is what keeps a derivative from spanning a sample the mask excludes:

            samples   0     1     2     3   ...  n-1
            accel           k=0   k=1   k=2 ...  k=n-2   spans samples k, k+1
            jerk                  j=0   j=1 ...  j=n-3   touches j, j+1, j+2

        With an all-True ``sample`` both masks are all-True and every value below
        is the value the unsplit feature produced.
        """
        p = self.params
        social = s.social & sample
        m_acc = sample[:-1] & sample[1:]
        m_jerk = m_acc[:-1] & m_acc[1:]

        row: dict[str, object] = {}

        # --- NN heading alignment ---
        if s.nn_angle is not None:
            ang = s.nn_angle[social]
            row["nn_align"] = (
                float(np.nanmean(np.cos(ang))) if ang.size else float("nan")
            )
        else:
            row["nn_align"] = float("nan")

        # --- Neighbor bearing preference ---
        if s.ego_x is not None and s.ego_y is not None:
            dxe = s.ego_x[social]
            dye = s.ego_y[social]
            if dxe.size:
                row["frac_nn_ahead"] = float(np.nanmean((dxe > 0).astype(float)))
                norm = np.sqrt(dxe**2 + dye**2)
                norm = np.where(norm > 0, norm, np.nan)
                row["nn_bearing_x"] = float(np.nanmean(dxe / norm))
                row["nn_bearing_y"] = float(np.nanmean(dye / norm))
            else:
                row["frac_nn_ahead"] = float("nan")
                row["nn_bearing_x"] = float("nan")
                row["nn_bearing_y"] = float("nan")
        else:
            row["frac_nn_ahead"] = float("nan")
            row["nn_bearing_x"] = float("nan")
            row["nn_bearing_y"] = float("nan")

        # --- Speed matching to nearest neighbor (group-free) ---
        if s.has_nn:
            diff = np.abs(s.speed[social] - s.neighbor_speed[social])
            row["speed_match_nn"] = (
                float(np.nanmean(diff))
                if diff.size and np.any(np.isfinite(diff))
                else float("nan")
            )
        else:
            row["speed_match_nn"] = float("nan")

        # --- Speed matching to group mean (needs group membership) ---
        if s.has_gm:
            dg = np.abs(s.dev_group[social])
            row["speed_match_group"] = (
                float(np.nanmean(dg))
                if dg.size and np.any(np.isfinite(dg))
                else float("nan")
            )
        else:
            row["speed_match_group"] = float("nan")

        # --- Motion / locomotor style ---
        cv, rcv = _dispersion(s.speed[sample])
        row["speed_cv"] = cv
        row["speed_rcv"] = rcv
        row.update(_magnitude_stats("accel", np.abs(s.accel[m_acc])))
        row.update(_magnitude_stats("jerk", np.abs(s.jerk[m_jerk])))

        if p.compute_burst_coast:
            row.update(self._burst_coast(s.accel, s.dframe, m_acc, m_jerk, p.fps))

        if p.subgroup_col is not None:
            row["n_frames"] = int(sample.sum())
            row["n_social_frames"] = int(social.sum())
            row["n_accel"] = int(m_acc.sum())
            row["n_jerk"] = int(m_jerk.sum())

        return row

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        p = self.params
        order_col = resolve_order_col(df)
        required = [C.id_col, order_col, p.speed_col]
        if p.subgroup_col is not None:
            required.append(p.subgroup_col)
        ensure_columns(df, required)
        df = df.sort_values([C.id_col, order_col]).reset_index(drop=True)

        has_gm = "group_membership" in df.columns
        has_gs = "group_size" in df.columns
        has_nn = "nn_id" in df.columns
        has_angle = "nn_delta_angle" in df.columns
        has_ego = "nn_delta_x_ego" in df.columns and "nn_delta_y_ego" in df.columns

        # These two are positional into the whole frame, and the per-fish `idx`
        # below indexes them. That is valid only because the reset_index above
        # runs first -- do not reorder these three statements.
        neighbor_speed = (
            self._neighbor_speed(df, order_col) if has_nn else np.full(len(df), np.nan)
        )
        if has_gm:
            group_mean = df.groupby([order_col, "group_membership"])[
                p.speed_col
            ].transform("mean")
            dev_group = df[p.speed_col].to_numpy(dtype=float) - group_mean.to_numpy(
                dtype=float
            )
        else:
            dev_group = np.full(len(df), np.nan)

        rows: list[dict[str, object]] = []
        for fid, g in df.groupby(C.id_col, sort=True):
            idx = g.index.to_numpy()
            speed = g[p.speed_col].to_numpy(dtype=float)
            frames = g[order_col].to_numpy(dtype=float)

            # --- social mask: frames with a valid nearest neighbor ---
            if has_nn:
                social = np.isfinite(g["nn_id"].to_numpy(dtype=float))
            else:
                social = np.zeros(len(g), dtype=bool)
            if has_gs:
                social = social & (
                    g["group_size"].to_numpy(dtype=float) >= p.social_min_group_size
                )

            accel = _derivative(speed, frames, p.fps)
            # jerk = d(accel)/dt over the intervals between accel samples
            jerk = (
                _derivative(accel, frames[1:], p.fps)
                if accel.size >= 2
                else np.empty(0, dtype=float)
            )
            series = _FishSeries(
                speed=speed,
                dframe=np.diff(frames),
                accel=accel,
                jerk=jerk,
                social=social,
                neighbor_speed=neighbor_speed[idx],
                dev_group=dev_group[idx],
                nn_angle=(
                    g["nn_delta_angle"].to_numpy(dtype=float) if has_angle else None
                ),
                ego_x=g["nn_delta_x_ego"].to_numpy(dtype=float) if has_ego else None,
                ego_y=g["nn_delta_y_ego"].to_numpy(dtype=float) if has_ego else None,
                has_nn=has_nn,
                has_gm=has_gm,
            )

            for value, sample in _subgroups(g, p.subgroup_col):
                row: dict[str, object] = {C.id_col: fid}
                if p.subgroup_col is not None:
                    row[p.subgroup_col] = value
                row.update(self._summarize(series, sample))
                rows.append(row)

        out = pd.DataFrame(rows)

        # Attach sequence / group metadata (constant within a sequence)
        for meta_col in (C.seq_col, C.group_col):
            if meta_col in df.columns and meta_col not in out.columns:
                out[meta_col] = df[meta_col].iloc[0]

        return out.reset_index(drop=True)

    @staticmethod
    def _burst_coast(
        accel: np.ndarray,
        dframe: np.ndarray,
        m_acc: np.ndarray,
        m_jerk: np.ndarray,
        fps: float,
    ) -> dict[str, float]:
        """Minimal burst-and-coast gait summary, over the masked intervals.

        ``kick_rate`` = number of acceleration peaks (a burst onset, where
        acceleration crosses from positive to non-positive) per second.
        ``burst_coast_ratio`` = fraction of intervals with positive acceleration.

        A peak is a transition *between* two adjacent intervals, so it lives on
        the jerk axis and takes the jerk mask. The rate's denominator is the
        elapsed time the retained intervals actually cover, not the span from
        first to last frame: under a split the latter would count the minutes
        spent outside the subgroup in the denominator of a rate measured inside
        it. With an all-True mask the sum telescopes back to that span.
        """
        selected = accel[m_acc]
        if selected.size == 0 or not np.any(np.isfinite(selected)):
            return {"kick_rate": float("nan"), "burst_coast_ratio": float("nan")}
        pos = accel > 0
        # peaks: a positive-accel sample immediately followed by non-positive
        peaks = int(np.sum(pos[:-1] & ~pos[1:] & m_jerk))
        duration_s = float(np.sum(dframe[m_acc]) / fps)
        kick_rate = _safe_ratio(float(peaks), duration_s)
        burst_coast_ratio = float(np.mean(pos[m_acc]))
        return {"kick_rate": kick_rate, "burst_coast_ratio": burst_coast_ratio}
