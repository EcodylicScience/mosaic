"""
ApproachAvoidance feature.

Detects approach-avoidance (AA) events for all C(n,2) unordered pairs per sequence.

Default decision logic follows trajognize AA:
  - role-specific speed thresholds (approacher vs avoider)
  - distance threshold
  - cosine thresholds between velocity and pair direction
  - approacher forward-motion gate vs body orientation
  - minimum event continuity (min_event_count of min_event_length frames)

Optional sliding-window averaging can be enabled, but it is OFF by default
to preserve trajognize-style framewise behavior.

Output columns (per frame × pair):
  - frame, id1, id2 (canonical order: id1 < id2)
  - label_id: primary non-directional AA label for visualization compatibility
  - aa_event: 1 if either direction is active
  - aa_event_12: 1 if id1 approaches and id2 avoids
  - aa_event_21: 1 if id2 approaches and id1 avoids
  - sequence, group (metadata pass-through)
"""

from __future__ import annotations
from itertools import combinations
from math import cos, radians
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from mosaic.core.dataset import register_feature
from .helpers import _merge_params


def _contiguous_runs(
    binary: np.ndarray, frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find start/end frame numbers for contiguous runs of 1s."""
    if len(binary) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    diff = np.diff(binary, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return frames[starts], frames[ends]


@register_feature
class ApproachAvoidance:
    """
    'approach-avoidance' — per-sequence AA event detection for all pairs.

    For N animals per sequence, evaluates all N*(N-1)/2 unique unordered pairs.
    The output stores directional events as aa_event_12 and aa_event_21 over
    canonical (id1,id2), plus aa_event/label_id as non-directional union.
    """

    name = "approach-avoidance"
    version = "0.1"
    parallelizable = True
    output_type = "per_frame"

    _defaults = dict(
        # Column conventions
        x_col="X",
        y_col="Y",
        orientation_col="ANGLE",
        id_col="id",
        seq_col="sequence",
        group_col="group",
        order_pref=("frame", "time"),

        # Sampling
        fps_default=30.0,
        velocity_units="per_frame",  # "per_frame" (trajognize-like) | "per_second"
        angle_units="radians",       # "radians" | "degrees" | "auto"
        consecutive_frame_delta=1.0,  # expected delta for consecutive samples

        # Cleaning / interpolation (per-animal)
        linear_interp_limit=10,
        edge_fill_limit=3,
        max_missing_fraction=0.10,

        # trajognize-style AA thresholds (position units/frame for velocity,
        # position units for distance). Example rat config used 200px / 5px/frame.
        distance_threshold=200.0,
        approacher_velocity_threshold=5.0,
        avoider_velocity_threshold=5.0,
        cos_approacher_threshold=0.8,
        cos_avoider_threshold=0.5,

        # Continuity gate (trajognize min-count-in-window)
        min_event_length=10,
        min_event_count=5,

        # Approacher body orientation gate (trajognize version >=7 behavior)
        use_approacher_orientation_gate=True,
        approacher_forward_cos_threshold=0.8660254037844386,  # cos(30 deg)

        # Optional smoothing extension (disabled by default)
        use_sliding_window=False,
        window_sec=0.4,
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = _merge_params(params, self._defaults)
        self._ds = None
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True
        self.skip_existing_outputs = False

    # ----------------------- Dataset hooks -----------------------

    def bind_dataset(self, ds):
        self._ds = ds

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        self._scope_filter = scope or {}

    # ----------------------- Fit protocol ------------------------

    def needs_fit(self) -> bool:
        return False

    def supports_partial_fit(self) -> bool:
        return False

    def fit(self, X_iter) -> None:
        return

    def partial_fit(self, df: pd.DataFrame) -> None:
        return

    def finalize_fit(self) -> None:
        return

    def save_model(self, path) -> None:
        raise NotImplementedError("Stateless feature; no model to save.")

    def load_model(self, path) -> None:
        raise NotImplementedError("Stateless feature; no model to load.")

    # ----------------------- Core logic --------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        p = self.params
        order_col = self._order_col(df)
        use_ori_gate = bool(p.get("use_approacher_orientation_gate", True))

        need = [p["id_col"], p["seq_col"], order_col, p["x_col"], p["y_col"]]
        if p["group_col"] in df.columns:
            need.append(p["group_col"])
        if use_ori_gate:
            need.append(p["orientation_col"])
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"[approach-avoidance] Missing columns: {missing}")

        cols = need.copy()
        df_small = df[cols].copy()
        if order_col == "frame":
            df_small[order_col] = df_small[order_col].astype(int, errors="ignore")

        # Clean per-animal, per-sequence
        group_cols = [p["seq_col"], p["id_col"]]
        data_cols = [p["x_col"], p["y_col"]]
        if p["orientation_col"] in df_small.columns:
            data_cols.append(p["orientation_col"])

        def clean_animal(g):
            result = self._clean_one_animal(g, data_cols, order_col)
            if isinstance(g.name, tuple):
                for col, val in zip(group_cols, g.name):
                    result[col] = val
            else:
                result[group_cols[0]] = g.name
            return result

        df_small = (
            df_small
            .groupby(group_cols, group_keys=False)
            .apply(clean_animal, include_groups=False)
        )

        # Build all pairs for each sequence
        out_frames: List[pd.DataFrame] = []

        for seq, gseq in df_small.groupby(p["seq_col"]):
            ids = sorted(gseq[p["id_col"]].unique())
            if len(ids) < 2:
                continue
            for id_a, id_b in combinations(ids, 2):
                pair_df = self._compute_pair(gseq, id_a, id_b, order_col, df)
                if pair_df is not None and not pair_df.empty:
                    out_frames.append(pair_df)

        if not out_frames:
            return pd.DataFrame(columns=[
                "frame", "id1", "id2",
                "label_id", "aa_event", "aa_event_12", "aa_event_21",
            ])

        out = pd.concat(out_frames, ignore_index=True)
        out = out.sort_values(["id1", "id2", "frame"]).reset_index(drop=True)
        return out

    # ----------------------- Pair computation --------------------

    def _compute_pair(
        self,
        gseq: pd.DataFrame,
        id_a: int,
        id_b: int,
        order_col: str,
        orig_df: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        p = self.params

        use_ori_gate = bool(p.get("use_approacher_orientation_gate", True))
        cols = [order_col, p["x_col"], p["y_col"]]
        if use_ori_gate and p["orientation_col"] in gseq.columns:
            cols.append(p["orientation_col"])
        A = gseq[gseq[p["id_col"]] == id_a][cols].copy()
        B = gseq[gseq[p["id_col"]] == id_b][cols].copy()

        if A.empty or B.empty:
            return None

        A = A.sort_values(order_col).rename(columns={order_col: "frame"})
        B = B.sort_values(order_col).rename(columns={order_col: "frame"})

        j = A.merge(B, on="frame", suffixes=("_A", "_B"))
        if j.empty or len(j) < 2:
            return None

        # Get fps
        fps = float(p["fps_default"])
        if "fps" in orig_df.columns:
            try:
                c = orig_df["fps"].dropna().unique()
                if len(c) == 1:
                    fps = float(c[0])
            except Exception:
                pass

        # Extract positions
        x_a = j[f"{p['x_col']}_A"].to_numpy(dtype=np.float64)
        y_a = j[f"{p['y_col']}_A"].to_numpy(dtype=np.float64)
        x_b = j[f"{p['x_col']}_B"].to_numpy(dtype=np.float64)
        y_b = j[f"{p['y_col']}_B"].to_numpy(dtype=np.float64)
        frames = j["frame"].to_numpy().astype(int)
        n = len(frames)
        eps = 1e-12
        expected_delta = float(p.get("consecutive_frame_delta", 1.0))

        # Trajognize-style velocity: frame-to-frame difference (not centered gradient).
        v_ax = np.full(n, np.nan, dtype=np.float64)
        v_ay = np.full(n, np.nan, dtype=np.float64)
        v_bx = np.full(n, np.nan, dtype=np.float64)
        v_by = np.full(n, np.nan, dtype=np.float64)
        if n > 1:
            frame_delta = np.diff(frames.astype(np.float64))
            valid_step = np.isclose(frame_delta, expected_delta)
            idx = np.where(valid_step)[0] + 1
            v_ax[idx] = x_a[idx] - x_a[idx - 1]
            v_ay[idx] = y_a[idx] - y_a[idx - 1]
            v_bx[idx] = x_b[idx] - x_b[idx - 1]
            v_by[idx] = y_b[idx] - y_b[idx - 1]

        if str(p.get("velocity_units", "per_frame")).lower() == "per_second":
            v_ax *= fps
            v_ay *= fps
            v_bx *= fps
            v_by *= fps

        speed_a = np.sqrt(v_ax * v_ax + v_ay * v_ay)
        speed_b = np.sqrt(v_bx * v_bx + v_by * v_by)

        # Pair direction d_12 points from id1 to id2 after canonicalization.
        # We'll canonicalize identities first, then remap signals accordingly.
        dx_ab = x_b - x_a
        dy_ab = y_b - y_a
        dist = np.sqrt(dx_ab * dx_ab + dy_ab * dy_ab)

        # Canonical pair order: id1 = min, id2 = max
        id1 = min(id_a, id_b)
        id2 = max(id_a, id_b)
        a_is_id1 = (id_a == id1)

        if a_is_id1:
            dx12 = dx_ab
            dy12 = dy_ab
            v1x, v1y = v_ax, v_ay
            v2x, v2y = v_bx, v_by
            speed1, speed2 = speed_a, speed_b
            ori1 = j.get(f"{p['orientation_col']}_A")
            ori2 = j.get(f"{p['orientation_col']}_B")
        else:
            dx12 = -dx_ab
            dy12 = -dy_ab
            v1x, v1y = v_bx, v_by
            v2x, v2y = v_ax, v_ay
            speed1, speed2 = speed_b, speed_a
            ori1 = j.get(f"{p['orientation_col']}_B")
            ori2 = j.get(f"{p['orientation_col']}_A")

        # Cosine metrics equivalent to trajognize:
        # direction 12: id1 approaches id2, id2 avoids id1
        cos_app_12 = (v1x * dx12 + v1y * dy12) / (np.maximum(dist * speed1, eps))
        cos_avoid_12 = (v2x * dx12 + v2y * dy12) / (np.maximum(dist * speed2, eps))
        # direction 21: id2 approaches id1, id1 avoids id2 (flip direction vector)
        cos_app_21 = (v2x * (-dx12) + v2y * (-dy12)) / (np.maximum(dist * speed2, eps))
        cos_avoid_21 = (v1x * (-dx12) + v1y * (-dy12)) / (np.maximum(dist * speed1, eps))

        # Optional approacher orientation gate (trajognize v7 behavior)
        use_ori_gate = bool(p.get("use_approacher_orientation_gate", True))
        forward_thr = float(p.get("approacher_forward_cos_threshold", cos(radians(30.0))))
        cos_ori_vel_1 = np.ones(n, dtype=np.float64)
        cos_ori_vel_2 = np.ones(n, dtype=np.float64)
        if use_ori_gate:
            if ori1 is None or ori2 is None:
                raise ValueError(
                    "[approach-avoidance] Orientation gate enabled but orientation column "
                    f"'{p['orientation_col']}' not available for merged pair."
                )
            th1 = ori1.to_numpy(dtype=np.float64)
            th2 = ori2.to_numpy(dtype=np.float64)
            unit_mode = str(p.get("angle_units", "radians")).lower()
            if unit_mode == "degrees":
                th1 = np.deg2rad(th1)
                th2 = np.deg2rad(th2)
            elif unit_mode == "auto":
                if np.nanmax(np.abs(th1)) > 2 * np.pi or np.nanmax(np.abs(th2)) > 2 * np.pi:
                    th1 = np.deg2rad(th1)
                    th2 = np.deg2rad(th2)
            o1x, o1y = np.cos(th1), np.sin(th1)
            o2x, o2y = np.cos(th2), np.sin(th2)
            cos_ori_vel_1 = (v1x * o1x + v1y * o1y) / np.maximum(speed1, eps)
            cos_ori_vel_2 = (v2x * o2x + v2y * o2y) / np.maximum(speed2, eps)

        # Optional smoothing extension (disabled by default).
        use_sliding = bool(p.get("use_sliding_window", False))
        if use_sliding:
            win_frames = max(1, int(round(float(p["window_sec"]) * fps)))
            dist_eval = self._sliding_nanmean(dist, win_frames)
            speed1_eval = self._sliding_nanmean(speed1, win_frames)
            speed2_eval = self._sliding_nanmean(speed2, win_frames)
            cos_app_12_eval = self._sliding_nanmean(cos_app_12, win_frames)
            cos_avoid_12_eval = self._sliding_nanmean(cos_avoid_12, win_frames)
            cos_app_21_eval = self._sliding_nanmean(cos_app_21, win_frames)
            cos_avoid_21_eval = self._sliding_nanmean(cos_avoid_21, win_frames)
            cos_ori_vel_1_eval = self._sliding_nanmean(cos_ori_vel_1, win_frames)
            cos_ori_vel_2_eval = self._sliding_nanmean(cos_ori_vel_2, win_frames)
        else:
            dist_eval = dist
            speed1_eval = speed1
            speed2_eval = speed2
            cos_app_12_eval = cos_app_12
            cos_avoid_12_eval = cos_avoid_12
            cos_app_21_eval = cos_app_21
            cos_avoid_21_eval = cos_avoid_21
            cos_ori_vel_1_eval = cos_ori_vel_1
            cos_ori_vel_2_eval = cos_ori_vel_2

        # Thresholds
        dist_thr = float(p["distance_threshold"])
        va_thr = float(p["approacher_velocity_threshold"])
        vb_thr = float(p["avoider_velocity_threshold"])
        cos_app_thr = float(p["cos_approacher_threshold"])
        cos_avoid_thr = float(p["cos_avoider_threshold"])

        base_valid = np.isfinite(speed1) & np.isfinite(speed2)

        common_ok = (
            base_valid
            & np.isfinite(dist_eval)
            & np.isfinite(speed1_eval)
            & np.isfinite(speed2_eval)
            & np.isfinite(cos_app_12_eval)
            & np.isfinite(cos_avoid_12_eval)
            & np.isfinite(cos_app_21_eval)
            & np.isfinite(cos_avoid_21_eval)
            & (dist_eval > 0.0)
            & (dist_eval <= dist_thr)
        )

        cand_12 = (
            common_ok
            & (speed1_eval >= va_thr)
            & (speed2_eval >= vb_thr)
            & (cos_app_12_eval >= cos_app_thr)
            & (cos_avoid_12_eval >= cos_avoid_thr)
            & (cos_ori_vel_1_eval >= forward_thr)
        )

        cand_21 = (
            common_ok
            & (speed2_eval >= va_thr)
            & (speed1_eval >= vb_thr)
            & (cos_app_21_eval >= cos_app_thr)
            & (cos_avoid_21_eval >= cos_avoid_thr)
            & (cos_ori_vel_2_eval >= forward_thr)
        )

        min_event_length = max(1, int(p.get("min_event_length", 1)))
        min_event_count = max(1, int(p.get("min_event_count", 1)))
        if min_event_count > min_event_length:
            min_event_count = min_event_length

        aa_event_12 = self._min_count_window_gate(
            cand_12, frames, min_event_length, min_event_count
        ).astype(np.int8)
        aa_event_21 = self._min_count_window_gate(
            cand_21, frames, min_event_length, min_event_count
        ).astype(np.int8)

        aa_event = np.maximum(aa_event_12, aa_event_21)

        out = pd.DataFrame(
            {
                "frame": frames,
                "id1": id1,
                "id2": id2,
                # Keep this as the preferred label column for viz/overlay loaders.
                "label_id": aa_event.astype(np.int8, copy=False),
                "aa_event": aa_event,
                "aa_event_12": aa_event_12,
                "aa_event_21": aa_event_21,
            }
        )

        for col in (p["seq_col"], p["group_col"]):
            if col in gseq.columns:
                out[col] = gseq[col].iloc[0]
            elif col in orig_df.columns:
                out[col] = orig_df[col].iloc[0]

        return out

    # ----------------------- Event extraction ---------------------

    @staticmethod
    def extract_events(
        aa_df: pd.DataFrame,
        min_duration: int = 1,
    ) -> pd.DataFrame:
        """Convert per-frame AA output into a compact event table.

        Parameters
        ----------
        aa_df : DataFrame
            Per-frame output with columns: frame, id1, id2,
            aa_event, aa_event_12, aa_event_21.  May span multiple
            sequences/groups (they are handled independently).
        min_duration : int
            Minimum event length in frames.  Events shorter than this
            are discarded.

        Returns
        -------
        DataFrame with columns:
            id1, id2, start_frame, end_frame, duration,
            direction ('12' if id1→id2, '21' if id2→id1, 'both'),
            approacher_id, avoider_id,
            sequence (if present), group (if present).
        """
        group_cols = ["id1", "id2"]
        for c in ("sequence", "group"):
            if c in aa_df.columns:
                group_cols.append(c)

        events: List[dict] = []
        for keys, g in aa_df.groupby(group_cols, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            meta = dict(zip(group_cols, keys))
            id1, id2 = int(meta["id1"]), int(meta["id2"])
            g = g.sort_values("frame").reset_index(drop=True)
            frames = g["frame"].to_numpy(dtype=int)

            for col, direction in [
                ("aa_event_12", "12"),
                ("aa_event_21", "21"),
            ]:
                if col not in g.columns:
                    continue
                active = g[col].to_numpy(dtype=int)
                starts, ends = _contiguous_runs(active, frames)
                for s, e in zip(starts, ends):
                    dur = e - s + 1
                    if dur < min_duration:
                        continue
                    if direction == "12":
                        appr, avoid = id1, id2
                    else:
                        appr, avoid = id2, id1
                    row = {
                        **meta,
                        "start_frame": int(s),
                        "end_frame": int(e),
                        "duration": int(dur),
                        "direction": direction,
                        "approacher_id": appr,
                        "avoider_id": avoid,
                    }
                    events.append(row)

        if not events:
            cols = group_cols + [
                "start_frame", "end_frame", "duration",
                "direction", "approacher_id", "avoider_id",
            ]
            return pd.DataFrame(columns=cols)

        out = pd.DataFrame(events)
        out = out.sort_values(
            group_cols + ["start_frame"]
        ).reset_index(drop=True)
        return out

    # ----------------------- Helpers -----------------------------

    def _order_col(self, df: pd.DataFrame) -> str:
        for c in self.params["order_pref"]:
            if c in df.columns:
                return c
        raise ValueError("Need either 'frame' or 'time' column to order rows.")

    def _clean_one_animal(
        self, g: pd.DataFrame, data_cols: List[str], order_col: str,
    ) -> pd.DataFrame:
        p = self.params
        g = g.sort_values(order_col).copy()
        g = g.set_index(order_col)

        g[data_cols] = g[data_cols].replace([np.inf, -np.inf], np.nan)
        g[data_cols] = g[data_cols].interpolate(
            method="linear",
            limit=int(p["linear_interp_limit"]),
            limit_direction="both",
        )
        g[data_cols] = g[data_cols].ffill(limit=int(p["edge_fill_limit"]))
        g[data_cols] = g[data_cols].bfill(limit=int(p["edge_fill_limit"]))

        miss_frac = g[data_cols].isna().mean(axis=1)
        g = g.loc[miss_frac <= float(p["max_missing_fraction"])].copy()

        if g[data_cols].isna().any().any():
            med = g[data_cols].median()
            g[data_cols] = g[data_cols].fillna(med)

        g = g.reset_index()
        return g

    @staticmethod
    def _sliding_mean(x: np.ndarray, win: int) -> np.ndarray:
        if win <= 1:
            return x
        # For even windows, asymmetric padding preserves output length == input length.
        left = win // 2
        right = win - 1 - left
        xp = np.pad(x, pad_width=(left, right), mode="reflect")
        ker = np.ones(win, dtype=np.float64) / float(win)
        return np.convolve(xp, ker, mode="valid")

    @staticmethod
    def _sliding_nanmean(x: np.ndarray, win: int) -> np.ndarray:
        if win <= 1:
            return x
        finite = np.isfinite(x)
        x0 = np.where(finite, x, 0.0)
        w = finite.astype(np.float64)
        num = ApproachAvoidance._sliding_mean(x0, win)
        den = ApproachAvoidance._sliding_mean(w, win)
        out = np.full_like(num, np.nan, dtype=np.float64)
        good = den > 0
        out[good] = num[good] / den[good]
        return out

    @staticmethod
    def _min_count_window_gate(
        candidate: np.ndarray,
        frames: np.ndarray,
        window_len: int,
        min_count: int,
    ) -> np.ndarray:
        """Trajognize-style continuity gate: >=min_count hits in last window_len frames."""
        if len(candidate) == 0:
            return np.zeros(0, dtype=bool)
        if window_len <= 1 and min_count <= 1:
            return candidate.astype(bool)

        cand = candidate.astype(np.int32, copy=False)
        fr = frames.astype(np.int64, copy=False)
        csum = np.concatenate([[0], np.cumsum(cand, dtype=np.int64)])
        left = np.searchsorted(fr, fr - int(window_len) + 1, side="left")
        idx = np.arange(len(fr), dtype=np.int64)
        counts = csum[idx + 1] - csum[left]
        return candidate & (counts >= int(min_count))
