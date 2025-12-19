from pathlib import Path
from typing import Optional, Dict, Any, Iterable
import numpy as np
import pandas as pd

from behavior.dataset import register_feature
from behavior.helpers import chunk_sequence


def _merge_params(overrides: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out

def _wrap_angle(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def _mean_angle(angles: np.ndarray) -> float:
    return float(np.arctan2(np.nanmean(np.sin(angles)), np.nanmean(np.cos(angles))))


@register_feature
class FFGroupsMetrics:
    """
    Per-sequence summary of focal-fish group metrics.

    Per-frame computed (internal):
      - distance_from_centroid, xrot_to_centroid, yrot_to_centroid, dev_speed_to_mean
    Summaries (output: one row per id within sequence):
      - fractime_norm2
      - avg_duration_frame
      - med_duration_frame
      - ftime_periphery
      - ftime_periphery_norm
    """

    name = "ffgroups-metrics"
    version = "0.1"
    parallelizable = True

    _defaults = dict(
        id_col="id",
        seq_col="sequence",
        group_col="event",
        x_col="x",
        y_col="y",
        heading_col="ANGLE",
        speed_col="speed",
        order_pref=("frame", "time"),
        time_chunk_sec=None,
        frame_chunk=None,
        centroid_heading_col="centroid_heading",
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

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        return

    def partial_fit(self, df: pd.DataFrame) -> None:
        return

    def finalize_fit(self) -> None:
        return

    def save_model(self, path: Path) -> None:
        return

    def load_model(self, path: Path) -> None:
        return

    # ----------------------- Core logic --------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        p = self.params
        order_col = self._order_col(df)
        df = df.sort_values(order_col).reset_index(drop=True)

        required = [p["x_col"], p["y_col"], p["heading_col"], p["speed_col"], p["id_col"]]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for FFGroupsMetrics: {missing}")

        # Split into chunks (default: whole sequence)
        chunks = list(chunk_sequence(
            df,
            time_chunk_sec=p.get("time_chunk_sec"),
            frame_chunk=p.get("frame_chunk"),
        ))  
        summaries = []
        for chunk_id, chunk_df, meta in chunks:
            # Group per frame within group_col if present
            group_keys = []
            if p["group_col"] in chunk_df.columns:
                group_keys.append(p["group_col"])
            frame_key = "frame" if "frame" in chunk_df.columns else "time"
            if frame_key not in chunk_df.columns:
                raise ValueError("Need either 'frame' or 'time' column.")
            group_keys.append(frame_key)
            per_frame = self._compute_per_frame(chunk_df, p, group_keys)
            summary = self._compute_summary(per_frame, p)
            summary["chunk_id"] = chunk_id
            summary["chunk_start_frame"] = meta.get("start_frame")
            summary["chunk_end_frame"] = meta.get("end_frame")
            summary["chunk_start_time"] = meta.get("start_time")
            summary["chunk_end_time"] = meta.get("end_time")
            summaries.append(summary)

        if not summaries:
            return pd.DataFrame()
        return pd.concat(summaries, ignore_index=True)

    # ------------------ Internal helpers ------------------------
    def _order_col(self, df: pd.DataFrame) -> str:
        for c in self.params["order_pref"]:
            if c in df.columns:
                return c
        raise ValueError("Need either 'frame' or 'time' column to order rows.")

    def _compute_per_frame(self, df: pd.DataFrame, p: dict, group_keys: list) -> pd.DataFrame:
        # Vectorized per-group computations to avoid slow Python loops.
        grouped = df.groupby(group_keys, sort=False)
        agg_dict = {
            p["x_col"]: "mean",
            p["y_col"]: "mean",
            p["speed_col"]: "mean",
        }
        if p["centroid_heading_col"] in df.columns:
            agg_dict[p["centroid_heading_col"]] = [
                lambda a: np.nanmean(np.sin(a)),
                lambda a: np.nanmean(np.cos(a)),
            ]
        stats = grouped.agg(agg_dict).reset_index()

        # Normalize multiindex columns and rename
        flat_cols = []
        for col in stats.columns:
            if isinstance(col, tuple):
                base, func = col
                if base == p["centroid_heading_col"] and func == "<lambda_0>":
                    flat_cols.append("_sin_mean")
                elif base == p["centroid_heading_col"] and func == "<lambda_1>":
                    flat_cols.append("_cos_mean")
                else:
                    flat_cols.append(base)
            else:
                flat_cols.append(col)
        stats.columns = flat_cols
        stats = stats.rename(columns={
            p["x_col"]: "_cx",
            p["y_col"]: "_cy",
            p["speed_col"]: "_mean_speed",
        })

        df = df.merge(stats, on=group_keys, how="left")

        dx = df[p["x_col"]].to_numpy(dtype=float) - df["_cx"].to_numpy(dtype=float)
        dy = df[p["y_col"]].to_numpy(dtype=float) - df["_cy"].to_numpy(dtype=float)
        if p["centroid_heading_col"] in df.columns:
            chead = np.arctan2(df["_sin_mean"].to_numpy(dtype=float), df["_cos_mean"].to_numpy(dtype=float))
        else:
            chead = 0.0
        ct, st = np.cos(-chead), np.sin(-chead)
        df["distance_from_centroid"] = np.sqrt(dx * dx + dy * dy)
        df["xrot_to_centroid"] = dx * ct - dy * st
        df["yrot_to_centroid"] = dx * st + dy * ct
        df["dev_speed_to_mean"] = df[p["speed_col"]] - df["_mean_speed"]

        drop_cols = ["_cx", "_cy", "_mean_speed"]
        if p["centroid_heading_col"] in df.columns:
            drop_cols.extend(["_sin_mean", "_cos_mean"])
        return df.drop(columns=drop_cols).reset_index(drop=True)

    def _compute_summary(self, df: pd.DataFrame, p: dict) -> pd.DataFrame:
        id_col = p["id_col"]
        frame_key = "frame" if "frame" in df.columns else "time"

        # Total frames per fish
        total_frames = df.groupby(id_col)[frame_key].count()

        # fractime_norm2: fraction of frames in each group_size (computed as count of frames sharing group in frame)
        if "group_size" in df.columns:
            gsize = df["group_size"]
        else:
            # infer group_size per frame group
            df["group_size"] = df.groupby(frame_key)[id_col].transform("count")
            gsize = df["group_size"]

        frame_counts = df.groupby([id_col, "group_size"])[frame_key].count()
        fractime_norm2 = (frame_counts / total_frames.reindex(frame_counts.index.get_level_values(id_col)).to_numpy())
        fractime_norm2 = fractime_norm2.reset_index(name="fractime_norm2")

        # durations of contiguous group_size runs per fish
        dur_rows = []
        for fish, sub in df.sort_values(frame_key).groupby(id_col):
            gsz = sub["group_size"].to_numpy()
            frames = sub[frame_key].to_numpy()
            if len(gsz) == 0:
                continue
            # find change points
            change = np.where(np.diff(gsz) != 0)[0] + 1
            starts = np.r_[0, change]
            ends = np.r_[change, len(gsz)]
            for s, e in zip(starts, ends):
                dur = frames[e - 1] - frames[s] + 1
                dur_rows.append((fish, gsz[s], dur))
        durations = pd.DataFrame(dur_rows, columns=[id_col, "group_size", "duration"])
        if durations.empty:
            durations["avg_duration_frame"] = []
            durations["med_duration_frame"] = []
        agg_dur = durations.groupby([id_col, "group_size"])["duration"].agg(
            avg_duration_frame="mean", med_duration_frame="median"
        ).reset_index()

        # periphery time: rank by distance, count frames where farthest
        rank_groups = [frame_key]
        if p["group_col"] in df.columns:
            rank_groups.insert(0, p["group_col"])
        df["rank_centroid_distance"] = df.groupby(rank_groups)["distance_from_centroid"].rank(
            ascending=True, method="max"
        )
        farthest = df["rank_centroid_distance"] == df["group_size"]
        ftime_counts = df.loc[farthest].groupby([id_col, "group_size"])[frame_key].count()
        # Normalize by frames spent in that group_size; fill missing combos with 0
        ftime_periphery = (ftime_counts / frame_counts).reindex(frame_counts.index, fill_value=0).reset_index()
        ftime_periphery = ftime_periphery.rename(columns={frame_key: "ftime_periphery"})

        ftime_periphery_norm_counts = (
            df.loc[farthest, "group_size"]
            .groupby([df.loc[farthest, id_col], df.loc[farthest, "group_size"]])
            .sum()
        )
        ftime_periphery_norm = (
            (ftime_periphery_norm_counts / frame_counts)
            .reindex(frame_counts.index, fill_value=0)
            .reset_index(name="ftime_periphery_norm")
        )

        # Merge summaries
        out = fractime_norm2.merge(agg_dur, on=[id_col, "group_size"], how="left")

        out = out.merge(ftime_periphery, on=[id_col, "group_size"], how="left")
        out = out.merge(ftime_periphery_norm, on=[id_col, "group_size"], how="left")

        # Attach meta
        for meta_col in (p["seq_col"], p["group_col"]):
            if meta_col in df.columns and meta_col not in out.columns:
                out[meta_col] = df[meta_col].iloc[0]

        return out.reset_index(drop=True)
