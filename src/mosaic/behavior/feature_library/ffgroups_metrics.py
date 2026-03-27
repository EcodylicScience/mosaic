from __future__ import annotations

from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.helpers import chunk_sequence
from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    Inputs,
    InputStream,
    Params,
    TrackInput,
    resolve_order_col,
)

from .helpers import apply_exclude_cols, ensure_columns
from .registry import register_feature


@final
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

    Params:
        group_col: Column name that identifies group events (e.g. from
            FFGroups output). Default: "event".
        speed_col: Column name for speed values. Default: "speed".
        time_chunk_sec: If set, split each sequence into time-based
            chunks of this duration (seconds) and compute summaries per
            chunk. Default: None (whole sequence).
        frame_chunk: If set, split each sequence into frame-based chunks
            of this size and compute summaries per chunk. Default: None.
        centroid_heading_col: Column for centroid heading used in rotation
            calculations. Default: "centroid_heading".
        exclude_cols: List of boolean column names (e.g. "bad_frame")
            whose truthy rows are dropped before computation.
            Default: [].
    """

    name = "ffgroups-metrics"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        group_col: str = "event"
        speed_col: str = "speed"
        time_chunk_sec: float | None = None
        frame_chunk: int | None = None
        centroid_heading_col: str = "centroid_heading"
        exclude_cols: list[str] = Field(default_factory=list)

    def __init__(
        self,
        inputs: FFGroupsMetrics.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

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

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        order_col = resolve_order_col(df)
        df = df.sort_values(order_col).reset_index(drop=True)

        # Drop rows where any exclude_col is truthy (e.g. bad_frame)
        df = apply_exclude_cols(df, self.params.exclude_cols)
        if df.empty:
            return pd.DataFrame()

        ensure_columns(df, [C.x_col, C.y_col, self.params.speed_col, C.id_col])

        summaries: list[pd.DataFrame] = []
        for chunk_id, chunk_df, meta in chunk_sequence(
            df,
            time_chunk_sec=self.params.time_chunk_sec,
            frame_chunk=self.params.frame_chunk,
        ):
            group_keys: list[str] = []
            if self.params.group_col in chunk_df.columns:
                group_keys.append(self.params.group_col)
            frame_key = resolve_order_col(chunk_df)
            group_keys.append(frame_key)

            per_frame = self._compute_per_frame(chunk_df, group_keys)
            summary = self._compute_summary(per_frame)
            summary["chunk_id"] = chunk_id
            summary["chunk_start_frame"] = meta.get("start_frame")
            summary["chunk_end_frame"] = meta.get("end_frame")
            summary["chunk_start_time"] = meta.get("start_time")
            summary["chunk_end_time"] = meta.get("end_time")
            summaries.append(summary)

        if not summaries:
            return pd.DataFrame()
        return pd.concat(summaries, ignore_index=True)

    def _compute_per_frame(
        self, df: pd.DataFrame, group_keys: list[str]
    ) -> pd.DataFrame:
        p = self.params
        grouped = df.groupby(group_keys, sort=False)

        stats = (
            grouped[[C.x_col, C.y_col, p.speed_col]]
            .mean()
            .reset_index()
            .rename(
                columns={C.x_col: "_cx", C.y_col: "_cy", p.speed_col: "_mean_speed"}
            )
        )

        if p.centroid_heading_col in df.columns:
            heading = grouped[p.centroid_heading_col]
            stats["_sin_mean"] = heading.agg(
                lambda v: float(np.nanmean(np.sin(v)))
            ).values
            stats["_cos_mean"] = heading.agg(
                lambda v: float(np.nanmean(np.cos(v)))
            ).values

        df = df.merge(stats, on=group_keys, how="left")

        dx = df[C.x_col].to_numpy(dtype=float) - df["_cx"].to_numpy(dtype=float)
        dy = df[C.y_col].to_numpy(dtype=float) - df["_cy"].to_numpy(dtype=float)
        if p.centroid_heading_col in df.columns:
            chead = np.arctan2(
                df["_sin_mean"].to_numpy(dtype=float),
                df["_cos_mean"].to_numpy(dtype=float),
            )
        else:
            chead = 0.0
        ct, st = np.cos(-chead), np.sin(-chead)
        df["distance_from_centroid"] = np.sqrt(dx * dx + dy * dy)
        df["xrot_to_centroid"] = dx * ct - dy * st
        df["yrot_to_centroid"] = dx * st + dy * ct
        df["dev_speed_to_mean"] = df[p.speed_col] - df["_mean_speed"]

        drop_cols = ["_cx", "_cy", "_mean_speed"]
        if p.centroid_heading_col in df.columns:
            drop_cols.extend(["_sin_mean", "_cos_mean"])
        return df.drop(columns=drop_cols).reset_index(drop=True)

    def _compute_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        frame_key = resolve_order_col(df)

        # Total frames per fish
        total_frames = df.groupby(C.id_col)[frame_key].count()

        # fractime_norm2: fraction of frames in each group_size (computed as count of frames sharing group in frame)
        if "group_size" not in df.columns:
            # infer group_size per frame group
            df["group_size"] = df.groupby(frame_key)[C.id_col].transform("count")

        frame_counts = df.groupby([C.id_col, "group_size"])[frame_key].count()
        fractime_norm2 = (
            frame_counts
            / total_frames.reindex(
                frame_counts.index.get_level_values(C.id_col)
            ).to_numpy()
        )
        fractime_norm2 = fractime_norm2.reset_index(name="fractime_norm2")

        # durations of contiguous group_size runs per fish
        sorted_df = df.sort_values([C.id_col, frame_key])
        ids = sorted_df[C.id_col].to_numpy()
        group_sizes = sorted_df["group_size"].to_numpy()
        frames = sorted_df[frame_key].to_numpy()

        # A new run starts where either id or group_size changes
        new_run = np.concatenate(
            [[True], (ids[1:] != ids[:-1]) | (group_sizes[1:] != group_sizes[:-1])]
        )
        starts = np.flatnonzero(new_run)
        ends = np.append(starts[1:], len(ids))

        durations = pd.DataFrame(
            {
                C.id_col: ids[starts],
                "group_size": group_sizes[starts],
                "duration": frames[ends - 1] - frames[starts] + 1,
            }
        )
        agg_dur = (
            durations.groupby([C.id_col, "group_size"])["duration"]
            .agg(avg_duration_frame="mean", med_duration_frame="median")
            .reset_index()
        )

        # periphery time: rank by distance, count frames where farthest
        rank_groups = [frame_key]
        if p.group_col in df.columns:
            rank_groups.insert(0, p.group_col)
        df["rank_centroid_distance"] = df.groupby(rank_groups)[
            "distance_from_centroid"
        ].rank(ascending=True, method="max")
        farthest_df = df.loc[df["rank_centroid_distance"] == df["group_size"]]
        ftime_counts = farthest_df.groupby([C.id_col, "group_size"])[frame_key].count()
        # Normalize by frames spent in that group_size; fill missing combos with 0
        ftime_periphery = (
            (ftime_counts / frame_counts)
            .reindex(frame_counts.index, fill_value=0)
            .reset_index()
            .rename(columns={frame_key: "ftime_periphery"})
        )
        ftime_periphery_norm_counts = farthest_df.groupby([C.id_col, "group_size"])[
            "group_size"
        ].sum()
        ftime_periphery_norm = (
            (ftime_periphery_norm_counts / frame_counts)
            .reindex(frame_counts.index, fill_value=0)
            .reset_index(name="ftime_periphery_norm")
        )

        # Merge summaries
        out = fractime_norm2.merge(agg_dur, on=[C.id_col, "group_size"], how="left")
        out = out.merge(ftime_periphery, on=[C.id_col, "group_size"], how="left")
        out = out.merge(ftime_periphery_norm, on=[C.id_col, "group_size"], how="left")

        # Attach meta
        for meta_col in (C.seq_col, p.group_col):
            if meta_col in df.columns and meta_col not in out.columns:
                out[meta_col] = df[meta_col].iloc[0]

        return out.reset_index(drop=True)
