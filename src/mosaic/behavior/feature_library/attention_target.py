"""
AttentionTarget -- per-frame attention target and group-membership label.

Consumes PairFacing output. For each (focal_id, frame), picks the facing
target with the smallest ``angle_diff_deg`` (NaN if no target satisfies the
facing thresholds), then optionally labels the focal-target relationship
against a generic ``id -> group`` mapping.

This generalizes Valerie's BeesInADish in_pair/out_pair attention metric:
for the petri-dish experiment the group map encodes scent role
(scented_1, scented_2 -> "scented"; unscented_1, unscented_2 -> "unscented"),
so attention_type "in_group" corresponds to her "in_pair" and "out_group" to
her "out_pair". The feature is experiment-agnostic -- any id->group mapping
works.
"""

from __future__ import annotations

from pathlib import Path
from typing import final

import pandas as pd

from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    Inputs,
    InputStream,
    Params,
    Result,
)

from .helpers import ensure_columns
from .registry import register_feature


def _id_key(value) -> str | None:
    """Normalize an id to the canonical str key used in id_group_map lookups.

    Floats that are whole numbers (e.g. 2.0 from a NaN-upcast merge column)
    map to "2", not "2.0".
    """
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


@final
@register_feature
class AttentionTarget:
    """
    Per-frame attention target with group-membership labeling.

    Output columns (one row per (focal_id, frame)):
      - frame
      - focal_id
      - attention_target_id              (NA if no facing target)
      - attention_target_angle_diff_deg  (NA if no facing target)
      - focal_group                      (NA if id_group_map is None or id absent)
      - target_group                     (NA if no facing target / id absent)
      - attention_type ∈ {"in_group", "out_group", "none", "unknown"}
            "none"    -- no target met the facing thresholds
            "unknown" -- a target was picked but at least one group label is
                         missing from id_group_map
            otherwise compares focal_group vs target_group

    Params
    ------
    id_group_map : dict[str, str] | None
        Mapping from individual id (as string) to group label. Default None,
        in which case focal_group / target_group are NA and attention_type is
        "none" (no facing target) or "unknown" (target picked).
    """

    name = "attention-target"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[Result]):
        pass

    class Params(Params):
        id_group_map: dict[str, str] | None = None

    def __init__(
        self,
        inputs: AttentionTarget.Inputs = Inputs((Result(feature="pair-facing"),)),
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
            return self._empty_output()

        ensure_columns(
            df,
            [C.frame_col, "focal_id", "target_id", "angle_diff_deg", "is_facing"],
        )

        # Restrict to facing candidates; for each (focal, frame) pick the one
        # with smallest angle_diff_deg.
        facing = df[df["is_facing"].astype(bool)]
        if not facing.empty:
            idx = facing.groupby(["focal_id", C.frame_col])["angle_diff_deg"].idxmin()
            picked = facing.loc[
                idx, ["focal_id", C.frame_col, "target_id", "angle_diff_deg"]
            ].rename(columns={
                "target_id": "attention_target_id",
                "angle_diff_deg": "attention_target_angle_diff_deg",
            })
        else:
            picked = pd.DataFrame(
                columns=[
                    "focal_id", C.frame_col,
                    "attention_target_id", "attention_target_angle_diff_deg",
                ]
            )

        # Outer-join onto the full (focal_id, frame) grid so frames without a
        # facing target appear with NA target.
        full_index = (
            df[["focal_id", C.frame_col]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        out = full_index.merge(picked, on=["focal_id", C.frame_col], how="left")

        # Group labels. Normalize ids to canonical string keys -- the left-join
        # may upcast attention_target_id to float (NaN-bearing), so str(2.0)
        # would otherwise fail to match a dict keyed on "2".
        group_map = self.params.id_group_map
        if group_map:
            out["focal_group"] = out["focal_id"].map(
                lambda i: group_map.get(_id_key(i))
            )
            out["target_group"] = out["attention_target_id"].map(
                lambda i: group_map.get(_id_key(i))
            )
        else:
            out["focal_group"] = pd.NA
            out["target_group"] = pd.NA

        def _attention_type(row):
            if pd.isna(row["attention_target_id"]):
                return "none"
            fg = row["focal_group"]
            tg = row["target_group"]
            if pd.isna(fg) or pd.isna(tg):
                return "unknown"
            return "in_group" if fg == tg else "out_group"

        out["attention_type"] = out.apply(_attention_type, axis=1)

        for col in (C.group_col, C.seq_col):
            if col in df.columns:
                out[col] = df[col].iloc[0]

        cols = [
            C.frame_col, "focal_id",
            "attention_target_id", "attention_target_angle_diff_deg",
            "focal_group", "target_group", "attention_type",
        ]
        extra = [c for c in out.columns if c not in cols]
        return (
            out[cols + extra]
            .sort_values(["focal_id", C.frame_col])
            .reset_index(drop=True)
        )

    def _empty_output(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                C.frame_col, "focal_id",
                "attention_target_id", "attention_target_angle_diff_deg",
                "focal_group", "target_group", "attention_type",
            ]
        )
