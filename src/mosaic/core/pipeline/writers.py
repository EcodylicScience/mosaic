from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ._utils import FeatureMeta

FeatureOutput = pd.DataFrame | None


# --- Trimming ---


def trim_feature_output(
    df_feat: FeatureOutput,
    core_start: int,
    core_end: int,
) -> FeatureOutput:
    """Trim feature output to original segment bounds (removing overlap regions)."""
    if df_feat is None:
        return df_feat

    if core_start == 0 and core_end >= len(df_feat):
        return df_feat
    return df_feat.iloc[core_start:core_end].reset_index(drop=True)


# --- Parquet writing ---


def write_output(
    meta: FeatureMeta,
    df_feat: FeatureOutput,
) -> int:
    """Write feature output to parquet. Returns n_rows written."""
    out_path = meta.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df_feat is None:
        df_feat = pd.DataFrame()

    n_rows = len(df_feat)
    df_feat.to_parquet(out_path, index=False)
    return n_rows


# --- Output validation (cache-hit checks) ---


def output_n_rows(out_path: Path) -> int:
    """Footer-only row count for an existing output parquet (fast)."""
    return int(pq.read_metadata(out_path).num_rows)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]


def default_check_output(meta: FeatureMeta, run_root: Path) -> bool:
    """Default deep validator: the output parquet is fully readable.

    Materializes all column chunks (unlike the footer-only fast path), so
    truncated/corrupt data pages are detected. ``run_root`` is accepted for
    signature parity with per-feature ``check_output`` overrides but unused
    here. Returns False on any read error.
    """
    try:
        pq.read_table(meta.out_path)
    except Exception:
        return False
    return True
