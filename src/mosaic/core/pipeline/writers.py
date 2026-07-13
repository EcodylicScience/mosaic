from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ._utils import FeatureMeta, atomic_write

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
    """Write feature output to parquet atomically. Returns n_rows written.

    The parquet is written to a temp file and renamed onto ``meta.out_path``, so
    a concurrent/interrupted write never leaves a torn file at the final path.
    """
    df = pd.DataFrame() if df_feat is None else df_feat
    n_rows = len(df)
    atomic_write(meta.out_path, lambda p: df.to_parquet(p, index=False))
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
