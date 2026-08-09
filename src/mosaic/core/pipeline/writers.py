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


# --- Parquet reading ---


def read_parquet_table(path: Path) -> pd.DataFrame:
    """Read a whole parquet as a frame.

    A one-line delegation, and the reason is the type checker rather than the
    behavior: ``pandas.read_parquet`` carries an unknown return through its
    ``**kwargs`` overload, so every direct call leaves its result partially
    unknown and the unknown spreads into whatever reads the frame. Confining it
    to one annotated function keeps that at one site rather than at each caller.
    """
    return pd.read_parquet(path)


# --- Parquet writing ---


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> int:
    """Write *df* to *path* as parquet, atomically. Returns the row count.

    **The only sanctioned way to write a parquet anywhere in the toolkit**, and the
    reason it exists as one function rather than an idiom: this used to be
    ``df.to_parquet(final_path)`` at eleven independent sites, so a kill mid-write
    left a half-written file exactly where a whole one belongs. A torn table is
    worse than an absent one, because every reuse gate in the tracking layer tests
    for *presence*.

    ``atomic_write`` writes a temp file in the destination directory and renames it
    over the target, so the addressed path only ever holds a complete file, and
    creates the parent directory itself -- an adjacent ``mkdir`` beside a call here
    is redundant.

    No constraint on where a caller puts it. This used to owe ``index_lock`` a
    rule -- never inside a locked block except as its last act, because the
    rename replaced the inode the lock was held on -- and the rule is gone:
    ``index_lock`` holds a sidecar that no rename touches.
    """
    n_rows = len(df)
    atomic_write(path, lambda p: df.to_parquet(p, index=False))
    return n_rows


def write_output(
    meta: FeatureMeta,
    df_feat: FeatureOutput,
) -> int:
    """Write feature output to parquet atomically. Returns n_rows written."""
    df = pd.DataFrame() if df_feat is None else df_feat
    return write_parquet_atomic(df, meta.out_path)


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
