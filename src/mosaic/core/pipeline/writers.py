from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow.parquet as pq

from ._utils import FeatureMeta, atomic_write

if TYPE_CHECKING:
    from .manifest import CoreSelector

FeatureOutput = pd.DataFrame | None


# --- Trimming ---


def trim_feature_output(
    df_feat: FeatureOutput,
    selector: CoreSelector,
) -> FeatureOutput:
    """Keep the rows of an overlapped output that belong to the entry.

    Selects on the frame interval the entry covers, rather than slicing by the
    row offsets the input happened to have. Those offsets assumed ``apply``
    returned one row per input row in input order -- a contract the ``Feature``
    protocol has never stated and roughly half the library breaks, by sorting, by
    filtering, or by reducing to a row per frame. The old positional slice
    returned the right *number* of rows and the wrong ones, silently; and its
    ``core_start == 0`` fast path skipped the trim entirely for the first
    sequence of every group, writing the next segment's rows into it.

    Two refusals rather than a plausible answer:

    - **No order column.** A per-sequence summary carries no frame, so there is
      nothing to select on and no honest fallback -- a positional guess is the
      thing this replaced.
    - **A non-empty output, none of which is inside the interval.** The feature
      returned rows addressed to frames the entry does not cover, which means it
      rewrote or re-based the frame axis. Writing zero rows would look like an
      entry that legitimately produced nothing.

    ``None`` and an empty frame pass through: both mean the feature produced
    nothing, which is a real result and distinguishable from the case above.
    """
    if df_feat is None or df_feat.empty:
        return df_feat

    if selector.order_col not in df_feat.columns:
        msg = (
            f"overlap trimming needs the {selector.order_col!r} column to tell "
            f"this entry's rows from its neighbours', and "
            f"{selector.entry_key or 'this feature'}'s output does not carry it. "
            f"A feature whose output has no frame axis -- a per-sequence summary "
            f"-- cannot be run with overlap_frames > 0."
        )
        raise ValueError(msg)

    keep = selector.mask(df_feat)
    if not bool(keep.any()):
        msg = (
            f"overlap trimming kept no rows of {selector.entry_key or 'the'} "
            f"output: it returned {len(df_feat)} rows, none inside frames "
            f"{selector.first}-{selector.last}, which the entry covers. The "
            f"feature re-based or replaced the frame axis it was given, so its "
            f"rows can no longer be told from its neighbours'."
        )
        raise ValueError(msg)
    return df_feat[keep].reset_index(drop=True)


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


def read_parquet_table_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read only *columns* of a parquet.

    The projected sibling of :func:`read_parquet_table`, and confined here for
    the same typing reason. A column the file does not hold is not an error --
    pyarrow raises, and the callers of this are asking a question whose honest
    answer is "unknown", so they catch rather than let it escape.
    """
    return pd.read_parquet(path, columns=columns)


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
