"""TRex per-id .npz track converter.

Converts TRex-exported per-individual NPZ files to the standardized
trex_v1 parquet schema.
"""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd

from mosaic.core.dataset import register_track_converter
from mosaic.core.schema import ensure_track_schema


# --- TRex per-id NPZ support ---
# Matches: _id0, _id1, _fish0, _fish1, _bee0, _bee1, etc.
_TREX_ID_SUFFIX = re.compile(r"_(?:id|fish|bee|animal|ind)(\d+)$", re.IGNORECASE)


def _strip_trex_seq(stem: str) -> str:
    """Return filename stem with trailing individual ID suffix removed, if present.

    Handles patterns like: _id0, _fish2, _bee1, _animal3, _ind0
    Examples:
        hex_7_fish2 -> hex_7
        OCI_1_fish0 -> OCI_1
        video1_id3 -> video1
    """
    m = _TREX_ID_SUFFIX.search(stem)
    if m:
        return stem[: m.start()]
    return stem


def _load_npz_to_df(filepath: Path) -> pd.DataFrame:
    """
    Flatten a TRex-like NPZ (per-id) into a DataFrame, robust to arrays with
    slightly different lengths in the same file. We pick the most common length
    across arrays as the target 'n', and truncate longer arrays to fit.
    """
    data = np.load(filepath, allow_pickle=True)
    keys = list(data.files)

    skip_keys = {} # including all for now.  Could make this a parameter to pass

    # Determine candidate lengths per key
    lens = []
    for k in keys:
        if k in skip_keys:
            continue
        v = data[k]
        if getattr(v, "ndim", 0) > 0:
            lens.append(int(v.shape[0]))
    if not lens:
        raise ValueError(f"No array-like keys with length found in NPZ: {filepath}")

    # Prefer 'time' length if present and 1D; else use the mode length
    if "time" in data.files and getattr(data["time"], "ndim", 0) == 1:
        n = int(data["time"].shape[0])
    else:
        # mode (most common) length among arrays
        vals, counts = np.unique(np.array(lens), return_counts=True)
        n = int(vals[np.argmax(counts)])

    cols: dict[str, np.ndarray] = {}

    for k in sorted(keys):
        if k in skip_keys:
            continue
        v = data[k]

        if np.ndim(v) == 0:
            # scalar -> broadcast
            cols[k] = np.repeat(v.item(), n)
            continue

        # 1D or ND: align first dimension to n by truncation (or pad if you prefer)
        if v.shape[0] < n:
            # If you want to pad instead of skip, do it here; for now, we take the shorter n
            # but to keep a consistent table width we'll just truncate n down for this column.
            vi = v  # keep as-is; we will align by slicing [:vi.shape[0]] and then pad
            # simple pad with NaN to length n
            pad = np.full((n - vi.shape[0],), np.nan, dtype=float) if v.ndim == 1 else \
                  np.full((n - vi.shape[0],) + v.shape[1:], np.nan, dtype=float)
            vi = np.concatenate([vi, pad], axis=0)
        else:
            vi = v[:n]

        if vi.ndim == 1:
            cols[k] = vi
        else:
            flat = vi.reshape(n, -1)
            for i in range(flat.shape[1]):
                cols[f"{k}_{i}"] = flat[:, i]

    df = pd.DataFrame(cols)

    # id
    if "id" in data.files and np.ndim(data["id"]) > 0:
        try:
            id_val = int(np.array(data["id"]).ravel()[0])
        except Exception:
            # fallback for weird dtypes
            id_val = int(np.array(data["id"]).ravel()[0].astype(np.int64))
    else:
        # derive from filename suffix _id<digits>
        m = _TREX_ID_SUFFIX.search(filepath.stem)
        id_val = int(m.group(1)) if m else 0
    df["id"] = id_val

    # frame/time
    if "frame" not in df.columns:
        df["frame"] = np.arange(len(df), dtype=int)
    if "time" not in df.columns:
        if "frame_rate" in data.files:
            fr = float(np.array(data["frame_rate"]).ravel()[0])
            fr = fr if fr > 0 else 1.0
            df["time"] = df["frame"] / fr
        else:
            df["time"] = df["frame"].astype(float)

    return df


def _trex_npz_converter(path: Path, params: dict) -> pd.DataFrame:
    """
    Convert a per-id TRex NPZ into our standard T-Rex-like DataFrame.
    Ensures 'group' and 'sequence' columns; derives sequence from stem by
    stripping '_id\\d+' unless explicitly provided in params.
    """
    df = _load_npz_to_df(path)

    group = params.get("group") or ""
    sequence = params.get("sequence") or _strip_trex_seq(path.stem)

    df["group"] = group
    df["sequence"] = sequence

    # Validate (non-strict) against trex_v1 schema
    ensure_track_schema(df, "trex_v1", strict=False)
    return df


# Register converter
register_track_converter("trex_npz", _trex_npz_converter)
