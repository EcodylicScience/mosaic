"""Data loading functions for visualization.

This module contains functions for loading tracks and labels:
- load_tracks_and_labels: Load tracks + feature/model labels for a sequence
- load_ground_truth_labels: Load GT labels from labels/<kind>/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from mosaic.core.pipeline.index import feature_index_path, latest_feature_run_root
from mosaic.core.pipeline.iteration import yield_sequences


def _pick_label_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristic to pick a label-like column from a feature/model output frame.
    Prioritizes common names, falls back to the first non-index column.
    """
    preferred = ["label_id", "label", "prediction", "cluster", "behavior", "state"]
    for col in preferred:
        if col in df.columns:
            return col
    skip = {
        "frame",
        "sequence",
        "group",
        "id",
        "id1",
        "id2",
        "id_a",
        "id_b",
        "id_A",
        "id_B",
        "entity_level",
    }
    for col in df.columns:
        if col not in skip:
            return col
    return None


def _normalize_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize identity columns to canonical id1/id2 when possible.
    """
    out = df.copy()
    if "id1" not in out.columns and "id" in out.columns:
        out["id1"] = out["id"]
    if "id1" not in out.columns and "id_a" in out.columns:
        out["id1"] = out["id_a"]
    if "id2" not in out.columns and "id_b" in out.columns:
        out["id2"] = out["id_b"]
    if "id1" not in out.columns and "id_A" in out.columns:
        out["id1"] = out["id_A"]
    if "id2" not in out.columns and "id_B" in out.columns:
        out["id2"] = out["id_B"]

    if "id1" in out.columns:
        out["id1"] = pd.to_numeric(out["id1"], errors="coerce")
    if "id2" in out.columns:
        out["id2"] = pd.to_numeric(out["id2"], errors="coerce")
    return out


def load_tracks_and_labels(
    ds,
    group: str,
    sequence: str,
    feature_runs: Dict[str, Optional[str]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load a single sequence's tracks plus per-frame labels from feature/model runs.

    Parameters
    ----------
    ds : Dataset
        Loaded Dataset instance.
    group, sequence : str
        The scope to load.
    feature_runs : dict[str, str | None]
        Mapping of feature/model storage names -> run_id.
        If run_id is None, the latest finished run is used.

    Returns
    -------
    tracks_df : pd.DataFrame
        Standard tracks for the requested (group, sequence).
    labels : dict
        {
          "per_id": {feature_name: {id_value: Series}},
          "per_pair": {feature_name: {(id1, id2): Series}},
          "raw": {feature_name: DataFrame}  # full frame per feature for bespoke use
        }
        Series are indexed by frame and hold the chosen label column.
    """
    tracks_df = None
    for _, _, df in yield_sequences(ds, groups=[group], sequences=[sequence]):
        tracks_df = df
        break
    if tracks_df is None:
        raise FileNotFoundError(
            f"No tracks found for group='{group}', sequence='{sequence}'."
        )

    per_id: dict[str, dict[Any, pd.Series]] = {}
    per_pair: dict[str, dict[Tuple[Any, Any], pd.Series]] = {}
    raw: dict[str, pd.DataFrame] = {}

    for feature_name, run_id in feature_runs.items():
        # Resolve run_id if not provided
        resolved_run_id = run_id
        if not resolved_run_id:
            resolved_run_id, _ = latest_feature_run_root(ds, feature_name)

        idx_path = feature_index_path(ds, feature_name)
        if not idx_path.exists():
            raise FileNotFoundError(
                f"Missing feature index for '{feature_name}': {idx_path}"
            )
        df_idx = pd.read_csv(idx_path)

        # Normalize NaNs/None to empty strings so blank/absent groups still match
        for col in ("sequence", "group"):
            if col in df_idx.columns:
                df_idx[col] = df_idx[col].fillna("").astype(str)

        df_idx = df_idx[df_idx["run_id"].astype(str) == str(resolved_run_id)]
        df_idx = df_idx[df_idx["sequence"].astype(str) == str(sequence)]
        if "group" in df_idx.columns:
            df_idx = df_idx[df_idx["group"].astype(str) == str(group)]

        if df_idx.empty:
            raise FileNotFoundError(
                f"No rows in feature index for '{feature_name}' run_id='{resolved_run_id}' "
                f"group='{group}' sequence='{sequence}'."
            )

        abs_path_raw = df_idx.iloc[0]["abs_path"]
        path = ds.resolve_path(abs_path_raw)
        df_feat = pd.read_parquet(path)
        raw[feature_name] = df_feat

        label_col = _pick_label_column(df_feat)
        if not label_col or "frame" not in df_feat.columns:
            continue  # nothing label-like to index

        df_norm = _normalize_identity_columns(df_feat)
        if "id1" in df_norm.columns:
            has_id1 = df_norm["id1"].notna()
            has_id2 = (
                df_norm["id2"].notna()
                if "id2" in df_norm.columns
                else pd.Series(False, index=df_norm.index)
            )

            # Per-pair rows (id1 + id2 present)
            pair_rows = df_norm[has_id1 & has_id2]
            if not pair_rows.empty:
                pairs = pair_rows[["id1", "id2"]].apply(
                    lambda row: tuple(sorted((int(row["id1"]), int(row["id2"])))),
                    axis=1,
                )
                pair_rows = pair_rows.assign(_pair=pairs)
                for pair, sub in pair_rows.groupby("_pair"):
                    series = sub.sort_values("frame").groupby("frame")[label_col].last()
                    per_pair.setdefault(feature_name, {})[pair] = series.sort_index()

            # Per-individual rows (id1 present, id2 missing)
            indiv_rows = df_norm[has_id1 & ~has_id2]
            if not indiv_rows.empty:
                for id_val, sub in indiv_rows.groupby("id1"):
                    id_key = int(id_val)
                    series = sub.sort_values("frame").groupby("frame")[label_col].last()
                    per_id.setdefault(feature_name, {})[id_key] = series.sort_index()

            # Global rows (no id1) stay under None
            global_rows = df_norm[~has_id1]
            if not global_rows.empty:
                series = (
                    global_rows.sort_values("frame").groupby("frame")[label_col].last()
                )
                per_id.setdefault(feature_name, {})[None] = series.sort_index()
        else:
            # No identity columns: global series
            series = df_norm.sort_values("frame").groupby("frame")[label_col].last()
            per_id.setdefault(feature_name, {})[None] = series.sort_index()

    labels = {"per_id": per_id, "per_pair": per_pair, "raw": raw}
    return tracks_df, labels


def load_ground_truth_labels(
    ds,
    label_kind: str,
    group: str,
    sequence: str,
) -> pd.DataFrame:
    """
    Load per-frame ground-truth labels for a given kind/group/sequence.

    Returns a DataFrame with columns:
        frame, label_id, label_name (if mapping provided in the npz).
        For individual_pair_v1 format, also includes id1, id2 columns.
    """
    labels_root = Path(ds.get_root("labels")) / label_kind
    idx_path = labels_root / "index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"Label index not found for kind='{label_kind}': {idx_path}"
        )
    df_idx = pd.read_csv(idx_path)
    if df_idx.empty:
        raise FileNotFoundError(f"No labels indexed for kind='{label_kind}'.")

    hits = df_idx[
        (df_idx["group"].astype(str) == str(group))
        & (df_idx["sequence"].astype(str) == str(sequence))
    ]
    if hits.empty:
        raise FileNotFoundError(
            f"No GT labels for kind='{label_kind}' group='{group}' sequence='{sequence}'."
        )

    path = ds.resolve_path(hits.iloc[0]["abs_path"])
    payload = np.load(path, allow_pickle=True)
    label_ids = payload["labels"]
    if "frames" in payload.files:
        frames = payload["frames"]
    else:
        # Dense label format (e.g. mabe22_behavior): one entry per frame,
        # implicitly indexed 0..N-1.
        frames = np.arange(len(label_ids), dtype=np.int64)
    label_id_list = payload.get("label_ids")
    label_name_list = payload.get("label_names")
    id_to_name: dict[int, str] = {}
    if label_id_list is not None and label_name_list is not None:
        for lid, name in zip(label_id_list, label_name_list):
            id_to_name[int(lid)] = str(name)
    label_names = [id_to_name.get(int(val), str(val)) for val in label_ids]

    result = {
        "frame": frames.astype(int, copy=False),
        "label_id": label_ids.astype(int, copy=False),
        "label_name": label_names,
    }

    # Include individual_ids for pair-aware labels (individual_pair_v1 format)
    if "individual_ids" in payload.files:
        individual_ids = np.asarray(payload["individual_ids"])
        if individual_ids.ndim == 1:
            individual_ids = individual_ids.reshape(-1, 2)
        result["id1"] = individual_ids[:, 0].astype(int, copy=False)
        result["id2"] = individual_ids[:, 1].astype(int, copy=False)

    return pd.DataFrame(result)


def demo_load_visual_inputs(
    ds, group: str, sequence: str, features: Dict[str, Optional[str]]
):
    """
    Small wrapper to quickly inspect what load_tracks_and_labels returns.
    Usage (notebook):
        tracks, labels = demo_load_visual_inputs(dataset, "G1", "S1",
                                                 {"temporal-stack": None,
                                                  "behavior-xgb-pred": "<run_id>"})
    """
    tracks, labels = load_tracks_and_labels(ds, group, sequence, features)
    print(f"Tracks shape: {tracks.shape}")
    for kind in ("per_id", "per_pair"):
        print(f"{kind}:")
        for feat, mapping in labels[kind].items():
            print(f"  {feat}: {len(mapping)} series")
    return tracks, labels
