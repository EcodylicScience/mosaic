"""Cluster-versus-label scoring for notebooks, over :mod:`mosaic.core.evaluation`.

The alignment and the scorers live in :mod:`mosaic.core.evaluation`; this module
keeps the notebook entry points and the run listing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from .dataset import Dataset
from .evaluation import (
    LABEL_COLUMN,
    PREDICTION_COLUMN,
    align_predictions_to_labels,
    cluster_label_scores,
    perspective_consistency,
)
from .helpers import UNLABELED, make_entry_key, to_safe_name
from .scope import Scope


def cluster_gt_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    """Quantify agreement between ground-truth labels and cluster assignments.

    :func:`~mosaic.core.evaluation.cluster_label_scores` under its older name:
    rows labeled ``-1`` (unlabeled) are dropped and counted in ``n_unlabeled``,
    and ``majority_accuracy`` (equal to ``purity``) is the accuracy to read when
    there are more clusters than classes.
    """
    return cluster_label_scores(y_true, y_pred)


@dataclass(frozen=True)
class SequenceBundle:
    sequence_safe: str
    path: Path
    sequence: str = ""
    group: str = ""


def compute_cluster_label_agreement(
    ds: Dataset,
    cluster_feature: str,
    cluster_run_id: Optional[str] = None,
    label_kind: str = "behavior",
    cluster_column: str = "cluster",
    sequences: Optional[Sequence[str]] = None,
    max_frames: Optional[int] = None,
    rng_seed: Optional[int] = 42,
    include_per_sequence: bool = False,
    frame_source_feature: Optional[str] = None,
    *,
    perspective: int | Literal["each"] | None = None,
    scope: Scope | None = None,
    labels_run_id: str | None = None,
    tracks_run_id: str | None = None,
) -> dict[str, object]:
    """Score a per-frame clustering run against ground-truth labels.

    Each output row of the run is paired with the label of that row through
    :func:`~mosaic.core.evaluation.align_predictions_to_labels`: by group and
    sequence, in one labels variant, and by the row's own frame and individuals.
    Rows labeled ``-1`` (unlabeled) are dropped and counted, so a frame nobody
    annotated is never scored as a class.

    Parameters
    ----------
    ds : Dataset
        Bound dataset instance.
    cluster_feature : str
        Feature storage name under features/ (e.g.
        "global-ward__from__global-tsne__from__social+ego@v1").
    cluster_run_id : str, optional
        Specific run_id to use. If None, the most recent finished run is selected.
    label_kind : str
        Folder under labels/ (defaults to "behavior").
    cluster_column : str
        Column name inside the feature parquet holding cluster IDs (default "cluster").
    sequences : Sequence[str], optional
        Optional subset of sequences to analyze. Accepts raw names or already-safe
        names, and keeps those sequences in every group. ``scope`` narrows by group
        and entry as well.
    max_frames : int, optional
        If provided, subsample the scored rows to this many, after the perspective
        filter and the unlabeled drop.
    rng_seed : int, optional
        Seed passed to numpy.random.default_rng for subsampling.
    include_per_sequence : bool
        If True, returns per-sequence metrics (with confusion matrices stripped).
    frame_source_feature : str, optional
        Deprecated and ignored. The run's own ``frame`` column aligns its rows; a
        run without one is refused by ``load_values``.
    perspective : int or "each", optional
        For a pair run, which ``perspective`` to score: ``0`` scores the rows
        focal on the lower id of each pair (the resident in CalMS21), and
        ``"each"`` scores every perspective separately into
        ``metrics_by_perspective``. ``None`` pools both, which counts each pair
        twice.
    scope : Scope, optional
        The entries to read. ``None`` reads every entry of the run.
    labels_run_id : str, optional
        The labels variant, when the kind has more than one.
    tracks_run_id : str, optional
        The tracks variant, when an entry has more than one. Unset, the one the
        clustering run read is used.

    Returns
    -------
    dict
        ``metrics`` (see :func:`~mosaic.core.evaluation.cluster_label_scores`),
        ``n_sequences``, ``n_frames_total`` (scored rows before subsampling),
        ``n_frames_used``, ``sampled``, ``n_unlabeled_dropped``,
        ``n_missing_prediction``, ``aligned_sequences`` and ``per_sequence``
        (rows carrying ``group``, ``sequence``, ``sequence_safe`` and
        ``entry_key``), plus ``perspective_consistency``,
        ``n_perspective_pairs`` and ``n_perspectives`` when the run writes pair
        rows, and ``metrics_by_perspective`` for ``perspective="each"``.
    """
    from .pipeline.types import ResultColumn

    if frame_source_feature is not None:
        warnings.warn(
            "frame_source_feature is ignored: a run's rows are aligned by their own "
            "frame column",
            DeprecationWarning,
            stacklevel=2,
        )
    _, resolved_run_id, _ = _load_feature_sequence_index(
        ds, cluster_feature, cluster_run_id
    )
    aligned = align_predictions_to_labels(
        ds,
        ResultColumn(
            feature=cluster_feature, run_id=resolved_run_id, column=cluster_column
        ),
        label_kind=label_kind,
        scope=scope,
        labels_run_id=labels_run_id,
        tracks_run_id=tracks_run_id,
        drop_unlabeled=False,
    )
    frame = aligned.frame
    if sequences:
        wanted = {str(seq) for seq in sequences}
        names = frame["sequence"].astype(str)
        frame = frame.loc[names.isin(wanted) | names.map(to_safe_name).isin(wanted)]
        if frame.empty:
            raise RuntimeError("No cluster outputs match the requested sequences.")
    if frame.empty:
        raise RuntimeError(
            f"No rows of '{cluster_feature}' (run_id={resolved_run_id}) aligned to "
            f"'{label_kind}' labels."
        )

    result: dict[str, object] = {
        "cluster_feature": cluster_feature,
        "cluster_run_id": resolved_run_id,
        "label_kind": label_kind,
        "perspective": perspective,
        "n_missing_prediction": aligned.n_missing_prediction,
    }
    has_perspective = "perspective" in frame.columns
    if has_perspective:
        consistency = perspective_consistency(frame, PREDICTION_COLUMN)
        assert consistency is not None
        result["perspective_consistency"] = consistency[0]
        result["n_perspective_pairs"] = consistency[1]
        result["n_perspectives"] = int(frame["perspective"].nunique())

    if perspective == "each" and has_perspective:
        by_perspective: dict[int, dict[str, object]] = {}
        for value in sorted(int(v) for v in frame["perspective"].unique().tolist()):
            part = frame.loc[frame["perspective"] == value]
            by_perspective[value] = cluster_label_scores(
                part[LABEL_COLUMN].to_numpy(), part[PREDICTION_COLUMN].to_numpy()
            )
        result["metrics_by_perspective"] = by_perspective
    elif isinstance(perspective, int) and has_perspective:
        frame = frame.loc[frame["perspective"] == perspective]

    unlabeled = (frame[LABEL_COLUMN] == UNLABELED).to_numpy()
    result["n_unlabeled_dropped"] = int(unlabeled.sum())
    frame = frame.loc[~unlabeled]
    if frame.empty:
        raise RuntimeError("Every aligned row is unlabeled.")

    total = int(frame.shape[0])
    used = frame
    if max_frames and total > max_frames:
        rng = np.random.default_rng(rng_seed)
        used = frame.iloc[np.sort(rng.choice(total, size=max_frames, replace=False))]

    entries = frame.groupby(["group", "sequence"], sort=True)
    result.update(
        n_sequences=int(entries.ngroups),
        n_frames_total=total,
        n_frames_used=int(used.shape[0]),
        sampled=used.shape[0] < total,
        metrics=cluster_label_scores(
            used[LABEL_COLUMN].to_numpy(), used[PREDICTION_COLUMN].to_numpy()
        ),
        aligned_sequences=[
            {**_entry_fields(str(group), str(sequence)), "n_frames": int(len(part))}
            for (group, sequence), part in entries
        ],
        per_sequence=(
            [
                {
                    **_entry_fields(str(group), str(sequence)),
                    **_strip_confusion(
                        cluster_label_scores(
                            part[LABEL_COLUMN].to_numpy(),
                            part[PREDICTION_COLUMN].to_numpy(),
                        )
                    ),
                }
                for (group, sequence), part in entries
            ]
            if include_per_sequence
            else None
        ),
    )
    return result


def _entry_fields(group: str, sequence: str) -> dict[str, str]:
    """The names a reported entry carries."""
    return {
        "group": group,
        "sequence": sequence,
        "sequence_safe": to_safe_name(sequence),
        "entry_key": make_entry_key(group, sequence),
    }


def list_feature_runs(ds: Dataset, feature_name: str) -> pd.DataFrame:
    """
    Inspect features/<feature_name>/index.csv and summarize cached runs (run_id,
    timestamps, model params). Useful to discover available run_ids before looping
    over compute_cluster_label_agreement.
    """
    root = ds.get_root("features") / feature_name
    idx_path = root / "index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(f"Feature index missing: {idx_path}")
    df = pd.read_csv(idx_path)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "run_id",
                "started_at",
                "finished_at",
                "n_entries",
                "params_hash",
                "model_path",
                "k",
                "n_clusters",
                "params",
            ]
        )

    runs = []
    for run_id, group in df.groupby("run_id"):
        run_root = root / str(run_id)
        started = _first_value(group.get("started_at"))
        finished = _first_value(group.get("finished_at"))
        params_hash = _first_value(group.get("params_hash"))
        model_info = _load_model_summary(run_root)
        runs.append(
            {
                "feature": feature_name,
                "run_id": str(run_id),
                "started_at": started,
                "finished_at": finished,
                "n_entries": int(group.shape[0]),
                "params_hash": params_hash,
                **model_info,
            }
        )

    runs_df = pd.DataFrame(runs)
    if not runs_df.empty and "finished_at" in runs_df.columns:
        runs_df = runs_df.sort_values(by="finished_at", ascending=False, kind="stable")
    return runs_df.reset_index(drop=True)


def _load_feature_sequence_index(
    ds: Dataset, feature: str, run_id: Optional[str]
) -> tuple[Dict[str, SequenceBundle], str, Path]:
    root = ds.get_root("features") / feature
    idx_path = root / "index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(f"Feature index missing: {idx_path}")
    df = pd.read_csv(idx_path)
    resolved_run = _resolve_run_id(df, run_id)
    df = df[df["run_id"].astype(str) == resolved_run]
    run_root = root / resolved_run
    mapping: Dict[str, SequenceBundle] = {}
    for _, row in df.iterrows():
        abs_raw = str(row.get("abs_path", "")).strip()
        if not abs_raw:
            continue
        safe_seq = _safe_sequence_name(row)
        if not safe_seq:
            continue
        resolved = ds.resolve_path(abs_raw)
        mapping[safe_seq] = SequenceBundle(
            sequence_safe=safe_seq,
            path=resolved,
            sequence=str(row.get("sequence", "") or ""),
            group=str(row.get("group", "") or ""),
        )
    return mapping, resolved_run, run_root


def _resolve_run_id(df: pd.DataFrame, run_id: Optional[str]) -> str:
    if run_id is not None:
        return str(run_id)
    if "finished_at" in df.columns:
        done = df[df["finished_at"].fillna("").astype(str) != ""]
        if not done.empty:
            df = done
            order_col = "finished_at"
        else:
            order_col = "started_at" if "started_at" in df.columns else None
    else:
        order_col = "started_at" if "started_at" in df.columns else None

    if order_col:
        df = df.sort_values(by=[order_col], ascending=False, kind="stable")
    if df.empty:
        raise RuntimeError("Feature index is empty; run the feature first.")
    return str(df.iloc[0]["run_id"])


def _safe_sequence_name(row: pd.Series) -> str:
    safe_val = str(row.get("sequence_safe", "") or "").strip()
    if safe_val:
        return safe_val
    raw = str(row.get("sequence", "") or "").strip()
    if raw:
        return to_safe_name(raw)
    abs_path = str(row.get("abs_path", "") or "")
    if abs_path:
        return to_safe_name(Path(abs_path).stem)
    return ""


def _strip_confusion(metrics: dict[str, object]) -> dict[str, object]:
    return {
        k: v
        for k, v in metrics.items()
        if k not in {"confusion", "mapping", "majority_mapping"}
    }


def _load_model_summary(run_root: Path) -> dict:
    model_path = run_root / "model.joblib"
    summary = {
        "model_path": str(model_path) if model_path.exists() else "",
        "k": None,
        "n_clusters": None,
        "params": None,
    }
    if not model_path.exists():
        return summary
    try:
        bundle = joblib.load(model_path)
    except Exception:
        return summary
    params_info = _summarize_model_params(bundle)
    summary.update(params_info)
    summary["model_path"] = str(model_path)
    return summary


def _summarize_model_params(bundle: dict) -> dict:
    info = {"k": None, "n_clusters": None, "params": None}
    if not isinstance(bundle, dict):
        return info
    params = bundle.get("params")
    if isinstance(params, dict):
        info["params"] = params
        info["k"] = params.get("k", info["k"])
        info["n_clusters"] = params.get("n_clusters", info["n_clusters"])
    info["k"] = bundle.get("k", info["k"])
    info["n_clusters"] = bundle.get("n_clusters", info["n_clusters"])
    return info


def _first_value(series: Optional[pd.Series]) -> Optional[str]:
    if series is None or series.empty:
        return None
    for val in series:
        if pd.notna(val):
            return val
    return None
