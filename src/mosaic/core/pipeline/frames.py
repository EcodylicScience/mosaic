from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.helpers import make_entry_key, to_safe_name

from ._utils import hash_params, json_ready
from .index_csv import IndexCSV, RunIndexRowBase

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


# --- Frame extraction index helpers ---


def frames_run_root(ds: Dataset, method: str, run_id: str) -> Path:
    return ds.get_root("frames") / method / run_id


def frames_index_path(ds: Dataset, method: str) -> Path:
    return ds.get_root("frames") / method / "index.csv"


@dataclass(frozen=True, slots=True)
class FramesIndexRow(RunIndexRowBase):
    """Typed row for the frames index CSV."""

    method: str
    group: str
    sequence: str
    video_abs_path: str
    params_hash: str
    n_frames_extracted: int = 0
    n_frames_requested: int = 0


def frames_index(path: Path) -> IndexCSV[FramesIndexRow]:
    return IndexCSV(
        path,
        FramesIndexRow,
        dedup_keys=["run_id", "group", "sequence"],
    )


def list_media_pairs(
    ds: Dataset,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return filtered media index DataFrame with (group, sequence, abs_path, ..., video_order)."""
    idx_path = ds.get_root("media") / "index.csv"
    if not idx_path.exists():
        raise FileNotFoundError("media/index.csv not found; run index_media() first.")
    df = pd.read_csv(idx_path)
    df["group"] = df["group"].fillna("").astype(str)
    df["sequence"] = df["sequence"].fillna("").astype(str)
    df["group_safe"] = df["group_safe"].fillna("").astype(str)
    df["sequence_safe"] = df["sequence_safe"].fillna("").astype(str)
    # Ensure video_order column exists (backward compat with old indexes)
    if "video_order" not in df.columns:
        df["video_order"] = 0
    else:
        df["video_order"] = df["video_order"].fillna(0).astype(int)
    mask = pd.Series(True, index=df.index)
    if groups is not None:
        mask &= df["group"].isin({str(g) for g in groups})
    if sequences is not None:
        mask &= df["sequence"].isin({str(s) for s in sequences})
    return df[mask].reset_index(drop=True)


# --- Frame extraction Dataset methods ---


def extract_frames(
    ds,
    n_frames: int,
    method: str = "uniform",
    *,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    overwrite: bool = False,
    start_frame: int | None = None,
    end_frame: int | None = None,
    candidate_step: int = 1,
    crop=None,
    # k-means params
    kmeans_resize: tuple[int, int] = (64, 64),
    kmeans_grayscale: bool = True,
    kmeans_max_candidates: int | None = 5000,
    kmeans_batch_size: int = 1024,
    kmeans_max_iter: int = 100,
    kmeans_n_init="auto",
    random_state: int = 42,
    # parallelism
    parallel_workers: int | str | None = "auto",
    parallel_mode: str = "thread",
) -> str:
    """
    Extract representative frames from media files in the dataset.

    Parameters
    ----------
    n_frames : int
        Number of frames to extract per video.
    method : str
        "uniform" or "kmeans".
    groups, sequences : optional iterables
        Scope filter -- only process matching entries in media/index.csv.
    overwrite : bool
        If True, re-extract even if the output directory already exists.
    start_frame, end_frame : optional int
        Inclusive frame range; defaults to full video.
    candidate_step : int
        Frame stride for candidate selection (>=1).
    crop : optional
        Crop rectangle as (x, y, w, h) or {"x","y","w","h"}.
    kmeans_* : various
        K-means sampling parameters (only used when method="kmeans").
    random_state : int
        Random seed for reproducibility.
    parallel_workers : int, "auto", or None
        Number of videos to process concurrently. ``"auto"`` (default)
        uses ``min(cpu_count, 8)`` workers. Set to ``1`` or ``None``
        to disable parallelism.
    parallel_mode : str
        "thread" (default) or "process".

    Returns
    -------
    str
        The run_id for this extraction batch.
    """
    from mosaic.media.extraction import extract_frames as _extract_frames
    from mosaic.media.extraction import extract_frames_multi as _extract_frames_multi

    method_norm = str(method).strip().lower()
    if method_norm not in {"uniform", "kmeans"}:
        raise ValueError("method must be one of: 'uniform', 'kmeans'")

    # Build params dict for hashing and persistence
    extraction_params = {
        "n_frames": int(n_frames),
        "method": method_norm,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "candidate_step": int(candidate_step),
        "crop": crop,
        "random_state": int(random_state),
    }
    if method_norm == "kmeans":
        extraction_params.update(
            {
                "kmeans_resize": [int(kmeans_resize[0]), int(kmeans_resize[1])],
                "kmeans_grayscale": bool(kmeans_grayscale),
                "kmeans_max_candidates": kmeans_max_candidates,
                "kmeans_batch_size": int(kmeans_batch_size),
                "kmeans_max_iter": int(kmeans_max_iter),
                "kmeans_n_init": kmeans_n_init,
            }
        )

    params_hash = hash_params(extraction_params)
    run_id = f"{method_norm}-{params_hash}"
    run_root = frames_run_root(ds, method_norm, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    # Persist params
    params_path = run_root / "run_params.json"
    try:
        params_path.write_text(json.dumps(json_ready(extraction_params), indent=2))
    except Exception as exc:
        print(
            f"[extract_frames:{method_norm}] failed to save run_params.json: {exc}",
            file=sys.stderr,
        )

    # Resolve scope from media index
    media_df = list_media_pairs(ds, groups=groups, sequences=sequences)
    if media_df.empty:
        print(
            "[extract_frames] No media entries match the given scope.", file=sys.stderr
        )
        return run_id

    idx_path = frames_index_path(ds, method_norm)
    idx = frames_index(idx_path)
    idx.ensure()

    if parallel_workers == "auto":
        import os as _os

        max_workers = min(_os.cpu_count() or 1, 8)
    elif parallel_workers and int(parallel_workers) > 1:
        max_workers = int(parallel_workers)
    else:
        max_workers = 1
    p_mode = (parallel_mode or "thread").lower()
    if p_mode not in {"thread", "process"}:
        p_mode = "thread"

    # Group media rows by (group, sequence) to handle multi-video sequences
    def _extract_sequence(
        group: str, sequence: str, video_paths: list[Path]
    ) -> FramesIndexRow | None:
        seq_label = make_entry_key(group, sequence)
        seq_dir = run_root / seq_label

        if seq_dir.exists() and not overwrite:
            print(f"[extract_frames] skip {seq_label} (exists, overwrite=False)")
            return None

        if overwrite and seq_dir.exists():
            import shutil

            shutil.rmtree(seq_dir)

        try:
            if len(video_paths) == 1:
                result = _extract_frames(
                    video_path=video_paths[0],
                    n_frames=int(n_frames),
                    method=method_norm,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    candidate_step=int(candidate_step),
                    crop=crop,
                    kmeans_resize=kmeans_resize,
                    kmeans_grayscale=kmeans_grayscale,
                    kmeans_max_candidates=kmeans_max_candidates,
                    kmeans_batch_size=int(kmeans_batch_size),
                    kmeans_max_iter=int(kmeans_max_iter),
                    kmeans_n_init=kmeans_n_init,
                    random_state=int(random_state),
                    run_id=run_id,
                    output_dir=seq_dir,
                )
            else:
                result = _extract_frames_multi(
                    video_paths=video_paths,
                    n_frames=int(n_frames),
                    method=method_norm,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    candidate_step=int(candidate_step),
                    crop=crop,
                    kmeans_resize=kmeans_resize,
                    kmeans_grayscale=kmeans_grayscale,
                    kmeans_max_candidates=kmeans_max_candidates,
                    kmeans_batch_size=int(kmeans_batch_size),
                    kmeans_max_iter=int(kmeans_max_iter),
                    kmeans_n_init=kmeans_n_init,
                    random_state=int(random_state),
                    run_id=run_id,
                    output_dir=seq_dir,
                )
            # Re-write run_info.json with relative paths for portability
            _manifest_path = seq_dir / "run_info.json"
            if _manifest_path.exists():
                _mdata = json.loads(_manifest_path.read_text())
                _mdata["output_dir"] = ds._relative_to_root(seq_dir)
                if "video_path" in _mdata:
                    _mdata["video_path"] = ds._relative_to_root(
                        Path(_mdata["video_path"])
                    )
                _manifest_path.write_text(json.dumps(_mdata, indent=2))
            return FramesIndexRow(
                run_id=run_id,
                method=method_norm,
                group=group,
                sequence=sequence,
                abs_path=seq_dir,
                n_frames_extracted=result.n_extracted,
                n_frames_requested=result.n_requested,
                video_abs_path=json.dumps([str(p) for p in video_paths])
                if len(video_paths) > 1
                else str(video_paths[0]),
                params_hash=params_hash,
            )
        except Exception as exc:
            print(
                f"[extract_frames] ERROR processing {seq_label}: {exc}", file=sys.stderr
            )
            return None

    # Build per-sequence work items from (possibly multi-video) media index
    work_items = []
    grouped = media_df.groupby(["group", "sequence"])
    for (g, s), sub in grouped:
        g, s = str(g), str(s)
        sub = sub.sort_values("video_order")
        paths = [ds.resolve_path(r["abs_path"]) for _, r in sub.iterrows()]

        # When group/sequence are empty (no tracks indexed yet), use video stem
        if not g and not to_safe_name(s):
            video_stem = paths[0].stem
            s = video_stem

        work_items.append((g, s, paths))

    # Execute extraction
    index_rows: list[FramesIndexRow] = []

    if max_workers > 1:
        PoolCls = ProcessPoolExecutor if p_mode == "process" else ThreadPoolExecutor
        with PoolCls(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_extract_sequence, g, s, vp): (g, s)
                for g, s, vp in work_items
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    index_rows.append(result)
    else:
        for g, s, vp in work_items:
            result = _extract_sequence(g, s, vp)
            if result is not None:
                index_rows.append(result)

    # Update index
    if index_rows:
        idx.append(index_rows)
        idx.mark_finished(run_id)

    print(
        f"[extract_frames:{method_norm}] completed run_id={run_id} "
        f"({len(index_rows)}/{len(work_items)} sequences) -> {run_root}"
    )
    return run_id


def list_frame_runs(ds: Dataset, method: str | None = None) -> pd.DataFrame:
    """
    List all frame extraction runs tracked in the frames index.

    Parameters
    ----------
    method : str, optional
        Filter to a specific method ("uniform" or "kmeans").
        If None, returns runs across all methods.

    Returns
    -------
    pd.DataFrame
        Index rows for matching extraction runs.
    """
    frames_root = ds.get_root("frames")
    if not frames_root.exists():
        return pd.DataFrame(
            columns=[f.name for f in dataclasses.fields(FramesIndexRow)]
        )

    methods = (
        [method] if method else [d.name for d in frames_root.iterdir() if d.is_dir()]
    )
    dfs = []
    for m in methods:
        idx_path = frames_root / m / "index.csv"
        if idx_path.exists():
            dfs.append(pd.read_csv(idx_path))
    if not dfs:
        return pd.DataFrame(
            columns=[f.name for f in dataclasses.fields(FramesIndexRow)]
        )
    return pd.concat(dfs, ignore_index=True)


def get_frame_paths(
    ds,
    method: str,
    run_id: str | None = None,
    group: str | None = None,
    sequence: str | None = None,
) -> list[Path]:
    """
    Return paths to extracted frame PNGs for a given scope.

    Parameters
    ----------
    method : str
        Extraction method ("uniform" or "kmeans").
    run_id : str, optional
        Specific run_id. If None, uses the latest run.
    group, sequence : str, optional
        Filter to a specific (group, sequence).

    Returns
    -------
    list[Path]
        Sorted list of PNG file paths.
    """
    frames_root = ds.get_root("frames")
    method_root = frames_root / method
    if not method_root.exists():
        return []

    # Resolve run_id
    if run_id is None:
        idx_path = method_root / "index.csv"
        if not idx_path.exists():
            return []
        df = pd.read_csv(idx_path)
        if df.empty:
            return []
        run_id = df["run_id"].iloc[-1]

    run_root = method_root / run_id
    if not run_root.exists():
        return []

    # Collect PNG paths
    paths = []
    if group is not None or sequence is not None:
        seq_label = make_entry_key(group or "", sequence or "")
        seq_dir = run_root / seq_label
        if seq_dir.exists():
            paths = sorted(seq_dir.glob("*.png"))
    else:
        for seq_dir in sorted(run_root.iterdir()):
            if seq_dir.is_dir():
                paths.extend(sorted(seq_dir.glob("*.png")))
    return paths


def get_frame_manifests(
    ds,
    method: str,
    run_id: str | None = None,
    group: str | None = None,
    sequence: str | None = None,
) -> list[dict[str, object]]:
    """
    Load run_info.json manifests from extracted frame directories.

    Parameters
    ----------
    method : str
        Extraction method ("uniform" or "kmeans").
    run_id : str, optional
        Specific run_id. If None, uses the latest run.
    group, sequence : str, optional
        Filter to a specific (group, sequence).

    Returns
    -------
    list[dict]
        List of manifest dicts loaded from run_info.json files,
        one per sequence directory. Each dict contains video_path,
        files, video_meta, selected_frame_indices, etc.
    """
    frames_root = ds.get_root("frames")
    method_root = frames_root / method
    if not method_root.exists():
        return []

    # Resolve run_id
    if run_id is None:
        idx_path = method_root / "index.csv"
        if not idx_path.exists():
            return []
        df = pd.read_csv(idx_path)
        if df.empty:
            return []
        run_id = df["run_id"].iloc[-1]

    run_root = method_root / run_id
    if not run_root.exists():
        return []

    # Collect sequence directories
    if group is not None or sequence is not None:
        seq_label = make_entry_key(group or "", sequence or "")
        seq_dirs = [run_root / seq_label]
    else:
        seq_dirs = sorted(d for d in run_root.iterdir() if d.is_dir())

    manifests = []
    for seq_dir in seq_dirs:
        manifest_path = seq_dir / "run_info.json"
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text())
            # Resolve relative paths so callers always see absolute paths
            manifest_dir = manifest_path.parent
            for f in data.get("files", []):
                if "path" in f:
                    fp = Path(f["path"])
                    if not fp.is_absolute():
                        f["path"] = str((manifest_dir / fp).resolve())
            if "output_dir" in data:
                od = Path(data["output_dir"])
                if not od.is_absolute():
                    data["output_dir"] = str(ds.resolve_path(od))
            if "video_path" in data:
                vp = Path(data["video_path"])
                if not vp.is_absolute():
                    data["video_path"] = str(ds.resolve_path(vp))
            manifests.append(data)
    return manifests
