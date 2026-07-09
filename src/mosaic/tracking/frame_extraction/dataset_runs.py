from __future__ import annotations

import dataclasses
import json
import shutil
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import pandas as pd

from mosaic.core.helpers import make_entry_key, to_safe_name
from mosaic.core.pipeline._utils import hash_params, json_ready
from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase
from mosaic.core.pipeline.job import Cancelled, JobContext
from mosaic.core.pipeline.types import HASH_EXCLUDE, Params
from mosaic.tracking.registry import TrackingOp, register_tracking_op, run_tracking_op

from .extraction import extract_frames as _extract_frames
from .extraction import extract_frames_multi as _extract_frames_multi

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.progress import ProgressCallback


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
    media_key = ds.resolve_media_root()
    idx_path = ds.get_root(media_key) / "index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"{media_key}/index.csv not found; run index_media() first."
        )
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


# --- Frame extraction op (registered under the Job Contract) ---


class ExtractFramesParams(Params):
    """Typed parameters for the ``extract-frames`` tracking op.

    Scope selectors (``groups``/``sequences``), ``overwrite``, and the
    parallelism knobs are ``HASH_EXCLUDE``: they select *which* work runs or
    *how fast*, but the run_id addresses only the extraction *settings* (so the
    same settings share a run_id and add per-sequence subdirs, like frames/trex).
    """

    n_frames: int
    method: str = "uniform"
    start_frame: int | None = None
    end_frame: int | None = None
    candidate_step: int = 1
    crop: tuple[int, int, int, int] | None = None
    random_state: int = 42
    kmeans_resize: tuple[int, int] = (64, 64)
    kmeans_grayscale: bool = True
    kmeans_max_candidates: int | None = 5000
    kmeans_batch_size: int = 1024
    kmeans_max_iter: int = 100
    kmeans_n_init: str | int = "auto"
    # scope + throughput (excluded from the content run_id)
    groups: Annotated[list[str] | None, HASH_EXCLUDE] = None
    sequences: Annotated[list[str] | None, HASH_EXCLUDE] = None
    overwrite: Annotated[bool, HASH_EXCLUDE] = False
    parallel_workers: Annotated[int | str | None, HASH_EXCLUDE] = "auto"
    parallel_mode: Annotated[str, HASH_EXCLUDE] = "thread"


@dataclass(frozen=True, slots=True)
class _ExtractSpec:
    """Picklable unit of work for one (group, sequence) -- process-safe."""

    group: str
    sequence: str
    video_paths: tuple[Path, ...]
    seq_dir: Path
    run_id: str
    params_hash: str
    n_frames: int
    method: str
    start_frame: int | None
    end_frame: int | None
    candidate_step: int
    crop: tuple[int, int, int, int] | None
    random_state: int
    kmeans_resize: tuple[int, int]
    kmeans_grayscale: bool
    kmeans_max_candidates: int | None
    kmeans_batch_size: int
    kmeans_max_iter: int
    kmeans_n_init: str | int
    overwrite: bool


def _extract_one(spec: _ExtractSpec) -> FramesIndexRow | None:
    """Extract one sequence. Module-scope (picklable) so process mode works.

    Manifest path-rewriting (which needs the Dataset) is done by the caller.
    """
    seq_dir = spec.seq_dir
    if seq_dir.exists() and not spec.overwrite:
        print(
            f"[extract_frames] skip {make_entry_key(spec.group, spec.sequence)} "
            f"(exists, overwrite=False)"
        )
        return None
    if spec.overwrite and seq_dir.exists():
        shutil.rmtree(seq_dir)

    kmeans_kw = dict(
        kmeans_resize=spec.kmeans_resize,
        kmeans_grayscale=spec.kmeans_grayscale,
        kmeans_max_candidates=spec.kmeans_max_candidates,
        kmeans_batch_size=spec.kmeans_batch_size,
        kmeans_max_iter=spec.kmeans_max_iter,
        kmeans_n_init=spec.kmeans_n_init,
    )
    try:
        if len(spec.video_paths) == 1:
            result = _extract_frames(
                video_path=spec.video_paths[0],
                n_frames=spec.n_frames,
                method=spec.method,
                start_frame=spec.start_frame,
                end_frame=spec.end_frame,
                candidate_step=spec.candidate_step,
                crop=spec.crop,
                random_state=spec.random_state,
                run_id=spec.run_id,
                output_dir=seq_dir,
                **kmeans_kw,
            )
        else:
            result = _extract_frames_multi(
                video_paths=list(spec.video_paths),
                n_frames=spec.n_frames,
                method=spec.method,
                start_frame=spec.start_frame,
                end_frame=spec.end_frame,
                candidate_step=spec.candidate_step,
                crop=spec.crop,
                random_state=spec.random_state,
                run_id=spec.run_id,
                output_dir=seq_dir,
                **kmeans_kw,
            )
    except Exception as exc:
        print(
            f"[extract_frames] ERROR processing "
            f"{make_entry_key(spec.group, spec.sequence)}: {exc}",
            file=sys.stderr,
        )
        return None

    return FramesIndexRow(
        run_id=spec.run_id,
        method=spec.method,
        group=spec.group,
        sequence=spec.sequence,
        abs_path=seq_dir,
        n_frames_extracted=result.n_extracted,
        n_frames_requested=result.n_requested,
        video_abs_path=json.dumps([str(p) for p in spec.video_paths])
        if len(spec.video_paths) > 1
        else str(spec.video_paths[0]),
        params_hash=spec.params_hash,
    )


def _rewrite_manifest(ds: Dataset, seq_dir: Path) -> None:
    """Rewrite run_info.json with dataset-relative paths for portability."""
    manifest_path = seq_dir / "run_info.json"
    if not manifest_path.exists():
        return
    try:
        mdata = json.loads(manifest_path.read_text())
        mdata["output_dir"] = ds._relative_to_root(seq_dir)
        if "video_path" in mdata:
            mdata["video_path"] = ds._relative_to_root(Path(mdata["video_path"]))
        manifest_path.write_text(json.dumps(mdata, indent=2))
    except Exception as exc:
        print(
            f"[extract_frames] manifest rewrite failed for {seq_dir}: {exc}",
            file=sys.stderr,
        )


def _resolve_max_workers(parallel_workers: int | str | None) -> int:
    if parallel_workers == "auto":
        import os as _os

        return min(_os.cpu_count() or 1, 8)
    try:
        n = int(parallel_workers)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1
    return n if n > 1 else 1


def _run_extract_frames(ds: Dataset, p: ExtractFramesParams, ctx: JobContext) -> str:
    """Extraction body executed inside a job_context (the op's payload)."""
    method_norm = str(p.method).strip().lower()
    if method_norm not in {"uniform", "kmeans"}:
        raise ValueError("method must be one of: 'uniform', 'kmeans'")

    params_hash = hash_params(p.identity_dump())
    run_id = f"{method_norm}-{params_hash}"
    ctx.set_run_id(run_id)

    run_root = frames_run_root(ds, method_norm, run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        (run_root / "run_params.json").write_text(
            json.dumps(json_ready(p.identity_dump()), indent=2)
        )
    except Exception as exc:
        print(
            f"[extract_frames:{method_norm}] failed to save run_params.json: {exc}",
            file=sys.stderr,
        )

    media_df = list_media_pairs(ds, groups=p.groups, sequences=p.sequences)
    if media_df.empty:
        print(
            "[extract_frames] No media entries match the given scope.", file=sys.stderr
        )
        return run_id

    # Build picklable per-sequence work specs (multi-video sequences merged).
    specs: list[_ExtractSpec] = []
    for (g, s), sub in media_df.groupby(["group", "sequence"]):
        g, s = str(g), str(s)
        sub = sub.sort_values("video_order")
        paths = tuple(ds.resolve_path(r["abs_path"]) for _, r in sub.iterrows())
        if not g and not to_safe_name(s):
            s = paths[0].stem
        specs.append(
            _ExtractSpec(
                group=g,
                sequence=s,
                video_paths=paths,
                seq_dir=run_root / make_entry_key(g, s),
                run_id=run_id,
                params_hash=params_hash,
                n_frames=int(p.n_frames),
                method=method_norm,
                start_frame=p.start_frame,
                end_frame=p.end_frame,
                candidate_step=int(p.candidate_step),
                crop=p.crop,
                random_state=int(p.random_state),
                kmeans_resize=p.kmeans_resize,
                kmeans_grayscale=p.kmeans_grayscale,
                kmeans_max_candidates=p.kmeans_max_candidates,
                kmeans_batch_size=int(p.kmeans_batch_size),
                kmeans_max_iter=int(p.kmeans_max_iter),
                kmeans_n_init=p.kmeans_n_init,
                overwrite=p.overwrite,
            )
        )

    ctx.set_total(len(specs))
    idx = frames_index(frames_index_path(ds, method_norm))
    idx.ensure()
    index_rows: list[FramesIndexRow] = []

    def _collect(row: FramesIndexRow | None) -> None:
        if row is not None:
            _rewrite_manifest(ds, row.abs_path)
            index_rows.append(row)

    max_workers = _resolve_max_workers(p.parallel_workers)
    p_mode = (p.parallel_mode or "thread").lower()
    if p_mode not in {"thread", "process"}:
        p_mode = "thread"

    try:
        if max_workers > 1:
            PoolCls = ProcessPoolExecutor if p_mode == "process" else ThreadPoolExecutor
            with PoolCls(max_workers=max_workers) as pool:
                futures = {pool.submit(_extract_one, spec): spec for spec in specs}
                done = 0
                for future in as_completed(futures):
                    if ctx.cancel_token.is_cancelled():
                        for f in futures:
                            f.cancel()
                        raise Cancelled()
                    row = future.result()
                    done += 1
                    spec = futures[future]
                    ctx.progress.on_entry_end(
                        done, len(specs), make_entry_key(spec.group, spec.sequence)
                    )
                    ctx.heartbeat(done)
                    _collect(row)
        else:
            for i, spec in enumerate(specs):
                ctx.check_cancel()
                key = make_entry_key(spec.group, spec.sequence)
                ctx.progress.on_entry_start(i, len(specs), key)
                _collect(_extract_one(spec))
                ctx.progress.on_entry_end(i + 1, len(specs), key)
                ctx.heartbeat(i + 1)
    finally:
        if index_rows:
            idx.append(index_rows)
            idx.mark_finished(run_id)

    print(
        f"[extract_frames:{method_norm}] completed run_id={run_id} "
        f"({len(index_rows)}/{len(specs)} sequences) -> {run_root}"
    )
    return run_id


@register_tracking_op
class ExtractFramesOp(TrackingOp[ExtractFramesParams]):
    kind = "extract-frames"
    category = "extract"
    version = "0.1"
    Params = ExtractFramesParams

    def target(self, params: ExtractFramesParams) -> str:
        return f"extract-{params.method}"

    def run(self, ds: Dataset, params: ExtractFramesParams, ctx: JobContext) -> str:
        return _run_extract_frames(ds, params, ctx)


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
    crop: tuple[int, int, int, int] | None = None,
    kmeans_resize: tuple[int, int] = (64, 64),
    kmeans_grayscale: bool = True,
    kmeans_max_candidates: int | None = 5000,
    kmeans_batch_size: int = 1024,
    kmeans_max_iter: int = 100,
    kmeans_n_init: str | int = "auto",
    random_state: int = 42,
    parallel_workers: int | str | None = "auto",
    parallel_mode: str = "thread",
    # Job Contract
    execution_id: str | None = None,
    owner: str = "",
    track: bool = True,
    progress_callback: "ProgressCallback | None" = None,
    cancel_token=None,
) -> str:
    """Extract representative frames from media as a tracked Job-Contract run.

    Ergonomic typed front door for the ``extract-frames`` tracking op: builds
    :class:`ExtractFramesParams` and dispatches via
    :func:`mosaic.tracking.run_tracking_op`, which records the attempt, reports
    per-sequence progress, and supports cooperative cancellation. Returns the
    content ``run_id``.

    Parameters mirror the previous signature (``method`` = "uniform"|"kmeans",
    ``groups``/``sequences`` scope, k-means knobs, ``parallel_workers``/
    ``parallel_mode``) plus the standard contract knobs
    (``execution_id``/``owner``/``track``/``progress_callback``/``cancel_token``).
    """
    params = ExtractFramesParams(
        n_frames=n_frames,
        method=str(method).strip().lower(),
        start_frame=start_frame,
        end_frame=end_frame,
        candidate_step=candidate_step,
        crop=crop,
        random_state=random_state,
        kmeans_resize=kmeans_resize,
        kmeans_grayscale=kmeans_grayscale,
        kmeans_max_candidates=kmeans_max_candidates,
        kmeans_batch_size=kmeans_batch_size,
        kmeans_max_iter=kmeans_max_iter,
        kmeans_n_init=kmeans_n_init,
        groups=list(groups) if groups is not None else None,
        sequences=list(sequences) if sequences is not None else None,
        overwrite=overwrite,
        parallel_workers=parallel_workers,
        parallel_mode=parallel_mode,
    )
    return run_tracking_op(
        ds,
        "extract-frames",
        params,
        execution_id=execution_id,
        owner=owner,
        track=track,
        progress_callback=progress_callback,
        cancel_token=cancel_token,
    )


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
