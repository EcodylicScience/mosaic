"""Dataset-level TREx runs: content-addressed, tracked, tracks-integrated.

``run_trex(ds, ...)`` is the first-class entry point that turns the standalone
TREx CLI wrappers (:mod:`mosaic.tracking.trex.run`) into a Job-Contract stage,
mirroring :func:`mosaic.tracking.extract_frames`:

* it resolves input videos through ``Dataset.resolve_media_scope``, routing
  each entry by its transcode verdict (an analysis-required entry tracks its
  constant-rate analysis derivative, not the defective original);
* computes a content-addressed ``run_id = "trex-<hash(settings)>"`` and writes
  run-addressed artifacts under ``<trex_root>/<run_id>/<group>__<seq>/``;
* records the attempt in its JSONL run-log (``kind="trex"``, under
  ``<dataset_root>/.mosaic/runs/``), reports coarse convert/track phase progress,
  and is cancellable (the subprocess runs in a killable process group);
* bridges the per-individual NPZ outputs into standardized
  ``tracks/<group>__<seq>.parquet`` via the registered ``trex_npz`` converter.

The ``run_id`` addresses the *tracking settings* (detect model + thresholds,
track params, calibration); the input video identity is captured per entry in
the index (``video_abs_path``). This matches the ``extract_frames`` precedent;
folding a full video content-hash into the ``run_id`` is a later refinement that
needs no schema change.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mosaic.core.helpers import make_entry_key, to_safe_name
from mosaic.core.pipeline._utils import hash_params, json_ready
from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase
from mosaic.core.pipeline.job import Cancelled, CancelToken, JobContext, job_context
from mosaic.core.pipeline.subprocess_util import ProcessCancelled
from mosaic.core.schema import ensure_track_schema

from .run import run_trex_convert, run_trex_track

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.progress import ProgressCallback


# --- TREx run index -------------------------------------------------------


def trex_run_root(ds: Dataset, run_id: str) -> Path:
    return ds.get_root("trex") / run_id


def trex_index_path(ds: Dataset) -> Path:
    return ds.get_root("trex") / "index.csv"


@dataclass(frozen=True, slots=True)
class TRexIndexRow(RunIndexRowBase):
    """Typed row for the TREx run index CSV."""

    group: str
    sequence: str
    video_abs_path: str
    params_hash: str
    n_individuals: int = 0
    pv_path: str = ""


def trex_index(path: Path) -> IndexCSV[TRexIndexRow]:
    return IndexCSV(path, TRexIndexRow, dedup_keys=["run_id", "group", "sequence"])


# --- NPZ -> standardized tracks bridge ------------------------------------


def _bridge_npz_to_tracks(
    ds: Dataset,
    group: str,
    sequence: str,
    npz_paths: list[Path],
    *,
    overwrite: bool,
) -> int | None:
    """Merge per-individual TREx NPZ into ``tracks/<group>__<seq>.parquet``.

    Reuses the registered ``trex_npz`` converter and mirrors the merge that
    ``Dataset.convert_all_tracks`` performs, but with the authoritative
    (group, sequence) known from the media index (no filename guessing).
    Returns the row count written, or ``None`` if skipped/failed.
    """
    from mosaic.core.dataset import TRACK_CONVERTERS  # lazy: avoids import cycle

    if not npz_paths:
        return None

    out_path = ds.get_root("tracks") / f"{make_entry_key(group, sequence)}.parquet"
    if out_path.exists() and not overwrite:
        return None

    converter = TRACK_CONVERTERS["trex_npz"]
    dfs: list[pd.DataFrame] = []
    for npz in npz_paths:
        try:
            dfs.append(converter(npz, {"group": group, "sequence": sequence}))
        except Exception as exc:
            print(
                f"[run_trex] convert failed for {npz}: {exc}; "
                f"skipping ({group}, {sequence})",
                file=sys.stderr,
            )
            return None
    if not dfs:
        return None

    all_cols = sorted(set().union(*[set(d.columns) for d in dfs]))
    aligned = []
    for d in dfs:
        for mc in [c for c in all_cols if c not in d.columns]:
            d[mc] = np.nan
        aligned.append(d[all_cols])
    merged = pd.concat(aligned, ignore_index=True)
    ensure_track_schema(merged, "trex_v1", strict=False, source=f"{group}/{sequence}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    ds._write_tracks_index_row(
        {
            "group": group,
            "sequence": sequence,
            "group_safe": to_safe_name(group) if group else "",
            "sequence_safe": to_safe_name(sequence),
            "collection": "",
            "collection_safe": "",
            "abs_path": ds._relative_to_root(out_path),
            "std_format": "trex_v1",
            "source_abs_path": str(npz_paths[0].parent),
            "source_md5": "",
            "n_rows": int(len(merged)),
        }
    )
    return int(len(merged))


# --- Public entry point ---------------------------------------------------


def run_trex(
    ds: Dataset,
    *,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    entries: Iterable[tuple[str, str]] | None = None,
    # detection / conversion
    detect_model: Path | str | None = None,
    detect_type: str = "yolo",
    detect_conf_threshold: float = 0.5,
    detect_iou_threshold: float = 0.1,
    cm_per_pixel: float = 1.0,
    meta_encoding: str = "gray",
    convert_extra_settings: dict[str, Any] | None = None,
    # tracking
    track_max_individuals: int = 1,
    track_max_speed: float = 80.0,
    track_max_reassign_time: float = 2.0,
    track_trusted_probability: float = 0.1,
    analysis_range: tuple[int, int] | None = None,
    visual_identification_model_path: Path | str | None = None,
    auto_train: bool = False,
    track_extra_settings: dict[str, Any] | None = None,
    # execution
    timeout: int = 600,
    trex_conda_env: str | None = None,
    trex_bin: Path | str | None = None,
    display: str | None = None,
    overwrite: bool = False,
    convert_to_tracks: bool = True,
    # Job Contract
    execution_id: str | None = None,
    owner: str = "",
    track: bool = True,
    progress_callback: ProgressCallback | None = None,
    cancel_token: CancelToken | None = None,
    # When set, run inside this already-open JobContext instead of opening one -- the
    # ``mosaic run --kind trex`` path (``TrexOp``) hands its ctx here so TREx rides the
    # standard tracking-op runner without double-wrapping the Job Contract. The
    # standalone/CLI path leaves it None; execution_id/owner/track/callbacks then open one.
    ctx: JobContext | None = None,
) -> str:
    """Run TREx (convert + track) over scoped videos as a tracked job.

    Parameters mirror :func:`mosaic.tracking.trex.run_trex_convert` /
    :func:`~mosaic.tracking.trex.run_trex_track`, plus scope
    (``groups``/``sequences``/``entries``) and the Job-Contract knobs
    (``execution_id``/``owner``/``track``/``progress_callback``/``cancel_token``).

    Returns the content-addressed ``run_id``.
    """
    if not ds.has_root("trex"):
        ds.set_root("trex", "tracks_raw/trex")

    # Settings that define the tracking result -> the content hash.
    settings = {
        "detect_model": str(detect_model) if detect_model is not None else None,
        "detect_type": detect_type,
        "detect_conf_threshold": detect_conf_threshold,
        "detect_iou_threshold": detect_iou_threshold,
        "cm_per_pixel": cm_per_pixel,
        "meta_encoding": meta_encoding,
        "convert_extra_settings": convert_extra_settings,
        "track_max_individuals": track_max_individuals,
        "track_max_speed": track_max_speed,
        "track_max_reassign_time": track_max_reassign_time,
        "track_trusted_probability": track_trusted_probability,
        "analysis_range": list(analysis_range) if analysis_range else None,
        "visual_identification_model_path": (
            str(visual_identification_model_path)
            if visual_identification_model_path is not None
            else None
        ),
        "auto_train": auto_train,
        "track_extra_settings": track_extra_settings,
    }
    params_hash = hash_params(settings)
    run_id = f"trex-{params_hash}"
    run_root = trex_run_root(ds, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    # Resolve a training run_id (e.g. "train-points-<hash>") to its best.pt weights for the
    # trex invocation -- the train->track handoff. The run_id hash above intentionally keys on
    # the original reference (portable across machines), not the resolved absolute path.
    detect_model_exec: Path | str | None = detect_model
    if detect_model is not None and not Path(str(detect_model)).exists():
        from mosaic.tracking.model_refs import resolve_model

        ref = str(detect_model)
        model_kind = ref.rsplit("-", 1)[0] if ref.count("-") >= 2 else "train-points"
        try:
            detect_model_exec, _ = resolve_model(ds, ref, model_kind)
        except (FileNotFoundError, KeyError):
            detect_model_exec = (
                detect_model  # let TREx surface a clear "not found" error
            )

    params_path = run_root / "run_params.json"
    try:
        params_path.write_text(json.dumps(json_ready(settings), indent=2))
    except Exception as exc:
        print(f"[run_trex] failed to save run_params.json: {exc}", file=sys.stderr)

    # Route each scoped entry through the transcode verdict: a clean entry
    # resolves to its original, an analysis-required entry to its constant-rate
    # analysis derivative (so tracks land in the same frame space as the rest of
    # the pipeline), and a required-but-unlinked entry raises MediaProbeError
    # here -- before any TREx subprocess opens a known-defective original. TREx
    # decodes the file itself, so only the routed path is needed, not the facts.
    scope = ds.resolve_media_scope(groups, sequences, entries)
    if not scope:
        print("[run_trex] No media entries match the given scope.", file=sys.stderr)
        return run_id

    # One work item per (group, sequence); first video when several exist.
    work_items: list[tuple[str, str, Path]] = []
    for group, sequence, resolved in scope:
        paths = resolved.paths
        if len(paths) > 1:
            print(
                f"[run_trex] ({group}, {sequence}) has {len(paths)} videos; using "
                f"the first ({paths[0].name}). Multi-video sequences are not yet "
                f"merged.",
                file=sys.stderr,
            )
        work_items.append((group, sequence, paths[0]))

    idx = trex_index(trex_index_path(ds))
    idx.ensure()

    index_rows: list[TRexIndexRow] = []
    # Reuse a caller-provided context (TrexOp / run_op) or open our own.
    managed: AbstractContextManager[JobContext] = (
        nullcontext(ctx)
        if ctx is not None
        else job_context(
            ds,
            kind="trex",
            target="trex-track",
            execution_id=execution_id,
            owner=owner,
            track=track,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
        )
    )
    with managed as ctx:
        ctx.set_run_id(run_id)
        ctx.set_total(len(work_items))
        cancel_check = ctx.cancel_token.is_cancelled

        try:
            for i, (group, sequence, video_path) in enumerate(work_items):
                ctx.check_cancel()
                key = make_entry_key(group, sequence)
                ctx.progress.on_entry_start(i, len(work_items), key)
                seq_dir = run_root / key

                if not (seq_dir.exists() and not overwrite):
                    if overwrite and seq_dir.exists():
                        import shutil

                        shutil.rmtree(seq_dir)
                    seq_dir.mkdir(parents=True, exist_ok=True)
                    ctx.progress.on_phase("convert", key)
                    convert_result = run_trex_convert(
                        video_path,
                        seq_dir,
                        detect_model=detect_model_exec,
                        detect_type=detect_type,
                        detect_conf_threshold=detect_conf_threshold,
                        detect_iou_threshold=detect_iou_threshold,
                        track_max_individuals=track_max_individuals,
                        cm_per_pixel=cm_per_pixel,
                        meta_encoding=meta_encoding,
                        extra_settings=convert_extra_settings,
                        timeout=timeout,
                        trex_conda_env=trex_conda_env,
                        trex_bin=trex_bin,
                        display=display,
                        cancel_check=cancel_check,
                    )
                    ctx.progress.on_phase("track", key)
                    run_trex_track(
                        convert_result.pv_path,
                        seq_dir,
                        track_max_individuals=track_max_individuals,
                        track_max_speed=track_max_speed,
                        track_max_reassign_time=track_max_reassign_time,
                        track_trusted_probability=track_trusted_probability,
                        analysis_range=analysis_range,
                        visual_identification_model_path=visual_identification_model_path,
                        auto_train=auto_train,
                        extra_settings=track_extra_settings,
                        timeout=timeout,
                        trex_conda_env=trex_conda_env,
                        trex_bin=trex_bin,
                        display=display,
                        cancel_check=cancel_check,
                    )

                data_dir = seq_dir / "data"
                npz_paths = sorted(data_dir.glob("*.npz")) if data_dir.is_dir() else []
                pv_matches = sorted(seq_dir.glob("*.pv"))
                index_rows.append(
                    TRexIndexRow(
                        run_id=run_id,
                        group=group,
                        sequence=sequence,
                        abs_path=Path(ds.relative_to_root(seq_dir)),
                        video_abs_path=str(video_path),
                        params_hash=params_hash,
                        n_individuals=len(npz_paths),
                        pv_path=(
                            ds.relative_to_root(pv_matches[0]) if pv_matches else ""
                        ),
                    )
                )

                if convert_to_tracks:
                    _bridge_npz_to_tracks(
                        ds, group, sequence, npz_paths, overwrite=overwrite
                    )

                ctx.progress.on_entry_end(i + 1, len(work_items), key)
                ctx.heartbeat(i + 1)
        except ProcessCancelled as exc:
            # A killed TREx subprocess -> mark the attempt cancelled.
            raise Cancelled() from exc
        finally:
            if index_rows:
                idx.append(index_rows)
                idx.mark_finished(run_id)

    print(
        f"[run_trex] completed run_id={run_id} "
        f"({len(index_rows)}/{len(work_items)} sequences) -> {run_root}"
    )
    return run_id


def list_trex_runs(ds: Dataset) -> pd.DataFrame:
    """List TREx runs tracked in the trex index."""
    if not ds.has_root("trex"):
        return pd.DataFrame(columns=[f.name for f in dataclasses.fields(TRexIndexRow)])
    idx_path = trex_index_path(ds)
    if not idx_path.exists():
        return pd.DataFrame(columns=[f.name for f in dataclasses.fields(TRexIndexRow)])
    return pd.read_csv(idx_path)
