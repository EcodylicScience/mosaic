"""Dataset-level TREx runs: content-addressed, tracked, tracks-integrated.

``run_trex(ds, ...)`` is the first-class entry point that turns the standalone
TREx CLI wrappers (:mod:`mosaic.tracking.trex.run`) into a Job-Contract stage,
mirroring :func:`mosaic.tracking.extract_frames`:

* it resolves input videos from ``media/index.csv``;
* computes a content-addressed ``run_id = "trex-<hash(settings)>"`` and writes
  run-addressed artifacts under ``<trex_root>/<run_id>/<group>__<seq>/``;
* records the attempt in ``.mosaic.db`` (``runs`` row, ``kind="trex"``), reports
  coarse convert/track phase progress, and is cancellable (the subprocess runs
  in a killable process group);
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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mosaic.core.helpers import make_entry_key, to_safe_name
from mosaic.core.pipeline._utils import hash_params, json_ready
from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase
from mosaic.core.pipeline.job import Cancelled, job_context
from mosaic.core.pipeline.subprocess_util import ProcessCancelled
from mosaic.core.schema import ensure_track_schema

from .run import run_trex_convert, run_trex_track

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


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


# --- media resolution -----------------------------------------------------


def _list_media(
    ds: Dataset,
    groups: Iterable[str] | None,
    sequences: Iterable[str] | None,
    entries: Iterable[tuple[str, str]] | None,
) -> pd.DataFrame:
    """Return the scoped media rows (group, sequence, abs_path, video_order)."""
    media_key = ds.resolve_media_root()
    idx_path = ds.get_root(media_key) / "index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"{media_key}/index.csv not found; run index_media() first."
        )
    df = pd.read_csv(idx_path)
    df["group"] = df["group"].fillna("").astype(str)
    df["sequence"] = df["sequence"].fillna("").astype(str)
    if "video_order" not in df.columns:
        df["video_order"] = 0
    else:
        df["video_order"] = df["video_order"].fillna(0).astype(int)

    mask = pd.Series(True, index=df.index)
    if groups is not None:
        mask &= df["group"].isin({str(g) for g in groups})
    if sequences is not None:
        mask &= df["sequence"].isin({str(s) for s in sequences})
    if entries is not None:
        wanted = {(str(g), str(s)) for g, s in entries}
        mask &= df.apply(lambda r: (r["group"], r["sequence"]) in wanted, axis=1)
    return df[mask].reset_index(drop=True)


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
    progress_callback=None,
    cancel_token=None,
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

    params_path = run_root / "run_params.json"
    try:
        params_path.write_text(json.dumps(json_ready(settings), indent=2))
    except Exception as exc:
        print(f"[run_trex] failed to save run_params.json: {exc}", file=sys.stderr)

    media_df = _list_media(ds, groups, sequences, entries)
    if media_df.empty:
        print("[run_trex] No media entries match the given scope.", file=sys.stderr)
        return run_id

    # One work item per (group, sequence); first video when several exist.
    work_items: list[tuple[str, str, Path]] = []
    for (g, s), sub in media_df.groupby(["group", "sequence"]):
        g, s = str(g), str(s)
        sub = sub.sort_values("video_order")
        paths = [ds.resolve_path(r["abs_path"]) for _, r in sub.iterrows()]
        if len(paths) > 1:
            print(
                f"[run_trex] ({g}, {s}) has {len(paths)} videos; using the first "
                f"({paths[0].name}). Multi-video sequences are not yet merged.",
                file=sys.stderr,
            )
        if not g and not to_safe_name(s):
            s = paths[0].stem
        work_items.append((g, s, paths[0]))

    idx = trex_index(trex_index_path(ds))
    idx.ensure()

    index_rows: list[TRexIndexRow] = []
    with job_context(
        ds,
        kind="trex",
        target="trex-track",
        execution_id=execution_id,
        owner=owner,
        track=track,
        progress_callback=progress_callback,
        cancel_token=cancel_token,
        total=len(work_items),
    ) as ctx:
        ctx.set_run_id(run_id)
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
                        detect_model=detect_model,
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
                        abs_path=seq_dir,
                        video_abs_path=str(video_path),
                        params_hash=params_hash,
                        n_individuals=len(npz_paths),
                        pv_path=str(pv_matches[0]) if pv_matches else "",
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
