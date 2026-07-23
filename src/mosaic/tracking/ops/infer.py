"""Model-inference tracking ops: pose, points (POLO), localizer.

Each op runs a trained model over the scoped videos, writes raw per-video
predictions under ``predictions/<kind>/<run_id>/``, and (by default) bridges the
resulting DataFrame into standardized ``tracks/<group>__<seq>.parquet`` -- the
extract -> train -> infer -> tracks loop. The ``model`` param is a weights path
OR a prior training ``run_id`` (resolved via the model index). Heavy backends
import lazily inside ``run()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable

import pandas as pd
from mosaic_media import MediaFacts

from mosaic.core.helpers import make_entry_key, to_safe_name
from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.types import HASH_EXCLUDE, Params
from mosaic.core.pipeline.ops import Op, register_op
from mosaic.core.schema import ensure_track_schema
from mosaic.tracking.model_refs import resolve_model

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


# --- Inference index -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class InferenceIndexRow(RunIndexRowBase):
    """Typed row for an inference index CSV (``predictions/<kind>/index.csv``)."""

    model_run_id: str
    group: str
    sequence: str
    video_abs_path: str
    start_frame: int
    end_frame: int
    n_rows: int


def inference_index(path: Path) -> IndexCSV[InferenceIndexRow]:
    return IndexCSV(path, InferenceIndexRow, dedup_keys=["run_id", "group", "sequence"])


def prediction_run_root(ds: Dataset, kind: str, run_id: str) -> Path:
    return ds.get_root("predictions") / kind / run_id


def prediction_index_path(ds: Dataset, kind: str) -> Path:
    return ds.get_root("predictions") / kind / "index.csv"


# --- Params --------------------------------------------------------------


class _InferParamsBase(Params):
    model: str  # weights path OR a prior training run_id
    conf_threshold: float = 0.25
    imgsz: int = 640
    frame_step: int = 1
    start_frame: int = 0
    end_frame: int | None = None
    max_frames: int | None = None
    convert_to_tracks: bool = True
    overwrite: Annotated[bool, HASH_EXCLUDE] = False
    groups: Annotated[list[str] | None, HASH_EXCLUDE] = None
    sequences: Annotated[list[str] | None, HASH_EXCLUDE] = None
    device: Annotated[str, HASH_EXCLUDE] = "0"
    batch_size: Annotated[int, HASH_EXCLUDE] = 8
    save_images: Annotated[bool, HASH_EXCLUDE] = False


class PoseInferParams(_InferParamsBase):
    pass


class PointInferParams(_InferParamsBase):
    dor: float = 0.8


class LocalizerInferParams(_InferParamsBase):
    num_classes: int = 4
    initial_channels: int = 32
    thresholds: float = 0.5


# --- Shared machinery ----------------------------------------------------


def _bridge_df_to_tracks(
    ds: Dataset,
    df: pd.DataFrame | None,
    group: str,
    sequence: str,
    *,
    overwrite: bool,
) -> int:
    """Write an inference DataFrame as a standardized ``tracks/`` parquet."""
    if df is None or df.empty:
        return 0
    out_path = ds.get_root("tracks") / f"{make_entry_key(group, sequence)}.parquet"
    if out_path.exists() and not overwrite:
        return 0
    df = df.copy()
    df["group"] = group
    df["sequence"] = sequence
    if "id" not in df.columns:
        df["id"] = 0
    if "time" not in df.columns:
        df["time"] = df["frame"] if "frame" in df.columns else range(len(df))
    ensure_track_schema(df, "trex_v1", strict=False, source=f"{group}/{sequence}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
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
            "source_abs_path": "",
            "source_md5": "",
            "n_rows": int(len(df)),
        }
    )
    return int(len(df))


def _run_inference_op(
    ds: Dataset,
    params: _InferParamsBase,
    ctx: JobContext,
    *,
    kind: str,
    train_kind: str,
    per_video: Callable[[str, Path, Path, MediaFacts | None], pd.DataFrame | None],
) -> str:
    """Shared scaffold: resolve model, loop scoped videos, predict, bridge."""
    if not ds.has_root("predictions"):
        ds.set_root("predictions", "predictions")

    model_pt, base_run_id = resolve_model(ds, params.model, train_kind)
    model_id = base_run_id or hash_params({"path": str(model_pt)})
    run_id = "{}-{}".format(
        kind, hash_params({"params": params.identity_dump(), "model": model_id})
    )
    ctx.set_run_id(run_id)

    scope = ds.resolve_media_scope(params.groups, params.sequences)
    if not scope:
        print(f"[{kind}] No media entries match the given scope.")
        return run_id

    work: list[tuple[str, str, Path, MediaFacts | None]] = []
    for entry in scope:
        # Each scope entry is one camera; per-camera inference output pathing +
        # index dedup is Phase 2, so single-camera behavior here is unchanged.
        # The op reads the first path; carry the routed entry's stored facts (the
        # analysis derivative's when the verdict routed there, else the
        # original's) so the reader injects them instead of re-probing. A
        # required-but-unlinked entry already raised in resolve_media_scope,
        # before any defective original was opened.
        group, sequence, resolved = entry.group, entry.sequence, entry.resolved
        facts = resolved.facts[0] if resolved.facts is not None else None
        work.append((group, sequence, resolved.paths[0], facts))

    ctx.set_total(len(work))
    run_root = prediction_run_root(ds, kind, run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    idx = inference_index(prediction_index_path(ds, kind))
    idx.ensure()
    rows: list[InferenceIndexRow] = []

    try:
        for i, (group, sequence, video_path, facts) in enumerate(work):
            ctx.check_cancel()
            key = make_entry_key(group, sequence)
            ctx.progress.on_entry_start(i, len(work), key)
            ctx.progress.on_phase("infer", key)

            seq_dir = run_root / key
            seq_dir.mkdir(parents=True, exist_ok=True)
            df = per_video(str(model_pt), video_path, seq_dir, facts)
            pred_path = seq_dir / "predictions.parquet"
            if df is not None and not df.empty:
                df.to_parquet(pred_path, index=False)

            n_rows = 0
            if params.convert_to_tracks and df is not None and not df.empty:
                ctx.progress.on_phase("bridge", key)
                n_rows = _bridge_df_to_tracks(
                    ds, df, group, sequence, overwrite=params.overwrite
                )

            rows.append(
                InferenceIndexRow(
                    run_id=run_id,
                    model_run_id=base_run_id,
                    group=group,
                    sequence=sequence,
                    abs_path=Path(ds.relative_to_root(seq_dir)),
                    video_abs_path=str(video_path),
                    start_frame=int(params.start_frame),
                    end_frame=int(params.end_frame)
                    if params.end_frame is not None
                    else -1,
                    n_rows=n_rows,
                )
            )
            ctx.progress.on_entry_end(i + 1, len(work), key)
            ctx.heartbeat(i + 1)
    finally:
        if rows:
            idx.append(rows)
            idx.mark_finished(run_id)

    print(f"[{kind}] completed run_id={run_id} ({len(rows)}/{len(work)}) -> {run_root}")
    return run_id


# --- Ops -----------------------------------------------------------------


@register_op
class InferPoseOp(Op[PoseInferParams]):
    kind = "infer-pose"
    category = "infer"
    domain = "tracking"
    version = "0.1"
    Params = PoseInferParams

    def run(self, ds: Dataset, params: PoseInferParams, ctx: JobContext) -> str:
        from mosaic.tracking.pose_training.inference import (
            inference_to_dataframe,
            run_inference,
        )

        def per_video(
            model: str, video: Path, out_dir: Path, facts: MediaFacts | None
        ) -> pd.DataFrame:
            results = run_inference(
                model,
                video,
                output_dir=out_dir if params.save_images else None,
                start_frame=params.start_frame,
                end_frame=params.end_frame,
                frame_step=params.frame_step,
                conf_threshold=params.conf_threshold,
                max_frames=params.max_frames,
                device=params.device,
                save_images=params.save_images,
                imgsz=params.imgsz,
                batch_size=params.batch_size,
                verbose=False,
                facts=facts,
            )
            return inference_to_dataframe(results)

        return _run_inference_op(
            ds,
            params,
            ctx,
            kind=self.kind,
            train_kind="train-pose",
            per_video=per_video,
        )


@register_op
class InferPointsOp(Op[PointInferParams]):
    kind = "infer-points"
    category = "infer"
    domain = "tracking"
    version = "0.1"
    Params = PointInferParams

    def run(self, ds: Dataset, params: PointInferParams, ctx: JobContext) -> str:
        from mosaic.tracking.pose_training.inference import (
            locations_to_dataframe,
            run_point_inference,
        )

        def per_video(
            model: str, video: Path, out_dir: Path, facts: MediaFacts | None
        ) -> pd.DataFrame:
            results = run_point_inference(
                model,
                video,
                output_dir=out_dir if params.save_images else None,
                start_frame=params.start_frame,
                end_frame=params.end_frame,
                frame_step=params.frame_step,
                conf_threshold=params.conf_threshold,
                dor=params.dor,
                max_frames=params.max_frames,
                device=params.device,
                save_images=params.save_images,
                imgsz=params.imgsz,
                batch_size=params.batch_size,
                verbose=False,
                facts=facts,
            )
            return locations_to_dataframe(results)

        return _run_inference_op(
            ds,
            params,
            ctx,
            kind=self.kind,
            train_kind="train-points",
            per_video=per_video,
        )


@register_op
class InferLocalizerOp(Op[LocalizerInferParams]):
    kind = "infer-localizer"
    category = "infer"
    domain = "tracking"
    version = "0.1"
    Params = LocalizerInferParams

    def run(self, ds: Dataset, params: LocalizerInferParams, ctx: JobContext) -> str:
        from mosaic.tracking.pose_training.localizer_inference import (
            localizer_detections_to_dataframe,
            run_localizer_inference,
        )

        def per_video(
            model: str, video: Path, out_dir: Path, facts: MediaFacts | None
        ) -> pd.DataFrame:
            detections = run_localizer_inference(
                model,
                video,
                output_dir=out_dir if params.save_images else None,
                num_classes=params.num_classes,
                initial_channels=params.initial_channels,
                thresholds=params.thresholds,
                start_frame=params.start_frame,
                end_frame=params.end_frame,
                frame_step=params.frame_step,
                max_frames=params.max_frames,
                device=params.device,
                save_images=params.save_images,
                facts=facts,
            )
            return localizer_detections_to_dataframe(detections)

        return _run_inference_op(
            ds,
            params,
            ctx,
            kind=self.kind,
            train_kind="train-localizer",
            per_video=per_video,
        )
