"""Model-inference tracking ops: pose, points (POLO), localizer.

Each op runs a trained model over the scoped videos, writes raw per-video
predictions under ``_tracking/<infer-kind>/<run_id>/``, and (by default) bridges
the resulting DataFrame into standardized ``tracks/<group>__<seq>.parquet`` -- the
extract -> train -> infer -> tracks loop. The ``model`` param is a weights path
OR a prior training ``run_id`` (resolved via the model index). Heavy backends
import lazily inside ``run()``.

**The parquet is an audit artifact, not a cache** (item 8.7). Inference re-runs
unconditionally: what the file is for is showing what a detector emitted *before*
schema coercion, which is what you want when debugging a bad model. It has no
index of its own -- it used to, and that index was written, never read, and
reached by no portability pass, so its one unique column was silently wrong on
any moved dataset. The edge from a tracks table back to the run that produced it
is ``producer`` / ``producer_run_id`` on the tracks row, with this directory as
its ``source_abs_path``.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Final

import pandas as pd
from mosaic_media import MediaFacts

from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.identity_scheme import write_identity_scheme
from mosaic.core.pipeline.markers import (
    PhaseMarker,
    clear_inflight,
    inflight_state,
    new_inflight,
    read_inflight,
    write_inflight,
    write_phase_marker,
)
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME, op_run_id
from mosaic.core.pipeline.tracking_roots import tracking_root_default
from mosaic.core.pipeline.tracks_identity import (
    infer_variant_payload,
    tracks_run_id,
    tracks_variant_root,
    write_tracks_variant,
)
from mosaic.core.pipeline.tracks_index import consumed_roots_for, write_tracks_row
from mosaic.core.pipeline.types import HASH_EXCLUDE, Params
from mosaic.core.pipeline.ops import Op, register_op
from mosaic.core.schema import ensure_track_schema
from mosaic.runlog import now_iso
from mosaic.tracking.model_refs import resolve_model
from mosaic.core.pipeline.writers import write_parquet_atomic

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


# --- Where inference output lands ----------------------------------------


def infer_run_root(ds: Dataset, kind: str, run_id: str) -> Path:
    """``_tracking/<infer-kind>/<run_id>/`` -- item 8.7's relocation.

    Formerly ``predictions/<kind>/<run_id>/`` under a root of its own. That root
    was written and never read: no resolver, no CLI command, no reindex, prune or
    portability pass, and no reference in any sibling repository. Its index was
    not merely unread but unmaintainable -- ``video_abs_path`` was stored
    absolute and reached by no portability pass, so it was silently wrong on
    every moved or synced dataset.

    Nothing of value went with it. Every column of the deleted row survives
    elsewhere: the run identifier and the entry as ``producer_run_id`` and
    ``group``/``sequence`` on the tracks row, the model as ``params.model`` in
    ``tracks/<variant>/params.json`` (a superset -- it carries a weights digest
    where the old column carried ``""`` for bare-path weights), the frame range
    in the same params blob, ``n_rows`` on the tracks row, and this directory as
    its ``source_abs_path``. Only ``video_abs_path`` had no survivor, and it was
    the wrong one.

    The parquet stays as an audit artifact under the shorter retention window of
    item 8.4: it is what shows what a detector emitted before schema coercion,
    which is exactly what a bad model needs to be diagnosed against.
    """
    return ds.get_root(kind) / run_id


# The inactivity bound for one video's inference. Generous, because a large model
# over a long video can be silent for a long time and a claim that expires under
# a live run invites a second execution into the same directory. The sweeper
# reads the expiry the claim carries rather than this constant (item 8.3).
_INFER_IDLE_SECONDS: Final = 3600.0


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


def infer_run_id(kind: str, version: str, params: Params, model_id: str) -> str:
    """Mint an inference run identifier.

    Args:
        kind: The op kind, e.g. ``"infer-points"``.
        version: The op's declared version -- a visible segment, not hashed.
        params: Op params; only ``identity_dump()`` enters the digest.
        model_id: The training run that produced the weights, or a digest of
            the weights path when they were given as a bare path. The model is
            what determined the predictions, so leaving it out would let two
            detectors share one identifier.
    """
    return op_run_id(
        kind, version, {"params": params.identity_dump(), "model": model_id}
    )


def _bridge_df_to_tracks(
    ds: Dataset,
    df: pd.DataFrame | None,
    group: str,
    sequence: str,
    *,
    tracks_variant: str,
    producer_run_id: str,
    kind: str,
    seq_dir: Path,
    video_path: Path,
    model_pt: Path,
    overwrite: bool,
) -> int:
    """Write an inference DataFrame as a standardized ``tracks/`` parquet.

    ``tracks_variant`` names the directory as well as the row, so two models (or
    two parameter sets) no longer target one path.
    """
    if df is None or df.empty:
        return 0
    variant_root = tracks_variant_root(ds.get_root("tracks"), tracks_variant)
    out_path = variant_root / f"{make_entry_key(group, sequence)}.parquet"
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
    _ = write_parquet_atomic(df, out_path)
    # source_abs_path was empty here, because the frame is built in memory and
    # there is no raw file. It now points at the prediction directory this run
    # wrote -- the row-level pointer from a tracks table back to the predictions
    # that produced it, which is what item 8.7 needs to retire that root.
    write_tracks_row(
        ds,
        run_id=tracks_variant,
        group=group,
        sequence=sequence,
        out_path=out_path,
        producer=kind,
        std_format="trex_v1",
        n_rows=int(len(df)),
        producer_run_id=producer_run_id,
        source=seq_dir,
        consumed_source_roots=consumed_roots_for(ds, [video_path, model_pt]),
    )
    return int(len(df))


def _run_inference_op(
    ds: Dataset,
    params: _InferParamsBase,
    ctx: JobContext,
    *,
    kind: str,
    version: str,
    train_kind: str,
    per_video: Callable[[str, Path, Path, MediaFacts | None], pd.DataFrame | None],
) -> str:
    """Shared scaffold: resolve model, loop scoped videos, predict, bridge."""
    if not ds.has_root(kind):
        ds.set_root(kind, tracking_root_default(kind))

    model = resolve_model(ds, params.model, train_kind)
    # The training run when there is one, the weights' digest otherwise -- never
    # the path. Hashing the path meant swapping best.pt in place reused the same
    # identifier, and moving unchanged weights minted a new one; both were wrong
    # in the direction that reports a cache hit over the wrong model.
    model_id = model.model_id
    run_id = infer_run_id(kind, version, params, model_id)
    ctx.set_run_id(run_id)

    # The tracks variant these predictions will be bridged into, minted once for
    # the whole run and recorded beside the tables. Same payload as the op run
    # above, so the two identifiers coincide -- which is what makes
    # ``predictions/<kind>/<run_id>/`` and the tracks variant obviously the same
    # run at a glance, while the index still keeps them in separate columns.
    tracks_variant = tracks_run_id(
        kind, version, infer_variant_payload(params.identity_dump(), model_id)
    )
    _ = write_tracks_variant(
        ds.get_root("tracks"),
        tracks_variant,
        kind,
        version,
        infer_variant_payload(params.identity_dump(), model_id),
    )

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
        facts = resolved.facts[0]
        work.append((group, sequence, resolved.paths[0], facts))

    ctx.set_total(len(work))
    run_root = infer_run_root(ds, kind, run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    write_identity_scheme(run_root, OP_IDENTITY_SCHEME)

    done = 0
    for i, (group, sequence, video_path, facts) in enumerate(work):
        ctx.check_cancel()
        key = make_entry_key(group, sequence)
        ctx.progress.on_entry_start(i, len(work), key)
        ctx.progress.on_phase("infer", key)

        seq_dir = run_root / key
        seq_dir.mkdir(parents=True, exist_ok=True)

        # A claim, not a cache. Inference still re-infers unconditionally -- the
        # completion marker below records that output is whole, and nothing gates
        # on it, because turning this into a cache is a behaviour change with its
        # own failure mode (a silently skipped re-run over a corrected video) and
        # is not what item 8.7 asked for. What the claim prevents is two
        # executions writing one ``predictions.parquet`` at once, which is the
        # guard the three trackers already have and this path did not.
        if (
            inflight_state(
                read_inflight(seq_dir),
                run_log_base=ds.base_dir,
                execution_id=ctx.execution_id,
            )
            == "live"
        ):
            print(
                f"[{kind}] ({group}, {sequence}) is held by another execution; "
                "skipping it.",
                file=sys.stderr,
            )
            ctx.progress.on_entry_end(i + 1, len(work), key)
            continue

        write_inflight(
            seq_dir,
            new_inflight(
                execution_id=ctx.execution_id,
                host=socket.gethostname(),
                pid=os.getpid(),
                phase="infer",
                idle_seconds=_INFER_IDLE_SECONDS,
            ),
        )
        try:
            df = per_video(str(model.path), video_path, seq_dir, facts)
            pred_path = seq_dir / "predictions.parquet"
            if df is not None and not df.empty:
                _ = write_parquet_atomic(df, pred_path)

            if params.convert_to_tracks and df is not None and not df.empty:
                ctx.progress.on_phase("bridge", key)
                _ = _bridge_df_to_tracks(
                    ds,
                    df,
                    group,
                    sequence,
                    tracks_variant=tracks_variant,
                    producer_run_id=run_id,
                    kind=kind,
                    seq_dir=seq_dir,
                    video_path=video_path,
                    model_pt=model.path,
                    overwrite=params.overwrite,
                )

            # Written after the bridge, not after the parquet. A directory whose
            # output has not reached ``tracks/`` is not finished, and the sweeper
            # reads this marker to decide what may be reclaimed -- so marking it
            # complete a moment early is how an unbridged run gets deleted.
            write_phase_marker(
                seq_dir,
                PhaseMarker(
                    phase="infer",
                    run_id=run_id,
                    execution_id=ctx.execution_id,
                    completed_at=now_iso(),
                    source=str(ds.relative_to_root(video_path)),
                    source_uid=facts.video_uuid if facts else "",
                    recorded_output=str(ds.relative_to_root(pred_path))
                    if pred_path.exists()
                    else "",
                ),
            )
            done += 1
        finally:
            # Released whatever happened, including a cancel: a claim outliving
            # its process is what makes the next run read a dead directory as
            # busy, and only its expiry would ever free it.
            clear_inflight(seq_dir)

        ctx.progress.on_entry_end(i + 1, len(work), key)
        ctx.heartbeat(i + 1)

    print(f"[{kind}] completed run_id={run_id} ({done}/{len(work)}) -> {run_root}")
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
            version=self.version,
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
            version=self.version,
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
            version=self.version,
            train_kind="train-localizer",
            per_video=per_video,
        )
