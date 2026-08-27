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

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Final, TypeAlias

import pandas as pd
from mosaic_media import MediaFacts

from mosaic.core.helpers import make_entry_key
from mosaic.core.media.read_target import verified_read_facts
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.identity_scheme import write_identity_scheme
from mosaic.core.pipeline.markers import (
    PhaseMarker,
    write_phase_marker,
)
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME, op_run_id
from mosaic.core.pipeline.tracking_roots import (
    tracking_output_schema,
    tracking_root_default,
)
from mosaic.core.pipeline.tracks_identity import (
    infer_variant_payload,
    tracks_run_id,
    tracks_variant_root,
    write_tracks_variant,
)
from mosaic.core.pipeline.tracks_index import consumed_roots_for, write_tracks_row
from mosaic.core.pipeline.types import OpParams
from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
    Params,
)
from mosaic.core.pipeline.ops import Op, OpIdentity, register_op
from mosaic.core.schema import ensure_track_schema
from mosaic.runlog import now_iso
from mosaic.tracking.common.entry import open_entry, phase_activity, release_entry
from mosaic.tracking.common.tool_input import resolve_entry_input
from mosaic.tracking.common.ultralytics_env import progress_activity
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

_PREDICTIONS_NAME: Final = "predictions.parquet"
"""What one entry's raw predictions are called, inside its working directory.

Spelled here and again in ``TRACKING_ROOTS``, which is in ``core`` and cannot
import this module."""


# --- The per-video seam ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class VideoInput:
    """One entry's video, and what running a model over it is given.

    *facts* are **gated**: the caller has already derived the read-target verdict
    and refused a file that needs an analysis transcode first. That check used to
    happen a layer down, inside ``open_frame_reader``; two of the three ops no
    longer open the video in this process at all, so it moved up here where every
    op passes through it.
    """

    video_path: Path
    work_dir: Path
    facts: MediaFacts
    on_output: Callable[[str], None]
    cancel_check: Callable[[], bool] | None


@dataclass(frozen=True, slots=True)
class VideoPredictions:
    """One video's raw predictions, and where they were published.

    ``published_path`` is ``None`` when the caller computed the frame in this
    process and the scaffold must still write it -- which is what the heatmap
    localizer does. The two Ultralytics ops run out of process and the runner
    writes the parquet itself, atomically, so re-writing it here would copy a
    whole table to the path it already occupies.
    """

    frame: pd.DataFrame | None
    published_path: Path | None = None


PerVideo: TypeAlias = Callable[[VideoInput], VideoPredictions]
"""What an op does with one entry's video."""


# --- Params --------------------------------------------------------------


_MODEL_DESCRIPTION = (
    "The weights to predict with, as a path or as the run identifier of the "
    "training run that produced them."
)

_CONF_THRESHOLD_DESCRIPTION = (
    "Minimum detection confidence a prediction must reach to be written."
)

_IMGSZ_DESCRIPTION = (
    "Longest side a frame is resized to before the model reads it. It must "
    "match what the model was trained at."
)

_FRAME_STEP_DESCRIPTION = (
    "Stride between the frames predicted on. A wider stride covers a long "
    "recording without predicting on every frame of it."
)

_START_FRAME_DESCRIPTION = "First frame to predict on, inclusive."

_END_FRAME_DESCRIPTION = (
    "Last frame to predict on, inclusive. Unset runs to the end of the video."
)

_MAX_FRAMES_DESCRIPTION = (
    "Ceiling on how many frames are predicted on per entry. Unset predicts on "
    "the whole range."
)

_CONVERT_TO_TRACKS_DESCRIPTION = (
    "Bridge the predictions into a standardized tracks table once inference "
    "finishes, instead of leaving them in the run directory alone."
)

_DEVICE_DESCRIPTION = (
    "Which accelerator the model runs on, in the tool's own spelling: a GPU "
    "index, or 'cpu'."
)

_BATCH_SIZE_DESCRIPTION = "How many frames the model reads in one forward pass."

_SAVE_IMAGES_DESCRIPTION = (
    "Write an annotated image per predicted frame beside the predictions, which "
    "is for inspecting a model rather than for any consumer downstream."
)


class _InferParamsBase(OpParams):
    """What every inference op predicts with, over the scope ``OpParams`` names.

    ``convert_to_tracks`` reaches ``identity_dump()`` where the same knob on
    :class:`~mosaic.tracking.common.params.TrackerOpParams` is ``HASH_EXCLUDE``.
    Whether the two should agree is open: bridging writes a second artifact from
    predictions that are already on disk, which argues for excluding it, and the
    exclusion would move every ``infer-*`` run identifier and every tracks
    variant minted from one.
    """

    model: Annotated[str, Declared(_MODEL_DESCRIPTION)]
    conf_threshold: Annotated[float, Declared(_CONF_THRESHOLD_DESCRIPTION)] = 0.25
    imgsz: Annotated[int, Declared(_IMGSZ_DESCRIPTION, unit="px")] = 640
    frame_step: Annotated[int, Declared(_FRAME_STEP_DESCRIPTION)] = 1
    start_frame: Annotated[int, Declared(_START_FRAME_DESCRIPTION)] = 0
    end_frame: Annotated[int | None, Declared(_END_FRAME_DESCRIPTION)] = None
    max_frames: Annotated[int | None, Declared(_MAX_FRAMES_DESCRIPTION)] = None
    convert_to_tracks: Annotated[bool, Declared(_CONVERT_TO_TRACKS_DESCRIPTION)] = True
    device: Annotated[str, HASH_EXCLUDE, Declared(_DEVICE_DESCRIPTION)] = "0"
    batch_size: Annotated[int, HASH_EXCLUDE, Declared(_BATCH_SIZE_DESCRIPTION)] = 8
    save_images: Annotated[bool, HASH_EXCLUDE, Declared(_SAVE_IMAGES_DESCRIPTION)] = (
        False
    )


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
    # Declared by the producing root, like every other tracks write path.
    std_format = tracking_output_schema(kind)
    ensure_track_schema(df, std_format, strict=False, source=f"{group}/{sequence}")
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
        std_format=std_format,
        n_rows=int(len(df)),
        producer_run_id=producer_run_id,
        source=seq_dir,
        consumed_source_roots=consumed_roots_for(ds, [video_path, model_pt]),
        # A bridge opens the entry's media, so its row records what that
        # media was. The variant identity has no term for the pixels, so
        # this cell is the only thing that notices a re-transcode.
        records_media=True,
    )
    return int(len(df))


def infer_identity(
    ds: Dataset, kind: str, version: str, params: _InferParamsBase, train_kind: str
) -> OpIdentity:
    """What an inference run with these params will be called. Writes nothing.

    The identity half of :func:`_run_inference_op`, which calls this rather than
    computing the same two identifiers a second way. A planner needs them before
    the run happens, because a feature step reading the produced tables hashes
    the *variant*, so the whole chain resolves in one walk or not at all.

    The model reference resolves through ``planned_model_id``, so an inference
    step whose weights are a training step's output in the same graph is
    resolvable before that training has run: a reference that is a run identifier
    is its own identity, which is exactly what ``resolve_model`` returns for it.
    """
    from mosaic.tracking.common.mint import planned_model_id

    model_id = planned_model_id(ds, kind, [params.model], train_kind)
    return OpIdentity(
        run_id=infer_run_id(kind, version, params, model_id),
        tracks_variant=tracks_run_id(
            kind, version, infer_variant_payload(params.identity_dump(), model_id)
        ),
    )


def _run_inference_op(
    ds: Dataset,
    params: _InferParamsBase,
    ctx: JobContext,
    *,
    kind: str,
    version: str,
    train_kind: str,
    opens_by_path: bool,
    per_video_for: Callable[[str], PerVideo],
) -> str:
    """Shared scaffold: resolve model, preflight, loop scoped videos, predict, bridge.

    *opens_by_path* says whether this op hands its model's runner a path to open,
    which decides two things about every entry: whether an imgstore has to have
    been exported first, and which file's verdict is the one that gates the read.
    Producer knowledge, declared by the op rather than inferred from its kind.
    """
    if not ds.has_root(kind):
        ds.set_root(kind, tracking_root_default(kind))

    # Resolved for its *artifact* -- the weights this run loads. The identity it
    # carries comes from ``infer_identity`` below, which is the one place these
    # two identifiers are minted; computing them here as well would be a second
    # answer to what this run is called.
    model = resolve_model(ds, params.model, train_kind)
    model_id = model.model_id

    identity = infer_identity(ds, kind, version, params, train_kind)
    run_id = identity.run_id
    ctx.set_run_id(run_id)

    scope = ds.resolve_media_scope(params.entries)
    if not scope:
        print(f"[{kind}] No media entries match the given scope.")
        return run_id

    work: list[tuple[str, str, Path, MediaFacts]] = []
    for entry in scope:
        # Each scope entry is one camera; per-camera inference output pathing +
        # index dedup is Phase 2, so single-camera behavior here is unchanged.
        # The op reads the first path; a required-but-unlinked entry already
        # raised in resolve_media_scope, before any defective original was opened.
        group, sequence, resolved = entry.group, entry.sequence, entry.resolved
        source = resolved.paths[0]
        # An op that hands a tool a path cannot hand it an imgstore, which is a
        # directory of chunk files -- so a store resolves to the video
        # ``export-store`` wrote for it, or raises naming that command.
        target = (
            resolve_entry_input(ds, group, sequence, source, kind=kind)
            if opens_by_path
            else source
        )
        # The gate, run here rather than inside the reader, because two of the
        # three ops no longer open the video in this process. The facts must
        # describe the file that will actually be read: for an export that is not
        # the file the index measured, so it is probed and gated on its own.
        facts = verified_read_facts(
            target, resolved.facts[0] if target == source else None, "analysis"
        )[0]
        work.append((group, sequence, target, facts))

    # Preflight after the scope and before anything is written. Two orderings
    # matter here. It follows the scope because resolving media is local and
    # cheap while a probe spawns an interpreter and loads a checkpoint, so a run
    # whose media is unreadable -- or whose scope is empty, which returned
    # above -- should not pay for one. It precedes every write because what a
    # preflight refuses -- an environment that is not there, an upstream build
    # asked for point detection, a checkpoint the running build cannot load, a
    # model whose task is not this op's -- describes a run that never happened,
    # and that must not leave a tracks variant behind naming it.
    #
    # Nothing it reports reaches an identifier. Which environment ran, and what
    # version it holds, are properties of the machine.
    per_video = per_video_for(str(model.path))

    # The tracks variant these predictions will be bridged into, recorded beside
    # the tables. Same payload as the op run above, so the two identifiers
    # coincide -- which is what makes the predictions directory and the tracks
    # variant obviously the same run at a glance, while the index still keeps
    # them in separate columns.
    tracks_variant = identity.tracks_variant
    _ = write_tracks_variant(
        ds.get_root("tracks"),
        tracks_variant,
        kind,
        version,
        infer_variant_payload(params.identity_dump(), model_id),
    )

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

        # A claim, not a cache. Inference still re-infers unconditionally -- the
        # completion marker below records that output is whole, and nothing gates
        # on it, because turning this into a cache is a behaviour change with its
        # own failure mode (a silently skipped re-run over a corrected video). What
        # the claim prevents is two executions writing one ``predictions.parquet``
        # at once. Through ``open_entry`` rather than a fifth inline copy of it, so
        # the exclusive create and the ownership-checked release are the same ones
        # every tracker gets.
        opened = open_entry(
            ds,
            ctx,
            run_root,
            key,
            kind=kind,
            overwrite=False,
            idle_seconds=_INFER_IDLE_SECONDS,
        )
        if opened is None:
            ctx.progress.on_entry_end(i + 1, len(work), key)
            continue
        seq_dir, held = opened
        try:
            outcome = per_video(
                VideoInput(
                    video_path=video_path,
                    work_dir=seq_dir,
                    facts=facts,
                    # Every line the model's runner writes refreshes the claim and
                    # reports position. Without it a video longer than
                    # ``_INFER_IDLE_SECONDS`` expires its own claim mid-run and a
                    # concurrent execution reads the directory as abandoned; an
                    # in-process op simply never had a line to hang this on.
                    on_output=progress_activity(
                        ctx,
                        key,
                        "infer",
                        phase_activity(ctx, seq_dir, held, _INFER_IDLE_SECONDS),
                    ),
                    cancel_check=ctx.cancel_token.is_cancelled,
                )
            )
            df = outcome.frame
            pred_path = outcome.published_path or seq_dir / _PREDICTIONS_NAME
            # Written here only when the caller did not publish it. An op that ran
            # out of process wrote the table itself, atomically, at this same path;
            # copying it back over itself would double the write for nothing.
            if outcome.published_path is None and df is not None and not df.empty:
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
                    source_uid=facts.video_uuid,
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
            release_entry(seq_dir, ctx.execution_id)

        ctx.progress.on_entry_end(i + 1, len(work), key)
        ctx.heartbeat(i + 1)

    print(f"[{kind}] completed run_id={run_id} ({done}/{len(work)}) -> {run_root}")
    return run_id


# --- Ops -----------------------------------------------------------------


@register_op
class InferPoseOp(Op[PoseInferParams]):
    """Run a trained YOLO pose model over scoped videos, bridging into ``tracks/``."""

    kind = "infer-pose"
    category = "infer"
    domain = "tracking"
    # 0.2 because this op's output moved. It used to resize each frame at decode
    # time to fit ``imgsz`` and return coordinates in that smaller space, which
    # reached ``tracks/`` under a schema whose every spatial column is video
    # pixels. Frames are now fed at their native size, as the tracker's have
    # always been, so the numbers mean what the schema says they mean. The
    # version is a visible segment and not a hash term, so nothing is re-derived
    # -- but tables written under 0.1 hold the smaller coordinates and should be
    # re-run.
    version = "0.2"
    Params = PoseInferParams

    def plan_identity(self, ds: Dataset, params: PoseInferParams) -> OpIdentity:
        """What this run and the tracks variant it bridges into are called."""
        return infer_identity(ds, self.kind, self.version, params, "train-pose")

    def run(self, ds: Dataset, params: PoseInferParams, ctx: JobContext) -> str:
        from mosaic.tracking.common.ultralytics_env import (
            ULTRALYTICS_ENV,
            UltralyticsError,
        )
        from mosaic.tracking.external.runner.ultralytics_protocol import (
            InferPoseRequest,
            pose_columns,
        )
        from mosaic.tracking.common.ultralytics_env import probe_environment
        from mosaic.tracking.pose_training.ultralytics_infer import (
            require_pose_model,
            run_pose_inference_tool,
        )

        def per_video_for(model_path: str) -> PerVideo:
            probe = probe_environment(
                model_path,
                env=ULTRALYTICS_ENV,
                failure=UltralyticsError,
                cancel_check=ctx.cancel_token.is_cancelled,
            )
            require_pose_model(probe, model_path)
            n_keypoints = probe.n_keypoints

            def per_video(item: VideoInput) -> VideoPredictions:
                published = item.work_dir / _PREDICTIONS_NAME
                request = InferPoseRequest(
                    model_path=model_path,
                    video_path=str(item.video_path),
                    output_parquet=str(published),
                    annotated_dir=str(item.work_dir) if params.save_images else "",
                    columns=pose_columns(n_keypoints),
                    task="pose",
                    conf=params.conf_threshold,
                    imgsz=params.imgsz,
                    device=params.device,
                    start_frame=params.start_frame,
                    end_frame=params.end_frame,
                    frame_step=params.frame_step,
                    max_frames=params.max_frames,
                    batch_size=params.batch_size,
                    prefetch=True,
                    media_facts=dataclasses.asdict(item.facts),
                    n_keypoints=n_keypoints,
                )
                outcome = run_pose_inference_tool(
                    request,
                    work_dir=item.work_dir,
                    idle_timeout=_INFER_IDLE_SECONDS,
                    cancel_check=item.cancel_check,
                    on_output=item.on_output,
                )
                return VideoPredictions(
                    pd.read_parquet(outcome.predictions_path), outcome.predictions_path
                )

            return per_video

        return _run_inference_op(
            ds,
            params,
            ctx,
            kind=self.kind,
            version=self.version,
            train_kind="train-pose",
            opens_by_path=True,
            per_video_for=per_video_for,
        )


@register_op
class InferPointsOp(Op[PointInferParams]):
    """Run a trained POLO point model over scoped videos, bridging into ``tracks/``."""

    kind = "infer-points"
    category = "infer"
    domain = "tracking"
    # 0.2 for the reason `infer-pose` gives above: the coordinates moved.
    version = "0.2"
    Params = PointInferParams

    def plan_identity(self, ds: Dataset, params: PointInferParams) -> OpIdentity:
        """What this run and the tracks variant it bridges into are called."""
        return infer_identity(ds, self.kind, self.version, params, "train-points")

    def run(self, ds: Dataset, params: PointInferParams, ctx: JobContext) -> str:
        from mosaic.tracking.common.ultralytics_env import POLO_ENV, PoloError
        from mosaic.tracking.external.runner.ultralytics_protocol import (
            POINT_COLUMNS,
            InferPointsRequest,
        )
        from mosaic.tracking.common.ultralytics_env import probe_environment
        from mosaic.tracking.pose_training.ultralytics_infer import (
            require_points_model,
            run_point_inference_tool,
        )

        def per_video_for(model_path: str) -> PerVideo:
            probe = probe_environment(
                model_path,
                env=POLO_ENV,
                failure=PoloError,
                cancel_check=ctx.cancel_token.is_cancelled,
            )
            require_points_model(probe, model_path)

            def per_video(item: VideoInput) -> VideoPredictions:
                published = item.work_dir / _PREDICTIONS_NAME
                request = InferPointsRequest(
                    model_path=model_path,
                    video_path=str(item.video_path),
                    output_parquet=str(published),
                    annotated_dir=str(item.work_dir) if params.save_images else "",
                    columns=list(POINT_COLUMNS),
                    task="locate",
                    conf=params.conf_threshold,
                    imgsz=params.imgsz,
                    device=params.device,
                    start_frame=params.start_frame,
                    end_frame=params.end_frame,
                    frame_step=params.frame_step,
                    max_frames=params.max_frames,
                    batch_size=params.batch_size,
                    prefetch=True,
                    media_facts=dataclasses.asdict(item.facts),
                    # `params.dor` is deliberately not sent. It reaches no
                    # Ultralytics argument on the in-process path either, so
                    # forwarding it would change what this op computes rather than
                    # move where it computes it. It remains a term of the run
                    # identifier and nothing else; that it is one is its own defect.
                    radii=None,
                )
                outcome = run_point_inference_tool(
                    request,
                    work_dir=item.work_dir,
                    idle_timeout=_INFER_IDLE_SECONDS,
                    cancel_check=item.cancel_check,
                    on_output=item.on_output,
                )
                return VideoPredictions(
                    pd.read_parquet(outcome.predictions_path), outcome.predictions_path
                )

            return per_video

        return _run_inference_op(
            ds,
            params,
            ctx,
            kind=self.kind,
            version=self.version,
            train_kind="train-points",
            opens_by_path=True,
            per_video_for=per_video_for,
        )


@register_op
class InferLocalizerOp(Op[LocalizerInferParams]):
    """Run a trained heatmap localizer over scoped videos, bridging into ``tracks/``."""

    kind = "infer-localizer"
    category = "infer"
    domain = "tracking"
    # Unmoved at 0.1, unlike its two siblings: the localizer is mosaic's own
    # PyTorch, it never resized at decode time, and its coordinates were already
    # in source pixels.
    version = "0.1"
    Params = LocalizerInferParams

    def plan_identity(self, ds: Dataset, params: LocalizerInferParams) -> OpIdentity:
        """What this run and the tracks variant it bridges into are called."""
        return infer_identity(ds, self.kind, self.version, params, "train-localizer")

    def run(self, ds: Dataset, params: LocalizerInferParams, ctx: JobContext) -> str:
        from mosaic.tracking.pose_training.localizer_inference import (
            localizer_detections_to_dataframe,
            run_localizer_inference,
        )

        def per_video_for(model_path: str) -> PerVideo:
            def per_video(item: VideoInput) -> VideoPredictions:
                detections = run_localizer_inference(
                    model_path,
                    item.video_path,
                    output_dir=item.work_dir if params.save_images else None,
                    num_classes=params.num_classes,
                    initial_channels=params.initial_channels,
                    thresholds=params.thresholds,
                    start_frame=params.start_frame,
                    end_frame=params.end_frame,
                    frame_step=params.frame_step,
                    max_frames=params.max_frames,
                    device=params.device,
                    save_images=params.save_images,
                    facts=item.facts,
                )
                # No published path: this one ran here, so the scaffold writes it.
                return VideoPredictions(localizer_detections_to_dataframe(detections))

            return per_video

        return _run_inference_op(
            ds,
            params,
            ctx,
            kind=self.kind,
            version=self.version,
            train_kind="train-localizer",
            # Reads a store natively, through mosaic's own reader.
            opens_by_path=False,
            per_video_for=per_video_for,
        )
