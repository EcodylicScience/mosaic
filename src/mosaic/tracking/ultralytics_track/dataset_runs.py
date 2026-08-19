"""Running Ultralytics tracking across a dataset's scoped media.

The dataset-level half of the integration: resolve the model to a content
identity, probe the Ultralytics environment, mint a run, and drive one gated
``track`` phase per entry through the shared runner, bridging each entry's raw
predictions into ``tracks/<variant>/``.

One expensive gated phase, like Lightning Pose, and like every other tracker the
tool runs in an environment of its own and opens a path -- so each entry resolves
through :func:`resolve_tool_input`, and every identifier, marker, reuse decision
and index write is the shared machinery, unchanged.

**What precedes minting, and what does not.** Resolving the model reads its bytes
for a digest, and the probe loads its weights where they will run; both come
before the mint, because a variant naming weights that cannot be loaded -- or a
backend this environment does not know -- describes a run that never happened.
The cost is one probe for a run whose scope matches nothing. Tracking itself is
per entry and after the scope, so such a run spawns nothing else.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

from mosaic.core.helpers import make_entry_key
from mosaic.core.json_value import JsonValue
from mosaic.core.media.read_target import verified_read_facts
from mosaic.core.pipeline.dataset_indexes import register_reconcilable_index
from mosaic.core.pipeline.index_csv import IndexCSV
from mosaic.core.pipeline.job import Cancelled, CancelToken, JobContext
from mosaic.core.pipeline.markers import clear_phase_marker
from mosaic.core.pipeline.op_identity import op_run_id, parse_op_run_id
from mosaic.core.pipeline.subprocess_util import ProcessCancelled
from mosaic.core.track_library.ultralytics_tracks import raw_columns
from mosaic.tracking.common.bridge import (
    BridgeCounts,
    publish_or_record,
    publish_tracks_table,
    readable_tracks_table,
    tracks_table_path,
)
from mosaic.tracking.common.driver import EntryJob, run_tracker
from mosaic.tracking.common.entry import (
    claim,
    clear_outputs,
    phase_activity,
    record_phase,
    reusable_output,
)
from mosaic.tracking.common.index import (
    media_composition_cell,
    register_tracker_row_class,
    TrackerRunRowBase,
    list_tracker_runs,
    tracker_index,
    tracker_index_path,
)
from mosaic.tracking.common.mint import mint_tracker_run, tracker_run_root
from mosaic.tracking.common.scope import build_work_items
from mosaic.tracking.common.tool_input import resolve_tool_input
from mosaic.tracking.external.runner.ultralytics_protocol import TrackRequest
from mosaic.tracking.model_refs import resolve_model
from mosaic.tracking.ultralytics_track.tracker_defaults import (
    TrackerName,
    TrackerSetting,
    resolve_tracker_config,
)
from mosaic.tracking.ultralytics_track.version import (
    TRAIN_POSE_KIND,
    ULTRALYTICS_KIND,
    ULTRALYTICS_VERSION,
)

from .run import (
    ModelTask,
    UnsupportedTaskError,
    effective_tracker_table,
    probe_ultralytics,
    require_supported_task,
    require_ultralytics,
    run_ultralytics_tool,
    write_tracker_yaml,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.progress import ProgressCallback

__all__ = [
    "UltralyticsIndexRow",
    "list_ultralytics_runs",
    "run_ultralytics",
    "ultralytics_index",
    "ultralytics_index_path",
    "ultralytics_run_id",
    "ultralytics_run_root",
    "ultralytics_settings",
]

PREDICTIONS_SUFFIX = ".predictions.parquet"
TRACKER_CONFIG_NAME = "tracker.yaml"


# --- the run index ---------------------------------------------------------


def ultralytics_run_root(ds: Dataset, run_id: str) -> Path:
    return tracker_run_root(ds, ULTRALYTICS_KIND, run_id)


def ultralytics_index_path(ds: Dataset) -> Path:
    return tracker_index_path(ds, ULTRALYTICS_KIND)


@dataclass(frozen=True, slots=True)
class UltralyticsIndexRow(TrackerRunRowBase):
    """One tracked entry.

    ``model_task`` is the declared task rather than anything read off the
    weights, so a reuse run -- which never loads a model -- fills it correctly.
    ``n_frames`` and ``n_keypoints`` are likewise re-derived from the recorded
    parquet on reuse, because a reuse run reporting zeros would overwrite a good
    row with an empty one.
    """

    model_id: str = ""
    model_task: str = ""
    tracker: str = ""
    n_frames: int = 0
    n_keypoints: int = 0
    predictions_path: str = ""


def ultralytics_index(path: Path) -> IndexCSV[UltralyticsIndexRow]:
    return tracker_index(path, UltralyticsIndexRow)


# --- identity --------------------------------------------------------------


def ultralytics_run_id(settings: Mapping[str, object]) -> str:
    return op_run_id(ULTRALYTICS_KIND, ULTRALYTICS_VERSION, dict(settings))


def ultralytics_settings(
    *,
    model_id: str,
    task: ModelTask,
    tracker: TrackerName,
    tracker_config: Mapping[str, TrackerSetting],
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
    classes: Sequence[int] | None,
    agnostic_nms: bool,
    start_frame: int,
    end_frame: int | None,
    frame_step: int,
) -> dict[str, object]:
    """Everything that determines what this run produces, and nothing else.

    Scope-free: no dataset, no video, no entry, no output location -- and no
    device, precision or batch size, which change how a run happens rather than
    what it produces. So one value names one variant across every sequence the
    run covered.

    The model is its content identity, never a path. The tracker is named twice
    on purpose: by backend, which is what a reader wants, and as the fully
    resolved table, which is what actually ran.
    """
    return {
        "model": model_id,
        "task": task,
        "tracker": tracker,
        "tracker_config": dict(tracker_config),
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "max_det": max_det,
        # Sorted and de-duplicated here rather than on the field: the identity
        # payload must not depend on the order a user typed a filter in, while
        # `run_params.json` should keep what they wrote.
        "classes": sorted(set(classes)) if classes is not None else None,
        "agnostic_nms": agnostic_nms,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_step": frame_step,
    }


# --- the bridge ------------------------------------------------------------


def _bridge_predictions_to_tracks(
    ds: Dataset,
    group: str,
    sequence: str,
    predictions_path: Path,
    *,
    tracks_variant: str,
    producer_run_id: str,
    video_path: Path,
    model_files: Sequence[Path],
    fps: float,
    overwrite: bool,
) -> BridgeCounts | None:
    """Convert one entry's raw predictions into its standardized table."""
    from mosaic.core.track_converter import EntryHints, get_track_converter
    from mosaic.core.track_library.ultralytics_tracks import UltralyticsTracksParams

    out_path = tracks_table_path(ds, tracks_variant, make_entry_key(group, sequence))
    if out_path.exists() and not overwrite:
        # Reuse only what reads. An unreadable table -- torn by a kill before the
        # writes here became atomic, or by an external tool -- falls through and is
        # republished, rather than being adopted as a valid empty result.
        reusable = readable_tracks_table(out_path)
        if reusable is not None:
            return reusable

    converter = get_track_converter("ultralytics_tracks")
    # A table one entry cannot convert must not end a batch whose other entries
    # converted fine -- but it must not vanish either. The exception now
    # propagates to `publish_or_record`, which keeps the batch going *and*
    # records the entry as failed on the attempt. Tracking failures still
    # propagate past that and end the run.
    df = converter.convert(
        predictions_path,
        UltralyticsTracksParams(fps=fps),
        EntryHints(group=group, sequence=sequence),
    )

    return publish_tracks_table(
        ds,
        df,
        kind=ULTRALYTICS_KIND,
        group=group,
        sequence=sequence,
        tracks_variant=tracks_variant,
        producer_run_id=producer_run_id,
        source=predictions_path.parent,
        consumed=[predictions_path, video_path, *model_files],
    )


def _keypoint_count_of(path: Path) -> int:
    """How many keypoint triples a recorded predictions table carries."""
    columns = pd.read_parquet(path).columns
    return sum(1 for name in columns if str(name).startswith("kpx"))


def _frame_count_of(path: Path) -> int:
    """How many distinct frames a recorded predictions table covers."""
    table = pd.read_parquet(path, columns=["frame"])
    return int(table["frame"].nunique())


def _context_token(ctx: JobContext | None) -> CancelToken | None:
    """The cancel token an already-open job carries, if a run was handed one.

    The op path passes its ``ctx`` and no token; a library caller passes a token
    and no ``ctx``. Reading both is what lets work done before ``run_tracker``
    opens -- the probe -- be cancellable on either.
    """
    return ctx.cancel_token if ctx is not None else None


# --- the run ---------------------------------------------------------------


def run_ultralytics(
    ds: Dataset,
    *,
    model_path: Path | str,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    entries: Iterable[tuple[str, str]] | None = None,
    task: ModelTask = "pose",
    tracker: TrackerName = "bytetrack",
    tracker_overrides: Mapping[str, JsonValue] | None = None,
    conf: float = 0.1,
    iou: float = 0.7,
    imgsz: int = 640,
    max_det: int = 300,
    classes: Sequence[int] | None = None,
    agnostic_nms: bool = False,
    start_frame: int = 0,
    end_frame: int | None = None,
    frame_step: int = 1,
    # execution
    device: str = "0",
    precision: Literal["fp32", "fp16"] = "fp32",
    batch_size: int = 8,
    prefetch: bool = True,
    idle_timeout: float = 900,
    max_runtime: float | None = None,
    ultralytics_conda_env: str | None = None,
    ultralytics_bin: Path | str | None = None,
    overwrite: bool = False,
    convert_to_tracks: bool = True,
    # Job Contract
    execution_id: str | None = None,
    owner: str = "",
    track: bool = True,
    progress_callback: ProgressCallback | None = None,
    cancel_token: CancelToken | None = None,
    ctx: JobContext | None = None,
) -> str:
    """Track every scoped sequence with *model_path*, and return the run id.

    ``idle_timeout`` bounds silence and sizes the in-flight claim on each entry's
    working directory: the runner prints a progress line per decoded batch, so a
    process that has printed nothing for that long is hung and is killed.
    ``max_runtime`` is the absolute ceiling. Cancellation kills the process
    group, so its latency is a poll interval rather than a video.

    ``ultralytics_conda_env`` and ``ultralytics_bin`` are the top two rungs of
    the location ladder, for a caller holding more than one Ultralytics
    environment. They name a machine rather than a result, so they reach no
    identifier and the op deliberately does not carry them: a queued job would
    ship a path that means something else on the host that runs it.
    """
    tracker_config = resolve_tracker_config(tracker, tracker_overrides)

    # Content, never a path -- and the kind comes from the reference itself,
    # because both `train-pose` and `train-points` produce runnable weights and a
    # run id resolves against the index its own training op wrote.
    parsed = parse_op_run_id(str(model_path))
    model_kind = parsed.kind if parsed is not None else TRAIN_POSE_KIND
    resolved_model = resolve_model(ds, str(model_path), model_kind)

    # Once per run, before anything is minted: what the environment holds, what
    # the backend ships, what the weights declare and how many keypoints they
    # carry. Every refusal below is decided from what it reported, rather than
    # from an import mosaic must not make.
    # A cancel raised while the weights load is answered by killing the probe,
    # the same way it is answered during tracking. The translation is local
    # because this call happens before `run_tracker` opens, and it is what makes
    # a cancelled attempt read as cancelled rather than failed.
    cancel = cancel_token if cancel_token is not None else _context_token(ctx)
    try:
        probe = probe_ultralytics(
            resolved_model.path,
            tracker=tracker,
            idle_timeout=idle_timeout,
            max_runtime=max_runtime,
            conda_env=ultralytics_conda_env,
            bin_path=ultralytics_bin,
            cancel_check=cancel.is_cancelled if cancel is not None else None,
        )
    except ProcessCancelled as exc:
        raise Cancelled() from exc
    require_ultralytics(probe, tracker)
    weights_task = require_supported_task(probe.model_task)
    if weights_task != task:
        raise UnsupportedTaskError(
            f"{resolved_model.path} is a {weights_task!r} model but the run "
            f"declares task={task!r}. The declared task is part of the run "
            "identifier, so it has to match what the weights actually are."
        )

    settings = ultralytics_settings(
        model_id=resolved_model.model_id,
        task=task,
        tracker=tracker,
        tracker_config=tracker_config,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        classes=classes,
        agnostic_nms=agnostic_nms,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_step=frame_step,
    )
    minted = mint_tracker_run(
        ds,
        kind=ULTRALYTICS_KIND,
        version=ULTRALYTICS_VERSION,
        settings=settings,
        observed={
            "model_id": resolved_model.model_id,
            "task": task,
            "tracker": tracker,
            # Which Ultralytics ran is a property of the machine, so it is
            # recorded beside the variant as provenance and enters no digest.
            "ultralytics_version": probe.ultralytics_version,
        },
    )
    scope = ds.resolve_media_scope(groups, sequences, entries)
    if not scope:
        print(
            "[run_ultralytics] No media entries match the given scope.",
            file=sys.stderr,
        )
        return minted.run_id

    # Written once per run beside run_params.json: the configuration is run-wide,
    # and Ultralytics reads it exactly once, on the first tracking call.
    tracker_yaml = write_tracker_yaml(
        minted.run_root / TRACKER_CONFIG_NAME,
        effective_tracker_table(probe.installed_tracker_table, tracker_config),
    )
    # Mosaic owns the raw-parquet column contract -- the converter that reads the
    # table is mosaic's -- so the runner is told the columns rather than deriving
    # them. One list for the run: the keypoint count belongs to the model.
    columns = list(raw_columns(probe.n_keypoints))

    def track_one(job: EntryJob) -> UltralyticsIndexRow | None:
        item, work_dir, seq_ctx = job.item, job.work_dir, job.ctx
        predictions_path = work_dir / f"{item.key}{PREDICTIONS_SUFFIX}"

        reusable = reusable_output(
            job.ds,
            work_dir,
            "track",
            params_hash=minted.params_hash,
            video_path=item.video_path,
            video_uid=item.video_uid,
        )
        if reusable is None:
            clear_phase_marker(work_dir, "track")
            clear_outputs(work_dir, ULTRALYTICS_KIND, "track")
            phase_claim = claim(seq_ctx, work_dir, "track", idle_timeout)
            seq_ctx.progress.on_phase("track", item.key)
            # The runner opens the path itself, so an imgstore recording resolves
            # to the plain video export-store wrote for it. Resolved here rather
            # than before the reuse gate: an entry already tracked needs no
            # export, and demanding one would fail a re-run over finished work.
            tool_input = resolve_tool_input(job.ds, item, kind=ULTRALYTICS_KIND)
            # The facts have to describe the file the tool will open, and the
            # index row measured the source. For an exported store those are two
            # files, so the indexed facts travel only when the resolved path is
            # the source itself; otherwise the export is probed and gated on its
            # own. The gate is mosaic's alone -- the runner cannot call it.
            facts = verified_read_facts(
                tool_input,
                item.facts if tool_input == item.video_path else None,
                "analysis",
            )[0]
            result = run_ultralytics_tool(
                TrackRequest(
                    model_path=str(resolved_model.path),
                    video_path=str(tool_input),
                    output_parquet=str(predictions_path),
                    tracker_yaml=str(tracker_yaml),
                    # Ultralytics computes its own run directory eagerly even
                    # with nothing saved, so it is pinned at the run root rather
                    # than left to walk the shared `runs/` tree.
                    project_dir=str(minted.run_root),
                    columns=columns,
                    n_keypoints=probe.n_keypoints,
                    task=task,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    max_det=max_det,
                    classes=list(classes) if classes is not None else None,
                    agnostic_nms=agnostic_nms,
                    device=device,
                    precision=precision,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    frame_step=frame_step,
                    batch_size=batch_size,
                    prefetch=prefetch,
                    media_facts=dataclasses.asdict(facts),
                ),
                work_dir=work_dir,
                idle_timeout=idle_timeout,
                max_runtime=max_runtime,
                conda_env=ultralytics_conda_env,
                bin_path=ultralytics_bin,
                cancel_check=seq_ctx.cancel_token.is_cancelled,
                on_output=phase_activity(seq_ctx, work_dir, phase_claim, idle_timeout),
            )
            marker = record_phase(
                job.ds,
                work_dir,
                "track",
                ctx=seq_ctx,
                run_id=minted.run_id,
                params_hash=minted.params_hash,
                video_path=item.video_path,
                video_uid=item.video_uid,
                output=result.predictions_path,
            )
            out_path = result.predictions_path
            n_frames, n_ids = result.n_frames, result.n_ids
            n_keypoints = _keypoint_count_of(out_path)
            recomputed = True
        else:
            marker, out_path = reusable
            # Re-derived from disk: the phase that knew these did not run.
            counts = readable_tracks_table(out_path) if out_path.exists() else None
            n_ids = counts.n_ids if counts is not None else 0
            n_frames = _frame_count_of(out_path)
            n_keypoints = _keypoint_count_of(out_path)
            recomputed = False

        row = UltralyticsIndexRow(
            run_id=minted.run_id,
            group=item.group,
            sequence=item.sequence,
            # What this entry's media was when the run read it. The
            # tracker identity carries no media term, so without this a
            # re-transcode leaves the run reading as current over
            # different pixels.
            consumed_media_composition=media_composition_cell(
                job.ds, item.group, item.sequence
            ),
            abs_path=Path(job.ds.relative_to_root(work_dir)),
            video_abs_path=(
                marker.source
                if marker.source
                else job.ds.relative_to_root(item.video_path)
            ),
            params_hash=minted.params_hash,
            model_id=resolved_model.model_id,
            model_task=task,
            tracker=tracker,
            n_ids=n_ids,
            n_frames=n_frames,
            n_keypoints=n_keypoints,
            predictions_path=job.ds.relative_to_root(out_path),
        )

        if not convert_to_tracks:
            return row
        bridged = publish_or_record(
            job.ctx,
            item.key,
            lambda: _bridge_predictions_to_tracks(
                job.ds,
                item.group,
                item.sequence,
                out_path,
                tracks_variant=minted.tracks_variant,
                producer_run_id=minted.run_id,
                video_path=item.video_path,
                model_files=list(resolved_model.significant_files),
                fps=item.fps,
                overwrite=job.overwrite or recomputed,
            ),
            kind=ULTRALYTICS_KIND,
        )
        return row if bridged is None else dataclasses.replace(row, n_ids=bridged.n_ids)

    return run_tracker(
        ds,
        kind=ULTRALYTICS_KIND,
        target="ultralytics-track",
        minted=minted,
        work_items=build_work_items(ds, scope, kind=ULTRALYTICS_KIND),
        index=ultralytics_index(ultralytics_index_path(ds)),
        run_entry=track_one,
        overwrite=overwrite,
        execution_id=execution_id,
        owner=owner,
        track=track,
        progress_callback=progress_callback,
        cancel_token=cancel_token,
        ctx=ctx,
    )


def list_ultralytics_runs(ds: Dataset) -> pd.DataFrame:
    """Every recorded Ultralytics tracking run, as a dataframe."""
    return list_tracker_runs(ds, ULTRALYTICS_KIND, UltralyticsIndexRow)


register_reconcilable_index(ULTRALYTICS_KIND, ultralytics_index)

# The row class this root's index holds, so an inventory can ask about every
# tracker generically. Registered rather than tabled in ``common``, which is
# imported by this module and cannot import it back.
register_tracker_row_class(ULTRALYTICS_KIND, UltralyticsIndexRow)
