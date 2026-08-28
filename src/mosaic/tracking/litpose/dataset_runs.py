"""Dataset-level Lightning Pose runs: content-addressed, tracked, tracks-integrated.

``run_litpose(ds, ...)`` is the first-class entry point that turns the standalone
Lightning Pose CLI wrapper (:mod:`mosaic.tracking.litpose.run`) into a
Job-Contract stage, mirroring :func:`mosaic.tracking.sleap.run_sleap`:

* it resolves input videos through ``Dataset.resolve_media_scope``, routing each
  entry by its transcode verdict (an analysis-required entry tracks its
  constant-rate analysis derivative, not the defective original);
* resolves the trained Lightning Pose model directory to a content digest -- what
  *names* the weights + config, never the path they sit at -- and computes a
  content-addressed ``run_id`` over the resolved settings, writing run-addressed
  artifacts under ``<litpose_root>/<run_id>/<group>__<seq>/``;
* records the attempt in its JSONL run-log (``kind="litpose"``), reports coarse
  progress, and is cancellable (the subprocess runs in a killable process group);
* bridges the DeepLabCut-style predictions CSV into standardized
  ``tracks/<variant>/<group>__<seq>.parquet`` via the registered ``deeplabcut``
  converter.

Lightning Pose is single-animal, per-frame: each video yields one ``id=0`` track
and the op carries no tracker knobs. There is one expensive, gated phase --
``track`` (``predict_on_video_file`` inference, producing the CSV). Its completion
marker lets a killed run resume without re-running inference; a killed run leaves
no marker, so its partial CSV is discarded and cleared on the next attempt.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.index_csv import IndexCSV
from mosaic.core.pipeline.job import CancelToken, JobContext
from mosaic.core.pipeline.markers import (
    clear_phase_marker,
)
from mosaic.core.pipeline.dataset_indexes import register_reconcilable_index
from mosaic.core.pipeline.op_identity import op_run_id
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
from mosaic.tracking.litpose.params import LitposeParams
from mosaic.tracking.litpose.version import (
    LITPOSE_KIND,
    LITPOSE_VERSION,
    TRAIN_LITPOSE_KIND,
)
from mosaic.tracking.model_refs import resolve_model_set

from .run import run_litpose_predict

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope
    from mosaic.core.pipeline.progress import ProgressCallback


# --- Lightning Pose run index --------------------------------------------


def litpose_run_root(ds: Dataset, run_id: str) -> Path:
    """Where one litpose run keeps its per-entry working directories."""
    return tracker_run_root(ds, LITPOSE_KIND, run_id)


def litpose_index_path(ds: Dataset) -> Path:
    """Where the litpose run index lives."""
    return tracker_index_path(ds, LITPOSE_KIND)


@dataclass(frozen=True, slots=True)
class LitposeIndexRow(TrackerRunRowBase):
    """Typed row for the litpose run index CSV.

    ``video_abs_path`` and ``csv_path`` are the tool-specific path columns; see
    :class:`TrackerRunRowBase` for how they are stored and where they must be
    declared.
    """

    model_id: str = ""
    model_type: str = ""
    csv_path: str = ""


def litpose_index(path: Path) -> IndexCSV[LitposeIndexRow]:
    """The litpose run index, one row per (run, entry)."""
    return tracker_index(path, LitposeIndexRow)


# --- Settings -------------------------------------------------------------


def litpose_run_id(settings: Mapping[str, object]) -> str:
    """Mint a tracker run identifier from the resolved Lightning Pose settings."""
    return op_run_id(LITPOSE_KIND, LITPOSE_VERSION, dict(settings))


def litpose_settings(params: LitposeParams, *, model_id: str) -> dict[str, object]:
    """Build the settings that define a Lightning Pose result -- the ``run_id`` payload.

    The model is carried as its content digest (``model_id``), never a path.
    Lightning Pose is pose-only, so there are no tracker knobs; the Hydra
    ``litpose_overrides`` are identity because they change the produced keypoints.

    Args:
        params: The run's parameters.
            :class:`~mosaic.tracking.litpose.params.LitposeParams` describes
            every field. The scope and execution knobs are excluded from
            identity and stay out of the payload.
        model_id: The model's identity -- a training ``run_id`` or a content
            digest, never a path. The caller resolves it, because a bare model
            path is a mutable key: swapping the checkpoint in place would let two
            different runs share one identifier.
    """
    overrides = params.litpose_overrides
    return {
        "model": model_id,
        "litpose_overrides": dict(overrides) if overrides else None,
    }


# --- Per-entry reuse ------------------------------------------------------


# Unlike SLEAP/TREx, Lightning Pose is a new integration with no pre-marker run
# directories to adopt, and its single output cannot distinguish a complete CSV
# from a partial one a killed predict left behind. So there is deliberately no
# "adopt a marker-less directory" step: a run is reusable only on the strength of
# a completion marker its own predict wrote.


# --- predictions-CSV -> standardized tracks bridge ------------------------


def _bridge_csv_to_tracks(
    ds: Dataset,
    group: str,
    sequence: str,
    csv_path: Path,
    *,
    tracks_variant: str,
    producer_run_id: str,
    video_path: Path,
    model_files: Sequence[Path],
    fps: float,
    overwrite: bool,
) -> BridgeCounts | None:
    """Bridge a Lightning Pose CSV into ``tracks/<variant>/<group>__<seq>.parquet``.

    Reuses the registered ``deeplabcut`` converter -- Lightning Pose exports the
    same ``(scorer, bodypart, coord)`` layout -- with the authoritative (group,
    sequence) known from the media index, so no name is guessed from a filename.
    Returns ``None`` when the conversion failed and nothing was published.
    """
    from mosaic.core.track_converter import EntryHints, get_track_converter
    from mosaic.core.track_library.deeplabcut import DlcParams

    out_path = tracks_table_path(ds, tracks_variant, make_entry_key(group, sequence))
    if out_path.exists() and not overwrite:
        # Reuse only what reads. An unreadable table -- torn by a kill before the
        # writes here became atomic, or by an external tool -- falls through and is
        # republished, rather than being adopted as a valid empty result.
        reusable = readable_tracks_table(out_path)
        if reusable is not None:
            return reusable

    converter = get_track_converter("deeplabcut")
    # Propagates to `publish_or_record` -- see the note in the sleap bridge.
    df = converter.convert(
        csv_path, DlcParams(fps=fps), EntryHints(group=group, sequence=sequence)
    )

    return publish_tracks_table(
        ds,
        df,
        kind=LITPOSE_KIND,
        group=group,
        sequence=sequence,
        tracks_variant=tracks_variant,
        producer_run_id=producer_run_id,
        source=csv_path.parent,
        consumed=[csv_path, video_path, *model_files],
    )


# --- Public entry point ---------------------------------------------------


def run_litpose(
    ds: Dataset,
    params: LitposeParams,
    scope: "ResolvedScope | None" = None,
    *,
    overwrite: bool = False,
    litpose_conda_env: str | None = None,
    litpose_bin: Path | str | None = None,
    # Job Contract
    execution_id: str | None = None,
    owner: str = "",
    track: bool = True,
    progress_callback: ProgressCallback | None = None,
    cancel_token: CancelToken | None = None,
    # When set, run inside this already-open JobContext instead of opening one --
    # the ``mosaic run --kind litpose`` path (``LitposeOp``) hands its ctx here so
    # Lightning Pose rides the standard runner without double-wrapping the Job
    # Contract.
    ctx: JobContext | None = None,
) -> str:
    """Run Lightning Pose inference over scoped videos as a tracked job.

    *params* states what to run and which entries to run it over;
    :class:`~mosaic.tracking.litpose.params.LitposeParams` describes every field.
    The Job-Contract knobs beside it (``execution_id`` / ``owner`` / ``track`` /
    ``progress_callback`` / ``cancel_token``) open the run's context, and *ctx*
    runs inside one that is already open instead.

    ``litpose_conda_env`` and ``litpose_bin`` are the top two rungs of the
    location ladder, for a caller holding more than one Lightning Pose
    environment; below them the entry point is found through
    ``MOSAIC_LITPOSE_CONDA_ENV`` / ``MOSAIC_LITPOSE_BIN`` and then ``$PATH``.
    They name a machine rather than a result, so they reach no identifier and the
    op does not carry them: a queued job would ship a path that means something
    else on the host that runs it.

    Returns the content-addressed ``run_id``.
    """
    # Resolve the model *before* the settings that name it, because what the
    # settings carry is the weights' identity, not the path that pointed at them.
    # An unresolvable reference aborts here, before any run root or tracks variant
    # is recorded.
    #
    # Resolved under the *training* kind, not this tracker's. A reference may be
    # a path or a registered training ``run_id``, and a run_id resolves against
    # ``models/<kind>/index.csv`` -- the index the row was written into, which
    # ``train-litpose`` owns. Passing ``LITPOSE_KIND`` here sent every run_id to a
    # ``models/litpose/`` index nothing writes, so only a path ever resolved and
    # "train here, track with it there" could not be spelled by name. The artifact
    # shape is unaffected: ``MODEL_KINDS`` declares ``train-litpose`` as Lightning
    # Pose's own spec for exactly this.
    resolved_model = resolve_model_set(ds, [str(params.model_path)], TRAIN_LITPOSE_KIND)

    settings = litpose_settings(params, model_id=resolved_model.model_id)
    minted = mint_tracker_run(
        ds,
        kind=LITPOSE_KIND,
        version=LITPOSE_VERSION,
        settings=settings,
        # Provenance, never identity: the model digest (already the identity term)
        # and the model type, recorded so a variant is explicable from disk.
        observed={
            "model_id": resolved_model.model_id,
            "model_type": resolved_model.model_type,
        },
    )
    scope_entries = scope.op_entries if scope is not None else None
    media_scope = ds.resolve_media_scope(scope_entries)
    if not media_scope:
        print("[run_litpose] No media entries match the given scope.", file=sys.stderr)
        return minted.run_id

    def predict_one(job: EntryJob) -> LitposeIndexRow | None:
        """One entry: the gated inference phase, then the bridge."""
        item, work_dir, seq_ctx = job.item, job.work_dir, job.ctx
        csv_path = work_dir / f"{item.key}.predictions.csv"

        # Lightning Pose deliberately does not adopt a marker-less directory.
        # Its single output cannot distinguish a complete CSV from a partial one
        # a killed predict left behind, so a run is reusable only on the strength
        # of a marker its own predict wrote.
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
            clear_outputs(work_dir, LITPOSE_KIND, "track")
            track_claim = claim(seq_ctx, work_dir, "track", params.idle_timeout)
            seq_ctx.progress.on_phase("track", item.key)
            # Lightning Pose opens the path itself, so an imgstore recording
            # resolves to the plain video export-store wrote for it. Resolved
            # here rather than before the reuse gate: an entry already predicted
            # needs no export, and demanding one would fail a re-run over
            # finished work.
            predict_result = run_litpose_predict(
                resolve_tool_input(job.ds, item, kind=LITPOSE_KIND),
                csv_path,
                model_dir=resolved_model.path,
                precision=params.precision,
                overrides=params.litpose_overrides,
                idle_timeout=params.idle_timeout,
                max_runtime=params.max_runtime,
                litpose_conda_env=litpose_conda_env,
                litpose_bin=litpose_bin,
                cancel_check=seq_ctx.cancel_token.is_cancelled,
                on_output=phase_activity(
                    seq_ctx, work_dir, track_claim, params.idle_timeout
                ),
            )
            csv_out = predict_result.csv_path
            track_marker = record_phase(
                job.ds,
                work_dir,
                "track",
                ctx=seq_ctx,
                run_id=minted.run_id,
                params_hash=minted.params_hash,
                video_path=item.video_path,
                video_uid=item.video_uid,
                output=csv_out,
            )
            recomputed = True
        else:
            track_marker, csv_out = reusable
            recomputed = False

        row = LitposeIndexRow(
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
            # From the marker, so the row names what produced the data rather
            # than what the scope resolves to now. The two can only differ when
            # the marker does not know.
            video_abs_path=(
                track_marker.source
                if track_marker.source
                else job.ds.relative_to_root(item.video_path)
            ),
            params_hash=minted.params_hash,
            model_id=resolved_model.model_id,
            model_type=resolved_model.model_type,
            n_ids=1,
            csv_path=job.ds.relative_to_root(csv_out),
        )

        if not params.convert_to_tracks:
            return row
        # A recomputed entry must replace its parquet: the bridge otherwise
        # declines to overwrite, and the table would keep the results of the run
        # just invalidated.
        bridged = publish_or_record(
            job.ctx,
            item.key,
            lambda: _bridge_csv_to_tracks(
                job.ds,
                item.group,
                item.sequence,
                csv_out,
                tracks_variant=minted.tracks_variant,
                producer_run_id=minted.run_id,
                video_path=item.video_path,
                model_files=list(resolved_model.significant_files),
                fps=item.fps,
                overwrite=job.overwrite or recomputed,
            ),
            kind=LITPOSE_KIND,
        )
        return row if bridged is None else dataclasses.replace(row, n_ids=bridged.n_ids)

    return run_tracker(
        ds,
        kind=LITPOSE_KIND,
        target="litpose-predict",
        minted=minted,
        work_items=build_work_items(ds, media_scope, kind=LITPOSE_KIND),
        index=litpose_index(litpose_index_path(ds)),
        run_entry=predict_one,
        overwrite=overwrite,
        execution_id=execution_id,
        owner=owner,
        track=track,
        progress_callback=progress_callback,
        cancel_token=cancel_token,
        ctx=ctx,
    )


def list_litpose_runs(ds: Dataset) -> pd.DataFrame:
    """List litpose runs tracked in the litpose index."""
    return list_tracker_runs(ds, LITPOSE_KIND, LitposeIndexRow)


# Item 6.1: the reconciler opens this root's index through the registry, so
# ``core`` never imports ``tracking`` to reach a row class.
register_reconcilable_index(LITPOSE_KIND, litpose_index)

# The row class this root's index holds, so an inventory can ask about every
# tracker generically. Registered rather than tabled in ``common``, which is
# imported by this module and cannot import it back.
register_tracker_row_class(LITPOSE_KIND, LitposeIndexRow)
