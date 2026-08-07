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
track params, calibration). What each entry was actually computed *from* -- the
source video, and the parameters of each phase -- is recorded in that entry's
completion markers (:mod:`mosaic.core.pipeline.markers`), and re-entry compares
against them. So changing which video a sequence resolves to forces a
recompute even though the settings, and therefore the ``run_id``, are unchanged.
Folding a video content-hash into the ``run_id`` itself is a later refinement
that needs no schema change.

**What the per-phase gate buys today, and what it does not.** The run root is
named by a hash over *every* setting, so a track-only parameter change already
moves the whole working directory. Within this layout the split therefore buys
resume-after-interruption -- a convert that finished and a track that was
killed re-runs only the track -- and not cross-run reuse of a conversion. That
is item 8.5, which keys the conversion cache on ``(video uid, convert-params
hash)``; the convert marker records both so 8.5 inherits a correct gate rather
than bolting one on.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline._utils import hash_params
from mosaic.tracking.model_refs import resolve_model
from mosaic.core.pipeline.dataset_indexes import register_reconcilable_index
from mosaic.core.pipeline.op_identity import (
    op_run_id,
    parse_op_run_id,
)
from mosaic.tracking.common.bridge import (
    BridgeCounts,
    publish_tracks_table,
    tracks_table_path,
)
from mosaic.tracking.common.driver import EntryJob, run_tracker
from mosaic.tracking.common.entry import (
    AdoptEvidence,
    adopt_completed_directory,
    claim,
    clear_outputs,
    phase_activity,
    record_phase,
    reusable_marker,
    reusable_output,
)
from mosaic.tracking.common.index import (
    TrackerRunRowBase,
    list_tracker_runs,
    tracker_index,
    tracker_index_path,
)
from mosaic.tracking.common.mint import mint_tracker_run, tracker_run_root
from mosaic.tracking.common.scope import build_work_items
from mosaic.tracking.common.tool_input import resolve_tool_input
from mosaic.tracking.trex.version import TREX_KIND, TREX_VERSION
from mosaic.core.pipeline.index_csv import IndexCSV
from mosaic.core.pipeline.job import CancelToken, JobContext
from mosaic.core.pipeline.markers import (
    PhaseName,
    clear_phase_markers,
)

from .run import run_trex_convert, run_trex_track

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.progress import ProgressCallback


# --- TREx run index -------------------------------------------------------


def trex_run_root(ds: Dataset, run_id: str) -> Path:
    """Where one trex run keeps its per-entry working directories."""
    return tracker_run_root(ds, TREX_KIND, run_id)


def trex_index_path(ds: Dataset) -> Path:
    """Where the trex run index lives."""
    return tracker_index_path(ds, TREX_KIND)


@dataclass(frozen=True, slots=True)
class TRexIndexRow(TrackerRunRowBase):
    """Typed row for the trex run index CSV.

    ``pv_path`` is the tool-specific path column; see :class:`TrackerRunRowBase`
    for how it is stored and where it must be declared.
    """

    pv_path: str = ""


def trex_index(path: Path) -> IndexCSV[TRexIndexRow]:
    """The trex run index, one row per (run, entry)."""
    return tracker_index(path, TRexIndexRow)


# --- Settings, whole and per phase ----------------------------------------

PHASES: Final[tuple[PhaseName, ...]] = ("convert", "track")


def trex_run_id(settings: Mapping[str, object]) -> str:
    """Mint a tracker run identifier from the resolved TREx settings."""
    return op_run_id(TREX_KIND, TREX_VERSION, dict(settings))


# Which settings each TREx task actually consumes, from the two parameter dicts
# built in ``trex/run.py``. ``track_max_individuals`` appears in **both**: it is
# a conversion input despite its name, so changing it must invalidate a
# conversion as well as a tracking. A key in neither set would silently stop
# invalidating anything, which ``trex_settings`` and its test guard against.
CONVERT_KEYS: Final[tuple[str, ...]] = (
    "detect_model",
    "detect_type",
    "detect_conf_threshold",
    "detect_iou_threshold",
    "cm_per_pixel",
    "meta_encoding",
    "convert_extra_settings",
    "track_max_individuals",
)
TRACK_KEYS: Final[tuple[str, ...]] = (
    "track_max_individuals",
    "track_max_speed",
    "track_max_reassign_time",
    "track_trusted_probability",
    "analysis_range",
    "visual_identification_model_path",
    "auto_train",
    "track_extra_settings",
)


def trex_settings(
    *,
    detect_model: Path | str | None,
    detect_type: str,
    detect_conf_threshold: float,
    detect_iou_threshold: float,
    cm_per_pixel: float,
    meta_encoding: str,
    convert_extra_settings: dict[str, Any] | None,
    track_max_individuals: int,
    track_max_speed: float,
    track_max_reassign_time: float,
    track_trusted_probability: float,
    analysis_range: tuple[int, int] | None,
    visual_identification_model_path: Path | str | None,
    auto_train: bool,
    track_extra_settings: dict[str, Any] | None,
) -> dict[str, object]:
    """Build the settings that define a tracking result -- the ``run_id`` payload.

    **Both model references arrive already resolved to an identity** -- a
    training ``run_id`` or a weights content digest, never a path. The caller
    does the resolving, because what these settings must carry is what the model
    *is*: a bare weights path is a mutable key, and swapping ``best.pt`` in place
    would let two different runs share one identifier and report the second as
    already done. This docstring said the opposite for three milestones, which is
    the state the caller had already left behind for ``detect_model``.
    """
    return {
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


def phase_settings(
    settings: Mapping[str, object], phase: PhaseName
) -> dict[str, object]:
    """Project *settings* onto the subset one phase consumes."""
    keys = CONVERT_KEYS if phase == "convert" else TRACK_KEYS
    return {key: settings[key] for key in keys}


# --- Per-entry reuse ------------------------------------------------------


# --- NPZ -> standardized tracks bridge ------------------------------------


def _bridge_npz_to_tracks(
    ds: Dataset,
    group: str,
    sequence: str,
    npz_paths: list[Path],
    *,
    tracks_variant: str,
    producer_run_id: str,
    video_path: Path,
    overwrite: bool,
) -> BridgeCounts | None:
    """Merge per-individual TREx NPZ into ``tracks/<variant>/<group>__<seq>.parquet``.

    TREx is the one tracker whose output is several files per entry: one NPZ per
    individual, which ``merge_on_column_union`` concatenates so a field present
    for one individual and absent for another survives as NaN rather than
    dropping the column. The conversion stays here rather than in the shared
    publisher because the publisher takes one frame, not a set of them.

    Returns ``None`` when there was nothing to convert or the conversion failed.
    """
    from mosaic.core.track_converter import (
        EntryHints,
        get_track_converter,
        merge_on_column_union,
    )

    if not npz_paths:
        return None

    out_path = tracks_table_path(ds, tracks_variant, make_entry_key(group, sequence))
    if out_path.exists() and not overwrite:
        return None

    converter = get_track_converter("trex_npz")
    # The tracker knows the authoritative entry from the media index, so the
    # hints are exact rather than guessed from a filename. No params: the TRex
    # NPZ conversion has none.
    conv_params = type(converter).Params()
    hints = EntryHints(group=group, sequence=sequence)
    frames: list[pd.DataFrame] = []
    for npz in npz_paths:
        try:
            frames.append(converter.convert(npz, conv_params, hints))
        except Exception as exc:
            print(
                f"[run_trex] convert failed for {npz}: {exc}; "
                f"skipping ({group}, {sequence})",
                file=sys.stderr,
            )
            return None
    if not frames:
        return None

    return publish_tracks_table(
        ds,
        merge_on_column_union(frames),
        kind=TREX_KIND,
        group=group,
        sequence=sequence,
        tracks_variant=tracks_variant,
        producer_run_id=producer_run_id,
        source=npz_paths[0].parent,
        consumed=[npz_paths[0], video_path],
    )


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
    idle_timeout: float = 900,
    max_runtime: float | None = None,
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
    # Resolve the detection model *before* the settings that name it, because
    # what the settings must carry is the model's identity and not the string
    # that pointed at it. A bare weights path is a mutable key: swap best.pt and
    # two tracker runs share one variant directory, reporting the second as
    # already done.
    #
    # This is a reordering, and it changes behaviour in one visible way worth
    # stating: an unresolvable model reference now aborts before any run root or
    # tracks variant is recorded, where it used to be swallowed and handed to
    # TREx to complain about. Failing before anything is written is the better
    # half of that trade -- a recorded variant naming a model that could not be
    # found describes a run that never happened.
    detect_model_exec: Path | str | None = detect_model
    detect_model_id: str | None = None
    if detect_model is not None:
        ref = str(detect_model)
        # Ask the identity module, not the string. The old `ref.rsplit("-", 1)[0]`
        # read "train-points.0.1-<digest>" as the kind "train-points.0.1", which
        # is not registered, so resolve_model looked for a path that never
        # existed. A ref that is not a run identifier at all (a bare weights
        # path) falls back rather than guessing, for the same reason.
        parsed = parse_op_run_id(ref)
        model_kind = parsed.kind if parsed is not None else "train-points"
        resolved_model = resolve_model(ds, ref, model_kind)
        detect_model_exec = resolved_model.path
        detect_model_id = resolved_model.model_id

    # The *second* model reference in the same settings dict, and the half item
    # 8.5's "hashed as a string rather than resolved content" describes that was
    # still open: the visual-identification weights were carried as a bare path
    # straight into `TRACK_KEYS`. It is the identical defect -- swap the file and
    # two runs share one identifier -- and it bites harder here than for the
    # detector, because item 8.5 makes the working directory a durable cache and
    # a mutable key on a durable cache never expires.
    vi_model_exec: Path | str | None = visual_identification_model_path
    vi_model_id: str | None = None
    if visual_identification_model_path is not None:
        vi_ref = str(visual_identification_model_path)
        vi_parsed = parse_op_run_id(vi_ref)
        vi_kind = vi_parsed.kind if vi_parsed is not None else "train-identity"
        resolved_vi = resolve_model(ds, vi_ref, vi_kind)
        vi_model_exec = resolved_vi.path
        vi_model_id = resolved_vi.model_id

    # Settings that define the tracking result -> the content hash.
    settings = trex_settings(
        detect_model=detect_model_id,
        detect_type=detect_type,
        detect_conf_threshold=detect_conf_threshold,
        detect_iou_threshold=detect_iou_threshold,
        cm_per_pixel=cm_per_pixel,
        meta_encoding=meta_encoding,
        convert_extra_settings=convert_extra_settings,
        track_max_individuals=track_max_individuals,
        track_max_speed=track_max_speed,
        track_max_reassign_time=track_max_reassign_time,
        track_trusted_probability=track_trusted_probability,
        analysis_range=analysis_range,
        visual_identification_model_path=vi_model_id,
        auto_train=auto_train,
        track_extra_settings=track_extra_settings,
    )
    minted = mint_tracker_run(
        ds, kind=TREX_KIND, version=TREX_VERSION, settings=settings
    )

    # TREx alone gates two phases on different parameter subsets, so it projects
    # its own per-phase digests. The whole-settings params_hash above is what the
    # index row records; these are what each phase marker records, and what a
    # later run compares against to decide the phase may be skipped.
    phase_hashes: dict[PhaseName, str] = {
        phase: hash_params(phase_settings(settings, phase)) for phase in PHASES
    }

    # Route each scoped entry through the transcode verdict: a clean entry
    # resolves to its original, an analysis-required entry to its constant-rate
    # analysis derivative (so tracks land in the same frame space as the rest of
    # the pipeline), and a required-but-unlinked entry raises MediaProbeError
    # here -- before any TREx subprocess opens a known-defective original. TREx
    # decodes the file itself, so only the routed path is needed, not the facts.
    scope = ds.resolve_media_scope(groups, sequences, entries)
    if not scope:
        print("[run_trex] No media entries match the given scope.", file=sys.stderr)
        return minted.run_id

    def convert_and_track(job: EntryJob) -> TRexIndexRow | None:
        """One entry: the gated convert phase, the gated track phase, the bridge.

        TREx is the one tracker with two gated phases, and they are gated on
        *different* parameter subsets, so retuning a track-only knob reuses an
        existing conversion.
        """
        item, work_dir, seq_ctx = job.item, job.work_dir, job.ctx
        cancel_check = seq_ctx.cancel_token.is_cancelled

        # The .results file is the only output TREx writes at the *end* of
        # tracking; the .pv and the per-individual files appear as processing
        # proceeds, so neither distinguishes a finished run from one killed
        # partway. A finished tracking implies a finished conversion, which is
        # what lets one signal adopt both phases.
        adopt_completed_directory(
            job.ds,
            work_dir,
            minted.run_id,
            required=("data/*.npz", "*.results", "*.pv"),
            record=(
                AdoptEvidence("convert", "*.pv"),
                AdoptEvidence("track", "*.results"),
            ),
        )

        reusable_convert = reusable_output(
            job.ds,
            work_dir,
            "convert",
            params_hash=phase_hashes["convert"],
            video_path=item.video_path,
            video_uid=item.video_uid,
        )
        if reusable_convert is None:
            # The tracking phase consumes this phase's output, so a
            # re-conversion invalidates it too.
            clear_phase_markers(work_dir)
            clear_outputs(work_dir, TREX_KIND, "convert")
            clear_outputs(work_dir, TREX_KIND, "track")
            convert_claim = claim(seq_ctx, work_dir, "convert", idle_timeout)
            seq_ctx.progress.on_phase("convert", item.key)
            # T-Rex opens the path itself, so an imgstore recording resolves to
            # the plain video export-store wrote for it. Resolved here rather
            # than before the reuse gate: an entry whose conversion is already
            # reusable needs no export, and demanding one would fail a re-run
            # over work that is finished.
            convert_result = run_trex_convert(
                resolve_tool_input(job.ds, item, kind=TREX_KIND),
                work_dir,
                detect_model=detect_model_exec,
                detect_type=detect_type,
                detect_conf_threshold=detect_conf_threshold,
                detect_iou_threshold=detect_iou_threshold,
                track_max_individuals=track_max_individuals,
                cm_per_pixel=cm_per_pixel,
                meta_encoding=meta_encoding,
                extra_settings=convert_extra_settings,
                idle_timeout=idle_timeout,
                max_runtime=max_runtime,
                trex_conda_env=trex_conda_env,
                trex_bin=trex_bin,
                display=display,
                cancel_check=cancel_check,
                on_output=phase_activity(
                    seq_ctx, work_dir, convert_claim, idle_timeout
                ),
            )
            pv_path = convert_result.pv_path
            _ = record_phase(
                job.ds,
                work_dir,
                "convert",
                ctx=seq_ctx,
                run_id=minted.run_id,
                params_hash=phase_hashes["convert"],
                video_path=item.video_path,
                video_uid=item.video_uid,
                output=pv_path,
            )
        else:
            _convert_marker, pv_path = reusable_convert

        track_marker = reusable_marker(
            job.ds,
            work_dir,
            "track",
            params_hash=phase_hashes["track"],
            video_path=item.video_path,
            video_uid=item.video_uid,
        )
        if track_marker is None:
            clear_outputs(work_dir, TREX_KIND, "track")
            track_claim = claim(seq_ctx, work_dir, "track", idle_timeout)
            seq_ctx.progress.on_phase("track", item.key)
            run_trex_track(
                pv_path,
                work_dir,
                track_max_individuals=track_max_individuals,
                track_max_speed=track_max_speed,
                track_max_reassign_time=track_max_reassign_time,
                track_trusted_probability=track_trusted_probability,
                analysis_range=analysis_range,
                visual_identification_model_path=vi_model_exec,
                auto_train=auto_train,
                extra_settings=track_extra_settings,
                idle_timeout=idle_timeout,
                max_runtime=max_runtime,
                trex_conda_env=trex_conda_env,
                trex_bin=trex_bin,
                display=display,
                cancel_check=cancel_check,
                on_output=phase_activity(seq_ctx, work_dir, track_claim, idle_timeout),
            )
            results = sorted(work_dir.glob("*.results"))
            track_marker = record_phase(
                job.ds,
                work_dir,
                "track",
                ctx=seq_ctx,
                run_id=minted.run_id,
                params_hash=phase_hashes["track"],
                video_path=item.video_path,
                video_uid=item.video_uid,
                output=results[0] if results else None,
            )
            recomputed = True
        else:
            recomputed = False

        data_dir = work_dir / "data"
        npz_paths = sorted(data_dir.glob("*.npz")) if data_dir.is_dir() else []
        row = TRexIndexRow(
            run_id=minted.run_id,
            group=item.group,
            sequence=item.sequence,
            abs_path=Path(job.ds.relative_to_root(work_dir)),
            # From the marker, so the row names what produced the data rather
            # than what the scope resolves to now. The two can only differ when
            # the marker does not know (an adopted directory), since a known
            # mismatch forced the recompute above.
            video_abs_path=(
                track_marker.source or job.ds.relative_to_root(item.video_path)
            ),
            params_hash=minted.params_hash,
            # Re-globbed rather than taken from the bridge, so the count is right
            # even when convert_to_tracks is off and nothing was published.
            n_ids=len(npz_paths),
            pv_path=job.ds.relative_to_root(pv_path),
        )

        if convert_to_tracks:
            # A recomputed entry must replace its parquet: the bridge otherwise
            # declines to overwrite, and the table would keep the results of the
            # run just invalidated.
            _ = _bridge_npz_to_tracks(
                job.ds,
                item.group,
                item.sequence,
                npz_paths,
                tracks_variant=minted.tracks_variant,
                producer_run_id=minted.run_id,
                video_path=item.video_path,
                overwrite=job.overwrite or recomputed,
            )
        return row

    return run_tracker(
        ds,
        kind=TREX_KIND,
        target="trex-track",
        minted=minted,
        work_items=build_work_items(ds, scope, kind=TREX_KIND),
        index=trex_index(trex_index_path(ds)),
        run_entry=convert_and_track,
        overwrite=overwrite,
        execution_id=execution_id,
        owner=owner,
        track=track,
        progress_callback=progress_callback,
        cancel_token=cancel_token,
        ctx=ctx,
    )


def list_trex_runs(ds: Dataset) -> pd.DataFrame:
    """List trex runs tracked in the trex index."""
    return list_tracker_runs(ds, TREX_KIND, TRexIndexRow)


# Item 6.1: the reconciler opens this root's index through the registry, so
# ``core`` never imports ``tracking`` to reach a row class.
register_reconcilable_index(TREX_KIND, trex_index)
