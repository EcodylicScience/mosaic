"""Dataset-level SLEAP runs: content-addressed, tracked, tracks-integrated.

``run_sleap(ds, ...)`` is the first-class entry point that turns the standalone
SLEAP CLI wrappers (:mod:`mosaic.tracking.sleap.run`) into a Job-Contract stage,
mirroring :func:`mosaic.tracking.trex.run_trex`:

* it resolves input videos through ``Dataset.resolve_media_scope``, routing each
  entry by its transcode verdict (an analysis-required entry tracks its
  constant-rate analysis derivative, not the defective original);
* resolves the trained SLEAP model directory(ies) to a content digest -- what
  *names* the weights, never the path they sit at -- and computes a
  content-addressed ``run_id`` over the resolved settings, writing run-addressed
  artifacts under ``<sleap_root>/<run_id>/<group>__<seq>/``;
* records the attempt in its JSONL run-log (``kind="sleap"``), reports coarse
  progress, and is cancellable (the subprocess runs in a killable process group);
* bridges the analysis-HDF5 export into standardized
  ``tracks/<variant>/<group>__<seq>.parquet`` via the registered
  ``sleap_analysis_h5`` converter.

There is one expensive, gated phase -- ``track`` (``sleap-track`` inference +
identity tracking, producing a ``.slp``). Its completion marker lets a killed
run resume without re-running inference. The cheap, deterministic analysis
export (``sleap-convert``, producing the ``.h5`` the converter reads) is not
marker-gated: it is re-run only when its output is missing or the inference was
recomputed.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd

from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.index_csv import IndexCSV
from mosaic.core.pipeline.job import CancelToken, JobContext
from mosaic.core.pipeline.markers import (
    clear_phase_marker,
)
from mosaic.core.pipeline.dataset_indexes import register_reconcilable_index
from mosaic.core.pipeline.op_identity import op_run_id
from mosaic.tracking.common.bridge import (
    BridgeCounts,
    existing_counts,
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
from mosaic.tracking.sleap.version import SLEAP_KIND, SLEAP_VERSION

from .run import run_sleap_convert, run_sleap_track

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.progress import ProgressCallback


# --- SLEAP run index ------------------------------------------------------


def sleap_run_root(ds: Dataset, run_id: str) -> Path:
    """Where one sleap run keeps its per-entry working directories."""
    return tracker_run_root(ds, SLEAP_KIND, run_id)


def sleap_index_path(ds: Dataset) -> Path:
    """Where the sleap run index lives."""
    return tracker_index_path(ds, SLEAP_KIND)


@dataclass(frozen=True, slots=True)
class SleapIndexRow(TrackerRunRowBase):
    """Typed row for the sleap run index CSV.

    ``slp_path`` and ``analysis_h5_path`` are the tool-specific path columns; see
    :class:`TrackerRunRowBase` for how they are stored and where they must be
    declared.
    """

    model_id: str = ""
    model_type: str = ""
    slp_path: str = ""
    analysis_h5_path: str = ""


def sleap_index(path: Path) -> IndexCSV[SleapIndexRow]:
    """The sleap run index, one row per (run, entry)."""
    return tracker_index(path, SleapIndexRow)


# --- Model resolution -----------------------------------------------------

# SLEAP checkpoint filenames, in preference order. ``best.ckpt`` is current
# sleap-nn; ``best_model.h5`` is a classic (<=1.4) UNet checkpoint.
_CHECKPOINT_NAMES: Final[tuple[str, ...]] = ("best.ckpt", "best_model.h5")
_CONFIG_NAMES: Final[tuple[str, ...]] = ("training_config.yaml", "training_config.json")
# SLEAP head types, matched against the config text. Longest-first so a
# ``multi_class_topdown`` config is not misread as ``centered_instance``.
_HEAD_TYPES: Final[tuple[str, ...]] = (
    "multi_class_topdown",
    "multi_class_bottomup",
    "centered_instance",
    "single_instance",
    "centroid",
    "bottomup",
)


@dataclass(frozen=True, slots=True)
class ResolvedSleapModels:
    """A resolved SLEAP model reference: what runs, and what names it.

    ``paths`` are the model directories handed to ``sleap-track -m`` (order
    preserved -- top-down passes centroid then centered-instance). ``checkpoints``
    is the weights file inside each, used both for the content digest and for
    ``consumed_source_roots``. ``model_id`` is the identity term the settings
    carry -- a digest of the weights, **never a path**. ``model_type`` is
    provenance read from the config, not identity.
    """

    paths: list[Path]
    checkpoints: list[Path]
    model_id: str
    model_type: str


def _find_checkpoint(model_dir: Path) -> Path:
    """Return the weights file inside *model_dir*, or raise."""
    for name in _CHECKPOINT_NAMES:
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    others = sorted(model_dir.glob("*.ckpt"))
    if others:
        return others[0]
    raise FileNotFoundError(
        f"No SLEAP checkpoint ({' / '.join(_CHECKPOINT_NAMES)}) found in "
        f"model directory: {model_dir}"
    )


def _read_model_type(model_dir: Path) -> str:
    """Best-effort model-type (head) name from the training config, for provenance.

    A token scan over the config *text* rather than a structured parse: it works
    for both the current ``.yaml`` and classic ``.json`` config without a YAML
    dependency, and this is provenance recorded on the row, not identity, so an
    empty string when it cannot be read is acceptable.
    """
    for name in _CONFIG_NAMES:
        config = model_dir / name
        if not config.exists():
            continue
        try:
            text = config.read_text()
        except OSError:
            return ""
        for head in _HEAD_TYPES:
            if head in text:
                return head
        return ""
    return ""


def resolve_sleap_models(model_paths: Sequence[Path | str]) -> ResolvedSleapModels:
    """Resolve external SLEAP model directory(ies) to weights + a content digest.

    Each reference is an external model *directory* (SLEAP models are not mosaic
    training runs), so identity is a digest over the weights, order-sensitive
    across the directories -- top-down's two folders are not interchangeable.
    Resolving here, before anything is minted, means an unresolvable reference
    aborts before any run root or tracks variant is written.
    """
    if not model_paths:
        raise ValueError("run_sleap requires at least one model directory")
    dirs = [Path(m) for m in model_paths]
    checkpoints: list[Path] = []
    digests: list[str] = []
    model_type = ""
    for model_dir in dirs:
        if not model_dir.exists():
            raise FileNotFoundError(
                f"SLEAP model directory does not exist: {model_dir}"
            )
        if not model_dir.is_dir():
            raise NotADirectoryError(
                f"SLEAP model reference is not a directory: {model_dir}"
            )
        checkpoint = _find_checkpoint(model_dir)
        checkpoints.append(checkpoint)
        digests.append(file_digest(checkpoint))
        if not model_type:
            model_type = _read_model_type(model_dir)
    model_id = hash_params({"sleap_weights": digests})
    return ResolvedSleapModels(
        paths=dirs, checkpoints=checkpoints, model_id=model_id, model_type=model_type
    )


# --- Settings -------------------------------------------------------------


def sleap_run_id(settings: Mapping[str, object]) -> str:
    """Mint a tracker run identifier from the resolved SLEAP settings."""
    return op_run_id(SLEAP_KIND, SLEAP_VERSION, dict(settings))


def sleap_settings(
    *,
    model_id: str,
    tracking: bool,
    tracker: str,
    similarity: str,
    match: str,
    track_window: int,
    max_instances: int | None,
    max_tracking: int | None,
    peak_threshold: float,
    analysis_range: tuple[int, int] | None,
    sleap_extra_settings: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build the settings that define a SLEAP tracking result -- the ``run_id`` payload.

    The model is carried as its content digest (``model_id``), never a path. When
    tracking is off, the tracker knobs are dropped from identity so retuning them
    cannot bust a cache they never fed.
    """
    return {
        "model": model_id,
        "tracking": tracking,
        "tracker": tracker if tracking else None,
        "similarity": similarity if tracking else None,
        "match": match if tracking else None,
        "track_window": track_window if tracking else None,
        "max_instances": max_instances,
        "max_tracking": max_tracking,
        "peak_threshold": peak_threshold,
        "analysis_range": list(analysis_range) if analysis_range else None,
        "sleap_extra_settings": sleap_extra_settings,
    }


# --- Per-entry reuse ------------------------------------------------------


# --- analysis-HDF5 -> standardized tracks bridge --------------------------


def _bridge_analysis_h5_to_tracks(
    ds: Dataset,
    group: str,
    sequence: str,
    h5_path: Path,
    *,
    tracks_variant: str,
    producer_run_id: str,
    video_path: Path,
    model_checkpoints: Sequence[Path],
    fps: float,
    overwrite: bool,
) -> BridgeCounts | None:
    """Bridge a SLEAP analysis HDF5 into ``tracks/<variant>/<group>__<seq>.parquet``.

    Uses the registered ``sleap_analysis_h5`` converter with the authoritative
    (group, sequence) known from the media index, so no name is guessed from a
    filename. Returns ``None`` when the conversion failed and nothing was
    published.
    """
    from mosaic.core.track_converter import EntryHints, get_track_converter
    from mosaic.core.track_library.sleap import SleapConvertParams

    out_path = tracks_table_path(ds, tracks_variant, make_entry_key(group, sequence))
    if out_path.exists() and not overwrite:
        return existing_counts(out_path)

    converter = get_track_converter("sleap_analysis_h5")
    try:
        df = converter.convert(
            h5_path,
            SleapConvertParams(fps=fps),
            EntryHints(group=group, sequence=sequence),
        )
    except Exception as exc:
        print(
            f"[run_sleap] convert failed for {h5_path}: {exc}; "
            f"skipping ({group}, {sequence})",
            file=sys.stderr,
        )
        return None

    return publish_tracks_table(
        ds,
        df,
        kind=SLEAP_KIND,
        group=group,
        sequence=sequence,
        tracks_variant=tracks_variant,
        producer_run_id=producer_run_id,
        source=h5_path.parent,
        consumed=[h5_path, video_path, *model_checkpoints],
    )


# --- Public entry point ---------------------------------------------------


def run_sleap(
    ds: Dataset,
    *,
    model_paths: Sequence[Path | str],
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    entries: Iterable[tuple[str, str]] | None = None,
    # tracking
    tracking: bool = True,
    tracker: str = "flow",
    similarity: str = "instance",
    match: str = "hungarian",
    track_window: int = 5,
    max_instances: int | None = None,
    max_tracking: int | None = None,
    peak_threshold: float = 0.2,
    analysis_range: tuple[int, int] | None = None,
    sleap_extra_settings: Mapping[str, object] | None = None,
    # execution
    batch_size: int = 4,
    device: str | None = None,
    idle_timeout: float = 900,
    max_runtime: float | None = None,
    sleap_conda_env: str | None = None,
    sleap_bin: Path | str | None = None,
    overwrite: bool = False,
    convert_to_tracks: bool = True,
    # Job Contract
    execution_id: str | None = None,
    owner: str = "",
    track: bool = True,
    progress_callback: ProgressCallback | None = None,
    cancel_token: CancelToken | None = None,
    # When set, run inside this already-open JobContext instead of opening one --
    # the ``mosaic run --kind sleap`` path (``SleapOp``) hands its ctx here so
    # SLEAP rides the standard runner without double-wrapping the Job Contract.
    ctx: JobContext | None = None,
) -> str:
    """Run SLEAP (infer + track) over scoped videos as a tracked job.

    Returns the content-addressed ``run_id``.
    """
    # Resolve the model *before* the settings that name it, because what the
    # settings carry is the weights' identity, not the paths that pointed at
    # them. An unresolvable reference aborts here, before any run root or tracks
    # variant is recorded -- a recorded variant naming weights that could not be
    # found describes a run that never happened.
    resolved_models = resolve_sleap_models(model_paths)

    settings = sleap_settings(
        model_id=resolved_models.model_id,
        tracking=tracking,
        tracker=tracker,
        similarity=similarity,
        match=match,
        track_window=track_window,
        max_instances=max_instances,
        max_tracking=max_tracking,
        peak_threshold=peak_threshold,
        analysis_range=analysis_range,
        sleap_extra_settings=sleap_extra_settings,
    )
    minted = mint_tracker_run(
        ds,
        kind=SLEAP_KIND,
        version=SLEAP_VERSION,
        settings=settings,
        # Provenance, never identity: the weights digest (already the identity
        # term) and the model type, recorded so a variant is explicable from disk.
        observed={
            "model_id": resolved_models.model_id,
            "model_type": resolved_models.model_type,
        },
    )
    scope = ds.resolve_media_scope(groups, sequences, entries)
    if not scope:
        print("[run_sleap] No media entries match the given scope.", file=sys.stderr)
        return minted.run_id

    # sleap-track takes a frame selection as one "start-end" token.
    frames_arg = f"{analysis_range[0]}-{analysis_range[1]}" if analysis_range else None

    def track_one(job: EntryJob) -> SleapIndexRow | None:
        """One entry: the gated inference phase, the ensured export, the bridge."""
        item, work_dir, seq_ctx = job.item, job.work_dir, job.ctx
        slp_path = work_dir / f"{item.key}.predictions.slp"
        h5_path = work_dir / f"{item.key}.analysis.h5"

        # A .slp alone appears as inference proceeds; the .h5 only after the
        # export finished, so requiring both is what distinguishes a finished
        # pre-marker directory from one killed partway.
        adopt_completed_directory(
            job.ds,
            work_dir,
            minted.run_id,
            required=("*.predictions.slp", "*.analysis.h5"),
            record=(AdoptEvidence("track", "*.predictions.slp"),),
        )

        reusable = reusable_output(
            job.ds,
            work_dir,
            "track",
            params_hash=minted.params_hash,
            video_path=item.video_path,
        )
        if reusable is None:
            clear_phase_marker(work_dir, "track")
            clear_outputs(work_dir, SLEAP_KIND, "track")
            track_claim = claim(seq_ctx, work_dir, "track", idle_timeout)
            seq_ctx.progress.on_phase("track", item.key)
            track_result = run_sleap_track(
                item.video_path,
                slp_path,
                model_paths=resolved_models.paths,
                tracking=tracking,
                tracker=tracker,
                similarity=similarity,
                match=match,
                track_window=track_window,
                max_instances=max_instances,
                max_tracking=max_tracking,
                peak_threshold=peak_threshold,
                batch_size=batch_size,
                frames=frames_arg,
                device=device,
                extra_settings=sleap_extra_settings,
                idle_timeout=idle_timeout,
                max_runtime=max_runtime,
                sleap_conda_env=sleap_conda_env,
                sleap_bin=sleap_bin,
                cancel_check=seq_ctx.cancel_token.is_cancelled,
                on_output=phase_activity(seq_ctx, work_dir, track_claim, idle_timeout),
            )
            slp_out = track_result.slp_path
            track_marker = record_phase(
                job.ds,
                work_dir,
                "track",
                ctx=seq_ctx,
                run_id=minted.run_id,
                params_hash=minted.params_hash,
                video_path=item.video_path,
                video_uid=item.video_uid,
                output=slp_out,
            )
            recomputed = True
        else:
            track_marker, slp_out = reusable
            recomputed = False

        # Analysis export: deterministic and cheap, so ensured rather than
        # marker-gated. Re-run only when its output is missing or the inference
        # it derives from was just recomputed. Published atomically:
        # sleap-convert truncates its output the instant h5py opens it, so a
        # killed export would otherwise leave a partial .h5 at the canonical
        # path -- and this step, unlike inference, is gated on existence rather
        # than on a completion marker. The rename also settles the case where
        # sleap-convert names its output per video rather than by -o.
        if recomputed or not h5_path.exists():
            h5_path.unlink(missing_ok=True)
            h5_tmp = work_dir / f"{item.key}.analysis.h5.partial"
            h5_tmp.unlink(missing_ok=True)
            export_claim = claim(seq_ctx, work_dir, "track", idle_timeout)
            convert_result = run_sleap_convert(
                slp_out,
                h5_tmp,
                idle_timeout=idle_timeout,
                max_runtime=max_runtime,
                sleap_conda_env=sleap_conda_env,
                sleap_bin=sleap_bin,
                cancel_check=seq_ctx.cancel_token.is_cancelled,
                on_output=phase_activity(seq_ctx, work_dir, export_claim, idle_timeout),
            )
            os.replace(convert_result.analysis_h5_path, h5_path)
            h5_tmp.unlink(missing_ok=True)

        row = SleapIndexRow(
            run_id=minted.run_id,
            group=item.group,
            sequence=item.sequence,
            abs_path=Path(job.ds.relative_to_root(work_dir)),
            # From the marker, so the row names what produced the data rather
            # than what the scope resolves to now.
            video_abs_path=(
                track_marker.source
                if track_marker.source
                else job.ds.relative_to_root(item.video_path)
            ),
            params_hash=minted.params_hash,
            model_id=resolved_models.model_id,
            model_type=resolved_models.model_type,
            n_ids=0,
            slp_path=job.ds.relative_to_root(slp_out),
            analysis_h5_path=job.ds.relative_to_root(h5_path),
        )

        if not convert_to_tracks:
            return row
        bridged = _bridge_analysis_h5_to_tracks(
            job.ds,
            item.group,
            item.sequence,
            h5_path,
            tracks_variant=minted.tracks_variant,
            producer_run_id=minted.run_id,
            video_path=item.video_path,
            model_checkpoints=resolved_models.checkpoints,
            fps=item.fps,
            overwrite=job.overwrite or recomputed,
        )
        return row if bridged is None else dataclasses.replace(row, n_ids=bridged.n_ids)

    return run_tracker(
        ds,
        kind=SLEAP_KIND,
        target="sleap-track",
        minted=minted,
        work_items=build_work_items(ds, scope, kind=SLEAP_KIND),
        index=sleap_index(sleap_index_path(ds)),
        run_entry=track_one,
        overwrite=overwrite,
        execution_id=execution_id,
        owner=owner,
        track=track,
        progress_callback=progress_callback,
        cancel_token=cancel_token,
        ctx=ctx,
    )


def list_sleap_runs(ds: Dataset) -> pd.DataFrame:
    """List sleap runs tracked in the sleap index."""
    return list_tracker_runs(ds, SLEAP_KIND, SleapIndexRow)


# Item 6.1: the reconciler opens this root's index through the registry, so
# ``core`` never imports ``tracking`` to reach a row class.
register_reconcilable_index("sleap", sleap_index)
