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

import dataclasses
import json
import os
import shutil
import socket
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from mosaic.core.helpers import make_entry_key, to_safe_name
from mosaic.core.pipeline._utils import hash_params, json_ready
from mosaic.core.pipeline.op_identity import op_run_id, parse_op_run_id
from mosaic.tracking.trex.version import TREX_KIND, TREX_VERSION
from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase
from mosaic.core.pipeline.job import Cancelled, CancelToken, JobContext, job_context
from mosaic.core.pipeline.markers import (
    InflightMarker,
    PhaseMarker,
    PhaseName,
    clear_inflight,
    clear_phase_markers,
    inflight_state,
    new_inflight,
    read_inflight,
    read_phase_marker,
    refresh_inflight,
    write_inflight,
    write_phase_marker,
)
from mosaic.core.pipeline.subprocess_util import ProcessCancelled
from mosaic.core.schema import ensure_track_schema
from mosaic.runlog import now_iso

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
    """Typed row for the TREx run index CSV.

    ``video_abs_path`` and ``pv_path`` are stored the same way ``abs_path`` is:
    dataset-root-relative when the file is inside the dataset, absolute when it
    is not. Readers resolve them with :meth:`Dataset.resolve_path`. A stored
    absolute would never match a freshly resolved one after a move or a sync
    between machines -- which is the comparison the tracker's reuse guard makes,
    so it would invert into a permanent recompute. A new path column here must
    also be added to ``_TREX_INDEX_PATH_COLUMNS`` in ``core/dataset.py``.
    """

    group: str
    sequence: str
    video_abs_path: str
    params_hash: str
    n_individuals: int = 0
    pv_path: str = ""


def trex_index(path: Path) -> IndexCSV[TRexIndexRow]:
    return IndexCSV(path, TRexIndexRow, dedup_keys=["run_id", "group", "sequence"])


# --- Settings, whole and per phase ----------------------------------------

PHASES: Final[tuple[PhaseName, ...]] = ("convert", "track")


def trex_run_id(settings: Mapping[str, object]) -> str:
    """Mint a tracker run identifier from the resolved TREx settings."""
    return op_run_id(TREX_KIND, TREX_VERSION, dict(settings))


# How often the per-line activity callback re-stamps the claim / heartbeat. A
# TREx progress bar can redraw many times a second, so the throttle keeps that
# from becoming a per-line disk write while staying well inside the run-log
# heartbeat cadence and any plausible idle window.
_INFLIGHT_REFRESH_SECONDS: Final = 15.0


def _phase_activity(
    ctx: JobContext,
    work_dir: Path,
    claim: InflightMarker,
    idle_seconds: float,
) -> Callable[[str], None]:
    """Build the per-output-line liveness callback for a running TREx phase.

    Every line TREx prints is proof the phase is alive. On a throttle it (a)
    advances the run-log heartbeat, so the queue reaper does not read a live
    multi-hour subprocess as lost, and (b) re-stamps the in-flight claim, so a
    concurrent execution does not read the working directory as abandoned. Both
    are the same activity-based liveness the inactivity watchdog uses, and both
    are best-effort: a missed refresh only shortens the claim, never aborts the
    run. Runs on the subprocess reader thread; ``ctx``/``run_log`` are untouched
    by the main thread for the duration of the phase.
    """
    last_refresh = [0.0]

    def _on_line(_line: str) -> None:
        now = time.monotonic()
        if now - last_refresh[0] < _INFLIGHT_REFRESH_SECONDS:
            return
        last_refresh[0] = now
        ctx.heartbeat()
        try:
            _ = refresh_inflight(work_dir, claim, idle_seconds)
        except OSError:
            pass

    return _on_line


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

    A model reference is kept as written rather than resolved: the reference is
    portable across machines, the path it resolves to is not.
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


@dataclass(frozen=True, slots=True)
class _WorkItem:
    """One sequence to track, and what it resolved to.

    ``key`` is the ``<group>__<sequence>`` working-directory name; there is
    exactly one item per key.
    """

    group: str
    sequence: str
    key: str
    video_path: Path
    video_uid: str


# Outputs each phase leaves in the working directory. Used to clear a phase
# before re-running it -- the .pv and the per-individual files are written as
# processing proceeds, so a killed phase leaves partial ones behind that must
# not be mistaken for, or merged with, the new run's.
_CONVERT_OUTPUT_GLOBS: Final[tuple[str, ...]] = ("*.pv", "*.settings", "average_*.png")
_TRACK_OUTPUT_GLOBS: Final[tuple[str, ...]] = ("*.results", "data/*.npz")


def _clear_phase_outputs(work_dir: Path, phase: PhaseName) -> None:
    globs = _CONVERT_OUTPUT_GLOBS if phase == "convert" else _TRACK_OUTPUT_GLOBS
    for pattern in globs:
        for path in sorted(work_dir.glob(pattern)):
            path.unlink(missing_ok=True)


def _same_video(ds: Dataset, stored: str, video_path: Path) -> bool:
    """Is *stored* the path *video_path* now resolves to?

    Both sides go through resolution: a stored value may be root-relative (the
    portable form) or a legacy absolute, and the dataset may have moved. A raw
    string comparison would call every relocated dataset a source change.
    """
    return ds.resolve_path(stored).resolve() == video_path.resolve()


def _reusable_marker(
    ds: Dataset,
    work_dir: Path,
    phase: PhaseName,
    *,
    phase_hash: str,
    video_path: Path,
) -> PhaseMarker | None:
    """The marker proving *phase* need not run again, or None.

    An empty ``source`` or ``params_hash`` means *unknown* rather than
    *mismatched*, and is not grounds for a recompute -- a marker adopted from a
    directory that predates markers cannot know either.
    """
    marker = read_phase_marker(work_dir, phase)
    if marker is None:
        return None
    if marker.params_hash and marker.params_hash != phase_hash:
        return None
    if marker.source and not _same_video(ds, marker.source, video_path):
        return None
    return marker


def _adopt_completed_directory(ds: Dataset, work_dir: Path, run_id: str) -> None:
    """Mark a pre-marker directory complete, when it demonstrably holds a finished run.

    Without this, every sequence already tracked before markers existed re-runs
    once -- hours of TREx apiece on a real dataset.

    The evidence required is a ``.results`` file, because it is the only output
    TREx writes at the *end* of tracking: the ``.pv`` and the per-individual
    files appear as processing proceeds, so neither distinguishes a finished run
    from one killed partway. A finished tracking also implies a finished
    conversion, which is what lets one signal adopt both phases.

    The adopted markers record no source and no parameter hash. Nothing on disk
    says what the directory was computed from, and an honest unknown is better
    than a confident guess that would then serve as a cache key. The
    consequence, which is inherent rather than a gap to close: an adopted
    directory is not protected by the source-video guard, because there is
    nothing to compare against. It is no worse off than before markers existed,
    and any directory produced *after* this lands carries a real source.
    """
    if any(read_phase_marker(work_dir, phase) is not None for phase in PHASES):
        return
    data_dir = work_dir / "data"
    npz_paths = sorted(data_dir.glob("*.npz")) if data_dir.is_dir() else []
    results = sorted(work_dir.glob("*.results"))
    pv_matches = sorted(work_dir.glob("*.pv"))
    if not (npz_paths and results and pv_matches):
        return

    stamp = now_iso()
    write_phase_marker(
        work_dir,
        PhaseMarker(
            phase="convert",
            run_id=run_id,
            completed_at=stamp,
            recorded_output=ds.relative_to_root(pv_matches[0]),
            backfilled=True,
        ),
    )
    write_phase_marker(
        work_dir,
        PhaseMarker(
            phase="track",
            run_id=run_id,
            completed_at=stamp,
            recorded_output=ds.relative_to_root(results[0]),
            backfilled=True,
        ),
    )


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
    from mosaic.core.track_converter import EntryHints, get_track_converter

    if not npz_paths:
        return None

    out_path = ds.get_root("tracks") / f"{make_entry_key(group, sequence)}.parquet"
    if out_path.exists() and not overwrite:
        return None

    converter = get_track_converter("trex_npz")
    # The tracker knows the authoritative entry from the media index, so the
    # hints are exact rather than guessed from a filename. No params: the TRex
    # NPZ conversion has none.
    conv_params = type(converter).Params()
    hints = EntryHints(group=group, sequence=sequence)
    dfs: list[pd.DataFrame] = []
    for npz in npz_paths:
        try:
            dfs.append(converter.convert(npz, conv_params, hints))
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
    if not ds.has_root("trex"):
        ds.set_root("trex", "tracks_raw/trex")

    # Settings that define the tracking result -> the content hash.
    settings = trex_settings(
        detect_model=detect_model,
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
        visual_identification_model_path=visual_identification_model_path,
        auto_train=auto_train,
        track_extra_settings=track_extra_settings,
    )
    params_hash = hash_params(settings)
    phase_hashes: dict[PhaseName, str] = {
        phase: hash_params(phase_settings(settings, phase)) for phase in PHASES
    }
    run_id = trex_run_id(settings)
    run_root = trex_run_root(ds, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    # Resolve a training run_id (e.g. "train-points-<hash>") to its best.pt weights for the
    # trex invocation -- the train->track handoff. The run_id hash above intentionally keys on
    # the original reference (portable across machines), not the resolved absolute path.
    detect_model_exec: Path | str | None = detect_model
    if detect_model is not None and not Path(str(detect_model)).exists():
        from mosaic.tracking.model_refs import resolve_model

        ref = str(detect_model)
        # Ask the identity module, not the string. The old `ref.rsplit("-", 1)[0]`
        # read "train-points.0.1-<digest>" as the kind "train-points.0.1", which
        # is not registered, so resolve_model looked for a path that never
        # existed. A ref that is not a run identifier at all (a bare weights
        # path) falls back rather than guessing, for the same reason.
        parsed = parse_op_run_id(ref)
        model_kind = parsed.kind if parsed is not None else "train-points"
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

    # One work item per (group, sequence); first video when several exist. Each
    # scope entry is one camera; per-camera TREx output (a camera-qualified
    # seq_dir and a camera in the trex index dedup) is Phase 2, gated on the
    # store->mp4 transcode a store-directory sequence needs to be TREx-readable
    # at all -- so single-camera behavior here is unchanged.
    #
    # The working directory is keyed on (group, sequence) with no camera, so a
    # multi-camera sequence's entries all resolve to *one* directory. Collapsing
    # them here keeps that one-to-one: left as several, the second entry would
    # see the first's source, call it a change, recompute over the first's
    # outputs and replace its index row -- on every run, forever.
    work_items: list[_WorkItem] = []
    claimed_keys: set[str] = set()
    for entry in scope:
        group, sequence, resolved = entry.group, entry.sequence, entry.resolved
        paths = resolved.paths
        if len(paths) > 1:
            print(
                f"[run_trex] ({group}, {sequence}) has {len(paths)} videos; using "
                f"the first ({paths[0].name}). Multi-video sequences are not yet "
                f"merged.",
                file=sys.stderr,
            )
        key = make_entry_key(group, sequence)
        if key in claimed_keys:
            print(
                f"[run_trex] ({group}, {sequence}) camera "
                f"{entry.camera or '<unnamed>'} shares one output directory with "
                f"an earlier camera; skipping it. Per-camera tracker output is "
                f"Phase 2.",
                file=sys.stderr,
            )
            continue
        claimed_keys.add(key)
        facts = resolved.facts
        work_items.append(
            _WorkItem(
                group=group,
                sequence=sequence,
                key=key,
                video_path=paths[0],
                # Advisory in this milestone: recorded so items 4.2 and 8.5 have
                # it, not yet compared. A media index written before the identity
                # columns existed carries none, and comparing on an absent value
                # would change behavior for every such dataset.
                video_uid=facts[0].video_uuid if facts else "",
            )
        )

    idx = trex_index(trex_index_path(ds))
    idx.ensure()

    index_rows: list[TRexIndexRow] = []
    skipped: list[str] = []
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
            for i, item in enumerate(work_items):
                ctx.check_cancel()
                group, sequence = item.group, item.sequence
                key, video_path, video_uid = item.key, item.video_path, item.video_uid
                ctx.progress.on_entry_start(i, len(work_items), key)
                seq_dir = run_root / key

                # The directory is no longer a completion signal, so creating it
                # up front costs nothing -- and the in-flight claim lives in it.
                seq_dir.mkdir(parents=True, exist_ok=True)

                # Before any destructive step, including overwrite's rmtree: a
                # live claim means another execution is writing here right now,
                # and clearing the directory first would delete both its work
                # and the claim that says so.
                claim = inflight_state(
                    read_inflight(seq_dir),
                    run_log_base=ds.base_dir,
                    execution_id=ctx.execution_id,
                )
                if claim == "live":
                    # Skip rather than raise: one contended sequence must not
                    # end a batch, and its directory is not ours to write.
                    print(
                        f"[run_trex] ({group}, {sequence}) is held by another "
                        f"execution; skipping it.",
                        file=sys.stderr,
                    )
                    skipped.append(key)
                    ctx.progress.on_entry_end(i + 1, len(work_items), key)
                    continue

                if overwrite:
                    shutil.rmtree(seq_dir)
                    seq_dir.mkdir(parents=True, exist_ok=True)

                _adopt_completed_directory(ds, seq_dir, run_id)
                try:
                    write_inflight(
                        seq_dir,
                        new_inflight(
                            execution_id=ctx.execution_id,
                            host=socket.gethostname(),
                            pid=os.getpid(),
                            phase=None,
                            idle_seconds=idle_timeout,
                        ),
                    )

                    convert_marker = _reusable_marker(
                        ds,
                        seq_dir,
                        "convert",
                        phase_hash=phase_hashes["convert"],
                        video_path=video_path,
                    )
                    # A conversion is only reusable if its output is still
                    # there to reuse -- and where it is is recorded, not
                    # globbed, since TREx may leave it beside the source video.
                    reusable_pv: Path | None = None
                    if convert_marker is not None and convert_marker.recorded_output:
                        candidate = ds.resolve_path(convert_marker.recorded_output)
                        if candidate.exists():
                            reusable_pv = candidate
                    if reusable_pv is None:
                        # The tracking phase consumes this phase's output, so a
                        # re-conversion invalidates it too.
                        clear_phase_markers(seq_dir)
                        _clear_phase_outputs(seq_dir, "convert")
                        _clear_phase_outputs(seq_dir, "track")
                        convert_claim = new_inflight(
                            execution_id=ctx.execution_id,
                            host=socket.gethostname(),
                            pid=os.getpid(),
                            phase="convert",
                            idle_seconds=idle_timeout,
                        )
                        write_inflight(seq_dir, convert_claim)
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
                            idle_timeout=idle_timeout,
                            max_runtime=max_runtime,
                            trex_conda_env=trex_conda_env,
                            trex_bin=trex_bin,
                            display=display,
                            cancel_check=cancel_check,
                            on_output=_phase_activity(
                                ctx, seq_dir, convert_claim, idle_timeout
                            ),
                        )
                        pv_path = convert_result.pv_path
                        write_phase_marker(
                            seq_dir,
                            PhaseMarker(
                                phase="convert",
                                run_id=run_id,
                                params_hash=phase_hashes["convert"],
                                execution_id=ctx.execution_id,
                                completed_at=now_iso(),
                                source=ds.relative_to_root(video_path),
                                source_uid=video_uid,
                                recorded_output=ds.relative_to_root(pv_path),
                            ),
                        )

                    else:
                        pv_path = reusable_pv

                    track_marker = _reusable_marker(
                        ds,
                        seq_dir,
                        "track",
                        phase_hash=phase_hashes["track"],
                        video_path=video_path,
                    )
                    if track_marker is None:
                        _clear_phase_outputs(seq_dir, "track")
                        track_claim = new_inflight(
                            execution_id=ctx.execution_id,
                            host=socket.gethostname(),
                            pid=os.getpid(),
                            phase="track",
                            idle_seconds=idle_timeout,
                        )
                        write_inflight(seq_dir, track_claim)
                        ctx.progress.on_phase("track", key)
                        run_trex_track(
                            pv_path,
                            seq_dir,
                            track_max_individuals=track_max_individuals,
                            track_max_speed=track_max_speed,
                            track_max_reassign_time=track_max_reassign_time,
                            track_trusted_probability=track_trusted_probability,
                            analysis_range=analysis_range,
                            visual_identification_model_path=visual_identification_model_path,
                            auto_train=auto_train,
                            extra_settings=track_extra_settings,
                            idle_timeout=idle_timeout,
                            max_runtime=max_runtime,
                            trex_conda_env=trex_conda_env,
                            trex_bin=trex_bin,
                            display=display,
                            cancel_check=cancel_check,
                            on_output=_phase_activity(
                                ctx, seq_dir, track_claim, idle_timeout
                            ),
                        )
                        results = sorted(seq_dir.glob("*.results"))
                        track_marker = PhaseMarker(
                            phase="track",
                            run_id=run_id,
                            params_hash=phase_hashes["track"],
                            execution_id=ctx.execution_id,
                            completed_at=now_iso(),
                            source=ds.relative_to_root(video_path),
                            source_uid=video_uid,
                            recorded_output=(
                                ds.relative_to_root(results[0]) if results else ""
                            ),
                        )
                        write_phase_marker(seq_dir, track_marker)
                        recomputed = True
                    else:
                        recomputed = False
                finally:
                    clear_inflight(seq_dir)

                data_dir = seq_dir / "data"
                npz_paths = sorted(data_dir.glob("*.npz")) if data_dir.is_dir() else []
                index_rows.append(
                    TRexIndexRow(
                        run_id=run_id,
                        group=group,
                        sequence=sequence,
                        abs_path=Path(ds.relative_to_root(seq_dir)),
                        # From the marker, so the row names what produced the
                        # data rather than what the scope resolves to now. The
                        # two can only differ when the marker does not know
                        # (an adopted directory), since a known mismatch forced
                        # the recompute above.
                        video_abs_path=(
                            track_marker.source or ds.relative_to_root(video_path)
                        ),
                        params_hash=params_hash,
                        n_individuals=len(npz_paths),
                        pv_path=ds.relative_to_root(pv_path),
                    )
                )

                if convert_to_tracks:
                    # A recomputed entry must replace its parquet: the bridge
                    # otherwise declines to overwrite, and the tracks table
                    # would keep the results of the run just invalidated.
                    _bridge_npz_to_tracks(
                        ds,
                        group,
                        sequence,
                        npz_paths,
                        overwrite=overwrite or recomputed,
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

    held = f", {len(skipped)} held by another execution" if skipped else ""
    print(
        f"[run_trex] completed run_id={run_id} "
        f"({len(index_rows)}/{len(work_items)} sequences{held}) -> {run_root}"
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
