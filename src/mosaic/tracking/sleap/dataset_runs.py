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
import shutil
import socket
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd

from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase
from mosaic.core.pipeline.job import Cancelled, CancelToken, JobContext, job_context
from mosaic.core.pipeline.markers import (
    InflightMarker,
    PhaseMarker,
    clear_inflight,
    clear_phase_marker,
    inflight_state,
    new_inflight,
    read_inflight,
    read_phase_marker,
    refresh_inflight,
    write_inflight,
    write_phase_marker,
)
from mosaic.core.pipeline.dataset_indexes import register_reconcilable_index
from mosaic.core.pipeline.op_identity import op_run_id
from mosaic.core.pipeline.subprocess_util import ProcessCancelled
from mosaic.core.pipeline.tracks_identity import tracks_variant_root
from mosaic.tracking.common.mint import mint_tracker_run
from mosaic.core.pipeline.tracks_index import consumed_roots_for, write_tracks_row
from mosaic.core.schema import ensure_track_schema
from mosaic.runlog import now_iso
from mosaic.tracking.sleap.version import SLEAP_KIND, SLEAP_VERSION

from .run import run_sleap_convert, run_sleap_track

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.progress import ProgressCallback


# --- SLEAP run index ------------------------------------------------------


def sleap_run_root(ds: Dataset, run_id: str) -> Path:
    return ds.get_root("sleap") / run_id


def sleap_index_path(ds: Dataset) -> Path:
    return ds.get_root("sleap") / "index.csv"


@dataclass(frozen=True, slots=True)
class SleapIndexRow(RunIndexRowBase):
    """Typed row for the SLEAP run index CSV.

    ``video_abs_path``, ``slp_path`` and ``analysis_h5_path`` are stored the way
    ``abs_path`` is: dataset-root-relative when inside the dataset, absolute when
    not. Readers resolve them with :meth:`Dataset.resolve_path`. A stored
    absolute would never match a freshly resolved one after a move or a sync
    between machines -- which is the comparison the tracker's reuse guard makes,
    so it would invert into a permanent recompute. A new path column here must
    also be added to ``_SLEAP_INDEX_PATH_COLUMNS`` in ``core/dataset.py``.
    """

    group: str
    sequence: str
    video_abs_path: str
    params_hash: str
    model_id: str = ""
    model_type: str = ""
    n_tracks: int = 0
    slp_path: str = ""
    analysis_h5_path: str = ""


def sleap_index(path: Path) -> IndexCSV[SleapIndexRow]:
    return IndexCSV(path, SleapIndexRow, dedup_keys=["run_id", "group", "sequence"])


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


# How often the per-line activity callback re-stamps the claim / heartbeat.
_INFLIGHT_REFRESH_SECONDS: Final = 15.0


def _phase_activity(
    ctx: JobContext,
    work_dir: Path,
    claim: InflightMarker,
    idle_seconds: float,
) -> Callable[[str], None]:
    """Build the per-output-line liveness callback for a running SLEAP phase.

    Every line SLEAP prints is proof the phase is alive. On a throttle it
    advances the run-log heartbeat and re-stamps the in-flight claim, both
    best-effort -- a missed refresh only shortens the claim, never aborts the run.
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


# --- Per-entry reuse ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _WorkItem:
    """One sequence to track, and what it resolved to."""

    group: str
    sequence: str
    key: str
    video_path: Path
    video_uid: str
    fps: float


# Outputs the inference phase leaves in the working directory: the predictions
# and the analysis export derived from them. A killed inference leaves a partial
# ``.slp`` that must not be mistaken for a finished one, and the ``.h5`` derived
# from a superseded ``.slp`` is stale.
_TRACK_OUTPUT_GLOBS: Final[tuple[str, ...]] = ("*.predictions.slp", "*.analysis.h5")


def _clear_track_outputs(work_dir: Path) -> None:
    for pattern in _TRACK_OUTPUT_GLOBS:
        for path in sorted(work_dir.glob(pattern)):
            path.unlink(missing_ok=True)


def _same_video(ds: Dataset, stored: str, video_path: Path) -> bool:
    """Is *stored* the path *video_path* now resolves to? Both sides are resolved."""
    return ds.resolve_path(stored).resolve() == video_path.resolve()


def _reusable_track_marker(
    ds: Dataset,
    work_dir: Path,
    *,
    params_hash: str,
    video_path: Path,
) -> PhaseMarker | None:
    """The marker proving inference need not run again, or None.

    An empty ``source`` or ``params_hash`` means *unknown* rather than
    *mismatched*, and is not grounds for a recompute -- a marker adopted from a
    directory that predates markers cannot know either.
    """
    marker = read_phase_marker(work_dir, "track")
    if marker is None:
        return None
    if marker.params_hash and marker.params_hash != params_hash:
        return None
    if marker.source and not _same_video(ds, marker.source, video_path):
        return None
    return marker


def _adopt_completed_directory(ds: Dataset, work_dir: Path, run_id: str) -> None:
    """Mark a pre-marker directory complete when it demonstrably holds a finished run.

    Without this, every sequence tracked before markers existed re-runs its
    inference once. The evidence required is both a ``.slp`` and an
    ``.analysis.h5``: the ``.slp`` proves inference finished, the ``.h5`` that the
    analysis export finished, so one signal adopts the whole pipeline. The
    adopted marker records no source and no parameter hash -- an honest unknown
    over a confident guess that would then serve as a cache key.
    """
    if read_phase_marker(work_dir, "track") is not None:
        return
    slp = sorted(work_dir.glob("*.predictions.slp"))
    h5 = sorted(work_dir.glob("*.analysis.h5"))
    if not (slp and h5):
        return
    write_phase_marker(
        work_dir,
        PhaseMarker(
            phase="track",
            run_id=run_id,
            completed_at=now_iso(),
            recorded_output=ds.relative_to_root(slp[0]),
            backfilled=True,
        ),
    )


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
) -> tuple[int, int] | None:
    """Bridge a SLEAP analysis HDF5 into ``tracks/<variant>/<group>__<seq>.parquet``.

    Reuses the registered ``sleap_analysis_h5`` converter with the authoritative
    (group, sequence) known from the media index (no filename guessing). Returns
    ``(n_rows, n_tracks)`` written, or ``None`` if skipped/failed.
    """
    from mosaic.core.track_converter import EntryHints, get_track_converter
    from mosaic.core.track_library.sleap import SleapConvertParams

    variant_root = tracks_variant_root(ds.get_root("tracks"), tracks_variant)
    out_path = variant_root / f"{make_entry_key(group, sequence)}.parquet"
    if out_path.exists() and not overwrite:
        # The table is already there; re-derive its counts from disk rather than
        # returning "unknown", so a reuse run records the same n_tracks the fresh
        # run did instead of overwriting the index row with a zero.
        return _parquet_counts(out_path)

    converter = get_track_converter("sleap_analysis_h5")
    conv_params = SleapConvertParams(fps=fps)
    hints = EntryHints(group=group, sequence=sequence)
    try:
        df = converter.convert(h5_path, conv_params, hints)
    except Exception as exc:
        print(
            f"[run_sleap] convert failed for {h5_path}: {exc}; "
            f"skipping ({group}, {sequence})",
            file=sys.stderr,
        )
        return None
    ensure_track_schema(df, "trex_v1", strict=False, source=f"{group}/{sequence}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    write_tracks_row(
        ds,
        run_id=tracks_variant,
        group=group,
        sequence=sequence,
        out_path=out_path,
        producer=SLEAP_KIND,
        std_format="trex_v1",
        n_rows=int(len(df)),
        producer_run_id=producer_run_id,
        source=h5_path.parent,
        # The video (media root) and the SLEAP predictions (sleap root) are what
        # this table was derived from. The external model directories sit under
        # no dataset root, so they contribute nothing -- their identity is
        # already in the run_id via the settings digest.
        consumed_source_roots=consumed_roots_for(
            ds, [h5_path, video_path, *model_checkpoints]
        ),
    )
    return _frame_counts(df)


def _frame_counts(df: pd.DataFrame) -> tuple[int, int]:
    """``(n_rows, n_distinct_ids)`` for a tracks frame."""
    n_tracks = int(df["id"].nunique()) if "id" in df.columns and len(df) else 0
    return int(len(df)), n_tracks


def _parquet_counts(path: Path) -> tuple[int, int]:
    """``(n_rows, n_tracks)`` read from an existing tracks parquet.

    Reads only the ``id`` column so a reuse run pays a column read, not a full
    table load, to keep the index row's ``n_tracks`` correct.
    """
    try:
        existing = pd.read_parquet(path, columns=["id"])
    except (OSError, ValueError, KeyError):
        return 0, 0
    return _frame_counts(existing)


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
    params_hash = minted.params_hash
    run_id = minted.run_id
    run_root = minted.run_root
    tracks_variant = minted.tracks_variant

    scope = ds.resolve_media_scope(groups, sequences, entries)
    if not scope:
        print("[run_sleap] No media entries match the given scope.", file=sys.stderr)
        return run_id

    fps_default = float(ds.meta.get("fps_default", 30.0))
    frames_arg = f"{analysis_range[0]}-{analysis_range[1]}" if analysis_range else None

    # One work item per (group, sequence); first video when several exist, and
    # cameras collapse onto one output directory (per-camera output is a later
    # phase, as in the TREx integration).
    work_items: list[_WorkItem] = []
    claimed_keys: set[str] = set()
    for entry in scope:
        group, sequence, resolved = entry.group, entry.sequence, entry.resolved
        paths = resolved.paths
        if len(paths) > 1:
            print(
                f"[run_sleap] ({group}, {sequence}) has {len(paths)} videos; using "
                f"the first ({paths[0].name}). Multi-video sequences are not yet "
                f"merged.",
                file=sys.stderr,
            )
        key = make_entry_key(group, sequence)
        if key in claimed_keys:
            print(
                f"[run_sleap] ({group}, {sequence}) camera "
                f"{entry.camera or '<unnamed>'} shares one output directory with "
                f"an earlier camera; skipping it.",
                file=sys.stderr,
            )
            continue
        claimed_keys.add(key)
        facts = resolved.facts
        entry_fps = facts[0].fps if facts and facts[0].fps > 0 else fps_default
        work_items.append(
            _WorkItem(
                group=group,
                sequence=sequence,
                key=key,
                video_path=paths[0],
                video_uid=facts[0].video_uuid if facts else "",
                fps=entry_fps,
            )
        )

    idx = sleap_index(sleap_index_path(ds))
    idx.ensure()

    index_rows: list[SleapIndexRow] = []
    skipped: list[str] = []
    managed: AbstractContextManager[JobContext] = (
        nullcontext(ctx)
        if ctx is not None
        else job_context(
            ds,
            kind="sleap",
            target="sleap-track",
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
                seq_dir.mkdir(parents=True, exist_ok=True)
                slp_path = seq_dir / f"{key}.predictions.slp"
                h5_path = seq_dir / f"{key}.analysis.h5"

                claim = inflight_state(
                    read_inflight(seq_dir),
                    run_log_base=ds.base_dir,
                    execution_id=ctx.execution_id,
                )
                if claim == "live":
                    print(
                        f"[run_sleap] ({group}, {sequence}) is held by another "
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

                    track_marker = _reusable_track_marker(
                        ds, seq_dir, params_hash=params_hash, video_path=video_path
                    )
                    reusable_slp: Path | None = None
                    if track_marker is not None and track_marker.recorded_output:
                        candidate = ds.resolve_path(track_marker.recorded_output)
                        if candidate.exists():
                            reusable_slp = candidate
                    if reusable_slp is None:
                        clear_phase_marker(seq_dir, "track")
                        _clear_track_outputs(seq_dir)
                        track_claim = new_inflight(
                            execution_id=ctx.execution_id,
                            host=socket.gethostname(),
                            pid=os.getpid(),
                            phase="track",
                            idle_seconds=idle_timeout,
                        )
                        write_inflight(seq_dir, track_claim)
                        ctx.progress.on_phase("track", key)
                        track_result = run_sleap_track(
                            video_path,
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
                            cancel_check=cancel_check,
                            on_output=_phase_activity(
                                ctx, seq_dir, track_claim, idle_timeout
                            ),
                        )
                        slp_out = track_result.slp_path
                        write_phase_marker(
                            seq_dir,
                            PhaseMarker(
                                phase="track",
                                run_id=run_id,
                                params_hash=params_hash,
                                execution_id=ctx.execution_id,
                                completed_at=now_iso(),
                                source=ds.relative_to_root(video_path),
                                source_uid=video_uid,
                                recorded_output=ds.relative_to_root(slp_out),
                            ),
                        )
                        recomputed = True
                    else:
                        slp_out = reusable_slp
                        recomputed = False

                    # Analysis export: deterministic and cheap, so ensured rather
                    # than marker-gated. Re-run only when its output is missing or
                    # the inference it derives from was just recomputed.
                    #
                    # Published atomically. sleap-convert (via h5py's "w" mode)
                    # truncates its output the instant it opens, so a killed
                    # export would otherwise leave a partial .h5 at the canonical
                    # path -- and this step, unlike inference, is gated on
                    # existence, not a completion marker. Writing to a temp path
                    # and renaming means the canonical .h5 appears only after a
                    # complete export, so it cannot be a partial file a later
                    # reuse run trusts. The rename to the canonical name also
                    # settles the case where sleap-convert names its output per
                    # video rather than by -o.
                    if recomputed or not h5_path.exists():
                        h5_path.unlink(missing_ok=True)
                        h5_tmp = seq_dir / f"{key}.analysis.h5.partial"
                        h5_tmp.unlink(missing_ok=True)
                        convert_claim = new_inflight(
                            execution_id=ctx.execution_id,
                            host=socket.gethostname(),
                            pid=os.getpid(),
                            phase="track",
                            idle_seconds=idle_timeout,
                        )
                        write_inflight(seq_dir, convert_claim)
                        convert_result = run_sleap_convert(
                            slp_out,
                            h5_tmp,
                            idle_timeout=idle_timeout,
                            max_runtime=max_runtime,
                            sleap_conda_env=sleap_conda_env,
                            sleap_bin=sleap_bin,
                            cancel_check=cancel_check,
                            on_output=_phase_activity(
                                ctx, seq_dir, convert_claim, idle_timeout
                            ),
                        )
                        os.replace(convert_result.analysis_h5_path, h5_path)
                        h5_tmp.unlink(missing_ok=True)
                        h5_out = h5_path
                    else:
                        h5_out = h5_path
                finally:
                    clear_inflight(seq_dir)

                index_rows.append(
                    SleapIndexRow(
                        run_id=run_id,
                        group=group,
                        sequence=sequence,
                        abs_path=Path(ds.relative_to_root(seq_dir)),
                        video_abs_path=(
                            track_marker.source
                            if track_marker is not None and track_marker.source
                            else ds.relative_to_root(video_path)
                        ),
                        params_hash=params_hash,
                        model_id=resolved_models.model_id,
                        model_type=resolved_models.model_type,
                        n_tracks=0,
                        slp_path=ds.relative_to_root(slp_out),
                        analysis_h5_path=ds.relative_to_root(h5_out),
                    )
                )

                if convert_to_tracks:
                    bridged = _bridge_analysis_h5_to_tracks(
                        ds,
                        group,
                        sequence,
                        h5_out,
                        tracks_variant=tracks_variant,
                        producer_run_id=run_id,
                        video_path=video_path,
                        model_checkpoints=resolved_models.checkpoints,
                        fps=item.fps,
                        overwrite=overwrite or recomputed,
                    )
                    if bridged is not None:
                        _n_rows, n_tracks = bridged
                        index_rows[-1] = dataclasses.replace(
                            index_rows[-1], n_tracks=n_tracks
                        )

                ctx.progress.on_entry_end(i + 1, len(work_items), key)
                ctx.heartbeat(i + 1)
        except ProcessCancelled as exc:
            raise Cancelled() from exc
        finally:
            if index_rows:
                idx.append(index_rows)
                idx.mark_finished(run_id)

    held = f", {len(skipped)} held by another execution" if skipped else ""
    print(
        f"[run_sleap] completed run_id={run_id} "
        f"({len(index_rows)}/{len(work_items)} sequences{held}) -> {run_root}"
    )
    return run_id


def list_sleap_runs(ds: Dataset) -> pd.DataFrame:
    """List SLEAP runs tracked in the sleap index."""
    if not ds.has_root("sleap"):
        return pd.DataFrame(columns=[f.name for f in dataclasses.fields(SleapIndexRow)])
    idx_path = sleap_index_path(ds)
    if not idx_path.exists():
        return pd.DataFrame(columns=[f.name for f in dataclasses.fields(SleapIndexRow)])
    return pd.read_csv(idx_path)


# Item 6.1: the reconciler opens this root's index through the registry, so
# ``core`` never imports ``tracking`` to reach a row class.
register_reconcilable_index("sleap", sleap_index)
