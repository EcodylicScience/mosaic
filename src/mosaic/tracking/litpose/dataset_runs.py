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
import json
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
from mosaic.core.pipeline._utils import hash_params, json_ready
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.identity_scheme import write_identity_scheme
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
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME, op_run_id
from mosaic.core.pipeline.subprocess_util import ProcessCancelled
from mosaic.core.pipeline.tracks_identity import (
    litpose_variant_payload,
    tracks_run_id,
    tracks_variant_root,
    write_tracks_variant,
)
from mosaic.core.pipeline.tracks_index import consumed_roots_for, write_tracks_row
from mosaic.core.schema import ensure_track_schema
from mosaic.runlog import now_iso
from mosaic.tracking.litpose.version import LITPOSE_KIND, LITPOSE_VERSION

from .run import run_litpose_predict

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.progress import ProgressCallback


# --- Lightning Pose run index --------------------------------------------


def litpose_run_root(ds: Dataset, run_id: str) -> Path:
    return ds.get_root("litpose") / run_id


def litpose_index_path(ds: Dataset) -> Path:
    return ds.get_root("litpose") / "index.csv"


@dataclass(frozen=True, slots=True)
class LitposeIndexRow(RunIndexRowBase):
    """Typed row for the Lightning Pose run index CSV.

    ``video_abs_path`` and ``csv_path`` are stored the way ``abs_path`` is:
    dataset-root-relative when inside the dataset, absolute when not. Readers
    resolve them with :meth:`Dataset.resolve_path`. A stored absolute would never
    match a freshly resolved one after a move or a sync between machines -- which
    is the comparison the tracker's reuse guard makes, so it would invert into a
    permanent recompute. A new path column here must also be added to
    ``_LITPOSE_INDEX_PATH_COLUMNS`` in ``core/dataset.py``.
    """

    group: str
    sequence: str
    video_abs_path: str
    params_hash: str
    model_id: str = ""
    model_type: str = ""
    n_individuals: int = 1
    csv_path: str = ""


def litpose_index(path: Path) -> IndexCSV[LitposeIndexRow]:
    return IndexCSV(path, LitposeIndexRow, dedup_keys=["run_id", "group", "sequence"])


# --- Model resolution -----------------------------------------------------

# Lightning Pose model-type (head) names, matched against the config text.
# Longest-first so ``heatmap_mhcrnn`` is not misread as ``heatmap``.
_MODEL_TYPES: Final[tuple[str, ...]] = (
    "heatmap_multiview_transformer",
    "heatmap_mhcrnn",
    "regression",
    "heatmap",
)


@dataclass(frozen=True, slots=True)
class ResolvedLitposeModel:
    """A resolved Lightning Pose model reference: what runs, and what names it.

    ``path`` is the model directory handed to inference. ``checkpoint`` and
    ``config`` are the weights and the ``config.yaml`` inside it, used both for the
    content digest and for ``consumed_source_roots``. ``model_id`` is the identity
    term the settings carry -- a digest of the weights *and* config (the config
    carries ``image_resize_dims`` / ``keypoint_names``, which change the output),
    **never a path**. ``model_type`` is provenance read from the config, not
    identity.
    """

    path: Path
    checkpoint: Path
    config: Path
    model_id: str
    model_type: str


def _find_checkpoint(model_dir: Path) -> Path:
    """Return the weights file inside *model_dir*, or raise.

    Prefers a ``*best*`` checkpoint under the canonical
    ``tb_logs/<name>/version_*/checkpoints/`` layout, deterministically (sorted),
    so a fixed model directory always resolves to the same weights.
    """
    candidates = sorted(model_dir.glob("tb_logs/*/version_*/checkpoints/*.ckpt"))
    if not candidates:
        candidates = sorted(model_dir.glob("**/*.ckpt"))
    if not candidates:
        raise FileNotFoundError(
            f"No Lightning Pose checkpoint (tb_logs/.../checkpoints/*.ckpt) found "
            f"in model directory: {model_dir}"
        )
    best = [c for c in candidates if "best" in c.name.lower()]
    return best[-1] if best else candidates[-1]


def _read_model_type(config: Path) -> str:
    """Best-effort model-type (head) name from the config text, for provenance.

    A token scan over the config *text* rather than a structured parse: it works
    without a YAML dependency, and this is provenance recorded on the row, not
    identity, so an empty string when it cannot be read is acceptable.
    """
    try:
        text = config.read_text()
    except OSError:
        return ""
    for model_type in _MODEL_TYPES:
        if model_type in text:
            return model_type
    return ""


def resolve_litpose_model(model_path: Path | str) -> ResolvedLitposeModel:
    """Resolve an external Lightning Pose model directory to weights + a content digest.

    The reference is an external model *directory* (Lightning Pose models are not
    mosaic training runs), so identity is a digest over the checkpoint and the
    ``config.yaml``. Resolving here, before anything is minted, means an
    unresolvable reference aborts before any run root or tracks variant is written.
    The whole directory is deliberately *not* fingerprinted: Lightning Pose writes
    ``video_preds/`` into it, which would spuriously bust identity.
    """
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Lightning Pose model directory does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(
            f"Lightning Pose model reference is not a directory: {model_dir}"
        )
    config = model_dir / "config.yaml"
    if not config.exists():
        raise FileNotFoundError(
            f"Lightning Pose model directory has no config.yaml: {model_dir}"
        )
    checkpoint = _find_checkpoint(model_dir)
    model_id = hash_params(
        {"litpose_config": file_digest(config), "litpose_weights": file_digest(checkpoint)}
    )
    return ResolvedLitposeModel(
        path=model_dir,
        checkpoint=checkpoint,
        config=config,
        model_id=model_id,
        model_type=_read_model_type(config),
    )


# --- Settings -------------------------------------------------------------


def litpose_run_id(settings: Mapping[str, object]) -> str:
    """Mint a tracker run identifier from the resolved Lightning Pose settings."""
    return op_run_id(LITPOSE_KIND, LITPOSE_VERSION, dict(settings))


def litpose_settings(
    *,
    model_id: str,
    litpose_overrides: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build the settings that define a Lightning Pose result -- the ``run_id`` payload.

    The model is carried as its content digest (``model_id``), never a path.
    Lightning Pose is pose-only, so there are no tracker knobs; the Hydra
    ``litpose_overrides`` are identity because they change the produced keypoints.
    """
    return {
        "model": model_id,
        "litpose_overrides": dict(litpose_overrides) if litpose_overrides else None,
    }


# How often the per-line activity callback re-stamps the claim / heartbeat.
_INFLIGHT_REFRESH_SECONDS: Final = 15.0


def _phase_activity(
    ctx: JobContext,
    work_dir: Path,
    claim: InflightMarker,
    idle_seconds: float,
) -> Callable[[str], None]:
    """Build the per-output-line liveness callback for a running Lightning Pose phase.

    Every line Lightning Pose prints is proof the phase is alive. On a throttle it
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


# The output the inference phase leaves in the working directory: the predictions
# CSV. A killed inference leaves a partial CSV that must not be mistaken for a
# finished one.
_TRACK_OUTPUT_GLOBS: Final[tuple[str, ...]] = ("*.predictions.csv",)


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
    inference once. The evidence required is the predictions CSV -- Lightning Pose
    has a single output, so one signal adopts the whole run. The adopted marker
    records no source and no parameter hash -- an honest unknown over a confident
    guess that would then serve as a cache key.
    """
    if read_phase_marker(work_dir, "track") is not None:
        return
    csv = sorted(work_dir.glob("*.predictions.csv"))
    if not csv:
        return
    write_phase_marker(
        work_dir,
        PhaseMarker(
            phase="track",
            run_id=run_id,
            completed_at=now_iso(),
            recorded_output=ds.relative_to_root(csv[0]),
            backfilled=True,
        ),
    )


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
) -> tuple[int, int] | None:
    """Bridge a Lightning Pose CSV into ``tracks/<variant>/<group>__<seq>.parquet``.

    Reuses the registered ``deeplabcut`` converter (Lightning Pose exports the same
    ``(scorer, bodypart, coord)`` layout) with the authoritative (group, sequence)
    known from the media index (no filename guessing). Returns ``(n_rows,
    n_individuals)`` written, or ``None`` if skipped/failed.
    """
    from mosaic.core.track_converter import EntryHints, get_track_converter
    from mosaic.core.track_library.deeplabcut import DlcParams

    variant_root = tracks_variant_root(ds.get_root("tracks"), tracks_variant)
    out_path = variant_root / f"{make_entry_key(group, sequence)}.parquet"
    if out_path.exists() and not overwrite:
        # The table is already there; re-derive its counts from disk rather than
        # returning "unknown", so a reuse run records the same count the fresh run
        # did instead of overwriting the index row with a zero.
        return _parquet_counts(out_path)

    converter = get_track_converter("deeplabcut")
    conv_params = DlcParams(fps=fps)
    hints = EntryHints(group=group, sequence=sequence)
    try:
        df = converter.convert(csv_path, conv_params, hints)
    except Exception as exc:
        print(
            f"[run_litpose] convert failed for {csv_path}: {exc}; "
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
        producer=LITPOSE_KIND,
        std_format="trex_v1",
        n_rows=int(len(df)),
        producer_run_id=producer_run_id,
        source=csv_path.parent,
        # The video (media root) and the predictions CSV (litpose root) are what
        # this table was derived from. The external model directory sits under no
        # dataset root, so it contributes nothing -- its identity is already in the
        # run_id via the settings digest.
        consumed_source_roots=consumed_roots_for(ds, [csv_path, video_path, *model_files]),
    )
    return _frame_counts(df)


def _frame_counts(df: pd.DataFrame) -> tuple[int, int]:
    """``(n_rows, n_distinct_ids)`` for a tracks frame."""
    n_individuals = int(df["id"].nunique()) if "id" in df.columns and len(df) else 0
    return int(len(df)), n_individuals


def _parquet_counts(path: Path) -> tuple[int, int]:
    """``(n_rows, n_individuals)`` read from an existing tracks parquet.

    Reads only the ``id`` column so a reuse run pays a column read, not a full
    table load, to keep the index row's count correct.
    """
    try:
        existing = pd.read_parquet(path, columns=["id"])
    except (OSError, ValueError, KeyError):
        return 0, 0
    return _frame_counts(existing)


# --- Public entry point ---------------------------------------------------


def run_litpose(
    ds: Dataset,
    *,
    model_path: Path | str,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    entries: Iterable[tuple[str, str]] | None = None,
    litpose_overrides: Mapping[str, object] | None = None,
    # execution
    precision: str = "fp32",
    idle_timeout: float = 900,
    max_runtime: float | None = None,
    litpose_conda_env: str | None = None,
    litpose_bin: Path | str | None = None,
    overwrite: bool = False,
    convert_to_tracks: bool = True,
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

    Returns the content-addressed ``run_id``.
    """
    if not ds.has_root("litpose"):
        ds.set_root("litpose", "_tracking/litpose")

    # Resolve the model *before* the settings that name it, because what the
    # settings carry is the weights' identity, not the path that pointed at them.
    # An unresolvable reference aborts here, before any run root or tracks variant
    # is recorded.
    resolved_model = resolve_litpose_model(model_path)

    settings = litpose_settings(
        model_id=resolved_model.model_id, litpose_overrides=litpose_overrides
    )
    params_hash = hash_params(settings)
    run_id = litpose_run_id(settings)
    run_root = litpose_run_root(ds, run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    write_identity_scheme(run_root, OP_IDENTITY_SCHEME)

    # What names the *tracks variant* these tables belong to, as distinct from the
    # tracker run that produced them. Minted once: the settings are scope-free, so
    # one value covers every sequence the run touches. Passed unwrapped, so it is
    # byte-identical to ``run_id``.
    tracks_variant = tracks_run_id(
        LITPOSE_KIND, LITPOSE_VERSION, litpose_variant_payload(settings)
    )
    _ = write_tracks_variant(
        ds.get_root("tracks"),
        tracks_variant,
        LITPOSE_KIND,
        LITPOSE_VERSION,
        settings,
        # Provenance, never identity: the model digest (already the identity term)
        # and the model type, recorded so a variant is explicable from disk.
        observed={
            "model_id": resolved_model.model_id,
            "model_type": resolved_model.model_type,
        },
    )

    params_path = run_root / "run_params.json"
    try:
        params_path.write_text(json.dumps(json_ready(settings), indent=2))
    except Exception as exc:
        print(f"[run_litpose] failed to save run_params.json: {exc}", file=sys.stderr)

    scope = ds.resolve_media_scope(groups, sequences, entries)
    if not scope:
        print("[run_litpose] No media entries match the given scope.", file=sys.stderr)
        return run_id

    fps_default = float(ds.meta.get("fps_default", 30.0))

    # One work item per (group, sequence); first video when several exist, and
    # cameras collapse onto one output directory (per-camera output is a later
    # phase, as in the SLEAP / TREx integrations).
    work_items: list[_WorkItem] = []
    claimed_keys: set[str] = set()
    for entry in scope:
        group, sequence, resolved = entry.group, entry.sequence, entry.resolved
        paths = resolved.paths
        if len(paths) > 1:
            print(
                f"[run_litpose] ({group}, {sequence}) has {len(paths)} videos; using "
                f"the first ({paths[0].name}). Multi-video sequences are not yet "
                f"merged.",
                file=sys.stderr,
            )
        key = make_entry_key(group, sequence)
        if key in claimed_keys:
            print(
                f"[run_litpose] ({group}, {sequence}) camera "
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

    idx = litpose_index(litpose_index_path(ds))
    idx.ensure()

    index_rows: list[LitposeIndexRow] = []
    skipped: list[str] = []
    managed: AbstractContextManager[JobContext] = (
        nullcontext(ctx)
        if ctx is not None
        else job_context(
            ds,
            kind="litpose",
            target="litpose-predict",
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
                csv_path = seq_dir / f"{key}.predictions.csv"

                claim = inflight_state(
                    read_inflight(seq_dir),
                    run_log_base=ds.base_dir,
                    execution_id=ctx.execution_id,
                )
                if claim == "live":
                    print(
                        f"[run_litpose] ({group}, {sequence}) is held by another "
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
                    reusable_csv: Path | None = None
                    if track_marker is not None and track_marker.recorded_output:
                        candidate = ds.resolve_path(track_marker.recorded_output)
                        if candidate.exists():
                            reusable_csv = candidate
                    if reusable_csv is None:
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
                        predict_result = run_litpose_predict(
                            video_path,
                            csv_path,
                            model_dir=resolved_model.path,
                            precision=precision,
                            overrides=litpose_overrides,
                            idle_timeout=idle_timeout,
                            max_runtime=max_runtime,
                            litpose_conda_env=litpose_conda_env,
                            litpose_bin=litpose_bin,
                            cancel_check=cancel_check,
                            on_output=_phase_activity(
                                ctx, seq_dir, track_claim, idle_timeout
                            ),
                        )
                        csv_out = predict_result.csv_path
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
                                recorded_output=ds.relative_to_root(csv_out),
                            ),
                        )
                        recomputed = True
                    else:
                        csv_out = reusable_csv
                        recomputed = False
                finally:
                    clear_inflight(seq_dir)

                index_rows.append(
                    LitposeIndexRow(
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
                        model_id=resolved_model.model_id,
                        model_type=resolved_model.model_type,
                        n_individuals=1,
                        csv_path=ds.relative_to_root(csv_out),
                    )
                )

                if convert_to_tracks:
                    bridged = _bridge_csv_to_tracks(
                        ds,
                        group,
                        sequence,
                        csv_out,
                        tracks_variant=tracks_variant,
                        producer_run_id=run_id,
                        video_path=video_path,
                        model_files=[resolved_model.checkpoint, resolved_model.config],
                        fps=item.fps,
                        overwrite=overwrite or recomputed,
                    )
                    if bridged is not None:
                        _n_rows, n_individuals = bridged
                        index_rows[-1] = dataclasses.replace(
                            index_rows[-1], n_individuals=n_individuals
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
        f"[run_litpose] completed run_id={run_id} "
        f"({len(index_rows)}/{len(work_items)} sequences{held}) -> {run_root}"
    )
    return run_id


def list_litpose_runs(ds: Dataset) -> pd.DataFrame:
    """List Lightning Pose runs tracked in the litpose index."""
    if not ds.has_root("litpose"):
        return pd.DataFrame(columns=[f.name for f in dataclasses.fields(LitposeIndexRow)])
    idx_path = litpose_index_path(ds)
    if not idx_path.exists():
        return pd.DataFrame(columns=[f.name for f in dataclasses.fields(LitposeIndexRow)])
    return pd.read_csv(idx_path)
