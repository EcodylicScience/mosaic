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

import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    publish_or_record,
    publish_tracks_table,
    readable_tracks_table,
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
    media_composition_cell,
    register_tracker_row_class,
    TrackerRunRowBase,
    list_tracker_runs,
    tracker_index,
    tracker_index_path,
)
from mosaic.tracking.common.mint import mint_tracker_run, tracker_run_root
from mosaic.tracking.common.scope import TrackerWorkItem, build_work_items
from mosaic.tracking.common.tool_input import resolve_tool_inputs
from mosaic.core.media.timeline import ConcatenatedTimeline, concatenated_timeline
from mosaic.tracking.trex.conversion_cache import (
    CONVERSION_STEM,
    CONVERT_KIND,
    SLOT_POLL_SECONDS,
    ConversionIndexRow,
    ConversionPublishError,
    adopt_by_link,
    claim_slot,
    conversion_index,
    conversion_index_path,
    conversion_slot,
    publish_conversion,
    release_slot,
    reusable_slot,
    slot_marker_is_usable,
    staging_dir,
)
from mosaic.tracking.trex.joined import retime_joined_frame
from mosaic.tracking.trex.version import TREX_KIND, TREX_VERSION
from mosaic.core.pipeline.index_csv import IndexCSV
from mosaic.core.pipeline.job import CancelToken, JobContext
from mosaic.core.pipeline.markers import (
    InflightMarker,
    PhaseMarker,
    PhaseName,
    clear_phase_markers,
    refresh_inflight,
    write_phase_marker,
)
from mosaic.runlog import now_iso

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

    The last four record **what the run actually consumed**, which
    ``video_abs_path`` alone cannot say once a session's clips are converted
    together: it holds the first clip, as every tracker's does.

    ``video_sources`` is a JSON array of dataset-root-relative paths, in
    ``video_order``. It is deliberately *not* named ``*_path``, so it is not
    swept into ``path_columns`` -- the portability passes rewrite such a cell as
    one path and would corrupt an array. It is portable by construction instead:
    the paths go in relative and stay relative.

    ``video_uuids`` is the same arrangement by content identity, comma-joined and
    **never sorted** -- sorting would make two orderings of one session record
    alike. ``media_composition`` is the digest of that arrangement, and is what
    the reuse gate compared.

    **Not to be confused with ``consumed_media_composition`` on the base row**,
    which every tracker carries. That one is the per-entry ``media_raw``
    composition from ``sequences.csv`` -- what the *sources* were -- and answers
    "did the media move under this run". This one is the digest of the clip
    arrangement TREx joined, computed from the work item, and only TREx has one
    because only TREx joins. Two similar names for two facts is the shape of
    mistake that gets more expensive to correct, so the rename belongs on a
    branch that can migrate the reuse gate with it.

    Both are empty when any clip carries no identity,
    which is unestablishable rather than "no videos" -- and that is exactly what
    ``n_source_videos`` is for, since it is the only cell that tells the two
    apart.
    """

    pv_path: str = ""
    video_sources: str = ""
    video_uuids: str = ""
    media_composition: str = ""
    n_source_videos: int = 0


def trex_index(path: Path) -> IndexCSV[TRexIndexRow]:
    """The trex run index, one row per (run, entry)."""
    return tracker_index(path, TRexIndexRow)


def _phase_label(item: TrackerWorkItem) -> str:
    """The entry key, plus the clip count when there is more than one.

    Progress is per entry, so a seventeen-clip session otherwise reports ``1/1``
    for hours with nothing to say why it is taking so long.
    """
    if item.n_sources == 1:
        return item.key
    return f"{item.key} ({item.n_sources} clips)"


# --- Settings, whole and per phase ----------------------------------------

PHASES: Final[tuple[PhaseName, ...]] = ("convert", "track")


def trex_run_id(settings: Mapping[str, object]) -> str:
    """Mint a tracker run identifier from the resolved TREx settings."""
    return op_run_id(TREX_KIND, TREX_VERSION, dict(settings))


# Which settings each TREx task actually consumes, from the two parameter dicts
# built in ``trex/run.py``. A key in neither set would silently stop invalidating
# anything, which ``trex_settings`` and its test guard against.
#
# ``track_max_individuals`` appears in **both**, and the reason is narrower than
# it looks. Nothing in TRex's conversion reads it: it reaches the ``.pv`` only
# through the trailing metadata blob, which sits after the index table and shifts
# no frame data, and a later ``-task track`` overrides it from the command line
# anyway. It stays a convert key because every convert marker already on disk
# recorded a ``params_hash`` computed from this projection, so moving it would
# re-key every existing conversion -- the reconversion the shared cache exists to
# avoid. The cost is only that changing it reconverts when it need not.
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
    detect_type: str | None,
    detect_conf_threshold: float | None,
    detect_iou_threshold: float | None,
    cm_per_pixel: float | None,
    meta_encoding: str | None,
    convert_extra_settings: dict[str, Any] | None,
    track_max_individuals: int | None,
    track_max_speed: float | None,
    track_max_reassign_time: float | None,
    track_trusted_probability: float | None,
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

    **``None`` means "not sent", and is recorded as such.** It is a distinct
    setting from any value, because the run it describes is one where TREx's own
    default governed -- so it hashes to a distinct ``run_id``, which is what
    stops a run made under mosaic's old imposed defaults from being reused for a
    run that leaves TREx to decide. The key is kept rather than dropped, so the
    payload's shape does not depend on what was set.
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
    video_paths: Sequence[Path],
    timeline: ConcatenatedTimeline | None,
    overwrite: bool,
) -> BridgeCounts | None:
    """Merge per-individual TREx NPZ into ``tracks/<variant>/<group>__<seq>.parquet``.

    TREx is the one tracker whose output is several files per entry: one NPZ per
    individual, which ``merge_on_column_union`` concatenates so a field present
    for one individual and absent for another survives as NaN rather than
    dropping the column. The conversion stays here rather than in the shared
    publisher because the publisher takes one frame, not a set of them.

    *timeline* is the concatenation the conversion was built from, and is what
    puts a joined entry's ``time`` on the clips' own measured rates instead of
    the single rate TREx took from the first of them. ``None`` (or a
    single-segment timeline) leaves the export exactly as it was.

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
        # The same answer the other three trackers give. This used to ``return
        # None``, which this function also uses for "nothing was published" -- so a
        # reuse was indistinguishable from a failed conversion, and the counts the
        # caller records came from nowhere. An unreadable table falls through and
        # is reconverted.
        reusable = readable_tracks_table(out_path)
        if reusable is not None:
            return reusable

    converter = get_track_converter("trex_npz")
    # The tracker knows the authoritative entry from the media index, so the
    # hints are exact rather than guessed from a filename. No params: the TRex
    # NPZ conversion has none.
    conv_params = type(converter).Params()
    hints = EntryHints(group=group, sequence=sequence)
    frames: list[pd.DataFrame] = []
    for npz in npz_paths:
        # A conversion failure propagates to `publish_or_record`, which records
        # it on the attempt as a failed entry. Caught and printed here, it was
        # discarded by the caller: a real run tracked a 3.5-hour session, refused
        # all four NPZ on an unclassified column, and reported `finished` having
        # published nothing.
        frames.append(converter.convert(npz, conv_params, hints))
    if not frames:
        return None

    merged = merge_on_column_union(frames)
    if timeline is not None:
        merged = retime_joined_frame(merged, timeline)

    return publish_tracks_table(
        ds,
        merged,
        kind=TREX_KIND,
        group=group,
        sequence=sequence,
        tracks_variant=tracks_variant,
        producer_run_id=producer_run_id,
        source=npz_paths[0].parent,
        consumed=[npz_paths[0], *video_paths],
    )


# --- Public entry point ---------------------------------------------------


def _conversion_from_cache(
    job: EntryJob,
    *,
    slot: Path,
    convert_hash: str,
    idle_timeout: float,
    entry_claim: InflightMarker,
    convert_into: Callable[[Path, str | None, Path, InflightMarker], Path],
) -> Path:
    """This entry's ``.pv`` from the shared cache, converting into it if need be.

    Loops rather than deciding once, because between looking and claiming
    another execution may publish the very slot this one was about to build.
    Each pass re-reads the slot first, so the winner's work is picked up instead
    of duplicated.

    A held slot is **waited for, not worked around**. The alternative is two
    executions each spending the detector hours on identical pixels, which is
    the cost this whole cache exists to remove.

    **Waiting still re-stamps this entry's own claim.** A claim carries its
    expiry, and nothing else refreshes one while this run is blocked on a peer:
    a run that waited out a multi-hour conversion in silence would have its
    working directory read as abandoned -- stolen by the next execution, and
    reclaimed by a concurrent sweep as ``expired_claim``, which is deletable.
    """
    seq_ctx = job.ctx
    while True:
        seq_ctx.check_cancel()
        cached = reusable_slot(job.ds, slot, params_hash=convert_hash, item=job.item)
        if cached is not None:
            return cached

        claim_marker = claim_slot(job.ds, seq_ctx, slot, idle_seconds=idle_timeout)
        if claim_marker is None:
            print(
                f"[{CONVERT_KIND}] {job.item.key}: another execution is building "
                f"this conversion; waiting for it rather than repeating it.",
                file=sys.stderr,
            )
            seq_ctx.heartbeat()
            _keep_entry_claim_alive(job.work_dir, entry_claim, idle_timeout)
            time.sleep(SLOT_POLL_SECONDS)
            continue

        staging = staging_dir(slot, seq_ctx.execution_id)
        try:
            # Every `.incoming-*`, not only this execution's: one left by a run
            # that died mid-conversion would otherwise sit here forever, and a
            # slot holding a staging tree and no marker is refused by the
            # sweeper rather than reclaimed. Safe to do under the claim, which
            # is what says no peer is writing.
            _clear_staging(slot)
            staging.mkdir(parents=True, exist_ok=True)
            _ = convert_into(staging, CONVERSION_STEM, slot, claim_marker)
            try:
                pv_path = publish_conversion(staging, slot)
            except ConversionPublishError as exc:
                return _keep_unshareable_conversion(job, staging=staging, why=str(exc))
            # The slot's own marker, written after every rename: reuse needs the
            # marker *and* the output, so a torn publish leaves nothing to bind.
            write_phase_marker(
                slot,
                PhaseMarker(
                    phase="convert",
                    run_id=slot.parent.name,
                    params_hash=convert_hash,
                    execution_id=seq_ctx.execution_id,
                    completed_at=now_iso(),
                    source=job.ds.relative_to_root(job.item.video_path),
                    source_uid=job.item.source_uid,
                    recorded_output=job.ds.relative_to_root(pv_path),
                ),
            )
            _record_conversion_row(
                job, slot=slot, pv_path=pv_path, convert_hash=convert_hash
            )
        finally:
            # A conversion that raised must not leave its staging tree behind:
            # the slot would then hold a partial `.pv` and no marker, which the
            # sweeper refuses as `foreign` rather than reclaiming.
            shutil.rmtree(staging, ignore_errors=True)
            release_slot(slot, seq_ctx.execution_id)
            _discard_empty_slot(slot)
        return pv_path


def _keep_entry_claim_alive(
    work_dir: Path, marker: InflightMarker, idle_seconds: float
) -> None:
    """Re-stamp this entry's claim. Best-effort, like every other refresh."""
    try:
        _ = refresh_inflight(work_dir, marker, idle_seconds)
    except OSError:
        pass


def _clear_staging(slot: Path) -> None:
    """Remove every staging tree under *slot*. Call only while holding its claim."""
    for stale in slot.glob(".incoming-*"):
        shutil.rmtree(stale, ignore_errors=True)


def _discard_empty_slot(slot: Path) -> None:
    """Remove a slot directory nothing was ever published into.

    ``claim_slot`` has to create the directory before it can take a claim in it,
    so a conversion that declined to publish leaves an empty one behind -- and an
    empty directory carries no marker, which the sweeper classifies ``foreign``
    and refuses permanently, telling the operator a root mosaic created is not a
    tracker root.
    """
    try:
        if not any(slot.iterdir()):
            slot.rmdir()
    except OSError:
        pass


def _keep_unshareable_conversion(job: EntryJob, *, staging: Path, why: str) -> Path:
    """Move a conversion that cannot be published into this run's own directory.

    A conversion is hours of detector inference. Refusing to *share* one that
    did not leave a settings file is right -- tracking from it would silently
    fall back to whatever ``detect_type`` implies -- but throwing it away is not,
    and neither is failing the entry: the run can still use it exactly where a
    run before this cache existed would have, and TREx's implicit settings lookup
    then finds whatever is beside it, which is the behaviour that has always been
    in force.

    Raises:
        ConversionPublishError: If there is no ``.pv`` either, in which case
            nothing was produced and there is nothing to keep.
    """
    staged_pv = staging / f"{CONVERSION_STEM}.pv"
    if not staged_pv.exists():
        raise ConversionPublishError(why)

    kept = job.work_dir / f"{CONVERSION_STEM}.pv"
    os.replace(staged_pv, kept)
    for name in (f"{CONVERSION_STEM}.settings", f"average_{CONVERSION_STEM}.png"):
        spare = staging / name
        if spare.exists():
            os.replace(spare, job.work_dir / name)
    shutil.rmtree(staging, ignore_errors=True)
    _discard_empty_slot(staging.parent)
    print(
        f"[{CONVERT_KIND}] {job.item.key}: {why}. Keeping the conversion in "
        f"{job.work_dir} for this run instead of sharing it; later runs with "
        f"these detection settings will convert again.",
        file=sys.stderr,
    )
    return kept


def _adopt_into_cache(
    job: EntryJob,
    *,
    slot: Path | None,
    marker: PhaseMarker,
    local_pv: Path,
    convert_hash: str,
    idle_timeout: float,
) -> Path | None:
    """Hard-link a conversion already on disk into the cache, and return its new path.

    ``None`` means it stays where it is, which is always a correct outcome: the
    run then uses the ``.pv`` in place exactly as it did before this cache
    existed.

    The marker must *prove* the match -- a non-empty ``params_hash`` and
    ``source_uid``, both equal to this run's. ``reusable_marker`` deliberately
    reads an empty one as "unknown is not mismatched", which is right for reusing
    a directory where it stands and wrong for promoting its contents into a
    durable shared address. A directory adopted from before markers existed
    records neither, so it is reused and never cached.
    """
    if slot is None or not slot_marker_is_usable(marker):
        return None
    if marker.params_hash != convert_hash or marker.source_uid != job.item.source_uid:
        return None
    if reusable_slot(job.ds, slot, params_hash=convert_hash, item=job.item) is not None:
        # Already published, by an earlier run or a peer. Adoption is a
        # convenience, so the published copy simply wins.
        return None

    seq_ctx = job.ctx
    claim_marker = claim_slot(job.ds, seq_ctx, slot, idle_seconds=idle_timeout)
    if claim_marker is None:
        _discard_empty_slot(slot)
        return None
    staging = staging_dir(slot, seq_ctx.execution_id)
    try:
        _clear_staging(slot)
        if not adopt_by_link(local_pv, staging):
            return None
        pv_path = publish_conversion(staging, slot)
        write_phase_marker(
            slot,
            PhaseMarker(
                phase="convert",
                run_id=slot.parent.name,
                params_hash=convert_hash,
                execution_id=seq_ctx.execution_id,
                completed_at=marker.completed_at or now_iso(),
                source=marker.source,
                source_uid=marker.source_uid,
                recorded_output=job.ds.relative_to_root(pv_path),
            ),
        )
        _record_conversion_row(
            job, slot=slot, pv_path=pv_path, convert_hash=convert_hash
        )
    except ConversionPublishError as exc:
        print(f"[{CONVERT_KIND}] {job.item.key}: {exc}", file=sys.stderr)
        return None
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        release_slot(slot, seq_ctx.execution_id)
        _discard_empty_slot(slot)
    return pv_path


def _record_conversion_row(
    job: EntryJob, *, slot: Path, pv_path: Path, convert_hash: str
) -> None:
    """Append this slot's index row. Best-effort, and deliberately so.

    Without a row the slot reads as ``unrowed``, which the sweeper *refuses* --
    so a failure here costs disk, never data.
    """
    row = ConversionIndexRow(
        run_id=slot.parent.name,
        group="",
        sequence=job.item.source_uid,
        abs_path=Path(job.ds.relative_to_root(slot)),
        video_abs_path=job.ds.relative_to_root(job.item.video_path),
        params_hash=convert_hash,
        pv_path=job.ds.relative_to_root(pv_path),
        settings_path=job.ds.relative_to_root(pv_path.with_suffix(".settings")),
        n_source_videos=job.item.n_sources,
    )
    try:
        index = conversion_index(conversion_index_path(job.ds))
        index.ensure()
        index.append([row])
        index.mark_finished(slot.parent.name)
    except OSError as exc:
        print(
            f"[{CONVERT_KIND}] failed to record {slot} in the index: {exc}",
            file=sys.stderr,
        )


def run_trex(
    ds: Dataset,
    *,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    entries: Iterable[tuple[str, str]] | None = None,
    # detection / conversion
    detect_model: Path | str | None = None,
    detect_type: str | None = None,
    detect_conf_threshold: float | None = None,
    detect_iou_threshold: float | None = None,
    cm_per_pixel: float | None = None,
    meta_encoding: str | None = None,
    convert_extra_settings: dict[str, Any] | None = None,
    # tracking
    track_max_individuals: int | None = None,
    track_max_speed: float | None = None,
    track_max_reassign_time: float | None = None,
    track_trusted_probability: float | None = None,
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

    Every TREx parameter defaults to ``None``, which means *do not send it*, so
    TREx's own default governs. See :class:`~mosaic.tracking.ops.trex.TrexParams`
    for why mosaic declares no default of its own.

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
    # The same projection the convert digest is taken over, so the conversion
    # run root's name and every convert marker's `params_hash` cannot disagree.
    convert_settings = phase_settings(settings, "convert")

    # Route each scoped entry through the transcode verdict: a clean entry
    # resolves to its original, an analysis-required entry to its constant-rate
    # analysis derivative (so tracks land in the same frame space as the rest of
    # the pipeline), and a required-but-unlinked entry raises MediaProbeError
    # here -- before any TREx subprocess opens a known-defective original.
    #
    # TREx decodes the files itself, so the routed *paths* are what it is given.
    # The routed facts are still read, for a different job: they are what the
    # concatenated timeline is built from, and TREx cannot supply that -- it
    # takes one frame rate from the first clip and never checks the others.
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
        joined = item.n_sources > 1
        # Built from the routed facts, which is what TREx will actually read.
        # It is what puts a joined entry's `time` on the clips' own measured
        # rates -- TREx takes one rate from the first clip and never checks the
        # others. `None` when the facts are absent, which leaves the export as
        # it is rather than guessing at a timeline.
        timeline = (
            concatenated_timeline(item.source_facts) if item.source_facts else None
        )

        # The .results file is the only output TREx writes at the *end* of
        # tracking; the .pv and the per-individual files appear as processing
        # proceeds, so neither distinguishes a finished run from one killed
        # partway. A finished tracking implies a finished conversion, which is
        # what lets one signal adopt both phases.
        #
        # Not for a joined entry. A pre-marker directory cannot say how many
        # clips it covered, and a joined entry's directory is shape-identical to
        # a single-video one -- so adopting would silently keep one clip's tracks
        # for a whole session, under a marker asserting the session was done.
        # This is the documented rule for a tracker whose output cannot make that
        # distinction: do not adopt.
        if not joined:
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

        def convert_into(
            out_dir: Path,
            name: str | None,
            claim_dir: Path,
            claim_marker: InflightMarker,
        ) -> Path:
            """Run the conversion into *out_dir*, keeping every claim alive.

            Two claims are held during a shared conversion -- this entry's
            working directory and the slot -- and **both** have to be re-stamped
            from the tool's output. Refreshing only the slot leaves the entry's
            own claim to expire on its fixed window, after which a concurrent
            sweep reads a live run as ``expired_claim`` and deletes its working
            directory, and the next execution steals it and tracks into the same
            place. When the two are the same directory (nothing is being shared)
            one callback is enough.
            """
            seq_ctx.progress.on_phase("convert", _phase_label(item))
            entry_tick = phase_activity(seq_ctx, work_dir, convert_claim, idle_timeout)
            slot_tick = (
                None
                if claim_dir == work_dir
                else phase_activity(seq_ctx, claim_dir, claim_marker, idle_timeout)
            )

            def on_output(line: str) -> None:
                entry_tick(line)
                if slot_tick is not None:
                    slot_tick(line)

            # T-Rex opens the path itself, so an imgstore recording resolves to
            # the plain video export-store wrote for it. Resolved here rather
            # than before the reuse gate: an entry whose conversion is already
            # reusable needs no export, and demanding one would fail a re-run
            # over work that is finished.
            return run_trex_convert(
                resolve_tool_inputs(job.ds, item, kind=TREX_KIND),
                out_dir,
                # TREx names a single-source `.pv` after its stem, which is where
                # mosaic already looks -- but for several sources sharing a
                # parent it uses *the parent directory's* name, which is neither
                # chosen nor looked for. A cached conversion pins the stem
                # outright, because the slot is shared and the source file's name
                # must not decide what is inside it.
                output_name=name,
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
                on_output=on_output,
            ).pv_path

        def pin(where: Path, pv: Path) -> None:
            """Record in *where* that this entry's conversion is *pv*.

            On the trex working directory this is also what pins the slot: the
            sweeper reads convert markers, not index rows, to decide a shared
            conversion is still in use.
            """
            _ = record_phase(
                job.ds,
                where,
                "convert",
                ctx=seq_ctx,
                run_id=minted.run_id,
                params_hash=phase_hashes["convert"],
                video_path=item.video_path,
                video_uid=item.source_uid,
                output=pv,
            )

        # --- the conversion -------------------------------------------------
        #
        # Local first, and that ordering is the whole compatibility story: this
        # is the call that was here before, with the arguments it had, so every
        # directory already on disk resolves exactly as it used to -- including
        # one whose markers the adoption above backfilled, since "unknown is not
        # mismatched". Consulting the cache first would send a run that already
        # holds a good `.pv` looking for a slot it might have to create.
        slot = conversion_slot(job.ds, convert_settings, item)
        reusable_convert = reusable_output(
            job.ds,
            work_dir,
            "convert",
            params_hash=phase_hashes["convert"],
            video_path=item.video_path,
            video_uid=item.source_uid,
        )
        if reusable_convert is not None:
            local_marker, pv_path = reusable_convert
            adopted = _adopt_into_cache(
                job,
                slot=slot,
                marker=local_marker,
                local_pv=pv_path,
                convert_hash=phase_hashes["convert"],
                idle_timeout=idle_timeout,
            )
            if adopted is not None:
                pv_path = adopted
                # Repointed with `write_phase_marker`, not `record_phase`: this
                # phase did not just complete, and restamping `completed_at`
                # would push out the trex directory's own reclamation.
                write_phase_marker(
                    work_dir,
                    local_marker.model_copy(
                        update={"recorded_output": job.ds.relative_to_root(pv_path)}
                    ),
                )
        else:
            # The tracking phase consumes this phase's output, so a
            # re-conversion invalidates it too.
            clear_phase_markers(work_dir)
            clear_outputs(work_dir, TREX_KIND, "convert")
            clear_outputs(work_dir, TREX_KIND, "track")
            # Taken before either route, and held across the whole conversion --
            # including any wait on a peer's. This entry's directory is claimed
            # for the duration whether the conversion happens here or elsewhere.
            convert_claim = claim(seq_ctx, work_dir, "convert", idle_timeout)
            if slot is None:
                # No content identity for this media, so there is no durable key
                # -- only a path, and a path is a mutable key. Convert in place,
                # as every run did before the cache existed.
                print(
                    f"[{TREX_KIND}] {item.key}: media carries no content "
                    f"identity, so this conversion cannot be shared. Run "
                    f"`mosaic reprobe-media` and re-run to make it cacheable.",
                    file=sys.stderr,
                )
                pv_path = convert_into(
                    work_dir, item.key if joined else None, work_dir, convert_claim
                )
            else:
                pv_path = _conversion_from_cache(
                    job,
                    slot=slot,
                    convert_hash=phase_hashes["convert"],
                    idle_timeout=idle_timeout,
                    entry_claim=convert_claim,
                    convert_into=convert_into,
                )
            pin(work_dir, pv_path)

        track_marker = reusable_marker(
            job.ds,
            work_dir,
            "track",
            params_hash=phase_hashes["track"],
            video_path=item.video_path,
            video_uid=item.source_uid,
        )
        if track_marker is None:
            clear_outputs(work_dir, TREX_KIND, "track")
            track_claim = claim(seq_ctx, work_dir, "track", idle_timeout)
            seq_ctx.progress.on_phase("track", _phase_label(item))
            # Named explicitly rather than left to TRex's implicit lookup, which
            # composes `<output_dir>/<pv stem>.settings` and so finds nothing
            # once the conversion is shared. Without it the detection parameters
            # -- posture, the skeleton, the size filter, everything
            # `convert_extra_settings` carried -- fall back to what `detect_type`
            # implies, and the run says nothing about it.
            conversion_settings = pv_path.with_suffix(".settings")
            run_trex_track(
                pv_path,
                work_dir,
                settings_path=(
                    conversion_settings if conversion_settings.exists() else None
                ),
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
                video_uid=item.source_uid,
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
            # What the run consumed, which video_abs_path cannot say for a
            # session: it holds clip 0. Relative going in, so the array stays
            # portable without the path passes -- which would corrupt it.
            video_sources=json.dumps(
                [job.ds.relative_to_root(path) for path in item.video_paths]
            ),
            # Never sorted: this is the arrangement. Empty when any clip carries
            # no identity, matching what the reuse gate could establish.
            video_uuids=(",".join(item.video_uids) if all(item.video_uids) else ""),
            media_composition=item.source_uid if joined else "",
            n_source_videos=item.n_sources,
        )

        if convert_to_tracks:
            # A recomputed entry must replace its parquet: the bridge otherwise
            # declines to overwrite, and the table would keep the results of the
            # run just invalidated.
            _ = publish_or_record(
                job.ctx,
                item.key,
                lambda: _bridge_npz_to_tracks(
                    job.ds,
                    item.group,
                    item.sequence,
                    npz_paths,
                    tracks_variant=minted.tracks_variant,
                    producer_run_id=minted.run_id,
                    video_paths=item.video_paths,
                    timeline=timeline,
                    overwrite=job.overwrite or recomputed,
                ),
                kind=TREX_KIND,
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

# The row class this root's index holds, so an inventory can ask about every
# tracker generically. Registered rather than tabled in ``common``, which is
# imported by this module and cannot import it back.
register_tracker_row_class(TREX_KIND, TRexIndexRow)
