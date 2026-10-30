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

There is one expensive, gated phase -- ``track`` (``sleap-nn track`` inference
+ identity tracking, producing a ``.slp``). Its completion marker lets a killed
run resume without re-running inference. The cheap, deterministic analysis
export (``sleap-convert``, producing the ``.h5`` the converter reads) is not
marker-gated: it is re-run only when its output is missing or the inference was
recomputed.

A model list whose last model is an embedding model makes an embedding run. Its
``track`` phase runs ``sleap-nn predict``, which tracks by appearance and saves
each detection's vector in the ``.slp``. After the export, the vectors are read
back into ``<entry_key>.embeddings.npz``, which is ensured like the export, and
the bridge publishes them as an appearance table beside the tracks table.

An entry in which sleap-nn finds nothing to embed is an empty entry. Its
``track`` phase records ``<entry_key>.no-detections.json`` in place of a
``.slp``, nothing is exported or read back, and an empty tracks table and an
empty appearance table are published. sleap-nn prints nothing while it embeds,
so each entry of an embedding run waits for silence in proportion to its
frames (:func:`_entry_idle_limit`).
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from mosaic.core.helpers import make_entry_key, text_cell
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.appearance import read_appearance
from mosaic.core.pipeline.index_csv import IndexCSV
from mosaic.core.pipeline.job import CancelToken, JobContext
from mosaic.core.pipeline.media_input import media_identity_terms
from mosaic.core.pipeline.markers import (
    clear_phase_marker,
)
from mosaic.core.pipeline.dataset_indexes import register_reconcilable_index
from mosaic.core.pipeline.entry_claim import claim, phase_activity
from mosaic.core.pipeline.op_identity import op_run_id
from mosaic.core.pipeline.placement import EntryAxis
from mosaic.core.pipeline.tracks_index import media_composition_for
from mosaic.core.track_library.sleap import (
    AppearanceReadback,
    appearance_rows,
    empty_sleap_tracks,
    read_analysis_track_names,
    read_appearance_readback,
)
from mosaic.tracking.common.bridge import (
    BridgeCounts,
    publish_or_record,
    publish_tracks_table,
    reusable_tracks_table,
    tracks_table_path,
)
from mosaic.tracking.common.driver import EntryJob, run_tracker
from mosaic.tracking.common.entry import (
    AdoptEvidence,
    adopt_completed_directory,
    clear_outputs,
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
from mosaic.tracking.common.tool_input import DecodeProbe, resolve_tool_input
from mosaic.tracking.model_refs import (
    SLEAP_EMBEDDING_HEAD,
    embedding_dim,
    model_kind_for,
    observed_model_source,
    resolve_model_set,
)
from mosaic.tracking.sleap.params import SLEAP_MODEL, SleapParams
from mosaic.tracking.sleap import probe as sleap_probe
from mosaic.tracking.sleap.version import (
    SLEAP_KIND,
    SLEAP_NN_EMBEDDING_MIN,
    SLEAP_VERSION,
    require_sleap_nn_at_least,
)

from .run import (
    SLEAP_ENV,
    SleapNoDetections,
    read_back_embeddings,
    run_sleap_convert,
    run_sleap_predict,
    run_sleap_track,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope
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
    declared. Both are blank for an entry without a detection, which has
    neither file.

    ``n_vectors_unmatched`` is the count of appearance vectors dropped because
    their detection has no tracks row
    (:attr:`~mosaic.tracking.common.bridge.BridgeCounts.n_vectors_unmatched`).
    It is a string so that it reads blank where nothing was measured: a row
    written before the column and a run without an embedding model. A reused
    tracks table is not measured again, and its row keeps the recorded count.
    """

    model_id: str = ""
    model_type: str = ""
    slp_path: str = ""
    analysis_h5_path: str = ""
    n_vectors_unmatched: str = ""


def sleap_index(path: Path) -> IndexCSV[SleapIndexRow]:
    """The sleap run index, one row per (run, entry)."""
    return tracker_index(path, SleapIndexRow)


def _recorded_unmatched(ds: Dataset, run_id: str) -> dict[tuple[str, str], str]:
    """Return the ``n_vectors_unmatched`` cell of *run_id*'s row of each entry."""
    path = sleap_index_path(ds)
    if not path.exists():
        return {}
    index = sleap_index(path).read()
    return {
        (text_cell(row.get("group")), text_cell(row.get("sequence"))): text_cell(
            row.get("n_vectors_unmatched")
        )
        for _, row in index[index["run_id"] == run_id].iterrows()
    }


# --- Settings -------------------------------------------------------------


def sleap_run_id(settings: Mapping[str, object]) -> str:
    """Mint a tracker run identifier from the resolved SLEAP settings."""
    return op_run_id(SLEAP_KIND, SLEAP_VERSION, dict(settings))


def sleap_settings(params: SleapParams, *, model_id: str) -> dict[str, object]:
    """Build the settings that define a SLEAP tracking result -- the ``run_id`` payload.

    The model is carried as its content digest (``model_id``), never a path. When
    tracking is off, the tracker knobs are dropped from identity so retuning them
    cannot bust a cache they never fed. ``media`` enters the payload only when it
    names a variant.

    Args:
        params: The run's parameters.
            :class:`~mosaic.tracking.sleap.params.SleapParams` describes every
            field. The execution knobs are excluded from identity and stay out
            of the payload.
        model_id: The weights' identity -- a training ``run_id`` or a content
            digest, never a path. The caller resolves it, because a bare weights
            path is a mutable key: swapping the checkpoint in place would let two
            different runs share one identifier.
    """
    tracking = params.tracking
    return {
        "model": model_id,
        "tracking": tracking,
        "use_flow": params.use_flow if tracking else None,
        "candidates_method": params.candidates_method if tracking else None,
        "features": params.features if tracking else None,
        "scoring_method": params.scoring_method if tracking else None,
        "track_matching_method": params.track_matching_method if tracking else None,
        "tracking_window_size": params.tracking_window_size if tracking else None,
        "max_tracks": params.max_tracks if tracking else None,
        "max_instances": params.max_instances,
        "peak_threshold": params.peak_threshold,
        "analysis_range": (
            list(params.analysis_range) if params.analysis_range else None
        ),
        "sleap_extra_settings": params.sleap_extra_settings,
        **media_identity_terms(params),
    }


# --- Per-entry reuse ------------------------------------------------------


# --- analysis-HDF5 -> standardized tracks bridge --------------------------


def bridge_analysis_h5_to_tracks(
    ds: Dataset,
    group: str,
    sequence: str,
    h5_path: Path,
    *,
    tracks_variant: str,
    producer_run_id: str,
    consumed_media: Sequence[Path],
    model_checkpoints: Sequence[Path],
    fps: float,
    overwrite: bool,
    axis: EntryAxis,
    recorded_first_frame: int | None,
    readback: Path | None = None,
) -> BridgeCounts:
    """Bridge a SLEAP analysis HDF5 into ``tracks/<variant>/<group>__<seq>.parquet``.

    Uses the registered ``sleap_analysis_h5`` converter with the authoritative
    (group, sequence) known from the media index, so no name is guessed from a
    filename. *consumed_media* are the media files that the table derives from,
    and *axis* places the table on the entry's axes. *readback* is the
    ``.npz`` of an embedding run's tracked instances, whose vectors are
    published as the entry's appearance table (:func:`_appearance_for`). An
    existing table is reused as :func:`_reusable_table` decides, against
    *recorded_first_frame*, and otherwise both tables are republished.

    Returns:
        The counts of the published or reused table.

    Raises:
        ValueError: A tracks row has no appearance vector in *readback*.
        AppearanceTableError: The appearance table breaks the appearance
            contract against the placed tracks table. Nothing is written.
    """
    from mosaic.core.track_converter import EntryHints, get_track_converter
    from mosaic.core.track_library.sleap import SleapConvertParams

    reusable = _reusable_table(
        ds,
        tracks_variant,
        group,
        sequence,
        overwrite=overwrite,
        axis=axis,
        recorded_first_frame=recorded_first_frame,
        with_appearance=readback is not None,
    )
    if reusable is not None:
        return reusable

    converter = get_track_converter("sleap_analysis_h5")
    # A conversion failure propagates to `publish_or_record`, which records it on
    # the attempt as a failed entry. Caught and printed here, it was discarded by
    # the caller and the run reported success having published nothing.
    df = converter.convert(
        h5_path,
        SleapConvertParams(fps=fps),
        EntryHints(group=group, sequence=sequence),
    )
    appearance: pd.DataFrame | None = None
    unmatched: int | None = None
    if readback is not None:
        appearance, unmatched = _appearance_for(
            df, readback, h5_path, group=group, sequence=sequence
        )

    counts = publish_tracks_table(
        ds,
        df,
        kind=SLEAP_KIND,
        group=group,
        sequence=sequence,
        tracks_variant=tracks_variant,
        producer_run_id=producer_run_id,
        source=h5_path.parent,
        consumed=[h5_path, *consumed_media, *model_checkpoints],
        axis=axis,
        # Unknown. The analysis export's frame axis runs to the video's length
        # when sleap-io can measure the video, and otherwise to the last frame
        # with an instance, and the file does not say which.
        frames_read=None,
        appearance=appearance,
    )
    return dataclasses.replace(counts, n_vectors_unmatched=unmatched)


def _bridge_no_detections(
    ds: Dataset,
    group: str,
    sequence: str,
    marker: Path,
    *,
    tracks_variant: str,
    producer_run_id: str,
    consumed_media: Sequence[Path],
    model_checkpoints: Sequence[Path],
    overwrite: bool,
    axis: EntryAxis,
    recorded_first_frame: int | None,
    dim: int,
) -> BridgeCounts:
    """Publish the empty tracks and appearance tables of an entry without a detection.

    *marker* is the entry's ``<entry_key>.no-detections.json``, and *dim* the
    vector length of the run's embedding model. The existing tables are reused
    as :func:`bridge_analysis_h5_to_tracks` reuses them.

    Returns:
        The counts of the published or reused table.
    """
    reusable = _reusable_table(
        ds,
        tracks_variant,
        group,
        sequence,
        overwrite=overwrite,
        axis=axis,
        recorded_first_frame=recorded_first_frame,
        with_appearance=True,
    )
    if reusable is not None:
        return reusable
    appearance = appearance_rows(
        AppearanceReadback.empty(dim), (), group=group, sequence=sequence
    ).drop(columns="has_vector")
    counts = publish_tracks_table(
        ds,
        empty_sleap_tracks(),
        kind=SLEAP_KIND,
        group=group,
        sequence=sequence,
        tracks_variant=tracks_variant,
        producer_run_id=producer_run_id,
        source=marker.parent,
        consumed=[marker, *consumed_media, *model_checkpoints],
        axis=axis,
        frames_read=None,
        appearance=appearance,
    )
    return dataclasses.replace(counts, n_vectors_unmatched=0)


def _reusable_table(
    ds: Dataset,
    variant: str,
    group: str,
    sequence: str,
    *,
    overwrite: bool,
    axis: EntryAxis,
    recorded_first_frame: int | None,
    with_appearance: bool,
) -> BridgeCounts | None:
    """Return the counts of the entry's published table when a re-run may keep it.

    A table is kept when *overwrite* is false, it reads and was numbered from
    the first frame that *axis* places the entry at
    (:func:`~mosaic.tracking.common.bridge.reusable_tracks_table`), and, when
    *with_appearance* is set, its row records an appearance table that reads.
    """
    out_path = tracks_table_path(ds, variant, make_entry_key(group, sequence))
    if overwrite or not out_path.exists():
        return None
    # Reuse only what reads. A table torn by a kill during a non-atomic write,
    # or by an external tool, is republished instead of adopted as an empty
    # result.
    reusable = reusable_tracks_table(
        out_path, recorded_first_frame=recorded_first_frame, axis=axis
    )
    if reusable is None:
        return None
    if with_appearance and not _appearance_reads(ds, variant, group, sequence):
        return None
    return reusable


def _appearance_reads(ds: Dataset, variant: str, group: str, sequence: str) -> bool:
    """Return whether the tracks row of the entry records an appearance table that reads."""
    try:
        return read_appearance(ds, variant, (group, sequence)) is not None
    except (OSError, ValueError):
        return False


def _appearance_for(
    tracks: pd.DataFrame,
    readback: Path,
    h5_path: Path,
    *,
    group: str,
    sequence: str,
) -> tuple[pd.DataFrame, int]:
    """Join the vectors of *readback* to the converted *tracks* of one entry.

    The vectors are keyed by the analysis HDF5's slots. The converter drops a
    slot whose keypoints are all NaN, and the vector of such a slot is dropped
    here and counted.

    Args:
        tracks: The converted tracks table, before placement.
        readback: The ``.npz`` of the entry's tracked instances.
        h5_path: The analysis HDF5 that *tracks* was converted from.
        group: The entry's group.
        sequence: The entry's sequence.

    Returns:
        The appearance table, one row per row of *tracks*, and the number of
        vectors dropped because their slot has no tracks row.

    Raises:
        ValueError: A row of *tracks* has no vector. The message gives the
            count.
    """
    rows = appearance_rows(
        read_appearance_readback(readback),
        read_analysis_track_names(h5_path),
        group=group,
        sequence=sequence,
    )
    keys = ["frame", "id"]
    tracked = pd.MultiIndex.from_frame(tracks[keys].astype("int64"))
    matched = pd.MultiIndex.from_frame(rows[keys]).isin(tracked)
    has_vector = rows["has_vector"].to_numpy(dtype=bool)
    unmatched = int((~matched & has_vector).sum())
    if unmatched:
        print(
            f"[run_sleap] {group}/{sequence}: dropped {unmatched} appearance "
            "vectors of detections without keypoints, which have no tracks row.",
            file=sys.stderr,
        )
    kept = rows.iloc[np.flatnonzero(matched & has_vector)].drop(columns="has_vector")
    missing = len(tracks) - len(kept)
    if missing:
        msg = (
            f"{missing} tracks rows of {group}/{sequence} have no appearance "
            f"vector in {readback}, and every row of an embedding run needs one."
        )
        raise ValueError(msg)
    return kept.reset_index(drop=True), unmatched


# --- Model checks ---------------------------------------------------------


_EMBEDDING_TRACKER: Final = (
    ("features", "embeddings"),
    ("scoring_method", "cosine_sim"),
    ("candidates_method", "local_queues"),
)
"""The tracker settings of an embedding run, the configuration that appearance
tracking was validated with."""

_IDENTITY_CLASS_HEADS: Final = frozenset(
    {"multi_class_bottomup", "multi_class_topdown"}
)
"""The model types that predict an identity class for each detection."""

_EMBEDDING_DETECTION_HEADS: Final = frozenset(
    {("single_instance",), ("bottomup",), ("centroid", "centered_instance")}
)
"""The model types, in run order, that an embedding model may follow."""


def sleap_nn_command(params: SleapParams) -> str:
    """Name the sleap-nn command a run with *params* runs, as its attempt records it.

    Provenance, never identity. ``features="embeddings"`` is the setting that
    every embedding run needs and that only an embedding run is allowed
    (:func:`_embedding_run_shape`), so it names the command before the models
    are resolved.
    """
    if params.features == "embeddings":
        return "sleap-nn predict"
    return "sleap-nn track"


def _embedding_run_shape(
    refs: Sequence[str], model_types: Sequence[str], params: SleapParams
) -> bool:
    """Return whether *refs* is an embedding model set, and refuse a wrong shape.

    An embedding model set ends in one embedding model after its detection
    models. Its run tracks by appearance and writes an appearance table beside
    each tracks table it publishes. The run is refused before it is named when
    its models or its settings cannot do that.

    Args:
        refs: The model references, in run order.
        model_types: The model type of each reference, in the same order.
        params: The run's parameters.

    Returns:
        ``True`` for an embedding model set, ``False`` for any other model set.

    Raises:
        ValueError: The models or the settings cannot make an embedding run, or
            ``features="embeddings"`` is set without an embedding model. The
            message names the models or the settings to change.
    """
    positions = [
        index
        for index, model_type in enumerate(model_types)
        if model_type == SLEAP_EMBEDDING_HEAD
    ]
    if not positions:
        if params.features == "embeddings":
            msg = (
                "features='embeddings' compares appearance vectors, which only an "
                "embedding model predicts. Add an embedding model as the last "
                "model, or choose other features."
            )
            raise ValueError(msg)
        return False
    if len(positions) > 1:
        named = " and ".join(repr(refs[index]) for index in positions)
        msg = (
            f"The models name more than one embedding model: {named}. A run "
            "takes one, as its last model."
        )
        raise ValueError(msg)
    position = positions[0]
    reference = refs[position]
    if position != len(refs) - 1:
        msg = (
            f"{reference!r} is at position {position + 1} of {len(refs)}, and an "
            "embedding model goes last, after the detection models."
        )
        raise ValueError(msg)
    for ref, model_type in zip(refs, model_types, strict=True):
        if model_type in _IDENTITY_CLASS_HEADS:
            msg = (
                f"{ref!r} is a {model_type} model, and sleap-nn predict -t "
                "retracks the identity classes that it predicts. Track by "
                "appearance after a single_instance, bottomup or "
                "centered_instance model."
            )
            raise ValueError(msg)
    if tuple(model_types[:position]) not in _EMBEDDING_DETECTION_HEADS:
        msg = (
            f"The embedding model {reference!r} needs a detection model before it: "
            "a single_instance or bottomup model, or a centroid model and its "
            "centered_instance model."
        )
        raise ValueError(msg)
    if not params.tracking:
        msg = (
            "An embedding run needs tracking=True: the appearance vectors are "
            "written only for tracked detections."
        )
        raise ValueError(msg)
    if params.use_flow:
        msg = (
            "An embedding run needs use_flow=False: the appearance tracker does "
            "not take optical flow."
        )
        raise ValueError(msg)
    if not params.convert_to_tracks:
        msg = (
            "An embedding run needs convert_to_tracks=True: the appearance tables "
            "are written only beside a published tracks table."
        )
        raise ValueError(msg)
    settings = params.model_dump(include={key for key, _ in _EMBEDDING_TRACKER})
    if any(settings[key] != value for key, value in _EMBEDDING_TRACKER):
        stated = [f"{key}={value!r}" for key, value in _EMBEDDING_TRACKER]
        required = f"{', '.join(stated[:-1])} and {stated[-1]}"
        msg = (
            f"An embedding run needs {required}, the tracker configuration that "
            "appearance tracking was validated with. Set the three explicitly."
        )
        raise ValueError(msg)
    return True


def _require_embedding_release(
    *,
    sleap_conda_env: str | None,
    sleap_bin: Path | str | None,
    cancel_token: CancelToken | None,
) -> str:
    """Refuse a SLEAP environment whose sleap-nn cannot predict with an embedding model.

    Returns:
        The installed sleap-nn version.

    Raises:
        SleapNotFoundError: sleap_nn does not import in the SLEAP environment.
        ValueError: The installed sleap-nn is older than
            :data:`~mosaic.tracking.sleap.version.SLEAP_NN_EMBEDDING_MIN`.
    """
    probe = sleap_probe.probe_sleap(
        sleap_conda_env=sleap_conda_env,
        sleap_bin=sleap_bin,
        cancel_check=cancel_token.is_cancelled if cancel_token is not None else None,
    )
    sleap_probe.require_sleap_nn(probe)
    require_sleap_nn_at_least(probe, SLEAP_NN_EMBEDDING_MIN)
    return probe.sleap_nn_version


@dataclass(frozen=True, slots=True)
class _EmbeddingRun:
    """What an embedding run knows of its embedding model and its sleap-nn."""

    dim: int
    sleap_nn_version: str


EMBEDDING_IDLE_SECONDS_PER_FRAME: Final = 0.05
"""How long an entry of an embedding run may stay silent per frame, about twice
the measured time per frame of the embedding stage."""

EMBEDDING_IDLE_UNKNOWN_LENGTH_SECONDS: Final = 3600.0
"""How long an entry of an embedding run of unknown length may stay silent."""


def _entry_idle_limit(
    params: SleapParams, axis: EntryAxis, *, embedding_run: bool
) -> float:
    """Return how long one entry's SLEAP calls may run without output.

    sleap-nn prints nothing while it embeds. An embedding run therefore allows
    :data:`EMBEDDING_IDLE_SECONDS_PER_FRAME` per frame when that exceeds
    ``params.idle_timeout``. The frames are the analyzed range under
    ``analysis_range``, and otherwise the entry's
    :attr:`~mosaic.core.pipeline.placement.EntryAxis.media_frames`. An entry of
    unknown length allows :data:`EMBEDDING_IDLE_UNKNOWN_LENGTH_SECONDS`. Any
    other run allows ``params.idle_timeout``.

    Returns:
        The idle limit in seconds.
    """
    if not embedding_run:
        return params.idle_timeout
    if params.analysis_range is not None:
        first, last = params.analysis_range
        frames: int | None = last - first + 1
    else:
        frames = axis.media_frames
    if frames is None:
        return max(params.idle_timeout, EMBEDDING_IDLE_UNKNOWN_LENGTH_SECONDS)
    return max(params.idle_timeout, EMBEDDING_IDLE_SECONDS_PER_FRAME * frames)


def _write_no_detections(
    path: Path, report: SleapNoDetections, *, sleap_nn_version: str
) -> None:
    """Write *report* to *path*, the ``track`` output of an entry without a detection."""
    document = {
        "sleap_nn_version": sleap_nn_version,
        "argv": list(report.argv),
        "message": report.message,
    }
    text = json.dumps(document, indent=2) + "\n"
    atomic_write(path, lambda temp: temp.write_text(text, encoding="utf-8"))


# --- Public entry point ---------------------------------------------------


def run_sleap(
    ds: Dataset,
    params: SleapParams,
    scope: "ResolvedScope | None" = None,
    *,
    overwrite: bool = False,
    sleap_conda_env: str | None = None,
    sleap_bin: Path | str | None = None,
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

    *params* states what to run, *scope* which entries to run it over, and
    *overwrite* whether a finished entry is redone;
    :class:`~mosaic.tracking.sleap.params.SleapParams` describes every field.
    The Job-Contract knobs beside it (``execution_id`` / ``owner`` / ``track`` /
    ``progress_callback`` / ``cancel_token``) open the run's context, and *ctx*
    runs inside one that is already open instead.

    ``sleap_conda_env`` and ``sleap_bin`` are the top two rungs of the location
    ladder, for a caller holding more than one SLEAP environment; below them the
    console scripts are found through ``MOSAIC_SLEAP_CONDA_ENV`` /
    ``MOSAIC_SLEAP_BIN`` and then ``$PATH``. They name a machine rather than a
    result, so they reach no identifier and the op does not carry them: a queued
    job would ship a path that means something else on the host that runs it.

    Returns the content-addressed ``run_id``.
    """
    # Resolve the model *before* the settings that name it, because what the
    # settings carry is the weights' identity, not the paths that pointed at
    # them. An unresolvable reference aborts here, before any run root or tracks
    # variant is recorded -- a recorded variant naming weights that could not be
    # found describes a run that never happened.
    #
    # Resolved under the *training* kind the declaration accepts, not this
    # tracker's. A reference may be a path or a registered training ``run_id``,
    # and a run_id resolves against ``models/<kind>/index.csv``, the index the
    # row was written into, which ``train-sleap`` owns. No op writes a
    # ``models/sleap/`` index, so under ``SLEAP_KIND`` only a path would resolve
    # and a model trained here could not be named by its run id. The artifact
    # shape is unaffected: ``MODEL_KINDS`` declares ``train-sleap`` as SLEAP's own
    # spec for this.
    refs = [str(m) for m in params.model_paths]
    resolved_models = resolve_model_set(
        ds, refs, model_kind_for(SLEAP_KIND, SLEAP_MODEL, refs)
    )
    # A run of the wrong shape is refused before the mint, as an unresolvable
    # reference is. So is a model without a vector length, and a sleap-nn
    # release that cannot run the embedding model.
    embedding: _EmbeddingRun | None = None
    embedding_observed: dict[str, str] = {}
    if _embedding_run_shape(refs, resolved_models.model_types, params):
        dim = embedding_dim(resolved_models)
        release = _require_embedding_release(
            sleap_conda_env=sleap_conda_env,
            sleap_bin=sleap_bin,
            cancel_token=ctx.cancel_token if ctx is not None else cancel_token,
        )
        embedding_observed = {
            # The shape check puts the embedding model last.
            "embedding_model_id": resolved_models.members[-1],
            "embedding_dim": str(dim),
            "sleap_nn_version": release,
        }
        embedding = _EmbeddingRun(dim=dim, sleap_nn_version=release)

    settings = sleap_settings(params, model_id=resolved_models.model_id)
    scope_entries = scope.op_entries if scope is not None else None
    media_scope = ds.resolve_media_scope(scope_entries)
    # Before the mint, so a run refused for its media records no variant.
    work_items = build_work_items(ds, media_scope, kind=SLEAP_KIND, media=params.media)
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
            **observed_model_source(resolved_models),
            **embedding_observed,
        },
    )
    if not media_scope:
        print("[run_sleap] No media entries match the given scope.", file=sys.stderr)
        return minted.run_id

    # sleap-nn track and predict take a frame selection as one "start-end" token.
    analysis_range = params.analysis_range
    frames_arg = f"{analysis_range[0]}-{analysis_range[1]}" if analysis_range else None
    # One probe for the run, in the environment that the run places SLEAP in. A
    # codec outside SLEAP's declaration that decodes is tested once per run.
    decode_probe = DecodeProbe(
        SLEAP_ENV.placed(conda_env=sleap_conda_env, bin_path=sleap_bin)
    )
    recorded_unmatched = _recorded_unmatched(ds, minted.run_id)

    def track_one(job: EntryJob) -> SleapIndexRow | None:
        """One entry: the gated inference phase, the ensured export and readback, the bridge."""
        item, work_dir, seq_ctx = job.item, job.work_dir, job.ctx
        slp_path = work_dir / f"{item.key}.predictions.slp"
        h5_path = work_dir / f"{item.key}.analysis.h5"
        readback_path = work_dir / f"{item.key}.embeddings.npz"
        no_detections_path = work_dir / f"{item.key}.no-detections.json"
        axis = item.entry_axis(windowed=bool(params.frame_window))
        idle = _entry_idle_limit(params, axis, embedding_run=embedding is not None)

        # A .slp alone appears as inference proceeds; the .h5 only after the
        # export finished, so requiring both is what distinguishes a finished
        # pre-marker directory from one killed partway.
        adopt_completed_directory(
            job.ds,
            work_dir,
            minted.run_id,
            item=item,
            required=("*.predictions.slp", "*.analysis.h5"),
            record=(AdoptEvidence("track", "*.predictions.slp"),),
        )

        reusable = reusable_output(
            job.ds,
            work_dir,
            "track",
            params_hash=minted.params_hash,
            item=item,
        )
        if reusable is None:
            clear_phase_marker(work_dir, "track")
            clear_outputs(work_dir, SLEAP_KIND, "track")
            track_claim = claim(seq_ctx, work_dir, "track", idle)
            seq_ctx.progress.on_phase("track", item.key)
            # SLEAP opens the path itself, so an imgstore recording resolves to
            # the plain video export-store wrote for it. Resolved here rather
            # than before the reuse gate: an entry already tracked needs no
            # export, and demanding one would fail a re-run over finished work.
            tool_input = resolve_tool_input(
                job.ds,
                item,
                kind=SLEAP_KIND,
                decode_probe=decode_probe,
                cancel_check=seq_ctx.cancel_token.is_cancelled,
            )
            on_activity = phase_activity(seq_ctx, work_dir, track_claim, idle)
            if embedding is not None:
                nothing_to_embed = run_sleap_predict(
                    tool_input,
                    slp_path,
                    model_paths=resolved_models.paths,
                    track_matching_method=params.track_matching_method,
                    tracking_window_size=params.tracking_window_size,
                    max_tracks=params.max_tracks,
                    max_instances=params.max_instances,
                    peak_threshold=params.peak_threshold,
                    batch_size=params.batch_size,
                    frames=frames_arg,
                    device=params.device,
                    extra_settings=params.sleap_extra_settings,
                    idle_timeout=idle,
                    max_runtime=params.max_runtime,
                    sleap_conda_env=sleap_conda_env,
                    sleap_bin=sleap_bin,
                    cancel_check=seq_ctx.cancel_token.is_cancelled,
                    on_activity=on_activity,
                )
                if nothing_to_embed is None:
                    slp_out = slp_path
                else:
                    _write_no_detections(
                        no_detections_path,
                        nothing_to_embed,
                        sleap_nn_version=embedding.sleap_nn_version,
                    )
                    slp_out = no_detections_path
            else:
                slp_out = run_sleap_track(
                    tool_input,
                    slp_path,
                    model_paths=resolved_models.paths,
                    tracking=params.tracking,
                    use_flow=params.use_flow,
                    candidates_method=params.candidates_method,
                    features=params.features,
                    scoring_method=params.scoring_method,
                    track_matching_method=params.track_matching_method,
                    tracking_window_size=params.tracking_window_size,
                    max_tracks=params.max_tracks,
                    max_instances=params.max_instances,
                    peak_threshold=params.peak_threshold,
                    batch_size=params.batch_size,
                    frames=frames_arg,
                    device=params.device,
                    extra_settings=params.sleap_extra_settings,
                    idle_timeout=params.idle_timeout,
                    max_runtime=params.max_runtime,
                    sleap_conda_env=sleap_conda_env,
                    sleap_bin=sleap_bin,
                    cancel_check=seq_ctx.cancel_token.is_cancelled,
                    on_activity=on_activity,
                ).slp_path
            track_marker = record_phase(
                job.ds,
                work_dir,
                "track",
                ctx=seq_ctx,
                run_id=minted.run_id,
                params_hash=minted.params_hash,
                item=item,
                output=slp_out,
            )
            recomputed = True
        else:
            track_marker, slp_out = reusable
            recomputed = False
        # The marker's recorded output names the no-detections file of an
        # entry without a detection, and nothing derives from that file.
        no_detections = (
            embedding is not None and slp_out.name == no_detections_path.name
        )

        # Analysis export: deterministic and cheap, so ensured rather than
        # marker-gated. Re-run only when its output is missing or the inference
        # it derives from was just recomputed. Published atomically:
        # sleap-convert truncates its output the instant h5py opens it, so a
        # killed export would otherwise leave a partial .h5 at the canonical
        # path -- and this step, unlike inference, is gated on existence rather
        # than on a completion marker. The rename also settles the case where
        # sleap-convert names its output per video rather than by -o.
        if not no_detections and (recomputed or not h5_path.exists()):
            h5_path.unlink(missing_ok=True)
            h5_tmp = work_dir / f"{item.key}.analysis.h5.partial"
            h5_tmp.unlink(missing_ok=True)
            export_claim = claim(seq_ctx, work_dir, "track", idle)
            convert_result = run_sleap_convert(
                slp_out,
                h5_tmp,
                idle_timeout=idle,
                max_runtime=params.max_runtime,
                sleap_conda_env=sleap_conda_env,
                sleap_bin=sleap_bin,
                cancel_check=seq_ctx.cancel_token.is_cancelled,
                on_activity=phase_activity(seq_ctx, work_dir, export_claim, idle),
            )
            os.replace(convert_result.analysis_h5_path, h5_path)
            h5_tmp.unlink(missing_ok=True)

        # The readback of an embedding run's vectors is ensured on the same
        # rule as the export, and publishes through its own ``.partial`` file.
        if (
            embedding is not None
            and not no_detections
            and (recomputed or not readback_path.exists())
        ):
            readback_claim = claim(seq_ctx, work_dir, "track", idle)
            _ = read_back_embeddings(
                slp_out,
                readback_path,
                sleap_conda_env=sleap_conda_env,
                sleap_bin=sleap_bin,
                cancel_check=seq_ctx.cancel_token.is_cancelled,
                on_activity=phase_activity(seq_ctx, work_dir, readback_claim, idle),
            )

        row = SleapIndexRow(
            run_id=minted.run_id,
            group=item.group,
            sequence=item.sequence,
            # What this entry's media was when the run read it. The
            # tracker identity carries no media term, so without this a
            # re-transcode leaves the run reading as current over
            # different pixels.
            consumed_media_composition=media_composition_for(
                job.ds, item.group, item.sequence
            ),
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
            # Blank for an entry without a detection, which has neither file.
            slp_path="" if no_detections else job.ds.relative_to_root(slp_out),
            analysis_h5_path=(
                "" if no_detections else job.ds.relative_to_root(h5_path)
            ),
            media=item.media,
            # A reused table measures nothing, so its recorded count stands.
            n_vectors_unmatched=recorded_unmatched.get((item.group, item.sequence), ""),
        )

        if not params.convert_to_tracks:
            return row

        def bridge() -> BridgeCounts:
            """Publish the entry's tables, empty for an entry without a detection."""
            overwrite_table = job.overwrite or recomputed
            checkpoints = list(resolved_models.significant_files)
            if no_detections and embedding is not None:
                return _bridge_no_detections(
                    job.ds,
                    item.group,
                    item.sequence,
                    slp_out,
                    tracks_variant=minted.tracks_variant,
                    producer_run_id=minted.run_id,
                    consumed_media=item.consumed_media,
                    model_checkpoints=checkpoints,
                    overwrite=overwrite_table,
                    axis=axis,
                    recorded_first_frame=job.recorded_first_frame,
                    dim=embedding.dim,
                )
            return bridge_analysis_h5_to_tracks(
                job.ds,
                item.group,
                item.sequence,
                h5_path,
                tracks_variant=minted.tracks_variant,
                producer_run_id=minted.run_id,
                consumed_media=item.consumed_media,
                model_checkpoints=checkpoints,
                fps=item.fps,
                overwrite=overwrite_table,
                axis=axis,
                recorded_first_frame=job.recorded_first_frame,
                readback=readback_path if embedding is not None else None,
            )

        bridged = publish_or_record(job.ctx, item.key, bridge, kind=SLEAP_KIND)
        if bridged is None:
            return row
        unmatched = bridged.n_vectors_unmatched
        return dataclasses.replace(
            row,
            n_ids=bridged.n_ids,
            n_vectors_unmatched=(
                row.n_vectors_unmatched if unmatched is None else str(unmatched)
            ),
        )

    return run_tracker(
        ds,
        kind=SLEAP_KIND,
        target=sleap_nn_command(params),
        minted=minted,
        work_items=work_items,
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
register_reconcilable_index(SLEAP_KIND, sleap_index)
