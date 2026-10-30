"""``prepare-embedding-data``: a sleap tracker run in, SLEAP training labels out.

An embedding model learns from tracks, and a dataset without identity labels has
tracker tracks only. This op cuts short windows from one sleap tracker run and
writes them to ``models/prepare-embedding-data/<run_id>/labels.slp`` and
``labels.val.slp``, one video per window, which ``train-sleap`` reads by this run's
id.

Each entry's tracked instances are read back from its ``predictions.slp`` in the
SLEAP environment, the windows are planned here
(:func:`~mosaic.tracking.sleap.embedding_windows.plan_windows`), and a second
snippet in that environment writes the planned instances as user instances, with
their frames embedded.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Final, Self

from pydantic import Field, ValidationInfo, field_validator, model_validator

from mosaic.core.entry import Entry, make_entry_key, validate_entry_name
from mosaic.core.params import Declared, Params
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.entry_claim import phase_activity
from mosaic.core.pipeline.error_kind import ClassifiedError, ErrorKind
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.index_csv import (
    IndexCSV,
    RunIndexRowBase,
    finished_rows,
    index_records,
    project_to_schema,
)
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.markers import clear_inflight
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.core.pipeline.op_identity import op_run_id, parse_op_run_id
from mosaic.core.pipeline.ops import Op, OpIdentity, register_op
from mosaic.core.pipeline.subprocess_util import command_summary, run_supervised
from mosaic.core.recording import RecordingKey, recording_name
from mosaic.core.pipeline.toolenv import subprocess_env
from mosaic.core.track_library.sleap import read_appearance_readback
from mosaic.tracking.common.entry_ref import EntryRef
from mosaic.tracking.sleap.dataset_runs import sleap_index_path
from mosaic.tracking.sleap.embedding_windows import EntryWindows, plan_windows
from mosaic.tracking.sleap.labels import validation_labels_path
from mosaic.tracking.sleap.run import (
    SleapError,
    read_back_embeddings,
    sleap_invocation,
)
from mosaic.tracking.sleap.version import PREPARE_EMBEDDING_DATA_KIND, SLEAP_KIND

from ._common import claim_run_root, empty_claimed_run_root, ensure_models_root

if TYPE_CHECKING:
    import pandas as pd

    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingDataIndexRow",
    "EmbeddingLabelsFile",
    "EmbeddingSource",
    "PrepareEmbeddingDataConflict",
    "PrepareEmbeddingDataInvalid",
    "PrepareEmbeddingDataNotFound",
    "PrepareEmbeddingDataOp",
    "PrepareEmbeddingDataParams",
    "embedding_data_index",
    "prepare_embedding_data_run_id",
    "write_embedding_labels",
]

PREPARE_EMBEDDING_VERSION: Final = "0.1"

_PREPARE_IDLE_SECONDS: Final = 900.0
"""How long a snippet of the preparation may print nothing.

Each snippet prints once per predictions file it has loaded, and the writer
prints every :data:`_HEARTBEAT_SECONDS` while it embeds a file's frames.
"""

_HEARTBEAT_SECONDS: Final = 60.0
"""How often the writer prints while it embeds the frames of one labels file."""

_MIN_TRACKS_PER_FILE: Final = 2
"""The fewest tracks a labels file holds. One track gives no negative pair."""

_SLEAP_RUN_ID_DESCRIPTION = "The run id of the sleap tracker run to cut windows from."

_TRAIN_ENTRIES_DESCRIPTION = "The entries written to labels.slp."

_VAL_ENTRIES_DESCRIPTION = (
    "The entries written to labels.val.slp. No entry shares a recording with "
    "train_entries, so validation measures another recording."
)

_WINDOW_FRAMES_DESCRIPTION = "The length of one window."

_FRAMES_PER_WINDOW_DESCRIPTION = (
    "How many frames are taken from a window, evenly spaced over its "
    "qualifying frames. A window with fewer qualifying frames gives all of them."
)

_MIN_TRACKS_PRESENT_DESCRIPTION = (
    "A frame qualifies when at least this many tracks have an instance in it."
)

_SPLIT_TRACKS_AT_GAP_DESCRIPTION = (
    "A track with two consecutive instances inside one window further apart "
    "than this is cut between them into two tracks. Unset keeps each track whole "
    "within a window."
)


class PrepareEmbeddingDataParams(Params):
    """Declares how training windows are cut from one sleap tracker run.

    Both entry lists are stored sorted. The order they are given in moves no
    identifier. Frames are chosen by frame index alone, and nothing is drawn at
    random.
    """

    sleap_run_id: Annotated[
        str, Field(min_length=1), Declared(_SLEAP_RUN_ID_DESCRIPTION)
    ]
    train_entries: Annotated[
        list[EntryRef], Field(min_length=1), Declared(_TRAIN_ENTRIES_DESCRIPTION)
    ]
    val_entries: Annotated[
        list[EntryRef], Field(min_length=1), Declared(_VAL_ENTRIES_DESCRIPTION)
    ]
    window_frames: Annotated[
        int, Field(ge=1), Declared(_WINDOW_FRAMES_DESCRIPTION, unit="frames")
    ] = 300
    frames_per_window: Annotated[
        int, Field(ge=1), Declared(_FRAMES_PER_WINDOW_DESCRIPTION)
    ] = 16
    min_tracks_present: Annotated[
        int, Field(ge=2), Declared(_MIN_TRACKS_PRESENT_DESCRIPTION)
    ] = 2
    split_tracks_at_gap: Annotated[
        int | None,
        Field(ge=1),
        Declared(_SPLIT_TRACKS_AT_GAP_DESCRIPTION, unit="frames"),
    ] = None

    @field_validator("train_entries", "val_entries")
    @classmethod
    def _sorted_without_repeats(
        cls, value: list[EntryRef], info: ValidationInfo
    ) -> list[EntryRef]:
        """Sort the entries, and refuse one named twice."""
        ordered = sorted(value, key=lambda ref: ref.entry)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous.entry == current.entry:
                msg = f"{info.field_name} names {current.entry!r} more than once"
                raise ValueError(msg)
        return ordered

    @model_validator(mode="after")
    def _lists_are_disjoint(self) -> Self:
        """Refuse an entry that is both trained and validated on."""
        validated = {ref.entry for ref in self.val_entries}
        shared = [ref.entry for ref in self.train_entries if ref.entry in validated]
        if shared:
            named = ", ".join(repr(entry) for entry in shared)
            msg = f"train_entries and val_entries both name {named}"
            raise ValueError(msg)
        return self


# --- Errors ----------------------------------------------------------------------


class PrepareEmbeddingDataInvalid(ClassifiedError, ValueError):
    """Refuses a ``sleap_run_id`` that names a run of another op.

    Also refuses train and validation entries of one recording.
    """

    error_kind: ClassVar[ErrorKind] = "invalid"


class PrepareEmbeddingDataNotFound(ClassifiedError, LookupError):
    """Reports a sleap run, or an entry of one, that the dataset does not hold.

    Raised for a run that no sleap index row names, an entry without a finished
    row of the run, and an entry whose ``predictions.slp`` is gone.
    """

    error_kind: ClassVar[ErrorKind] = "not_found"


class PrepareEmbeddingDataConflict(ClassifiedError, ValueError):
    """Refuses a labels file whose windows hold fewer than two tracks."""

    error_kind: ClassVar[ErrorKind] = "conflict"


# --- Index -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EmbeddingDataIndexRow(RunIndexRowBase):
    """Typed row for ``models/prepare-embedding-data/index.csv``.

    ``artifact_path`` is the root-relative ``labels.slp``, the cell that
    ``prepared_artifact_cell`` reads. ``consumed_predictions`` is a JSON list of
    ``{group, sequence, digest, size, mtime_ns}``, one per ``predictions.slp``
    read, sorted by entry: the file's content digest, its size and its
    modification time when it was read. No cell enters the run identifier.

    ``n_windows`` counts the windows of both files, one video each, and
    ``n_frames`` their frames, of which ``n_train`` are in ``labels.slp`` and
    ``n_valid`` in ``labels.val.slp``. ``n_tracks`` counts the distinct window
    tracks of both files.
    """

    kind: str
    artifact_path: str
    sleap_run_id: str
    consumed_predictions: str
    n_windows: int
    n_frames: int
    n_train: int
    n_valid: int
    n_tracks: int
    status: str


_EMBEDDING_DATA_INDEX_COLUMNS: Final = tuple(
    field.name for field in fields(EmbeddingDataIndexRow)
)


def _adopt_embedding_data_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bring an index read off disk up to the current schema, inside the write lock."""
    return project_to_schema(df, _EMBEDDING_DATA_INDEX_COLUMNS)


def embedding_data_index(path: Path) -> IndexCSV[EmbeddingDataIndexRow]:
    """The embedding preparation index at *path*, one row per run."""
    return IndexCSV(
        path,
        EmbeddingDataIndexRow,
        dedup_keys=["run_id"],
        adopt=_adopt_embedding_data_columns,
    )


def prepare_embedding_data_run_id(params: PrepareEmbeddingDataParams) -> str:
    """Mint an embedding preparation's run identifier from its params alone.

    ``sleap_run_id`` names the tracker's models and settings. No path and no
    digest of the tracker's output is hashed: ``consumed_predictions`` records
    the files read, and the train-sleap identity digests the labels written.
    """
    return op_run_id(
        PREPARE_EMBEDDING_DATA_KIND,
        PREPARE_EMBEDDING_VERSION,
        {"params": params.identity_dump()},
    )


# --- Labels writer ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EmbeddingSource:
    """One entry's predictions file and the windows planned from it.

    Attributes:
        predictions: The entry's ``predictions.slp``.
        windows: Its planned windows.
    """

    predictions: Path
    windows: EntryWindows


@dataclass(frozen=True, slots=True)
class EmbeddingLabelsFile:
    """One labels file and the entries written to it, in order.

    Attributes:
        out: Where the file is published.
        sources: The entries, each window of which becomes one video.
    """

    out: Path
    sources: tuple[EmbeddingSource, ...]


# Run by the SLEAP interpreter. Its only input is argv: a JSON request naming,
# per labels file, where to write it and, per entry, the predictions .slp and the
# planned windows, and how often to print while a file's frames are embedded.
# Per (frame, track), a user instance is kept over a predicted one, and the last
# predicted instance over an earlier one, the rule of the analysis export.
# Each kept instance is written as a user instance, the only kind that sleap-nn
# trains on. Each window becomes its own Video over the source file, and embed=True
# copies its frames into the labels file. A window's video opens a reader for
# each frame it reads and closes it again, since a reader kept open would leave
# one open per window. The save decodes and encodes every frame. A thread prints
# while it runs, and the supervisor counts each line as activity.
_WRITE_LABELS_SNIPPET: str = """
import json
import os
import sys
import threading
import time

import sleap_io

with open(sys.argv[1], encoding="utf-8") as _handle:
    _request = json.load(_handle)


def _beat(stop, out, every):
    started = time.monotonic()
    while not stop.wait(every):
        elapsed = time.monotonic() - started
        print(f"embedding frames into {out}: {elapsed:.0f} s", flush=True)


_skeleton = None
for _file in _request["files"]:
    _tracks = {}
    _videos = []
    _frames = []
    for _source in _file["sources"]:
        _predicted = _source["predictions"]
        _labels = sleap_io.load_slp(_predicted)
        print(f"loaded {_predicted}", flush=True)
        if len(_labels.videos) != 1:
            sys.exit(f"{_predicted} holds {len(_labels.videos)} videos, not one")
        _video = _labels.videos[0]
        _named = _video.filename
        for _path in _named if isinstance(_named, list) else [_named]:
            if not os.path.exists(_path):
                sys.exit(f"the video {_path} that {_predicted} names is missing")
        if _skeleton is None:
            _skeleton = _labels.skeletons[0]
        _slots = {}
        for _frame in _labels.labeled_frames:
            for _instance in _frame.instances:
                if _instance.track is None:
                    continue
                _key = (int(_frame.frame_idx), str(_instance.track.name))
                _held = _slots.get(_key)
                _user = not isinstance(_instance, sleap_io.PredictedInstance)
                if (
                    _held is None
                    or _user
                    or isinstance(_held, sleap_io.PredictedInstance)
                ):
                    _slots[_key] = _instance
        for _window in _source["windows"]:
            _window_video = sleap_io.Video.from_filename(_named, keep_open=False)
            _videos.append(_window_video)
            _kept = {}
            for _planned in _window["instances"]:
                _instance = _slots.get((_planned["frame"], _planned["track"]))
                if _instance is None:
                    sys.exit(
                        f"{_predicted} has no instance of track "
                        f"{_planned['track']!r} in frame {_planned['frame']}"
                    )
                _name = _planned["window_track"]
                if _name not in _tracks:
                    _tracks[_name] = sleap_io.Track(name=_name)
                _kept.setdefault(_planned["frame"], []).append(
                    sleap_io.Instance.from_numpy(
                        _instance.numpy(), skeleton=_skeleton, track=_tracks[_name]
                    )
                )
            for _frame_idx in _window["frames"]:
                _frames.append(
                    sleap_io.LabeledFrame(
                        video=_window_video,
                        frame_idx=_frame_idx,
                        instances=_kept.get(_frame_idx, []),
                    )
                )
    _written = sleap_io.Labels(
        labeled_frames=_frames,
        videos=_videos,
        skeletons=[] if _skeleton is None else [_skeleton],
        tracks=list(_tracks.values()),
    )
    _stop = threading.Event()
    _beater = threading.Thread(
        target=_beat,
        args=(_stop, _file["out"], _request["heartbeat_seconds"]),
        daemon=True,
    )
    _beater.start()
    try:
        sleap_io.save_slp(_written, _file["out"], embed=True)
    finally:
        _stop.set()
        _beater.join()
    print(
        f"wrote {len(_videos)} videos and {len(_tracks)} tracks to {_file['out']}",
        flush=True,
    )
"""


def _partial(path: Path) -> Path:
    return path.with_name(f"{path.name}.partial")


def _request_document(
    files: Sequence[EmbeddingLabelsFile], heartbeat_seconds: float
) -> str:
    """The writer's request: per file, its staging path and its planned windows."""
    document = {
        "heartbeat_seconds": heartbeat_seconds,
        "files": [
            {
                "out": str(_partial(each.out)),
                "sources": [
                    {
                        "predictions": str(source.predictions),
                        "windows": [
                            {
                                "index": window.index,
                                "frames": list(window.frames),
                                "instances": [
                                    {
                                        "frame": instance.frame,
                                        "track": instance.track,
                                        "window_track": instance.window_track,
                                    }
                                    for instance in window.instances
                                ],
                            }
                            for window in source.windows.windows
                        ],
                    }
                    for source in each.sources
                ],
            }
            for each in files
        ],
    }
    return json.dumps(document, sort_keys=True)


def write_embedding_labels(
    files: Sequence[EmbeddingLabelsFile],
    request: Path,
    *,
    sleap_conda_env: str | None = None,
    sleap_bin: Path | str | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_activity: Callable[[str], None] | None = None,
    heartbeat_seconds: float = _HEARTBEAT_SECONDS,
) -> None:
    """Write each of *files* from its planned windows, in the SLEAP environment.

    The SLEAP interpreter reads *request*, written here, and writes each file to
    ``<out>.partial``. Once it exits cleanly, every staged file is moved onto its
    ``out`` with ``os.replace``. *request* and the staged files are removed on
    every path.

    Args:
        files: The labels files to write.
        request: Where the request is written.
        sleap_conda_env: Run SLEAP inside this conda env (overrides
            ``MOSAIC_SLEAP_CONDA_ENV``).
        sleap_bin: Path to a SLEAP console script (overrides
            ``MOSAIC_SLEAP_BIN``).
        cancel_check: Polled while the snippet runs.
        on_activity: Called with each line of either output stream.
        heartbeat_seconds: How often the snippet prints while it embeds the
            frames of one file.

    Raises:
        SleapNotFoundError: No SLEAP install could be located.
        SleapError: The snippet exited non-zero, for instance because the video
            that a predictions file names is missing. The message names it.
        FileNotFoundError: It exited cleanly and staged no file.
    """
    staged = [_partial(each.out) for each in files]
    document = _request_document(files, heartbeat_seconds)
    atomic_write(request, lambda path: path.write_text(document, encoding="utf-8"))
    invocation = sleap_invocation(
        "python", sleap_conda_env=sleap_conda_env, sleap_bin=sleap_bin
    )
    cmd = [*invocation, "-c", _WRITE_LABELS_SNIPPET, str(request)]
    logger.info("Running: %s", command_summary(cmd, SleapError.head))
    try:
        stdout, stderr, returncode = run_supervised(
            cmd,
            env=subprocess_env(),
            cancel_check=cancel_check,
            idle_timeout=_PREPARE_IDLE_SECONDS,
            on_activity=on_activity,
        )
        if returncode != 0:
            raise SleapError(cmd, returncode, stdout, stderr)
        for partial in staged:
            if not partial.is_file():
                msg = f"sleap-io reported success but wrote no .slp at {partial}"
                raise FileNotFoundError(msg)
        for each, partial in zip(files, staged, strict=True):
            os.replace(partial, each.out)
    finally:
        request.unlink(missing_ok=True)
        for partial in staged:
            partial.unlink(missing_ok=True)


# --- The run ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SourceEntry:
    """One entry of the tracker run, and its predictions file on disk."""

    entry: Entry
    predictions: Path


def _sleap_run_rows(ds: Dataset, run_id: str) -> list[dict[str, str]]:
    """Every index row of sleap run *run_id*, or none without an index."""
    import pandas as pd

    index_path = sleap_index_path(ds)
    if not index_path.exists():
        return []
    frame = pd.read_csv(index_path, dtype="string", keep_default_na=False)
    return [row for row in index_records(frame) if row.get("run_id") == run_id]


def _source_entries(
    ds: Dataset, params: PrepareEmbeddingDataParams
) -> dict[Entry, _SourceEntry]:
    """Find the predictions file of every entry the params name, or refuse.

    Raises:
        PrepareEmbeddingDataInvalid: ``sleap_run_id`` names a run of another op.
        PrepareEmbeddingDataNotFound: No sleap index row names the run, an entry
            has no finished row of it, an entry's row records no predictions
            because its run found no detections, or an entry's
            ``predictions.slp`` is gone.
    """
    run_id = params.sleap_run_id
    parsed = parse_op_run_id(run_id)
    if parsed is None or parsed.kind != SLEAP_KIND:
        msg = (
            f"[{PREPARE_EMBEDDING_DATA_KIND}] sleap_run_id {run_id!r} is not the run id "
            f"of a {SLEAP_KIND} run"
        )
        raise PrepareEmbeddingDataInvalid(msg)
    rows = _sleap_run_rows(ds, run_id)
    if not rows:
        msg = (
            f"[{PREPARE_EMBEDDING_DATA_KIND}] no {SLEAP_KIND} run {run_id!r} is indexed"
        )
        raise PrepareEmbeddingDataNotFound(msg)
    finished = finished_rows(rows, run_id, ("group", "sequence"))

    found: dict[Entry, _SourceEntry] = {}
    for ref in (*params.train_entries, *params.val_entries):
        key = make_entry_key(*ref.entry)
        row = finished.get(ref.entry)
        if row is None:
            msg = (
                f"[{PREPARE_EMBEDDING_DATA_KIND}] {SLEAP_KIND} run {run_id!r} has no "
                f"finished row for entry {key!r}"
            )
            raise PrepareEmbeddingDataNotFound(msg)
        stored = row.get("slp_path", "")
        if not stored:
            msg = (
                f"[{PREPARE_EMBEDDING_DATA_KIND}] entry {key!r} of {SLEAP_KIND} run "
                f"{run_id!r} has no detections to cut windows from"
            )
            raise PrepareEmbeddingDataNotFound(msg)
        predictions = Path(ds.resolve_path(stored))
        if not predictions.is_file():
            msg = (
                f"[{PREPARE_EMBEDDING_DATA_KIND}] the predictions of entry {key!r} in "
                f"{SLEAP_KIND} run {run_id!r} are gone: {predictions}"
            )
            raise PrepareEmbeddingDataNotFound(msg)
        found[ref.entry] = _SourceEntry(entry=ref.entry, predictions=predictions)
    return found


def _refuse_shared_recordings(ds: Dataset, params: PrepareEmbeddingDataParams) -> None:
    """Refuse train and validation entries that belong to one recording.

    The divisions of a divided recording are one recording, so a validation
    score of one of them would measure a recording the model trained on.

    Raises:
        PrepareEmbeddingDataInvalid: A train entry and a validation entry
            belong to one recording. The first such pair is named.
    """
    trained: dict[RecordingKey, Entry] = {}
    for ref in params.train_entries:
        _ = trained.setdefault(ds.recording_of(*ref.entry), ref.entry)
    for ref in params.val_entries:
        recording = ds.recording_of(*ref.entry)
        train_entry = trained.get(recording)
        if train_entry is not None:
            msg = (
                f"[{PREPARE_EMBEDDING_DATA_KIND}] train entry "
                f"{make_entry_key(*train_entry)!r} and validation entry "
                f"{make_entry_key(*ref.entry)!r} belong to recording "
                f"{recording_name(recording)!r}. Validation measures another "
                "recording: put every entry of one recording on one side."
            )
            raise PrepareEmbeddingDataInvalid(msg)


def _consumed_predictions_cell(sources: Sequence[_SourceEntry]) -> str:
    """The JSON list of every predictions file read, sorted by entry."""
    consumed: list[dict[str, object]] = []
    for source in sorted(sources, key=lambda each: each.entry):
        stat = source.predictions.stat()
        consumed.append(
            {
                "group": source.entry[0],
                "sequence": source.entry[1],
                "digest": file_digest(source.predictions),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return json.dumps(consumed, sort_keys=True)


@dataclass(frozen=True, slots=True)
class _ConsumedFile:
    """One predictions file as it was when the preparation read it."""

    entry: Entry
    digest: str
    size: int
    mtime_ns: int


def _consumed_files(cell: str) -> list[_ConsumedFile] | None:
    """Parse a ``consumed_predictions`` cell, or return None when it is unreadable."""
    try:
        records = json.loads(cell)
        return [
            _ConsumedFile(
                entry=(str(record["group"]), str(record["sequence"])),
                digest=str(record["digest"]),
                size=int(record["size"]),
                mtime_ns=int(record["mtime_ns"]),
            )
            for record in records
        ]
    except (ValueError, KeyError, TypeError):
        return None


def _changed_predictions(
    ds: Dataset, sleap_run_id: str, consumed: Sequence[_ConsumedFile]
) -> list[Entry]:
    """The entries whose predictions file now has other bytes than when it was read.

    A file is hashed only when its size or modification time differs from the
    recorded ones. A file that is gone, swept with its tracker run, or that the
    run's index no longer names, is not compared.
    """
    finished = finished_rows(
        _sleap_run_rows(ds, sleap_run_id), sleap_run_id, ("group", "sequence")
    )
    changed: list[Entry] = []
    for each in consumed:
        stored = finished.get(each.entry, {}).get("slp_path", "")
        if not stored:
            continue
        path = Path(ds.resolve_path(stored))
        try:
            stat = path.stat()
        except OSError:
            continue
        if (stat.st_size, stat.st_mtime_ns) == (each.size, each.mtime_ns):
            continue
        if file_digest(path) != each.digest:
            changed.append(each.entry)
    return changed


def _reusable_predictions(
    ds: Dataset, index: IndexCSV[EmbeddingDataIndexRow], run_id: str
) -> str | None:
    """The ``consumed_predictions`` of run *run_id* when it can be reused, else None.

    A run is reusable when its row is finished and both labels files are still
    there. The row is the only evidence that the writer returned, and a row can
    outlive the files it names. An index that cannot be read is no evidence, and
    the preparation runs again.
    """
    if not index.path.exists():
        return None
    try:
        rows = index.read(run_id=run_id)
    except (OSError, ValueError, KeyError):
        return None
    if rows.empty:
        return None
    row = rows.iloc[-1]
    stored = str(row.get("artifact_path", "") or "")
    if not str(row.get("finished_at", "") or "") or not stored:
        return None
    labels = ds.resolve_path(stored)
    if not labels.is_file() or not validation_labels_path(labels).is_file():
        return None
    return str(row.get("consumed_predictions", "") or "")


def _reuse_message(run_id: str, sleap_run_id: str, changed: Sequence[Entry]) -> str:
    named = ", ".join(repr(make_entry_key(*entry)) for entry in changed)
    return (
        f"[{PREPARE_EMBEDDING_DATA_KIND}] reusing {run_id}, but the predictions of "
        f"{named} in {SLEAP_KIND} run {sleap_run_id!r} changed after it was "
        f"prepared. Run again with overwrite to prepare again."
    )


def _reuse(
    ds: Dataset,
    params: PrepareEmbeddingDataParams,
    run_id: str,
    consumed_cell: str,
    ctx: JobContext,
) -> None:
    """Report a finished preparation as reused, naming predictions that changed."""
    consumed = _consumed_files(consumed_cell)
    changed = (
        _changed_predictions(ds, params.sleap_run_id, consumed) if consumed else []
    )
    if changed:
        logger.warning(_reuse_message(run_id, params.sleap_run_id, changed))
    ctx.cache_hit()


def _window_tracks(labels: EmbeddingLabelsFile) -> set[str]:
    return {
        instance.window_track
        for source in labels.sources
        for window in source.windows.windows
        for instance in window.instances
    }


def _n_frames(labels: EmbeddingLabelsFile) -> int:
    return sum(
        len(window.frames)
        for source in labels.sources
        for window in source.windows.windows
    )


@register_op
class PrepareEmbeddingDataOp(Op[PrepareEmbeddingDataParams]):
    """Cut SLEAP training windows for an embedding model from a sleap tracker run.

    Without ``overwrite``, a finished preparation whose two labels files are
    still there is reused. The labels embed their frames. A swept tracker
    output leaves the preparation valid, and a predictions file that changed
    since is named in a warning.
    """

    kind = PREPARE_EMBEDDING_DATA_KIND
    category = "convert"
    domain = "tracking"
    version = PREPARE_EMBEDDING_VERSION
    scope_takes = "none"
    scope_dependent = False
    Params = PrepareEmbeddingDataParams

    def plan_identity(
        self,
        ds: Dataset,
        params: PrepareEmbeddingDataParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """Name the preparation by its params, without reading the dataset."""
        _ = (ds, scope, require_data)
        return OpIdentity(run_id=prepare_embedding_data_run_id(params))

    def run(
        self,
        ds: Dataset,
        params: PrepareEmbeddingDataParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        ensure_models_root(ds)
        run_id = self.plan_identity(ds, params, scope).run_id
        ctx.set_run_id(run_id)
        _refuse_shared_recordings(ds, params)
        if not overwrite:
            consumed = _reusable_predictions(
                ds, embedding_data_index(model_index_path(ds, self.kind)), run_id
            )
            if consumed is not None:
                _reuse(ds, params, run_id, consumed, ctx)
                return run_id
        sources = _source_entries(ds, params)
        for ref in (*params.train_entries, *params.val_entries):
            _ = validate_entry_name(ref.group, "group")
            _ = validate_entry_name(ref.sequence, "sequence")

        out = model_run_root(ds, self.kind, run_id)
        # Claimed before anything is cleared: two executions of one identifier
        # would otherwise have one delete the other's files mid-write.
        out.mkdir(parents=True, exist_ok=True)
        marker = claim_run_root(
            ds, ctx.execution_id, out, self.kind, _PREPARE_IDLE_SECONDS
        )
        empty_claimed_run_root(out)
        on_activity = phase_activity(ctx, out, marker, _PREPARE_IDLE_SECONDS)

        readbacks = out / "readback"
        planned: dict[Entry, EmbeddingSource] = {}
        for entry, source in sources.items():
            key = make_entry_key(*entry)
            readback = read_back_embeddings(
                source.predictions,
                readbacks / f"{key}.embeddings.npz",
                cancel_check=ctx.cancel_token.is_cancelled,
                on_activity=on_activity,
            )
            windows = plan_windows(
                read_appearance_readback(readback),
                entry_key=key,
                window_frames=params.window_frames,
                frames_per_window=params.frames_per_window,
                min_tracks_present=params.min_tracks_present,
                split_tracks_at_gap=params.split_tracks_at_gap,
            )
            planned[entry] = EmbeddingSource(
                predictions=source.predictions, windows=windows
            )
        shutil.rmtree(readbacks)

        labels = out / "labels.slp"
        files = (
            EmbeddingLabelsFile(
                out=labels,
                sources=tuple(planned[ref.entry] for ref in params.train_entries),
            ),
            EmbeddingLabelsFile(
                out=validation_labels_path(labels),
                sources=tuple(planned[ref.entry] for ref in params.val_entries),
            ),
        )
        write_embedding_labels(
            files,
            out / "labels.request.json",
            cancel_check=ctx.cancel_token.is_cancelled,
            on_activity=on_activity,
        )
        ctx.check_cancel()

        for each in files:
            n_tracks = len(_window_tracks(each))
            if n_tracks < _MIN_TRACKS_PER_FILE:
                empty_claimed_run_root(out)
                clear_inflight(out, execution_id=ctx.execution_id)
                msg = (
                    f"[{self.kind}] {each.out.name} holds {n_tracks} track(s), "
                    f"fewer than {_MIN_TRACKS_PER_FILE}: no window of its entries "
                    f"has a frame with {params.min_tracks_present} tracks present"
                )
                raise PrepareEmbeddingDataConflict(msg)

        train, valid = files
        index = embedding_data_index(model_index_path(ds, self.kind))
        index.append(
            [
                EmbeddingDataIndexRow(
                    run_id=run_id,
                    kind=self.kind,
                    artifact_path=ds.relative_to_root(labels),
                    sleap_run_id=params.sleap_run_id,
                    consumed_predictions=_consumed_predictions_cell(
                        list(sources.values())
                    ),
                    n_windows=sum(
                        len(source.windows.windows)
                        for each in files
                        for source in each.sources
                    ),
                    n_frames=_n_frames(train) + _n_frames(valid),
                    n_train=_n_frames(train),
                    n_valid=_n_frames(valid),
                    n_tracks=len(_window_tracks(train) | _window_tracks(valid)),
                    status="finished",
                    abs_path=Path(ds.relative_to_root(out)),
                )
            ]
        )
        index.mark_finished(run_id)
        ctx.entries_written(len(sources))
        return run_id
