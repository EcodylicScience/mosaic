"""The program that imports Ultralytics inside its own environment, and never imports mosaic.

Ultralytics is AGPL-3.0. A mosaic that imports it is one work with it, so
Ultralytics runs in an environment of its own and this program is what runs
inside that environment. Mosaic and this file exchange JSON files and
command-line arguments, which is what keeps them two programs. Nothing here may
grow a ``mosaic`` import; the shared vocabulary lives in ``ultralytics_protocol``
beside this file, which imports neither side.

Invoked as::

    <env>/bin/python ultralytics_runner.py probe --request <req.json> --out <resp.json>
    <env>/bin/python ultralytics_runner.py tracker-defaults --request <req.json> --out <resp.json>
    <env>/bin/python ultralytics_runner.py track --request <req.json> --out <result.json>

Every subcommand reads its whole request from a JSON file and writes its whole
response to a JSON file, with nothing else on the command line. The response
goes to a **file** rather than to standard output because Ultralytics' own
logger is a stream handler on standard output -- a weights download, a warning,
anything ``verbose=False`` does not suppress lands there -- so a response parsed
from that stream would be fragile. Standard output carries progress lines only.

Importing torch and Ultralytics, by contrast, writes nothing to standard output
at all. That is why ``probe`` and ``tracker-defaults`` are silent from spawn to
answer, and why mosaic gives them a deadline floor rather than bounding their
inactivity.

``ultralytics_protocol`` is imported as a bare top-level module: running a script
puts the script's own directory first on the module search path, the same
arrangement ``kpms_server.py`` uses with ``kpms_protocol``. It follows that
mosaic never imports this module -- only spawns it.

Every Ultralytics import is deferred into a function body, so this file imports
in an environment that has no Ultralytics at all and the argument parser can
still report a usage error there.
"""

# Four rules, for two things about the Ultralytics surface that no code here can
# fix. ``YOLO.track`` is partially typed and a keypoint shape is attribute access
# on a detection head the checker cannot see into, so three ``reportUnknown``
# rules fire across those reads. And a tracked ``Boxes`` carries
# ``Tensor | ndarray`` where :class:`Result` asks for ``ndarray`` -- true only
# after ``.cpu().numpy()``, which is a fact about the call order that no
# annotation expresses, so binding the loaded model to :class:`_Model` is
# reported. Each was checked by removing it: all four still fire.
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportAssignmentType=false

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import sys
import tempfile
import threading
from collections.abc import Callable, Generator, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, Protocol, TypeAlias, TypeVar

import numpy as np
import pandas as pd
from mosaic_media import MediaFacts
from mosaic_media.io import VideoReader
from pydantic import BaseModel, TypeAdapter

from ultralytics_protocol import (
    POINT_COLUMNS,
    InferenceResult,
    InferPointsRequest,
    InferPoseRequest,
    InferRequestBase,
    InferResponse,
    PointResult,
    PoseResult,
    Precision,
    ProbeRequest,
    ProbeResponse,
    ProgressEvent,
    ProgressEventKind,
    Result,
    TrackerDefaultsRequest,
    TrackerDefaultsResponse,
    TrackerSetting,
    TrackRequest,
    TrackResponse,
    point_rows_from_result,
    pose_rows_from_result,
    rows_from_result,
)

ResultT = TypeVar("ResultT", bound=InferenceResult)

QuantizeSpelling: TypeAlias = Literal["quantize", "half"]
"""Which keyword the installed Ultralytics takes half precision under.

Declared here rather than in the protocol because it never crosses the wire:
mosaic sends a precision and this program spells it for the Ultralytics it is
running against, so nothing on the other side has an opinion about it.
"""


# Process umask, read once at import (single-threaded startup). Used to restore
# sensible permissions on temp files, which ``mkstemp`` creates mode 0600. The
# same two lines mosaic's own atomic writer runs, for the same reason.
_UMASK: Final = os.umask(0)
_ = os.umask(_UMASK)


class RunnerError(RuntimeError):
    """The runner cannot answer honestly, and will not answer approximately."""


class _Model(Protocol):
    """The tracking call."""

    def track(self, source: list[np.ndarray], **kwargs: object) -> list[Result]: ...


class _Predictor(Protocol[ResultT]):
    """The inference call, over whichever result the loaded task produces."""

    def predict(self, source: list[np.ndarray], **kwargs: object) -> list[ResultT]: ...


# --- probe -----------------------------------------------------------------


def _is_importable(module_name: str) -> bool:
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


def _installed_tracker_table(
    package_root: Path, tracker: str
) -> dict[str, TrackerSetting]:
    """The backend's own shipped defaults, as scalars.

    Mosaic merges its resolved settings over these. Only ``bool`` / ``int`` /
    ``float`` / ``str`` entries are reported, because a null reaches a backend as
    a missing attribute and every backend reads its knobs off an object that
    raises on one.
    """
    import yaml

    path = package_root / "cfg" / "trackers" / f"{tracker}.yaml"
    table: dict[str, TrackerSetting] = {}
    if not path.is_file():
        return table
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        return table
    for key, value in loaded.items():
        if isinstance(value, (bool, int, float, str)):
            table[str(key)] = value
    return table


def _keypoint_count(model: object, task: str) -> int:
    """How many keypoints this model predicts; 1 for a box-only model.

    Read off the detection **head**, not off the model. ``PoseModel`` never
    assigns ``kpt_shape`` to itself -- only the head does -- so the obvious read
    yields nothing for a checkpoint that did not come through Ultralytics'
    trainer, and a one-keypoint table would be written for a pose model.
    """
    if task != "pose":
        return 1
    inner = getattr(model, "model", None)
    layers = getattr(inner, "model", None)
    head = layers[-1] if layers is not None else None
    shape = getattr(head, "kpt_shape", None) or getattr(inner, "kpt_shape", None)
    if shape is None:
        raise RunnerError(
            "this pose model does not declare a keypoint shape, so the number of "
            "keypoints its predictions carry cannot be told."
        )
    return int(shape[0])


def _has_locate() -> bool:
    """Whether this environment's ``ultralytics`` is the POLO fork.

    Read as the presence of ``LocalizationModel``, which is what the fork adds
    and upstream does not define. There is nothing else to read it from: the two
    share a distribution name, overlap in version, and install the same console
    scripts, so no version comparison and no resolved path distinguishes them.

    ``hasattr`` on the module rather than a ``from ... import``, so the check
    needs no bound name it then has to explain away.
    """
    try:
        from ultralytics.nn import tasks
    except ImportError:
        return False
    return hasattr(tasks, "LocalizationModel")


def _load_model(model_path: str) -> tuple[object | None, str]:
    """The loaded weights, or ``None`` and why not.

    Every exception is caught, which is the point rather than an oversight. What
    the probe is reporting is *whether this environment can load these weights*,
    and the ways it cannot are open-ended: unpickling a class the running build
    does not define raises ``AttributeError``, a missing module raises
    ``ModuleNotFoundError``, a truncated checkpoint raises from ``torch.load``
    itself, an absent file raises ``FileNotFoundError``. A closed list would
    catch four of those and let the fifth reach the user as the traceback this
    exists to replace.
    """
    from ultralytics import YOLO

    try:
        return YOLO(model_path), ""
    except Exception as failure:
        return None, f"{type(failure).__name__}: {failure}"


def _quantize_spelling() -> QuantizeSpelling:
    """Which keyword the installed Ultralytics takes half precision under.

    ``half`` was replaced by ``quantize`` and now emits a deprecation warning, but
    the older spelling is what an older install understands. Probed from the
    shipped default configuration rather than from a version comparison, so this
    keeps working across the transition in both directions.

    ``DEFAULT_CFG_DICT`` is imported from ``ultralytics.utils``, which defines it,
    rather than from ``ultralytics.cfg``, which re-exports it without an
    ``__all__``; the two are the same object.
    """
    from ultralytics.utils import DEFAULT_CFG_DICT

    return "quantize" if "quantize" in DEFAULT_CFG_DICT else "half"


def run_probe(request: ProbeRequest) -> ProbeResponse:
    """Report what this environment holds, refusing nothing.

    Every finding is data mosaic decides about: which install hint to print, and
    whether the installed Ultralytics knows the requested backend. A model that
    fails to load is the one thing not reported -- the exception propagates, the
    process exits non-zero, and mosaic surfaces the captured streams.
    """
    has_lap = _is_importable("lap")
    if not _is_importable("ultralytics"):
        return ProbeResponse(
            has_ultralytics=False,
            has_lap=has_lap,
            has_locate=False,
            ultralytics_version="",
            tracker_names=[],
            model_task="",
            n_keypoints=0,
            model_load_error="",
            installed_tracker_table={},
        )

    import ultralytics
    from ultralytics.trackers.track import TRACKER_MAP

    # Reported whether or not the weights load: what this environment holds is
    # not a question about the checkpoint it was handed.
    model, load_error = _load_model(request.model_path)
    task = "" if model is None else str(getattr(model, "task", ""))
    return ProbeResponse(
        has_ultralytics=True,
        has_lap=has_lap,
        has_locate=_has_locate(),
        ultralytics_version=str(getattr(ultralytics, "__version__", "unknown")),
        tracker_names=sorted(TRACKER_MAP),
        model_task=task,
        n_keypoints=0 if model is None else _keypoint_count(model, task),
        model_load_error=load_error,
        installed_tracker_table=_installed_tracker_table(
            Path(ultralytics.__file__).parent, request.tracker
        ),
    )


# --- tracker defaults ------------------------------------------------------


def run_tracker_defaults(request: TrackerDefaultsRequest) -> TrackerDefaultsResponse:
    """Every backend's shipped configuration table, read in one process.

    *request* carries nothing; it is validated by the caller so that this
    subcommand takes the same shape as the other two. What decides the answer is
    the installed release alone.

    Mosaic transcribes these tables so that an upstream retune cannot silently
    re-mean an identifier already on disk, and reads them back through here to
    find out when one has moved. Every backend in one answer because each spawn
    pays a cold torch import, which is the whole cost of asking.
    """
    del request  # nothing to read: the installed release decides the whole answer

    import ultralytics
    from ultralytics.trackers.track import TRACKER_MAP

    package_root = Path(ultralytics.__file__).parent
    return TrackerDefaultsResponse(
        tables={
            name: _installed_tracker_table(package_root, name)
            for name in sorted(TRACKER_MAP)
        }
    )


# --- track -----------------------------------------------------------------


def _precision_kwarg(precision: Precision) -> dict[str, object]:
    """The half-precision argument, spelled for the installed Ultralytics."""
    if _quantize_spelling() == "quantize":
        return {"quantize": precision}
    return {"half": precision == "fp16"}


def _track_kwargs(request: TrackRequest) -> dict[str, object]:
    """The arguments every frame of this video is tracked with.

    Every detection-affecting argument is passed explicitly rather than left to
    Ultralytics' shipped defaults, so an upstream retune cannot silently re-mean
    an identifier mosaic already minted.

    Frozen once, before the loop, rather than rebuilt per call: ``device`` must be
    byte-identical on every ``track`` call, because Ultralytics rebuilds its
    predictor when the device argument changes and takes the live trackers --
    pools, Kalman state, identity numbering -- with it, mid-video and silently.
    """
    return {
        "persist": True,
        "tracker": request.tracker_yaml,
        "stream": False,
        "conf": request.conf,
        "iou": request.iou,
        "imgsz": request.imgsz,
        "max_det": request.max_det,
        "classes": list(request.classes) if request.classes is not None else None,
        "agnostic_nms": request.agnostic_nms,
        "augment": False,
        "device": request.device,
        # False so the tool does not chatter over the progress lines below, which
        # are what mosaic's inactivity watchdog reads.
        "verbose": False,
        "save": False,
        "save_txt": False,
        "show": False,
        # Pin the run directory even though nothing is saved: Ultralytics computes
        # it eagerly, and an unpinned one walks the shared `runs/` tree.
        "project": request.project_dir,
        "name": "ultralytics",
        "exist_ok": True,
        **_precision_kwarg(request.precision),
    }


_MEDIA_FACTS: Final = TypeAdapter(MediaFacts)
"""Rebuilds a ``MediaFacts`` from its flattened form.

Validating rather than splatting the payload straight into the constructor:
these fields crossed a process boundary as JSON, so a field that arrived under
the wrong type is caught here rather than at frame forty thousand, and the
result is a ``MediaFacts`` the checker can see is one.
"""


def _media_facts(payload: dict[str, object]) -> MediaFacts:
    """Rebuild the facts mosaic measured, gated and flattened for the wire.

    Never ``None``, and the request field is not nullable either. The read-target
    gate -- probe the file, derive its verdict, refuse one that needs transcoding
    before it can be read for analysis -- lives in mosaic and cannot be called
    from here. A reader handed no facts probes with no gate, so a rotated or
    variable-frame-rate original would be read silently at misindexed
    coordinates under a valid identifier. Requiring the payload is what makes the gate
    unskippable from this side of the boundary.

    Injecting measured facts rather than probing here is also what keeps a raw
    ``.h264`` reading with its true frame count instead of the garbage count its
    header declares -- the container that would carry the truth is not there to
    read. An incomplete payload raises out of this validation rather than
    degrading to an ungated probe.
    """
    return _MEDIA_FACTS.validate_python(payload)


@contextlib.contextmanager
def _decoded_batches(
    reader: VideoReader, batch_size: int, prefetch: bool
) -> Generator[Iterator[tuple[np.ndarray, np.ndarray]]]:
    """Yield ``(indices, frames)`` batches, optionally decoded a batch ahead.

    The prefetch producer is written out here rather than borrowed from
    ``mosaic.core.media.video_io.prefetch_batches`` because it has to be
    *stoppable*: that one runs until the reader is empty, so a cancelled run
    would decode the rest of the video on its way out. Naming it is a docstring
    reference and not an import; this program still takes none from mosaic.
    """
    if not prefetch or batch_size <= 1:
        yield _direct_batches(reader, batch_size)
        return

    pending: queue.Queue[tuple[np.ndarray, np.ndarray] | None] = queue.Queue(maxsize=2)
    stop = threading.Event()

    def offer(item: tuple[np.ndarray, np.ndarray] | None) -> None:
        """Hand *item* to the consumer, giving up only if the run is stopping.

        Retried rather than attempted once. A single bounded ``put`` drops the
        item whenever the consumer has not freed a slot within the timeout, and
        for the ``None`` sentinel below that is a deadlock rather than a lost
        frame: the consumer blocks on a ``get`` that never arrives, and the whole
        process sleeps until something kills it. Whether it happens is decided by
        how long one batch takes against a half-second timer -- so it does not
        happen on a fast GPU and does happen on a busy machine, on CPU, and on a
        large model, which is the worst possible way for it to be distributed.
        """
        while not stop.is_set():
            try:
                pending.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def produce() -> None:
        try:
            while not stop.is_set():
                indices, frames = reader.read_batch(batch_size)
                if len(indices) == 0:
                    break
                offer((indices, frames))
        finally:
            # Abandoned only when `stop` is set, which the context manager below
            # does before it drains the queue -- at which point nothing is reading
            # and the sentinel has nobody to reach.
            offer(None)

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    try:
        yield _queued_batches(pending)
    finally:
        stop.set()
        while True:
            try:
                _ = pending.get_nowait()
            except queue.Empty:
                break
        worker.join(timeout=5)


def _direct_batches(
    reader: VideoReader, batch_size: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    while True:
        indices, frames = reader.read_batch(max(1, batch_size))
        if len(indices) == 0:
            return
        yield indices, frames


def _queued_batches(
    pending: queue.Queue[tuple[np.ndarray, np.ndarray] | None],
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    while True:
        item = pending.get()
        if item is None:
            return
        yield item


def _report(event: ProgressEventKind, done: int = 0, total: int = 0) -> None:
    """One JSON line on standard output, flushed.

    Not cosmetic. Mosaic supervises this process with an inactivity watchdog that
    reads its output stream and kills a process that has produced nothing for the
    idle timeout, and ``verbose=False`` in the tracking arguments is what stops
    Ultralytics from chattering. Without these lines a long video is killed as
    idle.

    ``started`` is written as soon as Ultralytics is imported, before the weights
    load and before the reader opens, and it exists to decouple that timeout from
    startup latency: the torch import and a cold checkpoint load each take far
    longer than a batch, and without a line between them the timeout has to be
    chosen against the sum of the two rather than against the work it is meant to
    supervise. It does not abolish the problem -- the load still runs silently,
    between this line and the first ``progress`` -- so the caller's bound must
    still exceed a cold model load. What it does is split that stretch in two, so
    that a slow *import* can no longer be read as a hung tool.
    """
    line = json.dumps(ProgressEvent(event=event, done=done, total=total).model_dump())
    _ = sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _publish_parquet(table: pd.DataFrame, output_parquet: str) -> None:
    """Write *table* to *output_parquet* atomically: temp file, then rename.

    Mosaic's reuse gate proves a phase complete by finding this file, so a torn
    parquet left by a killed process would be adopted as a finished result. The
    temp name is a leading-dot, ``.tmp``-suffixed hidden file in the destination
    directory, so an orphan left by a hard kill never matches a ``*.parquet``
    filter and the rename is a same-filesystem one.

    A deliberate copy of ``mosaic.core.pipeline._utils.atomic_write``, down to
    the permission fix below, because this program may take no import from
    mosaic -- the duplication is what the separation costs, and the alternative
    is a mosaic import in the Ultralytics process. Mosaic's structural guard
    against a direct ``to_parquet`` on a final path spells its escape as the
    literal ``"(tmp,"``, so the local below has to keep that name.
    """
    final_path = Path(output_parquet)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        dir=final_path.parent, prefix=f".{final_path.stem}-", suffix=".tmp"
    )
    os.close(handle)  # mkstemp returns an open fd; pandas reopens the path
    tmp = Path(name)
    try:
        # mkstemp is 0600 and pandas truncates the file it was given rather than
        # recreating it, so the rename would carry 0600 to the published name --
        # a predictions table only the identity that wrote it can read, where
        # every other parquet mosaic writes is world-readable. A queue worker,
        # an impersonated pod and a sweeper under a service account are all the
        # other identity.
        os.chmod(tmp, 0o666 & ~_UMASK)
        table.to_parquet(tmp, index=False)
        os.replace(tmp, final_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def run_track(request: TrackRequest) -> TrackResponse:
    """Track one video and write its raw predictions.

    Batching decodes and infers several frames per call while still tracking them
    one at a time in order: a list source puts Ultralytics in image mode, where
    one tracker is created and every result in the batch is fed through it in
    sequence. So a batch is a throughput choice, not a behavioral one.

    Track identity is numbered from a counter on a class shared by every backend,
    and this process tracks exactly one video -- so that counter starts at zero by
    construction and there is deliberately nothing here that resets it.

    The table is written **even when it is empty**, with its full column set. The
    reuse gate proves a phase complete by finding the output it recorded, so an
    absent file for a video with no detections would re-run that video forever.
    """
    from ultralytics import YOLO

    # Announced before the weights load, not after it. Loading a cold checkpoint
    # -- off a network mount, with the torch runtime still warming -- is the
    # longest silence a run contains, and mosaic's watchdog bounds *silence*: a
    # line written after the load would put the caller's `idle_timeout` on work
    # proceeding exactly as intended, and kill it with a message about the tool
    # producing no output. Written here it says what is true, which is that
    # Ultralytics is imported and the process is loading.
    _report("started")

    loaded = YOLO(request.model_path)
    task = str(loaded.task)
    if task != request.task:
        raise RunnerError(
            f"{request.model_path} is a {task!r} model but the run declares "
            f"task={request.task!r}. The declared task is part of the run "
            "identifier, so it has to match what the weights actually are."
        )
    model: _Model = loaded
    kwargs = _track_kwargs(request)

    blocks: list[np.ndarray] = []
    n_frames = 0
    reader = VideoReader(
        request.video_path,
        start_frame=request.start_frame,
        end_frame=request.end_frame,
        frame_step=request.frame_step,
        # No decode-time resize: Ultralytics letterboxes to `imgsz` itself and
        # maps predictions back to the frame it was given, so feeding native
        # frames is what puts the coordinates in source pixels.
        resize=None,
        facts=_media_facts(request.media_facts),
    )
    with reader:
        total = len(reader)
        with _decoded_batches(reader, request.batch_size, request.prefetch) as batches:
            for indices, frames in batches:
                results = model.track(
                    [frames[i] for i in range(len(indices))], **kwargs
                )
                for offset, result in enumerate(results):
                    block = rows_from_result(
                        result, int(indices[offset]), n_keypoints=request.n_keypoints
                    )
                    if block is not None:
                        blocks.append(block)
                n_frames += len(indices)
                _report("progress", n_frames, total)

    columns = list(request.columns)
    stacked = (
        np.concatenate(blocks, axis=0)
        if blocks
        else np.empty((0, len(columns)), dtype=np.float64)
    )
    table = pd.DataFrame(stacked, columns=columns)
    table = table.astype({"frame": "int64", "track_id": "int64", "cls": "int64"})
    _publish_parquet(table, request.output_parquet)

    n_ids = int(np.unique(stacked[:, 1]).size) if blocks else 0
    return TrackResponse(n_frames=n_frames, n_ids=n_ids)


# --- inference -------------------------------------------------------------


def _infer_kwargs(request: InferRequestBase) -> dict[str, object]:
    """The arguments every frame of this video is predicted with.

    Four, and deliberately not the full explicit set :func:`_track_kwargs`
    passes. Single-model inference has always left ``iou``, ``max_det``,
    ``agnostic_nms``, ``augment`` and half precision at Ultralytics' shipped
    defaults, and pinning them here would change what these ops produce rather
    than move where they produce it. That leaves the exposure the tracker closed
    -- an upstream retune can still re-mean an identifier already on disk -- which
    is recorded as its own defect rather than fixed in passing.
    """
    return {
        "device": request.device,
        "conf": request.conf,
        "imgsz": request.imgsz,
        # False so the tool does not chatter over the progress lines, which are
        # what mosaic's inactivity watchdog reads.
        "verbose": False,
    }


def _write_annotated(
    result: InferenceResult, directory: Path, frame_index: int
) -> None:
    """Draw one frame the way Ultralytics draws it, and save it.

    Named for the **source** frame index, not the run's position, which is what
    the in-process path wrote and the one place the two numbers differ visibly:
    a windowed or stepped run's images stay addressable against the video they
    came from.

    ``cv2`` is imported here rather than at module scope so a run with no
    annotated output does not pay for it, and so an environment without it still
    answers every other subcommand.
    """
    import cv2

    _ = cv2.imwrite(str(directory / f"frame_{frame_index:08d}.jpg"), result.plot())


@dataclass(slots=True)
class _InferOutcome:
    """What one video's inference produced, before it is given column names."""

    blocks: list[np.ndarray] = field(default_factory=list)
    n_frames: int = 0
    names: dict[int, str] = field(default_factory=dict)


def _infer_frames(
    request: InferRequestBase,
    model: _Predictor[ResultT],
    kwargs: dict[str, object],
    rows: Callable[[ResultT, int], np.ndarray | None],
) -> _InferOutcome:
    """Read *request*'s window in batches, predict each, and keep only the rows.

    The results themselves are dropped as each batch is converted. That is the
    one thing this loop does that the in-process path did not: it accumulated
    every ``Results`` object for a whole video and returned them all, each
    holding live torch tensors, so a long recording's peak memory grew with its
    length. A process boundary cannot carry those objects at all, so converting
    per batch is forced -- and it is also the fix.

    No decode-time resize. Ultralytics letterboxes to ``imgsz`` itself and maps
    predictions back to the frame it was given, so feeding native frames is what
    puts the coordinates in source pixels. The in-process path resized at decode
    time instead and returned coordinates in that smaller space, which is what
    reached ``tracks/`` under a schema whose every spatial column is video
    pixels.
    """
    outcome = _InferOutcome()
    annotated = Path(request.annotated_dir) if request.annotated_dir else None
    if annotated is not None:
        annotated.mkdir(parents=True, exist_ok=True)

    reader = VideoReader(
        request.video_path,
        start_frame=request.start_frame,
        end_frame=request.end_frame,
        frame_step=request.frame_step,
        resize=None,
        facts=_media_facts(request.media_facts),
    )
    with reader:
        total = len(reader)
        if request.max_frames is not None:
            total = min(total, request.max_frames)
        with _decoded_batches(reader, request.batch_size, request.prefetch) as batches:
            for indices, frames in batches:
                if request.max_frames is not None:
                    remaining = request.max_frames - outcome.n_frames
                    if remaining <= 0:
                        break
                    indices = indices[:remaining]
                    frames = frames[:remaining]
                results = model.predict(
                    source=[frames[i] for i in range(len(indices))], **kwargs
                )
                for offset, result in enumerate(results):
                    if not outcome.names:
                        outcome.names = dict(result.names)
                    block = rows(result, outcome.n_frames + offset)
                    if block is not None:
                        outcome.blocks.append(block)
                    if annotated is not None:
                        _write_annotated(result, annotated, int(indices[offset]))
                outcome.n_frames += len(indices)
                _report("progress", outcome.n_frames, total)
                if (
                    request.max_frames is not None
                    and outcome.n_frames >= request.max_frames
                ):
                    break
    return outcome


def _numeric_frame(
    blocks: list[np.ndarray], columns: Sequence[str]
) -> tuple[pd.DataFrame, int]:
    """*blocks* stacked and labelled, plus how many rows that is.

    Built even when there is nothing in it, with its full column set, so an empty
    result is a legible empty table rather than an absent file -- the same reason
    ``track`` writes one.
    """
    stacked = (
        np.concatenate(blocks, axis=0)
        if blocks
        else np.empty((0, len(columns)), dtype=np.float64)
    )
    if stacked.shape[1] != len(columns):
        raise RunnerError(
            f"the predictions carry {stacked.shape[1]} columns and mosaic named "
            f"{len(columns)}. The names decide what every column means, so a "
            "mismatch cannot be written."
        )
    return pd.DataFrame(stacked, columns=list(columns)), int(stacked.shape[0])


def _loaded_for(request: InferRequestBase) -> object:
    """The weights this run declares, loaded, with the declared task checked.

    Announced before the load for the reason :func:`run_track` gives: a cold
    checkpoint is the longest silence a healthy run contains, and mosaic's
    watchdog bounds silence.
    """
    from ultralytics import YOLO

    _report("started")
    loaded = YOLO(request.model_path)
    task = str(getattr(loaded, "task", ""))
    if task != request.task:
        raise RunnerError(
            f"{request.model_path} is a {task!r} model but the run declares "
            f"task={request.task!r}. The declared task is part of the run "
            "identifier, so it has to match what the weights actually are."
        )
    return loaded


def run_infer_pose(request: InferPoseRequest) -> InferResponse:
    """Run one video through a YOLO pose model and write its raw predictions."""
    model: _Predictor[PoseResult] = _loaded_for(request)
    outcome = _infer_frames(
        request,
        model,
        _infer_kwargs(request),
        lambda result, position: pose_rows_from_result(
            result, position, n_keypoints=request.n_keypoints
        ),
    )
    table, n_rows = _numeric_frame(outcome.blocks, request.columns)
    table = table.astype({"frame": "int64", "id": "int64"})
    _publish_parquet(table, request.output_parquet)
    return InferResponse(n_frames=outcome.n_frames, n_rows=n_rows)


def run_infer_points(request: InferPointsRequest) -> InferResponse:
    """Run one video through a POLO point model and write its raw predictions.

    ``class_name`` is mapped from ``class_id`` after the numeric block is
    assembled, because it is the one column of either table that is not a number
    and so cannot travel in one.
    """
    model: _Predictor[PointResult] = _loaded_for(request)
    kwargs = _infer_kwargs(request)
    if request.radii is not None:
        kwargs["radii"] = dict(request.radii)
    outcome = _infer_frames(request, model, kwargs, point_rows_from_result)

    numeric = list(POINT_COLUMNS[:-1])
    table, n_rows = _numeric_frame(outcome.blocks, numeric)
    table = table.astype(
        {"frame": "int64", "detection_id": "int64", "class_id": "int64"}
    )
    # ``dtype=object`` explicitly: an empty run would otherwise get a float64
    # ``class_name`` from an empty list, and the table is now written even when it
    # holds no rows -- so a reader concatenating an empty run with a full one
    # would meet two dtypes for one column.
    table["class_name"] = pd.Series(
        [
            outcome.names.get(class_id, f"class_{class_id}")
            for class_id in table["class_id"]
        ],
        dtype=object,
        index=table.index,
    )
    _publish_parquet(table[list(request.columns)], request.output_parquet)
    return InferResponse(n_frames=outcome.n_frames, n_rows=n_rows)


# --- entry point -----------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ultralytics_runner",
        description="Probe the Ultralytics environment, or run one video through it.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("probe", "report what this environment holds"),
        ("tracker-defaults", "report every backend's shipped configuration table"),
        ("track", "track one video and write its raw predictions"),
        ("infer-pose", "run a pose model over one video"),
        ("infer-points", "run a point model over one video"),
    ):
        subcommand = subcommands.add_parser(name, help=help_text)
        _ = subcommand.add_argument(
            "--request", required=True, help="path to the JSON request"
        )
        _ = subcommand.add_argument(
            "--out", required=True, help="path the JSON response is written to"
        )
    return parser


def _answer(command: str, payload: str) -> BaseModel:
    """Run the named subcommand against its own request model."""
    if command == "probe":
        return run_probe(ProbeRequest.model_validate_json(payload))
    if command == "tracker-defaults":
        return run_tracker_defaults(TrackerDefaultsRequest.model_validate_json(payload))
    if command == "infer-pose":
        return run_infer_pose(InferPoseRequest.model_validate_json(payload))
    if command == "infer-points":
        return run_infer_points(InferPointsRequest.model_validate_json(payload))
    return run_track(TrackRequest.model_validate_json(payload))


def main(argv: Sequence[str] | None = None) -> int:
    namespace = _parser().parse_args(argv)
    command: str = namespace.command
    request_path = Path(str(namespace.request))
    output_path = Path(str(namespace.out))

    response = _answer(command, request_path.read_text())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = output_path.write_text(response.model_dump_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
