"""Driving Ultralytics multi-object tracking over one video, in process.

The only module here that imports Ultralytics, and it does so inside functions --
so registering the op, describing its parameters and reconciling its index all
work in an environment that has no Ultralytics at all.

Three module-level functions are the seam the tests replace:
:func:`load_tracking_model`, :func:`reset_trackers` and
:func:`run_ultralytics_track`. Patching them on ``dataset_runs`` exercises the
whole run protocol -- identifiers, markers, reuse, the bridge -- with no weights,
no torch and no GPU, exactly as the three subprocess trackers patch their
``run_*`` wrappers.

**Mosaic decodes; Ultralytics never sees a path.** Handing ``model.track`` a
video would put OpenCV on the read path, which cannot open an imgstore directory
at all and reads a raw ``.h264`` with a garbage frame count. Reading through
``open_frame_reader`` instead keeps imgstore, raw streams, the true frame index
and mosaic's own frame window working, and costs nothing: Ultralytics tracks
frame by frame regardless.
"""

from __future__ import annotations

import contextlib
import importlib
import queue
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol

import numpy as np
import pandas as pd

from mosaic.core.media.video_io import open_frame_reader
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.track_library.ultralytics_tracks import raw_columns
from mosaic.tracking.ultralytics_track.tracker_defaults import (
    TrackerName,
    TrackerSetting,
)

if TYPE_CHECKING:
    from mosaic_media import MediaFacts

    from mosaic.core.media.video_io import FrameReader

ModelTask = Literal["pose", "detect"]
"""What mosaic bridges. A segment mask and a rotated box have no ``trex_v1``
mapping, POLO's point detection has its own op, and Ultralytics refuses to track
a classifier.
"""

_SUPPORTED_TASKS: Final[tuple[ModelTask, ...]] = ("pose", "detect")

_TRACK_COLUMNS: Final = 7
"""Columns a tracked ``Boxes`` carries: x1, y1, x2, y2, track id, confidence, class."""


class UltralyticsNotFoundError(ImportError):
    """Ultralytics, or a dependency of its tracker, is not importable.

    Deliberately not a ``ToolNotFoundError``: there is no ``ToolEnv``, no
    ``MOSAIC_ULTRALYTICS_*`` and no location ladder, and borrowing that
    vocabulary would promise a placement mechanism this tracker does not have.
    """


class UnsupportedTaskError(ValueError):
    """The model is not one mosaic can bridge into a ``trex_v1`` table."""


class UnsupportedTrackerError(ValueError):
    """The installed Ultralytics does not know the requested backend."""


class UltralyticsInteropError(RuntimeError):
    """Ultralytics behaved in a way this integration's correctness depends on not happening."""


@dataclass(frozen=True, slots=True)
class UltralyticsTrackResult:
    """What one entry's tracking produced."""

    predictions_path: Path
    n_frames: int
    n_ids: int


# --- protocols standing in for the Ultralytics surface ---------------------


class _Detections(Protocol):
    """The ``Boxes`` / ``Keypoints`` surface, after ``.cpu().numpy()``."""

    @property
    def data(self) -> np.ndarray: ...


class _Boxes(_Detections, Protocol):
    @property
    def id(self) -> np.ndarray | None: ...

    def cpu(self) -> _Boxes: ...
    def numpy(self) -> _Boxes: ...


class _Keypoints(_Detections, Protocol):
    def cpu(self) -> _Keypoints: ...
    def numpy(self) -> _Keypoints: ...


class _Result(Protocol):
    @property
    def boxes(self) -> _Boxes | None: ...
    @property
    def keypoints(self) -> _Keypoints | None: ...


class _Model(Protocol):
    def track(self, source: list[np.ndarray], **kwargs: object) -> list[_Result]: ...


# --- preflight -------------------------------------------------------------


def _installed_version() -> str:
    import ultralytics

    return str(getattr(ultralytics, "__version__", "unknown"))


def require_ultralytics(tracker: TrackerName) -> None:
    """Refuse, by name, anything this run needs and does not have.

    Runs before a model is loaded or a video is opened, so a missing dependency
    or an Ultralytics too old for the requested backend is a message rather than
    a traceback from inside a callback on frame zero.
    """
    try:
        _ = importlib.import_module("ultralytics")
    except ImportError as exc:
        raise UltralyticsNotFoundError(
            "ultralytics is required for the 'ultralytics' tracker. Install it "
            'with: pip install "mosaic-behavior[pose]".'
        ) from exc

    try:
        _ = importlib.import_module("lap")
    except ImportError as exc:
        raise UltralyticsNotFoundError(
            "lap is required for Ultralytics multi-object tracking -- it is the "
            "linear-assignment solver every backend associates with. It appears "
            "in no ultralytics extra, so without it ultralytics tries to "
            "pip-install it mid-run. Install it with: pip install "
            '"mosaic-behavior[pose]".'
        ) from exc

    from ultralytics.trackers.track import TRACKER_MAP

    if tracker not in TRACKER_MAP:
        raise UnsupportedTrackerError(
            f"the installed ultralytics ({_installed_version()}) knows "
            f"{sorted(TRACKER_MAP)}, not {tracker!r}. The four newer backends "
            "arrived in 8.4.63; mosaic declares that floor on its [pose] extra."
        )


def require_supported_task(task: str) -> ModelTask:
    """Narrow a model's task to one mosaic bridges, or refuse it by name."""
    if task in _SUPPORTED_TASKS:
        return "pose" if task == "pose" else "detect"
    hint = {
        "locate": "POLO point detection has its own op: mosaic run --kind infer-points.",
        "segment": "mosaic bridges boxes and keypoints; a mask has no trex_v1 mapping.",
        "obb": "mosaic bridges axis-aligned boxes; a rotated box has no trex_v1 mapping.",
        "classify": "a classifier localizes nothing, so there is nothing to track.",
    }.get(task, "")
    raise UnsupportedTaskError(
        f"the 'ultralytics' tracker runs {' and '.join(_SUPPORTED_TASKS)} models, "
        f"not {task!r}. {hint}".strip()
    )


def precision_kwarg(precision: Literal["fp32", "fp16"]) -> dict[str, object]:
    """How the installed Ultralytics spells half precision.

    ``half`` was replaced by ``quantize`` and now emits a deprecation warning, but
    the older spelling is what an older install understands. Probed from the
    shipped default configuration rather than from a version comparison, so this
    keeps working across the transition in both directions.
    """
    from ultralytics.cfg import DEFAULT_CFG_DICT

    if "quantize" in DEFAULT_CFG_DICT:
        return {"quantize": precision}
    return {"half": precision == "fp16"}


# --- the tracker configuration file ----------------------------------------


def effective_tracker_table(
    tracker: TrackerName, resolved: Mapping[str, TrackerSetting]
) -> dict[str, TrackerSetting]:
    """The installed backend's own defaults, with mosaic's resolved values on top.

    Mosaic's table is what identity is taken over, but it must not be what gets
    *written*: each backend reads its settings off an object that raises on a
    missing attribute, so writing mosaic's table alone would fail inside
    Ultralytics -- after the model loaded and the video opened -- the first time
    an upstream release added a required setting. Merging leaves a setting mosaic
    has not transcribed at its upstream default, and the preflight drift test is
    what turns that into a decision rather than a surprise.
    """
    import yaml

    import ultralytics

    path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{tracker}.yaml"
    installed: dict[str, TrackerSetting] = {}
    if path.is_file():
        loaded = yaml.safe_load(path.read_text())
        if isinstance(loaded, dict):
            for key, value in loaded.items():  # pyright: ignore[reportUnknownVariableType]
                if isinstance(value, (bool, int, float, str)):
                    installed[str(key)] = value
    return {**installed, **resolved}


def write_tracker_yaml(path: Path, table: Mapping[str, TrackerSetting]) -> Path:
    """Write *table* where Ultralytics will read it, atomically.

    Four details are load-bearing. The suffix is ``.yaml`` and the path absolute,
    because Ultralytics' config check returns an existing path untouched and
    otherwise goes looking for -- or downloading -- a name. Keys are sorted, so
    the file is a function of its contents. And the write is atomic, because the
    check gates on existence alone: a truncated file would be *loaded*, its
    missing settings quietly filled from each backend's own fallbacks, and a
    differently-configured tracker would run under the right identifier.
    """
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)

    def write(temp: Path) -> None:
        _ = temp.write_text(yaml.safe_dump(dict(table), sort_keys=True))

    atomic_write(path, write)
    return path


# --- the model, held for the whole run -------------------------------------


@dataclass(frozen=True, slots=True)
class TrackSession:
    """A loaded model plus the arguments every frame of the run is tracked with.

    One session per run rather than per entry, so weights load once however many
    sequences the run covers. The keyword arguments are frozen here for a reason
    beyond tidiness: ``device`` must be byte-identical on every call, because
    Ultralytics rebuilds its predictor when the device argument changes and would
    take the live trackers -- pools, Kalman state, identity numbering -- with it,
    mid-video and silently.
    """

    model: _Model
    task: ModelTask
    n_keypoints: int
    kwargs: Mapping[str, object]

    def track(self, frames: list[np.ndarray]) -> list[_Result]:
        return self.model.track(frames, **self.kwargs)


def load_tracking_model(
    model_path: Path,
    *,
    tracker_yaml: Path,
    task: ModelTask,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
    classes: Sequence[int] | None,
    agnostic_nms: bool,
    device: str,
    precision: Literal["fp32", "fp16"],
    work_dir: Path,
) -> TrackSession:
    """Load *model_path* and freeze the arguments its whole run will use.

    Every detection-affecting argument is passed explicitly rather than left to
    Ultralytics' shipped defaults, so an upstream retune cannot silently re-mean
    an identifier mosaic already minted.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    resolved_task = require_supported_task(str(model.task))
    if resolved_task != task:
        raise UnsupportedTaskError(
            f"{model_path} is a {resolved_task!r} model but the run declares "
            f"task={task!r}. The declared task is part of the run identifier, so "
            "it has to match what the weights actually are."
        )

    n_keypoints = _keypoint_count(model, resolved_task)
    kwargs: dict[str, object] = {
        "persist": True,
        "tracker": str(tracker_yaml),
        "stream": False,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "max_det": max_det,
        "classes": list(classes) if classes is not None else None,
        "agnostic_nms": agnostic_nms,
        "augment": False,
        "device": device,
        "verbose": False,
        "save": False,
        "save_txt": False,
        "show": False,
        # Pin the run directory even though nothing is saved: Ultralytics computes
        # it eagerly, and an unpinned one walks the shared `runs/` tree.
        "project": str(work_dir),
        "name": "ultralytics",
        "exist_ok": True,
        **precision_kwarg(precision),
    }
    return TrackSession(
        model=model, task=resolved_task, n_keypoints=n_keypoints, kwargs=kwargs
    )


def _keypoint_count(model: object, task: ModelTask) -> int:
    """How many keypoints this model predicts; 1 for a box-only model.

    Read off the detection **head**, not off the model. ``PoseModel`` never
    assigns ``kpt_shape`` to itself -- only the head does -- so the obvious read
    yields nothing for a checkpoint that did not come through Ultralytics'
    trainer, and a one-keypoint table would be written for a pose model.
    """
    if task == "detect":
        return 1
    inner = getattr(model, "model", None)
    layers = getattr(inner, "model", None)
    head = layers[-1] if layers is not None else None
    shape = getattr(head, "kpt_shape", None) or getattr(inner, "kpt_shape", None)
    if shape is None:
        raise UnsupportedTaskError(
            "this pose model does not declare a keypoint shape, so mosaic cannot "
            "tell how many keypoints its predictions carry."
        )
    return int(shape[0])


def reset_trackers(session: TrackSession) -> None:
    """Return every attached tracker to its frame-zero state, between entries.

    Necessary, not defensive. Track identity is numbered from a counter on a
    class shared by all backends, so without this a run's second video continues
    the first video's numbering and the output depends on what ran before it in
    the process. ``reset()`` clears that counter along with each tracker's pools,
    its Kalman filter and -- where there is one -- its camera-motion history,
    which optical flow would otherwise carry across a cut between two videos.

    The reset is a method call rather than ``persist=False`` on the first frame:
    Ultralytics binds the persist flag into its callbacks at the first tracking
    call and ignores it thereafter, so alternating would rebuild the trackers on
    every frame instead of once per entry.
    """
    from ultralytics.trackers.basetrack import BaseTrack

    predictor = getattr(session.model, "predictor", None)
    trackers = getattr(predictor, "trackers", None)
    for tracker in trackers or ():
        tracker.reset()
    # Covers the first entry, where no tracker exists yet to reset the shared
    # counter a long-lived worker process may already have advanced.
    BaseTrack.reset_id()


# --- one entry -------------------------------------------------------------


def rows_from_result(
    result: _Result, frame_index: int, *, n_keypoints: int
) -> np.ndarray | None:
    """One frame's tracked detections as a ``(n, 8 + 3K)`` block, or None.

    ``boxes.id is None`` is not an empty frame in disguise. Ultralytics' tracking
    callback returns early when the tracker produced no tracks, leaving the
    result holding its *raw, untracked* detections -- so treating that as data
    would put identity-less rows into a table whose whole subject is identity.
    """
    boxes = result.boxes
    if boxes is None:
        return None
    boxes = boxes.cpu().numpy()
    if boxes.id is None:
        return None
    data = boxes.data
    if data.shape[0] == 0:
        return None
    if data.shape[1] < _TRACK_COLUMNS:
        raise UltralyticsInteropError(
            f"a tracked detection should carry {_TRACK_COLUMNS} columns "
            f"(box, track id, confidence, class); got {data.shape[1]}."
        )

    n = data.shape[0]
    xyxy = data[:, 0:4].astype(np.float64)
    track_id = data[:, 4].astype(np.float64)
    conf = data[:, 5].astype(np.float64)
    cls = data[:, 6].astype(np.float64)

    if result.keypoints is not None:
        keypoints = result.keypoints.cpu().numpy().data
        if keypoints.shape[0] != n:
            raise UltralyticsInteropError(
                f"frame {frame_index}: {keypoints.shape[0]} keypoint sets for "
                f"{n} tracked boxes. Keypoints are reindexed with the boxes, so "
                "a mismatch means the two are no longer the same detections."
            )
        kp = np.empty((n, n_keypoints, 3), dtype=np.float64)
        kp[:, :, 0:2] = keypoints[:, :n_keypoints, 0:2]
        kp[:, :, 2] = keypoints[:, :n_keypoints, 2] if keypoints.shape[2] > 2 else 1.0
    else:
        # A box-only model localizes nothing finer, so the box centre is the
        # honest single keypoint and the detection confidence is its score.
        kp = np.empty((n, 1, 3), dtype=np.float64)
        kp[:, 0, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
        kp[:, 0, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
        kp[:, 0, 2] = conf

    block = np.empty((n, 8 + 3 * kp.shape[1]), dtype=np.float64)
    block[:, 0] = float(frame_index)
    block[:, 1] = track_id
    block[:, 2:6] = xyxy
    block[:, 6] = conf
    block[:, 7] = cls
    block[:, 8:] = kp.reshape(n, -1)
    # List order is not a contract -- two backends override how they format their
    # output -- so sort by the tracker's own numbering to make the file a
    # function of the frame rather than of the backend's bookkeeping.
    return block[np.argsort(block[:, 1], kind="stable")]


@contextlib.contextmanager
def _decoded_batches(
    reader: FrameReader, batch_size: int, prefetch: bool
) -> Iterator[Iterator[tuple[np.ndarray, np.ndarray]]]:
    """Yield ``(indices, frames)`` batches, optionally decoded a batch ahead.

    The prefetch producer is local rather than
    :func:`~mosaic.core.media.video_io.prefetch_batches` because it has to be
    *stoppable*: that one runs until the reader is empty, so a cancelled run
    would decode the rest of the video on its way out.
    """
    if not prefetch or batch_size <= 1:
        yield _direct_batches(reader, batch_size)
        return

    pending: queue.Queue[tuple[np.ndarray, np.ndarray] | None] = queue.Queue(maxsize=2)
    stop = threading.Event()

    def produce() -> None:
        try:
            while not stop.is_set():
                indices, frames = reader.read_batch(batch_size)
                if len(indices) == 0:
                    break
                while not stop.is_set():
                    try:
                        pending.put((indices, frames), timeout=0.5)
                        break
                    except queue.Full:
                        continue
        finally:
            with contextlib.suppress(queue.Full):
                pending.put(None, timeout=0.5)

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
    reader: FrameReader, batch_size: int
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


def run_ultralytics_track(
    session: TrackSession,
    video_path: Path,
    out_parquet: Path,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    frame_step: int = 1,
    batch_size: int = 8,
    prefetch: bool = True,
    facts: MediaFacts | None = None,
    on_tick: object = None,
) -> UltralyticsTrackResult:
    """Track one video and write its raw predictions.

    Batching decodes and infers several frames per call while still tracking
    them one at a time in order: a list source puts Ultralytics in image mode,
    where one tracker is created and every result in the batch is fed through it
    in sequence. So a batch is a throughput choice, not a behavioral one.

    The table is written **even when it is empty**, with its full column set. The
    reuse gate proves a phase complete by finding the output it recorded, so an
    absent file for a video with no detections would re-run that video forever.
    """
    n_keypoints = session.n_keypoints
    blocks: list[np.ndarray] = []
    n_frames = 0

    reader = open_frame_reader(
        video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_step=frame_step,
        # No decode-time resize: Ultralytics letterboxes to `imgsz` itself and
        # maps predictions back to the frame it was given, so feeding native
        # frames is what puts the coordinates in source pixels.
        resize=None,
        facts=facts,
        target="analysis",
    )
    with reader:
        total = len(reader)
        with _decoded_batches(reader, batch_size, prefetch) as batches:
            for indices, frames in batches:
                results = session.track([frames[i] for i in range(len(indices))])
                for offset, result in enumerate(results):
                    block = rows_from_result(
                        result, int(indices[offset]), n_keypoints=n_keypoints
                    )
                    if block is not None:
                        blocks.append(block)
                n_frames += len(indices)
                if callable(on_tick):
                    on_tick(n_frames, total)

    columns = list(raw_columns(n_keypoints))
    stacked = (
        np.concatenate(blocks, axis=0)
        if blocks
        else np.empty((0, len(columns)), dtype=np.float64)
    )
    table = pd.DataFrame(stacked, columns=columns)
    table = table.astype({"frame": "int64", "track_id": "int64", "cls": "int64"})

    def write(temp: Path) -> None:
        table.to_parquet(temp, index=False)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(out_parquet, write)

    n_ids = int(np.unique(stacked[:, 1]).size) if blocks else 0
    return UltralyticsTrackResult(
        predictions_path=out_parquet, n_frames=n_frames, n_ids=n_ids
    )


__all__ = [
    "ModelTask",
    "TrackSession",
    "UltralyticsInteropError",
    "UltralyticsNotFoundError",
    "UltralyticsTrackResult",
    "UnsupportedTaskError",
    "UnsupportedTrackerError",
    "effective_tracker_table",
    "load_tracking_model",
    "precision_kwarg",
    "require_supported_task",
    "require_ultralytics",
    "reset_trackers",
    "rows_from_result",
    "run_ultralytics_track",
    "write_tracker_yaml",
]
