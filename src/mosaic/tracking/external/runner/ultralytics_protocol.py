"""The wire contract between mosaic and the Ultralytics tracking runner.

Imported from **both** sides of the boundary: mosaic's own environment builds
the requests and reads the responses, and the runner program inside the
Ultralytics environment does the reverse. Its dependencies are therefore the
intersection of the two -- the standard library, numpy and pydantic, and nothing
else.

It imports neither ``ultralytics`` nor ``mosaic``, and must not grow either
import. An ``ultralytics`` import here would pull Ultralytics into mosaic's
process; a ``mosaic`` import here would put mosaic's code into the Ultralytics
process. Either one recreates the single combined work the separation exists to
prevent, which is the whole reason Ultralytics runs in an environment of its
own.
"""

from __future__ import annotations

from typing import Final, Literal, Protocol, TypeAlias

import numpy as np
from pydantic import BaseModel

# Mirror of ``mosaic.tracking.ultralytics_track.tracker_defaults.TrackerSetting``.
# Duplicated deliberately rather than imported: this module may take no import
# from mosaic, so the two cannot be one declaration.
TrackerSetting: TypeAlias = bool | int | float | str

ModelTask: TypeAlias = Literal["pose", "detect"]
"""What a run declares its weights to be. Closed here rather than left a bare
``str`` because a wire contract is where a closed set is worth closing: a
mistyped task would pass validation and then be compared, unequal, against what
the weights actually declare.
"""

Precision: TypeAlias = Literal["fp32", "fp16"]
"""Closed for the sharpest reason of the four. The runner computes
``{"half": precision == "fp16"}`` on an older Ultralytics, so a mangled value
silently means fp32 there and travels as ``{"quantize": "fp61"}`` into a newer
one -- a run that read as full precision under an identifier minted for half.
"""

ProgressEventKind: TypeAlias = Literal["started", "progress"]
"""The two lines the runner writes to standard output."""


_TRACK_COLUMNS: Final = 7
"""Columns a tracked ``Boxes`` carries: x1, y1, x2, y2, track id, confidence, class."""


class UltralyticsInteropError(RuntimeError):
    """Ultralytics behaved in a way this integration's correctness depends on not happening."""


# --- protocols standing in for the Ultralytics surface ---------------------


class Detections(Protocol):
    """The ``Boxes`` / ``Keypoints`` surface, after ``.cpu().numpy()``."""

    @property
    def data(self) -> np.ndarray: ...


class Boxes(Detections, Protocol):
    @property
    def id(self) -> np.ndarray | None: ...

    def cpu(self) -> Boxes: ...
    def numpy(self) -> Boxes: ...


class Keypoints(Detections, Protocol):
    def cpu(self) -> Keypoints: ...
    def numpy(self) -> Keypoints: ...


class Result(Protocol):
    @property
    def boxes(self) -> Boxes | None: ...
    @property
    def keypoints(self) -> Keypoints | None: ...


def rows_from_result(
    result: Result, frame_index: int, *, n_keypoints: int
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


# --- probe -----------------------------------------------------------------


class ProbeRequest(BaseModel):
    """What mosaic wants to know about the Ultralytics environment."""

    model_path: str
    """The weights the run will load, so the probe can report their task."""

    tracker: str
    """Which backend's shipped configuration file to read back."""


class ProbeResponse(BaseModel):
    """What the environment holds. Findings only -- every refusal is mosaic's.

    The runner reports and never refuses, for two reasons. The refusal messages
    name mosaic commands and mosaic's own installation documentation, which the
    runner cannot know; and a refusal decided from reported data is testable with
    no Ultralytics installed at all.
    """

    has_ultralytics: bool
    """Whether ``import ultralytics`` succeeded."""

    has_lap: bool
    """Whether ``lap`` is importable.

    Probed separately from Ultralytics because it is the linear-assignment solver
    every tracker backend uses for the detection-to-track association step, and
    it appears in no Ultralytics extra. Missing, Ultralytics pip-installs it
    *during* the run -- a network write inside a queued job, and a hard failure in
    a locked environment.
    """

    ultralytics_version: str
    """``ultralytics.__version__``; empty when Ultralytics is absent."""

    tracker_names: list[str]
    """The backends the installed ``TRACKER_MAP`` knows; empty when absent."""

    model_task: str
    """The task the weights declare (``pose``, ``detect``, ``segment``, ...);
    empty when Ultralytics is absent.

    A bare ``str`` where the other closed sets in this module are ``Literal``
    aliases, and deliberately so: it reports whatever Ultralytics declares,
    including the tasks mosaic refuses, and closing it to the two mosaic bridges
    would turn the report this field exists to make into a validation error.
    """

    n_keypoints: int
    """How many keypoints the model predicts, 1 for a box-only detection model;
    0 when Ultralytics is absent."""

    installed_tracker_table: dict[str, TrackerSetting]
    """The requested backend's shipped defaults; empty when absent.

    Mosaic merges its own resolved settings over these, so a knob mosaic has not
    transcribed stays at its upstream default instead of reaching a backend that
    raises on a missing attribute.
    """


# --- tracker defaults ------------------------------------------------------


class TrackerDefaultsRequest(BaseModel):
    """Ask the environment for every backend's shipped configuration table.

    Carries no fields at all, and no model path in particular: reading the
    configuration files loads no weights. It exists so that all three
    subcommands take the same ``--request`` / ``--out`` shape, which is what
    keeps one launcher able to drive any of them.

    The answer is not narrowed to the backends mosaic knows either. A setting
    the installed release added and mosaic has not transcribed is precisely what
    the comparison is for, and a request naming the backends would hide a whole
    backend the same way.
    """


class TrackerDefaultsResponse(BaseModel):
    """Every backend's shipped defaults, read in one process.

    One response rather than one per backend because each spawn pays a cold
    torch import, which is the whole cost of the answer.
    """

    tables: dict[str, dict[str, TrackerSetting]]
    """Each backend the installed ``TRACKER_MAP`` knows, mapped to its shipped
    configuration file read as scalars -- the same reading
    :class:`ProbeResponse.installed_tracker_table` reports for one backend.

    A backend whose configuration file is absent maps to an empty table rather
    than being left out, so the key set stays the set of backends that exist and
    a missing file reads as a missing file.
    """


# --- track -----------------------------------------------------------------


class TrackRequest(BaseModel):
    """One video, tracked once, in a process of its own."""

    model_path: str
    video_path: str
    output_parquet: str
    """Where the raw predictions table is published, atomically."""

    tracker_yaml: str
    """The merged settings file mosaic wrote for this run."""

    project_dir: str
    """Where Ultralytics computes its own run directory.

    Named for the ``project`` argument it is passed rather than for a directory
    of mosaic's, because mosaic writes nothing here: Ultralytics computes that
    path eagerly even with saving off, and an unpinned one walks the shared
    ``runs/`` tree. Not the entry working directory -- that is what ``work_dir``
    means on the mosaic side, and one name for the two would be a trap.
    """

    columns: list[str]
    """The raw-parquet column names, in write order.

    Supplied by mosaic rather than derived here, because mosaic owns the
    raw-parquet column contract (``raw_columns`` in
    ``mosaic.core.track_library.ultralytics_tracks``) -- the converter that reads
    this parquet is mosaic's. The runner is told the columns; it never computes
    them. That is what stops the column list existing in two places that can
    disagree.
    """

    n_keypoints: int
    """How many keypoints each row carries. Supplied by mosaic for the same
    reason as ``columns``: it is a term of the column contract."""

    task: ModelTask
    """The task the run was minted under, checked against the loaded weights."""

    conf: float
    iou: float
    imgsz: int
    max_det: int
    classes: list[int] | None
    agnostic_nms: bool
    device: str
    precision: Precision
    """Spelled for the installed Ultralytics by the runner."""

    start_frame: int
    end_frame: int | None
    frame_step: int
    batch_size: int
    prefetch: bool
    media_facts: dict[str, object]
    """A ``mosaic_media.MediaFacts`` flattened with ``dataclasses.asdict``.

    **Required, and not nullable.** Mosaic owns the read-target gate: it probes
    the file, derives its verdict and raises when the measured verdict says the
    file needs transcoding before it can be read for analysis. The runner cannot
    call that gate -- it lives in mosaic -- so an omitted payload would leave the
    reader probing with no gate at all, and a rotated or variable-frame-rate
    original would track silently to misindexed coordinates under a perfectly
    valid identifier. Making the field required is what stops the boundary being
    crossed ungated.

    Passing measured facts rather than letting the runner probe also keeps a raw
    ``.h264`` reading with its true frame count instead of the garbage count its
    header declares. The runner rebuilds the dataclass by validating this
    payload, so a field that arrived under the wrong type is refused here rather
    than read as data.
    """


class TrackResponse(BaseModel):
    """What the tracked video produced."""

    n_frames: int
    """Frames read and tracked."""

    n_ids: int
    """Distinct track identities in the written table."""


class ProgressEvent(BaseModel):
    """One line of the runner's standard output.

    Two kinds. ``started`` is written once, as soon as the weights are loaded,
    and ``progress`` after every decoded batch. Splitting them is what lets a
    supervising inactivity timeout be chosen against one batch rather than
    against a cold torch import and model load, which is far longer and has
    nothing to do with whether the run is stuck.
    """

    event: ProgressEventKind

    done: int = 0
    """Frames tracked so far. Zero on ``started``."""

    total: int = 0
    """Frames the run will read. Zero on ``started``: the reader is not open
    yet, and zero means *not reported* rather than *no frames*.
    """


__all__ = [
    "Boxes",
    "Detections",
    "Keypoints",
    "ModelTask",
    "Precision",
    "ProbeRequest",
    "ProbeResponse",
    "ProgressEvent",
    "ProgressEventKind",
    "Result",
    "TrackRequest",
    "TrackResponse",
    "TrackerDefaultsRequest",
    "TrackerDefaultsResponse",
    "TrackerSetting",
    "UltralyticsInteropError",
    "rows_from_result",
]
