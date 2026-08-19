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
from pydantic import BaseModel, JsonValue

# Mirror of ``mosaic.tracking.ultralytics_track.tracker_defaults.TrackerSetting``.
# Duplicated deliberately rather than imported: this module may take no import
# from mosaic, so the two cannot be one declaration.
TrackerSetting: TypeAlias = bool | int | float | str

ModelTask: TypeAlias = Literal["pose", "detect"]
"""What a run declares its weights to be. Closed here rather than left a bare
``str`` because a wire contract is where a closed set is worth closing: a
mistyped task would pass validation and then be compared, unequal, against what
the weights actually declare.

Declared once, here, and **imported** by mosaic's side rather than mirrored the
way :data:`TrackerSetting` above is. The rule this module lives under is
one-directional -- it may take no import from mosaic, while mosaic may import it
freely -- so a name mosaic can adopt as-is should be one declaration.
``TrackerSetting`` cannot be: its mosaic-side home is
``ultralytics_track.tracker_defaults``, which is a pure mosaic module and would
have to reach into the runner directory to share this one.
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
    """The weights the run will load, so the probe can report their task.

    **Empty asks about the environment alone**, the reading :attr:`tracker` below
    already has. Training sends empty whenever it is not fine-tuning: its model is
    a bare asset name Ultralytics fetches from a release, so loading it here would
    put a network download inside a preflight whose whole value is being cheap,
    and would report a load error for a question nobody asked. When there *is* a
    base checkpoint mosaic sends its resolved path, and the load answers a real
    question -- whether these weights open in this environment.
    """

    tracker: str = ""
    """Which backend's shipped configuration file to read back, or empty for none.

    Empty is what single-model inference sends: it runs no tracker, so there is no
    table for it to merge its settings over. The runner reads a configuration file
    named for this value and reports an empty table when there is none, so an
    empty name needs no branch of its own.
    """


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

    has_locate: bool
    """Whether this environment's ``ultralytics`` is the POLO fork.

    Probed by importing ``ultralytics.nn.tasks.LocalizationModel``, which is the
    only thing that tells the fork from upstream at run time: the two ship under
    one distribution name, at overlapping versions, with the same console
    scripts. So neither the version nor the resolved path answers the question,
    and the ``$PATH`` rung of the location ladder cannot know which it found.

    Point detection needs it and refuses without it. Tracking does not, and reads
    it only to say which environment a fork-only checkpoint belongs to.
    """

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

    model_load_error: str
    """Why the weights would not load, or empty when they loaded.

    **Reported rather than raised**, which is the whole reason this probe can
    diagnose a checkpoint from the wrong fork. POLO pickles its weights under a
    class upstream does not define, so an upstream build meets
    ``AttributeError: Can't get attribute 'LocalizationModel'`` inside
    ``torch.load`` -- *before* the task the checkpoint declares can be read. A
    propagating exception therefore reaches the user as a raw traceback naming a
    class they have never heard of, and the refusal mosaic already wrote for
    ``locate`` weights is unreachable, because it runs on a task string that only
    exists once the load succeeded.

    Non-empty leaves ``model_task`` and ``n_keypoints`` at their empty values:
    nothing was loaded, so there is nothing to report about it. Mosaic pairs this
    with :attr:`has_locate` to name the environment the weights want.
    """

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


# --- inference -------------------------------------------------------------


InferTask: TypeAlias = Literal["pose", "locate"]
"""What a single-model inference run declares its weights to be.

Disjoint from :data:`ModelTask`, which is what the *tracker* bridges. The two
overlap on ``pose`` and agree about nothing else: a tracker runs a box model and
refuses ``locate``, while point inference runs ``locate`` alone and a detection
model has no point output to read.
"""

_POINT_TRACKED_COLUMNS: Final = 5
"""Columns a POLO location carries when it also carries a track id."""


def pose_columns(n_keypoints: int) -> list[str]:
    """The pose predictions parquet's column names, in write order.

    One declaration, imported by both sides, rather than a list mosaic builds and
    the runner rebuilds: the runner writes this file and mosaic reads it back to
    bridge it, so a disagreement would be a column mismatch discovered at the
    bridge rather than at the write.
    """
    columns = ["frame", "id"]
    for k in range(n_keypoints):
        columns += [f"poseX{k}", f"poseY{k}", f"poseP{k}"]
    return columns


POINT_COLUMNS: Final = (
    "frame",
    "detection_id",
    "x",
    "y",
    "confidence",
    "class_id",
    "class_name",
)
"""The point predictions parquet's column names, in write order.

Fixed rather than a function of the model, unlike :func:`pose_columns`: a point
detector localizes one thing per detection whatever it was trained on.
``class_name`` is the one non-numeric column either table carries, so it is
appended after the numeric block rather than being part of it.
"""


class Locations(Detections, Protocol):
    """POLO's ``locations`` surface, after ``.cpu().numpy()``."""

    def cpu(self) -> Locations: ...
    def numpy(self) -> Locations: ...


class InferenceResult(Protocol):
    """What an inference run reads off every frame's result, whatever the task.

    ``plot`` is Ultralytics' own renderer, and the reason annotated frames are
    drawn inside the runner: it reads live tensors off the result, so there is
    nothing to draw from once the boundary is crossed.
    """

    @property
    def names(self) -> dict[int, str]: ...
    def plot(self) -> np.ndarray: ...


class PoseResult(InferenceResult, Protocol):
    """What pose inference additionally reads."""

    @property
    def keypoints(self) -> Keypoints | None: ...


class PointResult(InferenceResult, Protocol):
    """What point inference additionally reads."""

    @property
    def locations(self) -> Locations | None: ...


def pose_rows_from_result(
    result: PoseResult, position: int, *, n_keypoints: int
) -> np.ndarray | None:
    """One frame's pose detections as a ``(n, 2 + 3K)`` block, or None.

    *position* is how many frames the run has already processed, **not** the
    source frame index. That is what the column has always held -- the in-process
    converter numbered rows by their place in the results list -- and a run with
    ``start_frame`` or ``frame_step`` set therefore numbers from zero by ones
    whatever it read. Preserved here rather than corrected, so this move changes
    the coordinate space and nothing else.
    """
    keypoints = result.keypoints
    if keypoints is None:
        return None
    data = keypoints.cpu().numpy().data
    if data.shape[0] == 0:
        return None
    if data.shape[1] != n_keypoints:
        raise UltralyticsInteropError(
            f"the model declares {n_keypoints} keypoints and this frame's "
            f"predictions carry {data.shape[1]}. The column names were built "
            "from the declared count, so writing these would put one keypoint's "
            "coordinates under another's name."
        )

    n = data.shape[0]
    block = np.empty((n, 2 + 3 * n_keypoints), dtype=np.float64)
    block[:, 0] = float(position)
    block[:, 1] = np.arange(n, dtype=np.float64)
    kp = np.empty((n, n_keypoints, 3), dtype=np.float64)
    kp[:, :, 0:2] = data[:, :, 0:2]
    kp[:, :, 2] = data[:, :, 2] if data.shape[2] > 2 else 1.0
    block[:, 2:] = kp.reshape(n, -1)
    return block


def point_rows_from_result(result: PointResult, position: int) -> np.ndarray | None:
    """One frame's point detections as a ``(n, 6)`` numeric block, or None.

    Six columns, not seven: ``class_name`` is a string and is mapped from
    ``class_id`` once, when the table is assembled. *position* means what it does
    in :func:`pose_rows_from_result`.

    POLO writes a location as ``[x, y, conf, cls]`` or, when it also tracked it,
    ``[x, y, track_id, conf, cls]``. Both shapes are read, and the track id is
    dropped: ``infer-points`` runs a detector, so a per-frame ordinal is the only
    identity these rows honestly carry.
    """
    locations = result.locations
    if locations is None:
        return None
    data = locations.cpu().numpy().data
    if data.shape[0] == 0:
        return None

    n = data.shape[0]
    tracked = data.shape[1] >= _POINT_TRACKED_COLUMNS
    block = np.empty((n, 6), dtype=np.float64)
    block[:, 0] = float(position)
    block[:, 1] = np.arange(n, dtype=np.float64)
    block[:, 2:4] = data[:, 0:2].astype(np.float64)
    block[:, 4] = data[:, 3 if tracked else 2].astype(np.float64)
    block[:, 5] = data[:, 4 if tracked else 3].astype(np.float64)
    return block


class InferRequestBase(BaseModel):
    """What both inference subcommands take. Never sent on its own.

    The fields ``track`` also carries mean what they mean there. What is absent
    is the whole tracking half -- no ``tracker_yaml``, no ``persist``, no
    ``project_dir`` -- because a detector holds no state between frames and
    Ultralytics computes no run directory for ``predict``.
    """

    model_path: str
    video_path: str
    output_parquet: str
    """Where the raw predictions table is published, atomically."""

    annotated_dir: str
    """Where annotated frames are written, or empty to write none.

    The drawing happens here because the object it draws from cannot cross the
    boundary: ``Results.plot()`` is Ultralytics' own renderer over live tensors.
    Empty rather than nullable, so the field is one type on both sides.
    """

    columns: list[str]
    """The raw-parquet column names, in write order.

    Built by mosaic from :func:`pose_columns` or :data:`POINT_COLUMNS` and sent,
    rather than rebuilt here, for the reason ``TrackRequest.columns`` gives: the
    file this program writes is read back by mosaic, so the column list must
    exist in one place.
    """

    task: InferTask
    """The task the run was minted under, checked against the loaded weights."""

    conf: float
    imgsz: int
    device: str
    start_frame: int
    end_frame: int | None
    frame_step: int
    max_frames: int | None
    batch_size: int
    prefetch: bool
    media_facts: dict[str, object]
    """A ``mosaic_media.MediaFacts`` flattened with ``dataclasses.asdict``.

    **Required, and not nullable**, for the reason
    :attr:`TrackRequest.media_facts` gives at length: mosaic owns the read-target
    gate and this program cannot call it, so an omitted payload would leave the
    reader probing ungated.
    """


class InferPoseRequest(InferRequestBase):
    """One video, run through a YOLO pose model, in a process of its own."""

    n_keypoints: int
    """How many keypoints each row carries, from the probe. A term of the column
    contract, like ``columns`` itself, and the count this run's predictions are
    checked against."""


class InferPointsRequest(InferRequestBase):
    """One video, run through a POLO point model, in a process of its own.

    Carries no ``dor``. ``PointInferParams.dor`` reaches no Ultralytics argument
    on the in-process path either -- it is declared, documented, and never
    passed -- so sending it would not preserve behavior but change it, under a
    keyword whose effect on the fork is unverified. It stays a term of the run
    identifier and nothing else.
    """

    radii: dict[int, float] | None
    """Per-class detection radii, or None for the fork's own defaults."""


class InferResponse(BaseModel):
    """What one inference video produced.

    ``n_rows`` rather than the tracker's ``n_ids``: these are detections, and the
    identity column is a per-frame ordinal, so counting distinct values of it
    would report the largest number of detections in any one frame and read like
    a count of individuals.
    """

    n_frames: int
    """Frames read and run through the model."""

    n_rows: int
    """Detections written across every frame."""


# --- training --------------------------------------------------------------


TrainStop: TypeAlias = Literal["completed", "early_stopped", "cancelled"]
"""How a training run ended.

``completed`` ran its epoch target. ``early_stopped`` is Ultralytics' own
``patience``. ``cancelled`` is the epoch callback finding the sentinel and
setting ``trainer.stop``, and is the only one of the three that is not a
finished model.
"""


class TrainRequestBase(BaseModel):
    """What both training subcommands take. Never sent on its own.

    Everything here is already resolved. Mosaic decides which base weights win,
    where the run lands, and what a preset name meant, and sends the answers; this
    program constructs one model, calls ``train`` once, and reports how it ended.

    What is absent is the whole video-reading half the inference requests carry --
    no ``media_facts``, no frame window, no ``prefetch`` -- because training reads
    the image files ``data_yaml`` declares and opens no reader, so mosaic's
    read-target gate has nothing to gate.
    """

    model: str
    """What ``YOLO()`` is constructed from, already chosen.

    Three things collapse into this one field, mosaic-side, in the order the op
    applies them: the ``last.pt`` mosaic found when the run resumes; else the
    resolved path of the base model when fine-tuning; else the caller's ``model``
    verbatim. Verbatim means it may be a bare Ultralytics asset name
    (``yolo11n-pose.pt``) or a fork configuration name (``polo26n.yaml``) rather
    than a path, which Ultralytics resolves itself -- fetching the first from a
    release when the environment does not hold it.

    One field because the in-process path had one winner and only ever passed one
    value. Two would let this program disagree with the identity mosaic minted
    from both.
    """

    data_yaml: str
    """The training data declaration, resolved against the dataset root.

    Absolute, because this program's working directory is not mosaic's, and the
    paths inside the file are resolved against the file rather than against
    whoever opened it.
    """

    epochs: int
    """The epoch target, and the denominator every progress line reports against."""

    imgsz: int
    batch: int
    device: str
    patience: int

    project_dir: str
    """The claimed run root, passed as Ultralytics' ``project``.

    Named for the argument it becomes, as :attr:`TrackRequest.project_dir` is, but
    unlike there this is a directory of mosaic's: the run root the op claimed,
    which already holds this request file. Ultralytics computes ``<project>/<name>``
    under it and writes its weights and ``results.csv`` there.
    """

    run_name: str
    """Ultralytics' ``name``, the subdirectory it writes into.

    Sent rather than assumed on either side because mosaic reads ``best.pt`` and
    ``results.csv`` back from underneath it, and a directory name spelled
    independently in two processes is one that can disagree. What Ultralytics
    *chose* comes back in :attr:`TrainResponse.save_dir`.
    """

    resume: bool
    """Whether to continue the checkpoint in :attr:`model` rather than start from it.

    Mosaic finds the checkpoint; this says only what to do with it. The split is
    deliberate: the checkpoint lives under mosaic's claimed run root by mosaic's
    own directory convention, and the refusal when there is none names that root.
    Letting this program search would put the convention in two places.
    """

    augment: dict[str, JsonValue]
    """The augmentation keywords, already resolved from the preset name.

    The preset tables live in mosaic and import no Ultralytics, so they stay where
    they are read and an unknown preset name is refused at submit -- before a run
    root is claimed and before an interpreter is spawned. Empty means *apply none
    of mosaic's*, which is what both an unset augmentation and a resuming run
    send: a resume restores the augmentation its checkpoint was trained under, and
    overriding that is not what resuming means.
    """

    train_overrides: dict[str, JsonValue]
    """The caller's own ``train`` keywords, applied last.

    A second bag rather than merged into :attr:`augment` by mosaic, for two
    reasons. Precedence: overrides beat the resolved augmentation, and one merged
    bag would have to record that ordering somewhere else. Provenance: this file
    stays on disk beside the run, and the two bags are what tell a reader which
    key came from a preset mosaic chose and which came from their own request.

    Mosaic has already refused the keys it supplies itself, so nothing here can
    retarget the run.
    """

    cancel_sentinel: str
    """A path this program stats at each epoch boundary; its appearance stops the run.

    Written by mosaic when the job's cancel token fires, and never by this
    program. Stopping this way is what leaves ``last.pt`` and ``results.csv``
    complete up to the epoch that finished, where killing the process loses
    whichever epoch was in flight. Ultralytics honors the flag between epochs and
    nowhere else, so one stat per epoch is the whole check.
    """


class TrainPoseRequest(TrainRequestBase):
    """One YOLO pose model, trained in a process of its own.

    Adds nothing. It exists so the two subcommands take two request types rather
    than one carrying a task field: the subcommand already names the task, and a
    payload that validated as either kind is one a mistyped subcommand would
    happily run.
    """


class TrainPointsRequest(TrainRequestBase):
    """One POLO point model, trained in a process of its own."""

    loc: float
    """The localization loss weight, a POLO ``train`` keyword."""

    loc_loss: str
    """Which localization loss, e.g. ``mse``.

    A bare ``str`` where the other closed sets here are ``Literal`` aliases, for
    the reason :attr:`ProbeResponse.model_task` gives: mosaic's own field is a free
    string and a term of the run identifier, so closing it at the boundary would
    refuse a run that has already been named on disk.
    """

    dor: float
    """The Distance of Reference threshold POLO evaluates against.

    Carried, where :class:`InferPointsRequest` deliberately drops the identically
    named field: on the inference path it reaches no Ultralytics argument at all,
    so sending it would change what that op computes. Here it is a ``train``
    keyword the in-process path always passed, so *not* sending it would change
    what this one computes.
    """


class TrainResponse(BaseModel):
    """How the run ended. The artifacts are on disk, not in here.

    Deliberately neither the Ultralytics results object nor a copy of the final
    metrics. ``best.pt`` and ``results.csv`` are written under :attr:`save_dir` and
    mosaic reads both from there -- it already did on the in-process path, where
    the returned object was assigned to nothing -- and every per-epoch number has
    already travelled on the progress channel into the run-log. A metric spelled
    here as well would be a second copy of a number ``results.csv`` holds, and the
    two would be read by different code.

    What is left is the three facts that exist only inside the training process.
    """

    save_dir: str
    """The directory Ultralytics actually wrote into.

    Reported rather than assumed. Mosaic composes ``<project_dir>/<run_name>`` and
    reads ``weights/best.pt`` and ``results.csv`` from under it, and mosaic also
    asks for ``exist_ok`` so that path is the one used -- but a build that
    increments the name anyway would leave mosaic registering some other attempt's
    weights under this run's identifier. The composed path is what mosaic expects;
    this is what happened, and the two disagreeing is a refusal rather than a
    shrug.
    """

    epochs_completed: int
    """Epochs that finished, which is not always the epochs that were asked for.

    ``patience`` stops a run early and a cancel stops it earlier still. The index
    row records this rather than the request's ``epochs``, so a model that trained
    for forty of three hundred does not read as a three-hundred-epoch model.
    """

    stop: TrainStop
    """Which of the three ways this run ended.

    The one fact mosaic cannot recover from disk: all three leave ``best.pt``,
    ``last.pt`` and a ``results.csv``, and the cancel token that used to answer the
    question lived in the trainer's own process. It no longer does, and the token
    being set is not the answer either -- a sentinel written during the final epoch
    loses the race, the run completes, and that is a model to register rather than
    an attempt to discard.
    """


class ProgressEvent(BaseModel):
    """One line of the runner's standard output.

    Two kinds. ``started`` is written once, as soon as Ultralytics is imported
    and before the weights load; ``progress`` after every decoded batch.
    Splitting them is what lets a supervising inactivity timeout be chosen
    against one batch rather than against a cold torch import as well, which is
    far longer and has nothing to do with whether the run is stuck. The weights
    load still falls between the two, so the bound must exceed it.
    """

    event: ProgressEventKind

    done: int = 0
    """Frames tracked so far. Zero on ``started``."""

    total: int = 0
    """Frames the run will read. Zero on ``started``: the reader is not open
    yet, and zero means *not reported* rather than *no frames*.
    """


class EpochEvent(BaseModel):
    """One completed training epoch, as a line on standard output.

    A **sibling** of :class:`ProgressEvent` rather than a third kind on it.
    ``done`` and ``total`` there are frames, and a model whose fields mean
    something on one kind and nothing on the next is how a plausible number gets
    recorded -- an epoch carrying ``done=0`` would read as "no frames" rather than
    as "not applicable".

    Splitting them also makes the addition invisible to every reader that came
    before it, by construction: mosaic validates a line against one model, and an
    epoch line fails :class:`ProgressEvent`'s validation and is answered ``None``.
    """

    event: Literal["epoch"]

    epoch: int
    """Zero-based, the trainer's own index.

    Absolute over the model's training history, so a resumed attempt's first epoch
    is not zero. That reads as a jump from nothing to 181/300 in the ledger, and it
    is correct: the numerator names an epoch of the model, not of the attempt.
    """

    total_epochs: int
    """The trainer's own total, not the request's.

    On a resume Ultralytics restores its arguments from the checkpoint, so what
    was asked for is not necessarily what runs, and a denominator taken from the
    request would not match this numerator.
    """

    metrics: dict[str, float] = {}
    """Finite floats only. What will not cast, and what is not finite, is dropped
    before it is written.

    The filtering happens on this side because the values start as torch tensors
    and a dict of arbitrary objects is exactly the untyped bag this module refuses
    everywhere else. It is also what keeps the line valid JSON: a NaN metric
    serializes as ``null``, which fails this annotation on arrival and would cost
    the whole epoch rather than the one key.
    """


class HeartbeatEvent(BaseModel):
    """Proof the training process is alive, between epochs.

    Carries no numbers at all, deliberately. Its whole job is to be a line: mosaic
    bounds *silence* with an inactivity watchdog and refreshes the run root's claim
    on every line, and an epoch on a large dataset can outlast both windows while
    everything is perfectly healthy. Ultralytics' own progress bar cannot serve --
    it writes carriage returns, which a line-oriented reader never sees as lines.

    Giving it a position would be worse than giving it none: the only number
    available mid-epoch is the last epoch that finished, and reporting that
    repeatedly is a counter that stalls rather than a counter that waits.
    """

    event: Literal["heartbeat"]


# `Boxes`, `Detections` and `Keypoints` are deliberately absent. They exist to
# give `Result` and `rows_from_result` something to say about the Ultralytics
# surface, and nothing on either side of the boundary names them: `Result` is
# what a caller passes. Exporting them would advertise a surface no caller has.
__all__ = [
    "POINT_COLUMNS",
    "EpochEvent",
    "HeartbeatEvent",
    "InferPointsRequest",
    "InferPoseRequest",
    "InferRequestBase",
    "InferResponse",
    "InferTask",
    "InferenceResult",
    "ModelTask",
    "PointResult",
    "PoseResult",
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
    "TrainPointsRequest",
    "TrainPoseRequest",
    "TrainRequestBase",
    "TrainResponse",
    "TrainStop",
    "UltralyticsInteropError",
    "point_rows_from_result",
    "pose_columns",
    "pose_rows_from_result",
    "rows_from_result",
]
