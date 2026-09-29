"""The pre-processing op, its parameters, and the run identifier they mint.

A media variant is named ``preprocess.<version>-<digest>``, where the digest is
taken over :func:`preprocess_identity_payload`: each step with its version, in
order, the upstream variant, the labeled rate when one is set, the codec and the
quality the encode resolves to. The identifier is therefore a function of the
parameters alone, and a variant chained after another names it through
``media``.

:class:`PreprocessOp` writes one variant file per scoped entry. It reads the
entry media of the camera a tracker reads, one clip at a time, or the upstream
variant's file when ``media`` names one. It applies the steps to every selected
frame and encodes the frames to a partial file, which is counted before it is
renamed into place. The row written beside the variants records where the file
sits in its entry's source, the file's probed facts and what the entry's media
was, so a consumer reads the row instead of probing the file.
"""

from __future__ import annotations

import dataclasses
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal, Protocol, Self

import cv2
import numpy as np
from mosaic_media import MediaFacts, MediaProbeError, probe_media
from mosaic_media.ffmpeg import (
    failed_message,
    not_found_message,
    timed_out_message,
)
from mosaic_media.hwaccel import encoder_available
from mosaic_media.transcode import TranscodeError
from pydantic import Field, TypeAdapter, field_validator, model_validator

from mosaic.core.entry import Entry
from mosaic.core.helpers import make_entry_key
from mosaic.core.json_value import JsonValue
from mosaic.core.media.preprocess.crop import MIN_CROP_SIDE
from mosaic.core.media.preprocess.geometry import FrameMap, Placement
from mosaic.core.media.preprocess.registry import Frame, FrameFn, MediaStep
from mosaic.core.media.preprocess.specs import MediaStepSpec
from mosaic.core.media.timeline import (
    ConcatenatedTimeline,
    TimelineSegment,
    concatenated_timeline,
)
from mosaic.core.media.video_io import FFmpegVideoWriter, open_frame_reader
from mosaic.core.params import HASH_EXCLUDE, Declared, Params
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.composition import compositions_disagree
from mosaic.core.pipeline.consumed_camera import one_camera_per_entry
from mosaic.core.pipeline.entry_claim import (
    open_entry,
    release_entry,
    throttled_refresh,
)
from mosaic.core.pipeline.job import Cancelled
from mosaic.core.pipeline.op_identity import op_run_id
from mosaic.core.pipeline.ops import Op, OpIdentity, register_op
from mosaic.core.pipeline.preprocess_index import (
    media_variant_index,
    media_variant_row,
    variant_row,
    write_media_variant_row,
)
from mosaic.core.pipeline.preprocess_layout import (
    media_variant_index_path,
    media_variant_path,
    media_variant_run_root,
    media_variant_work_root,
)
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.pipeline.stream_copy import coded_frame_count
from mosaic.core.pipeline.tracks_index import media_composition_for
from mosaic.core.pipeline.variant_source import VariantSource, resolve_variant_source

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset, ResolvedScopeEntry
    from mosaic.core.pipeline.job import JobContext

__all__ = [
    "AV1_DEFAULT_QUALITY",
    "H264_DEFAULT_CRF",
    "PREPROCESS_KIND",
    "PREPROCESS_VERSION",
    "H264PipeWriter",
    "PreprocessOp",
    "PreprocessParams",
    "PreprocessRefused",
    "VariantCodec",
    "VariantWriter",
    "open_variant_writer",
    "preprocess_identity",
    "preprocess_identity_payload",
    "resolved_quality",
]

PREPROCESS_KIND: Final = "preprocess"
"""The op kind, which leads every variant's run identifier."""

PREPROCESS_VERSION: Final = "0.1"
"""The op version, the visible segment of every variant's run identifier."""

# Equal to mosaic-media's ANALYSIS_ENCODING.quality, the rate an analysis
# transcode encodes at. Held here rather than read from there because the
# resolved quality enters the run identifier, which must not move when the
# installed mosaic-media changes its default.
AV1_DEFAULT_QUALITY: Final = 14
"""The SVT-AV1 CRF an ``av1`` variant is encoded at when ``quality`` is unset."""

# Provisional, until the rate at which Lightning Pose's detections on an H.264
# variant match its detections on the source is measured.
H264_DEFAULT_CRF: Final = 16
"""The x264 CRF an ``h264`` variant is encoded at when ``quality`` is unset."""

type VariantCodec = Literal["av1", "h264"]
"""The codecs a variant can be encoded in."""


@dataclass(frozen=True, slots=True)
class _QualityScale:
    """One codec's constant-rate-factor scale and the default taken on it."""

    name: str
    maximum: int
    default: int


_QUALITY_SCALES: Final[dict[VariantCodec, _QualityScale]] = {
    "av1": _QualityScale("SVT-AV1 CRF", 63, AV1_DEFAULT_QUALITY),
    "h264": _QualityScale("x264 CRF", 51, H264_DEFAULT_CRF),
}
"""Each codec's quality scale, which starts at 0 on both."""

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
"""Reads a step's identity dump as JSON, so a polygon's tuples become lists."""

_STEPS_DESCRIPTION = (
    "The steps applied to each frame, in order. Every position and frame number a "
    "step names is in the entry media's pixels and frames, wherever the step sits "
    "in the list."
)

_MEDIA_DESCRIPTION = (
    "The run identifier of a variant to read in place of the entry media, so "
    "these steps apply after that variant's. Empty reads the entry media."
)

_FPS_DESCRIPTION = (
    "The frame rate the output file is labeled at, which sets its frames' "
    "timestamps and so what a tool with per-second thresholds reads. Unset, it is "
    "the rate the kept frames were recorded at: the first clip's rate divided by "
    "the decimation. Under any other label the tracker's per-second columns are "
    "wrong, and they are dropped when its table is mapped back."
)

_CODEC_DESCRIPTION = (
    "The output codec. 'h264' is a fallback for a decoder that cannot read AV1, "
    "such as Lightning Pose on a GPU below compute capability 8.6, and needs an "
    "ffmpeg built with libx264."
)

_QUALITY_DESCRIPTION = (
    "The encoder's constant rate factor, on the chosen codec's own scale, where "
    f"lower is better: SVT-AV1 CRF from 0 to {_QUALITY_SCALES['av1'].maximum} for "
    f"'av1', x264 CRF from 0 to {_QUALITY_SCALES['h264'].maximum} for 'h264'. "
    f"Unset uses mosaic's default, {AV1_DEFAULT_QUALITY} for AV1 and "
    f"{H264_DEFAULT_CRF} for H.264."
)

_ALLOW_HARDWARE_DESCRIPTION = (
    "Permit the av1_nvenc hardware encoder where the machine offers a usable one. "
    "The encode falls back to the CPU encoder where it does not, and an 'h264' "
    "variant always encodes on the CPU. A permission rather than a setting, so it "
    "does not change the variant's run identifier."
)


class PreprocessParams(Params):
    """How one media variant is made from the entry media or another variant.

    The settings alone. Which entries a run covers is an argument to the run,
    and the run identifier is these settings, so covering more entries writes
    more files under the same identifier.
    """

    steps: Annotated[list[MediaStepSpec], Declared(_STEPS_DESCRIPTION)]
    # Not checked for the shape of a run identifier here: a recipe validates its
    # steps with a placeholder standing for a reference to another step.
    media: Annotated[str, Declared(_MEDIA_DESCRIPTION)] = ""
    fps: Annotated[
        float | None,
        Field(gt=0.0, allow_inf_nan=False),
        Declared(_FPS_DESCRIPTION, unit="fps"),
    ] = None
    codec: Annotated[VariantCodec, Declared(_CODEC_DESCRIPTION)] = "av1"
    quality: Annotated[int | None, Field(ge=0), Declared(_QUALITY_DESCRIPTION)] = None
    allow_hardware: Annotated[
        bool, HASH_EXCLUDE, Declared(_ALLOW_HARDWARE_DESCRIPTION)
    ] = False

    @field_validator("steps")
    @classmethod
    def _refuse_no_steps(cls, steps: list[MediaStepSpec]) -> list[MediaStepSpec]:
        if not steps:
            message = (
                "a variant needs at least one step. The step-less variant is the "
                "entry media itself: leave `media` empty on the consumer instead"
            )
            raise ValueError(message)
        return steps

    @model_validator(mode="after")
    def _refuse_a_quality_off_the_scale(self) -> Self:
        # The lower bound is the field's own, and 0 on every scale.
        scale = _QUALITY_SCALES[self.codec]
        if self.quality is not None and self.quality > scale.maximum:
            message = (
                f"quality {self.quality} is off the {self.codec} scale: "
                f"{scale.name} runs from 0 to {scale.maximum}"
            )
            raise ValueError(message)
        return self


def resolved_quality(params: PreprocessParams) -> int:
    """The quality *params* encodes at: its own, or the codec's default."""
    if params.quality is not None:
        return params.quality
    return _QUALITY_SCALES[params.codec].default


def _step_terms(step: MediaStep) -> dict[str, JsonValue]:
    """*step* as identity terms: its parameters, name and version."""
    # The dump already holds the ``step`` discriminator, which equals the name.
    terms = _JSON_OBJECT.validate_python(step.identity_dump())
    return {**terms, "step": step.name, "version": step.version}


def preprocess_identity_payload(params: PreprocessParams) -> dict[str, JsonValue]:
    """Everything a variant's run identifier is a digest of.

    The steps keep their order, since a crop then a mask and a mask then a crop
    are different recipes. ``quality`` is the resolved value, so leaving it unset
    and naming the default are one variant. ``fps`` enters only when it is set,
    and ``allow_hardware`` never does.
    """
    payload: dict[str, JsonValue] = {
        "steps": [_step_terms(step) for step in params.steps],
        "media": params.media,
        "codec": params.codec,
        "quality": resolved_quality(params),
    }
    if params.fps is not None:
        payload["fps"] = params.fps
    return payload


def preprocess_identity(params: PreprocessParams) -> OpIdentity:
    """The run identifier of the variant *params* describes."""
    payload = preprocess_identity_payload(params)
    return OpIdentity(run_id=op_run_id(PREPROCESS_KIND, PREPROCESS_VERSION, payload))


# --- the op ------------------------------------------------------------------


_ENCODE_IDLE_SECONDS: Final = 600.0
"""The longest one entry goes without a decoded frame refreshing its claim.

The quiet stretches are opening a clip or seeking into it, and counting and
probing the encoded file. Frames arrive far more often than this while the
encode runs.
"""

_H264_FINISH_TIMEOUT_SECONDS: Final = 3600.0
"""How long ffmpeg may take to finish an H.264 variant once its input is closed."""


class PreprocessRefused(ValueError):
    """A recipe the op cannot apply to one of its entries.

    Raised for the whole run, before any entry is decoded: a crop or mask outside
    an entry's frame, steps that select no frame, or a frame size the encoder
    cannot hold. The message names the entry.
    """


class VariantWriter(Protocol):
    """What the op encodes a variant's frames through.

    ``FFmpegVideoWriter`` writes AV1 in this process, and :class:`H264PipeWriter`
    writes H.264 through an ffmpeg subprocess.
    """

    def write(self, frame: Frame) -> None:
        """Encode one ``H x W x 3`` BGR frame."""
        ...

    def close(self) -> None:
        """Finish the file."""
        ...

    @property
    def frames_written(self) -> int:
        """How many frames :meth:`write` has taken."""
        ...

    @property
    def encoder_name(self) -> str:
        """The encoder the file is written with, recorded on the variant's row."""
        ...


class H264PipeWriter:
    """Write BGR frames to an H.264 mp4 by piping them to an ffmpeg subprocess.

    libx264 runs in its own process, so no GPL encoder is linked into this one.
    The file is yuv420p, the pixel format an AV1 variant has too. ffmpeg's error
    output goes to a temporary file rather than a pipe, so a chatty encoder never
    blocks on a pipe nobody drains, and it is named in the error when ffmpeg
    fails.
    """

    def __init__(
        self, path: Path, width: int, height: int, fps: float, *, crf: int
    ) -> None:
        self._path = path
        self._shape = (height, width, 3)
        self._frames_written = 0
        self._closed = False
        self._action = f"encoding {path.name} as H.264"
        rate = Fraction(fps).limit_denominator(1_000_000)
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{rate.numerator}/{rate.denominator}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        self._stderr = tempfile.TemporaryFile()
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
            )
        except FileNotFoundError as exc:
            self._stderr.close()
            raise TranscodeError(not_found_message("ffmpeg", exc)) from exc

    @property
    def frames_written(self) -> int:
        """How many frames :meth:`write` has handed to ffmpeg."""
        return self._frames_written

    @property
    def encoder_name(self) -> str:
        """Always ``libx264``."""
        return "libx264"

    def write(self, frame: Frame) -> None:
        """Hand one ``H x W x 3`` BGR frame to ffmpeg.

        Raises:
            ValueError: If *frame* is not a uint8 image of the writer's size.
            TranscodeError: If the writer is closed, or ffmpeg has exited.
        """
        if self._closed:
            message = f"the H.264 writer for {self._path.name} is closed"
            raise TranscodeError(message)
        if frame.shape != self._shape or frame.dtype != np.uint8:
            message = (
                f"a frame of shape {tuple(frame.shape)} and type {frame.dtype} does "
                f"not fit the writer's {self._shape} uint8 frames"
            )
            raise ValueError(message)
        stdin = self._process.stdin
        if stdin is None:
            message = f"ffmpeg was started without an input pipe for {self._path.name}"
            raise TranscodeError(message)
        try:
            _ = stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            self._closed = True
            _ = self._process.wait()
            raise TranscodeError(self._failure()) from exc
        self._frames_written += 1

    def close(self) -> None:
        """Close ffmpeg's input and wait for it to finish the file.

        Raises:
            TranscodeError: If ffmpeg exits with an error or does not finish in
                time. The message carries ffmpeg's own error output.
        """
        if self._closed:
            return
        self._closed = True
        stdin = self._process.stdin
        try:
            if stdin is not None:
                stdin.close()
        except BrokenPipeError:
            pass
        try:
            returncode = self._process.wait(timeout=_H264_FINISH_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            self._process.kill()
            _ = self._process.wait()
            self._stderr.close()
            message = timed_out_message(
                "ffmpeg", self._action, timeout=_H264_FINISH_TIMEOUT_SECONDS
            )
            raise TranscodeError(message) from exc
        if returncode != 0:
            raise TranscodeError(self._failure())
        self._stderr.close()

    def _failure(self) -> str:
        """The failure message, carrying what ffmpeg wrote to its error output."""
        _ = self._stderr.seek(0)
        detail = self._stderr.read().decode(errors="replace")
        self._stderr.close()
        return failed_message("ffmpeg", self._action, detail)


def open_variant_writer(
    params: PreprocessParams, path: Path, width: int, height: int, fps: float
) -> VariantWriter:
    """The writer that encodes a *width* x *height* variant to *path* at *fps*.

    The codec and quality are *params*'s own. ``allow_hardware`` reaches only the
    AV1 writer, which falls back to the CPU encoder where the hardware one is not
    usable.
    """
    quality = resolved_quality(params)
    if params.codec == "h264":
        return H264PipeWriter(path, width, height, fps, crf=quality)
    return FFmpegVideoWriter(
        path, width, height, fps, av1_crf=quality, hwaccel=params.allow_hardware
    )


def _require_libx264() -> None:
    """Refuse an H.264 variant when ffmpeg cannot encode H.264.

    Raises:
        TranscodeError: If there is no ffmpeg on ``PATH``, or it lists no
            ``libx264``.
    """
    if encoder_available("libx264"):
        return
    message = (
        "an 'h264' variant is written with libx264, and the ffmpeg on PATH is "
        "missing or built without it. H.264 is only needed for a decoder that "
        "cannot read AV1: use the default 'av1' codec, or install an ffmpeg that "
        "includes libx264."
    )
    raise TranscodeError(message)


@dataclass(frozen=True, slots=True)
class _EntryMedia:
    """An entry's clips, read one at a time and placed on one frame axis."""

    paths: tuple[Path, ...]
    facts: tuple[MediaFacts, ...]
    timeline: ConcatenatedTimeline


@dataclass(frozen=True, slots=True)
class _EntryPlan:
    """Everything the run knows about one entry before decoding it."""

    group: str
    sequence: str
    camera: str
    source: _EntryMedia | VariantSource
    start: Placement
    final: Placement
    composition: str

    @property
    def key(self) -> str:
        return make_entry_key(self.group, self.sequence)


def _consumed_media(ds: Dataset, entry: Entry) -> ResolvedScopeEntry:
    """The media of *entry*'s consumed camera, the one a tracker reads.

    Raises:
        FileNotFoundError: If no media is indexed for *entry*, or there is no
            media index at all.
        MediaProbeError: If the entry's media needs a transcode that has not run.
    """
    # A multi-camera entry's skipped-camera line prints twice in a run whose
    # `media` is empty, once from `plan_identity` and once from `_plan_entry`:
    # the cost of `run` minting its identifier through `plan_identity`.
    resolved = one_camera_per_entry(PREPROCESS_KIND, ds.resolve_media_scope([entry]))
    if not resolved:
        group, sequence = entry
        message = f"{make_entry_key(group, sequence)}: no media is indexed for it"
        raise FileNotFoundError(message)
    return resolved[0]


def _entry_media(key: str, media: ResolvedScopeEntry) -> tuple[_EntryMedia, Placement]:
    """*media*'s clips on one frame axis, and the identity placement over them.

    Raises:
        MediaProbeError: If the clips differ in frame size, or one has no frame
            rate, so the entry cannot be read as one source.
    """
    # Local: `uniformity` imports this package through `media_index`, and this
    # module is imported by the package's `__init__`.
    from mosaic.core.media.uniformity import geometry_mismatch

    paths = tuple(media.resolved.paths)
    facts = tuple(media.resolved.facts)
    mismatch = geometry_mismatch(facts)
    if mismatch is not None:
        message = (
            f"{key}: {paths[mismatch.index].name} has {mismatch.field} "
            f"{mismatch.other} where {paths[0].name} has {mismatch.first}. The "
            f"clips of one entry must decode to one frame size."
        )
        raise MediaProbeError(message)
    try:
        timeline = concatenated_timeline(facts)
        first = facts[0]
        start = Placement.identity(
            first.width, first.height, timeline.total_frames, first.fps
        )
    except ValueError as exc:
        raise MediaProbeError(f"{key}: {exc}") from exc
    return _EntryMedia(paths, facts, timeline), start


def _place(
    key: str, params: PreprocessParams, start: Placement, first_clip_fps: float
) -> Placement:
    """The placement after every step, labeled at the variant's rate.

    The rate is *params*'s ``fps`` when set, and otherwise the rate the kept
    frames were recorded at: the first clip's rate divided by the frame map's
    step.

    Raises:
        PreprocessRefused: If a step does not fit the entry, the steps select no
            frame, or the final frame size cannot be encoded.
    """
    placement = start
    try:
        for step in params.steps:
            placement = step.place(placement)
    except ValueError as exc:
        message = f"preprocess cannot make a variant of {key}: {exc}"
        raise PreprocessRefused(message) from exc
    if placement.frames.count == 0:
        message = (
            f"preprocess cannot make a variant of {key}: the steps select no frame "
            f"of its {start.source_frame_count}-frame media"
        )
        raise PreprocessRefused(message)
    width, height = placement.width, placement.height
    if width % 2 or height % 2 or min(width, height) < MIN_CROP_SIDE:
        message = (
            f"preprocess cannot make a variant of {key}: its frames are "
            f"{width}x{height}, and a variant is encoded as yuv420p, which needs "
            f"both sides even and at least {MIN_CROP_SIDE}. Add a crop step that "
            f"keeps an even rectangle."
        )
        raise PreprocessRefused(message)
    labeled = (
        params.fps if params.fps is not None else first_clip_fps / placement.frames.step
    )
    return dataclasses.replace(placement, fps=labeled)


def _plan_entry(ds: Dataset, params: PreprocessParams, entry: Entry) -> _EntryPlan:
    """Resolve and check one entry, reading no frame.

    Raises:
        PreprocessRefused: If the recipe does not fit the entry.
        FileNotFoundError: If the entry has no media, or the upstream variant
            has no file for it.
        MediaProbeError: If the entry's media cannot be read as one source.
        MediaVariantDriftedError: If the upstream variant is out of date.
    """
    media = _consumed_media(ds, entry)
    key = make_entry_key(media.group, media.sequence)
    composition = media_composition_for(ds, media.group, media.sequence)
    source: _EntryMedia | VariantSource
    if params.media:
        source = resolve_variant_source(ds, params.media, media)
        start = source.placement
    else:
        source, start = _entry_media(key, media)
    final = _place(key, params, start, media.resolved.facts[0].fps)
    if isinstance(source, VariantSource):
        try:
            _ = final.frames.file_indices(source.placement.frames)
        except ValueError as exc:
            message = f"preprocess cannot make a variant of {key}: {exc}"
            raise PreprocessRefused(message) from exc
    return _EntryPlan(
        group=media.group,
        sequence=media.sequence,
        camera=media.camera,
        source=source,
        start=start,
        final=final,
        composition=composition,
    )


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _clip_window(frames: FrameMap, segment: TimelineSegment) -> tuple[int, int] | None:
    """The first and last frame of *segment* the map selects, clip-local, or None."""
    if frames.count == 0:
        return None
    last_selected = frames.start + frames.step * (frames.count - 1)
    low = max(frames.start, segment.start_frame)
    high = min(last_selected, segment.end_frame - 1)
    first = frames.start + frames.step * _ceil_div(low - frames.start, frames.step)
    if first > high:
        return None
    last = frames.start + frames.step * ((high - frames.start) // frames.step)
    return first - segment.start_frame, last - segment.start_frame


def _clip_frames(
    path: Path, facts: MediaFacts, first: int, last: int, step: int
) -> Iterator[Frame]:
    """Frames ``first, first + step, ..., last`` of one clip, in order.

    A clip is entered at *first* by seeking, which positions the reader through
    the clip's packet index. A raw elementary stream carries no timestamps and so
    has no packet index, and ``VideoReader`` refuses to seek one rather than
    decode from a wrong position. Such a clip is read from its start, and the
    frames before *first* are discarded. Its analysis verdict asks for a
    transcode, so today an entry resolves to the derivative instead and this
    path is not reached.
    """
    seekable = facts.timing_source != "absent"
    start, stride = (first, step) if seekable else (0, 1)
    with open_frame_reader(
        path,
        start_frame=start,
        end_frame=last + 1,
        frame_step=stride,
        facts=facts,
        target="analysis",
    ) as reader:
        for index, frame in reader:
            if index >= first and (index - first) % step == 0:
                yield np.asarray(frame, dtype=np.uint8)


def _entry_frames(source: _EntryMedia, frames: FrameMap) -> Iterator[Frame]:
    """The frames *frames* selects from an entry's clips, clip by clip.

    A clip holding no selected frame is not opened.
    """
    for segment, path, facts in zip(
        source.timeline.segments, source.paths, source.facts, strict=True
    ):
        window = _clip_window(frames, segment)
        if window is not None:
            yield from _clip_frames(path, facts, window[0], window[1], frames.step)


def _upstream_frames(source: VariantSource, frames: FrameMap) -> Iterator[Frame]:
    """The frames *frames* selects from the upstream variant's file."""
    first, stride, count = frames.file_indices(source.placement.frames)
    with open_frame_reader(
        source.path,
        start_frame=first,
        end_frame=first + stride * (count - 1) + 1,
        frame_step=stride,
        facts=source.facts,
        target="analysis",
    ) as reader:
        for _, frame in reader:
            yield np.asarray(frame, dtype=np.uint8)


def _source_frames(plan: _EntryPlan) -> Iterator[Frame]:
    """The frames of *plan*'s entry its final frame map selects, before any step."""
    if isinstance(plan.source, VariantSource):
        return _upstream_frames(plan.source, plan.final.frames)
    return _entry_frames(plan.source, plan.final.frames)


def _bind(steps: Sequence[MediaStep], start: Placement) -> list[FrameFn]:
    """Each step's per-frame function, bound at the placement before it."""
    functions: list[FrameFn] = []
    placement = start
    for step in steps:
        functions.append(step.bind(placement))
        placement = step.place(placement)
    return functions


def _render(frame: Frame, functions: Sequence[FrameFn]) -> Frame:
    """*frame* through every step, as the contiguous BGR image an encoder takes."""
    for function in functions:
        frame = function(frame)
    if frame.ndim == 2:
        frame = np.asarray(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), dtype=np.uint8)
    return np.ascontiguousarray(frame)


def _reusable(ds: Dataset, run_id: str, plan: _EntryPlan, dest: Path) -> bool:
    """Whether the entry's file and row are current, so the entry is not encoded.

    A blank composition on either side is unknown, and unknown is not drift. A
    chained variant is current only while its upstream file is the one it read.
    """
    if not dest.is_file():
        return False
    row = variant_row(ds, run_id, plan.group, plan.sequence, plan.camera)
    if row is None:
        return False
    if compositions_disagree(row["consumed_media_composition"], plan.composition):
        return False
    if isinstance(plan.source, VariantSource):
        return row["upstream_video_uuid"] == plan.source.facts.video_uuid
    return True


def _write_variant(
    ds: Dataset,
    ctx: JobContext,
    params: PreprocessParams,
    run_id: str,
    plan: _EntryPlan,
    work_dir: Path,
    refresh_claim: Callable[[], bool],
) -> None:
    """Encode one entry's variant, count it, publish it and record its row.

    The frames go to a partial file in the entry's claimed working directory
    *work_dir*, never beside the destination, where it could sit at the path of
    another entry's variant file. A partial whose coded frame count, or probed
    frame count, is not the placement's count is kept for inspection and
    refused. Otherwise it is renamed into place and its row written: a row is
    the claim that the file exists.

    Raises:
        TranscodeError: If the encode holds another number of frames than the
            steps select.
    """
    dest = media_variant_path(ds, run_id, plan.group, plan.sequence, plan.camera)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = work_dir / f"{dest.stem}.partial{dest.suffix}"
    final = plan.final
    expected = final.frames.count
    keep_partial = False
    try:
        writer = open_variant_writer(
            params, partial, final.width, final.height, final.fps
        )
        try:
            functions = _bind(params.steps, plan.start)
            for frame in _source_frames(plan):
                writer.write(_render(frame, functions))
                if refresh_claim():
                    ctx.heartbeat()
                    ctx.check_cancel()
        finally:
            writer.close()
        coded = coded_frame_count(partial)
        if coded != expected:
            keep_partial = True
            message = (
                f"{plan.key}: the encode holds {coded} frames where the steps "
                f"select {expected}, so it is not published. It is kept at "
                f"{partial} for inspection."
            )
            raise TranscodeError(message)
        facts = probe_media(partial)
        if facts.frame_count != expected:
            keep_partial = True
            message = (
                f"{plan.key}: the encode holds {expected} frames but its timestamps "
                f"place only {facts.frame_count}, so a reader seeking by time would "
                f"miss frames. It is kept at {partial} for inspection."
            )
            raise TranscodeError(message)
        _ = partial.replace(dest)
    finally:
        if not keep_partial:
            partial.unlink(missing_ok=True)
    upstream_video_uuid = (
        plan.source.facts.video_uuid if isinstance(plan.source, VariantSource) else ""
    )
    write_media_variant_row(
        ds,
        media_variant_row(
            ds,
            path=dest,
            run_id=run_id,
            group=plan.group,
            sequence=plan.sequence,
            camera=plan.camera,
            upstream=params.media,
            upstream_video_uuid=upstream_video_uuid,
            placement=final,
            facts=facts,
            encoder=writer.encoder_name,
            consumed_media_composition=plan.composition,
        ),
    )
    ctx.progress.on_phase(
        PREPROCESS_KIND,
        f"{plan.key}: {writer.frames_written} frames -> {dest.name} "
        f"({writer.encoder_name})",
    )


@register_op
class PreprocessOp(Op[PreprocessParams]):
    """Write one media variant file per scoped entry.

    Every refusal the recipe earns is raised before any entry is decoded. An
    entry whose media or upstream file cannot be read fails alone, and the rest
    carry on. An entry is reused while its file and row exist, its media is the
    media the file was written from, and, for a chained variant, the upstream
    file is the one it read. ``overwrite`` encodes every entry in scope again.
    """

    kind = PREPROCESS_KIND
    domain = "media"
    category = "preprocess"
    version = PREPROCESS_VERSION
    # ffmpeg-bound, like a transcode: one encode uses a machine's cores well, and
    # several side by side contend for them.
    resource_class = "heavy"
    scope_takes = "at-least-one"
    scope_dependent = False
    Params = PreprocessParams

    def target(self, params: PreprocessParams, scope: ResolvedScope) -> str:
        """The entry when there is one, and otherwise how many there are."""
        entries = sorted(scope.entries)
        if len(entries) == 1:
            group, sequence = entries[0]
            return f"{group}/{sequence}"
        return f"{len(entries)} entries"

    def plan_identity(
        self,
        ds: Dataset,
        params: PreprocessParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """The variant's run identifier, after checking the recipe where it can.

        The identifier is :func:`preprocess_identity`, a function of the
        parameters alone. When ``media`` is empty, the recipe is checked against
        each scoped entry whose media resolves, reading no frame, and a recipe
        that does not fit one is refused here as ``run`` would refuse it. An
        entry whose media needs a transcode not yet run, and a chained variant
        whose upstream is not yet written, are not checked: a graph that runs
        that transcode or that upstream first must still plan.

        Raises:
            PreprocessRefused: If the recipe does not fit a checked entry.
        """
        if not params.media:
            for entry in sorted(scope.entries):
                try:
                    media = _consumed_media(ds, entry)
                    key = make_entry_key(media.group, media.sequence)
                    _, start = _entry_media(key, media)
                except (FileNotFoundError, MediaProbeError):
                    continue
                _ = _place(key, params, start, media.resolved.facts[0].fps)
        return preprocess_identity(params)

    def run(
        self,
        ds: Dataset,
        params: PreprocessParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        run_id = self.plan_identity(ds, params, scope).run_id
        ctx.set_run_id(run_id)
        entries = sorted(scope.entries)
        ctx.set_total(len(entries))
        if params.codec == "h264":
            _require_libx264()

        # Every entry is resolved and checked before any is decoded, so a recipe
        # that does not fit one entry is refused before a file is written.
        plans: list[_EntryPlan] = []
        failed = 0
        for group, sequence in entries:
            try:
                plans.append(_plan_entry(ds, params, (group, sequence)))
            except PreprocessRefused:
                raise
            except Exception as exc:
                ctx.entry_failed(make_entry_key(group, sequence), exc)
                failed += 1

        work_root = media_variant_work_root(ds, run_id)
        written = 0
        reused = 0
        held = 0
        for position, plan in enumerate(plans):
            ctx.check_cancel()
            ctx.progress.on_entry_start(position, len(plans), plan.key)
            opened = open_entry(
                ds,
                ctx,
                work_root,
                plan.key,
                kind=PREPROCESS_KIND,
                overwrite=overwrite,
                idle_seconds=_ENCODE_IDLE_SECONDS,
            )
            if opened is None:
                held += 1
            else:
                work_dir, marker = opened
                try:
                    dest = media_variant_path(
                        ds, run_id, plan.group, plan.sequence, plan.camera
                    )
                    if not overwrite and _reusable(ds, run_id, plan, dest):
                        reused += 1
                        ctx.progress.on_phase(PREPROCESS_KIND, f"{plan.key}: reused")
                    else:
                        _write_variant(
                            ds,
                            ctx,
                            params,
                            run_id,
                            plan,
                            work_dir,
                            throttled_refresh(work_dir, marker, _ENCODE_IDLE_SECONDS),
                        )
                    written += 1
                except Cancelled:
                    raise
                except Exception as exc:
                    ctx.entry_failed(plan.key, exc)
                    failed += 1
                finally:
                    release_entry(work_dir, ctx.execution_id)
                # Cache hits included, so a resumed run and a fresh one report
                # the same coverage.
                ctx.entries_written(written)
            ctx.progress.on_entry_end(position + 1, len(plans), plan.key)
            ctx.heartbeat(done=len(entries) - len(plans) + position + 1)

        media_variant_index(media_variant_index_path(ds)).mark_finished(run_id)
        ctx.entries_written(written)
        if entries and reused == len(entries):
            ctx.cache_hit()
        attempted = len(entries) - held
        if attempted and failed == attempted:
            message = (
                f"[{PREPROCESS_KIND}] every one of {attempted} attempted entries "
                f"failed, so {run_id} wrote no variant file. The per-entry errors "
                f"are in this attempt's run-log."
            )
            raise AllEntriesFailed(message)
        held_note = f", {held} held by another execution" if held else ""
        run_root = media_variant_run_root(ds, run_id)
        print(
            f"[{PREPROCESS_KIND}] completed run_id={run_id} "
            f"({written}/{len(entries)} entries{held_note}) -> {run_root}"
        )
        return run_id
