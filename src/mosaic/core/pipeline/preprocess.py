"""Define the pre-processing op, its parameters, and the run identifier they mint.

A media variant is named ``preprocess.<version>-<digest>``, where the digest is
taken over :func:`preprocess_identity_payload`: each step with its version, in
order, the upstream variant, the labeled rate when one is set, the codec and the
quality the encode resolves to. The identifier is therefore a function of the
parameters alone, and a variant chained after another names it through
``media``.

:class:`PreprocessOp` writes one variant file per scoped entry. It reads the
entry media of the camera that a tracker reads, one clip at a time, or the
upstream variant's file when ``media`` names one. It applies the steps to every
selected frame and encodes the frames to a partial file, which is counted before
it is renamed into place. The row written beside the variants records the file's
placement in its entry's source, the file's probed facts and the composition of
the entry's media. A consumer therefore reads the row instead of probing the file.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
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

from mosaic.core.entry import CameraEntry, Entry
from mosaic.core.helpers import is_nameless_entry, make_entry_key
from mosaic.core.json_value import JsonValue
from mosaic.core.media.preprocess.crop import MIN_CROP_SIDE
from mosaic.core.media.preprocess.geometry import FrameMap, Placement
from mosaic.core.media.preprocess.registry import Frame, FrameFn, MediaStep
from mosaic.core.media.preprocess.specs import MediaStepSpec
from mosaic.core.media.timeline import ConcatenatedTimeline, concatenated_timeline
from mosaic.core.media.video_io import FFmpegVideoWriter, open_frame_reader
from mosaic.core.params import HASH_EXCLUDE, Declared, Params
from mosaic.core.pipeline._utils import ResolvedScope, atomic_write
from mosaic.core.pipeline.composition import compositions_disagree
from mosaic.core.pipeline.consumed_camera import one_camera_per_entry
from mosaic.core.pipeline.entry_claim import (
    open_entry,
    release_entry,
    throttled_refresh,
)
from mosaic.core.pipeline.identity_scheme import write_identity_scheme
from mosaic.core.pipeline.job import Cancelled
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME, op_run_id
from mosaic.core.pipeline.ops import Op, OpIdentity, register_op
from mosaic.core.pipeline.preprocess_index import (
    media_variant_index,
    build_media_variant_row,
    media_variant_rows,
    write_media_variant_row,
)
from mosaic.core.pipeline.preprocess_layout import (
    PREPROCESS_KIND,
    media_variant_index_path,
    media_variant_path,
    media_variant_recipe_path,
    media_variant_run_root,
    media_variant_work_root,
)
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.pipeline.sequence_index import media_compositions_for
from mosaic.core.pipeline.stream_copy import coded_frame_count
from mosaic.core.pipeline.variant_source import VariantLookup, VariantSource

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset, ResolvedScopeEntry
    from mosaic.core.pipeline.job import JobContext

__all__ = [
    "AV1_DEFAULT_QUALITY",
    "H264_DEFAULT_CRF",
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

PREPROCESS_VERSION: Final = "0.1"
"""The op version, the visible segment of every variant's run identifier."""

# Equal to mosaic-media's ANALYSIS_ENCODING.quality, the rate that an analysis
# transcode encodes at. It is copied here rather than read from there, because
# the resolved quality enters the run identifier, which must stay fixed when the
# installed mosaic-media changes its default.
AV1_DEFAULT_QUALITY: Final = 14
"""The SVT-AV1 CRF that an ``av1`` variant is encoded at when ``quality`` is unset."""

# Provisional, until the rate at which Lightning Pose's detections on an H.264
# variant match its detections on the source is measured.
H264_DEFAULT_CRF: Final = 16
"""The x264 CRF that an ``h264`` variant is encoded at when ``quality`` is unset."""

type VariantCodec = Literal["av1", "h264"]
"""The codecs that a variant can be encoded in."""


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
"""The adapter that reads a step's identity dump as JSON, turning tuples into lists."""

_STEPS_DESCRIPTION = (
    "The steps applied to each frame, in order. Every position and frame number "
    "that a step names is in the entry media's pixels and frames, whatever the "
    "step's position in the list."
)

_MEDIA_DESCRIPTION = (
    "The run identifier of a variant to read in place of the entry media. These "
    "steps then apply after that variant's. Empty reads the entry media."
)

_FPS_DESCRIPTION = (
    "The frame rate that the output file is labeled at, which sets its frames' "
    "timestamps and therefore the time base of a tool's per-second thresholds. "
    "Unset, it is the rate that the kept frames were recorded at: the first clip's "
    "rate divided by the decimation. Under any other label the tracker's "
    "per-second columns are scaled by the label instead of the recording rate, and "
    "they are dropped when its table is mapped back."
)

_CODEC_DESCRIPTION = (
    "The output codec. 'h264' is a fallback for a decoder without AV1 support, "
    "such as the DALI 2.3 video reader in Lightning Pose. It needs an ffmpeg built "
    "with libx264."
)

_QUALITY_DESCRIPTION = (
    "The encoder's constant rate factor, on the chosen codec's scale, where "
    f"lower is better: SVT-AV1 CRF from 0 to {_QUALITY_SCALES['av1'].maximum} for "
    f"'av1', x264 CRF from 0 to {_QUALITY_SCALES['h264'].maximum} for 'h264'. "
    f"Unset uses mosaic's default, {AV1_DEFAULT_QUALITY} for AV1 and "
    f"{H264_DEFAULT_CRF} for H.264."
)

_ALLOW_HARDWARE_DESCRIPTION = (
    "Permit the av1_nvenc hardware encoder where the machine offers a usable one. "
    "The encode falls back to the CPU encoder where it does not, and an 'h264' "
    "variant always encodes on the CPU. As a permission, it leaves the variant's "
    "run identifier unchanged."
)


class PreprocessParams(Params):
    """Declare the settings that make one media variant from its source.

    The source is the entry media or another variant. The run's entries are an
    argument to the run and are not part of these settings. The run identifier is
    these settings. Covering more entries therefore writes more files under the
    same identifier.
    """

    steps: Annotated[list[MediaStepSpec], Declared(_STEPS_DESCRIPTION)]
    # The shape of a run identifier is not checked here, because a recipe validates
    # its steps with a placeholder that stands for a reference to another step.
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
                "entry media itself. Leave `media` empty on the consumer instead"
            )
            raise ValueError(message)
        return steps

    @model_validator(mode="after")
    def _refuse_a_quality_off_the_scale(self) -> Self:
        # The field declares the lower bound, which is 0 on every scale.
        scale = _QUALITY_SCALES[self.codec]
        if self.quality is not None and self.quality > scale.maximum:
            message = (
                f"quality {self.quality} is off the {self.codec} scale. "
                f"{scale.name} runs from 0 to {scale.maximum}"
            )
            raise ValueError(message)
        return self


def resolved_quality(params: PreprocessParams) -> int:
    """Return the quality that *params* encodes at, or else the codec's default."""
    if params.quality is not None:
        return params.quality
    return _QUALITY_SCALES[params.codec].default


def _step_terms(step: MediaStep) -> dict[str, JsonValue]:
    """Return *step* as identity terms: its parameters, name and version."""
    # The dump already contains the ``step`` discriminator, which equals the name.
    terms = _JSON_OBJECT.validate_python(step.identity_dump())
    return {**terms, "step": step.name, "version": step.version}


def preprocess_identity_payload(params: PreprocessParams) -> dict[str, JsonValue]:
    """Return the payload that a variant's run identifier digests.

    The steps keep their order, since a crop then a mask and a mask then a crop
    are different recipes. ``quality`` is the resolved value. Leaving it unset and
    naming the default are therefore one variant. ``fps`` enters only when it is
    set, and ``allow_hardware`` never does.
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
    """Return the run identifier of the variant that *params* describes."""
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
"""The time that ffmpeg may take to finish an H.264 variant after its input closes."""


class PreprocessRefused(ValueError):
    """The op cannot apply the recipe to one of its entries.

    It is raised for the whole run, before any entry is decoded, for a crop or
    mask outside an entry's frame, steps that do not select a frame, or a frame
    size that the encoder cannot encode. The message names the entry.
    """


class VariantWriter(Protocol):
    """Define the writer that the op encodes a variant's frames through.

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
        """The number of frames that :meth:`write` has taken."""
        ...

    @property
    def encoder_name(self) -> str:
        """The encoder that the file is written with, recorded on the variant's row."""
        ...


class H264PipeWriter:
    """Write BGR frames to an H.264 mp4 by piping them to an ffmpeg subprocess.

    libx264 runs in a separate process. A GPL encoder is therefore not linked into
    this one. The file is yuv420p, the pixel format that an AV1 variant has too.
    ffmpeg converts the BGR frames with swscale's ``accurate_rnd`` rounding. Its
    default rounding darkens each channel by up to 3 gray levels per encode.
    ffmpeg's error output goes to a temporary file instead of a pipe. A verbose
    encoder therefore does not block on an undrained pipe, and the output is quoted
    in the error when ffmpeg fails.
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
            "-sws_flags",
            "accurate_rnd",
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
        """The number of frames that :meth:`write` has passed to ffmpeg."""
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
                time. The message contains ffmpeg's error output.
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
        """Return the failure message, with ffmpeg's error output."""
        _ = self._stderr.seek(0)
        detail = self._stderr.read().decode(errors="replace")
        self._stderr.close()
        return failed_message("ffmpeg", self._action, detail)


def open_variant_writer(
    params: PreprocessParams, path: Path, width: int, height: int, fps: float
) -> VariantWriter:
    """Return the writer that encodes a *width* x *height* variant to *path* at *fps*.

    *params* sets the codec and quality. ``allow_hardware`` applies only to the AV1
    writer, which falls back to the CPU encoder where the hardware one is not
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
        TranscodeError: If ffmpeg is not on ``PATH``, or does not list
            ``libx264``.
    """
    if encoder_available("libx264"):
        return
    message = (
        "an 'h264' variant is written with libx264, and the ffmpeg on PATH is "
        "missing or built without it. H.264 is only needed for a decoder that "
        "cannot read AV1. Use the default 'av1' codec, or install an ffmpeg that "
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
    """The run's plan for one entry, fixed before the entry is decoded."""

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

    @property
    def camera_entry(self) -> CameraEntry:
        return (self.group, self.sequence, self.camera)


type _ConsumedMedia = ResolvedScopeEntry | FileNotFoundError | MediaProbeError
"""An entry's consumed media, or the error that explains its absence."""


def _consumed_media(
    ds: Dataset, entries: Sequence[Entry], *, report_skipped: bool
) -> dict[Entry, _ConsumedMedia]:
    """Return the media of each entry's consumed camera, the one that a tracker reads.

    One read of the media index resolves every entry. An entry whose media needs
    a transcode that has not run fails alone. Without a media index every entry
    fails with the one error that the read raised. An entry with neither group
    nor sequence resolves under its first file's stem, a name that the scope's
    read cannot key back to the entry. Each such entry is therefore read
    separately.

    Args:
        ds: The dataset.
        entries: The entries, each keyed in the result.
        report_skipped: Print a line for each camera dropped beside the one
            read. The run's identity check passes ``False``. A run therefore
            prints each line once.

    Returns:
        Per entry, its media, or the ``FileNotFoundError`` or
        ``MediaProbeError`` that resolving it raised.
    """
    failed: dict[Entry, MediaProbeError] = {}

    def consumed_cameras(scoped: Sequence[Entry]) -> list[ResolvedScopeEntry]:
        return one_camera_per_entry(
            PREPROCESS_KIND,
            ds.resolve_media_scope(scoped, errors=failed),
            report_skipped=report_skipped,
        )

    nameless = [
        (group, sequence)
        for group, sequence in entries
        if is_nameless_entry(group, sequence)
    ]
    try:
        kept = {
            (media.group, media.sequence): media
            for media in consumed_cameras(
                [entry for entry in entries if entry not in nameless]
            )
        }
        for entry in nameless:
            cameras = consumed_cameras([entry])
            if cameras:
                kept[entry] = cameras[0]
    except FileNotFoundError as exc:
        return {entry: exc for entry in entries}
    consumed: dict[Entry, _ConsumedMedia] = {}
    for entry in entries:
        media = kept.get(entry)
        if media is not None:
            consumed[entry] = media
        elif entry in failed:
            consumed[entry] = failed[entry]
        else:
            message = f"{make_entry_key(*entry)}: no media is indexed for it"
            consumed[entry] = FileNotFoundError(message)
    return consumed


def _entry_media(key: str, media: ResolvedScopeEntry) -> tuple[_EntryMedia, Placement]:
    """Return *media*'s clips on one frame axis, and the identity placement.

    Raises:
        MediaProbeError: If the clips differ in frame size, or one lacks a frame
            rate. Either prevents reading the entry as one source.
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
    """Return the placement after every step, labeled at the variant's rate.

    The rate is *params*'s ``fps`` when set, and otherwise the rate that the kept
    frames were recorded at: the first clip's rate divided by the frame map's
    step.

    Raises:
        PreprocessRefused: If a step does not fit the entry, the steps do not
            select a frame, or the final frame size cannot be encoded.
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
            f"preprocess cannot make a variant of {key}: the steps do not select a "
            f"frame of its {start.source_frame_count}-frame media"
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


def _plan_entry(
    ds: Dataset,
    params: PreprocessParams,
    media: _ConsumedMedia,
    compositions: Mapping[Entry, str],
    upstream: VariantLookup | None,
) -> _EntryPlan:
    """Check one entry against the recipe without reading a frame.

    Args:
        ds: The dataset.
        params: The recipe.
        media: The entry's consumed media, or the error that explains its
            absence.
        compositions: The current media composition of each entry in scope.
        upstream: The variant that ``params.media`` names, read for every entry
            in scope, or ``None`` when the recipe reads the entry media.

    Raises:
        PreprocessRefused: If the recipe does not fit the entry.
        FileNotFoundError: If the entry lacks media, or the upstream variant
            lacks a file for it.
        MediaProbeError: If the entry's media cannot be read as one source.
        MediaVariantDriftedError: If the upstream variant is out of date.
    """
    if isinstance(media, (FileNotFoundError, MediaProbeError)):
        raise media
    key = make_entry_key(media.group, media.sequence)
    composition = compositions.get((media.group, media.sequence), "")
    source: _EntryMedia | VariantSource
    if upstream is not None:
        source = upstream.resolve(ds, media)
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


def _clip_frames(
    path: Path, facts: MediaFacts, first: int, last: int, step: int
) -> Iterator[Frame]:
    """Yield frames ``first, first + step, ..., last`` of one clip, in order.

    A clip is entered at *first* by seeking, which positions the reader through
    the clip's packet index. A raw elementary stream lacks timestamps and
    therefore a packet index. ``VideoReader`` refuses to seek one, because a seek
    without an index decodes from an unverified position. Such a clip is read from
    its start, and the frames before *first* are discarded. Its analysis verdict
    requires a transcode. An entry therefore resolves to the derivative instead,
    and this path is currently not reached.
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
    """Yield the frames that *frames* selects from an entry's clips, clip by clip.

    A clip without a selected frame is not opened.
    """
    for segment, path, facts in zip(
        source.timeline.segments, source.paths, source.facts, strict=True
    ):
        window = frames.within_segment(segment.start_frame, segment.end_frame)
        if window is not None:
            first, last = window
            yield from _clip_frames(
                path,
                facts,
                first - segment.start_frame,
                last - segment.start_frame,
                frames.step,
            )


def _upstream_frames(source: VariantSource, frames: FrameMap) -> Iterator[Frame]:
    """Yield the frames that *frames* selects from the upstream variant's file."""
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
    """Return the frames of *plan*'s entry that its final frame map selects."""
    if isinstance(plan.source, VariantSource):
        return _upstream_frames(plan.source, plan.final.frames)
    return _entry_frames(plan.source, plan.final.frames)


def _bind(steps: Sequence[MediaStep], start: Placement) -> list[FrameFn]:
    """Return each step's per-frame function, bound at the placement before it."""
    functions: list[FrameFn] = []
    placement = start
    for step in steps:
        functions.append(step.bind(placement))
        placement = step.place(placement)
    return functions


def _render(frame: Frame, functions: Sequence[FrameFn]) -> Frame:
    """Pass *frame* through every step, into the contiguous BGR image for an encoder."""
    for function in functions:
        frame = function(frame)
    if frame.ndim == 2:
        frame = np.asarray(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), dtype=np.uint8)
    return np.ascontiguousarray(frame)


def _reusable(row: Mapping[str, str] | None, plan: _EntryPlan) -> bool:
    """Return whether *row* records the entry's file as current.

    A current file is not encoded again. A blank composition on either side is
    unknown, and unknown is not drift. The recorded placement is therefore compared
    as well. An entry whose media now has other frames or another frame size places
    the steps elsewhere, and its file is written again. A chained variant is
    current only while its upstream file is the one that it read.
    """
    if row is None:
        return False
    if compositions_disagree(row["consumed_media_composition"], plan.composition):
        return False
    if row["placement"] != plan.final.to_json():
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

    The frames are written to a partial file in the entry's claimed working
    directory *work_dir* and not beside the destination, where the partial's name
    can collide with another entry's variant file. A partial whose coded frame
    count, or probed frame count, is not the placement's count is kept for
    inspection and refused. Otherwise it is renamed into place, and then its row
    is written, because a row asserts that the file exists.

    Raises:
        TranscodeError: If the encoded file has a different number of frames from
            the number that the steps select.
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
                f"{plan.key}: the encode has {coded} frames where the steps "
                f"select {expected}. It is not published, and it is kept at "
                f"{partial} for inspection."
            )
            raise TranscodeError(message)
        facts = probe_media(partial)
        if facts.frame_count != expected:
            keep_partial = True
            message = (
                f"{plan.key}: the encode has {expected} frames but its timestamps "
                f"place only {facts.frame_count}, and a reader that seeks by time "
                f"misses frames. It is kept at {partial} for inspection."
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
        build_media_variant_row(
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


def _record_recipe(ds: Dataset, params: PreprocessParams, run_id: str) -> None:
    """Mark variant *run_id*'s directory with its identity scheme and save its recipe.

    The recipe is *params* in the form that ``mosaic run --params`` reads. The
    command that rewrites an entry's variant therefore names this file. The save
    is best-effort. The run identifier already fixes the recipe, and a failure to
    write the readable copy does not fail an otherwise successful run.
    """
    run_root = media_variant_run_root(ds, run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    write_identity_scheme(run_root, OP_IDENTITY_SCHEME)
    path = media_variant_recipe_path(ds, run_id)
    recipe = params.model_dump_json(indent=2) + "\n"
    try:
        atomic_write(path, lambda target: target.write_text(recipe))
    except OSError as exc:
        print(f"[{PREPROCESS_KIND}] failed to save {path}: {exc}", file=sys.stderr)


@register_op
class PreprocessOp(Op[PreprocessParams]):
    """Write one media variant file per scoped entry.

    Every refusal of the recipe is raised before any entry is decoded. An entry
    whose media or upstream file cannot be read fails alone, and the rest
    continue. An entry is reused while its file and row exist, its media is the
    media that the file was written from, the steps place it where its row
    records, and, for a chained variant, the upstream file is the one that it
    read. ``overwrite`` encodes every entry in scope again. The run saves its
    parameters in the variant's directory, as the recipe that a remedy names.
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
        """Return the entry when there is one, and otherwise the entry count."""
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
        """Return the variant's run identifier, after checking the recipe if possible.

        The identifier is :func:`preprocess_identity`, a function of the
        parameters alone. When ``media`` is empty, the recipe is checked against
        each scoped entry whose media resolves, without reading a frame. A recipe
        that does not fit one is refused here, as ``run`` refuses it. An entry
        whose media needs a transcode not yet run, and a chained variant whose
        upstream is not yet written, are not checked, because a graph that runs
        that transcode or that upstream first must still plan.

        Raises:
            PreprocessRefused: If the recipe does not fit a checked entry.
        """
        if not params.media:
            entries = sorted(scope.entries)
            consumed = _consumed_media(ds, entries, report_skipped=False)
            for media in consumed.values():
                if isinstance(media, (FileNotFoundError, MediaProbeError)):
                    continue
                key = make_entry_key(media.group, media.sequence)
                try:
                    _, start = _entry_media(key, media)
                except MediaProbeError:
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
        # `plan_identity` resolves the scope without reporting a skipped
        # camera. The resolution below therefore reports each one once.
        run_id = self.plan_identity(ds, params, scope).run_id
        ctx.set_run_id(run_id)
        entries = sorted(scope.entries)
        ctx.set_total(len(entries))
        if params.codec == "h264":
            _require_libx264()

        # Every entry is resolved and checked before any is decoded. A recipe that
        # does not fit one entry is therefore refused before a file is written. The
        # media index, the compositions and the upstream variant's rows are each
        # read once for the whole scope.
        consumed = _consumed_media(ds, entries, report_skipped=True)
        resolved = [
            (media.group, media.sequence)
            for media in consumed.values()
            if not isinstance(media, (FileNotFoundError, MediaProbeError))
        ]
        compositions = media_compositions_for(ds, resolved)
        upstream = (
            VariantLookup(
                run_id=params.media,
                rows=media_variant_rows(ds, params.media),
                compositions=compositions,
            )
            if params.media
            else None
        )
        plans: list[_EntryPlan] = []
        failed = 0
        for group, sequence in entries:
            try:
                plans.append(
                    _plan_entry(
                        ds, params, consumed[(group, sequence)], compositions, upstream
                    )
                )
            except PreprocessRefused:
                raise
            except Exception as exc:
                ctx.entry_failed(make_entry_key(group, sequence), exc)
                failed += 1

        _record_recipe(ds, params, run_id)
        rows = media_variant_rows(ds, run_id)
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
                    reusable = False
                    if not overwrite and dest.is_file():
                        reusable = _reusable(rows.get(plan.camera_entry), plan)
                        if not reusable:
                            # Read again, since another execution may have
                            # written the entry after the rows were read. Only
                            # an entry about to be encoded reads them again.
                            rows = media_variant_rows(ds, run_id)
                            reusable = _reusable(rows.get(plan.camera_entry), plan)
                    if reusable:
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
                # The count includes cache hits. A resumed run and a fresh one
                # therefore report the same coverage.
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
                f"failed, and {run_id} did not write a variant file. The per-entry "
                f"errors are in this attempt's run-log."
            )
            raise AllEntriesFailed(message)
        held_note = f", {held} held by another execution" if held else ""
        run_root = media_variant_run_root(ds, run_id)
        print(
            f"[{PREPROCESS_KIND}] completed run_id={run_id} "
            f"({written}/{len(entries)} entries{held_note}) -> {run_root}"
        )
        return run_id
