"""The division of an imgstore recording into sequences of whole chunks.

A recording is one store, or every store of one camera serial in clock start
order. Its rule (:class:`DivisionRule`) gives a division as a whole multiple of
the stores' ``chunksize``, in frames. Chunk ids alone decide which frames a
division contains. :func:`plan_division` applies the rule to chunk records
(:class:`ChunkRecord`) without opening a store:

- **Runs.** A run is a stretch of one store's chunks with no gap between them.
  A store boundary ends a run, and so does a step between neighboring chunks
  wider than the rule's ``gap_seconds``. Sequence ``k`` of a run starting at
  chunk ``s`` contains chunks ``s + k * n`` to ``s + (k + 1) * n - 1``, where ``n``
  is the chunks per division.
- **Closure.** A sequence is written once it is closed: it has ``n`` chunks, a
  gap or a later sequence of its store follows it, or its store has ended. A
  store has ended when its last chunk is indexed and short, or when a later
  store of the recording exists. A chunk without an index refuses its
  division, except the trailing chunk of a store still recording, which is
  not counted. The trailing sequence that is not closed is held back unless
  the caller releases it.
- **Placement.** A division's ``first_frame`` on the recording frame axis is
  its first chunk times ``chunksize``, plus the frames of the recording's
  earlier stores. A chunk of an earlier store that has no index counts as
  ``chunksize`` frames.
- **Names.** A division is named by the UTC start of its first frame
  (:func:`division_token`). The store's clock (:class:`StoreClock`) names
  divisions and does not decide membership.
- **Frozen divisions.** The rows written by an earlier plan are passed back as
  ``frozen``. Their names and starts are reused, and the plan refuses one it
  does not reproduce. A frozen division shorter than ``n`` chunks ends its run,
  so the chunks recorded after a released partial division start a new run on
  the same frame axis.

The module imports only the standard library and other standard-library leaves,
so a dataset read can use its types without loading a frame reader.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePath
from typing import Final, Literal, get_args
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from typing_extensions import TypeIs

from mosaic.core.entry import validate_entry_name
from mosaic.core.narrowing import is_str_mapping

from .store_rate import store_rate_mismatch

__all__ = [
    "DIVISION_IDENTITY_SCHEME",
    "GAP_INTERVALS",
    "ChunkRecord",
    "DivisionPlan",
    "DivisionRefusalReason",
    "DivisionRefused",
    "DivisionRule",
    "HeldBack",
    "HoldBackReason",
    "MediaRowKey",
    "PlannedDivision",
    "RecordingIdentity",
    "RefusedSpan",
    "RowDivision",
    "SpanRefusalReason",
    "StartSource",
    "StoreClock",
    "StorePlanInput",
    "StoreWindow",
    "division_rule",
    "division_token",
    "division_uid",
    "plan_division",
    "store_base_name",
    "store_clock",
]

DIVISION_IDENTITY_SCHEME: Final = "imgstore-division/1"
"""The scheme of a division's ``video_uuid`` and of its rule cell."""

GAP_INTERVALS: Final = 10
"""The nominal frame intervals that a step between frames exceeds to be a gap."""

StartSource = Literal["created_utc", "frame_time", "extra_data", "dirname"]
"""Where a store's clock takes its anchor, in the order :func:`store_clock` tries."""

RecordingIdentity = Literal["store", "camera"]
"""Whether a recording is one store, or every store of one camera serial."""

DivisionRefusalReason = Literal[
    "frames_not_chunk_multiple",
    "chunksize_mismatch",
    "rate_mismatch",
    "several_stores",
    "empty_camera_serial",
    "camera_serial_mismatch",
    "overlapping_stores",
    "empty_store_uuid",
    "invalid_name",
    "duplicate_token",
    "frozen_mismatch",
    "no_clock_source",
    "local_time_without_timezone",
    "unknown_timezone",
    "ambiguous_local_time",
    "nonexistent_local_time",
]
"""Why a rule, a recording or a store clock is refused as a whole."""

SpanRefusalReason = Literal[
    "missing_chunk_index", "short_interior_chunk", "gap_inside_chunk"
]
"""Why one division or one store of a plan is refused while the rest is planned."""

HoldBackReason = Literal["store_recording"]
"""Why a sequence is held back: its store is still recording it."""

_RECORDING_IDENTITIES: Final = frozenset(get_args(RecordingIdentity))
_FRAME_TIME_FLOOR: Final = datetime(2000, 1, 1, tzinfo=UTC)
_FRAME_TIME_AGREEMENT: Final = timedelta(hours=1)
_DIRNAME_TIME: Final = re.compile(r"(?<!\d)(\d{8}_\d{6})(?!\d)")
_TOKEN_FORMAT: Final = "%Y%m%dT%H%M%SZ"
_UID_LENGTH: Final = 32


class DivisionRefused(ValueError):
    """Raised when a rule, a recording or a store clock cannot be divided.

    Attributes:
        reason: The kind of refusal, stable across releases.
    """

    def __init__(self, reason: DivisionRefusalReason, message: str) -> None:
        super().__init__(message)
        self.reason: DivisionRefusalReason = reason


@dataclass(frozen=True, slots=True)
class StoreWindow:
    """The chunks ``chunk_first`` to ``chunk_last`` of one store, both included."""

    chunk_first: int
    chunk_last: int

    def __post_init__(self) -> None:
        if self.chunk_first < 0 or self.chunk_last < self.chunk_first:
            message = (
                f"a store window runs from chunk {self.chunk_first} to "
                f"{self.chunk_last}; it needs 0 <= first <= last"
            )
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """One chunk of a store, as its index records it.

    The four measured fields are ``None`` for a chunk without an index, which is
    still recording or damaged.

    Attributes:
        chunk_id: The chunk's number.
        path: The chunk's video file, or its directory in an image store.
        frame_count: The frames that the index lists.
        first_time: The ``frame_time`` of the first frame, ``None`` also for an
            indexed chunk of no frames.
        last_time: The ``frame_time`` of the last frame, ``None`` as
            ``first_time`` is.
        max_interval: The widest step between neighboring frame times inside the
            chunk, ``0.0`` for fewer than two frames.
    """

    chunk_id: int
    path: Path
    frame_count: int | None
    first_time: float | None
    last_time: float | None
    max_interval: float | None


@dataclass(frozen=True, slots=True)
class StoreClock:
    """The wall clock of one store, anchored at the ``frame_time`` of one frame.

    Attributes:
        anchor_utc: The UTC time of the anchor frame, timezone-aware.
        anchor_time: The ``frame_time`` of the anchor frame, in seconds.
        source: Where ``anchor_utc`` comes from.
        timezone_local: The store's declared IANA time zone, if any.
    """

    anchor_utc: datetime
    anchor_time: float
    source: StartSource
    timezone_local: str | None

    def utc_at(self, frame_time: float) -> datetime:
        """Return the UTC time of the frame recorded at *frame_time*.

        The offset from the anchor is the difference of the two frame times.
        """
        return self.anchor_utc + timedelta(seconds=frame_time - self.anchor_time)


@dataclass(frozen=True, slots=True)
class DivisionRule:
    """The declared division of one recording, frozen when it is first divided.

    Attributes:
        frames: The frames of a full division, a multiple of ``chunksize``.
        chunksize: The frames per chunk of the recording's stores.
        identity: Whether the recording is one store or one camera.
        gap_seconds: The widest step between frames that is not a gap.
    """

    frames: int
    chunksize: int
    identity: RecordingIdentity
    gap_seconds: float

    def __post_init__(self) -> None:
        if not self.gap_seconds > 0 or not math.isfinite(self.gap_seconds):
            message = (
                f"a division rule's gap threshold of {self.gap_seconds!r} seconds "
                "is not positive and finite"
            )
            raise ValueError(message)

    @property
    def chunks_per_division(self) -> int:
        """The chunks of a full division."""
        return self.frames // self.chunksize

    def to_cell(self) -> str:
        """Return the rule as canonical JSON for a media index cell."""
        document = {
            "scheme": DIVISION_IDENTITY_SCHEME,
            "frames": self.frames,
            "chunksize": self.chunksize,
            "identity": self.identity,
            "gap_seconds": self.gap_seconds,
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_cell(cls, cell: str) -> DivisionRule:
        """Read a rule that :meth:`to_cell` wrote.

        The frames are not checked against the chunksize here.
        :func:`plan_division` refuses such a rule.

        Raises:
            ValueError: If *cell* is not a rule of :data:`DIVISION_IDENTITY_SCHEME`,
                or its gap threshold is not positive and finite.
        """
        try:
            document: object = json.loads(cell)
        except json.JSONDecodeError as exc:
            message = f"the division rule cell is not JSON: {cell!r}"
            raise ValueError(message) from exc
        if not is_str_mapping(document):
            message = f"the division rule cell is not a JSON object: {cell!r}"
            raise ValueError(message)
        frames = document.get("frames")
        chunksize = document.get("chunksize")
        identity = document.get("identity")
        gap_seconds = document.get("gap_seconds")
        if (
            document.get("scheme") != DIVISION_IDENTITY_SCHEME
            or not _is_int(frames)
            or not _is_int(chunksize)
            or identity not in _RECORDING_IDENTITIES
            or not isinstance(gap_seconds, (int, float))
            or isinstance(gap_seconds, bool)
        ):
            message = f"the division rule cell is malformed: {cell!r}"
            raise ValueError(message)
        return cls(
            frames=frames,
            chunksize=chunksize,
            identity="camera" if identity == "camera" else "store",
            gap_seconds=float(gap_seconds),
        )


def _is_int(value: object) -> TypeIs[int]:
    """Whether *value* is an ``int`` other than a ``bool``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _check_frames(frames: int, chunksize: int) -> None:
    """Refuse *frames* unless they are a positive multiple of *chunksize*."""
    if chunksize <= 0 or frames <= 0 or frames % chunksize:
        message = (
            f"a division of {frames} frames is not a whole number of "
            f"{chunksize}-frame chunks. Give the division as a positive "
            "multiple of the store's chunksize."
        )
        raise DivisionRefused("frames_not_chunk_multiple", message)


def division_rule(
    frames: int,
    *,
    chunksize: int,
    nominal_interval: float,
    identity: RecordingIdentity = "store",
) -> DivisionRule:
    """Return the rule that first divides a recording.

    Args:
        frames: The frames of a full division.
        chunksize: The frames per chunk of the recording's stores.
        nominal_interval: The median step between frame times, in seconds.
        identity: Whether the recording is one store or one camera.

    Raises:
        DivisionRefused: If *frames* is not a positive multiple of *chunksize*.
        ValueError: If *nominal_interval* is not positive.
    """
    _check_frames(frames, chunksize)
    if not nominal_interval > 0 or not math.isfinite(nominal_interval):
        message = f"the nominal frame interval {nominal_interval!r} is not positive"
        raise ValueError(message)
    return DivisionRule(
        frames=frames,
        chunksize=chunksize,
        identity=identity,
        gap_seconds=GAP_INTERVALS * nominal_interval,
    )


def store_base_name(dirname: str, camera_serial: str) -> str:
    """Return a store directory name without its ``.<camera_serial>`` suffix.

    ``sound_2_20210930_172126.23739656`` of camera ``23739656`` is
    ``sound_2_20210930_172126``. A name without the suffix, or a store without a
    serial, keeps the whole name. ``Path.stem`` would strip a dotted serial as a
    file extension.
    """
    if not camera_serial:
        return dirname
    return dirname.removesuffix("." + camera_serial)


def division_uid(
    store_uuid: str, chunk_first: int, chunk_last: int, frame_count: int
) -> str:
    """Return the ``video_uuid`` of one division of a store.

    It is the first 32 hex characters of a sha256 over the scheme and the four
    arguments.
    """
    text = "\0".join(
        (
            DIVISION_IDENTITY_SCHEME,
            store_uuid,
            str(chunk_first),
            str(chunk_last),
            str(frame_count),
        )
    )
    return hashlib.sha256(text.encode()).hexdigest()[:_UID_LENGTH]


def division_token(start_utc: datetime) -> str:
    """Return the sequence name of a division starting at *start_utc*.

    The name is the UTC time ``YYYYMMDDTHHMMSSZ`` with the seconds floored.

    Raises:
        ValueError: If *start_utc* is naive.
    """
    if start_utc.tzinfo is None or start_utc.utcoffset() is None:
        message = f"a division start needs a timezone: {start_utc!r}"
        raise ValueError(message)
    return start_utc.astimezone(UTC).strftime(_TOKEN_FORMAT)


# The store clock


def _created_utc(descriptor: Mapping[str, object]) -> datetime | None:
    """Return ``__store.created_utc`` as an aware UTC time, a naive one read as UTC."""
    value = descriptor.get("created_utc")
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _epoch_utc(seconds: float) -> datetime | None:
    """Return *seconds* since the Unix epoch as UTC, ``None`` when out of range."""
    try:
        return datetime.fromtimestamp(seconds, UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        message = f"the store's timezone_local {name!r} is not a known time zone"
        raise DivisionRefused("unknown_timezone", message) from exc


def _local_to_utc(wall: datetime, zone: ZoneInfo, dirname: str) -> datetime:
    """Return the UTC time of the wall-clock time *wall* in *zone*.

    Raises:
        DivisionRefused: If the wall time occurs twice in *zone*, or is skipped.
    """
    early = wall.replace(tzinfo=zone, fold=0)
    late = wall.replace(tzinfo=zone, fold=1)
    if early.utcoffset() != late.utcoffset():
        round_trip = early.astimezone(UTC).astimezone(zone).replace(tzinfo=None)
        if round_trip != wall:
            message = (
                f"the store directory {dirname!r} names {wall} in {zone.key}, a "
                "local time that the clock change skips"
            )
            raise DivisionRefused("nonexistent_local_time", message)
        message = (
            f"the store directory {dirname!r} names {wall} in {zone.key}, a "
            "local time that occurs twice at the clock change"
        )
        raise DivisionRefused("ambiguous_local_time", message)
    return early.astimezone(UTC)


def _dirname_wall_time(dirname: str) -> datetime | None:
    """Return the ``YYYYMMDD_HHMMSS`` time in a store directory name, if any."""
    match = _DIRNAME_TIME.search(dirname)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _dirname_start(dirname: str, timezone_local: str | None) -> datetime | None:
    """The directory name's start for corroborating a frame time, if it converts."""
    wall = _dirname_wall_time(dirname)
    if wall is None or timezone_local is None:
        return None
    try:
        return _local_to_utc(wall, _zone(timezone_local), dirname)
    except DivisionRefused:
        return None


def store_clock(
    metadata: Mapping[str, object],
    dirname: str,
    first_frame_time: float,
    extra_recording_start: float | None,
) -> StoreClock:
    """Return the clock of a store whose first frame is at *first_frame_time*.

    The anchor is the first of these sources:

    1. ``__store.created_utc``, a naive value read as UTC.
    2. The first ``frame_time`` as Unix time, when it is on or after 2000-01-01
       and within an hour of the ``recording_start`` or of the directory name.
    3. *extra_recording_start*, the ``recording_start`` of the first chunk's
       extra data, as Unix time.
    4. The ``YYYYMMDD_HHMMSS`` in *dirname*, read in ``__store.timezone_local``.

    The anchor is the time of the first frame. A store's modification time is
    not read.

    Args:
        metadata: The parsed ``metadata.yaml`` of the store.
        dirname: The store's directory name.
        first_frame_time: The ``frame_time`` of the first frame of chunk 0.
        extra_recording_start: The ``recording_start`` of the store's extra
            data, if it records one.

    Raises:
        DivisionRefused: If every source is missing, or the directory name is
            the source and its local time lacks a zone, names an unknown zone,
            or is a time that a clock change skips or repeats.
    """
    descriptor = metadata.get("__store")
    store_fields: Mapping[str, object] = (
        descriptor if is_str_mapping(descriptor) else {}
    )
    zone_value = store_fields.get("timezone_local")
    timezone_local = zone_value if isinstance(zone_value, str) and zone_value else None

    def anchored(anchor_utc: datetime, source: StartSource) -> StoreClock:
        return StoreClock(anchor_utc, first_frame_time, source, timezone_local)

    created = _created_utc(store_fields)
    if created is not None:
        return anchored(created, "created_utc")

    recording_start = (
        None if extra_recording_start is None else _epoch_utc(extra_recording_start)
    )
    declared = [
        start
        for start in (recording_start, _dirname_start(dirname, timezone_local))
        if start is not None
    ]
    frame_utc = _epoch_utc(first_frame_time)
    if (
        frame_utc is not None
        and frame_utc >= _FRAME_TIME_FLOOR
        and any(abs(frame_utc - start) <= _FRAME_TIME_AGREEMENT for start in declared)
    ):
        return anchored(frame_utc, "frame_time")

    if recording_start is not None:
        return anchored(recording_start, "extra_data")

    wall = _dirname_wall_time(dirname)
    if wall is None:
        message = (
            f"the store {dirname!r} records neither created_utc nor "
            "recording_start, and its directory name is not YYYYMMDD_HHMMSS. "
            "Its start time is unknown."
        )
        raise DivisionRefused("no_clock_source", message)
    if timezone_local is None:
        message = (
            f"the store directory {dirname!r} names a local time, and the store "
            "lacks a timezone_local to read it in"
        )
        raise DivisionRefused("local_time_without_timezone", message)
    return anchored(_local_to_utc(wall, _zone(timezone_local), dirname), "dirname")


# Plan inputs and results


@dataclass(frozen=True, slots=True)
class StorePlanInput:
    """One store of a recording, as :func:`plan_division` reads it.

    Attributes:
        abs_path: The store's path, as its media index rows record it.
        store_uuid: The store's ``__store.uuid``.
        camera_serial: The store's camera serial, ``""`` without one.
        chunksize: The frames per chunk that the store declares.
        fps: The store's measured frame rate.
        clock: The store's clock.
        chunks: Every chunk of the store, an unindexed one included.
    """

    abs_path: str
    store_uuid: str
    camera_serial: str
    chunksize: int
    fps: float
    clock: StoreClock
    chunks: tuple[ChunkRecord, ...]

    @property
    def dirname(self) -> str:
        """The store's directory name."""
        return PurePath(self.abs_path).name


@dataclass(frozen=True, slots=True)
class MediaRowKey:
    """The key of one media index row, with blank chunk cells for an undivided row."""

    abs_path: str
    chunk_first: int | None
    chunk_last: int | None


@dataclass(frozen=True, slots=True)
class RowDivision:
    """One division as its media index row records it.

    Attributes:
        abs_path: The store's path.
        chunk_first: The division's first chunk.
        chunk_last: The division's last chunk.
        first_frame: The division's first frame on the recording frame axis.
        frame_count: The frames of the division.
        follows_gap: Whether a gap or a store boundary precedes the division.
        sequence: The division's name, a :func:`division_token`.
        start_utc: The UTC time of the division's first frame.
        start_source: The source of the clock that named the division.
    """

    abs_path: str
    chunk_first: int
    chunk_last: int
    first_frame: int
    frame_count: int
    follows_gap: bool
    sequence: str
    start_utc: datetime
    start_source: StartSource

    @property
    def key(self) -> MediaRowKey:
        """The key of the division's media index row."""
        return MediaRowKey(self.abs_path, self.chunk_first, self.chunk_last)

    @property
    def window(self) -> StoreWindow:
        """The division's chunks."""
        return StoreWindow(self.chunk_first, self.chunk_last)


@dataclass(frozen=True, slots=True)
class PlannedDivision:
    """One division that a plan writes.

    Attributes:
        row: The division's row.
        uid: The division's ``video_uuid`` (:func:`division_uid`).
        frozen: Whether the row repeats a frozen division.
    """

    row: RowDivision
    uid: str
    frozen: bool


@dataclass(frozen=True, slots=True)
class HeldBack:
    """The trailing sequence of a recording that is not yet closed.

    Attributes:
        abs_path: The store's path.
        chunk_first: The sequence's first chunk.
        chunk_last: The sequence's last indexed chunk.
        frame_count: The frames of its indexed chunks.
        reason: Why it is held back.
        message: The reason, for a person.
    """

    abs_path: str
    chunk_first: int
    chunk_last: int
    frame_count: int
    reason: HoldBackReason
    message: str


@dataclass(frozen=True, slots=True)
class RefusedSpan:
    """A division, or a whole store, that a plan refuses to write.

    Attributes:
        abs_path: The store's path.
        chunk_first: The span's first chunk.
        chunk_last: The span's last chunk.
        reason: Why it is refused.
        message: The reason, for a person.
    """

    abs_path: str
    chunk_first: int
    chunk_last: int
    reason: SpanRefusalReason
    message: str


@dataclass(frozen=True, slots=True)
class DivisionPlan:
    """The divisions of one recording under one rule.

    Attributes:
        recording: The recording's name, the group of its sequences.
        rule: The rule the plan applies.
        divisions: The divisions to write, in recording order.
        held_back: The trailing sequence still recording, if any.
        refused: The divisions and stores that are not written.
    """

    recording: str
    rule: DivisionRule
    divisions: tuple[PlannedDivision, ...]
    held_back: tuple[HeldBack, ...]
    refused: tuple[RefusedSpan, ...]


# Planning


def _check_stores(stores: Sequence[StorePlanInput], rule: DivisionRule) -> None:
    """Refuse stores that cannot form one recording under *rule*."""
    for candidate in stores:
        if candidate.chunksize != rule.chunksize:
            message = (
                f"the store {candidate.abs_path!r} has {candidate.chunksize}-frame "
                f"chunks, and its recording is divided in {rule.chunksize}-frame "
                "chunks"
            )
            raise DivisionRefused("chunksize_mismatch", message)
        if not candidate.store_uuid:
            message = (
                f"the store {candidate.abs_path!r} records no __store.uuid to "
                "name its divisions by"
            )
            raise DivisionRefused("empty_store_uuid", message)
    _refuse_rate_mismatch(stores)
    if rule.identity == "store":
        if len(stores) != 1:
            message = f"a recording by store is one store, and {len(stores)} were given"
            raise DivisionRefused("several_stores", message)
        return
    serials = sorted({candidate.camera_serial for candidate in stores})
    if "" in serials:
        message = "a recording by camera needs a camera serial on every store"
        raise DivisionRefused("empty_camera_serial", message)
    if len(serials) > 1:
        message = f"a recording by camera is one camera, and the stores name {serials}"
        raise DivisionRefused("camera_serial_mismatch", message)


def _refuse_rate_mismatch(stores: Sequence[StorePlanInput]) -> None:
    """Refuse stores whose slowest and fastest rates are two rates.

    The two extremes differ the most, so the outcome does not depend on the
    order of *stores*. A rate that is not finite differs from every rate.
    """
    if len(stores) < 2:
        return
    unmeasured = sorted(
        candidate.abs_path for candidate in stores if not math.isfinite(candidate.fps)
    )
    if unmeasured:
        message = (
            f"the store {unmeasured[0]!r} records no finite frame rate, and a "
            "recording has one frame rate"
        )
        raise DivisionRefused("rate_mismatch", message)
    by_rate = sorted(stores, key=lambda candidate: (candidate.fps, candidate.abs_path))
    slowest, fastest = by_rate[0], by_rate[-1]
    if store_rate_mismatch([slowest.fps, fastest.fps]) is not None:
        message = (
            f"the store {fastest.abs_path!r} records {fastest.fps} fps, and "
            f"{slowest.abs_path!r} records {slowest.fps} fps. A recording has one "
            "frame rate."
        )
        raise DivisionRefused("rate_mismatch", message)


def _indexed_times(candidate: StorePlanInput) -> list[tuple[float, float]]:
    """The first and last frame time of every chunk of *candidate* that has both."""
    return [
        (chunk.first_time, chunk.last_time)
        for chunk in candidate.chunks
        if chunk.first_time is not None and chunk.last_time is not None
    ]


def _start_order(candidate: StorePlanInput) -> tuple[datetime, str]:
    """Sort key placing stores in clock start order."""
    times = _indexed_times(candidate)
    start = (
        candidate.clock.utc_at(min(first for first, _ in times))
        if times
        else candidate.clock.anchor_utc
    )
    return start, candidate.abs_path


def _refuse_overlaps(ordered: Sequence[StorePlanInput]) -> None:
    """Refuse two stores of one camera whose time spans overlap."""
    previous: tuple[StorePlanInput, datetime] | None = None
    for candidate in ordered:
        times = _indexed_times(candidate)
        if not times:
            continue
        start = candidate.clock.utc_at(min(first for first, _ in times))
        end = candidate.clock.utc_at(max(last for _, last in times))
        if previous is not None and start <= previous[1]:
            message = (
                f"the store {candidate.abs_path!r} starts at {start.isoformat()}, "
                f"before the store {previous[0].abs_path!r} of its camera ends at "
                f"{previous[1].isoformat()}"
            )
            raise DivisionRefused("overlapping_stores", message)
        previous = (candidate, end)


def _recording_name(
    ordered: Sequence[StorePlanInput], identity: RecordingIdentity, name: str | None
) -> str:
    """Return the recording's name: *name*, else the one its identity derives."""
    first = ordered[0]
    if name is not None:
        candidate = name
    elif identity == "camera":
        candidate = first.camera_serial
    elif first.camera_serial:
        base = store_base_name(first.dirname, first.camera_serial)
        candidate = f"{first.camera_serial}_{base}"
    else:
        candidate = first.dirname
    problem: str | None = None
    if not candidate.strip():
        problem = "is empty"
    else:
        try:
            _ = validate_entry_name(candidate, "recording")
        except ValueError as exc:
            problem = str(exc)
        for separator in ("__", ":"):
            if problem is None and separator in candidate:
                problem = f"contains {separator!r}, which joins a group to a sequence"
    if problem is not None:
        message = f"the recording name {candidate!r} {problem}"
        raise DivisionRefused("invalid_name", message)
    return candidate


@dataclass(frozen=True, slots=True)
class _StorePlan:
    divisions: tuple[PlannedDivision, ...]
    held_back: tuple[HeldBack, ...]
    refused: tuple[RefusedSpan, ...]


def _store_refusal(
    candidate: StorePlanInput, rule: DivisionRule, last_id: int
) -> RefusedSpan | None:
    """The refusal of a whole store with a short interior chunk or a gap inside one."""
    for chunk in candidate.chunks:
        if chunk.frame_count is None or chunk.max_interval is None:
            continue
        if chunk.chunk_id < last_id and chunk.frame_count < rule.chunksize:
            message = (
                f"chunk {chunk.chunk_id} of the store {candidate.abs_path!r} contains "
                f"{chunk.frame_count} of {rule.chunksize} frames, and later chunks "
                "follow it"
            )
            return RefusedSpan(
                candidate.abs_path, 0, last_id, "short_interior_chunk", message
            )
        if chunk.max_interval > rule.gap_seconds:
            message = (
                f"chunk {chunk.chunk_id} of the store {candidate.abs_path!r} has a "
                f"{chunk.max_interval:.3f} s step between frames, wider than the "
                f"{rule.gap_seconds:.3f} s gap threshold, and a gap inside a chunk "
                "cannot end a sequence"
            )
            return RefusedSpan(
                candidate.abs_path, 0, last_id, "gap_inside_chunk", message
            )
    return None


def _frame_total(candidate: StorePlanInput, chunksize: int, *, ended: bool) -> int:
    """The frames of a store on its recording's frame axis.

    A chunk without an index counts as full, as an indexed interior chunk must
    be, unless it is the trailing chunk of a store still recording, which is not
    counted. When *ended* is true, the trailing chunk is counted as well, since
    no chunk of an ended store is still recording.
    """
    if not candidate.chunks:
        return 0
    last_id = max(chunk.chunk_id for chunk in candidate.chunks)
    counts = {chunk.chunk_id: chunk.frame_count for chunk in candidate.chunks}
    total = 0
    for chunk_id in range(last_id + 1):
        count = counts.get(chunk_id)
        if count is not None:
            total += count
        elif chunk_id < last_id or ended:
            total += chunksize
    return total


def _run_starts(
    chunks: Mapping[int, ChunkRecord],
    counted_last: int,
    rule: DivisionRule,
    frozen: Iterable[RowDivision],
) -> tuple[set[int], set[int]]:
    """Return the run starts of a store, and the run starts after a gap."""
    gaps: set[int] = set()
    for chunk_id in range(1, counted_last + 1):
        before = chunks.get(chunk_id - 1)
        after = chunks.get(chunk_id)
        if before is None or after is None:
            continue
        if before.last_time is None or after.first_time is None:
            continue
        if after.first_time - before.last_time > rule.gap_seconds:
            gaps.add(chunk_id)
    starts = {0} | gaps
    for row in frozen:
        if row.chunk_last - row.chunk_first + 1 < rule.chunks_per_division:
            starts.add(row.chunk_last + 1)
    return starts, gaps


def _windows(
    starts: set[int], counted_last: int, per_division: int
) -> list[StoreWindow]:
    """The sequences of the runs that begin at *starts*, in chunk order."""
    bounds = sorted(start for start in starts if start <= counted_last)
    result: list[StoreWindow] = []
    for position, start in enumerate(bounds):
        end = bounds[position + 1] - 1 if position + 1 < len(bounds) else counted_last
        for first in range(start, end + 1, per_division):
            result.append(StoreWindow(first, min(first + per_division - 1, end)))
    return result


def _plan_store(
    candidate: StorePlanInput,
    rule: DivisionRule,
    *,
    offset: int,
    after_boundary: bool,
    later_store: bool,
    include_partial: bool,
    frozen: Sequence[RowDivision],
) -> _StorePlan:
    """Plan the divisions of one store of a recording."""
    chunks = {chunk.chunk_id: chunk for chunk in candidate.chunks}
    last_id = max(chunks, default=-1)
    refused_store = _store_refusal(candidate, rule, last_id)
    if last_id < 0 or refused_store is not None:
        if frozen:
            _refuse_unreproduced(frozen[0], None)
        refusals = () if refused_store is None else (refused_store,)
        return _StorePlan((), (), refusals)
    last = chunks[last_id]
    trailing_open = last.frame_count is None and not later_store
    counted_last = last_id - 1 if trailing_open else last_id
    ended = later_store or (
        last.frame_count is not None and last.frame_count < rule.chunksize
    )
    starts, gaps = _run_starts(chunks, counted_last, rule, frozen)
    per_division = rule.chunks_per_division
    frozen_by_first = {row.chunk_first: row for row in frozen}

    divisions: list[PlannedDivision] = []
    held_back: list[HeldBack] = []
    refused: list[RefusedSpan] = []
    for window in _windows(starts, counted_last, per_division):
        span = [
            chunks.get(chunk_id)
            for chunk_id in range(window.chunk_first, window.chunk_last + 1)
        ]
        records = [
            chunk
            for chunk in span
            if chunk is not None and chunk.frame_count is not None
        ]
        if len(records) < len(span):
            message = (
                f"chunks {window.chunk_first}-{window.chunk_last} of the store "
                f"{candidate.abs_path!r} include a chunk without an index"
            )
            refused.append(
                RefusedSpan(
                    candidate.abs_path,
                    window.chunk_first,
                    window.chunk_last,
                    "missing_chunk_index",
                    message,
                )
            )
            continue
        frame_count = sum(chunk.frame_count or 0 for chunk in records)
        first_time = records[0].first_time
        if frame_count == 0 or first_time is None:
            continue
        full = window.chunk_last - window.chunk_first + 1 == per_division
        closed = (
            full
            or window.chunk_last < counted_last
            or ended
            or window.chunk_first in frozen_by_first
        )
        if not closed and not include_partial:
            message = (
                f"chunks {window.chunk_first}-{window.chunk_last} of the store "
                f"{candidate.abs_path!r} contain {frame_count} of {rule.frames} "
                "frames, and the store is still recording"
            )
            held_back.append(
                HeldBack(
                    candidate.abs_path,
                    window.chunk_first,
                    window.chunk_last,
                    frame_count,
                    "store_recording",
                    message,
                )
            )
            continue
        start_utc = candidate.clock.utc_at(first_time)
        row = RowDivision(
            abs_path=candidate.abs_path,
            chunk_first=window.chunk_first,
            chunk_last=window.chunk_last,
            first_frame=offset + window.chunk_first * rule.chunksize,
            frame_count=frame_count,
            follows_gap=window.chunk_first in gaps
            or (window.chunk_first == 0 and after_boundary),
            sequence=division_token(start_utc),
            start_utc=start_utc,
            start_source=candidate.clock.source,
        )
        previous = frozen_by_first.pop(window.chunk_first, None)
        if previous is not None:
            _refuse_unreproduced(previous, row)
            row = previous
        divisions.append(
            PlannedDivision(
                row=row,
                uid=division_uid(
                    candidate.store_uuid, row.chunk_first, row.chunk_last, frame_count
                ),
                frozen=previous is not None,
            )
        )
    if frozen_by_first:
        _refuse_unreproduced(frozen_by_first[min(frozen_by_first)], None)
    return _StorePlan(tuple(divisions), tuple(held_back), tuple(refused))


def _refuse_unreproduced(frozen: RowDivision, planned: RowDivision | None) -> None:
    """Refuse a frozen division unless *planned* places the same frames."""
    if planned is not None and (
        planned.chunk_last,
        planned.first_frame,
        planned.frame_count,
        planned.follows_gap,
    ) == (
        frozen.chunk_last,
        frozen.first_frame,
        frozen.frame_count,
        frozen.follows_gap,
    ):
        return
    found = (
        "no division"
        if planned is None
        else (
            f"chunks {planned.chunk_first}-{planned.chunk_last} at frame "
            f"{planned.first_frame} with {planned.frame_count} frames"
        )
    )
    message = (
        f"the division {frozen.sequence!r} of the store {frozen.abs_path!r} contains "
        f"chunks {frozen.chunk_first}-{frozen.chunk_last} at frame "
        f"{frozen.first_frame} with {frozen.frame_count} frames, and the store now "
        f"gives {found}. A division's name and frames do not move once written."
    )
    raise DivisionRefused("frozen_mismatch", message)


def plan_division(
    stores: Sequence[StorePlanInput],
    rule: DivisionRule,
    *,
    name: str | None = None,
    include_partial: bool = False,
    frozen: Iterable[RowDivision] = (),
) -> DivisionPlan:
    """Divide one recording's stores into sequences of whole chunks.

    Args:
        stores: The recording's one store, or every store of its camera.
        rule: The recording's rule.
        name: The recording's name, in place of the one its identity derives.
        include_partial: Whether to write the trailing sequence that is not
            yet closed.
        frozen: The divisions written by earlier plans of the recording.

    Raises:
        DivisionRefused: If the rule, the stores, the name or a frozen division
            refuses the recording as a whole, or two divisions share a name.
        ValueError: If *stores* is empty.
    """
    if not stores:
        message = "a recording is divided from at least one store"
        raise ValueError(message)
    _check_frames(rule.frames, rule.chunksize)
    _check_stores(stores, rule)
    ordered = sorted(stores, key=_start_order)
    if rule.identity == "camera":
        _refuse_overlaps(ordered)
    recording = _recording_name(ordered, rule.identity, name)

    frozen_by_store: dict[str, list[RowDivision]] = {}
    for row in frozen:
        frozen_by_store.setdefault(row.abs_path, []).append(row)
    unknown = sorted(
        set(frozen_by_store) - {candidate.abs_path for candidate in ordered}
    )
    if unknown:
        row = frozen_by_store[unknown[0]][0]
        _refuse_unreproduced(row, None)

    divisions: list[PlannedDivision] = []
    held_back: list[HeldBack] = []
    refused: list[RefusedSpan] = []
    offset = 0
    for position, candidate in enumerate(ordered):
        later_store = position + 1 < len(ordered)
        planned = _plan_store(
            candidate,
            rule,
            offset=offset,
            after_boundary=position > 0,
            later_store=later_store,
            include_partial=include_partial,
            frozen=frozen_by_store.get(candidate.abs_path, ()),
        )
        divisions.extend(planned.divisions)
        held_back.extend(planned.held_back)
        refused.extend(planned.refused)
        offset += _frame_total(candidate, rule.chunksize, ended=later_store)

    seen: dict[str, RowDivision] = {}
    for division in divisions:
        token = division.row.sequence
        if token in seen:
            message = (
                f"two divisions of the recording {recording!r} start in the second "
                f"{token}: chunks {seen[token].chunk_first}-{seen[token].chunk_last} "
                f"and {division.row.chunk_first}-{division.row.chunk_last}"
            )
            raise DivisionRefused("duplicate_token", message)
        seen[token] = division.row
    return DivisionPlan(
        recording=recording,
        rule=rule,
        divisions=tuple(divisions),
        held_back=tuple(held_back),
        refused=tuple(refused),
    )
