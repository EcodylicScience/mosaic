"""The ``lmt_sqlite`` label converters: a database's animals and its events.

A Live Mouse Tracker database is one recording, and the ``lmt_sqlite`` format has
two label converters, ``id_tags`` for its animals and ``lmt_events`` for its
events. Both read the file through
:func:`~mosaic.core.track_library.lmt.open_lmt_database`, which opens it
``mode=ro&immutable=1``, so a conversion never changes the database, and a file with
a journal beside it is refused.

The ``id_tags`` converter returns one entry per database: the ``ANIMAL`` table as
the ``ids`` array, the ascending ``ANIMAL.ID``, and one text array per column present
among ``RFID``, ``NAME``, ``GENOTYPE``, ``AGE``, ``SEX``, ``STRAIN``, ``SETUP``,
``TREATMENT`` and ``IND``, named in lower case. Every cell is the ``str`` of its
value whatever SQLite stored, so an RFID keeps its leading zeros, and a NULL is the
empty string. The arrays load without unpickling. The entry is the raw row's group
under the conversion's ``group_from`` policy and the raw row's sequence, the entry
that the tracks of the same file convert to.

The ``lmt_events`` converter returns one entry per database, in the entry of its
tracks, and writes the segment sidecar alone (``segments_v1``). Every ``EVENT``
row of a name it reads is one segment: the actor ``IDANIMALA`` is
``seg_individual``, and ``IDANIMALB``, ``IDANIMALC`` and ``IDANIMALD`` are
``seg_partner``, ``seg_member_c`` and ``seg_member_d``, each ``-1`` when NULL. A pair
event stored in both directions is two segments. The segments are in ``EVENT.ID``
order, and the labels are the whole vocabulary of the params, so every file of a
variant declares the same ids. Events are placed by their seconds on the frames of
the entry's tracks when the dataset has them, and by frame number less the origin
otherwise. An event whose seconds hold no tracks frame is left out and its id is
listed in ``dropped_event_ids``. The file records the database's ``LOG`` rows as
``rebuild_log``, the builds that made its events.

With ``rebuild_run_id`` set, ``lmt_events`` reads the copy that a
``rebuild-lmt-events`` run made of the database in its place (``input_file``), so
``rebuild_log`` lists the builds of the source followed by the rebuild's. The file
then also holds the keys ``rebuild_run_id`` and ``lmt_analysis_commit`` as text, the
run and the lmt-analysis commit that the copy's manifest names, and both are ``""``
without a rebuild. ``id_tags`` always reads the raw database.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Self, override

import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator, model_validator

from mosaic.core.entry import entry_token
from mosaic.core.label_converter import (
    ClockedLabelConverter,
    LabelConvertParams,
    LabelConverter,
    LabelEntry,
    TrackClock,
    entry_group,
)
from mosaic.core.label_intervals import SEGMENTS_FORMAT, LabelSegments
from mosaic.core.params import Declared, has_fields
from mosaic.core.root_lookup import RootLookup
from mosaic.core.track_library.lmt import (
    LMT_EVENT_NAMES,
    LmtFrames,
    frame_seconds,
    open_lmt_database,
    table_columns,
)
from mosaic.core.track_library.lmt_rebuild import (
    REBUILD_RUN_ID_DESCRIPTION,
    read_rebuild_manifest,
    rebuild_input,
    refuse_foreign_rebuild_run_id,
)

__all__ = [
    "LmtAnimalsConverter",
    "LmtEventsConverter",
    "LmtEventsParams",
    "comma_free",
]

_FIELD_COLUMNS: Final = (
    "RFID",
    "NAME",
    "GENOTYPE",
    "AGE",
    "SEX",
    "STRAIN",
    "SETUP",
    "TREATMENT",
    "IND",
)
_SPELLING_NOTE = (
    "Each name is the stored spelling of EVENT.NAME, comma included, compared "
    "with EVENT.NAME character for character."
)
_EXTRA_EVENT_NAMES_DESCRIPTION = (
    "Event names given label ids after the built-in vocabulary, in the order "
    "written, from id 67. A name must not be in the built-in vocabulary or in "
    "skip_event_names. " + _SPELLING_NOTE
)
_SKIP_EVENT_NAMES_DESCRIPTION = (
    "Event names left out of the conversion. Their rows are not read past the "
    "name, and the run counts them. " + _SPELLING_NOTE
)
_QUALITY_EVENTS_DESCRIPTION = (
    "Event names marked as quality events instead of behavior. A name must be "
    "in the vocabulary. label-statistics ignores their frames unless its "
    "states name them. A name also in skip_event_names is never written, so "
    "it labels no frame. " + _SPELLING_NOTE
)


def comma_free(name: str) -> str:
    """Return *name* with each ``", "`` and ``","`` replaced by one space.

    The labels index joins ``label_names`` with commas, so a name with a comma
    would shift every later name onto the wrong id.
    """
    return name.replace(", ", " ").replace(",", " ")


class LmtEventsParams(LabelConvertParams):
    """What an event conversion chooses: the vocabulary and the quality events."""

    extra_event_names: Annotated[
        list[str], Declared(_EXTRA_EVENT_NAMES_DESCRIPTION)
    ] = []
    skip_event_names: Annotated[list[str], Declared(_SKIP_EVENT_NAMES_DESCRIPTION)] = []
    quality_events: Annotated[list[str], Declared(_QUALITY_EVENTS_DESCRIPTION)] = [
        "Flickering",
        "Detection",
        "Undetected",
    ]
    rebuild_run_id: Annotated[str | None, Declared(REBUILD_RUN_ID_DESCRIPTION)] = None

    @field_validator("rebuild_run_id", check_fields=False)
    @classmethod
    def _refuse_a_run_of_another_kind(cls, value: str | None) -> str | None:
        """Refuse a value that is not the id of a ``rebuild-lmt-events`` run."""
        return refuse_foreign_rebuild_run_id(value)

    @model_validator(mode="after")
    def _refuse_a_vocabulary_that_cannot_hold(self) -> Self:
        """Refuse names that two lists claim or that one index spelling shares.

        Raises:
            ValueError: A name is in both ``extra_event_names`` and
                ``skip_event_names``, or in ``extra_event_names`` and the
                built-in vocabulary; two vocabulary names have one comma-free
                spelling; or a ``quality_events`` name is not in the
                vocabulary. The message names the names and the param.
        """
        if not has_fields(
            self, "extra_event_names", "skip_event_names", "quality_events"
        ):
            return self
        skipped = set(self.skip_event_names)
        both = [name for name in self.extra_event_names if name in skipped]
        if both:
            raise ValueError(f"extra_event_names and skip_event_names both list {both}")
        built_in = set(LMT_EVENT_NAMES)
        repeated = [name for name in self.extra_event_names if name in built_in]
        if repeated:
            raise ValueError(
                f"extra_event_names lists {repeated}, which the built-in "
                "vocabulary already has"
            )
        vocabulary = self.vocabulary()
        spelled: dict[str, str] = {}
        for name in vocabulary:
            spelling = comma_free(name)
            if spelling in spelled:
                raise ValueError(
                    f"the vocabulary names {spelled[spelling]!r} and {name!r}, "
                    f"which both read {spelling!r} in the labels index; change "
                    "extra_event_names"
                )
            spelled[spelling] = name
        known = set(vocabulary)
        unknown = [name for name in self.quality_events if name not in known]
        if unknown:
            raise ValueError(
                f"quality_events lists {unknown}, which are not in the vocabulary"
            )
        return self

    def vocabulary(self) -> tuple[str, ...]:
        """The built-in names, then ``extra_event_names`` in order; the position is the id."""
        return (*LMT_EVENT_NAMES, *self.extra_event_names)


class LmtAnimalsConverter(LabelConverter[LabelConvertParams]):
    """A Live Mouse Tracker database -> its animals as individual tags."""

    src_format = "lmt_sqlite"
    label_kind = "id_tags"
    label_format = "id_tags_v1"
    version = "0.1"
    Params = LabelConvertParams
    refuses_shared_entry_keys = True

    def convert(
        self,
        src_path: Path,
        params: LabelConvertParams,
        raw_row: Mapping[str, object],
    ) -> list[LabelEntry]:
        """Read ``ANIMAL`` into one :class:`LabelEntry`.

        Raises:
            ValueError: The database has no ``ANIMAL`` table or no ``ID`` column, or
                a journal sits beside it. The message names the file.
        """
        connection = open_lmt_database(src_path)
        try:
            columns = table_columns(connection, "ANIMAL", str(src_path))
            if "ID" not in columns:
                raise ValueError(f"{src_path}: the ANIMAL table has no ID column")
            present = [name for name in _FIELD_COLUMNS if name in columns]
            selected = ", ".join(["ID", *present])
            rows = connection.execute(
                f"SELECT {selected} FROM ANIMAL ORDER BY ID"
            ).fetchall()
        finally:
            connection.close()

        ids = np.array([row[0] for row in rows], dtype=np.int64)
        payload: dict[str, object] = {"ids": ids}
        for position, name in enumerate(present, start=1):
            payload[name.lower()] = np.array(
                [("" if row[position] is None else str(row[position])) for row in rows],
                dtype=np.str_,
            )
        return [
            LabelEntry(
                group=entry_group(params.group_from, raw_row, ""),
                sequence=str(raw_row.get("sequence", "") or ""),
                payload=payload,
                n_frames=len(rows),
                label_ids=tuple(int(value) for value in ids),
                label_names=tuple(sorted(key for key in payload if key != "ids")),
            )
        ]


_ACTOR_COLUMNS: Final = ("IDANIMALA", "IDANIMALB", "IDANIMALC", "IDANIMALD")
_LOG_COLUMNS: Final = ("process", "version", "date", "tmin", "tmax")
_NO_ANIMAL: Final = -1


@dataclass(frozen=True, slots=True, eq=False)
class _EventRows:
    """The ``EVENT`` rows of the names a conversion reads, in key order."""

    key: NDArray[np.int64]
    name: tuple[str, ...]
    start_frame: NDArray[np.float64]
    end_frame: NDArray[np.float64]
    animals: NDArray[np.int64]


@dataclass(frozen=True, slots=True, eq=False)
class _Spans:
    """Each event's placed span, and which events have one.

    ``start`` and ``end`` hold a span at the events ``kept`` marks, end exclusive.
    """

    start: NDArray[np.int64]
    end: NDArray[np.int64]
    kept: NDArray[np.bool_]
    n_frames: int


def _refuse_unknown_names(
    connection: sqlite3.Connection, file: str, params: LmtEventsParams
) -> tuple[int, int]:
    """Count the events and those with a skipped name, refusing a name outside the lists.

    Returns:
        The number of ``EVENT`` rows and the number whose name is skipped.

    Raises:
        ValueError: A name is in neither the vocabulary nor ``skip_event_names``.
            The message names every such name, the file and both params.
    """
    counts = connection.execute(
        "SELECT NAME, COUNT(*) FROM EVENT GROUP BY NAME ORDER BY NAME"
    ).fetchall()
    known = set(params.vocabulary())
    skipped = set(params.skip_event_names)
    unknown = [
        f"{'NULL' if name is None else repr(name)} ({count} events)"
        for name, count in counts
        if name is None or (name not in known and name not in skipped)
    ]
    if unknown:
        raise ValueError(
            f"{file}: the EVENT table has names outside the vocabulary: "
            f"{', '.join(unknown)}. A label id follows from a name, so none is "
            "given on its own. List a name in extra_event_names to count it as "
            "behavior, or in skip_event_names to leave it out of the conversion."
        )
    total = sum(int(count) for _, count in counts)
    skipped_events = sum(int(count) for name, count in counts if name in skipped)
    return total, skipped_events


def _read_events(
    connection: sqlite3.Connection, file: str, skip: list[str]
) -> _EventRows:
    """Read the events whose name is not in *skip*, ordered by ``ID``.

    ``METADATA`` and ``DESCRIPTION`` are not read. A missing ``IDANIMAL`` column
    reads as NULL.

    Raises:
        ValueError: The ``EVENT`` table lacks ``NAME``, ``STARTFRAME`` or
            ``ENDFRAME``.
    """
    columns = table_columns(connection, "EVENT", file)
    for required in ("NAME", "STARTFRAME", "ENDFRAME"):
        if required not in columns:
            raise ValueError(f"{file}: the EVENT table has no {required} column")
    key = "ID" if "ID" in columns else "rowid"
    actors = [name if name in columns else f"NULL AS {name}" for name in _ACTOR_COLUMNS]
    where = f"WHERE NAME NOT IN ({', '.join('?' * len(skip))}) " if skip else ""
    rows = connection.execute(
        f"SELECT {key}, NAME, STARTFRAME, ENDFRAME, {', '.join(actors)} FROM EVENT "
        f"{where}ORDER BY {key}",
        skip,
    ).fetchall()
    return _EventRows(
        key=np.array([row[0] for row in rows], dtype=np.int64),
        name=tuple(str(row[1]) for row in rows),
        start_frame=np.array([row[2] for row in rows], dtype=np.float64),
        end_frame=np.array([row[3] for row in rows], dtype=np.float64),
        animals=np.array(
            [
                [_NO_ANIMAL if cell is None else cell for cell in row[4:]]
                for row in rows
            ],
            dtype=np.int64,
        ).reshape(len(rows), len(_ACTOR_COLUMNS)),
    )


def _refuse_unplaceable(
    file: str, events: _EventRows, frames: LmtFrames
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The start and end second of each event, refusing one that has no seconds.

    Raises:
        ValueError: An event's ``STARTFRAME`` or ``ENDFRAME`` has no ``FRAME`` row,
            or its ``ENDFRAME`` is below its ``STARTFRAME``. The message names the
            file and the first such event's id.
    """
    start_row = frames.rows_of(events.start_frame)
    end_row = frames.rows_of(events.end_frame)
    reversed_span = events.end_frame < events.start_frame
    for position in np.flatnonzero((start_row < 0) | (end_row < 0) | reversed_span):
        event = f"event {int(events.key[position])} ({events.name[position]})"
        if start_row[position] < 0:
            raise ValueError(
                f"{file}: {event} starts on frame {events.start_frame[position]:g}, "
                "which has no FRAME row"
            )
        if end_row[position] < 0:
            raise ValueError(
                f"{file}: {event} ends on frame {events.end_frame[position]:g}, "
                "which has no FRAME row"
            )
        raise ValueError(
            f"{file}: {event} ends on frame {events.end_frame[position]:g}, before "
            f"its start frame {events.start_frame[position]:g}"
        )
    return frames.seconds[start_row], frames.seconds[end_row]


def _place_events(
    events: _EventRows,
    frames: LmtFrames,
    seconds: tuple[NDArray[np.float64], NDArray[np.float64]],
    clock: TrackClock | None,
    entry: tuple[str, str],
) -> _Spans:
    """Place each event on the frames of the entry's tracks, or on the file's numbers.

    With a clock that has the entry's tracks, a span starts at the first tracks
    frame at or after its start second and ends one past the last tracks frame at or
    before its end second, and a span with no such frame is not kept. The clock is
    asked once for the entry, with no seconds when no event is read, so an entry
    that has tracks is recorded as having read them. Without tracks a span is the
    file's frame numbers less the origin, the end exclusive.
    """
    start_second, end_second = seconds
    if clock is not None:
        first = clock.frames_at_or_after(*entry, start_second)
        bounds = clock.bounds(*entry)
        last = clock.frames_at_or_before(*entry, end_second)
        if first is not None and last is not None and bounds is not None:
            kept = ~np.isnan(first) & ~np.isnan(last) & (last >= first)
            return _Spans(
                start=np.where(kept, first, 0).astype(np.int64),
                end=np.where(kept, last, -1).astype(np.int64) + 1,
                kept=kept,
                n_frames=bounds.last_frame + 1,
            )
    origin = int(frames.numbers[0])
    return _Spans(
        start=events.start_frame.astype(np.int64) - origin,
        end=events.end_frame.astype(np.int64) - origin + 1,
        kept=np.ones(events.key.size, dtype=np.bool_),
        n_frames=int(frames.numbers[-1]) - origin + 1,
    )


def _rebuild_log(connection: sqlite3.Connection, file: str) -> NDArray[np.str_]:
    """The ``LOG`` rows as ``(n, 5)`` text, ordered by ``id``; ``(0, 5)`` without a table."""
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'LOG'"
    ).fetchone()
    if table is None:
        return np.empty((0, len(_LOG_COLUMNS)), dtype=np.str_)
    present = table_columns(connection, "LOG", file)
    key = "id" if "id" in present else "rowid"
    selects = [name if name in present else f"NULL AS {name}" for name in _LOG_COLUMNS]
    rows = connection.execute(
        f"SELECT {', '.join(selects)} FROM LOG ORDER BY {key}"
    ).fetchall()
    return np.array(
        [["" if cell is None else str(cell) for cell in row] for row in rows],
        dtype=np.str_,
    ).reshape(len(rows), len(_LOG_COLUMNS))


class LmtEventsConverter(ClockedLabelConverter[LmtEventsParams]):
    """A Live Mouse Tracker database -> its events as segments on its tracks."""

    src_format = "lmt_sqlite"
    label_kind = "lmt_events"
    label_format = SEGMENTS_FORMAT
    version = "0.1"
    Params = LmtEventsParams
    refuses_shared_entry_keys = True

    @override
    def input_file(
        self,
        path: Path,
        params: LmtEventsParams,
        source_md5: str,
        root_of: RootLookup,
    ) -> Path:
        return rebuild_input(path, params.rebuild_run_id, source_md5, root_of)

    @override
    def convert(
        self,
        src_path: Path,
        params: LmtEventsParams,
        raw_row: Mapping[str, object],
        clock: TrackClock | None = None,
    ) -> list[LabelEntry]:
        """Read ``EVENT`` into one :class:`LabelEntry` of segments.

        The stderr line counts the events kept, those with a skipped name and those
        left out because their seconds hold no tracks frame.

        Raises:
            ValueError: An event name is outside the vocabulary and the skipped
                names, an event's ``STARTFRAME`` or ``ENDFRAME`` has no ``FRAME``
                row, an ``ENDFRAME`` is below its ``STARTFRAME``, a table or
                column is missing, a journal sits beside the file, or with
                ``rebuild_run_id`` set the manifest beside the file names another
                run. The message names the file and the event id or name, or both
                runs.
        """
        group = entry_group(params.group_from, raw_row, "")
        sequence = str(raw_row.get("sequence", "") or "")
        file = str(src_path)
        commit = ""
        if params.rebuild_run_id is not None:
            manifest = read_rebuild_manifest(src_path)
            if manifest.run_id != params.rebuild_run_id:
                raise ValueError(
                    f"{file}: the manifest beside it names the run {manifest.run_id}, "
                    f"not the run {params.rebuild_run_id} that was asked for"
                )
            commit = manifest.lmt_analysis_commit
        connection = open_lmt_database(src_path)
        try:
            total, skipped = _refuse_unknown_names(connection, file, params)
            frames = frame_seconds(connection)
            events = _read_events(connection, file, params.skip_event_names)
            rebuild_log = _rebuild_log(connection, file)
        finally:
            connection.close()

        seconds = _refuse_unplaceable(file, events, frames)
        spans = _place_events(events, frames, seconds, clock, (group, sequence))
        kept = np.flatnonzero(spans.kept)
        dropped = events.key[~spans.kept]
        print(
            f"[lmt_events] {entry_token(group, sequence)}: kept {kept.size} of "
            f"{total} events; {skipped} with a skipped name and {dropped.size} "
            "without a tracks frame left out",
            file=sys.stderr,
        )

        vocabulary = params.vocabulary()
        label_of = {name: position for position, name in enumerate(vocabulary)}
        animals = events.animals[kept]
        segments = LabelSegments(
            start_frame=spans.start[kept],
            end_frame=spans.end[kept],
            label=np.array([label_of[events.name[at]] for at in kept], dtype=np.int32),
            individual=animals[:, 0],
            extra={
                "seg_partner": animals[:, 1],
                "seg_member_c": animals[:, 2],
                "seg_member_d": animals[:, 3],
                "seg_event_id": events.key[kept],
            },
        )
        label_names = tuple(comma_free(name) for name in vocabulary)
        payload: dict[str, object] = {
            **segments.to_payload(),
            "dropped_event_ids": dropped.astype(np.int64),
            "label_ids": np.arange(len(vocabulary), dtype=np.int32),
            "label_names": np.array(label_names, dtype=np.str_),
            "source_label_names": np.array(vocabulary, dtype=np.str_),
            "quality_label_ids": np.array(
                sorted({label_of[name] for name in params.quality_events}),
                dtype=np.int32,
            ),
            "background_id": np.int32(-1),
            "rebuild_log": rebuild_log,
            "rebuild_run_id": params.rebuild_run_id or "",
            "lmt_analysis_commit": commit,
            "label_format": self.label_format,
            "group": group,
            "sequence": sequence,
        }
        return [
            LabelEntry(
                group=group,
                sequence=sequence,
                payload=payload,
                n_frames=spans.n_frames,
                label_ids=tuple(range(len(vocabulary))),
                label_names=label_names,
            )
        ]
