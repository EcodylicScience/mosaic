"""Live Mouse Tracker databases, written the way the tracker writes them.

The statements of ``ANIMAL``, ``DETECTION``, ``FRAME``, ``EVENT`` and ``LOG``
are the ones of a database that Live Mouse Tracker wrote and that lmt-analysis
rebuilt. A recording is the rows of those tables as records, and
:func:`write_lmt_sqlite` writes them. A recording that a test needs slightly
different is the default one with a record replaced.

:func:`default_lmt_recording` is small and covers the shapes a reader meets: an
RFID with leading zeros, uneven timestamp steps, a missing frame number, a
paused frame that has detections, detections without an animal inserted under
explicit descending ids, and events of one, two and three animals, of none,
with a comma in the name, and overlapping. :func:`lmt_detection_grid` streams
detections without holding them, for the tests that write millions of rows.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import batched
from pathlib import Path
from typing import Final

__all__ = [
    "DEFAULT_ANIMAL_COLUMNS",
    "NINE_ANIMAL_COLUMNS",
    "LmtAnimal",
    "LmtDetection",
    "LmtEvent",
    "LmtFrame",
    "LmtLog",
    "LmtRecording",
    "default_lmt_recording",
    "lmt_detection_grid",
    "write_lmt_sqlite",
]

DEFAULT_ANIMAL_COLUMNS: Final = ("ID", "RFID", "GENOTYPE", "NAME")
"""The columns of the ``ANIMAL`` table that Live Mouse Tracker writes."""

NINE_ANIMAL_COLUMNS: Final = (
    "ID",
    "RFID",
    "NAME",
    "GENOTYPE",
    "AGE",
    "SEX",
    "STRAIN",
    "SETUP",
    "TREATMENT",
)
"""The columns of the ``ANIMAL`` table of a newer Live Mouse Tracker."""

_ANIMAL_STATEMENT: Final = (
    "CREATE TABLE IF NOT EXISTS 'ANIMAL' (  'ID'\tINTEGER NOT NULL PRIMARY KEY "
    "AUTOINCREMENT, 'RFID'\tTEXT, 'GENOTYPE'\tTEXT, 'NAME'\tTEXT )"
)
_NINE_COLUMN_ANIMAL_STATEMENT: Final = (
    "CREATE TABLE IF NOT EXISTS 'ANIMAL' (  'ID'\tINTEGER NOT NULL PRIMARY KEY "
    "AUTOINCREMENT, 'RFID'\tTEXT, 'NAME'\tTEXT, 'GENOTYPE'\tTEXT, 'AGE'\tINTEGER, "
    "'SEX'\tTEXT, 'STRAIN'\tTEXT, 'SETUP'\tTEXT, 'TREATMENT'\tTEXT )"
)
_DETECTION_STATEMENT: Final = (
    "CREATE TABLE IF NOT EXISTS 'DETECTION' ( 'ID' INTEGER NOT NULL PRIMARY KEY "
    "AUTOINCREMENT,  'FRAMENUMBER' INTEGER,  'ANIMALID' INTEGER,  'MASS_X'\tREAL,  "
    "'MASS_Y'\tREAL,  'MASS_Z'\tREAL,  'FRONT_X'\tREAL,  'FRONT_Y'\tREAL,  "
    "'FRONT_Z'\tREAL,  'BACK_X'\tREAL,  'BACK_Y'\tREAL,  'BACK_Z'\tREAL,  "
    "'REARING'\tINTEGER,  'LOOK_UP'\tINTEGER,  'LOOK_DOWN'\tINTEGER,  "
    "'DATA'\tTEXT NOT NULL,  FOREIGN KEY('ANIMALID') REFERENCES ANIMAL('ID'),  "
    "FOREIGN KEY('FRAMENUMBER') REFERENCES FRAME('FRAMENUMBER') )"
)
_FRAME_STATEMENT: Final = (
    "CREATE TABLE IF NOT EXISTS 'FRAME' ('FRAMENUMBER'\tINTEGER NOT NULL UNIQUE,"
    "'TIMESTAMP'\tINTEGER,'NUMPARTICLE'\tINTEGER,'PAUSED'\tINTEGER,"
    "PRIMARY KEY(FRAMENUMBER) )"
)
_EVENT_STATEMENT: Final = (
    "CREATE TABLE IF NOT EXISTS 'EVENT' ('ID'\tINTEGER NOT NULL PRIMARY KEY "
    "AUTOINCREMENT,'NAME'\tTEXT,'DESCRIPTION'\tTEXT,'STARTFRAME'\tINTEGER,"
    "'ENDFRAME'\tINTEGER,'IDANIMALA'\tINTEGER,'IDANIMALB'\tINTEGER,"
    "'IDANIMALC'\tINTEGER,'IDANIMALD'\tINTEGER, METADATA TEXT,"
    "FOREIGN KEY(`IDANIMALA`) REFERENCES ANIMAL(id),"
    "FOREIGN KEY(`IDANIMALB`) REFERENCES ANIMAL(id) )"
)
_LOG_STATEMENT: Final = (
    "CREATE TABLE LOG ( id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, process TEXT, "
    "version TEXT,date TEXT,tmin INTEGER, tmax INTEGER)"
)

_DETECTION_COLUMNS: Final = (
    "ID",
    "FRAMENUMBER",
    "ANIMALID",
    "MASS_X",
    "MASS_Y",
    "MASS_Z",
    "FRONT_X",
    "FRONT_Y",
    "FRONT_Z",
    "BACK_X",
    "BACK_Y",
    "BACK_Z",
    "REARING",
    "LOOK_UP",
    "LOOK_DOWN",
    "DATA",
)

_DATA: Final = "<root/>"
"""What the ``DATA`` column holds. The tracker stores the mask's XML there."""

_BATCH_ROWS: Final = 50_000
_FIRST_TIMESTAMP: Final = 1515570902366


@dataclass(frozen=True, slots=True)
class LmtAnimal:
    """One ``ANIMAL`` row.

    ``extra`` holds the columns that only the nine-column table has, by name.
    """

    id: int
    rfid: str | None
    genotype: str | None = None
    name: str | None = None
    extra: Mapping[str, str | None] = field(default_factory=dict[str, str | None])


@dataclass(frozen=True, slots=True)
class LmtFrame:
    """One ``FRAME`` row."""

    number: int
    timestamp: int | None
    paused: bool = False


@dataclass(frozen=True, slots=True)
class LmtDetection:
    """One ``DETECTION`` row, without its mask.

    ``id`` is the primary key, and ``None`` lets the table assign it.
    """

    frame: int
    animal: int | None
    mass_x: float | None = None
    mass_y: float | None = None
    mass_z: float | None = None
    front_x: float | None = None
    front_y: float | None = None
    front_z: float | None = None
    back_x: float | None = None
    back_y: float | None = None
    back_z: float | None = None
    rearing: bool | None = None
    look_up: bool | None = None
    look_down: bool | None = None
    id: int | None = None


@dataclass(frozen=True, slots=True)
class LmtEvent:
    """One ``EVENT`` row. ``id`` of ``None`` lets the table assign it."""

    name: str
    start: int
    end: int
    a: int | None = None
    b: int | None = None
    c: int | None = None
    d: int | None = None
    id: int | None = None


@dataclass(frozen=True, slots=True)
class LmtLog:
    """One ``LOG`` row, a build of lmt-analysis."""

    process: str
    version: str
    date: str
    tmin: int
    tmax: int


@dataclass(frozen=True, slots=True)
class LmtRecording:
    """The rows of every table of one database."""

    animals: Sequence[LmtAnimal]
    frames: Sequence[LmtFrame]
    detections: Sequence[LmtDetection]
    events: Sequence[LmtEvent]
    log: Sequence[LmtLog]


def _without_column(statement: str, column: str) -> str:
    """The statement of ``DETECTION`` without *column* and its foreign key."""
    removed = re.sub(rf"\s*'{column}'\s+\w+(?: NOT NULL)?,", "", statement, count=1)
    without = re.sub(
        rf",\s*FOREIGN KEY\('{column}'\) REFERENCES [^)]*\)", "", removed, count=1
    )
    if f"'{column}'" in without:
        raise ValueError(f"DETECTION cannot be created without its {column} column")
    return without


def _animal_values(animal: LmtAnimal, columns: Sequence[str]) -> tuple[object, ...]:
    own: dict[str, object] = {
        "ID": animal.id,
        "RFID": animal.rfid,
        "GENOTYPE": animal.genotype,
        "NAME": animal.name,
    }
    return tuple(
        own[column] if column in own else animal.extra.get(column) for column in columns
    )


def _flag(value: bool | None) -> int | None:
    return None if value is None else int(value)


def _detection_values(detection: LmtDetection) -> dict[str, object]:
    return {
        "ID": detection.id,
        "FRAMENUMBER": detection.frame,
        "ANIMALID": detection.animal,
        "MASS_X": detection.mass_x,
        "MASS_Y": detection.mass_y,
        "MASS_Z": detection.mass_z,
        "FRONT_X": detection.front_x,
        "FRONT_Y": detection.front_y,
        "FRONT_Z": detection.front_z,
        "BACK_X": detection.back_x,
        "BACK_Y": detection.back_y,
        "BACK_Z": detection.back_z,
        "REARING": _flag(detection.rearing),
        "LOOK_UP": _flag(detection.look_up),
        "LOOK_DOWN": _flag(detection.look_down),
        "DATA": _DATA,
    }


def write_lmt_sqlite(
    path: Path,
    *,
    animals: Sequence[LmtAnimal],
    frames: Sequence[LmtFrame],
    events: Sequence[LmtEvent],
    detections: Iterable[LmtDetection],
    log: Sequence[LmtLog] | None = None,
    animal_columns: Sequence[str] = DEFAULT_ANIMAL_COLUMNS,
    drop_detection_columns: Sequence[str] = (),
) -> Path:
    """Write a Live Mouse Tracker database and return its path.

    The tables are created with the statements of a database that Live Mouse
    Tracker wrote. Rows go in through ``executemany`` in batches, so *detections*
    may be any iterable, including a generator of millions of rows. The connection writes without a
    journal and is closed on return, so the directory gains no other file.

    Args:
        path: The database file. It must not exist.
        animals: The ``ANIMAL`` rows.
        frames: The ``FRAME`` rows, in the order written.
        events: The ``EVENT`` rows.
        detections: The ``DETECTION`` rows.
        log: The ``LOG`` rows. ``None`` creates no ``LOG`` table.
        animal_columns: The columns of ``ANIMAL``: ``DEFAULT_ANIMAL_COLUMNS`` or
            ``NINE_ANIMAL_COLUMNS``.
        drop_detection_columns: Columns that ``DETECTION`` is created without,
            with the foreign key of each.

    Raises:
        ValueError: *animal_columns* is neither of the two column sets, or
            *drop_detection_columns* names a column ``DETECTION`` does not have.
        FileExistsError: *path* exists.
    """
    animal_statements: dict[tuple[str, ...], str] = {
        DEFAULT_ANIMAL_COLUMNS: _ANIMAL_STATEMENT,
        NINE_ANIMAL_COLUMNS: _NINE_COLUMN_ANIMAL_STATEMENT,
    }
    animal_statement = animal_statements.get(tuple(animal_columns))
    if animal_statement is None:
        raise ValueError(
            f"animal_columns must be {DEFAULT_ANIMAL_COLUMNS} or "
            f"{NINE_ANIMAL_COLUMNS}, got {tuple(animal_columns)}"
        )
    unknown = sorted(set(drop_detection_columns) - set(_DETECTION_COLUMNS))
    if unknown:
        raise ValueError(f"DETECTION has no column {unknown}")
    if path.exists():
        raise FileExistsError(path)

    detection_statement = _DETECTION_STATEMENT
    for column in drop_detection_columns:
        detection_statement = _without_column(detection_statement, column)
    detection_columns = [
        column for column in _DETECTION_COLUMNS if column not in drop_detection_columns
    ]

    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        for statement in (
            animal_statement,
            detection_statement,
            _FRAME_STATEMENT,
            _EVENT_STATEMENT,
        ):
            connection.execute(statement)
        if log is not None:
            connection.execute(_LOG_STATEMENT)

        marks = ", ".join("?" * len(animal_columns))
        connection.executemany(
            f"INSERT INTO ANIMAL ({', '.join(animal_columns)}) VALUES ({marks})",
            [_animal_values(animal, animal_columns) for animal in animals],
        )
        connection.executemany(
            "INSERT INTO FRAME (FRAMENUMBER, TIMESTAMP, NUMPARTICLE, PAUSED) "
            "VALUES (?, ?, ?, ?)",
            [
                (frame.number, frame.timestamp, len(animals), int(frame.paused))
                for frame in frames
            ],
        )
        connection.executemany(
            "INSERT INTO EVENT (ID, NAME, STARTFRAME, ENDFRAME, IDANIMALA, "
            "IDANIMALB, IDANIMALC, IDANIMALD) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(e.id, e.name, e.start, e.end, e.a, e.b, e.c, e.d) for e in events],
        )
        if log is not None:
            connection.executemany(
                "INSERT INTO LOG (process, version, date, tmin, tmax) "
                "VALUES (?, ?, ?, ?, ?)",
                [(x.process, x.version, x.date, x.tmin, x.tmax) for x in log],
            )
        insert_detection = (
            f"INSERT INTO DETECTION ({', '.join(detection_columns)}) "
            f"VALUES ({', '.join('?' * len(detection_columns))})"
        )
        for batch in batched(detections, _BATCH_ROWS):
            rows = [
                tuple(values[column] for column in detection_columns)
                for values in map(_detection_values, batch)
            ]
            connection.executemany(insert_detection, rows)
        connection.commit()
    finally:
        connection.close()
    return path


def _grid_detection(animal: int, number: int) -> LmtDetection:
    """A detection whose positions follow from its animal and frame number."""
    x = 100.0 + 10.0 * animal + 0.25 * number
    y = 200.0 + 5.0 * animal - 0.125 * number
    return LmtDetection(
        frame=number,
        animal=animal,
        mass_x=x,
        mass_y=y,
        mass_z=50.0 + animal,
        front_x=x + 12.0,
        front_y=y - 3.0,
        front_z=30.0 + animal,
        back_x=x - 12.0,
        back_y=y + 3.0,
        back_z=25.0 + animal,
        rearing=(animal + number) % 5 == 0,
        look_up=number % 7 == 0,
        look_down=number % 11 == 0,
    )


def lmt_detection_grid(
    animal_ids: Sequence[int], frame_numbers: Iterable[int]
) -> Iterator[LmtDetection]:
    """One detection per frame number and animal, frame by frame, as a stream.

    The stream holds one detection at a time, so a test writes 30 million rows
    of it without a list.
    """
    for number in frame_numbers:
        for animal in animal_ids:
            yield _grid_detection(animal, number)


def default_lmt_recording() -> LmtRecording:
    """A small recording that has each shape a reader meets.

    Frame numbers run from 101 to 401, with 250 missing, so 300 frames are
    present. Frame 180 is paused and has detections. Timestamps start at
    1515570902366 and step by 33 or 34 ms between consecutive frame numbers,
    so the missing number leaves a double step. Each present frame has one
    detection per animal. Frame 120 also has two detections without an animal,
    which carry the explicit ids 10001 and 10000, inserted in that order and
    after the others.

    No event starts or ends on frame 180 or frame 250.
    """
    animals = (
        LmtAnimal(id=1, rfid="000004395862", genotype="WT", name="mouse one"),
        LmtAnimal(id=2, rfid="000004395870", genotype="KO", name="mouse two"),
        LmtAnimal(id=3, rfid="000004395881", genotype=None, name="mouse three"),
    )
    origin, last, missing, paused = 101, 401, 250, 180
    timestamp = _FIRST_TIMESTAMP
    frames: list[LmtFrame] = []
    for number in range(origin, last + 1):
        if number != missing:
            frames.append(LmtFrame(number, timestamp, paused=number == paused))
        timestamp += 33 if number % 2 == 0 else 34
    present = [frame.number for frame in frames]
    animal_ids = [animal.id for animal in animals]
    detections = [
        *lmt_detection_grid(animal_ids, present),
        replace(_grid_detection(1, 120), animal=None, id=10001),
        replace(_grid_detection(2, 120), animal=None, id=10000),
    ]
    events = (
        LmtEvent("Move isolated", 110, 130, a=1),
        LmtEvent("Contact", 140, 160, a=1, b=2),
        LmtEvent("Contact", 140, 160, a=2, b=1),
        LmtEvent("Group3 make", 200, 230, a=1, b=2, c=3),
        LmtEvent("night", 120, 380),
        LmtEvent("Flickering", 260, 262, a=3),
        LmtEvent("Side by side Contact, opposite way", 270, 290, a=1, b=3),
        LmtEvent("Move in contact", 330, 350, a=2, b=3),
        LmtEvent("Stop isolated", 340, 360, a=2),
    )
    log = (
        LmtLog("Build DataBase Index", "0", "2022-09-08 13:35:55", 0, -1),
        LmtLog("Build Event Detection", "0", "2022-09-08 13:36:40", 0, 1188000),
    )
    return LmtRecording(
        animals=animals,
        frames=frames,
        detections=detections,
        events=events,
        log=log,
    )
