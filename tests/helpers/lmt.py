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
:func:`lmt_dataset` indexes databases as a dataset's raw files, and
:func:`place_rebuilt_copy` writes the copy that a rebuild of one of them leaves
under the ``rebuild-lmt-events`` root.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import batched
from pathlib import Path
from typing import Final, Literal

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.markers import PhaseMarker, write_phase_marker
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.track_library.lmt_rebuild import REBUILD_KIND, RebuildManifest
from tests.helpers.datasets import make_dataset
from tests.helpers.lmt_analysis import LmtAnalysisInstaller

__all__ = [
    "DEFAULT_ANIMAL_COLUMNS",
    "LMT_REBUILD_COMMIT",
    "LMT_REBUILD_RUN_ID",
    "NINE_ANIMAL_COLUMNS",
    "REBUILD_SPOILS",
    "LmtAnimal",
    "LmtDetection",
    "LmtEvent",
    "LmtFrame",
    "LmtLog",
    "LmtRecording",
    "RebuildSpoil",
    "alter_lmt_sqlite",
    "default_lmt_recording",
    "lmt_dataset",
    "lmt_detection_grid",
    "lmt_source_state",
    "place_rebuilt_copy",
    "rebuilt_lmt_dataset",
    "spoil_rebuilt_copy",
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


def _insert_events(connection: sqlite3.Connection, events: Sequence[LmtEvent]) -> None:
    """Insert *events* into ``EVENT``."""
    connection.executemany(
        "INSERT INTO EVENT (ID, NAME, STARTFRAME, ENDFRAME, IDANIMALA, "
        "IDANIMALB, IDANIMALC, IDANIMALD) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(e.id, e.name, e.start, e.end, e.a, e.b, e.c, e.d) for e in events],
    )


def _insert_log(connection: sqlite3.Connection, log: Sequence[LmtLog]) -> None:
    """Insert *log* into ``LOG``."""
    connection.executemany(
        "INSERT INTO LOG (process, version, date, tmin, tmax) VALUES (?, ?, ?, ?, ?)",
        [(x.process, x.version, x.date, x.tmin, x.tmax) for x in log],
    )


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
        _insert_events(connection, events)
        if log is not None:
            _insert_log(connection, log)
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

    The nine events have the ids 1 to 9 and names of the event vocabulary. No
    event starts or ends on frame 180 or frame 250.
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
        LmtEvent("Move isolated", 110, 130, a=1, id=1),
        LmtEvent("Contact", 140, 160, a=1, b=2, id=2),
        LmtEvent("Contact", 140, 160, a=2, b=1, id=3),
        LmtEvent("Group 3 make", 200, 230, a=1, b=2, c=3, id=4),
        LmtEvent("night", 120, 380, id=5),
        LmtEvent("Flickering", 260, 262, a=3, id=6),
        LmtEvent("Side by side Contact, opposite way", 270, 290, a=1, b=3, id=7),
        LmtEvent("Move in contact", 330, 350, a=2, b=3, id=8),
        LmtEvent("Stop isolated", 340, 360, a=2, id=9),
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


def alter_lmt_sqlite(path: Path, sql: str) -> None:
    """Run the SQL statements of *sql* on the database at *path* and commit them.

    A test breaks a database the writer made, by dropping a table or a column or by
    changing a cell, to see what a reader refuses.
    """
    connection = sqlite3.connect(path)
    try:
        connection.executescript(sql)
        connection.commit()
    finally:
        connection.close()


def lmt_dataset(
    base: Path,
    *,
    recordings: Mapping[str, LmtRecording] | None = None,
    tracks: bool = True,
    labels: bool = False,
    group: str = "",
    labels_group: str | None = None,
    animal_columns: Sequence[str] = DEFAULT_ANIMAL_COLUMNS,
    drop_detection_columns: Sequence[str] = (),
    alter: str = "",
    md5: bool = True,
) -> Dataset:
    """A dataset whose ``raw/`` holds Live Mouse Tracker databases, indexed with md5s.

    Nothing is converted.

    Args:
        base: The dataset's directory.
        recordings: Each database file's stem and rows. Defaults to the one file
            ``recording`` that holds :func:`default_lmt_recording`.
        tracks: Index ``raw/`` as the tracks source.
        labels: Index ``raw/`` as the labels source.
        group: The group of the tracks rows.
        labels_group: The group of the labels rows, ``group`` when ``None``.
        animal_columns: The columns of ``ANIMAL`` in every file.
        drop_detection_columns: Columns that ``DETECTION`` is created without.
        alter: SQL run on each file before it is indexed.
        md5: Whether the scan hashes the files. ``False`` leaves the ``md5`` cell empty.
    """
    ds = make_dataset(base)
    raw = ds.base_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if recordings is None:
        recordings = {"recording": default_lmt_recording()}
    for stem, rows in recordings.items():
        path = write_lmt_sqlite(
            raw / f"{stem}.sqlite",
            animals=rows.animals,
            frames=rows.frames,
            events=rows.events,
            detections=rows.detections,
            log=rows.log,
            animal_columns=animal_columns,
            drop_detection_columns=drop_detection_columns,
        )
        if alter:
            alter_lmt_sqlite(path, alter)
    if tracks:
        _ = ds.index_tracks_raw(
            [raw],
            patterns=["*.sqlite"],
            src_format="lmt_sqlite",
            group_from_path=(lambda _path: group) if group else None,
            compute_md5=md5,
        )
    if labels:
        labels_named = group if labels_group is None else labels_group
        _ = ds.index_labels_raw(
            [raw],
            patterns=["*.sqlite"],
            src_format="lmt_sqlite",
            group_from_path=(lambda _path: labels_named) if labels_named else None,
            compute_md5=md5,
        )
    return ds


def lmt_source_state(ds: Dataset) -> dict[Path, tuple[bytes, int]]:
    """Every raw database's bytes and modification time, to compare after a run."""
    return {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted((ds.base_dir / "raw").glob("*.sqlite*"))
    }


def rebuilt_lmt_dataset(
    base: Path, install_lmt: LmtAnalysisInstaller, *, raises: bool = False
) -> tuple[Dataset, Path, dict[Path, tuple[bytes, int]]]:
    """A dataset whose one entry the rebuild op ran on, the entry's directory and the raw state.

    The raw state is :func:`lmt_source_state` before the run, for the test to compare
    with afterwards. With *raises* the stand-in fails, and the entry's directory holds
    the logs of the failed attempt and nothing else.
    """
    from mosaic.tracking import register_ops

    register_ops()
    checkout = install_lmt(raises=raises)
    ds = lmt_dataset(base)
    before = lmt_source_state(ds)
    params = {"lmt_analysis_commit": checkout.commit}
    if raises:
        with pytest.raises(AllEntriesFailed):
            _ = run_op(ds, REBUILD_KIND, params)
        [run_root] = ds.get_root(REBUILD_KIND).glob("rebuild-lmt-events.*")
        run_id = run_root.name
    else:
        run_id = run_op(ds, REBUILD_KIND, params)
    entry_dir = ds.get_root(REBUILD_KIND) / run_id / "recording"
    assert entry_dir.is_dir()
    return ds, entry_dir, before


def place_rebuilt_copy(
    ds: Dataset,
    run_id: str,
    source: Path,
    *,
    entry_key: str,
    commit: str,
    added_event: LmtEvent | None = None,
    log: LmtLog | None = None,
    alter: str = "",
    finished: bool = True,
) -> Path:
    """Write the entry directory that a rebuild of *source* leaves, and return the copy.

    The directory is ``<rebuild-lmt-events root>/<run_id>/<entry_key>``. It holds the
    copy of *source* under its name, with *added_event* in ``EVENT`` and *log* in
    ``LOG`` as a rebuild adds them and the SQL of *alter* run on it, the manifest
    that names the copy's size and modification time, and with *finished* the
    ``rebuild`` marker.

    Args:
        ds: The dataset that holds the root.
        run_id: The run the copy belongs to.
        source: The raw database.
        entry_key: The name of the entry's directory.
        commit: The lmt-analysis commit the manifest records.
        added_event: An event the copy has and *source* lacks.
        log: A ``LOG`` row the copy has and *source* lacks.
        alter: SQL run on the copy after the rows above, for a change they do
            not make.
        finished: Whether to write the marker.
    """
    entry_dir = ds.get_root(REBUILD_KIND) / run_id / entry_key
    entry_dir.mkdir(parents=True)
    copy = entry_dir / source.name
    _ = shutil.copyfile(source, copy)
    if added_event is not None or log is not None:
        connection = sqlite3.connect(copy)
        try:
            if added_event is not None:
                _insert_events(connection, [added_event])
            if log is not None:
                _insert_log(connection, [log])
            connection.commit()
        finally:
            connection.close()
    if alter:
        alter_lmt_sqlite(copy, alter)
    stat = copy.stat()
    RebuildManifest(
        manifest_format=1,
        run_id=run_id,
        op_version="0.1",
        lmt_analysis_commit=commit,
        builders=("BuildEventStop",),
        source_md5=hashlib.md5(source.read_bytes()).hexdigest(),
        source_name=source.name,
        source_size=source.stat().st_size,
        copy_name=copy.name,
        copy_size=stat.st_size,
        copy_mtime_ns=stat.st_mtime_ns,
        max_frame=0,
        script_max_frame=0,
        python="3.12.3",
        packages={},
        completed_at="2026-11-26T12:00:00+00:00",
    ).write(entry_dir)
    if finished:
        write_phase_marker(entry_dir, PhaseMarker(phase="rebuild", run_id=run_id))
    return copy


LMT_REBUILD_RUN_ID: Final = "rebuild-lmt-events.0.1-0123456789"
"""A well-formed id of a ``rebuild-lmt-events`` run, for the copies a test places."""

LMT_REBUILD_COMMIT: Final = "0123456789abcdef0123456789abcdef01234567"
"""An lmt-analysis commit, for the manifests a test places."""

RebuildSpoil = Literal["no-run", "other-file", "unfinished", "missing", "changed"]
"""How a copy placed by :func:`place_rebuilt_copy` stops being readable."""

REBUILD_SPOILS: Final[tuple[RebuildSpoil, ...]] = (
    "no-run",
    "other-file",
    "unfinished",
    "missing",
    "changed",
)


def spoil_rebuilt_copy(
    ds: Dataset, run_id: str, source: Path, how: RebuildSpoil
) -> None:
    """Make the copy of *source* that :func:`place_rebuilt_copy` placed unreadable.

    Args:
        ds: The dataset that holds the root.
        run_id: The run the copy was placed under.
        source: The raw database.
        how: ``no-run`` removes the run's directory, ``other-file`` replaces its copy
            with the copy of a database with other content, ``unfinished`` replaces
            it with one that has no marker, ``missing`` deletes the copy file, and
            ``changed`` appends a byte to the copy.
    """
    run_dir = ds.get_root(REBUILD_KIND) / run_id
    if how in ("missing", "changed"):
        (copy,) = run_dir.glob(f"*/{source.name}")
        if how == "missing":
            copy.unlink()
            return
        with copy.open("ab") as handle:
            _ = handle.write(b"x")
        return
    shutil.rmtree(run_dir)
    if how == "no-run":
        return
    if how == "unfinished":
        _ = place_rebuilt_copy(
            ds, run_id, source, entry_key="spoiled", commit="0" * 40, finished=False
        )
        return
    recording = default_lmt_recording()
    other = write_lmt_sqlite(
        ds.base_dir / "other.sqlite",
        animals=recording.animals,
        frames=recording.frames,
        events=[],
        detections=[],
    )
    _ = place_rebuilt_copy(ds, run_id, other, entry_key="other", commit="0" * 40)
