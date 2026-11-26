"""Live Mouse Tracker databases -> tracks, labels and tags, one database per recording.

**The format.** Live Mouse Tracker writes one SQLite database per recording.
``ANIMAL`` lists the mice and their RFIDs, ``FRAME`` the frames and their
timestamps, ``DETECTION`` one row per animal and frame with its body positions,
and ``EVENT`` the behavioral events that lmt-analysis detected. ``LOG`` lists the
builds that lmt-analysis ran on the file. ``DETECTION.DATA`` holds the XML of the
mask, which no reader here selects. ``RFIDEVENT`` and the other tables are not
read.

**The database is only read.** :func:`open_lmt_database` opens it with
``mode=ro&immutable=1`` on every path, so SQLite takes no lock, writes no
``-shm`` file and reads no journal. A raw input does not change under a
conversion. Because ``immutable=1`` ignores a hot rollback journal and a
write-ahead log, a file with either beside it is refused, naming the sibling.
No code here adds an index, a table or a persistent pragma.

**The track converter.** ``lmt_sqlite`` writes one ``lmt_v1`` table per database,
one row per identified detection of a running frame, numbered and timed as below.
``DETECTION`` is read through one cursor in chunks of ``read_chunk_rows`` rows
into preallocated column arrays, and sorted once in memory by frame and id. A
detection whose ``ANIMALID`` is NULL is dropped and counted, or under
``keep_unidentified`` numbered ``-2``, ``-3`` and so on within its frame in
``DETECTION.ID`` order (``rowid`` without that column), which leaves ``-1`` free
for "no individual". A detection on a paused frame is dropped and counted. The
mask in ``DATA`` is not selected. The converter prints its counts to standard
error, which a converter has no other channel for.

**Frames and time.** The frame origin is the smallest ``FRAMENUMBER`` of the
``FRAME`` table, paused or not. A row's frame is its ``FRAMENUMBER`` less the
origin and its time is its ``TIMESTAMP`` less the origin's, in seconds. A number
missing from ``FRAME`` is a gap with no row. :func:`frame_seconds` refuses a
``FRAME`` table that is empty, a NULL ``TIMESTAMP``, and a ``TIMESTAMP`` that
does not increase along ``FRAMENUMBER``, each naming the file and the frame.

**Sequence tags.** :func:`write_lmt_sequence_tags` writes a CSV of each recording's
wall-clock start, genotypes and animal count, for ``mosaic sequence-tags import``.
It reads the databases and writes the CSV alone.

The module reads SQLite through the standard library and unpickles nothing.
"""

from __future__ import annotations

import csv
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Final, override

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field

from mosaic.core.entry import entry_token
from mosaic.core.params import HASH_EXCLUDE, Declared
from mosaic.core.track_converter import (
    EntryHints,
    TrackConverter,
    TrackConvertParams,
    register_track_converter,
)
from mosaic.core.track_library.helpers import norm_hint

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "LmtFrames",
    "LmtSqliteConverter",
    "LmtSqliteParams",
    "frame_origin",
    "frame_seconds",
    "open_lmt_database",
    "write_lmt_sequence_tags",
]

_JOURNAL_SUFFIXES = ("-journal", "-wal")


@dataclass(frozen=True, slots=True, eq=False)
class LmtFrames:
    """The ``FRAME`` table as three aligned arrays, in ascending frame number.

    Attributes:
        numbers: ``FRAMENUMBER``, as stored (the origin is the first).
        seconds: ``(TIMESTAMP - first TIMESTAMP) / 1000``.
        paused: Whether the frame's ``PAUSED`` is true. All false when the table
            has no ``PAUSED`` column.
    """

    numbers: NDArray[np.int64]
    seconds: NDArray[np.float64]
    paused: NDArray[np.bool_]


def open_lmt_database(path: Path) -> sqlite3.Connection:
    """Open a Live Mouse Tracker database for reading, and nothing else.

    The connection is ``mode=ro&immutable=1``. The caller closes it.

    Raises:
        ValueError: A ``-journal`` or ``-wal`` file sits beside the file that *path*
            resolves to.
        sqlite3.OperationalError: *path* is not an openable database.
    """
    resolved = path.resolve()
    for suffix in _JOURNAL_SUFFIXES:
        sibling = resolved.with_name(resolved.name + suffix)
        if sibling.exists():
            raise ValueError(
                f"{path}: the sibling {sibling.name} exists. SQLite ignores a hot "
                "journal under immutable=1, and the file would read without its "
                "last writes. Open it once in SQLite, or copy it, to finish the "
                "journal."
            )
    return sqlite3.connect(f"{resolved.as_uri()}?mode=ro&immutable=1", uri=True)


def _database_file(connection: sqlite3.Connection) -> str:
    """The file the connection reads, for an error message."""
    row = connection.execute("PRAGMA database_list").fetchone()
    return str(row[2]) if row is not None and row[2] else "<database>"


def frame_seconds(connection: sqlite3.Connection) -> LmtFrames:
    """Read the ``FRAME`` table, ordered by ``FRAMENUMBER``.

    Raises:
        ValueError: The table or its ``FRAMENUMBER`` or ``TIMESTAMP`` column is
            missing, the table is empty, a ``TIMESTAMP`` is NULL, or a
            ``TIMESTAMP`` is equal to or below its predecessor's. The message
            names the file and the frame.
    """
    file = _database_file(connection)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(FRAME)")}
    if not columns:
        raise ValueError(f"{file}: the database has no FRAME table")
    for required in ("FRAMENUMBER", "TIMESTAMP"):
        if required not in columns:
            raise ValueError(f"{file}: the FRAME table has no {required} column")
    paused_column = "PAUSED" if "PAUSED" in columns else "0"
    rows = connection.execute(
        f"SELECT FRAMENUMBER, TIMESTAMP, {paused_column} FROM FRAME "
        "ORDER BY FRAMENUMBER"
    ).fetchall()
    if not rows:
        raise ValueError(f"{file}: the FRAME table has no rows")

    numbers = np.array([row[0] for row in rows], dtype=np.int64)
    timestamps = np.array([row[1] for row in rows], dtype=np.float64)
    paused = np.array([bool(row[2]) for row in rows], dtype=np.bool_)

    null = np.flatnonzero(np.isnan(timestamps))
    if null.size:
        raise ValueError(f"{file}: frame {int(numbers[null[0]])} has a NULL TIMESTAMP")
    not_increasing = np.flatnonzero(np.diff(timestamps) <= 0)
    if not_increasing.size:
        later = int(not_increasing[0]) + 1
        raise ValueError(
            f"{file}: the TIMESTAMP of frame {int(numbers[later])} does not exceed "
            f"that of frame {int(numbers[later - 1])}"
        )
    return LmtFrames(
        numbers=numbers,
        seconds=(timestamps - timestamps[0]) / 1000.0,
        paused=paused,
    )


def frame_origin(connection: sqlite3.Connection) -> int:
    """The smallest ``FRAMENUMBER``, paused or not, read as :func:`frame_seconds` does.

    Raises:
        ValueError: As :func:`frame_seconds`.
    """
    return int(frame_seconds(connection).numbers[0])


_SEQUENCE_TAG_COLUMNS: Final = (
    "group",
    "sequence",
    "recording_start",
    "genotypes",
    "n_animals",
)
_GENOTYPE_SEPARATOR: Final = "|"


def _recording_start(connection: sqlite3.Connection) -> str:
    """The origin frame's ``TIMESTAMP`` as UTC ISO 8601 with milliseconds and a ``Z``."""
    origin = frame_origin(connection)
    row = connection.execute(
        "SELECT TIMESTAMP FROM FRAME WHERE FRAMENUMBER = ?", (origin,)
    ).fetchone()
    start = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(
        milliseconds=round(float(row[0]))
    )
    return start.isoformat(timespec="milliseconds").removesuffix("+00:00") + "Z"


def _animal_summary(connection: sqlite3.Connection, file: str) -> tuple[str, int]:
    """The joined distinct genotypes and the animal count of the ``ANIMAL`` table."""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(ANIMAL)")}
    if not columns:
        raise ValueError(f"{file}: the database has no ANIMAL table")
    row = connection.execute("SELECT COUNT(*) FROM ANIMAL").fetchone()
    genotypes: set[str] = set()
    if "GENOTYPE" in columns:
        for (value,) in connection.execute("SELECT DISTINCT GENOTYPE FROM ANIMAL"):
            if value is not None and str(value) != "":
                genotypes.add(str(value))
    return _GENOTYPE_SEPARATOR.join(sorted(genotypes)), int(row[0])


def write_lmt_sequence_tags(ds: Dataset, csv_path: Path) -> int:
    """Write the sequence tags of the dataset's Live Mouse Tracker recordings as a CSV.

    One row per ``lmt_sqlite`` row of ``tracks_raw/index.csv``, in index order, under
    the columns ``group``, ``sequence``, ``recording_start``, ``genotypes`` and
    ``n_animals``. ``recording_start`` is the origin frame's ``TIMESTAMP`` as UTC ISO
    8601 with milliseconds, such as ``2018-01-10T07:55:02.366Z``, and the file records
    no time zone. ``genotypes`` is the sorted distinct non-empty ``ANIMAL.GENOTYPE``
    values joined by ``|``, empty when the table has no such column or every value is
    NULL or empty. ``n_animals`` is the row count of ``ANIMAL``.

    The databases are read through :func:`open_lmt_database`. Only *csv_path* is
    written, and only once every database has been read. The dataset is unchanged.
    Run ``mosaic sequence-tags import`` on the CSV to store the tags.

    Args:
        ds: The dataset whose raw tracks index lists the databases.
        csv_path: The CSV to write.

    Returns:
        The number of rows written.

    Raises:
        ValueError: A database has no ``ANIMAL`` table, or breaks a rule of
            :func:`frame_seconds`, or has a journal beside it. The message names the
            file.
    """
    from mosaic.core.pipeline.tracks_raw_index import read_tracks_raw_index

    records = read_tracks_raw_index(ds.get_root("tracks_raw") / "index.csv")
    rows: list[list[str]] = []
    for record in records:
        if record.get("src_format", "").strip() != "lmt_sqlite":
            continue
        path = ds.resolve_path(record["abs_path"].strip())
        connection = open_lmt_database(path)
        try:
            start = _recording_start(connection)
            genotypes, animals = _animal_summary(connection, str(path))
        finally:
            connection.close()
        rows.append(
            [
                record.get("group", ""),
                record["sequence"],
                start,
                genotypes,
                str(animals),
            ]
        )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_SEQUENCE_TAG_COLUMNS)
        writer.writerows(rows)
    return len(rows)


_KEEP_UNIDENTIFIED_DESCRIPTION = (
    "Keep the detections whose ANIMALID is NULL, under the negative ids -2, -3 "
    "and so on within each frame, in DETECTION.ID order. Off, they are dropped "
    "and counted. On, the ids no longer follow animals."
)
_READ_CHUNK_DESCRIPTION = (
    "Detections fetched per read. A larger chunk reads faster and holds more "
    "of them as Python tuples beside the column arrays. It does not change the "
    "table."
)

_REQUIRED_COLUMNS: Final = (
    "FRAMENUMBER",
    "ANIMALID",
    "MASS_X",
    "MASS_Y",
    "FRONT_X",
    "FRONT_Y",
    "BACK_X",
    "BACK_Y",
)
_OPTIONAL_COLUMNS: Final = (
    "MASS_Z",
    "FRONT_Z",
    "BACK_Z",
    "REARING",
    "LOOK_UP",
    "LOOK_DOWN",
)
_FLOAT_OUTPUTS: Final = (
    ("X", "MASS_X"),
    ("Y", "MASS_Y"),
    ("poseX0", "FRONT_X"),
    ("poseY0", "FRONT_Y"),
    ("poseX1", "BACK_X"),
    ("poseY1", "BACK_Y"),
    ("Z#mass", "MASS_Z"),
    ("Z#front", "FRONT_Z"),
    ("Z#back", "BACK_Z"),
)
_FLAG_OUTPUTS: Final = (
    ("rearing", "REARING"),
    ("look_up", "LOOK_UP"),
    ("look_down", "LOOK_DOWN"),
)
_SELECTED: Final = (
    "FRAMENUMBER",
    "ANIMALID",
    "KEY",
    *(source for _, source in _FLOAT_OUTPUTS),
    *(source for _, source in _FLAG_OUTPUTS),
)
_FLOAT_START: Final = 3
_FLAG_START: Final = _FLOAT_START + len(_FLOAT_OUTPUTS)
_NULL_FLAG: Final = -1


class LmtSqliteParams(TrackConvertParams):
    """What a Live Mouse Tracker conversion chooses.

    Everything else is in the database.
    """

    keep_unidentified: Annotated[bool, Declared(_KEEP_UNIDENTIFIED_DESCRIPTION)] = False
    read_chunk_rows: Annotated[
        int,
        Field(ge=1),
        HASH_EXCLUDE,
        Declared(_READ_CHUNK_DESCRIPTION, unit="rows"),
    ] = 50_000


@dataclass(frozen=True, slots=True, eq=False)
class _Detections:
    """The ``DETECTION`` rows as column arrays, in the order the scan returned them."""

    frame_number: NDArray[np.int64]
    animal: NDArray[np.int64]
    unidentified: NDArray[np.bool_]
    key: NDArray[np.int64]
    floats: NDArray[np.float64]
    flags: NDArray[np.int8]


def _detection_columns(connection: sqlite3.Connection, file: str) -> set[str]:
    """The columns of ``DETECTION``, refusing a table that lacks a required one."""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(DETECTION)")}
    if not columns:
        raise ValueError(f"{file}: the database has no DETECTION table")
    for required in _REQUIRED_COLUMNS:
        if required not in columns:
            raise ValueError(f"{file}: the DETECTION table has no {required} column")
    return columns


def _read_detections(
    connection: sqlite3.Connection, file: str, chunk_rows: int
) -> _Detections:
    """Read ``DETECTION`` through one cursor into preallocated arrays.

    A NULL cell reads as NaN. The id columns are integers and the flags are
    stored as ``-1`` for NULL, so that a chunk is one float block and no column
    is held twice.
    """
    columns = _detection_columns(connection, file)
    key = "ID" if "ID" in columns else "rowid"
    selects = [
        f"{key} AS KEY"
        if name == "KEY"
        else name
        if name in columns
        else f"NULL AS {name}"
        for name in _SELECTED
    ]
    row = connection.execute("SELECT COUNT(*) FROM DETECTION").fetchone()
    total = int(row[0])
    detections = _Detections(
        frame_number=np.empty(total, dtype=np.int64),
        animal=np.empty(total, dtype=np.int64),
        unidentified=np.empty(total, dtype=np.bool_),
        key=np.empty(total, dtype=np.int64),
        floats=np.empty((len(_FLOAT_OUTPUTS), total), dtype=np.float64),
        flags=np.empty((len(_FLAG_OUTPUTS), total), dtype=np.int8),
    )
    cursor = connection.execute(f"SELECT {', '.join(selects)} FROM DETECTION")
    start = 0
    while True:
        rows = cursor.fetchmany(chunk_rows)
        if not rows:
            break
        try:
            block = np.array(rows, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{file}: a DETECTION cell is not a number ({error})"
            ) from error
        stop = start + len(rows)
        missing_frame = np.flatnonzero(np.isnan(block[:, 0]))
        if missing_frame.size:
            raise ValueError(
                f"{file}: the DETECTION row with key {int(block[missing_frame[0], 2])} "
                "has a NULL FRAMENUMBER"
            )
        unidentified = np.isnan(block[:, 1])
        detections.frame_number[start:stop] = block[:, 0]
        detections.unidentified[start:stop] = unidentified
        detections.animal[start:stop] = np.where(unidentified, 0, block[:, 1])
        detections.key[start:stop] = block[:, 2]
        detections.floats[:, start:stop] = block[:, _FLOAT_START:_FLAG_START].T
        flags = block[:, _FLAG_START:]
        detections.flags[:, start:stop] = np.where(
            np.isnan(flags), _NULL_FLAG, flags != 0
        ).T
        start = stop
    if start != total:
        raise ValueError(f"{file}: DETECTION held {start} rows after counting {total}")
    return detections


def _unidentified_ids(
    frame: NDArray[np.int64], key: NDArray[np.int64]
) -> NDArray[np.int64]:
    """``-2, -3, ...`` within each frame, in key order, for rows given in any order.

    Returns the ids in the order of the arguments.
    """
    order = np.lexsort((key, frame))
    ordered = frame[order]
    first_of_frame = np.empty(ordered.size, dtype=np.bool_)
    first_of_frame[:1] = True
    first_of_frame[1:] = ordered[1:] != ordered[:-1]
    positions = np.arange(ordered.size, dtype=np.int64)
    start_of_frame = np.maximum.accumulate(np.where(first_of_frame, positions, 0))
    ids = np.empty(frame.size, dtype=np.int64)
    ids[order] = -2 - (positions - start_of_frame)
    return ids


@register_track_converter
class LmtSqliteConverter(TrackConverter[LmtSqliteParams]):
    """A Live Mouse Tracker database -> an ``lmt_v1`` table, in pixels."""

    src_format = "lmt_sqlite"
    version = "0.1"
    output_schema = "lmt_v1"
    carries_tracklets = True
    enumerable = False
    merges_per_sequence = False
    Params: ClassVar[type[TrackConvertParams]] = LmtSqliteParams

    @override
    def tracklets(self, params: LmtSqliteParams) -> bool | None:
        return not params.keep_unidentified

    @override
    def convert(
        self, path: Path, params: LmtSqliteParams, hints: EntryHints
    ) -> pd.DataFrame:
        group = norm_hint(hints.group) or ""
        sequence = norm_hint(hints.sequence) or self.sequence_from_stem(path.stem)
        connection = open_lmt_database(path)
        try:
            file = str(path)
            frames = frame_seconds(connection)
            detections = _read_detections(connection, file, params.read_chunk_rows)
        finally:
            connection.close()

        placed = np.searchsorted(frames.numbers, detections.frame_number)
        clipped = np.minimum(placed, frames.numbers.size - 1)
        absent = frames.numbers[clipped] != detections.frame_number
        if absent.any():
            first = int(detections.frame_number[absent].min())
            raise ValueError(
                f"{file}: a detection is on frame {first}, which has no FRAME row"
            )

        on_paused = frames.paused[clipped]
        dropped_unidentified = detections.unidentified & ~on_paused
        keep = ~on_paused
        if not params.keep_unidentified:
            keep &= ~detections.unidentified
        total = int(detections.frame_number.size)
        kept = int(keep.sum())
        unidentified_count = (
            0 if params.keep_unidentified else int(dropped_unidentified.sum())
        )
        paused_count = int(on_paused.sum())
        print(
            f"[lmt_sqlite] {entry_token(group, sequence)}: kept {kept} of {total} "
            f"detections; {unidentified_count} unidentified and {paused_count} on "
            "paused frames dropped",
            file=sys.stderr,
        )

        rows = np.flatnonzero(keep)
        frame_number = detections.frame_number[rows]
        animal = detections.animal[rows].copy()
        unidentified = detections.unidentified[rows]
        if unidentified.any():
            animal[unidentified] = _unidentified_ids(
                frame_number[unidentified], detections.key[rows][unidentified]
            )
        order = np.lexsort((animal, frame_number))
        rows = rows[order]
        animal = animal[order]
        frame_number = frame_number[order]

        columns: dict[str, object] = {
            "frame": frame_number - frames.numbers[0],
            "time": frames.seconds[clipped[rows]],
            "id": animal,
            "group": np.full(rows.size, group, dtype=object),
            "sequence": np.full(rows.size, sequence, dtype=object),
        }
        for index, (name, _) in enumerate(_FLOAT_OUTPUTS):
            columns[name] = detections.floats[index][rows]
        for index, (name, _) in enumerate(_FLAG_OUTPUTS):
            code = detections.flags[index][rows]
            columns[name] = pd.arrays.BooleanArray(code == 1, code == _NULL_FLAG)
        return pd.DataFrame(columns, copy=False)
