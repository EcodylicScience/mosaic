"""The ``lmt_sqlite`` converter and the reader it stands on.

A Live Mouse Tracker database is one recording. The reader opens it with
``mode=ro&immutable=1``, which ignores a rollback journal and a write-ahead log,
so a file that has either beside it is refused. The frame table gives each frame
its number relative to the first one and its time in seconds, and refuses a table
that does not order its frames in time. The converter writes one ``lmt_v1`` table
per database: the identified detections of the running frames, in pixels and
LMT's heights and flags, under the frame numbers and times of the file.
"""

from __future__ import annotations

import csv
import os
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.annotations.sequence_tags import (
    import_sequence_tags,
    list_sequence_tags,
)
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.tracks_index import read_carries_tracklets, read_tracks_index
from mosaic.core.schema import ensure_track_schema
from mosaic.core.track_converter import TRACK_CONVERTERS, EntryHints
from mosaic.core.track_library.lmt import (
    LmtFrames,
    LmtSqliteConverter,
    LmtSqliteParams,
    frame_origin,
    frame_seconds,
    journals_beside,
    open_lmt_database,
    table_columns,
    write_lmt_sequence_tags,
)
from tests.helpers import (
    LMT_REBUILD_COMMIT,
    LMT_REBUILD_RUN_ID,
    DEFAULT_ANIMAL_COLUMNS,
    NINE_ANIMAL_COLUMNS,
    REBUILD_SPOILS,
    LmtDetection,
    LmtFrame,
    LmtRecording,
    RebuildSpoil,
    alter_lmt_sqlite,
    assert_tree_unchanged,
    default_lmt_recording,
    file_digests,
    lmt_dataset,
    make_dataset,
    place_rebuilt_copy,
    spoil_rebuilt_copy,
    tree_snapshot,
    write_lmt_sqlite,
)

_MISSING_FRAME = 250
_PAUSED_FRAME = 180


def _write(
    path: Path,
    recording: LmtRecording,
    *,
    animal_columns: Sequence[str] = DEFAULT_ANIMAL_COLUMNS,
    drop_detection_columns: Sequence[str] = (),
) -> Path:
    return write_lmt_sqlite(
        path,
        animals=recording.animals,
        frames=recording.frames,
        events=recording.events,
        detections=recording.detections,
        log=recording.log,
        animal_columns=animal_columns,
        drop_detection_columns=drop_detection_columns,
    )


def _frames_of(tmp_path: Path, frames: Sequence[LmtFrame]) -> Path:
    recording = replace(default_lmt_recording(), frames=frames, detections=())
    return _write(tmp_path / "frames.sqlite", recording)


def _read_frames(path: Path) -> LmtFrames:
    connection = open_lmt_database(path)
    try:
        return frame_seconds(connection)
    finally:
        connection.close()


def test_frame_seconds_places_each_frame_after_the_first(tmp_path: Path) -> None:
    """Seconds count from the first timestamp, and a missing number has no row."""
    recording = default_lmt_recording()
    path = _write(tmp_path / "recording.sqlite", recording)

    frames = _read_frames(path)

    numbers = [frame.number for frame in recording.frames]
    timestamps = np.array(
        [frame.timestamp for frame in recording.frames], dtype=np.float64
    )
    assert frames.numbers.dtype == np.int64
    assert frames.numbers.tolist() == numbers
    assert _MISSING_FRAME not in frames.numbers
    assert frames.seconds[0] == 0.0
    assert frames.seconds.dtype == np.float64
    np.testing.assert_array_equal(frames.seconds, (timestamps - timestamps[0]) / 1000)
    assert frames.paused.dtype == np.bool_
    assert frames.numbers[frames.paused].tolist() == [_PAUSED_FRAME]


def test_the_missing_frame_number_leaves_a_double_step(tmp_path: Path) -> None:
    """The timestamps step by 33 or 34 ms, and twice that across the gap."""
    frames = _read_frames(_write(tmp_path / "a.sqlite", default_lmt_recording()))

    steps = np.diff(frames.seconds)
    across = steps[frames.numbers[1:] == _MISSING_FRAME + 1]

    assert set(np.round(steps[frames.numbers[1:] != _MISSING_FRAME + 1] * 1000)) == {
        33.0,
        34.0,
    }
    assert np.round(across[0] * 1000) == 67.0


def test_frame_origin_is_the_smallest_frame_number(tmp_path: Path) -> None:
    path = _write(tmp_path / "recording.sqlite", default_lmt_recording())
    connection = open_lmt_database(path)
    try:
        origin = frame_origin(connection)
    finally:
        connection.close()

    assert origin == min(frame.number for frame in default_lmt_recording().frames)


def test_frame_origin_is_the_smallest_number_whatever_the_row_order(
    tmp_path: Path,
) -> None:
    """Rows written out of order read in frame-number order."""
    frames = [LmtFrame(12, 3000), LmtFrame(10, 1000), LmtFrame(11, 2000)]
    path = _frames_of(tmp_path, frames)

    connection = open_lmt_database(path)
    try:
        origin = frame_origin(connection)
        read = frame_seconds(connection)
    finally:
        connection.close()

    assert origin == 10
    assert read.numbers.tolist() == [10, 11, 12]
    assert read.seconds.tolist() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    ("timestamps", "named"),
    [
        pytest.param([1000, None, 3000], 2, id="null"),
        pytest.param([1000, 2000, 2000], 3, id="equal"),
        pytest.param([1000, 3000, 2000], 3, id="decreasing"),
        pytest.param([None, 2000, 3000], 1, id="null-first"),
    ],
)
def test_a_timestamp_out_of_order_names_its_frame(
    tmp_path: Path, timestamps: list[int | None], named: int
) -> None:
    """A NULL timestamp, or one that does not exceed its predecessor, raises."""
    path = _frames_of(
        tmp_path, [LmtFrame(n, t) for n, t in enumerate(timestamps, start=1)]
    )

    with pytest.raises(ValueError, match=rf"frame {named}\b") as raised:
        _read_frames(path)

    assert str(path) in str(raised.value)


def test_an_empty_frame_table_names_the_table(tmp_path: Path) -> None:
    path = _frames_of(tmp_path, [])
    connection = open_lmt_database(path)
    try:
        with pytest.raises(ValueError, match="FRAME"):
            frame_seconds(connection)
        with pytest.raises(ValueError, match="FRAME"):
            frame_origin(connection)
    finally:
        connection.close()


def test_a_frame_table_without_paused_reads_every_frame_as_running(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path / "recording.sqlite", default_lmt_recording())
    alter_lmt_sqlite(path, "ALTER TABLE FRAME DROP COLUMN PAUSED")

    frames = _read_frames(path)

    assert frames.numbers.size == 300
    assert not frames.paused.any()


def test_a_database_without_a_frame_table_names_the_file_and_table(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path / "recording.sqlite", default_lmt_recording())
    alter_lmt_sqlite(path, "DROP TABLE FRAME")

    with pytest.raises(ValueError, match="FRAME") as raised:
        _read_frames(path)

    assert str(path) in str(raised.value)


def test_a_frame_table_without_timestamps_names_the_file_and_column(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path / "recording.sqlite", default_lmt_recording())
    alter_lmt_sqlite(path, "ALTER TABLE FRAME DROP COLUMN TIMESTAMP")

    with pytest.raises(ValueError, match="TIMESTAMP") as raised:
        _read_frames(path)

    assert str(path) in str(raised.value)


def test_a_journal_beside_the_target_of_a_symlink_is_refused(tmp_path: Path) -> None:
    """SQLite opens the resolved file, so its journal is the one that counts."""
    (tmp_path / "data").mkdir()
    target = _write(tmp_path / "data" / "recording.sqlite", default_lmt_recording())
    link = tmp_path / "link.sqlite"
    link.symlink_to(target)
    (tmp_path / "data" / "recording.sqlite-journal").write_bytes(b"")

    with pytest.raises(ValueError, match="recording.sqlite-journal"):
        open_lmt_database(link)


def test_journals_beside_names_each_sibling_of_the_resolved_file(
    tmp_path: Path,
) -> None:
    (tmp_path / "data").mkdir()
    target = _write(tmp_path / "data" / "recording.sqlite", default_lmt_recording())
    link = tmp_path / "link.sqlite"
    link.symlink_to(target)
    assert journals_beside(link) == []
    journal = target.with_name("recording.sqlite-journal")
    wal = target.with_name("recording.sqlite-wal")
    journal.write_bytes(b"")
    wal.write_bytes(b"")

    assert journals_beside(link) == [journal.resolve(), wal.resolve()]
    assert journals_beside(target) == [journal.resolve(), wal.resolve()]


def test_the_writer_refuses_to_drop_the_primary_key(tmp_path: Path) -> None:
    recording = default_lmt_recording()

    with pytest.raises(ValueError, match="ID"):
        _write(
            tmp_path / "recording.sqlite",
            recording,
            drop_detection_columns=("ID",),
        )


@pytest.mark.parametrize("suffix", ["-journal", "-wal"])
def test_a_file_with_a_journal_sibling_is_refused(tmp_path: Path, suffix: str) -> None:
    """Under ``immutable=1`` SQLite would read the file without that sibling."""
    path = _write(tmp_path / "recording.sqlite", default_lmt_recording())
    sibling = tmp_path / f"recording.sqlite{suffix}"
    sibling.write_bytes(b"")

    with pytest.raises(ValueError, match=sibling.name) as raised:
        open_lmt_database(path)

    assert "immutable" in str(raised.value)


def test_reading_a_database_leaves_the_file_and_its_directory_unchanged(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path / "recording.sqlite", default_lmt_recording())
    before = tree_snapshot(tmp_path)
    digests = file_digests(tmp_path)
    mtime_ns = path.stat().st_mtime_ns

    connection = open_lmt_database(path)
    try:
        for table in ("DETECTION", "FRAME", "EVENT"):
            assert connection.execute(f"SELECT * FROM {table}").fetchall()
        frame_seconds(connection)
    finally:
        connection.close()

    assert path.stat().st_mtime_ns == mtime_ns
    assert file_digests(tmp_path) == digests
    assert_tree_unchanged(tmp_path, before)


def test_the_connection_refuses_a_write(tmp_path: Path) -> None:
    path = _write(tmp_path / "recording.sqlite", default_lmt_recording())
    connection = open_lmt_database(path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("CREATE INDEX probe ON DETECTION (FRAMENUMBER)")
    finally:
        connection.close()


def test_the_writer_leaves_one_file(tmp_path: Path) -> None:
    _write(tmp_path / "recording.sqlite", default_lmt_recording())

    assert [path.name for path in tmp_path.iterdir()] == ["recording.sqlite"]


def test_the_default_recording_has_the_shapes_a_reader_meets(tmp_path: Path) -> None:
    path = _write(tmp_path / "recording.sqlite", default_lmt_recording())
    connection = open_lmt_database(path)
    try:
        rfids = [
            r[0] for r in connection.execute("SELECT RFID FROM ANIMAL ORDER BY ID")
        ]
        unidentified = connection.execute(
            "SELECT ID FROM DETECTION WHERE ANIMALID IS NULL"
        ).fetchall()
        paused_detections = connection.execute(
            "SELECT COUNT(*) FROM DETECTION WHERE FRAMENUMBER = ?", (_PAUSED_FRAME,)
        ).fetchone()
        night = connection.execute(
            "SELECT IDANIMALA FROM EVENT WHERE NAME = 'night'"
        ).fetchone()
        builds = connection.execute("SELECT COUNT(*) FROM LOG").fetchone()
    finally:
        connection.close()

    assert rfids[0] == "000004395862"
    assert all(rfid.startswith("0000") for rfid in rfids)
    assert unidentified == [(10000,), (10001,)]
    assert paused_detections == (3,)
    assert night == (None,)
    assert builds == (2,)


def test_nine_animal_columns_write_the_nine_column_table(tmp_path: Path) -> None:
    recording = default_lmt_recording()
    animals = [
        replace(animal, extra={"AGE": "12", "SEX": "F", "TREATMENT": None})
        for animal in recording.animals
    ]
    path = _write(
        tmp_path / "nine.sqlite",
        replace(recording, animals=animals),
        animal_columns=NINE_ANIMAL_COLUMNS,
    )

    connection = open_lmt_database(path)
    try:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(ANIMAL)")]
        first = connection.execute(
            "SELECT RFID, NAME, GENOTYPE, AGE, SEX, STRAIN, TREATMENT "
            "FROM ANIMAL ORDER BY ID"
        ).fetchone()
    finally:
        connection.close()

    assert columns == list(NINE_ANIMAL_COLUMNS)
    assert first == ("000004395862", "mouse one", "WT", 12, "F", None, None)


def test_a_dropped_detection_column_leaves_the_table_without_it(
    tmp_path: Path,
) -> None:
    recording = default_lmt_recording()
    path = _write(
        tmp_path / "dropped.sqlite",
        replace(recording, detections=recording.detections[:5]),
        drop_detection_columns=("LOOK_UP", "MASS_Z"),
    )

    connection = open_lmt_database(path)
    try:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(DETECTION)")]
        count = connection.execute("SELECT COUNT(*) FROM DETECTION").fetchone()
    finally:
        connection.close()

    assert "LOOK_UP" not in columns
    assert "MASS_Z" not in columns
    assert "LOOK_DOWN" in columns
    assert count == (5,)


def test_a_database_without_a_log_has_no_log_table(tmp_path: Path) -> None:
    recording = default_lmt_recording()
    path = write_lmt_sqlite(
        tmp_path / "nolog.sqlite",
        animals=recording.animals,
        frames=recording.frames,
        events=recording.events,
        detections=iter(recording.detections),
    )

    connection = open_lmt_database(path)
    try:
        tables = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master")
        }
    finally:
        connection.close()

    assert "LOG" not in tables
    assert {"ANIMAL", "DETECTION", "FRAME", "EVENT"} <= tables


# --- the converter ---------------------------------------------------------------

_FIRST_FRAME = 101
_COLUMNS = [
    "frame",
    "time",
    "id",
    "group",
    "sequence",
    "X",
    "Y",
    "poseX0",
    "poseY0",
    "poseX1",
    "poseY1",
    "Z#mass",
    "Z#front",
    "Z#back",
    "rearing",
    "look_up",
    "look_down",
]
_DTYPES = {
    "frame": "int64",
    "time": "float64",
    "id": "int64",
    "X": "float64",
    "Y": "float64",
    "poseX0": "float64",
    "poseY0": "float64",
    "poseX1": "float64",
    "poseY1": "float64",
    "Z#mass": "float64",
    "Z#front": "float64",
    "Z#back": "float64",
    "rearing": "boolean",
    "look_up": "boolean",
    "look_down": "boolean",
}
_TEXT = ("group", "sequence")


def _convert(
    path: Path, params: LmtSqliteParams | None = None, hints: EntryHints | None = None
) -> pd.DataFrame:
    return LmtSqliteConverter().convert(
        path, params or LmtSqliteParams(), hints or EntryHints()
    )


def _recording_file(tmp_path: Path, recording: LmtRecording | None = None) -> Path:
    return _write(tmp_path / "recording.sqlite", recording or default_lmt_recording())


def _expected_rows(recording: LmtRecording) -> pd.DataFrame:
    """The identified detections of the running frames, from the records.

    Returns the frame, id, the nine floats and the three flags, sorted by frame
    and id, with the time of each row.
    """
    timestamps = {frame.number: frame.timestamp for frame in recording.frames}
    paused = {frame.number for frame in recording.frames if frame.paused}
    first = min(frame.number for frame in recording.frames)
    origin = timestamps[first]
    assert origin is not None
    rows: list[dict[str, object]] = []
    for detection in recording.detections:
        if detection.animal is None or detection.frame in paused:
            continue
        timestamp = timestamps[detection.frame]
        assert timestamp is not None
        rows.append(
            {
                "frame": detection.frame - first,
                "time": (timestamp - origin) / 1000,
                "id": detection.animal,
                "X": detection.mass_x,
                "Y": detection.mass_y,
                "poseX0": detection.front_x,
                "poseY0": detection.front_y,
                "poseX1": detection.back_x,
                "poseY1": detection.back_y,
                "Z#mass": detection.mass_z,
                "Z#front": detection.front_z,
                "Z#back": detection.back_z,
                "rearing": detection.rearing,
                "look_up": detection.look_up,
                "look_down": detection.look_down,
            }
        )
    return pd.DataFrame(rows).sort_values(["frame", "id"], ignore_index=True)


def _assert_columns_and_dtypes(table: pd.DataFrame) -> None:
    assert list(table.columns) == _COLUMNS
    for name, dtype in _DTYPES.items():
        assert str(table[name].dtype) == dtype, name
    for name in _TEXT:
        assert pd.api.types.is_string_dtype(table[name].dtype), name


def test_the_converter_declares_its_contract() -> None:
    converter = LmtSqliteConverter()

    assert TRACK_CONVERTERS["lmt_sqlite"] is LmtSqliteConverter
    assert converter.version == "0.1"
    assert converter.output_schema == "lmt_v1"
    assert converter.carries_tracklets is True
    assert converter.enumerable is False
    assert converter.merges_per_sequence is False
    assert converter.tracklets(LmtSqliteParams()) is True
    assert converter.tracklets(LmtSqliteParams(keep_unidentified=True)) is False


def test_the_columns_and_dtypes_follow_the_declared_table(tmp_path: Path) -> None:
    table = _convert(_recording_file(tmp_path))

    _assert_columns_and_dtypes(table)
    for forbidden in ("fps", "frame_rate", "FRAME"):
        assert forbidden not in table.columns


def test_the_table_answers_to_lmt_v1(tmp_path: Path) -> None:
    table = _convert(_recording_file(tmp_path))

    _, report = ensure_track_schema(table, "lmt_v1", strict=True)

    assert report["missing_required"] == []


def test_positions_heights_and_flags_equal_the_stored_values(tmp_path: Path) -> None:
    recording = default_lmt_recording()
    table = _convert(_recording_file(tmp_path, recording))
    expected = _expected_rows(recording)

    assert len(table) == len(expected)
    for name in (
        "frame",
        "id",
        "X",
        "Y",
        "poseX0",
        "poseY0",
        "poseX1",
        "poseY1",
        "Z#mass",
        "Z#front",
        "Z#back",
    ):
        np.testing.assert_array_equal(
            table[name].to_numpy(), expected[name].to_numpy(), err_msg=name
        )
    for name in ("rearing", "look_up", "look_down"):
        assert table[name].tolist() == expected[name].tolist(), name


def test_time_is_the_timestamp_less_the_first_in_seconds(tmp_path: Path) -> None:
    recording = default_lmt_recording()
    table = _convert(_recording_file(tmp_path, recording))
    timestamps = {
        frame.number: frame.timestamp
        for frame in recording.frames
        if frame.timestamp is not None
    }
    first = timestamps[_FIRST_FRAME]

    expected = [
        (timestamps[int(frame) + _FIRST_FRAME] - first) / 1000
        for frame in table["frame"]
    ]

    assert table["time"].tolist() == expected
    assert table["time"].iloc[0] == 0.0


def test_the_missing_number_has_no_row_and_the_later_frames_keep_their_numbers(
    tmp_path: Path,
) -> None:
    table = _convert(_recording_file(tmp_path))
    frames = set(table["frame"])

    assert _MISSING_FRAME - _FIRST_FRAME not in frames
    assert _MISSING_FRAME + 1 - _FIRST_FRAME in frames
    assert table["frame"].max() == 401 - _FIRST_FRAME


def test_a_paused_frame_and_an_unidentified_detection_leave_no_row(
    tmp_path: Path,
) -> None:
    table = _convert(_recording_file(tmp_path))

    assert _PAUSED_FRAME - _FIRST_FRAME not in set(table["frame"])
    assert (table["id"] > 0).all()
    assert len(table) == 300 * 3 - 3


def test_the_stderr_line_counts_what_was_kept_and_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _ = _convert(_recording_file(tmp_path))

    assert capsys.readouterr().err.splitlines() == [
        "[lmt_sqlite] recording: kept 897 of 902 detections; "
        "2 unidentified and 3 on paused frames dropped"
    ]


def test_the_hints_name_the_group_and_sequence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    table = _convert(
        _recording_file(tmp_path), hints=EntryHints(group="cage", sequence="night1")
    )

    assert set(table["group"]) == {"cage"}
    assert set(table["sequence"]) == {"night1"}
    assert "[lmt_sqlite] cage:night1: kept" in capsys.readouterr().err


def test_the_group_and_sequence_default_to_nothing_and_the_stem(
    tmp_path: Path,
) -> None:
    table = _convert(_recording_file(tmp_path))

    assert set(table["group"]) == {""}
    assert set(table["sequence"]) == {"recording"}


def test_a_null_cell_is_nan_in_a_float_column_and_na_in_a_flag(tmp_path: Path) -> None:
    recording = default_lmt_recording()
    nulled = replace(
        recording.detections[0], mass_z=None, front_z=None, rearing=None, look_up=None
    )
    table = _convert(
        _recording_file(
            tmp_path, replace(recording, detections=[nulled, *recording.detections[1:]])
        )
    )

    row = table[(table["frame"] == 0) & (table["id"] == nulled.animal)].iloc[0]
    assert np.isnan(row["Z#mass"])
    assert np.isnan(row["Z#front"])
    assert pd.isna(row["rearing"])
    assert pd.isna(row["look_up"])
    assert not np.isnan(row["Z#back"])
    assert not pd.isna(row["look_down"])


def test_a_detection_table_without_the_look_columns_gives_them_as_na(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "recording.sqlite",
        default_lmt_recording(),
        drop_detection_columns=("LOOK_UP", "LOOK_DOWN"),
    )

    table = _convert(path)

    _assert_columns_and_dtypes(table)
    assert table["look_up"].isna().all()
    assert table["look_down"].isna().all()
    assert not table["rearing"].isna().all()


def test_a_detection_table_without_the_heights_gives_them_as_nan(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "recording.sqlite",
        default_lmt_recording(),
        drop_detection_columns=("MASS_Z", "FRONT_Z", "BACK_Z", "REARING"),
    )

    table = _convert(path)

    _assert_columns_and_dtypes(table)
    for name in ("Z#mass", "Z#front", "Z#back"):
        assert table[name].isna().all(), name
    assert table["rearing"].isna().all()


def test_the_chunk_size_does_not_change_the_table(tmp_path: Path) -> None:
    path = _recording_file(tmp_path)

    whole = _convert(path)
    chunked = _convert(path, LmtSqliteParams(read_chunk_rows=7))

    pd.testing.assert_frame_equal(whole, chunked)


def test_unidentified_detections_take_negative_ids_in_key_order(
    tmp_path: Path,
) -> None:
    """Frame 120's two were inserted as ID 10001 and then ID 10000."""
    path = _recording_file(tmp_path)

    table = _convert(path, LmtSqliteParams(keep_unidentified=True))

    frame = table[table["frame"] == 120 - _FIRST_FRAME]
    assert frame["id"].tolist() == [-3, -2, 1, 2, 3]
    by_id = frame.set_index("id")
    grid_x = {10000: 100.0 + 20.0 + 0.25 * 120, 10001: 100.0 + 10.0 + 0.25 * 120}
    assert by_id.loc[-2, "X"] == grid_x[10000]
    assert by_id.loc[-3, "X"] == grid_x[10001]
    assert len(table) == 300 * 3 - 3 + 2
    _assert_columns_and_dtypes(table)


def test_unidentified_ids_restart_at_minus_two_in_each_frame(tmp_path: Path) -> None:
    recording = default_lmt_recording()
    extra = replace(recording.detections[0], animal=None, frame=130, id=None)
    path = _recording_file(
        tmp_path, replace(recording, detections=[*recording.detections, extra])
    )

    table = _convert(path, LmtSqliteParams(keep_unidentified=True))

    frame = table[table["frame"] == 130 - _FIRST_FRAME]
    assert frame["id"].tolist() == [-2, 1, 2, 3]


def test_the_stderr_line_under_keep_unidentified_counts_none_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _ = _convert(_recording_file(tmp_path), LmtSqliteParams(keep_unidentified=True))

    assert capsys.readouterr().err.splitlines() == [
        "[lmt_sqlite] recording: kept 899 of 902 detections; "
        "0 unidentified and 3 on paused frames dropped"
    ]


def test_an_unidentified_detection_on_a_paused_frame_counts_as_paused_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recording = _with_detection(_grid_like(_PAUSED_FRAME, None))
    path = _recording_file(tmp_path, recording)

    table = _convert(path)

    assert capsys.readouterr().err.splitlines() == [
        "[lmt_sqlite] recording: kept 897 of 903 detections; "
        "2 unidentified and 4 on paused frames dropped"
    ]
    assert _PAUSED_FRAME - _FIRST_FRAME not in set(table["frame"])


def test_keep_unidentified_gives_a_paused_frame_no_row_of_any_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recording = _with_detection(_grid_like(_PAUSED_FRAME, None))
    path = _recording_file(tmp_path, recording)

    table = _convert(path, LmtSqliteParams(keep_unidentified=True))

    assert capsys.readouterr().err.splitlines() == [
        "[lmt_sqlite] recording: kept 899 of 903 detections; "
        "0 unidentified and 4 on paused frames dropped"
    ]
    on_paused = table[table["frame"] == _PAUSED_FRAME - _FIRST_FRAME]
    assert on_paused.empty
    assert not (on_paused["id"] < 0).any()


def test_a_null_frame_number_names_the_file_and_the_row_key(tmp_path: Path) -> None:
    path = _recording_file(tmp_path)
    alter_lmt_sqlite(path, "UPDATE DETECTION SET FRAMENUMBER = NULL WHERE ID = 5")

    with pytest.raises(ValueError, match="NULL FRAMENUMBER") as raised:
        _convert(path)

    assert str(path) in str(raised.value)
    assert "key 5" in str(raised.value)


def test_a_non_numeric_cell_names_the_file(tmp_path: Path) -> None:
    path = _recording_file(tmp_path)
    alter_lmt_sqlite(path, "UPDATE DETECTION SET MASS_X = 'left' WHERE ID = 7")

    with pytest.raises(ValueError, match="not a number") as raised:
        _convert(path)

    assert str(path) in str(raised.value)


# --- through a dataset -----------------------------------------------------------


def test_a_dataset_converts_the_database_to_an_lmt_v1_table(tmp_path: Path) -> None:
    ds = lmt_dataset(tmp_path / "ds")

    outcome = ds.convert_all_tracks()

    assert (outcome.converted, outcome.failed) == (1, 0)
    (row,) = (row for _, row in read_tracks_index(ds).iterrows())
    assert row["std_format"] == "lmt_v1"
    assert read_carries_tracklets(row) is True
    written = pd.read_parquet(ds.resolve_path(str(row["abs_path"])))
    _assert_columns_and_dtypes(written)
    pd.testing.assert_frame_equal(
        written, _convert(ds.base_dir / "raw" / "recording.sqlite"), check_dtype=False
    )
    assert int(row["n_rows"]) == len(written)


def test_keep_unidentified_changes_the_variant_and_records_no_tracklets(
    tmp_path: Path,
) -> None:
    ds = lmt_dataset(tmp_path / "ds")
    default = ds.track_conversion_variant("lmt_sqlite")
    keeping = ds.track_conversion_variant("lmt_sqlite", {"keep_unidentified": True})

    outcome = ds.convert_all_tracks({"keep_unidentified": True})

    assert keeping != default
    assert outcome.variants == (keeping,)
    (row,) = (row for _, row in read_tracks_index(ds).iterrows())
    assert row["run_id"] == keeping
    assert read_carries_tracklets(row) is False


def test_the_default_chunk_holds_50000_rows() -> None:
    assert LmtSqliteParams().read_chunk_rows == 50_000


def test_the_chunk_size_keeps_the_variant(tmp_path: Path) -> None:
    ds = lmt_dataset(tmp_path / "ds")

    default = ds.track_conversion_variant("lmt_sqlite")
    chunked = ds.track_conversion_variant("lmt_sqlite", {"read_chunk_rows": 7})
    outcome = ds.convert_all_tracks({"read_chunk_rows": 7})

    assert chunked == default
    assert outcome.variants == (default,)


def test_conversion_leaves_the_database_and_its_directory_unchanged(
    tmp_path: Path,
) -> None:
    ds = lmt_dataset(tmp_path / "ds")
    raw = ds.base_dir / "raw"
    path = raw / "recording.sqlite"
    before = tree_snapshot(raw)
    digests = file_digests(raw)
    mtime_ns = path.stat().st_mtime_ns

    _ = _convert(path)
    outcome = ds.convert_all_tracks()

    assert outcome.failed == 0
    assert path.stat().st_mtime_ns == mtime_ns
    assert file_digests(raw) == digests
    assert_tree_unchanged(raw, before)


def _failure_text(ds: Dataset, params: dict[str, object] | None = None) -> str:
    """Convert and return the one failure's text, asserting the entry failed."""
    failures: list[str] = []
    outcome = ds.convert_all_tracks(
        params, on_failure=lambda _path, error: failures.append(str(error))
    )
    assert (outcome.converted, outcome.failed) == (0, 1)
    (text,) = failures
    return text


def _grid_like(frame: int, animal: int | None) -> LmtDetection:
    return LmtDetection(
        frame=frame,
        animal=animal,
        mass_x=1.0,
        mass_y=2.0,
        front_x=3.0,
        front_y=4.0,
        back_x=5.0,
        back_y=6.0,
    )


def _with_detection(extra: LmtDetection) -> LmtRecording:
    recording = default_lmt_recording()
    return replace(recording, detections=[*recording.detections, extra])


def _with_timestamps(changes: dict[int, int | None]) -> LmtRecording:
    recording = default_lmt_recording()
    frames = [
        replace(frame, timestamp=changes.get(frame.number, frame.timestamp))
        for frame in recording.frames
    ]
    return replace(recording, frames=frames)


def test_a_repeated_frame_and_id_fails_the_entry_at_the_write_path(
    tmp_path: Path,
) -> None:
    ds = lmt_dataset(
        tmp_path / "ds", recordings={"recording": _with_detection(_grid_like(130, 1))}
    )

    text = _failure_text(ds)

    assert "appears in more than one row" in text
    assert "frame 29 of id 1" in text


def test_a_detection_on_a_frame_without_a_frame_row_names_the_frame(
    tmp_path: Path,
) -> None:
    ds = lmt_dataset(
        tmp_path / "ds", recordings={"recording": _with_detection(_grid_like(250, 1))}
    )

    text = _failure_text(ds)

    assert "frame 250" in text
    assert "recording.sqlite" in text


def test_a_null_timestamp_fails_the_entry_naming_the_frame(tmp_path: Path) -> None:
    ds = lmt_dataset(
        tmp_path / "ds", recordings={"recording": _with_timestamps({150: None})}
    )

    text = _failure_text(ds)

    assert "frame 150" in text
    assert "NULL TIMESTAMP" in text


def test_an_equal_timestamp_fails_the_entry_naming_the_frame(tmp_path: Path) -> None:
    stamps = {frame.number: frame.timestamp for frame in default_lmt_recording().frames}
    ds = lmt_dataset(
        tmp_path / "ds", recordings={"recording": _with_timestamps({151: stamps[150]})}
    )

    text = _failure_text(ds)

    assert "frame 151" in text


def test_a_missing_required_column_fails_the_entry_naming_the_column(
    tmp_path: Path,
) -> None:
    ds = lmt_dataset(tmp_path / "ds", drop_detection_columns=("BACK_Y",))

    text = _failure_text(ds)

    assert "BACK_Y" in text
    assert "recording.sqlite" in text


@pytest.mark.parametrize("suffix", ["-journal", "-wal"])
def test_a_journal_sibling_fails_the_entry_naming_the_sibling(
    tmp_path: Path, suffix: str
) -> None:
    ds = lmt_dataset(tmp_path / "ds")
    sibling = ds.base_dir / "raw" / f"recording.sqlite{suffix}"
    sibling.write_bytes(b"")

    text = _failure_text(ds)

    assert sibling.name in text


# --- sequence tags ---------------------------------------------------------------

_TAG_HEADER = ["group", "sequence", "recording_start", "genotypes", "n_animals"]
_START = "2018-01-10T07:55:02.366Z"


def _without_genotypes(recording: LmtRecording) -> LmtRecording:
    return replace(
        recording,
        animals=[replace(animal, genotype=None) for animal in recording.animals],
    )


def _two_recording_dataset(base: Path) -> Dataset:
    """A dataset indexing ``raw/with.sqlite`` and ``raw/without.sqlite``."""
    ds = make_dataset(base)
    raw = ds.base_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    recording = default_lmt_recording()
    _ = _write(raw / "with.sqlite", recording)
    _ = _write(raw / "without.sqlite", _without_genotypes(recording))
    _ = ds.index_tracks_raw([raw], patterns=["*.sqlite"], src_format="lmt_sqlite")
    return ds


def _read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as handle:
        return list(csv.reader(handle))


def test_sequence_tags_write_one_row_per_recording(tmp_path: Path) -> None:
    ds = _two_recording_dataset(tmp_path / "ds")
    csv_path = tmp_path / "tags.csv"

    count = write_lmt_sequence_tags(ds, csv_path)

    assert count == 2
    assert _read_rows(csv_path) == [
        _TAG_HEADER,
        ["", "with", _START, "KO|WT", "3"],
        ["", "without", _START, "", "3"],
    ]


def test_the_genotypes_of_a_table_without_the_column_are_empty(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    raw = ds.base_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = _write(raw / "bare.sqlite", default_lmt_recording())
    alter_lmt_sqlite(path, "ALTER TABLE ANIMAL DROP COLUMN GENOTYPE")
    _ = ds.index_tracks_raw([raw], patterns=["*.sqlite"], src_format="lmt_sqlite")
    csv_path = tmp_path / "tags.csv"

    _ = write_lmt_sequence_tags(ds, csv_path)

    assert _read_rows(csv_path)[1] == ["", "bare", _START, "", "3"]


def test_sequence_tags_import_as_three_tags(tmp_path: Path) -> None:
    ds = _two_recording_dataset(tmp_path / "ds")
    assert ds.convert_all_tracks().failed == 0
    csv_path = tmp_path / "tags.csv"
    _ = write_lmt_sequence_tags(ds, csv_path)

    result = import_sequence_tags(ds, csv_path)

    assert result.rows_matched == 2
    assert result.findings == ()
    listing = {tag.name: tag for tag in list_sequence_tags(ds).tags}
    assert sorted(listing) == ["genotypes", "n_animals", "recording_start"]
    assert {
        (value.value, value.entries) for value in listing["recording_start"].values
    } == {(_START, 2)}
    assert {(value.value, value.entries) for value in listing["genotypes"].values} == {
        ("KO|WT", 1)
    }
    assert {(value.value, value.entries) for value in listing["n_animals"].values} == {
        (3, 2)
    }


def test_writing_sequence_tags_leaves_the_dataset_and_the_databases_unchanged(
    tmp_path: Path,
) -> None:
    ds = _two_recording_dataset(tmp_path / "ds")
    raw = ds.base_dir / "raw"
    sources = sorted(raw.glob("*.sqlite"))
    mtimes = [path.stat().st_mtime_ns for path in sources]
    dataset_digests = file_digests(ds.base_dir)
    dataset_before = tree_snapshot(ds.base_dir)

    _ = write_lmt_sequence_tags(ds, tmp_path / "tags.csv")

    assert file_digests(ds.base_dir) == dataset_digests
    assert_tree_unchanged(ds.base_dir, dataset_before)
    assert [path.stat().st_mtime_ns for path in sources] == mtimes


def test_a_database_without_an_animal_table_fails_the_sequence_tags_and_writes_no_csv(
    tmp_path: Path,
) -> None:
    ds = _two_recording_dataset(tmp_path / "ds")
    broken = ds.base_dir / "raw" / "without.sqlite"
    alter_lmt_sqlite(broken, "DROP TABLE ANIMAL")
    csv_path = tmp_path / "tags.csv"

    with pytest.raises(ValueError, match="no ANIMAL table") as raised:
        _ = write_lmt_sequence_tags(ds, csv_path)

    assert str(broken) in str(raised.value)
    assert not csv_path.exists()


def test_table_columns_names_the_columns_of_the_table(tmp_path: Path) -> None:
    path = _recording_file(tmp_path)
    connection = open_lmt_database(path)
    try:
        columns = table_columns(connection, "ANIMAL", str(path))
    finally:
        connection.close()

    assert columns == set(DEFAULT_ANIMAL_COLUMNS)


def test_table_columns_refuses_a_missing_table_naming_the_file(tmp_path: Path) -> None:
    path = _recording_file(tmp_path)
    connection = open_lmt_database(path)
    try:
        with pytest.raises(ValueError, match="no NOPE table") as raised:
            _ = table_columns(connection, "NOPE", str(path))
    finally:
        connection.close()

    assert str(path) in str(raised.value)


# --- a rebuilt copy ----------------------------------------------------------------

_ADDED_DETECTION = (
    "INSERT INTO DETECTION (FRAMENUMBER, ANIMALID, MASS_X, MASS_Y, DATA) "
    "VALUES (130, 4, 1.0, 2.0, '<root/>');"
)


def _rebuilt(tmp_path: Path) -> tuple[Dataset, Path, Path]:
    """A dataset and its database, with a rebuilt copy that has one more animal."""
    ds = lmt_dataset(tmp_path / "ds")
    source = ds.base_dir / "raw" / "recording.sqlite"
    copy = place_rebuilt_copy(
        ds,
        LMT_REBUILD_RUN_ID,
        source,
        entry_key="recording",
        commit=LMT_REBUILD_COMMIT,
        alter=_ADDED_DETECTION,
    )
    return ds, source, copy


def _tracks_of(ds: Dataset) -> pd.DataFrame:
    (path,) = read_tracks_index(ds)["abs_path"].tolist()
    return pd.read_parquet(ds.resolve_path(str(path)))


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="needs a file that its owner cannot read",
)
def test_a_rebuilt_copy_is_read_in_place_of_the_raw_file(tmp_path: Path) -> None:
    ds, source, _ = _rebuilt(tmp_path)
    source.chmod(0)
    try:
        outcome = ds.convert_all_tracks({"rebuild_run_id": LMT_REBUILD_RUN_ID})
    finally:
        source.chmod(0o644)

    assert (outcome.converted, outcome.failed) == (1, 0)
    assert 4 in set(_tracks_of(ds)["id"])


def test_the_raw_file_is_unchanged_by_a_conversion_from_its_copy(
    tmp_path: Path,
) -> None:
    ds, source, _ = _rebuilt(tmp_path)
    digests = file_digests(source.parent)
    mtime_ns = source.stat().st_mtime_ns

    _ = ds.convert_all_tracks({"rebuild_run_id": LMT_REBUILD_RUN_ID})

    assert file_digests(source.parent) == digests
    assert source.stat().st_mtime_ns == mtime_ns


def test_the_rebuild_run_changes_the_variant(tmp_path: Path) -> None:
    ds, _, _ = _rebuilt(tmp_path)

    default = ds.track_conversion_variant("lmt_sqlite")
    rebuilt = ds.track_conversion_variant(
        "lmt_sqlite", {"rebuild_run_id": LMT_REBUILD_RUN_ID}
    )

    assert rebuilt != default


def test_the_stderr_line_names_the_rebuild(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, _, copy = _rebuilt(tmp_path)

    _ = _convert(copy, LmtSqliteParams(rebuild_run_id=LMT_REBUILD_RUN_ID))

    (line,) = capsys.readouterr().err.splitlines()
    assert line.endswith(f" from rebuild {LMT_REBUILD_RUN_ID}")


def test_a_run_id_of_another_kind_is_refused_by_the_params() -> None:
    with pytest.raises(ValueError, match="rebuild-lmt-events"):
        _ = LmtSqliteParams(rebuild_run_id="convert-tracks.0.1-0123456789")


@pytest.mark.parametrize("how", REBUILD_SPOILS)
def test_a_copy_that_cannot_be_read_fails_the_entry_naming_the_file_and_the_run(
    tmp_path: Path, how: RebuildSpoil
) -> None:
    ds, source, _ = _rebuilt(tmp_path)
    spoil_rebuilt_copy(ds, LMT_REBUILD_RUN_ID, source, how)

    text = _failure_text(ds, {"rebuild_run_id": LMT_REBUILD_RUN_ID})

    assert str(source) in text
    assert LMT_REBUILD_RUN_ID in text
    assert read_tracks_index(ds).empty


def test_a_file_the_scan_did_not_hash_fails_the_entry_naming_the_file_and_the_run(
    tmp_path: Path,
) -> None:
    ds = lmt_dataset(tmp_path / "ds", md5=False)
    source = ds.base_dir / "raw" / "recording.sqlite"
    _ = place_rebuilt_copy(
        ds, LMT_REBUILD_RUN_ID, source, entry_key="recording", commit=LMT_REBUILD_COMMIT
    )

    text = _failure_text(ds, {"rebuild_run_id": LMT_REBUILD_RUN_ID})

    assert str(source) in text
    assert LMT_REBUILD_RUN_ID in text
    assert "mosaic scan" in text
