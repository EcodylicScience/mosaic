"""A divided store has one media index row per division, keyed by its chunks.

The store ``rec`` records 22 frames at 4 fps in chunks of 5, 5, 5, 5 and 2 frames,
and is divided every 10 frames. Its divisions are chunks 0-1, 2-3 and 4, which
start 2.5 s apart. The short last chunk ends the store, so its division is closed.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

import mosaic.core.dataset as dataset_module
from mosaic.core.dataset import Dataset
from mosaic.core.dataset_summary import dataset_summary
from mosaic.core.media import imgstore_io
from mosaic.core.media.facts_columns import (
    MEDIA_INDEX_COLUMNS,
    media_row_division_key,
    media_row_key,
    media_row_path_key,
    row_division,
)
from mosaic.core.media.imgstore_io import StoreProbe
from mosaic.core.media.store_division import (
    DIVISION_COLUMNS,
    DIVISION_IDENTITY_SCHEME,
    MediaRowKey,
    RowDivision,
    StoreWindow,
)
from mosaic.core.pipeline.media_index import (
    frame_from_rows,
    load_media_index_frame,
    write_media_index_rows,
)
from tests.helpers import (
    DIVISION_CREATED_UTC,
    MediaClip,
    add_transcode_derivative,
    make_dataset,
    store_dataset,
    video_store_maker,
    write_division_rows,
    write_media_index,
)

STORE_CHUNKS = (5, 5, 5, 5, 2)
RATE = 4.0
DIVISION = 10
TOKENS = ("20210219T192253Z", "20210219T192255Z", "20210219T192258Z")
WINDOWS = (StoreWindow(0, 1), StoreWindow(2, 3), StoreWindow(4, 4))


@pytest.fixture
def divided(tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]) -> Dataset:
    """A dataset whose one store ``rec`` is divided into three sequences."""
    dataset = store_dataset(
        tmp_path,
        make_media_dataset,
        video_store_maker(STORE_CHUNKS, fps=RATE, created_utc=DIVISION_CREATED_UTC),
    )
    _ = write_division_rows(dataset, divided_store(dataset), DIVISION)
    return dataset


def divided_store(dataset: Dataset) -> Path:
    """The store of the :func:`divided` dataset."""
    return dataset.get_root("media_raw") / "recordings" / "rec"


def index_path(dataset: Dataset) -> Path:
    """The originals index of *dataset*."""
    return dataset.get_root(dataset.resolve_media_root()) / "index.csv"


def test_an_undivided_row_is_written_with_blank_division_cells(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    write_media_index(dataset, [MediaClip(sequence="s", video_uuid="uid-s")])
    path = index_path(dataset)

    write_media_index_rows(path, load_media_index_frame(path))

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == MEDIA_INDEX_COLUMNS
        assert list(DIVISION_COLUMNS) == MEDIA_INDEX_COLUMNS[-len(DIVISION_COLUMNS) :]
        (row,) = list(reader)
    assert all(row[column] == "" for column in DIVISION_COLUMNS)
    assert row_division(row) is None
    assert media_row_key(row) == MediaRowKey(row["abs_path"], None, None)
    assert media_row_key(row).window is None


def test_a_half_filled_division_row_is_refused() -> None:
    chunks_only = {"abs_path": "rec", "chunk_first": "0", "chunk_last": ""}
    with pytest.raises(ValueError, match="names both"):
        _ = media_row_key(chunks_only)

    without_start = {
        "abs_path": "rec",
        "sequence": "s",
        "frame_count": "10",
        "chunk_first": "0",
        "chunk_last": "1",
        "first_frame": "0",
    }
    with pytest.raises(ValueError, match="fills every one"):
        _ = row_division(without_start)


def test_division_cells_round_trip_through_the_index(divided: Dataset) -> None:
    path = index_path(divided)
    before = path.read_bytes()

    write_media_index_rows(path, load_media_index_frame(path))

    assert path.read_bytes() == before
    rows = divided.read_media_index()
    found = [row_division(row) for row in rows]
    assert [division.sequence for division in found if division] == list(TOKENS)
    assert [division.window for division in found if division] == list(WINDOWS)
    assert [division.first_frame for division in found if division] == [0, 10, 20]
    assert [division.follows_gap for division in found if division] == [False] * 3
    assert {row["group"] for row in rows} == {"rec"}
    assert {row["camera"] for row in rows} == {""}
    assert {row["assignment_source"] for row in rows} == {"assigned"}


def test_each_division_row_records_its_window(divided: Dataset) -> None:
    rows = divided.read_media_index()

    assert [int(row["frame_count"]) for row in rows] == [10, 10, 2]
    facts = [divided.resolve_media("rec", token).facts[0] for token in TOKENS]
    assert {clip.identity_scheme for clip in facts} == {DIVISION_IDENTITY_SCHEME}
    assert len({clip.video_uuid for clip in facts}) == 3
    assert [row["video_uuid"] for row in rows] == [clip.video_uuid for clip in facts]
    assert [clip.fps for clip in facts] == pytest.approx([RATE] * 3)
    sizes = [int(row["size_bytes"]) for row in rows]
    assert sum(sizes) < imgstore_io.store_stat(divided_store(divided)).size


def test_division_rows_share_a_path_key_and_differ_in_their_division_key(
    divided: Dataset,
) -> None:
    rows = divided.read_media_index()

    assert len({media_row_path_key(row) for row in rows}) == 1
    keys = [media_row_division_key(row) for row in rows]
    assert [key[2:] for key in keys] == [(0, 1), (2, 3), (4, 4)]


def test_a_division_resolves_to_its_window_and_first_frame(divided: Dataset) -> None:
    store = divided_store(divided).resolve()

    for token, window, first_frame in zip(TOKENS, WINDOWS, (0, 10, 20), strict=True):
        resolved = divided.resolve_media("rec", token)
        assert resolved.paths == [store]
        assert resolved.windows == [window]
        assert resolved.first_frame == first_frame
        assert resolved.row_keys == [MediaRowKey.of(str(store), window)]
        rows = divided.media_rows_of(resolved.row_keys)
        assert [row["sequence"] for row in rows.values()] == [token]
        assert divided.indexed_entry(resolved.row_keys[0], ("", "")) == ("rec", token)


def test_a_scan_keeps_each_division_its_assigned_identity_and_calibration(
    divided: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = index_path(divided)
    frame = load_media_index_frame(path)
    frame["assignment_source"] = "assigned"
    frame["sequence"] = ["first", "second", "third"]
    frame["cm_per_pixel"] = ["0.1", "0.2", "0.3"]
    write_media_index_rows(path, frame)
    guessed = [
        {**row, "sequence": f"guess{at}", "cm_per_pixel": ""}
        for at, row in enumerate(divided.read_media_index())
    ]
    rescanned = [*guessed, guessed[0]]

    def scanned_store_rows(*_args: object) -> list[dict[str, object]]:
        return [dict(row) for row in rescanned]

    monkeypatch.setattr(Dataset, "_imgstore_rows", scanned_store_rows)
    _ = divided.index_media([divided_store(divided).parent])

    rows = divided.read_media_index()
    assert [row["sequence"] for row in rows] == ["first", "second", "third"]
    assert [row["cm_per_pixel"] for row in rows] == ["0.1", "0.2", "0.3"]
    assert [media_row_key(row).window for row in rows] == list(WINDOWS)


def test_a_whole_store_rescan_keeps_the_assigned_divisions_and_their_calibration(
    divided: Dataset,
) -> None:
    """A source that does not divide proposes the store's row, and the divisions stay."""
    assert divided.set_media_calibration(0.25, group="rec") == 3

    _ = divided.index_media([divided_store(divided).parent])

    rows = divided.read_media_index()
    assert [media_row_key(row).window for row in rows] == list(WINDOWS)
    assert [row["cm_per_pixel"] for row in rows] == ["0.25"] * 3
    assert divided.continuous_groups == ("rec",)


def test_a_store_divided_after_its_calibration_keeps_it_on_each_division(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    dataset = store_dataset(
        tmp_path,
        make_media_dataset,
        video_store_maker(STORE_CHUNKS, fps=RATE, created_utc=DIVISION_CREATED_UTC),
    )
    assert dataset.set_media_calibration(0.25, sequence="rec") == 1

    _ = write_division_rows(dataset, divided_store(dataset), DIVISION)

    rows = dataset.read_media_index()
    assert [row_division(row) is not None for row in rows] == [True] * 3
    assert [row["cm_per_pixel"] for row in rows] == ["0.25"] * 3


def test_reprobe_probes_a_divided_store_once_and_finds_it_unchanged(
    divided: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    probes: list[Path] = []
    probe_store = imgstore_io.probe_store

    def counted(path: Path | str) -> StoreProbe:
        probes.append(Path(path))
        return probe_store(path)

    monkeypatch.setattr(imgstore_io, "probe_store", counted)
    before = index_path(divided).read_bytes()

    report = divided.reprobe_media(apply=True)

    assert len(probes) == 1
    assert (report.rows_total, report.unchanged, report.rows_patched) == (3, 3, 0)
    assert not report.changed
    assert index_path(divided).read_bytes() == before


def test_the_summary_counts_a_divided_store_once(divided: Dataset) -> None:
    summary = dataset_summary(divided)

    assert summary.counts.videos == 1
    assert [rate.videos for rate in summary.frame_rates] == [1]
    (recording,) = summary.recordings
    assert recording.name == "rec"
    assert recording.kind == "divided"


def read_cells(path: Path) -> list[dict[str, str]]:
    """The rows of the index at *path*, as the text of each cell."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def add_plain_clip(dataset: Dataset, abs_path: str) -> None:
    """Append an undivided clip row at *abs_path* to the originals index."""
    path = index_path(dataset)
    rows: list[dict[str, object]] = [dict(row) for row in read_cells(path)]
    plain: dict[str, object] = {
        **rows[0],
        **dict.fromkeys(DIVISION_COLUMNS, ""),
        "abs_path": abs_path,
        "group": "plain",
        "sequence": "clip",
        "group_safe": "plain",
        "sequence_safe": "clip",
        "video_uuid": "uid-plain",
        "assignment_source": "assigned",
    }
    write_media_index_rows(path, frame_from_rows([*rows, plain]))


def without_paths(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """*rows* without their ``abs_path`` cells."""
    return [{k: v for k, v in row.items() if k != "abs_path"} for row in rows]


def placements(rows: list[dict[str, str]]) -> list[tuple[int, bool] | None]:
    """The first frame and gap of the division each of *rows* records."""
    found = [row_division(row) for row in rows]
    return [(d.first_frame, d.follows_gap) if d else None for d in found]


@pytest.mark.parametrize("mixed", [False, True], ids=["divided", "mixed"])
def test_a_path_rewrite_keeps_every_division_cell(
    divided: Dataset, mixed: bool
) -> None:
    path = index_path(divided)
    store = str(divided_store(divided).resolve())
    frame = load_media_index_frame(path)
    frame["abs_path"] = f"/elsewhere/{divided_store(divided).name}"
    write_media_index_rows(path, frame)
    if mixed:
        add_plain_clip(divided, "/elsewhere/clip.mp4")
    before = read_cells(path)

    changed = divided.rewrite_index_paths({"/elsewhere": str(Path(store).parent)})

    after = read_cells(path)
    assert changed[str(path)] == len(before)
    assert without_paths(after) == without_paths(before)
    assert placements(after)[:3] == [(0, False), (10, False), (20, False)]
    assert [media_row_key(row).window for row in after][:3] == list(WINDOWS)


@pytest.mark.parametrize("mixed", [False, True], ids=["divided", "mixed"])
def test_making_an_index_portable_keeps_every_division_cell(
    divided: Dataset, mixed: bool
) -> None:
    path = index_path(divided)
    frame = load_media_index_frame(path)
    frame["abs_path"] = str(divided_store(divided).resolve())
    write_media_index_rows(path, frame)
    if mixed:
        add_plain_clip(divided, str(divided_store(divided).parent / "clip.mp4"))
    before = read_cells(path)

    changed = divided.make_portable()

    after = read_cells(path)
    assert changed[str(path)] == len(before)
    assert all(not Path(row["abs_path"]).is_absolute() for row in after)
    assert without_paths(after) == without_paths(before)
    assert placements(after)[:3] == [(0, False), (10, False), (20, False)]
    assert [media_row_key(row).window for row in after][:3] == list(WINDOWS)
    assert [divided.resolve_media("rec", token).windows for token in TOKENS] == [
        [window] for window in WINDOWS
    ]


def test_a_scope_reads_a_divided_recordings_rows_once(
    divided: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One read of the recording's rows, then one of each sequence's own row."""
    reads: list[str] = []

    def counted(row: Mapping[str, object]) -> RowDivision | None:
        reads.append(str(row["sequence"]))
        return row_division(row)

    monkeypatch.setattr(dataset_module, "row_division", counted)

    scope = divided.resolve_media_scope([("rec", token) for token in TOKENS])

    assert [media.resolved.first_frame for media in scope] == [0, 10, 20]
    assert sorted(reads) == sorted([*TOKENS, *TOKENS])


def test_prune_keeps_the_derivative_of_each_division(divided: Dataset) -> None:
    """Each division's derivative is named for its uid and linked from its row."""
    derivatives = [
        add_transcode_derivative(divided, token, target="analysis") for token in TOKENS
    ]
    rows = divided.read_media_index()
    assert [path.name.split(".")[0] for path in derivatives] == [
        row["video_uuid"] for row in rows
    ]
    derivative_index = divided.get_root("media") / "index.csv"
    derivative_rows = derivative_index.read_bytes()

    for apply in (False, True):
        report = divided.prune_media(apply=apply, min_age_hours=0.0)

        assert report.considered
        assert report.counts() == {"live": 3}
        assert not report.files_deleted
    assert all(path.exists() for path in derivatives)
    assert derivative_index.read_bytes() == derivative_rows
    assert [row["analysis_derivative_path"] for row in divided.read_media_index()] == [
        f"transcode/{path.name}" for path in derivatives
    ]
