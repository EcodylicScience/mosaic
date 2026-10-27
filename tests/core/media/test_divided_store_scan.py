"""A media scan or a division scope divides an imgstore recording into sequences.

The store ``rec`` of camera ``123`` records frames at 4 fps in chunks of 5, from
2021-02-19 19:22:53 UTC, and is divided every 10 frames: chunks 0-1, 2-3 and 4,
which start 2.5 s apart. Its recording is ``123_rec``. A store whose last chunk
is full is still recording, and one whose last chunk is short has ended.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import mosaic.core.pipeline.inventory.scan as inventory_scan
import mosaic.core.pipeline.media_index as media_index_module
from mosaic.agent.mcp.local import LocalBackend
from mosaic.agent.mcp.spec import DatasetChange, call_tool
from mosaic.cli import app
from mosaic.core.dataset import Dataset, new_dataset_manifest, open_dataset
from mosaic.core.entry import EntryKeyCollisionError
from mosaic.core.manifest import MediaDivision, MediaScanSource, read_manifest
from mosaic.core.media.facts_columns import row_division
from mosaic.core.media.store_division import DivisionRefused, StoreWindow
from mosaic.core.pipeline.artifact_kinds import ArtifactKind
from mosaic.core.pipeline.media_index import MediaIndexScope
from tests.helpers import (
    DIVISION_CREATED_UTC,
    RecordingStarter,
    add_media_variant,
    add_tracks_table,
    track_table,
    video_store_maker,
    write_h264_mp4,
)

pytestmark = pytest.mark.media

RATE = 4.0
SERIAL = "123"
RECORDING = "123_rec"
TOKENS = ("20210219T192253Z", "20210219T192255Z", "20210219T192258Z")
ENDED = (5, 5, 5, 5, 2)
"""A store whose short last chunk ended it."""
RECORDING_STILL = (5, 5, 5, 5, 5)
"""A store whose full last chunk may still grow."""
DIVISION = MediaDivision(frames=10)

runner = CliRunner()


def make_store(
    parent: Path,
    chunks: Sequence[int],
    *,
    name: str = "rec",
    serial: str = SERIAL,
    created_utc: str = DIVISION_CREATED_UTC,
    extra: Mapping[str, object] | None = None,
    store_metadata: Mapping[str, object] | None = None,
) -> Path:
    """Write the store *name* of camera *serial* under *parent*, and return it."""
    make = video_store_maker(
        chunks, fps=RATE, created_utc=created_utc, store_metadata=store_metadata
    )
    metadata: dict[str, object] = {"camera_serial": serial} if serial else {}
    metadata.update(extra or {})
    store, _ = make(name=name, parent=parent, extra_metadata=metadata)
    return store


def scanned_dataset(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    stores: Path,
    division: MediaDivision = DIVISION,
) -> Dataset:
    """A dataset that declares *stores* a media source dividing by *division*."""
    dataset = make_media_dataset(tmp_path / "dataset")
    dataset.add_scan_source(
        MediaScanSource(id="stores", path=str(stores), division=division)
    )
    return dataset


def division_scope(store: Path, group: str = "", **division: object) -> MediaIndexScope:
    """The division scope of *store*, every 10 frames unless *division* says."""
    rule = MediaDivision.model_validate({"frames": 10, **division})
    return MediaIndexScope(directory=store, group=group, sequence="", division=rule)


def placed(dataset: Dataset) -> list[tuple[str, str, int, bool, StoreWindow]]:
    """Each division row's group, sequence, first frame, gap and window."""
    found: list[tuple[str, str, int, bool, StoreWindow]] = []
    for row in dataset.read_media_index():
        division = row_division(row)
        assert division is not None, row
        found.append(
            (
                row["group"],
                division.sequence,
                division.first_frame,
                division.follows_gap,
                division.window,
            )
        )
    return found


def index_bytes(dataset: Dataset) -> bytes:
    """The originals index of *dataset*."""
    return (dataset.get_root("media_raw") / "index.csv").read_bytes()


# --- rows, names and closure ----------------------------------------------------


def test_a_scan_writes_one_row_per_division_on_the_recording_axis(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    store = make_store(tmp_path / "stores", ENDED)
    dataset = scanned_dataset(tmp_path, make_media_dataset, store.parent)

    result = dataset.scan_media()

    assert placed(dataset) == [
        (RECORDING, TOKENS[0], 0, False, StoreWindow(0, 1)),
        (RECORDING, TOKENS[1], 10, False, StoreWindow(2, 3)),
        (RECORDING, TOKENS[2], 20, False, StoreWindow(4, 4)),
    ]
    rows = dataset.read_media_index()
    assert [int(row["frame_count"]) for row in rows] == [10, 10, 2]
    assert {row["assignment_source"] for row in rows} == {"scan-imgstore-division"}
    assert {row["camera"] for row in rows} == {""}
    assert {row["camera_serial"] for row in rows} == {SERIAL}
    (plan,) = result.divisions
    assert plan.recording == RECORDING
    assert (result.held_back, result.refused) == ((), ())
    assert dataset.continuous_groups == (RECORDING,)
    assert read_manifest(dataset.manifest_path).manifest_version == 4


def test_a_full_last_chunk_is_held_back_until_it_is_released(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    store = make_store(tmp_path / "stores", RECORDING_STILL)
    dataset = scanned_dataset(tmp_path, make_media_dataset, store.parent)

    held = dataset.scan_media()
    released = dataset.scan_media(include_partial=True)

    (held_back,) = held.held_back
    assert (held_back.chunk_first, held_back.chunk_last) == (4, 4)
    assert (held_back.reason, held_back.frame_count) == ("store_recording", 5)
    assert released.held_back == ()
    assert [token for _, token, *_ in placed(dataset)] == list(TOKENS)
    assert [
        plan_division.frozen for plan_division in released.divisions[0].divisions
    ] == [
        True,
        True,
        False,
    ]


def test_a_rescan_writes_the_same_bytes(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    store = make_store(tmp_path / "stores", ENDED)
    dataset = scanned_dataset(tmp_path, make_media_dataset, store.parent)
    _ = dataset.scan_media()
    before = index_bytes(dataset)

    _ = dataset.scan_media()

    assert index_bytes(dataset) == before


def test_a_rescan_keeps_its_divisions_and_adds_those_that_closed(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    """Chunks 5 and 6 are recorded after the first scan, and 6 is short."""
    whole = make_store(tmp_path / "whole", (5, 5, 5, 5, 5, 5, 2))
    store = tmp_path / "stores" / "rec"
    store.mkdir(parents=True)
    for path in sorted(whole.iterdir()):
        if path.name == "metadata.yaml" or not path.name.startswith(
            ("000005", "000006")
        ):
            _ = shutil.copy2(path, store / path.name)
    dataset = scanned_dataset(tmp_path, make_media_dataset, store.parent)
    _ = dataset.scan_media()
    first = dataset.read_media_index()
    for path in sorted(whole.glob("00000[56].*")):
        _ = shutil.copy2(path, store / path.name)

    grown = dataset.scan_media()

    rows = dataset.read_media_index()
    assert rows[:2] == first
    assert [(start, window) for _, _, start, _, window in placed(dataset)] == [
        (0, StoreWindow(0, 1)),
        (10, StoreWindow(2, 3)),
        (20, StoreWindow(4, 5)),
        (30, StoreWindow(6, 6)),
    ]
    assert [division.frozen for division in grown.divisions[0].divisions] == [
        True,
        True,
        False,
        False,
    ]


def test_a_division_keeps_its_name_when_the_store_clock_source_changes(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    """Without ``created_utc`` the directory name, 20:00 UTC, would name the store."""
    store = make_store(
        tmp_path / "stores",
        ENDED,
        name="20210219_120000",
        store_metadata={"timezone_local": "America/Los_Angeles"},
    )
    dataset = scanned_dataset(tmp_path, make_media_dataset, store.parent)
    _ = dataset.scan_media()
    metadata = store / "metadata.yaml"
    loaded = yaml.safe_load(metadata.read_text())
    del loaded["__store"]["created_utc"]
    _ = metadata.write_text(yaml.safe_dump(loaded))

    _ = dataset.scan_media()

    assert [token for _, token, *_ in placed(dataset)] == list(TOKENS)
    assert {row["start_source"] for row in dataset.read_media_index()} == {
        "created_utc"
    }


def test_by_camera_a_later_scope_continues_the_divided_stores_axis(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    """The second store's scope finds the first among the divided rows."""
    stores = tmp_path / "stores"
    first = make_store(stores, (5, 5, 5, 5, 2), name="a")
    second = make_store(stores, (5, 5, 2), name="b", created_utc="2021-02-19T20:22:53")
    dataset = make_media_dataset(tmp_path / "dataset")

    _ = dataset.write_media_index([division_scope(first, identity="camera")])
    _ = dataset.write_media_index([division_scope(second, identity="camera")])

    assert [(token, start, gap) for _, token, start, gap, _ in placed(dataset)] == [
        (TOKENS[0], 0, False),
        (TOKENS[1], 10, False),
        (TOKENS[2], 20, False),
        ("20210219T202253Z", 22, True),
        ("20210219T202255Z", 32, False),
    ]
    assert dataset.continuous_groups == (SERIAL,)


def test_by_camera_the_stores_of_a_serial_continue_one_frame_axis(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    """The second store starts an hour later, after a boundary."""
    stores = tmp_path / "stores"
    _ = make_store(stores, (5, 5, 5, 5, 2), name="a")
    _ = make_store(stores, (5, 5, 2), name="b", created_utc="2021-02-19T20:22:53")
    dataset = scanned_dataset(
        tmp_path,
        make_media_dataset,
        stores,
        MediaDivision(frames=10, identity="camera"),
    )

    _ = dataset.scan_media()

    assert placed(dataset) == [
        (SERIAL, TOKENS[0], 0, False, StoreWindow(0, 1)),
        (SERIAL, TOKENS[1], 10, False, StoreWindow(2, 3)),
        (SERIAL, TOKENS[2], 20, False, StoreWindow(4, 4)),
        (SERIAL, "20210219T202253Z", 22, True, StoreWindow(0, 1)),
        (SERIAL, "20210219T202255Z", 32, False, StoreWindow(2, 2)),
    ]
    assert dataset.continuous_groups == (SERIAL,)


def test_the_scope_and_the_scan_write_the_same_rows(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    store = make_store(tmp_path / "stores", ENDED)
    scanned = scanned_dataset(tmp_path, make_media_dataset, store.parent)
    scoped = make_media_dataset(tmp_path / "scoped")

    _ = scanned.scan_media()
    result = scoped.write_media_index([division_scope(store)])

    def without_source(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        return [
            {k: v for k, v in row.items() if k != "assignment_source"} for row in rows
        ]

    assert without_source(scoped.read_media_index()) == without_source(
        scanned.read_media_index()
    )
    assert {row["assignment_source"] for row in scoped.read_media_index()} == {
        "assigned"
    }
    assert result.divisions[0].recording == RECORDING
    assert scoped.continuous_groups == (RECORDING,)


def test_a_division_scope_replaces_the_whole_store_in_one_index_write(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = make_store(tmp_path / "stores", ENDED)
    dataset = make_media_dataset(tmp_path / "dataset")
    _ = dataset.index_media([store.parent])
    assert [row_division(row) for row in dataset.read_media_index()] == [None]
    writes: list[Path] = []
    write = media_index_module.write_media_index_rows

    def counted(index_path: Path, frame: pd.DataFrame) -> None:
        writes.append(index_path)
        write(index_path, frame)

    monkeypatch.setattr(media_index_module, "write_media_index_rows", counted)
    monkeypatch.setattr("mosaic.core.dataset.write_media_index_rows", counted)

    _ = dataset.write_media_index([division_scope(store)])

    assert len(writes) == 1
    assert [token for _, token, *_ in placed(dataset)] == list(TOKENS)


def test_a_division_scope_raises_the_manifest_to_4_before_it_writes_division_rows(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older mosaic never opens an index of division rows as whole stores."""
    store = make_store(tmp_path / "stores", ENDED)
    dataset = make_media_dataset(tmp_path / "dataset")
    seen: list[tuple[int, tuple[str, ...]]] = []
    write = media_index_module.write_media_index_rows

    def recorded(index_path: Path, frame: pd.DataFrame) -> None:
        manifest = read_manifest(dataset.manifest_path)
        seen.append((manifest.manifest_version, manifest.continuous_groups))
        write(index_path, frame)

    monkeypatch.setattr("mosaic.core.dataset.write_media_index_rows", recorded)

    _ = dataset.write_media_index([division_scope(store)])

    assert seen == [(4, ())]
    assert read_manifest(dataset.manifest_path).continuous_groups == (RECORDING,)


def refuse_every_entry_key(dataset: Dataset, written: object, existing: object) -> None:
    """A collision check that refuses every write."""
    _ = (dataset, written, existing)
    message = "the entry collides"
    raise EntryKeyCollisionError(message)


def test_a_division_scope_refused_inside_the_lock_leaves_the_manifest(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused write leaves a manifest that an older mosaic still opens."""
    store = make_store(tmp_path / "stores", ENDED)
    dataset = make_media_dataset(tmp_path / "dataset")
    before = dataset.manifest_path.read_bytes()
    monkeypatch.setattr(Dataset, "refuse_entry_key_collision", refuse_every_entry_key)

    with pytest.raises(EntryKeyCollisionError):
        _ = dataset.write_media_index([division_scope(store)])

    assert dataset.manifest_path.read_bytes() == before
    assert read_manifest(dataset.manifest_path).manifest_version == 3


def test_a_scan_refused_inside_the_lock_leaves_the_manifest(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = make_store(tmp_path / "stores", ENDED)
    dataset = scanned_dataset(tmp_path, make_media_dataset, store.parent)
    before = dataset.manifest_path.read_bytes()
    monkeypatch.setattr(Dataset, "refuse_entry_key_collision", refuse_every_entry_key)

    with pytest.raises(EntryKeyCollisionError):
        _ = dataset.scan_media()

    assert dataset.manifest_path.read_bytes() == before


def test_a_recording_whose_store_is_divided_stays_declared(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    store = make_store(tmp_path / "stores", ENDED)
    dataset = scanned_dataset(tmp_path, make_media_dataset, store.parent)
    _ = dataset.scan_media()

    with pytest.raises(ValueError, match="leaves out \\['123_rec'\\]"):
        dataset.set_continuous_groups([])

    assert read_manifest(dataset.manifest_path).continuous_groups == (RECORDING,)


def test_a_whole_store_scan_keeps_a_scopes_divisions(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    store = make_store(tmp_path / "stores", ENDED)
    dataset = make_media_dataset(tmp_path / "dataset")
    _ = dataset.write_media_index([division_scope(store)])
    before = dataset.read_media_index()

    _ = dataset.index_media([store.parent])

    assert dataset.read_media_index() == before


def test_a_store_without_outputs_is_divided_again_under_a_new_rule(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    store = make_store(tmp_path / "stores", ENDED)
    dataset = make_media_dataset(tmp_path / "dataset")
    _ = dataset.index_media([store.parent])
    _ = dataset.write_media_index([division_scope(store)])

    _ = dataset.write_media_index([division_scope(store, frames=20)])

    assert [(token, window) for _, token, _, _, window in placed(dataset)] == [
        (TOKENS[0], StoreWindow(0, 3)),
        (TOKENS[2], StoreWindow(4, 4)),
    ]


# --- refusals -------------------------------------------------------------------


def refused(call: Callable[[], object]) -> DivisionRefused:
    """The refusal that *call* raises."""
    with pytest.raises(DivisionRefused) as error:
        _ = call()
    return error.value


def test_a_division_that_is_not_whole_chunks_is_refused_before_a_write(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    store = make_store(tmp_path / "stores", ENDED)
    dataset = scanned_dataset(
        tmp_path, make_media_dataset, store.parent, MediaDivision(frames=12)
    )

    error = refused(dataset.scan_media)

    assert error.reason == "frames_not_chunk_multiple"
    assert dataset.read_media_index() == []
    assert dataset.continuous_groups == ()


def test_a_store_of_a_synchronized_set_is_refused(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    store = make_store(
        tmp_path / "stores",
        ENDED,
        extra={"synchronizationuuid": "f064", "synchronization": "framenumber"},
    )
    dataset = make_media_dataset(tmp_path / "dataset")

    error = refused(lambda: dataset.write_media_index([division_scope(store)]))

    assert error.reason == "synchronized_store"


def test_a_division_scope_names_one_store(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    stores = tmp_path / "stores"
    _ = make_store(stores, ENDED, name="a")
    _ = make_store(stores, ENDED, name="b")
    videos = tmp_path / "videos"
    write_h264_mp4(videos / "clip.mp4", frames=4, fps=RATE)
    dataset = make_media_dataset(tmp_path / "dataset")

    for directory in (stores, videos):
        error = refused(
            lambda directory=directory: dataset.write_media_index(
                [division_scope(directory)]
            )
        )
        assert error.reason == "not_one_store"
    with pytest.raises(ValueError, match="names the sequence 'x'"):
        _ = dataset.write_media_index(
            [
                MediaIndexScope(
                    directory=stores / "a", group="", sequence="x", division=DIVISION
                )
            ]
        )


def test_a_scan_refuses_a_store_a_caller_assigned_unless_it_reassigns(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    store = make_store(tmp_path / "stores", ENDED)
    dataset = scanned_dataset(tmp_path, make_media_dataset, store.parent)
    _ = dataset.write_media_index(
        [MediaIndexScope(directory=store, group="lab", sequence="rec")]
    )

    error = refused(dataset.scan_media)
    _ = dataset.scan_media(reassign=True)

    assert error.reason == "assigned_store"
    assert "division MediaIndexScope" in str(error)
    assert [token for _, token, *_ in placed(dataset)] == list(TOKENS)


def test_a_name_another_recording_has_is_refused(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    stores = tmp_path / "stores"
    first = make_store(stores / "one", ENDED)
    second = make_store(stores / "two", ENDED)
    dataset = make_media_dataset(tmp_path / "dataset")
    _ = dataset.write_media_index([division_scope(first)])

    error = refused(lambda: dataset.write_media_index([division_scope(second)]))
    both = refused(
        lambda: make_media_dataset(tmp_path / "both").write_media_index(
            [division_scope(first), division_scope(second)]
        )
    )

    assert error.reason == both.reason == "recording_name_taken"


def test_a_name_of_an_empty_group_sequence_is_refused(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    store = make_store(tmp_path / "stores", ENDED)
    videos = tmp_path / "videos"
    write_h264_mp4(videos / "clip.mp4", frames=4, fps=RATE)
    dataset = make_media_dataset(tmp_path / "dataset")
    _ = dataset.write_media_index(
        [MediaIndexScope(directory=videos, group="", sequence=RECORDING)]
    )

    error = refused(lambda: dataset.write_media_index([division_scope(store)]))

    assert error.reason == "recording_name_taken"


def test_a_recording_declared_by_hand_is_not_given_divisions(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    store = make_store(tmp_path / "stores", ENDED)
    videos = tmp_path / "videos"
    write_h264_mp4(videos / "clip.mp4", frames=4, fps=RATE)
    dataset = make_media_dataset(tmp_path / "dataset")
    _ = dataset.write_media_index(
        [MediaIndexScope(directory=videos, group="session", sequence="part1")]
    )
    dataset.set_continuous_groups(["session"])

    error = refused(
        lambda: dataset.write_media_index([division_scope(store, group="session")])
    )

    assert error.reason == "mixed_recording"


def test_dividing_again_is_refused_while_an_entry_has_outputs(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    store = make_store(tmp_path / "stores", ENDED)
    dataset = make_media_dataset(tmp_path / "dataset")
    _ = dataset.write_media_index([division_scope(store)])
    add_tracks_table(dataset, "v1", track_table(RECORDING, TOKENS[1], n_rows=10))
    before = index_bytes(dataset)

    error = refused(
        lambda: dataset.write_media_index([division_scope(store, frames=20)])
    )

    assert error.reason == "redivision_with_outputs"
    assert f"{RECORDING}:{TOKENS[1]}: tracks-variant v1" in str(error)
    assert index_bytes(dataset) == before


def test_dividing_a_store_indexed_whole_is_refused_while_it_has_outputs(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    store = make_store(tmp_path / "stores", ENDED)
    dataset = make_media_dataset(tmp_path / "dataset")
    _ = dataset.index_media([store.parent])
    _ = add_media_variant(dataset, "preprocess.0.1-abc", "rec")

    error = refused(lambda: dataset.write_media_index([division_scope(store)]))

    assert error.reason == "redivision_with_outputs"
    assert "rec: media-variant preprocess.0.1-abc" in str(error)


def test_dividing_a_store_indexed_whole_without_outputs_is_reported(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = make_store(tmp_path / "stores", ENDED)
    dataset = make_media_dataset(tmp_path / "dataset")
    _ = dataset.index_media([store.parent])
    _ = capsys.readouterr()

    _ = dataset.write_media_index([division_scope(store)])

    stderr = capsys.readouterr().err
    assert f"the recording {RECORDING!r} is divided again" in stderr
    assert "are indexed whole" in stderr


def camera_with_second_store_whole(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> tuple[Dataset, Path]:
    """A dataset whose camera's first store is divided and second indexed as ``b``."""
    stores = tmp_path / "stores"
    first = make_store(stores, (5, 5, 5, 5, 2), name="a")
    second = make_store(stores, (5, 5, 2), name="b", created_utc="2021-02-19T20:22:53")
    dataset = make_media_dataset(tmp_path / "dataset")
    _ = dataset.write_media_index([division_scope(first, identity="camera")])
    _ = dataset.write_media_index(
        [MediaIndexScope(directory=second, group="", sequence="b")]
    )
    return dataset, second


def test_by_camera_a_later_store_indexed_whole_is_refused_while_it_has_outputs(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    """The first store's divisions do not make the second store's division growth."""
    dataset, second = camera_with_second_store_whole(tmp_path, make_media_dataset)
    add_tracks_table(dataset, "v1", track_table("", "b", n_rows=10))
    before = index_bytes(dataset)

    error = refused(
        lambda: dataset.write_media_index([division_scope(second, identity="camera")])
    )

    assert error.reason == "redivision_with_outputs"
    assert "b: tracks-variant v1" in str(error)
    assert index_bytes(dataset) == before


def test_by_camera_a_later_store_indexed_whole_is_divided_after_the_first(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    dataset, second = camera_with_second_store_whole(tmp_path, make_media_dataset)
    first_rows = [row for row in dataset.read_media_index() if row["group"] == SERIAL]

    _ = dataset.write_media_index([division_scope(second, identity="camera")])

    assert dataset.read_media_index()[: len(first_rows)] == first_rows
    assert [(token, start, gap) for _, token, start, gap, _ in placed(dataset)] == [
        (TOKENS[0], 0, False),
        (TOKENS[1], 10, False),
        (TOKENS[2], 20, False),
        ("20210219T202253Z", 22, True),
        ("20210219T202255Z", 32, False),
    ]


def test_outputs_the_tracking_roots_hold_are_refused_unlisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process that never registered the tracking kinds cannot list their outputs."""
    store = make_store(tmp_path / "stores", ENDED)
    dataset = open_dataset(new_dataset_manifest("d", tmp_path / "dataset"))
    _ = dataset.index_media([store.parent])

    def unregistered(kind: ArtifactKind) -> None:
        _ = kind

    monkeypatch.setattr(inventory_scan, "inventory_contributor", unregistered)
    assert dataset.entry_outputs([("", "rec")]) == {("", "rec"): []}
    frames = dataset.get_root("frames") / "uniform" / "run"
    frames.mkdir(parents=True)
    _ = (frames / "index.csv").write_text("group,sequence\n")

    error = refused(lambda: dataset.write_media_index([division_scope(store)]))

    assert error.reason == "tracking_unregistered"
    assert "register_ops" in str(error)


def test_a_division_that_moved_under_outputs_is_refused_on_a_rescan(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    """An interior chunk index lost after the division leaves its division unmade."""
    store = make_store(tmp_path / "stores", ENDED)
    dataset = scanned_dataset(tmp_path, make_media_dataset, store.parent)
    _ = dataset.scan_media()
    add_tracks_table(dataset, "v1", track_table(RECORDING, TOKENS[0], n_rows=10))
    (store / "000002.npz").unlink()

    with pytest.warns(UserWarning, match="missing index for imgstore chunk 2"):
        error = refused(dataset.scan_media)

    assert error.reason == "redivision_with_outputs"
    assert "moved" in str(error)


def test_a_division_that_moved_without_outputs_is_reported_as_it_is_replaced(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = make_store(tmp_path / "stores", ENDED)
    dataset = scanned_dataset(tmp_path, make_media_dataset, store.parent)
    _ = dataset.scan_media()
    _ = capsys.readouterr()
    (store / "000002.npz").unlink()

    with pytest.warns(UserWarning, match="missing index for imgstore chunk 2"):
        result = dataset.scan_media()

    stderr = capsys.readouterr().err
    assert f"the recording {RECORDING!r} is divided again" in stderr
    assert f"the division {TOKENS[1]!r}" in stderr
    assert [(span.chunk_first, span.reason) for span in result.refused] == [
        (2, "missing_chunk_index")
    ]
    assert [token for _, token, *_ in placed(dataset)] == [TOKENS[0], TOKENS[2]]


# --- reports ----------------------------------------------------------------------


def test_scan_reports_its_divisions_as_json_and_its_hold_back_on_stderr(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    store = make_store(tmp_path / "stores", RECORDING_STILL)
    dataset = scanned_dataset(tmp_path, make_media_dataset, store.parent)
    manifest = str(dataset.manifest_path)

    as_json = runner.invoke(app, ["scan", "-m", manifest, "--kind", "media", "--json"])
    plain = runner.invoke(app, ["scan", "-m", manifest, "--kind", "media"])
    released = runner.invoke(
        app, ["scan", "-m", manifest, "--kind", "media", "--include-partial", "--json"]
    )

    assert as_json.exit_code == plain.exit_code == released.exit_code == 0
    payload = json.loads(as_json.stdout)
    assert [entry["sequence"] for entry in payload["divisions"]] == list(TOKENS[:2])
    (held,) = payload["held_back"]
    assert (held["group"], held["reason"], held["chunk_first"]) == (
        RECORDING,
        "store_recording",
        4,
    )
    assert payload["refused"] == []
    assert "held back: 123_rec (store_recording)" in plain.stderr
    assert "held back" not in plain.stdout
    assert len(json.loads(released.stdout)["divisions"]) == 3


def test_the_mcp_scan_reports_its_divisions(tmp_path: Path) -> None:
    store = make_store(tmp_path / "stores", RECORDING_STILL)
    dataset = open_dataset(new_dataset_manifest("d", tmp_path / "dataset"))
    dataset.add_scan_source(
        MediaScanSource(id="stores", path=str(store.parent), division=DIVISION)
    )
    backend = LocalBackend(
        [dataset.manifest_path], warm=False, start_driver=RecordingStarter()
    )
    uuid = dataset.manifest.uuid

    held = call_tool(backend, "manage_dataset", {"action": "scan", "dataset": uuid})
    released = call_tool(
        backend,
        "manage_dataset",
        {"action": "scan", "dataset": uuid, "include_partial": True},
    )

    assert isinstance(held, DatasetChange), held
    assert isinstance(released, DatasetChange), released
    assert (held.divisions, released.divisions) == (2, 3)
    (span,) = held.held_back
    assert (span.reason, span.chunk_first, span.chunk_last) == ("store_recording", 4, 4)
    assert released.held_back == released.refused == ()


def test_a_scan_names_each_division_by_its_start(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    store = make_store(
        tmp_path / "stores", ENDED, store_metadata={"timezone_local": "Europe/Berlin"}
    )
    dataset = scanned_dataset(tmp_path, make_media_dataset, store.parent)

    _ = dataset.scan_media()

    assert dataset.display_names() == {
        (RECORDING, TOKENS[0]): "2021-02-19 20:22:53 CET",
        (RECORDING, TOKENS[1]): "2021-02-19 20:22:55 CET",
        (RECORDING, TOKENS[2]): "2021-02-19 20:22:58 CET",
    }
