"""Tracks of a divided store land on its recording's axes, and overlap reads across.

A media scan divides the store ``rec`` of camera ``123`` (4 fps, chunks of 5)
every 10 frames into the recording ``123_rec``. A tracks table converted for a
division is numbered from the division's first frame and lands at it, on the
frame axis and on the time axis of the store's clock. An inference op lands at
it too. Neighbors exist only where one division's frames continue another's.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest, open_dataset
from mosaic.core.label_converter import LABEL_CONVERTERS
from mosaic.core.manifest import MediaDivision, MediaScanSource
from mosaic.core.pipeline.manifest import build_manifest
from mosaic.core.pipeline.labels_index import read_labels_index
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.pipeline.tracks_raw_index import TracksRawIndexScope
from mosaic.core.pipeline.types import Inputs
from mosaic.core.track_converter import (
    TRACK_CONVERTERS,
    EntryHints,
    TrackConverter,
    TrackConvertParams,
)
from tests.helpers import (
    SecondsEntries,
    DIVISION_CREATED_UTC,
    LISTED_FRAMES,
    TracksAxisEntries,
    install_fake_pose_inference,
    pose_per_frame,
    video_store_maker,
    write_listed_entries,
    write_sleap_analysis_h5,
)

pytestmark = [pytest.mark.media, pytest.mark.slow]

RATE = 4.0
RECORDING = "123_rec"
LOCAL_CSV = "division_csv"


class _LocalParams(TrackConvertParams):
    """A CSV already in the tracks layout needs no knobs."""


class _LocalCsvConverter(TrackConverter[_LocalParams]):
    """Read a CSV of ``frame``, ``time``, ``id``, ``X`` and ``Y`` as it is written."""

    src_format = LOCAL_CSV
    version = "0.1"
    output_schema = "mosaic_v1"
    Params = _LocalParams

    def convert(
        self, path: Path, params: _LocalParams, hints: EntryHints
    ) -> pd.DataFrame:
        _ = params
        table = pd.read_csv(path)
        table["group"] = hints.group
        table["sequence"] = hints.sequence
        return table


@pytest.fixture
def local_csv(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register the CSV converter for one test."""
    monkeypatch.setitem(TRACK_CONVERTERS, LOCAL_CSV, _LocalCsvConverter)
    return LOCAL_CSV


def divided_dataset(
    tmp_path: Path,
    chunks: Sequence[int],
    *,
    frame_times: Sequence[float] | None = None,
    missing_index: Collection[int] = (),
) -> Dataset:
    """A dataset whose source divides the store ``rec`` of *chunks* every 10 frames."""
    make = video_store_maker(
        chunks,
        fps=RATE,
        created_utc=DIVISION_CREATED_UTC,
        frame_times=frame_times,
        missing_index=missing_index,
    )
    stores = tmp_path / "stores"
    _ = make(name="rec", parent=stores, extra_metadata={"camera_serial": "123"})
    dataset = open_dataset(new_dataset_manifest("divided", tmp_path / "dataset"))
    dataset.add_scan_source(
        MediaScanSource(
            id="stores", path=str(stores), division=MediaDivision(frames=10)
        )
    )
    _ = dataset.scan_media()
    return dataset


def divisions(dataset: Dataset) -> list[tuple[str, int, int]]:
    """Each division's sequence, first frame and frame count."""
    facts = dataset.division_facts(
        (row["group"], row["sequence"]) for row in dataset.read_media_index()
    )
    return [
        (sequence, fact.first_frame or 0, fact.frame_count or 0)
        for (_, sequence), fact in sorted(
            facts.items(), key=lambda item: item[1].first_frame or 0
        )
    ]


def index_tracks_raw_dirs(dataset: Dataset, src_format: str, pattern: str) -> None:
    """Index each sequence directory of ``tracks_raw`` as *src_format* files of *pattern*."""
    rows = [
        TracksRawIndexScope(
            directory=dataset.get_root("tracks_raw") / found,
            group=RECORDING,
            sequence=found,
            src_format=src_format,
        )
        for found in sorted(
            path.name
            for path in dataset.get_root("tracks_raw").iterdir()
            if path.is_dir()
        )
    ]
    _ = dataset.write_tracks_raw_index(rows, patterns=pattern)


def write_local_tracks(
    dataset: Dataset, sequence: str, frames: int, *, start: int = 0
) -> None:
    """Index a CSV of one animal over *frames* frames from *start*, for *sequence*."""
    raw = dataset.get_root("tracks_raw") / sequence
    raw.mkdir(parents=True, exist_ok=True)
    frame = np.arange(start, start + frames)
    pd.DataFrame(
        {
            "frame": frame,
            "time": frame / RATE,
            "id": 0,
            "X": frame.astype(float),
            "Y": 1.0,
        }
    ).to_csv(raw / f"{sequence}.csv", index=False)
    index_tracks_raw_dirs(dataset, LOCAL_CSV, "*.csv")


def write_sleap_tracks(dataset: Dataset, sequence: str, frames: int) -> None:
    """Index a SLEAP analysis file of one animal over *frames* frames, for *sequence*."""
    path = dataset.get_root("tracks_raw") / sequence / f"{sequence}.h5"
    write_sleap_analysis_h5(path, np.random.default_rng(0).random((frames, 1, 1, 2)))
    index_tracks_raw_dirs(dataset, "sleap_analysis_h5", "*.h5")


def converted(dataset: Dataset, sequence: str) -> pd.DataFrame:
    """The converted table of *sequence*."""
    return dataset.load_tracks(RECORDING, sequence)


def test_a_converted_table_lands_at_its_division_on_both_axes(
    tmp_path: Path, requires_ffmpeg: None, local_csv: str
) -> None:
    dataset = divided_dataset(tmp_path, (5, 5, 5, 5, 2))
    placed = divisions(dataset)
    for sequence, _, count in placed:
        write_local_tracks(dataset, sequence, count)

    outcome = dataset.convert_all_tracks(source_format=local_csv)

    assert outcome.failed == 0
    for sequence, first_frame, count in placed:
        table = converted(dataset, sequence)
        assert list(table["frame"]) == list(range(first_frame, first_frame + count))
        assert list(table["time"]) == pytest.approx(
            [frame / RATE for frame in range(first_frame, first_frame + count)]
        )
    rows = read_tracks_index(dataset)
    extents = sorted(
        zip(
            rows["frame_min"].to_numpy(np.int64).tolist(),
            rows["frame_max"].to_numpy(np.int64).tolist(),
            strict=True,
        )
    )
    assert extents == [(0, 9), (10, 19), (20, 21)]


def test_time_counts_the_gap_before_a_division(
    tmp_path: Path, requires_ffmpeg: None, local_csv: str
) -> None:
    """Chunk 2 starts 100 s after chunk 1 ends, which ends the first run."""
    times = np.arange(22, dtype=np.float64) / RATE
    times[10:] += 100.0
    dataset = divided_dataset(tmp_path, (5, 5, 5, 5, 2), frame_times=times.tolist())
    (_, first), (later, second), (last, third) = [
        (sequence, (start, count)) for sequence, start, count in divisions(dataset)
    ]
    write_local_tracks(dataset, later, second[1])
    write_local_tracks(dataset, last, third[1])

    _ = dataset.convert_all_tracks(source_format=local_csv)

    table = converted(dataset, later)
    assert first == (0, 10)
    assert table["frame"].iloc[0] == 10
    assert table["time"].iloc[0] == pytest.approx(times[10])
    assert converted(dataset, last)["time"].iloc[0] == pytest.approx(times[20])


def test_time_continues_across_a_closed_boundary_at_the_division_rate(
    tmp_path: Path, requires_ffmpeg: None, local_csv: str
) -> None:
    """Frame 10 is stamped 0.05 s late, short of a gap, and the axis keeps its rate."""
    times = np.arange(22, dtype=np.float64) / RATE
    times[10:] += 0.05
    dataset = divided_dataset(tmp_path, (5, 5, 5, 5, 2), frame_times=times.tolist())
    placed = divisions(dataset)
    for sequence, _, count in placed:
        write_local_tracks(dataset, sequence, count)

    _ = dataset.convert_all_tracks(source_format=local_csv)

    for sequence, first_frame, count in placed:
        assert list(converted(dataset, sequence)["time"]) == pytest.approx(
            [frame / RATE for frame in range(first_frame, first_frame + count)]
        )


def test_a_sleap_table_of_a_division_is_timed_at_the_stores_rate(
    tmp_path: Path, requires_ffmpeg: None
) -> None:
    """The measured rate times each division's table and the division's offset agrees."""
    dataset = divided_dataset(tmp_path, (5, 5, 5, 5, 2))
    placed = divisions(dataset)
    for sequence, _, count in placed:
        write_sleap_tracks(dataset, sequence, count)

    outcome = dataset.convert_all_tracks(source_format="sleap_analysis_h5")

    assert outcome.ok
    for sequence, first_frame, count in placed:
        table = converted(dataset, sequence)
        assert list(table["frame"]) == list(range(first_frame, first_frame + count))
        assert list(table["time"]) == pytest.approx(
            [frame / RATE for frame in range(first_frame, first_frame + count)]
        )


def test_a_table_numbered_on_the_recording_axis_is_refused(
    tmp_path: Path, requires_ffmpeg: None, local_csv: str
) -> None:
    dataset = divided_dataset(tmp_path, (5, 5, 5, 5, 2))
    sequence, first_frame, count = divisions(dataset)[1]
    write_local_tracks(dataset, sequence, count, start=first_frame)

    outcome = dataset.convert_all_tracks(source_format=local_csv)

    assert outcome.failed == 1
    assert read_tracks_index(dataset).empty


def test_an_inference_table_lands_at_each_divisions_first_frame(
    tmp_path: Path,
    requires_ffmpeg: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = divided_dataset(tmp_path, (5, 5, 5, 5, 2))
    _ = install_fake_pose_inference(monkeypatch, pose_per_frame)
    model = tmp_path / "weights" / "best.pt"
    model.parent.mkdir(parents=True)
    _ = model.write_bytes(b"weights")

    _ = run_op(dataset, "infer-pose", {"model": str(model)})

    rows = read_tracks_index(dataset)
    found = sorted(
        zip(
            rows["first_frame"].to_numpy(np.int64).tolist(),
            rows["frame_min"].to_numpy(np.int64).tolist(),
            rows["frame_max"].to_numpy(np.int64).tolist(),
            strict=True,
        )
    )
    assert found == [(0, 0, 9), (10, 10, 19), (20, 20, 21)]


def label_frames(
    dataset: Dataset, source_format: str, kind: str = "behavior"
) -> list[int]:
    """The event frames of the one entry that *source_format* converted."""
    rows = read_labels_index(dataset, kind)
    (path,) = rows.loc[rows["run_id"].str.contains(source_format), "abs_path"]
    with np.load(dataset.resolve_path(str(path)), allow_pickle=True) as npz:
        return np.asarray(npz["frames"], dtype=np.int64).tolist()


def test_labels_on_the_tracks_axis_keep_their_frames_and_others_move(
    tmp_path: Path, requires_ffmpeg: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every conversion labels the second division, which starts at frame 10."""
    monkeypatch.setitem(
        LABEL_CONVERTERS, (TracksAxisEntries.src_format, "behavior"), TracksAxisEntries
    )
    dataset = divided_dataset(tmp_path, (5, 5, 5, 5, 2))
    sequence, first_frame, _ = divisions(dataset)[1]
    on_tracks = write_listed_entries(
        tmp_path / "tracks_axis", {"predictions": [(RECORDING, sequence)]}
    )
    calms21 = tmp_path / "calms21"
    calms21.mkdir()
    annotations = {RECORDING: {sequence: {"annotations": [0, 1, 2]}}}
    np.save(calms21 / "labels.npy", np.array(annotations, dtype=object))
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    pd.DataFrame(
        {
            "group": [RECORDING] * 2,
            "sequence": [sequence] * 2,
            "frame": [12, 15],
            "predicted_label": [0, 1],
        }
    ).to_csv(predictions / "own.csv", index=False)
    dataset.index_labels_raw(
        [on_tracks], patterns=["*.npy"], src_format=TracksAxisEntries.src_format
    )
    dataset.index_labels_raw([calms21], patterns=["*.npy"], src_format="calms21_npy")
    dataset.index_labels_raw(
        [predictions], patterns=["*.csv"], src_format="predictions_table"
    )

    for source_format in (TracksAxisEntries.src_format, "calms21_npy"):
        _ = dataset.convert_all_labels(kind="behavior", source_format=source_format)
    _ = dataset.convert_all_labels(
        kind="predictions", source_format="predictions_table"
    )

    assert first_frame == 10
    assert label_frames(dataset, TracksAxisEntries.src_format) == list(LISTED_FRAMES)
    assert label_frames(dataset, "calms21_npy") == [10, 11, 12]
    assert label_frames(dataset, "predictions_table", "predictions") == [12, 15]


def test_event_seconds_of_a_division_are_read_on_its_own_clock(
    tmp_path: Path,
    requires_ffmpeg: None,
    local_csv: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seconds 1.0 to 1.5 of the second division are 2.5 s later on the recording.

    The tracks of a division are timed on the recording's clock, which a division's
    events are not, and the frames the clock gives are already on the recording's
    axis. A comparison against the raw track times would put the whole event on the
    division's first frame.
    """
    monkeypatch.setitem(
        LABEL_CONVERTERS, (SecondsEntries.src_format, "behavior"), SecondsEntries
    )
    dataset = divided_dataset(tmp_path, (5, 5, 5, 5, 2))
    placed = divisions(dataset)
    for sequence, _, count in placed:
        write_local_tracks(dataset, sequence, count)
    _ = dataset.convert_all_tracks(source_format=local_csv)
    (first, _, _), (second, second_start, _), _ = placed
    listed = write_listed_entries(
        tmp_path / "seconds", {"events": [(RECORDING, first), (RECORDING, second)]}
    )
    dataset.index_labels_raw(
        [listed], patterns=["*.npy"], src_format=SecondsEntries.src_format
    )

    _ = dataset.convert_all_labels(
        kind="behavior", source_format=SecondsEntries.src_format
    )

    rows = read_labels_index(dataset, "behavior")
    frames: dict[str, list[int]] = {}
    for sequence, path in zip(rows["sequence"], rows["abs_path"], strict=True):
        with np.load(dataset.resolve_path(str(path)), allow_pickle=True) as npz:
            frames[str(sequence)] = np.asarray(npz["frames"], dtype=np.int64).tolist()
    assert second_start == 10
    assert frames[first] == [4, 5, 6]
    assert frames[second] == [14, 15, 16]
    roots = {str(cell) for cell in rows["consumed_source_roots"]}
    assert all("tracks_raw" in root for root in roots)


def neighbors(dataset: Dataset) -> list[tuple[str, str]]:
    """Each pair of sequences that one entry's next neighbor makes."""
    manifest, _ = build_manifest(dataset, Inputs(("tracks",)))
    return sorted(
        (entry.sequence, manifest[entry.next_entry_key].sequence)
        for entry in manifest.values()
        if entry.next_entry_key is not None
    )


def test_overlap_reads_across_a_closed_boundary_and_not_across_a_gap(
    tmp_path: Path, requires_ffmpeg: None, local_csv: str
) -> None:
    times = np.arange(32, dtype=np.float64) / RATE
    times[20:] += 100.0
    dataset = divided_dataset(
        tmp_path, (5, 5, 5, 5, 5, 5, 2), frame_times=times.tolist()
    )
    placed = divisions(dataset)
    for sequence, _, count in placed:
        write_local_tracks(dataset, sequence, count)
    _ = dataset.convert_all_tracks(source_format=local_csv)

    found = neighbors(dataset)

    tokens = [sequence for sequence, *_ in placed]
    assert [start for _, start, _ in placed] == [0, 10, 20, 30]
    assert found == [(tokens[0], tokens[1]), (tokens[2], tokens[3])]


def test_overlap_does_not_read_across_a_refused_division(
    tmp_path: Path, requires_ffmpeg: None, local_csv: str
) -> None:
    """Chunk 2 has no index, so the division of chunks 2-3 is refused."""
    with pytest.warns(UserWarning, match="missing index for imgstore chunk 2"):
        dataset = divided_dataset(tmp_path, (5, 5, 5, 5, 5, 2), missing_index={2})
    placed = divisions(dataset)
    for sequence, _, count in placed:
        write_local_tracks(dataset, sequence, count)
    _ = dataset.convert_all_tracks(source_format=local_csv)

    assert [start for _, start, _ in placed] == [0, 20]
    assert neighbors(dataset) == []
