"""Mosaic's Ultralytics runner reads an entry's files directly, with no join or export.

Ultralytics tracking, ``infer-pose`` and ``infer-points`` run in mosaic's runner
program, which decodes frames itself. It is handed the entry's files in order and
reads them on one frame axis: an entry's clips, and an imgstore's chunk files when
those hold the frames mosaic reads. So nothing is copied before a run. A store
whose chunks are not those frames is still read through its ``export-store``
video.

What makes this safe is that the runner reads the frames, the frame numbers and
the batches that it would read from the join, and the frames that mosaic's own
store reader reads. The runner program runs in this process, on real clips and
stores, against stand-in weights.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from mosaic_media.io import VideoReader

import mosaic.tracking.common.tool_input as tool_input
import mosaic.tracking.ultralytics_track.dataset_runs as ultralytics_runs
from mosaic.core.dataset import Dataset
from mosaic.core.media.imgstore_native import NativeStore
from mosaic.core.media.read_target import verified_read_facts
from mosaic.core.media.video_io import open_frame_reader
from mosaic.core.pipeline.joined_export import JOINED_KIND_DIRECTORY
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.store_export import StoreExportParams, readable_chunks
from mosaic.core.pipeline.transcode import TRANSCODE_KIND_DIRECTORY
from mosaic.core.scope import Scope
from mosaic.core.track_library.ultralytics_tracks import raw_columns
from mosaic.tracking.common.tool_input import (
    StoreExportMissingError,
    ToolFile,
    entry_runner_sources,
    required_media_ops,
)
from mosaic.tracking.common.ultralytics_env import request_sources
from mosaic.tracking.external.runner.ultralytics_protocol import (
    SourceWindow,
    TrackRequest,
    source_windows,
)
from mosaic.tracking.ultralytics_track.params import UltralyticsParams
from tests.helpers import (
    FakeUltralytics,
    MakeStore,
    MediaClip,
    gray_level,
    install_fake_pose_inference,
    install_fake_ultralytics,
    make_dataset,
    paint_gray,
    point_at_a_store,
    pose_per_frame,
    published_table,
    store_dataset,
    stub_join,
    write_media_index,
    write_painted_entry,
)
from tests.test_ultralytics_rows import FakeDetections, FakeResult

pytestmark = pytest.mark.tracker

MakeMediaDataset = Callable[[Path], Dataset]

_ENTRY = ("", "sess")
_CLIP_FRAMES = 10
_LEVEL_TOLERANCE = 4
"""How far a decoded painted frame's mean gray level may sit from the painted one."""

_STORE_FRAMES = 12
_STORE_CHUNK = 5


# --- the frame window, divided among the files -----------------------------


@pytest.mark.parametrize(
    ("counts", "start", "end", "step", "expected"),
    [
        ([10], 0, None, 1, [SourceWindow(0, 0, 0, 10)]),
        (
            [10, 10],
            5,
            None,
            4,
            [SourceWindow(0, 0, 5, 10), SourceWindow(1, 10, 3, 10)],
        ),
        (
            [5, 5, 5],
            7,
            None,
            1,
            [SourceWindow(1, 5, 2, 5), SourceWindow(2, 10, 0, 5)],
        ),
        (
            [5, 5, 5],
            0,
            6,
            1,
            [SourceWindow(0, 0, 0, 5), SourceWindow(1, 5, 0, 1)],
        ),
        ([5], -3, None, 0, [SourceWindow(0, 0, 0, 5)]),
        ([5, 5], 20, None, 1, []),
    ],
    ids=["one-file", "stride-across", "starts-later", "ends-early", "clamped", "past"],
)
def test_a_window_is_divided_among_the_files(
    counts: list[int],
    start: int,
    end: int | None,
    step: int,
    expected: list[SourceWindow],
) -> None:
    assert source_windows(counts, start=start, end=end, step=step) == expected


@pytest.mark.parametrize("counts", [[7], [7, 3], [4, 1, 6], [1, 1, 1, 1]])
@pytest.mark.parametrize("step", [1, 2, 3, 5])
@pytest.mark.parametrize("start", [0, 1, 4, 9])
@pytest.mark.parametrize("end", [None, 3, 8, 11])
def test_the_files_read_the_frames_their_join_reads(
    counts: list[int], step: int, start: int, end: int | None
) -> None:
    """Each window reads, file by file, exactly what one reader of the join reads."""
    total = sum(counts)
    joined = list(range(start, total if end is None else min(end, total), step))

    read: list[int] = []
    for window in source_windows(counts, start=start, end=end, step=step):
        assert window.end <= counts[window.index]
        frames = range(window.start, window.end, step)
        read.extend(window.first_frame + frame for frame in frames)

    assert read == joined


# --- the runner program over real clips ------------------------------------


@dataclass
class _Weights:
    """Weights that find one animal in every frame, recording what they are given.

    Attributes:
        task: What the weights declare themselves to be.
        loads: How many times the weights were loaded.
        batches: The mean gray level of each frame of each call, in order.
    """

    task: str = "detect"
    loads: int = 0
    batches: list[list[int]] = field(default_factory=list)

    def load(self, _path: str) -> _Weights:
        self.loads += 1
        return self

    def track(
        self, source: list[npt.NDArray[np.uint8]], **kwargs: object
    ) -> list[FakeResult]:
        assert kwargs["persist"] is True, "the tracker persists across calls"
        self.batches.append([round(float(frame.mean())) for frame in source])
        box = np.array([[1.0, 2.0, 5.0, 6.0, 1.0, 0.9, 0.0]])
        return [FakeResult(boxes=FakeDetections(box)) for _ in source]

    @property
    def levels(self) -> list[int]:
        """Every frame's mean gray level, over every call."""
        return [level for batch in self.batches for level in batch]


@pytest.fixture
def weights(monkeypatch: pytest.MonkeyPatch) -> _Weights:
    """Install an ``ultralytics`` package whose ``YOLO`` loads one :class:`_Weights`.

    ``ultralytics.utils`` holds the default configuration the runner reads to
    spell half precision, as a current release ships it.
    """
    loaded = _Weights()
    package = ModuleType("ultralytics")
    monkeypatch.setattr(package, "__path__", [], raising=False)
    monkeypatch.setattr(package, "YOLO", loaded.load, raising=False)
    utils = ModuleType("ultralytics.utils")
    monkeypatch.setattr(utils, "DEFAULT_CFG_DICT", {"quantize": None}, raising=False)
    monkeypatch.setitem(sys.modules, "ultralytics", package)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", utils)
    return loaded


def _painted_entry(
    tmp_path: Path, rates: tuple[float, ...]
) -> tuple[Dataset, list[Path]]:
    """One entry, ``sess``, of one painted clip of ten frames per rate in *rates*."""
    ds = make_dataset(tmp_path / "ds")
    clips = write_painted_entry(
        ds, "sess", [(_CLIP_FRAMES, rate) for rate in rates], paint_gray
    )
    return ds, clips


def _runner_files(
    ds: Dataset, group: str, sequence: str, kind: str = "ultralytics"
) -> tuple[ToolFile, ...]:
    """The files that *kind*'s runner is handed for one entry."""
    resolved = ds.resolve_media(group, sequence)
    return entry_runner_sources(
        ds, group, sequence, resolved.paths, resolved.facts, kind=kind
    )


def _track(
    runner_module: ModuleType,
    tmp_path: Path,
    files: tuple[ToolFile, ...],
    *,
    start_frame: int = 0,
    frame_step: int = 1,
) -> list[int]:
    """Run the runner's ``track`` over *files*, and return the frames it tracked."""
    request = TrackRequest(
        model_path="best.pt",
        sources=request_sources(
            [file.path for file in files], [file.facts for file in files]
        ),
        output_parquet=str(tmp_path / "predictions.parquet"),
        tracker_yaml=str(tmp_path / "tracker.yaml"),
        project_dir=str(tmp_path),
        columns=list(raw_columns(1)),
        n_keypoints=1,
        task="detect",
        conf=0.25,
        iou=0.7,
        imgsz=64,
        max_det=10,
        classes=None,
        agnostic_nms=False,
        device="cpu",
        precision="fp32",
        start_frame=start_frame,
        end_frame=None,
        frame_step=frame_step,
        batch_size=4,
        prefetch=True,
    )
    request_path = tmp_path / "track-request.json"
    _ = request_path.write_text(request.model_dump_json())
    response = tmp_path / "track-response.json"
    argv = ["track", "--request", str(request_path), "--out", str(response)]
    assert runner_module.main(argv) == 0
    table = pd.read_parquet(request.output_parquet)
    return [int(frame) for frame in table["frame"]]


@pytest.mark.media
@pytest.mark.parametrize("rates", [(30.0, 30.0), (25.0, 30.0)], ids=str)
def test_the_runner_tracks_an_entrys_clips_on_one_frame_axis(
    runner_module: ModuleType,
    weights: _Weights,
    tmp_path: Path,
    rates: tuple[float, float],
    requires_ffmpeg: None,
) -> None:
    """One load, one persisting tracker, and a batch that spans the boundary.

    Every frame's pixels are those of the clip frame that its entry frame names,
    so the second clip is numbered after the first, whatever the two rates.
    """
    ds, clips = _painted_entry(tmp_path, rates)
    files = _runner_files(ds, *_ENTRY)
    assert [file.path for file in files] == clips

    frames = _track(runner_module, tmp_path, files)

    assert frames == list(range(2 * _CLIP_FRAMES))
    assert weights.loads == 1
    assert [len(batch) for batch in weights.batches] == [4, 4, 4, 4, 4]
    for frame, level in zip(frames, weights.levels, strict=True):
        assert abs(level - gray_level(frame)) <= _LEVEL_TOLERANCE, frame


@pytest.mark.media
def test_a_window_on_the_entry_axis_reads_across_the_boundary(
    runner_module: ModuleType, weights: _Weights, tmp_path: Path, requires_ffmpeg: None
) -> None:
    ds, _clips = _painted_entry(tmp_path, (25.0, 30.0))

    files = _runner_files(ds, *_ENTRY)
    frames = _track(runner_module, tmp_path, files, start_frame=7, frame_step=3)

    assert frames == [7, 10, 13, 16, 19]
    for frame, level in zip(frames, weights.levels, strict=True):
        assert abs(level - gray_level(frame)) <= _LEVEL_TOLERANCE, frame


class _Batches(Protocol):
    """A reader of ``(indices, frames)`` batches, empty at the end."""

    def read_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]: ...


def _read_batches(
    reader: _Batches, batch_size: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every batch of *reader*, until it runs dry."""
    batches: list[tuple[np.ndarray, np.ndarray]] = []
    while True:
        indices, frames = reader.read_batch(batch_size)
        if len(indices) == 0:
            return batches
        batches.append((indices, frames))


@pytest.mark.media
def test_the_runner_reads_the_clips_as_it_reads_their_join(
    runner_module: ModuleType, tmp_path: Path, requires_ffmpeg: None
) -> None:
    """The same batches of frame numbers, and the same pixels, decoded either way."""
    ds, _clips = _painted_entry(tmp_path, (30.0, 30.0))
    _ = run_op(ds, "export-joined", {}, scope=Scope(entries=[_ENTRY]))
    (joined,) = sorted((ds.get_root("media") / JOINED_KIND_DIRECTORY).glob("*.mp4"))
    files = _runner_files(ds, *_ENTRY)

    with runner_module.EntryReader(
        request_sources([file.path for file in files], [file.facts for file in files]),
        start_frame=0,
        end_frame=None,
        frame_step=1,
    ) as entry_reader:
        from_clips = _read_batches(entry_reader, 4)
    join_facts = verified_read_facts(joined, None, "analysis")[0]
    with VideoReader(joined, facts=join_facts) as join_reader:
        from_join = _read_batches(join_reader, 4)

    assert len(from_clips) == len(from_join) == 5
    for (clip_indices, clip_frames), (join_indices, join_frames) in zip(
        from_clips, from_join, strict=True
    ):
        assert clip_indices.tolist() == join_indices.tolist()
        assert np.array_equal(clip_frames, join_frames), clip_indices.tolist()


# --- an imgstore, read as its chunk files ----------------------------------


def _store_entry(
    tmp_path: Path,
    make_media_dataset: MakeMediaDataset,
    make_imgstore: MakeStore,
    *,
    fmt: str = "avc1/mp4",
) -> tuple[Dataset, str, str, Path]:
    """A dataset of one store of 12 frames in chunks of 5, its entry and the store."""
    ds = store_dataset(
        tmp_path,
        make_media_dataset,
        make_imgstore,
        nframes=_STORE_FRAMES,
        chunksize=_STORE_CHUNK,
        fill=True,
        fmt=fmt,
    )
    (entry,) = ds.resolve_media_scope(None)
    return ds, entry.group, entry.sequence, entry.resolved.paths[0]


def _export_store(ds: Dataset, group: str, sequence: str) -> Path:
    """Run ``export-store`` over the entry, and return the export it wrote."""
    _ = run_op(
        ds,
        "export-store",
        StoreExportParams(),
        scope=Scope(entries=[(group, sequence)]),
    )
    (export,) = sorted((ds.get_root("media") / TRANSCODE_KIND_DIRECTORY).glob("*.mp4"))
    return export


def _nothing_copied(ds: Dataset) -> bool:
    """Whether no join and no store export were written."""
    media = ds.get_root("media")
    return not any(
        (media / directory).is_dir() and any((media / directory).iterdir())
        for directory in (JOINED_KIND_DIRECTORY, TRANSCODE_KIND_DIRECTORY)
    )


def test_a_store_lists_its_chunks_with_their_frame_counts(
    tmp_path: Path, make_media_dataset: MakeMediaDataset, make_imgstore: MakeStore
) -> None:
    _ds, _group, _sequence, store = _store_entry(
        tmp_path, make_media_dataset, make_imgstore
    )

    spans = readable_chunks(store)

    assert [count for _, count in spans] == [5, 5, 2]
    with NativeStore(store) as native:
        assert [path for path, _ in spans] == native.chunk_paths()
        assert sum(count for _, count in spans) == native.frame_count


def test_a_store_whose_chunks_are_not_video_has_none_to_read(
    tmp_path: Path, make_media_dataset: MakeMediaDataset, make_imgstore: MakeStore
) -> None:
    _ds, _group, _sequence, store = _store_entry(
        tmp_path, make_media_dataset, make_imgstore, fmt="npy"
    )

    assert readable_chunks(store) == []


@pytest.mark.media
def test_the_runner_reads_a_stores_chunks_as_mosaic_reads_the_store(
    runner_module: ModuleType,
    tmp_path: Path,
    make_media_dataset: MakeMediaDataset,
    make_imgstore: MakeStore,
) -> None:
    """Store frame ``i`` is entry frame ``i``, with the pixels mosaic's reader gives."""
    ds, group, sequence, store = _store_entry(
        tmp_path, make_media_dataset, make_imgstore
    )
    files = _runner_files(ds, group, sequence)
    assert [file.path for file in files] == [path for path, _ in readable_chunks(store)]

    with runner_module.EntryReader(
        request_sources([file.path for file in files], [file.facts for file in files]),
        start_frame=0,
        end_frame=None,
        frame_step=1,
    ) as entry_reader:
        batches = _read_batches(entry_reader, 4)
    with open_frame_reader(store, target="raw") as store_reader:
        expected = list(store_reader)

    read = [
        (int(index), frame)
        for indices, frames in batches
        for index, frame in zip(indices, frames, strict=True)
    ]
    assert [index for index, _ in read] == list(range(_STORE_FRAMES))
    for (index, frame), (store_index, store_frame) in zip(read, expected, strict=True):
        assert index == store_index
        assert np.array_equal(frame, store_frame), index


def _install_ultralytics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[FakeUltralytics, UltralyticsParams]:
    weights_file = tmp_path / "yolo" / "best.pt"
    weights_file.parent.mkdir(parents=True)
    _ = weights_file.write_bytes(b"weights")
    params = UltralyticsParams.model_validate({"model_path": str(weights_file)})
    return install_fake_ultralytics(monkeypatch), params


def test_ultralytics_tracks_a_store_from_its_chunks_with_no_export(
    tmp_path: Path,
    make_media_dataset: MakeMediaDataset,
    make_imgstore: MakeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ds, _group, _sequence, store = _store_entry(
        tmp_path, make_media_dataset, make_imgstore
    )
    fake, params = _install_ultralytics(tmp_path, monkeypatch)

    _ = ultralytics_runs.run_ultralytics(ds, params)

    assert fake.tracked == [tuple(path for path, _ in readable_chunks(store))]
    assert _nothing_copied(ds)


def test_infer_pose_reads_a_store_from_its_chunks_with_no_export(
    tmp_path: Path,
    make_media_dataset: MakeMediaDataset,
    make_imgstore: MakeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The table spans the store: each chunk's frames follow the chunks before it."""
    ds, group, sequence, store = _store_entry(
        tmp_path, make_media_dataset, make_imgstore
    )
    fake = install_fake_pose_inference(monkeypatch, pose_per_frame)
    model = tmp_path / "weights" / "best.pt"
    model.parent.mkdir()
    _ = model.write_bytes(b"weights")

    _ = run_op(
        ds,
        "infer-pose",
        {"model": str(model)},
        scope=Scope(entries=[(group, sequence)]),
    )

    assert fake.videos == [path for path, _ in readable_chunks(store)]
    frames = published_table(ds, "infer-pose")["frame"]
    assert sorted({int(frame) for frame in frames}) == list(range(_STORE_FRAMES))
    assert _nothing_copied(ds)


def test_a_store_whose_chunks_are_not_video_is_read_through_its_export(
    tmp_path: Path,
    make_media_dataset: MakeMediaDataset,
    make_imgstore: MakeStore,
    monkeypatch: pytest.MonkeyPatch,
    requires_ffmpeg: None,
) -> None:
    ds, group, sequence, _store = _store_entry(
        tmp_path, make_media_dataset, make_imgstore, fmt="npy"
    )
    fake, params = _install_ultralytics(tmp_path, monkeypatch)

    with pytest.raises(StoreExportMissingError, match="--kind export-store"):
        _ = ultralytics_runs.run_ultralytics(ds, params)
    assert fake.tracked == []

    export = _export_store(ds, group, sequence)
    _ = ultralytics_runs.run_ultralytics(ds, params)

    assert fake.tracked == [(export,)]


def test_chunks_that_disagree_with_the_stores_index_need_its_export(
    tmp_path: Path,
    make_media_dataset: MakeMediaDataset,
    make_imgstore: MakeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chunk measured short of its index cannot hold its frames on the store's axis.

    Its export is read instead, as for a store whose chunks are not video, and
    without one the run refuses naming the command.
    """
    ds, _group, _sequence, _store = _store_entry(
        tmp_path, make_media_dataset, make_imgstore
    )

    def overcounted(store: Path) -> list[tuple[Path, int]]:
        return [(path, count + 1) for path, count in readable_chunks(store)]

    monkeypatch.setattr(tool_input, "readable_chunks", overcounted)
    fake, params = _install_ultralytics(tmp_path, monkeypatch)

    with pytest.raises(StoreExportMissingError) as refused:
        _ = ultralytics_runs.run_ultralytics(ds, params)
    message = str(refused.value)
    assert "holds 5 frames by its own measure and 6 by the store's index" in message
    assert "--kind export-store" in message
    assert fake.tracked == []


def test_the_chunks_are_read_while_an_export_exists_and_a_tracked_entry_needs_neither(
    tmp_path: Path,
    make_media_dataset: MakeMediaDataset,
    make_imgstore: MakeStore,
    monkeypatch: pytest.MonkeyPatch,
    requires_ffmpeg: None,
) -> None:
    """An export changes nothing the runner reads, and the reuse gate names neither."""
    ds, group, sequence, store = _store_entry(
        tmp_path, make_media_dataset, make_imgstore
    )
    export = _export_store(ds, group, sequence)
    fake, params = _install_ultralytics(tmp_path, monkeypatch)

    _ = ultralytics_runs.run_ultralytics(ds, params)
    export.unlink()
    _ = ultralytics_runs.run_ultralytics(ds, params)

    assert fake.tracked == [tuple(path for path, _ in readable_chunks(store))]


# --- what has to be built before a tool can read an entry -----------------


def _two_clips(*, rates: tuple[float, float] = (30.0, 30.0)) -> list[MediaClip]:
    return [
        MediaClip(
            filename=f"c{order}.mp4",
            video_order=order,
            video_uuid=f"uid-{order}",
            fps=rate,
        )
        for order, rate in enumerate(rates)
    ]


@pytest.mark.parametrize(
    ("kind", "needed"),
    [
        ("trex", ("export-joined",)),
        ("sleap", ("export-joined",)),
        ("ultralytics", ()),
        ("infer-pose", ()),
        ("infer-localizer", ()),
    ],
)
def test_several_clips_need_a_join_only_for_a_tool_that_opens_one_file(
    tmp_path: Path, kind: str, needed: tuple[str, ...]
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, _two_clips())

    assert required_media_ops(ds, kind=kind) == ({_ENTRY: needed} if needed else {})

    _ = stub_join(ds, ["uid-0", "uid-1"])
    assert required_media_ops(ds, kind=kind) == {}


def test_clips_at_two_rates_need_a_join_for_the_localizer(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, _two_clips(rates=(30.0, 31.0)))

    assert required_media_ops(ds, kind="infer-localizer") == {
        _ENTRY: ("export-joined",)
    }
    assert required_media_ops(ds, kind="ultralytics") == {}


@pytest.mark.parametrize(
    ("kind", "needed"),
    [("trex", ("export-store",)), ("ultralytics", ()), ("infer-localizer", ())],
)
def test_a_video_store_needs_an_export_only_for_a_tool_that_opens_one_file(
    tmp_path: Path,
    make_media_dataset: MakeMediaDataset,
    make_imgstore: MakeStore,
    kind: str,
    needed: tuple[str, ...],
) -> None:
    ds, group, sequence, _store = _store_entry(
        tmp_path, make_media_dataset, make_imgstore
    )

    expected = {(group, sequence): needed} if needed else {}
    assert required_media_ops(ds, kind=kind) == expected


@pytest.mark.parametrize(
    ("kind", "needed"),
    [
        ("trex", ("export-store",)),
        ("ultralytics", ("export-store",)),
        ("infer-localizer", ()),
    ],
)
def test_a_store_whose_chunks_are_not_video_needs_an_export_for_every_runner(
    tmp_path: Path, kind: str, needed: tuple[str, ...]
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip()])
    _ = point_at_a_store(ds, "sess", ds.get_root("media_raw") / "sess.store")

    assert required_media_ops(ds, kind=kind) == ({_ENTRY: needed} if needed else {})


def test_a_built_export_needs_nothing(
    tmp_path: Path,
    make_media_dataset: MakeMediaDataset,
    make_imgstore: MakeStore,
    requires_ffmpeg: None,
) -> None:
    ds, group, sequence, _store = _store_entry(
        tmp_path, make_media_dataset, make_imgstore
    )
    _ = _export_store(ds, group, sequence)

    assert required_media_ops(ds, kind="trex") == {}
