"""A division of a store reads the frames of its window of chunks, from 0.

The store ``rec`` records 22 frames at 4 fps in chunks of 5, 5, 5, 5 and 2 frames,
each painted with its store frame, and is divided every 10 frames. Its divisions
are chunks 0-1, 2-3 and 4. Local frame ``k`` of a division is store frame
``chunk_first * 5 + k``, which is also its frame on the recording's axis.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from mosaic_media import MediaProbeError

from mosaic.behavior.visualization_library.egocentric_crop import EgocentricCrop
from mosaic.behavior.visualization_library.interaction_crop import (
    InteractionCropPipeline,
)
from mosaic.behavior.visualization_library.playback import play_video
from mosaic.behavior.visualization_library.video_stream import render_stream
from mosaic.core.dataset import Dataset, ResolvedMedia
from mosaic.core.media.imgstore_io import ImgStoreCapture, ImgStoreFrameReader
from mosaic.core.media.imgstore_native import NativeStore
from mosaic.core.media.store_division import StoreWindow
from mosaic.core.media.video_io import (
    MultiVideoReader,
    extract_candidate_features,
    open_frame_reader,
    read_entry_frames,
    read_frames_at,
)
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.preprocess import PreprocessParams
from mosaic.core.pipeline.preprocess_layout import media_variant_path
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.recording import FrameOutsideSequenceError
from mosaic.core.scope import Scope
from mosaic.tracking.common.scope import (
    JoinedSourceMismatchError,
    build_work_items,
    refuse_unjoinable,
)
from mosaic.tracking.common.tool_input import entry_runner_sources
from mosaic.tracking.frame_extraction.frame_runs import FrameFile
from mosaic.tracking.pose_training import localizer_inference
from mosaic.tracking.pose_training.inference import visualize_inference
from mosaic.tracking.prerequisites import plan_prerequisites
from tests.helpers import (
    DIVISION_CREATED_UTC,
    add_tracks_table,
    read_frame_code,
    sole_frames_row_files,
    store_dataset,
    track_table,
    video_store_maker,
    write_division_rows,
    write_h264_mp4,
)

pytestmark = pytest.mark.media

STORE_CHUNKS = (5, 5, 5, 5, 2)
CHUNKSIZE = 5
RATE = 4.0
SIZE = (64, 48)
SECOND = StoreWindow(2, 3)
"""The window of the second division, store frames 10 to 19."""
TOKENS = ("20210219T192253Z", "20210219T192255Z", "20210219T192258Z")
WINDOWS = (StoreWindow(0, 1), SECOND, StoreWindow(4, 4))


@pytest.fixture
def divided(tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]) -> Dataset:
    """The dataset whose store ``rec`` is divided into three sequences."""
    dataset = store_dataset(
        tmp_path,
        make_media_dataset,
        video_store_maker(STORE_CHUNKS, fps=RATE, created_utc=DIVISION_CREATED_UTC),
    )
    _ = write_division_rows(dataset, store_of(dataset), 2 * CHUNKSIZE)
    return dataset


def store_of(dataset: Dataset) -> Path:
    """The store of the :func:`divided` dataset."""
    return dataset.get_root("media_raw") / "recordings" / "rec"


def store_frames(window: StoreWindow, frames: int) -> list[int]:
    """The store frames that the first *frames* local frames of *window* are."""
    first = window.chunk_first * CHUNKSIZE
    return list(range(first, first + frames))


def codes(frames: Iterable[tuple[int, npt.NDArray[np.uint8]]]) -> list[int]:
    """The code painted in each image of *frames*."""
    return [read_frame_code(image) for _, image in frames]


def second(dataset: Dataset) -> ResolvedMedia:
    """The media of the second division."""
    return dataset.resolve_media("rec", TOKENS[1])


# --- the readers ---------------------------------------------------------------


@pytest.mark.parametrize("window", WINDOWS, ids=["first", "second", "last"])
def test_the_store_reader_counts_a_windows_frames_from_its_first(
    divided: Dataset, window: StoreWindow
) -> None:
    with NativeStore(store_of(divided), window) as native:
        frames = [native.frame(k) for k in range(native.frame_count)]
        count = native.frame_count

    expected = 2 if window.chunk_first == 4 else 10
    assert count == expected
    assert [read_frame_code(frame) for frame in frames] == store_frames(
        window, expected
    )


def test_every_reader_of_a_store_reads_a_window_from_its_first_frame(
    divided: Dataset,
) -> None:
    store = store_of(divided)
    expected = store_frames(SECOND, 10)

    capture = ImgStoreCapture(store, SECOND)
    capture.seek(3)
    _, seeked = capture.read()
    capture.close()
    assert seeked is not None and read_frame_code(seeked) == expected[3]
    with ImgStoreFrameReader(store, frame_step=2, window=SECOND) as reader:
        assert codes(reader) == expected[::2]
    with open_frame_reader(store, target="raw", window=SECOND) as reader:
        assert reader.frame_count == 10
        assert codes(reader) == expected
    assert codes(read_frames_at(store, [7, 1], window=SECOND)) == [
        expected[1],
        expected[7],
    ]
    assert codes(read_entry_frames(store, windows=[SECOND], target="raw")) == expected
    indices, _ = extract_candidate_features(
        store, 0, 9, 3, (8, 8), True, None, window=SECOND
    )
    assert indices.tolist() == [0, 3, 6, 9]


def test_a_multi_reader_places_each_window_after_the_one_before(
    divided: Dataset,
) -> None:
    store = store_of(divided)

    reader = MultiVideoReader(
        [store, store], windows=[WINDOWS[2], SECOND], target="analysis"
    )
    read = [reader.read()[1] for _ in range(reader.total_frames)]
    reader.close()

    assert [read_frame_code(image) for image in read if image is not None] == [
        20,
        21,
        *store_frames(SECOND, 10),
    ]


def test_a_read_outside_a_window_is_refused(divided: Dataset) -> None:
    store = store_of(divided)

    with NativeStore(store, SECOND) as native, pytest.raises(MediaProbeError):
        _ = native.frame(10)
    with pytest.raises(MediaProbeError, match="frame 10"):
        _ = codes(read_frames_at(store, [10], window=SECOND))
    with pytest.raises(FrameOutsideSequenceError):
        _ = second(divided).sequence_frame(20)
    with pytest.raises(MediaProbeError, match="no chunk 5"):
        _ = NativeStore(store, StoreWindow(4, 5))


def test_a_window_on_a_plain_video_is_refused(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    write_h264_mp4(clip, frames=4)

    with pytest.raises(ValueError, match="is not an imgstore"):
        _ = open_frame_reader(clip, target="raw", window=SECOND)
    with pytest.raises(ValueError, match="is not an imgstore"):
        _ = MultiVideoReader([clip], windows=[SECOND], target="raw")


def test_a_missing_interior_index_leaves_the_other_windows_intact(
    divided: Dataset,
) -> None:
    store = store_of(divided)
    (store / "000002.npz").unlink()

    for window in (WINDOWS[0], WINDOWS[2]):
        with NativeStore(store, window) as native:
            read = [native.frame(k) for k in range(native.frame_count)]
        assert [read_frame_code(frame) for frame in read] == store_frames(
            window, len(read)
        )
    with pytest.raises(MediaProbeError, match="chunk 2 .* has no index"):
        _ = NativeStore(store, SECOND)
    assert divided.resolve_media("rec", TOKENS[2]).first_frame == 20


# --- the dataset's consumers ------------------------------------------------------


@pytest.fixture
def tracked(divided: Dataset) -> Dataset:
    """The divided dataset, with the second division's tracks at frames 10 to 13."""
    table = track_table("rec", TOKENS[1], n_rows=4, start_frame=10).assign(
        X=SIZE[0] / 2, Y=SIZE[1] / 2
    )
    add_tracks_table(divided, "convert.a", table, std_format="mosaic_v1")
    return divided


def test_the_renderer_reads_a_divisions_frames_on_the_recording_axis(
    divided: Dataset,
) -> None:
    media = second(divided)

    stream = render_stream(
        media.paths,
        {"per_frame": {}},
        start=12,
        end=13,
        facts=media.facts,
        first_frame=media.first_frame,
        windows=media.windows,
    )

    read: list[tuple[int, npt.NDArray[np.uint8]]] = list(stream)
    assert [(frame, read_frame_code(image)) for frame, image in read] == [
        (12, 12),
        (13, 13),
    ]


def test_an_egocentric_crop_cuts_a_divisions_frames(
    tracked: Dataset, tmp_path: Path
) -> None:
    feature = EgocentricCrop(
        params={
            "crop_size": SIZE,
            "target_id": 0,
            "center_mode": "xy",
            "rotate_to_heading": False,
            "output_mode": "frames",
            "output_root": str(tmp_path / "out"),
        }
    )
    feature.bind_dataset(tracked)
    table = track_table("rec", TOKENS[1], n_rows=4, start_frame=10).assign(
        X=SIZE[0] / 2, Y=SIZE[1] / 2
    )

    _ = feature.transform(table)

    crops = sorted((tmp_path / "out").rglob("frame_*.png"))
    decoded = [
        read_frame_code(np.asarray(cv2.imread(str(path)), dtype=np.uint8))
        for path in crops
    ]
    assert decoded == [10, 11, 12, 13]


def test_an_interaction_crop_cuts_a_divisions_frames(
    tracked: Dataset, tmp_path: Path
) -> None:
    feature = InteractionCropPipeline(
        params={
            "crop_size": SIZE,
            "center_mode": "xy",
            "rotate_to_heading": False,
            "grayscale": True,
            "crop_both_individuals": False,
            "output_root": str(tmp_path / "out"),
        }
    )
    feature.bind_dataset(tracked)
    single = track_table("rec", TOKENS[1], n_rows=4, start_frame=10).assign(
        X=SIZE[0] / 2, Y=SIZE[1] / 2
    )
    pairs = pd.concat(
        [single.assign(id=0), single.assign(id=1)], ignore_index=True
    ).assign(
        id1=0,
        id2=1,
        perspective=lambda table: table["id"],
        interaction_id=0,
        interaction_start=11,
        interaction_end=12,
    )

    clips = feature.apply(pairs)

    (clip,) = clips["video_path"]
    with open_frame_reader(tmp_path / "out" / str(clip), target="raw") as reader:
        assert [read_frame_code(image) for _, image in reader] == [11, 12]


def test_extract_frames_writes_a_divisions_local_frames(divided: Dataset) -> None:
    divided.set_root("frames", "frames")
    _ = run_op(
        divided,
        "extract-frames",
        {
            "method": "list",
            "frames": [{"group": "rec", "sequence": TOKENS[1], "indices": [0, 7]}],
        },
        scope=Scope(entries=[("rec", TOKENS[1])]),
    )

    files: tuple[FrameFile, ...] = sole_frames_row_files(divided)
    images = [divided.resolve_path(file.image) for file in files]
    assert [file.frame_index for file in files] == [0, 7]
    assert [
        read_frame_code(np.asarray(cv2.imread(str(path)), dtype=np.uint8))
        for path in images
    ] == [10, 17]


def test_a_media_variant_holds_a_divisions_frames(divided: Dataset) -> None:
    run_id = run_op(
        divided,
        "preprocess",
        PreprocessParams(steps=[], codec="h264"),
        scope=Scope(entries=[("rec", TOKENS[1])]),
    )

    variant = media_variant_path(divided, run_id, "rec", TOKENS[1], "")
    with open_frame_reader(variant, target="raw") as reader:
        assert codes(reader) == store_frames(SECOND, 10)


def test_a_tracker_work_item_and_the_runner_read_a_divisions_chunks(
    divided: Dataset,
) -> None:
    scope = divided.resolve_media_scope([("rec", token) for token in TOKENS])
    items = build_work_items(divided, scope, kind="ultralytics").items

    assert [item.video_windows for item in items] == [(window,) for window in WINDOWS]
    assert [item.first_frame for item in items] == [0, 10, 20]
    middle = items[1]
    files = entry_runner_sources(
        divided,
        middle.group,
        middle.sequence,
        middle.video_paths,
        middle.source_facts,
        windows=middle.video_windows,
        kind="ultralytics",
    )
    store = store_of(divided)
    assert [file.path for file in files] == [
        store / "000002.mp4",
        store / "000003.mp4",
    ]
    assert [file.facts.frame_count for file in files] == [5, 5]


def test_each_division_plans_its_own_store_export(divided: Dataset) -> None:
    plan = plan_prerequisites(
        divided, "trex", entries=[("rec", token) for token in TOKENS]
    )

    exports = [run for run in plan.runs if run.kind == "export-store"]
    assert [run.entries for run in exports] == [(("rec", token),) for token in TOKENS]


def test_the_localizer_is_handed_a_divisions_frames(
    divided: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its table lands on the recording axis, at the division's first frame."""
    handed: list[int] = []

    def encoder(*_args: object, **_kwargs: object) -> object:
        return object()

    def detect(
        _model: object, frame: npt.NDArray[np.uint8], *_args: object, **_kwargs: object
    ) -> list[dict[str, float | int]]:
        handed.append(read_frame_code(frame))
        return [{"x": 1.0, "y": 2.0, "confidence": 0.9, "class_id": 0}]

    monkeypatch.setattr(localizer_inference, "_load_encoder", encoder)
    monkeypatch.setattr(localizer_inference, "detect_locations", detect)
    model = tmp_path / "localizer" / "best.pt"
    model.parent.mkdir()
    _ = model.write_bytes(b"weights")

    _ = run_op(
        divided,
        "infer-localizer",
        {"model": str(model)},
        scope=Scope(entries=[("rec", TOKENS[1])]),
    )

    assert handed == store_frames(SECOND, 10)
    (path,) = read_tracks_index(divided)["abs_path"].tolist()
    table = pd.read_parquet(divided.resolve_path(str(path)))
    assert sorted(set(table["frame"].tolist())) == store_frames(SECOND, 10)


@pytest.mark.filterwarnings("ignore:FFmpegVideoWriter's crf:DeprecationWarning")
def test_an_inference_overlay_draws_on_a_divisions_frames(
    divided: Dataset, tmp_path: Path
) -> None:
    store = store_of(divided)
    results: list[list[dict[str, float]]] = [[] for _ in range(10)]

    written = visualize_inference(
        store,
        results,
        result_type="localizer",
        output_path=tmp_path / "overlay.mp4",
        show_window=False,
        verbose=False,
        window=SECOND,
    )

    assert written is not None
    with open_frame_reader(written, target="raw") as reader:
        assert codes(reader) == store_frames(SECOND, 10)


def test_playback_draws_a_divisions_frames_on_the_recording_axis(
    tracked: Dataset, tmp_path: Path
) -> None:
    written = play_video(
        tracked,
        "rec",
        TOKENS[1],
        feature_runs={},
        label_kind=None,
        show_window=False,
        show_individual_bboxes=False,
        output_path=tmp_path / "played.mp4",
    )

    assert written is not None
    with open_frame_reader(written, target="raw") as reader:
        assert codes(reader) == store_frames(SECOND, 10)


def test_an_unjoinable_entry_of_two_divisions_is_named_by_its_first_row(
    divided: Dataset,
) -> None:
    """The remedy's command names the entry that the first division's row records."""
    second_media, last = second(divided), divided.resolve_media("rec", TOKENS[2])

    with pytest.raises(JoinedSourceMismatchError) as refused:
        refuse_unjoinable(
            divided,
            "trex",
            "elsewhere",
            "named",
            [*second_media.paths, *last.paths],
            [*second_media.facts, *last.facts],
            windows=[SECOND, WINDOWS[2]],
            hands_over_path=True,
        )

    assert f"--entries rec:{TOKENS[1]}" in str(refused.value)
