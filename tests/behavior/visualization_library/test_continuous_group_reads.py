"""The renderer and the crops read a continuous group's frames from the member's media.

A sequence of a continuous group has tracks on its recording's frame axis and
media of its frames only. Each reader subtracts the sequence's
``first_frame`` from a tracks frame to find the media frame, and refuses a frame
outside the sequence.

Each frame of the recording ``rec`` is painted with its recording frame:
``s1`` contains frames 0 to 4, and ``s2`` frames 5 to 8 in two clips.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from mosaic.behavior.visualization_library.egocentric_crop import EgocentricCrop
from mosaic.behavior.visualization_library.interaction_crop import (
    InteractionCropPipeline,
)
from mosaic.behavior.visualization_library.playback import play_video
from mosaic.behavior.visualization_library.video_stream import render_stream
from mosaic.core.dataset import Dataset
from mosaic.core.media.video_io import open_frame_reader
from mosaic.core.recording import FrameOutsideSequenceError
from tests.helpers import (
    add_tracks_table,
    make_dataset,
    read_frame_code,
    track_table,
    write_painted_recording,
)

pytestmark = pytest.mark.media

_SIZE = (64, 48)
_SECOND = range(5, 9)


@pytest.fixture
def recording(tmp_path: Path, requires_ffmpeg: None) -> Dataset:
    """The continuous group ``rec``, with ``s2``'s tracks at frames 5 to 8."""
    dataset = make_dataset(tmp_path / "ds")
    _ = write_painted_recording(
        dataset, "rec", [("s1", [(5, 30.0)]), ("s2", [(2, 30.0), (2, 30.0)])]
    )
    dataset.set_continuous_groups(["rec"])
    table = track_table("rec", "s2", n_rows=len(_SECOND), start_frame=_SECOND[0])
    add_tracks_table(dataset, "convert.a", _centered(table), std_format="mosaic_v1")
    return dataset


def _centered(table: pd.DataFrame) -> pd.DataFrame:
    """*table* with every individual at the center of the frame."""
    return table.assign(X=_SIZE[0] / 2, Y=_SIZE[1] / 2)


def _codes(
    frames: Iterable[tuple[int, npt.NDArray[np.uint8]]],
) -> list[tuple[int, int]]:
    """Each ``(frame, image)`` as ``(frame, the code painted in the image)``."""
    return [(frame, read_frame_code(image)) for frame, image in frames]


class TestRenderStream:
    def test_a_recording_frame_reads_its_member_frame(self, recording: Dataset) -> None:
        second = recording.resolve_media("rec", "s2")

        stream = render_stream(
            second.paths,
            {"per_frame": {}},
            start=6,
            end=7,
            facts=second.facts,
            first_frame=second.first_frame,
        )

        assert _codes(stream) == [(6, 6), (7, 7)]

    def test_a_start_before_the_sequence_reads_from_its_first_frame(
        self, recording: Dataset
    ) -> None:
        """The window is the part of the requested frames that the sequence contains."""
        second = recording.resolve_media("rec", "s2")

        stream = render_stream(
            second.paths,
            {"per_frame": {}},
            start=0,
            facts=second.facts,
            first_frame=second.first_frame,
        )

        assert _codes(stream) == [(frame, frame) for frame in _SECOND]

    @pytest.mark.parametrize(("start", "end"), [(0, 4), (9, None)])
    def test_a_window_outside_the_sequence_is_refused(
        self, recording: Dataset, start: int, end: int | None
    ) -> None:
        second = recording.resolve_media("rec", "s2")

        with pytest.raises(FrameOutsideSequenceError, match="5 to 8"):
            _ = render_stream(
                second.paths,
                {"per_frame": {}},
                start=start,
                end=end,
                facts=second.facts,
                first_frame=second.first_frame,
            )


class TestPlayVideo:
    def test_the_member_s_frames_are_drawn(
        self, recording: Dataset, tmp_path: Path
    ) -> None:
        written = play_video(
            recording,
            "rec",
            "s2",
            feature_runs={},
            label_kind=None,
            show_window=False,
            output_path=tmp_path / "s2.mp4",
        )

        assert written is not None
        with open_frame_reader(written, target="raw") as reader:
            assert sum(1 for _ in reader) == len(_SECOND)

    def test_a_window_outside_the_sequence_is_refused(self, recording: Dataset) -> None:
        with pytest.raises(FrameOutsideSequenceError, match="5 to 8"):
            _ = play_video(
                recording,
                "rec",
                "s2",
                feature_runs={},
                label_kind=None,
                show_window=False,
                start=0,
                end=2,
            )


def _egocentric(recording: Dataset, out: Path) -> EgocentricCrop:
    """A crop of the whole frame, written as lossless frames under *out*."""
    feature = EgocentricCrop(
        params={
            "crop_size": _SIZE,
            "target_id": 0,
            "center_mode": "xy",
            "rotate_to_heading": False,
            "output_mode": "frames",
            "output_root": str(out),
        }
    )
    feature.bind_dataset(recording)
    return feature


class TestEgocentricCrop:
    def test_each_tracks_frame_crops_its_member_frame(
        self, recording: Dataset, tmp_path: Path
    ) -> None:
        import cv2

        table = _centered(track_table("rec", "s2", n_rows=4, start_frame=5))
        metadata = _egocentric(recording, tmp_path / "out").transform(table)

        assert list(metadata["frame"]) == list(_SECOND)
        crops = tmp_path / "out" / "rec__s2" / "frames_id0"
        decoded = {
            int(path.stem.split("_")[1]): read_frame_code(
                np.asarray(cv2.imread(str(path)), dtype=np.uint8)
            )
            for path in sorted(crops.glob("frame_*.png"))
        }
        assert decoded == {frame: frame for frame in _SECOND}

    def test_a_tracks_frame_outside_the_sequence_is_refused(
        self, recording: Dataset, tmp_path: Path
    ) -> None:
        table = _centered(track_table("rec", "s2", n_rows=4, start_frame=3))

        with pytest.raises(FrameOutsideSequenceError, match="frame 3"):
            _ = _egocentric(recording, tmp_path / "out").transform(table)


def _interactions(start: int, end: int) -> pd.DataFrame:
    """Two individuals at the frame's center, interacting from *start* to *end*."""
    frames = [
        _centered(track_table("rec", "s2", n_rows=4, start_frame=5)).assign(id=0),
        _centered(track_table("rec", "s2", n_rows=4, start_frame=5)).assign(id=1),
    ]
    return pd.concat(frames, ignore_index=True).assign(
        id1=0,
        id2=1,
        perspective=lambda table: table["id"],
        interaction_id=0,
        interaction_start=start,
        interaction_end=end,
    )


def _interaction_crop(recording: Dataset, out: Path) -> InteractionCropPipeline:
    feature = InteractionCropPipeline(
        params={
            "crop_size": _SIZE,
            "center_mode": "xy",
            "rotate_to_heading": False,
            "grayscale": True,
            "crop_both_individuals": False,
            "output_root": str(out),
        }
    )
    feature.bind_dataset(recording)
    return feature


class TestInteractionCrop:
    def test_a_segment_crops_its_member_frames(
        self, recording: Dataset, tmp_path: Path
    ) -> None:
        clips = _interaction_crop(recording, tmp_path / "out").apply(
            _interactions(6, 7)
        )

        (clip,) = clips["video_path"]
        with open_frame_reader(tmp_path / "out" / str(clip), target="raw") as reader:
            assert [read_frame_code(image) for _, image in reader] == [6, 7]

    def test_a_segment_outside_the_sequence_is_refused(
        self, recording: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(FrameOutsideSequenceError, match="frame 9"):
            _ = _interaction_crop(recording, tmp_path / "out").apply(
                _interactions(6, 9)
            )
