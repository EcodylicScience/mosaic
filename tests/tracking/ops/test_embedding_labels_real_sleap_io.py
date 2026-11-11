"""The labels writer of ``prepare-embedding-data`` saves through the real sleap-io.

The stand-in ``sleap_io`` of the other tests neither decodes nor embeds. These
tests run the writer's snippet in the located SLEAP environment, read each shard
back there, and skip when that environment has no sleap-io.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from mosaic.tracking.ops.prepare_embedding import (
    EmbeddingLabelsSplit,
    EmbeddingSource,
    write_embedding_labels,
)
from mosaic.tracking.sleap.embedding_windows import (
    EmbeddingWindow,
    EntryWindows,
    WindowInstance,
)
from mosaic.tracking.sleap.labels import sleap_label_files
from mosaic.tracking.sleap.probe import probe_sleap
from mosaic.tracking.sleap.run import SleapNotFoundError, sleap_invocation
from tests.helpers import CHILD_TIMEOUT_SECONDS, write_mpeg4_mp4

_PREDICTIONS_SNIPPET = """
import sys

import numpy as np
import sleap_io

_video_path, _out, _n_frames = sys.argv[1], sys.argv[2], int(sys.argv[3])
_skeleton = sleap_io.Skeleton(["head", "tail"])
_video = sleap_io.Video.from_filename(_video_path)
_tracks = {name: sleap_io.Track(name=name) for name in ("a", "b")}
_frames = []
for _frame_idx in range(_n_frames):
    _instances = []
    for _slot, (_name, _track) in enumerate(_tracks.items()):
        _points = np.array([[10.0 + _slot, 10.0], [20.0 + _slot, 30.0]])
        _instances.append(
            sleap_io.PredictedInstance.from_numpy(
                _points, skeleton=_skeleton, track=_track, score=1.0
            )
        )
    _frames.append(
        sleap_io.LabeledFrame(video=_video, frame_idx=_frame_idx, instances=_instances)
    )
sleap_io.save_slp(
    sleap_io.Labels(
        labeled_frames=_frames,
        videos=[_video],
        skeletons=[_skeleton],
        tracks=list(_tracks.values()),
    ),
    _out,
)
"""


def _require_sleap_io() -> None:
    """Skip unless the located SLEAP environment imports sleap-io."""
    try:
        version = probe_sleap().sleap_io_version
    except SleapNotFoundError:
        pytest.skip("no SLEAP environment is located")
    if not version:
        pytest.skip("the SLEAP environment has no sleap-io")


# Run by the SLEAP interpreter: argv is the source video, then each shard. It
# prints, as JSON, a digest of every source frame and, per shard, each video's
# backend and source, and each labeled frame's video, index, tracks and digest.
_READ_SHARDS_SNIPPET = """
import hashlib
import json
import sys

import numpy as np
import sleap_io


def _digest(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


_source = sleap_io.Video.from_filename(sys.argv[1])
_frames = [_digest(_source[_index]) for _index in range(len(_source))]
_source.close()
_shards = []
for _path in sys.argv[2:]:
    _labels = sleap_io.load_slp(_path)
    _written = []
    for _frame in _labels.labeled_frames:
        _written.append(
            {
                "video": _labels.videos.index(_frame.video),
                "frame_idx": int(_frame.frame_idx),
                "tracks": sorted(_each.track.name for _each in _frame.instances),
                "user": all(
                    not isinstance(_each, sleap_io.PredictedInstance)
                    for _each in _frame.instances
                ),
                "image": _digest(_frame.image),
            }
        )
    _videos = []
    for _video in _labels.videos:
        _named = None if _video.source_video is None else _video.source_video.filename
        _videos.append(
            {
                "backend": type(_video.backend).__name__,
                "filename": _video.filename,
                "source": _named,
            }
        )
    _shards.append({"videos": _videos, "frames": _written})
print(json.dumps({"source": _frames, "shards": _shards}))
"""


class _WrittenFrame(BaseModel):
    video: int
    frame_idx: int
    tracks: list[str]
    user: bool
    image: str


class _WrittenVideo(BaseModel):
    backend: str
    filename: str
    source: str | list[str] | None


class _Shard(BaseModel):
    videos: list[_WrittenVideo]
    frames: list[_WrittenFrame]


class _ReadBack(BaseModel):
    source: list[str]
    shards: list[_Shard]


def _read_back(video: Path, shards: Sequence[Path]) -> _ReadBack:
    """Read *shards* in the SLEAP environment, with the frames of *video*."""
    completed = subprocess.run(
        [
            *sleap_invocation("python"),
            "-c",
            _READ_SHARDS_SNIPPET,
            str(video),
            *(str(shard) for shard in shards),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS,
    )
    return _ReadBack.model_validate_json(completed.stdout.strip().splitlines()[-1])


def _windows(entry_key: str, frames: Sequence[Sequence[int]]) -> EntryWindows:
    """One window per item of *frames*, each holding tracks ``a`` and ``b``."""
    windows = tuple(
        EmbeddingWindow(
            index=index,
            frames=tuple(each),
            instances=tuple(
                WindowInstance(
                    frame=frame,
                    track=track,
                    window_track=f"{entry_key}/w{index:05d}/{track}",
                )
                for frame in each
                for track in ("a", "b")
            ),
        )
        for index, each in enumerate(frames)
    )
    return EntryWindows(entry_key=entry_key, windows=windows)


def _source(
    tmp_path: Path, entry_key: str, n_frames: int, frames: Sequence[Sequence[int]]
) -> tuple[EmbeddingSource, Path]:
    """A source of *n_frames* frames, shaded by frame, and its video."""
    video = tmp_path / "media" / f"{entry_key}.mp4"
    write_mpeg4_mp4(video, frames=n_frames, shade="by-frame")
    predictions = tmp_path / "work" / f"{entry_key}.predictions.slp"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    _ = subprocess.run(
        [
            *sleap_invocation("python"),
            "-c",
            _PREDICTIONS_SNIPPET,
            str(video),
            str(predictions),
            str(n_frames),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS,
    )
    source = EmbeddingSource(
        predictions=predictions, windows=_windows(entry_key, frames)
    )
    return source, video


def _assert_planned(read: _ReadBack, source: EmbeddingSource) -> None:
    """Each window is one video of its planned frames, tracks and pixels, in order."""
    written = [
        (shard.videos[frame.video], frame)
        for shard in read.shards
        for frame in shard.frames
    ]
    planned = [
        (window, position, frame)
        for window in source.windows.windows
        for position, frame in enumerate(window.frames)
    ]
    assert len(written) == len(planned)
    entry = source.windows.entry_key
    for (video, frame), (window, position, source_frame) in zip(
        written, planned, strict=True
    ):
        assert video.backend == "HDF5Video"
        assert frame.frame_idx == position
        assert frame.user
        assert frame.tracks == [
            f"{entry}/w{window.index:05d}/a",
            f"{entry}/w{window.index:05d}/b",
        ]
        assert frame.image == read.source[source_frame]


def test_each_split_is_written_as_shards_of_embedded_windows(tmp_path: Path) -> None:
    _require_sleap_io()
    windows = ((0, 2), (4, 6), (8, 11))
    train, train_video = _source(tmp_path, "s1", 12, windows)
    val, val_video = _source(tmp_path, "s2", 12, windows)
    labels = tmp_path / "out" / "labels"
    splits = [
        EmbeddingLabelsSplit(out=labels / "train", sources=(train,)),
        EmbeddingLabelsSplit(out=labels / "val", sources=(val,)),
    ]

    write_embedding_labels(splits, tmp_path / "out" / "labels.partial", max_frames=4)

    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == ["labels"]
    found = sleap_label_files(labels)
    assert [path.name for path in found.train] == ["shard-00000.slp", "shard-00001.slp"]
    assert [path.name for path in found.validation] == [
        "shard-00000.slp",
        "shard-00001.slp",
    ]
    for shards, source, video in (
        (found.train, train, train_video),
        (found.validation, val, val_video),
    ):
        read = _read_back(video, shards)
        assert [len(shard.videos) for shard in read.shards] == [2, 1]
        _assert_planned(read, source)


def test_sixty_windows_are_decoded_once_into_bounded_shards(tmp_path: Path) -> None:
    """No shard video reads the source video: each embeds the images of its window."""
    _require_sleap_io()
    windows = [(2 * index, 2 * index + 1) for index in range(60)]
    source, video = _source(tmp_path, "s1", 120, windows)
    labels = tmp_path / "out" / "labels"
    lines: list[str] = []

    write_embedding_labels(
        [EmbeddingLabelsSplit(out=labels / "train", sources=(source,))],
        tmp_path / "out" / "labels.partial",
        max_frames=8,
        on_activity=lines.append,
    )

    shards = sleap_label_files(labels).train
    assert len(shards) == 15
    assert [line.strip() for line in lines if line.startswith("decoded ")] == [
        f"decoded 120 frames from {video}"
    ]
    read = _read_back(video, shards)
    assert [len(shard.videos) for shard in read.shards] == [4] * 15
    for shard, path in zip(read.shards, shards, strict=True):
        for each in shard.videos:
            assert each.backend == "HDF5Video"
            assert each.filename == str(path)
            assert isinstance(each.source, list)
            assert len(each.source) == 2
            assert all(Path(name).suffix == ".png" for name in each.source)
    _assert_planned(read, source)
