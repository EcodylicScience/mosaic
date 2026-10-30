"""The labels writer of ``prepare-embedding-data`` saves through the real sleap-io.

The stand-in ``sleap_io`` of the other tests accepts any output name. The real
``save_slp(..., embed=True)`` reads a name ending in a video extension as a video,
so the name of the staged file matters. These tests run the writer's snippet in
the located SLEAP environment and skip when that environment has no sleap-io.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import h5py
import pytest

from mosaic.tracking.ops.prepare_embedding import (
    EmbeddingLabelsFile,
    EmbeddingSource,
    write_embedding_labels,
)
from mosaic.tracking.sleap.embedding_windows import (
    EmbeddingWindow,
    EntryWindows,
    WindowInstance,
)
from mosaic.tracking.sleap.probe import probe_sleap
from mosaic.tracking.sleap.run import SleapNotFoundError, sleap_invocation
from tests.helpers import write_mpeg4_mp4

_N_FRAMES = 10

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


def _windows(entry_key: str) -> EntryWindows:
    """Two windows of two frames, each holding tracks ``a`` and ``b``."""
    windows = tuple(
        EmbeddingWindow(
            index=index,
            frames=frames,
            instances=tuple(
                WindowInstance(
                    frame=frame,
                    track=track,
                    window_track=f"{entry_key}/w{index:05d}/{track}",
                )
                for frame in frames
                for track in ("a", "b")
            ),
        )
        for index, frames in enumerate(((0, 2), (5, 7)))
    )
    return EntryWindows(entry_key=entry_key, windows=windows)


def _source(tmp_path: Path, entry_key: str) -> EmbeddingSource:
    video = tmp_path / "media" / f"{entry_key}.mp4"
    write_mpeg4_mp4(video, frames=_N_FRAMES)
    predictions = tmp_path / "work" / f"{entry_key}.predictions.slp"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    _ = subprocess.run(
        [
            *sleap_invocation("python"),
            "-c",
            _PREDICTIONS_SNIPPET,
            str(video),
            str(predictions),
            str(_N_FRAMES),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return EmbeddingSource(predictions=predictions, windows=_windows(entry_key))


def _embedded_videos(path: Path) -> int:
    """How many videos of the labels file hold their frames inside it."""
    with h5py.File(path, "r") as handle:
        return sum(1 for index in range(8) if f"video{index}/video" in handle)


def test_the_real_save_embeds_the_frames_of_both_files(tmp_path: Path) -> None:
    _require_sleap_io()
    out = tmp_path / "out"
    out.mkdir()
    files = [
        EmbeddingLabelsFile(out=out / "labels.slp", sources=(_source(tmp_path, "s1"),)),
        EmbeddingLabelsFile(
            out=out / "labels.val.slp", sources=(_source(tmp_path, "s2"),)
        ),
    ]

    write_embedding_labels(files, out / "request.json")

    assert sorted(path.name for path in out.iterdir()) == [
        "labels.slp",
        "labels.val.slp",
    ]
    for each in files:
        assert _embedded_videos(each.out) == 2
        loaded = probe_sleap(each.out)
        assert loaded.labels_load_error == ""
        assert loaded.n_labeled_frames == 4
        assert loaded.n_tracks == 4
