"""Test a media variant tracked by its pixels and read back in its entry's frames.

A two-clip entry shows one bright dot moving along a known path. The ``preprocess`` op
crops, trims and decimates it into a variant, and ``infer-localizer`` runs over the
variant with a locator in place of a trained model. The locator decodes the file that it
is handed and reports each frame's brightest pixel. Every coordinate in the published
table is read from the variant's pixels, and must come back as the dot's path in the
entry's own frames and pixels. A feature's frame range over that table counts in the
same source frames.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from mosaic_media import MediaFacts

import mosaic.tracking.pose_training.localizer_inference as localizer_inference
from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
from mosaic.core.dataset import Dataset
from mosaic.core.media.video_io import open_frame_reader
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.preprocess_layout import media_variant_path
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.scope import Scope
from mosaic.tracking.pose_training.localizer_inference import LocalizerFrame

from tests.helpers import dot_image, make_dataset, write_painted_entry

pytestmark = pytest.mark.media

_SIZE = (96, 72)
_CLIP_FRAMES = 60
_FPS = 30.0


def _dot(frame: int) -> tuple[int, int]:
    """Return the dot's center in source frame *frame*, as ``(x, y)``.

    The dot moves one column right per frame and wraps every 60 frames, and one
    row down every fourth frame. Every frame of the entry has a distinct position.
    """
    return 10 + frame % 60, 20 + (frame // 4) % 30


_CROP_X, _CROP_Y = 6, 14
_STEPS: list[dict[str, object]] = [
    {"step": "crop", "x": _CROP_X, "y": _CROP_Y, "width": 72, "height": 44},
    {"step": "trim", "start": 7, "stop": 110},
    {"step": "decimate", "every": 3},
]
"""A 72x44 window at (6, 14) with the path, every third frame from 7 to 109.

The kept frames step over the boundary between the two clips, from 58 to 61.
"""

_KEPT = list(range(7, 110, 3))
"""The source frames in the variant, in order."""


@dataclass
class _BrightestPixel:
    """Stand in for the localizer's model by locating each frame's brightest pixel.

    It decodes the video that it is handed, as the localizer does, and records
    each path. A test can then identify the file that the reported coordinates
    were read from.
    """

    videos: list[Path] = field(default_factory=list)

    def run(
        self,
        _model_path: str,
        video_paths: Sequence[Path],
        *,
        facts: Sequence[MediaFacts] | None = None,
        **_kwargs: object,
    ) -> list[LocalizerFrame]:
        (video_path,) = video_paths
        self.videos.append(video_path)
        brightest = _brightest_pixels(video_path, facts[0] if facts else None)
        return [
            LocalizerFrame(
                frame,
                ({"x": float(x), "y": float(y), "confidence": 1.0, "class_id": 0},),
            )
            for frame, (x, y) in enumerate(brightest)
        ]


def _brightest_pixels(
    path: Path, facts: MediaFacts | None = None
) -> list[tuple[int, int]]:
    """Return the ``(x, y)`` of the brightest pixel of each frame of *path*."""
    found: list[tuple[int, int]] = []
    with open_frame_reader(path, facts=facts, target="analysis") as reader:
        for _, frame in reader:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            row, column = divmod(int(np.argmax(gray)), gray.shape[1])
            found.append((column, row))
    return found


@dataclass(frozen=True)
class _Tracked:
    """A dataset whose moving-dot entry was tracked on its variant."""

    ds: Dataset
    variant: str
    locator: _BrightestPixel


@pytest.fixture
def tracked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Tracked:
    ds = make_dataset(tmp_path / "ds")
    clips = write_painted_entry(
        ds,
        "s",
        [(_CLIP_FRAMES, _FPS), (_CLIP_FRAMES, _FPS)],
        lambda frame: dot_image(_SIZE, _dot(frame)),
        size=_SIZE,
    )
    # The source encode keeps the dot's brightest pixel on its path. Any departure
    # from the path below comes from the variant or its mapping.
    decoded = [position for clip in clips for position in _brightest_pixels(clip)]
    assert decoded == [_dot(frame) for frame in range(2 * _CLIP_FRAMES)]

    scope = Scope(entries=[("", "s")])
    variant = run_op(ds, "preprocess", {"steps": _STEPS}, scope=scope)
    locator = _BrightestPixel()
    monkeypatch.setattr(localizer_inference, "run_localizer_inference", locator.run)
    model = tmp_path / "localizer" / "best.pt"
    model.parent.mkdir()
    _ = model.write_bytes(b"weights")
    _ = run_op(
        ds, "infer-localizer", {"model": str(model), "media": variant}, scope=scope
    )
    return _Tracked(ds=ds, variant=variant, locator=locator)


def _published(ds: Dataset) -> pd.DataFrame:
    (path,) = read_tracks_index(ds)["abs_path"].tolist()
    table = pd.read_parquet(ds.resolve_path(str(path)))
    return table.sort_values("frame").reset_index(drop=True)


def test_the_published_table_is_the_dot_path_in_source_frames_and_pixels(
    tracked: _Tracked,
) -> None:
    ds, variant = tracked.ds, tracked.variant
    assert tracked.locator.videos == [media_variant_path(ds, variant, "", "s", "")]
    # In the variant's pixels the dot is up and left of its source position by
    # the crop offset.
    assert _brightest_pixels(tracked.locator.videos[0]) == [
        (x - _CROP_X, y - _CROP_Y) for x, y in map(_dot, _KEPT)
    ]

    table = _published(ds)

    assert table["frame"].tolist() == _KEPT
    positions = list(zip(table["X"].tolist(), table["Y"].tolist(), strict=True))
    assert positions == [_dot(frame) for frame in _KEPT]
    assert table["time"].to_numpy() == pytest.approx(table["frame"] / _FPS)


def test_a_feature_frame_range_selects_source_frames(tracked: _Tracked) -> None:
    """``[40, 80)`` names source frames, which the variant contains from 40 to 79.

    Counted in the variant's frames, the range would lie past its 35 frames and
    would not select a frame.
    """
    ds = tracked.ds

    result = run_feature(ds, SpeedAngvel(), filter_start_frame=40, filter_end_frame=80)

    assert result.run_id is not None
    run_root = feature_run_root(ds, result.feature, result.run_id)
    outputs = list(run_root.glob("*.parquet"))
    assert len(outputs) == 1, "the frame range selected no rows"
    table = pd.read_parquet(outputs[0]).sort_values("frame").reset_index(drop=True)
    selected = [frame for frame in _KEPT if 40 <= frame < 80]
    assert table["frame"].tolist() == selected
    # Each row's velocity is the dot's move since the row before, in source
    # pixels per second of source time. The first row lacks a predecessor.
    assert math.isnan(table["vx"].iloc[0]) and math.isnan(table["vy"].iloc[0])
    seconds = 3 / _FPS
    for axis, column in enumerate(("vx", "vy")):
        expected = [
            (_dot(frame)[axis] - _dot(frame - 3)[axis]) / seconds
            for frame in selected[1:]
        ]
        assert table[column].iloc[1:].tolist() == pytest.approx(expected), column
