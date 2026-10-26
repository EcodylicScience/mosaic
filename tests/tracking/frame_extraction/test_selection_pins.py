"""Pins of the frames that extraction selects and of the manifest that records them.

The fixtures decode to the pixels that were written. The videos are lossless H.264, and
the stores hold raw arrays. Each read then computes the same k-means features, and the
picks recorded here repeat.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest

from mosaic.core.json_value import JsonValue
from mosaic.core.media.video_io import MultiVideoReader
from mosaic.tracking.frame_extraction.extraction import (
    FrameExtractionResult,
    MultiClipFeatures,
    SingleFileFeatures,
    extract_frames,
    extract_frames_multi,
)
from mosaic.tracking.frame_extraction.sampling import CandidateFeatures, Exclusion
from tests.helpers import paint_frame_code, require_ffmpeg, write_h264_mp4

MakeImgstore = Callable[..., tuple[Path, list[npt.NDArray[np.uint8]]]]

MANIFEST_KEYS = [
    "run_id",
    "method",
    "video_path",
    "output_dir",
    "manifest_path",
    "n_requested",
    "n_extracted",
    "selected_frame_indices",
    "start_frame",
    "end_frame",
    "candidate_step",
    "crop",
    "created_utc",
    "files",
    "video_meta",
    "sampling",
    "random_state",
]
SINGLE_FILE_META_KEYS = ["width", "height", "fps", "frame_count"]
MULTI_FILE_META_KEYS = [
    "width",
    "height",
    "fps",
    "total_frames",
    "video_count",
    "video_paths",
]

LEVELS = (130, 80, 110, 130, 110, 110, 80, 80, 130, 70, 130, 110, 20)
"""The flat gray level of each frame of the levels video: five levels in 13 frames."""

type PinnedVideo = Literal["coded", "levels"]
type PinnedMethod = Literal["uniform", "kmeans"]


def kmeans_record(
    effective_step: int, candidate_count: int, *, max_candidates: int | None = 5000
) -> list[tuple[str, JsonValue]]:
    """Return a k-means run's ``sampling`` record under the default settings, in order."""
    return [
        ("kmeans_resize", [64, 64]),
        ("kmeans_grayscale", True),
        ("kmeans_max_candidates", max_candidates),
        ("kmeans_effective_candidate_step", effective_step),
        ("kmeans_batch_size", 1024),
        ("kmeans_max_iter", 100),
        ("kmeans_n_init", "auto"),
        ("candidate_count", candidate_count),
    ]


@dataclass(frozen=True)
class Pin:
    """One extraction call and the picks and ``sampling`` record that it writes."""

    method: PinnedMethod
    n_frames: int
    selected: list[int]
    sampling: list[tuple[str, JsonValue]]
    video: PinnedVideo = "coded"
    start_frame: int | None = None
    end_frame: int | None = None
    candidate_step: int = 1
    crop: tuple[int, int, int, int] | None = None
    kmeans_max_candidates: int | None = 5000


@dataclass(frozen=True)
class Media:
    """The lossless videos that the pins read, written once per module."""

    root: Path
    coded: Path
    levels: Path
    clips: tuple[Path, Path]

    def video(self, name: PinnedVideo) -> Path:
        """Return the video that *name* names."""
        return self.coded if name == "coded" else self.levels


def _flat_level(frame: int) -> npt.NDArray[np.uint8]:
    return np.full((48, 64, 3), LEVELS[frame], np.uint8)


def _second_clip(frame: int) -> npt.NDArray[np.uint8]:
    return paint_frame_code(16 + frame)


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> Media:
    """Write ``coded.mp4``, ``levels.mp4`` and two clips of one coded sequence.

    Each of the 30 frames of ``coded.mp4`` spells its index in a grid of cells. The two
    clips hold the same 30 codes, 16 in the first and 14 in the second. ``levels.mp4``
    is flat at :data:`LEVELS`.
    """
    require_ffmpeg()
    root = tmp_path_factory.mktemp("pinned-media")
    coded = root / "coded.mp4"
    write_h264_mp4(coded, frames=30, paint=paint_frame_code, lossless=True)
    levels = root / "levels.mp4"
    write_h264_mp4(levels, frames=len(LEVELS), paint=_flat_level, lossless=True)
    first, second = root / "clip0.mp4", root / "clip1.mp4"
    write_h264_mp4(first, frames=16, paint=paint_frame_code, lossless=True)
    write_h264_mp4(second, frames=14, paint=_second_clip, lossless=True)
    return Media(root=root, coded=coded, levels=levels, clips=(first, second))


def _extract_single(
    source: Path, pin: Pin, output_dir: Path, excluded: Exclusion | None = None
) -> FrameExtractionResult:
    return extract_frames(
        source,
        n_frames=pin.n_frames,
        method=pin.method,
        start_frame=pin.start_frame,
        end_frame=pin.end_frame,
        candidate_step=pin.candidate_step,
        crop=pin.crop,
        kmeans_max_candidates=pin.kmeans_max_candidates,
        run_id="pinned",
        output_dir=output_dir,
        excluded=excluded,
    )


def _extract_multi(
    sources: Sequence[Path],
    pin: Pin,
    output_dir: Path,
    excluded: Exclusion | None = None,
) -> FrameExtractionResult:
    return extract_frames_multi(
        sources,
        n_frames=pin.n_frames,
        method=pin.method,
        start_frame=pin.start_frame,
        end_frame=pin.end_frame,
        candidate_step=pin.candidate_step,
        crop=pin.crop,
        kmeans_max_candidates=pin.kmeans_max_candidates,
        run_id="pinned",
        output_dir=output_dir,
        excluded=excluded,
    )


def _manifest(result: FrameExtractionResult) -> dict[str, JsonValue]:
    manifest: dict[str, JsonValue] = json.loads(Path(result.manifest_path).read_text())
    return manifest


def _assert_pinned(
    result: FrameExtractionResult,
    pin: Pin,
    meta_keys: list[str],
    exclude_keys: list[tuple[str, JsonValue]] | None = None,
) -> None:
    manifest = _manifest(result)
    assert list(manifest) == MANIFEST_KEYS
    assert manifest["selected_frame_indices"] == pin.selected
    assert result.selected_frame_indices == pin.selected
    assert manifest["n_requested"] == manifest["n_extracted"] == pin.n_frames
    sampling = manifest["sampling"]
    assert isinstance(sampling, dict)
    assert list(sampling.items()) == pin.sampling + (exclude_keys or [])
    video_meta = manifest["video_meta"]
    assert isinstance(video_meta, dict)
    assert list(video_meta) == meta_keys


def _normalized(
    result: FrameExtractionResult, *, media_root: Path, output_dir: Path
) -> dict[str, JsonValue]:
    """Return the manifest without ``created_utc`` and the measured frame rate.

    The media and output directories read ``<media>`` and ``<out>``. The rate is
    checked to be 30 frames per second, to within the container's time base.
    """
    text = Path(result.manifest_path).read_text()
    text = text.replace(str(media_root), "<media>").replace(str(output_dir), "<out>")
    manifest: dict[str, JsonValue] = json.loads(text)
    assert isinstance(manifest.pop("created_utc"), str)
    video_meta = manifest["video_meta"]
    assert isinstance(video_meta, dict)
    fps = video_meta.pop("fps")
    assert isinstance(fps, float)
    assert fps == pytest.approx(30.0, abs=1e-3)
    return manifest


def _files(selected: list[int], width: int = 64) -> list[JsonValue]:
    return [
        {"frame_index": i, "path": f"frame_{i:06d}.png", "width": width, "height": 48}
        for i in selected
    ]


SINGLE_FILE_PINS = [
    pytest.param(
        Pin(
            "uniform",
            5,
            [1, 7, 13, 19, 25],
            [],
            start_frame=1,
            end_frame=25,
            candidate_step=2,
        ),
        id="uniform-in-range",
    ),
    pytest.param(
        Pin("kmeans", 5, [11, 12, 17, 18, 21], kmeans_record(1, 30)), id="kmeans"
    ),
    pytest.param(
        Pin(
            "kmeans",
            4,
            [0, 4, 8, 12],
            kmeans_record(4, 8, max_candidates=8),
            crop=(0, 0, 32, 48),
            kmeans_max_candidates=8,
        ),
        id="kmeans-widened-step-cropped",
    ),
    pytest.param(
        Pin("kmeans", 6, [0, 1, 2, 3, 9, 12], kmeans_record(1, 13), video="levels"),
        id="kmeans-farthest-point-fill",
    ),
]
"""Single-file picks.

In ``kmeans-farthest-point-fill`` the clusters cover frames 0, 1, 2 and 12, four of the
five levels. The farthest-point fill then adds frame 9, the level farthest from them,
and frame 3, the first of the remaining frames, which all tie at distance zero.
"""

MULTI_FILE_PINS = [
    pytest.param(Pin("uniform", 5, [0, 7, 14, 22, 29], []), id="uniform"),
    pytest.param(
        Pin(
            "kmeans",
            5,
            [4, 10, 16, 18, 28],
            kmeans_record(2, 15),
            candidate_step=2,
        ),
        id="kmeans",
    ),
]

STORE_PINS = [
    pytest.param(Pin("uniform", 4, [0, 6, 13, 19], []), id="uniform"),
    pytest.param(Pin("kmeans", 4, [3, 9, 13, 18], kmeans_record(1, 20)), id="kmeans"),
]

STORE_SEQUENCE_PINS = [
    pytest.param(Pin("uniform", 4, [0, 6, 13, 19], []), id="uniform"),
    pytest.param(Pin("kmeans", 5, [0, 1, 2, 5, 7], kmeans_record(1, 20)), id="kmeans"),
    pytest.param(
        Pin("kmeans", 12, list(range(12)), kmeans_record(1, 20)),
        id="kmeans-farthest-point-fill",
    ),
]
"""Picks over two stores of ten frames each.

The stores hold the same ten frames, so the sequence has ten distinct frames. In
``kmeans-farthest-point-fill`` the clusters cover frames 0 to 8. The fill adds frame 9,
the one distinct frame left, then frames 10 and 11, the first of the remaining frames,
which all tie at distance zero.
"""


@pytest.mark.media
@pytest.mark.parametrize("pin", SINGLE_FILE_PINS)
def test_single_file_picks_are_pinned(media: Media, tmp_path: Path, pin: Pin) -> None:
    result = _extract_single(media.video(pin.video), pin, tmp_path)
    _assert_pinned(result, pin, SINGLE_FILE_META_KEYS)


@pytest.mark.media
@pytest.mark.parametrize("pin", MULTI_FILE_PINS)
def test_multi_clip_picks_are_pinned(media: Media, tmp_path: Path, pin: Pin) -> None:
    result = _extract_multi(media.clips, pin, tmp_path)
    _assert_pinned(result, pin, MULTI_FILE_META_KEYS)


@pytest.mark.parametrize("pin", STORE_PINS)
def test_imgstore_picks_are_pinned(
    make_imgstore: MakeImgstore, tmp_path: Path, pin: Pin
) -> None:
    store, _ = make_imgstore(name="rec", nframes=20, fill=True, parent=tmp_path / "in")
    result = _extract_single(store, pin, tmp_path / "out")
    _assert_pinned(result, pin, SINGLE_FILE_META_KEYS)


@pytest.mark.parametrize("pin", STORE_SEQUENCE_PINS)
def test_imgstore_sequence_picks_are_pinned(
    make_imgstore: MakeImgstore, tmp_path: Path, pin: Pin
) -> None:
    stores = [
        make_imgstore(name=name, nframes=10, fill=True, parent=tmp_path / "in")[0]
        for name in ("a", "b")
    ]
    result = _extract_multi(stores, pin, tmp_path / "out")
    _assert_pinned(result, pin, MULTI_FILE_META_KEYS)


@pytest.mark.media
def test_a_single_file_manifest_is_pinned_whole(media: Media, tmp_path: Path) -> None:
    selected = [11, 12, 17, 18, 21]
    result = extract_frames(
        media.coded, n_frames=5, method="kmeans", run_id="pinned", output_dir=tmp_path
    )
    assert _normalized(result, media_root=media.root, output_dir=tmp_path) == {
        "run_id": "pinned",
        "method": "kmeans",
        "video_path": "<media>/coded.mp4",
        "output_dir": "<out>",
        "manifest_path": "<out>/run_info.json",
        "n_requested": 5,
        "n_extracted": 5,
        "selected_frame_indices": selected,
        "start_frame": 0,
        "end_frame": 29,
        "candidate_step": 1,
        "crop": None,
        "files": _files(selected),
        "video_meta": {"width": 64, "height": 48, "frame_count": 30},
        "sampling": dict(kmeans_record(1, 30)),
        "random_state": 42,
    }


@pytest.mark.media
def test_a_multi_clip_manifest_is_pinned_whole(media: Media, tmp_path: Path) -> None:
    selected = [0, 4, 12, 16]
    result = extract_frames_multi(
        media.clips,
        n_frames=4,
        method="uniform",
        start_frame=0,
        end_frame=16,
        candidate_step=4,
        crop=(0, 0, 32, 48),
        random_state=7,
        run_id="pinned",
        output_dir=tmp_path,
    )
    clips = ["<media>/clip0.mp4", "<media>/clip1.mp4"]
    assert _normalized(result, media_root=media.root, output_dir=tmp_path) == {
        "run_id": "pinned",
        "method": "uniform",
        "video_path": json.dumps(clips),
        "output_dir": "<out>",
        "manifest_path": "<out>/run_info.json",
        "n_requested": 4,
        "n_extracted": 4,
        "selected_frame_indices": selected,
        "start_frame": 0,
        "end_frame": 16,
        "candidate_step": 4,
        "crop": {"x": 0, "y": 0, "w": 32, "h": 48},
        "files": _files(selected, width=32),
        "video_meta": {
            "width": 64,
            "height": 48,
            "total_frames": 30,
            "video_count": 2,
            "video_paths": clips,
        },
        "sampling": {},
        "random_state": 7,
    }


NO_ANCHOR = Exclusion(indices=(), spacing=0, entries_ignored=4)
"""An exclusion that names no frame of the camera, from a run that ignored four entries."""


def _no_anchor_keys(method: PinnedMethod) -> list[tuple[str, JsonValue]]:
    """Return the exclude keys that :data:`NO_ANCHOR` appends to a ``sampling`` record."""
    keys: list[tuple[str, JsonValue]] = [
        ("exclude_count", 0),
        ("exclude_spacing", 0),
        ("candidates_removed", 0),
        ("exclude_entries_ignored", 4),
    ]
    return keys + [("clusters_covered", 0)] if method == "kmeans" else keys


@pytest.mark.media
@pytest.mark.parametrize("pin", SINGLE_FILE_PINS)
def test_single_file_picks_without_an_anchor_are_pinned(
    media: Media, tmp_path: Path, pin: Pin
) -> None:
    result = _extract_single(media.video(pin.video), pin, tmp_path, NO_ANCHOR)
    _assert_pinned(result, pin, SINGLE_FILE_META_KEYS, _no_anchor_keys(pin.method))


@pytest.mark.media
@pytest.mark.parametrize("pin", MULTI_FILE_PINS)
def test_multi_clip_picks_without_an_anchor_are_pinned(
    media: Media, tmp_path: Path, pin: Pin
) -> None:
    result = _extract_multi(media.clips, pin, tmp_path, NO_ANCHOR)
    _assert_pinned(result, pin, MULTI_FILE_META_KEYS, _no_anchor_keys(pin.method))


@pytest.mark.parametrize("pin", STORE_SEQUENCE_PINS)
def test_imgstore_sequence_picks_without_an_anchor_are_pinned(
    make_imgstore: MakeImgstore, tmp_path: Path, pin: Pin
) -> None:
    stores = [
        make_imgstore(name=name, nframes=10, fill=True, parent=tmp_path / "in")[0]
        for name in ("a", "b")
    ]
    result = _extract_multi(stores, pin, tmp_path / "out", NO_ANCHOR)
    _assert_pinned(result, pin, MULTI_FILE_META_KEYS, _no_anchor_keys(pin.method))


def _assert_exact_reads_match_the_candidates(
    source: CandidateFeatures, last_frame: int
) -> None:
    """Read frames 17, 3 and 9 at their indices, in that order, and compare them."""
    indices, features = source.candidates(0, last_frame, 1)
    assert indices.tolist() == list(range(last_frame + 1))
    wanted = np.array([17, 3, 9], dtype=np.int32)
    np.testing.assert_array_equal(source.at(wanted), features[wanted])


@pytest.mark.media
def test_exact_reads_of_videos_match_their_candidate_features(media: Media) -> None:
    cropped = (8, 0, 40, 48)
    _assert_exact_reads_match_the_candidates(
        SingleFileFeatures(
            path=media.coded,
            facts=None,
            resize=(16, 12),
            grayscale=False,
            crop_rect=cropped,
        ),
        29,
    )
    reader = MultiVideoReader(list(media.clips), target="analysis")
    try:
        _assert_exact_reads_match_the_candidates(
            MultiClipFeatures(
                reader=reader, resize=(16, 12), grayscale=True, crop_rect=cropped
            ),
            29,
        )
    finally:
        reader.close()


def test_exact_reads_of_imgstores_match_their_candidate_features(
    make_imgstore: MakeImgstore, tmp_path: Path
) -> None:
    store, _ = make_imgstore(name="rec", nframes=20, fill=True, parent=tmp_path / "one")
    _assert_exact_reads_match_the_candidates(
        SingleFileFeatures(
            path=store, facts=None, resize=(16, 12), grayscale=True, crop_rect=None
        ),
        19,
    )
    stores = [
        make_imgstore(name=name, nframes=10, fill=True, parent=tmp_path / "two")[0]
        for name in ("a", "b")
    ]
    reader = MultiVideoReader(stores, target="analysis")
    try:
        _assert_exact_reads_match_the_candidates(
            MultiClipFeatures(
                reader=reader, resize=(16, 12), grayscale=False, crop_rect=None
            ),
            19,
        )
    finally:
        reader.close()


GAP_FILL = Pin(
    "uniform",
    3,
    [3, 7, 22],
    [
        ("exclude_count", 1),
        ("exclude_spacing", 2),
        ("candidates_removed", 5),
        ("exclude_entries_ignored", 0),
    ],
)
"""Uniform picks around frame 15, which removes frames 13 to 17.

The gap from -1 to 15 is split at 7, the gap from 15 to 30 at 22, and the first of the
three gaps of 8 at 3.
"""


@pytest.mark.media
def test_uniform_picks_fill_the_gaps_around_an_excluded_frame(
    media: Media, tmp_path: Path
) -> None:
    excluded = Exclusion(indices=(15,), spacing=2, entries_ignored=0)
    single = _extract_single(media.coded, GAP_FILL, tmp_path / "single", excluded)
    _assert_pinned(single, GAP_FILL, SINGLE_FILE_META_KEYS)
    multi = _extract_multi(media.clips, GAP_FILL, tmp_path / "multi", excluded)
    _assert_pinned(multi, GAP_FILL, MULTI_FILE_META_KEYS)


@pytest.mark.media
@pytest.mark.parametrize("sequence", [False, True], ids=["single-file", "multi-clip"])
def test_kmeans_picks_avoid_the_excluded_frames(
    media: Media, tmp_path: Path, sequence: bool
) -> None:
    """Frames 10 to 12 and 17 to 19 are removed around the anchors 11 and 18.

    The two anchors cover one or two of the seven clusters.
    """
    pin = Pin("kmeans", 5, [], [])
    excluded = Exclusion(indices=(11, 18), spacing=1, entries_ignored=2)
    result = (
        _extract_multi(media.clips, pin, tmp_path, excluded)
        if sequence
        else _extract_single(media.coded, pin, tmp_path, excluded)
    )
    assert len(result.selected_frame_indices) == 5
    assert not set(result.selected_frame_indices) & {10, 11, 12, 17, 18, 19}
    sampling = _manifest(result)["sampling"]
    assert isinstance(sampling, dict)
    *settings, covered = sampling.items()
    assert settings == kmeans_record(1, 30) + [
        ("exclude_count", 2),
        ("exclude_spacing", 1),
        ("candidates_removed", 6),
        ("exclude_entries_ignored", 2),
    ]
    assert covered[0] == "clusters_covered"
    assert covered[1] in (1, 2)
