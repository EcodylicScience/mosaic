"""The runner's batches end, keep every frame, and contain images of one shape each.

``_decoded_batches`` hands decoded batches from a producer thread to the loop that
runs the model, through a two-slot queue. Its two tests cover the end of the
video, the only moment at which the two threads have to agree.

``ImageListReader`` reads the images of ``predict-images``. Ultralytics stacks a
batch into one array. Each batch therefore contains images of one shape. The
batches follow from the images in whatever order the request lists them.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Final

import cv2
import numpy as np
import pytest

from mosaic.tracking.external.runner.ultralytics_protocol import ImageSource

from tests.helpers import image_mark, write_marked_images

pytestmark = pytest.mark.tracker


class _Reader:
    """A frame reader that yields *n_batches* batches and then runs dry."""

    def __init__(self, n_batches: int, batch_size: int) -> None:
        self._remaining = n_batches
        self._batch_size = batch_size

    def read_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        del batch_size
        if self._remaining <= 0:
            return np.empty(0, dtype=np.int64), np.empty((0, 2, 2), dtype=np.uint8)
        self._remaining -= 1
        indices = np.arange(self._batch_size, dtype=np.int64)
        return indices, np.zeros((self._batch_size, 2, 2), dtype=np.uint8)


def test_a_consumer_slower_than_the_producer_still_reaches_the_end(
    runner_module: ModuleType,
) -> None:
    """The end-of-stream sentinel is delivered however long a batch takes.

    The producer fills a two-slot queue and then finishes. If it offers the
    sentinel once, with a timeout, a consumer that has not freed a slot within
    that window never receives it and blocks on a ``get`` that never arrives --
    the whole process asleep, no output, nothing to kill it but a supervisor.

    Whether that happens is decided by how long one batch takes against the
    producer's timer, so it never happens on a fast GPU and always happens on a
    busy machine or on CPU. This test buys that certainty back: every batch here
    takes longer than the producer's window, so the old behaviour cannot pass.
    """
    reader = _Reader(n_batches=4, batch_size=2)
    seen = 0
    with runner_module._decoded_batches(reader, 2, True) as batches:
        for _indices, _frames in batches:
            seen += 1
            # Longer than the 0.5s the producer is willing to wait for a slot.
            time.sleep(0.7)
    assert seen == 4, "the loop ended early or never ended at all"


def test_every_decoded_batch_reaches_the_consumer(runner_module: ModuleType) -> None:
    """Nothing is dropped on the way through the queue, prefetching or not."""
    for prefetch in (False, True):
        reader = _Reader(n_batches=5, batch_size=3)
        with runner_module._decoded_batches(reader, 3, prefetch) as batches:
            frames = sum(len(indices) for indices, _ in batches)
        assert frames == 15, f"prefetch={prefetch} lost frames"


# --- a list of images --------------------------------------------------------


_SHAPES: Final = ((8, 12), (6, 10), (8, 12), (6, 10), (6, 10), (8, 12), (4, 14))
"""The ``(height, width)`` of the image at each position: three shapes, interleaved."""


def _read_all(
    runner_module: ModuleType, images: Sequence[ImageSource], batch_size: int
) -> list[tuple[list[int], np.ndarray]]:
    """Return the positions and the frames of every batch that the reader returns."""
    batches: list[tuple[list[int], np.ndarray]] = []
    with runner_module.ImageListReader(images) as reader:
        while True:
            indices, frames = reader.read_batch(batch_size)
            if len(indices) == 0:
                return batches
            batches.append(([int(index) for index in indices], frames))


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [
        (1, [[6], [1], [3], [4], [0], [2], [5]]),
        (2, [[6], [1, 3], [4], [0, 2], [5]]),
        (10, [[6], [1, 3, 4], [0, 2, 5]]),
    ],
)
def test_images_are_batched_by_shape_then_position(
    runner_module: ModuleType,
    tmp_path: Path,
    batch_size: int,
    expected: list[list[int]],
) -> None:
    """Each batch contains images of one shape, by ascending shape, then position.

    A batch ends where its shape ends, short of the batch size when it has to. Each
    image is read in BGR order, the channel order of the video reader.
    """
    images = write_marked_images(tmp_path, _SHAPES)

    batches = _read_all(runner_module, images, batch_size)

    assert len(runner_module.ImageListReader(images)) == len(_SHAPES)
    assert [positions for positions, _ in batches] == expected
    for positions, frames in batches:
        assert frames.dtype == np.uint8
        assert frames.shape == (len(positions), *_SHAPES[positions[0]], 3)
        for position, frame in zip(positions, frames, strict=True):
            assert frame.shape[:2] == _SHAPES[position]
            assert (frame == image_mark(position)).all()


def test_two_readers_of_the_same_images_give_the_same_batches(
    runner_module: ModuleType, tmp_path: Path
) -> None:
    """The batches follow from the images, whatever order the request lists them in."""
    images = write_marked_images(tmp_path, _SHAPES)

    runs = [
        _read_all(runner_module, listed, batch_size=2)
        for listed in (images, images, images[::-1])
    ]

    first = runs[0]
    for run in runs[1:]:
        assert [positions for positions, _ in run] == [
            positions for positions, _ in first
        ]
        assert [frames.tobytes() for _, frames in run] == [
            frames.tobytes() for _, frames in first
        ]


def test_two_images_at_one_position_are_refused(
    runner_module: ModuleType, tmp_path: Path
) -> None:
    """A position numbers the rows of one image."""
    first, second = write_marked_images(tmp_path, [(4, 4), (4, 4)])

    with pytest.raises(runner_module.RunnerError, match=r"positions \[0\]"):
        _ = runner_module.ImageListReader(
            [first, ImageSource(position=0, path=second.path)]
        )


def test_an_image_that_is_not_a_png_is_refused(
    runner_module: ModuleType, tmp_path: Path
) -> None:
    """The reader reads the shape of an image from its PNG header."""
    path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(path), np.zeros((4, 4, 3), np.uint8))

    with pytest.raises(runner_module.RunnerError, match="is not a PNG"):
        _ = runner_module.ImageListReader([ImageSource(position=0, path=str(path))])


def test_an_image_that_cannot_be_opened_is_refused(
    runner_module: ModuleType, tmp_path: Path
) -> None:
    """The refusal names the path that the request gave."""
    path = tmp_path / "absent.png"

    with pytest.raises(runner_module.RunnerError, match="absent.png could not be"):
        _ = runner_module.ImageListReader([ImageSource(position=0, path=str(path))])


def test_a_png_that_does_not_decode_is_refused(
    runner_module: ModuleType, tmp_path: Path
) -> None:
    """A PNG with a header and without its pixels is refused.

    The file contains only the signature and the header chunk of a PNG.
    """
    (image,) = write_marked_images(tmp_path, [(4, 4)])
    path = Path(image.path)
    _ = path.write_bytes(path.read_bytes()[:33])

    with runner_module.ImageListReader([image]) as reader:
        with pytest.raises(runner_module.RunnerError, match="could not be decoded"):
            _ = reader.read_batch(1)
