"""The runner's decode loop must end, and must not lose a frame doing it.

``_decoded_batches`` hands decoded batches from a producer thread to the loop that
runs the model, through a two-slot queue. Both tests below are about the moment
the video runs out, which is the one moment the two threads have to agree about.
"""

from __future__ import annotations

import importlib
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from mosaic.tracking.external import runner as runner_package

pytestmark = pytest.mark.tracker


@pytest.fixture(scope="module")
def runner_module() -> Iterator[ModuleType]:
    """The runner program, imported into this process, and then unimported.

    Every Ultralytics import in it is deferred into a function body, so the module
    imports in an environment that has none -- which is what lets this run in
    mosaic's own environment against the real code rather than a copy of it.
    """
    directory = str(Path(runner_package.__file__).parent)
    inserted = directory not in sys.path
    if inserted:
        sys.path.insert(0, directory)
    try:
        yield importlib.import_module("ultralytics_runner")
    finally:
        if inserted:
            sys.path.remove(directory)
        for name in ("ultralytics_runner", "ultralytics_protocol"):
            _ = sys.modules.pop(name, None)


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
