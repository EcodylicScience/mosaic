"""A run over thousands of entries does its per-run work a fixed number of times.

A study of 8,400 entries converted at 4 entries a second, with idle workers, because
each published table reread and rewrote the whole tracks index. A feature step spent
170 ms an entry in the calling process, most of it in a full garbage collection after
each entry, in a map of every entry rebuilt for each one, and in a write of the feature
index every ten rows. These tests count that work instead of timing it, with a clock
that moves a fixed step on each read, so that a slow or a busy machine changes
nothing. A pass over everything for each entry, or for each handful of them, exceeds
the bounds, and a run that does each pass a few times does not.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pytest

from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
from mosaic.core.dataset import Dataset
from mosaic.core import dataset as dataset_module
from mosaic.core.pipeline import pacing, tracks_index
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.index_csv import IndexCSV
from mosaic.core.pipeline.parallelism import ParallelSettings
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.tracks_index import (
    TracksIndexRow,
    read_tracks_index,
    tracks_index_path,
)
from tests.helpers import (
    add_tracks_variant_in_bulk,
    make_dataset,
    write_tracktor_recording,
)

ENTRIES = 1200
SMALL = 300
VARIANT = "convert-tracktor_corrected_tsv.0.1-56a0218226"
WRITES_PER_ENTRY = 0.01
"""The most index writes for each entry that a run of many entries may make."""


class _Clock:
    """A clock that moves one step on each read, whatever the machine is doing."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1e-4
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """Every pacer made during the test reads this clock."""
    stepped = _Clock()
    monkeypatch.setattr(pacing, "now", stepped)
    return stepped


def _bulk_dataset(root: Path, entries: int) -> Dataset:
    dataset = make_dataset(root / f"bulk{entries}", name=f"bulk{entries}")
    add_tracks_variant_in_bulk(
        dataset, VARIANT, [f"seq_{index:05d}" for index in range(entries)]
    )
    return dataset


def _tracktor_dataset(root: Path, entries: int) -> Dataset:
    dataset = make_dataset(root / f"tracktor{entries}", name=f"tracktor{entries}")
    raw = dataset.base_dir / "raw"
    for index in range(entries):
        stem = f"20211117{index:06d}_900s_no12-DGRP{101 + index % 40}-female-2to4-6-1"
        _ = write_tracktor_recording(
            raw, stem=stem, frames=20, flies=2, write_csv=False
        )
    _ = dataset.index_tracks_raw(
        [raw],
        patterns=["*_tracked_fixed_final.tsv"],
        src_format="tracktor_corrected_tsv",
    )
    return dataset


def _convert(dataset: Dataset) -> None:
    outcome = dataset.convert_all_tracks(
        source_format="tracktor_corrected_tsv",
        parallel=ParallelSettings(workers=2, mode="thread"),
    )
    assert outcome.failed == 0
    assert len(read_tracks_index(dataset)) == outcome.converted


def _speed(dataset: Dataset) -> None:
    result = run_feature(
        dataset, SpeedAngvel(), parallel_workers=2, parallel_mode="thread"
    )
    assert not result.failed_entries


def _count_writes(
    monkeypatch: pytest.MonkeyPatch, method: str, within: Path
) -> list[int]:
    """Count the calls of ``IndexCSV.<method>`` on the indexes under *within*."""
    writes: list[int] = []
    original = getattr(IndexCSV[TracksIndexRow], method)

    def counting(self: IndexCSV[TracksIndexRow], rows: list[TracksIndexRow]) -> None:
        if self.path.is_relative_to(within):
            writes.append(len(rows))
        original(self, rows)

    monkeypatch.setattr(IndexCSV, method, counting)
    return writes


@pytest.mark.slow
@pytest.mark.usefixtures("clock")
def test_converting_many_entries_writes_the_tracks_index_a_few_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _tracktor_dataset(tmp_path, ENTRIES)
    writes = _count_writes(
        monkeypatch, "append_holding_lock", tracks_index_path(dataset).parent
    )

    _convert(dataset)

    assert sum(writes) == ENTRIES
    assert len(writes) <= WRITES_PER_ENTRY * ENTRIES, len(writes)


@pytest.mark.slow
@pytest.mark.usefixtures("clock")
def test_a_feature_run_over_many_entries_writes_its_index_a_few_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _bulk_dataset(tmp_path, ENTRIES)
    writes = _count_writes(monkeypatch, "append", dataset.get_root("features"))

    _speed(dataset)

    assert sum(writes) == ENTRIES
    assert len(writes) <= WRITES_PER_ENTRY * ENTRIES, len(writes)


COLLECTION_SECONDS = 0.05
"""What a collection of a run's heap costs, on the stepped clock."""


@pytest.mark.slow
def test_a_feature_run_does_not_collect_garbage_after_each_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    dataset = _bulk_dataset(tmp_path, SMALL)
    collections: list[int] = []

    def expensive_collection(generation: int = 2) -> int:
        collections.append(generation)
        clock.now += COLLECTION_SECONDS
        return 0

    monkeypatch.setattr(gc, "collect", expensive_collection)

    _speed(dataset)

    assert len(collections) < SMALL / 4, len(collections)


@pytest.mark.slow
def test_a_feature_run_maps_its_entries_a_fixed_number_of_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _bulk_dataset(tmp_path, SMALL)
    builds: list[int] = []
    original = ResolvedScope.entry_map.fget  # pyright: ignore[reportAttributeAccessIssue]
    assert original is not None

    def counting(self: ResolvedScope) -> dict[str, tuple[str, str]]:
        builds.append(1)
        return original(self)

    monkeypatch.setattr(ResolvedScope, "entry_map", property(counting))

    _speed(dataset)

    assert len(builds) < 10, len(builds)


RESUBMITTED = 40
READS_PER_RESUBMIT = 4
"""The most reads of an index or of the source compositions a rerun may make."""


def test_converting_entries_that_are_done_reads_the_indexes_a_fixed_number_of_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _tracktor_dataset(tmp_path, RESUBMITTED)
    settings = ParallelSettings(workers=2, mode="thread")
    _ = dataset.convert_all_tracks(
        source_format="tracktor_corrected_tsv", parallel=settings
    )
    index_reads: list[int] = []
    composition_reads: list[int] = []
    read_index = dataset_module.read_tracks_index
    read_compositions = tracks_index.read_entry_compositions

    def counting_index(ds: Dataset) -> pd.DataFrame:
        index_reads.append(1)
        return read_index(ds)

    def counting_compositions(
        ds: Dataset, entries: Iterable[tuple[str, str]]
    ) -> dict[tuple[str, str], dict[str, str]]:
        composition_reads.append(1)
        return read_compositions(ds, entries)

    monkeypatch.setattr(dataset_module, "read_tracks_index", counting_index)
    monkeypatch.setattr(tracks_index, "read_entry_compositions", counting_compositions)

    outcome = dataset.convert_all_tracks(
        source_format="tracktor_corrected_tsv", parallel=settings
    )

    assert outcome.converted == RESUBMITTED
    assert len(index_reads) <= READS_PER_RESUBMIT, len(index_reads)
    assert len(composition_reads) <= READS_PER_RESUBMIT, len(composition_reads)
