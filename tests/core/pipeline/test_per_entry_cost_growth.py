"""A run over thousands of entries does its per-run work a fixed number of times.

A dataset of 8,400 entries converts at 4 entries a second, with idle workers, when
each published table rereads and rewrites the whole tracks index. A feature step spends
170 ms an entry in the calling process, most of it in a full garbage collection after
each entry, in a map of every entry rebuilt for each one, in a write of the feature
index every ten rows, and in a parse of an input's parquet footer by each function that
asks for it. These tests count that work instead of timing it, with a clock
that moves a fixed step on each read, so that a slow or a busy machine changes
nothing. A pass over everything for each entry, or for each handful of them, exceeds
the bounds, and a run that does each pass a few times does not.
"""

from __future__ import annotations

import gc
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
from mosaic.core.dataset import Dataset
from mosaic.core import dataset as dataset_module
from mosaic.core.pipeline import loading, pacing, parquet_reader, tracks_index
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.index_csv import IndexCSV
from mosaic.core.pipeline.loading import input_projection
from mosaic.core.pipeline.parallelism import ParallelSettings
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import NNResult
from mosaic.core.pipeline.tracks_index import (
    TracksIndexRow,
    read_tracks_index,
    tracks_index_path,
)
from tests.helpers import (
    add_tracks_variant_in_bulk,
    count_property_reads,
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
    builds = count_property_reads(monkeypatch, ResolvedScope, "entry_map")

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


TIMED_ENTRIES = 40
READS_PER_CONVERSION = 3
"""The most reads of the tracks index or of division facts one conversion may make."""


def test_converting_timed_events_reads_the_tracks_index_a_fixed_number_of_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(tmp_path / "timed", name="timed")
    sequences = [f"o{index:03d}__ant" for index in range(TIMED_ENTRIES)]
    add_tracks_variant_in_bulk(dataset, VARIANT, sequences)
    events = pd.DataFrame(
        {
            "Observation id": [f"o{index:03d}" for index in range(TIMED_ENTRIES)],
            "Subject": ["ant"] * TIMED_ENTRIES,
            "Behavior": ["approach"] * TIMED_ENTRIES,
            "Behavior type": ["STATE"] * TIMED_ENTRIES,
            "Start (s)": [0.0] * TIMED_ENTRIES,
            "Stop (s)": [0.05] * TIMED_ENTRIES,
            "FPS": [30.0] * TIMED_ENTRIES,
        }
    )
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    events.to_pickle(uploads / "boris.pkl")
    dataset.index_labels_raw(
        [uploads], patterns=["*.pkl"], src_format="boris_pandas_pickle"
    )
    index_reads: list[int] = []
    division_reads: list[int] = []
    read_index = dataset_module.read_tracks_index
    division_facts = Dataset.division_facts

    def counting_index(ds: Dataset) -> pd.DataFrame:
        index_reads.append(1)
        return read_index(ds)

    def counting_divisions(self: Dataset, entries: Iterable[tuple[str, str]]):
        division_reads.append(1)
        return division_facts(self, entries)

    monkeypatch.setattr(dataset_module, "read_tracks_index", counting_index)
    monkeypatch.setattr(Dataset, "division_facts", counting_divisions)

    outcome = dataset.convert_all_labels(
        kind="behavior", source_format="boris_pandas_pickle"
    )

    assert outcome.written == TIMED_ENTRIES
    assert len(index_reads) <= READS_PER_CONVERSION, len(index_reads)
    assert len(division_reads) <= READS_PER_CONVERSION, len(division_reads)


NAN = float("nan")
PROJECTED_ENTRIES = 4
SPEED_READS = sorted(["frame", "time", "id", "group", "sequence", "X", "Y"])
"""The columns of a bulk table that ``speed-angvel``'s output depends on: its
identity columns and the columns it requires. ``feat_a`` is not among them."""


def _count_table_reads(
    monkeypatch: pytest.MonkeyPatch, within: Path
) -> list[list[str] | None]:
    """The columns each ``pd.read_parquet`` under *within* asks for."""
    reads: list[list[str] | None] = []
    original = pd.read_parquet

    def counting(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
        if Path(path).is_relative_to(within):
            reads.append(None if columns is None else list(columns))
        return original(path, columns=columns)

    monkeypatch.setattr(pd, "read_parquet", counting)
    return reads


def test_a_feature_run_reads_only_the_columns_its_output_depends_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _bulk_dataset(tmp_path, PROJECTED_ENTRIES)
    reads = _count_table_reads(monkeypatch, dataset.get_root("tracks"))

    _speed(dataset)

    assert len(reads) == PROJECTED_ENTRIES, reads
    assert all(
        columns is not None and sorted(columns) == SPEED_READS for columns in reads
    ), reads


def test_a_column_whose_presence_adds_an_output_column_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``speed-angvel`` writes ``angvel`` where the table holds ``ANGLE``."""
    dataset = _bulk_dataset(tmp_path, PROJECTED_ENTRIES)
    tables = sorted(dataset.get_root("tracks").rglob("*.parquet"))
    for table in tables:
        rows = pd.read_parquet(table)
        rows["ANGLE"] = rows["X"] / 10.0
        rows.to_parquet(table)
    reads = _count_table_reads(monkeypatch, dataset.get_root("tracks"))

    _speed(dataset)

    assert len(reads) == PROJECTED_ENTRIES, reads
    assert all(
        columns is not None and sorted(columns) == sorted([*SPEED_READS, "ANGLE"])
        for columns in reads
    ), reads
    outputs = sorted(dataset.get_root("features").rglob("*seq_*.parquet"))
    assert len(outputs) == PROJECTED_ENTRIES, outputs
    assert all("angvel" in pd.read_parquet(output).columns for output in outputs)


def test_an_aggregate_that_carries_columns_reads_every_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mosaic.behavior.feature_library.individual_aggregate import (
        IndividualAggregate,
    )

    dataset = _bulk_dataset(tmp_path, PROJECTED_ENTRIES)
    reads = _count_table_reads(monkeypatch, dataset.get_root("tracks"))

    params: dict[str, object] = {"columns": ["X"], "aggs": ["mean"]}
    result = run_feature(dataset, IndividualAggregate(params=params))

    assert not result.failed_entries
    assert reads == [None] * PROJECTED_ENTRIES, reads
    monkeypatch.setattr(IndividualAggregate, "declares_every_read", True, raising=False)
    assert input_projection(IndividualAggregate(params=params)) is None


def test_a_feature_that_reads_undeclared_columns_reads_every_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``social-motion-summary`` reads ``group_size`` where it is present."""
    from mosaic.behavior.feature_library.social_motion_summary import (
        SocialMotionSummary,
    )

    dataset = _bulk_dataset(tmp_path, PROJECTED_ENTRIES)
    reads = _count_table_reads(monkeypatch, dataset.get_root("tracks"))

    result = run_feature(dataset, SocialMotionSummary(params={"speed_col": "X"}))

    assert not result.failed_entries
    assert reads == [None] * PROJECTED_ENTRIES, reads


NN_STORAGE = "nearest-neighbor"
NN_RUN = "0.1-aaaaaaaaaa"


def _neighbor_run(dataset: Dataset, table: pd.DataFrame) -> tuple[NNResult, Path]:
    """A finished neighbor run holding *table* for the entry ``seq_a``."""
    from mosaic.core.pipeline.index import (
        FeatureIndexRow,
        feature_index,
        feature_index_path,
    )

    root = dataset.get_root("features") / NN_STORAGE / NN_RUN
    root.mkdir(parents=True)
    output = root / "seq_a.parquet"
    table.to_parquet(output)
    index = feature_index(feature_index_path(dataset, NN_STORAGE))
    index.ensure()
    index.append(
        [
            FeatureIndexRow(
                run_id=NN_RUN,
                feature=NN_STORAGE,
                version="0.1",
                group="",
                sequence="seq_a",
                abs_path=Path(dataset.relative_to_root(output)),
                params_hash="",
                n_rows=len(table),
            )
        ]
    )
    index.mark_finished(NN_RUN)
    return NNResult(feature=NN_STORAGE, run_id=NN_RUN), output


def _count_footer_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The files whose footer is parsed, memoized or not."""
    footers: list[Path] = []
    original = parquet_reader.parse_footer

    def counting(path: Path) -> parquet_reader.ParquetFooter:
        footers.append(path)
        return original(path)

    monkeypatch.setattr(parquet_reader, "parse_footer", counting)
    return footers


FOOTER_ENTRIES = 12
INPUTS_PER_ENTRY = 1
"""``speed-angvel`` reads one tracks table for each entry."""
PER_RUN_READS = 2
"""``is_track_shaped`` and ``carries_z``, which may read a file that no entry does."""


def test_a_feature_run_reads_each_input_footer_once_per_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _bulk_dataset(tmp_path, FOOTER_ENTRIES)
    footers = _count_footer_reads(monkeypatch)

    _speed(dataset)

    assert len(footers) <= FOOTER_ENTRIES * INPUTS_PER_ENTRY, footers

    footers.clear()
    _speed(dataset)

    assert len(footers) <= FOOTER_ENTRIES + PER_RUN_READS, footers


NEIGHBOR_ROWS = {"id": [0, 1, 0, 1], "nn_id": [1.0, 0.0, NAN, 0.0]}


@pytest.mark.parametrize(
    ("order", "values", "keys"),
    [
        ("frame", [0, 0, 1, 1], [0, 0, 1, 1]),
        ("time", [0.0, 0.0, 1.5, 1.5], [0, 0, 1, 1]),
    ],
)
def test_the_neighbor_lookup_reads_three_columns_of_its_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    order: str,
    values: list[float],
    keys: list[int],
) -> None:
    dataset = make_dataset(tmp_path / "nn", name="nn")
    table = pd.DataFrame({order: values, **NEIGHBOR_ROWS, "nn_dist": 1.0})
    spec, output = _neighbor_run(dataset, table)
    footers = _count_footer_reads(monkeypatch)
    reads = _count_table_reads(monkeypatch, output.parent)

    lookup = loading.build_nn_lookup(dataset, "", "seq_a", spec)

    assert footers == [output]
    assert reads == [[order, "id", "nn_id"]]
    found, neighbor = lookup.neighbor_of(
        np.asarray(keys, dtype=np.int64), np.asarray([0, 1, 0, 1], dtype=np.int64)
    )
    assert found.tolist() == [True, True, False, True]
    assert neighbor[found].tolist() == [1, 0, 0]


def test_a_neighbor_table_without_its_columns_gives_an_empty_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(tmp_path / "nn", name="nn")
    spec, output = _neighbor_run(
        dataset, pd.DataFrame({"frame": [0, 1], "id": [0, 0], "nn_dist": [1.0, 2.0]})
    )
    reads = _count_table_reads(monkeypatch, output.parent)

    lookup = loading.build_nn_lookup(dataset, "", "seq_a", spec)

    assert not lookup
    assert reads == []


def _reference_lookup(
    frames: np.ndarray, ids: np.ndarray, neighbors: np.ndarray
) -> dict[tuple[int, int], int]:
    """Each ``(frame, id)``'s neighbor, a row at a time: the last row wins."""
    lookup: dict[tuple[int, int], int] = {}
    for frame, identity, neighbor in zip(frames, ids, neighbors):
        if not np.isnan(neighbor):
            lookup[(int(frame), int(identity))] = int(neighbor)
    return lookup


def _reference_mask(
    pairs: pd.DataFrame, lookup: dict[tuple[int, int], int]
) -> np.ndarray:
    """Whether either individual of each pair row is the other's neighbor."""
    frames = pairs["frame"].to_numpy(dtype=int)
    first = pairs["id1"].to_numpy(dtype=int)
    second = pairs["id2"].to_numpy(dtype=int)
    mask = np.zeros(len(pairs), dtype=bool)
    for row in range(len(pairs)):
        frame, a, b = int(frames[row]), int(first[row]), int(second[row])
        if lookup.get((frame, a)) == b or lookup.get((frame, b)) == a:
            mask[row] = True
    return mask


NEIGHBOR_TABLES: dict[str, tuple[list[float], list[float], list[float]]] = {
    "a key repeated, the last row winning": (
        [0, 0, 0, 1, 1],
        [0, 1, 0, 0, 0],
        [1, 0, 2, 2, 1],
    ),
    "a missing neighbor after a present one": (
        [0, 0, 1, 1],
        [0, 0, 1, 2],
        [2, NAN, NAN, 1],
    ),
    "every neighbor missing": ([0, 1], [0, 0], [NAN, NAN]),
    "frames given as seconds": ([0.0, 0.4, 0.9, 1.2], [0, 0, 1, 1], [1, 2, 0, 0]),
    "no row": ([], [], []),
}


LOOKUP_CASES = [
    (case, dtype)
    for case, (_, _, neighbors) in NEIGHBOR_TABLES.items()
    for dtype in ("float64", "int64")
    if dtype == "float64" or not any(np.isnan(neighbors))
]
"""Each table with float neighbors, and with integers where none is missing."""


@pytest.mark.parametrize(("case", "neighbor_dtype"), LOOKUP_CASES)
def test_the_neighbor_lookup_matches_a_row_by_row_build(
    case: str, neighbor_dtype: str
) -> None:
    from mosaic.core.pipeline.loading import NeighborLookup, nn_pair_mask

    frame_values, id_values, neighbor_values = NEIGHBOR_TABLES[case]
    frames = np.asarray(frame_values, dtype=np.float64)
    ids = np.asarray(id_values, dtype=np.int64)
    neighbors = np.asarray(neighbor_values, dtype=neighbor_dtype)
    expected = _reference_lookup(frames, ids, neighbors)

    lookup = NeighborLookup.from_rows(frames, ids, neighbors)

    assert len(lookup) == len(expected)
    grid = [(frame, a) for frame in range(-1, 3) for a in range(-1, 4)]
    query_frames = np.asarray([frame for frame, _ in grid], dtype=np.int64)
    query_ids = np.asarray([a for _, a in grid], dtype=np.int64)
    found, neighbor = lookup.neighbor_of(query_frames, query_ids)
    for position, key in enumerate(grid):
        assert bool(found[position]) == (key in expected), key
        if key in expected:
            assert int(neighbor[position]) == expected[key], key
    pairs = pd.DataFrame(
        {
            "frame": [frame for frame, _, _ in _PAIR_ROWS],
            "id1": [a for _, a, _ in _PAIR_ROWS],
            "id2": [b for _, _, b in _PAIR_ROWS],
        }
    )
    if expected:
        assert nn_pair_mask(pairs, lookup).tolist() == (
            _reference_mask(pairs, expected).tolist()
        )
    else:
        assert nn_pair_mask(pairs, lookup).all()


_PAIR_ROWS = [(frame, a, b) for frame in range(3) for a in range(3) for b in range(3)]


class _CallCount:
    """Python and C function calls made while it is active, on this thread."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, frame: object, event: str, arg: object) -> None:
        del frame, arg
        if event in ("call", "c_call"):
            self.calls += 1


def _calls_to_filter(rows: int) -> int:
    """The calls that a lookup over *rows* and a mask of as many pairs make."""
    from mosaic.core.pipeline.loading import NeighborLookup, nn_pair_mask

    frames = np.repeat(np.arange(rows // 2, dtype=np.float64), 2)
    ids = np.tile(np.arange(2, dtype=np.int64), rows // 2)
    neighbors = (1 - ids).astype(np.float64)
    pairs = pd.DataFrame({"frame": frames, "id1": ids, "id2": 1 - ids})
    counter = _CallCount()
    sys.setprofile(counter)
    try:
        lookup = NeighborLookup.from_rows(frames, ids, neighbors)
        mask = nn_pair_mask(pairs, lookup)
    finally:
        sys.setprofile(None)
    assert mask.all()
    return counter.calls


FEW_ROWS = 1_000
MANY_ROWS = 20_000
"""A row-by-row filter makes a call a row at least: 20,325 over these rows."""


def test_the_neighbor_filter_makes_no_call_per_row() -> None:
    few = _calls_to_filter(FEW_ROWS)
    many = _calls_to_filter(MANY_ROWS)

    assert many < MANY_ROWS / 4, (few, many)
    assert many <= few + FEW_ROWS // 10, (few, many)
