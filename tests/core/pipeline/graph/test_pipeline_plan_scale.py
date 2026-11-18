"""Planning a recipe costs time in proportion to its entries, steps and runs.

A dataset of 8,400 entries waits a minute or more on each ``plan_recipe`` before any
step computes, because a pass over the tracks index did its work once per entry.
The bounds are far above what a plan costs on a quiet machine and far below
what a quadratic pass costs at this size, so only a return of one fails them.

A recipe of many cached steps planned slowly too, because each step read the
tracks index and each root's ``sequences.csv`` again, and the scan passed over
a feature storage's whole index once for each run it holds. Those are counted
rather than timed.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Iterator
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.arena_index import ARENA_LOOKUP_CACHE
from mosaic.core.pipeline.composition import SequenceComposition
from mosaic.core.pipeline.graph import (
    CleanupPolicy,
    DeclarationCatalog,
    Recipe,
    declaration_catalog,
    plan_pipeline,
    plan_with_cleanup,
    run_pipeline,
)
from mosaic.core.pipeline.graph import read_recipe
from mosaic.core.pipeline.inventory import InventoryCache
from mosaic.core.pipeline.inventory.scan import inventory as scan_inventory
from mosaic.core.pipeline.sequence_index import write_sequence_compositions
from mosaic.core.pipeline.tracks_raw_index import TRACKS_RAW_INDEX_COLUMNS
from tests.helpers import add_tracks_variant_in_bulk, make_dataset, speed_recipe

ENTRIES = 2000
SMALL_ENTRIES = 500
UNCONVERTED = 100
VARIANT = "convert-tracktor_corrected_tsv.0.1-56a0218226"
COLD_SECONDS = 8.0
WARM_SECONDS = 4.0
GROWTH = 6.0
"""The most that a plan may slow by when its entries grow fourfold.

A pass in proportion to the entries slows by about 4, and one over every pair of
entries by about 16.
"""


def _dataset(root: Path, entries: int) -> Dataset:
    dataset = make_dataset(root / f"ds{entries}", name=f"scale{entries}")
    add_tracks_variant_in_bulk(
        dataset, VARIANT, [f"seq_{index:05d}" for index in range(entries)]
    )
    return dataset


@pytest.fixture(scope="module")
def large(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return _dataset(tmp_path_factory.mktemp("large"), ENTRIES)


@pytest.fixture(scope="module")
def small(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return _dataset(tmp_path_factory.mktemp("small"), SMALL_ENTRIES)


def _cold_seconds(dataset: Dataset, catalog: DeclarationCatalog) -> float:
    """The best of two plans, each with an inventory that has not been built."""
    best = float("inf")
    for _ in range(2):
        started = time.perf_counter()
        _ = plan_pipeline(
            dataset, speed_recipe(), inventory=InventoryCache(dataset), catalog=catalog
        )
        best = min(best, time.perf_counter() - started)
    return best


@pytest.mark.slow
def test_a_plan_grows_in_proportion_to_its_entries(
    small: Dataset, large: Dataset
) -> None:
    catalog = declaration_catalog()

    few = _cold_seconds(small, catalog)
    many = _cold_seconds(large, catalog)

    assert many < GROWTH * few, (few, many)


@pytest.mark.slow
def test_a_plan_over_two_thousand_entries_is_not_quadratic(large: Dataset) -> None:
    held = InventoryCache(large)
    catalog = declaration_catalog()

    started = time.perf_counter()
    cold = plan_pipeline(large, speed_recipe(), inventory=held, catalog=catalog)
    cold_seconds = time.perf_counter() - started
    started = time.perf_counter()
    warm = plan_pipeline(large, speed_recipe(), inventory=held, catalog=catalog)
    warm_seconds = time.perf_counter() - started

    assert [step.step_id for step in cold.steps] == ["speed"]
    assert len(cold.steps[0].coverage.target) == ENTRIES
    assert len(warm.steps[0].coverage.target) == ENTRIES
    assert cold_seconds < COLD_SECONDS
    assert warm_seconds < WARM_SECONDS


@pytest.mark.slow
def test_a_plan_with_cleanup_over_two_thousand_entries_is_not_quadratic(
    large: Dataset,
) -> None:
    catalog = declaration_catalog()
    started = time.perf_counter()
    plan = plan_with_cleanup(
        large,
        speed_recipe(),
        cleanup=CleanupPolicy(),
        inventory=InventoryCache(large),
        catalog=catalog,
    )
    seconds = time.perf_counter() - started

    assert plan.cleanup is not None
    assert seconds < COLD_SECONDS


def _write_raw_rows(dataset: Dataset, sequences: list[str]) -> None:
    """Index one raw Tracktor file for each of *sequences*, converting none."""
    rows = [
        {
            "group": "",
            "sequence": sequence,
            "abs_path": f"raw/{sequence}.tsv",
            "src_format": "tracktor_corrected_tsv",
            "size_bytes": 1,
            "mtime_iso": "",
            "md5": "",
        }
        for sequence in sequences
    ]
    root = dataset.get_root("tracks_raw")
    root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=TRACKS_RAW_INDEX_COLUMNS)
    frame.to_csv(root / "index.csv", index=False)


@pytest.mark.slow
def test_a_conversion_step_over_two_thousand_raw_files_plans_its_remainder(
    tmp_path: Path,
) -> None:
    """The raw rows are measured against the variant's tables in one pass."""
    large = _dataset(tmp_path, ENTRIES)
    converted = [f"seq_{index:05d}" for index in range(ENTRIES)]
    extra = [f"late_{index:05d}" for index in range(UNCONVERTED)]
    _write_raw_rows(large, converted + extra)
    recipe = read_recipe(
        {
            "steps": [
                {
                    "id": "convert",
                    "type": "op",
                    "kind": "convert-tracks",
                    "params": {"source_format": "tracktor_corrected_tsv"},
                }
            ]
        }
    )

    started = time.perf_counter()
    plan = plan_pipeline(
        large, recipe, inventory=InventoryCache(large), catalog=declaration_catalog()
    )
    seconds = time.perf_counter() - started

    step = plan.step("convert")
    assert step.run_id == VARIANT
    assert len(step.coverage.target) == ENTRIES + UNCONVERTED
    assert len(step.coverage.missing) == UNCONVERTED
    assert seconds < COLD_SECONDS


FEW_STEPS = 5
MANY_STEPS = 50
CACHED_ENTRIES = 3
RUN_GROWTH_BOUND = 2 * MANY_STEPS // FEW_STEPS
"""The most that the rows a scan walks may grow by when its runs grow tenfold.

The index rows grow tenfold with the runs, so a walk in proportion to them grows
by about 10. A walk over a storage's whole index once for each of its runs grows
by about 100.
"""


def _stepping_recipe(steps: int, first: int = 0) -> Recipe:
    """Steps *first* to *steps* of a recipe of ``speed-angvel`` steps over the tracks.

    Each step has its own run. It reads the tracks and follows ids, so it
    reaches every per-step read of a plan: the identity's compositions and
    tracks variants, and the check for rows without tracklets. A shorter recipe
    is a prefix of a longer.
    """
    return read_recipe(
        {
            "schema_version": 1,
            "name": f"stepping-{first}-{steps}",
            "steps": [
                {
                    "id": f"speed_{index:02d}",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                    "params": {"step_size": index + 1},
                }
                for index in range(first, steps)
            ],
        }
    )


def _cached_dataset(root: Path, steps: int) -> Dataset:
    """A dataset that caches the runs of :func:`_stepping_recipe` of *steps*.

    Each source root records a composition for every entry, so a plan finds a
    ``sequences.csv`` to read for each. The steps run one at a time, so that
    building the dataset does not plan the whole recipe before each step.
    """
    dataset = make_dataset(root, name=f"cached{steps}")
    sequences = [f"seq_{index:02d}" for index in range(CACHED_ENTRIES)]
    add_tracks_variant_in_bulk(dataset, VARIANT, sequences)
    compositions = {
        ("", sequence): SequenceComposition(digest=f"{number:016x}", member_count=1)
        for number, sequence in enumerate(sequences)
    }
    for source_root in ("tracks_raw", "media_raw", "labels_raw"):
        dataset.get_root(source_root).mkdir(parents=True, exist_ok=True)
        _ = write_sequence_compositions(dataset, source_root, compositions=compositions)
    catalog = declaration_catalog()
    for index in range(steps):
        _ = run_pipeline(dataset, _stepping_recipe(index + 1, index), catalog=catalog)
    return dataset


@pytest.fixture(scope="module")
def few_runs(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return _cached_dataset(tmp_path_factory.mktemp("few_runs"), FEW_STEPS)


@pytest.fixture(scope="module")
def many_runs(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return _cached_dataset(tmp_path_factory.mktemp("many_runs"), MANY_STEPS)


def _count_index_reads(patch: pytest.MonkeyPatch, dataset: Dataset) -> Counter[str]:
    """Count each read of a file of *dataset*, by its path.

    Every index reader parses its file with ``pd.read_csv``: ``IndexCSV`` and
    the tracks, sequence and raw tracks readers alike. A read is counted by
    path, whichever function asked for it.
    """
    root = Path(dataset.base_dir).resolve()
    counts: Counter[str] = Counter()

    def record(path: object) -> None:
        if not isinstance(path, (str, Path)):
            return
        try:
            relative = Path(path).resolve().relative_to(root)
        except ValueError:
            return
        counts[relative.as_posix()] += 1

    # Held under a declared type: the call forwards whatever arguments it was
    # given, which no single overload of ``read_csv`` names.
    originals: dict[str, Callable[..., pd.DataFrame]] = {"read_csv": pd.read_csv}

    def counted_read_csv(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        record(path)
        return originals["read_csv"](path, *args, **kwargs)

    patch.setattr(pd, "read_csv", counted_read_csv)
    return counts


@pytest.mark.slow
def test_a_plan_reads_each_index_as_often_for_fifty_cached_steps_as_for_five(
    many_runs: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = declaration_catalog()
    _ = plan_pipeline(
        many_runs,
        _stepping_recipe(2),
        inventory=InventoryCache(many_runs),
        catalog=catalog,
    )
    reads: dict[int, Counter[str]] = {}
    for steps in (FEW_STEPS, MANY_STEPS):
        ARENA_LOOKUP_CACHE.clear()
        with monkeypatch.context() as patch:
            counts = _count_index_reads(patch, many_runs)
            plan = plan_pipeline(
                many_runs,
                _stepping_recipe(steps),
                inventory=InventoryCache(many_runs),
                catalog=catalog,
            )
        assert [step.status for step in plan.steps] == ["complete"] * steps
        reads[steps] = counts

    assert "tracks/index.csv" in reads[MANY_STEPS]
    assert "tracks_raw/sequences.csv" in reads[MANY_STEPS]
    assert reads[FEW_STEPS] == reads[MANY_STEPS]


def _count_rows_walked(patch: pytest.MonkeyPatch) -> Counter[str]:
    """Count the rows that a walk over a frame's rows in Python passes over."""
    walked: Counter[str] = Counter()
    itertuples = pd.DataFrame.itertuples
    iterrows = pd.DataFrame.iterrows

    def counted_itertuples(
        frame: pd.DataFrame, index: bool = True, name: str | None = "Pandas"
    ) -> Iterable[tuple[object, ...]]:
        walked["rows"] += len(frame)
        return itertuples(frame, index=index, name=name)

    def counted_iterrows(frame: pd.DataFrame) -> Iterator[tuple[Hashable, pd.Series]]:
        walked["rows"] += len(frame)
        return iterrows(frame)

    patch.setattr(pd.DataFrame, "itertuples", counted_itertuples)
    patch.setattr(pd.DataFrame, "iterrows", counted_iterrows)
    return walked


@pytest.mark.slow
def test_a_scan_walks_rows_in_proportion_to_the_runs_it_holds(
    few_runs: Dataset, many_runs: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    walked: dict[int, int] = {}
    for steps, dataset in ((FEW_STEPS, few_runs), (MANY_STEPS, many_runs)):
        with monkeypatch.context() as patch:
            counts = _count_rows_walked(patch)
            found = scan_inventory(dataset, kinds=("feature",))
        assert len(found.records) == steps
        walked[steps] = counts["rows"]

    assert walked[MANY_STEPS] < RUN_GROWTH_BOUND * walked[FEW_STEPS], walked


@pytest.mark.slow
def test_a_scan_reads_each_sequence_index_as_often_for_fifty_runs_as_for_five(
    few_runs: Dataset, many_runs: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: dict[int, dict[str, int]] = {}
    for steps, dataset in ((FEW_STEPS, few_runs), (MANY_STEPS, many_runs)):
        with monkeypatch.context() as patch:
            counts = _count_index_reads(patch, dataset)
            found = scan_inventory(dataset, kinds=("feature",))
        assert len(found.records) == steps
        reads[steps] = {
            path: count
            for path, count in counts.items()
            if path.endswith("sequences.csv")
        }

    assert "tracks_raw/sequences.csv" in reads[MANY_STEPS]
    assert reads[FEW_STEPS] == reads[MANY_STEPS]
