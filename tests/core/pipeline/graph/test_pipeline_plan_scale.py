"""Planning a recipe over thousands of entries costs time in proportion to them.

A study of 8,400 entries waited a minute or more on each ``plan_recipe`` before any
step computed, because a pass over the tracks index did its work once per entry.
The bounds are far above what a plan costs on a quiet machine and far below
what a quadratic pass costs at this size, so only a return of one fails them.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    CleanupPolicy,
    DeclarationCatalog,
    declaration_catalog,
    plan_pipeline,
    plan_with_cleanup,
)
from mosaic.core.pipeline.graph import read_recipe
from mosaic.core.pipeline.inventory import InventoryCache
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
