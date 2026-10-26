"""A select-frames step's coverage is bounded by the entries of its tracks variant.

The op selects frames only for the scoped entries that the variant has a row
for, and skips the rest. The planner reads a scoped entry outside the variant
as present, and an entry of the variant as present when it has a finished
selections row and its file.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.pipeline.graph import PlannedStep, Recipe, plan_pipeline
from mosaic.core.pipeline.inventory.model import FrameSelectionRef
from mosaic.core.scope import Scope
from mosaic.tracking import register_ops
from tests.helpers import add_tracks_variant, make_dataset, record_entry_selection

register_ops()

VARIANT: Final = "trex.0.1-abcdef0123"
STEP: Final = "select"

S1: Final[Entry] = ("", "s1")
S2: Final[Entry] = ("", "s2")
S3: Final[Entry] = ("", "s3")

RECIPE: Final = Recipe.model_validate(
    {
        "steps": [
            {
                "id": STEP,
                "type": "op",
                "kind": "select-frames",
                "params": {"source_tracks_run_id": VARIANT},
            }
        ]
    }
)


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return make_dataset(tmp_path / "ds")


def _planned(ds: Dataset, scope: Sequence[Entry]) -> PlannedStep:
    """Plan the select-frames step over *scope*."""
    return plan_pipeline(ds, RECIPE, scope=Scope(entries=list(scope))).step(STEP)


def _selected(ds: Dataset, entry: Entry, *, finished: bool = True) -> Path:
    """Record the planned run's selection of one entry, and return its file."""
    run_id = _planned(ds, [entry]).run_id
    assert run_id is not None
    group, sequence = entry
    return record_entry_selection(
        ds,
        run_id=run_id,
        group=group,
        sequence=sequence,
        frames=[3],
        finished=finished,
        source_tracks_run_id=VARIANT,
    )


def test_a_planned_step_names_its_selection_by_run_and_variant(
    dataset: Dataset,
) -> None:
    add_tracks_variant(dataset, VARIANT, "s1")

    planned = _planned(dataset, [S1])

    assert planned.run_id is not None
    assert planned.artifact == FrameSelectionRef(
        run_id=planned.run_id, source_tracks_run_id=VARIANT
    )


def test_a_scope_wider_than_the_variant_reads_covered_after_one_run(
    dataset: Dataset,
) -> None:
    add_tracks_variant(dataset, VARIANT, "s1", "s2")
    _ = _selected(dataset, S1)
    _ = _selected(dataset, S2)

    planned = _planned(dataset, [S1, S2, S3])

    assert planned.status == "complete"
    assert planned.coverage.missing == frozenset()
    assert planned.coverage.covered == {S1, S2, S3}


def test_a_scope_naming_another_entry_of_the_variant_reads_short(
    dataset: Dataset,
) -> None:
    add_tracks_variant(dataset, VARIANT, "s1", "s2", "s3")
    _ = _selected(dataset, S1)
    _ = _selected(dataset, S2)

    planned = _planned(dataset, [S1, S2, S3])

    assert planned.status == "partial"
    assert planned.coverage.missing == {S3}
    assert planned.coverage.covered == {S1, S2}


def test_an_unselected_run_reads_absent(dataset: Dataset) -> None:
    add_tracks_variant(dataset, VARIANT, "s1")

    planned = _planned(dataset, [S1, S2])

    assert planned.status == "absent"
    assert planned.coverage.covered == frozenset()


def _row_unfinished(ds: Dataset) -> None:
    _ = _selected(ds, S1, finished=False)


def _file_gone(ds: Dataset) -> None:
    _selected(ds, S1).unlink()


@pytest.mark.parametrize(
    "select", [_row_unfinished, _file_gone], ids=["row-unfinished", "file-gone"]
)
def test_an_entry_needs_a_finished_row_and_its_file(
    dataset: Dataset, select: Callable[[Dataset], None]
) -> None:
    add_tracks_variant(dataset, VARIANT, "s1", "s2")
    _ = _selected(dataset, S2)
    select(dataset)

    planned = _planned(dataset, [S1, S2])

    assert planned.coverage.missing == {S1}
    assert planned.coverage.covered == {S2}
