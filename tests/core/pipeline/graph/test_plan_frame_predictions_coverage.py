"""A predict-frames step's coverage is bounded by the entries of its extraction run.

The op predicts only the scoped entries that the extraction run has frames of.
It skips the rest. The planner reads a scoped entry outside the extraction run
as present, and an entry of the run as present when every camera that its
frames rows name has a finished predictions row and its file.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.pipeline.graph import PlannedStep, Recipe, plan_pipeline
from mosaic.core.scope import Scope
from mosaic.tracking import register_ops
from tests.helpers import (
    entry_predictions,
    make_dataset,
    record_entry_predictions,
    record_frames_row,
)

register_ops()

FRAMES_RUN: Final = "uniform-0123456789"
MODEL: Final = "train-pose.0.2-abcdef0123"
STEP: Final = "predict"

S1: Final[Entry] = ("G", "S1")
S2: Final[Entry] = ("G", "S2")
S3: Final[Entry] = ("G", "S3")

RECIPE: Final = Recipe.model_validate(
    {
        "steps": [
            {
                "id": STEP,
                "type": "op",
                "kind": "predict-frames",
                "params": {"model": [MODEL], "frames_run_id": FRAMES_RUN},
            }
        ]
    }
)


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return make_dataset(tmp_path / "ds")


def _extracted(
    ds: Dataset, entry: Entry, camera: str = "", *, composition: str = ""
) -> None:
    """Record the extraction run's frames row of one entry camera."""
    group, sequence = entry
    record_frames_row(
        ds,
        f"frames/uniform/{FRAMES_RUN}/{group}__{sequence}",
        run_id=FRAMES_RUN,
        group=group,
        sequence=sequence,
        camera=camera,
        media_composition=composition,
    )


def _planned(ds: Dataset, scope: Sequence[Entry]) -> PlannedStep:
    """Plan the predict-frames step over *scope*."""
    return plan_pipeline(ds, RECIPE, scope=Scope(entries=list(scope))).step(STEP)


def _predicted(
    ds: Dataset,
    entry: Entry,
    camera: str = "",
    *,
    finished: bool = True,
    consumed: str = "",
) -> Path:
    """Record the planned run's predictions of one entry camera, and return its file."""
    group, sequence = entry
    predictions = entry_predictions(
        frames_run_id=FRAMES_RUN, group=group, sequence=sequence, camera=camera
    )
    run_id = _planned(ds, [entry]).run_id
    assert run_id is not None
    return record_entry_predictions(
        ds,
        predictions,
        run_id=run_id,
        finished=finished,
        consumed_frames_composition=consumed,
    )


def test_a_scope_wider_than_the_extraction_run_reads_covered_after_one_run(
    dataset: Dataset,
) -> None:
    _extracted(dataset, S1)
    _extracted(dataset, S2)
    _ = _predicted(dataset, S1)
    _ = _predicted(dataset, S2)

    planned = _planned(dataset, [S1, S2, S3])

    assert planned.status == "complete"
    assert planned.coverage.missing == frozenset()
    assert planned.coverage.covered == {S1, S2, S3}


def test_a_scope_naming_another_entry_of_the_extraction_run_reads_short(
    dataset: Dataset,
) -> None:
    for entry in (S1, S2, S3):
        _extracted(dataset, entry)
    _ = _predicted(dataset, S1)
    _ = _predicted(dataset, S2)

    planned = _planned(dataset, [S1, S2, S3])

    assert planned.status == "partial"
    assert planned.coverage.missing == {S3}
    assert planned.coverage.covered == {S1, S2}


def test_an_unpredicted_run_reads_absent(dataset: Dataset) -> None:
    _extracted(dataset, S1)

    planned = _planned(dataset, [S1, S2])

    assert planned.status == "absent"
    assert planned.coverage.covered == frozenset()


def _camera_unpredicted(ds: Dataset) -> None:
    _ = _predicted(ds, S1, "c1")


def _row_unfinished(ds: Dataset) -> None:
    _ = _predicted(ds, S1, "c1")
    _ = _predicted(ds, S1, "c2", finished=False)


def _file_gone(ds: Dataset) -> None:
    _ = _predicted(ds, S1, "c1")
    _predicted(ds, S1, "c2").unlink()


@pytest.mark.parametrize(
    "predict",
    [_camera_unpredicted, _row_unfinished, _file_gone],
    ids=["camera-unpredicted", "row-unfinished", "file-gone"],
)
def test_an_entry_needs_a_finished_row_and_file_for_every_camera(
    dataset: Dataset, predict: Callable[[Dataset], None]
) -> None:
    _extracted(dataset, S1, "c1")
    _extracted(dataset, S1, "c2")
    _extracted(dataset, S2)
    _ = _predicted(dataset, S2)
    predict(dataset)

    planned = _planned(dataset, [S1, S2])

    assert planned.coverage.missing == {S1}
    assert planned.coverage.covered == {S2}


def test_an_entry_with_every_camera_predicted_reads_covered(
    dataset: Dataset,
) -> None:
    _extracted(dataset, S1, "c1")
    _extracted(dataset, S1, "c2")
    _ = _predicted(dataset, S1, "c1")
    _ = _predicted(dataset, S1, "c2")

    planned = _planned(dataset, [S1])

    assert (planned.status, planned.coverage.covered) == ("complete", {S1})


def test_a_reextracted_entry_stays_drifted_through_the_plan(
    dataset: Dataset,
) -> None:
    _extracted(dataset, S1, composition="new")
    _extracted(dataset, S2, composition="same")
    _ = _predicted(dataset, S1, consumed="old")
    _ = _predicted(dataset, S2, consumed="same")

    planned = _planned(dataset, [S1, S2, S3])

    assert planned.drift == (S1,)
    assert planned.status == "complete-but-drifted"
    assert planned.coverage.covered == {S1, S2, S3}
