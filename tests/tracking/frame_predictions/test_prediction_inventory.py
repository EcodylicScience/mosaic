"""The inventory's frame-predictions and prediction-score kinds.

A predictions run is covered by the entry cameras of its rows, and it drifts on an
entry whose extraction was cut again from other media. A score run is one artifact
that covers its own run id.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset, open_dataset
from mosaic.core.entry import Entry
from mosaic.core.pipeline.inventory import (
    ArtifactStatus,
    CoverageWire,
    FramePredictionsRef,
    InventoryWire,
    PredictionScoreRef,
    inventory_wire,
)
from mosaic.core.scope import Scope
from mosaic.tracking import dataset_inventory
from mosaic.tracking.frame_predictions.index import (
    SCORE_RUN_OUTPUTS,
    ScoreRunRow,
    prediction_index_path,
    prediction_kind_root,
    score_predictions_index,
)
from tests.helpers import (
    assert_tree_unchanged,
    entry_predictions,
    invoke_ok,
    make_dataset,
    record_entry_predictions,
    record_frames_row,
    tree_snapshot,
)

FRAMES_RUN = "uniform-0123456789"
OTHER_FRAMES_RUN = "uniform-abcdef0123"
RUN = "predict-frames.0.1-0123456789"
SCORE_RUN = "score-predictions.0.1-0123456789"
REF = FramePredictionsRef(run_id=RUN, frames_run_id=FRAMES_RUN)
SCORE_REF = PredictionScoreRef(run_id=SCORE_RUN)


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return make_dataset(tmp_path / "ds")


def _predict(
    ds: Dataset,
    sequence: str,
    camera: str = "",
    *,
    consumed: str = "",
    finished: bool = True,
) -> Path:
    """Record the run's predictions of one entry camera, read from *consumed*."""
    predictions = entry_predictions(
        frames_run_id=FRAMES_RUN, group="G", sequence=sequence, camera=camera
    )
    return record_entry_predictions(
        ds,
        predictions,
        run_id=RUN,
        finished=finished,
        consumed_frames_composition=consumed,
    )


def _extracted(
    ds: Dataset,
    sequence: str,
    camera: str = "",
    *,
    composition: str,
    run_id: str = FRAMES_RUN,
) -> None:
    """Record the frames row of one entry camera, cut from media of *composition*."""
    record_frames_row(
        ds,
        f"frames/uniform/{run_id}/G__{sequence}",
        run_id=run_id,
        group="G",
        sequence=sequence,
        camera=camera,
        media_composition=composition,
    )


def _drifted_run(ds: Dataset) -> None:
    """A run over two entries, the first of which was extracted again."""
    _extracted(ds, "S1", composition="new")
    _extracted(ds, "S2", composition="same")
    _ = _predict(ds, "S1", consumed="old")
    _ = _predict(ds, "S2", consumed="same")


def _score(ds: Dataset, *, finished: bool = True) -> Path:
    """Record a score run with every output, and return its directory."""
    run_root = prediction_kind_root(ds, "score-predictions") / SCORE_RUN
    run_root.mkdir(parents=True)
    for name in SCORE_RUN_OUTPUTS:
        _ = (run_root / name).write_bytes(b"")
    index = score_predictions_index(prediction_index_path(ds, "score-predictions"))
    index.append(
        [
            ScoreRunRow(
                run_id=SCORE_RUN,
                abs_path=Path(ds.relative_to_root(run_root)),
                prediction_runs=json.dumps([RUN]),
                model=json.dumps(["train-pose.0.1-0123456789"]),
                model_kind="train-pose",
                model_id="train-pose.0.1-0123456789",
                consumed_sets="[]",
                pose_id="1",
                n_scored=1,
                n_excluded_training=0,
                n_not_covered=0,
                training_frames_known=True,
                params_hash="h",
            )
        ]
    )
    if finished:
        index.mark_finished(SCORE_RUN)
    return run_root


def _measured(wire: InventoryWire) -> list[tuple[ArtifactStatus, CoverageWire, int]]:
    """Return each artifact's status, coverage and drift count."""
    return [
        (artifact.status, artifact.coverage, artifact.drift)
        for artifact in wire.artifacts
    ]


# --- frame predictions ------------------------------------------------------


def test_a_predictions_run_covers_the_entry_cameras_of_its_rows(
    dataset: Dataset,
) -> None:
    _ = _predict(dataset, "S1", "c1")
    _ = _predict(dataset, "S1", "c2")
    _ = _predict(dataset, "S2")

    found = dataset_inventory(dataset, kinds=["frame-predictions"])

    record = found.record(REF)
    assert found.errors == ()
    assert record is not None
    assert (record.name, record.status) == ("predict-frames", "complete")
    assert found.coverage(REF).present == {
        ("G", "S1", "c1"),
        ("G", "S1", "c2"),
        ("G", "S2", ""),
    }


def test_a_predictions_row_without_its_file_is_an_orphan_row(
    dataset: Dataset,
) -> None:
    _predict(dataset, "S1").unlink()
    _ = _predict(dataset, "S2")

    record = dataset_inventory(dataset, kinds=["frame-predictions"]).record(REF)

    assert record is not None
    assert record.orphan_rows == {("G", "S1", "")}
    assert record.status == "inconsistent"


def test_a_predictions_row_without_a_finish_is_an_unfinished_row(
    dataset: Dataset,
) -> None:
    _ = _predict(dataset, "S1")
    _ = _predict(dataset, "S2", finished=False)

    found = dataset_inventory(dataset, kinds=["frame-predictions"])

    record = found.record(REF)
    assert record is not None
    assert record.unfinished_rows == {("G", "S2", "")}
    assert found.coverage(REF).present == {("G", "S1", ""), ("G", "S2", "")}
    assert record.status == "complete"


def test_a_reextracted_entry_drifts_the_run(dataset: Dataset) -> None:
    _drifted_run(dataset)

    record = dataset_inventory(dataset, kinds=["frame-predictions"]).record(REF)

    assert record is not None
    assert record.drift == (("G", "S1"),)
    assert record.status == "complete-but-drifted"


def test_a_drifted_camera_drifts_its_entry(dataset: Dataset) -> None:
    _extracted(dataset, "S1", "c1", composition="new")
    _extracted(dataset, "S1", "c2", composition="same")
    _ = _predict(dataset, "S1", "c1", consumed="old")
    _ = _predict(dataset, "S1", "c2", consumed="same")

    record = dataset_inventory(dataset, kinds=["frame-predictions"]).record(REF)

    assert record is not None
    assert record.drift == (("G", "S1"),)


@pytest.mark.parametrize(
    ("consumed", "current"),
    [("", "new"), ("old", ""), ("old", None)],
    ids=["blank-consumed", "blank-current", "no-frames-row"],
)
def test_a_blank_composition_is_no_drift(
    dataset: Dataset, consumed: str, current: str | None
) -> None:
    if current is not None:
        _extracted(dataset, "S1", composition=current)
    _ = _predict(dataset, "S1", consumed=consumed)

    record = dataset_inventory(dataset, kinds=["frame-predictions"]).record(REF)

    assert record is not None
    assert (record.drift, record.status) == ((), "complete")


def test_only_the_runs_own_extraction_is_compared(dataset: Dataset) -> None:
    _extracted(dataset, "S1", composition="old")
    _extracted(dataset, "S1", composition="new", run_id=OTHER_FRAMES_RUN)
    _ = _predict(dataset, "S1", consumed="old")

    record = dataset_inventory(dataset, kinds=["frame-predictions"]).record(REF)

    assert record is not None
    assert record.drift == ()


def test_an_undeclared_frames_root_is_no_drift(dataset: Dataset) -> None:
    _extracted(dataset, "S1", composition="new")
    _ = _predict(dataset, "S1", consumed="old")
    reopened = open_dataset(dataset.manifest_path, ensure_roots=False)
    _ = reopened.roots.pop("frames")

    found = dataset_inventory(reopened, kinds=["frame-predictions"])

    record = found.record(REF)
    assert found.errors == ()
    assert record is not None
    assert record.drift == ()


@pytest.mark.parametrize(
    ("entry", "drift", "status"),
    [(("G", "S1"), 1, "complete-but-drifted"), (("G", "S2"), 0, "complete")],
)
def test_a_scoped_listing_keeps_the_drifted_entry_and_counts_it(
    dataset: Dataset, entry: Entry, drift: int, status: ArtifactStatus
) -> None:
    _drifted_run(dataset)
    group, sequence = entry

    result = invoke_ok(
        [
            "inventory",
            "--manifest",
            str(dataset.manifest_path),
            "--kind",
            "frame-predictions",
            "--entry",
            f"{group}:{sequence}",
            "--json",
        ]
    )

    (artifact,) = InventoryWire.model_validate_json(result.stdout).artifacts
    assert artifact.ref == f"frame-predictions/{RUN}/{FRAMES_RUN}"
    assert (artifact.coverage.target, artifact.drift, artifact.status) == (
        1,
        drift,
        status,
    )


@pytest.mark.parametrize(
    ("entry", "drift"), [(("G", "S1"), (("G", "S1"),)), (("G", "S2"), ())]
)
def test_a_scoped_scan_matches_the_scoped_projection(
    dataset: Dataset, entry: Entry, drift: tuple[Entry, ...]
) -> None:
    _drifted_run(dataset)
    scope = Scope(entries=[entry])

    scanned = dataset_inventory(dataset, kinds=["frame-predictions"], scope=scope)
    projected = inventory_wire(
        dataset, dataset_inventory(dataset), kinds=["frame-predictions"], scope=scope
    )

    record = scanned.record(REF)
    assert record is not None
    assert record.drift == drift
    assert _measured(inventory_wire(dataset, scanned)) == _measured(projected)


# --- prediction scores ------------------------------------------------------


def test_a_score_run_covers_its_own_run_id(dataset: Dataset) -> None:
    _ = _score(dataset)

    found = dataset_inventory(dataset, kinds=["prediction-score"])

    record = found.record(SCORE_REF)
    assert found.errors == ()
    assert record is not None
    assert (record.name, record.status) == ("score-predictions", "complete")
    assert found.coverage(SCORE_REF).present == {SCORE_RUN}


@pytest.mark.parametrize("output", SCORE_RUN_OUTPUTS)
def test_a_finished_score_run_without_an_output_is_inconsistent(
    dataset: Dataset, output: str
) -> None:
    (_score(dataset) / output).unlink()

    found = dataset_inventory(dataset, kinds=["prediction-score"])

    assert found.coverage(SCORE_REF).present == frozenset()
    assert found.status(SCORE_REF) == "inconsistent"


def test_an_unfinished_score_run_is_absent(dataset: Dataset) -> None:
    _ = _score(dataset, finished=False)

    found = dataset_inventory(dataset, kinds=["prediction-score"])

    assert found.status(SCORE_REF) == "absent"


# --- reads --------------------------------------------------------------------


def test_an_undeclared_root_reports_no_prediction_artifacts(
    dataset: Dataset,
) -> None:
    reopened = open_dataset(dataset.manifest_path, ensure_roots=False)
    _ = reopened.roots.pop("frame_predictions")

    found = dataset_inventory(reopened, kinds=["frame-predictions", "prediction-score"])

    assert (found.records, found.errors, found.unavailable_kinds) == (
        (),
        (),
        frozenset(),
    )


def test_a_read_only_inventory_creates_nothing(dataset: Dataset) -> None:
    _drifted_run(dataset)
    _ = _score(dataset)
    reopened = open_dataset(dataset.manifest_path, ensure_roots=False)
    before = tree_snapshot(dataset.base_dir)

    found = dataset_inventory(reopened, kinds=["frame-predictions", "prediction-score"])

    assert {record.ref for record in found.records} == {REF, SCORE_REF}
    assert_tree_unchanged(dataset.base_dir, before)
