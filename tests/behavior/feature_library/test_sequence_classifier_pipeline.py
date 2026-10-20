"""`sequence-classifier` as recipe steps: screening on the training groups, then the test groups.

The screening request scopes the classifier to the training groups, and one
`evaluate-labels` step over them is the pooled out-of-fold score. The final request
widens the same step to the test groups. A step that fits from its stream holds its
scope in its run id, so the widened step is a new run that fits again, unlike
`xgboost`, which fits from templates and keeps its run id.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import PipelineRun, Recipe, plan_pipeline, run_pipeline
from mosaic.core.scope import Scope
from tests.helpers import CONVERT_LABELS_STEP, CONVERT_TRACKS_STEP, calms21_dataset

pytestmark = pytest.mark.identity

TRAIN, TEST = Scope(groups=["train"]), Scope(groups=["test"])
BOTH = Scope(groups=["train", "test"])
TRAIN_ENTRIES = [("train", f"s{i}") for i in range(1, 7)]
TEST_ENTRIES = [("test", "t1"), ("test", "t2")]
SCREENING = {"labeled": TRAIN, "classifier": TRAIN, "evaluate": TRAIN}
FINAL = {"labeled": TRAIN, "classifier": BOTH, "evaluate": TEST}


def _dataset(base: Path) -> Dataset:
    return calms21_dataset(
        base,
        {
            "train": {"a": {name: 240 for _, name in TRAIN_ENTRIES}},
            "test": {"a": {name: 240 for _, name in TEST_ENTRIES}},
        },
        seed=11,
        n_classes=3,
    )


def _recipe() -> Recipe:
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "sequence-network",
            "steps": [
                CONVERT_TRACKS_STEP,
                CONVERT_LABELS_STEP,
                {
                    "id": "heading",
                    "type": "feature",
                    "feature": "heading",
                    "inputs": ["tracks"],
                    "tracks": {"step": "tracks"},
                    "params": {"method": "two_point", "front_idx": 0, "rear_idx": 3},
                },
                {
                    "id": "pairs",
                    "type": "feature",
                    "feature": "pair-position",
                    "inputs": ["tracks", {"step": "heading"}],
                    "tracks": {"step": "tracks"},
                },
                {
                    "id": "labeled",
                    "type": "feature",
                    "feature": "extract-labeled-templates",
                    "inputs": [{"step": "pairs"}],
                    "params": {
                        "labels": {"step": "labels"},
                        "n_total": 300,
                        "test_fraction": 0.0,
                    },
                },
                {
                    "id": "classifier",
                    "type": "feature",
                    "feature": "sequence-classifier",
                    "inputs": [{"step": "pairs"}],
                    "params": {
                        "templates": {"step": "labeled"},
                        "labels": {"step": "labels"},
                        "default_class": 0,
                        "device": "cpu",
                        "head": "tcn",
                        "head_config": {"hidden": 8, "blocks": 2},
                        "steps": 30,
                        "batch_size": 4,
                        "crop_length": 60,
                        "learning_rate": 0.01,
                        "cv_folds": 3,
                    },
                },
                {
                    "id": "evaluate",
                    "type": "feature",
                    "feature": "evaluate-labels",
                    "inputs": [{"step": "classifier"}],
                    "params": {
                        "labels": {"step": "labels"},
                        "perspective": 0,
                        "average_over": [0, 1, 2],
                    },
                },
            ],
        }
    )


def _root(ds: Dataset, run: PipelineRun, step: str, scopes: dict[str, Scope]) -> Path:
    outcome = next(o for o in run.outcomes if o.step_id == step)
    planned = plan_pipeline(ds, _recipe(), step_scopes=scopes).step(step)
    return ds.get_root("features") / planned.storage_name / str(outcome.run_id)


def test_the_screening_request_scores_every_training_entry_out_of_fold(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path / "ds")

    run = run_pipeline(ds, _recipe(), step_scopes=SCREENING)

    assert {o.state for o in run.outcomes} == {"ran"}
    root = _root(ds, run, "classifier", SCREENING)
    folds = pd.read_parquet(root / "cv_folds.parquet")
    assert sorted(zip(folds["group"], folds["sequence"])) == sorted(TRAIN_ENTRIES)
    assert sorted(folds["fold"].value_counts().tolist()) == [2, 2, 2]
    metrics = json.loads(
        (_root(ds, run, "evaluate", SCREENING) / "metrics.json").read_text()
    )
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_the_final_request_over_the_test_groups_is_a_new_run_that_fits_again(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path / "ds")
    screening = run_pipeline(ds, _recipe(), step_scopes=SCREENING)

    final = run_pipeline(ds, _recipe(), step_scopes=FINAL)

    old = next(o for o in screening.outcomes if o.step_id == "classifier")
    new = next(o for o in final.outcomes if o.step_id == "classifier")
    assert old.run_id != new.run_id
    scored = pd.read_parquet(
        next(_root(ds, final, "classifier", FINAL).glob("test__t1.parquet"))
    )
    assert {"prob_0", "prob_1", "prob_2", "predicted_label", "perspective"} <= set(
        scored.columns
    )
    assert len(scored) == 2 * 240


def test_a_rerun_of_one_request_is_a_cache_hit(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "ds")
    first = run_pipeline(ds, _recipe(), step_scopes=SCREENING)

    again = run_pipeline(ds, _recipe(), step_scopes=SCREENING)

    assert {o.state for o in again.outcomes} == {"cached"}
    assert {o.step_id: o.run_id for o in again.outcomes} == {
        o.step_id: o.run_id for o in first.outcomes
    }
