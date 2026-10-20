"""The ensemble over cross-validated classifiers as recipe steps, on the CalMS21 shape.

Three `xgboost` runs are fitted with the same folds. Two uniform averages combine
pairs of them, and a fitted blend combines the first run with both averages. The
averages pass the fold table on, so the blend finds a table for every member, and
`evaluate-labels` over the blend scores the same out-of-fold frames that the blend
chose its weights on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import PipelineRun, Recipe, plan_pipeline, run_pipeline
from mosaic.core.scope import Scope
from tests.helpers import CONVERT_LABELS_STEP, CONVERT_TRACKS_STEP, calms21_dataset

TRAIN = Scope(groups=["train"])
TRAIN_ENTRIES = [("train", f"s{i}") for i in range(1, 7)]
TEST_ENTRIES = [("test", "t1"), ("test", "t2")]
CLASSIFIERS = ("clf_a", "clf_b", "clf_c")
STEP_SCOPES = {
    step: TRAIN
    for step in (*CLASSIFIERS, "avg_ab", "avg_bc", "blend", "evaluate", "labeled")
}


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


def _classifier(step: str, seed: int) -> dict[str, object]:
    return {
        "id": step,
        "type": "feature",
        "feature": "xgboost",
        "inputs": [{"step": "pairs"}],
        "params": {
            "templates": {"step": "labeled"},
            "default_class": 0,
            "n_estimators": 5,
            "n_jobs": 1,
            "random_state": seed,
            "cv_folds": 3,
        },
    }


def _average(step: str, members: tuple[str, str]) -> dict[str, object]:
    return {
        "id": step,
        "type": "feature",
        "feature": "probability-ensemble",
        "inputs": [{"step": member} for member in members],
        "params": {"default_class": 0},
    }


def _recipe(**blend: object) -> Recipe:
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "blended",
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
                _classifier("clf_a", 1),
                _classifier("clf_b", 2),
                _classifier("clf_c", 3),
                _average("avg_ab", ("clf_a", "clf_b")),
                _average("avg_bc", ("clf_b", "clf_c")),
                {
                    "id": "blend",
                    "type": "feature",
                    "feature": "probability-ensemble",
                    "inputs": [
                        {"step": "clf_a"},
                        {"step": "avg_ab"},
                        {"step": "avg_bc"},
                    ],
                    "params": {
                        "default_class": 0,
                        "weights": "fit",
                        "labels": {"step": "labels"},
                        "score_classes": [0, 1, 2],
                        **blend,
                    },
                },
                {
                    "id": "evaluate",
                    "type": "feature",
                    "feature": "evaluate-labels",
                    "inputs": [{"step": "blend"}],
                    "params": {
                        "labels": {"step": "labels"},
                        "perspective": 0,
                        "average_over": [0, 1, 2],
                    },
                },
            ],
        }
    )


def _root(ds: Dataset, run: PipelineRun, step: str) -> Path:
    outcome = next(o for o in run.outcomes if o.step_id == step)
    planned = plan_pipeline(ds, _recipe(), step_scopes=STEP_SCOPES).step(step)
    return ds.get_root("features") / planned.storage_name / str(outcome.run_id)


def _folds(root: Path) -> dict[tuple[str, str], int]:
    table = pd.read_parquet(root / "cv_folds.parquet")
    return dict(zip(zip(table["group"], table["sequence"]), table["fold"]))


def test_a_fitted_blend_of_two_averages_and_a_classifier_runs_end_to_end(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path / "ds")

    run = run_pipeline(ds, _recipe(), step_scopes=STEP_SCOPES)

    assert {o.state for o in run.outcomes} == {"ran"}
    classifier_folds = _folds(_root(ds, run, "clf_a"))
    assert sorted(classifier_folds) == sorted(TRAIN_ENTRIES)
    for step in ("clf_b", "clf_c", "avg_ab", "avg_bc", "blend"):
        assert _folds(_root(ds, run, step)) == classifier_folds, step

    root = _root(ds, run, "blend")
    report = json.loads((root / "ensemble_report.json").read_text())
    weights = report["weights_used"]
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 1e-12
    assert report["objective"] >= max(report["member_objectives"])
    assert report["n_entries"] == len(TRAIN_ENTRIES)

    # The score of the blend over the entries it was fitted on is the objective.
    metrics = json.loads((_root(ds, run, "evaluate") / "metrics.json").read_text())
    assert metrics["macro_f1"] == report["objective"]


def test_an_average_of_the_same_inputs_is_a_cache_hit(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "ds")
    first = run_pipeline(ds, _recipe(), step_scopes=STEP_SCOPES)

    again = run_pipeline(ds, _recipe(), step_scopes=STEP_SCOPES)

    assert {o.state for o in again.outcomes} == {"cached"}
    assert {o.step_id: o.run_id for o in again.outcomes} == {
        o.step_id: o.run_id for o in first.outcomes
    }
