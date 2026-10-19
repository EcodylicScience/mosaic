"""`xgboost` cross-validation as recipe steps: screening on the training groups, then the test groups.

The screening request scopes the classifier to the training groups, and one
`evaluate-labels` step over them is the pooled out-of-fold score. The final request
widens the same step to the test groups, which is the same run, so the model is a cache
hit and only the added entries are computed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.metrics import f1_score  # pyright: ignore[reportUnknownVariableType]
from xgboost import XGBClassifier

from mosaic.behavior.feature_library.helpers import entry_identity
from mosaic.behavior.feature_library.xgboost_feature import XgboostFeature
from mosaic.core.dataset import Dataset
from mosaic.core.evaluation import (
    LABEL_COLUMN,
    PREDICTION_COLUMN,
    EvaluationReport,
    align_predictions_to_labels,
)
from mosaic.core.pipeline.graph import PipelineRun, Recipe, plan_pipeline, run_pipeline
from mosaic.core.pipeline.index import feature_index, feature_index_path
from mosaic.core.pipeline.types import ResultColumn
from mosaic.core.scope import Scope
from tests.helpers import CONVERT_LABELS_STEP, CONVERT_TRACKS_STEP, calms21_dataset

TRAIN, TEST = Scope(groups=["train"]), Scope(groups=["test"])
BOTH = Scope(groups=["train", "test"])
TRAIN_ENTRIES = [("train", f"s{i}") for i in range(1, 7)]
TEST_ENTRIES = [("test", "t1"), ("test", "t2")]


def dataset(base: Path) -> Dataset:
    return calms21_dataset(
        base,
        {
            "train": {"a": {name: 240 for _, name in TRAIN_ENTRIES}},
            "test": {"a": {name: 240 for _, name in TEST_ENTRIES}},
        },
        seed=11,
        n_classes=3,
    )


def recipe(**classifier: object) -> Recipe:
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "cross-validated",
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
                    "feature": "xgboost",
                    "inputs": [{"step": "pairs"}],
                    "params": {
                        "templates": {"step": "labeled"},
                        "default_class": 0,
                        "n_estimators": 5,
                        "n_jobs": 1,
                        "cv_folds": 3,
                        **classifier,
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


SCREENING = {"labeled": TRAIN, "classifier": TRAIN, "evaluate": TRAIN}
FINAL = {"labeled": TRAIN, "classifier": BOTH, "evaluate": TEST}


@pytest.fixture
def fit_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """One entry per `XGBClassifier.fit`."""
    calls: list[int] = []
    original = XGBClassifier.fit

    def spy(self: XGBClassifier, X: Any, y: Any, **kwargs: Any) -> XGBClassifier:
        calls.append(len(calls))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(XGBClassifier, "fit", spy)
    return calls


@pytest.fixture
def fit_invocations(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """One entry per `XgboostFeature.fit`."""
    calls: list[int] = []
    original = XgboostFeature.fit

    def spy(self: XgboostFeature, inputs: Any) -> None:
        calls.append(len(calls))
        original(self, inputs)

    monkeypatch.setattr(XgboostFeature, "fit", spy)
    return calls


@pytest.fixture
def scored(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """The entry of every `XgboostFeature.apply`."""
    entries: list[tuple[str, str]] = []
    original = XgboostFeature.apply

    def spy(self: XgboostFeature, df: pd.DataFrame) -> pd.DataFrame:
        entries.append(entry_identity(df))
        return original(self, df)

    monkeypatch.setattr(XgboostFeature, "apply", spy)
    return entries


def classifier_root(ds: Dataset, run: PipelineRun) -> Path:
    outcome = next(o for o in run.outcomes if o.step_id == "classifier")
    planned = plan_pipeline(ds, recipe(), step_scopes=SCREENING).step("classifier")
    return ds.get_root("features") / planned.storage_name / str(outcome.run_id)


def test_the_screening_run_fits_the_folds_and_the_full_model_in_one_fit(
    tmp_path: Path,
    fit_calls: list[int],
    fit_invocations: list[int],
    scored: list[tuple[str, str]],
) -> None:
    ds = dataset(tmp_path / "ds")

    run = run_pipeline(ds, recipe(), step_scopes=SCREENING)

    assert [outcome.state for outcome in run.outcomes] == ["ran"] * 7
    assert len(fit_invocations) == 1
    assert len(fit_calls) == 3 + 1
    assert sorted(scored) == sorted(TRAIN_ENTRIES)
    folds = pd.read_parquet(classifier_root(ds, run) / "cv_folds.parquet")
    assert sorted(zip(folds["group"], folds["sequence"])) == sorted(TRAIN_ENTRIES)
    assert sorted(folds["fold"].value_counts().tolist()) == [2, 2, 2]


def test_a_rerun_fits_nothing_and_keeps_its_run_id(
    tmp_path: Path, fit_calls: list[int], fit_invocations: list[int]
) -> None:
    ds = dataset(tmp_path / "ds")
    first = run_pipeline(ds, recipe(), step_scopes=SCREENING)
    fitted = len(fit_calls)

    again = run_pipeline(ds, recipe(), step_scopes=SCREENING)

    assert len(fit_invocations) == 1
    assert len(fit_calls) == fitted
    assert [o.run_id for o in again.outcomes] == [o.run_id for o in first.outcomes]


def test_widening_to_the_test_groups_keeps_the_run_and_computes_only_the_added_entries(
    tmp_path: Path, fit_calls: list[int], scored: list[tuple[str, str]]
) -> None:
    ds = dataset(tmp_path / "ds")
    first = run_pipeline(ds, recipe(), step_scopes=SCREENING)
    fitted, applied = len(fit_calls), len(scored)

    widened = run_pipeline(ds, recipe(), step_scopes=FINAL)

    run_id = {o.step_id: o.run_id for o in first.outcomes}["classifier"]
    assert {o.step_id: o.run_id for o in widened.outcomes}["classifier"] == run_id
    assert len(fit_calls) == fitted
    assert sorted(scored[applied:]) == sorted(TEST_ENTRIES)
    planned = plan_pipeline(ds, recipe(), step_scopes=FINAL).step("classifier")
    rows = feature_index(feature_index_path(ds, planned.storage_name)).read(
        run_id=run_id
    )
    assert sorted(zip(rows["group"], rows["sequence"])) == sorted(
        TRAIN_ENTRIES + TEST_ENTRIES
    )


def test_the_pooled_out_of_fold_score_is_the_f1_of_the_concatenated_predictions(
    tmp_path: Path,
) -> None:
    ds = dataset(tmp_path / "ds")
    run = run_pipeline(ds, recipe(), step_scopes=SCREENING)
    outcomes = {o.step_id: o for o in run.outcomes}
    planned = plan_pipeline(ds, recipe(), step_scopes=SCREENING)
    features = ds.get_root("features")
    evaluate = features / planned.step("evaluate").storage_name
    report = EvaluationReport.model_validate_json(
        (evaluate / str(outcomes["evaluate"].run_id) / "metrics.json").read_text()
    )

    aligned = align_predictions_to_labels(
        ds,
        ResultColumn(
            feature=planned.step("classifier").storage_name,
            column="predicted_label",
            run_id=str(outcomes["classifier"].run_id),
        ),
        scope=TRAIN,
        perspective=0,
    )

    assert report.n_entries == len(TRAIN_ENTRIES)
    assert report.n_scored == len(aligned.frame)
    pooled = f1_score(
        aligned.frame[LABEL_COLUMN],
        aligned.frame[PREDICTION_COLUMN],
        labels=[0, 1, 2],
        average="macro",
    )
    assert report.macro_f1 == pytest.approx(pooled)


@pytest.mark.parametrize(
    "changed",
    [
        {"cv_folds": 4},
        {"cv_folds": 3, "cv_seed": 1},
        {"cv_folds": 3, "random_state": 7},
    ],
    ids=["folds", "seed", "random_state"],
)
def test_the_folds_and_their_seed_are_part_of_the_run_id(
    tmp_path: Path, changed: dict[str, object]
) -> None:
    ds = dataset(tmp_path / "ds")
    _ = run_pipeline(ds, recipe(), step_scopes=SCREENING)

    base = plan_pipeline(ds, recipe(), step_scopes=SCREENING).step("classifier")
    other = plan_pipeline(ds, recipe(**changed), step_scopes=SCREENING).step(
        "classifier"
    )

    assert base.run_id is not None
    assert other.run_id != base.run_id
