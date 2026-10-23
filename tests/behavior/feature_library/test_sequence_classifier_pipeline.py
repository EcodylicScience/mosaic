"""`sequence-classifier` as recipe steps: screening on the training groups, then the test groups.

The screening request scopes the classifier to the training groups, and one
`evaluate-labels` step over them is the pooled out-of-fold score. The final request
widens the same step to the test groups. The `templates` run names the training set,
so the scope is not in the run id: the widened step is the same run, it fits nothing
again, and the final model scores the new entries, as with `xgboost`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from mosaic.behavior.feature_library.sequence_classifier import (
    SequenceClassifier,
    SequenceClassifierModelArtifact,
)
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.inventory.model import FeatureRunRef
from mosaic.core.pipeline.graph import PipelineRun, Recipe, plan_pipeline, run_pipeline
from mosaic.core.pipeline.graph.resolve import build_step_feature
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


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outputs(root: Path) -> dict[str, str]:
    return {path.name: _digest(path) for path in sorted(root.glob("*__*.parquet"))}


def test_the_final_request_over_the_test_groups_is_the_same_run_and_fits_nothing_again(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path / "ds")
    screening = run_pipeline(ds, _recipe(), step_scopes=SCREENING)
    screened = _root(ds, screening, "classifier", SCREENING)
    model = _digest(screened / "sequence_classifier_model.joblib")
    folds = _digest(screened / "cv_folds.parquet")
    train_outputs = _outputs(screened)
    assert len(train_outputs) == len(TRAIN_ENTRIES)

    final = run_pipeline(ds, _recipe(), step_scopes=FINAL)

    old = next(o for o in screening.outcomes if o.step_id == "classifier")
    new = next(o for o in final.outcomes if o.step_id == "classifier")
    assert old.run_id == new.run_id
    root = _root(ds, final, "classifier", FINAL)
    assert root == screened
    assert _digest(root / "sequence_classifier_model.joblib") == model
    assert _digest(root / "cv_folds.parquet") == folds
    after = _outputs(root)
    assert len(after) == len(TRAIN_ENTRIES) + len(TEST_ENTRIES)
    assert {name: after[name] for name in train_outputs} == train_outputs
    scored = pd.read_parquet(next(root.glob("test__t1.parquet")))
    assert {"prob_0", "prob_1", "prob_2", "predicted_label", "perspective"} <= set(
        scored.columns
    )
    assert len(scored) == 2 * 240


def test_a_first_fit_asked_over_the_wider_scope_is_the_fit_of_the_narrow_scope(
    tmp_path: Path,
) -> None:
    """The scope-free identity holds only if a test entry cannot move the fit."""
    narrow_ds, wide_ds = _dataset(tmp_path / "narrow"), _dataset(tmp_path / "wide")
    narrow = run_pipeline(narrow_ds, _recipe(), step_scopes=SCREENING)
    wide = run_pipeline(wide_ds, _recipe(), step_scopes=FINAL)

    old = _root(narrow_ds, narrow, "classifier", SCREENING)
    new = _root(wide_ds, wide, "classifier", FINAL)

    assert old.name == new.name
    for name in ("sequence_classifier_model.joblib", "cv_folds.parquet"):
        assert _digest(new / name) == _digest(old / name), name
    train_outputs = _outputs(old)
    assert {name: _outputs(new)[name] for name in train_outputs} == train_outputs


def test_a_train_entry_is_scored_out_of_fold_and_a_test_entry_by_the_final_model(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path / "ds")
    run = run_pipeline(ds, _recipe(), step_scopes=FINAL)
    root = _root(ds, run, "classifier", FINAL)
    bundle = SequenceClassifierModelArtifact().from_path(
        root / "sequence_classifier_model.joblib"
    )
    folds = pd.read_parquet(root / "cv_folds.parquet")
    fold_of = {
        (g, s): int(f)
        for g, s, f in zip(folds["group"], folds["sequence"], folds["fold"])
    }
    planned = plan_pipeline(ds, _recipe(), step_scopes=FINAL)
    feature = build_step_feature(planned.step("classifier").spec)
    assert isinstance(feature, SequenceClassifier)
    feature._adopt(bundle, folds=True)  # pyright: ignore[reportPrivateUsage]
    pairs = next(
        (ds.get_root("features") / planned.step("pairs").storage_name).glob("*/")
    )

    def table(group: str, sequence: str) -> pd.DataFrame:
        return pd.read_parquet(pairs / f"{group}__{sequence}.parquet")

    full = len(bundle["models"]) - 1
    train_df, test_df = table(*TRAIN_ENTRIES[0]), table(*TEST_ENTRIES[0])
    assert feature._scoring_model(train_df, bundle) == fold_of[TRAIN_ENTRIES[0]]  # pyright: ignore[reportPrivateUsage]
    assert feature._scoring_model(train_df, bundle) != full  # pyright: ignore[reportPrivateUsage]
    assert feature._scoring_model(test_df, bundle) == full  # pyright: ignore[reportPrivateUsage]
    stored_train = pd.read_parquet(root / "train__s1.parquet")
    stored_test = pd.read_parquet(root / "test__t1.parquet")
    for stored, source in ((stored_train, train_df), (stored_test, test_df)):
        again = feature.apply(source)
        pd.testing.assert_frame_equal(
            again[["prob_0", "prob_1", "prob_2"]].reset_index(drop=True),
            stored[["prob_0", "prob_1", "prob_2"]].reset_index(drop=True),
            atol=1e-5,
        )


def test_the_plan_of_the_final_request_applies_the_stored_fit_to_the_new_entries_only(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path / "ds")
    screening = run_pipeline(ds, _recipe(), step_scopes=SCREENING)
    stored = next(o for o in screening.outcomes if o.step_id == "classifier")

    planned = plan_pipeline(ds, _recipe(), step_scopes=FINAL).step("classifier")

    assert planned.scope_dependent is False
    assert planned.status == "partial"
    assert isinstance(planned.artifact, FeatureRunRef)
    assert planned.artifact.run_id == stored.run_id
    assert planned.coverage.missing == frozenset(TEST_ENTRIES)


def test_the_first_request_over_a_scope_without_the_train_entries_is_refused(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path / "ds")

    with pytest.raises(Exception, match="first run must be asked for over a scope"):
        _ = run_pipeline(
            ds,
            _recipe(),
            step_scopes={"labeled": TRAIN, "classifier": TEST, "evaluate": TEST},
        )


def test_a_rerun_of_one_request_is_a_cache_hit(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "ds")
    first = run_pipeline(ds, _recipe(), step_scopes=SCREENING)

    again = run_pipeline(ds, _recipe(), step_scopes=SCREENING)

    assert {o.state for o in again.outcomes} == {"cached"}
    assert {o.step_id: o.run_id for o in again.outcomes} == {
        o.step_id: o.run_id for o in first.outcomes
    }
