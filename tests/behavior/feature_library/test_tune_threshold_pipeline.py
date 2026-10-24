"""``tune-threshold`` as a recipe step, on the CalMS21 shape.

A cross-validated ``xgboost`` run is tuned, and a one-member ``probability-ensemble``
applies the thresholds through ``decision_threshold: {"step": "tune"}``.
``evaluate-labels`` over that ensemble scores the same out-of-fold frames the tuning
chose its thresholds on, so its score is the plain estimate of the report.

The thresholds come from the classifier's fold table, so no step hashes its scope:
widening the request to the test groups is the same runs, which score the new
entries with the thresholds they have.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from mosaic.behavior.feature_library.classifier_common import read_thresholds
from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    PipelineRun,
    Recipe,
    check_recipe,
    plan_pipeline,
    run_pipeline,
)
from mosaic.core.scope import Scope
from tests.helpers import CONVERT_LABELS_STEP, CONVERT_TRACKS_STEP, calms21_dataset

TRAIN = Scope(groups=["train"])
BOTH = Scope(groups=["train", "test"])
TRAIN_ENTRIES = [("train", f"s{i}") for i in range(1, 7)]
TEST_ENTRIES = [("test", "t1"), ("test", "t2")]
FITTED = ("clf", "tune", "apply", "evaluate", "labeled")
STEP_SCOPES = {step: TRAIN for step in FITTED}
WIDE_SCOPES = {**STEP_SCOPES, **{step: BOTH for step in ("clf", "tune", "apply")}}


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


def _steps(
    classifier: dict[str, JsonValue] | None = None, **tune: JsonValue
) -> list[dict[str, JsonValue]]:
    return [
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
            "id": "clf",
            "type": "feature",
            "feature": "xgboost",
            "inputs": [{"step": "pairs"}],
            "params": {
                "templates": {"step": "labeled"},
                "default_class": 0,
                "n_estimators": 5,
                "n_jobs": 1,
                "random_state": 1,
                "cv_folds": 3,
                **(classifier or {}),
            },
        },
        {
            "id": "tune",
            "type": "feature",
            "feature": "tune-threshold",
            "inputs": [{"step": "clf"}],
            "params": {
                "labels": {"step": "labels"},
                "default_class": 0,
                "average_over": [0, 1, 2],
                **tune,
            },
        },
        {
            "id": "apply",
            "type": "feature",
            "feature": "probability-ensemble",
            "inputs": [{"step": "clf"}],
            "params": {
                "default_class": 0,
                "decision_threshold": {"step": "tune"},
            },
        },
        {
            "id": "evaluate",
            "type": "feature",
            "feature": "evaluate-labels",
            "inputs": [{"step": "apply"}],
            "params": {
                "labels": {"step": "labels"},
                "perspective": 0,
                "average_over": [0, 1, 2],
            },
        },
    ]


def _recipe(
    classifier: dict[str, JsonValue] | None = None, **tune: JsonValue
) -> Recipe:
    return Recipe.model_validate(
        {"schema_version": 1, "name": "tuned", "steps": _steps(classifier, **tune)}
    )


def _root(
    ds: Dataset, run: PipelineRun, step: str, recipe: Recipe | None = None
) -> Path:
    outcome = next(o for o in run.outcomes if o.step_id == step)
    planned = plan_pipeline(
        ds, _recipe() if recipe is None else recipe, step_scopes=STEP_SCOPES
    ).step(step)
    return ds.get_root("features") / planned.storage_name / str(outcome.run_id)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_recipe_tunes_a_classifier_and_applies_the_thresholds_in_one_run(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path / "ds")

    run = run_pipeline(ds, _recipe(), step_scopes=STEP_SCOPES)

    assert {o.state for o in run.outcomes} == {"ran"}
    tune = _root(ds, run, "tune")
    thresholds = read_thresholds(tune / "thresholds.parquet")
    assert sorted(thresholds) == [0, 1, 2]
    assert thresholds[0] == 1.0
    report = json.loads((tune / "tune_report.json").read_text())
    assert report["n_entries"] == len(TRAIN_ENTRIES)
    assert report["nested"] is not None
    assert {int(c): t for c, t in report["thresholds"].items()} == {
        c: thresholds[c] for c in (1, 2)
    }
    # The step the thresholds feed scores the frames they were chosen on.
    metrics = json.loads((_root(ds, run, "evaluate") / "metrics.json").read_text())
    assert metrics["macro_f1"] == report["plain"]["value"]


def test_the_ensemble_and_the_tune_step_assign_the_same_classes(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "ds")
    run = run_pipeline(ds, _recipe(), step_scopes=STEP_SCOPES)

    for group, sequence in TRAIN_ENTRIES:
        name = f"{group}__{sequence}.parquet"
        tuned = pd.read_parquet(_root(ds, run, "tune") / name)
        applied = pd.read_parquet(_root(ds, run, "apply") / name)
        assert tuned["predicted_label"].tolist() == applied["predicted_label"].tolist()


def test_the_same_recipe_is_a_cache_hit(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "ds")
    first = run_pipeline(ds, _recipe(), step_scopes=STEP_SCOPES)

    again = run_pipeline(ds, _recipe(), step_scopes=STEP_SCOPES)

    assert {o.state for o in again.outcomes} == {"cached"}
    assert {o.step_id: o.run_id for o in again.outcomes} == {
        o.step_id: o.run_id for o in first.outcomes
    }


def test_widening_the_request_keeps_the_thresholds_and_scores_the_new_entries(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path / "ds")
    narrow = run_pipeline(ds, _recipe(), step_scopes=STEP_SCOPES)
    tune = _root(ds, narrow, "tune")
    thresholds, report = (
        _digest(tune / name) for name in ("thresholds.parquet", "tune_report.json")
    )

    wide = run_pipeline(ds, _recipe(), step_scopes=WIDE_SCOPES)

    assert {o.step_id: o.run_id for o in wide.outcomes} == {
        o.step_id: o.run_id for o in narrow.outcomes
    }
    assert _digest(tune / "thresholds.parquet") == thresholds
    assert _digest(tune / "tune_report.json") == report
    for step in ("tune", "apply"):
        names = {path.name for path in _root(ds, wide, step).glob("*__*.parquet")}
        assert names == {
            f"{group}__{sequence}.parquet"
            for group, sequence in (*TRAIN_ENTRIES, *TEST_ENTRIES)
        }, step


def test_a_first_tune_asked_over_the_wider_scope_is_the_tune_of_the_narrow_scope(
    tmp_path: Path,
) -> None:
    """The scope-free identity holds only if a test entry cannot move the thresholds."""
    narrow_ds, wide_ds = _dataset(tmp_path / "narrow"), _dataset(tmp_path / "wide")
    narrow = run_pipeline(narrow_ds, _recipe(), step_scopes=STEP_SCOPES)

    wide = run_pipeline(wide_ds, _recipe(), step_scopes=WIDE_SCOPES)

    assert {o.step_id: o.run_id for o in wide.outcomes} == {
        o.step_id: o.run_id for o in narrow.outcomes
    }
    old, new = _root(narrow_ds, narrow, "tune"), _root(wide_ds, wide, "tune")
    for name in ("thresholds.parquet", "tune_report.json"):
        assert _digest(new / name) == _digest(old / name), name


def test_the_plan_reads_the_step_as_scope_free_and_fitting(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "ds")
    _ = run_pipeline(ds, _recipe(), step_scopes=STEP_SCOPES)
    plan = plan_pipeline(ds, _recipe(), step_scopes=STEP_SCOPES)

    tune = plan.step("tune")

    assert tune.scope_dependent is False
    assert tune.fits is True
    assert not check_recipe(_recipe())


def test_a_classifier_cannot_take_its_own_tuning_as_its_threshold() -> None:
    """The step needs the classifier's run, so naming it back is a cycle."""
    steps = _steps()
    for step in steps:
        if step["id"] == "clf":
            params = step["params"]
            assert isinstance(params, dict)
            params["decision_threshold"] = {"step": "tune"}

    problems = check_recipe(
        Recipe.model_validate({"schema_version": 1, "name": "cycle", "steps": steps})
    )

    assert any("cycle" in problem.message for problem in problems)


def test_one_training_sequence_is_tuned_and_scored_out_of_fold_by_block(
    tmp_path: Path,
) -> None:
    """The acceptance combination: block folds, tune-threshold, a one-member ensemble.

    A single training sequence leaves no sequence to hold out, so the classifier cuts
    it into blocks. The tuning reads the fold table of blocks, and what the ensemble
    applies, scored by evaluate-labels, is the plain estimate of the report.
    """
    ds = calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": {"s1": 600}}, "test": {"a": {"t1": 240}}},
        seed=11,
        n_classes=3,
    )
    by_block = _recipe(
        {"cv_unit": "block", "cv_blocks_per_sequence": 6, "cv_gap_frames": 5}
    )

    run = run_pipeline(ds, by_block, step_scopes=STEP_SCOPES)

    assert {o.state for o in run.outcomes} == {"ran"}
    folds = pd.read_parquet(_root(ds, run, "clf", by_block) / "cv_folds.parquet")
    assert "start" in folds.columns
    tune = _root(ds, run, "tune", by_block)
    report = json.loads((tune / "tune_report.json").read_text())
    assert report["n_entries"] == 1
    assert report["nested"] is not None
    # Only the frames inside a block have an out-of-fold probability.
    inside = int((folds["stop"] - folds["start"]).sum())
    assert 0 < report["n_rows"] <= inside
    metrics = json.loads(
        (_root(ds, run, "evaluate", by_block) / "metrics.json").read_text()
    )
    assert metrics["macro_f1"] == report["plain"]["value"]
