"""The ensemble over cross-validated classifiers as recipe steps, on the CalMS21 shape.

Three `xgboost` runs are fitted with the same folds. Two uniform averages combine
pairs of them, and a fitted blend combines the first run with both averages. The
averages pass the fold table on, so the blend finds a table for every member, and
`evaluate-labels` over the blend scores the same out-of-fold frames that the blend
chose its weights on.

The fitted weights come from the members' fold table, so no step hashes its scope:
widening the request to the test groups is the same runs, which score the new
entries with the weights they have.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.inventory.model import FeatureRunRef
from mosaic.core.pipeline.graph import PipelineRun, Recipe, plan_pipeline, run_pipeline
from mosaic.core.scope import Scope
from tests.helpers import CONVERT_LABELS_STEP, CONVERT_TRACKS_STEP, calms21_dataset

TRAIN = Scope(groups=["train"])
BOTH = Scope(groups=["train", "test"])
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


def _classifier(step: str, seed: int, cv_folds: int | None = 3) -> dict[str, object]:
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
            **({} if cv_folds is None else {"cv_folds": cv_folds}),
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
    return Recipe.model_validate(_recipe_dict(**blend))


def _shared_steps() -> list[dict[str, JsonValue]]:
    """The steps from the conversions through the templates."""
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
    ]


def _recipe_dict(**blend: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "blended",
        "steps": [
            *_shared_steps(),
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


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_widening_the_request_keeps_the_blend_and_scores_the_new_entries_with_its_weights(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path / "ds")
    narrow = run_pipeline(ds, _recipe(), step_scopes=STEP_SCOPES)
    blend = _root(ds, narrow, "blend")
    weights, report, folds = (
        _digest(blend / name)
        for name in ("ensemble.json", "ensemble_report.json", "cv_folds.parquet")
    )
    wide_scopes = {
        **STEP_SCOPES,
        **{step: BOTH for step in (*CLASSIFIERS, "avg_ab", "avg_bc", "blend")},
    }

    wide = run_pipeline(ds, _recipe(), step_scopes=wide_scopes)

    assert {o.step_id: o.run_id for o in wide.outcomes} == {
        o.step_id: o.run_id for o in narrow.outcomes
    }
    assert next(o for o in wide.outcomes if o.step_id == "blend").state == "ran", (
        "the blend computes the two new entries"
    )
    assert _digest(blend / "ensemble.json") == weights
    assert _digest(blend / "ensemble_report.json") == report
    assert _digest(blend / "cv_folds.parquet") == folds
    names = {path.name for path in blend.glob("*__*.parquet")}
    assert names == {
        f"{group}__{sequence}.parquet"
        for group, sequence in (*TRAIN_ENTRIES, *TEST_ENTRIES)
    }


def test_a_first_blend_asked_over_the_wider_scope_is_the_blend_of_the_narrow_scope(
    tmp_path: Path,
) -> None:
    """The scope-free identity holds only if a test entry cannot move the weights."""
    narrow_ds, wide_ds = _dataset(tmp_path / "narrow"), _dataset(tmp_path / "wide")
    narrow = run_pipeline(narrow_ds, _recipe(), step_scopes=STEP_SCOPES)
    wide_scopes = {
        **STEP_SCOPES,
        **{step: BOTH for step in (*CLASSIFIERS, "avg_ab", "avg_bc", "blend")},
    }
    wide = run_pipeline(wide_ds, _recipe(), step_scopes=wide_scopes)

    assert {o.step_id: o.run_id for o in wide.outcomes} == {
        o.step_id: o.run_id for o in narrow.outcomes
    }
    old, new = _root(narrow_ds, narrow, "blend"), _root(wide_ds, wide, "blend")
    for name in ("ensemble.json", "ensemble_report.json", "cv_folds.parquet"):
        assert _digest(new / name) == _digest(old / name), name
    for path in old.glob("*__*.parquet"):
        assert _digest(new / path.name) == _digest(path), path.name


def test_a_first_blend_asked_over_a_scope_without_the_members_fold_entries_is_refused(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path / "ds")
    scopes = {
        **STEP_SCOPES,
        **{step: BOTH for step in (*CLASSIFIERS, "avg_ab", "avg_bc")},
        "blend": Scope(groups=["test"]),
    }

    with pytest.raises(Exception, match="first run must be asked for over a scope"):
        _ = run_pipeline(ds, _recipe(), step_scopes=scopes)


def test_the_plan_of_a_widened_blend_names_the_new_entries_and_the_stored_run(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path / "ds")
    narrow = run_pipeline(ds, _recipe(), step_scopes=STEP_SCOPES)
    stored = next(o for o in narrow.outcomes if o.step_id == "blend")
    wide_scopes = {
        **STEP_SCOPES,
        **{step: BOTH for step in (*CLASSIFIERS, "avg_ab", "avg_bc", "blend")},
    }

    planned = plan_pipeline(ds, _recipe(), step_scopes=wide_scopes).step("blend")

    assert planned.scope_dependent is False
    assert isinstance(planned.artifact, FeatureRunRef)
    assert planned.artifact.run_id == stored.run_id
    assert planned.coverage.missing == frozenset(TEST_ENTRIES)


def test_fixed_weights_average_members_of_which_only_some_were_cross_validated(
    tmp_path: Path,
) -> None:
    """Without `cv_folds` on one member there is no table, and an average needs none."""
    ds = _dataset(tmp_path / "ds")
    steps = [
        *_shared_steps(),
        _classifier("clf_a", 1),
        _classifier("clf_c", 3, cv_folds=None),
        {
            "id": "mixed",
            "type": "feature",
            "feature": "probability-ensemble",
            "inputs": [{"step": "clf_a"}, {"step": "clf_c"}],
            "params": {"default_class": 0},
        },
    ]
    recipe = Recipe.model_validate(
        {"schema_version": 1, "name": "mixed", "steps": steps}
    )
    scopes = {step: TRAIN for step in ("labeled", "clf_a", "clf_c", "mixed")}

    run = run_pipeline(ds, recipe, step_scopes=scopes)

    assert {o.state for o in run.outcomes} == {"ran"}
    planned = plan_pipeline(ds, recipe, step_scopes=scopes).step("mixed")
    outcome = next(o for o in run.outcomes if o.step_id == "mixed")
    root = ds.get_root("features") / planned.storage_name / str(outcome.run_id)
    assert (root / "ensemble.json").exists()
    assert not (root / "cv_folds.parquet").exists()
    assert len(list(root.glob("*__*.parquet"))) == len(TRAIN_ENTRIES)
