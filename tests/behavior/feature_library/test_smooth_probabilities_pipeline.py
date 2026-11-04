"""``smooth-probabilities`` as recipe steps over a cross-validated ``xgboost``, on the CalMS21 shape.

The classifier is smoothed twice, ``tune-threshold`` tunes over the first smoothing,
a ``method: "none"`` step applies the tuned thresholds and merges short bouts, and a
fitted ``probability-ensemble`` blends the two smoothings. Each smoothing passes the
classifier's fold table on, so the steps after it find one where they find a
classifier's.

A ``mask`` on the labels of the classifier's templates reaches the steps that score
against labels through the smoothing run's record of its input, so a step that
reads the labels without the mask is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import PipelineRun, Recipe, plan_pipeline, run_pipeline
from mosaic.core.scope import Scope
from tests.helpers import CONVERT_LABELS_STEP, CONVERT_TRACKS_STEP, calms21_dataset

TRAIN = Scope(groups=["train"])
TRAIN_ENTRIES = [("train", f"s{i}") for i in range(1, 7)]
SCOPED = (
    "labeled",
    "clf",
    "smooth",
    "smooth_wide",
    "tune",
    "apply",
    "blend",
    "average",
    "evaluate",
)
LABELS: dict[str, JsonValue] = {"step": "labels"}
MASKED: dict[str, JsonValue] = {"step": "labels", "mask": [2]}


def _dataset(base: Path) -> Dataset:
    return calms21_dataset(
        base,
        {"train": {"a": {name: 240 for _, name in TRAIN_ENTRIES}}},
        seed=11,
        n_classes=3,
    )


def _shared_steps(
    templates_labels: dict[str, JsonValue], classifier: dict[str, JsonValue]
) -> list[dict[str, JsonValue]]:
    """The steps from the conversions through the classifier and its two smoothings."""
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
                "labels": templates_labels,
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
                **classifier,
            },
        },
        {
            "id": "smooth",
            "type": "feature",
            "feature": "smooth-probabilities",
            "inputs": [{"step": "clf"}],
            "params": {"default_class": 0},
        },
        {
            "id": "smooth_wide",
            "type": "feature",
            "feature": "smooth-probabilities",
            "inputs": [{"step": "clf"}],
            "params": {"default_class": 0, "sigma": 4.0},
        },
    ]


def _tune(labels: dict[str, JsonValue], **params: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": "tune",
        "type": "feature",
        "feature": "tune-threshold",
        "inputs": [{"step": "smooth"}],
        "params": {"labels": labels, "default_class": 0, **params},
    }


def _recipe(*steps: dict[str, JsonValue], name: str = "smoothed") -> Recipe:
    return Recipe.model_validate(
        {"schema_version": 1, "name": name, "steps": list(steps)}
    )


def _full_recipe(classifier: dict[str, JsonValue] | None = None) -> Recipe:
    return _recipe(
        *_shared_steps(LABELS, classifier or {}),
        _tune(LABELS, average_over=[0, 1, 2]),
        {
            "id": "apply",
            "type": "feature",
            "feature": "smooth-probabilities",
            "inputs": [{"step": "smooth"}],
            "params": {
                "method": "none",
                "default_class": 0,
                "decision_threshold": {"step": "tune"},
                "min_bout_frames": 3,
            },
        },
        {
            "id": "blend",
            "type": "feature",
            "feature": "probability-ensemble",
            "inputs": [{"step": "smooth"}, {"step": "smooth_wide"}],
            "params": {
                "default_class": 0,
                "weights": "fit",
                "labels": LABELS,
                "score_classes": [0, 1, 2],
                "grid_step": 0.25,
            },
        },
    )


def _scopes(recipe: Recipe) -> dict[str, Scope]:
    """The training scope for each step of *recipe* past the conversions."""
    return {step.id: TRAIN for step in recipe.steps if step.id in SCOPED}


def _root(ds: Dataset, run: PipelineRun, recipe: Recipe, step: str) -> Path:
    outcome = next(o for o in run.outcomes if o.step_id == step)
    planned = plan_pipeline(ds, recipe, step_scopes=_scopes(recipe)).step(step)
    return ds.get_root("features") / planned.storage_name / str(outcome.run_id)


def _check_the_chain(ds: Dataset, recipe: Recipe) -> None:
    run = run_pipeline(ds, recipe, step_scopes=_scopes(recipe))

    assert {o.state for o in run.outcomes} == {"ran"}
    folds = pd.read_parquet(_root(ds, run, recipe, "clf") / "cv_folds.parquet")
    for step in ("smooth", "smooth_wide"):
        copied = pd.read_parquet(_root(ds, run, recipe, step) / "cv_folds.parquet")
        pd.testing.assert_frame_equal(copied, folds, obj=step)
    tune = _root(ds, run, recipe, "tune")
    report = json.loads((tune / "tune_report.json").read_text())
    assert report["n_entries"] == len(
        {tuple(row) for row in folds[["group", "sequence"]].to_numpy()}
    )
    blend = json.loads(
        (_root(ds, run, recipe, "blend") / "ensemble_report.json").read_text()
    )
    assert blend["weights"] == "fit"
    applied = _root(ds, run, recipe, "apply")
    assert sorted(path.name for path in applied.glob("*__*.parquet")) == sorted(
        f"{group}__{sequence}.parquet" for group, sequence in TRAIN_ENTRIES
    )


def test_a_smoothed_classifier_is_tuned_applied_and_blended(tmp_path: Path) -> None:
    _check_the_chain(_dataset(tmp_path / "ds"), _full_recipe())


def test_a_classifier_folded_by_block_is_smoothed_tuned_and_blended(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path / "ds")
    by_block = _full_recipe(
        {"cv_unit": "block", "cv_blocks_per_sequence": 4, "cv_gap_frames": 5}
    )

    _check_the_chain(ds, by_block)


# --- A mask upstream of the smoothing -------------------------------------------


def test_a_mask_on_the_templates_and_not_on_the_tuning_is_refused_by_class(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path / "ds")
    recipe = _recipe(*_shared_steps(MASKED, {}), _tune(LABELS))

    with pytest.raises(
        ValueError, match=r"\[tune-threshold\] labeled frames hold classes \[2\]"
    ):
        _ = run_pipeline(ds, recipe, step_scopes=_scopes(recipe))


def test_a_mask_on_the_templates_and_on_the_tuning_runs(tmp_path: Path) -> None:
    ds = _dataset(tmp_path / "ds")
    recipe = _recipe(*_shared_steps(MASKED, {}), _tune(MASKED))

    run = run_pipeline(ds, recipe, step_scopes=_scopes(recipe))

    assert {o.state for o in run.outcomes} == {"ran"}
    report = json.loads(
        (_root(ds, run, recipe, "tune") / "tune_report.json").read_text()
    )
    assert report["classes_tuned"] == [1]


def test_a_mask_upstream_of_an_average_of_smoothings_reaches_its_evaluation(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path / "ds")
    recipe = _recipe(
        *_shared_steps(MASKED, {}),
        {
            "id": "average",
            "type": "feature",
            "feature": "probability-ensemble",
            "inputs": [{"step": "smooth"}, {"step": "smooth_wide"}],
            "params": {"default_class": 0},
        },
        {
            "id": "evaluate",
            "type": "feature",
            "feature": "evaluate-labels",
            "inputs": [{"step": "average"}],
            "params": {"labels": LABELS, "perspective": 0},
        },
    )

    with pytest.raises(
        ValueError, match=r"\[evaluate-labels\] labeled frames hold classes \[2\]"
    ):
        _ = run_pipeline(ds, recipe, step_scopes=_scopes(recipe))
