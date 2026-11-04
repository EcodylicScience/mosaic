"""``remap`` and ``mask`` on a ground-truth labels source.

A remap maps a class id of the label file to the id every reader sees, and a
mask makes the frames of the listed ids unlabeled. The source validates and
canonicalizes both, relabels an aligned array, and leaves a run's identity
where it was while both are unset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.classifier_common import probability_classes
from mosaic.cli._features import feature_class_for_slug
from mosaic.core.dataset import Dataset
from mosaic.core.evaluation import EvaluationReport
from mosaic.core.json_value import JsonValue
from mosaic.core.params import Params
from mosaic.core.pipeline.graph import PipelineRun, Recipe, plan_pipeline, run_pipeline
from mosaic.core.pipeline.loading import labels_for_rows
from mosaic.core.pipeline.types import GroundTruthLabelsSource, LabelsSource
from mosaic.core.scope import Scope
from tests.helpers import (
    CONVERT_LABELS_STEP,
    CONVERT_TRACKS_STEP,
    calms21_dataset,
    write_sparse_labels,
)

# --- relabel ---


def test_a_masked_id_becomes_unlabeled() -> None:
    source = GroundTruthLabelsSource(mask=[2])
    assert source.relabel(np.array([0, 2, 1, 2])).tolist() == [0, -1, 1, -1]


def test_a_masked_id_takes_the_sentinel_given() -> None:
    source = GroundTruthLabelsSource(mask=[2])
    sentinel = np.iinfo(np.int64).min
    relabeled = source.relabel(np.array([0, 2]), unlabeled=sentinel)
    assert relabeled.tolist() == [0, sentinel]


def test_a_remap_applies_every_pair_at_once() -> None:
    source = GroundTruthLabelsSource(remap={1: 2, 2: 1})
    assert source.relabel(np.array([0, 1, 2, 1])).tolist() == [0, 2, 1, 2]


def test_the_unlabeled_value_stays() -> None:
    source = GroundTruthLabelsSource(remap={0: 1}, mask=[2])
    assert source.relabel(np.array([-1, 0, 2])).tolist() == [-1, 1, -1]


def test_relabel_returns_a_new_int64_array() -> None:
    labels = np.array([0, 1, 2], dtype=np.int32)
    relabeled = GroundTruthLabelsSource(remap={1: 0}, mask=[2]).relabel(labels)
    assert labels.tolist() == [0, 1, 2]
    assert relabeled.dtype == np.int64
    assert relabeled.tolist() == [0, 0, -1]


def test_relabel_without_controls_copies() -> None:
    labels = np.array([0, 1], dtype=np.int64)
    relabeled = GroundTruthLabelsSource().relabel(labels)
    assert relabeled.tolist() == [0, 1]
    assert relabeled is not labels


def test_masked_flags_the_masked_ids() -> None:
    source = GroundTruthLabelsSource(mask=[1, 3])
    assert source.masked(np.array([0, 1, 2, 3])).tolist() == [
        False,
        True,
        False,
        True,
    ]
    assert not GroundTruthLabelsSource().masked(np.array([0, 1])).any()


# --- validation ---


@pytest.mark.parametrize(
    "fields",
    [{"remap": {-1: 0}}, {"remap": {0: -2}}, {"mask": [-1]}],
    ids=["remap-key", "remap-target", "mask"],
)
def test_a_negative_id_is_refused(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _ = GroundTruthLabelsSource.model_validate(fields)


def test_an_id_both_masked_and_remapped_is_refused() -> None:
    with pytest.raises(ValidationError, match="2"):
        _ = GroundTruthLabelsSource(remap={2: 0}, mask=[2])


def test_a_json_key_validates_as_an_id() -> None:
    source = GroundTruthLabelsSource.model_validate({"remap": {"2": 0}})
    assert source.remap == {2: 0}


def test_the_controls_are_canonical() -> None:
    source = GroundTruthLabelsSource(remap={3: 0, 1: 1, 2: 0}, mask=[5, 4, 5])
    assert source.remap is not None
    assert list(source.remap.items()) == [(2, 0), (3, 0)]
    assert source.mask == [4, 5]


@pytest.mark.parametrize(
    "fields",
    [{"remap": {}}, {"remap": {1: 1}}, {"mask": []}],
    ids=["empty-remap", "identity-pairs", "empty-mask"],
)
def test_controls_that_change_nothing_become_none(fields: dict[str, object]) -> None:
    source = GroundTruthLabelsSource.model_validate(fields)
    assert (source.remap, source.mask) == (None, None)


# --- the one aligner ---

_NAMES = ("attack", "investigation", "mount", "other")


def _rows(frames: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": np.arange(frames),
            "id1": np.zeros(frames, dtype=np.int64),
            "id2": np.ones(frames, dtype=np.int64),
        }
    )


def test_the_aligner_relabels_a_ground_truth_source(tmp_path: Path) -> None:
    path = write_sparse_labels(
        tmp_path / "labels.npz",
        [(0, 0, 1, 1), (1, 0, 1, 2)],
        names=_NAMES,
        background_id=3,
    )
    plain = labels_for_rows(path, _rows(3), GroundTruthLabelsSource())
    assert plain.tolist() == [1, 2, 3]
    controlled = GroundTruthLabelsSource(remap={3: 0}, mask=[2])
    assert labels_for_rows(path, _rows(3), controlled).tolist() == [1, -1, 0]


def test_the_aligner_leaves_another_labels_source_alone(tmp_path: Path) -> None:
    path = write_sparse_labels(
        tmp_path / "labels.npz", [(0, 0, 1, 2)], names=_NAMES, background_id=-1
    )
    rows = _rows(2)
    assert labels_for_rows(path, rows, LabelsSource(kind="behavior")).tolist() == [
        2,
        -1,
    ]


# --- identity ---

_TEMPLATES: dict[str, object] = {
    "feature": "extract-templates",
    "pattern": "templates.parquet",
    "load": {"kind": "parquet"},
}

_CONSUMERS: dict[str, dict[str, object]] = {
    "extract-labeled-templates": {"n_total": 10},
    "sequence-classifier": {"templates": _TEMPLATES, "default_class": 0},
    "lightning-action": {"templates": _TEMPLATES, "default_class": 0},
    "evaluate-labels": {},
    "tune-threshold": {"default_class": 0},
    "probability-ensemble": {"default_class": 0, "weights": "fit"},
    "fit-label-map": {},
    "label-statistics": {},
}
"""Every feature that reads ground-truth labels, with the params it needs."""


def _identity(slug: str, labels: dict[str, object]) -> dict[str, object]:
    """The identity of *slug*'s params, built without the feature's optional packages."""
    feature = feature_class_for_slug(slug)
    model: object = getattr(feature, "Params", None)
    assert isinstance(model, type) and issubclass(model, Params)
    given = {**_CONSUMERS[slug], "labels": {"kind": "behavior", **labels}}
    return model.from_overrides(given).identity_dump()


@pytest.mark.parametrize("slug", sorted(_CONSUMERS))
@pytest.mark.parametrize(
    "labels",
    [
        {},
        {"remap": None, "mask": None},
        {"remap": {}, "mask": []},
        {"remap": {1: 1, "2": 2}},
    ],
    ids=["omitted", "none", "empty", "identity-pairs"],
)
def test_controls_that_change_nothing_leave_the_identity(
    slug: str, labels: dict[str, object]
) -> None:
    assert _identity(slug, labels) == _identity(slug, {})


@pytest.mark.parametrize("slug", sorted(_CONSUMERS))
@pytest.mark.parametrize(
    "labels", [{"remap": {2: 0}}, {"mask": [3]}], ids=["remap", "mask"]
)
def test_either_control_changes_the_identity(
    slug: str, labels: dict[str, object]
) -> None:
    assert _identity(slug, labels) != _identity(slug, {})


@pytest.mark.parametrize("slug", sorted(_CONSUMERS))
def test_spellings_of_one_control_give_one_identity(slug: str) -> None:
    assert _identity(slug, {"remap": {"2": 0}}) == _identity(slug, {"remap": {2: 0}})
    assert _identity(slug, {"mask": [3, 1, 3]}) == _identity(slug, {"mask": [1, 3]})


def test_params_json_keeps_both_fields() -> None:
    source = GroundTruthLabelsSource()
    assert {"remap", "mask"} <= set(source.model_dump())


# --- a chain whose steps control the labels ---

_TRAIN_ENTRIES = [("train", f"s{i}") for i in range(1, 5)]
_TRAIN = Scope(groups=["train"])
_SHARED_STEPS: list[dict[str, JsonValue]] = [
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
]


def _chain_dataset(base: Path) -> Dataset:
    """Three classes, with the labels converted before any step reads them."""
    ds = calms21_dataset(
        base,
        {"train": {"a": {name: 240 for _, name in _TRAIN_ENTRIES}}},
        seed=11,
        n_classes=3,
    )
    prepared = Recipe.model_validate(
        {"schema_version": 1, "name": "prepared", "steps": _SHARED_STEPS}
    )
    _ = run_pipeline(ds, prepared)
    return ds


def _chain(
    templates: dict[str, JsonValue],
    evaluate: dict[str, JsonValue],
    **template_params: JsonValue,
) -> Recipe:
    """Templates, ``xgboost`` and ``evaluate-labels``, each labels source given whole.

    The sources are literal rather than step references, so that each carries
    its own controls.
    """
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "controlled",
            "steps": [
                *_SHARED_STEPS,
                {
                    "id": "labeled",
                    "type": "feature",
                    "feature": "extract-labeled-templates",
                    "inputs": [{"step": "pairs"}],
                    "params": {
                        "labels": {"kind": "behavior", **templates},
                        "test_fraction": 0.0,
                        **(template_params or {"n_total": 300}),
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
                    },
                },
                {
                    "id": "evaluate",
                    "type": "feature",
                    "feature": "evaluate-labels",
                    "inputs": [{"step": "classifier"}],
                    "params": {
                        "labels": {"kind": "behavior", **evaluate},
                        "perspective": 0,
                    },
                },
            ],
        }
    )


def _steps_scoped() -> dict[str, Scope]:
    return {"labeled": _TRAIN, "classifier": _TRAIN, "evaluate": _TRAIN}


def _report(ds: Dataset, recipe: Recipe) -> EvaluationReport:
    run = run_pipeline(ds, recipe, step_scopes=_steps_scoped())
    root = _step_root(ds, recipe, run, "evaluate")
    return EvaluationReport.model_validate_json((root / "metrics.json").read_text())


def _step_root(ds: Dataset, recipe: Recipe, run: PipelineRun, step: str) -> Path:
    outcome = next(o for o in run.outcomes if o.step_id == step)
    planned = plan_pipeline(ds, recipe, step_scopes=_steps_scoped()).step(step)
    return ds.get_root("features") / planned.storage_name / str(outcome.run_id)


def test_a_mask_on_the_templates_alone_is_refused_two_steps_down(
    tmp_path: Path,
) -> None:
    ds = _chain_dataset(tmp_path / "ds")

    with pytest.raises(Exception, match=r"classes \[2\]"):
        _ = run_pipeline(ds, _chain({"mask": [2]}, {}), step_scopes=_steps_scoped())


def test_the_same_mask_on_every_labels_source_scores(tmp_path: Path) -> None:
    ds = _chain_dataset(tmp_path / "ds")

    report = _report(ds, _chain({"mask": [2]}, {"mask": [2]}))

    assert 2 not in report.classes
    assert report.n_scored > 0


def test_a_remap_on_the_templates_leaves_the_classifier_without_its_source_class(
    tmp_path: Path,
) -> None:
    ds = _chain_dataset(tmp_path / "ds")
    recipe = _chain({"remap": {"2": 0}}, {"remap": {"2": 0}})

    run = run_pipeline(ds, recipe, step_scopes=_steps_scoped())

    root = _step_root(ds, recipe, run, "classifier")
    columns = pd.read_parquet(next(root.glob("*.parquet"))).columns
    assert probability_classes(columns) == [0, 1]
    report = EvaluationReport.model_validate_json(
        (_step_root(ds, recipe, run, "evaluate") / "metrics.json").read_text()
    )
    assert 2 not in report.classes


def test_a_class_the_templates_leave_out_without_controls_scores_as_misses(
    tmp_path: Path,
) -> None:
    ds = _chain_dataset(tmp_path / "ds")

    report = _report(ds, _chain({}, {}, n_per_class={"0": 50, "2": 50}))

    assert 1 in report.classes
    one = next(score for score in report.per_class if score.label == 1)
    assert one.support > 0
    assert one.predicted == 0
