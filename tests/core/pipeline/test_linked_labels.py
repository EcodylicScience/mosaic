"""A classifier on a copy reads its source's converted labels through the link.

The source converts its labels and runs a classifier over them. A copy that
links it reads both, plans the source's recipe complete, and computes nothing.
An entry the copy converts again takes the copy's own variant, and a classifier
over it then computes a run of its own into the copy's tree.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import Plan, Recipe, plan_pipeline, run_pipeline
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.labels_index import read_labels_index
from mosaic.core.pipeline.run import (
    resolve_feature_identity,
    resolve_labels_variants,
    run_feature,
)
from mosaic.core.scope import Scope
from tests.helpers import (
    CONVERT_LABELS_STEP,
    CONVERT_TRACKS_STEP,
    LABELS_CLASSIFIER_STEP,
    LabelsClassifier,
    add_labels,
    assert_tree_unchanged,
    calms21_dataset,
    copy_of,
    file_digests,
    labels_classifier_registered,
    tree_snapshot,
)

STORAGE = f"{LabelsClassifier.name}__from__tracks"
"""The storage directory of the classifier's runs."""

FILES = {"train": {"a": {"s1": 20, "s2": 20}}}
"""One CalMS21 file of two sequences, with tracks and labels for each."""

OWN_VARIANT = "convert-labels-calms21_npy.0.3-bbbbbbbbbb"
"""The variant the copy converts one entry into."""

type Entry = tuple[str, str]


def _recipe(labels_step: dict[str, JsonValue]) -> Recipe:
    """The tracks conversion, *labels_step* and the classifier over both."""
    return Recipe.model_validate(
        {"steps": [CONVERT_TRACKS_STEP, labels_step, LABELS_CLASSIFIER_STEP]}
    )


CHANGED_LABELS_STEP: dict[str, JsonValue] = {
    **CONVERT_LABELS_STEP,
    "params": {
        "source_format": "calms21_npy",
        "params": {"resident_id": 7, "intruder_id": 8},
    },
}
"""The labels conversion under other params, which names another variant."""


class Project(NamedTuple):
    """A source that ran its recipe, a copy that links it, and the source's plan."""

    source: Dataset
    copy: Dataset
    recipe: Recipe
    source_plan: Plan


@pytest.fixture(autouse=True)
def registered() -> Iterator[None]:
    """Put the stand-in classifier in the registry for the length of one test."""
    with labels_classifier_registered():
        yield


@pytest.fixture
def project(tmp_path: Path) -> Project:
    """The source after its recipe, and a copy that links it."""
    source = calms21_dataset(tmp_path / "source", FILES, seed=3, n_classes=2)
    recipe = _recipe(CONVERT_LABELS_STEP)
    _ = run_pipeline(source, recipe)
    LabelsClassifier.read.clear()
    return Project(
        source=source,
        copy=copy_of(source, tmp_path / "copy"),
        recipe=recipe,
        source_plan=plan_pipeline(source, recipe),
    )


def _entries(ds: Dataset) -> list[Entry]:
    """The entries of *ds*'s own ``behavior`` labels index, in index order."""
    rows = read_labels_index(ds, "behavior")
    return list(zip(rows["group"], rows["sequence"], strict=True))


def _identity(ds: Dataset) -> str:
    run_id, _ = resolve_feature_identity(ds, LabelsClassifier(), scope=Scope())
    return run_id


def test_a_copy_resolves_its_sources_labels_variant_and_classifier_run(
    project: Project,
) -> None:
    source, copy, _, plan = project
    variant = plan.step("labels").labels_variant

    assert resolve_labels_variants(copy, LabelsClassifier()) == (variant,)
    assert _identity(copy) == _identity(source) == plan.step("classifier").run_id


def test_the_sources_recipe_plans_every_step_complete_on_the_copy(
    project: Project,
) -> None:
    plan = plan_pipeline(project.copy, project.recipe)

    assert {step.step_id: step.status for step in plan.steps} == {
        "tracks": "complete",
        "labels": "complete",
        "classifier": "complete",
    }
    assert [step.run_id for step in plan.steps] == [
        step.run_id for step in project.source_plan.steps
    ]
    assert run_pipeline(project.copy, project.recipe).ran == ()


def test_a_classifier_reads_an_own_conversion_and_the_sources_for_the_rest(
    project: Project,
) -> None:
    source, copy, _, plan = project
    first, second = _entries(source)
    _ = add_labels(copy, *first, [(0, 0, 1, 1)], run_id=OWN_VARIANT)
    variant = str(plan.step("labels").labels_variant)
    before, digests = tree_snapshot(source.base_dir), file_digests(source.base_dir)

    result = run_feature(copy, LabelsClassifier())

    assert (
        LabelsClassifier.read[first].resolve().is_relative_to(copy.base_dir.resolve())
    )
    assert (
        LabelsClassifier.read[second]
        .resolve()
        .is_relative_to(source.base_dir.resolve())
    )
    assert resolve_labels_variants(copy, LabelsClassifier()) == tuple(
        sorted((OWN_VARIANT, variant))
    )
    assert result.run_id not in (None, plan.step("classifier").run_id)
    assert feature_run_root(copy, STORAGE, str(result.run_id)).exists()
    assert_tree_unchanged(source.base_dir, before)
    assert file_digests(source.base_dir) == digests


def test_a_changed_conversion_step_writes_its_variant_into_the_copy_and_pins_it(
    project: Project,
) -> None:
    source, copy, *_ = project
    recipe = _recipe(CHANGED_LABELS_STEP)
    before, digests = tree_snapshot(source.base_dir), file_digests(source.base_dir)
    variant = plan_pipeline(copy, recipe).step("labels").labels_variant

    run = run_pipeline(copy, recipe)

    assert {outcome.step_id for outcome in run.ran} == {"labels", "classifier"}
    own = read_labels_index(copy, "behavior")
    assert set(own["run_id"]) == {variant}
    assert sorted(zip(own["group"], own["sequence"], strict=True)) == sorted(
        _entries(source)
    )
    assert sorted(LabelsClassifier.read) == sorted(_entries(source))
    assert all(
        path.is_relative_to(copy.get_root("labels") / "behavior" / str(variant))
        for path in LabelsClassifier.read.values()
    )
    assert_tree_unchanged(source.base_dir, before)
    assert file_digests(source.base_dir) == digests
