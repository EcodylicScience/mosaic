"""A graph that converts its own tracks and labels before computing on them.

The conversion names its variant before it runs, so a feature reading the
converted tracks or labels through a reference is named exactly at plan time.
What cannot be named before the conversion is a step whose identity covers its
entries, on a dataset with no tracks yet, and such a step is left unresolved
with the conversion as the reason rather than named over no entries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.manifest import LabelsScanSource
from mosaic.core.pipeline.graph import (
    IdentityUnresolved,
    Recipe,
    build_step_feature,
    plan_pipeline,
    run_pipeline,
)
from mosaic.core.pipeline.labels_index import read_labels_index
from tests.helpers import (
    CONVERT_LABELS_STEP,
    CONVERT_TRACKS_STEP,
    SPEED_OF_CONVERTED_STEP,
    calms21_dataset,
    converting_recipe,
    write_calms21_npy,
)

FILES = {"train": {"a": {"s1": 40, "s2": 30}}, "test": {"a": {"s3": 30}}}
"""Two CalMS21 files, filed as a train group of two sequences and a test group."""

TEMPLATES_STEP: dict[str, JsonValue] = {
    "id": "templates",
    "type": "feature",
    "feature": "extract-templates",
    "inputs": [{"step": "speed"}],
    "params": {"n_templates": 4},
}

LABELED_STEP: dict[str, JsonValue] = {
    "id": "labeled",
    "type": "feature",
    "feature": "extract-labeled-templates",
    "inputs": [{"step": "speed"}],
    "params": {"labels": {"step": "labels"}, "n_total": 12},
}


def _dataset(tmp_path: Path) -> Dataset:
    return calms21_dataset(tmp_path / "ds", FILES, seed=3, n_classes=2)


def test_a_graph_converting_its_own_tracks_without_media_defers_scope_dependent_steps(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path)
    recipe = converting_recipe(
        SPEED_OF_CONVERTED_STEP,
        TEMPLATES_STEP,
        {
            "id": "kmeans",
            "type": "feature",
            "feature": "global-kmeans",
            "inputs": [{"step": "speed"}],
            "params": {"templates": {"step": "templates"}, "k": 2},
        },
    )

    plan = plan_pipeline(ds, recipe)

    assert plan.scope == frozenset()
    assert plan.step("tracks").run_id == ds.track_conversion_variant("calms21_npy")
    speed = plan.step("speed")
    assert not build_step_feature(speed.spec).scope_dependent
    assert speed.run_id is not None
    templates = plan.step("templates")
    assert build_step_feature(templates.spec).scope_dependent
    assert templates.run_id is None
    assert isinstance(templates.reason, IdentityUnresolved)
    assert templates.reason.step == "tracks"
    kmeans = plan.step("kmeans")
    assert kmeans.run_id is None
    assert isinstance(kmeans.reason, IdentityUnresolved)

    run = run_pipeline(ds, recipe)

    after = plan_pipeline(ds, recipe)
    assert after.is_complete
    assert after.step("speed").run_id == speed.run_id
    executed = {outcome.step_id: outcome.run_id for outcome in run.outcomes}
    assert executed["templates"] == after.step("templates").run_id


def test_a_scoped_op_below_a_deferring_converter_is_deferred_not_refused(
    tmp_path: Path,
) -> None:
    """A transcode over an unset scope is refused, so it must not be asked yet."""
    ds = _dataset(tmp_path)
    recipe = converting_recipe(
        {"id": "transcode", "type": "op", "kind": "transcode", "after": ["tracks"]}
    )

    plan = plan_pipeline(ds, recipe)

    transcode = plan.step("transcode")
    assert transcode.run_id is None
    assert isinstance(transcode.reason, IdentityUnresolved)
    assert transcode.reason.step == "tracks"


def test_an_unscoped_converter_does_not_fall_back_to_the_media_universe(
    three_entry_dataset: Dataset,
) -> None:
    """A converter writes the entries its files hold, not those there is media for."""
    from mosaic.core.manifest import TracksScanSource

    ds = three_entry_dataset
    raw = ds.base_dir / "raw"
    write_calms21_npy(raw / "train.npy", {"a": {"s1": 10}})
    ds.add_scan_source(
        TracksScanSource(
            id="calms21",
            path=str(raw),
            patterns=("*.npy",),
            src_format="calms21_npy",
            multi_sequences_per_file=True,
            group_from="filename",
        )
    )
    ds.scan_tracks()

    plan = plan_pipeline(ds, converting_recipe(SPEED_OF_CONVERTED_STEP, TEMPLATES_STEP))

    assert plan.scope == frozenset()
    assert plan.step("templates").run_id is None


def test_a_labels_reference_pins_the_minted_labels_variant(tmp_path: Path) -> None:
    """Planned before the labels exist, the run id is the one execution writes."""
    ds = _dataset(tmp_path)
    _ = ds.convert_all_tracks()
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "labels",
            "steps": [
                CONVERT_LABELS_STEP,
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                },
                LABELED_STEP,
            ],
        }
    )

    plan = plan_pipeline(ds, recipe)

    variant = ds.label_conversion_variant("behavior", "calms21_npy")
    assert plan.step("labels").labels_variant == variant
    labeled = plan.step("labeled")
    assert labeled.spec.labels_run_id == variant
    assert labeled.run_id is not None
    assert read_labels_index(ds, "behavior").empty

    run = run_pipeline(ds, recipe)

    executed = {outcome.step_id: outcome.run_id for outcome in run.outcomes}
    assert executed["labeled"] == labeled.run_id


def test_the_labels_step_of_a_partly_labelled_dataset_reads_complete(
    tmp_path: Path,
) -> None:
    """Labels answer for the entries they annotate, not for the plan's target."""
    ds = calms21_dataset(tmp_path / "ds", FILES, seed=3, n_classes=2, scan=False)
    labels = ds.manifest.sources.labels[0]
    ds.remove_scan_source("labels", labels.id)
    ds.add_scan_source(
        LabelsScanSource(
            id=labels.id,
            path=labels.path,
            patterns=("train.npy",),
            src_format="calms21_npy",
            multi_sequences_per_file=True,
            group_from="filename",
        )
    )
    ds.scan_tracks()
    ds.scan_labels()
    _ = ds.convert_all_tracks()
    _ = ds.convert_all_labels(kind="behavior", source_format="calms21_npy")
    recipe = Recipe.model_validate(
        {"schema_version": 1, "name": "labels", "steps": [CONVERT_LABELS_STEP]}
    )

    plan = plan_pipeline(ds, recipe)

    assert len(plan.scope) == 3
    assert plan.step("labels").status == "complete"


@pytest.mark.parametrize("labels", [False, True])
def test_a_converting_plan_names_its_conversions_by_their_variants(
    tmp_path: Path, labels: bool
) -> None:
    ds = _dataset(tmp_path)

    plan = plan_pipeline(ds, converting_recipe(labels=labels))

    assert plan.step(str(CONVERT_TRACKS_STEP["id"])).tracks_variant == (
        ds.track_conversion_variant("calms21_npy")
    )
    if labels:
        assert plan.step("labels").labels_kind == "behavior"


def test_a_step_runs_with_the_labels_variant_its_reference_names(
    tmp_path: Path,
) -> None:
    """Two labels variants of one kind, and the step reads the one it names.

    Unpinned, two recipes for one entry refuse. The reference pins the variant
    its conversion writes, and the run records that variant as what it read.
    """
    import json

    ds = _dataset(tmp_path)
    _ = ds.convert_all_tracks()
    swapped = ds.convert_all_labels(
        kind="behavior",
        source_format="calms21_npy",
        params={"resident_id": 1, "intruder_id": 0},
    )
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "labels",
            "steps": [
                CONVERT_LABELS_STEP,
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                },
                LABELED_STEP,
            ],
        }
    )
    variant = ds.label_conversion_variant("behavior", "calms21_npy")
    assert variant != swapped.variant

    run = run_pipeline(ds, recipe)

    labeled = next(outcome for outcome in run.outcomes if outcome.step_id == "labeled")
    storage = plan_pipeline(ds, recipe).step("labeled").storage_name
    params_path = ds.get_root("features") / storage / labeled.run_id / "params.json"
    resolved = json.loads(params_path.read_text())["_resolved"]
    read = [item["run_id"] for item in resolved if item["feature"] == "labels"]
    assert read == [variant]
