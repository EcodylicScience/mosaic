"""A project copy reads what its source computed and writes only into its own tree.

The copy is a dataset with a fresh ``uuid``, a link to the source and raw index rows
that name the source's raw files. It plans the source's recipe with every step complete,
reads the source's feature outputs, tracks tables, label series, converted labels and
frame runs through their lookups, computes a run of its own that wins over the source's
at the same address, and removes, prunes and reconciles without changing the source.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pytest

from mosaic.behavior.feature_library import SpeedAngvel
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.feature_run_removal import remove_feature_run
from mosaic.core.pipeline.graph import Plan, Recipe, plan_pipeline, run_pipeline
from mosaic.core.pipeline.index import feature_rows, feature_run_root
from mosaic.core.pipeline.inventory.params import read_run_params
from mosaic.core.pipeline.inventory.scan import run_coverage
from mosaic.core.pipeline.label_series import TAGS_SERIES
from mosaic.core.pipeline.label_series_index import label_series_rows
from mosaic.core.pipeline.labels_index import labels_rows, read_labels_index
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.tracks_index import read_tracks_index, tracks_rows
from mosaic.runlog import JsonlRunLog, new_execution_id, run_log_path
from mosaic.tracking.frame_extraction.frame_runs import frame_rows
from tests.helpers import (
    add_labels,
    add_raw_track_files,
    add_tracks_table,
    assert_tree_unchanged,
    copy_of,
    extract_entry_camera,
    file_digests,
    make_dataset,
    tag_table,
    track_table,
    tree_snapshot,
    write_media_index,
    write_tags,
)

type Document = dict[str, object]

SEQUENCES = ("s1", "s2")
"""The sequences of the source, each with a video and a raw TREx export."""

FRAMES_RUN = "0.1-frames0000"
"""The frames run the source holds."""

TARGET = frozenset(("", sequence) for sequence in SEQUENCES)
"""The entries of the source."""

COPY_ROOTS = ("tracks", "features", "frames", "media")
"""The roots of the copy that a step writes into."""


class Project(NamedTuple):
    """A source that ran its recipe, a copy that links it, and the recipe's plan."""

    source: Dataset
    copy: Dataset
    recipe: Recipe
    source_plan: Plan


def _recipe() -> Recipe:
    """A conversion of the TREx exports and a speed feature on its tracks."""
    steps: list[Document] = [
        {
            "id": "convert",
            "type": "op",
            "kind": "convert-tracks",
            "params": {"source_format": "trex_npz"},
        },
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "inputs": ["tracks"],
            "tracks": {"step": "convert"},
        },
    ]
    return Recipe.model_validate({"steps": steps})


@pytest.fixture
def project(tmp_path: Path) -> Project:
    """The source of :data:`SEQUENCES` after its recipe, and a copy that links it."""
    source = make_dataset(tmp_path / "source", name="source")
    write_media_index(
        source,
        list(SEQUENCES),
        uids={sequence: f"uid-{sequence}" for sequence in SEQUENCES},
    )
    add_raw_track_files(source, trex=list(SEQUENCES))
    recipe = _recipe()
    _ = run_pipeline(source, recipe)
    _ = write_tags(source, tag_table({("", "s1"): {"site": "a"}}, {"site": "text"}))
    for sequence in SEQUENCES:
        _ = add_labels(source, "", sequence, [(0, 0, 1, 0)])
    _ = extract_entry_camera(
        source,
        run_id=FRAMES_RUN,
        group="",
        sequence="s1",
        camera="",
        sizes={0: (4, 4)},
    )
    return Project(
        source=source,
        copy=copy_of(source, tmp_path / "copy"),
        recipe=recipe,
        source_plan=plan_pipeline(source, recipe),
    )


def _roots(copy: Dataset) -> dict[str, dict[str, object] | None]:
    """The snapshot of each of :data:`COPY_ROOTS` that exists in *copy*."""
    return {
        name: dict(tree_snapshot(root)) if root.exists() else None
        for name, root in ((name, copy.get_root(name)) for name in COPY_ROOTS)
    }


def test_a_copy_has_a_uuid_of_its_own_a_link_and_the_raw_rows_of_its_source(
    project: Project,
) -> None:
    source, copy, *_ = project

    assert copy.uuid and copy.uuid != source.uuid
    (link,) = copy.manifest.links
    assert link.id == f"source-{source.uuid}"
    assert copy.resolve_link_path(link).resolve() == source.base_dir.resolve()
    for root in ("media_raw", "tracks_raw"):
        copied = pd.read_csv(copy.get_root(root) / "index.csv", dtype=str)
        assert sorted(copied["sequence"]) == list(SEQUENCES), root
    assert read_tracks_index(copy).empty


def test_the_sources_recipe_plans_every_step_complete_on_the_copy(
    project: Project,
) -> None:
    plan = plan_pipeline(project.copy, project.recipe)

    assert {step.step_id: step.status for step in plan.steps} == {
        "convert": "complete",
        "speed": "complete",
    }
    assert plan.is_complete
    assert [step.run_id for step in plan.steps] == [
        step.run_id for step in project.source_plan.steps
    ]


def test_running_the_sources_recipe_on_the_copy_writes_no_file_into_its_roots_or_the_sources(
    project: Project,
) -> None:
    source = project.source.base_dir
    before = _roots(project.copy)
    source_tree, source_digests = tree_snapshot(source), file_digests(source)

    run = run_pipeline(project.copy, project.recipe)

    assert run.ran == ()
    assert _roots(project.copy) == before
    assert_tree_unchanged(source, source_tree)
    assert file_digests(source) == source_digests


def test_the_copy_reads_the_sources_data_through_each_lookup(project: Project) -> None:
    source, copy, *_ = project
    link = f"source-{source.uuid}"
    speed = project.source_plan.step("speed")

    features = feature_rows(copy, speed.storage_name, run_id=str(speed.run_id))
    tracks = tracks_rows(copy)
    series = label_series_rows(copy, TAGS_SERIES)
    frames = frame_rows(copy, "uniform")
    labels = labels_rows(copy, "behavior")

    assert (set(features["sequence"]), set(features["link_id"])) == (
        set(SEQUENCES),
        {link},
    )
    assert (set(tracks["sequence"]), set(tracks["link_id"])) == (set(SEQUENCES), {link})
    assert list(series["link_id"]) == [link]
    assert list(frames["run_id"]) == [FRAMES_RUN]
    assert set(frames["link_id"]) == {link}
    assert (set(labels["sequence"]), set(labels["link_id"])) == (set(SEQUENCES), {link})
    assert read_labels_index(copy, "behavior").empty
    assert run_coverage(
        copy, speed.storage_name, str(speed.run_id), TARGET
    ).is_satisfied


def _compute_one_entry_into_the_copy(project: Project) -> str:
    """Replace the copy's tracks of ``s1`` and run the speed feature there.

    Returns:
        The speed run's id.
    """
    speed = project.source_plan.step("speed")
    variant = str(project.source_plan.step("convert").tracks_variant)
    add_tracks_table(
        project.copy,
        variant,
        track_table("", "s1", n_rows=40),
        consumed_composition="changed",
        std_format="mosaic_v1",
    )
    result = run_feature(project.copy, SpeedAngvel())
    assert result.run_id == speed.run_id
    return str(result.run_id)


def test_a_run_of_the_copy_wins_over_the_sources_run_at_the_same_address(
    project: Project,
) -> None:
    source, copy, *_ = project
    speed = project.source_plan.step("speed")

    run_id = _compute_one_entry_into_the_copy(project)

    rows = feature_rows(copy, speed.storage_name, run_id=run_id)
    assert dict(zip(rows["sequence"], rows["link_id"], strict=True)) == {
        "s1": "",
        "s2": f"source-{source.uuid}",
    }
    own = feature_run_root(copy, speed.storage_name, run_id)
    assert [path.name for path in own.glob("*.parquet")] != []
    assert read_run_params(own).params is not None
    assert run_coverage(copy, speed.storage_name, run_id, TARGET).is_satisfied


def test_removal_prune_and_reconcile_on_the_copy_leave_the_source_unchanged(
    project: Project,
) -> None:
    source, copy, *_ = project
    speed = project.source_plan.step("speed")
    run_id = _compute_one_entry_into_the_copy(project)
    snapshot = tree_snapshot(source.base_dir)
    digests = file_digests(source.base_dir)
    params = read_run_params(feature_run_root(copy, speed.storage_name, run_id)).params
    assert params is not None
    execution_id = new_execution_id()

    with JsonlRunLog(run_log_path(copy.base_dir, execution_id), execution_id) as log:
        removed = remove_feature_run(
            copy,
            speed.storage_name,
            run_id,
            params=params,
            log=log,
            reason="superseded",
            superseded_by="",
        )
    pruned = copy.prune_media(apply=True, min_age_hours=0.0, relink=True)
    _ = copy.reconcile(apply=True)

    assert removed.files > 0
    assert pruned.considered
    assert_tree_unchanged(source.base_dir, snapshot)
    assert file_digests(source.base_dir) == digests
    rows = feature_rows(copy, speed.storage_name, run_id=run_id)
    assert set(rows["link_id"]) == {f"source-{source.uuid}"}, (
        "the source's run serves again once the copy's own run is gone"
    )
