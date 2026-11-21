"""No removal, prune, reconcile or backfill on a copy changes the dataset it links.

The source holds feature runs, a fitted run, tracks, a frames run and transcoded
media, with a derivative no link cell names. Each test acts on a copy that links
it and holds data of its own, then compares the source's tree and file digests
with what they were before.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline import FeatureStep, Pipeline
from mosaic.core.pipeline.graph import (
    plan_fingerprint,
    plan_with_cleanup,
    run_cleanup,
    execute_step,
    submit_request,
)
from mosaic.core.pipeline.graph.cleanup import CleanupPolicy
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.job import CancelToken
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.tracks_index import (
    backfill_frame_extents,
    backfill_pair_frames,
)
from mosaic.behavior.feature_library import SpeedAngvel
from tests.helpers import (
    SOURCE_ENTRIES,
    SUPERSEDED_RECIPE,
    CenteringFeature,
    add_tracks_table,
    assert_tree_unchanged,
    copy_of,
    extract_entry_camera,
    file_digests,
    raw_source,
    run_speed,
    speed_recipe_stepping,
    track_table,
    tree_snapshot,
)

SOURCE_VARIANT = "convert-trex.0.2-1111111111"
"""The tracks variant the source holds for both of its entries."""

OWN_VARIANT = "convert-trex.0.2-2222222222"
"""The tracks variant the copy holds for an entry of its own."""

STORAGE = "speed-angvel__from__tracks"

LOOSE = CleanupPolicy(keep_newest=0, min_age_hours=0)
"""A policy that keeps no run for its age or its recency."""

type SourceUnchanged = Callable[[], None]


@pytest.fixture
def source(tmp_path: Path) -> Dataset:
    """A dataset with media, tracks, speed runs at 1 and 4, a fitted run and frames."""
    dataset = raw_source(tmp_path / "source")
    for group, sequence in SOURCE_ENTRIES:
        add_tracks_table(
            dataset,
            SOURCE_VARIANT,
            track_table(group, sequence, n_rows=40, n_ids=2),
            std_format="mosaic_v1",
        )
    _ = run_speed(dataset, 1)
    _ = run_speed(dataset, 4)
    _ = run_feature(dataset, CenteringFeature())
    _ = extract_entry_camera(
        dataset,
        run_id="0.1-frames0000",
        group="g",
        sequence="s1",
        camera="",
        sizes={0: (4, 4)},
    )
    return dataset


@pytest.fixture
def source_unchanged(source: Dataset) -> SourceUnchanged:
    """Assert that the source's tree and every file's digest are what they were.

    The source is recorded when the fixture is built, before the copy adopts its raw
    rows and runs its own feature, so a write from either one fails the check.
    """
    snapshot = tree_snapshot(source.base_dir)
    digests = file_digests(source.base_dir)

    def check() -> None:
        assert_tree_unchanged(source.base_dir, snapshot)
        assert file_digests(source.base_dir) == digests

    return check


@pytest.fixture
def copy(source: Dataset, source_unchanged: SourceUnchanged, tmp_path: Path) -> Dataset:
    """A copy of *source* that holds a tracks table and a speed run of its own.

    Depends on ``source_unchanged`` so the source is recorded before the copy exists.
    """
    _ = source_unchanged
    own = copy_of(source, tmp_path / "copy")
    add_tracks_table(
        own,
        OWN_VARIANT,
        track_table("g", "s3", n_rows=40, n_ids=2),
        std_format="mosaic_v1",
    )
    _ = run_speed(own, 3)
    return own


def _speed_runs(dataset: Dataset) -> set[str]:
    """The run directories of the speed storage under *dataset*'s own root."""
    root = dataset.get_root("features") / STORAGE
    return {path.name for path in root.iterdir() if path.is_dir()}


def test_a_cleanup_on_a_copy_removes_its_own_run_and_leaves_the_source_alone(
    source: Dataset, copy: Dataset, source_unchanged: SourceUnchanged
) -> None:
    (own,) = _speed_runs(copy)
    held = _speed_runs(source)
    recipe = speed_recipe_stepping(2)
    plan = plan_with_cleanup(copy, recipe, cleanup=LOOSE)
    assert plan.cleanup is not None
    assert [run.run_id for run in plan.cleanup.runs] == [own], (
        "the runs only the source holds are no candidates"
    )
    request = submit_request(
        copy, recipe, cleanup=LOOSE, expect_plan=plan_fingerprint(plan)
    ).request
    for step_id in request.step_executions:
        _ = execute_step(copy, request, step_id)

    outcome = run_cleanup(copy, request, token=CancelToken())

    assert [(run.storage, run.run_id) for run in outcome.removed] == [(STORAGE, own)]
    assert not feature_run_root(copy, STORAGE, own).exists()
    assert _speed_runs(source) == held
    source_unchanged()


def test_a_pipeline_clean_on_a_copy_neither_removes_nor_counts_the_sources_runs(
    source: Dataset, copy: Dataset, source_unchanged: SourceUnchanged
) -> None:
    (own,) = _speed_runs(copy)
    held = _speed_runs(source)
    pipeline = Pipeline()
    pipeline.add(FeatureStep("speed", SpeedAngvel, {"step_size": 1}))

    summary = pipeline.clean(copy, dry_run=False)

    assert dict(zip(summary["run_id"], summary["status"], strict=True)) == {
        own: "removed"
    }
    assert not feature_run_root(copy, STORAGE, own).exists()
    assert _speed_runs(source) == held, "the source's two runs are all still there"
    source_unchanged()


def test_a_prune_on_a_copy_leaves_the_sources_derivatives(
    source: Dataset, copy: Dataset, source_unchanged: SourceUnchanged
) -> None:
    superseded = list(
        (source.get_root("media") / "transcode").glob(f"*.{SUPERSEDED_RECIPE}.*")
    )
    assert superseded, "the source holds a derivative no link cell names"

    report = copy.prune_media(apply=True, min_age_hours=0.0, relink=True)

    assert report.considered
    assert all(path.exists() for path in superseded)
    source_unchanged()


def test_a_reconcile_on_a_copy_leaves_the_source_alone(
    copy: Dataset, source_unchanged: SourceUnchanged
) -> None:
    _ = copy.reconcile(apply=True)

    source_unchanged()


@pytest.mark.parametrize(
    "backfill", [backfill_frame_extents, backfill_pair_frames], ids=lambda f: f.__name__
)
def test_a_tracks_backfill_on_a_copy_leaves_the_source_alone(
    copy: Dataset,
    source_unchanged: SourceUnchanged,
    backfill: Callable[[Dataset], pd.DataFrame],
) -> None:
    _ = backfill(copy)

    source_unchanged()
