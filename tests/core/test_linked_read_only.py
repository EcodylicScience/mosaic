"""A copy reads a linked dataset whose tree it cannot write.

A dataset never writes, renames, locks or deletes a path under a dataset it
links. A source on a read-only share therefore serves a copy's plans, runs,
tracks reads and inventory as a writable one does, and no byte of it moves.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import Recipe, plan_pipeline, run_pipeline
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.inventory import inventory
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.tracks_index import located_tracks_table, tracks_rows
from tests.helpers import (
    CONVERT_LABELS_STEP,
    CONVERT_TRACKS_STEP,
    LABELS_CLASSIFIER_STEP,
    SPEED_OF_CONVERTED_STEP,
    LabelsClassifier,
    calms21_dataset,
    copy_of,
    file_digests,
    labels_classifier_registered,
)

FILES = {"train": {"a": {"s1": 20, "s2": 20}}}
"""One CalMS21 file of two sequences, with tracks and labels for each."""


class ReadOnlyProject(NamedTuple):
    """A source made read-only after its recipe, and a copy that links it."""

    source: Dataset
    copy: Dataset
    recipe: Recipe


def _set_writable(base: Path, *, writable: bool) -> None:
    """Grant or take away the owner's write bit on every path under *base*."""
    for path in [base, *base.rglob("*")]:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            continue
        bits = mode | stat.S_IWUSR if writable else mode & ~stat.S_IWUSR
        path.chmod(stat.S_IMODE(bits))


@pytest.fixture
def project(tmp_path: Path) -> Iterator[ReadOnlyProject]:
    """The source after its recipe, read-only, and a copy that links it."""
    with labels_classifier_registered():
        source = calms21_dataset(tmp_path / "source", FILES, seed=3, n_classes=2)
        recipe = Recipe.model_validate(
            {
                "steps": [
                    CONVERT_TRACKS_STEP,
                    CONVERT_LABELS_STEP,
                    SPEED_OF_CONVERTED_STEP,
                    LABELS_CLASSIFIER_STEP,
                ]
            }
        )
        _ = run_pipeline(source, recipe)
        copy = copy_of(source, tmp_path / "copy")
        _set_writable(source.base_dir, writable=False)
        try:
            if os.access(source.base_dir, os.W_OK):
                pytest.skip("the tree stays writable for this user")
            yield ReadOnlyProject(source=source, copy=copy, recipe=recipe)
        finally:
            _set_writable(source.base_dir, writable=True)


def test_a_read_only_source_serves_a_copy_and_is_left_unchanged(
    project: ReadOnlyProject,
) -> None:
    source, copy, recipe = project
    digests = file_digests(source.base_dir)

    plan = plan_pipeline(copy, recipe)
    run = run_pipeline(copy, recipe)
    remapped = run_feature(
        copy, LabelsClassifier(params={"labels": {"kind": "behavior", "remap": {1: 0}}})
    )
    rows = tracks_rows(copy)
    tables = [
        located_tracks_table(copy, variant, (group, sequence))
        for variant, group, sequence in zip(
            rows["run_id"].astype(str),
            rows["group"].astype(str),
            rows["sequence"].astype(str),
            strict=True,
        )
    ]
    found = inventory(copy)

    assert plan.is_complete
    assert run.ran == ()
    assert remapped.run_id is not None
    assert feature_run_root(
        copy, f"{LabelsClassifier.name}__from__tracks", remapped.run_id
    ).exists()
    assert LabelsClassifier.read
    assert all(
        path.resolve().is_relative_to(source.base_dir.resolve())
        for path in LabelsClassifier.read.values()
    )
    assert set(rows["link_id"]) == {f"source-{source.uuid}"}
    assert all(table is not None and table.is_file() for table in tables)
    assert found.errors == ()
    assert file_digests(source.base_dir) == digests
