"""Sizing a step of a request that names a feature definition."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.definitions.define import define_feature
from mosaic.core.pipeline.definitions.scope import current_scope, definition_scope
from mosaic.core.pipeline.definitions.store import FeatureDefinitionsNotAllowed
from mosaic.core.pipeline.graph import Recipe, plan_pipeline, submit_request
from mosaic.core.pipeline.graph.input_size import step_input_size
from mosaic.core.pipeline.input_size import manifest_input_sizes
from mosaic.core.pipeline.manifest import build_manifest
from mosaic.core.pipeline.types import Inputs
from mosaic.core.scope import Scope
from tests.helpers import (
    MINIMAL_NAME,
    MINIMAL_SOURCE,
    add_track_sequences,
    defined_dataset,
    make_dataset,
)

SLUG: Final = f"def:{MINIMAL_NAME}"
SECOND_SOURCE: Final = MINIMAL_SOURCE.replace(
    "scale: float = 1.0", "scale: float = 2.0"
).replace('version = "1"', 'version = "2"')
"""A second version of ``MINIMAL_SOURCE``, of another default and version."""
ENTRIES: Final = [("g", "s1"), ("g", "s2")]


def _recipe() -> Recipe:
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [
                {
                    "id": "d",
                    "type": "feature",
                    "feature": SLUG,
                    "inputs": ["tracks"],
                }
            ],
        }
    )


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return defined_dataset(tmp_path, {MINIMAL_NAME: MINIMAL_SOURCE})


def _rows(dataset: Dataset) -> int:
    """The footer rows of the tracks of both entries."""
    manifest, _ = build_manifest(dataset, Inputs(("tracks",)), Scope(entries=ENTRIES))
    return sum(size.rows for size in manifest_input_sizes(manifest).values())


def test_a_step_is_sized_against_the_version_its_request_pinned(
    dataset: Dataset,
) -> None:
    submitted = submit_request(dataset, _recipe())
    _ = define_feature(dataset, MINIMAL_NAME, SECOND_SOURCE)
    assert current_scope() is None
    moved = plan_pipeline(dataset, _recipe()).step("d").run_id
    assert moved != submitted.plan.step("d").run_id

    report = step_input_size(dataset, submitted.request, "d")

    assert report.inputs_ready, report.reason
    assert report.run_id == submitted.plan.step("d").run_id
    assert report.entries == 2
    assert [row.token for row in report.entries_to_compute_by_rows] == [
        "g:s1",
        "g:s2",
    ]
    assert report.rows == report.rows_to_compute == _rows(dataset)
    assert current_scope() is None


@pytest.fixture(params=["no-dataset", "same-dataset", "another-dataset"])
def disabled(
    request: pytest.FixtureRequest, dataset: Dataset, tmp_path: Path
) -> Dataset | None:
    """The dataset a disabled scope is entered for."""
    other = make_dataset(tmp_path / "other")
    add_track_sequences(other, ("g", "s1"))
    return {"no-dataset": None, "same-dataset": dataset, "another-dataset": other}[
        request.param
    ]


def test_sizing_inside_a_disabled_scope_keeps_it_disabled(
    dataset: Dataset, disabled: Dataset | None
) -> None:
    submitted = submit_request(dataset, _recipe())

    with definition_scope(disabled, load=False, allowed=False) as outer:
        with pytest.raises(FeatureDefinitionsNotAllowed) as sized:
            _ = step_input_size(dataset, submitted.request, "d")
        assert current_scope() is outer

    assert sized.value.error_reason() == "feature_definitions_not_allowed"
    assert sized.value.slug == SLUG
