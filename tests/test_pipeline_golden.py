"""The worked recipe and its plan, pinned so a rename is a visible diff.

The identity tests beside this one assert that a plan agrees with execution,
which is the property that matters and says nothing about the *shape* of the
answer. This corpus is the other half: every field a plan hands to whatever runs
it, written down, so a renamed field or a re-derived identifier shows up as a
line in a review rather than as a downstream surprise.

Same conventions as ``op_identity_golden.json`` -- one flat map of case id to
value, regenerated under ``MOSAIC_UPDATE_GOLDEN=1``, with a test that refuses a
stale entry. A moved digest during a refactor should always mean a mistake.

The dataset underneath is fixed on purpose: the tracks variant is a constant, the
entries are two named sequences, and the media rows carry pinned ``video_uuid``
cells -- so the only thing that can move a value here is a change to how mosaic
names things.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import Plan, Recipe, plan_pipeline, recipe_digest
from tests.helpers import add_tracks_variant, make_dataset, write_media_index

GOLDEN_PATH = Path(__file__).parent / "data" / "pipeline_plan_golden.json"
UPDATE_ENV = "MOSAIC_UPDATE_GOLDEN"

VARIANT = "convert-trex.0.2-1111111111"
"""The tracks recipe the fixture's tables answer to."""

WORKED_RECIPE: dict[str, object] = {
    "schema_version": 1,
    "name": "trex to global tsne",
    "steps": [
        {
            "id": "transcode",
            "type": "op",
            "kind": "transcode",
            "params": {"target": "analysis"},
        },
        {
            "id": "trex",
            "type": "op",
            "kind": "trex",
            "params": {"track_max_individuals": 4, "cm_per_pixel": 0.05},
            "after": ["transcode"],
        },
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "inputs": ["tracks"],
            "tracks": {"step": "trex"},
        },
        {
            "id": "templates",
            "type": "feature",
            "feature": "extract-templates",
            "inputs": [{"step": "speed"}],
            "params": {"n_templates": 8},
        },
        {
            "id": "tsne",
            "type": "feature",
            "feature": "global-tsne",
            "inputs": [{"step": "speed"}],
            "params": {
                "templates": {"step": "templates", "pattern": "templates.parquet"},
                "perplexity": 5,
            },
        },
    ],
}


@pytest.fixture
def worked_dataset(tmp_path: Path) -> Dataset:
    """Two tracked sequences and the media they came from, all pinned."""
    dataset = make_dataset(tmp_path / "worked")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    write_media_index(
        dataset, ["seq_a", "seq_b"], uids={"seq_a": "uuid-a", "seq_b": "uuid-b"}
    )
    return dataset


def _case_values(plan: Plan, recipe: Recipe) -> dict[str, str]:
    """Every pinned value, keyed by what it describes."""
    values: dict[str, str] = {"recipe/digest": recipe_digest(recipe)}
    for planned in plan.steps:
        values[f"{planned.step_id}/run_id"] = planned.run_id or ""
        values[f"{planned.step_id}/storage_name"] = planned.storage_name
        values[f"{planned.step_id}/tracks_variant"] = planned.tracks_variant
        values[f"{planned.step_id}/lane"] = planned.lane
        values[f"{planned.step_id}/entries"] = ";".join(
            f"{group}:{sequence}" for group, sequence in planned.spec.entries
        )
    return values


def _resolved(dataset: Dataset) -> dict[str, str]:
    recipe = Recipe.model_validate(WORKED_RECIPE)
    return _case_values(plan_pipeline(dataset, recipe), recipe)


def _load_golden() -> dict[str, str]:
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"No golden corpus at {GOLDEN_PATH}. Run "
            f"`{UPDATE_ENV}=1 pytest tests/test_pipeline_golden.py`."
        )
    loaded: object = json.loads(GOLDEN_PATH.read_text())
    assert isinstance(loaded, dict)
    return {str(key): str(value) for key, value in loaded.items()}


def test_the_worked_plan_matches_golden(worked_dataset: Dataset) -> None:
    """Every field of every step, byte for byte."""
    if os.environ.get(UPDATE_ENV) == "1":
        pytest.skip(f"{UPDATE_ENV}=1: regenerating, see test_regenerate_golden")

    assert _resolved(worked_dataset) == _load_golden()


def test_regenerate_golden(worked_dataset: Dataset) -> None:
    """Rewrite the golden file. Runs only under the update environment variable."""
    if os.environ.get(UPDATE_ENV) != "1":
        pytest.skip(f"set {UPDATE_ENV}=1 to regenerate")
    fresh = _resolved(worked_dataset)
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ = GOLDEN_PATH.write_text(json.dumps(fresh, indent=2, sort_keys=True) + "\n")
    assert fresh


def test_the_plan_is_the_same_on_a_second_call(worked_dataset: Dataset) -> None:
    """Planning is pure, so two calls over one dataset cannot disagree.

    Cheap, and it is what would notice an identity term picking up wall clock or
    iteration order -- the suite runs under a randomized hash seed, so an
    unordered term yields a different name per process rather than per call.
    """
    assert _resolved(worked_dataset) == _resolved(worked_dataset)
