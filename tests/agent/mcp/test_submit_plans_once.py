"""``submit_pipeline`` plans the recipe once, and plans it again when the dataset moved.

A submission reviews the plan it is given and records a plan. Resolving the same
recipe twice in one call doubled the cost of a submission on a dataset of
thousands of entries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.agent.mcp.local import LocalBackend
from mosaic.agent.mcp.spec import SubmittedPipeline, call_tool
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.graph import PlanWire
from tests.helpers import (
    SPEED_RECIPE,
    RecordingStarter,
    add_tracks_variant,
    make_dataset,
    run_speed,
)

VARIANT = "convert-trex.0.2-1111111111"

SPEED_RECIPE_AT_STEP_ONE: dict[str, object] = {
    "schema_version": 1,
    "name": "speed",
    "steps": [
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "inputs": ["tracks"],
            "params": {"step_size": 1},
        }
    ],
}


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "tracked", name="tracked")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


def _count_plans(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count every resolution of a plan, whichever module asks for it."""
    import mosaic.core.pipeline.graph.plan as plan_module
    import mosaic.core.pipeline.graph.request as request_module

    resolved: list[int] = []
    real = plan_module.plan_pipeline

    def counted(*args: object, **kwargs: object) -> object:
        resolved.append(1)
        return real(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(plan_module, "plan_pipeline", counted)
    monkeypatch.setattr(request_module, "plan_pipeline", counted)
    return resolved


def _fingerprint(
    backend: LocalBackend, dataset: Dataset, recipe: dict[str, object] = SPEED_RECIPE
) -> str:
    plan = call_tool(
        backend, "plan_recipe", {"dataset": dataset.manifest.uuid, "recipe": recipe}
    )
    assert isinstance(plan, PlanWire), plan
    return plan.fingerprint


def test_a_submission_resolves_the_plan_once(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    starter = RecordingStarter()
    backend = LocalBackend([dataset.manifest_path], warm=False, start_driver=starter)
    fingerprint = _fingerprint(backend, dataset)
    resolved = _count_plans(monkeypatch)

    submitted = call_tool(
        backend,
        "submit_pipeline",
        {
            "dataset": dataset.manifest.uuid,
            "recipe": SPEED_RECIPE,
            "plan_fingerprint": fingerprint,
        },
    )

    assert isinstance(submitted, SubmittedPipeline), submitted
    assert resolved == [1]
    assert starter.started == [submitted.request_id]


def test_a_submission_after_the_dataset_moved_is_refused_as_changed(
    dataset: Dataset,
) -> None:
    backend = LocalBackend(
        [dataset.manifest_path], warm=False, start_driver=RecordingStarter()
    )
    recipe = SPEED_RECIPE_AT_STEP_ONE
    fingerprint = _fingerprint(backend, dataset, recipe)
    _ = run_speed(dataset, 1)

    refused = call_tool(
        backend,
        "submit_pipeline",
        {
            "dataset": dataset.manifest.uuid,
            "recipe": recipe,
            "plan_fingerprint": fingerprint,
        },
    )

    assert isinstance(refused, ErrorEnvelope), refused
    assert refused.kind == "conflict"
    assert fingerprint in refused.message
