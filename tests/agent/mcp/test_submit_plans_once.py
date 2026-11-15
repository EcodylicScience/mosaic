"""``submit_pipeline`` plans the recipe once, and plans it again when the dataset moved.

A submission reviews the plan it is given and records a plan. Resolving the same
recipe twice in one call doubled the cost of a submission on a dataset of
thousands of entries. A feature definition whose ``current`` moves after the review
keeps the version the reviewed plan pinned.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest

import mosaic.core.pipeline.graph as graph_package
from mosaic.agent.mcp.local import LocalBackend
from mosaic.agent.mcp.spec import DatasetChange, SubmittedPipeline, call_tool
from mosaic.core.pipeline.cancel import read_request
from mosaic.core.pipeline.definitions.scope import clear_definition_caches
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.graph import PlanWire
from tests.helpers import (
    MINIMAL_NAME,
    MINIMAL_SOURCE,
    SPEED_RECIPE,
    RecordingStarter,
    add_tracks_variant,
    defined_dataset,
    forget_imported_definitions,
    imported_definitions,
    make_dataset,
    observed,
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

    def count(*_args: object, **_kwargs: object) -> None:
        resolved.append(1)

    counted = observed(plan_module.plan_pipeline, count)

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


DEFINITION_RECIPE: dict[str, object] = {
    "schema_version": 1,
    "name": "distance",
    "steps": [
        {
            "id": "d",
            "type": "feature",
            "feature": f"def:{MINIMAL_NAME}",
            "inputs": ["tracks"],
        }
    ],
}

EDITED_SOURCE = MINIMAL_SOURCE.replace("scale: float = 1.0", "scale: float = 2.0")


@pytest.fixture
def defined(tmp_path: Path) -> Iterator[Dataset]:
    forget_imported_definitions()
    clear_definition_caches()
    yield defined_dataset(tmp_path, {MINIMAL_NAME: MINIMAL_SOURCE})
    forget_imported_definitions()
    clear_definition_caches()


def _define_edit(dataset: Dataset) -> str:
    """Define the edited source, which moves ``current``, and return its digest."""
    changed = call_tool(
        LocalBackend(
            [dataset.manifest_path], warm=False, allow_feature_definitions=True
        ),
        "manage_dataset",
        {
            "action": "define_feature",
            "dataset": dataset.manifest.uuid,
            "name": MINIMAL_NAME,
            "source": EDITED_SOURCE,
        },
    )
    assert isinstance(changed, DatasetChange), changed
    assert changed.definition is not None
    return changed.definition.digest


def test_a_moved_current_between_plan_and_submit_keeps_the_planned_digest(
    defined: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    starter = RecordingStarter()
    backend = LocalBackend(
        [defined.manifest_path],
        warm=False,
        start_driver=starter,
        allow_feature_definitions=True,
    )
    planned = call_tool(
        backend,
        "plan_recipe",
        {"dataset": defined.manifest.uuid, "recipe": DEFINITION_RECIPE},
    )
    assert isinstance(planned, PlanWire), planned
    first = hashlib.sha256(MINIMAL_SOURCE.encode("utf-8")).hexdigest()
    edited: list[str] = []

    def define_edit(*_args: object, **_kwargs: object) -> None:
        edited.append(_define_edit(defined))

    # The edit lands after the submission reviewed its plan and before it records.
    monkeypatch.setattr(
        graph_package,
        "submit_request",
        observed(graph_package.submit_request, define_edit),
    )

    submitted = call_tool(
        backend,
        "submit_pipeline",
        {
            "dataset": defined.manifest.uuid,
            "recipe": DEFINITION_RECIPE,
            "plan_fingerprint": planned.fingerprint,
        },
    )

    assert isinstance(submitted, SubmittedPipeline), submitted
    request = read_request(defined.base_dir, submitted.request_id)
    (step,) = submitted.first_steps
    assert request.definitions == {MINIMAL_NAME: first}
    assert len(edited) == 1
    assert edited[0] != first
    assert step.run_id == planned.steps[0].run_id
    assert not imported_definitions()


def test_a_moved_current_after_the_plan_refuses_its_fingerprint(
    defined: Dataset,
) -> None:
    starter = RecordingStarter()
    backend = LocalBackend(
        [defined.manifest_path],
        warm=False,
        start_driver=starter,
        allow_feature_definitions=True,
    )
    planned = call_tool(
        backend,
        "plan_recipe",
        {"dataset": defined.manifest.uuid, "recipe": DEFINITION_RECIPE},
    )
    assert isinstance(planned, PlanWire), planned
    _ = _define_edit(defined)

    refused = call_tool(
        backend,
        "submit_pipeline",
        {
            "dataset": defined.manifest.uuid,
            "recipe": DEFINITION_RECIPE,
            "plan_fingerprint": planned.fingerprint,
        },
    )

    assert isinstance(refused, ErrorEnvelope), refused
    assert refused.kind == "conflict"
    assert starter.started == []
