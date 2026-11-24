"""A recipe naming a feature definition, run through the detached driver.

``submit_pipeline`` starts the detached driver, which starts one step subprocess,
which computes its entries in spawned process workers. The definition's top level
notes each importing process. The test therefore sees which processes ran its
code.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from mosaic.agent.mcp.local import LocalBackend
from mosaic.agent.mcp.outputs import PlanPage
from mosaic.agent.mcp.spec import SubmittedPipeline, call_tool
from mosaic.core.pipeline.definitions.scope import clear_definition_caches
from mosaic.core.pipeline.definitions.sidecar import DefinitionRef
from mosaic.core.pipeline.graph.status import RequestStatusWire
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.inventory.params import read_run_params
from mosaic.core.pipeline.markers import process_running
from mosaic.core.pipeline.parallelism import ParallelLimits
from tests.helpers import (
    CANARY_NAME,
    CANARY_SOURCE,
    CANARY_VARIABLE,
    defined_dataset,
    forget_imported_definitions,
    imported_definitions,
    wait_for,
)

STORAGE: Final = f"def.{CANARY_NAME}__from__tracks"

RECIPE: Final[dict[str, object]] = {
    "schema_version": 1,
    "name": "canary",
    "steps": [
        {
            "id": "d",
            "type": "feature",
            "feature": f"def:{CANARY_NAME}",
            "inputs": ["tracks"],
        }
    ],
}

REQUEST_SECONDS: Final = 360.0
"""The longest the test waits for the driver and its step process to finish."""

STEP_MARK: Final = "--graph-request"
"""What the command line of a step subprocess, and of each worker it spawns, contains."""


@pytest.fixture(autouse=True)
def no_imported_definitions() -> Iterator[None]:
    """Start and end the test with no definition imported and no sidecar cached."""
    forget_imported_definitions()
    clear_definition_caches()
    yield
    forget_imported_definitions()
    clear_definition_caches()


def _status(backend: LocalBackend, uuid: str, request_id: str) -> RequestStatusWire:
    status = call_tool(
        backend,
        "get_request_status",
        {"dataset": uuid, "request_id": request_id, "brief": False},
    )
    assert isinstance(status, RequestStatusWire), status
    return status


@pytest.mark.slow
def test_a_definition_runs_through_the_driver_in_process_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = defined_dataset(tmp_path, {CANARY_NAME: CANARY_SOURCE})
    # Set after the definition. The child that checks it then notes nothing.
    canary = tmp_path / "canary.jsonl"
    monkeypatch.setenv(CANARY_VARIABLE, str(canary))
    uuid = dataset.manifest.uuid
    assert uuid is not None
    digest = hashlib.sha256(CANARY_SOURCE.encode("utf-8")).hexdigest()
    backend = LocalBackend(
        [dataset.manifest_path],
        warm=False,
        parallel_limits=ParallelLimits(max_workers=2, allow_process_mode=True),
        allow_feature_definitions=True,
    )
    planned = call_tool(backend, "plan_recipe", {"dataset": uuid, "recipe": RECIPE})
    assert isinstance(planned, PlanPage), planned
    (planned_step,) = planned.steps

    submitted = call_tool(
        backend,
        "submit_pipeline",
        {
            "dataset": uuid,
            "recipe": RECIPE,
            "plan_fingerprint": planned.fingerprint,
            "workers": 2,
            "parallel_mode": "process",
        },
    )

    assert isinstance(submitted, SubmittedPipeline), submitted
    assert (submitted.workers, submitted.parallel_mode) == (2, "process")
    wait_for(
        lambda: _status(backend, uuid, submitted.request_id).terminal,
        REQUEST_SECONDS,
        "the detached driver never ended the request",
    )
    # The driver outlives this test's process otherwise.
    wait_for(
        lambda: not process_running(submitted.driver_pid),
        REQUEST_SECONDS,
        "the detached driver never exited",
    )
    status = _status(backend, uuid, submitted.request_id)
    assert status.verdict == "finished", status
    (step,) = status.steps
    assert step.run_id == planned_step.run_id
    root = feature_run_root(dataset, STORAGE, step.run_id)
    assert sorted(path.name for path in root.glob("*.parquet")) == [
        "g__s1.parquet",
        "g__s2.parquet",
    ]
    document = json.loads((root / "params.json").read_text(encoding="utf-8"))
    assert document["_definition"] == {"name": CANARY_NAME, "digest": digest}
    read = read_run_params(root)
    assert read.params is not None
    assert read.params.definition == DefinitionRef(CANARY_NAME, digest)
    imports = [json.loads(line) for line in canary.read_text().splitlines()]
    importers = {record["pid"] for record in imports}
    assert importers.isdisjoint({os.getpid(), submitted.driver_pid})
    # The step subprocess and each spawned worker import the definition. A spawned
    # worker inherits the step's command line. The count of distinct importers,
    # not the command line, therefore shows that the workers imported it.
    assert len(importers) >= 2, imports
    for record in imports:
        assert STEP_MARK in " ".join(record["argv"]), record["argv"]
    assert not imported_definitions()
