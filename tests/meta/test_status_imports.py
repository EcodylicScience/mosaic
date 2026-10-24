"""Reading a status report, a request listing or an attempt's lost entities is cheap.

Each reads a few small run-log, request and failure-record files. Importing the
modules that build them, and calling them, leaves the feature library, pandas,
cv2 and av unloaded. ``mosaic.runlog``, which mosaic-api and mosaic-queue read on
every sweep, leaves pydantic unloaded too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.pipeline.graph import (
    CleanupPolicy,
    plan_fingerprint,
    plan_with_cleanup,
    run_pipeline,
    submit_request,
)
from tests.helpers import (
    add_tracks_variant,
    make_dataset,
    modules_loaded_by,
    run_speed,
    speed_recipe,
    speed_recipe_stepping,
)

HEAVY_MODULES = ("pandas", "cv2", "av", "mosaic.behavior.feature_library")
"""The modules that a status import must not load."""

STATUS_IMPORTS = (
    "import mosaic.runlog",
    "import mosaic.core.pipeline.graph.rollup",
    "import mosaic.core.pipeline.graph.status",
    "from mosaic.core.pipeline.graph import request_status_wire, run_status_wire",
    "from mosaic.core.pipeline.graph import RequestStatusWire, RunStatusWire",
    "import mosaic.core.pipeline.graph.store",
    "from mosaic.core.pipeline.graph import list_requests, RequestPageWire",
    "import mosaic.core.pipeline.graph.entry_errors",
    "from mosaic.core.pipeline.graph import entry_errors_wire, EntryErrorsWire",
    "import mosaic.core.pipeline.graph.driver_lease",
    "from mosaic.core.pipeline.graph import driver_alive, read_driver_lease",
    "import mosaic.core.pipeline.graph.liveness",
    "import mosaic.core.pipeline.graph.retirement",
    "import mosaic.core.pipeline.graph.saved",
    "import mosaic.core.pipeline.graph.patch",
    "from mosaic.core.pipeline.graph import retirable_recipes, RetirableRecipeWire",
    "from mosaic.core.pipeline.cancel import cancel_attempt, cancel_request",
)
"""Each spelling that a status reader, or a cancel, is imported by."""


@pytest.mark.parametrize("statement", STATUS_IMPORTS)
def test_a_status_import_loads_no_heavy_module(statement: str) -> None:
    assert modules_loaded_by(statement, HEAVY_MODULES) == []


def test_the_run_log_module_loads_no_pydantic() -> None:
    assert modules_loaded_by("import mosaic.runlog", ("pydantic",)) == []


def test_the_status_reads_load_no_heavy_module(tmp_path: Path) -> None:
    """Each reader runs over a finished request, in one fresh interpreter."""
    ds = make_dataset(tmp_path / "ds")
    add_tracks_variant(ds, "convert-trex.0.2-1111111111", "seq_a")
    done = run_pipeline(ds, speed_recipe())
    root = str(ds.base_dir)
    statement = (
        "from mosaic.core.pipeline.graph import (\n"
        "    entry_errors_wire, list_requests, load_request,\n"
        "    request_status_wire, run_status_wire,\n"
        ")\n"
        "from mosaic.runlog import list_runs, run_log_dir\n"
        f"root = {root!r}\n"
        f"execution_id = {done.outcomes[0].execution_id!r}\n"
        "_ = run_status_wire(root, execution_id, progress=True)\n"
        "_ = entry_errors_wire(root, execution_id)\n"
        "_ = list_runs(run_log_dir(root), limit=1)\n"
        "page = list_requests(root, limit=1)\n"
        "request = load_request(root, page.items[0].request_id)\n"
        "_ = request_status_wire(root, request)"
    )

    assert modules_loaded_by(statement, HEAVY_MODULES) == []


def test_a_status_with_a_cleanup_loads_no_heavy_module(tmp_path: Path) -> None:
    """A request whose cleanup removed a run reports it without pandas."""
    ds = make_dataset(tmp_path / "ds")
    add_tracks_variant(ds, "convert-trex.0.2-1111111111", "seq_a")
    _ = run_speed(ds, 1)
    recipe = speed_recipe_stepping(2)
    policy = CleanupPolicy(keep_newest=0, min_age_hours=0)
    fingerprint = plan_fingerprint(plan_with_cleanup(ds, recipe, cleanup=policy))
    request = submit_request(
        ds, recipe, cleanup=policy, expect_plan=fingerprint
    ).request
    _ = run_pipeline(ds, recipe, request=request)
    statement = (
        "from mosaic.core.pipeline.graph import load_request, request_status_wire\n"
        f"root = {str(ds.base_dir)!r}\n"
        f"request = load_request(root, {request.request_id!r})\n"
        "status = request_status_wire(root, request)\n"
        "assert status.cleanup is not None and status.cleanup.removed_runs == 1"
    )

    assert modules_loaded_by(statement, HEAVY_MODULES) == []
