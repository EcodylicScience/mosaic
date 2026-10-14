"""Reading a status report, a request listing or an attempt's lost entities is cheap.

Each reads a few small run-log, request and failure-record files. Importing the
modules that build them, and calling them, leaves the feature library, pandas,
cv2 and av unloaded. ``mosaic.runlog``, which mosaic-api and mosaic-queue read on
every sweep, leaves pydantic unloaded too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.pipeline.graph import run_pipeline
from tests.helpers import (
    add_tracks_variant,
    make_dataset,
    modules_loaded_by,
    speed_recipe,
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
)
"""Each spelling that a status reader is imported by."""


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
