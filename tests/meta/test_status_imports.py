"""Reading a status report, a request listing or an attempt's lost entities is cheap.

Each reads a few small run-log, request and failure-record files. Importing the
modules that build them leaves the feature library, pandas, cv2 and av unloaded.
``mosaic.runlog``, which mosaic-api and mosaic-queue read on every sweep, leaves
pydantic unloaded too.

Each check runs in a fresh interpreter. An in-process probe answers about whatever
an earlier test already imported.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

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
)
"""Each spelling that a status reader is imported by."""


def _loaded(statement: str, candidates: tuple[str, ...]) -> str:
    """Which of *candidates* a fresh interpreter has loaded after *statement*."""
    code = (
        "import sys\n"
        f"{statement}\n"
        f"loaded = sorted(m for m in {candidates!r} if m in sys.modules)\n"
        "print(loaded or 'none')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


@pytest.mark.parametrize("statement", STATUS_IMPORTS)
def test_a_status_import_loads_no_heavy_module(statement: str) -> None:
    assert _loaded(statement, HEAVY_MODULES) == "none"


def test_the_run_log_module_loads_no_pydantic() -> None:
    assert _loaded("import mosaic.runlog", ("pydantic",)) == "none"
