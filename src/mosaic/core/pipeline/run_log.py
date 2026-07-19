"""Back-compat re-export of the run-log, now homed at :mod:`mosaic.runlog`.

The implementation moved to the dependency-light leaf ``mosaic.runlog`` so external
readers (mosaic-api's ledger sweeper, ``mosaic-queue`` workers) can import it without
dragging the heavy ``mosaic.core.pipeline`` package ``__init__`` (numpy / pandas /
matplotlib). This module keeps the historical ``mosaic.core.pipeline.run_log`` import path
working for the in-package writer (``job.py``) and the CLI.
"""

from __future__ import annotations

from mosaic.runlog import (
    JsonlRunLog as JsonlRunLog,
    read_run as read_run,
    read_run_progress as read_run_progress,
    read_runs as read_runs,
    read_runs_by_run_id as read_runs_by_run_id,
    reduce_run_log as reduce_run_log,
    run_log_dir as run_log_dir,
    run_log_path as run_log_path,
)

__all__ = [
    "JsonlRunLog",
    "read_run",
    "read_run_progress",
    "read_runs",
    "read_runs_by_run_id",
    "reduce_run_log",
    "run_log_dir",
    "run_log_path",
]
