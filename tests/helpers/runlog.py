"""Read an attempt's run-log the way a test asserts on it."""

from __future__ import annotations

from mosaic.core.dataset import Dataset
from mosaic.runlog import run_log_path


def entry_error_lines(ds: Dataset, execution_id: str) -> list[str]:
    """Return every ``entry_error`` event that attempt *execution_id* logged.

    The events stay raw lines instead of parsed records. A test can then assert on
    an error's class, its entry and its message with one substring each.
    """
    log = run_log_path(ds.base_dir, execution_id).read_text()
    return [line for line in log.splitlines() if '"entry_error"' in line]
