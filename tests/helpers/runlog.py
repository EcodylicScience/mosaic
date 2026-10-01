"""Read an attempt's run-log the way a test asserts on it."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import TypeAdapter

from mosaic.core.dataset import Dataset
from mosaic.runlog import RunLogSnapshot, reduce_run_log, run_log_dir, run_log_path

_EVENT: Final = TypeAdapter(dict[str, object])


def entry_error_lines(ds: Dataset, execution_id: str) -> list[str]:
    """Return every ``entry_error`` event that attempt *execution_id* logged.

    The events stay raw lines instead of parsed records. A test can then assert on
    an error's class, its entry and its message with one substring each.
    """
    log = run_log_path(ds.base_dir, execution_id).read_text()
    return [line for line in log.splitlines() if '"entry_error"' in line]


def latest_snapshot(ds: Dataset) -> RunLogSnapshot:
    """Return the folded run-log of the attempt in *ds* that logged last."""
    snapshot = reduce_run_log(_latest_log(ds))
    assert snapshot is not None
    return snapshot


def latest_events(ds: Dataset, ev: str) -> list[dict[str, object]]:
    """Return every *ev* event of the attempt in *ds* that logged last, in order."""
    lines = _latest_log(ds).read_text().splitlines()
    events = [_EVENT.validate_json(line) for line in lines]
    return [event for event in events if event.get("ev") == ev]


def _latest_log(ds: Dataset) -> Path:
    """Return the run-log in *ds* written last."""
    logs = run_log_dir(ds.base_dir).glob("*.jsonl")
    return max(logs, key=lambda path: path.stat().st_mtime)
