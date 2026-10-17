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


def record_open_attempt(ds: Dataset, run_id: str) -> str:
    """Record an attempt of this process that has named *run_id* and not ended.

    Its process is this one, so the attempt reads ``alive`` while the test runs.

    Returns:
        The attempt's execution id.
    """
    import os
    import socket

    from mosaic.runlog import JsonlRunLog, new_execution_id

    execution_id = new_execution_id()
    with JsonlRunLog(run_log_path(ds.base_dir, execution_id), execution_id) as log:
        log.started(
            kind="feature", target="x", host=socket.gethostname(), pid=os.getpid()
        )
        log.set_run_id(run_id)
    return execution_id


def write_reader_params(
    ds: Dataset, storage: str, run_id: str, reads: tuple[str, str]
) -> None:
    """Write the ``params.json`` of a run of *storage* that reads run *reads*.

    The run has no index rows, as an attempt that has named its run and not yet
    written an entry.
    """
    import json

    run_root = ds.get_root("features") / storage / run_id
    run_root.mkdir(parents=True)
    feature, read = reads
    document: dict[str, object] = {
        "_params": {},
        "_inputs": [{"feature": feature, "run_id": read}],
        "_resolved": [{"where": "inputs[0]", "feature": feature, "run_id": read}],
    }
    _ = (run_root / "params.json").write_text(json.dumps(document))
