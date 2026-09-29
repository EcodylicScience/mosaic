"""Read an index's runs and each run's finish state from its rows.

Every run index records a run's rows with ``run_id``, ``started_at`` and
``finished_at``. The feature scan and the media-variant records read both facts
here, the same way.
"""

from __future__ import annotations

import pandas as pd

from mosaic.core.pipeline.index_csv import index_records

__all__ = ["finish_state", "run_ids"]


def finish_state(frame: pd.DataFrame, run_id: str) -> tuple[str, str, bool]:
    """Return ``(started_at, finished_at, finished)`` for one run, from its rows.

    Each value is the first non-empty cell of its column. A run re-entered for more
    entries reads as finished once any of its rows recorded a finish.
    """
    started, finished = "", ""
    for record in index_records(frame):
        if str(record.get("run_id", "")) != run_id:
            continue
        started = started or str(record.get("started_at", ""))
        finished = finished or str(record.get("finished_at", ""))
    return started, finished, bool(finished)


def run_ids(frame: pd.DataFrame) -> list[str]:
    """Return every run identifier in an index, sorted, or an empty list."""
    if frame.empty or "run_id" not in frame.columns:
        return []
    return sorted({record.get("run_id", "") for record in index_records(frame)})
