"""Read the outcome that an entry task returns the way a test asserts on it."""

from __future__ import annotations

import pandas as pd

from mosaic.core.pipeline.entry_pool import EntryFailure
from mosaic.core.pipeline.entry_task import EntryReport, EntryTask


def report_of(outcome: EntryReport | EntryFailure) -> EntryReport:
    """Return *outcome* as the report it is, raising the error of a failure."""
    if isinstance(outcome, EntryFailure):
        outcome.reraise()
    return outcome


def written_by_worker(
    task: EntryTask, outcome: EntryReport | EntryFailure
) -> pd.DataFrame:
    """Return the output at *task*'s path, once *outcome* reports it written.

    A failed *outcome* raises its error.
    """
    _ = report_of(outcome)
    return pd.read_parquet(task.meta.out_path)
