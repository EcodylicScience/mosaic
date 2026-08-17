"""Dataset context helpers for the CLI (the library analogue of a DB session).

The library takes an explicit ``--manifest <dataset.yaml>`` -- there is no
``MOSAIC_DATA_ROOT`` here (that is a mosaic-api concept). Heavy imports stay
lazy so ``--help`` and the read-only commands never pull in the feature/tracking
stacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.cli._io import fail
from mosaic.user_paths import user_path

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


def load_dataset(manifest: Path) -> "Dataset":
    """Load the dataset at *manifest*, reporting a failure as a clean exit.

    The opening itself is :func:`~mosaic.core.dataset.open_dataset`. What stays
    here is the presentation: a CLI reports a missing or unreadable manifest as
    a one-line message and a non-zero exit, not as a traceback.
    """
    from mosaic.core.dataset import open_dataset

    # Every `--manifest` in the CLI arrives here, so this is the one place the
    # tilde is expanded. A shell expands an unquoted `~` itself, but not a quoted
    # one, and not one that reached the argument from a script or a config file.
    manifest = user_path(manifest)
    if not manifest.exists():
        fail(f"Manifest not found: {manifest}")
    try:
        return open_dataset(manifest)
    except Exception as exc:  # noqa: BLE001 - surface any load failure cleanly
        fail(f"Failed to load dataset from {manifest}: {exc}")


def run_log_dir_for(ds: "Dataset") -> Path:
    """Return the run-log directory for *ds* (``<dataset_root>/.mosaic/runs``).

    This is the append-only JSONL status/progress bridge that replaced the
    per-dataset ``.mosaic.db``. It is dataset-level (not under ``features/``), so
    it works for tracking-only datasets too.
    """
    from mosaic.core.pipeline.run_log import run_log_dir

    return run_log_dir(ds.base_dir)


def entry_failure_status(ds: "Dataset", execution_id: str) -> dict[str, object]:
    """``status`` and ``entries_failed`` for an attempt, read back from its log.

    An op that lost some entities but not all is ``partial``, not ``finished``:
    it exited 0, and saying ``finished`` would report a run that published
    nothing for those entries as a clean success. ``run_feature`` already
    reports this way; ops and trackers hardcoded the literal.

    Read from the run-log rather than returned by ``run_op``, whose ``-> str``
    is also called by mosaic-queue and mosaic-api -- and the log is the record
    that survives the queue giving the child's stderr to ``DEVNULL``.
    ``reduce_run_log`` already folds the ``entry_error`` events into
    ``entries_failed``, so nothing new is derived here.

    ``partial`` is deliberately **not** a terminal status and is absent from
    ``runlog.TERMINAL_STATUSES``: mosaic-api's sweeper treats that set as
    terminal and would reap a live run. It is a reporting word, computed here.
    """
    from mosaic.core.pipeline.run_log import read_run

    snapshot = read_run(run_log_dir_for(ds), execution_id)
    failed = int(snapshot["entries_failed"]) if snapshot else 0
    return {
        "status": "partial" if failed else "finished",
        "entries_failed": failed,
    }
