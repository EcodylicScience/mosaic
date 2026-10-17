"""Removing one whole feature run: its index rows, its directory, and a record of both.

:func:`remove_feature_run` is the one way a whole feature run is removed outside
``Pipeline.clean``. It goes in four steps, each before the next:

1. It refuses a run directory outside the dataset's ``features`` root, or an
   index row of the run that names a file outside that directory.
2. It records the intent in the cleanup's run-log, with the run's whole
   ``params.json``, from which the run can be computed again.
3. It drops the run's index rows.
4. It deletes the run's directory, and records that it did.

A crash leaves at worst an intent without an outcome, which reads as
``interrupted`` (:func:`~mosaic.runlog.read_removal_records`), or rows already
dropped beside a directory still on disk, which ``mosaic reindex`` and a resumed
cleanup repair. A deletion never happens without its record.

Importing this module imports the standard library and :mod:`mosaic.runlog`
alone. The index helpers, which import pandas, are imported where they are used,
so :mod:`~mosaic.core.pipeline.sweep` can measure a directory with
:func:`run_bytes`.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.runlog import JsonlRunLog

    from .inventory.params import RunParams

__all__ = [
    "RemovedRun",
    "finish_feature_run_removal",
    "remove_feature_run",
    "run_bytes",
]


@dataclass(frozen=True, slots=True)
class RemovedRun:
    """One feature run that was removed.

    Attributes:
        storage: Its storage name under ``features/``.
        run_id: The run.
        bytes: The bytes its directory held.
        files: The files its directory held.
        rows_dropped: The index rows dropped.
    """

    storage: str
    run_id: str
    bytes: int
    files: int
    rows_dropped: int


def run_bytes(run_root: Path) -> tuple[int, int]:
    """Return the bytes and the number of files under *run_root*.

    A file that disappears or cannot be read while it is counted is left out,
    and a directory that does not exist holds ``(0, 0)``.
    """
    total = 0
    files = 0
    for child in run_root.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
                files += 1
        except OSError:
            continue
    return total, files


def remove_feature_run(
    ds: Dataset,
    storage: str,
    run_id: str,
    *,
    params: RunParams,
    log: JsonlRunLog,
    reason: str,
    superseded_by: str,
) -> RemovedRun:
    """Remove one feature run: record it, drop its rows, then delete its files.

    The caller has read the run's ``params.json`` and refuses a run without
    one, because a run removed without that record could not be computed again.

    Args:
        ds: The dataset.
        storage: The run's storage name under ``features/``.
        run_id: The run.
        params: The run's recorded ``params.json``, written to the run-log.
        log: The run-log of the cleanup removing it.
        reason: Why it is removed.
        superseded_by: The run that replaces it.

    Returns:
        What was removed.

    Raises:
        RuntimeError: The run's directory is outside the ``features`` root, or
            one of its index rows names a file outside that directory. Nothing
            is recorded or removed.
        OSError: Its rows or its directory could not be removed. The intent is
            recorded, and the rows are dropped when only the directory failed.
    """
    from .index import feature_index, feature_index_path

    run_root = _checked_run_root(ds, storage, run_id)
    log.run_removing(
        feature=storage,
        run_id=run_id,
        reason=reason,
        superseded_by=superseded_by,
        params=params.model_dump(mode="json", by_alias=True),
    )
    dropped = feature_index(feature_index_path(ds, storage)).drop_runs({run_id})
    size, files = run_bytes(run_root)
    if run_root.exists():
        shutil.rmtree(run_root)
    log.run_removed(
        feature=storage, run_id=run_id, bytes=size, files=files, rows=len(dropped)
    )
    return RemovedRun(
        storage=storage,
        run_id=run_id,
        bytes=size,
        files=files,
        rows_dropped=len(dropped),
    )


def finish_feature_run_removal(
    ds: Dataset, storage: str, run_id: str, *, log: JsonlRunLog
) -> RemovedRun:
    """Finish a removal that its run-log records as begun and not ended.

    The intent is already recorded. Rows still in the index are dropped, the
    directory is deleted when it is still there, and the outcome is recorded.

    Raises:
        RuntimeError: As for :func:`remove_feature_run`.
    """
    from .index import feature_index, feature_index_path

    run_root = _checked_run_root(ds, storage, run_id)
    dropped = feature_index(feature_index_path(ds, storage)).drop_runs({run_id})
    size, files = run_bytes(run_root)
    if run_root.exists():
        shutil.rmtree(run_root)
    log.run_removed(
        feature=storage, run_id=run_id, bytes=size, files=files, rows=len(dropped)
    )
    return RemovedRun(
        storage=storage,
        run_id=run_id,
        bytes=size,
        files=files,
        rows_dropped=len(dropped),
    )


def _checked_run_root(ds: Dataset, storage: str, run_id: str) -> Path:
    """Return the run's directory, refusing one this module may not delete.

    It must lie strictly inside the ``features`` root, two levels down, reached
    through no symlink, and every index row of the run must name a file inside
    it. A run that fails either is
    evidence that what was asked for was computed wrongly, so this raises rather
    than skipping it, as ``delete_set`` does.

    Raises:
        RuntimeError: The directory or a row's file is outside.
    """
    from .index import feature_index, feature_index_path, feature_run_root

    features = ds.get_root("features").resolve()
    for part in (storage, run_id):
        if not part or part in (".", "..") or "/" in part or "\\" in part:
            raise RuntimeError(
                f"refusing to remove feature run {storage!r}/{run_id!r}: "
                f"{part!r} is not one path component"
            )
    named = feature_run_root(ds, storage, run_id)
    run_root = named.resolve()
    # Resolved, it must be the very path it names: a symlinked storage or run
    # directory resolves to another run, whose rows this would not drop.
    if (
        run_root != features / storage / run_id
        or named.is_symlink()
        or named.parent.is_symlink()
    ):
        raise RuntimeError(
            f"refusing to remove {named}: it is not a run directory of the "
            f"features root {features}, or it is reached through a symlink"
        )
    index_path = feature_index_path(ds, storage)
    if not index_path.exists():
        return run_root
    try:
        rows = feature_index(index_path).read(run_id=run_id)
    except FileNotFoundError:
        return run_root
    stored: list[object] = list(rows["abs_path"]) if "abs_path" in rows.columns else []
    for value in stored:
        if not isinstance(value, str) or not value:
            continue
        resolved = ds.resolve_path(value).resolve()
        if run_root not in resolved.parents:
            raise RuntimeError(
                f"refusing to remove feature run {storage}/{run_id}: its index "
                f"row names {resolved}, outside the run directory {run_root}, so "
                f"the rows are not the run's alone"
            )
    return run_root
