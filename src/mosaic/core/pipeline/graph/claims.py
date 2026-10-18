"""Failure records as files under ``<dataset>/.mosaic/claims``.

The implementation a bare compute node has: no Redis, no service, nothing to
deploy. A queue may put the same records somewhere faster, behind the same
protocol -- but this one has to exist, because a graph driven by a shell script
or a job array is the path that must work with nothing else running.

**Records are written through** ``atomic_write``, so a reader never sees half a
document and an interrupted write never destroys a whole one. That is the right
primitive here and the wrong one for a *claim*: ``os.replace`` cannot fail on an
existing target, so two executions both reading a directory as free would both
"take" it with the last writer winning. Where a create must not clobber, the
primitive is ``O_CREAT|O_EXCL`` -- and mosaic already has that in
:func:`~mosaic.core.pipeline.markers.try_create_inflight`, which the training ops
use to make one run root exclusive. This module does not introduce a second one.

**One file per addressable thing, and the address is percent-encoded.** An entry
name may hold any character a filesystem does not want, so it goes through
``to_safe_name`` like every other name mosaic turns into a path. A storage name
whose encoded form is longer than one path component is abbreviated with a
digest.

**Reads are tolerant.** A record that cannot be parsed reads as absent, because
it gates a *retry* rather than an artifact: the cost of ignoring one is at most a
repeated attempt, where the cost of raising is a branch that cannot run because a
bookkeeping file went bad.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeVar

from pydantic import BaseModel, ValidationError

from mosaic.core.entry import make_entry_key, to_safe_name
from mosaic.runlog import ERROR_MESSAGE_LIMIT

from .._utils import NAME_MAX_BYTES, abbreviate
from .failures import (
    QUARANTINE_AFTER,
    EntryFailureKey,
    Exclusions,
    FailureRecord,
    StepFailureKey,
    backoff_until,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mosaic.core.entry import Entry

__all__ = [
    "FileFailureStore",
    "claims_root",
    "entry_record_path",
    "exclusions_path",
    "step_record_path",
]

_ENTRIES_DIR: Final = "entries"
_STEPS_DIR: Final = "steps"
_REQUESTS_DIR: Final = "requests"
_EXCLUSIONS_FILE: Final = "exclusions.json"


def claims_root(base_dir: Path | str) -> Path:
    """``<base_dir>/.mosaic/claims`` -- what has been tried, and what was decided.

    Beside ``.mosaic/runs`` and ``.mosaic/pipelines`` rather than inside either: a
    run-log describes one attempt and a request describes one submission, while
    these outlive both.
    """
    return Path(base_dir) / ".mosaic" / "claims"


def entry_record_path(base_dir: Path | str, key: EntryFailureKey) -> Path:
    """The path of one entity's failure record for one run.

    The file is named by the entry key, the stem that the run's output files use.
    """
    storage_name, run_id, group, sequence = key
    return (
        claims_root(base_dir)
        / _ENTRIES_DIR
        / _storage_component(storage_name)
        / to_safe_name(run_id)
        / f"{make_entry_key(group, sequence)}.json"
    )


def step_record_path(base_dir: Path | str, key: StepFailureKey) -> Path:
    """Where one run's whole-step failure record sits."""
    storage_name, run_id = key
    return (
        claims_root(base_dir)
        / _STEPS_DIR
        / _storage_component(storage_name)
        / f"{to_safe_name(run_id)}.json"
    )


def _storage_component(storage_name: str) -> str:
    """The path component that names *storage_name*'s records.

    It is ``to_safe_name(storage_name)`` when that fits in one component.
    Percent-encoding lengthens a name by at least 2 bytes for every ``+``, ``@``
    or other character it escapes. A storage name that fits on disk can encode
    past the limit, and is then abbreviated to a head and a digest of the whole
    name, which fit once encoded.
    """
    safe = to_safe_name(storage_name)
    if len(safe) <= NAME_MAX_BYTES:
        return safe
    return to_safe_name(
        abbreviate(
            storage_name, lambda short: len(to_safe_name(short)) <= NAME_MAX_BYTES
        )
    )


def exclusions_path(base_dir: Path | str, request_id: str) -> Path:
    """Where one request's exclusion decisions sit."""
    return (
        claims_root(base_dir)
        / _REQUESTS_DIR
        / to_safe_name(request_id)
        / _EXCLUSIONS_FILE
    )


class FileFailureStore:
    """A :class:`~mosaic.core.pipeline.graph.failures.FailureStore` on the dataset."""

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir: Path = Path(base_dir)

    # -- reads -------------------------------------------------------------

    def entry_record(self, key: EntryFailureKey) -> FailureRecord:
        """This entity's record, or a fresh one where nothing has failed."""
        return _load(entry_record_path(self.base_dir, key), FailureRecord)

    def step_record(self, key: StepFailureKey) -> FailureRecord:
        """This step's record, or a fresh one where nothing has failed."""
        return _load(step_record_path(self.base_dir, key), FailureRecord)

    def quarantined_entries(
        self, storage_name: str, run_id: str, candidates: Iterable[Entry]
    ) -> frozenset[Entry]:
        """Which of *candidates* have stopped being attempted at all.

        Asked per candidate rather than by listing the directory, because the
        answer is wanted for a known entry set and a listing would also return
        entries this run was not asked about.
        """
        return frozenset(
            (group, sequence)
            for group, sequence in candidates
            if self.entry_record((storage_name, run_id, group, sequence)).quarantined
        )

    def waiting_entries(
        self,
        storage_name: str,
        run_id: str,
        candidates: Iterable[Entry],
        *,
        now: datetime.datetime | None = None,
    ) -> frozenset[Entry]:
        """Which of *candidates* have failed recently and are not due yet.

        Quarantined entries are **not** here. They are not waiting for anything,
        and a caller that treated the two alike would offer a decision about an
        entry that needs only a few more seconds.
        """
        waiting: set[Entry] = set()
        for group, sequence in candidates:
            record = self.entry_record((storage_name, run_id, group, sequence))
            if not record.quarantined and not record.ready(now=now):
                waiting.add((group, sequence))
        return frozenset(waiting)

    def exclusions(self, request_id: str) -> Exclusions:
        """What this request has decided to proceed without."""
        held = _load(exclusions_path(self.base_dir, request_id), Exclusions)
        return held.model_copy(update={"request_id": request_id})

    # -- writes ------------------------------------------------------------

    def note_entry_failure(
        self, key: EntryFailureKey, *, error: str, execution_id: str = ""
    ) -> FailureRecord:
        """Count one entity failure and return the record it produced."""
        path = entry_record_path(self.base_dir, key)
        record = _advanced(
            _load(path, FailureRecord), error=error, execution_id=execution_id
        )
        _store(path, record)
        return record

    def note_step_failure(
        self, key: StepFailureKey, *, error: str, execution_id: str = ""
    ) -> FailureRecord:
        """Count one step failure and return the record it produced."""
        path = step_record_path(self.base_dir, key)
        record = _advanced(
            _load(path, FailureRecord), error=error, execution_id=execution_id
        )
        _store(path, record)
        return record

    def clear_entry(self, key: EntryFailureKey) -> None:
        """Forget an entity's failures. Call when it succeeds."""
        entry_record_path(self.base_dir, key).unlink(missing_ok=True)

    def clear_step(self, key: StepFailureKey) -> None:
        """Forget a step's failures. Call when it succeeds."""
        step_record_path(self.base_dir, key).unlink(missing_ok=True)

    def exclude(self, request_id: str, step_id: str, entries: Iterable[Entry]) -> None:
        """Record that this request proceeds without *entries*.

        Additive: a second exclusion for one step joins the first rather than
        replacing it. The two are separate decisions about separate entries, and
        losing the earlier one would silently re-admit something already ruled
        out.
        """
        wanted = {(str(group), str(sequence)) for group, sequence in entries}
        if not wanted:
            return
        held = self.exclusions(request_id)
        merged = dict(held.by_step)
        merged[step_id] = sorted(set(merged.get(step_id, [])) | wanted)
        _store(
            exclusions_path(self.base_dir, request_id),
            Exclusions(request_id=request_id, by_step=merged),
        )


def _advanced(record: FailureRecord, *, error: str, execution_id: str) -> FailureRecord:
    """The record one more failure produces."""
    attempts = record.attempts + 1
    return FailureRecord(
        attempts=attempts,
        last_error=error[:ERROR_MESSAGE_LIMIT],
        not_before=backoff_until(attempts),
        last_execution_id=execution_id,
        quarantined=attempts >= QUARANTINE_AFTER,
    )


ModelT = TypeVar("ModelT", bound=BaseModel)


def _load(path: Path, model_cls: type[ModelT]) -> ModelT:
    """Read one record, or the model's own defaults where nothing is readable.

    Defaults rather than ``None``: every caller wants "what do we know about this"
    and a thing nothing is known about is a thing that has never failed, which is
    exactly what the defaults say.
    """
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError):
        return model_cls()
    try:
        return model_cls.model_validate_json(text)
    except ValidationError:
        return model_cls()


def _store(path: Path, record: BaseModel) -> None:
    """Write one record atomically, as indented JSON a person can read."""
    # Imported here so that a reader of failure records leaves numpy and pandas
    # unloaded (``tests/meta/test_status_imports.py``).
    from .._utils import atomic_write

    payload = record.model_dump(mode="json")
    atomic_write(path, lambda temp: temp.write_text(json.dumps(payload, indent=2)))
