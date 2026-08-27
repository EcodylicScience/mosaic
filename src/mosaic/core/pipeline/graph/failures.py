"""What went wrong, kept where re-deriving it from disk is impossible.

Step status is derived from the artifact record and never stored -- that is what
kills the state-machine class of bugs. A **failure** is the one thing that rule
cannot cover, because absent-because-quarantined and absent-because-never-run are
the same observation on disk. So a failure record is durable, and it is the only
bound on the resubmit-forever case: a deterministically bad sequence -- a corrupt
video, a pose with no keypoints -- cannot be retried into success, and nothing
else counts the attempts.

Two granularities, and they answer different questions:

* **Per entry** ``(storage_name, run_id, group, sequence)`` -- one sequence failed
  and the rest of the step is fine. A step-level backoff would quarantine a whole
  branch over one sequence, which is what this exists to avoid.
* **Per step** ``(storage_name, run_id)`` -- the attempt did not get far enough to
  blame an entry.

**Attempts are global; the decision to proceed without an entry is not.** A count
that reset on every resubmit would bound nothing, since a resubmit is the cheap
and expected recovery. But ``allow_partial`` is a scientific decision -- a model
fitted on 89 sequences is a different model from one fitted on 90 -- so one
request's answer must not silently bind another's. The exclusions therefore live
under the request that made them.

**Waiting and quarantined are kept apart, and which one a step meets matters.** An
entry inside its backoff has failed recently and is not due yet; an entry past the
attempt bound will not succeed. Only the second is something to decide about.
Treating a wait as a verdict would let one explicit gesture *permanently* drop an
entry that needed nothing but a few more seconds -- and an exclusion changes what
a fit is, with nothing later putting it back.

This module declares the vocabulary and nothing else: no filesystem, no registry,
no pandas. A release gate reads these far more often than a submission writes
them, so the cost of importing this has to stay near zero.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, ClassVar, Final, Protocol

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mosaic.core.entry import Entry

__all__ = [
    "BACKOFF_CEILING_SECONDS",
    "BACKOFF_SECONDS",
    "QUARANTINE_AFTER",
    "EntryFailureKey",
    "Exclusions",
    "FailureRecord",
    "FailureStore",
    "StepFailureKey",
    "backoff_until",
]

QUARANTINE_AFTER: Final = 3
"""Attempts after which an entry is quarantined rather than tried again.

Small on purpose. The case this bounds is a *deterministically* bad entry, which
the second attempt already establishes; the third is there so a genuine transient
-- a node evicted twice -- is not mistaken for one.
"""

BACKOFF_SECONDS: Final = 60.0
"""The first wait after a failure. Doubled per attempt, up to the ceiling."""

BACKOFF_CEILING_SECONDS: Final = 3600.0
"""The longest a backoff grows to, so a long-running request still converges."""


type EntryFailureKey = tuple[str, str, str, str]
"""``(storage_name, run_id, group, sequence)`` -- one entity of one run."""

type StepFailureKey = tuple[str, str]
"""``(storage_name, run_id)`` -- a whole step that failed before naming an entity."""


def backoff_until(attempts: int, *, now: datetime.datetime | None = None) -> str:
    """When an entry with this many failures may be tried again, ISO-8601 UTC.

    Exponential from :data:`BACKOFF_SECONDS` and capped at
    :data:`BACKOFF_CEILING_SECONDS`. The cap matters more than the growth rate: an
    uncapped doubling puts the fourth retry beyond any session a person is
    watching, which reads as a hang rather than as a wait.
    """
    moment = now or datetime.datetime.now(datetime.timezone.utc)
    wait = min(BACKOFF_SECONDS * (2 ** max(attempts - 1, 0)), BACKOFF_CEILING_SECONDS)
    return (moment + datetime.timedelta(seconds=wait)).isoformat()


class FailureRecord(BaseModel):
    """How often one thing has failed, why, and when it may be tried again.

    A model rather than a dataclass because it is read back off disk, and
    ``extra="ignore"`` is what lets a record written by a later mosaic be read by
    an earlier one -- the same call :class:`~mosaic.core.pipeline.markers.InflightMarker`
    makes, for the same reason.

    Attributes:
        attempts: How many failures have been recorded. Never decremented; a
            success clears the record outright.
        last_error: A short description of the most recent failure, for a human.
        not_before: ISO-8601 UTC instant before which a retry is pointless, or
            empty when there is no wait.
        last_execution_id: The attempt that recorded the most recent failure, so
            the full traceback is one run-log away.
        quarantined: Whether this has failed often enough to stop trying.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    attempts: int = 0
    last_error: str = ""
    not_before: str = ""
    last_execution_id: str = ""
    quarantined: bool = False

    def ready(self, *, now: datetime.datetime | None = None) -> bool:
        """May this be attempted right now?

        A quarantined record is never ready. An unparseable ``not_before`` reads
        as ready rather than as blocked: refusing to run because a timestamp
        could not be parsed would turn a cosmetic fault into a stall.
        """
        if self.quarantined:
            return False
        if not self.not_before:
            return True
        try:
            deadline = datetime.datetime.fromisoformat(self.not_before)
        except ValueError:
            return True
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=datetime.timezone.utc)
        moment = now or datetime.datetime.now(datetime.timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=datetime.timezone.utc)
        return moment >= deadline


class Exclusions(BaseModel):
    """The entries one request has decided to proceed without.

    Request-scoped desired state rather than a status, which is why it is durable
    without contradicting the rule against stored status: it records a decision a
    person made, and nothing derives it from the artifacts.

    Attributes:
        request_id: Whose decision this is.
        by_step: Entries excluded, keyed by the step that recorded the exclusion.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    request_id: str = ""
    by_step: dict[str, list[tuple[str, str]]] = Field(default_factory=dict)

    @property
    def entries(self) -> frozenset[Entry]:
        """Every excluded entry, whichever step recorded it.

        Flat because that is how a plan takes them: an exclusion narrows the
        scope of the whole graph, which is the honest reading -- a request run
        without a sequence is run without it everywhere, and a step below the one
        that excluded it must not silently see it again.
        """
        return frozenset(
            (group, sequence)
            for entries in self.by_step.values()
            for group, sequence in entries
        )


class FailureStore(Protocol):
    """Where failure records live, whatever that turns out to be.

    A protocol rather than a class because there are two implementations with
    nothing in common: a directory on the dataset for the path with no queue, and
    Redis keys where there is one. The dataset implementation is the one that has
    to exist, since it is what a bare compute node has.

    A store is never authoritative about what *exists* -- that is the artifact
    record's job. It is authoritative only about what has been tried.
    """

    def entry_record(self, key: EntryFailureKey) -> FailureRecord:
        """This entry's record. A thing that never failed reads as a fresh one."""
        ...

    def step_record(self, key: StepFailureKey) -> FailureRecord:
        """This step's record. A thing that never failed reads as a fresh one."""
        ...

    def note_entry_failure(
        self, key: EntryFailureKey, *, error: str, execution_id: str = ""
    ) -> FailureRecord:
        """Count one entry failure and return the record it produced."""
        ...

    def note_step_failure(
        self, key: StepFailureKey, *, error: str, execution_id: str = ""
    ) -> FailureRecord:
        """Count one step failure and return the record it produced."""
        ...

    def clear_entry(self, key: EntryFailureKey) -> None:
        """Forget an entry's failures, because it has now succeeded."""
        ...

    def clear_step(self, key: StepFailureKey) -> None:
        """Forget a step's failures, because it has now succeeded."""
        ...

    def quarantined_entries(
        self, storage_name: str, run_id: str, candidates: Iterable[Entry]
    ) -> frozenset[Entry]:
        """Which of *candidates* have stopped being attempted at all."""
        ...

    def waiting_entries(
        self,
        storage_name: str,
        run_id: str,
        candidates: Iterable[Entry],
        *,
        now: datetime.datetime | None = None,
    ) -> frozenset[Entry]:
        """Which of *candidates* have failed recently and are not due yet."""
        ...

    def exclusions(self, request_id: str) -> Exclusions:
        """What this request has decided to proceed without."""
        ...

    def exclude(self, request_id: str, step_id: str, entries: Iterable[Entry]) -> None:
        """Record that this request proceeds without *entries*."""
        ...
