"""What has been tried, kept where re-deriving it from disk is impossible.

Absent-because-quarantined and absent-because-never-run are the same observation
on the artifacts, so the count of attempts is the one thing derived status cannot
answer -- and it is the only bound on retrying a deterministically bad sequence
forever.

Two halves are tested apart because they have different lifetimes. Attempts are
global and survive a resubmit, since a resubmit is the cheap and expected
recovery and a counter that reset on one would bound nothing. The decision to
proceed *without* an entry is per request, because it is a scientific choice --
a model fitted on 89 sequences is a different model -- and one request's answer
must not silently bind another's.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from mosaic.core.pipeline.graph import (
    QUARANTINE_AFTER,
    FailureRecord,
    FileFailureStore,
    claims_root,
)
from mosaic.core.pipeline.graph.claims import entry_record_path, exclusions_path
from mosaic.core.pipeline.graph.failures import backoff_until

ENTRY = ("speed-angvel__from__tracks", "0.2-abcdef0123", "g1", "seq_a")
STEP = ("speed-angvel__from__tracks", "0.2-abcdef0123")

LATER = datetime.datetime(2099, 1, 1, tzinfo=datetime.timezone.utc)


def test_a_thing_that_never_failed_reads_as_a_fresh_record(tmp_path: Path) -> None:
    """Every caller asks "what do we know", and nothing known is nothing failed."""
    store = FileFailureStore(tmp_path)

    assert store.entry_record(ENTRY) == FailureRecord()
    assert store.step_record(STEP) == FailureRecord()
    assert store.entry_record(ENTRY).ready()


def test_failures_accumulate_and_then_quarantine(tmp_path: Path) -> None:
    """A deterministically bad sequence cannot be retried into success."""
    store = FileFailureStore(tmp_path)

    for attempt in range(1, QUARANTINE_AFTER + 1):
        record = store.note_entry_failure(ENTRY, error="boom", execution_id="01ABC")
        assert record.attempts == attempt

    held = store.entry_record(ENTRY)
    assert held.quarantined
    assert not held.ready(now=LATER), "quarantine outlasts any backoff"
    assert held.last_error == "boom"
    assert held.last_execution_id == "01ABC"


def test_a_backoff_holds_an_entry_back_and_then_releases_it(tmp_path: Path) -> None:
    """Before the quarantine, a failure is a wait rather than a verdict."""
    store = FileFailureStore(tmp_path)

    _ = store.note_entry_failure(ENTRY, error="transient")

    held = store.entry_record(ENTRY)
    assert not held.quarantined
    assert not held.ready(now=datetime.datetime.now(datetime.timezone.utc))
    assert held.ready(now=LATER)


def test_the_backoff_is_capped(tmp_path: Path) -> None:
    """An uncapped doubling puts a retry beyond any session a person is watching."""
    _ = tmp_path
    far = backoff_until(
        40, now=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    )

    assert far <= "2026-01-01T01:00:00+00:00"


def test_success_forgets_what_went_before(tmp_path: Path) -> None:
    """The record is about what is still being tried, not a history."""
    store = FileFailureStore(tmp_path)
    _ = store.note_entry_failure(ENTRY, error="boom")

    store.clear_entry(ENTRY)

    assert store.entry_record(ENTRY) == FailureRecord()


def test_attempts_survive_a_new_request(tmp_path: Path) -> None:
    """The key holds no request, which is what makes the count a real bound.

    A resubmit is the cheap recovery, so a counter reset by one would never stop
    a sequence that cannot succeed.
    """
    store = FileFailureStore(tmp_path)
    _ = store.note_entry_failure(ENTRY, error="boom")

    # A second store over the same dataset is what a second process is.
    again = FileFailureStore(tmp_path)

    assert again.note_entry_failure(ENTRY, error="boom again").attempts == 2


def test_a_recent_failure_is_waiting_and_not_quarantined(tmp_path: Path) -> None:
    """The distinction a caller must not collapse.

    An entry inside its backoff needs a few more seconds; an entry past the
    attempt bound will not succeed. Only the second is something to decide about,
    and a decision permanently drops an entry from a fit.
    """
    store = FileFailureStore(tmp_path)
    _ = store.note_entry_failure(ENTRY, error="boom")
    candidates = [("g1", "seq_a"), ("g1", "seq_b")]

    assert store.waiting_entries(ENTRY[0], ENTRY[1], candidates) == frozenset(
        {("g1", "seq_a")}
    )
    assert store.quarantined_entries(ENTRY[0], ENTRY[1], candidates) == frozenset()


def test_a_quarantined_entry_is_no_longer_merely_waiting(tmp_path: Path) -> None:
    """It is not waiting for anything, so it must not read as a wait."""
    store = FileFailureStore(tmp_path)
    for _ in range(QUARANTINE_AFTER):
        _ = store.note_entry_failure(ENTRY, error="boom")
    candidates = [("g1", "seq_a")]

    assert store.quarantined_entries(ENTRY[0], ENTRY[1], candidates) == frozenset(
        {("g1", "seq_a")}
    )
    assert store.waiting_entries(ENTRY[0], ENTRY[1], candidates) == frozenset()


def test_a_record_that_cannot_be_read_gates_nothing(tmp_path: Path) -> None:
    """It gates a retry, not an artifact, so ignoring it costs one attempt.

    Raising instead would make a branch unrunnable because a bookkeeping file
    went bad, which is the more expensive mistake by a long way.
    """
    store = FileFailureStore(tmp_path)
    _ = store.note_entry_failure(ENTRY, error="boom")
    path = entry_record_path(tmp_path, ENTRY)
    _ = path.write_text("{ not json at all")

    assert store.entry_record(ENTRY) == FailureRecord()


# --- exclusions, which are per request ---------------------------------------


def test_an_exclusion_belongs_to_the_request_that_made_it(tmp_path: Path) -> None:
    """One request's scientific decision must not bind another's."""
    store = FileFailureStore(tmp_path)

    store.exclude("req-a", "templates", [("g1", "seq_a")])

    assert store.exclusions("req-a").entries == frozenset({("g1", "seq_a")})
    assert store.exclusions("req-b").entries == frozenset()


def test_excluding_twice_joins_rather_than_replaces(tmp_path: Path) -> None:
    """Two decisions about two entries, and losing the first re-admits one."""
    store = FileFailureStore(tmp_path)

    store.exclude("req-a", "templates", [("g1", "seq_a")])
    store.exclude("req-a", "templates", [("g1", "seq_b")])

    assert store.exclusions("req-a").entries == frozenset(
        {("g1", "seq_a"), ("g1", "seq_b")}
    )


def test_an_exclusion_records_which_step_decided(tmp_path: Path) -> None:
    store = FileFailureStore(tmp_path)

    store.exclude("req-a", "templates", [("g1", "seq_a")])
    store.exclude("req-a", "scaler", [("g1", "seq_b")])

    held = store.exclusions("req-a")
    assert sorted(held.by_step) == ["scaler", "templates"]
    assert held.by_step["templates"] == [("g1", "seq_a")]


def test_excluding_nothing_writes_nothing(tmp_path: Path) -> None:
    """An empty decision is not a decision."""
    store = FileFailureStore(tmp_path)

    store.exclude("req-a", "templates", [])

    assert not exclusions_path(tmp_path, "req-a").exists()


# --- where it all lives -------------------------------------------------------


def test_records_live_beside_the_run_logs_and_not_inside_them(
    tmp_path: Path,
) -> None:
    """A run-log describes one attempt; these outlive every attempt they count."""
    store = FileFailureStore(tmp_path)
    _ = store.note_entry_failure(ENTRY, error="boom")
    store.exclude("req-a", "templates", [("g1", "seq_a")])

    assert claims_root(tmp_path) == tmp_path / ".mosaic" / "claims"
    assert entry_record_path(tmp_path, ENTRY).is_relative_to(claims_root(tmp_path))
    assert exclusions_path(tmp_path, "req-a").is_relative_to(claims_root(tmp_path))


def test_an_entry_name_a_filesystem_would_reject_is_encoded(tmp_path: Path) -> None:
    """An entry name goes through the same encoding every other path does."""
    store = FileFailureStore(tmp_path)
    awkward = ("storage", "0.1-aaaaaaaaaa", "g/1", "seq a")

    record = store.note_entry_failure(awkward, error="boom")

    assert record.attempts == 1
    path = entry_record_path(tmp_path, awkward)
    assert "/" not in path.name
    assert json.loads(path.read_text())["attempts"] == 1
