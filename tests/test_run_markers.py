"""Completion and in-flight markers, in isolation.

No ``Dataset``, no TREx: these exercise
:mod:`mosaic.core.pipeline.markers` alone, so a failure here localizes to the
marker contract rather than to whatever stage is using it.

The run-log side is built by writing real JSONL through
:class:`~mosaic.runlog.JsonlRunLog`, not by stubbing the reader, because the
liveness rule depends on how ``reduce_run_log`` folds events into a status.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from mosaic.core.pipeline.markers import (
    INFLIGHT_GRACE_SECONDS,
    INFLIGHT_MARKER_NAME,
    INFLIGHT_MIN_TTL_SECONDS,
    MARKER_SCHEMA_VERSION,
    InflightMarker,
    PhaseMarker,
    clear_inflight,
    clear_phase_marker,
    clear_phase_markers,
    inflight_expiry,
    inflight_marker_path,
    inflight_state,
    new_inflight,
    phase_marker_path,
    read_inflight,
    read_phase_marker,
    write_inflight,
    write_phase_marker,
)
from mosaic.runlog import JsonlRunLog, run_log_path

UTC = datetime.timezone.utc


def instant(offset_seconds: float) -> datetime.datetime:
    """A fixed reference instant, shifted by *offset_seconds*."""
    return datetime.datetime(2026, 7, 27, 12, 0, tzinfo=UTC) + datetime.timedelta(
        seconds=offset_seconds
    )


def write_run_log(base_dir: Path, execution_id: str, *, terminal: str | None) -> None:
    """Emit a real run-log for *execution_id*, optionally ending it."""
    log = JsonlRunLog(run_log_path(base_dir, execution_id), execution_id)
    log.started(kind="trex", target="trex-track")
    if terminal == "finished":
        log.finished()
    elif terminal == "failed":
        log.failed("boom")
    elif terminal == "cancelled":
        log.cancelled()
    log.close()


# --- Phase markers --------------------------------------------------------


def test_phase_marker_round_trips(tmp_path: Path) -> None:
    marker = PhaseMarker(
        phase="convert",
        run_id="trex-abc",
        params_hash="hash-convert",
        execution_id="EXEC1",
        completed_at="2026-07-27T12:00:00+00:00",
        source="media_raw/seq/vid1.mp4",
        recorded_output="tracks_raw/trex/trex-abc/seq/vid1.pv",
    )
    write_phase_marker(tmp_path, marker)

    assert read_phase_marker(tmp_path, "convert") == marker
    assert read_phase_marker(tmp_path, "track") is None


def test_absent_marker_reads_as_none(tmp_path: Path) -> None:
    assert read_phase_marker(tmp_path, "convert") is None


def test_torn_marker_reads_as_none(tmp_path: Path) -> None:
    """A half-written file must not be mistaken for a completion record."""
    path = phase_marker_path(tmp_path, "convert")
    _ = path.write_text('{"phase": "conv')

    assert read_phase_marker(tmp_path, "convert") is None


def test_a_marker_that_is_not_text_reads_as_none(tmp_path: Path) -> None:
    """``read_text`` raises UnicodeDecodeError, which is a ValueError, not an OSError.

    A marker exists to make a run resumable; one that crashes the run instead
    is worse than one that is ignored.
    """
    _ = phase_marker_path(tmp_path, "convert").write_bytes(b"\xff\xfe\x00binary")
    _ = inflight_marker_path(tmp_path).write_bytes(b"\xff\xfe\x00binary")

    assert read_phase_marker(tmp_path, "convert") is None
    assert read_inflight(tmp_path) is None


def test_newer_schema_marker_reads_as_none(tmp_path: Path) -> None:
    """Reuse needs a completion contract this version understands."""
    path = phase_marker_path(tmp_path, "track")
    _ = path.write_text(
        json.dumps({"schema_version": MARKER_SCHEMA_VERSION + 1, "phase": "track"})
    )

    assert read_phase_marker(tmp_path, "track") is None


def test_unknown_fields_are_ignored(tmp_path: Path) -> None:
    """Forward compatibility within a schema version: extra keys are additive."""
    path = phase_marker_path(tmp_path, "track")
    _ = path.write_text(
        json.dumps(
            {
                "schema_version": MARKER_SCHEMA_VERSION,
                "phase": "track",
                "source": "a/b.mp4",
                "something_a_later_version_added": 7,
            }
        )
    )

    marker = read_phase_marker(tmp_path, "track")
    assert marker is not None
    assert marker.source == "a/b.mp4"


def test_marker_naming_the_other_phase_reads_as_none(tmp_path: Path) -> None:
    """A file whose body disagrees with its name is not a trustworthy record."""
    path = phase_marker_path(tmp_path, "track")
    _ = path.write_text(
        json.dumps({"schema_version": MARKER_SCHEMA_VERSION, "phase": "convert"})
    )

    assert read_phase_marker(tmp_path, "track") is None


def test_writes_leave_no_temp_sibling(tmp_path: Path) -> None:
    """An orphan temp beside a marker would be swept as an unknown file."""
    write_phase_marker(tmp_path, PhaseMarker(phase="convert"))
    write_inflight(tmp_path, InflightMarker(execution_id="EXEC1"))

    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [".mosaic-convert.json", INFLIGHT_MARKER_NAME]
    )


def test_clearing_markers(tmp_path: Path) -> None:
    write_phase_marker(tmp_path, PhaseMarker(phase="convert"))
    write_phase_marker(tmp_path, PhaseMarker(phase="track"))

    clear_phase_marker(tmp_path, "track")
    assert read_phase_marker(tmp_path, "convert") is not None
    assert read_phase_marker(tmp_path, "track") is None

    clear_phase_markers(tmp_path)
    assert read_phase_marker(tmp_path, "convert") is None

    clear_phase_markers(tmp_path)  # idempotent


# --- In-flight claims -----------------------------------------------------


def test_inflight_round_trips_and_clears(tmp_path: Path) -> None:
    marker = new_inflight(
        execution_id="EXEC1", host="box", pid=42, phase="track", timeout_seconds=600
    )
    write_inflight(tmp_path, marker)

    assert read_inflight(tmp_path) == marker

    clear_inflight(tmp_path)
    assert read_inflight(tmp_path) is None
    assert not inflight_marker_path(tmp_path).exists()

    clear_inflight(tmp_path)  # idempotent


def test_newer_schema_inflight_is_still_honoured(tmp_path: Path) -> None:
    """Unlike a phase marker: refusing to read a claim would steal a live directory."""
    _ = inflight_marker_path(tmp_path).write_text(
        json.dumps(
            {
                "schema_version": MARKER_SCHEMA_VERSION + 1,
                "execution_id": "OTHER",
                "expires_at": instant(3600).isoformat(),
            }
        )
    )

    marker = read_inflight(tmp_path)
    assert marker is not None
    assert marker.execution_id == "OTHER"


def test_expiry_adds_the_grace_to_the_phase_timeout() -> None:
    """The phase cannot outlive its own enforced timeout, so timeout+grace covers it."""
    long_phase = 6 * 3600
    stamp = inflight_expiry(long_phase, instant(0))

    assert datetime.datetime.fromisoformat(stamp) == instant(
        long_phase + INFLIGHT_GRACE_SECONDS
    )


def test_expiry_has_a_floor() -> None:
    """A short timeout must not produce a claim that expires during setup."""
    stamp = inflight_expiry(1, instant(0))

    assert datetime.datetime.fromisoformat(stamp) == instant(INFLIGHT_MIN_TTL_SECONDS)


# --- Liveness -------------------------------------------------------------


def test_no_marker_is_free(tmp_path: Path) -> None:
    assert (
        inflight_state(
            None, run_log_base=tmp_path, execution_id="EXEC1", now=instant(0)
        )
        == "free"
    )


def test_own_claim_is_mine(tmp_path: Path) -> None:
    """Re-entry within one execution is normal, not contention."""
    marker = InflightMarker(execution_id="EXEC1", expires_at=instant(-99).isoformat())

    assert (
        inflight_state(
            marker, run_log_base=tmp_path, execution_id="EXEC1", now=instant(0)
        )
        == "mine"
    )


@pytest.mark.parametrize("terminal", ["finished", "failed", "cancelled"])
def test_terminal_run_log_orphans_a_future_dated_claim(
    tmp_path: Path, terminal: str
) -> None:
    """The log can prove death, so it outranks an expiry that has not passed."""
    write_run_log(tmp_path, "OTHER", terminal=terminal)
    marker = InflightMarker(execution_id="OTHER", expires_at=instant(3600).isoformat())

    assert (
        inflight_state(
            marker, run_log_base=tmp_path, execution_id="EXEC1", now=instant(0)
        )
        == "orphaned"
    )


def test_running_run_log_leaves_the_claim_live(tmp_path: Path) -> None:
    write_run_log(tmp_path, "OTHER", terminal=None)
    marker = InflightMarker(execution_id="OTHER", expires_at=instant(3600).isoformat())

    assert (
        inflight_state(
            marker, run_log_base=tmp_path, execution_id="EXEC1", now=instant(0)
        )
        == "live"
    )


def test_absent_run_log_falls_through_to_the_expiry(tmp_path: Path) -> None:
    """An untracked run writes no log; silence must not read as death."""
    marker = InflightMarker(execution_id="OTHER", expires_at=instant(3600).isoformat())

    assert (
        inflight_state(
            marker, run_log_base=tmp_path, execution_id="EXEC1", now=instant(0)
        )
        == "live"
    )
    assert (
        inflight_state(
            marker, run_log_base=tmp_path, execution_id="EXEC1", now=instant(7200)
        )
        == "expired"
    )


def test_unparseable_expiry_is_expired(tmp_path: Path) -> None:
    """A claim that cannot say when it lapses cannot hold a directory forever."""
    marker = InflightMarker(execution_id="OTHER", expires_at="not-a-timestamp")

    assert (
        inflight_state(
            marker, run_log_base=tmp_path, execution_id="EXEC1", now=instant(0)
        )
        == "expired"
    )
