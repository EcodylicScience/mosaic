"""Completion and in-flight markers for a per-entry working directory.

A stage that writes its outputs into a per-entry directory needs to answer two
questions on re-entry: *did this work already finish*, and *is someone else
doing it right now*. Directory existence answers neither. TREx creates the
directory before the first subprocess starts, so a timeout, cancellation or
eviction leaves one behind that makes every later identical run a silent no-op
reporting success with zero individuals (implementation item 8.2).

These markers answer both, per phase:

* ``<work_dir>/.mosaic-<phase>.json`` -- written only after the phase returns
  normally, and deleted immediately before the phase is re-run. So the marker
  is present exactly when that phase completed with the inputs it records.
* ``<work_dir>/.mosaic-inflight.json`` -- written on entry, cleared in a
  ``finally``. Names the execution that holds the directory (item 8.3).

Three properties are load-bearing.

**Writes are atomic; they are not the provenance-sidecar idiom.**
``params.json`` and ``run_params.json`` are non-atomic ``write_text`` calls
inside a swallowing ``except`` -- correct, because they record provenance and
gate nothing. A marker *gates reuse*, so a torn read is a bogus skip and a
silently-dropped write is hours of needless recompute. Every write here goes
through :func:`~mosaic.core.pipeline._utils.atomic_write`.

**"Complete" means the phase returned normally, not that its outputs are
non-empty.** A legitimate tracking run over a video with no detected
individuals writes zero output files; an outputs-must-be-non-empty rule would
re-run it forever.

**Liveness comes from the run-log, plus an expiry the marker carries itself.**
:mod:`mosaic.runlog` already records ``execution_id`` / ``host`` /
``heartbeat_at`` for every tracked attempt, and mosaic-queue's reapers already
treat it as *the* liveness signal. A second liveness record that could disagree
with it, with no reconciliation rule, would be worse than the problem. So the
marker stores only the link -- the ``execution_id`` -- and
:func:`inflight_state` resolves it.

The expiry is still needed, for two reasons the run-log cannot cover. A TREx
phase is one multi-hour subprocess that writes no run-log record while it
runs, so run-log silence is not evidence of death for this kind of work; and
an untracked run (``track=False``, the notebook path) has no run-log at all.
The expiry is written by whoever created the marker and derived from that
caller's own enforced subprocess timeout, so the sweeper never has to consult
a queue lease interval -- mosaic does not import mosaic-queue, and a run
started from a notebook holds no lease.

**Storage constraint.** ``atomic_write`` is an ``os.replace`` and is atomic on
one filesystem. Two hosts writing one sync-service folder can therefore both
claim the same directory. These markers are safe for many writers on one
mount, or many machines one at a time, never both.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Final, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from mosaic.core.pipeline._utils import atomic_write
from mosaic.runlog import TERMINAL_STATUSES, now_iso, read_run, run_log_dir

__all__ = [
    "INFLIGHT_GRACE_SECONDS",
    "INFLIGHT_MARKER_NAME",
    "INFLIGHT_MIN_TTL_SECONDS",
    "MARKER_SCHEMA_VERSION",
    "InflightMarker",
    "InflightState",
    "PhaseMarker",
    "PhaseName",
    "clear_inflight",
    "clear_phase_marker",
    "clear_phase_markers",
    "inflight_expiry",
    "inflight_marker_path",
    "inflight_state",
    "new_inflight",
    "phase_marker_path",
    "read_inflight",
    "read_phase_marker",
    "write_inflight",
    "write_phase_marker",
]

PhaseName = Literal["convert", "track"]
PHASE_NAMES: Final[tuple[PhaseName, ...]] = ("convert", "track")

InflightState = Literal["free", "mine", "live", "expired", "orphaned"]

MARKER_SCHEMA_VERSION: Final = 1

# The prefix marks these files as mosaic's own: they sit among a tool's output
# (``*.pv``, ``*.settings``, ``data/*.npz``) and a future sweeper walks the
# directory generically, so ownership must be unambiguous.
MARKER_PREFIX: Final = ".mosaic-"
INFLIGHT_MARKER_NAME: Final = f"{MARKER_PREFIX}inflight.json"

# Added to the caller's own phase timeout to get the claim's expiry. The
# timeout is an enforced wall-clock kill, so the phase cannot outlive it; the
# grace covers process teardown and clock skew between hosts on a shared mount.
INFLIGHT_GRACE_SECONDS: Final = 900
# Floor, so a caller who passes a very short timeout does not produce a claim
# that expires while its own successor phase is still setting up.
INFLIGHT_MIN_TTL_SECONDS: Final = 1800


class PhaseMarker(BaseModel):
    """Proof that one phase of one entry completed, and on what.

    Attributes:
        schema_version: Marker format version.
        phase: Which phase completed.
        run_id: Run whose directory this is.
        params_hash: Digest of *this phase's* parameter subset. Today a run
            root already keys on every parameter, so this can only agree --
            it is recorded so a future cross-run conversion cache (item 8.5)
            inherits a correct gate rather than bolting one on.
        execution_id: Attempt that completed the phase.
        completed_at: ISO-8601 UTC timestamp.
        source: Dataset-root-relative source the phase consumed. Empty means
            *unknown*, not *mismatched* -- a marker written by the one-time
            adoption of a directory that predates markers cannot know it.
        source_uid: Content identity of that source, when the media index
            carries one. Advisory: recorded so items 4.2 / 8.5 have it, not
            yet compared.
        recorded_output: Dataset-root-relative primary output, when the phase
            has one whose location is not derivable from *work_dir*.
        backfilled: True when adopted from a pre-marker directory rather than
            written by a completing phase.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    schema_version: int = MARKER_SCHEMA_VERSION
    phase: PhaseName
    run_id: str = ""
    params_hash: str = ""
    execution_id: str = ""
    completed_at: str = ""
    source: str = ""
    source_uid: str = ""
    recorded_output: str = ""
    backfilled: bool = False


class InflightMarker(BaseModel):
    """A claim on a working directory by one execution.

    Attributes:
        schema_version: Marker format version.
        execution_id: Attempt holding the claim; the link into the run-log.
        host: Machine that wrote it, so a stale-claim message can name it.
        pid: Process that wrote it.
        phase: Phase in progress, when known.
        started_at: ISO-8601 UTC timestamp of the claim.
        expires_at: ISO-8601 UTC instant after which the claim is void. Written
            by whoever created it (see the module docstring).
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    schema_version: int = MARKER_SCHEMA_VERSION
    execution_id: str = ""
    host: str = ""
    pid: int = 0
    phase: PhaseName | None = None
    started_at: str = ""
    expires_at: str = ""


MarkerT = TypeVar("MarkerT", bound=BaseModel)


def _load(path: Path, model_cls: type[MarkerT]) -> MarkerT | None:
    """Read and validate a marker, or return None if it cannot be trusted.

    Every way a marker can be unreadable resolves to None, including a file
    that is not UTF-8 at all: ``read_text`` raises ``UnicodeDecodeError`` (a
    ``ValueError``, not an ``OSError``) for one, and a marker that crashes the
    run it was meant to make resumable is worse than one that is ignored.
    """
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError):
        return None
    try:
        return model_cls.model_validate_json(text)
    except ValidationError:
        return None


def _store(path: Path, marker: BaseModel) -> None:
    atomic_write(path, lambda tmp: tmp.write_text(marker.model_dump_json(indent=2)))


# --- Phase completion -----------------------------------------------------


def phase_marker_path(work_dir: Path, phase: PhaseName) -> Path:
    """Return the completion-marker path for *phase* within *work_dir*."""
    return work_dir / f"{MARKER_PREFIX}{phase}.json"


def read_phase_marker(work_dir: Path, phase: PhaseName) -> PhaseMarker | None:
    """Read *phase*'s completion marker, or None if absent or untrustworthy.

    A marker written by a **newer** schema reads as absent, so the phase
    recomputes. Refusing to reuse outputs whose completion contract this
    version does not understand is the conservative direction; the cost is one
    recompute, and the alternative is honouring a guarantee that may not mean
    what it says.
    """
    marker = _load(phase_marker_path(work_dir, phase), PhaseMarker)
    if marker is None or marker.schema_version > MARKER_SCHEMA_VERSION:
        return None
    if marker.phase != phase:
        return None
    return marker


def write_phase_marker(work_dir: Path, marker: PhaseMarker) -> None:
    """Write *marker*. Call only after the phase's outputs are on disk."""
    _store(phase_marker_path(work_dir, marker.phase), marker)


def clear_phase_marker(work_dir: Path, phase: PhaseName) -> None:
    """Remove *phase*'s marker. Call immediately before re-running the phase."""
    phase_marker_path(work_dir, phase).unlink(missing_ok=True)


def clear_phase_markers(work_dir: Path) -> None:
    """Remove every phase marker in *work_dir*."""
    for phase in PHASE_NAMES:
        clear_phase_marker(work_dir, phase)


# --- In-flight claims -----------------------------------------------------


def inflight_marker_path(work_dir: Path) -> Path:
    """Return the in-flight claim path within *work_dir*."""
    return work_dir / INFLIGHT_MARKER_NAME


def read_inflight(work_dir: Path) -> InflightMarker | None:
    """Read the in-flight claim, or None if absent or unparseable.

    Unlike :func:`read_phase_marker` this does **not** reject a newer schema
    version: the fields it needs are additive by contract, and treating a claim
    it does not fully understand as absent would steal a directory from a live
    run. Waiting out an expiry is the cheaper mistake.
    """
    return _load(inflight_marker_path(work_dir), InflightMarker)


def write_inflight(work_dir: Path, marker: InflightMarker) -> None:
    """Write the in-flight claim for *work_dir*."""
    _store(inflight_marker_path(work_dir), marker)


def clear_inflight(work_dir: Path) -> None:
    """Release the claim on *work_dir*. Belongs in a ``finally``."""
    inflight_marker_path(work_dir).unlink(missing_ok=True)


def inflight_expiry(
    timeout_seconds: float, now: datetime.datetime | None = None
) -> str:
    """Return the ``expires_at`` for a claim covering a *timeout_seconds* phase.

    *timeout_seconds* is the caller's own enforced subprocess wall-clock limit,
    so the phase provably cannot outlive it. The grace and the floor cover
    teardown, setup between phases, and modest clock skew.
    """
    moment = now or datetime.datetime.now(datetime.timezone.utc)
    ttl = max(timeout_seconds + INFLIGHT_GRACE_SECONDS, INFLIGHT_MIN_TTL_SECONDS)
    return (moment + datetime.timedelta(seconds=ttl)).isoformat()


def new_inflight(
    *,
    execution_id: str,
    host: str,
    pid: int,
    phase: PhaseName | None,
    timeout_seconds: float,
) -> InflightMarker:
    """Build a claim expiring after a *timeout_seconds* phase."""
    return InflightMarker(
        execution_id=execution_id,
        host=host,
        pid=pid,
        phase=phase,
        started_at=now_iso(),
        expires_at=inflight_expiry(timeout_seconds),
    )


def _parse_instant(stamp: str) -> datetime.datetime | None:
    try:
        moment = datetime.datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=datetime.timezone.utc)
    return moment


def inflight_state(
    marker: InflightMarker | None,
    *,
    run_log_base: Path,
    execution_id: str,
    now: datetime.datetime | None = None,
) -> InflightState:
    """Classify a claim: may this execution take the directory?

    ``free``, ``mine``, ``expired`` and ``orphaned`` all mean *take it*;
    ``live`` means someone else holds it.

    Resolution order, and why:

    1. No marker -- ``free``.
    2. The marker is ours -- ``mine``. Re-entry within one execution is normal
       (a retry, or two scope entries resolving to one directory) and must not
       be mistaken for contention.
    3. The run-log says the attempt reached a terminal status -- ``orphaned``.
       The log can *prove* death, and does so immediately, so it outranks the
       expiry.
    4. Past ``expires_at``, or the stamp is unparseable -- ``expired``.
    5. Otherwise ``live``.

    An **absent** run-log is deliberately not evidence of anything: a job may
    legitimately run untracked (``track=False``), and a run on another machine
    may not share ``.mosaic/runs``. Such a claim falls through to its expiry,
    which is the case the expiry exists for.

    Args:
        marker: The claim, as read by :func:`read_inflight`.
        run_log_base: Dataset base directory holding ``.mosaic/runs``.
        execution_id: The execution asking.
        now: Override for the current instant, for tests.

    Returns:
        The claim's state with respect to *execution_id*.
    """
    if marker is None:
        return "free"
    if marker.execution_id and marker.execution_id == execution_id:
        return "mine"

    if marker.execution_id:
        snapshot = read_run(run_log_dir(run_log_base), marker.execution_id)
        if snapshot is not None and snapshot["status"] in TERMINAL_STATUSES:
            return "orphaned"

    deadline = _parse_instant(marker.expires_at)
    if deadline is None:
        return "expired"
    moment = now or datetime.datetime.now(datetime.timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.timezone.utc)
    return "expired" if moment >= deadline else "live"
