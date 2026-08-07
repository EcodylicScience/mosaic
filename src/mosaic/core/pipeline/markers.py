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

The expiry is still needed, for two reasons the run-log cannot cover. An
untracked run (``track=False``, the notebook path) has no run-log at all, and a
tracked one's run-log heartbeat is coarse. So the claim carries its own expiry,
**refreshed while the phase keeps producing output** (:func:`refresh_inflight`)
and derived from the caller's *inactivity* bound rather than a total-runtime one
-- the same liveness-from-activity rule mosaic-queue's reaper already applies. A
live multi-hour phase re-stamps its claim as it prints; a holder that dies stops
refreshing and the claim self-voids one idle window (plus grace) after its last
line. The sweeper never has to consult a queue lease interval -- mosaic does not
import mosaic-queue, and a run started from a notebook holds no lease.

**Storage constraint.** ``atomic_write`` is an ``os.replace`` and is atomic on
one filesystem. Two hosts writing one sync-service folder can therefore both
claim the same directory. These markers are safe for many writers on one
mount, or many machines one at a time, never both.
"""

from __future__ import annotations

import datetime
import os
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
    "refresh_inflight",
    "write_inflight",
    "write_phase_marker",
]

PhaseName = Literal["convert", "track", "infer"]
PHASE_NAMES: Final[tuple[PhaseName, ...]] = ("convert", "track", "infer")
"""Every gated phase any producer under ``_tracking`` can be in.

``infer`` joined when item 8.7 moved model inference under that root. It is not a
tracker phase and shares nothing with the other two beyond this protocol -- which
is the point: the sweeper reads markers, not producers, so a fourth kind of
output joins by writing one rather than by teaching the sweeper about itself.
"""

InflightState = Literal["free", "mine", "live", "expired", "orphaned"]

MARKER_SCHEMA_VERSION: Final = 1

# The prefix marks these files as mosaic's own: they sit among a tool's output
# (``*.pv``, ``*.settings``, ``data/*.npz``) and a future sweeper walks the
# directory generically, so ownership must be unambiguous.
MARKER_PREFIX: Final = ".mosaic-"
INFLIGHT_MARKER_NAME: Final = f"{MARKER_PREFIX}inflight.json"

# Added to the caller's inactivity window to get the claim's expiry: silence up
# to the idle bound is normal (the phase refreshes its claim on output), so the
# claim must outlast one idle window. The grace covers process teardown and
# clock skew between hosts on a shared mount.
INFLIGHT_GRACE_SECONDS: Final = 900
# Floor, so a caller who passes a very short idle bound does not produce a claim
# that expires while its own successor phase is still setting up, before its
# first output line refreshes it.
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


def try_create_inflight(work_dir: Path, marker: InflightMarker) -> bool:
    """Take the claim on *work_dir*, or ``False`` because someone else holds it.

    ``O_EXCL`` rather than :func:`write_inflight`, whose ``os.replace`` cannot fail
    on an existing target: two executions reading a directory as free would both
    "claim" it, last writer winning. Single-mount only, the bound
    :mod:`index_lock` documents -- across a synced folder the marker's expiry and
    the run-log check in :func:`inflight_state` remain the cross-machine guard.
    """
    path = inflight_marker_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        _ = handle.write(marker.model_dump_json(indent=2))
    return True


def clear_inflight(work_dir: Path, *, execution_id: str = "") -> None:
    """Release the claim on *work_dir*. Belongs in a ``finally``.

    *execution_id* makes the release ownership-checked. Without it the driver's
    unconditional ``finally`` unlinked whatever marker was present -- including a
    live peer's, on any path where this execution never claimed anything.
    """
    if execution_id:
        held = read_inflight(work_dir)
        if held is not None and held.execution_id != execution_id:
            return
    inflight_marker_path(work_dir).unlink(missing_ok=True)


def inflight_expiry(idle_seconds: float, now: datetime.datetime | None = None) -> str:
    """Return the ``expires_at`` for a claim refreshed every *idle_seconds*.

    *idle_seconds* is the caller's *inactivity* bound -- the longest the phase
    goes silent before it is declared hung and killed -- so a claim refreshed on
    output activity (:func:`refresh_inflight`) provably outlives any real gap
    between refreshes. The grace and the floor cover teardown, setup before the
    first output line, and modest clock skew.
    """
    moment = now or datetime.datetime.now(datetime.timezone.utc)
    ttl = max(idle_seconds + INFLIGHT_GRACE_SECONDS, INFLIGHT_MIN_TTL_SECONDS)
    return (moment + datetime.timedelta(seconds=ttl)).isoformat()


def new_inflight(
    *,
    execution_id: str,
    host: str,
    pid: int,
    phase: PhaseName | None,
    idle_seconds: float,
) -> InflightMarker:
    """Build a claim tolerating *idle_seconds* of silence between refreshes."""
    return InflightMarker(
        execution_id=execution_id,
        host=host,
        pid=pid,
        phase=phase,
        started_at=now_iso(),
        expires_at=inflight_expiry(idle_seconds),
    )


def refresh_inflight(
    work_dir: Path,
    marker: InflightMarker,
    idle_seconds: float,
    *,
    now: datetime.datetime | None = None,
) -> InflightMarker | None:
    """Re-stamp *marker*'s expiry while its phase is still producing output.

    Called on the activity signal (each line the tool prints), so a live
    multi-hour phase keeps its claim fresh; a holder that dies stops refreshing
    and the claim self-voids ``idle_seconds`` (plus grace) after its last line.
    Every identity field is preserved -- only ``expires_at`` moves.

    Absent is restored: something deleted our claim while the phase ran, and this
    callback is what puts it back. Foreign is left alone and reported as ``None``:
    a peer has taken the directory over, and re-stamping would steal it back.
    """
    held = read_inflight(work_dir)
    if held is not None and held.execution_id != marker.execution_id:
        return None
    refreshed = marker.model_copy(
        update={"expires_at": inflight_expiry(idle_seconds, now)}
    )
    write_inflight(work_dir, refreshed)
    return refreshed


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
