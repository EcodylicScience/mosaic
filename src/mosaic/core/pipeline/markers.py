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

:class:`Phase` and :func:`phase_fields` carry the other sense of the word. They
mark nothing on disk: :class:`Phase` is ``Annotated`` metadata naming which
phases of a multi-phase op consume a parameter field, and it lives here because
it reuses the :data:`PhaseName` vocabulary the on-disk markers already own.
"""

from __future__ import annotations

import datetime
import functools
import os
import secrets
from pathlib import Path
from typing import Final, Literal, TypeVar, final

from pydantic import BaseModel, ConfigDict, ValidationError

from mosaic.runlog import TERMINAL_STATUSES, now_iso, read_run, run_log_dir

__all__ = [
    "BOOT_TIME_TOLERANCE_SECONDS",
    "INFLIGHT_GRACE_SECONDS",
    "INFLIGHT_MARKER_NAME",
    "INFLIGHT_MIN_TTL_SECONDS",
    "MARKER_SCHEMA_VERSION",
    "PROCESS_START_SLACK_SECONDS",
    "InflightMarker",
    "InflightState",
    "Phase",
    "PhaseMarker",
    "PhaseName",
    "boots_differ",
    "clear_inflight",
    "clear_phase_marker",
    "clear_phase_markers",
    "create_exclusive",
    "current_boot_id",
    "current_pid_namespace",
    "inflight_expiry",
    "inflight_marker_path",
    "inflight_state",
    "new_inflight",
    "parse_instant",
    "phase_fields",
    "phase_marker_path",
    "process_running",
    "process_running_since",
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


@final
class Phase:
    """Names which phases of a multi-phase op consume a parameter field.

    Goes in a tool-facing field's ``Annotated``, beside the field's ``Declared``
    and any pydantic ``Field``::

        cm_per_pixel: Annotated[float | None, Declared("..."), Phase("convert")]
        track_max_individuals: Annotated[
            int | None,
            Declared("the maximum number of simultaneous individuals to track"),
            Phase("convert", "track"),
        ]

    The first phase is a required positional parameter. A declaration that names
    no phase fails to type-check where it is written. A plain final class rather
    than a dataclass, because a dataclass cannot express varargs.

    Two markers naming the same phases are still two objects: instances compare
    by identity, and :func:`phase_fields` selects by ``isinstance`` and
    membership in :attr:`names` rather than by comparing markers.

    It declares no JSON Schema hook. :func:`phase_fields` reads it from
    ``FieldInfo.metadata`` to project a settings dictionary per phase, and a
    client groups a form by the layout it chooses instead of by an op's
    execution phases.

    Attributes:
        names: The phases that consume the field. Read-only.
    """

    __slots__ = ("_names",)

    def __init__(self, first: PhaseName, *rest: PhaseName) -> None:
        self._names: frozenset[PhaseName] = frozenset((first, *rest))

    @property
    def names(self) -> frozenset[PhaseName]:
        """The phases that consume the field."""
        return self._names

    def __repr__(self) -> str:
        named = ", ".join(repr(name) for name in sorted(self._names))
        return f"Phase({named})"


def phase_fields(model: type[BaseModel], phase: PhaseName) -> tuple[str, ...]:
    """Return the fields of *model* that *phase* consumes, in declaration order.

    Args:
        model: Parameter model to read.
        phase: Phase to select.

    Returns:
        Field names, in the order *model* declares them. A field that declares
        no :class:`Phase` is omitted.
    """
    return tuple(
        name
        for name, info in model.model_fields.items()
        if any(
            isinstance(marker, Phase) and phase in marker.names
            for marker in info.metadata
        )
    )


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

PROCESS_START_SLACK_SECONDS: Final = 2.0
"""How far a process's start time may lie after a stamp recorded once it began.

A process's start time is read from the kernel's clock, which on Linux is the
boot time plus clock ticks and drifts from the wall clock by up to a second.
"""


BOOT_TIME_TOLERANCE_SECONDS: Final = 120
"""How far two readings of one boot's start time may differ.

Off Linux a boot is named by its start time, which the system derives from the
wall clock and uptime, so a clock adjustment moves it.
"""


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
    # Imported in the call because its module loads pandas, and reading a marker
    # or a driver lease leaves pandas unloaded.
    from mosaic.core.pipeline._utils import atomic_write

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

    An exclusive create (:func:`create_exclusive`) rather than
    :func:`write_inflight`, whose ``os.replace`` cannot fail on an existing
    target: two executions reading a directory as free would both "claim" it,
    last writer winning. Single-mount only, the bound :mod:`index_lock`
    documents -- across a synced folder the marker's expiry and the run-log
    check in :func:`inflight_state` remain the cross-machine guard.
    """
    return create_exclusive(
        inflight_marker_path(work_dir), marker.model_dump_json(indent=2)
    )


def create_exclusive(path: Path, text: str) -> bool:
    """Create *path* holding *text*, or return ``False`` because it exists.

    *text* is written to a temporary file beside *path*, which is then
    hard-linked to *path*. A reader therefore finds *path* absent or whole, never
    created and still empty. Of two processes creating one path, exactly one
    succeeds. It is exclusive on one mount, the bound that
    :mod:`~mosaic.core.pipeline.index_lock` documents. The parent directories
    are created first.

    When the link fails because of anything but *path* existing, as on a
    filesystem without hard links, *path* is created with ``O_CREAT | O_EXCL``
    and filled after, and a reader may find it empty in between.

    A link reported as failing on an existing *path* succeeded when the
    temporary file has two links. NFS reports that for a retransmitted link
    whose first reply was lost.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}-{os.getpid()}-{secrets.token_hex(4)}.tmp")
    fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        _ = handle.write(text)
    try:
        os.link(temp, path)
    except FileExistsError:
        return os.stat(temp).st_nlink == 2
    except OSError:
        return _create_then_fill(path, text)
    finally:
        temp.unlink(missing_ok=True)
    return True


def _create_then_fill(path: Path, text: str) -> bool:
    """Create *path* with ``O_CREAT | O_EXCL``, then write *text* to it."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        _ = handle.write(text)
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


def parse_instant(stamp: str) -> datetime.datetime | None:
    """Return the ISO-8601 instant *stamp*, read as UTC when it names no zone.

    A stamp that does not parse returns ``None``.
    """
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

    deadline = parse_instant(marker.expires_at)
    if deadline is None:
        return "expired"
    moment = now or datetime.datetime.now(datetime.timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.timezone.utc)
    return "expired" if moment >= deadline else "live"


def process_running(pid: int) -> bool:
    """Return whether *pid* is a live process on this host.

    ``signal 0`` performs the existence and permission checks and delivers no
    signal. A process owned by another user raises ``PermissionError`` and is
    live. A pid of 0 or below names no process. A zombie has exited and is not
    live: a detached driver that ends, or is killed, stays a zombie of the
    long-lived process that started it until that process reaps it.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    import psutil

    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:
        return True


def process_running_since(pid: int, started_by: str) -> bool:
    """Return whether *pid* is a live process on this host that began by *started_by*.

    A process that began later reuses the pid of one that ended, as after a
    reboot. :data:`PROCESS_START_SLACK_SECONDS` is allowed between the process
    clock and the wall clock.

    Args:
        pid: The process.
        started_by: An ISO-8601 instant recorded after the process began. A
            stamp that does not parse leaves :func:`process_running` alone to
            decide.
    """
    if not process_running(pid):
        return False
    deadline = parse_instant(started_by)
    if deadline is None:
        return True
    import psutil

    try:
        began = psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:
        return True
    return began <= deadline.timestamp() + PROCESS_START_SLACK_SECONDS


@functools.cache
def current_boot_id() -> str:
    """Return a name for this host's current boot, ``""`` when none can be read.

    On Linux it is the kernel's random boot id, ``linux:<uuid>``. Elsewhere it is
    the boot's start time to the second, ``boot:<epoch seconds>``, which
    :func:`boots_differ` compares with :data:`BOOT_TIME_TOLERANCE_SECONDS`. A
    process records it in its run-log, so a reader on the same host can tell that
    the host rebooted since, and that the process is gone.
    """
    try:
        text = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii")
    except OSError:
        pass
    else:
        if text.strip():
            return f"linux:{text.strip()}"
    try:
        import psutil

        return f"boot:{int(psutil.boot_time())}"
    except Exception:
        return ""


def current_pid_namespace() -> str:
    """Return this process's pid namespace, ``""`` off Linux or when unreadable.

    Two containers can share a host name and a boot, and a pid recorded in one
    names nothing in the other, so a reader checks a pid only within its own
    namespace.
    """
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return ""


def boots_differ(recorded: str, current: str) -> bool:
    """Return whether two boot names (:func:`current_boot_id`) prove a reboot.

    Args:
        recorded: The boot that a run-log recorded.
        current: The boot of the host reading it.

    Returns:
        ``False`` when either name is empty or the two are of different kinds,
        because neither proves anything then. Two ``boot:`` names differ when
        their start times are more than :data:`BOOT_TIME_TOLERANCE_SECONDS`
        apart, and two other names when they are unequal.
    """
    if not recorded or not current:
        return False
    kind, _, value = recorded.partition(":")
    current_kind, _, current_value = current.partition(":")
    if kind != current_kind:
        return False
    if kind != "boot":
        return value != current_value
    try:
        gap = abs(int(value) - int(current_value))
    except ValueError:
        return False
    return gap > BOOT_TIME_TOLERANCE_SECONDS
