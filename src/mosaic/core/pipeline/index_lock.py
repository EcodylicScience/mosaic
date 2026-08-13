"""Serialization for index read-modify-writes.

Every index write in the toolkit is a full-file read-modify-write: read the
whole CSV, add or drop rows, write it back. ``atomic_write`` makes the write
half indivisible, so a reader never sees a partial file -- but it does nothing
about two writers whose *reads* interleave. Both read the same starting state,
both write a complete file, and the second silently erases the first's rows. No
error is raised on either side, and the loss is invisible until someone notices
a run missing from an index.

That is rule P7: every index write is serialized by a lock scoped to the file
being written.

**The guarantee, and its limit.** A POSIX advisory lock is per-inode on one
filesystem -- here, the sidecar's inode. It serializes any number of processes
and threads writing through one mount, which covers the real case -- parallel
feature workers and queue workers on a host. It does **not** reach across
machines: two hosts writing the same synced folder (a dataset shared between a
Mac and a Linux box, as they are today) each lock their own local copy, both
write, and the sync service produces conflicted copies with no error. So the
rule is a constraint rather than a promise: **many writers on one mount, or many
machines one at a time, never both.**

**Why a sidecar and not the index file itself.** The lock is held on an inode,
and ``atomic_write`` renames a **new** inode over the index path on every write.
Locking the index directly therefore cost two things, and the second is why this
module no longer does it.

The first is portability. Holding an open handle on the file being replaced is
legal on POSIX and refused under Windows' delete semantics -- which reach into
Linux through WSL's ``drvfs``, so every index write to a dataset under ``/mnt/*``
failed, and failed by naming the temp file ``atomic_write``'s own cleanup had
just removed. That whole class of "works on ext4, not on the share the data
actually lives on" goes away once the locked inode is never the renamed one.

The second is correctness, and it held on every platform. The first
``atomic_write`` inside a locked block silently dropped that block's grip: a
second process opened the new inode, flocked it uncontended, and interleaved
with the first -- both holding "the lock", one erasing the other. It was
contained by a convention -- at most one ``atomic_write`` per locked block, and
it is the last thing the block does -- which is a rule a reader has to know and
a reviewer has to enforce. A sidecar removes the hazard rather than documenting
it: **a locked block may rewrite its index as often as it likes and still holds
the lock at the end.**

The cost is one zero-byte ``<index>.lock`` beside each index, and it is the
right trade twice over. It is *visible* to anyone who can see the dataset, which
a lock in a temp directory is not; and it is keyed by the dataset's own path,
which a lock in a temp directory is not either -- SLURM sets ``TMPDIR`` per job,
a container gets its own ``/tmp``, and Windows' ``%TEMP%`` is per user, so a
temp-directory lock silently fails to serialize precisely the multi-writer cases
this module exists for. An adjacent sidecar cannot split that way: two writers
who can see the same index can see the same lock, by construction.

**The lock file is created once and never unlinked**, not on release and not by
a later tidying pass. Removing it reopens the hazard the sidecar was adopted to
close: a holder's inode is unlinked while it still holds the lock, a third
process creates a fresh one at the same name, flocks it uncontended, and the two
interleave. A zero-byte file per index is the price of the guarantee, not litter.

**Cross-platform, and now one strategy rather than two.** POSIX takes an
advisory ``fcntl.flock`` on the sidecar; Windows takes a mandatory
``msvcrt.locking`` on the same file. Windows' locks being mandatory no longer
matters, because the locked file is not one anybody reads -- readers open
``index.csv``, which is never locked and never held open. So the Windows branch
narrows to two three-line functions, and the ``%TEMP%`` keying it used to need,
with the per-user split that came with it, is gone.

Per file rather than per root, which is finer than P7's wording and never less
safe. Feature indexes are per feature (``features/<name>/index.csv``), so a
root-wide lock would serialize every feature worker in a dataset against every
other -- worse than the problem it solves.
"""

from __future__ import annotations

import errno
import os
import sys
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

# Platform lock backend. ``_lock_exclusive_nb`` / ``_release`` take and drop an
# exclusive lock on byte 0 of the fd ``_open_lock_fd`` returns. Both raise
# ``OSError`` when the lock is already held -- what the poll loop in ``_acquire``
# expects -- and both are released by the OS on process death, so there is never
# a stale lock to clean up. POSIX's flock is advisory and Windows'
# ``msvcrt.locking`` is mandatory; that difference used to force two strategies
# and no longer does, because neither is ever taken on a file something reads.
if sys.platform == "win32":
    import msvcrt

    def _lock_exclusive_nb(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _release(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock_exclusive_nb(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _release(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


LOCK_SUFFIX: Final = ".lock"


def lock_path_for(resolved: Path) -> Path:
    """The sidecar whose inode holds the lock for the index at *resolved*.

    Derived from the **resolved** path, not the path as written, so two callers
    reaching one index by different routes -- a symlinked root, a relative path,
    a ``..`` -- name one lock file rather than two that serialize against
    nothing. Public because a test pins it and because anything that ever has to
    recognise the file should ask rather than spell it.
    """
    return resolved.with_name(resolved.name + LOCK_SUFFIX)


def _open_lock_fd(resolved: Path) -> int:
    """Open (creating if absent) the sidecar for the index at *resolved*.

    Two side effects, both load-bearing.

    **The index is created empty if it is absent.** That was a by-product of
    locking the index inode with ``O_CREAT``, and three call sites are written
    around it: ``IndexCSV.append`` and ``mark_finished`` order ``ensure()`` or an
    existence check *before* acquiring because of it, ``load_media_index_frame``
    reads a zero-byte index as an empty frame because of it, and
    ``Dataset._read_media_index`` uses ``csv.DictReader`` rather than pandas
    inside a locked block for the same reason. Moving the lock off the index
    removed the mechanism, so the side effect is now deliberate and is kept.

    **The sidecar is created if absent and never removed** -- not here, not on
    release, not by a later cleanup. Unlinking a held lock file is the same
    lost-grip race the sidecar exists to close: the holder's inode goes away
    while it still holds it, the next process creates a fresh one at the same
    name and flocks it uncontended, and two writers proceed at once.
    """
    resolved.touch(exist_ok=True)
    return os.open(lock_path_for(resolved), os.O_RDWR | os.O_CREAT, 0o666)


# Default ceiling on how long a writer waits for its turn. Generous, because the
# operation being serialized is a CSV rewrite measured in milliseconds: reaching
# this means something is wedged, not merely busy.
DEFAULT_TIMEOUT_S = 60.0

_POLL_S = 0.01


# What the OS says when a lock is merely *held by someone else*. Everything in
# ``_UNSUPPORTED`` says the filesystem will not do locking at all. The split
# matters because the two look identical from a poll loop: an ``nolock`` NFS
# export or a FUSE mount refuses every attempt, so a loop that treats refusal as
# contention spins the whole timeout on every index write and then reports a
# phantom other writer.
_CONTENDED: Final[frozenset[int]] = frozenset(
    {errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES}
)

_UNSUPPORTED: Final[frozenset[int]] = frozenset(
    {errno.ENOLCK, errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOSYS, errno.EINVAL}
)


class IndexLockUnsupported(RuntimeError):
    """This filesystem will not lock, so a dataset on it cannot be written safely.

    Its own type rather than a timeout, because the remedy is different in kind:
    contention passes, and this does not. A caller can only move the dataset to a
    mount that locks, or accept that concurrent writers will silently drop each
    other's rows.

    **Never degraded into an unlocked write.** That is the lost update this
    module exists to prevent, and concluding "locking is unsupported" from an
    unfamiliar error and then proceeding would cause the corruption the check
    was added to detect. An errno this does not recognise keeps polling, which
    is the previous behaviour and the conservative one.
    """


class IndexLockTimeout(RuntimeError):
    """A writer could not acquire an index lock within its timeout.

    Raised rather than proceeding unlocked. An unlocked write is the lost-update
    this module exists to prevent, so degrading to one on contention would make
    the failure mode worse exactly when the system is under load.
    """


# Thread-level serialization, keyed by resolved path. The OS file lock is held
# per open file handle, so two threads in one process would each open their own
# fd and neither would block on the other -- the file lock alone is not
# thread-safe. RLock also makes nesting safe: `IndexCSV.append` calls `ensure`,
# and both take the lock.
_thread_locks: dict[Path, threading.RLock] = {}
_registry_guard = threading.Lock()

# Depth per (path, thread), so a nested acquisition reuses the held file lock
# instead of trying to take it twice on a second descriptor -- which would
# deadlock against itself with no error.
_depth: dict[tuple[Path, int], int] = {}


def _thread_lock_for(path: Path) -> threading.RLock:
    with _registry_guard:
        lock = _thread_locks.get(path)
        if lock is None:
            lock = threading.RLock()
            _thread_locks[path] = lock
        return lock


@contextmanager
def index_lock(path: Path, timeout: float = DEFAULT_TIMEOUT_S) -> Generator[None]:
    """Serialize a read-modify-write of the index at *path*.

    Re-entrant within a thread, so a locked method may call another. The lock is
    released on the way out even if the body raises, and by the OS if the
    process dies -- there is no stale lock to clean up.

    The lock is taken on a sidecar, ``<name>.lock`` in the same directory, which
    is created once and never removed. **A locked block may therefore perform
    any number of ``atomic_write``s to *path* and still hold the lock when it
    exits** -- the sidecar is never the inode a rename replaces. The old "one
    atomic_write, and it is the last thing the block does" convention is no
    longer load-bearing; the call sites that still follow it do so because
    holding this lock across an expensive read phase would serialize writers on
    work that is not the write.

    Args:
        path: The index file. **Created empty if absent** -- a by-product of the
            old design that callers depend on and that is now kept on purpose
            (see ``_open_lock_fd``). Callers that distinguish "missing" from
            "empty" must therefore check existence, or call ``ensure()``,
            *before* acquiring -- both ``IndexCSV.append`` and ``mark_finished``
            do, and getting that order wrong leaves a headerless file that reads
            back as ``EmptyDataError``.
        timeout: Seconds to wait before giving up.

    Raises:
        IndexLockTimeout: if the lock is not acquired within *timeout*.
    """
    resolved = path.resolve() if path.exists() else _resolved_parent(path)
    key = (resolved, threading.get_ident())

    with _thread_lock_for(resolved):
        if _depth.get(key, 0) > 0:
            # Already held by this thread further up the stack.
            _depth[key] += 1
            try:
                yield
            finally:
                _depth[key] -= 1
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        fd = _open_lock_fd(resolved)
        try:
            _acquire(fd, path, timeout)
            _depth[key] = 1
            try:
                yield
            finally:
                _depth.pop(key, None)
                _release(fd)
        finally:
            os.close(fd)


def _resolved_parent(path: Path) -> Path:
    """Resolve *path* through its parent, for a file that does not exist yet.

    ``Path.resolve()`` on a missing file still normalizes, but resolving the
    parent is what makes two callers reaching the same file by different routes
    (a symlinked root, a relative path) agree on one key.
    """
    return path.parent.resolve() / path.name


def _acquire(fd: int, path: Path, timeout: float) -> None:
    """Take an exclusive lock on *fd*, polling until *timeout*."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            _lock_exclusive_nb(fd)
            return
        except OSError as exc:
            if exc.errno in _UNSUPPORTED and exc.errno not in _CONTENDED:
                # Fail now rather than at the deadline. Waiting cannot help --
                # the mount refuses every attempt -- and the timeout message
                # would name a competing writer that does not exist, sending a
                # reader looking for a wedged process.
                raise IndexLockUnsupported(
                    f"the filesystem holding {path} does not support locking "
                    f"({errno.errorcode.get(exc.errno, exc.errno)}). mosaic "
                    f"refuses to write an index unlocked: concurrent writers "
                    f"silently drop each other's rows. Move the dataset to a "
                    f"mount with working advisory locks."
                ) from None
            if time.monotonic() >= deadline:
                raise IndexLockTimeout(
                    f"could not acquire the index lock for {path} within "
                    f"{timeout:g}s. Another writer is holding it, or a process "
                    f"holding it is wedged. Refusing to write unlocked: a "
                    f"concurrent read-modify-write silently drops the other "
                    f"writer's rows."
                ) from None
            time.sleep(_POLL_S)


def probe_lock_support(directory: Path) -> None:
    """Check that *directory* can hold an index lock. Raises if it cannot.

    For a caller that wants to know up front rather than at the first write --
    loading a dataset, starting a worker -- because the first write may be
    minutes of compute later and the failure is not one retrying fixes.

    Takes and releases a real lock on a throwaway sidecar rather than inspecting
    the mount: the filesystem type is not the question, and reading it would mean
    keeping a list of which ones lock. The probe file is removed, which is safe
    where removing a *live* index lock is not -- nothing else knows this name.

    Raises:
        IndexLockUnsupported: The filesystem refused to lock.
    """
    probe = directory / f".mosaic-lock-probe{LOCK_SUFFIX}"
    fd = os.open(probe, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        try:
            _lock_exclusive_nb(fd)
        except OSError as exc:
            if exc.errno in _UNSUPPORTED and exc.errno not in _CONTENDED:
                raise IndexLockUnsupported(
                    f"the filesystem holding {directory} does not support "
                    f"locking ({errno.errorcode.get(exc.errno, exc.errno)}). "
                    f"mosaic refuses to write an index unlocked: concurrent "
                    f"writers silently drop each other's rows."
                ) from None
            # Anything else means the probe met a real holder, which can only
            # happen if two probes race -- and proves locking works.
            return
        _release(fd)
    finally:
        os.close(fd)
        probe.unlink(missing_ok=True)
