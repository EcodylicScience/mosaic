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
filesystem. It serializes any number of processes and threads writing through
one mount, which covers the real case -- parallel feature workers and queue
workers on a host. It does **not** reach across machines: two hosts writing the
same synced folder (a dataset shared between a Mac and a Linux box, as they are
today) each lock their own local copy, both write, and the sync service produces
conflicted copies with no error. So the rule is a constraint rather than a
promise: **many writers on one mount, or many machines one at a time, never
both.**

**Why the index file itself and not a sidecar.** A sidecar ``.lock`` is another
file in a root that users browse and that tests enumerate -- ``test_tracks_index``
asserts an index directory holds exactly ``index.csv``. Locking the index inode
directly keeps the guarantee without adding anything to disk.

**Cross-platform.** POSIX locks the index inode directly with an advisory
``fcntl.flock`` -- it survives the atomic rename and adds nothing to the
dataset. Windows locks are *mandatory* (they would block index readers) and an
open handle blocks the ``os.replace`` that ``atomic_write`` performs, so there
the index inode cannot be the lock. Windows instead locks a dedicated file in
the OS temp directory, keyed by the resolved index path so every caller and
process on the host agrees on it -- outside the dataset, so it blocks neither
readers nor the rename and the "exactly ``index.csv``" invariant still holds.

Per file rather than per root, which is finer than P7's wording and never less
safe. Feature indexes are per feature (``features/<name>/index.csv``), so a
root-wide lock would serialize every feature worker in a dataset against every
other -- worse than the problem it solves.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

# Platform lock backend. ``_open_lock_fd`` returns the fd whose byte 0 is locked
# for the index at *resolved*; ``_lock_exclusive_nb`` / ``_release`` take and
# drop the lock. Both backends raise ``OSError`` when the lock is already held
# (what the poll loop in ``_acquire`` expects) and are released by the OS on
# process death, so there is never a stale lock to clean up.
#
# POSIX locks the index inode itself: an advisory ``flock`` survives the atomic
# rename and leaves nothing on disk. Windows locks are mandatory and an open
# handle blocks ``os.replace``, so the index inode cannot be the lock there --
# it would block readers and the rename. Windows locks a dedicated file in the
# OS temp directory keyed by the resolved index path (outside the dataset, so
# ``test_tracks_index``'s "exactly index.csv" still holds), and still creates
# the empty index so the POSIX side effect callers rely on is preserved.
if sys.platform == "win32":
    import tempfile
    from hashlib import sha1

    import msvcrt

    _WIN_LOCK_DIR = Path(tempfile.gettempdir()) / "mosaic-index-locks"

    def _open_lock_fd(resolved: Path) -> int:
        resolved.touch(exist_ok=True)
        _WIN_LOCK_DIR.mkdir(parents=True, exist_ok=True)
        digest = sha1(str(resolved).encode("utf-8"), usedforsecurity=False).hexdigest()
        return os.open(_WIN_LOCK_DIR / f"{digest}.lock", os.O_RDWR | os.O_CREAT, 0o666)

    def _lock_exclusive_nb(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _release(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _open_lock_fd(resolved: Path) -> int:
        return os.open(resolved, os.O_RDWR | os.O_CREAT, 0o666)

    def _lock_exclusive_nb(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _release(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


# Default ceiling on how long a writer waits for its turn. Generous, because the
# operation being serialized is a CSV rewrite measured in milliseconds: reaching
# this means something is wedged, not merely busy.
DEFAULT_TIMEOUT_S = 60.0

_POLL_S = 0.01


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

    Args:
        path: The index file. **Created empty if absent**, because a lock needs
            an inode to hold. Callers that distinguish "missing" from "empty"
            must therefore check existence, or call ``ensure()``, *before*
            acquiring -- both ``IndexCSV.append`` and ``mark_finished`` do, and
            getting that order wrong leaves a headerless file that reads back as
            ``EmptyDataError``.
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
        except OSError:
            if time.monotonic() >= deadline:
                raise IndexLockTimeout(
                    f"could not acquire the index lock for {path} within "
                    f"{timeout:g}s. Another writer is holding it, or a process "
                    f"holding it is wedged. Refusing to write unlocked: a "
                    f"concurrent read-modify-write silently drops the other "
                    f"writer's rows."
                ) from None
            time.sleep(_POLL_S)
