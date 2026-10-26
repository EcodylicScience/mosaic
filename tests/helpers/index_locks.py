"""Whether another process can take an index's lock at this moment.

The probe runs in a child process because ``index_lock`` serializes the threads
of one process with an ``RLock`` and is re-entrant within a thread. A thread of
the test process would therefore block whether or not the file lock is held.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

__all__ = ["index_locked_out"]

_LOCK_PROBE = """
import sys
from pathlib import Path
from mosaic.core.pipeline.index_lock import IndexLockTimeout, index_lock

try:
    with index_lock(Path(sys.argv[1]), timeout=float(sys.argv[2])):
        pass
except IndexLockTimeout:
    sys.exit(0)   # still held by the parent
sys.exit(3)       # acquired: the parent's grip is gone
"""


def index_locked_out(index_path: Path, timeout: float = 1.0) -> bool:
    """Return whether a separate process fails to acquire *index_path*'s lock now.

    Args:
        index_path: The index whose lock is probed.
        timeout: How long the child waits for the lock before it reports it held.

    Raises:
        AssertionError: The child failed for a reason other than the lock.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _LOCK_PROBE, str(index_path), str(timeout)],
        capture_output=True,
        timeout=120,
        check=False,
    )
    if proc.returncode not in (0, 3):
        raise AssertionError(proc.stderr.decode()[-800:])
    return proc.returncode == 0
