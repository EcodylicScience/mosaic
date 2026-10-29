"""Locate the global heavy lock, its holder, and the holder's record."""

import fcntl
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

from quiet_paths import read_quietly

HOLDER_RECORD = re.compile(r"cmd=(.*?)\s+pid=\d+\s+started_epoch=(\d+)")


class LockPaths(NamedTuple):
    lock_file: Path
    current_file: Path


class HolderRecord(NamedTuple):
    label: str
    started_epoch: int


def lock_directory(environment: Mapping[str, str], home: Path) -> Path:
    """Return HEAVY_TASK_DIR, or the default under `home` when unset or
    empty."""
    configured = environment.get("HEAVY_TASK_DIR")
    return Path(configured) if configured else home / ".claude" / "heavy-task"


def lock_paths(directory: Path) -> LockPaths:
    return LockPaths(directory / "lock", directory / "current")


def holder_record(label: str, pid: int, started_epoch: int) -> str:
    """Format the `current` file's content for the run that owns the lock."""
    return f"cmd={label} pid={pid} started_epoch={started_epoch}\n"


def read_holder(current_file: Path) -> str:
    """Return the `current` file's content, empty when it cannot be read."""
    return (read_quietly(current_file) or b"").decode("utf-8", "replace")


def parse_holder_record(text: str) -> HolderRecord | None:
    match = HOLDER_RECORD.search(text)
    if match is None:
        return None
    return HolderRecord(match.group(1).strip(), int(match.group(2)))


def lock_held_by_other(lock_file: Path) -> bool:
    """Return True iff some other process currently owns the global heavy lock.

    A non-blocking flock probe is the source of truth. The lock is bound to the
    holder's process, and the probe is accurate even when a hard-killed run
    left a stale `current` file behind. Any error degrades to False.
    """
    try:
        if not lock_file.exists():
            return False
        descriptor = os.open(lock_file, os.O_RDWR)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
