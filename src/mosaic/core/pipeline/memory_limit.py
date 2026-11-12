"""Reads the memory limit that applies to this process: a container's cgroup or the host's.

A process in a pod or a SLURM job sees the node's memory through ``psutil`` and is
killed at the cgroup's limit, which is lower. :func:`read_memory_limit` finds that
limit and how much of it is in use, so a floor and an admission check measure against
what the kernel enforces. A host without a cgroup limit reads as ``host``.

Imports the standard library only. ``psutil`` is imported inside the function, as
``markers`` does, and a host where it cannot be imported reports nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal

__all__ = [
    "LimitSource",
    "MemoryLimit",
    "read_memory_limit",
    "usable_memory_bytes",
]

type LimitSource = Literal["cgroup_v2", "cgroup_v1", "host"]

_DEFAULT_CGROUP_ROOT: Final = Path("/sys/fs/cgroup")
_DEFAULT_PROC_SELF_CGROUP: Final = Path("/proc/self/cgroup")


@dataclass(frozen=True, slots=True)
class MemoryLimit:
    """The memory limit that applies to this process, in bytes.

    Attributes:
        limit_bytes: The limit that applies: the smallest cgroup limit at or below
            the host's memory, else the host's physical memory.
        source: Where ``limit_bytes`` came from.
        in_use_bytes: The working set under the limit, the cgroup's memory less its
            inactive file cache, which is what a kubelet evicts by. For ``host`` it
            is the total less the available memory.
        peak_bytes: The limiting cgroup's high-water mark, 0 when the kernel does
            not report one.
        host_total_bytes: The host's physical memory.
        host_available_bytes: The memory the host can give a process without
            swapping.
    """

    limit_bytes: int
    source: LimitSource
    in_use_bytes: int
    peak_bytes: int
    host_total_bytes: int
    host_available_bytes: int

    @property
    def available_bytes(self) -> int:
        """The memory left under the limit, or the host's available memory for ``host``."""
        if self.source == "host":
            return self.host_available_bytes
        return max(0, self.limit_bytes - self.in_use_bytes)


@dataclass(frozen=True, slots=True)
class _Level:
    """One cgroup that has a limit, with the figures read beside it."""

    limit_bytes: int
    in_use_bytes: int
    peak_bytes: int


def _read_int(path: Path) -> int | None:
    """Return the integer in *path*, or ``None`` when it is missing, unreadable or not one."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_stat_value(path: Path, key: str) -> int:
    """Return the value of *key* in a ``memory.stat`` file, 0 when it cannot be read."""
    try:
        text = path.read_text()
    except OSError:
        return 0
    for line in text.splitlines():
        name, _, value = line.partition(" ")
        if name == key:
            try:
                return int(value.strip())
            except ValueError:
                return 0
    return 0


def _cgroup_paths(proc_self_cgroup: Path) -> tuple[str | None, str | None]:
    """Return this process's cgroup path for version 2 and for the version 1 memory controller."""
    try:
        text = proc_self_cgroup.read_text()
    except OSError:
        return None, None
    v2: str | None = None
    v1: str | None = None
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if hierarchy == "0" and controllers == "":
            v2 = path
        elif "memory" in controllers.split(","):
            v1 = path
    return v2, v1


def _levels(mount: Path, cgroup_path: str) -> list[Path]:
    """Return the directories from the process's cgroup up to *mount*, the process's first.

    A container whose cgroup namespace is not its own sees a host-relative path that
    does not exist under its mount, and then the mount itself is the only level.
    """
    parts = PurePosixPath(cgroup_path.strip()).parts
    parts = tuple(part for part in parts if part != "/")
    if ".." in parts:
        return [mount]
    deepest = mount.joinpath(*parts)
    if not deepest.is_dir():
        return [mount]
    return [mount.joinpath(*parts[:depth]) for depth in range(len(parts), -1, -1)]


def _level_v2(directory: Path) -> _Level | None:
    raw = directory / "memory.max"
    try:
        text = raw.read_text().strip()
    except OSError:
        return None
    if text == "max":
        return None
    try:
        limit = int(text)
    except ValueError:
        return None
    current = _read_int(directory / "memory.current")
    if current is None:
        return None
    inactive = _read_stat_value(directory / "memory.stat", "inactive_file")
    peak = _read_int(directory / "memory.peak") or 0
    return _Level(limit, max(0, current - inactive), peak)


def _level_v1(directory: Path) -> _Level | None:
    limit = _read_int(directory / "memory.limit_in_bytes")
    usage = _read_int(directory / "memory.usage_in_bytes")
    if limit is None or usage is None:
        return None
    inactive = _read_stat_value(directory / "memory.stat", "total_inactive_file")
    peak = _read_int(directory / "memory.max_usage_in_bytes") or 0
    return _Level(limit, max(0, usage - inactive), peak)


def _tightest(levels: list[_Level]) -> _Level | None:
    """Return the level with the smallest limit, the one nearest the process on a tie."""
    tightest: _Level | None = None
    for level in levels:
        if tightest is None or level.limit_bytes < tightest.limit_bytes:
            tightest = level
    return tightest


def read_memory_limit(
    *,
    cgroup_root: Path = _DEFAULT_CGROUP_ROOT,
    proc_self_cgroup: Path = _DEFAULT_PROC_SELF_CGROUP,
) -> MemoryLimit | None:
    """Return the memory limit that applies to this process now.

    The limit is the smallest of the ``memory.max`` values from the process's cgroup
    up to the root, so a job limit above a step's cgroup applies. A limit at or above
    the host's physical memory is no limit. An unreadable or malformed file at one
    level skips that level.

    Args:
        cgroup_root: Where the cgroup file system is mounted.
        proc_self_cgroup: The file that names this process's cgroup.

    Returns:
        The limit, or ``None`` where ``psutil`` is not importable.
    """
    try:
        import psutil
    except ImportError:
        return None
    host = psutil.virtual_memory()
    host_total = int(host.total)
    host_available = int(host.available)

    v2_path, v1_path = _cgroup_paths(proc_self_cgroup)
    candidates: list[tuple[_Level, LimitSource]] = []
    if v2_path is not None:
        found = _tightest(
            [
                level
                for directory in _levels(cgroup_root, v2_path)
                if (level := _level_v2(directory)) is not None
            ]
        )
        if found is not None:
            candidates.append((found, "cgroup_v2"))
    if v1_path is not None:
        found = _tightest(
            [
                level
                for directory in _levels(cgroup_root / "memory", v1_path)
                if (level := _level_v1(directory)) is not None
            ]
        )
        if found is not None:
            candidates.append((found, "cgroup_v1"))

    for level, source in sorted(candidates, key=lambda pair: pair[0].limit_bytes):
        if level.limit_bytes < host_total:
            return MemoryLimit(
                limit_bytes=level.limit_bytes,
                source=source,
                in_use_bytes=level.in_use_bytes,
                peak_bytes=level.peak_bytes,
                host_total_bytes=host_total,
                host_available_bytes=host_available,
            )
        break
    return MemoryLimit(
        limit_bytes=host_total,
        source="host",
        in_use_bytes=max(0, host_total - host_available),
        peak_bytes=0,
        host_total_bytes=host_total,
        host_available_bytes=host_available,
    )


def usable_memory_bytes() -> tuple[int, int]:
    """Return the memory limit that applies and the memory left under it, in bytes.

    Returns:
        ``(limit, available)``: the container's limit and what is left under it, or
        the host's total and available memory where no limit applies. ``(0, 0)``
        where ``psutil`` is not importable.
    """
    limit = read_memory_limit()
    if limit is None:
        return 0, 0
    return limit.limit_bytes, limit.available_bytes
