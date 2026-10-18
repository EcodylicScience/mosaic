"""Declares how many entries of a feature run are computed at once, and the limits on it.

The setting changes how fast a run finishes and how much memory it uses. It never
changes what the run produces, so it is not part of a recipe or a run id. It belongs
to the machine a step runs on, and the process that starts the step supplies it.

Imports pydantic and the standard library only. The graph, the command line and the
agent server all import it.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, ClassVar, Final, Literal

from pydantic import ConfigDict, Field

from mosaic.core.pipeline.error_kind import ClassifiedError, ErrorKind
from mosaic.core.strict_model import StrictModel

if TYPE_CHECKING:
    from mosaic.core.json_value import JsonValue

__all__ = [
    "MAX_WORKERS_CEILING",
    "ParallelLimits",
    "ParallelMode",
    "ParallelRefused",
    "ParallelSettings",
    "MemoryFloorReached",
    "WorkerProcessDied",
    "effective_parallel",
    "host_cpus",
    "memory_use_text",
    "parallel_flags",
]

type ParallelMode = Literal["thread", "process"]
"""How parallel entries are computed.

``thread``
    Threads share one feature instance and one copy of the imported libraries. Memory
    grows by about one in-flight table per worker. Work that holds the interpreter
    lock gains little.
``process``
    Each worker is a spawned interpreter that imports the feature stack and receives
    a pickled copy of each entry's table. It uses every core, and memory grows by a
    whole interpreter and several copies of a table per worker.
"""

MAX_WORKERS_CEILING: Final = 32
"""The most workers any setting may name."""


class ParallelSettings(StrictModel):
    """How one run computes its entries.

    Attributes:
        workers: How many entries are computed at once. 1 computes them in turn.
        mode: How the entries are computed once ``workers`` is above 1.
        min_free_memory_fraction: The share of the host's memory that must stay
            available. A run with several workers waits for its in-flight entries
            before it starts another while the share is lower, and fails when it
            is still lower with none in flight. 0 turns the check off.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    workers: int = Field(default=1, ge=1, le=MAX_WORKERS_CEILING)
    mode: ParallelMode = "thread"
    min_free_memory_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)


class ParallelRefused(ClassifiedError, ValueError):
    """A request for more parallelism than the operator allows."""

    error_kind: ClassVar[ErrorKind] = "unavailable"

    def __init__(
        self, message: str, *, limit: Literal["max_workers", "allow_process_mode"]
    ) -> None:
        super().__init__(message)
        self.limit: Literal["max_workers", "allow_process_mode"] = limit

    def error_reason(self) -> str | None:
        return "parallel_limit"

    def error_detail(self) -> dict[str, JsonValue] | None:
        return {"limit": self.limit}


class ParallelLimits(StrictModel):
    """What the operator allows a caller to ask for.

    Attributes:
        max_workers: The most workers a request may name, and the count a request
            that names none receives.
        allow_process_mode: Whether a request may name ``process`` mode. A request
            that names no mode receives ``thread`` whatever this is.
        min_free_memory_fraction: Applied to every request, as in
            :class:`ParallelSettings`.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    max_workers: int = Field(default=1, ge=1, le=MAX_WORKERS_CEILING)
    allow_process_mode: bool = False
    min_free_memory_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)

    def settings_for(
        self, workers: int | None = None, mode: ParallelMode | None = None
    ) -> ParallelSettings:
        """Return the settings a request receives, or refuse a request over a limit.

        Args:
            workers: The count the caller asks for, ``max_workers`` when ``None``.
            mode: The mode the caller asks for, ``thread`` when ``None``.

        Returns:
            The settings, within the limits.

        Raises:
            ParallelRefused: ``workers`` is above ``max_workers``, or ``mode`` is
                ``process`` and the operator does not allow it.
        """
        count = self.max_workers if workers is None else workers
        if count > self.max_workers:
            msg = (
                f"workers {count} is above this server's limit of {self.max_workers}. "
                "Ask for no more than the limit, or report this to the user."
            )
            raise ParallelRefused(msg, limit="max_workers")
        chosen: ParallelMode = "thread" if mode is None else mode
        if chosen == "process" and not self.allow_process_mode:
            msg = (
                "parallel_mode 'process' is not allowed on this server. The operator "
                "sets allow_process_mode in its configuration. Use 'thread', or "
                "report this to the user."
            )
            raise ParallelRefused(msg, limit="allow_process_mode")
        return ParallelSettings(
            workers=count,
            mode=chosen,
            min_free_memory_fraction=self.min_free_memory_fraction,
        )


def parallel_flags(settings: ParallelSettings | None) -> list[str]:
    """Return the command-line flags that carry *settings* to a ``mosaic`` process.

    Args:
        settings: The settings, or ``None`` for none.

    Returns:
        The flags of ``mosaic run`` and ``mosaic pipeline run``, empty unless the
        settings name more than one worker. The memory floor matters to a run with
        several workers only, so it travels with them.
    """
    if settings is None or settings.workers <= 1:
        return []
    return [
        "--workers",
        str(settings.workers),
        "--parallel-mode",
        settings.mode,
        "--min-free-memory-fraction",
        str(settings.min_free_memory_fraction),
    ]


def host_cpus() -> int:
    """Return how many processors this process may use, at least 1."""
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return max(1, len(affinity(0)))
    return max(1, os.cpu_count() or 1)


def effective_parallel(
    settings: ParallelSettings | None,
    *,
    parallelizable: bool,
    needs_scope: bool = False,
    cpus: int | None = None,
) -> ParallelSettings:
    """Return the settings a feature actually runs with.

    Args:
        settings: What the caller asked for. ``None`` asks for no parallelism.
        parallelizable: The feature's ``parallelizable`` attribute. A feature that
            is not parallelizable runs one entry at a time.
        needs_scope: Whether the feature defines ``set_scope``. A process worker
            rebuilds the feature without calling it, so such a feature runs in
            threads.
        cpus: How many processors there are, :func:`host_cpus` when ``None``.

    Returns:
        The settings, with at most one worker per processor.
    """
    if settings is None or not parallelizable:
        return ParallelSettings()
    available = host_cpus() if cpus is None else max(1, cpus)
    mode: ParallelMode = "thread" if needs_scope else settings.mode
    return ParallelSettings(
        workers=min(settings.workers, available),
        mode=mode,
        min_free_memory_fraction=settings.min_free_memory_fraction,
    )


_GIGABYTE: Final = 1024**3


def memory_use_text(
    *,
    workers: int,
    mode: ParallelMode,
    peak_rss_bytes: int,
    total_bytes: int,
) -> str:
    """Describe a run's workers and the memory it used, for a failure message.

    Args:
        workers: How many entries the run computed at once.
        mode: How it computed them.
        peak_rss_bytes: The largest resident memory measured, 0 when none was.
        total_bytes: The host's physical memory, 0 when it was not measured.

    Returns:
        A clause such as ``workers=4, mode=process, peak memory 38.2 GB of 64.0 GB``.
    """
    clause = f"workers={workers}, mode={mode}"
    if peak_rss_bytes > 0 and total_bytes > 0:
        used = peak_rss_bytes / _GIGABYTE
        total = total_bytes / _GIGABYTE
        clause += f", peak memory {used:.1f} GB of {total:.1f} GB"
    return clause


_REMEDY: Final = (
    "Run again with fewer workers, or in thread mode. Entries already written are kept."
)


class WorkerProcessDied(RuntimeError):
    """A process-mode worker ended without returning, so the pool cannot take more work.

    The usual cause is the host stopping the worker for lack of memory.
    """

    def __init__(self, use: str) -> None:
        msg = f"a worker process died ({use}). This is usually out of memory. {_REMEDY}"
        super().__init__(msg)


class MemoryFloorReached(RuntimeError):
    """The host's available memory is under the floor and no entry is in flight to free it."""

    def __init__(self, use: str, *, free_fraction: float, floor: float) -> None:
        msg = (
            f"the host has {free_fraction:.0%} of its memory available, under the "
            f"floor of {floor:.0%}, with no entry in flight ({use}). {_REMEDY}"
        )
        super().__init__(msg)
