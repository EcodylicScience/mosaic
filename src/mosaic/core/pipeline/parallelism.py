"""Declares how many entries of a feature run are computed at once, and the limits on it.

The setting changes how fast a run finishes and how much memory it uses. It never
changes what the run produces, so it is not part of a recipe or a run id. It belongs
to the machine a step runs on, and the process that starts the step supplies it.

Imports pydantic, psutil and the standard library only. The graph, the command line
and the agent server all import it.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, ClassVar, Final, Literal

import psutil
from pydantic import ConfigDict, Field, field_validator

from mosaic.core.pipeline.error_kind import ClassifiedError, ErrorKind
from mosaic.core.strict_model import StrictModel

if TYPE_CHECKING:
    from mosaic.core.json_value import JsonValue
    from mosaic.core.pipeline.memory_limit import LimitSource

__all__ = [
    "DEFAULT_ADMISSION_MARGIN",
    "DEFAULT_MIN_FREE_MEMORY_FRACTION",
    "MAX_ADMISSION_MARGIN",
    "MAX_WORKERS_CEILING",
    "EntryOverMemoryLimit",
    "ParallelLimits",
    "ParallelMode",
    "ParallelRefused",
    "ParallelSettings",
    "MemoryFloorReached",
    "WorkerProcessDied",
    "default_max_workers",
    "effective_parallel",
    "host_cpus",
    "memory_use_text",
    "parallel_flags",
    "physical_cpus",
    "process_capable",
    "valid_admission_margin",
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
    whole interpreter and several copies of a table per worker. Choose it for a
    feature whose per-entry work is a Python loop, which holds the interpreter lock
    in threads.
"""

MAX_WORKERS_CEILING: Final = 32
"""The most workers any setting may name."""

MAX_ADMISSION_MARGIN: Final = 4.0
"""The largest admission margin. A larger factor means an estimate too wrong to admit by."""

DEFAULT_ADMISSION_MARGIN: Final = 1.5
"""The admission margin a caller that turns the gate on without a figure of its own uses."""


def valid_admission_margin(margin: float) -> bool:
    """Whether *margin* is an admission margin: 0 (off), or from 1 to 4."""
    return margin == 0.0 or 1.0 <= margin <= MAX_ADMISSION_MARGIN


DEFAULT_MIN_FREE_MEMORY_FRACTION: Final = 0.10
"""The memory floor of an operator who sets none."""


class ParallelSettings(StrictModel):
    """How one run computes its entries.

    Attributes:
        workers: How many entries are computed at once. 1 computes them in turn.
        mode: How the entries are computed once ``workers`` is above 1.
        min_free_memory_fraction: The share of the memory limit that applies, a
            container's or else the host's, that must stay available. A run with
            several workers waits for its in-flight entries before it starts
            another while the share is lower, and fails when it is still lower
            with none in flight. 0 turns the check off.
        admission_margin: The factor on an entry's predicted memory before the
            gate admits it beside the entries in flight. With a margin the first
            entry runs alone and entries start largest first. 0 turns the gate
            off; otherwise from 1 to 4.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    workers: int = Field(default=1, ge=1, le=MAX_WORKERS_CEILING)
    mode: ParallelMode = "thread"
    min_free_memory_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    admission_margin: float = Field(default=0.0, ge=0.0, le=MAX_ADMISSION_MARGIN)

    @field_validator("admission_margin")
    @classmethod
    def _off_or_from_one(cls, margin: float) -> float:
        if not valid_admission_margin(margin):
            msg = f"admission_margin is 0 (off) or from 1 to 4, not {margin}"
            raise ValueError(msg)
        return margin


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


def default_max_workers() -> int:
    """Return the worker limit of an operator who sets none: the physical cores."""
    return min(physical_cpus(), MAX_WORKERS_CEILING)


class ParallelLimits(StrictModel):
    """What the operator allows a caller to ask for.

    Attributes:
        max_workers: The most workers a request may name, and the count a request
            that names none receives. The default is the host's physical cores,
            at most :data:`MAX_WORKERS_CEILING`.
        allow_process_mode: Whether a request may name ``process`` mode. A request
            that names no mode receives ``thread`` whatever this is.
        min_free_memory_fraction: Applied to every request, as in
            :class:`ParallelSettings`. The default keeps a tenth of the host's
            memory available.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    max_workers: int = Field(
        default_factory=default_max_workers, ge=1, le=MAX_WORKERS_CEILING
    )
    allow_process_mode: bool = True
    min_free_memory_fraction: float = Field(
        default=DEFAULT_MIN_FREE_MEMORY_FRACTION, ge=0.0, lt=1.0
    )

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
        settings name more than one worker. The memory floor and the admission
        margin matter to a run with several workers only, so they travel with them.
        The margin is added only when it turns the gate on.
    """
    if settings is None or settings.workers <= 1:
        return []
    flags = [
        "--workers",
        str(settings.workers),
        "--parallel-mode",
        settings.mode,
        "--min-free-memory-fraction",
        str(settings.min_free_memory_fraction),
    ]
    if settings.admission_margin > 0.0:
        flags += ["--admission-margin", str(settings.admission_margin)]
    return flags


def host_cpus() -> int:
    """Return how many processors this process may use, at least 1."""
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return max(1, len(affinity(0)))
    return max(1, os.cpu_count() or 1)


def process_capable(feature: object) -> bool:
    """Whether *feature* may compute its entries in worker processes.

    It must be ``parallelizable`` and define no ``set_scope``: a process worker
    rebuilds the feature without calling it.
    """
    return bool(getattr(feature, "parallelizable", False)) and not hasattr(
        feature, "set_scope"
    )


def physical_cpus() -> int:
    """Return how many physical cores this process may use, at least 1.

    The processors of :func:`host_cpus` divided by the host's threads per core, so a
    host that runs two hardware threads per core counts each core once. Process
    workers of numeric code gain nothing from a second thread on a core. A host
    that does not report its physical cores counts every processor.
    """
    logical = psutil.cpu_count(logical=True)
    physical = psutil.cpu_count(logical=False)
    threads_per_core = 1 if not logical or not physical else max(1, logical // physical)
    return max(1, host_cpus() // threads_per_core)


def effective_parallel(
    settings: ParallelSettings | None,
    *,
    parallelizable: bool,
    process_capable: bool = True,
    cpus: int | None = None,
) -> ParallelSettings:
    """Return the settings a feature actually runs with.

    Args:
        settings: What the caller asked for. ``None`` asks for no parallelism.
        parallelizable: The feature's ``parallelizable`` attribute. A feature that
            is not parallelizable runs one entry at a time.
        process_capable: :func:`process_capable` of the feature. A feature that is
            not runs in threads.
        cpus: How many processors there are, :func:`host_cpus` when ``None``.

    Returns:
        The settings, with at most one worker per processor.
    """
    if settings is None or not parallelizable:
        return ParallelSettings()
    available = host_cpus() if cpus is None else max(1, cpus)
    mode: ParallelMode = settings.mode if process_capable else "thread"
    return ParallelSettings(
        workers=min(settings.workers, available),
        mode=mode,
        min_free_memory_fraction=settings.min_free_memory_fraction,
        admission_margin=settings.admission_margin,
    )


_GIGABYTE: Final = 1024**3


def memory_use_text(
    *,
    workers: int,
    mode: ParallelMode,
    peak_rss_bytes: int,
    limit_bytes: int,
) -> str:
    """Describe a run's workers and the memory it used, for a failure message.

    Args:
        workers: How many entries the run computed at once.
        mode: How it computed them.
        peak_rss_bytes: The largest resident memory measured, 0 when none was.
        limit_bytes: The memory limit that applied, a container's or the host's, 0
            when it was not measured.

    Returns:
        A clause such as ``workers=4, mode=process, peak memory 38.2 GB of 64.0 GB``.
    """
    clause = f"workers={workers}, mode={mode}"
    if peak_rss_bytes > 0 and limit_bytes > 0:
        used = peak_rss_bytes / _GIGABYTE
        limit = limit_bytes / _GIGABYTE
        clause += f", peak memory {used:.1f} GB of {limit:.1f} GB"
    return clause


_KEPT: Final = "Entries already written are kept."

_REMEDY: Final = f"Run again with fewer workers, or in thread mode. {_KEPT}"


class WorkerProcessDied(RuntimeError):
    """A process-mode worker ended without returning, so the pool cannot take more work.

    The usual cause is the host stopping the worker for lack of memory.
    """

    def __init__(self, use: str) -> None:
        msg = f"a worker process died ({use}). This is usually out of memory. {_REMEDY}"
        super().__init__(msg)


class MemoryFloorReached(RuntimeError):
    """The available memory is under the floor and no entry is in flight to free it."""

    def __init__(
        self,
        use: str,
        *,
        free_fraction: float,
        floor: float,
        limit_bytes: int,
        source: LimitSource,
    ) -> None:
        size = f"{limit_bytes / _GIGABYTE:.1f} GB"
        if source == "host":
            held = f"the host has {free_fraction:.0%} of its {size} memory available"
        else:
            held = (
                f"the container has {free_fraction:.0%} of its {size} limit available"
            )
        msg = (
            f"{held}, under the floor of {floor:.0%}, with no entry in flight "
            f"({use}). {_REMEDY}"
        )
        super().__init__(msg)


class EntryOverMemoryLimit(RuntimeError):
    """One entry is predicted to need more memory than the limit leaves, even alone.

    The prediction has no margin, and is compared with the limit less the memory
    floor, never with the memory available at the moment, so it is a property of
    the entry and the limit. In process mode the prediction includes the live
    workers' own memory, so the message offers thread mode as well.
    """

    def __init__(
        self,
        entry: str,
        *,
        predicted_bytes: int,
        limit_bytes: int,
        source: LimitSource,
        parallel_mode: ParallelMode = "thread",
    ) -> None:
        size = f"{limit_bytes / _GIGABYTE:.1f} GB"
        if source == "host":
            held = f"the host's {size} memory"
        else:
            held = f"the container's {size} limit"
        remedy = (
            "a larger memory limit, or in thread mode"
            if parallel_mode == "process"
            else "a larger memory limit"
        )
        msg = (
            f"entry {entry} is predicted to need "
            f"{predicted_bytes / _GIGABYTE:.1f} GB alone, more than {held} leaves "
            f"above the memory floor. Run again with {remedy}. {_KEPT}"
        )
        super().__init__(msg)
        self.entry: str = entry
        self.predicted_bytes: int = predicted_bytes
        self.limit_bytes: int = limit_bytes
