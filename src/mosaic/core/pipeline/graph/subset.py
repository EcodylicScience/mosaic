"""A run of a step over some of its planned entries, under an attempt of its own.

Two callers need one: a queue that measures a step on a few of its largest entries
before sizing the rest (a **probe**), and an array of attempts that splits a step's
entries between its indexes (a **named** subset). Both run under an execution id
other than the step's own, so the step's status, which reads only its own attempt,
is unchanged by them. The step's own attempt afterward computes what is left and
serves the subset's entries from the cache.

A subset is a choice of the machine running it, like the worker count: it never
enters a recipe, a request, a run id or ``params.json``.
"""

from __future__ import annotations

from typing import Annotated, Final, Literal

from pydantic import Field

from mosaic.core.entry import Entry

from .model import GraphModel

__all__ = [
    "MAX_PROBE_ENTRIES",
    "PROBE_THREAD_ENTRIES",
    "PROBE_WORKERS",
    "NamedSubset",
    "ProbeSubset",
    "StepSubset",
]

MAX_PROBE_ENTRIES: Final = 5
"""The most entries a probe computes.

Five is the least that measures levels 1 and 2 of both modes, since an entry is
computed once: the largest alone, the next two as a thread pair
(:data:`PROBE_WORKERS` workers), then two as a process pair. A probe of
:data:`PROBE_THREAD_ENTRIES` or fewer runs in threads only, and so does a probe of
four, whose fourth entry would load while the pair computes and run beside a third
entry in flight.
"""

PROBE_THREAD_ENTRIES: Final = 3
"""The entries a probe computes in threads: one alone and then one pair.

A feature that is not process-capable computes at most this many. For a
process-capable feature asked for :data:`MAX_PROBE_ENTRIES`, these are the largest
entries, and the rest run as a process pair.
"""

PROBE_WORKERS: Final = 2
"""How many entries a probe computes at once, in threads and again in processes.

The first thread entry runs alone, because the gate has no estimate yet, and the
next two run together once the gate admits them, so one probe measures a feature
alone and in a pair. The process pair has no gate: both entries are admitted at
once and the second loads right after the first, so the process phase also reports
a short level 1.
"""


class ProbeSubset(GraphModel):
    """The largest entries of the step's attempt, to measure the step by.

    Attributes:
        kind: ``probe``.
        count: How many of the attempt's entries to compute, largest by input
            rows first.
    """

    kind: Literal["probe"] = "probe"
    count: int = Field(ge=1, le=MAX_PROBE_ENTRIES)


class NamedSubset(GraphModel):
    """Entries of the step's target that a caller chose.

    Attributes:
        kind: ``named``.
        entries: The entries, each in the step's target.
    """

    kind: Literal["named"] = "named"
    entries: tuple[Entry, ...] = Field(min_length=1)


type StepSubset = Annotated[ProbeSubset | NamedSubset, Field(discriminator="kind")]
"""Which of a step's entries an attempt other than the step's own computes."""
