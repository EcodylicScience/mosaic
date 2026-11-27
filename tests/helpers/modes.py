"""The ways a run computes its entries, and the check that a run computed in one."""

from __future__ import annotations

from typing import NamedTuple, TypedDict

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.parallelism import ParallelMode
from tests.helpers.runlog import latest_events


class RunModeKwargs(TypedDict):
    """The ``run_feature`` arguments that ask for one run mode."""

    parallel_workers: int
    parallel_mode: ParallelMode


class RunMode(NamedTuple):
    """How a run computes its entries: serially, in threads or in processes."""

    name: str
    workers: int
    kind: ParallelMode

    def run_kwargs(self) -> RunModeKwargs:
        """Return the ``run_feature`` arguments that ask for this mode."""
        return {"parallel_workers": self.workers, "parallel_mode": self.kind}


def run_modes(workers: int) -> tuple[RunMode, RunMode, RunMode]:
    """Return the serial, thread and process modes, the last two on *workers* workers."""
    return (
        RunMode("serial", 1, "thread"),
        RunMode("thread", workers, "thread"),
        RunMode("process", workers, "process"),
    )


def assert_ran_in_mode(
    ds: Dataset, mode: RunMode, *, entries: int | None = None
) -> None:
    """Assert that the last attempt computed each entry in the kind of *mode*.

    A cache hit writes an ``entry_end`` without a ``mode``, and a run that fell
    back to threads writes ``thread`` where processes were asked for. Either
    fails the check.

    Args:
        ds: The dataset the attempt ran in.
        mode: The mode the run asked for.
        entries: The number of entries the attempt computed. When given, the
            attempt must have ended exactly that many, so a missing entry fails
            where a run that ended none would pass.
    """
    ends = latest_events(ds, "entry_end")
    if entries is not None:
        assert len(ends) == entries, (mode.name, ends)
    assert [end.get("mode") for end in ends] == [mode.kind] * len(ends), mode.name
