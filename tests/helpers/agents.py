"""A stand-in for the detached driver that an MCP submission starts.

:class:`RecordingStarter` is passed to ``LocalBackend(start_driver=...)``. It
records each request it is given and starts nothing, so a submission is
recorded and no step runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.graph import DetachedDriver, Request
    from mosaic.core.pipeline.parallelism import ParallelSettings

__all__ = ["RecordingStarter"]


@dataclass
class RecordingStarter:
    """A stand-in for the detached driver that records each request it is given.

    Attributes:
        started: The ids of the requests given, in order.
        parallel: The parallelism given with each request, in order.
    """

    started: list[str] = field(default_factory=list)
    parallel: list[ParallelSettings] = field(default_factory=list)

    def __call__(
        self, dataset: Dataset, request: Request, parallel: ParallelSettings
    ) -> DetachedDriver:
        from mosaic.core.pipeline.graph import DetachedDriver

        self.started.append(request.request_id)
        self.parallel.append(parallel)
        return DetachedDriver(
            request_id=request.request_id,
            pid=4242,
            log_path=dataset.base_dir / "driver.log",
        )
