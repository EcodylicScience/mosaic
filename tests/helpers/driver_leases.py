"""Driver leases written the way a running or a dead driver leaves them."""

from __future__ import annotations

import datetime
import os
import socket
from pathlib import Path

from mosaic.core.pipeline.graph import DriverLease, driver_lease_path


def write_driver_lease(
    base_dir: Path | str,
    request_id: str,
    *,
    stale: bool,
    pid: int = 0,
    step_id: str = "",
    step_pid: int = 0,
) -> DriverLease:
    """Write a lease for *request_id*, and return it.

    A stale lease was last refreshed an hour ago, and any other now. The lease
    names *pid*, or this process for 0. A lease refreshed now and naming an
    exited process is stale too, as a killed driver leaves it. *step_id* and
    *step_pid* name the step process that the lease's driver left running.
    """
    refreshed = datetime.datetime.now(datetime.UTC)
    if stale:
        refreshed -= datetime.timedelta(hours=1)
    lease = DriverLease(
        request_id=request_id,
        host=socket.gethostname(),
        pid=pid or os.getpid(),
        started_at=refreshed.isoformat(),
        refreshed_at=refreshed.isoformat(),
        step_id=step_id,
        step_pid=step_pid,
    )
    path = driver_lease_path(base_dir, request_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(lease.model_dump_json(indent=2), encoding="utf-8")
    return lease
