"""Shared, dependency-light helpers for tracking ops.

Factored out of ``ops/train.py`` so the training ops and the ``convert-points`` op
share one copy of the models-root guard and the copy-stable dataset fingerprint used
in content ``run_id`` computation. Behavior is identical to the original private
helpers -- training ``run_id``s are unchanged by the move.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.markers import (
    clear_inflight,
    inflight_state,
    new_inflight,
    read_inflight,
    try_create_inflight,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


def ensure_models_root(ds: Dataset) -> None:
    """Ensure the dataset has a ``models`` root (default ``models/``)."""
    if not ds.has_root("models"):
        ds.set_root("models", "models")


def fingerprint_dataset(path: Path) -> str:
    """Cheap, copy-stable digest of a training/converted dataset (file text + size listing).

    Uses relative paths + file sizes (not mtimes) so a copied/moved dataset with
    identical contents fingerprints identically -- keeping run_ids deterministic
    across machines.
    """
    path = Path(path)
    parts: dict[str, object] = {}
    if path.is_file():
        parts["file"] = path.name
        try:
            parts["text"] = path.read_text(errors="ignore")
        except Exception:
            parts["text"] = ""
        base = path.parent
    else:
        base = path
    listing: list[str] = []
    if base.exists():
        for f in sorted(base.rglob("*")):
            if f.is_file():
                try:
                    size = f.stat().st_size
                except OSError:
                    size = -1
                listing.append(f"{f.relative_to(base).as_posix()}:{size}")
    parts["listing"] = listing
    return hash_params(parts)


class RunRootHeld(RuntimeError):
    """Another execution is already producing this run, so this one must not."""


def claim_run_root(
    ds: Dataset, ctx: JobContext, run_root: Path, kind: str, idle_seconds: float
) -> None:
    """Take *run_root* exclusively for a one-shot op, or raise.

    Per-entry work skips a contended item so one sequence cannot end a batch; a
    one-shot op *is* the batch, and returning its run_id would hand back a model
    another execution is mid-write. Two executions of one identifier are not merely
    wasted: a nondeterministic trainer interleaves ``best.pt`` / ``last.pt`` /
    ``results.csv`` into one root, which is corrupt rather than slow.

    No ``finally`` release -- ``inflight_state`` reads a holder whose run-log went
    terminal as ``orphaned``, so a dead execution frees the root by itself.
    """
    marker = new_inflight(
        execution_id=ctx.execution_id,
        host=socket.gethostname(),
        pid=os.getpid(),
        phase=None,
        idle_seconds=idle_seconds,
    )
    for attempt in (0, 1):
        if try_create_inflight(run_root, marker):
            return
        held = read_inflight(run_root)
        state = inflight_state(
            held, run_log_base=ds.base_dir, execution_id=ctx.execution_id
        )
        if state == "mine":
            return
        if state in {"expired", "orphaned"} and attempt == 0:
            clear_inflight(run_root)
            continue
        where = f"{held.host}:{held.pid}" if held is not None else "another host"
        raise RunRootHeld(
            f"[{kind}] {run_root.name} is being produced by execution "
            f"{held.execution_id if held else '?'} on {where}; not training it again."
        )
