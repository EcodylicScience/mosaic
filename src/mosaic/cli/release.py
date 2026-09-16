"""``mosaic release``: free a run root whose claim outlived the run that took it.

An op claims its run root with an exclusive create and never releases it in a
``finally``: a holder whose run-log has gone terminal reads as ``orphaned``, so
a finished or cancelled execution frees the directory by itself. That covers
every ordinary ending, including the one ``mosaic cancel`` now records for a
process that had already exited.

What it does not cover is a claim with **no terminal record to find**: an
execution that ran untracked, or whose run-log never reached the dataset. There
``inflight_state`` can only wait out the marker's own expiry, which is half an
hour after the claim was last refreshed, and until then the next attempt is
refused with ``RunRootHeld``. Deleting the marker by hand was the only way out.

**This never overrides a live run.** A claim held by another host is refused,
because deciding it from here means guessing about a machine this process
cannot see, and a claim whose process is still running is refused unless the
caller says otherwise.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail, log

if TYPE_CHECKING:
    from mosaic.core.pipeline.markers import InflightMarker

_SKIPPED_ROOTS: frozenset[str] = frozenset({"media", "media_raw", "labels_raw"})
"""Directories a claim never sits under, pruned from the walk.

A run root is always a derived directory. The media roots are where a dataset's
bulk lives, and walking them to look for a marker that cannot be there is most
of what this command would otherwise cost.
"""


def release_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    execution_id: Annotated[
        str, typer.Option("--execution-id", help="Attempt ULID whose claim to release.")
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Release even though the claiming process is still running here.",
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the result as JSON on stdout.")
    ] = False,
) -> None:
    """Release the run roots claimed by an attempt that is no longer running."""
    from mosaic.core.pipeline.markers import clear_inflight, read_inflight

    ds = load_dataset(manifest)
    held = [
        (root, marker)
        for root, marker in _claims(ds.base_dir, read_inflight)
        if marker.execution_id == execution_id
    ]
    if not held:
        _emit(
            {"execution_id": execution_id, "released": []},
            as_json,
            f"[mosaic] no run root is claimed by {execution_id}.",
        )
        return

    here = socket.gethostname()
    for root, marker in held:
        if marker.host and marker.host != here:
            fail(
                f"{root.name} is claimed from host {marker.host!r}; releasing it "
                "from here would take a directory a run on that machine may "
                "still be writing."
            )
        if not force and _is_running(marker.pid):
            fail(
                f"{root.name} is claimed by pid {marker.pid}, which is still "
                "running here. Cancel it first with 'mosaic cancel "
                f"--execution-id {execution_id}', or pass --force."
            )

    for root, _ in held:
        clear_inflight(root, execution_id=execution_id)

    released = [str(root) for root, _ in held]
    _emit(
        {"execution_id": execution_id, "released": released},
        as_json,
        f"[mosaic] released {len(released)} run root(s) claimed by {execution_id}.",
    )


def _claims(
    base_dir: Path, read: Callable[[Path], "InflightMarker | None"]
) -> list[tuple[Path, "InflightMarker"]]:
    """Every in-flight claim under *base_dir*, as ``(run root, marker)``."""
    from mosaic.core.pipeline.markers import INFLIGHT_MARKER_NAME

    found: list[tuple[Path, InflightMarker]] = []
    for child in sorted(base_dir.iterdir()) if base_dir.is_dir() else []:
        if not child.is_dir() or child.name in _SKIPPED_ROOTS:
            continue
        for path in sorted(child.rglob(INFLIGHT_MARKER_NAME)):
            marker = read(path.parent)
            if marker is not None:
                found.append((path.parent, marker))
    return found


def _is_running(pid: int) -> bool:
    """Is *pid* a live process on this host?

    ``signal 0`` performs the permission and existence checks and delivers
    nothing, which is exactly the question. A process owned by someone else
    answers ``PermissionError``, and that is still a live process.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _emit(payload: dict[str, object], as_json: bool, human: str) -> None:
    if as_json:
        emit_json(payload)
    else:
        log(human)
