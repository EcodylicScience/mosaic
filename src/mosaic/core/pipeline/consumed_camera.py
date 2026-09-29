"""Choose the camera of an entry that a per-entry consumer reads.

``Dataset.resolve_media_scope`` yields one entry per ``(group, sequence, camera)``,
while a tracker, an ``infer-*`` op and the ``tracks/`` layer they publish into all
address an entry by ``(group, sequence)`` alone. The reducer is in ``core``
because ``core`` may not import ``tracking``. Every consumer that must agree with a
tracker about the camera that it reads therefore calls this one implementation.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from mosaic.core.helpers import make_entry_key

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mosaic.core.dataset import ResolvedScopeEntry

__all__ = ["one_camera_per_entry"]


def one_camera_per_entry(
    kind: str, scope: Sequence[ResolvedScopeEntry], *, report_skipped: bool = True
) -> list[ResolvedScopeEntry]:
    """*scope* with a second camera of an entry dropped, and reported.

    ``Dataset.resolve_media_scope`` yields one entry per
    ``(group, sequence, camera)``. A per-entry working directory is keyed on
    ``(group, sequence)`` with no camera. Two cameras of one sequence therefore
    resolve to one directory. Left as two items, the second reads the first's
    source, records that as a change, recomputes over the first's outputs and
    replaces its index row, on every run. Dropping the second is what stops that,
    and the line on stderr is what stops it being invisible.

    Per-camera output needs the tracks layer to address a camera, and it does
    not. ``tracks_table_path`` names one parquet per
    ``(variant, group, sequence)``, the tracks index holds one row per
    ``(run_id, group, sequence)``, and no registered track schema declares a
    ``camera`` column.

    The kept camera is the one that the trackers and the ``infer-*`` ops read, and
    anything that must agree with them about that camera reduces here too. The
    rule was once written inline in both the tracker loop and the inference
    loop, and only the tracker's copy ran.

    Args:
        kind: The caller's op kind, prefixing each message so it names the op
            the user invoked rather than the shared machinery.
        scope: What ``Dataset.resolve_media_scope`` returned.
        report_skipped: Print the line for each camera dropped. A caller that
            reduces the same scope again later, as a plan does before its run,
            passes ``False`` so the line prints once.

    Returns:
        The entries to work on, in the order they arrived, one per
        ``(group, sequence)``.
    """
    claimed: set[str] = set()
    kept: list[ResolvedScopeEntry] = []
    for entry in scope:
        key = make_entry_key(entry.group, entry.sequence)
        if key in claimed:
            if report_skipped:
                print(
                    f"[{kind}] ({entry.group}, {entry.sequence}) camera "
                    f"{entry.camera or '<unnamed>'} shares one output directory "
                    f"with an earlier camera; skipping it.",
                    file=sys.stderr,
                )
            continue
        claimed.add(key)
        kept.append(entry)
    return kept
