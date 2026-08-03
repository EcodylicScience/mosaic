"""Turning a resolved media scope into the list of entries a tracker will run.

``Dataset.resolve_media_scope`` answers "which entries, and which file does each
resolve to" -- routing an analysis-required entry to its constant-rate derivative
rather than the defective original. What every tracker then does with that answer
is identical, and was written three times.

Two collapses happen here, and both are load-bearing rather than tidy-up:

* **Several videos under one entry** warn and use the first. Merging them is not
  yet a thing any tracker does, and silently tracking only part of a sequence
  would be worse than saying so.
* **Several cameras under one entry** collapse onto one work item. The working
  directory is keyed on ``(group, sequence)`` with no camera, so a multi-camera
  sequence's entries all resolve to one directory. Left as several, the second
  entry would see the first's source, call it a change, recompute over the first's
  outputs and replace its index row -- on every run, forever.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.core.helpers import make_entry_key

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset, ResolvedScopeEntry

__all__ = ["TrackerWorkItem", "build_work_items"]


@dataclass(frozen=True, slots=True)
class TrackerWorkItem:
    """One sequence to track, and what it resolved to.

    There is exactly one item per ``key``, which is the ``<group>__<sequence>``
    working-directory name.

    ``video_uid`` is the media index's content identity for the file. It is what
    the reuse gate compares first, because it answers "are these the same bytes"
    where a path can only answer "is this the same location" -- and a video
    replaced in place has the same location. It is empty for media indexed before
    the identity columns existed, which is why the path comparison stays as a
    fallback rather than being replaced.
    """

    group: str
    sequence: str
    key: str
    video_path: Path
    video_uid: str
    fps: float


def build_work_items(
    ds: Dataset,
    scope: list[ResolvedScopeEntry],
    *,
    kind: str,
    fps_default: float | None = None,
) -> list[TrackerWorkItem]:
    """Collapse a resolved media scope into one work item per entry.

    Args:
        ds: The dataset, read for its default frame rate when an entry's media
            index carries none.
        scope: What ``Dataset.resolve_media_scope`` returned.
        kind: The tracker's kind, used to prefix warnings so a message names the
            tool the user invoked rather than the shared machinery.
        fps_default: Frame rate for an entry whose facts carry none. Defaults to
            the dataset's ``fps_default``.
    """
    fallback_fps = (
        ds.meta_float("fps_default", 30.0) if fps_default is None else fps_default
    )
    items: list[TrackerWorkItem] = []
    claimed: set[str] = set()

    for entry in scope:
        group, sequence, resolved = entry.group, entry.sequence, entry.resolved
        paths = resolved.paths
        if len(paths) > 1:
            print(
                f"[{kind}] ({group}, {sequence}) has {len(paths)} videos; using "
                f"the first ({paths[0].name}). Multi-video sequences are not yet "
                f"merged.",
                file=sys.stderr,
            )
        key = make_entry_key(group, sequence)
        if key in claimed:
            print(
                f"[{kind}] ({group}, {sequence}) camera "
                f"{entry.camera or '<unnamed>'} shares one output directory with "
                f"an earlier camera; skipping it. Per-camera tracker output is a "
                f"later phase.",
                file=sys.stderr,
            )
            continue
        claimed.add(key)

        facts = resolved.facts
        items.append(
            TrackerWorkItem(
                group=group,
                sequence=sequence,
                key=key,
                video_path=paths[0],
                video_uid=facts[0].video_uuid if facts else "",
                fps=facts[0].fps if facts and facts[0].fps > 0 else fallback_fps,
            )
        )

    return items
