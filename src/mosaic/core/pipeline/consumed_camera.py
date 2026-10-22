"""Resolve each entry's cameras, and choose the one that a per-entry consumer reads.

``Dataset.resolve_media_scope`` yields one entry per ``(group, sequence, camera)``,
while a tracker, an ``infer-*`` op and the ``tracks/`` layer they publish into all
address an entry by ``(group, sequence)`` alone. The reducer is in ``core``
because ``core`` may not import ``tracking``. Every consumer that must agree with a
tracker about the camera that it reads therefore calls this one implementation.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from mosaic.core.helpers import is_nameless_entry, make_entry_key

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from mosaic_media import MediaProbeError

    from mosaic.core.dataset import Dataset, ResolvedScopeEntry
    from mosaic.core.entry import Entry

__all__ = ["one_camera_per_entry", "resolve_entry_media"]


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


def resolve_entry_media(
    ds: Dataset, entries: Sequence[Entry] | None
) -> dict[Entry, list[ResolvedScopeEntry] | FileNotFoundError | MediaProbeError]:
    """Return each entry's cameras, or the error that explains why it has none.

    One read of the media index resolves the entries. An entry whose media needs
    a transcode that has not run fails alone.

    Args:
        ds: The dataset.
        entries: The entries, each keyed in the result in the order given.
            Without a media index each fails with the one error that the read
            raised. An entry with neither group nor sequence resolves under its
            first file's stem, a name that the scope's read cannot key back to
            the entry. Each such entry is read separately.
            ``None`` resolves every indexed entry, as an unscoped run reads them,
            and keys each by the label that ``Dataset.resolve_media_scope`` gives
            it, in sorted order. An entry that fails to resolve is keyed as
            indexed, which differs from its label only for a nameless entry.
            Without a media index the result is empty.

    Returns:
        Per entry, its cameras in ``(group, sequence, camera)`` order, or the
        ``FileNotFoundError`` or ``MediaProbeError`` that resolving it raised.
    """
    failed: dict[Entry, MediaProbeError] = {}
    if entries is None:
        try:
            scope = ds.resolve_media_scope(None, errors=failed)
        except FileNotFoundError:
            return {}
        everything = {**_cameras_by_entry(scope), **failed}
        return {entry: everything[entry] for entry in sorted(everything)}
    nameless = [entry for entry in entries if is_nameless_entry(*entry)]
    named = [entry for entry in entries if not is_nameless_entry(*entry)]
    try:
        cameras = _cameras_by_entry(ds.resolve_media_scope(named, errors=failed))
        for entry in nameless:
            alone = ds.resolve_media_scope([entry], errors=failed)
            if alone:
                cameras[entry] = alone
    except FileNotFoundError as exc:
        return {entry: exc for entry in entries}
    resolved: dict[
        Entry, list[ResolvedScopeEntry] | FileNotFoundError | MediaProbeError
    ] = {}
    for entry in entries:
        if entry in cameras:
            resolved[entry] = cameras[entry]
        elif entry in failed:
            resolved[entry] = failed[entry]
        else:
            message = f"{make_entry_key(*entry)}: no media is indexed for it"
            resolved[entry] = FileNotFoundError(message)
    return resolved


def _cameras_by_entry(
    scope: Iterable[ResolvedScopeEntry],
) -> dict[Entry, list[ResolvedScopeEntry]]:
    """Group a resolved scope's cameras under their ``(group, sequence)`` label."""
    cameras: dict[Entry, list[ResolvedScopeEntry]] = {}
    for media in scope:
        cameras.setdefault((media.group, media.sequence), []).append(media)
    return cameras
