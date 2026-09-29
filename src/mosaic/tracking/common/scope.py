"""Turning a resolved media scope into the list of entries a tracker will run.

``Dataset.resolve_media_scope`` answers "which entries, and which file does each
resolve to" -- routing an analysis-required entry to its constant-rate derivative
rather than the defective original. What every tracker then does with that answer
is identical, and was written three times.

Two collapses happen here, and both are load-bearing rather than tidy-up:

* **Several videos under one entry** are handed over whole to a tracker that
  declares ``joins_sources``, and **truncated to the first** for one that does
  not. A recorder that chops a session into clips leaves a boundary that is a
  filesystem artifact, not an event, so a tool able to read the clips as one
  video should not be made to see only the first of them. A tool that cannot is
  told, on stderr, that the rest were dropped -- silently tracking part of a
  sequence would be worse than saying so.
* **Several cameras under one entry** collapse onto one work item. The working
  directory is keyed on ``(group, sequence)`` with no camera, so a multi-camera
  sequence's entries all resolve to one directory. Left as several, the second
  entry would see the first's source, call it a change, recompute over the first's
  outputs and replace its index row -- on every run, forever. The rule is
  :func:`~mosaic.core.pipeline.consumed_camera.one_camera_per_entry`, which the
  ``infer-*`` ops apply as well.

**A media variant replaces the source when the item is built.** Every tracker
checks reuse before it resolves the file it hands its tool, against the item's
``video_uid`` or ``source_uid``, and both are read from the item's facts. An item
that named a variant only when the file was resolved would pass those checks on
the entry media's identity, and reuse output made from the entry media. So a
variant item carries the variant file and its facts from the start, and an entry
whose variant is missing or out of date fails alone rather than ending the run.

**Joining is refused on geometry and accepted on frame rate.** The two
disagreements have opposite consequences. Clips that decode to different frame
shapes cannot be one video at all -- TRex says so itself, and mosaic says it
first, with a message naming the clip. Clips that were recorded at different
rates are a real and common property of a session (30, then 29.95, then 31 fps is
a measured example), and refusing them would refuse the data; they are carried
instead, and the consumer reconstructs time per clip through
:mod:`mosaic.core.media.timeline`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.core.helpers import make_entry_key
from mosaic.core.media.uniformity import geometry_mismatch
from mosaic.core.pipeline.composition import MediaMember, media_composition
from mosaic.core.pipeline.consumed_camera import one_camera_per_entry
from mosaic.core.pipeline.placement import SourceMapping
from mosaic.core.pipeline.preprocess_index import (
    MediaVariantDriftedError,
    MediaVariantMissingError,
)
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS
from mosaic.core.pipeline.variant_source import VariantSource, resolve_variant_source

if TYPE_CHECKING:
    from mosaic_media import MediaFacts

    from mosaic.core.dataset import Dataset, ResolvedScopeEntry

__all__ = [
    "JoinedSourceMismatchError",
    "TrackerWorkItem",
    "UnresolvedEntry",
    "WorkItems",
    "build_work_items",
]


class JoinedSourceMismatchError(ValueError):
    """An entry's clips cannot be handed to one tool as one video.

    Deliberately not a ``MediaProbeError``: that error's remedy is "transcode
    it", and no transcode mosaic performs rescales a frame or repairs a rotation
    difference. This one's remedy is to fix the arrangement.
    """


@dataclass(frozen=True, slots=True)
class TrackerWorkItem:
    """One sequence to track, and what it resolved to.

    There is exactly one item per ``key``, which is the ``<group>__<sequence>``
    working-directory name.

    ``video_paths`` are the entry's clips in ``video_order`` -- one element for
    the ordinary single-file sequence, several for a session a recorder split.
    ``source_facts`` is parallel to it. The single-source views every tracker
    already reads (``video_path``, ``video_uid``, ``facts``) are **derived** from
    element 0 rather than stored beside it, so "the first source" has one
    spelling that cannot drift from the list it comes from.
    """

    group: str
    sequence: str
    key: str
    video_paths: tuple[Path, ...]
    fps: float
    """The frame rate of ``video_path``, i.e. of the **first** clip.

    Deliberately not a mean over the clips. The three trackers that read this
    pass it to their converter and track only the first clip anyway, so a
    session-wide average would mistime exactly the output it reached. A consumer
    that joins the clips must ignore it and build a
    :class:`~mosaic.core.media.timeline.ConcatenatedTimeline` instead, because no
    single rate indexes a session whose clips disagree.
    """

    source_facts: tuple[MediaFacts, ...] = ()
    """The probed facts of each file in ``video_paths``, for a tracker that decodes.

    Read from the media index for the entry media, and from the variant index
    for a variant.

    ``open_frame_reader`` takes them so that a raw stream is read with measured
    values rather than trusted header ones -- a raw ``.h264`` reports a garbage
    frame count and cannot be seeked. Defaulted, because the three subprocess
    trackers hand a path to their tool and never open the file themselves.
    """

    camera: str = ""
    """The camera of the entry this item reads, ``""`` for single-camera media."""

    variant: VariantSource | None = None
    """The media variant this item reads, or ``None`` when it reads the entry media.

    When set, ``video_paths`` is the variant file and ``source_facts`` its stored
    facts, so every view derived from them (``video_uid``, ``source_uid``,
    ``facts``, ``n_sources``) describes the file the tool reads. The reuse gates
    compare those, so a tracker run over a variant never reuses output made from
    the entry media, and recomputes when the variant file is rewritten.
    """

    def __post_init__(self) -> None:
        """Facts are absent, or there is one per clip.

        Never fires on the production path -- ``_resolve_matched_rows`` appends a
        path and its facts in lockstep -- but a short tuple would silently place
        a session's later clips on the first clip's rate, which is the class of
        error the timeline exists to prevent.
        """
        if not self.video_paths:
            raise ValueError("a work item needs at least one video path")
        if self.source_facts and len(self.source_facts) != len(self.video_paths):
            raise ValueError(
                f"({self.group}, {self.sequence}) has {len(self.video_paths)} "
                f"videos but {len(self.source_facts)} facts; they must be parallel"
            )

    @property
    def video_path(self) -> Path:
        """The first clip -- what a tracker that reads one file gets."""
        return self.video_paths[0]

    @property
    def n_sources(self) -> int:
        """How many clips this item covers."""
        return len(self.video_paths)

    @property
    def facts(self) -> MediaFacts | None:
        """The first clip's probed facts, or ``None`` when there are none."""
        return self.source_facts[0] if self.source_facts else None

    @property
    def video_uid(self) -> str:
        """The first clip's content identity, empty when it carries none."""
        return self.source_facts[0].video_uuid if self.source_facts else ""

    @property
    def video_uids(self) -> tuple[str, ...]:
        """Every clip's content identity, in ``video_order``.

        Derived rather than stored: a second copy of what ``source_facts``
        already says is a second thing to keep in step.
        """
        return tuple(clip.video_uuid for clip in self.source_facts)

    @property
    def source_uid(self) -> str:
        """What the reuse gate compares -- the identity of *the whole input*.

        One clip: that clip's ``video_uuid``, unchanged, so nothing already on
        disk is invalidated by this concept existing. Several: the ordered
        composition digest, which is what notices a clip being replaced, added,
        removed or reordered -- none of which the first clip's uid can see.

        ``""`` when any clip carries no identity, which sends the gate to its
        path fallback. That fallback compares **the first clip only**, so a
        joined entry over unidentified media will not notice a later clip
        changing. It is the same trade the uid-less populations already make, and
        it is stated here rather than papered over.
        """
        if not self.source_facts:
            return ""
        if len(self.source_facts) == 1:
            return self.source_facts[0].video_uuid
        members = [
            MediaMember(camera="", video_order=order, uid=clip.video_uuid)
            for order, clip in enumerate(self.source_facts)
        ]
        return media_composition(members).digest

    @property
    def media(self) -> str:
        """The run identifier of the media variant read, ``""`` for the entry media."""
        return self.variant.run_id if self.variant is not None else ""

    @property
    def consumed_media(self) -> tuple[Path, ...]:
        """Every media file a table from this item derives from.

        The files the tool reads, or for a variant item the variant's
        :attr:`~mosaic.core.pipeline.variant_source.VariantSource.consumed_paths`.
        """
        if self.variant is not None:
            return self.variant.consumed_paths
        return self.video_paths

    @property
    def source_mapping(self) -> SourceMapping | None:
        """How a table tracked on this item maps into source space, if it must.

        ``None`` when the item reads the entry media, whose table is already in
        source space.
        """
        return self.variant.mapping() if self.variant is not None else None


@dataclass(frozen=True, slots=True)
class UnresolvedEntry:
    """An entry whose media variant could not be read, and why.

    Attributes:
        key: The entry's ``<group>__<sequence>`` key.
        error: What resolving the variant raised.
    """

    key: str
    error: MediaVariantMissingError | MediaVariantDriftedError


@dataclass(frozen=True, slots=True)
class WorkItems:
    """What a resolved media scope becomes: the items to run, and the entries lost.

    Attributes:
        items: One work item per entry, in scope order.
        failures: The entries whose media variant is missing or out of date.
            They fail alone, and the other entries run.
        media: The media variant the items read, ``""`` for the entry media.
    """

    items: tuple[TrackerWorkItem, ...]
    failures: tuple[UnresolvedEntry, ...] = ()
    media: str = ""


def build_work_items(
    ds: Dataset,
    scope: list[ResolvedScopeEntry],
    *,
    kind: str,
    media: str = "",
    fps_default: float | None = None,
) -> WorkItems:
    """Collapse a resolved media scope into one work item per entry.

    Args:
        ds: The dataset, read for its default frame rate when an entry's media
            index carries none.
        scope: What ``Dataset.resolve_media_scope`` returned.
        kind: The tracker's kind. It selects the tool's ``joins_sources``
            capability and prefixes warnings, so a message names the tool the
            user invoked rather than the shared machinery.
        media: The media variant each entry is read from, or ``""`` to read the
            entry media. A variant's file is resolved for the camera each entry
            keeps.
        fps_default: Frame rate for an entry whose facts carry none. Defaults to
            the dataset's ``fps_default``.

    Returns:
        The work items, and each entry whose variant is missing or out of date.

    Raises:
        JoinedSourceMismatchError: If a joining tracker's entry has clips that
            disagree on frame geometry, or one whose frame rate is unknown.
    """
    fallback_fps = (
        ds.meta_float("fps_default", 30.0) if fps_default is None else fps_default
    )
    root = TRACKING_ROOTS.get(kind)
    joins = root is not None and root.joins_sources
    items: list[TrackerWorkItem] = []
    failures: list[UnresolvedEntry] = []

    # Reduced first. An entry that is dropped is then not also warned about for
    # video count, a warning that would describe work this tracker will not do.
    for entry in one_camera_per_entry(kind, scope):
        group, sequence, resolved = entry.group, entry.sequence, entry.resolved
        key = make_entry_key(group, sequence)
        if media:
            try:
                variant = resolve_variant_source(ds, media, entry)
            except (MediaVariantMissingError, MediaVariantDriftedError) as exc:
                failures.append(UnresolvedEntry(key=key, error=exc))
                continue
            items.append(_variant_item(entry, key, variant))
            continue
        paths = list(resolved.paths)
        facts = list(resolved.facts)
        if len(paths) > 1 and not joins:
            print(
                f"[{kind}] ({group}, {sequence}) has {len(paths)} videos; using "
                f"the first ({paths[0].name}). {kind} reads one video file, so "
                f"the rest are not tracked.",
                file=sys.stderr,
            )
            paths, facts = paths[:1], facts[:1]

        if len(paths) > 1:
            _refuse_unjoinable(kind, group, sequence, paths, facts)

        items.append(
            TrackerWorkItem(
                group=group,
                sequence=sequence,
                key=key,
                video_paths=tuple(paths),
                fps=facts[0].fps if facts and facts[0].fps > 0 else fallback_fps,
                source_facts=tuple(facts),
                camera=entry.camera,
            )
        )

    return WorkItems(items=tuple(items), failures=tuple(failures), media=media)


def _variant_item(
    entry: ResolvedScopeEntry, key: str, variant: VariantSource
) -> TrackerWorkItem:
    """The work item reading *variant*'s file for *entry*.

    One source, the variant file, at the rate the file is labeled at. An entry
    of several clips is read as one file, so nothing is joined.
    """
    return TrackerWorkItem(
        group=entry.group,
        sequence=entry.sequence,
        key=key,
        video_paths=(variant.path,),
        fps=variant.placement.fps,
        source_facts=(variant.facts,),
        camera=entry.camera,
        variant=variant,
    )


def _refuse_unjoinable(
    kind: str,
    group: str,
    sequence: str,
    paths: list[Path],
    facts: list[MediaFacts],
) -> None:
    """Raise unless *facts* describe clips that can be read as one video.

    Checked here, before a work item exists, so a run dies naming the file and
    the field rather than inside a subprocess whose traceback names neither.
    """
    entry = f"({group}, {sequence})"
    if len(facts) != len(paths):
        raise JoinedSourceMismatchError(
            f"[{kind}] {entry} has {len(paths)} clips but measurements for "
            f"{len(facts)}, so they cannot be joined into one video. Re-probe the "
            f"sequence with 'mosaic reprobe-media'."
        )

    mismatch = geometry_mismatch(facts)
    if mismatch is not None:
        raise JoinedSourceMismatchError(
            f"[{kind}] {entry} cannot be tracked as one video: "
            f"{paths[mismatch.index].name} has {mismatch.field} "
            f"{mismatch.other} where {paths[0].name} has {mismatch.first}. "
            f"Clips of one sequence must decode to the same frame. "
            f"See ds.sequence_uniformity({group!r}, {sequence!r}) for the whole "
            f"picture."
        )

    for position, clip in enumerate(facts):
        if clip.fps <= 0:
            raise JoinedSourceMismatchError(
                f"[{kind}] {entry} cannot be tracked as one video: "
                f"{paths[position].name} reports no frame rate, so its frames "
                f"cannot be placed on the sequence's time axis. A default would "
                f"put a wrong slope on one clip of an otherwise measured "
                f"session. Re-probe it with 'mosaic reprobe-media'."
            )
