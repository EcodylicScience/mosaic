"""Turning a resolved media scope into the list of entries a tracker will run.

``Dataset.resolve_media_scope`` answers "which entries, and which file does each
resolve to" -- routing an analysis-required entry to its constant-rate derivative
rather than the defective original. What every tracker then does with that answer
is identical, and was written three times.

Two collapses happen here, and both are load-bearing rather than tidy-up:

* **Several videos under one entry** stay on one work item, which covers them
  all. A recorder that chops a session into clips leaves a boundary that is a
  filesystem artifact, not an event, and every tool is handed the clips' join as
  one video (:mod:`mosaic.core.pipeline.joined_export`).
* **Several cameras under one entry** collapse onto one work item. The working
  directory is keyed on ``(group, sequence)`` with no camera, so a multi-camera
  sequence's entries all resolve to one directory. Left as several, the second
  entry would see the first's source, call it a change, recompute over the first's
  outputs and replace its index row -- on every run, forever. The rule is
  :func:`~mosaic.core.pipeline.consumed_camera.one_camera_per_entry`, which the
  ``infer-*`` ops apply as well.

**A media variant replaces the source when the item is built.** Every tracker
checks reuse against the item's ``video_uid`` or ``source_uid`` before it resolves
the file that it hands its tool, and both are read from the item's facts. A variant
item therefore contains the variant file and its facts from the start, and every
reuse check compares the variant's identity instead of the entry media's. An
entry whose variant is missing or out of date fails alone rather than ending the
run.

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

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.core.helpers import make_entry_key
from mosaic.core.media.uniformity import geometry_mismatch
from mosaic.core.pipeline.composition import MediaMember, media_composition
from mosaic.core.pipeline.consumed_camera import one_camera_per_entry
from mosaic.core.pipeline.placement import EntryAxis
from mosaic.core.pipeline.preprocess_index import (
    MediaVariantDriftedError,
    MediaVariantMissingError,
)
from mosaic.core.pipeline.variant_source import VariantLookup, VariantSource

if TYPE_CHECKING:
    from mosaic_media import MediaFacts

    from mosaic.core.dataset import Dataset, ResolvedScopeEntry

__all__ = [
    "JoinedSourceMismatchError",
    "TrackerWorkItem",
    "UnresolvedEntry",
    "WorkItems",
    "build_work_items",
    "refuse_unjoinable",
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

    Deliberately not a mean over the clips, because a mean describes none of
    them. It is the rate a tool reads the clips' join at, since the join is
    labeled at its first clip's rate. No single rate indexes a session whose
    clips disagree, so the shared bridge retimes a table from several clips by
    each clip's own rate (:meth:`entry_axis`).
    """

    source_facts: tuple[MediaFacts, ...] = ()
    """The probed facts of each file in ``video_paths``, for a tracker that decodes.

    They are read from the media index for the entry media, and from the variant
    index for a variant.

    ``open_frame_reader`` takes them so that a raw stream is read with measured
    values rather than trusted header ones -- a raw ``.h264`` reports a garbage
    frame count and cannot be seeked. Defaulted, because the three subprocess
    trackers hand a path to their tool and never open the file themselves.
    """

    camera: str = ""
    """The camera of the entry that this item reads, ``""`` for single-camera media."""

    variant: VariantSource | None = None
    """The media variant that this item reads, or ``None`` for the entry media.

    When set, ``video_paths`` is the variant file and ``source_facts`` its stored
    facts. Every view derived from them (``video_uid``, ``source_uid``,
    ``facts``, ``n_sources``) describes the file that the tool reads, and the reuse
    gates compare those. A tracker run over a variant therefore never reuses output
    made from the entry media, and it recomputes when the variant file is
    rewritten.
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
        """Every media file that a table from this item derives from.

        They are the files that the tool reads, or for a variant item the variant's
        :attr:`~mosaic.core.pipeline.variant_source.VariantSource.consumed_media`.
        """
        if self.variant is not None:
            return self.variant.consumed_media
        return self.video_paths

    def entry_axis(self, *, windowed: bool) -> EntryAxis:
        """Where a table tracked on this item sits on its entry's axes.

        Args:
            windowed: Whether the run reads fewer than every frame, as
                :attr:`~mosaic.core.pipeline.media_input.MediaInputParams.frame_window`
                says. A variant item ignores it, because a frame window is
                refused beside a variant.

        Returns:
            The variant's mapping for a variant item, and otherwise the entry's
            clips, whose join a tool read when there are several.
        """
        if self.variant is not None:
            return EntryAxis.of_variant(self.variant.mapping())
        return EntryAxis.of_entry_media(self.source_facts, windowed=windowed)


@dataclass(frozen=True, slots=True)
class UnresolvedEntry:
    """An entry whose media variant could not be read, and why.

    Attributes:
        group: The entry's group.
        sequence: The entry's sequence.
        error: The error that resolving the variant raised.
    """

    group: str
    sequence: str
    error: MediaVariantMissingError | MediaVariantDriftedError

    @property
    def key(self) -> str:
        """The entry's ``<group>__<sequence>`` key."""
        return make_entry_key(self.group, self.sequence)


@dataclass(frozen=True, slots=True)
class WorkItems:
    """The work from a resolved media scope: the items to run, and the entries lost.

    Attributes:
        items: One work item per entry, in scope order.
        failures: The entries whose media variant is missing or out of date.
            They fail alone, and the other entries run.
        media: The media variant that the items read, or ``""`` for the entry media.
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
        kind: The tracker's kind. It prefixes warnings and refusals, so a
            message names the tool the user invoked rather than the shared
            machinery.
        media: The media variant that each entry is read from, or ``""`` to read
            the entry media. A variant's file is resolved for the camera that each
            entry keeps.
        fps_default: Frame rate for an entry whose facts carry none. Defaults to
            the dataset's ``fps_default``.

    Returns:
        The work items, and each entry whose variant is missing or out of date.

    Raises:
        JoinedSourceMismatchError: If an entry has clips that disagree on frame
            geometry, or one whose frame rate is unknown.
    """
    fallback_fps = (
        ds.meta_float("fps_default", 30.0) if fps_default is None else fps_default
    )
    items: list[TrackerWorkItem] = []
    failures: list[UnresolvedEntry] = []

    # Reduced first. An entry that is dropped is then not also warned about for
    # video count, a warning that would describe work this tracker will not do.
    kept = one_camera_per_entry(kind, scope)
    variants = (
        VariantLookup.read(ds, media, [(entry.group, entry.sequence) for entry in kept])
        if media
        else None
    )
    for entry in kept:
        group, sequence, resolved = entry.group, entry.sequence, entry.resolved
        key = make_entry_key(group, sequence)
        if variants is not None:
            try:
                variant = variants.resolve(ds, entry)
            except (MediaVariantMissingError, MediaVariantDriftedError) as exc:
                failures.append(
                    UnresolvedEntry(group=group, sequence=sequence, error=exc)
                )
                continue
            items.append(_variant_item(entry, key, variant))
            continue
        paths = list(resolved.paths)
        facts = list(resolved.facts)
        refuse_unjoinable(kind, group, sequence, paths, facts)

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
    """Return the work item that reads *variant*'s file for *entry*.

    The item has one source, the variant file, at the rate that the file is
    labeled at. An entry of several clips is read as one file and is not joined.
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


def refuse_unjoinable(
    kind: str,
    group: str,
    sequence: str,
    paths: Sequence[Path],
    facts: Sequence[MediaFacts],
) -> None:
    """Raise unless *facts* describe clips that can be read as one video.

    Checked before any work starts, by the trackers as they build their work
    items and by the inference ops as they resolve their entries, so a run dies
    naming the file and the field rather than inside a subprocess whose
    traceback names neither. One clip is its own video and passes.

    Raises:
        JoinedSourceMismatchError: If the clips lack one set of facts each,
            disagree on frame geometry, or include one whose frame rate is
            unknown.
    """
    if len(paths) < 2:
        return
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
            f"[{kind}] {entry} cannot be read as one video: "
            f"{paths[mismatch.index].name} has {mismatch.field} "
            f"{mismatch.other} where {paths[0].name} has {mismatch.first}. "
            f"Clips of one sequence must decode to the same frame. "
            f"See ds.sequence_uniformity({group!r}, {sequence!r}) for the whole "
            f"picture."
        )

    for position, clip in enumerate(facts):
        if clip.fps <= 0:
            raise JoinedSourceMismatchError(
                f"[{kind}] {entry} cannot be read as one video: "
                f"{paths[position].name} reports no frame rate, so its frames "
                f"cannot be placed on the sequence's time axis. A default would "
                f"put a wrong slope on one clip of an otherwise measured "
                f"session. Re-probe it with 'mosaic reprobe-media'."
            )
