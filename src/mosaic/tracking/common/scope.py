"""Turning a resolved media scope into the list of entries a tracker will run.

``Dataset.resolve_media_scope`` answers "which entries, and which file does each
resolve to" -- routing an analysis-required entry to its constant-rate derivative
rather than the defective original. What every tracker then does with that answer
is identical, and was written three times.

Two collapses happen here, and both are load-bearing rather than tidy-up:

* **Several videos under one entry** stay on one work item, which covers them
  all. A recorder that chops a session into clips leaves a boundary that is a
  filesystem artifact, not an event. A tool that opens one file is handed the
  clips' join (:mod:`mosaic.core.pipeline.joined_export`), and mosaic's runner
  reads the clips themselves in order, on the same frame axis.
* **Several cameras under one entry** collapse onto one work item. The working
  directory is keyed on ``(group, sequence)`` with no camera, so a multi-camera
  sequence's entries all resolve to one directory. Left as several, the second
  entry would see the first's source, call it a change, recompute over the first's
  outputs and replace its index row -- on every run, forever. The rule is
  :func:`~mosaic.core.pipeline.consumed_camera.one_camera_per_entry`, which the
  ``infer-*`` ops apply as well.

**A media variant replaces the source when the item is built.** Every tracker
checks reuse against the item's ``source_uid``, read from the item's facts, before
it resolves the file that it hands its tool. A variant item therefore contains the
variant file and its facts from the start, and every reuse check compares the
variant's identity instead of the entry media's. An entry whose variant is
missing or out of date fails alone rather than ending the run.

**Joining is refused on geometry and accepted on frame rate.** The two
disagreements have opposite consequences. Clips that decode to different frame
shapes cannot be one video at all -- TRex says so itself, and mosaic says it
first, with a message naming the clip. Clips that were recorded at different
rates are a real and common property of a session (30, then 29.95, then 31 fps is
a measured example), and refusing them would refuse the data; they are carried
instead, and the consumer reconstructs time per clip through
:mod:`mosaic.core.media.timeline`. Imgstore recordings are the exception. A
tracker reads a store only as an entry's one clip, because ``export-joined`` does
not join stores, so it refuses an entry of several clips that includes a store.
A reader in this process refuses stores at different rates, because the store
reader needs one rate.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mosaic_media import MediaProbeError

from mosaic.core.helpers import make_entry_key
from mosaic.core.media.imgstore_io import is_imgstore
from mosaic.core.media.store_division import MediaRowKey, StoreWindow
from mosaic.core.media.store_rate import store_rate_mismatch
from mosaic.core.media.uniformity import geometry_mismatch
from mosaic.core.pipeline.consumed_camera import one_camera_per_entry
from mosaic.core.pipeline.joined_export import (
    entry_source_uid,
    refuse_unidentified_clips,
)
from mosaic.core.pipeline.placement import EntryAxis
from mosaic.core.pipeline.preprocess_index import (
    MediaVariantDriftedError,
    MediaVariantMissingError,
)
from mosaic.core.pipeline.run_commands import mosaic_run_command
from mosaic.core.pipeline.variant_source import VariantLookup, VariantSource

if TYPE_CHECKING:
    from mosaic_media import MediaFacts

    from mosaic.core.dataset import Dataset, ResolvedScopeEntry
    from mosaic.core.entry import Entry

__all__ = [
    "JoinedSourceMismatchError",
    "TrackerWorkItem",
    "UnresolvedEntry",
    "WorkItems",
    "build_work_items",
    "refuse_unjoinable",
    "run_entries",
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
    ``source_facts`` is parallel to it. The single-source views (``video_path``
    and ``facts``) are **derived** from element 0 rather than stored beside it,
    so "the first source" has one spelling that cannot drift from the list it
    comes from.
    """

    group: str
    sequence: str
    key: str
    video_paths: tuple[Path, ...]
    fps: float
    """The frame rate of ``video_path``, i.e. of the **first** clip.

    Not a mean over the clips, because a mean describes none of them. It is the
    rate that a tool reads the clips' join at, because the join is labeled at its
    first clip's rate. No single rate indexes a session whose clips disagree, so
    the shared bridge retimes a table from several clips by each clip's rate
    (:meth:`entry_axis`).
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

    windows: tuple[StoreWindow | None, ...] = ()
    """The chunks of each store in ``video_paths`` that the item reads.

    Parallel to ``video_paths``, ``None`` for a file read whole. Empty reads every
    file whole. A division of a store reads its window.
    """

    first_frame: int = 0
    """The sequence's first frame on its recording's frame axis.

    The tool reads the sequence's media, and :meth:`entry_axis` moves its
    table onto the recording's axis by this many frames. 0 for an undivided
    recording.
    """

    variant: VariantSource | None = None
    """The media variant that this item reads, or ``None`` for the entry media.

    When set, ``video_paths`` is the variant file and ``source_facts`` its stored
    facts. Every view derived from them (``source_uid``, ``facts``,
    ``n_sources``) describes the file that the tool reads, and the reuse gates
    compare those. A tracker run over a variant therefore never reuses output
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
        if self.windows and len(self.windows) != len(self.video_paths):
            raise ValueError(
                f"({self.group}, {self.sequence}) has {len(self.video_paths)} "
                f"videos but {len(self.windows)} windows; they must be parallel"
            )

    @property
    def video_windows(self) -> tuple[StoreWindow | None, ...]:
        """The window of each file in ``video_paths``, ``None`` for a file read whole."""
        return self.windows or (None,) * len(self.video_paths)

    @property
    def row_keys(self) -> tuple[MediaRowKey, ...]:
        """The media index key of each file's row, parallel to ``video_paths``."""
        return tuple(
            MediaRowKey.of(str(path), window)
            for path, window in zip(self.video_paths, self.video_windows, strict=True)
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
    def measured_fps(self) -> float | None:
        """The measured frame rate of the file the tool reads, or ``None``.

        The first clip's probed rate, which a join of the clips is labeled at, or
        the variant file's. ``None`` where the facts are absent or report no
        rate. :attr:`fps` falls back to the dataset's default there, and a
        default is not a measurement to correct a tool's timing by.
        """
        facts = self.facts
        if facts is None or facts.fps <= 0:
            return None
        return facts.fps

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

        ``""`` when any clip carries no identity. For one clip that sends the
        gate to its path fallback. An entry of several clips with one unidentified
        is refused by :func:`build_work_items` instead, because the fallback
        compares the first clip's path only, and no join of the clips can be
        addressed.

        It is :func:`~mosaic.core.pipeline.joined_export.entry_source_uid`, the
        identity an inference op records for the same media.
        """
        return entry_source_uid(self.source_facts)

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
        """Return how a table tracked on this item is placed on its entry's axes.

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
            return EntryAxis.of_variant(
                self.variant.mapping(), first_frame=self.first_frame
            )
        return EntryAxis.of_entry_media(
            self.source_facts, windowed=windowed, first_frame=self.first_frame
        )


@dataclass(frozen=True, slots=True)
class UnresolvedEntry:
    """An entry that a tool cannot be handed, and why.

    Either its media variant could not be read, or its sequence could not be
    placed on its recording's frame axis.

    Attributes:
        group: The entry's group.
        sequence: The entry's sequence.
        run_entry: The entry as a run's scope names it, by its indexed key. It
            differs from *group* and *sequence* for a nameless entry, which a
            scope labels by its first file's stem.
        error: The error that resolving the variant, or placing the sequence,
            raised.
    """

    group: str
    sequence: str
    run_entry: Entry
    error: MediaVariantMissingError | MediaVariantDriftedError | MediaProbeError

    @property
    def variant_unreadable(self) -> bool:
        """Whether the entry was lost because its media variant could not be read."""
        return not isinstance(self.error, MediaProbeError)

    @property
    def key(self) -> str:
        """The entry's ``<group>__<sequence>`` key."""
        return make_entry_key(self.group, self.sequence)


def never_handed_refusal(kind: str, run_id: str, lost: Iterable[str]) -> str:
    """Return the refusal for a run of *kind* whose tool was handed no entry it lost.

    Each lost entry's media variant could not be read, or its sequence could not
    be placed on its recording's frame axis. The tool did not run for it and
    wrote no output to keep.

    Args:
        kind: The op kind of the run.
        run_id: The run's identifier.
        lost: The keys of every entry that the run attempted and lost.
    """
    return (
        f"[{kind}] run_id={run_id} did not produce tracks, because no attempted "
        f"entry could be handed to the tool: {', '.join(sorted(lost))}. The "
        f"per-entry errors are in this attempt's run-log."
    )


@dataclass(frozen=True, slots=True)
class WorkItems:
    """The work from a resolved media scope: the items to run, and the entries lost.

    Attributes:
        items: One work item per entry, in scope order.
        failures: The entries whose media variant is missing or out of date,
            and those whose sequence cannot be placed on its recording's frame
            axis. They fail alone, and the other entries run.
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
            index records none, and for the indexed key that a refusal's command
            names.
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
        The work items, each entry whose variant is missing or out of date, and
        each entry whose sequence cannot be placed on its recording's frame axis.

    Raises:
        JoinedSourceMismatchError: If an entry has clips that disagree on frame
            geometry, one whose frame rate is unknown, or several clips of which
            one is a store.
        JoinedExportMissingError: If an entry has several clips and one carries
            no content identity. No join of them can be addressed, and no marker
            can prove which clips earlier output was made from.
    """
    fallback_fps = (
        ds.meta_float("fps_default", 30.0) if fps_default is None else fps_default
    )
    items: list[TrackerWorkItem] = []
    lost: list[
        tuple[
            ResolvedScopeEntry,
            MediaVariantMissingError | MediaVariantDriftedError | MediaProbeError,
        ]
    ] = []

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
        try:
            first_frame = resolved.first_frame
        except MediaProbeError as exc:
            lost.append((entry, exc))
            continue
        if variants is not None:
            try:
                variant = variants.resolve(ds, entry)
            except (MediaVariantMissingError, MediaVariantDriftedError) as exc:
                lost.append((entry, exc))
                continue
            items.append(_variant_item(entry, key, variant, first_frame))
            continue
        paths = list(resolved.paths)
        facts = list(resolved.facts)
        windows = resolved.windows
        refuse_unjoinable(
            ds,
            kind,
            group,
            sequence,
            paths,
            facts,
            windows=windows,
            hands_over_path=True,
        )
        refuse_unidentified_clips(
            ds, group, sequence, paths, facts, windows=windows, asker=kind
        )

        items.append(
            TrackerWorkItem(
                group=group,
                sequence=sequence,
                key=key,
                video_paths=tuple(paths),
                fps=facts[0].fps if facts and facts[0].fps > 0 else fallback_fps,
                source_facts=tuple(facts),
                camera=entry.camera,
                first_frame=first_frame,
                windows=tuple(resolved.windows),
            )
        )

    named = run_entries(
        ds, [((entry.group, entry.sequence), entry) for entry, _ in lost]
    )
    failures = tuple(
        UnresolvedEntry(
            group=entry.group, sequence=entry.sequence, run_entry=run_entry, error=exc
        )
        for (entry, exc), run_entry in zip(lost, named, strict=True)
    )
    return WorkItems(items=tuple(items), failures=failures, media=media)


def run_entries(
    ds: Dataset, cameras: Sequence[tuple[Entry, ResolvedScopeEntry]]
) -> list[Entry]:
    """Return the entry that a run's scope names for each of *cameras*, in order.

    It is the indexed key of the camera's first row
    (:meth:`~mosaic.core.dataset.Dataset.indexed_entries_of`). A media scope
    labels a nameless entry by its first file's stem, and a run's scope does not
    resolve that label. The entry paired with a camera names it when the media
    index does not list the file.
    """
    keys = ds.indexed_entries_of(camera.resolved.row_keys[0] for _, camera in cameras)
    return [keys.get(camera.resolved.row_keys[0], named) for named, camera in cameras]


def _variant_item(
    entry: ResolvedScopeEntry, key: str, variant: VariantSource, first_frame: int
) -> TrackerWorkItem:
    """Return the work item that reads *variant*'s file for *entry*.

    The item has one source, the variant file, at the rate that the file is
    labeled at, and starts at *first_frame* on the recording's axis. An entry of
    several clips is read as one file and is not joined.
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
        first_frame=first_frame,
    )


def refuse_unjoinable(
    ds: Dataset,
    kind: str,
    group: str,
    sequence: str,
    paths: Sequence[Path],
    facts: Sequence[MediaFacts],
    *,
    windows: Sequence[StoreWindow | None],
    hands_over_path: bool,
) -> None:
    """Raise unless *facts* describe clips that can be read as one video.

    Checked before any work starts, by the trackers as they build their work
    items and by the inference ops as they resolve their entries, so a run dies
    naming the file and the field rather than inside a subprocess whose
    traceback names neither. One clip is its own video and passes.

    An entry of several clips that holds a store is refused for a consumer whose
    tool reads files outside this process (*hands_over_path*): such a tool reads
    a store only as an entry's one clip, and ``export-joined`` does not join
    stores. A consumer that reads the stores in its own process reads them as one
    only at one rate, by the rule the store reader applies
    (:func:`~mosaic.core.media.store_rate.store_rate_mismatch`).
    Either refusal says how to make a media variant that contains the entry as
    one file, with a command that names the entry by its first clip's indexed
    key (:meth:`~mosaic.core.dataset.Dataset.indexed_entry`), its path and its
    window in *windows*, which are parallel to *paths*.

    Raises:
        JoinedSourceMismatchError: If the clips lack one set of facts each,
            disagree on frame geometry, or include one whose frame rate is
            unknown; if they include a store and *hands_over_path*; or if they
            are stores whose frame rates differ.
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

    stores = [path.name for path in paths if is_imgstore(path)]
    if not stores:
        return
    if hands_over_path:
        raise JoinedSourceMismatchError(
            f"[{kind}] {entry} cannot be read as one video: {kind} reads a store "
            f"only as an entry's one clip, and export-joined does not join stores, "
            f"so nothing holds {', '.join(stores)} with the entry's other clips. "
            f"{_variant_remedy(ds, group, sequence, paths, facts, windows)}"
        )
    other = store_rate_mismatch([clip.fps for clip in facts])
    if other is not None:
        raise JoinedSourceMismatchError(
            f"[{kind}] {entry} cannot be read as one video: "
            f"{paths[other].name} was recorded at {facts[other].fps:g} fps and "
            f"{paths[0].name} at {facts[0].fps:g}. Stores cannot be joined, so no "
            f"reader places their frames on one axis. Re-record the stores at one "
            f"rate, or read them through a media variant. "
            f"{_variant_remedy(ds, group, sequence, paths, facts, windows)}"
        )


def _variant_remedy(
    ds: Dataset,
    group: str,
    sequence: str,
    paths: Sequence[Path],
    facts: Sequence[MediaFacts],
    windows: Sequence[StoreWindow | None],
) -> str:
    """Return how to make a media variant that reads an entry of stores as one file.

    The variant reads each store at its recorded rate. Variants are AV1 by
    default, and preprocess refuses an AV1 variant with no step. The remedy
    therefore names a trim over every frame of the entry, which keeps them all.
    The trim's end is the entry's frame count from *facts*. When a store's length
    is unknown, the remedy first names a measurement of the stores. The command
    names the entry by the indexed key of its first file in *paths*, read on
    its window in *windows*.
    """
    counts = [int(clip.frame_count) for clip in facts]
    if min(counts) <= 0:
        return (
            "Measure the stores with 'mosaic reprobe-media --apply'. Then make a "
            "media variant of the entry, which reads each store at its own rate, "
            "and name it with the 'media' parameter. A trim from frame 0 to the "
            "entry's frame count keeps every frame."
        )
    total = sum(counts)
    steps = json.dumps(
        {"steps": [{"step": "trim", "start": 0, "stop": total}]},
        separators=(",", ":"),
    )
    key = MediaRowKey.of(str(paths[0]), windows[0])
    entry = ds.indexed_entry(key, (group, sequence))
    command = mosaic_run_command("preprocess", [entry], params=steps)
    return (
        f"Make a media variant of the entry, which reads each store at its own "
        f"rate, and name it with the 'media' parameter. A trim over all {total} "
        f"frames keeps every frame:\n    {command}"
    )
