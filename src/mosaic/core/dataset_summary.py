"""One dataset described in one read: its manifest, its counts, its schemas and frame rates.

:func:`dataset_summary` returns a :class:`DatasetSummary`. The CLI's
``mosaic describe --json`` is that model dumped.

A summary reads the manifest, the tracks index, the originals media index and
an inventory of three artifact kinds. Its recordings are named by
:mod:`mosaic.core.recording`. It writes nothing, takes no lock and
creates no root.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Final

from pydantic import Field

from mosaic.core.json_value import JsonValue
from mosaic.core.manifest import DatasetTag, LibraryLink, SplitRole
from mosaic.core.pipeline.inventory import (
    ArtifactKind,
    DatasetInventory,
    existing_tracks_records,
    inventory,
    reportable_universe,
)
from mosaic.core.entry import Entry
from mosaic.core.pipeline.dataset_indexes import root_subdirectories
from mosaic.core.pipeline.label_series_index import series_index_path
from mosaic.core.pipeline.labels_index import read_labels_index
from mosaic.core.pipeline.tracks_raw_index import read_tracks_raw_index
from mosaic.core.pipeline.tracks_index import (
    FrameExtents,
    read_frame_extents,
    read_tracks_index,
)
from mosaic.core.recording import Recording, facts_from_frame_extents
from mosaic.core.schema import LEGACY_SCHEMA
from mosaic.core.strict_model import StrictModel

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "SUMMARY_KINDS",
    "DatasetCounts",
    "DatasetSummary",
    "FrameRateCount",
    "RecordingSequence",
    "RecordingSummary",
    "TrackSchemaCount",
    "dataset_next_steps",
    "raw_label_rows",
    "dataset_summary",
]

SUMMARY_KINDS: Final[tuple[ArtifactKind, ...]] = (
    "tracks-variant",
    "feature",
    "trained-model",
)
"""The artifact kinds whose records a summary counts."""


class DatasetCounts(StrictModel):
    """How much a dataset holds.

    Attributes:
        sequences: The ``(group, sequence)`` entries whose tracks table exists,
            across every tracks variant.
        recordings: The recordings of the entries that the media index or the
            tracks index names. The sequences of a continuous group are one
            divided recording, and every other entry is an undivided one.
        divided_recordings: The recordings among them that are divided.
        videos: The rows of the originals media index, one per video file or
            imgstore store.
        tracks_variants: The tracks variants in the tracks index. ``None`` when
            the inventory did not report the kind.
        feature_runs: The feature runs, one per storage directory and run id.
            ``None`` when the inventory did not report the kind.
        models: The trained models. ``None`` when the inventory did not report
            the kind, as an inventory read without the tracking contributors
            registered does not.
    """

    sequences: int
    recordings: int
    divided_recordings: int
    videos: int
    tracks_variants: int | None
    feature_runs: int | None
    models: int | None


class TrackSchemaCount(StrictModel):
    """One track schema and the number of tracks tables recorded under it.

    Attributes:
        name: The schema name. A row with no recorded schema counts as the
            legacy schema, ``trex_v1``.
        tables: The tracks index rows that record it. A row is counted only
            when its table exists on disk.
    """

    name: str
    tables: int


class FrameRateCount(StrictModel):
    """One frame rate and the number of indexed videos recorded at it.

    Attributes:
        fps: The frame rate, as the media index records it.
        videos: The media index rows that record it.
    """

    fps: float
    videos: int


class RecordingSequence(StrictModel):
    """One sequence of a divided recording.

    Attributes:
        sequence: The sequence's name.
        first_frame: Its first frame on the recording's frame axis: the first
            frame of its tracks table, ``None`` when unrecorded.
        frame_count: The frames from its tracks table's first frame to its last,
            ``None`` when unrecorded.
        start_utc: When it starts, ``None`` when unrecorded.
    """

    sequence: str
    first_frame: int | None = None
    frame_count: int | None = None
    start_utc: datetime | None = None


class RecordingSummary(StrictModel):
    """One divided recording and its sequences.

    Attributes:
        name: The recording's name, its continuous group.
        n_sequences: The sequences it is divided into.
        sequences: Its sequences in recording order: by their tracks tables'
            first frames, then by a natural sort of their names. A member with
            two labelled tracks variants leaves its recording in name order and
            unmeasured.
    """

    name: str
    n_sequences: int
    sequences: tuple[RecordingSequence, ...]


class DatasetSummary(StrictModel):
    """What a dataset is and how much it holds.

    Attributes:
        manifest: The absolute path of the manifest file.
        name: The manifest's ``name``.
        uuid: The manifest's ``uuid``, ``None`` for a manifest without one.
        manifest_version: The format version of the file on disk, before any
            in-memory migration.
        version: The dataset's own version string.
        created_at: When the manifest was first written, ISO-8601 UTC.
        notes: The manifest's free text.
        tags: The typed dataset tags, in display order.
        libraries: The linked library datasets, in lookup order.
        continuous_groups: The groups whose sequences divide one recording.
        splits: The role of each named group in a held-out protocol, ``train``,
            ``test`` or ``sealed``. Empty when the manifest declares none.
        divided_recordings: The divided recordings that hold an entry, ordered
            by name. An undivided recording is one entry, which ``counts`` counts
            and ``mosaic sequences`` lists.
        managed_by: The manifest's top-level ``managed_by`` value, verbatim.
        api_managed: Whether ``managed_by`` names a managing service.
        counts: Sequences, videos and artifacts.
        track_schemas: The schemas of the tracks tables, by name.
        frame_rates: The frame rates of the indexed videos, ascending. Rows
            with a blank or non-positive rate are left out.
        inventory_errors: The inventory's read failures, one per kind, each
            prefixed with its kind. That kind's count omits what failed.
        next_steps: What the dataset needs before a pipeline reads its tracks
            or labels, from its declared sources, its raw indexes and its
            converted tables (:func:`dataset_next_steps`). Empty when nothing
            is missing.
    """

    manifest: str
    name: str
    uuid: str | None
    manifest_version: int
    version: str
    created_at: str | None
    notes: str
    tags: tuple[DatasetTag, ...]
    libraries: tuple[LibraryLink, ...]
    continuous_groups: tuple[str, ...]
    divided_recordings: tuple[RecordingSummary, ...]
    managed_by: JsonValue
    api_managed: bool
    splits: dict[str, SplitRole] = Field(default_factory=dict)
    counts: DatasetCounts
    track_schemas: tuple[TrackSchemaCount, ...]
    frame_rates: tuple[FrameRateCount, ...]
    inventory_errors: tuple[str, ...]
    next_steps: tuple[str, ...] = ()


def dataset_summary(
    ds: Dataset, *, found: DatasetInventory | None = None
) -> DatasetSummary:
    """Describe *ds*: its manifest fields, counts, track schemas and frame rates.

    Args:
        ds: The dataset to describe. Never written to.
        found: An inventory of *ds* to count artifacts from, such as
            ``mosaic.tracking.dataset_inventory`` returns. ``None`` reads
            :data:`SUMMARY_KINDS` with the contributors already registered, so
            ``counts.models`` is ``None`` unless the tracking contributors are.

    Returns:
        The summary.
    """
    held = found if found is not None else inventory(ds, kinds=SUMMARY_KINDS)
    manifest = ds.manifest
    tracks = existing_tracks_records(ds, read_tracks_index(ds))
    media = _media_rows(ds)
    entries = ds.recording_entries()
    extents = _member_extents(ds, entries)
    every_recording = ds.recordings(
        entries, order_facts=facts_from_frame_extents(extents)
    )
    divided = [
        recording for recording in every_recording if recording.kind == "divided"
    ]
    return DatasetSummary(
        manifest=str(ds.manifest_path.absolute()),
        name=manifest.name,
        uuid=manifest.uuid,
        manifest_version=manifest.migrated_from
        if manifest.migrated_from is not None
        else manifest.manifest_version,
        version=manifest.version,
        created_at=manifest.created_at,
        notes=manifest.notes,
        tags=manifest.ordered_tags(),
        libraries=manifest.libraries,
        continuous_groups=ds.continuous_groups,
        divided_recordings=tuple(
            _recording_summary(recording, extents) for recording in divided
        ),
        splits=dict(sorted(manifest.splits.items())),
        managed_by=manifest.managed_by,
        api_managed=manifest.is_managed,
        counts=DatasetCounts(
            sequences=len(reportable_universe(ds)),
            recordings=len(every_recording),
            divided_recordings=len(divided),
            videos=len(media),
            tracks_variants=_count(held, "tracks-variant"),
            feature_runs=_count(held, "feature"),
            models=_count(held, "trained-model"),
        ),
        track_schemas=tuple(
            TrackSchemaCount(name=name, tables=tables)
            for name, tables in sorted(
                Counter(
                    record.get("std_format", "") or LEGACY_SCHEMA for record in tracks
                ).items()
            )
        ),
        frame_rates=tuple(
            FrameRateCount(fps=fps, videos=videos)
            for fps, videos in sorted(
                Counter(
                    fps
                    for fps in (_frame_rate(row.get("fps") or "") for row in media)
                    if fps is not None
                ).items()
            )
        ),
        inventory_errors=held.errors,
        next_steps=dataset_next_steps(ds, converted_tracks=len(tracks)),
    )


def _member_extents(ds: Dataset, entries: Iterable[Entry]) -> FrameExtents:
    """Return the tracks frame extents of the divided recordings' members.

    Each member's extent is read as ``build_manifest`` reads it without a pinned
    variant. A recording with a member of two labelled variants contributes no
    extent, because a summary chooses no variant and an overlapped run over it
    refuses.
    """
    groups = sorted(
        {group for group, _ in entries if ds.recording_kind(group) == "divided"}
    )
    extents: FrameExtents = {}
    for group in groups:
        try:
            extents.update(read_frame_extents(ds, None, groups={group}))
        except ValueError:
            continue
    return extents


def _recording_summary(
    recording: Recording, extents: Mapping[Entry, tuple[int, int]]
) -> RecordingSummary:
    """Return *recording*'s name and its sequences in member order, with their extents."""
    sequences: list[RecordingSequence] = []
    for member in recording.members:
        extent = extents.get(member)
        sequences.append(
            RecordingSequence(sequence=member[1])
            if extent is None
            else RecordingSequence(
                sequence=member[1],
                first_frame=extent[0],
                frame_count=extent[1] - extent[0] + 1,
            )
        )
    return RecordingSummary(
        name=recording.name,
        n_sequences=len(recording.members),
        sequences=tuple(sequences),
    )


def _count(held: DatasetInventory, kind: ArtifactKind) -> int | None:
    """Return the records of *kind*, ``None`` when *held* did not report the kind."""
    if kind not in held.scope.kinds or kind in held.unavailable_kinds:
        return None
    return len(held.of_kind(kind))


def _media_rows(ds: Dataset) -> list[dict[str, str]]:
    """Return the originals media index rows, empty when the media root is undeclared."""
    if not ds.has_root(ds.resolve_media_root()):
        return []
    return ds.read_media_index()


def _frame_rate(cell: str) -> float | None:
    """Return a media index ``fps`` cell as a positive finite rate, else ``None``."""
    try:
        rate = float(cell)
    except ValueError:
        return None
    return rate if math.isfinite(rate) and rate > 0 else None


def dataset_next_steps(
    ds: Dataset, *, converted_tracks: int | None = None
) -> tuple[str, ...]:
    """What *ds* needs before a pipeline reads its tracks or labels.

    A dataset that is scanned and not converted has no entries, and a plan over
    it covers none, so it reads as empty. Each step names what is missing:
    a tracks source to declare, sources to scan, raw files to convert.

    Args:
        ds: The dataset. Never written to.
        converted_tracks: The tracks tables that exist, when the caller has
            counted them.
    """
    if converted_tracks is None:
        converted_tracks = len(existing_tracks_records(ds, read_tracks_index(ds)))
    steps: list[str] = []
    sources = ds.manifest.sources
    raw_tracks = _raw_rows(ds, "tracks_raw")
    declared = sources.media or sources.tracks or sources.labels
    if converted_tracks == 0:
        if not raw_tracks and not declared:
            steps.append(
                "declare a source of raw tracks files (mosaic sources add, or "
                "manage_dataset add_source), then scan it"
            )
        elif sources.tracks and not raw_tracks:
            steps.append(
                "scan the declared tracks sources (mosaic scan, or manage_dataset "
                "scan): the raw tracks index is empty"
            )
        elif raw_tracks:
            steps.append(
                f"convert the {raw_tracks} raw tracks files: begin a recipe with a "
                "convert-tracks step and read its output with tracks: {step: <id>}"
            )
    files = [source for source in sources.labels if source.series is None]
    series = [source.series for source in sources.labels if source.series is not None]
    raw_labels = _raw_rows(ds, "labels_raw")
    unscanned = (files and not raw_labels) or any(
        not _series_rows(ds, name) for name in series
    )
    if unscanned:
        steps.append(
            "scan the declared labels sources (mosaic scan, or manage_dataset "
            "scan): a raw labels index is empty"
        )
    if files and raw_labels:
        if not _has_converted_labels(ds):
            steps.append(
                f"convert the {raw_labels} raw labels files with a convert-labels "
                "step, and read them with labels: {step: <id>}"
            )
    return tuple(steps)


def raw_label_rows(ds: Dataset) -> int:
    """The raw labels index rows: the uploaded files' and every series'."""
    names = {source.series for source in ds.manifest.sources.labels if source.series}
    return _raw_rows(ds, "labels_raw") + sum(
        _series_rows(ds, name) for name in sorted(names)
    )


def _series_rows(ds: Dataset, series: str) -> int:
    """The rows of one label series' index, 0 when it is absent."""
    if not ds.has_root("labels_raw"):
        return 0
    return len(read_tracks_raw_index(series_index_path(ds, series)))


def _raw_rows(ds: Dataset, root: str) -> int:
    """The rows of a raw index, 0 when its root is undeclared or it is absent."""
    if not ds.has_root(root):
        return 0
    return len(read_tracks_raw_index(ds.get_root(root) / "index.csv"))


def _has_converted_labels(ds: Dataset) -> bool:
    """Whether any labels kind has a converted table in its index."""
    return any(
        len(read_labels_index(ds, kind)) > 0
        for kind in root_subdirectories(ds, "labels", require_index=True)
    )
