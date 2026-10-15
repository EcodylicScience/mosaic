"""One dataset described in one read: its manifest, its counts, its schemas and frame rates.

:func:`dataset_summary` returns a :class:`DatasetSummary`. The CLI's
``mosaic describe --json`` is that model dumped.

A summary reads the manifest, the tracks index, the originals media index and
an inventory of three artifact kinds. It writes nothing, takes no lock and
creates no root.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING, Final

from mosaic.core.json_value import JsonValue
from mosaic.core.manifest import DatasetTag, LibraryLink
from mosaic.core.pipeline.inventory import (
    ArtifactKind,
    DatasetInventory,
    existing_tracks_records,
    inventory,
    reportable_universe,
)
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.schema import LEGACY_SCHEMA
from mosaic.core.strict_model import StrictModel

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "SUMMARY_KINDS",
    "DatasetCounts",
    "DatasetSummary",
    "FrameRateCount",
    "TrackSchemaCount",
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
        managed_by: The manifest's top-level ``managed_by`` value, verbatim.
        api_managed: Whether ``managed_by`` names a managing service.
        counts: Sequences, videos and artifacts.
        track_schemas: The schemas of the tracks tables, by name.
        frame_rates: The frame rates of the indexed videos, ascending. Rows
            with a blank or non-positive rate are left out.
        inventory_errors: The inventory's read failures, one per kind, each
            prefixed with its kind. That kind's count omits what failed.
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
    managed_by: JsonValue
    api_managed: bool
    counts: DatasetCounts
    track_schemas: tuple[TrackSchemaCount, ...]
    frame_rates: tuple[FrameRateCount, ...]
    inventory_errors: tuple[str, ...]


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
        continuous_groups=manifest.continuous_groups,
        managed_by=manifest.managed_by,
        api_managed=manifest.is_managed,
        counts=DatasetCounts(
            sequences=len(reportable_universe(ds)),
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
