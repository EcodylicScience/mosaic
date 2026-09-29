"""The media variant index: one row per variant file, read instead of the file.

``media/preprocess/index.csv`` records, per variant, entry and camera, what a
consumer would otherwise have to measure or cannot measure at all:

- the placement mapping the file's pixels and frames back to the entry's source,
  composed through every upstream variant, so map-back reads one row;
- the file's probed facts, stored the way the media index stores them, so a
  reader is given them rather than probing the file again;
- what the entry's media was when the file was written, compared against the
  current composition to tell a current variant from a drifted one.

One writer, :func:`write_media_variant_row`, and one reader,
:func:`read_media_variant_index`. Where the files and the index sit is
:mod:`mosaic.core.pipeline.preprocess_layout`'s to say.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd
from mosaic_media import MediaFacts

from mosaic.core.helpers import validate_entry_name
from mosaic.core.media.facts_columns import row_to_facts
from mosaic.core.media.preprocess.geometry import Placement
from mosaic.core.media.probe_row import row_from_facts
from mosaic.core.pipeline.dataset_indexes import register_reconcilable_index
from mosaic.core.pipeline.index_csv import (
    IndexCSV,
    RunIndexRowBase,
    index_records,
    project_to_schema,
)
from mosaic.core.pipeline.preprocess_layout import (
    PREPROCESS_KIND_DIRECTORY,
    media_variant_index_path,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "MediaVariantDriftedError",
    "MediaVariantMissingError",
    "MediaVariantRow",
    "media_variant_index",
    "media_variant_row",
    "read_media_variant_index",
    "variant_facts",
    "variant_placement",
    "variant_row",
    "write_media_variant_row",
]


class MediaVariantMissingError(FileNotFoundError):
    """An entry has no file of the variant a step asked to read.

    A ``FileNotFoundError`` like the missing-export errors a tracker raises,
    because the remedy is the same kind: run the op that writes the file.
    """


class MediaVariantDriftedError(ValueError):
    """An entry's variant file depicts media the entry no longer holds.

    The file exists and reads, so this is not a missing file. It is a recorded
    value that disagrees with the current one, and the remedy differs: rewrite
    the variant, rather than write one that was never there.
    """


@dataclass(frozen=True, slots=True)
class MediaVariantRow(RunIndexRowBase):
    """One variant file of one entry and camera.

    ``abs_path`` is the file, stored dataset-root-relative. The columns from
    ``frame_count`` to ``media_facts`` are the media index's facts columns
    (:data:`~mosaic.core.media.facts_columns.FACTS_COLUMNS`), filled from the
    file's probed facts, with ``encoder`` naming the encoder that wrote the file.

    Attributes:
        group: The entry's group.
        sequence: The entry's sequence.
        camera: The camera of the entry the variant was made from, or ``""``.
        upstream: The variant this one was made from, or ``""`` when it was made
            from the entry media.
        upstream_video_uuid: The upstream file's ``video_uuid`` when it was read,
            or ``""``.
        placement: The file's placement in the entry's source, as the canonical
            JSON :meth:`~mosaic.core.media.preprocess.geometry.Placement.to_json`
            writes.
        width: The file's coded width.
        height: The file's coded height.
        fps: The file's frame rate.
        codec: The file's codec.
        consumed_media_composition: The entry's media composition when the file
            was written. Compared, never hashed: blank means not establishable.
    """

    group: str
    sequence: str
    camera: str
    upstream: str
    upstream_video_uuid: str
    placement: str
    width: int
    height: int
    fps: float
    codec: str
    frame_count: int
    analysis_transcode: str
    stream_transcode: str
    analysis_derivative_path: str
    playback_derivative_path: str
    source_path: str
    video_uuid: str
    content_digest: str
    source_video_uuid: str
    recipe_hash: str
    encoder: str
    media_facts: str
    consumed_media_composition: str


_COLUMNS: Final = tuple(field.name for field in dataclasses.fields(MediaVariantRow))
"""Every column of the index, in order."""


def _adopt(frame: pd.DataFrame) -> pd.DataFrame:
    """*frame* projected onto the current columns, for an index written earlier."""
    return project_to_schema(frame, _COLUMNS)


def media_variant_index(path: Path) -> IndexCSV[MediaVariantRow]:
    """The variant index at *path*, one row per variant, entry and camera."""
    return IndexCSV(
        path,
        MediaVariantRow,
        dedup_keys=["run_id", "group", "sequence", "camera"],
        adopt=_adopt,
    )


def media_variant_row(
    ds: Dataset,
    *,
    path: Path,
    run_id: str,
    group: str,
    sequence: str,
    camera: str,
    upstream: str,
    upstream_video_uuid: str,
    placement: Placement,
    facts: MediaFacts,
    encoder: str,
    consumed_media_composition: str,
) -> MediaVariantRow:
    """The row recording variant file *path*, built from its probed *facts*.

    The facts columns come from the builder the media index uses, so a consumer
    rebuilds *facts* from the row with :func:`variant_facts`. Keyword-only
    throughout: several arguments are strings a transposition would not catch.
    """
    probe = row_from_facts(facts)
    return MediaVariantRow(
        abs_path=Path(ds.relative_to_root(path)),
        run_id=run_id,
        group=group,
        sequence=sequence,
        camera=camera,
        upstream=upstream,
        upstream_video_uuid=upstream_video_uuid,
        placement=placement.to_json(),
        width=probe["width"],
        height=probe["height"],
        fps=probe["fps"],
        codec=probe["codec"],
        frame_count=probe["frame_count"],
        analysis_transcode=probe["analysis_transcode"],
        stream_transcode=probe["stream_transcode"],
        analysis_derivative_path=probe["analysis_derivative_path"],
        playback_derivative_path=probe["playback_derivative_path"],
        source_path=probe["source_path"],
        video_uuid=probe["video_uuid"],
        content_digest=probe["content_digest"],
        source_video_uuid=probe["source_video_uuid"],
        recipe_hash=probe["recipe_hash"],
        encoder=encoder,
        media_facts=probe["media_facts"],
        consumed_media_composition=consumed_media_composition,
    )


def write_media_variant_row(ds: Dataset, row: MediaVariantRow) -> None:
    """Record *row*, replacing the row of the same variant, entry and camera.

    Under the index lock, so two entries finishing at once both land. Call it
    after the file is in place: a row is the claim that the file exists.

    Raises:
        ValueError: If the entry's group or sequence is not one path component.
    """
    _ = validate_entry_name(row.group, "group")
    _ = validate_entry_name(row.sequence, "sequence")
    media_variant_index(media_variant_index_path(ds)).append([row])


def read_media_variant_index(ds: Dataset) -> pd.DataFrame:
    """Every variant row of *ds*, in the current columns.

    An absent index reads as an empty one carrying every column. Never writes, so
    reading an index written before a column existed leaves it as it is.
    """
    path = media_variant_index_path(ds)
    frame = media_variant_index(path).read() if path.exists() else pd.DataFrame()
    return project_to_schema(frame, _COLUMNS)


def variant_row(
    ds: Dataset, run_id: str, group: str, sequence: str, camera: str
) -> dict[str, str] | None:
    """The row of variant *run_id* for one entry and camera, or ``None``."""
    for record in index_records(read_media_variant_index(ds)):
        if (
            record["run_id"] == run_id
            and record["group"] == group
            and record["sequence"] == sequence
            and record["camera"] == camera
        ):
            return record
    return None


def variant_placement(row: Mapping[str, str]) -> Placement:
    """Where the file of *row* sits in its entry's source."""
    return Placement.from_json(row["placement"])


def variant_facts(row: Mapping[str, str]) -> MediaFacts:
    """The probed facts of the file of *row*, rebuilt without probing it."""
    return row_to_facts(row)


# Registered so the dataset-wide passes see the index: reindex drops a row whose
# file is gone, and the portability passes rewrite its paths.
register_reconcilable_index(PREPROCESS_KIND_DIRECTORY, media_variant_index)
