"""Record one row per media variant file, which a consumer reads instead of the file.

``media/preprocess/index.csv`` records, per variant, entry and camera, the facts
that a consumer must otherwise measure or cannot measure at all:

- the placement that maps the file's pixels and frames back to the entry's
  source, composed through every upstream variant. Map-back reads this one row.
- the file's probed facts, stored the way the media index stores them. A reader
  is given them and does not probe the file again.
- the composition of the entry's media when the file was written, compared
  against the current composition to tell a current variant from a drifted one.

:func:`write_media_variant_row` is the one writer and
:func:`read_media_variant_index` the one reader.
:mod:`mosaic.core.pipeline.preprocess_layout` names the paths of the files and the
index.
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
    PREPROCESS_KIND,
    media_variant_index_path,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.entry import CameraEntry

__all__ = [
    "MediaVariantDriftedError",
    "MediaVariantMissingError",
    "MediaVariantRow",
    "build_media_variant_row",
    "media_variant_facts",
    "media_variant_index",
    "media_variant_placement",
    "media_variant_rows",
    "read_media_variant_index",
    "write_media_variant_row",
]


class MediaVariantMissingError(FileNotFoundError):
    """An entry lacks a file of the variant that a step asked to read.

    It is a ``FileNotFoundError`` like the missing-export errors that a tracker
    raises, because the remedy is of the same kind: run the op that writes the
    file.
    """


class MediaVariantDriftedError(ValueError):
    """An entry's variant file depicts media that the entry no longer has.

    The file exists and is readable, and the error is not a missing file. It is a
    recorded value that disagrees with the current one, and its remedy is to
    rewrite the variant instead of writing a missing one.
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
        camera: The camera of the entry that the variant was made from, or
            ``""``.
        upstream: The variant that this one was made from, or ``""`` when it was
            made from the entry media.
        upstream_video_uuid: The upstream file's ``video_uuid`` when it was read,
            or ``""``.
        placement: The file's placement in the entry's source, as the canonical
            JSON that
            :meth:`~mosaic.core.media.preprocess.geometry.Placement.to_json`
            writes.
        width: The file's coded width.
        height: The file's coded height.
        fps: The file's frame rate.
        codec: The file's codec.
        consumed_media_composition: The entry's media composition when the file
            was written. It is compared and not hashed. Blank means not
            establishable.
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
    """Project *frame* onto the current columns, for an index written earlier."""
    return project_to_schema(frame, _COLUMNS)


def media_variant_index(path: Path) -> IndexCSV[MediaVariantRow]:
    """Return the variant index at *path*, one row per variant, entry and camera."""
    return IndexCSV(
        path,
        MediaVariantRow,
        dedup_keys=["run_id", "group", "sequence", "camera"],
        adopt=_adopt,
    )


def build_media_variant_row(
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
    """Return the row that records variant file *path*, built from its *facts*.

    The facts columns come from the builder that the media index uses. A consumer
    rebuilds *facts* from the row with :func:`media_variant_facts`.
    Every argument is keyword-only, because several are strings, and a type check
    does not catch two transposed strings.
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

    The write takes the index lock, and two entries that finish at once are both
    recorded. Call it after the file is in place, because a row asserts that the
    file exists.

    Raises:
        ValueError: If the entry's group or sequence is not one path component.
    """
    _ = validate_entry_name(row.group, "group")
    _ = validate_entry_name(row.sequence, "sequence")
    media_variant_index(media_variant_index_path(ds)).append([row])


def read_media_variant_index(ds: Dataset) -> pd.DataFrame:
    """Return every variant row of *ds*, in the current columns.

    An absent index reads as an empty one with every column. The read does not
    write, and an index written before a column existed stays as it is.
    """
    path = media_variant_index_path(ds)
    frame = media_variant_index(path).read() if path.exists() else pd.DataFrame()
    return project_to_schema(frame, _COLUMNS)


def media_variant_rows(ds: Dataset, run_id: str) -> dict[CameraEntry, dict[str, str]]:
    """Return every row of variant *run_id*, keyed by entry and camera, in one read.

    A caller that looks up many entries reads the index here once, rather than
    once per entry.
    """
    frame = read_media_variant_index(ds)
    return {
        (record["group"], record["sequence"], record["camera"]): record
        for record in index_records(frame[frame["run_id"] == run_id])
    }


def media_variant_placement(row: Mapping[str, str]) -> Placement:
    """Return the placement of the file of *row* in its entry's source."""
    return Placement.from_json(row["placement"])


def media_variant_facts(row: Mapping[str, str]) -> MediaFacts:
    """Return the probed facts of the file of *row*, rebuilt without probing it."""
    return row_to_facts(row)


# The dataset-wide passes read every registered index. Reindex drops a row whose
# file is gone, and the portability passes rewrite its paths.
register_reconcilable_index(PREPROCESS_KIND, media_variant_index)
