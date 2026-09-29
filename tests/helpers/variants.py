"""Write media variant files and their index rows, as the ``preprocess`` op does.

A consumer of a variant's row reads its path, its placement and its compositions,
and does not read its pixels. The file here is therefore a placeholder. The row is
built by the index's builder from facts that a test can state, and a test that is
not about encoding does not run the encoder.

A consumer reads the originals index, the variant index and the recorded
compositions once per call over its whole scope. :func:`count_index_reads`
counts those reads for a test to assert.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
import pytest
from mosaic_media import MediaFacts, MediaProbeError

from mosaic.core.dataset import Dataset, ResolvedScopeEntry
from mosaic.core.entry import Entry
from mosaic.core.media.facts_columns import store_facts
from mosaic.core.media.preprocess import Placement
from mosaic.core.pipeline import preprocess_index, sequence_index
from mosaic.core.pipeline.preprocess_index import (
    MediaVariantRow,
    media_variant_index,
    build_media_variant_row,
    write_media_variant_row,
)
from mosaic.core.pipeline.preprocess_layout import (
    media_variant_index_path,
    media_variant_path,
)

_SIZE = (64, 48)
_FRAMES = 12
_FPS = 30.0


def add_media_variant(
    ds: Dataset,
    run_id: str,
    sequence: str,
    *,
    group: str = "",
    camera: str = "",
    composition: str = "",
    upstream: str = "",
    placement: Placement | None = None,
    facts: MediaFacts | None = None,
) -> MediaVariantRow:
    """Write one entry's variant file and then its row, the order that the op uses.

    The file contains the bytes ``b"variant"``. *composition* is recorded as the
    entry's media composition when the file was written, and an empty one as not
    establishable.

    Args:
        ds: The dataset.
        run_id: The variant.
        sequence: The entry's sequence.
        group: The entry's group.
        camera: The camera that the variant was made from.
        composition: The entry's media composition to record.
        upstream: The variant that this one was made from, or ``""``.
        placement: The file's placement in its source, or the identity
            placement of a small file when not given.
        facts: The file's probed facts. When not given, the facts of that
            small file, naming a ``video_uuid`` unique to the run, entry and
            camera.

    Returns:
        The row written.
    """
    path = media_variant_path(ds, run_id, group, sequence, camera)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(b"variant")
    width, height = _SIZE
    row = build_media_variant_row(
        ds,
        path=path,
        run_id=run_id,
        group=group,
        sequence=sequence,
        camera=camera,
        upstream=upstream,
        upstream_video_uuid="",
        placement=(
            Placement.identity(width, height, _FRAMES, _FPS)
            if placement is None
            else placement
        ),
        facts=(
            store_facts(
                width,
                height,
                _FPS,
                _FRAMES,
                "av1",
                _FRAMES / _FPS,
                f"uuid-{run_id}-{group}-{sequence}-{camera}",
                "",
            )
            if facts is None
            else facts
        ),
        encoder="libsvtav1",
        consumed_media_composition=composition,
    )
    write_media_variant_row(ds, row)
    return row


def finish_media_variant(ds: Dataset, run_id: str) -> None:
    """Record that variant *run_id* finished, as the op does when its run ends."""
    media_variant_index(media_variant_index_path(ds)).mark_finished(run_id)


@dataclass
class IndexReads:
    """Counts of the reads of each index that a media variant's consumers read.

    Attributes:
        media_scopes: Resolutions of a media scope, each a read of the
            originals index.
        variant_indexes: Reads of the media variant index.
        compositions: Reads of the recorded media compositions.
    """

    media_scopes: int = 0
    variant_indexes: int = 0
    compositions: int = 0


def count_index_reads(monkeypatch: pytest.MonkeyPatch) -> IndexReads:
    """Count every read of the three indexes from now on, and return the counts."""
    reads = IndexReads()
    resolve_media_scope = Dataset.resolve_media_scope
    read_variants = preprocess_index.read_media_variant_index
    read_compositions = sequence_index.read_entry_compositions

    def scope(
        ds: Dataset,
        entries: Iterable[Entry] | None,
        index_filename: str = "index.csv",
        *,
        errors: dict[Entry, MediaProbeError] | None = None,
    ) -> list[ResolvedScopeEntry]:
        reads.media_scopes += 1
        return resolve_media_scope(ds, entries, index_filename, errors=errors)

    def variants(ds: Dataset) -> pd.DataFrame:
        reads.variant_indexes += 1
        return read_variants(ds)

    def compositions(
        ds: Dataset, entries: Iterable[tuple[str, str]]
    ) -> dict[tuple[str, str], dict[str, str]]:
        reads.compositions += 1
        return read_compositions(ds, entries)

    monkeypatch.setattr(Dataset, "resolve_media_scope", scope)
    monkeypatch.setattr(preprocess_index, "read_media_variant_index", variants)
    monkeypatch.setattr(sequence_index, "read_entry_compositions", compositions)
    return reads
