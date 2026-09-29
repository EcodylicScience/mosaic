"""Media variant files and their index rows, as the ``preprocess`` op leaves them.

What reads a variant's row reads its path, its placement and its compositions,
never its pixels. So the file here is a placeholder, and the row is built by the
index's own builder from facts a test can state, which keeps a test off the
encoder when encoding is not what it is about.
"""

from __future__ import annotations

from mosaic_media import MediaFacts

from mosaic.core.dataset import Dataset
from mosaic.core.media.facts_columns import store_facts
from mosaic.core.media.preprocess import Placement
from mosaic.core.pipeline.preprocess_index import (
    MediaVariantRow,
    media_variant_index,
    media_variant_row,
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
    """Write one entry's variant file and then its row, the order the op uses.

    The file holds the bytes ``b"variant"``. *composition* is recorded as the
    entry's media composition when the file was written, and an empty one as not
    establishable.

    Args:
        ds: The dataset.
        run_id: The variant.
        sequence: The entry's sequence.
        group: The entry's group.
        camera: The camera the variant was made from.
        composition: The entry's media composition to record.
        upstream: The variant this one was made from, or ``""``.
        placement: Where the file sits in its source. The identity placement
            of a small file when not given.
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
    row = media_variant_row(
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
