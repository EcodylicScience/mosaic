"""What the inventory says about versioned label series.

One record per ``(series, origin, key)``: an annotation set, say, with the
revisions the index holds as its coverage target and the ones whose payload still
resolves as what is present. That is the shape the question takes -- "which
versions of this set does the dataset hold, and are they all there" -- and it is
why the coverage key is the revision rather than an entry.

A library's record looks the same as a project's. Its revisions resolve into
other datasets, so a missing one reads here as a row with no file, which is what
a contributing project being archived or unmounted looks like from the library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.label_series import LABEL_SERIES, SERIES_MARKER
from mosaic.core.pipeline.label_series_index import (
    read_label_series,
    series_index_path,
)

from ._read import IndexReader
from .model import (
    ArtifactRecord,
    Coverage,
    InventoryScope,
    LabelSeriesRef,
    classify,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = ["label_series_records"]


def label_series_records(
    ds: Dataset, scope: InventoryScope, reader: IndexReader
) -> list[ArtifactRecord[str]]:
    """Every key of every label series *ds* holds, with its revisions.

    ``scope`` does not narrow this. A keypoint set spans sequences, so filtering
    by entry would drop every set from a scoped query rather than report the ones
    that exist.
    """
    _ = scope
    if not ds.has_root("labels_raw"):
        return []
    records: list[ArtifactRecord[str]] = []
    for series in sorted(LABEL_SERIES):
        index_path = series_index_path(ds, series)
        if not (index_path.parent / SERIES_MARKER).is_file():
            continue
        frame = reader.frame(index_path, lambda s=series: read_label_series(ds, s))
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in index_records(frame):
            owner = (row.get("origin_uuid", ""), row.get("key", ""))
            grouped.setdefault(owner, []).append(row)
        for (origin_uuid, key), rows in sorted(grouped.items()):
            target = frozenset(row.get("revision", "") for row in rows)
            present = frozenset(
                row.get("revision", "")
                for row in rows
                if row.get("abs_path", "")
                and ds.resolve_path(row["abs_path"]).is_file()
            )
            missing = target - present
            newest = max(rows, key=lambda row: row.get("exported_at", ""))
            records.append(
                ArtifactRecord[str](
                    ref=LabelSeriesRef(series=series, key=key, origin_uuid=origin_uuid),
                    name=series,
                    run_id="",
                    coverage=Coverage(target=target, present=present),
                    status=classify(
                        satisfied=not missing,
                        any_covered=bool(present),
                        orphan_rows=bool(missing),
                        orphan_files=False,
                        drifted=False,
                        # A revision is written whole before its row is appended,
                        # so there is no in-progress state for a row to be in.
                        finished=True,
                    ),
                    run_root=index_path.parent / key
                    if origin_uuid == (ds.uuid or "")
                    else None,
                    index_path=index_path,
                    rows=target,
                    orphan_rows=missing,
                    finished_at=newest.get("exported_at", ""),
                )
            )
    return records
