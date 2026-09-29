"""Report media coverage: transcode derivatives, and the variants of ``preprocess``.

Every other artifact is addressed by a run identifier naming a directory, and
its coverage is which outputs that directory holds. Transcode is not.
``transcode_run_id`` says so of itself -- "This value addresses nothing. It names
no directory and gates no reuse" -- because the output is named by its recipe and
reuse is gated by that filename plus the forward link on the source row.

So its coverage is a property of the **media index**: which in-scope rows can be
read for this target, either because they need no derivative or because the
derivative they need is registered and present. Nothing else needs to exist.

**This is the case a single coverage signature gets wrong**, and it gets it wrong
in the worst direction. Asked for a run directory that was never supposed to
exist, a directory-shaped check reports zero of N -- so a corpus that is entirely
clean, with nothing to transcode and nothing missing, reads as permanently
incomplete. Anything acting on that resubmits the same work every tick, forever.

Media variants are the opposite case: a run directory per variant, one file per
entry and camera, and an index that records each file's row. A variant is keyed
and reported the way a frame run is.

**Built in ``core`` rather than through the contributor registry**, unlike the
ops kinds. The registry exists to carry what lives above the layering line;
the media index, its verdict columns and the variant index belong to ``core``. A
registration for them adds a seam without a boundary behind it, and makes the
kinds unavailable to a core-only caller.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.core.entry import CameraEntry, Entry
from mosaic.core.media.facts_columns import (
    derivative_path_for_target,
    media_row_uuid,
    read_link_cell,
    transcode_required,
)
from mosaic.core.pipeline.composition import composition_drift
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.media_index import read_media_index
from mosaic.core.pipeline.preprocess_index import read_media_variant_index
from mosaic.core.pipeline.preprocess_layout import (
    MEDIA_ROOT_KEY,
    PREPROCESS_KIND,
    media_variant_index_path,
    media_variant_path,
    media_variant_run_root,
)
from mosaic.core.pipeline.sequence_index import media_compositions_for

from ._runs import finish_state, run_ids
from .model import (
    ArtifactRecord,
    Coverage,
    InventoryScope,
    MediaDerivativeRef,
    MediaVariantRef,
    Target,
    classify,
)

if TYPE_CHECKING:
    import pandas as pd

    from ._read import IndexReader
    from mosaic.core.dataset import Dataset

__all__ = [
    "media_derivative_record",
    "media_variant_records",
]


def _media_index_path(ds: Dataset, reader: IndexReader) -> Path | None:
    """Return the originals index, noted on *reader*, or ``None`` without media.

    ``resolve_media_root`` falls back to ``"media"`` when ``media_raw`` is unset,
    and returns that name whether or not ``media`` is set either. A tracks-only
    dataset, which declares both roots and fills neither, therefore names a root
    that ``get_root`` refuses. Such a dataset cannot be short of media, and it
    reads as having none instead of raising out of a read.
    """
    root_key = ds.resolve_media_root()
    if not ds.has_root(root_key):
        return None
    index_path = ds.get_root(root_key) / "index.csv"
    reader.note(index_path)
    return index_path


def _nothing_to_cover(target: Target) -> ArtifactRecord[str]:
    """The record for a dataset holding no media at all: nothing is missing."""
    coverage = Coverage[str](target=frozenset(), present=frozenset())
    return ArtifactRecord[str](
        ref=MediaDerivativeRef(target=target),
        name=f"transcode:{target}",
        run_id="",
        coverage=coverage,
        status="absent",
        extra={"needs_transcode": frozenset(), "needs_probe": frozenset()},
    )


def media_derivative_record(
    ds: Dataset, target: Target, scope: InventoryScope, reader: IndexReader
) -> ArtifactRecord[str]:
    """Whether every in-scope media row can be read for *target*.

    Keyed on ``video_uuid``, which is what a derivative links back by and what
    survives a rename. A row carrying none -- an imgstore, or a row not yet
    probed -- falls back to its stored path so it is still nameable rather than
    silently dropped from the target.

    Two remedies are reported separately rather than as one "incomplete" count,
    mirroring the two textually distinct errors the read path already raises: a
    row needing a transcode wants ``mosaic run --kind transcode``, and a row with
    no reconstructable measurement wants ``mosaic reprobe-media``. Collapsing
    them would tell a user their corpus is short without saying what to do.
    """
    index_path = _media_index_path(ds, reader)
    if index_path is None:
        return _nothing_to_cover(target)
    # Derivatives are anchored under the ``media`` root. Without one, nothing can
    # be registered, so a row needing a transcode reads as needing it still.
    media_root = ds.get_root("media") if ds.has_root("media") else None

    covered: set[str] = set()
    target_keys: set[str] = set()
    needs_transcode: set[str] = set()
    needs_probe: set[str] = set()
    wanted = scope.selector.entry_pairs

    for row in read_media_index(index_path):
        if read_link_cell(row, "media_type") == "imgstore":
            # A store is read natively and has no elementary stream to transcode,
            # so it is not a row this coverage can be short of.
            continue
        entry = (read_link_cell(row, "group"), read_link_cell(row, "sequence"))
        if wanted is not None and entry not in wanted:
            continue
        key = media_row_uuid(row) or read_link_cell(row, "abs_path")
        if not key:
            continue
        target_keys.add(key)

        if transcode_required(row, target):
            linked = (
                derivative_path_for_target(row, target, media_root)
                if media_root is not None
                else None
            )
            # Both halves, matching the reuse gate the transcode op itself
            # applies: the link records the registration and the file is the
            # output. Registration writes the back-link row first and the
            # forward link last, so an unlinked file is the recoverable state
            # and reads here as still needing the work.
            if linked is not None and linked.exists():
                covered.add(key)
            else:
                needs_transcode.add(key)
            continue

        if read_link_cell(row, "media_facts"):
            covered.add(key)
        else:
            needs_probe.add(key)

    coverage = Coverage(target=frozenset(target_keys), present=frozenset(covered))
    return ArtifactRecord[str](
        ref=MediaDerivativeRef(target=target),
        name=f"transcode:{target}",
        run_id="",
        coverage=coverage,
        status=classify(
            satisfied=coverage.is_satisfied,
            any_covered=bool(coverage.covered),
            orphan_rows=False,
            orphan_files=False,
            drifted=False,
            finished=True,
        ),
        index_path=index_path,
        rows=frozenset(target_keys),
        extra={
            "needs_transcode": frozenset(needs_transcode),
            "needs_probe": frozenset(needs_probe),
        },
    )


def media_variant_records(
    ds: Dataset, scope: InventoryScope, reader: IndexReader
) -> list[ArtifactRecord[CameraEntry]]:
    """Return each media variant's coverage, keyed by ``(group, sequence, camera)``.

    The function returns one record per variant that the index names. A variant
    covers the entries that its rows name and the variant files found beside them.
    A variant is legitimately made for a subset. Measured against the dataset's
    whole universe, a finished variant reads as short.

    An entry is covered when it has both a row and a file. The op renames a file
    into place and then writes its row, and a consumer reads the row. A file ahead
    of its row is therefore not yet usable. On a run still writing that is
    progress and reads as partial, and on a finished run it is damage. A file is
    looked for where
    :func:`~mosaic.core.pipeline.preprocess_layout.media_variant_path` puts one,
    for every entry and camera that the run's rows or the media index name. An
    entry's claim and its encode in progress are under ``.work/``, outside every
    path that ``media_variant_path`` returns.

    A row whose recorded media composition differs from the entry's current one
    is drift, under the rule that a blank on either side is not.

    The media index and the entries' current compositions are read once for
    every variant rather than once for each, and are not read when the index does
    not name a variant. A dataset without a media root cannot contain a variant,
    because every variant is stored under that root.

    Args:
        ds: The dataset. Read only.
        scope: The inventory request. Its selector narrows the entries
            reported.
        reader: This scan's view of the index files.

    Returns:
        One record per variant.
    """
    if not ds.has_root(MEDIA_ROOT_KEY):
        return []
    frame = _variant_index(ds, reader)
    variants = run_ids(frame)
    if not variants:
        return []
    wanted = scope.selector.entry_pairs
    rows = _wanted_rows(frame, wanted)
    media_entries = _media_entries(ds, wanted, reader)
    compositions = media_compositions_for(ds, _entries_of(rows))
    return [
        _variant_record(
            ds,
            frame,
            run_id,
            [record for record in rows if record.get("run_id", "") == run_id],
            media_entries,
            compositions,
        )
        for run_id in variants
    ]


def _variant_record(
    ds: Dataset,
    frame: pd.DataFrame,
    run_id: str,
    run_rows: list[dict[str, str]],
    media_entries: frozenset[CameraEntry],
    compositions: Mapping[Entry, str],
) -> ArtifactRecord[CameraEntry]:
    """Return the record of variant *run_id*, built from its rows and shared reads.

    Args:
        ds: The dataset.
        frame: The whole variant index, for when the run started and finished.
        run_id: The variant.
        run_rows: The variant's rows within the scope.
        media_entries: Every entry and camera that the media index names within
            the scope. A file is looked for at each, beside the run's rows.
        compositions: The current media composition of each entry that the rows
            name.
    """
    rows = frozenset(
        (
            record.get("group", ""),
            record.get("sequence", ""),
            record.get("camera", ""),
        )
        for record in run_rows
    )
    files = frozenset(
        key
        for key in rows | media_entries
        if media_variant_path(ds, run_id, *key).is_file()
    )
    coverage = Coverage(target=rows | files, present=rows & files)
    drift = composition_drift(
        {
            (record.get("group", ""), record.get("sequence", "")): record.get(
                "consumed_media_composition", ""
            )
            for record in run_rows
        },
        compositions,
    )
    started_at, finished_at, finished = finish_state(frame, run_id)
    return ArtifactRecord[CameraEntry](
        ref=MediaVariantRef(run_id=run_id),
        name=PREPROCESS_KIND,
        run_id=run_id,
        coverage=coverage,
        status=classify(
            satisfied=coverage.is_satisfied,
            any_covered=bool(coverage.covered),
            orphan_rows=bool(rows - files),
            orphan_files=bool(files - rows),
            drifted=bool(drift),
            finished=finished,
        ),
        run_root=media_variant_run_root(ds, run_id),
        index_path=media_variant_index_path(ds),
        rows=rows,
        orphan_rows=rows - files,
        orphan_files=files - rows,
        drift=drift,
        started_at=started_at,
        finished_at=finished_at,
        upstreams=tuple(
            sorted({record.get("upstream", "") for record in run_rows} - {""})
        ),
    )


def _variant_index(ds: Dataset, reader: IndexReader) -> pd.DataFrame:
    """Return the variant index, read once per scan through *reader*."""
    return reader.frame(
        media_variant_index_path(ds), lambda: read_media_variant_index(ds)
    )


def _wanted_rows(
    frame: pd.DataFrame, wanted: set[Entry] | None
) -> list[dict[str, str]]:
    """Return the rows of *frame* whose entry is in *wanted*, or every row if unset."""
    return [
        record
        for record in index_records(frame)
        if wanted is None
        or (record.get("group", ""), record.get("sequence", "")) in wanted
    ]


def _entries_of(rows: list[dict[str, str]]) -> set[Entry]:
    """Return the ``(group, sequence)`` pairs that *rows* name."""
    return {(record.get("group", ""), record.get("sequence", "")) for record in rows}


def _media_entries(
    ds: Dataset, wanted: set[Entry] | None, reader: IndexReader
) -> frozenset[CameraEntry]:
    """Return each entry and camera in *wanted* that the media index names."""
    index_path = _media_index_path(ds, reader)
    if index_path is None:
        return frozenset()
    found: set[CameraEntry] = set()
    for row in read_media_index(index_path):
        group = read_link_cell(row, "group")
        sequence = read_link_cell(row, "sequence")
        if wanted is None or (group, sequence) in wanted:
            found.add((group, sequence, read_link_cell(row, "camera")))
    return frozenset(found)
