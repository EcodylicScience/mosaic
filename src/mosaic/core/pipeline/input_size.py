"""The size of an entry's input, read from its input files' footers.

Rows are the proxy a run's memory estimate is learned against
(:class:`~mosaic.core.pipeline.admission.RowCostEstimate`), so the run, the order its
entries start in and a report of a step's input size all count them by this one
rule: the footer rows of the entry's own input files, its neighbors left out. No
table is read. A file whose footer cannot be read counts nothing.

The other covariates of an entry's work (ids and pair frames per frame, frames,
keypoints, frame rate) come from its tracks index row and the media index, so no
table is opened for them either. A covariate that cannot be read is ``None``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mosaic.core.entry import Entry

from ._utils import ResolvedScope
from .manifest import Manifest, ManifestEntry, tracks_variants_read
from .parquet_reader import ParquetUnreadable, read_footer
from .tracks_index import (
    AmbiguousTracksVariant,
    CellRow,
    count_cell,
    read_frame_extent,
    read_ids_per_frame,
    read_pair_frames,
    read_tracks_index,
    records_no_ids,
    rows_in_scope,
    select_variant_rows,
)

if TYPE_CHECKING:
    import pandas as pd

    from mosaic.core.dataset import Dataset

__all__ = [
    "NO_COVARIATES",
    "EntryCovariates",
    "EntryInputSize",
    "by_rows",
    "entry_covariates",
    "entry_input_size",
    "manifest_input_sizes",
    "step_tracks_variants",
    "tracks_covariates",
]


@dataclass(frozen=True, slots=True)
class EntryInputSize:
    """How much one entry reads.

    Attributes:
        rows: The footer rows summed over the entry's input files, neighbors
            left out.
        columns: The columns summed over the same files.
    """

    rows: int
    columns: int


def entry_input_size(entry: ManifestEntry) -> EntryInputSize:
    """Return the rows and columns of *entry*'s input files, from their footers.

    Args:
        entry: The manifest entry. Its neighbors' files are not counted.

    Returns:
        The sums over its readable files. An unreadable file adds nothing.
    """
    rows = 0
    columns = 0
    for path, _spec in entry.file_specs:
        try:
            footer = read_footer(path)
        except ParquetUnreadable:
            continue
        rows += footer.n_rows
        columns += len(footer.columns)
    return EntryInputSize(rows=rows, columns=columns)


def manifest_input_sizes(manifest: Manifest) -> dict[str, EntryInputSize]:
    """Return :func:`entry_input_size` of every entry, keyed as *manifest* is."""
    return {key: entry_input_size(entry) for key, entry in manifest.items()}


def by_rows(rows: Mapping[Entry, int]) -> list[Entry]:
    """Return the entries of *rows* largest first, ties by ``(group, sequence)``."""
    return sorted(rows, key=lambda entry: (-rows[entry], entry))


@dataclass(frozen=True, slots=True)
class EntryCovariates:
    """What one entry's work scales with, read from the indexes.

    Attributes:
        ids_per_frame: Distinct non-null ``(frame, id)`` pairs over the frames
            of the entry's extent, or ``None``.
        pair_frames: Ordered pairs summed over the frames, or ``None``. When
            ``pair_frames_bound`` is true, the lower bound that the rows and
            frames give.
        pair_frames_bound: Whether ``pair_frames`` is that lower bound, because
            the tracks row records no pair count.
        frames: ``frame_max - frame_min + 1`` of the tracks row, or ``None``.
        keypoints: ``n_keypoints`` of the tracks row, or ``None``.
        fps: The frame rate that the entry's indexed media measured, or
            ``None``.
    """

    ids_per_frame: float | None
    pair_frames: int | None
    pair_frames_bound: bool
    frames: int | None
    keypoints: int | None
    fps: float | None


NO_COVARIATES = EntryCovariates(
    ids_per_frame=None,
    pair_frames=None,
    pair_frames_bound=False,
    frames=None,
    keypoints=None,
    fps=None,
)
"""The covariates of an entry whose tracks row cannot be read."""


def entry_covariates(row: CellRow, *, fps: float | None) -> EntryCovariates:
    """Return the covariates of the entry that tracks index row *row* records.

    A row that records no pair count reads its lower bound: with ``m`` the rows
    per frame, ``frames * m * (m - 1)`` is below the pair frames because
    ``n * (n - 1)`` is convex, clamped at 0, and ``ids_per_frame`` reads ``m``.
    A row that records a table without ids reads neither, and no bound.

    Args:
        row: The entry's tracks index row.
        fps: The entry's measured frame rate, or ``None``.
    """
    extent = read_frame_extent(row)
    frames = None if extent is None else extent[1] - extent[0] + 1
    pair_frames = read_pair_frames(row)
    ids_per_frame = read_ids_per_frame(row)
    bound = False
    rows = count_cell(row.get("n_rows"))
    if (
        pair_frames is None
        and not records_no_ids(row)
        and rows is not None
        and frames is not None
        and frames > 0
    ):
        pair_frames = max(0, rows * (rows - frames) // frames)
        bound = True
        if ids_per_frame is None:
            ids_per_frame = rows / frames
    return EntryCovariates(
        ids_per_frame=ids_per_frame,
        pair_frames=pair_frames,
        pair_frames_bound=bound,
        frames=frames,
        keypoints=count_cell(row.get("n_keypoints")),
        fps=fps,
    )


def tracks_covariates(
    ds: Dataset,
    entries: Collection[Entry],
    *,
    tracks_run_id: str | None,
    variants: Collection[str] | None,
) -> dict[Entry, EntryCovariates]:
    """Return the covariates of each of *entries*, reading the tracks index once.

    Nothing is written and no table is opened.

    Args:
        ds: The dataset, which may be opened read-only.
        entries: The entries to read.
        tracks_run_id: The tracks variant to read, or ``None`` for the variant
            each entry resolves to (:func:`select_variant_rows`).
        variants: Under ``tracks_run_id=None``, resolve each entry among these
            variants and its unlabeled rows, or among every variant for
            ``None``. An empty collection names the unlabeled rows alone.

    Returns:
        Each entry's covariates. An entry with no row, or with two variants that
        neither *tracks_run_id* nor *variants* choose between, reads
        :data:`NO_COVARIATES`.
    """
    rows = read_tracks_index(ds)
    if tracks_run_id is None and variants is not None:
        rows = rows[rows["run_id"].astype(str).isin({*variants, ""})]
    rows = rows_in_scope(rows, set(entries))
    chosen = _chosen_rows(rows, tracks_run_id)
    found: dict[Entry, EntryCovariates] = {}
    for _, series in chosen.iterrows():
        entry = (str(series["group"]), str(series["sequence"]))
        found[entry] = entry_covariates(series, fps=ds.measured_entry_fps(*entry))
    return {entry: found.get(entry, NO_COVARIATES) for entry in entries}


def _chosen_rows(rows: pd.DataFrame, tracks_run_id: str | None) -> pd.DataFrame:
    """Reduce *rows* to one per entry, leaving out each entry that is ambiguous."""
    while True:
        try:
            return select_variant_rows(rows, tracks_run_id)
        except AmbiguousTracksVariant as exc:
            group, sequence = exc.entry
            rows = rows[
                (rows["group"].astype(str) != group)
                | (rows["sequence"].astype(str) != sequence)
            ]


def step_tracks_variants(
    ds: Dataset, resolved: ResolvedScope
) -> tuple[str, ...] | None:
    """Return the tracks variants that a step's inputs read, sorted.

    The variants a ``tracks`` input resolved to, joined with the variants each
    feature input's upstream chain read. An empty tuple names the unlabeled
    tables.

    Args:
        ds: The dataset holding the upstream runs.
        resolved: The step's resolved scope.

    Returns:
        The variants, or ``None`` when an upstream's variants are unknown, which
        leaves every variant in play.
    """
    read = tracks_variants_read(ds, resolved.tracks_variants, resolved.feature_inputs)
    return None if read is None else tuple(sorted(read))
