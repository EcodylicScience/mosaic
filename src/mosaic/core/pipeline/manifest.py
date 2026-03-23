"""Unified manifest builder for all input types."""

from __future__ import annotations

import gc
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, overload

import pandas as pd
import pyarrow as pa

from ...core.helpers import make_entry_key
from ._utils import Scope
from .index import feature_index, feature_index_path, latest_feature_run_root
from .loading import load_entry_data
from .types import (
    InputsLike,
    LoadSpec,
    ParquetLoadSpec,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


FileSpecs = list[tuple[Path, LoadSpec]]

FilterFactory = Callable[[str], Iterable[Callable[[pd.DataFrame], pd.DataFrame]]]


@dataclass(slots=True)
class ManifestEntry:
    file_specs: FileSpecs = field(default_factory=list)
    prev_file_specs: FileSpecs | None = None
    prev_entry_key: str | None = None
    next_file_specs: FileSpecs | None = None
    next_entry_key: str | None = None


Manifest = dict[str, ManifestEntry]


def build_manifest(
    ds: Dataset,
    inputs: InputsLike,
    groups: set[str] | None = None,
    sequences: set[str] | None = None,
) -> tuple[Manifest, Scope]:
    """Build unified manifest for all input types.

    Returns the manifest (entry_key -> ManifestEntry) and the
    resolved Scope (entries present in ALL inputs after intersection).
    """
    per_input_entries: list[set[tuple[str, str]]] = []
    per_input_paths: list[dict[tuple[str, str], tuple[Path, LoadSpec]]] = []
    per_input_paths_all: list[dict[tuple[str, str], tuple[Path, LoadSpec]]] = []
    first_full_order: list[tuple[str, str]] = []

    for i, item in enumerate(inputs.root):
        if item == "tracks":
            entries, path_map, full_order, path_map_all = _resolve_tracks(
                ds, groups, sequences
            )
        else:
            entries, path_map, full_order, path_map_all = _resolve_feature(
                ds,
                item.feature,
                item.run_id,
                groups,
                sequences,
            )
        per_input_entries.append(entries)
        per_input_paths.append(path_map)
        per_input_paths_all.append(path_map_all)
        if i == 0:
            first_full_order = full_order

    # Intersect entries across all inputs
    shared_entries: set[tuple[str, str]] = set()
    if per_input_entries:
        shared_entries = per_input_entries[0].intersection(*per_input_entries[1:])

    scope = Scope(entries=shared_entries)

    # Build per-group ordering from first input's full order
    group_order: dict[str, list[tuple[str, str]]] = {}
    for entry in first_full_order:
        group_order.setdefault(entry[0], []).append(entry)

    # Build manifest with adjacency
    manifest: Manifest = {}
    for group, sequence in sorted(shared_entries):
        key = make_entry_key(group, sequence)
        specs: FileSpecs = []
        for path_map in per_input_paths:
            if (group, sequence) in path_map:
                specs.append(path_map[(group, sequence)])

        # Find prev/next in the same group using the full ordering
        current = (group, sequence)
        ordered = group_order.get(group, [])
        try:
            idx = ordered.index(current)
        except ValueError:
            idx = -1

        prev_entry: tuple[str, str] | None = ordered[idx - 1] if idx > 0 else None
        next_entry: tuple[str, str] | None = (
            ordered[idx + 1] if 0 <= idx < len(ordered) - 1 else None
        )

        prev_specs: FileSpecs | None = None
        prev_key: str | None = None
        if prev_entry is not None:
            prev_key = make_entry_key(prev_entry[0], prev_entry[1])
            collected: FileSpecs = []
            for path_map_all in per_input_paths_all:
                if prev_entry in path_map_all:
                    collected.append(path_map_all[prev_entry])
            prev_specs = collected if collected else None

        next_specs: FileSpecs | None = None
        next_key: str | None = None
        if next_entry is not None:
            next_key = make_entry_key(next_entry[0], next_entry[1])
            collected_next: FileSpecs = []
            for path_map_all in per_input_paths_all:
                if next_entry in path_map_all:
                    collected_next.append(path_map_all[next_entry])
            next_specs = collected_next if collected_next else None

        manifest[key] = ManifestEntry(
            file_specs=specs,
            prev_file_specs=prev_specs,
            prev_entry_key=prev_key,
            next_file_specs=next_specs,
            next_entry_key=next_key,
        )

    return manifest, scope


def _resolve_tracks(
    ds: Dataset,
    groups: set[str] | None,
    sequences: set[str] | None,
) -> tuple[
    set[tuple[str, str]],
    dict[tuple[str, str], tuple[Path, LoadSpec]],
    list[tuple[str, str]],
    dict[tuple[str, str], tuple[Path, LoadSpec]],
]:
    """Resolve track entries and paths from tracks/index.csv.

    Returns (scoped_entries, scoped_path_map, full_order, path_map_all).
    """
    tracks_root = ds.get_root("tracks")
    idx_path = tracks_root / "index.csv"
    if not idx_path.exists():
        return set(), {}, [], {}

    df = pd.read_csv(idx_path, keep_default_na=False)

    # Build full (unscoped) path map and order
    path_map_all: dict[tuple[str, str], tuple[Path, LoadSpec]] = {}
    all_entries: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        g, s = row["group"], row["sequence"]
        p = ds.resolve_path(row["abs_path"])
        if not p.exists():
            continue
        entry = (g, s)
        path_map_all[entry] = (p, ParquetLoadSpec())
        all_entries.append(entry)

    # Sort by (group, sequence) for stable ordering
    full_order = sorted(set(all_entries))

    # Filter for scoped subset
    entries: set[tuple[str, str]] = set()
    path_map: dict[tuple[str, str], tuple[Path, LoadSpec]] = {}
    for entry, spec in path_map_all.items():
        g, s = entry
        if groups and g not in groups:
            continue
        if sequences and s not in sequences:
            continue
        entries.add(entry)
        path_map[entry] = spec

    return entries, path_map, full_order, path_map_all


def _resolve_feature(
    ds: Dataset,
    feature_name: str,
    run_id: str | None,
    groups: set[str] | None,
    sequences: set[str] | None,
) -> tuple[
    set[tuple[str, str]],
    dict[tuple[str, str], tuple[Path, LoadSpec]],
    list[tuple[str, str]],
    dict[tuple[str, str], tuple[Path, LoadSpec]],
]:
    """Resolve feature result entries and paths from the feature index CSV.

    Returns (scoped_entries, scoped_path_map, full_order, path_map_all).
    """
    idx_path = feature_index_path(ds, feature_name)
    if not idx_path.exists():
        return set(), {}, [], {}

    if run_id is None:
        run_id, _ = latest_feature_run_root(ds, feature_name)

    idx = feature_index(idx_path)

    # Read full (unscoped) index for all entries
    df_all = idx.read(run_id=run_id, filter_ext=".parquet")
    path_map_all: dict[tuple[str, str], tuple[Path, LoadSpec]] = {}
    all_entries: list[tuple[str, str]] = []
    for _, row in df_all.iterrows():
        entry = (row["group"], row["sequence"])
        path_map_all[entry] = (Path(row["abs_path"]), ParquetLoadSpec())
        all_entries.append(entry)

    full_order = sorted(set(all_entries))

    # Read scoped subset
    df = idx.read(
        run_id=run_id,
        filter_ext=".parquet",
        groups=groups,
        sequences=sequences,
    )

    entries: set[tuple[str, str]] = set()
    path_map: dict[tuple[str, str], tuple[Path, LoadSpec]] = {}
    for _, row in df.iterrows():
        entry = (row["group"], row["sequence"])
        entries.add(entry)
        path_map[entry] = (Path(row["abs_path"]), ParquetLoadSpec())

    return entries, path_map, full_order, path_map_all


def _load_neighbor(
    file_specs: FileSpecs | None,
    entry_key: str | None,
    filter_factory: FilterFactory | None,
) -> pd.DataFrame | None:
    """Load a neighbor segment with its own filters applied."""
    if file_specs is None:
        return None
    filters: Iterable[Callable[[pd.DataFrame], pd.DataFrame]] = ()
    if filter_factory is not None and entry_key is not None:
        filters = filter_factory(entry_key)
    return load_entry_data(file_specs, filters=filters)


@overload
def iter_manifest(
    manifest: Manifest,
    *,
    filter_factory: FilterFactory | None = None,
    overlap_frames: None = None,
    progress_label: str = "",
    progress_interval: int = 10,
) -> Iterator[tuple[str, pd.DataFrame]]: ...


@overload
def iter_manifest(
    manifest: Manifest,
    *,
    filter_factory: FilterFactory | None = None,
    overlap_frames: int,
    progress_label: str = "",
    progress_interval: int = 10,
) -> Iterator[tuple[str, pd.DataFrame, int, int]]: ...


def iter_manifest(
    manifest: Manifest,
    *,
    filter_factory: FilterFactory | None = None,
    overlap_frames: int | None = None,
    progress_label: str = "",
    progress_interval: int = 10,
) -> Iterator[tuple[str, pd.DataFrame] | tuple[str, pd.DataFrame, int, int]]:
    """Iterate manifest entries, yielding data per sequence.

    When overlap_frames is None (default), yields (entry_key, df).
    When overlap_frames is an int, yields (entry_key, df, core_start, core_end).
    """
    n_entries = len(manifest)
    for i, (entry_key, entry) in enumerate(manifest.items()):
        # Build filters for this entry
        filters: Iterable[Callable[[pd.DataFrame], pd.DataFrame]] = ()
        if filter_factory is not None:
            filters = filter_factory(entry_key)

        # Load current segment
        df = load_entry_data(entry.file_specs, filters=filters)
        if df is None:
            continue

        if overlap_frames is None:
            yield entry_key, df
        else:
            if overlap_frames > 0:
                # Load and filter neighbor segments
                prev_df = _load_neighbor(
                    entry.prev_file_specs,
                    entry.prev_entry_key,
                    filter_factory,
                )
                next_df = _load_neighbor(
                    entry.next_file_specs,
                    entry.next_entry_key,
                    filter_factory,
                )
                # Trim neighbors to overlap_frames
                if prev_df is not None:
                    prev_df = prev_df.iloc[-overlap_frames:]
                if next_df is not None:
                    next_df = next_df.iloc[:overlap_frames]
                # Concatenate
                core_start = len(prev_df) if prev_df is not None else 0
                core_end = core_start + len(df)
                parts = [p for p in (prev_df, df, next_df) if p is not None]
                df = pd.concat(parts, ignore_index=True)
            else:
                # overlap_frames == 0: no neighbor loading, trivial bounds
                core_start = 0
                core_end = len(df)

            yield entry_key, df, core_start, core_end

        del df
        gc.collect()
        pa.default_memory_pool().release_unused()

        if progress_label and ((i + 1) % progress_interval == 0 or i == n_entries - 1):
            print(
                f"[{progress_label}] Processed {i + 1}/{n_entries} sequences",
                file=sys.stderr,
            )
