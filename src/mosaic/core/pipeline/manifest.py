"""Unified manifest builder for all input types."""

from __future__ import annotations

import gc
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

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


Manifest = dict[str, list[tuple[Path, "LoadSpec"]]]


def build_manifest(
    ds: Dataset,
    inputs: InputsLike,
    groups: set[str] | None = None,
    sequences: set[str] | None = None,
) -> tuple[Manifest, Scope]:
    """Build unified manifest for all input types.

    Returns the manifest (entry_key -> [(path, load_spec)]) and the
    resolved Scope (entries present in ALL inputs after intersection).
    """
    per_input_entries: list[set[tuple[str, str]]] = []
    per_input_paths: list[dict[tuple[str, str], tuple[Path, LoadSpec]]] = []

    for item in inputs.root:
        if item == "tracks":
            entries, path_map = _resolve_tracks(ds, groups, sequences)
        else:
            entries, path_map = _resolve_feature(
                ds,
                item.feature,
                item.run_id,
                groups,
                sequences,
            )
        per_input_entries.append(entries)
        per_input_paths.append(path_map)

    # Intersect entries across all inputs
    shared_entries: set[tuple[str, str]] = set()
    if per_input_entries:
        shared_entries = per_input_entries[0].intersection(*per_input_entries[1:])

    scope = Scope(entries=shared_entries)

    # Build manifest: for each shared entry, collect all input file specs
    manifest: Manifest = {}
    for group, sequence in sorted(shared_entries):
        key = make_entry_key(group, sequence)
        specs: list[tuple[Path, LoadSpec]] = []
        for path_map in per_input_paths:
            if (group, sequence) in path_map:
                specs.append(path_map[(group, sequence)])
        manifest[key] = specs

    return manifest, scope


def _resolve_tracks(
    ds: Dataset,
    groups: set[str] | None,
    sequences: set[str] | None,
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], tuple[Path, LoadSpec]]]:
    """Resolve track entries and paths from tracks/index.csv."""
    tracks_root = ds.get_root("tracks")
    idx_path = tracks_root / "index.csv"
    if not idx_path.exists():
        return set(), {}

    df = pd.read_csv(idx_path, keep_default_na=False)
    entries: set[tuple[str, str]] = set()
    path_map: dict[tuple[str, str], tuple[Path, LoadSpec]] = {}

    for _, row in df.iterrows():
        g, s = row["group"], row["sequence"]
        if groups and g not in groups:
            continue
        if sequences and s not in sequences:
            continue
        # TODO: tracks/index.csv abs_path is actually relative despite the
        # column name -- ds.resolve_path resolves it against the dataset root.
        p = ds.resolve_path(row["abs_path"])
        if not p.exists():
            continue
        entry = (g, s)
        entries.add(entry)
        path_map[entry] = (p, ParquetLoadSpec())

    return entries, path_map


def _resolve_feature(
    ds: Dataset,
    feature_name: str,
    run_id: str | None,
    groups: set[str] | None,
    sequences: set[str] | None,
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], tuple[Path, LoadSpec]]]:
    """Resolve feature result entries and paths from the feature index CSV."""
    idx_path = feature_index_path(ds, feature_name)
    if not idx_path.exists():
        return set(), {}

    if run_id is None:
        run_id, _ = latest_feature_run_root(ds, feature_name)

    idx = feature_index(idx_path)
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

    return entries, path_map


def iter_manifest(
    manifest: Manifest,
    progress_label: str = "",
    progress_interval: int = 10,
) -> Iterator[tuple[str, tuple[pd.DataFrame, str]]]:
    """Iterate manifest entries, yielding (entry_key, EntryData) per sequence.

    Loads each entry's file specs, merges via inner join on alignment
    columns, extracts numeric features, and yields EntryData. Runs
    gc.collect + PyArrow pool release between iterations.
    """

    n_entries = len(manifest)
    for i, (entry_key, file_specs) in enumerate(manifest.items()):
        entry_data = load_entry_data(file_specs)
        if entry_data is None:
            continue

        yield entry_key, entry_data

        del entry_data
        gc.collect()
        pa.default_memory_pool().release_unused()

        if progress_label and ((i + 1) % progress_interval == 0 or i == n_entries - 1):
            print(
                f"[{progress_label}] Processed {i + 1}/{n_entries} sequences",
                file=sys.stderr,
            )
