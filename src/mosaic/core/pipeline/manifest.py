"""Unified manifest builder for all input types."""

from __future__ import annotations

import gc
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, overload

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ...core.helpers import make_entry_key
from ._utils import Scope
from .index import (
    feature_index,
    feature_index_path,
    latest_feature_run_root,
    missing_outputs_error,
)
from .loading import load_entry_data
from .tracks_index import read_tracks_index, tracks_index_path
from .types import (
    COLUMNS,
    InputsLike,
    LoadSpec,
    ParquetLoadSpec,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


FileSpecs = list[tuple[Path, LoadSpec]]

# What to do about a pinned upstream run the index does not hold. "raise" is
# for resolving inputs a run will read; "empty" is for predicting an identifier,
# where naming a run that does not exist yet is the ordinary case.
MissingRunPolicy = Literal["raise", "empty"]

FilterFactory = Callable[[str], Iterable[Callable[[pd.DataFrame], pd.DataFrame]]]


@dataclass(slots=True)
class ManifestEntry:
    file_specs: FileSpecs = field(default_factory=list)
    prev_file_specs: FileSpecs | None = None
    prev_entry_key: str | None = None
    next_file_specs: FileSpecs | None = None
    next_entry_key: str | None = None


Manifest = dict[str, ManifestEntry]


def _ensure_track_shaped(
    feature_name: str,
    path_map_all: dict[tuple[str, str], tuple[Path, LoadSpec]],
) -> None:
    """Enforce the track-input contract for a ``Result`` feeding a track feature.

    A feature declaring ``TrackInputs`` consumes a *track-shaped* table, so a
    ``Result`` input must come from a **track-producing** feature -- one that
    passes the track frame through (keeping ``X``/``Y``), e.g. ``trajectory-smooth``
    or ``track-subsample``. Derived features (``speed-angvel``, ``nearest-neighbor``,
    ``pair-*``) join back only ``COLUMNS.meta_set()`` and drop ``X``/``Y``, so their
    output is a valid ``Result`` but not a valid track input.

    The check is truth-based -- it peeks the resolved output's parquet schema for
    the position columns -- so it needs no producer registry (``core`` must not
    import ``behavior``) and self-maintains as new track producers are added. It is
    a no-op when nothing resolved (empty upstream -> empty manifest, as before).
    """
    if not path_map_all:
        return
    path = next(iter(path_map_all.values()))[0]
    if not path.exists():
        return
    # pyarrow ships no type stubs here, so read_schema().names is Unknown; pin it.
    names = cast("list[str]", pq.read_schema(path).names)  # pyright: ignore[reportUnknownMemberType]
    columns = set(names)
    required = {COLUMNS.x_col, COLUMNS.y_col}
    missing = required - columns
    if missing:
        raise ValueError(
            f"Feature {feature_name!r} does not produce a track-shaped table "
            f"(missing {sorted(missing)}); it cannot be used as a track input. "
            f"A track input must be raw 'tracks' or the Result of a track-producing "
            f"feature (trajectory-smooth, track-subsample, movement-smooth, "
            f"movement-filter-interpolate)."
        )


def build_manifest(
    ds: Dataset,
    inputs: InputsLike,
    groups: set[str] | None = None,
    sequences: set[str] | None = None,
    entries: set[tuple[str, str]] | None = None,
    *,
    tracks_run_id: str | None = None,
    on_missing_run: MissingRunPolicy = "raise",
) -> tuple[Manifest, Scope]:
    """Build unified manifest for all input types.

    Returns the manifest (entry_key -> ManifestEntry) and the
    resolved Scope (entries present in ALL inputs after intersection).

    Scope can be narrowed three ways (all applied, intersecting):

    - ``groups`` / ``sequences`` -- keep entries whose group / sequence is in
      the given set. These combine as a *cross-product* filter.
    - ``entries`` -- keep only these explicit ``(group, sequence)`` pairs. Use
      this when an arbitrary subset is required (e.g. a tag-resolved selection),
      especially when sequence names are not unique across groups, where a bare
      ``sequences`` filter would be ambiguous.

    ``tracks_run_id`` selects which tracks *variant* the ``"tracks"`` input
    resolves to. ``None``, the default, means **every row** -- see
    :func:`_resolve_tracks` for why that is not "the latest run".

    ``on_missing_run`` is forwarded to both resolvers; leave it at ``"raise"``
    unless you are predicting an identifier rather than resolving inputs to read.

    Both are keyword-only, so a positional argument cannot land in the wrong one.
    """
    per_input_entries: list[set[tuple[str, str]]] = []
    per_input_paths: list[dict[tuple[str, str], tuple[Path, LoadSpec]]] = []
    per_input_paths_all: list[dict[tuple[str, str], tuple[Path, LoadSpec]]] = []
    first_full_order: list[tuple[str, str]] = []

    for i, item in enumerate(inputs.root):
        if item == "tracks":
            resolved, path_map, full_order, path_map_all = _resolve_tracks(
                ds, tracks_run_id, groups, sequences, entries, on_missing_run
            )
        else:
            resolved, path_map, full_order, path_map_all = _resolve_feature(
                ds,
                item.feature,
                item.run_id,
                groups,
                sequences,
                entries,
                on_missing_run,
            )
            if getattr(type(inputs), "_track_input", False):
                _ensure_track_shaped(item.feature, path_map_all)
        per_input_entries.append(resolved)
        per_input_paths.append(path_map)
        per_input_paths_all.append(path_map_all)
        if i == 0:
            first_full_order = full_order

    # Intersect entries across all inputs
    shared_entries: set[tuple[str, str]] = set()
    if per_input_entries:
        shared_entries = per_input_entries[0].intersection(*per_input_entries[1:])

    scope = Scope(entries=shared_entries)

    # Build per-group ordering from first input's full order.
    #
    # `group` here defines *temporal contiguity*: the prev/next adjacency below
    # (used for overlap_frames) is computed only within a group, never across
    # group boundaries. This is the one structural role of `group`; it is dormant
    # for discrete datasets and becomes load-bearing for the future `continuous`
    # dataset type (arbitrary time-window sequences). Preserve when softening
    # `group` elsewhere. See also iteration.yield_sequences_with_overlap.
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
    run_id: str | None,
    groups: set[str] | None,
    sequences: set[str] | None,
    entries: set[tuple[str, str]] | None = None,
    on_missing_run: MissingRunPolicy = "raise",
) -> tuple[
    set[tuple[str, str]],
    dict[tuple[str, str], tuple[Path, LoadSpec]],
    list[tuple[str, str]],
    dict[tuple[str, str], tuple[Path, LoadSpec]],
]:
    """Resolve track entries and paths from tracks/index.csv.

    Returns (scoped_entries, scoped_path_map, full_order, path_map_all).

    ``run_id`` names one tracks *variant*. It takes ``_resolve_feature``'s
    second-positional slot, and mirrors it -- with one deliberate difference:

    **``None`` means every row, not "the latest run".** For a feature, "latest"
    is well defined because one feature run covers the scope it was given. A
    tracks index is not like that: a mixed dataset carries different variants on
    different entries -- some converted, some tracked, some inferred -- so
    "latest" would silently collapse the universe to whichever recipe was written
    last, and would erase every adopted legacy row (whose ``run_id`` is honestly
    empty) the moment one new row appeared.

    ``None`` is well defined *because* M1 holds one row per ``(group, sequence)``.
    **That expires at Stage 3.4**, which makes a second row legal; ``None``
    becomes ambiguous there and should raise until item 9.4 widens the selector.

    Unlike ``_resolve_feature`` this does not raise when *every* output resolves
    missing. An empty tracks manifest is a legitimate cold-start state that
    ``_read_track_universe``'s glob fallback and ``Dataset.load_tracks``'s
    auto-convert both depend on; it warns instead.
    """
    df = read_tracks_index(ds)
    if run_id is not None:
        selected = df[df["run_id"] == run_id]
        if selected.empty:
            if on_missing_run == "raise":
                msg = f"No tracks rows for run_id {run_id!r} in {tracks_index_path(ds)}"
                raise FileNotFoundError(msg)
            return set(), {}, [], {}
        df = selected

    # Build full (unscoped) path map and order
    # One row per (group, sequence) is guaranteed upstream: read_tracks_index
    # collapses duplicates keep-last, so this loop cannot see two rows for one
    # entry and does not have to choose between them. That is what makes a
    # ``run_id`` of None well defined. At Stage 3.4 the collapse goes and this
    # loop becomes the place that must choose -- or refuse to.
    path_map_all: dict[tuple[str, str], tuple[Path, LoadSpec]] = {}
    all_entries: list[tuple[str, str]] = []
    missing: list[Path] = []
    for _, row in df.iterrows():
        g, s = str(row["group"]), str(row["sequence"])
        entry = (g, s)
        p = ds.resolve_path(row["abs_path"])
        if not p.exists():
            missing.append(p)
            continue
        path_map_all[entry] = (p, ParquetLoadSpec())
        all_entries.append(entry)

    if missing and not all_entries:
        print(
            f"[manifest] all {len(missing)} tracks row(s) resolve to missing "
            f"files; first: {missing[0]}. A moved or synced dataset needs "
            "ds.make_portable() / ds.rewrite_index_paths().",
            file=sys.stderr,
        )

    # Sort by (group, sequence) for stable ordering
    full_order = sorted(set(all_entries))

    # Filter for scoped subset
    scoped: set[tuple[str, str]] = set()
    path_map: dict[tuple[str, str], tuple[Path, LoadSpec]] = {}
    for entry, spec in path_map_all.items():
        g, s = entry
        if groups and g not in groups:
            continue
        if sequences and s not in sequences:
            continue
        if entries is not None and entry not in entries:
            continue
        scoped.add(entry)
        path_map[entry] = spec

    return scoped, path_map, full_order, path_map_all


def _resolve_feature(
    ds: Dataset,
    feature_name: str,
    run_id: str | None,
    groups: set[str] | None,
    sequences: set[str] | None,
    entries: set[tuple[str, str]] | None = None,
    on_missing_run: MissingRunPolicy = "raise",
) -> tuple[
    set[tuple[str, str]],
    dict[tuple[str, str], tuple[Path, LoadSpec]],
    list[tuple[str, str]],
    dict[tuple[str, str], tuple[Path, LoadSpec]],
]:
    """Resolve feature result entries and paths from the feature index CSV.

    Returns (scoped_entries, scoped_path_map, full_order, path_map_all).

    ``on_missing_run`` governs a pinned run that the index does not hold. It is
    ``"raise"`` for execution -- consuming a run that is not there is a wrong
    answer, and an empty manifest would turn it into a silent no-op. It is
    ``"empty"`` for the chain runner's prediction, which asks what a step's
    identifier *would* be and therefore names runs that do not exist yet: after
    any params change the upstream index holds only the previous run, which is
    an ordinary state rather than a failure.
    """
    idx_path = feature_index_path(ds, feature_name)
    if not idx_path.exists():
        return set(), {}, [], {}

    if run_id is None:
        # Unreachable from run_feature and from the chain runner: both call
        # resolve.resolve_references first, so `item.run_id` arrives concrete
        # (item 1.1). Kept for direct callers of build_manifest -- it is public
        # API -- and because the resolution rule here and there is the same
        # function, so the two cannot disagree about which run "latest" means.
        run_id, _ = latest_feature_run_root(ds, feature_name)

    idx = feature_index(idx_path)

    # Read full (unscoped) index for all entries. Resolve each stored path
    # through the dataset (relative -> root; remap for relocated absolutes) and
    # check existence there -- mirroring _resolve_tracks -- rather than trusting
    # the raw string via IndexCSV's naive validate_paths (which false-fails on a
    # moved/synced dataset whose index carries another machine's paths).
    try:
        df_all = idx.read(run_id=run_id, filter_ext=".parquet", validate_paths=False)
    except FileNotFoundError:
        if on_missing_run == "raise":
            raise
        # The scoped read below filters on the same run_id, so it would fail
        # identically; there is nothing left to resolve.
        return set(), {}, [], {}
    path_map_all: dict[tuple[str, str], tuple[Path, LoadSpec]] = {}
    all_entries: list[tuple[str, str]] = []
    missing_all: list[Path] = []
    for _, row in df_all.iterrows():
        entry = (row["group"], row["sequence"])
        resolved = ds.resolve_path(row["abs_path"])
        if not resolved.exists():
            missing_all.append(resolved)
            continue
        path_map_all[entry] = (resolved, ParquetLoadSpec())
        all_entries.append(entry)

    if missing_all and not all_entries:
        # Every output resolved missing: dataset moved / non-portable paths.
        # Fail loudly and actionably instead of computing over an empty scope.
        raise missing_outputs_error(feature_name, run_id, missing_all, len(df_all))
    if missing_all:
        # Partial: skip the missing entries. In a Pipeline the upstream step's
        # completeness check reports it not-cached and recomputes them.
        print(
            f"[manifest] feature {feature_name!r} run {run_id!r}: "
            f"{len(missing_all)} of {len(df_all)} output(s) missing; skipping "
            f"(will be recomputed upstream).",
            file=sys.stderr,
        )

    full_order = sorted(set(all_entries))

    # Read scoped subset (same resolve-then-skip policy).
    df = idx.read(
        run_id=run_id,
        filter_ext=".parquet",
        groups=groups,
        sequences=sequences,
        entries=entries,
        validate_paths=False,
    )

    scoped: set[tuple[str, str]] = set()
    path_map: dict[tuple[str, str], tuple[Path, LoadSpec]] = {}
    for _, row in df.iterrows():
        entry = (row["group"], row["sequence"])
        resolved = ds.resolve_path(row["abs_path"])
        if not resolved.exists():
            continue
        scoped.add(entry)
        path_map[entry] = (resolved, ParquetLoadSpec())

    return scoped, path_map, full_order, path_map_all


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
