"""Unified manifest builder for all input types."""

from __future__ import annotations

import gc
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import pandas as pd
import pyarrow as pa

from ...core.helpers import make_entry_key, text_cell
from ...core.schema import LEGACY_SCHEMA, schema_family
from .sequence_index import read_entry_compositions
from ._utils import Scope
from .index import (
    feature_index,
    feature_index_path,
    missing_outputs_error,
)
from .loading import load_entry_data
from .track_universe import current_run_id, is_track_shaped
from .tracks_index import read_tracks_index, select_variant_rows, tracks_index_path
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


@dataclass(frozen=True, slots=True)
class ResolvedInput:
    """What one input of a feature resolved to.

    Both resolvers return this rather than a bare tuple. They are unpacked into
    the same four names by one dispatch loop, so a positional 4-tuple beside a
    5-tuple would put the new field in a different slot depending on which branch
    ran -- exactly the kind of difference that reads correctly and is not.

    Attributes:
        entries: The (group, sequence) pairs surviving the scope narrowing.
        path_map: Those entries' files.
        full_order: Every resolvable entry, sorted -- the ordering prev/next
            adjacency is computed from, which is why it ignores the narrowing.
        path_map_all: Every resolvable entry's file, likewise unnarrowed.
        tracks_variants: Which tracks recipes produced them. Only ever non-empty
            for the tracks resolver; a feature input's own identity is already
            pinned in ``Inputs`` by ``resolve_references``.
    """

    entries: set[tuple[str, str]]
    path_map: dict[tuple[str, str], tuple[Path, LoadSpec]]
    full_order: list[tuple[str, str]]
    path_map_all: dict[tuple[str, str], tuple[Path, LoadSpec]]
    tracks_variants: tuple[str, ...] = ()


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

    The predicate itself lives in ``track_universe.is_track_shaped``, because item
    9.4's enumerator needs the same question in boolean form. Two spellings of
    "what is a track" would eventually disagree, and the way they would show it is
    a table the selector offers and this then rejects.
    """
    if not path_map_all:
        return
    path = next(iter(path_map_all.values()))[0]
    if not path.exists():
        return
    if not is_track_shaped(path):
        raise ValueError(
            f"Feature {feature_name!r} does not produce a track-shaped table "
            f"(needs {sorted({COLUMNS.x_col, COLUMNS.y_col})}); it cannot be used "
            f"as a track input. A track input must be raw 'tracks' or the Result "
            f"of a track-producing feature (trajectory-smooth, track-subsample, "
            f"movement-smooth, movement-filter-interpolate)."
        )


def _leaf_run_of(ds: Dataset, feature_name: str) -> str:
    """Which run of *feature_name* an unpinned reference reads.

    Delegates: this used to hold its own three rules -- leaf when the runs were
    track-shaped, recorded time otherwise, and recorded time again when a cycle left
    no leaf -- while ``resolve`` used a fourth and claimed they agreed. One function
    now answers, and it needs neither a materialised nor a track-shaped run because
    the edges come from ``params.json``.
    """
    return current_run_id(ds, feature_name)


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

    variants: tuple[str, ...] = ()
    for i, item in enumerate(inputs.root):
        if item == "tracks":
            input_result = _resolve_tracks(
                ds, tracks_run_id, groups, sequences, entries, on_missing_run
            )
            variants = input_result.tracks_variants
        else:
            input_result = _resolve_feature(
                ds,
                item.feature,
                item.run_id,
                groups,
                sequences,
                entries,
                on_missing_run,
            )
            if getattr(type(inputs), "_track_input", False):
                _ensure_track_shaped(item.feature, input_result.path_map_all)
        per_input_entries.append(input_result.entries)
        per_input_paths.append(input_result.path_map)
        per_input_paths_all.append(input_result.path_map_all)
        if i == 0:
            first_full_order = input_result.full_order

    # Intersect entries across all inputs
    shared_entries: set[tuple[str, str]] = set()
    if per_input_entries:
        shared_entries = per_input_entries[0].intersection(*per_input_entries[1:])

    # Per-entry, and therefore read *after* the narrowing -- the opposite of
    # tracks_variants above, which is a property of the dataset under the
    # selector. Reading every source root rather than only the ones this feature
    # declares keeps the resolver ignorant of features: what is *hashed* is
    # decided at the one payload site, and what is merely *recorded* on the index
    # row wants the others.
    scope = Scope(
        entries=shared_entries,
        tracks_variants=variants,
        compositions=read_entry_compositions(ds, shared_entries),
    )

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
) -> ResolvedInput:
    """Resolve track entries and paths from tracks/index.csv.

    ``run_id`` names one tracks *variant*. It takes ``_resolve_feature``'s
    second-positional slot, and mirrors it -- with one deliberate difference:

    **``None`` means "whichever variant each entry has", not "the latest run".**
    For a feature, "latest" is well defined because one feature run covers the
    scope it was given. A tracks index is not like that: a mixed dataset carries
    different variants on different entries -- some converted, some tracked, some
    inferred -- so "latest" would silently collapse the universe to whichever
    recipe was written last, and would erase every adopted legacy row (whose
    ``run_id`` is honestly empty) the moment one new row appeared.

    That stays well defined once an entry can carry two rows because
    :func:`~mosaic.core.pipeline.tracks_index.select_variant_rows` decides per
    entry rather than per index: an unlabelled row loses to a labelled one, and
    two genuinely different recipes for one entry raise rather than guess. So
    ``None`` is legal on every dataset except the one where it has no answer.

    Unlike ``_resolve_feature`` this does not raise when *every* output resolves
    missing. An empty tracks manifest is a legitimate cold-start state that
    ``_read_track_universe``'s glob fallback and ``Dataset.load_tracks``'s
    auto-convert both depend on; it warns instead.

    **``tracks_variants`` is collected before the narrowing, not after.** It is
    the set of recipes behind every entry this dataset resolves, not behind the
    subset this call happens to want, and that distinction is what keeps it out
    of trouble: it feeds the feature identifier, and a scope-free feature must
    get one identifier for every scope. Were it scoped, ``run_feature(ds, f)``,
    ``run_feature(ds, f, sequences=["a"])`` and ``...=["b"]`` would mint three
    identifiers for one computation on a mixed dataset, and ``Pipeline.clean``
    -- whose keep set is the identifiers it predicted -- would delete two of
    them. Rows whose ``run_id`` is empty contribute nothing, so a dataset that
    predates variants yields an empty tuple and hashes exactly as it always has.
    """
    df = select_variant_rows(read_tracks_index(ds), run_id)
    if run_id is not None and df.empty:
        if on_missing_run == "raise":
            msg = f"No tracks rows for run_id {run_id!r} in {tracks_index_path(ds)}"
            raise FileNotFoundError(msg)
        return ResolvedInput(set(), {}, [], {})

    # Build full (unscoped) path map and order.
    # One row per (group, sequence) is guaranteed by select_variant_rows, which
    # is where the choice between two variants of one entry is made -- or
    # refused. This loop therefore still cannot see two rows for one entry, and
    # the resolver keeps its single-answer shape whatever the index holds.
    path_map_all: dict[tuple[str, str], tuple[Path, LoadSpec]] = {}
    all_entries: list[tuple[str, str]] = []
    missing: list[Path] = []
    resolved_variants: set[str] = set()
    schema_by_entry: dict[tuple[str, str], str] = {}
    for _, series in df.iterrows():
        # Materialized as a string-keyed mapping once per row. A pandas ``Series``
        # is keyed by an untyped index, so every cell read off it is untyped too
        # -- and the columns here are all read as text. Cheap beside the ``Series``
        # ``iterrows`` already builds.
        row: dict[str, object] = {str(key): value for key, value in series.items()}
        g, s = str(row["group"]), str(row["sequence"])
        entry = (g, s)
        p = ds.resolve_path(str(row["abs_path"]))
        if not p.exists():
            missing.append(p)
            continue
        path_map_all[entry] = (p, ParquetLoadSpec())
        all_entries.append(entry)
        # Read straight off the row rather than through ``.get``: the reader
        # projects onto the typed schema, so the column is always present and
        # an index predating it carries "" rather than being absent.
        schema_by_entry[entry] = text_cell(row["std_format"])
        # Collected here, off the *unscoped* rows, and deliberately not below
        # in the narrowing loop -- see the note in the docstring.
        variant = str(row["run_id"])
        if variant:
            resolved_variants.add(variant)

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

    _refuse_mixed_schemas(scoped, schema_by_entry)

    return ResolvedInput(
        scoped, path_map, full_order, path_map_all, tuple(sorted(resolved_variants))
    )


def _refuse_mixed_schemas(
    scoped: set[tuple[str, str]],
    schema_by_entry: dict[tuple[str, str], str],
) -> None:
    """Refuse a scope whose tables answer to incompatible schemas.

    ``select_variant_rows`` refuses two variants for **one entry**, and blesses
    different entries carrying different variants -- which is right, because a
    mixed dataset of converted and tracked entries is ordinary. What it cannot
    see is that those variants may mean different *things*: a ``trex_v1`` table
    is centimetres with ``X`` at the head, a ``mosaic_v1`` table is pixels with
    ``X`` at the body centre. Reading both into one feature run silently
    compares the two, and every number that comes out is wrong by a factor
    nobody recorded.

    Scoped rather than dataset-wide on purpose. A migration converts entries a
    batch at a time, and refusing on the whole index would make the dataset
    unusable from the first reconverted entry until the last. What must not
    happen is one *run* spanning both, and that is exactly what this asks.

    The check is on the schema *family* -- the root of the ``extends`` chain --
    so ``mosaic_v1`` and ``trex_v2`` mix freely: a feature reading what
    ``mosaic_v1`` guarantees gets the same meaning from either.

    **An unrecorded schema is read as the legacy one, never as a family of its
    own.** Every row with an empty cell was written before the column existed,
    and the only schema that existed then was ``trex_v1`` -- so it *is* that,
    and pairing the two is not a mixture. Counting blank as its own family
    instead would fire on every dataset converted before this column, which is
    the trap ``select_variant_rows`` documents for an unlabelled ``run_id``:
    the ambiguity is with the historical default, not between two recipes. A
    blank beside a ``mosaic_v1`` row still refuses, which is the case that
    matters -- one is centimetres and the other pixels.
    """
    families: dict[str, list[tuple[str, str]]] = {}
    for entry in scoped:
        recorded = schema_by_entry.get(entry, "") or LEGACY_SCHEMA
        families.setdefault(schema_family(recorded), []).append(entry)
    if len(families) < 2:
        return

    def describe(family: str) -> str:
        entries = sorted(families[family])
        shown = ", ".join(make_entry_key(g, s) for g, s in entries[:3])
        more = f" (+{len(entries) - 3} more)" if len(entries) > 3 else ""
        name = family or "<unrecorded>"
        return f"  {name}: {shown}{more}"

    listing = "\n".join(describe(family) for family in sorted(families))
    raise ValueError(
        "This scope resolves tracks tables of incompatible schemas, which do not "
        "mean the same thing and must not be read together:\n"
        f"{listing}\n"
        f"A row with no recorded schema is read as {LEGACY_SCHEMA!r}, the only one "
        "there was before the column existed. Reconvert the odd entries out, or "
        "narrow the scope with groups=/sequences=/entries= so one run reads one "
        "schema."
    )


def _resolve_feature(
    ds: Dataset,
    feature_name: str,
    run_id: str | None,
    groups: set[str] | None,
    sequences: set[str] | None,
    entries: set[tuple[str, str]] | None = None,
    on_missing_run: MissingRunPolicy = "raise",
) -> ResolvedInput:
    """Resolve feature result entries and paths from the feature index CSV.

    Leaves ``tracks_variants`` empty: a feature input's identity is the upstream
    ``run_id``, which ``resolve_references`` has already pinned into ``Inputs``
    where the digest reads it.

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
        return ResolvedInput(set(), {}, [], {})

    if run_id is None:
        # Unreachable from run_feature and from the chain runner: both call
        # resolve.resolve_references first, so `item.run_id` arrives concrete
        # (item 1.1). Kept for direct callers of build_manifest -- it is public
        # API.
        #
        # **The leaf of the chain, not the newest run** (item 9.4). It used to be
        # `latest_feature_run_root`, which sorts on the `finished_at` /
        # `started_at` strings -- wall-clock ordering, and not reproducible
        # across a synced dataset: two machines that ran the same work in a
        # different order disagree about which table is current. The leaf is a
        # property of the data instead, and two leaves refuse rather than
        # tiebreak. A dataset with a single chain per feature is unaffected,
        # which is every dataset that has only ever run one recipe.
        run_id = _leaf_run_of(ds, feature_name)

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
        return ResolvedInput(set(), {}, [], {})
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

    return ResolvedInput(scoped, path_map, full_order, path_map_all)


def _load_neighbor(
    file_specs: FileSpecs | None,
    entry_key: str | None,
    filter_factory: FilterFactory | None,
    cross_join: bool = False,
) -> pd.DataFrame | None:
    """Load a neighbor segment with its own filters applied."""
    if file_specs is None:
        return None
    filters: Iterable[Callable[[pd.DataFrame], pd.DataFrame]] = ()
    if filter_factory is not None and entry_key is not None:
        filters = filter_factory(entry_key)
    return load_entry_data(file_specs, filters=filters, cross_join=cross_join)


@overload
def iter_manifest(
    manifest: Manifest,
    *,
    filter_factory: FilterFactory | None = None,
    overlap_frames: None = None,
    progress_label: str = "",
    progress_interval: int = 10,
    cross_join: bool = False,
) -> Iterator[tuple[str, pd.DataFrame]]: ...


@overload
def iter_manifest(
    manifest: Manifest,
    *,
    filter_factory: FilterFactory | None = None,
    overlap_frames: int,
    progress_label: str = "",
    progress_interval: int = 10,
    cross_join: bool = False,
) -> Iterator[tuple[str, pd.DataFrame, int, int]]: ...


def iter_manifest(
    manifest: Manifest,
    *,
    filter_factory: FilterFactory | None = None,
    overlap_frames: int | None = None,
    progress_label: str = "",
    progress_interval: int = 10,
    cross_join: bool = False,
) -> Iterator[tuple[str, pd.DataFrame] | tuple[str, pd.DataFrame, int, int]]:
    """Iterate manifest entries, yielding data per sequence.

    When overlap_frames is None (default), yields (entry_key, df).
    When overlap_frames is an int, yields (entry_key, df, core_start, core_end).

    *cross_join* passes the frame-only-merge escape down to the loader, for the one
    feature that declares it (``loading.CROSS_JOIN_FEATURES``).
    """
    n_entries = len(manifest)
    for i, (entry_key, entry) in enumerate(manifest.items()):
        # Build filters for this entry
        filters: Iterable[Callable[[pd.DataFrame], pd.DataFrame]] = ()
        if filter_factory is not None:
            filters = filter_factory(entry_key)

        # Load current segment
        df = load_entry_data(entry.file_specs, filters=filters, cross_join=cross_join)
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
