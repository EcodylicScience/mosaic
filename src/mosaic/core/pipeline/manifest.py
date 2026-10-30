"""Unified manifest builder for all input types."""

from __future__ import annotations

import gc
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa

from ...core.helpers import make_entry_key, text_cell
from ...core.recording import (
    OrderFacts,
    RecordingKey,
    adjacent,
    facts_from_frame_extents,
)
from ...core.schema import LEGACY_SCHEMA, schema_family
from ...core.scope import Scope
from .sequence_index import read_entry_compositions
from ._utils import ResolvedScope
from .index import (
    feature_index,
    feature_index_path,
    missing_outputs_error,
)
from .loading import load_entry_data
from .track_universe import current_run_id, is_track_shaped
from .tracks_index import (
    FrameExtents,
    read_frame_extents,
    read_tracks_index,
    select_variant_rows,
    tracks_index_path,
    variant_candidate_rows,
)
from .types import (
    COLUMNS,
    InputsLike,
    LoadSpec,
    ParquetLoadSpec,
    resolve_order_col,
)
from .parquet_reader import read_parquet_column_names

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
    """One entry's inputs, its neighbours, and what overlap needs to judge them.

    The extents and ``continuous`` are carried rather than looked up so
    :func:`iter_manifest` can gate and slice without a ``Dataset`` -- it is handed
    a manifest and nothing else, and reaching back for the index per entry would
    put a read inside the iteration.
    """

    file_specs: FileSpecs = field(default_factory=list)
    prev_file_specs: FileSpecs | None = None
    prev_entry_key: str | None = None
    next_file_specs: FileSpecs | None = None
    next_entry_key: str | None = None
    entry_key: str = ""
    group: str = ""
    sequence: str = ""
    continuous: bool = False
    core_extent: tuple[int, int] | None = None
    prev_extent: tuple[int, int] | None = None
    next_extent: tuple[int, int] | None = None


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
        tracks_schemas: The schema each entry in ``entries`` records, for the
            tracks resolver only. What ``ResolvedScope.tracks_schemas`` is
            narrowed from.
    """

    entries: set[tuple[str, str]]
    path_map: dict[tuple[str, str], tuple[Path, LoadSpec]]
    full_order: list[tuple[str, str]]
    path_map_all: dict[tuple[str, str], tuple[Path, LoadSpec]]
    tracks_variants: tuple[str, ...] = ()
    tracks_schemas: dict[tuple[str, str], str] = field(default_factory=dict)


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
    scope: Scope | None = None,
    *,
    tracks_run_id: str | None = None,
    on_missing_run: MissingRunPolicy = "raise",
    stack: bool = False,
) -> tuple[Manifest, ResolvedScope]:
    """Build unified manifest for all input types.

    Returns the manifest (entry_key -> ManifestEntry) and the resolved
    ``ResolvedScope`` (entries present in ALL inputs after intersection, or in
    ANY input under *stack*), recording the selector it came from.

    *stack* is the reading of a feature that declares ``stacks_inputs``: its
    inputs are runs over disjoint parts of the data, and the entries are their
    **union**, each entry carrying the one file that holds it. An entry that two
    inputs both hold is refused with :class:`StackedEntryOverlap`, because
    reading it from either would silently drop the other and reading both would
    count it twice. A single input is unchanged by it.

    *scope* is applied to every input and the results are intersected.
    ``Scope(groups=..., sequences=...)`` keeps an entry whose group and whose
    sequence are both named, combining as a cross product.
    ``Scope(entries=...)`` keeps the pairs it lists, which an arbitrary subset
    needs -- a tag-resolved selection, or a sequence name that repeats across
    groups and a bare ``sequences`` filter cannot separate. ``None``, the
    default, keeps every entry each input resolves.

    The selector narrows each input against its **own** index rather than
    against one enumeration shared between them. Resolving ``groups`` to entries
    up front would need one index naming every entry, and a dataset whose tracks
    predate its media index does not have one.

    ``tracks_run_id`` selects which tracks *variant* the ``"tracks"`` input
    resolves to. ``None``, the default, means **every row** -- see
    :func:`_resolve_tracks` for why that is not "the latest run".

    ``on_missing_run`` is forwarded to both resolvers; leave it at ``"raise"``
    unless you are predicting an identifier rather than resolving inputs to read.

    Both are keyword-only, so a positional argument cannot land in the wrong one.
    """
    selector = scope if scope is not None else Scope()
    per_input_entries: list[set[tuple[str, str]]] = []
    per_input_paths: list[dict[tuple[str, str], tuple[Path, LoadSpec]]] = []
    per_input_paths_all: list[dict[tuple[str, str], tuple[Path, LoadSpec]]] = []
    first_full_order: list[tuple[str, str]] = []
    stacked_full_order: set[tuple[str, str]] = set()

    variants: tuple[str, ...] = ()
    tracks_schemas: dict[tuple[str, str], str] = {}
    upstream_runs: list[tuple[str, str | None]] = []
    for i, item in enumerate(inputs.root):
        if item == "tracks":
            input_result = _resolve_tracks(ds, tracks_run_id, selector, on_missing_run)
            variants = input_result.tracks_variants
            tracks_schemas = input_result.tracks_schemas
        else:
            input_result = _resolve_feature(
                ds,
                item.feature,
                item.run_id,
                selector,
                on_missing_run,
            )
            if getattr(type(inputs), "_track_input", False):
                _ensure_track_shaped(item.feature, input_result.path_map_all)
            upstream_runs.append((item.feature, item.run_id))
        per_input_entries.append(input_result.entries)
        per_input_paths.append(input_result.path_map)
        per_input_paths_all.append(input_result.path_map_all)
        if i == 0:
            first_full_order = input_result.full_order
        stacked_full_order.update(input_result.full_order)

    shared_entries: set[tuple[str, str]] = set()
    if stack:
        _refuse_stacked_overlap(per_input_entries)
        shared_entries = {entry for entries in per_input_entries for entry in entries}
        first_full_order = sorted(stacked_full_order)
    elif per_input_entries:
        # Intersect entries across all inputs
        shared_entries = per_input_entries[0].intersection(*per_input_entries[1:])

    # Per-entry, and therefore read *after* the narrowing -- the opposite of
    # tracks_variants above, which is a property of the dataset under the
    # selector. Reading every source root rather than only the ones this feature
    # declares keeps the resolver ignorant of features: what is *hashed* is
    # decided at the one payload site, and what is merely *recorded* on the index
    # row wants the others.
    resolved = ResolvedScope(
        entries=shared_entries,
        # What was asked for, beside what it resolved to. An empty resolution
        # under an unset selector and one under a named group are different
        # states, and the entry set alone does not distinguish them.
        selector=selector,
        tracks_variants=variants,
        compositions=read_entry_compositions(ds, shared_entries),
        tracks_schemas={
            entry: schema
            for entry, schema in tracks_schemas.items()
            if entry in shared_entries
        },
    )

    # Neighbors are the adjacent members of a divided recording
    # (`mosaic.core.recording`). An undivided recording, whatever its group, has
    # one member and no prev/next, and adjacency does not cross from one
    # recording into another.
    #
    # Members come from the first input's universe, before the scope narrows it,
    # and are ordered by their recorded first frame, then by a natural sort on
    # their names. The resolvers above skip a sequence whose parquet is missing.
    # A name order alone would then make `seg3` and `seg5` adjacent across the
    # hole.
    #
    # A division of a store is placed by its media row, and its neighbor is the
    # division whose frames it continues (`adjacent`). A gap, a store boundary,
    # or a division that is refused or missing leaves the division before it
    # without a next neighbor.
    frame_extents = _continuous_frame_extents(
        ds, first_full_order, tracks_run_id, variants, upstream_runs
    )
    order_facts = {
        **facts_from_frame_extents(frame_extents),
        **ds.division_facts(first_full_order),
    }
    divided: dict[RecordingKey, tuple[tuple[str, str], ...]] = {
        recording.key: recording.members
        for recording in ds.recordings(first_full_order, order_facts=order_facts)
        if recording.kind == "divided"
    }

    manifest: Manifest = {}
    for group, sequence in sorted(shared_entries):
        key = make_entry_key(group, sequence)
        specs: FileSpecs = []
        for path_map in per_input_paths:
            if (group, sequence) in path_map:
                specs.append(path_map[(group, sequence)])

        members = divided.get(ds.recording_of(group, sequence))
        continuous = members is not None
        prev_entry: tuple[str, str] | None = None
        next_entry: tuple[str, str] | None = None
        if members is not None:
            prev_entry, next_entry = _neighbors((group, sequence), members, order_facts)

        prev_key, prev_specs = _neighbor_specs(prev_entry, per_input_paths_all)
        next_key, next_specs = _neighbor_specs(next_entry, per_input_paths_all)

        manifest[key] = ManifestEntry(
            file_specs=specs,
            prev_file_specs=prev_specs,
            prev_entry_key=prev_key,
            next_file_specs=next_specs,
            next_entry_key=next_key,
            entry_key=key,
            group=group,
            sequence=sequence,
            continuous=continuous,
            core_extent=frame_extents.get((group, sequence)),
            prev_extent=None if prev_entry is None else frame_extents.get(prev_entry),
            next_extent=None if next_entry is None else frame_extents.get(next_entry),
        )

    return manifest, resolved


class StackedEntryOverlap(ValueError):
    """Two inputs of a stacking step both hold one entry."""


def _refuse_stacked_overlap(per_input_entries: list[set[tuple[str, str]]]) -> None:
    """Raise :class:`StackedEntryOverlap` when an entry is in more than one input.

    Args:
        per_input_entries: The entries each input resolves under the scope, by
            declared position.
    """
    held_by: dict[tuple[str, str], list[int]] = {}
    for position, entries in enumerate(per_input_entries):
        for entry in entries:
            held_by.setdefault(entry, []).append(position)
    repeated = sorted(entry for entry, held in held_by.items() if len(held) > 1)
    if not repeated:
        return
    shown = ", ".join(
        f"({group!r}, {sequence!r}) in inputs "
        + " and ".join(str(position) for position in held_by[(group, sequence)])
        for group, sequence in repeated[:3]
    )
    more = f" and {len(repeated) - 3} more" if len(repeated) > 3 else ""
    msg = (
        f"inputs that stack must hold disjoint entries, and {len(repeated)} "
        f"entries appear in more than one input: {shown}{more}. Scoring an "
        "entry twice would count it twice. Narrow the step's scope or give each "
        "input its own entries."
    )
    raise StackedEntryOverlap(msg)


def _continuous_frame_extents(
    ds: Dataset,
    universe: Iterable[tuple[str, str]],
    tracks_run_id: str | None,
    tracks_variants: Iterable[str],
    upstream_runs: Iterable[tuple[str, str | None]],
) -> FrameExtents:
    """Return the recorded frame extents of the continuous-group members of *universe*.

    Only a continuous group's members have neighbors, and only their extents
    are read. Two variants of another entry do not refuse the read. A pinned
    *tracks_run_id* names the variant. Otherwise each member resolves among the
    variants that the inputs read: *tracks_variants* from a ``tracks`` input,
    and the variants each feature run of *upstream_runs* read, from its
    ``params.json``. A run whose variants are unknown leaves every variant in
    play.

    Raises:
        ValueError: If a member has two labeled variants and neither the pin
            nor the inputs choose between them.
    """
    groups = {group for group, _ in universe if ds.recording_kind(group) == "divided"}
    if not groups:
        return {}
    if tracks_run_id is not None:
        return read_frame_extents(ds, tracks_run_id, groups=groups)
    read: set[str] | None = set(tracks_variants)
    for storage, run_id in upstream_runs:
        upstream = _upstream_tracks_variants(ds, storage, run_id)
        if upstream is None:
            read = None
            break
        read |= upstream
    return read_frame_extents(ds, None, groups=groups, variants=read)


def _upstream_tracks_variants(
    ds: Dataset, storage: str, run_id: str | None
) -> frozenset[str] | None:
    """Return the tracks variants a run of *storage* read, ``None`` if unknown.

    An unpinned *run_id* is the run :func:`_resolve_feature` reads. An empty set
    names the unlabeled tables, which a run records no variant for.
    """
    from .inventory.params import run_tracks_variants

    if run_id is None:
        if not feature_index_path(ds, storage).exists():
            return frozenset()
        run_id = _leaf_run_of(ds, storage)
    return run_tracks_variants(ds, storage, run_id)


def _neighbors(
    entry: tuple[str, str],
    ordered: tuple[tuple[str, str], ...],
    order_facts: Mapping[tuple[str, str], OrderFacts],
) -> tuple[tuple[str, str] | None, tuple[str, str] | None]:
    """The entries before and after *entry* in its recording's order.

    An entry absent from *ordered* has neither. A neighbor that *entry* does not
    continue, or that does not continue *entry* (:func:`adjacent`), is none.
    """
    try:
        idx = ordered.index(entry)
    except ValueError:
        return None, None
    prev_entry = ordered[idx - 1] if idx > 0 else None
    next_entry = ordered[idx + 1] if idx < len(ordered) - 1 else None
    facts = order_facts.get(entry)
    if prev_entry is not None and not adjacent(order_facts.get(prev_entry), facts):
        prev_entry = None
    if next_entry is not None and not adjacent(facts, order_facts.get(next_entry)):
        next_entry = None
    return prev_entry, next_entry


def _neighbor_specs(
    neighbor: tuple[str, str] | None,
    per_input_paths_all: list[dict[tuple[str, str], tuple[Path, LoadSpec]]],
) -> tuple[str | None, FileSpecs | None]:
    """A neighbor's entry key and the files every input holds for it."""
    if neighbor is None:
        return None, None
    collected: FileSpecs = [
        path_map_all[neighbor]
        for path_map_all in per_input_paths_all
        if neighbor in path_map_all
    ]
    return make_entry_key(neighbor[0], neighbor[1]), collected or None


def order_values(df: pd.DataFrame, order_col: str) -> npt.NDArray[np.float64]:
    """The order column as a float array, with unreadable cells as NaN.

    Confined to one annotated function for the reason ``read_parquet_table`` is:
    ``pd.to_numeric`` is typed as returning a scalar-or-array union, so every
    direct call leaves its result partially unknown and the unknown spreads into
    whatever compares against it.

    Float rather than int because the order column may be ``time``, and because a
    NaN is how an unreadable cell has to arrive -- it compares false against both
    ends of an interval, which is the right answer for a row whose frame cannot
    be read.
    """
    return np.asarray(pd.to_numeric(df[order_col], errors="coerce"), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class CoreSelector:
    """Which of an overlapped output's rows belong to the entry being computed.

    Replaces the ``(core_start, core_end)`` row offsets, which were measured on
    the *input* and applied to the *output*. That was only ever sound for a
    feature returning one row per input row in input order, and the ``Feature``
    protocol has never required either: a feature that sorts by ``(id, frame)``
    scatters the core rows into one block per individual, one that filters
    returns fewer, and one that reduces to a row per frame returns N times fewer.
    Each produced a right-sized, wrong-content output with no error.

    A frame interval says the same thing about the output that it says about the
    input, whatever the feature did to the rows in between, and it is exact
    because ``verify_overlap_supported`` has already established that the three
    segments do not share frame numbers.

    Attributes:
        order_col: The column the interval is read on.
        first: Lowest frame belonging to the entry, inclusive.
        last: Highest frame belonging to the entry, inclusive.
        entry_key: Named only so a refusal can say which entry it was trimming.
    """

    order_col: str
    first: int
    last: int
    entry_key: str = ""

    def mask(self, df: pd.DataFrame) -> npt.NDArray[np.bool_]:
        """Which rows of *df* fall inside the interval.

        An array rather than a ``Series`` so the result carries no index: it is
        applied to the frame it was computed from, and an index-aligned boolean
        would silently reindex against an output whose index the feature reset.
        """
        values = order_values(df, self.order_col)
        return (values >= self.first) & (values <= self.last)


class OverlapUnsupported(ValueError):
    """``overlap_frames > 0`` was asked of a continuous group that cannot support it.

    Raised when an entry of a continuous group, or a neighbor it reads, records
    no frame extent, or when two neighboring extents share frame numbers. Each
    case would otherwise measure the context window in rows or trim the output
    against a frame axis that several sequences share. It is a ``ValueError``
    because the request is wrong, not the state of the world.
    """


def _describe(entry: str, extent: tuple[int, int] | None) -> str:
    """One entry and its frame range, for a refusal a reader can act on."""
    if extent is None:
        return f"{entry} (frame range not recorded)"
    return f"{entry} (frames {extent[0]}-{extent[1]})"


def verify_overlap_supported(manifest: Manifest, overlap_frames: int) -> None:
    """Refuse *manifest* if its continuous groups cannot be read across a boundary.

    Checked once over the whole manifest before any entry is computed. A run
    that cannot do what was asked then fails before it writes half its outputs.

    An entry outside a continuous group has no neighbors (:func:`build_manifest`)
    and reads only its own frames. It is not checked. Every entry of a
    continuous group must pass two checks:

    - **It and each neighbor it reads record a frame extent.** Without one there
      is no way to know which neighbor comes first, or whether the axis is shared.
    - **The extents are disjoint and increase across each boundary.** Sequences
      that restart their numbering, which is what every mosaic converter writes,
      do not form one axis. Concatenating them hands ``apply`` three rows for
      frame 7, and every feature that sorts or groups by frame then interleaves
      or merges three recordings. No trim can repair numbers computed from that.

    The message of a failed check names both sequences and both ranges.

    Raises:
        OverlapUnsupported: An entry of a continuous group fails a check.
        RuntimeError: An entry outside a continuous group names a neighbor,
            which :func:`build_manifest` does not produce.
    """
    if overlap_frames <= 0:
        return
    for entry in manifest.values():
        if not entry.continuous:
            if entry.prev_entry_key is not None or entry.next_entry_key is not None:
                msg = (
                    f"{entry.entry_key} is outside a continuous group but names "
                    f"a neighbor ({entry.prev_entry_key!r} before, "
                    f"{entry.next_entry_key!r} after). The manifest is "
                    f"inconsistent."
                )
                raise RuntimeError(msg)
            continue
        if entry.core_extent is None:
            msg = (
                f"overlap_frames={overlap_frames} but {entry.entry_key} records "
                f"no frame range, so mosaic cannot tell which sequences neighbour "
                f"it in time. Re-run the conversion, or measure the existing "
                f"tables, so the tracks index records each entry's frame extent."
            )
            raise OverlapUnsupported(msg)
        core = entry.core_extent
        for side, key, extent in (
            ("precedes", entry.prev_entry_key, entry.prev_extent),
            ("follows", entry.next_entry_key, entry.next_extent),
        ):
            if key is None:
                continue
            if extent is None:
                msg = (
                    f"overlap_frames={overlap_frames} but {key}, which {side} "
                    f"{entry.entry_key}, records no frame range. Every sequence a "
                    f"continuous group reads across has to record one."
                )
                raise OverlapUnsupported(msg)
            before, after = (extent, core) if side == "precedes" else (core, extent)
            if before[1] >= after[0]:
                msg = (
                    f"overlap_frames={overlap_frames} needs one frame axis across "
                    f"group {entry.group!r}, but "
                    f"{_describe(entry.entry_key, core)} and "
                    f"{_describe(key, extent)} overlap. Sequences of a continuous "
                    f"group have to be numbered across the whole recording, not "
                    f"restarted at zero for each -- otherwise the same frame "
                    f"number names a different moment in each of them."
                )
                raise OverlapUnsupported(msg)


@dataclass(frozen=True, slots=True)
class _TracksTable:
    """One tracks-index row whose table is on disk."""

    entry: tuple[str, str]
    path: Path
    schema: str
    variant: str


def _tables_on_disk(
    ds: Dataset, df: pd.DataFrame
) -> tuple[list[_TracksTable], list[Path]]:
    """Split the rows of *df* into the tables on disk and the paths that are not."""
    tables: list[_TracksTable] = []
    missing: list[Path] = []
    for _, series in df.iterrows():
        # Materialized as a string-keyed mapping once per row. A pandas ``Series``
        # is keyed by an untyped index, so every cell read off it is untyped too
        # -- and the columns here are all read as text. Cheap beside the ``Series``
        # ``iterrows`` already builds.
        row: dict[str, object] = {str(key): value for key, value in series.items()}
        path = ds.resolve_path(str(row["abs_path"]))
        if not path.exists():
            missing.append(path)
            continue
        tables.append(
            _TracksTable(
                entry=(str(row["group"]), str(row["sequence"])),
                path=path,
                # Read straight off the row rather than through ``.get``: the
                # reader projects onto the typed schema, so the column is always
                # present and an index predating it carries "" rather than being
                # absent.
                schema=text_cell(row["std_format"]),
                variant=str(row["run_id"]),
            )
        )
    return tables, missing


def _entry_selected(entry: tuple[str, str], selector: Scope) -> bool:
    """Whether *selector* covers *entry*.

    Each test is against ``None`` and never against emptiness. A selector that
    lists nothing names nothing, the rule every other narrowing follows, from
    ``IndexCSV.read`` to ``narrow_target``.
    """
    group, sequence = entry
    if selector.groups is not None and group not in selector.groups:
        return False
    if selector.sequences is not None and sequence not in selector.sequences:
        return False
    pairs = selector.entry_pairs
    return pairs is None or entry in pairs


def _resolve_tracks(
    ds: Dataset,
    run_id: str | None,
    scope: Scope | None = None,
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
    ``run_feature(ds, f, scope=Scope(sequences=["a"]))`` and the same with
    ``["b"]`` would mint three identifiers for one computation on a mixed
    dataset, and ``Pipeline.clean``
    -- whose keep set is the identifiers it predicted -- would delete two of
    them. Rows whose ``run_id`` is empty contribute nothing, so a dataset that
    predates variants yields an empty tuple and hashes exactly as it always has.
    """
    selector = scope if scope is not None else Scope()
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
    resolved_variants: set[str] = set()
    schema_by_entry: dict[tuple[str, str], str] = {}
    tables, missing = _tables_on_disk(ds, df)
    for table in tables:
        path_map_all[table.entry] = (table.path, ParquetLoadSpec())
        all_entries.append(table.entry)
        schema_by_entry[table.entry] = table.schema
        # Collected here, off the *unscoped* rows, and deliberately not below
        # in the narrowing loop -- see the note in the docstring.
        if table.variant:
            resolved_variants.add(table.variant)

    if missing and not all_entries:
        print(
            f"[manifest] all {len(missing)} tracks row(s) resolve to missing "
            f"files; first: {missing[0]}. A moved or synced dataset needs "
            "ds.make_portable() / ds.rewrite_index_paths().",
            file=sys.stderr,
        )

    # Sort by (group, sequence) for stable ordering
    full_order = sorted(set(all_entries))

    # Filter for scoped subset.
    scoped: set[tuple[str, str]] = set()
    path_map: dict[tuple[str, str], tuple[Path, LoadSpec]] = {}
    for entry, spec in path_map_all.items():
        if _entry_selected(entry, selector):
            scoped.add(entry)
            path_map[entry] = spec

    _refuse_mixed_schemas((entry, schema_by_entry[entry]) for entry in scoped)

    return ResolvedInput(
        scoped,
        path_map,
        full_order,
        path_map_all,
        tuple(sorted(resolved_variants)),
        {entry: schema_by_entry[entry] for entry in scoped},
    )


def tracks_variants_for(
    ds: Dataset, tracks_run_id: str | None = None
) -> tuple[str, ...]:
    """Which tracks recipes this dataset's tables answer to under *tracks_run_id*.

    The ``_tracks`` term of a run identifier, asked without building a manifest.
    A planner needs exactly this and nothing else a manifest carries: it is
    hashing what a run *would* read, and the paths it would read them from are
    not part of the answer.

    It goes through the resolver rather than reading the index itself, because
    which rows count is a rule with two halves that must not drift -- a variant
    is collected before the scope narrowing, and only from rows whose table is
    actually on disk. A second reader would eventually disagree with the first,
    and the way it would show is a hash that moved without the data moving.
    """
    return _resolve_tracks(ds, tracks_run_id, None, "empty").tracks_variants


def tracks_table_columns(
    ds: Dataset, tracks_run_id: str | None, scope: Scope
) -> dict[tuple[str, str], tuple[str, ...]]:
    """The column names of each tracks table that *scope* resolves to.

    Read from each parquet's footer and not its data. The resolver chooses the
    table of each entry, as it does for a run. An entry without a table is left
    out, and so is a table whose footer does not parse.

    Args:
        ds: The dataset whose tracks index to read.
        tracks_run_id: Which variant to resolve, or ``None`` for whichever each
            entry has. A variant without rows yet does not resolve a table.
        scope: The entries to read.
    """
    resolved = _resolve_tracks(ds, tracks_run_id, scope, "empty")
    found: dict[tuple[str, str], tuple[str, ...]] = {}
    for entry, (path, _) in sorted(resolved.path_map.items()):
        names = read_parquet_column_names(path)
        if names is not None:
            found[entry] = names
    return found


def refuse_mixed_track_schemas(
    ds: Dataset,
    *,
    tracks_run_id: str | None = None,
    scope: Scope | None = None,
) -> None:
    """Raise if this scope holds tracks tables of incompatible schemas.

    The same question ``build_manifest`` asks on its way to loading, asked on its
    own so a caller can refuse *before* doing work rather than partway through
    it. It reads rows through what the resolver reads them through, because a
    second rule would eventually disagree with the first about which rows count,
    and the disagreement would show as a run that loaded a mixture the check said
    was fine.

    **It asks about families and never chooses a variant.** A dataset holding two
    variants of an entry leaves that choice to whoever reads the entry, so every
    candidate counts here, and a tracker sweep of one family passes. Two families
    on one entry raise, whichever one a reader would have named.

    Args:
        ds: The dataset whose tracks index to read.
        tracks_run_id: Which variant to check, or ``None`` for every candidate of
            each entry.
        scope: What to check, or ``None`` for everything.

    Raises:
        ValueError: Naming the families and a sample of the entries in each.
    """
    selector = scope if scope is not None else Scope()
    candidates = variant_candidate_rows(read_tracks_index(ds), tracks_run_id)
    tables, _ = _tables_on_disk(ds, candidates)
    _refuse_mixed_schemas(
        (table.entry, table.schema)
        for table in tables
        if _entry_selected(table.entry, selector)
    )


def _refuse_mixed_schemas(tables: Iterable[tuple[tuple[str, str], str]]) -> None:
    """Refuse *tables*, ``(entry, recorded schema)`` pairs, of incompatible schemas.

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
    families: dict[str, set[tuple[str, str]]] = {}
    for entry, recorded in tables:
        families.setdefault(schema_family(recorded or LEGACY_SCHEMA), set()).add(entry)
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
        "narrow the scope with a Scope naming groups, sequences or entries "
        "until one run reads one schema."
    )


def _resolve_feature(
    ds: Dataset,
    feature_name: str,
    run_id: str | None,
    scope: Scope | None = None,
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
    selector = scope if scope is not None else Scope()
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
        scope=selector,
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


def _frame_window(
    df: pd.DataFrame | None, overlap_frames: int, *, keep_tail: bool
) -> pd.DataFrame | None:
    """The last (or first) *overlap_frames* frame numbers of a neighbour.

    Frames, not rows. ``.iloc[-N:]`` took N *rows*, so a table of M individuals
    yielded ceil(N/M) frames of context, fewer where the last frames had
    dropouts, and it assumed the parquet was stored in temporal order -- which
    the id-major converters do not guarantee. Hence ``max()``/``min()`` over the
    column rather than a position.

    Gaps make the row count vary, which is what "N frames of context" should
    mean. A window that selects nothing is legal -- the neighbour's last N frame
    numbers can all be dropouts -- and the caller treats it as an absent
    neighbour. ``overlap_frames`` wider than the neighbour selects all of it,
    with no special case.
    """
    if df is None or df.empty or COLUMNS.frame_col not in df.columns:
        return df if df is not None and not df.empty else None
    frames = pd.to_numeric(df[COLUMNS.frame_col], errors="coerce")
    if frames.isna().all():
        return None
    if keep_tail:
        window = df[frames > int(frames.max()) - overlap_frames]
    else:
        window = df[frames < int(frames.min()) + overlap_frames]
    return None if window.empty else window


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
) -> Iterator[tuple[str, pd.DataFrame, CoreSelector]]: ...


def iter_manifest(
    manifest: Manifest,
    *,
    filter_factory: FilterFactory | None = None,
    overlap_frames: int | None = None,
    progress_label: str = "",
    progress_interval: int = 10,
    cross_join: bool = False,
) -> Iterator[tuple[str, pd.DataFrame] | tuple[str, pd.DataFrame, CoreSelector]]:
    """Iterate manifest entries, yielding data per sequence.

    When overlap_frames is None (default), yields (entry_key, df).
    When overlap_frames is an int, yields (entry_key, df, selector) -- see
    :class:`CoreSelector`, which names the rows of the *output* that belong to
    this entry.

    Call :func:`verify_overlap_supported` before iterating with
    ``overlap_frames > 0``. This does not re-check per entry: the selector's
    exactness rests on the segments not sharing frame numbers, and that is
    established once for the whole manifest rather than rediscovered here.

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
            order_col = resolve_order_col(df)
            core_frames = pd.to_numeric(df[order_col], errors="coerce")
            selector = CoreSelector(
                order_col=order_col,
                first=int(core_frames.min()),
                last=int(core_frames.max()),
                entry_key=entry_key,
            )
            if overlap_frames > 0:
                # Neighbours are loaded whole, with their own filters, and then
                # windowed by frame value. The filters have to run first: the
                # nearest-neighbour filter is built from a feature table keyed on
                # the frames as written, so it must meet them unaltered.
                prev_df = _frame_window(
                    _load_neighbor(
                        entry.prev_file_specs,
                        entry.prev_entry_key,
                        filter_factory,
                    ),
                    overlap_frames,
                    keep_tail=True,
                )
                next_df = _frame_window(
                    _load_neighbor(
                        entry.next_file_specs,
                        entry.next_entry_key,
                        filter_factory,
                    ),
                    overlap_frames,
                    keep_tail=False,
                )
                parts = [p for p in (prev_df, df, next_df) if p is not None]
                df = pd.concat(parts, ignore_index=True)

            yield entry_key, df, selector

        del df
        gc.collect()
        pa.default_memory_pool().release_unused()

        if progress_label and ((i + 1) % progress_interval == 0 or i == n_entries - 1):
            print(
                f"[{progress_label}] Processed {i + 1}/{n_entries} sequences",
                file=sys.stderr,
            )
