"""Data loading, artifact resolution, and key-data helpers.

Moved from ``mosaic.behavior.feature_library.helpers`` to live closer to the
pipeline infrastructure they depend on.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..filter_bounds import TimeWindow
from ..helpers import (
    RowEvents,
    events_for_frames,
    filter_time_range,
    load_labels_for_feature_frames,
)
from ..schema import ARENA_COLUMN
from ..scope import Scope
from ._loaders import JoblibLoadSpec, LoadSpec, ParquetLoadSpec
from .column_contract import (
    ParamColumn,
    Provides,
    contract_fields,
    declared_provides,
    declared_requires,
    needs_of,
    param_values,
    resolve_names,
)
from .parquet_reader import ParquetUnreadable, read_footer, read_parquet_column_names
from .index import (
    feature_run_roots,
    feature_run_rows,
    latest_feature_run_root,
    resolve_artifact_file,
)
from .types import ArtifactSpec, NNResult
from .types.labels import GroundTruthLabelsSource, LabelsSource
from .types.data_config import (
    ALIGN_COLS,
    COLUMNS,
    ID_COLS,
    NON_PAYLOAD_COLS,
    SUMMARY_KEY_ORDER,
    SUMMARY_KEYS,
)

if TYPE_CHECKING:
    from ..dataset import Dataset
    from .types import Feature

_ALIGN_COLS = ALIGN_COLS

# Features whose inputs legitimately align on frame alone. A closed set rather than
# a per-feature flag: ``Feature`` is a Protocol with 44 standalone implementations,
# so a required attribute is 44 edits, and it cannot live on ``Inputs`` whose dump
# feeds every downstream identifier. ``interaction-crop-pipeline`` merges tracks
# against a pair-level filter and groups afterwards, so its fan-out costs memory
# rather than correctness.
CROSS_JOIN_FEATURES: frozenset[str] = frozenset({"interaction-crop-pipeline"})


def stacks_inputs(feature: object) -> bool:
    """Whether a feature (or feature class) reads several inputs as one stack.

    The other reading of a multi-input step. A merge joins the inputs
    column-wise on the entries they all hold. A **stack** unions their entries:
    each entry is read from the one input that holds it, and the inputs are
    alike runs over disjoint parts of the data, such as one classifier run per
    fold. A feature declares it with the optional class attribute
    ``stacks_inputs = True``, which :func:`~mosaic.core.pipeline.manifest.build_manifest`
    and the recipe planner both read. Unlike :data:`CROSS_JOIN_FEATURES` it is
    open: it changes which entries a step sees, not how rows are joined, and a
    feature outside the registry can declare it.
    """
    return getattr(feature, "stacks_inputs", False) is True


_WIDENING_EMITS: frozenset[str] = frozenset({"as-input", "pair"})


def expands_individuals(feature: object) -> bool:
    """Whether a feature (or feature class) widens individual inputs onto pair rows.

    A feature whose ``emits`` is ``as-input`` or ``pair`` and whose input 0 is
    pair-level reads each individual-level input joined twice onto the pair
    rows, once as the focal ``id1`` and once as the other ``id2`` (see
    :func:`merge_naming`). Its output stays keyed by the pair rows, so an
    ``as-input`` feature still resolves to its first input's level. A feature
    declared ``individual`` or ``unidentified`` keeps the alignment refusal,
    and so does a feature in :data:`CROSS_JOIN_FEATURES`, whose frame-only
    merge is a different join. ``run_feature``, ``mosaic pipeline verify`` and
    the recipe declarations all read it here.
    """
    return (
        getattr(feature, "emits", None) in _WIDENING_EMITS
        and getattr(feature, "name", None) not in CROSS_JOIN_FEATURES
    )


class MultiInputAlignmentError(ValueError):
    """Two inputs cannot be aligned without inventing or dropping rows."""


class EntryRowsMissing(MultiInputAlignmentError):
    """An input holds no rows for an entry whose merge needs them.

    Dropping the empty input would change what the merged frame means. The run
    records the entry as an ``entry_error`` instead of merging the rest.
    """


class PairRowsMissing(EntryRowsMissing):
    """Input 0 of a widening merge holds no pair rows for an entry.

    Dropping the empty input would leave the individual rows of a later input
    as the entry's merged frame, under a consumer whose output is keyed by pair
    rows.
    """


class BroadcastRowsMissing(EntryRowsMissing):
    """An input that joins by the entry or by its summary keys holds no rows.

    Leaving it out would give the entry a merged frame without its columns,
    which differs from the run's other entries.
    """


@dataclass(frozen=True, slots=True)
class AlignmentVerdict:
    """Whether two column sets can be joined, and on what.

    Exported so a caller composing a chain before running it uses the same rule the
    merge enforces, rather than learning at execution.

    Attributes:
        keys: The shared alignment columns.
        levels: The entity level of each input.
        compatible: Whether the inputs can be joined.
        reason: The refusal's text, ``""`` when compatible.
        cause: Why they cannot be joined. ``"levels"`` for inputs at different
            concrete entity levels that share no identity column, ``"no_keys"``
            for inputs that share no alignment column, ``""`` when compatible.
            A caller branches on the cause and never on the reason's text.
    """

    keys: frozenset[str]
    levels: tuple[str, ...]
    compatible: bool
    reason: str
    cause: Literal["", "levels", "no_keys"]


def entity_level_of(columns: Iterable[str]) -> str:
    """``"pair"`` / ``"individual"`` / ``"global"`` from column names alone.

    Name-based, so it answers for a parquet schema without reading data -- which is
    what lets a submit-time check resolve a run's level cheaply.

    **There is one pair spelling, and that is the point.** It used to accept
    ``id_a``/``id_b`` and ``id_A``/``id_B`` beside ``id1``/``id2``, and still
    missed the ``focal_id``-with-a-target pair that ``pair-facing`` and
    ``attention-target`` wrote -- so both read as ``"global"``, no identity, and a
    join against an individual-level input was permitted on ``frame`` alone. A
    list of names that grows every time a feature invents one is a list somebody
    forgets to extend; every producer now writes ``id1`` (the focal) and ``id2``
    (the other), and a feature that does not is caught by
    ``tests/behavior/feature_library/test_pair_identity_convention.py`` rather
    than silently read as unidentified here.
    """
    present = set(columns)
    if "id1" in present and "id2" in present:
        return "pair"
    return "individual" if COLUMNS.id_col in present else "global"


_NO_SHARED_ALIGNMENT = "no shared alignment columns"


def alignment_keys(column_sets: Sequence[Iterable[str]]) -> frozenset[str]:
    """The columns that a merge of these inputs joins on: the shared ``ALIGN_COLS``."""
    sets = [set(columns) for columns in column_sets]
    return frozenset(_ALIGN_COLS.intersection(*sets)) if sets else frozenset()


def alignment_verdict(column_sets: Sequence[Iterable[str]]) -> AlignmentVerdict:
    """Can these inputs be joined? The rule behind :class:`MultiInputAlignmentError`.

    Incompatible when two inputs carry identity at different levels and the keys
    they share carry no identity at all: joining an individual frame to a pair frame
    on ``frame`` alone is a per-frame cartesian product, not an alignment.

    "Carries identity" is ``ID_COLS``, not "a key that is not ``frame`` or ``time``".
    The two were the same set until ``perspective`` joined ``ALIGN_COLS``, and they
    mean different things: ``perspective`` separates two rows of one pair but names
    no individual, so an individual-level frame that happens to carry one must not
    read as sharing identity with a pair-level frame.
    """
    sets = [set(columns) for columns in column_sets]
    keys = alignment_keys(sets)
    levels = tuple(entity_level_of(columns) for columns in sets)
    identity = keys & ID_COLS
    concrete = {level for level in levels if level != "global"}
    if len(concrete) > 1 and not identity:
        return AlignmentVerdict(
            keys,
            levels,
            False,
            f"inputs are at different entity levels ({', '.join(levels)}) and share "
            f"no identity column, so joining on {sorted(keys) or 'nothing'} would "
            f"pair every row of one with every row of the other",
            "levels",
        )
    if not keys:
        return AlignmentVerdict(keys, levels, False, _NO_SHARED_ALIGNMENT, "no_keys")
    return AlignmentVerdict(keys, levels, True, "", "")


@dataclass(frozen=True, slots=True)
class MergeNaming:
    """What merging one more input does to the column names.

    Attributes:
        keys: The columns joined on, sorted.
        dropped: The incoming identity and bookkeeping columns that the merged
            side already contains, dropped from the incoming input. For a
            widening merge, every one of them except the order column and ``id``.
        renamed: Each incoming payload column that the merged side already
            contains, mapped to its numbered name ``<column>__<position>``.
        columns: Every column of the merged frame afterwards.
        widened: For a widening merge, each incoming payload column mapped to
            its names for the focal and the other individual,
            ``<column>__id1`` and ``<column>__id2``. Empty otherwise.
    """

    keys: tuple[str, ...]
    dropped: tuple[str, ...]
    renamed: Mapping[str, str]
    columns: frozenset[str]
    widened: Mapping[str, tuple[str, str]] = field(
        default_factory=dict[str, tuple[str, str]]
    )


def widens(merged: Iterable[str], incoming: Iterable[str]) -> bool:
    """Whether merging *incoming* into *merged* widens individual rows onto pair rows.

    Only under :func:`expands_individuals`, which the caller asks first: the
    merged side is pair-level and the incoming input is individual-level.
    """
    return (
        entity_level_of(merged) == "pair" and entity_level_of(incoming) == "individual"
    )


def widening_hint(levels: Iterable[str]) -> str:
    """The refusal's remedy for a widening consumer that reads its pair input late.

    ``""`` unless *levels* mix individual and pair inputs, so the merge and
    ``can_join`` append the same words to the same refusal.
    """
    present = set(levels)
    if {"individual", "pair"} <= present:
        return (
            "; widening individual rows onto pair rows needs the pair-level input first"
        )
    return ""


def _shared_order_column(left: frozenset[str], right: frozenset[str]) -> str | None:
    """The order column both sides carry, preferred as ``resolve_order_col`` prefers."""
    if COLUMNS.order_by == "frames":
        preference = (COLUMNS.frame_col, COLUMNS.time_col)
    else:
        preference = (COLUMNS.time_col, COLUMNS.frame_col)
    return next((name for name in preference if name in left and name in right), None)


_PAIR_SLOTS: tuple[str, str] = ("id1", "id2")


def _widened_naming(
    left: frozenset[str], right: frozenset[str], position: int
) -> MergeNaming:
    """The naming of a widening merge: payload twice, identity and bookkeeping dropped.

    The keys are the shared order column alone, or, where the sides share none,
    ``id`` followed by the summary keys both sides have in ``SUMMARY_KEY_ORDER``.
    """
    order = _shared_order_column(left, right)
    payload = sorted(right - NON_PAYLOAD_COLS)
    widened: dict[str, tuple[str, str]] = {}
    for column in payload:
        names = tuple(f"{column}__{slot}" for slot in _PAIR_SLOTS)
        if any(name in left for name in names):
            names = tuple(f"{column}__{position}__{slot}" for slot in _PAIR_SLOTS)
        widened[column] = (names[0], names[1])
    dropped = tuple(sorted((right & NON_PAYLOAD_COLS) - {order, COLUMNS.id_col}))
    added = frozenset(name for names in widened.values() for name in names)
    if order is not None:
        keys = (order,)
    else:
        keys = (
            COLUMNS.id_col,
            *(key for key in SUMMARY_KEY_ORDER if key in left and key in right),
        )
    return MergeNaming(
        keys=keys,
        dropped=dropped,
        renamed={},
        columns=left | added,
        widened=widened,
    )


def merge_naming(
    merged: Iterable[str],
    incoming: Iterable[str],
    position: int,
    *,
    widen: bool = False,
    keys: Collection[str] | None = None,
) -> MergeNaming:
    """The column names after merging *incoming*, the input at *position*.

    The naming rule of a multi-input merge, apart from the data. The merge
    applies it, and pipeline validation applies it to declared column sets. The
    two then agree on the columns of a step's merged input.

    A collided ``NON_PAYLOAD_COLS`` column is **dropped** from the incoming
    input, and not numbered. Identity is either joined on or constant within an
    entry, and the two copies state the same value. Of two bookkeeping copies
    the merged frame keeps the first input's, and a later ``nn_id`` that names
    a different neighbor is lost. A numbered copy such as ``perspective__1`` or
    ``nn_id__1`` matches no exclusion list, and ``perspective__1`` was once read
    downstream as a measurement and fitted as data. Only payload keeps the
    ``__<position>`` suffix.

    A widening merge (*widen*, see :func:`widens`) joins an individual-level
    input onto pair rows, once on ``id1`` and once on ``id2``, with the order
    column, or with the summary keys that both sides have where they share no
    order column. Each of its payload columns appears twice, as ``<column>__id1``
    for the focal and ``<column>__id2`` for the other, or as
    ``<column>__<position>__id1`` and ``<column>__<position>__id2`` when the
    merged side already holds one of those names. Its identity and bookkeeping
    columns are dropped, and the pair side's stand.

    Args:
        merged: The columns merged so far.
        incoming: The next input's columns.
        position: The next input's declared position among the step's inputs.
        widen: Whether the merge widens *incoming* onto pair rows.
        keys: The columns joined on, in place of the shared ``ALIGN_COLS``. A
            collided column in *keys* is a key and is neither dropped nor
            renamed. Ignored for a widening merge.
    """
    left = frozenset(merged)
    right = frozenset(incoming)
    if widen:
        return _widened_naming(left, right, position)
    joined = alignment_keys([left, right]) if keys is None else frozenset(keys)
    collided = sorted((right - joined) & left)
    dropped = tuple(column for column in collided if column in NON_PAYLOAD_COLS)
    renamed = {
        column: f"{column}__{position}"
        for column in collided
        if column not in NON_PAYLOAD_COLS
    }
    kept = (right - frozenset(dropped) - frozenset(renamed)) | frozenset(
        renamed.values()
    )
    return MergeNaming(
        keys=tuple(sorted(joined)),
        dropped=dropped,
        renamed=renamed,
        columns=left | kept,
    )


FallbackKind = Literal["summary", "entry"]
"""How inputs that share no alignment column join: on summary keys, or by the entry."""


FallbackSide = Literal["merged", "incoming"]
"""Which side of a merge position: the inputs merged so far, or the next input."""


@dataclass(frozen=True, slots=True)
class FallbackJoin:
    """How two inputs that share no alignment column join, decided from names.

    Attributes:
        keys: The summary keys joined on, sorted. Empty for a join by the entry
            and where no join is decided.
        kind: ``"summary"`` for a join on the summary keys both sides have,
            ``"entry"`` for a join by the entry (a cross join within it), and
            ``None`` where neither applies and the merge's own refusal stands.
        side: The unidentified side that has a summary key the other side
            lacks, ``None`` when no side does. With *kind* ``None`` the
            position refuses.
        unshared: That side's summary keys that the other side lacks, in
            ``SUMMARY_KEY_ORDER``. Empty exactly when *side* is ``None``.
        single: For a join by the entry that carries *side*'s summary keys
            onto the other side's rows, the other side, which must then have
            one row. ``None`` elsewhere.
    """

    keys: tuple[str, ...]
    kind: FallbackKind | None
    side: FallbackSide | None = None
    unshared: tuple[str, ...] = ()
    single: FallbackSide | None = None


def fallback_join(merged: Iterable[str], incoming: Iterable[str]) -> FallbackJoin:
    """How *incoming* joins *merged* where they share no alignment column.

    A pure function of two column sets, which the merge and the plan-time check
    both read. A side is unidentified when it has none of ``ID_COLS``.

    An unidentified side that has a summary key the other side lacks refuses,
    because its value cannot be repeated onto rows it does not describe: the
    result names the side and every such key. The exception is a join by the
    entry where the other side is unidentified and has none of
    ``SUMMARY_KEYS``: the result keeps the side and the keys, and names the
    other side as *single*, the one that must have one row. Else the sides join
    on the ``SUMMARY_KEYS`` that both have. Else a join by the entry applies
    where one side is unidentified. With neither side unidentified and no
    shared key nothing is decided.

    Args:
        merged: The columns merged so far.
        incoming: The next input's columns.
    """
    left = frozenset(merged)
    right = frozenset(incoming)
    sides: tuple[tuple[FallbackSide, frozenset[str], frozenset[str]], ...] = (
        ("merged", left, right),
        ("incoming", right, left),
    )
    for side, columns, other in sides:
        if columns & ID_COLS:
            continue
        unshared = tuple(
            key for key in SUMMARY_KEY_ORDER if key in columns and key not in other
        )
        if not unshared:
            continue
        if other & ID_COLS or other & SUMMARY_KEYS:
            return FallbackJoin((), None, side, unshared)
        single: FallbackSide = "incoming" if side == "merged" else "merged"
        return FallbackJoin((), "entry", side, unshared, single)
    shared = tuple(sorted(SUMMARY_KEYS & left & right))
    if shared:
        return FallbackJoin(shared, "summary")
    if not left & ID_COLS or not right & ID_COLS:
        return FallbackJoin((), "entry")
    return FallbackJoin((), None)


def _unshared_refusal(fallback: FallbackJoin, position: int) -> str:
    """The merge's refusal of a *fallback* that names an unshared summary key."""
    merged_side, incoming_side = "the inputs before it", f"input {position}"
    if fallback.side == "merged":
        side, other = merged_side, incoming_side
    else:
        side, other = incoming_side, merged_side
    key = fallback.unshared[0]
    return (
        f"cannot merge input {position}: {side} names nobody and has {key}, "
        f"which {other} lacks. Its {key} cannot be repeated onto rows it does "
        "not describe"
    )


MergeKind = Literal["shared", "widen", "summary", "entry"]
"""How a merge position joins.

On shared keys, by widening, on summary keys alone, or by the entry.
"""


@dataclass(frozen=True, slots=True)
class MergeStep:
    """The column-level decision of one merge position.

    Attributes:
        kind: ``"shared"`` for a join on the shared alignment columns and the
            summary keys both sides have, ``"widen"`` for individual rows
            widened onto pair rows, ``"summary"`` for a join on the summary
            keys alone, ``"entry"`` for a join by the entry (a cross join).
        keys: The columns joined on, sorted for a shared or summary join. For
            a widening join the shared order column, or ``id`` followed by the
            shared summary keys in ``SUMMARY_KEY_ORDER``, so that the join also
            takes ``id1`` and ``id2`` as the pair rows' slots. Empty for a join
            by the entry.
        naming: What the merge does to the column names.
        summary_keys: The summary keys a shared join uses as keys, sorted. Empty
            for any other kind.
        refusal: The refusal's text, ``""`` when the position merges.
        single: For a join by the entry that carries an unidentified side's
            summary keys, the side that must have one row, else ``None``.
        single_refusal: The text with which a merge refuses when the *single*
            side has more than one row, ``""`` when *single* is ``None``.
    """

    kind: MergeKind
    keys: tuple[str, ...]
    naming: MergeNaming
    summary_keys: tuple[str, ...]
    refusal: str
    single: FallbackSide | None = None
    single_refusal: str = ""


def merge_step(
    merged: Iterable[str],
    incoming: Iterable[str],
    position: int,
    *,
    cross_join: bool,
    expands_individuals: bool,
) -> MergeStep:
    """How merging *incoming*, the input at *position*, into *merged* joins.

    A pure function of the two column sets: the one decision a merge position
    makes from column names. The merge folds it over the columns of the loaded
    frames and applies each decision to the data, adding only the checks that
    read rows.

    Under *expands_individuals* an individual-level input after pair rows
    widens onto them, on the shared order column or else on ``id`` and the
    summary keys both sides have, and joins no summary key in step 1 either
    way. Otherwise :func:`alignment_verdict` decides, and a
    *cross_join* consumer joins on the shared alignment columns it finds. A
    join on shared keys uses the shared ``ALIGN_COLS`` and every
    ``SUMMARY_KEYS`` column that both sides have, except for a *cross_join*
    consumer, which uses the alignment columns alone. Where the sides share no
    alignment column and the consumer is not a *cross_join* one,
    :func:`fallback_join` decides: a join on the summary keys, a join by the
    entry, or its refusal. Neither adds to ``summary_keys``. A join by the entry
    that carries an unidentified side's summary keys also names the side that
    must have one row and the text that refuses otherwise.

    Args:
        merged: The columns merged so far.
        incoming: The next input's columns.
        position: The next input's declared position among the step's inputs.
        cross_join: Whether the consumer declares a frame-only merge.
        expands_individuals: Whether the consumer widens individual inputs onto
            pair rows.
    """
    left = frozenset(merged)
    right = frozenset(incoming)
    prefix = f"cannot merge input {position}: "
    if expands_individuals and widens(left, right):
        naming = merge_naming(left, right, position, widen=True)
        return MergeStep("widen", naming.keys, naming, (), "")
    verdict = alignment_verdict([left, right])
    if verdict.cause == "no_keys" and not cross_join:
        fallback = fallback_join(left, right)
        naming = merge_naming(left, right, position, keys=fallback.keys)
        if fallback.kind is None and fallback.unshared:
            refusal = _unshared_refusal(fallback, position)
            return MergeStep("shared", (), naming, (), refusal)
        if fallback.kind is not None:
            single_refusal = (
                _unshared_refusal(fallback, position) if fallback.single else ""
            )
            return MergeStep(
                fallback.kind,
                fallback.keys,
                naming,
                (),
                "",
                fallback.single,
                single_refusal,
            )
    summary_keys = (
        ()
        if cross_join or not verdict.keys
        else tuple(sorted(SUMMARY_KEYS & left & right))
    )
    naming = merge_naming(left, right, position, keys=verdict.keys.union(summary_keys))
    refusal = ""
    if not verdict.compatible and not cross_join:
        hint = (
            widening_hint(verdict.levels)
            if expands_individuals and verdict.cause == "levels"
            else ""
        )
        refusal = f"{prefix}{verdict.reason}{hint}"
    elif not naming.keys:
        refusal = f"{prefix}{_NO_SHARED_ALIGNMENT}"
    return MergeStep("shared", naming.keys, naming, summary_keys, refusal)


def summary_join_cell(
    column_sets: Sequence[Iterable[str] | None],
    *,
    cross_join: bool,
    expands_individuals: bool,
) -> str:
    """The summary keys that step 1 of the merge joins, as an entry's recorded cell.

    Folds :func:`merge_step` over *column_sets* as the merge folds it over the
    loaded frames, and returns the sorted union of each position's
    ``summary_keys`` joined with ``,``. A refusal ends the fold, since the
    entry's merge fails there. A cross-join consumer, a widening position and a
    position that shares no summary key add none.

    Args:
        column_sets: The columns of each input, by declared position. ``None``
            for an input that is left out of the merge because it loads no rows.
        cross_join: Whether the consumer declares a frame-only merge.
        expands_individuals: Whether the consumer widens individual inputs onto
            pair rows.

    Returns:
        The keys, and ``""`` when none is joined.
    """
    merged: frozenset[str] | None = None
    joined: set[str] = set()
    for position, columns in enumerate(column_sets):
        if columns is None:
            continue
        if merged is None:
            merged = frozenset(columns)
            continue
        step = merge_step(
            merged,
            columns,
            position,
            cross_join=cross_join,
            expands_individuals=expands_individuals,
        )
        if step.refusal:
            break
        joined.update(step.summary_keys)
        merged = step.naming.columns
    return ",".join(sorted(joined))


def entry_summary_join(
    file_specs: Sequence[tuple[Path, LoadSpec]],
    *,
    cross_join: bool,
    expands_individuals: bool,
) -> str:
    """The summary keys step 1 joins for one entry, read from its inputs' footers.

    Each input's column names and row count come from one
    :func:`~.parquet_reader.read_footer`, which a footer memo answers without a
    second parse. An input is left out as :func:`load_entry_data` leaves it out:
    a spec that is not a parquet, a footer that records no rows, and a file whose
    footer cannot be read.

    Args:
        file_specs: The entry's input files, in declared order.
        cross_join: Whether the consumer declares a frame-only merge.
        expands_individuals: Whether the consumer widens individual inputs onto
            pair rows.

    Returns:
        What :func:`summary_join_cell` returns for those columns.
    """
    column_sets: list[tuple[str, ...] | None] = []
    for path, load_spec in file_specs:
        names: tuple[str, ...] | None = None
        if isinstance(load_spec, ParquetLoadSpec):
            try:
                footer = read_footer(path)
            except ParquetUnreadable:
                footer = None
            if footer is not None and footer.n_rows > 0:
                names = tuple(column.name for column in footer.columns)
        column_sets.append(names)
    return summary_join_cell(
        column_sets, cross_join=cross_join, expands_individuals=expands_individuals
    )


@dataclass(frozen=True, slots=True)
class ColumnProjection:
    """The columns of an input table that a step's output depends on.

    Attributes:
        names: The columns read where the table has them.
        prefixes: The prefixes of column families, each family read whole.
    """

    names: frozenset[str]
    prefixes: tuple[str, ...] = ()

    def select(self, available: Sequence[str]) -> list[str]:
        """The columns of *available* to read, in their order there."""
        return [
            name
            for name in available
            if name in self.names or name.startswith(self.prefixes)
        ]


_CARRIED_COLUMNS_FIELD = "keep_columns"
"""The params field that names input columns an aggregate's output carries."""


def _carries_columns(provides: Provides) -> bool:
    """Whether an output carries the columns its ``keep_columns`` names."""
    refs = [
        *provides.adds,
        *(conditional.column for conditional in provides.conditional),
    ]
    return any(
        isinstance(ref, ParamColumn) and ref.field == _CARRIED_COLUMNS_FIELD
        for ref in refs
    )


def declares_every_read(feature: object) -> bool:
    """Whether a feature (or feature class) declares every column it reads.

    A feature declares it with the optional class attribute
    ``declares_every_read = True``, and its input tables are then read for the
    columns of :func:`input_projection` alone. Before setting it, audit
    ``apply``, ``fit`` and every helper they call for each column they read:
    each must be named by ``requires``, be the ``when_present`` column of a
    ``provides`` condition, or be an identity, bookkeeping or ``arena`` column.
    A column read only where present, or named by a param that no declaration
    reads, rules the attribute out until it is declared. Undeclared, a feature
    reads every column.
    """
    return getattr(feature, "declares_every_read", False) is True


def input_projection(feature: Feature) -> ColumnProjection | None:
    """The columns of its input table that *feature*'s output depends on.

    They are the identity, bookkeeping and ``arena`` columns, the columns its
    ``requires`` names under its params, and the input columns whose presence
    its ``provides`` makes a condition of an output column.

    ``None`` means every column. It is returned for a feature that does not
    declare every column it reads (:func:`declares_every_read`), and when the
    output may depend on columns the declarations do not list: a step of more
    than one input, whose merge names collided columns by what each input
    holds; a feature that declares no ``requires`` or no ``provides``; a
    requirement of any payload column, or one whose params value cannot be read
    as column names; an output that passes its input through, writes a family
    per payload column, carries the columns that ``keep_columns`` names, or
    holds a column the data alone decides.
    """
    if not declares_every_read(feature) or len(feature.inputs.root) != 1:
        return None
    requires = declared_requires(type(feature))
    provides = declared_provides(type(feature))
    if requires is None or provides is None:
        return None
    if provides.passthrough or provides.payload_families or _carries_columns(provides):
        return None
    if any(
        conditional.when_present is None and not conditional.when_param
        for conditional in provides.conditional
    ):
        return None
    values = param_values(feature.params, contract_fields(requires, provides))
    named = [
        *needs_of(requires, values),
        *(
            resolve_names(conditional.when_present, values)
            for conditional in provides.conditional
            if conditional.when_present is not None
        ),
    ]
    if any(one.unknown or one.payload for one in named):
        return None
    names = set(NON_PAYLOAD_COLS | {ARENA_COLUMN})
    prefixes: list[str] = []
    for one in named:
        names.update(one.columns)
        prefixes.extend((*one.prefixes, *one.possible_prefixes))
    return ColumnProjection(frozenset(names), tuple(dict.fromkeys(prefixes)))


def _projected_columns(path: Path, projection: ColumnProjection) -> list[str] | None:
    """The columns of the table at *path* to read, ``None`` for every column.

    All when the footer cannot be read, when the projection keeps every column,
    or when it keeps none, so that the read is the one it would be without it.
    """
    available = read_parquet_column_names(path)
    if available is None:
        return None
    selected = projection.select(available)
    if not selected or len(selected) == len(available):
        return None
    return selected


def load_parquet_dataframe(
    path: Path,
    load_spec: LoadSpec,
    df_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    projection: ColumnProjection | None = None,
) -> pd.DataFrame | None:
    """Load a parquet file as a DataFrame, applying filter if given.

    With *projection*, only the columns it selects are read (see
    :func:`input_projection`), and otherwise every column.

    Returns None for non-parquet specs or if the result is empty after
    filtering.
    """
    if not isinstance(load_spec, ParquetLoadSpec):
        return None
    columns = None if projection is None else _projected_columns(path, projection)
    df = (
        pd.read_parquet(path)
        if columns is None
        else pd.read_parquet(path, columns=columns)
    )
    if df_filter is not None:
        df = df_filter(df)
    if df.empty:
        return None
    return df


def _column_names(df: pd.DataFrame) -> list[str]:
    """*df*'s column labels as strings, the only labels a parquet file stores."""
    labels: list[object] = list(df.columns)
    return [str(label) for label in labels]


def _widen_onto_pairs(
    merged: pd.DataFrame, individuals: pd.DataFrame, step: MergeStep, declared: int
) -> pd.DataFrame:
    """Join *individuals* onto the pair rows of *merged*, as ``id1`` and as ``id2``.

    The keys of *step* are the order column alone, which joins with ``id``, or
    ``id`` and the summary keys both sides have.
    """
    keyed = list(step.keys)
    if COLUMNS.id_col not in keyed:
        keyed.append(COLUMNS.id_col)
    _refuse_missing_summary_keys(merged, individuals, step, declared)
    if individuals.duplicated(keyed).any():
        raise MultiInputAlignmentError(
            f"cannot merge input {declared}: it repeats a ({', '.join(keyed)}) "
            "row, so a pair row would match more than one"
        )
    others = [key for key in keyed if key != COLUMNS.id_col]
    for slot_index, slot in enumerate(_PAIR_SLOTS):
        part = individuals[keyed].rename(columns={COLUMNS.id_col: slot})
        for column, pair in step.naming.widened.items():
            part[pair[slot_index]] = individuals[column]
        merged = merged.merge(part, how="inner", on=[*others, slot])
    return merged


def _refuse_missing_summary_keys(
    merged: pd.DataFrame, incoming: pd.DataFrame, step: MergeStep, declared: int
) -> None:
    """Refuse a join on a summary key that is ``NaN`` in some row of either side.

    pandas matches a ``NaN`` key to a ``NaN`` key, so such a row would join
    rows that name no bin or arena.
    """
    for key in (key for key in step.keys if key in SUMMARY_KEYS):
        for side, table in (
            ("the inputs before it", merged),
            (f"input {declared}", incoming),
        ):
            missing = int(table[key].isna().sum())
            if missing:
                raise MultiInputAlignmentError(
                    f"cannot merge input {declared}: {key} is missing in {missing} "
                    f"rows of {side}, and a summary key names a bin or an arena"
                )


def _refuse_uncovered(
    merged: pd.DataFrame, incoming: pd.DataFrame, keys: list[str], declared: int
) -> None:
    """Refuse a row of the covered side that matches no row of the broadcast side.

    The broadcast side is the one unique on *keys*, the incoming side when both
    are. A row of the broadcast side that matches nothing is not refused, and
    the inner join drops it.
    """
    if incoming.duplicated(keys).any():
        broadcast, covered = merged, incoming
        broadcast_side, covered_side = "the inputs before it", f"input {declared}"
    else:
        broadcast, covered = incoming, merged
        broadcast_side, covered_side = f"input {declared}", "the inputs before it"
    probe = covered[keys].merge(
        broadcast[keys].drop_duplicates(), how="left", on=keys, indicator=True
    )
    unmatched = probe[probe["_merge"] == "left_only"]
    if unmatched.empty:
        return
    first = unmatched.iloc[0]
    values = ", ".join(f"{key}={first[key]}" for key in keys)
    raise MultiInputAlignmentError(
        f"cannot merge input {declared}: {len(unmatched)} rows of {covered_side} "
        f"have no row with the same {', '.join(keys)} in {broadcast_side}, the "
        f"first at {values}"
    )


def _named(incoming: pd.DataFrame, naming: MergeNaming) -> pd.DataFrame:
    """*incoming* with the columns that *naming* drops removed and its renames applied."""
    if naming.dropped:
        incoming = incoming.drop(columns=list(naming.dropped))
    if naming.renamed:
        incoming = incoming.rename(columns=dict(naming.renamed))
    return incoming


def _merge_on_shared_keys(
    merged: pd.DataFrame,
    incoming: pd.DataFrame,
    step: MergeStep,
    declared: int,
    *,
    cross_join: bool,
) -> pd.DataFrame:
    """Inner-join *incoming* onto *merged* on the keys of *step*.

    Applies the naming of :func:`merge_step` and adds the checks that read rows:
    a ``NaN`` summary key, a key set that is not unique on either side, and, for
    a join on summary keys alone, a row of one side that the other does not
    cover.
    """
    on_cols = list(step.keys)
    _refuse_missing_summary_keys(merged, incoming, step, declared)
    if not cross_join and (
        merged.duplicated(on_cols).any() and incoming.duplicated(on_cols).any()
    ):
        raise MultiInputAlignmentError(
            f"cannot merge input {declared}: {on_cols} is not unique on either "
            "side, so the join would multiply rows"
        )
    if step.kind == "summary":
        _refuse_uncovered(merged, incoming, on_cols, declared)
    return merged.merge(_named(incoming, step.naming), how="inner", on=on_cols)


def _merge_by_entry(
    merged: pd.DataFrame, incoming: pd.DataFrame, step: MergeStep, declared: int
) -> pd.DataFrame:
    """Join every row of one side to the row of the other, which holds the entry's value.

    A *step* that names a ``single`` side refuses unless that side has one row.
    The side's value is the entry's, and the other side carries summary keys that
    describe its own rows. Otherwise one side must have one row.
    """
    if step.single is not None:
        holder = merged if step.single == "merged" else incoming
        if len(holder) > 1:
            raise MultiInputAlignmentError(step.single_refusal)
    if len(merged) > 1 and len(incoming) > 1:
        raise MultiInputAlignmentError(
            f"cannot merge input {declared}: it shares no key with the inputs "
            "before it and joins by the entry, and both sides have more than one "
            "row. An input joined by the entry has one row per entry, or one per "
            "bin or arena when both sides have bin_start and bin_end or arena"
        )
    return merged.merge(_named(incoming, step.naming), how="cross")


def _merge_parquet_inputs(
    dfs: Iterable[tuple[int, pd.DataFrame]],
    *,
    cross_join: bool = False,
    expands_individuals: bool = False,
    empty_inputs: Sequence[tuple[int, tuple[str, ...]]] = (),
) -> pd.DataFrame | None:
    """Merge parquet inputs on the identity columns they share.

    Each item carries the input's **declared** position, not its position among the
    ones that happened to load: the suffix on a collided column used to count
    survivors, so with an empty middle input a later input's column was renamed to
    an earlier input's index and read back as the wrong column entirely.

    Refuses rather than inventing rows -- see :func:`alignment_verdict`. Two escapes,
    both narrow: *cross_join* for a feature that declares a frame-only merge, and
    the multiplicity check below, which allows a one-to-many join (a per-frame table
    against a per-frame-per-id one) while refusing many-to-many.

    With *expands_individuals* (see :func:`expands_individuals`), an
    individual-level input after pair rows is widened onto them instead of
    refused. Inputs that share no alignment column join on their summary keys,
    or by the entry (see :func:`fallback_join`), each row of one side taking the
    row of the other that it matches.

    Each position folds :func:`merge_step` over the columns of the frames, which
    decides the keys, the names and the refusals that columns alone settle. The
    merge applies that decision and checks the rows.

    *empty_inputs* holds the declared position and footer names of each input that
    loaded no rows. It is read once *dfs* is exhausted, so a caller that fills it
    while *dfs* loads passes the list itself.

    Raises:
        BroadcastRowsMissing: Another input loaded rows, and an empty input names
            no identity column and shares no alignment column with the merged
            inputs (see :func:`_unplaceable_empty_input`).
    """
    it = iter(dfs)
    first = next(it, None)
    if first is None:
        return None
    merged = first[1]

    for declared, df_next in it:
        step = merge_step(
            _column_names(merged),
            _column_names(df_next),
            declared,
            cross_join=cross_join,
            expands_individuals=expands_individuals,
        )
        if step.refusal:
            raise MultiInputAlignmentError(step.refusal)
        if step.kind == "widen":
            merged = _widen_onto_pairs(merged, df_next, step, declared)
        elif step.kind == "entry":
            merged = _merge_by_entry(merged, df_next, step, declared)
        else:
            merged = _merge_on_shared_keys(
                merged, df_next, step, declared, cross_join=cross_join
            )

    lost = _unplaceable_empty_input(merged, empty_inputs)
    if lost is not None:
        raise BroadcastRowsMissing(
            f"input {lost} has no rows for this entry, and the other inputs "
            "join it by the entry or by its summary keys. Their rows cannot be "
            "written without its columns"
        )
    if merged.empty:
        return None
    return merged


def entry_bounds_filters(
    entry_key: str,
    frame_start: int | None,
    frame_end: int | None,
    time_window: TimeWindow,
) -> list[Callable[[pd.DataFrame], pd.DataFrame]]:
    """The filters of an entry's frame bounds and time window, in that order.

    An unset bound adds no filter. The run's filter factory and a worker that
    loads its entry both filter through these.

    Args:
        entry_key: The manifest key of the entry, which the time window's error
            names.
        frame_start: The first frame kept, ``None`` for no lower bound.
        frame_end: The frame after the last kept, ``None`` for no upper bound.
        time_window: The window of ``time`` kept.

    Returns:
        The filters. The time window's filter raises :class:`ValueError` for a
        table without a ``time`` column.
    """
    filters: list[Callable[[pd.DataFrame], pd.DataFrame]] = []
    if frame_start is not None or frame_end is not None:

        def frame_filter(df: pd.DataFrame) -> pd.DataFrame:
            return filter_time_range(
                df, filter_start_frame=frame_start, filter_end_frame=frame_end
            )

        filters.append(frame_filter)
    if time_window.is_set:

        def window_filter(df: pd.DataFrame) -> pd.DataFrame:
            if COLUMNS.time_col not in df.columns:
                raise ValueError(
                    f"a time filter reads the {COLUMNS.time_col!r} column, and "
                    f"the rows of {entry_key!r} have none. Filter by frame "
                    "with filter_start_frame and filter_end_frame"
                )
            return filter_time_range(
                df,
                filter_start_time=time_window.start,
                filter_end_time=time_window.end,
                time_col=COLUMNS.time_col,
            )

        filters.append(window_filter)
    return filters


def _unplaceable_empty_input(
    merged: pd.DataFrame, empty: Sequence[tuple[int, tuple[str, ...]]]
) -> int | None:
    """The first empty input whose columns the merged frame would lose, else ``None``.

    An empty input is lost when its footer names none of ``ID_COLS`` and shares
    none of ``ALIGN_COLS`` with the columns of the merged inputs. Such an input
    joins by the entry or by its summary keys, so it adds its columns to every
    row of the entry.

    Args:
        merged: The merge of the inputs that loaded rows, named as step 1 names
            them.
        empty: Each empty input's declared position and footer names.
    """
    merged_columns = frozenset(_column_names(merged))
    for position, names in empty:
        columns = frozenset(names)
        if not columns & ID_COLS and not columns & merged_columns & _ALIGN_COLS:
            return position
    return None


def load_entry_data(
    file_specs: list[tuple[Path, LoadSpec]],
    filters: Iterable[Callable[[pd.DataFrame], pd.DataFrame]] = (),
    *,
    cross_join: bool = False,
    expands_individuals: bool = False,
    projection: ColumnProjection | None = None,
) -> pd.DataFrame | None:
    """Load and merge data for a single manifest entry.

    All manifest entries are parquet. Multiple inputs are merged via
    inner join on shared alignment columns. DataFrames are loaded
    lazily and merged incrementally to minimize peak memory. An input that
    loads no rows is left out of the merge, unless the merge would then lose its
    columns. *projection* selects the columns each table is read for, and ``None``
    reads every column.

    Raises:
        PairRowsMissing: Under *expands_individuals*, input 0 is pair-level and
            holds no rows, and a later individual-level input does. Leaving
            input 0 out would merge individual rows for a consumer keyed by
            pair rows.
        BroadcastRowsMissing: Another input loaded rows, and an input that loaded
            none names no identity column and shares no alignment column with the
            merged inputs. Leaving it out would give the entry a frame without the
            columns that the run's other entries have.
    """
    empty: list[tuple[int, tuple[str, ...]]] = []

    def _load_dfs() -> Iterator[tuple[int, pd.DataFrame]]:
        pair_input_empty = False
        for declared, (path, load_spec) in enumerate(file_specs):
            df = load_parquet_dataframe(path, load_spec, projection=projection)
            if df is None:
                names = (
                    read_parquet_column_names(path)
                    if isinstance(load_spec, ParquetLoadSpec)
                    else None
                )
                if names is not None:
                    empty.append((declared, names))
                pair_input_empty = pair_input_empty or (
                    declared == 0
                    and expands_individuals
                    and names is not None
                    and entity_level_of(names) == "pair"
                )
                continue
            if pair_input_empty and entity_level_of(_column_names(df)) == "individual":
                raise PairRowsMissing(
                    f"input 0 holds no pair rows for this entry, so the "
                    f"individual rows of input {declared} have none to widen onto"
                )
            yield declared, df

    merged = _merge_parquet_inputs(
        _load_dfs(),
        cross_join=cross_join,
        expands_individuals=expands_individuals,
        empty_inputs=empty,
    )
    if merged is None:
        return None
    for fn in filters:
        merged = fn(merged)
        if merged.empty:
            return None
    return merged


def _integers(values: npt.NDArray[np.generic]) -> npt.NDArray[np.int64]:
    """*values* as integers, a float truncated toward zero as ``int()`` does.

    Raises:
        ValueError: A float value is NaN or infinite, which ``int()`` refuses.
    """
    if values.dtype.kind == "f" and not bool(np.isfinite(values).all()):
        raise ValueError("cannot convert a NaN or infinite value to an integer")
    return values.astype(np.int64)


@dataclass(frozen=True, slots=True)
class NeighborLookup:
    """Each individual's nearest neighbor in each frame, from a neighbor table.

    A key is an integer ``(frame, id)`` and holds one neighbor id. Of the rows
    naming one key, the last row with a neighbor sets it, and a row whose
    neighbor is NaN sets nothing. An empty lookup is false.

    Attributes:
        keys: Each ``(frame, id)`` key once.
        neighbors: The neighbor id of each key, in the order of *keys*.
    """

    keys: pd.MultiIndex
    neighbors: npt.NDArray[np.int64]

    @classmethod
    def from_rows(
        cls,
        frames: npt.NDArray[np.generic],
        ids: npt.NDArray[np.generic],
        neighbors: npt.NDArray[np.generic],
    ) -> NeighborLookup:
        """The lookup of a neighbor table's ``frame``, ``id`` and ``nn_id``.

        A float frame or id is truncated toward zero, as ``int()`` does.

        Raises:
            ValueError: A row with a neighbor has a NaN or infinite frame or
                id.
        """
        present = ~np.isnan(neighbors)
        keys = pd.MultiIndex.from_arrays(
            [_integers(frames[present]), _integers(ids[present])]
        )
        last = ~keys.duplicated(keep="last")
        return cls(keys=keys[last], neighbors=_integers(neighbors[present])[last])

    @classmethod
    def empty(cls) -> NeighborLookup:
        """The lookup of no row, which keeps every pair."""
        none = np.zeros(0, np.int64)
        return cls.from_rows(none, none, none.astype(np.float64))

    def __len__(self) -> int:
        return len(self.neighbors)

    def neighbor_of(
        self, frames: npt.NDArray[np.int64], ids: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
        """Whether each ``(frame, id)`` has a neighbor, and which one.

        Returns:
            A boolean array of the keys found, and an array of their neighbor
            ids that holds 0 where a key is not found.
        """
        if not len(self):
            return np.zeros(len(frames), dtype=bool), np.zeros(len(frames), np.int64)
        positions = self.keys.get_indexer(pd.MultiIndex.from_arrays([frames, ids]))
        found = positions >= 0
        return found, np.where(found, self.neighbors[positions], 0)


def build_nn_lookup(
    ds: Dataset,
    group: str,
    sequence: str,
    pair_filter_spec: NNResult,
) -> NeighborLookup:
    """
    Build a per-frame nearest-neighbor lookup for a given sequence.

    Returns each ``(frame, individual_id)``'s ``nn_id`` from the
    nearest-neighbor feature parquet, which is read for those columns alone.
    Returns an empty lookup when the NN feature has not been computed for this
    sequence (fail-open).

    ``NNResult`` is a ``Result``, so on the ``run_feature`` path
    :func:`~mosaic.core.pipeline.resolve.resolve_references` has already pinned
    ``pair_filter_spec.run_id`` and the fallback below does not fire. It used to:
    a pair filter resolved to "latest" here, at load time, while the identifier
    that named the run knew only ``None`` -- the same defect as an unpinned
    ``templates``, in a fourth place.
    """

    feature_name = pair_filter_spec.feature
    run_id = pair_filter_spec.run_id
    if run_id is None:
        run_id, _ = latest_feature_run_root(ds, feature_name)

    idx_df = feature_run_rows(
        ds,
        feature_name,
        run_id,
        filter_ext=".parquet",
        scope=Scope(entries=[(group, sequence)]),
    )
    if idx_df.empty:
        return NeighborLookup.empty()

    nn_path = ds.resolve_path(idx_df.iloc[0]["abs_path"])
    names = read_parquet_column_names(nn_path)
    if names is None:
        # An absent or unreadable file raises here, as a read of it does.
        names = tuple(_column_names(pd.read_parquet(nn_path)))

    # below is probably too defensive, NNResult (as a per sequence result) has a known parquet schema
    frame_col = "frame" if "frame" in names else ("time" if "time" in names else None)
    if frame_col is None or "id" not in names or "nn_id" not in names:
        return NeighborLookup.empty()

    df_nn = pd.read_parquet(nn_path, columns=[frame_col, "id", "nn_id"])
    return NeighborLookup.from_rows(
        df_nn[frame_col].to_numpy(), df_nn["id"].to_numpy(), df_nn["nn_id"].to_numpy()
    )


def nn_pair_mask(df: pd.DataFrame, nn_lookup: NeighborLookup) -> npt.NDArray[np.bool_]:
    """
    Return a boolean mask for rows in a pair-feature DataFrame where at
    least one individual in the pair considers the other its nearest neighbor.

    For pair (id1, id2) at frame F the row is kept when:
      - the neighbor of ``(F, id1)`` is ``id2`` (id1's NN is id2), **or**
      - the neighbor of ``(F, id2)`` is ``id1`` (id2's NN is id1).

    Returns all-True when the DataFrame lacks the required columns or
    when *nn_lookup* is empty (fail-open).
    """
    frame_col = (
        "frame" if "frame" in df.columns else ("time" if "time" in df.columns else None)
    )
    if (
        not nn_lookup
        or frame_col is None
        or "id1" not in df.columns
        or "id2" not in df.columns
    ):
        return np.ones(len(df), dtype=bool)

    frames = df[frame_col].to_numpy(dtype=np.int64)
    id1s = df["id1"].to_numpy(dtype=np.int64)
    id2s = df["id2"].to_numpy(dtype=np.int64)

    first_found, first_neighbor = nn_lookup.neighbor_of(frames, id1s)
    second_found, second_neighbor = nn_lookup.neighbor_of(frames, id2s)
    return (first_found & (first_neighbor == id2s)) | (
        second_found & (second_neighbor == id1s)
    )


def nn_pair_filter(
    ds: Dataset, group: str, sequence: str, spec: NNResult
) -> Callable[[pd.DataFrame], pd.DataFrame] | None:
    """Build the filter that keeps the nearest-neighbor pairs of one entry.

    The run's filter factory and an entry's task both build the filter here, so
    each keeps the same rows of an entry.

    Args:
        ds: The dataset that holds the neighbor run, directly or through a link.
        group: The entry's group.
        sequence: The entry's sequence.
        spec: The nearest-neighbor run whose table gives each individual's
            neighbor.

    Returns:
        ``None`` for an empty lookup, which an entry the neighbor run lacks gives
        and which filters nothing. Otherwise a function that keeps the rows of
        :func:`nn_pair_mask` and resets the index.

    Raises:
        Exception: Whatever :func:`build_nn_lookup` raises, such as the error of
            a neighbor table that cannot be read.
    """
    nn_lookup = build_nn_lookup(ds, group, sequence, spec)
    if not nn_lookup:
        return None

    def keep_neighbor_pairs(df: pd.DataFrame) -> pd.DataFrame:
        mask = nn_pair_mask(df, nn_lookup)
        return df.loc[mask].reset_index(drop=True)

    return keep_neighbor_pairs


def row_individual_pairs(df: pd.DataFrame) -> np.ndarray | None:
    """Each row's ``(id1, id2)``, for aligning labels to rows rather than to frames.

    A pair row gives its ``id1`` (the focal) and ``id2``. An individual row gives
    ``(id, -1)``, so an individual label and every pair label focal on that
    individual match it. A missing id is ``-1``, a wildcard.

    Returns:
        An ``(len(df), 2)`` int64 array, or ``None`` when *df* carries no id
        column, or carries ids that are not integers (names, say). ``None``
        means the frame alone aligns a row to its label, as it did before rows
        carried individuals.
    """
    if "id1" in df.columns and "id2" in df.columns:
        columns = [df["id1"], df["id2"]]
    elif COLUMNS.id_col in df.columns:
        columns = [df[COLUMNS.id_col], pd.Series(-1, index=df.index)]
    else:
        return None
    pairs = np.empty((len(df), 2), dtype=np.int64)
    for position, column in enumerate(columns):
        numeric = pd.to_numeric(column, errors="coerce")
        if bool((numeric.isna() & column.notna()).any()):
            return None
        filled = numeric.fillna(-1).to_numpy(dtype=np.float64)
        if not np.all(np.mod(filled, 1) == 0):
            return None
        pairs[:, position] = filled.astype(np.int64)
    return pairs


def labels_for_rows(
    path: Path,
    df: pd.DataFrame,
    labels: LabelsSource,
    *,
    default_label: int | None = None,
) -> np.ndarray:
    """One label per row of *df*, from the label file at *path*.

    The one per-row alignment: extract-labeled-templates, ``load_values`` and the
    scoring helpers all call it, so a row is labeled one way everywhere. A row is
    matched by its frame and its own individuals (:func:`row_individual_pairs`),
    so a pair row takes the label of its own direction, and a row no event
    matches takes the file's background class -- :data:`~mosaic.core.helpers.UNLABELED`
    (``-1``) for a file with none. A :class:`GroundTruthLabelsSource` then
    applies its ``mask`` and ``remap`` (:meth:`GroundTruthLabelsSource.relabel`).

    Args:
        path: The ``.npz`` label file of the row's entry.
        df: Rows with a ``frame`` column.
        labels: The labels source *path* was resolved for.
        default_label: The label for an unmatched row. ``None`` takes the
            file's background class.

    Returns:
        An int64 array of ``len(df)`` labels.

    Raises:
        ValueError: *df* has no ``frame`` column.
    """
    aligned = load_labels_for_feature_frames(
        path,
        _row_frames(df),
        default_label=default_label,
        individual_filter=row_individual_pairs(df),
    )
    if isinstance(labels, GroundTruthLabelsSource):
        return labels.relabel(aligned)
    return aligned


def events_for_rows(path: Path, df: pd.DataFrame) -> RowEvents:
    """The event of the label file at *path* that labels each row of *df*.

    Rows are matched as :func:`labels_for_rows` matches them, so
    ``events_for_rows(path, df).labels`` is its result with ``default_label``
    unset, before label controls. A reader gathers any other per-event column of
    the file from the rows' ``positions``, and ``matches`` counts the events each
    row matched.

    Args:
        path: The ``.npz`` label file of the rows' entry.
        df: Rows with a ``frame`` column.

    Returns:
        Each row's event position, match count and label
        (:class:`~mosaic.core.helpers.RowEvents`).

    Raises:
        ValueError: *df* has no ``frame`` column.
    """
    return events_for_frames(
        path, _row_frames(df), individual_filter=row_individual_pairs(df)
    )


def _row_frames(df: pd.DataFrame) -> np.ndarray:
    """The ``frame`` column of *df*, which aligns its rows to a label file."""
    if COLUMNS.frame_col not in df.columns:
        raise ValueError(
            f"labels align to rows by their {COLUMNS.frame_col!r} column, which "
            f"these rows lack"
        )
    return df[COLUMNS.frame_col].to_numpy()


def resolve_sequence_identity(
    entry_key: str,
    entry_map: dict[str, tuple[str, str]],
) -> tuple[str, str]:
    """Map an entry key to (group, sequence).

    Looks up in entry_map, falls back to ("", entry_key).
    """
    if entry_key in entry_map:
        return entry_map[entry_key]
    return "", entry_key


def get_feature_run_root(
    ds: Dataset,
    feature_name: str,
    run_id: str | None = None,
) -> tuple[str, Path]:
    """
    Resolve (run_id, run_root_path) for a feature.

    If run_id is None, picks the latest finished run. A run that a linked dataset
    holds resolves to that dataset's run root.
    """
    if run_id is None:
        run_id, run_root = latest_feature_run_root(ds, feature_name)
    else:
        from .inventory.params import linked_run_root

        run_root = linked_run_root(ds, feature_name, run_id)
    return str(run_id), run_root


def load_joblib_artifact(ds: Dataset, artifact: ArtifactSpec) -> object:
    """
    Load a joblib artifact from a feature run root.

    Parameters
    ----------
    ds : Dataset
        Dataset instance
    artifact : ArtifactSpec
        Typed artifact specification with feature, run_id, pattern, and load spec.

    Returns
    -------
    object
        The loaded object, or obj[key] if load spec has a key
    """
    if not isinstance(artifact.load, JoblibLoadSpec):
        raise ValueError(
            f"load_joblib_artifact requires JoblibLoadSpec, "
            f"got {type(artifact.load).__name__}"
        )
    run_id = artifact.run_id
    if run_id is None:
        run_id, _ = latest_feature_run_root(ds, artifact.feature)
    for root in feature_run_roots(ds, artifact.feature, run_id):
        resolved = resolve_artifact_file(
            "artifact", artifact.feature, root, artifact.pattern
        )
        if resolved is not None:
            break
    else:
        raise FileNotFoundError(
            f"No files matching '{artifact.pattern}' in the run roots of "
            f"{artifact.feature!r} run {run_id!r}"
        )
    return artifact.from_path(resolved)
