"""The columns of each step of a recipe, propagated from declarations.

A step's output columns follow from its declared ``provides``, its params and its
merged input. Its merged input follows from its inputs' outputs by the naming rule
of the merge (:func:`~mosaic.core.pipeline.loading.merge_naming`). A topological
walk of the recipe therefore states, for every step and with no dataset, the
columns its input is guaranteed to contain and the columns it may contain.

**Two sets per step, because a recipe cannot decide every column.** A column is
*guaranteed* when the recipe decides every condition on it: a params field set, an
upstream column itself guaranteed. It is *possible* when a condition is left to an
upstream column that is only possible, or to the data. The output of a step
without a ``provides`` declaration is *unbounded*: it may contain any column. A
requirement reading it is a caution and not a problem, and an undeclared feature
does not cause a refusal.

The registered track schemas describe the tracks (:func:`tracks_provides`), and
this module does not restate them. A converter also writes columns that its
schema does not name, such as SLEAP's ``poseP0`` and TRex's ``X#head``, and a
step's params may name one. The tracks therefore may contain any column that no
declaration names literally (:func:`tracks_columns`), and a requirement on one is
a caution. A plan reads the tables that a step reads, and states their columns
(:func:`measured_columns`).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal

from mosaic.core.schema import STANDARD_COLUMNS, TRACK_SCHEMAS

from ..column_contract import (
    Column,
    ColumnRef,
    Conditional,
    NamedColumns,
    PAIR_COUNTING_AGGS,
    PairRead,
    PairRule,
    ParamValues,
    Prefix,
    Provides,
    Requirement,
    declared_names,
    is_set,
    needs_of,
    resolve_names,
)
from ..loading import merge_naming, widens
from ..types.data_config import NON_PAYLOAD_COLS
from .compatibility import (
    TRACKS_DECLARATION,
    ConsumerDecl,
    Declaration,
    DeclarationCatalog,
    can_connect,
)
from .model import TRACKS_INPUT
from .wire import (
    ColumnNames,
    ColumnProvider,
    ConditionalColumns,
    NeedStatus,
    ProvidedColumns,
)

__all__ = [
    "ColumnSet",
    "UNKNOWN_COLUMNS",
    "declared_vocabulary",
    "measured_columns",
    "merged_input",
    "names_wire",
    "need_status",
    "output_columns",
    "pair_rule_read",
    "provided_wire",
    "provider_phrase",
    "providers_of",
    "requires_wire",
    "resolve_pair_rule",
    "tracks_columns",
    "tracks_provides",
]

_NO_NAMES: Final[frozenset[str]] = frozenset()

_FAMILY_SEPARATOR: Final = "__"
"""What joins a payload family's column to the rest of a member's name."""

_PERSPECTIVE: Final = "perspective"
"""The column that tells the two rows of a pair apart."""


@dataclass(frozen=True, slots=True)
class ColumnSet:
    """The columns of one step's output, or of one step's merged input.

    Attributes:
        guaranteed: Columns it always contains.
        possible: Columns it may contain, the guaranteed ones included.
        guaranteed_prefixes: Column families it always contains a member of.
        possible_prefixes: Column families it may contain members of, the
            guaranteed ones included.
        unbounded: Whether it may also contain undeclared columns, because a
            step behind it declares none.
        unlisted: Whether it may also contain columns outside *vocabulary*, as
            the tracks may.
        vocabulary: The columns that some declaration names literally. Their
            sources are declared, and an unlisted column is any other.
        symmetric: Columns that hold one value on the two perspective rows of
            a pair in a frame, as a declaration behind them states
            (``Provides.symmetric``).
        symmetric_prefixes: Column families whose every member is symmetric.
        mirrored: Whether the rows hold both perspective rows of every pair
            they hold. A fact of the rows, apart from any column's value.
    """

    guaranteed: frozenset[str] = _NO_NAMES
    possible: frozenset[str] = _NO_NAMES
    guaranteed_prefixes: frozenset[str] = _NO_NAMES
    possible_prefixes: frozenset[str] = _NO_NAMES
    unbounded: bool = False
    unlisted: bool = False
    vocabulary: frozenset[str] = _NO_NAMES
    symmetric: frozenset[str] = _NO_NAMES
    symmetric_prefixes: frozenset[str] = _NO_NAMES
    mirrored: bool = False

    def admits(self, column: str) -> bool:
        """Whether this set may contain *column*."""
        if self.unbounded or column in self.possible:
            return True
        if self.unlisted and column not in self.vocabulary:
            return True
        return any(column.startswith(prefix) for prefix in self.possible_prefixes)

    def is_symmetric(self, column: str) -> bool:
        """Whether *column* holds one value on the two perspective rows of a pair."""
        return column in self.symmetric or any(
            column.startswith(prefix) for prefix in self.symmetric_prefixes
        )


UNKNOWN_COLUMNS: Final = ColumnSet(unbounded=True)
"""The output of an undeclared step, or of a step that did not check out."""


@dataclass(frozen=True, slots=True)
class _Symmetry:
    """The symmetric columns and families of a set, and whether its rows are mirrored."""

    columns: frozenset[str] = _NO_NAMES
    prefixes: frozenset[str] = _NO_NAMES
    mirrored: bool = False


def _column_set(
    guaranteed: Iterable[str],
    possible: Iterable[str],
    guaranteed_prefixes: Iterable[str],
    possible_prefixes: Iterable[str],
    *,
    unbounded: bool,
    unlisted: bool,
    vocabulary: Iterable[str],
    symmetry: _Symmetry = _Symmetry(),
) -> ColumnSet:
    """A set whose possible columns and prefixes include its guaranteed ones."""
    sure = frozenset(guaranteed)
    sure_prefixes = frozenset(guaranteed_prefixes)
    return ColumnSet(
        guaranteed=sure,
        possible=sure | frozenset(possible),
        guaranteed_prefixes=sure_prefixes,
        possible_prefixes=sure_prefixes | frozenset(possible_prefixes),
        unbounded=unbounded,
        unlisted=unlisted,
        vocabulary=frozenset(vocabulary),
        symmetric=symmetry.columns,
        symmetric_prefixes=symmetry.prefixes,
        mirrored=symmetry.mirrored,
    )


# --- one requirement against one set ----------------------------------------------


def _meets_a_family(need: NamedColumns, families: Collection[str]) -> bool:
    """Whether a member of one of *families* may meet *need*.

    A member of any family meets a payload requirement, because a family is a
    measurement such as the pose keypoints.
    """
    if need.payload and families:
        return True
    if any(column.startswith(family) for column in need.columns for family in families):
        return True
    return any(
        family.startswith(prefix) or prefix.startswith(family)
        for prefix in need.prefixes
        for family in families
    )


def _may_be_unlisted(need: NamedColumns, available: ColumnSet) -> bool:
    """Whether a column that no declaration names may meet *need* in *available*."""
    if not available.unlisted:
        return False
    if need.prefixes or need.payload:
        return True
    return any(column not in available.vocabulary for column in need.columns)


def need_status(need: NamedColumns, available: ColumnSet) -> NeedStatus:
    """How a merged input described by *available* answers *need*."""
    if (
        need.met_by(available.guaranteed)
        or (need.payload and bool(available.guaranteed_prefixes))
        or any(
            family.startswith(prefix)
            for prefix in need.prefixes
            for family in available.guaranteed_prefixes
        )
    ):
        return "guaranteed"
    if need.met_by(available.possible) or _meets_a_family(
        need, available.possible_prefixes
    ):
        return "possible"
    if available.unbounded or need.unknown:
        return "unknown"
    if _may_be_unlisted(need, available):
        return "possible"
    return "absent"


# --- the merge ----------------------------------------------------------------------


def _shared(sets: Iterable[frozenset[str]]) -> frozenset[str]:
    """The names every one of *sets* holds."""
    shared: frozenset[str] | None = None
    for one in sets:
        shared = one if shared is None else shared & one
    return _NO_NAMES if shared is None else shared


def merged_input(
    inputs: Sequence[ColumnSet], *, stack: bool = False, widen: bool = False
) -> ColumnSet:
    """The columns of a step's merged input, from the columns of each input.

    With *stack*, for a step that reads its inputs as one stack of disjoint
    entries, no column is renamed: a column is guaranteed when every input
    guarantees it, and possible when any input may contain it.

    Applies the naming rule of the merge once to the guaranteed sets and once to
    the possible ones. A column of the first input keeps its name. A colliding
    payload column of a later input is numbered ``<column>__<position>``, and a
    colliding identity or bookkeeping column is dropped. Over the guaranteed sets
    the rule gives the columns of every merge. Over the possible sets it gives the
    columns that some merge may contain.

    With *widen*, for a step that expands individuals, an individual-level
    input after a pair-level input 0 is named as the widening merge names it,
    each payload column as ``<column>__id1`` and ``<column>__id2``. The levels
    are read from the guaranteed sets, and the possible sets follow them.

    A symmetric column keeps its symmetry under the name the merge gives it. A
    widened column is not symmetric, because its two names hold two
    individuals' values. Under *stack*, a column is symmetric when every input
    declares it so. The merge is an inner join, so the rows are mirrored only
    when every pair-level input's are: an input missing a mirror row drops that
    row's partner from the join. A widened individual-level input drops a
    pair's two rows together, since both need the same two individuals. A
    stack is mirrored when every input is.
    """
    if not inputs:
        return ColumnSet()
    if stack and len(inputs) > 1:
        return _column_set(
            _shared(one.guaranteed for one in inputs),
            (name for one in inputs for name in one.possible),
            _shared(one.guaranteed_prefixes for one in inputs),
            (prefix for one in inputs for prefix in one.possible_prefixes),
            unbounded=any(one.unbounded for one in inputs),
            unlisted=any(one.unlisted for one in inputs),
            vocabulary=(name for one in inputs for name in one.vocabulary),
            symmetry=_Symmetry(
                columns=_shared(one.symmetric for one in inputs),
                prefixes=_shared(one.symmetric_prefixes for one in inputs),
                mirrored=all(one.mirrored for one in inputs),
            ),
        )
    first, *rest = inputs
    guaranteed = first.guaranteed
    possible = first.possible
    symmetric = set(first.symmetric)
    symmetric_prefixes = set(first.symmetric_prefixes)
    for position, incoming in enumerate(rest, start=1):
        widened = widen and widens(guaranteed, incoming.guaranteed)
        guaranteed = merge_naming(
            guaranteed, incoming.guaranteed, position, widen=widened
        ).columns
        naming = merge_naming(possible, incoming.possible, position, widen=widened)
        possible = naming.columns
        if widened:
            continue
        symmetric_prefixes.update(incoming.symmetric_prefixes)
        for column in incoming.symmetric:
            if column in naming.dropped:
                continue
            symmetric.add(naming.renamed.get(column, column))
    pair_inputs = [one for one in inputs if one.admits(_PERSPECTIVE)]
    input_prefixes = (prefix for one in inputs for prefix in one.possible_prefixes)
    return _column_set(
        guaranteed,
        possible,
        (prefix for one in inputs for prefix in one.guaranteed_prefixes),
        (prefix for one in inputs for prefix in one.possible_prefixes),
        unbounded=any(one.unbounded for one in inputs),
        unlisted=any(one.unlisted for one in inputs),
        vocabulary=(name for one in inputs for name in one.vocabulary),
        symmetry=_Symmetry(
            columns=frozenset(symmetric),
            prefixes=_unshadowed(
                symmetric_prefixes, possible, input_prefixes, symmetric
            ),
            mirrored=bool(pair_inputs) and all(one.mirrored for one in pair_inputs),
        ),
    )


# --- a step's output ----------------------------------------------------------------

type _Outcome = Literal["met", "unmet", "undecided"]
"""Whether a condition is met, as far as the recipe decides it."""


def _combined(outcomes: Iterable[_Outcome]) -> _Outcome:
    """The outcome of several conditions that must all be met."""
    seen = set(outcomes)
    if "unmet" in seen:
        return "unmet"
    if "undecided" in seen:
        return "undecided"
    return "met"


def _presence(need: NamedColumns, merged: ColumnSet) -> _Outcome:
    """Whether the merged input contains *need*, as far as the recipe decides."""
    status = need_status(need, merged)
    if status == "guaranteed":
        return "met"
    if status == "absent":
        return "unmet"
    return "undecided"


def _param_outcome(field_name: str, values: ParamValues) -> _Outcome:
    """Whether the params field *field_name* is set, ``met`` when none is named."""
    if not field_name:
        return "met"
    if field_name not in values:
        return "undecided"
    return "met" if is_set(values[field_name]) else "unmet"


def _presence_outcomes(
    ref: ColumnRef | None, count: int, values: ParamValues, merged: ColumnSet
) -> tuple[list[_Outcome], _Outcome]:
    """The presence condition, per output column and for the entry as a whole.

    Pairs by position when the condition names as many columns as the entry
    does, and otherwise requires every column it names of every output column.
    """
    if ref is None:
        return ["met"] * count, "met"
    named = resolve_names(ref, values)
    if named.unknown:
        return ["undecided"] * count, "undecided"
    each: list[_Outcome] = [
        _presence(NamedColumns(columns=(column,)), merged) for column in named.columns
    ]
    each.extend(
        _presence(NamedColumns(prefixes=(prefix,)), merged) for prefix in named.prefixes
    )
    whole = _combined(each) if each else "unmet"
    if count > 1 and len(named.columns) == count and not named.prefixes:
        return each, whole
    return [whole] * count, whole


def _conditional_outcomes(
    conditional: Conditional, count: int, values: ParamValues, merged: ColumnSet
) -> tuple[list[_Outcome], _Outcome]:
    """Whether a conditional entry's columns appear, per column and as a whole."""
    if not conditional.when_param and conditional.when_present is None:
        return ["undecided"] * count, "undecided"
    param = _param_outcome(conditional.when_param, values)
    each, whole = _presence_outcomes(conditional.when_present, count, values, merged)
    return [_combined((param, one)) for one in each], _combined((param, whole))


def output_columns(
    provides: Provides | None, merged: ColumnSet, values: ParamValues
) -> ColumnSet:
    """The columns of a step's output, from its declaration, params and input.

    A family named after an input payload column is a prefix of the output:
    guaranteed when the input guarantees the column, possible when it may
    contain it. A family meets a payload requirement and makes a named column
    possible. It does not guarantee a named column.

    A passthrough output keeps the input's symmetric columns, less those it
    writes again, and a family named after a symmetric input column is a
    symmetric family: ``temporal-stack`` groups pair rows by pair and
    perspective, and ``global-scaler``, ``scale-to-cm`` and ``scale-to-bl``
    scale a column by one value per run or entry. ``scale-to-cm`` and
    ``scale-to-bl`` in derive mode, the default, write ``<column><suffix>``,
    which is no member of the column's family, so a sum over it reads every row.
    ``provides.symmetric`` adds its own. The rows are the input's, mirrored or
    not, for a passthrough or a family writer, which keep every row, and
    otherwise mirrored as ``provides`` declares.

    Args:
        provides: The step's declared output, ``None`` when undeclared.
        merged: The columns of the step's merged input.
        values: The step's params, every field the declaration reads included.
    """
    if provides is None:
        return UNKNOWN_COLUMNS
    start = merged if provides.passthrough else ColumnSet()
    guaranteed = set(start.guaranteed)
    possible = set(start.possible)
    sure_prefixes = set(start.guaranteed_prefixes)
    maybe_prefixes = set(start.possible_prefixes)
    unbounded = start.unbounded
    # The input that the output's columns are named after, by passthrough or by
    # family. Its unlisted columns name output columns outside the vocabulary.
    named_after = merged if provides.payload_families else start
    if provides.payload_families:
        sure_prefixes.update(_payload_names(merged.guaranteed))
        sure_prefixes.update(merged.guaranteed_prefixes)
        maybe_prefixes.update(_payload_names(merged.possible))
        maybe_prefixes.update(merged.possible_prefixes)
        unbounded = unbounded or merged.unbounded
    written: list[NamedColumns] = []
    for ref in provides.adds:
        named = resolve_names(ref, values)
        written.append(named)
        unbounded = unbounded or named.unknown
        guaranteed.update(named.columns)
        sure_prefixes.update(named.prefixes)
    for conditional in provides.conditional:
        named = resolve_names(conditional.column, values)
        written.append(named)
        unbounded = unbounded or named.unknown
        each, whole = _conditional_outcomes(
            conditional, len(named.columns), values, merged
        )
        for column, outcome in zip(named.columns, each, strict=True):
            if outcome == "met":
                guaranteed.add(column)
            elif outcome == "undecided":
                possible.add(column)
        if whole == "met":
            sure_prefixes.update(named.prefixes)
        elif whole == "undecided":
            maybe_prefixes.update(named.prefixes)
    return _column_set(
        guaranteed,
        possible,
        sure_prefixes,
        maybe_prefixes,
        unbounded=unbounded,
        unlisted=named_after.unlisted,
        vocabulary=named_after.vocabulary,
        symmetry=_output_symmetry(provides, merged, values, start, written),
    )


def _output_symmetry(
    provides: Provides,
    merged: ColumnSet,
    values: ParamValues,
    start: ColumnSet,
    written: Sequence[NamedColumns],
) -> _Symmetry:
    """The symmetric columns of a step's output, and whether its rows are mirrored.

    Args:
        provides: The step's declared output.
        merged: The columns of the step's merged input.
        values: The step's params.
        start: What the output copies of *merged*: all of it for a passthrough,
            nothing otherwise.
        written: The columns that ``adds`` and ``conditional`` name.
    """
    rewritten = {column for named in written for column in named.columns}
    rewritten_prefixes = {prefix for named in written for prefix in named.prefixes}
    columns = {
        column
        for column in start.symmetric
        if column not in rewritten
        and not any(column.startswith(prefix) for prefix in rewritten_prefixes)
    }
    prefixes = {
        prefix
        for prefix in start.symmetric_prefixes
        if not any(
            prefix.startswith(other) or other.startswith(prefix)
            for other in rewritten_prefixes
        )
    }
    if provides.payload_families:
        families = _symmetric_families(merged)
        columns.update(families.columns)
        prefixes.update(families.prefixes)
    for ref in provides.symmetric:
        named = resolve_names(ref, values)
        columns.update(named.columns)
        prefixes.update(named.prefixes)
    keeps_rows = provides.passthrough or provides.payload_families
    return _Symmetry(
        columns=frozenset(columns),
        prefixes=frozenset(prefixes),
        mirrored=provides.is_mirrored or (keeps_rows and merged.mirrored),
    )


def _symmetric_families(merged: ColumnSet) -> _Symmetry:
    """The families named after a symmetric payload column of *merged*.

    A family is its column, the exact name, and the names under
    ``<column>__`` (:data:`_FAMILY_SEPARATOR`), as ``temporal-stack`` writes
    ``speed__pool_mean`` from ``speed``. A longer name that only begins with
    the column's, such as ``distance_from_centroid`` beside ``distance``, is
    no member. A family is left out when a column or family that *merged* may
    contain falls under its prefix and is not symmetric, since its members
    would share the prefix.
    """
    columns = set(merged.symmetric)
    prefixes = set(merged.symmetric_prefixes)
    for column in _payload_names(merged.possible):
        if merged.is_symmetric(column):
            columns.add(column)
            prefixes.add(f"{column}{_FAMILY_SEPARATOR}")
    return _Symmetry(
        columns=frozenset(columns),
        prefixes=_unshadowed(
            prefixes, merged.possible, merged.possible_prefixes, columns
        ),
    )


def _unshadowed(
    prefixes: Iterable[str],
    possible: Iterable[str],
    possible_prefixes: Iterable[str],
    symmetric: Collection[str],
) -> frozenset[str]:
    """The symmetric *prefixes* that no column or family outside them falls under.

    A column under a prefix that is not itself symmetric, such as a widened
    individual measurement, would read as symmetric by the prefix, and the
    prefix is dropped. Dropping one leaves its members reading every row.
    """
    kept = frozenset(prefixes)
    others = [
        name
        for name in (*possible, *possible_prefixes)
        if name not in symmetric and name not in kept
    ]
    return frozenset(
        prefix
        for prefix in kept
        if not any(other.startswith(prefix) for other in others)
    )


def _payload_names(columns: Iterable[str]) -> list[str]:
    """The payload columns among *columns*: those outside ``NON_PAYLOAD_COLS``."""
    return [column for column in columns if column not in NON_PAYLOAD_COLS]


def resolve_pair_rule(
    merged: ColumnSet,
    column: str,
    agg: str,
    pairs: PairRule,
    filter_expr: str | None,
) -> Literal["once"] | None:
    """What an aggregate's ``auto`` pair rule reads: ``once``, or ``None`` for every row.

    ``once`` when the rule is ``auto``, *agg* is ``sum`` or ``count``, no
    *filter_expr* is set, *merged* may hold ``perspective``, its rows are
    mirrored, and *column* is symmetric. A filter can keep one row of a pair
    (on ``id1`` or ``perspective``, say), and ``once`` would then miss pairs.
    An undeclared or unbounded input has no symmetric column, and its ``auto``
    reads every row.

    Args:
        merged: The columns of the aggregate's merged input.
        column: The column it aggregates.
        agg: Its aggregation.
        pairs: Its pair rule as written.
        filter_expr: Its row filter, ``None`` or empty when unset.
    """
    if (
        pairs == "auto"
        and agg in PAIR_COUNTING_AGGS
        and not filter_expr
        and merged.admits(_PERSPECTIVE)
        and merged.mirrored
        and merged.is_symmetric(column)
    ):
        return "once"
    return None


def pair_rule_read(
    merged: ColumnSet,
    column: str,
    agg: str,
    pairs: PairRule,
    filter_expr: str | None,
) -> PairRead | None:
    """The rows that an aggregate reads of pair rows, ``None`` unless a sum or count.

    A rule written as ``once`` or ``each_perspective`` stands, and ``auto``
    reads what :func:`resolve_pair_rule` resolves, else every row. The other
    aggregations read every row whatever the rule, and the rule says nothing.
    Arguments as :func:`resolve_pair_rule` takes them.
    """
    if agg not in PAIR_COUNTING_AGGS:
        return None
    if pairs != "auto":
        return pairs
    return resolve_pair_rule(merged, column, agg, pairs, filter_expr) or (
        "each_perspective"
    )


def tracks_provides() -> Provides:
    """The dataset's tracks, as the registered track schemas describe them.

    The standard columns are guaranteed. Every schema that a built-in converter
    or tracker declares requires them. The legacy ``trex_v1`` requires neither
    ``X`` nor ``Y``, and a converter answers to it by default. Before ``apply``,
    a feature requiring one refuses an entry whose table lacks it
    (:func:`~mosaic.core.pipeline.column_contract.check_input_columns`). Each
    other column that a registered schema requires, recommends or allows as a
    measured quantity is possible. So is each column family that a schema
    requires a member of. The registry is read at each call, and a schema
    registered later is included.
    """
    known: set[str] = set()
    families: set[str] = set()
    for schema in TRACK_SCHEMAS.values():
        known.update(schema.required, schema.recommended, schema.allows)
        families.update(schema.required_prefixes)
    possible = sorted(known - STANDARD_COLUMNS)
    return Provides(
        adds=tuple(Column(name) for name in sorted(STANDARD_COLUMNS)),
        conditional=(
            *(Conditional(Column(name)) for name in possible),
            *(Conditional(Prefix(family)) for family in sorted(families)),
        ),
    )


def declared_vocabulary(catalog: DeclarationCatalog) -> frozenset[str]:
    """Every column that a declaration in *catalog* names literally."""
    return frozenset(
        name
        for declared in catalog.entries.values()
        for name in declared_names(declared.requires, declared.provides)
    )


def measured_columns(tables: Iterable[Collection[str]]) -> ColumnSet | None:
    """The columns of a step's tracks, from the column names of each table it reads.

    Guaranteed: each column in every table. Possible: each column in some table.
    The set admits only those, because the tables are read. A column that some
    tables contain and others lack stays possible, and the entries whose tables
    lack it fail at run time.

    ``None`` for an empty *tables*, which says nothing about the columns.
    """
    read = [frozenset(table) for table in tables]
    if not read:
        return None
    first, *rest = read
    return _column_set(
        first.intersection(*rest),
        first.union(*rest),
        (),
        (),
        unbounded=False,
        unlisted=False,
        vocabulary=(),
    )


def tracks_columns(catalog: DeclarationCatalog) -> ColumnSet:
    """The columns of the dataset's tracks, as an input of a step.

    The columns of :func:`tracks_provides`, and any column outside the
    vocabulary: the columns that a registered schema or a declaration in
    *catalog* names literally. A column in the vocabulary has its sources
    declared, and the tracks contain one only where a schema names it.
    """
    listed = output_columns(tracks_provides(), ColumnSet(), {})
    return replace(
        listed,
        unlisted=True,
        vocabulary=listed.possible | declared_vocabulary(catalog),
    )


# --- who provides a column ------------------------------------------------------------


def _supplies(named: NamedColumns, need: NamedColumns) -> bool:
    """Whether columns named by an output entry could meet *need*."""
    return need.met_by(named.columns) or _meets_a_family(need, named.prefixes)


def _provider(
    feature: str, provides: Provides, values: ParamValues, need: NamedColumns
) -> ColumnProvider | None:
    """How *feature* supplies *need* under its default params, if it does."""
    for ref in provides.adds:
        if _supplies(resolve_names(ref, values), need):
            return ColumnProvider(feature=feature, when_param="", when_present=())
    for conditional in provides.conditional:
        if not conditional.when_param and conditional.when_present is None:
            continue
        if conditional.when_present == conditional.column:
            continue
        if not _supplies(resolve_names(conditional.column, values), need):
            continue
        present = (
            ()
            if conditional.when_present is None
            else _labels(resolve_names(conditional.when_present, values))
        )
        return ColumnProvider(
            feature=feature, when_param=conditional.when_param, when_present=present
        )
    return None


def _labels(named: NamedColumns) -> tuple[str, ...]:
    """The names and families of *named*, as plain strings."""
    return (*named.columns, *(f"{prefix}*" for prefix in named.prefixes))


def providers_of(
    need: NamedColumns,
    catalog: DeclarationCatalog,
    tracks: ColumnSet,
    besides: str = "",
    consumer: ConsumerDecl | None = None,
) -> tuple[ColumnProvider, ...]:
    """The inputs whose output would contain *need*: the tracks, then features.

    The tracks come first when they guarantee *need*, and the features follow,
    sorted by slug and read under their default params. A feature passing its
    input through, or keeping a column of its input (``kept``), counts here only
    for the columns it adds. It copies a column of its input only when that input
    has it, and the tracks or another provider decide that.

    Args:
        need: The requirement.
        catalog: The declarations to search.
        tracks: The columns of the dataset's tracks.
        besides: A slug to leave out: the step's feature.
        consumer: The step's consumer side. An input that it cannot read
            (:func:`~mosaic.core.pipeline.graph.compatibility.can_connect`) is
            left out.
    """

    def readable(declaration: Declaration) -> bool:
        return consumer is None or bool(can_connect(declaration.produces, consumer))

    found: list[ColumnProvider] = []
    if need_status(need, tracks) == "guaranteed" and readable(TRACKS_DECLARATION):
        found.append(
            ColumnProvider(feature=TRACKS_INPUT, when_param="", when_present=())
        )
    for name in catalog.names():
        declared = catalog.get(name)
        if (
            declared is None
            or declared.produces.kind != "feature"
            or declared.provides is None
            or name == besides
            or not readable(declared)
        ):
            continue
        provider = _provider(name, declared.provides, declared.param_defaults, need)
        if provider is not None:
            found.append(provider)
    return tuple(found)


def provider_phrase(provider: ColumnProvider) -> str:
    """One provider as a message names it, its condition included."""
    if provider.feature == TRACKS_INPUT:
        return "the tracks input"
    conditions: list[str] = []
    if provider.when_param:
        conditions.append(f"{provider.when_param} is set")
    if provider.when_present:
        conditions.append(f"its input has {' and '.join(provider.when_present)}")
    if not conditions:
        return provider.feature
    return f"{provider.feature} (when {' and '.join(conditions)})"


# --- the wire -------------------------------------------------------------------------


def names_wire(named: NamedColumns) -> ColumnNames:
    """*named* as the JSON an editor reads."""
    return ColumnNames(
        label=named.label,
        columns=named.columns,
        prefixes=named.prefixes,
        params=named.params,
        unknown=named.unknown,
        payload=named.payload,
    )


def requires_wire(
    requires: Sequence[Requirement], values: ParamValues
) -> tuple[ColumnNames, ...]:
    """A declared ``requires``, read under *values*, as the JSON an editor reads."""
    return tuple(names_wire(need) for need in needs_of(requires, values))


def provided_wire(provides: Provides, values: ParamValues) -> ProvidedColumns:
    """A declared ``provides``, read under *values*, as the JSON an editor reads."""
    return ProvidedColumns(
        passthrough=provides.passthrough,
        adds=tuple(names_wire(resolve_names(ref, values)) for ref in provides.adds),
        conditional=tuple(
            ConditionalColumns(
                names=names_wire(resolve_names(conditional.column, values)),
                when_param=conditional.when_param,
                when_present=None
                if conditional.when_present is None
                else names_wire(resolve_names(conditional.when_present, values)),
            )
            for conditional in provides.conditional
        ),
        payload_families=provides.payload_families,
    )
