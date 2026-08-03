"""The typed index of the standardized tracks under ``tracks/``.

Not to be confused with :mod:`mosaic.core.pipeline.tracks_raw_index`, which
indexes the *raw* uploads under ``tracks_raw/``. Every public name there carries
``tracks_raw`` so the two never read alike.

``tracks/index.csv`` was the last index in the toolkit written by hand: an
eleven-name column list projected with ``row.get(k, "")``, concatenated onto
whatever was already on disk, and written with a bare ``to_csv`` -- no lock, no
atomic rename, no run identity, and read with four different NA policies by six
different callers. It is also the only index with three producers in two
packages (a registered converter, the TREx tracker, an inference op), which is
why the row lives here rather than beside any one of them.

**One row per ``(run_id, group, sequence)``.** The triple its six sibling indexes
use. It was the ``(group, sequence)`` pair while every writer targeted one flat
``tracks/<group>__<seq>.parquet``, where a second row for an entry would have
named a file the first had already overwritten; now each variant has its own
directory and two rows describe two real tables.

Writing a second row and *resolving* one are different questions, and only the
first belongs to the writer. :func:`select_variant_rows` answers the second: an
unlabelled row loses to a labelled one, and two genuinely different recipes for
one entry refuse to be guessed between.

**Adoption is on write, tolerance is on read.** An older index is brought up to
this schema by :func:`adopt_legacy_columns`, wired in as ``IndexCSV``'s ``adopt``
hook so it runs in memory inside the write lock (see ``index_lock`` for why a
second write there would be unsafe). Readers go through
:func:`read_tracks_index`, which projects in memory and never touches disk -- so
listing sequences on a read-only mount works, and a legacy dataset is not
silently rewritten by someone merely looking at it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Final, get_args

import pandas as pd

from mosaic.core.helpers import text_cell, to_safe_name, validate_entry_name
from mosaic.core.pipeline.dataset_indexes import register_reconcilable_index
from mosaic.core.pipeline.index_csv import (
    IndexCSV,
    RunIndexRowBase,
    project_to_schema,
)
from mosaic.core.pipeline.sequence_index import (
    SourceRoot,
    encode_entry_composition,
    read_entry_compositions,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "DROPPED_LEGACY_COLUMNS",
    "TRACKS_INDEX_COLUMNS",
    "TRACKS_INDEX_PATH_COLUMNS",
    "TracksIndexRow",
    "adopt_legacy_columns",
    "consumed_composition_for",
    "consumed_roots_for",
    "encode_source_roots",
    "legacy_view",
    "read_tracks_index",
    "select_variant_rows",
    "tracks_index",
    "tracks_index_path",
    "write_tracks_row",
]


@dataclass(frozen=True, slots=True)
class TracksIndexRow(RunIndexRowBase):
    """One row of ``tracks/index.csv``.

    Field order is the CSV column order, after ``abs_path``/``run_id``/
    ``started_at``/``finished_at`` from the bases.

    ``run_id`` is the *tracks variant* identity from
    :mod:`mosaic.core.pipeline.tracks_identity` -- what recipe produced this
    table -- and is deliberately not the producing op's run. ``producer_run_id``
    is that: the TREx or inference run whose output this table was bridged from,
    empty for a conversion, which has no op run at all. Keeping them apart is
    what stops an op-version bump relocating every tracks table, and it is the
    column that lets the ``predictions`` root be retired (item 8.7).

    ``producer`` is ``convert-<src_format>``, ``trex`` or ``infer-<kind>`` --
    exactly ``parse_op_run_id(run_id).kind``, and exactly the segment Stage 3.2
    turns into a directory name. Typed a bare ``str`` rather than a ``Literal``
    because the set is open: converters register through
    ``@register_track_converter`` and inference kinds through ``@register_op``,
    so a closed alias would be a claim this module cannot keep.

    ``source_abs_path`` is a ``str``, not a ``Path``: "this table has no source
    file" is a real state (the inference bridge builds from an in-memory frame),
    and ``Path("")`` renders as ``"."``. It holds a path, so it is registered in
    :data:`TRACKS_INDEX_PATH_COLUMNS` for the portability rewrites.

    ``consumed_source_roots`` names the dataset *roots* this table was derived
    from, not paths -- comma-joined and sorted by :func:`encode_source_roots`.
    Roots rather than files because the question it answers is "did a change
    under ``tracks_raw`` invalidate this?", and because a multi-valued *path*
    cell would need split/rewrite/rejoin support the portability passes do not
    have. Empty means "not establishable", never "none".

    ``consumed_composition`` is what those roots *held* for this entry when the
    table was written -- item 5.1's tracks half, and the value that closes the one
    dated gap in the Stage 4 design. ``consumed_source_roots`` answers "which root
    would a change have to be under?"; only this answers "has it changed?", which
    is the question item 6.2's walk actually asks. Encoded by
    :func:`encode_entry_composition`, the same minter the feature row uses.

    **It sits on the row and never in ``run_id``**, and the distinction is forced
    rather than stylistic. A tracks variant is params-only and scope-free (item
    3.1): one identity names one recipe covering many sequences, which under one
    variant legitimately hold *different* compositions. Folding a per-sequence
    value into the name would rename one recipe's whole directory because one
    other sequence's source moved -- the same reason a per-frame feature records
    its composition here rather than in its identifier (rule P2d).
    """

    group: str
    sequence: str
    producer: str
    std_format: str
    producer_run_id: str = ""
    source_abs_path: str = ""
    source_md5: str = ""
    consumed_source_roots: str = ""
    n_rows: int = 0
    consumed_composition: str = ""


TRACKS_INDEX_COLUMNS: Final[list[str]] = [
    field.name for field in fields(TracksIndexRow)
]
"""The schema, in CSV order. Derived from the row so the two cannot drift."""

TRACKS_INDEX_PATH_COLUMNS: Final[tuple[str, ...]] = ("source_abs_path",)
"""Path-bearing columns beyond ``abs_path``.

Named here because both of ``Dataset``'s path-rewriting passes read raw CSVs and
have no row class to ask. A new path column on :class:`TracksIndexRow` belongs in
this tuple, or it silently stops being portable -- which is what happened to
``source_abs_path`` itself until now. ``consumed_source_roots`` is deliberately
absent: it holds root *keys*, which are machine-independent already.
"""

_ROOT_SEPARATOR: Final = ","
"""Comma, matching every other multi-valued index cell in the toolkit.

pandas quotes a comma-bearing cell and both readers restore it intact, and a
dataset root key can never contain one.
"""

DROPPED_LEGACY_COLUMNS: Final[tuple[str, ...]] = (
    "group_safe",
    "sequence_safe",
    "collection",
    "collection_safe",
)
"""What the hand-written writer emitted that this schema deliberately does not.

Recorded rather than merely omitted, so a reader of an old index knows these were
dropped on purpose and where the two that mattered went. ``group_safe`` and
``sequence_safe`` are pure functions of ``group``/``sequence`` -- the old writer
recomputed them on every write rather than trusting its caller, so they were a
cache, and :func:`legacy_view` re-derives them. ``collection`` and
``collection_safe`` duplicated either ``group`` or the raw file's grouping hint
and had no reader anywhere: not in the toolkit, its tests, its notebooks, or the
sibling repos.
"""


def tracks_index_path(ds: Dataset) -> Path:
    """Where the standardized-tracks index lives."""
    return ds.get_root("tracks") / "index.csv"


def tracks_index(path: Path) -> IndexCSV[TracksIndexRow]:
    """Factory: an ``IndexCSV`` configured for the tracks schema.

    ``dedup_keys`` is the ``run_id``-led triple its six sibling indexes use. Until
    Stage 3.2 it was the ``(group, sequence)`` pair, because that was the
    invariant the flat layout actually had: every writer targeted one
    ``tracks/<group>__<seq>.parquet``, so a second row for an entry would have
    named a file the first had already overwritten. Now each variant has its own
    directory, so two rows describe two real tables.

    Which row an entry *resolves* to is a separate question, and not one the
    writer can answer -- see :func:`select_variant_rows`.
    """
    return IndexCSV(
        path,
        TracksIndexRow,
        dedup_keys=["run_id", "group", "sequence"],
        adopt=adopt_legacy_columns,
    )


def encode_source_roots(roots: Iterable[str]) -> str:
    """Encode dataset root keys into one ``consumed_source_roots`` cell.

    Sorted and deduplicated, so one set of roots has one spelling however the
    caller ordered them -- the same reason identity payloads sort their
    collections.
    """
    return _ROOT_SEPARATOR.join(sorted({root for root in roots if root}))


def consumed_roots_for(ds: Dataset, paths: Iterable[Path | str]) -> tuple[str, ...]:
    """Which declared roots contain *paths*.

    Longest-prefix wins, because roots nest: ``trex`` defaults under
    ``tracks_raw/`` and ``frames`` under ``media/``, so a shortest-match answer
    would name the parent and lose what actually produced the file. A path under
    no declared root contributes nothing -- an honest omission rather than a
    guess.
    """
    roots: list[tuple[Path, str]] = []
    for key in ds.roots:
        try:
            roots.append((ds.get_root(key).resolve(), key))
        except (KeyError, OSError):
            continue
    found: set[str] = set()
    for raw in paths:
        if not raw:
            continue
        candidate = ds.resolve_path(raw).resolve()
        best: tuple[int, str] | None = None
        for root_path, key in roots:
            if candidate == root_path or root_path in candidate.parents:
                depth = len(root_path.parts)
                if best is None or depth > best[0]:
                    best = (depth, key)
        if best is not None:
            found.add(best[1])
    return tuple(sorted(found))


def adopt_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bring a frame read off disk up to the current schema, in memory.

    Wired in as ``IndexCSV``'s ``adopt`` hook, so it runs inside the write lock
    and its result is written by the single ``atomic_write`` that already ends
    that block. Idempotent: a frame already in schema is projected onto itself.

    Four things happen, and each is load-bearing:

    1. **Missing columns are added empty.** A row that predates a column has no
       honest value for it, and ``""`` is what the rest of the toolkit already
       uses to mean "predates the scheme".
    2. **NaN is coerced to ``""``.** Not merely defensive: the previous writer
       concatenated an eleven-key row onto a frame it had read with default NA
       handling, so real NaN is *already on disk* in the widened columns.
    3. **Off-schema columns are dropped**, including the four the old writer
       emitted (see ``_DROPPED_LEGACY_COLUMNS``). Destructive, deliberately, and
       recoverable: :func:`legacy_view` re-derives the two that had a reader.

    It deliberately does **not** collapse duplicates. It did while an entry could
    only have one row -- keep-last, which is what every reader effectively did
    with a duplicate written before string columns were read as strings. Now that
    two variants of one sequence are two real tables in two directories, dropping
    one here would discard a row silently on the way *in*, since this is
    ``IndexCSV``'s ``adopt`` hook and runs on write as well as on read. Choosing
    between rows moved to :func:`select_variant_rows`, which is a read concern and
    can refuse.

    Returns a frame carrying every schema column -- the contract ``IndexCSV``
    requires, because a partial adoption still leaves ``list_runs`` and
    ``latest_run_id`` raising on ``finished_at``. The projection itself is
    :func:`project_to_schema`, shared with the other typed indexes.
    """
    return project_to_schema(df, TRACKS_INDEX_COLUMNS)


def select_variant_rows(df: pd.DataFrame, run_id: str | None = None) -> pd.DataFrame:
    """Reduce a tracks frame to one row per ``(group, sequence)``.

    The single place that decides *which variant an entry resolves to*, so the
    resolver, ``Dataset.load_tracks`` and the chain runner's target universe
    cannot answer it three different ways. Order-preserving.

    ``run_id`` given selects that variant exactly. ``""`` is a legal argument and
    names the *unlabelled* tables -- rows written before variants existed -- which
    is how a pre-Stage-3 flat layout stays addressable by name.

    ``run_id`` of ``None`` means "whichever variant this entry has", and the rule
    it applies is that **an empty ``run_id`` is unknown, never a peer variant**.
    Every dataset converted before Stage 3 has a full index of empty ones, and
    the first ordinary re-conversion writes a labelled row beside each. Treating
    those as two competing recipes would make the ambiguity below fire on every
    existing dataset, with ``''`` as one of the candidates the user is asked to
    choose between -- so a labelled row supersedes an unlabelled one for the same
    entry. Nothing is deleted: the row stays in the index as a record, and
    reverting Stage 3 finds it again.

    Two genuinely different recipes for one entry is the case with no defensible
    default, and it raises. Different entries carrying different variants -- some
    converted, some tracked -- stays legal and is the mixed dataset the resolver
    has always been expected to handle.
    """
    if df.empty:
        return df.reset_index(drop=True)

    # The three identity columns as plain lists, once. Row-wise ``df.iloc[i]``
    # inside the loops below would be quadratic on an index with a row per
    # sequence, and this is on the path of every feature run.
    run_ids = [str(value) for value in df["run_id"]]
    groups = [str(value) for value in df["group"]]
    sequences = [str(value) for value in df["sequence"]]

    if run_id is not None:
        named = [i for i, value in enumerate(run_ids) if value == run_id]
        return df.iloc[named].reset_index(drop=True)

    positions_by_entry: dict[tuple[str, str], list[int]] = {}
    for position, entry in enumerate(zip(groups, sequences, strict=True)):
        positions_by_entry.setdefault(entry, []).append(position)

    keep: list[int] = []
    for entry, positions in positions_by_entry.items():
        labelled = [p for p in positions if run_ids[p]]
        variants = sorted({run_ids[p] for p in labelled})
        if len(variants) > 1:
            raise ValueError(_ambiguous_variant_message(entry, variants))
        # Last wins within one variant, matching what every reader effectively
        # did with a duplicate before the index enforced uniqueness.
        keep.append((labelled or positions)[-1])
    return df.iloc[sorted(keep)].reset_index(drop=True)


def _ambiguous_variant_message(entry: tuple[str, str], variants: list[str]) -> str:
    """Name the entry, both candidates, and the keyword that resolves it."""
    group, sequence = entry
    listing = ", ".join(repr(variant) for variant in variants)
    return (
        f"tracks/index.csv holds {len(variants)} variants of "
        f"(group={group!r}, sequence={sequence!r}): {listing}. "
        f"There is no defensible default between two recipes, so say which one "
        f"to read: pass tracks_run_id=<variant> to run_feature, build_manifest, "
        f"load_tracks or drop_entries, or --tracks-run-id on the command line."
    )


def legacy_view(df: pd.DataFrame) -> pd.DataFrame:
    """Add back the derived safe-name columns some callers still name.

    ``group_safe``/``sequence_safe`` are pure functions of ``group``/``sequence``
    -- the old writer recomputed them on every write rather than trusting what it
    was handed -- so storing them was caching, not recording. They are re-derived
    here for the two readers that ask for them by name, and never written back.

    Derived unconditionally rather than only when absent. A present-but-empty
    cell is the state a migrated index leaves behind, and the old
    ``row.get("group_safe") or ...`` fallback does not fire on one: a NaN is
    truthy, so it returned the NaN and the next ``.lower()`` raised.
    """
    out = df.copy()
    out["group_safe"] = [
        to_safe_name(str(value)) if str(value) else "" for value in out["group"]
    ]
    out["sequence_safe"] = [
        to_safe_name(str(value)) if str(value) else "" for value in out["sequence"]
    ]
    return out


def empty_tracks_frame() -> pd.DataFrame:
    """The full-schema, zero-row frame an absent index reads as.

    The column set is load-bearing, not cosmetic: callers filter on ``group`` and
    ``sequence`` straight away, and a column-less empty frame turns "there are no
    tracks yet" into ``KeyError: 'group'``.
    """
    return pd.DataFrame(
        {column: pd.Series(dtype="object") for column in TRACKS_INDEX_COLUMNS}
    )


def read_tracks_index(ds: Dataset) -> pd.DataFrame:
    """Read ``tracks/index.csv``, projected onto the current schema.

    The single reader. An absent index reads as an *empty* one -- absence and
    emptiness are two spellings of "this dataset has no standardized tracks", and
    answering them differently is what left six callers with four different
    policies. Callers that want to tell a human to run a conversion check for
    zero rows; that check is the same for both spellings.

    Never writes. Adoption on disk happens only when something appends, so
    reading a legacy dataset from a read-only mount works and merely listing its
    sequences does not rewrite it.

    Uses only ``ds.get_root``, so the duck-typed dataset stand-ins in the test
    suite keep working. An unset ``tracks`` root still raises ``KeyError`` from
    ``get_root``: that is a misconfigured dataset, not an absent file.
    """
    path = tracks_index_path(ds)
    if not path.exists():
        return empty_tracks_frame()
    raw = pd.read_csv(path, keep_default_na=False, dtype=str)
    return adopt_legacy_columns(raw)


def consumed_composition_for(
    ds: Dataset, group: str, sequence: str, roots: Iterable[str]
) -> str:
    """What *roots* recorded for this entry, as one cell -- item 5.1's tracks half.

    Only the **source** roots among *roots* can answer: a composition exists for
    a root that holds what cannot be recomputed, and ``consumed_source_roots``
    legitimately names derived ones too (the TREx bridge records ``trex``, the
    inference bridge ``models``). Filtering against :data:`SourceRoot` rather than
    against "has a ``sequences.csv``" keeps the answer the same on a dataset whose
    projection has not been written yet -- absent, not wrong.

    Read per entry rather than per variant, deliberately. One variant covers many
    sequences and their compositions differ, so there is no per-variant answer to
    cache. The two small reads this costs sit beside a parquet write and a locked
    whole-file index append, which dominate them.
    """
    source_roots = [root for root in roots if root in get_args(SourceRoot)]
    if not source_roots:
        return ""
    entry = (group, sequence)
    recorded = read_entry_compositions(ds, [entry]).get(entry, {})
    return encode_entry_composition(recorded, source_roots)


def write_tracks_row(
    ds: Dataset,
    *,
    run_id: str,
    group: object,
    sequence: object,
    out_path: Path,
    producer: str,
    std_format: str,
    n_rows: int,
    producer_run_id: str = "",
    source: Path | str = "",
    source_md5: str = "",
    consumed_source_roots: Sequence[str] = (),
) -> None:
    """Record one standardized-tracks table. The only way to write this index.

    Keyword-only throughout: five call sites in three files pass ``group`` and
    ``sequence`` adjacently, and transposing them would be silent.

    ``group``/``sequence`` are typed ``object`` and read through
    :func:`~mosaic.core.helpers.text_cell` here because one caller reads them
    straight off a pandas ``Series``, where they arrive as ``numpy`` scalars; a
    frozen dataclass does no coercion, so an ``np.int64`` would land as an
    integer CSV column and defeat the dedup that holds this index to one row per
    entry. ``str()`` alone was not enough: it spells a blank group -- a float
    NaN off a CSV -- as the word "nan", which then names an entry that has no
    composition recorded under it.

    Paths are stored root-relative via ``Dataset.relative_to_root`` so the index
    survives a move or a sync between machines. Call this *after* writing the
    parquet: ``IndexRowBase`` existence-checks an absolute ``abs_path``, which is
    what a ``tracks`` root outside the dataset tree produces.
    """
    # Validated first, because the composition is looked up by the same key the
    # row is written under: a lookup on the raw value would miss for the caller
    # that reads its group off a pandas Series as a numpy scalar.
    entry_group = validate_entry_name(text_cell(group), "group")
    entry_sequence = validate_entry_name(text_cell(sequence), "sequence")
    row = TracksIndexRow(
        abs_path=Path(ds.relative_to_root(out_path)),
        run_id=run_id,
        group=entry_group,
        sequence=entry_sequence,
        producer=producer,
        std_format=std_format,
        producer_run_id=producer_run_id,
        source_abs_path=ds.relative_to_root(source) if source else "",
        source_md5=source_md5,
        consumed_source_roots=encode_source_roots(consumed_source_roots),
        n_rows=int(n_rows),
        consumed_composition=consumed_composition_for(
            ds, entry_group, entry_sequence, consumed_source_roots
        ),
    )
    tracks_index(tracks_index_path(ds)).append([row])


# Item 6.1: reconciled through the shared registry, beside the tracker
# indexes, so one pass covers every root that has an ``IndexCSV`` behind it.
register_reconcilable_index("tracks", tracks_index)
