"""The per-sequence index: ``<source root>/sequences.csv``.

The row that has been missing. Every index in a dataset is per *file*
(``media_raw``, ``tracks_raw``) or per *run* (``features``, ``models``,
``tracks``), and neither can hold a fact about a sequence. This one can, and it
sits beside its root's ``index.csv``, one per source root.

**Why a stored row rather than a computed answer.** A composition could be
recomputed from ``index.csv`` on demand, and for a single lookup that would be
cheaper. It cannot serve the two things a composition is for: item 5.2's drift
check needs a *baseline* to compare the present against, and a value recomputed
from the present agrees with itself by construction; and item 6.2's block check
needs one value per sequence without walking the whole root. Storage is what
makes both of those a read.

**And why not on the per-file rows.** A composition is a property of the
sequence, so putting it on every member row means rewriting every row to record
one change, and leaves no way to tell a stale copy from a current one when they
disagree.

**Lock order, and it must not be inverted.** A writer commits its ``index.csv``
first, releases, and only then writes ``sequences.csv``. Two files, two locks,
never nested: ``atomic_write`` renames a new inode over ``index.csv``, so a
locked block that went on to write a second file would already have lost its grip
on the first (see :mod:`mosaic.core.pipeline.index_lock`), and ``index_lock`` is
re-entrant per resolved path so it will not catch the mistake. Writing
``sequences.csv`` first would be worse than unsafe: it would record a composition
for an index state that never committed -- a confident value nothing on disk
supports. A crash between the two leaves the projection absent or stale, which
over-reports on the next comparison and heals on the next write.

**The display name is deliberately not here**, though item 4.4 says all three
per-sequence facts share one row. A composition is a property of
``(sequence, root)``; a label is a property of the sequence alone. On one row per
root, a sequence with both media and tracks would carry two display names with no
rule saying which wins, and ``Dataset.set_display_name`` would have to pick a
root it has no basis to pick. The label lives in a dataset-level
``sequences.csv`` instead (item 4.1), and this deviation from the plan's wording
is recorded rather than left to be rediscovered.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, get_args

import pandas as pd

from mosaic.core.pipeline._utils import now_iso
from mosaic.core.pipeline.composition import (
    LABELS_RAW_COMPOSITION_SCHEME,
    MEDIA_COMPOSITION_SCHEME,
    TRACKS_RAW_COMPOSITION_SCHEME,
    SequenceComposition,
)
from mosaic.core.pipeline.index_csv import IndexCSV, SchemaRowBase

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "SEQUENCE_INDEX_COLUMNS",
    "SEQUENCE_INDEX_PATH_COLUMNS",
    "SequenceIndexRow",
    "SourceRoot",
    "adopt_sequence_columns",
    "empty_sequence_frame",
    "read_sequence_index",
    "sequence_index",
    "sequence_index_path",
    "write_sequence_compositions",
]

SourceRoot = Literal["media_raw", "tracks_raw", "labels_raw"]
"""A root holding what cannot be recomputed, and therefore has a composition.

A closed alias, unlike ``TracksIndexRow.producer`` which is deliberately a bare
``str`` because converters register at runtime. This set is closed by rule P1 --
nothing else in the dataset is a *source* -- so the claim is one this module can
keep, and it makes "which composition function applies" exhaustively checkable
rather than a dict lookup that can miss.

``labels_raw`` holds the raw label files a kind is converted from -- uploaded, or
in a later milestone projected from the Dolt scoring store. It is not a partition
of ``tracks_raw``: a format like ``calms21_npy`` is registered as both a track
converter and a label converter, so one raw file is at once a track source and a
label source. Membership is therefore by index, not by directory, and the two
roots hold overlapping files without either owning the other.
"""

_SCHEME_BY_ROOT: Final[dict[str, str]] = {
    "media_raw": MEDIA_COMPOSITION_SCHEME,
    "tracks_raw": TRACKS_RAW_COMPOSITION_SCHEME,
    "labels_raw": LABELS_RAW_COMPOSITION_SCHEME,
}


@dataclass(frozen=True, slots=True)
class SequenceIndexRow(SchemaRowBase):
    """One row of ``<source root>/sequences.csv``: what a sequence was made of.

    No ``abs_path``, which is why :class:`SchemaRowBase` exists. A sequence is
    not a file, and borrowing its directory to satisfy a base class would let
    ``prune_missing`` delete the drift baseline whenever that directory moved.

    ``composition`` is the digest, or ``""`` for *not establishable* -- a member
    carrying no identity, which is the honest state for a ``tracks_raw`` sequence
    whose checksums are turned off, or a media sequence holding a store with no
    uuid. ``member_count`` is what distinguishes that from *nothing here*, and it
    is the only cell a human can read without recomputing.

    ``identity_scheme`` is the per-family constant at mint time, typed ``str``
    and never ``int`` for the reason ``identity_scheme.py`` spells out: an
    integer column round-trips through a ``pd.concat`` against an all-NaN column
    as ``1.0`` and reads back as the string ``"1.0"``, which is two on-disk
    spellings of one scheme and silently defeats the detector.

    ``computed_at`` is provenance and never hashed. It is what makes a crash
    between the index write and this one diagnosable: an index newer than its own
    projection is visible rather than inferred.
    """

    group: str = ""
    sequence: str = ""
    composition: str = ""
    member_count: int = 0
    identity_scheme: str = ""
    computed_at: str = ""


SEQUENCE_INDEX_COLUMNS: Final[list[str]] = [
    field.name for field in fields(SequenceIndexRow)
]
"""The schema, in CSV order. Derived from the row so the two cannot drift."""

SEQUENCE_INDEX_PATH_COLUMNS: Final[tuple[str, ...]] = ()
"""Path-bearing columns: none, deliberately.

Recorded as an empty tuple rather than omitted so rule P7's "any new
path-bearing column joins the portability rewrite lists in the same change" has
somewhere to be added. Both of ``Dataset``'s rewrite passes hard-code
``<root>/index.csv`` and would skip ``sequences.csv`` anyway; this says that is
correct rather than forgotten.
"""


def sequence_index_path(ds: Dataset, root: SourceRoot) -> Path:
    """Where *root*'s per-sequence index lives, beside its ``index.csv``."""
    return ds.get_root(root) / "sequences.csv"


def _root_or_none(ds: Dataset, root: SourceRoot) -> Path | None:
    """*root*'s per-sequence index path, or ``None`` when the root is unset.

    Asks ``get_root`` and catches, rather than testing ``has_root`` first, for
    the reason ``read_tracks_index`` gives: the duck-typed dataset stand-ins in
    the test suite implement ``get_root`` and nothing else, and a reader that
    needs more of the ``Dataset`` surface than it uses is a reader those
    stand-ins cannot serve.
    """
    try:
        return sequence_index_path(ds, root)
    except KeyError:
        return None


def sequence_index(path: Path) -> IndexCSV[SequenceIndexRow]:
    """Factory: an ``IndexCSV`` configured for the per-sequence schema.

    ``dedup_keys`` is ``(group, sequence)`` and stays that way: one row per
    sequence per root is the whole invariant, and unlike the tracks index there
    is no variant axis that could ever make a second row legal.
    """
    return IndexCSV(
        path,
        SequenceIndexRow,
        dedup_keys=["group", "sequence"],
        adopt=adopt_sequence_columns,
    )


def empty_sequence_frame() -> pd.DataFrame:
    """The full-schema, zero-row frame an absent index reads as.

    The column set is load-bearing, exactly as it is for ``empty_tracks_frame``:
    callers filter on ``group``/``sequence`` immediately, and a column-less empty
    frame turns "this root has never been indexed" into ``KeyError: 'group'``.
    """
    return pd.DataFrame(
        {column: pd.Series(dtype="object") for column in SEQUENCE_INDEX_COLUMNS}
    )


def adopt_sequence_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bring a frame read off disk up to the current schema, in memory.

    Wired in as ``IndexCSV``'s ``adopt`` hook so it runs inside the write lock
    and its result is written by the single ``atomic_write`` that already ends
    that block. Idempotent.

    Every column is built with an explicit ``object`` dtype. An empty list
    assigned to a column lands as ``float64``, and a later ``pd.concat`` against
    a real row then widens ``member_count`` so ``3`` reaches disk as ``3.0`` --
    the same trap that made ``identity_scheme`` a ``str`` rather than an ``int``.
    """
    out = pd.DataFrame(index=df.index)
    for column in SEQUENCE_INDEX_COLUMNS:
        if column in df.columns:
            cells = ["" if pd.isna(cell) else cell for cell in df[column]]
        else:
            cells = [""] * len(df)
        out[column] = pd.Series(cells, index=df.index, dtype="object")
    deduped = out.drop_duplicates(subset=["group", "sequence"], keep="last")
    return deduped.reset_index(drop=True)


def read_sequence_index(ds: Dataset, root: SourceRoot) -> pd.DataFrame:
    """Read *root*'s per-sequence index, projected onto the current schema.

    The single reader. An absent index -- or an unset root, which is the state
    every dataset predating this milestone is in -- reads as an *empty* one:
    absence and emptiness are two spellings of "this root has recorded no
    compositions", and answering them differently is what left the tracks index
    with four policies across six callers.

    Never writes, so reading a legacy dataset from a read-only mount works and
    merely looking at it does not rewrite it.
    """
    path = _root_or_none(ds, root)
    if path is None or not path.exists():
        return empty_sequence_frame()
    raw = pd.read_csv(path, keep_default_na=False, dtype=str)
    return adopt_sequence_columns(raw)


def write_sequence_compositions(
    ds: Dataset,
    root: SourceRoot,
    *,
    compositions: Mapping[tuple[str, str], SequenceComposition],
) -> Path | None:
    """Replace *root*'s per-sequence index with exactly *compositions*.

    A projection of what the root's ``index.csv`` now says, so it *replaces*
    rather than appends: a sequence that has gone away must leave, and an append
    with dedup keys can only ever add. Returns the written path, or ``None`` when
    the root is unset and there is nowhere to write.

    Call this **after** the index write and **outside** its lock -- see the
    module docstring for why that order is not interchangeable.
    """
    path = _root_or_none(ds, root)
    if path is None:
        return None
    scheme = _SCHEME_BY_ROOT[root]
    stamped = now_iso()
    rows = [
        SequenceIndexRow(
            group=group,
            sequence=sequence,
            composition=value.digest,
            member_count=value.member_count,
            # Only a real digest was minted under a scheme. Stamping one on an
            # unestablishable row would claim it was produced by a contract that
            # never ran.
            identity_scheme=scheme if value.digest else "",
            computed_at=stamped,
        )
        for (group, sequence), value in sorted(compositions.items())
    ]
    sequence_index(path).replace(rows)
    return path


@dataclass(frozen=True, slots=True)
class SequenceLabelRow(SchemaRowBase):
    """One row of the dataset-level ``sequences.csv``: what a sequence is called.

    Separate from :class:`SequenceIndexRow`, and separate for a reason worth
    stating because item 4.4's wording says otherwise. A composition is a
    property of ``(sequence, root)`` -- a sequence has one per root and they
    change independently. A label is a property of the sequence alone. Put it on
    the per-root row and a sequence with both media and tracks carries two
    display names that can disagree, with no rule saying which wins, and
    ``Dataset.set_display_name`` has to pick a root it has no basis to pick.

    ``group``/``sequence`` are the stable **token**: minted once, part of every
    filename and every index join key, and never rewritten. ``display_group``
    and ``display_name`` are the label: relabelled freely, read by humans, and
    touching nothing on disk. An empty label means "no label recorded", and every
    reader falls back to the token -- so a dataset that never names anything
    behaves exactly as it did before this file existed.

    ``derived_from`` is reserved for item 8.6, where a promoted manual correction
    records the tracker run it was corrected from. Declared now rather than added
    later so the schema does not move once datasets hold rows.
    """

    group: str = ""
    sequence: str = ""
    display_group: str = ""
    display_name: str = ""
    derived_from: str = ""


SEQUENCE_LABEL_COLUMNS: Final[list[str]] = [
    field.name for field in fields(SequenceLabelRow)
]
"""The label schema, in CSV order. Derived from the row so the two cannot drift."""


def sequence_label_path(ds: Dataset) -> Path:
    """Where a dataset's sequence labels live: one file, at the dataset root.

    Not under a source root, because a label is not a property of a root. Beside
    the manifest, because it is a property of the dataset the manifest describes.
    """
    return ds.base_dir / "sequences.csv"


def sequence_labels(path: Path) -> IndexCSV[SequenceLabelRow]:
    """Factory: an ``IndexCSV`` configured for the label schema."""
    return IndexCSV(
        path,
        SequenceLabelRow,
        dedup_keys=["group", "sequence"],
        adopt=adopt_label_columns,
    )


def empty_label_frame() -> pd.DataFrame:
    """The full-schema, zero-row frame an absent label file reads as."""
    return pd.DataFrame(
        {column: pd.Series(dtype="object") for column in SEQUENCE_LABEL_COLUMNS}
    )


def adopt_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bring a label frame read off disk up to the current schema, in memory."""
    out = pd.DataFrame(index=df.index)
    for column in SEQUENCE_LABEL_COLUMNS:
        if column in df.columns:
            cells = ["" if pd.isna(cell) else cell for cell in df[column]]
        else:
            cells = [""] * len(df)
        out[column] = pd.Series(cells, index=df.index, dtype="object")
    deduped = out.drop_duplicates(subset=["group", "sequence"], keep="last")
    return deduped.reset_index(drop=True)


def read_sequence_labels(ds: Dataset) -> pd.DataFrame:
    """Read a dataset's sequence labels, projected onto the current schema.

    The single reader. Absent reads as the full-schema empty frame -- which every
    dataset predating item 4.1 is, and which is why the fallback to the token has
    to be the ordinary path rather than an error case.
    """
    path = sequence_label_path(ds)
    if not path.exists():
        return empty_label_frame()
    raw = pd.read_csv(path, keep_default_na=False, dtype=str)
    return adopt_label_columns(raw)


def decode_consumed_roots(cell: str) -> tuple[str, ...]:
    """The roots a ``consumed_roots`` cell names, in the order it names them.

    The reader for what ``encode_consumed_roots`` wrote, kept beside the
    composition encoder for the same reason it exists: three callers now parse
    this cell -- the drift comparison, ``run_feature``'s pre-pass and the
    reverse-dependency walk -- and three spellings of "which roots" would be
    three answers to the same question.

    An empty cell yields an empty tuple: a consumer that declared nothing, which
    a caller must not confuse with one whose declared root recorded nothing.
    """
    return tuple(root for root in cell.split(",") if root)


def encode_entry_composition(recorded: Mapping[str, str], roots: Iterable[str]) -> str:
    """Encode one entry's recorded compositions into a single index cell.

    The one minter for that cell, shared by the feature row and the tracks row,
    because two spellings of one answer would be two answers to item 6.2's walk.

    One root is the whole of today's reality -- two features declare
    ``media_raw``, a converted tracks table reads ``tracks_raw`` -- so the common
    forms are a bare digest and ``""``. Several roots join as ``root=digest``
    pairs, sorted, so the cell says which value came from where without a second
    column to keep in step.

    **The shape follows what was declared, not what was found.** A consumer
    declaring two roots writes a labelled pair for each even where one recorded
    nothing (``tracks_raw=abc,media_raw=``), because the alternative -- emitting
    only what was present -- collapses "two roots, one recorded" onto the same
    bare digest as "one root, recorded", and a reader comparing the two cannot
    tell which root the digest came from. That is the honest-empty rule one level
    up: an empty needs a companion saying which kind of empty it is, and here the
    declaration is the companion.

    Empty means **nothing recorded**, never "recorded as empty". It covers a
    consumer that declares no source root and one whose every declared root has
    recorded nothing, and a reader must draw no conclusion from it -- two entries
    whose compositions are both unrecorded are not known to be alike.

    Only the labelled branch changes shape, and no consumer takes it today: both
    tracks bridges filter to exactly one source root (a TREx run reads
    ``media_raw`` and ``trex``, an inference run ``media_raw`` and ``models``, and
    the derived halves drop out), and the two features that declare a root declare
    one. Item 8.6 is what takes it -- a promoted correction lands in ``tracks_raw``
    beside a video in ``media_raw`` -- so this is settled before the milestone that
    would otherwise settle it by accident.
    """
    declared = sorted({root for root in roots if root})
    found = [(root, recorded.get(root, "")) for root in declared]
    if not any(digest for _, digest in found):
        return ""
    if len(declared) == 1:
        return found[0][1]
    return ",".join(f"{root}={digest}" for root, digest in found)


def read_entry_compositions(
    ds: Dataset,
    entries: Iterable[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    """Every source root's recorded composition, for each of *entries*.

    Two levels -- entry, then root -- rather than one flattened value per entry.
    Flattening is what item 4.4's H3 case 2 exists to rule out: a track-only
    sequence gaining its first video would move a tracks-only identifier, and one
    combined hash cannot tell that apart from a change that mattered.

    Reads at most one ``sequences.csv`` per declared source root, once, rather
    than once per entry. A root with no index, an entry with no row in it, and an
    unestablishable composition all contribute **nothing** -- not an empty string.
    That distinction is the mechanism the omission rule rests on: an absent key
    digests differently from a key whose value is empty, which is what keeps an
    identifier still on a dataset that has recorded no compositions yet.
    """
    wanted = set(entries)
    if not wanted:
        return {}
    found: dict[tuple[str, str], dict[str, str]] = {}
    for root in get_args(SourceRoot):
        frame = read_sequence_index(ds, root)
        for _, row in frame.iterrows():
            key = (str(row["group"]), str(row["sequence"]))
            digest = str(row["composition"])
            if key not in wanted or not digest:
                continue
            found.setdefault(key, {})[root] = digest
    return found
