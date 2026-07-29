"""Which derived artifacts a source change reaches -- item 6.1's walk.

Every column this needs was recorded in Stage 5: a tracks table and a feature
entry each carry the composition they were built from, and ``<root>/sequences.csv``
carries what that sequence is made of now. What was missing is the join, and this
is it. ``drifted_entries`` answers the same question for one feature run; this
answers it across the dataset and follows the edge that run cannot see.

**A query, never a stored index.** Item 6.1 calls for a "derived index",
version-control-ignored and rebuildable. It is neither built nor stored, because
an edge list is a *join over the present* rather than a baseline: a value
recomputed from the present agrees with itself by construction, which is the
argument ``sequence_index`` makes for storing compositions and the same argument
against storing this. Freezing it would add a staleness class -- index newer than
projection -- with no reader able to tolerate it.

**Three arms, and the fourth is deliberately absent.** Tracks and features are
reached directly, by the composition each recorded. Feature runs are reached
*transitively*, through a tracks variant that moved under them -- which matters
because forty-two features declare no source root and reach media only through
tracks, so a walk without this arm stops at the tracks table and reports nothing
about everything built on it.

Extracted frames are **not** walked. They record a ``media_composition`` and a
reorder does move it, but a frame set is a P4 carve-out that is never destroyed,
so it cannot appear in a delete set -- and this walk exists to compute one.
Leaving frames out makes that guarantee structural rather than a filter someone
can forget. It also keeps ``core`` from importing ``tracking``, which owns the
frames schema. A caller that wants to *report* affected frame sets reads that
index itself, above both layers.

**Verdicts, not decisions.** A row says what a change reached and whether the
recorded value still matches; whether that justifies deleting anything is item
6.4's, behind an explicit force.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Literal

import pandas as pd

from .index import feature_index, feature_index_path, feature_run_root
from .sequence_index import decode_consumed_roots, encode_entry_composition
from .tracks_index import consumed_composition_for, read_tracks_index

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "PROVENANCE_COLUMNS",
    "index_records",
    "Verdict",
    "reached_by",
]

Verdict = Literal["current", "drifted", "unknown"]
"""What comparing a recorded composition against the present says.

``unknown`` is not a hedge. It is what either side being empty means, and the
two empties are different: a recorded ``""`` is a row written before item 5.1 or
one whose root was not establishable, a current ``""`` is a projection absent or
unestablishable now. Neither is evidence of change, and neither is evidence of
safety -- which is why item 6.2 fails closed on it rather than reading it as
``current``.
"""

PROVENANCE_COLUMNS: list[str] = [
    "kind",
    "name",
    "run_id",
    "group",
    "sequence",
    "abs_path",
    "consumed_roots",
    "recorded",
    "current",
    "verdict",
    "via",
]
"""The returned frame's columns, in order.

``kind``/``name``/``run_id`` locate the artifact class, ``abs_path`` the file.
``via`` says how the row was reached -- ``direct`` for a recorded composition
that moved, ``tracks`` for a feature run whose tracks variant moved under it --
because the two are answerable by different evidence and a reader deciding what
to delete needs to know which.
"""


def index_records(frame: pd.DataFrame) -> list[dict[str, str]]:
    """A frame's rows as plain string dicts.

    The one place pandas' partially-typed row access is turned into something the
    rest of the module can read without a cast at every cell. Every index walked
    here is written through :class:`IndexCSV`, whose dtype map pins each schema
    ``str`` column, so the cells are already strings and this states that rather
    than converting them.
    """
    columns = [str(name) for name in frame.columns]
    return [
        {column: str(value) for column, value in zip(columns, row, strict=True)}
        for row in frame.itertuples(index=False, name=None)
    ]


def _verdict(recorded: str, current: str) -> Verdict:
    """Compare one recorded composition against the present. See :data:`Verdict`."""
    if not recorded or not current:
        return "unknown"
    return "current" if recorded == current else "drifted"


def _row(
    *,
    kind: str,
    name: str,
    run_id: str,
    group: str,
    sequence: str,
    abs_path: str,
    consumed_roots: Iterable[str],
    recorded: str,
    current: str,
    via: str,
) -> dict[str, object]:
    return {
        "kind": kind,
        "name": name,
        "run_id": run_id,
        "group": group,
        "sequence": sequence,
        "abs_path": abs_path,
        "consumed_roots": ",".join(consumed_roots),
        "recorded": recorded,
        "current": current,
        "verdict": _verdict(recorded, current),
        "via": via,
    }


def _tracks_rows(
    ds: Dataset, wanted: set[tuple[str, str]], root: str
) -> list[dict[str, object]]:
    """Tracks tables for *wanted* whose producer read *root*.

    Every variant of an entry, not the one a selector would resolve to: a change
    under ``tracks_raw`` reaches every table converted from it, and asking which
    one a feature *would* read today is a different question from which ones were
    built from bytes that moved.
    """
    frame = read_tracks_index(ds)
    rows: list[dict[str, object]] = []
    for record in index_records(frame):
        entry = (str(record.get("group", "")), str(record.get("sequence", "")))
        if entry not in wanted:
            continue
        declared = decode_consumed_roots(str(record.get("consumed_source_roots", "")))
        if root not in declared:
            continue
        rows.append(
            _row(
                kind="tracks",
                name=str(record.get("producer", "")),
                run_id=str(record.get("run_id", "")),
                group=entry[0],
                sequence=entry[1],
                abs_path=str(record.get("abs_path", "")),
                consumed_roots=declared,
                recorded=str(record.get("consumed_composition", "")),
                current=consumed_composition_for(ds, entry[0], entry[1], declared),
                via="direct",
            )
        )
    return rows


def _feature_storages(ds: Dataset) -> list[str]:
    """Every feature storage directory, sorted. Empty when the root is unset."""
    if not ds.roots.get("features"):
        return []
    root = ds.get_root("features")
    if not root.exists():
        return []
    return sorted(child.name for child in root.iterdir() if child.is_dir())


def _feature_rows(
    ds: Dataset, wanted: set[tuple[str, str]], root: str
) -> list[dict[str, object]]:
    """Feature entries that declared *root* and recorded a composition under it."""
    rows: list[dict[str, object]] = []
    for storage in _feature_storages(ds):
        index = feature_index(feature_index_path(ds, storage))
        if not index.path.exists():
            continue
        frame = index.read(validate_paths=False)
        for record in index_records(frame):
            entry = (str(record.get("group", "")), str(record.get("sequence", "")))
            if entry not in wanted:
                continue
            declared = decode_consumed_roots(str(record.get("consumed_roots", "")))
            if root not in declared:
                continue
            current = encode_entry_composition(
                _current_compositions(ds, entry), declared
            )
            rows.append(
                _row(
                    kind="features",
                    name=storage,
                    run_id=str(record.get("run_id", "")),
                    group=entry[0],
                    sequence=entry[1],
                    abs_path=str(record.get("abs_path", "")),
                    consumed_roots=declared,
                    recorded=str(record.get("consumed_composition", "")),
                    current=current,
                    via="direct",
                )
            )
    return rows


def _current_compositions(ds: Dataset, entry: tuple[str, str]) -> dict[str, str]:
    """What every source root records for *entry* now."""
    from .sequence_index import read_entry_compositions

    return read_entry_compositions(ds, [entry]).get(entry, {})


def _consumed_variants(ds: Dataset, storage: str, run_id: str) -> set[str]:
    """The tracks variants a feature run read, from its ``params.json``.

    The identifier already carries them -- ``_tracks`` is a hash term -- so this
    is the readable copy item 0.3 persisted for exactly this walk rather than a
    second source of truth. A run whose file is missing or unreadable contributes
    nothing, which leaves it out of the blast radius rather than guessing it in:
    a run nothing can describe is item 6.2's to fail closed on, not this
    function's to invent an edge for.
    """
    path = feature_run_root(ds, storage, run_id) / "params.json"
    parsed: object
    try:
        parsed = json.loads(path.read_text())
    except (OSError, ValueError):
        return set()
    if not isinstance(parsed, dict):
        return set()
    # Narrowed, then widened to a fully-known mapping, the way ``fit_scope``
    # reads its own record: ``json.loads`` is untyped, so every read off the
    # narrowed dict would otherwise be an unknown the strict checker cannot see
    # through. A record on disk is untrusted input -- hand-edited, truncated, or
    # written by a future version -- so every read narrows before it converts.
    payload: Mapping[object, object] = parsed
    entries = payload.get("_resolved")
    if not isinstance(entries, list):
        return set()
    listed: list[object] = entries
    variants: set[str] = set()
    for item in listed:
        if not isinstance(item, dict):
            continue
        reference: Mapping[object, object] = item
        if reference.get("where") != "inputs[tracks]":
            continue
        variant = reference.get("run_id")
        if isinstance(variant, str) and variant:
            variants.add(variant)
    return variants


def _transitive_rows(
    ds: Dataset,
    wanted: set[tuple[str, str]],
    moved_variants: set[str],
    already: set[tuple[str, str, str, str]],
) -> list[dict[str, object]]:
    """Feature entries reached through a tracks variant that moved under them.

    The arm that matters. Forty of the forty-two registered features declare no
    source root, so their own cell is empty and the direct arm cannot see them --
    they read media only through tracks, and this is the edge that says so.
    """
    if not moved_variants:
        return []
    rows: list[dict[str, object]] = []
    for storage in _feature_storages(ds):
        index = feature_index(feature_index_path(ds, storage))
        if not index.path.exists():
            continue
        frame = index.read(validate_paths=False)
        records = index_records(frame)
        variants_by_run: dict[str, set[str]] = {}
        for record in records:
            entry = (str(record.get("group", "")), str(record.get("sequence", "")))
            if entry not in wanted:
                continue
            run_id = str(record.get("run_id", ""))
            if run_id not in variants_by_run:
                variants_by_run[run_id] = _consumed_variants(ds, storage, run_id)
            if not (variants_by_run[run_id] & moved_variants):
                continue
            if ("features", storage, run_id, _key(entry)) in already:
                continue
            rows.append(
                _row(
                    kind="features",
                    name=storage,
                    run_id=run_id,
                    group=entry[0],
                    sequence=entry[1],
                    abs_path=str(record.get("abs_path", "")),
                    consumed_roots=decode_consumed_roots(
                        str(record.get("consumed_roots", ""))
                    ),
                    # The tracks table moved, not this row's own record -- which
                    # is empty for a feature declaring no root. Carrying the
                    # feature's own values here would report `unknown` for a row
                    # whose upstream is known to have moved.
                    recorded="tracks",
                    current="moved",
                    via="tracks",
                )
            )
    return rows


def _key(entry: tuple[str, str]) -> str:
    return f"{entry[0]}\x00{entry[1]}"


def reached_by(
    ds: Dataset,
    changed: Iterable[tuple[str, str]],
    root: str,
) -> pd.DataFrame:
    """Every derived artifact a change to *root* under *changed* reaches.

    *changed* is the ``(group, sequence)`` entries whose source moved, and *root*
    the source root it moved under (``media_raw`` or ``tracks_raw``). Scoping by
    root is what keeps invalidation honest across roots: uploading a video for
    visualisation changes only the media composition, so kinematic features built
    from ``tracks_raw`` that never touched a pixel are not reached at all.

    Usable **before** the change as well as after. Run against a dataset the
    change has not yet touched, every row reads ``current`` and the answer is the
    membership -- what *would* be reached. Run after, the rows that moved read
    ``drifted``. One function serves the preview and the audit, because the two
    ask the same question at different times.

    Returns a frame with :data:`PROVENANCE_COLUMNS`, empty (with the full schema,
    so a caller may filter it without a ``KeyError``) when nothing is reached.
    """
    wanted = {(str(group), str(sequence)) for group, sequence in changed}
    if not wanted:
        return pd.DataFrame(columns=pd.Index(PROVENANCE_COLUMNS))

    tracks = _tracks_rows(ds, wanted, root)
    features = _feature_rows(ds, wanted, root)

    # Only a variant that actually moved propagates. A tracks table whose
    # composition still matches is not an edge to follow, and one whose verdict is
    # `unknown` is not either -- an absent record is not evidence of change, and
    # inventing an edge from it would delete on a guess.
    moved_variants = {
        str(row["run_id"]) for row in tracks if row["verdict"] == "drifted"
    }
    already = {
        (
            "features",
            str(row["name"]),
            str(row["run_id"]),
            _key((str(row["group"]), str(row["sequence"]))),
        )
        for row in features
    }
    transitive = _transitive_rows(ds, wanted, moved_variants, already)

    rows = [*tracks, *features, *transitive]
    if not rows:
        return pd.DataFrame(columns=pd.Index(PROVENANCE_COLUMNS))
    return pd.DataFrame(rows, columns=pd.Index(PROVENANCE_COLUMNS))
