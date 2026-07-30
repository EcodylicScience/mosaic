"""The typed per-kind index of converted labels under ``labels/<kind>/``.

Not to be confused with :mod:`mosaic.core.pipeline.labels_raw_index` -- there is
no such module, because ``labels_raw``'s raw-file index shares the
``tracks_raw_index`` schema and helpers. *This* index is the label sibling of
:mod:`mosaic.core.pipeline.tracks_index`: it names the converted ``.npz`` tables,
one row per ``(run_id, group, sequence)``, and it was the last index in the
toolkit written by hand -- a twelve-name column list projected with
``row.get(k, "")`` and concatenated with a bare ``to_csv``, no lock, no atomic
rename, no run identity.

**One index per kind.** ``labels/<kind>/index.csv``, because the kind is a real
namespace a consumer selects by (``behavior``, ``id_tags``, ...), unlike a tracks
variant, which is an anonymous recipe. The variant lives one directory below, at
``labels/<kind>/<run_id>/``, so two recipes for one kind coexist rather than
overwrite -- the central defect item 9.3 closes.

**Two provenances, one root.** A label kind can arrive scored (converted from
``labels_raw``, ``consumed_source_roots=("labels_raw",)``, source side the
``labels_raw`` composition) or derived (computed from tracks or features, source
side the upstream artifact identity). The ``consumed_source_roots`` cell
distinguishes them, exactly as it does for a tracks table converted from an
upload versus one tracked here. A third, quieter case is authored-in-place
(``id_tags`` from an external CSV in no raw index): an honest empty
``consumed_source_roots`` and no composition, neither scored nor derived.

Resolution mirrors tracks: :func:`select_label_variant_rows` is the one place
that answers "which variant does this entry resolve to", an unlabelled row loses
to a labelled one, and two genuine recipes refuse to be guessed between.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd

from mosaic.core.helpers import to_safe_name, validate_entry_name
from mosaic.core.pipeline.dataset_indexes import register_reconcilable_index
from mosaic.core.pipeline.index_csv import (
    IndexCSV,
    RunIndexRowBase,
    project_to_schema,
)
from mosaic.core.pipeline.tracks_index import (
    consumed_composition_for,
    encode_source_roots,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "DROPPED_LEGACY_COLUMNS",
    "LABELS_INDEX_COLUMNS",
    "LABELS_INDEX_PATH_COLUMNS",
    "LabelsIndexRow",
    "adopt_legacy_label_columns",
    "empty_labels_frame",
    "labels_index",
    "labels_index_path",
    "legacy_labels_view",
    "read_labels_index",
    "select_label_variant_rows",
    "write_labels_row",
]


@dataclass(frozen=True, slots=True)
class LabelsIndexRow(RunIndexRowBase):
    """One row of ``labels/<kind>/index.csv``.

    Field order is the CSV column order, after ``abs_path``/``run_id``/
    ``started_at``/``finished_at`` from the bases. It is :class:`TracksIndexRow`
    with the label-specific columns (``label_kind``, ``label_format``,
    ``label_ids``, ``label_names``) and ``n_frames`` in place of ``n_rows``.

    ``run_id`` is the *label variant* identity from
    :mod:`mosaic.core.pipeline.labels_identity` -- what recipe produced these
    labels -- not the producing op's run. ``producer`` is
    ``convert-labels-<src_format>`` for a scored kind, or an upstream op for a
    derived one; ``producer_run_id`` is the upstream feature/tracks run a derived
    kind chained from, empty for a conversion.

    ``consumed_source_roots`` names the roots these labels were derived from --
    ``labels_raw`` for a scored kind, a derived root for a derived one, empty for
    an authored one -- and ``consumed_composition`` is what those roots held for
    this entry when the labels were written. The pair sits on the row and never in
    ``run_id``, for the same reason it does on the tracks row: a variant is
    params-only and scope-free, so one identity covers many sequences whose source
    compositions differ.
    """

    group: str
    sequence: str
    producer: str
    label_kind: str
    label_format: str = ""
    producer_run_id: str = ""
    source_abs_path: str = ""
    source_md5: str = ""
    consumed_source_roots: str = ""
    n_frames: int = 0
    label_ids: str = ""
    label_names: str = ""
    consumed_composition: str = ""


LABELS_INDEX_COLUMNS: Final[list[str]] = [
    field.name for field in fields(LabelsIndexRow)
]
"""The schema, in CSV order. Derived from the row so the two cannot drift."""

LABELS_INDEX_PATH_COLUMNS: Final[tuple[str, ...]] = ("source_abs_path",)
"""Path-bearing columns beyond ``abs_path``.

Named here because ``Dataset``'s path-rewriting passes read raw CSVs and have no
row class to ask. A new path column belongs in this tuple or it silently stops
being portable. ``consumed_source_roots`` is deliberately absent: it holds root
*keys*, machine-independent already.
"""

DROPPED_LEGACY_COLUMNS: Final[tuple[str, ...]] = (
    "group_safe",
    "sequence_safe",
    "kind",
    "n_events",
)
"""What the hand-written labels writer emitted that this schema does not.

``group_safe``/``sequence_safe`` are pure functions of ``group``/``sequence`` --
a cache the old writer recomputed on every write -- and :func:`legacy_labels_view`
re-derives them. ``kind`` was the label kind, redundant with the ``labels/<kind>/``
directory the index lives in and superseded by the typed ``label_kind`` column.
``n_events`` was a per-converter extra already present inside every ``.npz``, so
dropping it from the index loses nothing.
"""


def labels_index_path(ds: Dataset, kind: str) -> Path:
    """Where a kind's converted-labels index lives: ``labels/<kind>/index.csv``."""
    return ds.get_root("labels") / kind / "index.csv"


def labels_index(path: Path) -> IndexCSV[LabelsIndexRow]:
    """Factory: an ``IndexCSV`` configured for the labels schema.

    ``dedup_keys`` is the ``run_id``-led triple every typed index uses. Which row
    an entry *resolves* to is a separate question -- see
    :func:`select_label_variant_rows`. Registered with the shared reconciler under
    the ``labels`` root key (below), so ``reindex`` and the sweep cover it too.
    """
    return IndexCSV(
        path,
        LabelsIndexRow,
        dedup_keys=["run_id", "group", "sequence"],
        adopt=adopt_legacy_label_columns,
    )


def adopt_legacy_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bring a labels frame read off disk up to the current schema, in memory.

    Wired in as ``IndexCSV``'s ``adopt`` hook, so it runs inside the write lock.
    Missing columns are added empty, NaN coerced to ``""``, off-schema columns
    (see :data:`DROPPED_LEGACY_COLUMNS`) dropped. The projection is
    :func:`project_to_schema`, shared with the other typed indexes; a legacy flat
    ``labels/<kind>/index.csv`` adopts as rows with an empty ``run_id``, which
    :func:`select_label_variant_rows` then supersedes on the first re-conversion.
    """
    return project_to_schema(df, LABELS_INDEX_COLUMNS)


def empty_labels_frame() -> pd.DataFrame:
    """The full-schema, zero-row frame an absent labels index reads as.

    The column set is load-bearing: callers filter on ``group``/``sequence``
    straight away, and a column-less empty frame turns "no labels of this kind
    yet" into ``KeyError: 'group'``.
    """
    return pd.DataFrame(
        {column: pd.Series(dtype="object") for column in LABELS_INDEX_COLUMNS}
    )


def read_labels_index(ds: Dataset, kind: str) -> pd.DataFrame:
    """Read ``labels/<kind>/index.csv``, projected onto the current schema.

    The single reader. An absent index -- or an unset ``labels`` root -- reads as
    an empty one, so listing a kind with no conversions yet does not raise and a
    read-only mount works. Never writes.
    """
    try:
        path = labels_index_path(ds, kind)
    except KeyError:
        return empty_labels_frame()
    if not path.exists():
        return empty_labels_frame()
    raw = pd.read_csv(path, keep_default_na=False, dtype=str)
    return adopt_legacy_label_columns(raw)


def select_label_variant_rows(
    df: pd.DataFrame, run_id: str | None = None
) -> pd.DataFrame:
    """Reduce a labels frame to one row per ``(group, sequence)``.

    The single place that decides which label variant an entry resolves to, so a
    reader and a consumer cannot answer it two ways. Order-preserving.

    ``run_id`` given selects that variant exactly. ``""`` names the *unlabelled*
    rows -- a flat ``labels/<kind>/`` written before variants existed -- which is
    how a pre-9.3 layout stays addressable.

    ``run_id`` of ``None`` means "whichever variant this entry has", applying the
    same rule as tracks: an empty ``run_id`` is unknown, never a peer variant, so
    a labelled row supersedes an unlabelled one and the first re-conversion of a
    migrated dataset does not fire the ambiguity below on every entry. Two
    genuinely different recipes for one entry has no defensible default and raises.
    """
    if df.empty:
        return df.reset_index(drop=True)

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
            raise ValueError(_ambiguous_label_variant_message(entry, variants))
        keep.append((labelled or positions)[-1])
    return df.iloc[sorted(keep)].reset_index(drop=True)


def _ambiguous_label_variant_message(
    entry: tuple[str, str], variants: list[str]
) -> str:
    """Name the entry, both candidates, and the keyword that resolves it."""
    group, sequence = entry
    listing = ", ".join(repr(variant) for variant in variants)
    return (
        f"labels index holds {len(variants)} variants of "
        f"(group={group!r}, sequence={sequence!r}): {listing}. "
        f"There is no defensible default between two recipes, so say which one to "
        f"read: pass labels_run_id=<variant> to run_feature or build_manifest."
    )


def legacy_labels_view(df: pd.DataFrame) -> pd.DataFrame:
    """Add back the columns the untyped labels writer emitted, for the reverse path.

    ``group_safe``/``sequence_safe`` are re-derived from ``group``/``sequence``,
    and ``kind`` from ``label_kind`` -- so a typed index reads back as the old
    twelve-column CSV when item 9.3's ``[break]`` is rolled back. Never written by
    the forward path.
    """
    out = df.copy()
    out["kind"] = [str(value) for value in out["label_kind"]]
    out["group_safe"] = [
        to_safe_name(str(value)) if str(value) else "" for value in out["group"]
    ]
    out["sequence_safe"] = [
        to_safe_name(str(value)) if str(value) else "" for value in out["sequence"]
    ]
    return out


def write_labels_row(
    ds: Dataset,
    *,
    run_id: str,
    group: object,
    sequence: object,
    out_path: Path,
    producer: str,
    label_kind: str,
    label_format: str = "",
    n_frames: int,
    label_ids: str = "",
    label_names: str = "",
    producer_run_id: str = "",
    source: Path | str = "",
    source_md5: str = "",
    consumed_source_roots: Sequence[str] = (),
) -> None:
    """Record one converted-labels table. The only way to write this index.

    Keyword-only, and ``group``/``sequence`` are stringified for the same reason
    the tracks writer does it: a caller reading them off a pandas ``Series`` gets
    ``numpy`` scalars, which would defeat the dedup that holds this index to one
    row per entry. ``consumed_composition`` is computed here from the declared
    ``consumed_source_roots`` via the shared :func:`consumed_composition_for`,
    which now covers ``labels_raw`` too.

    Call this *after* writing the ``.npz``: ``IndexRowBase`` existence-checks an
    absolute ``abs_path``.
    """
    entry_group = validate_entry_name(str(group) if group is not None else "", "group")
    entry_sequence = validate_entry_name(
        str(sequence) if sequence is not None else "", "sequence"
    )
    row = LabelsIndexRow(
        abs_path=Path(ds.relative_to_root(out_path)),
        run_id=run_id,
        group=entry_group,
        sequence=entry_sequence,
        producer=producer,
        label_kind=label_kind,
        label_format=label_format,
        producer_run_id=producer_run_id,
        source_abs_path=ds.relative_to_root(source) if source else "",
        source_md5=source_md5,
        consumed_source_roots=encode_source_roots(consumed_source_roots),
        n_frames=int(n_frames),
        label_ids=label_ids,
        label_names=label_names,
        consumed_composition=consumed_composition_for(
            ds, entry_group, entry_sequence, consumed_source_roots
        ),
    )
    labels_index(labels_index_path(ds, label_kind)).append([row])


# Item 6.1: reconciled through the shared registry, so ``reindex`` and the sweep
# cover every ``labels/<kind>/index.csv`` the way they cover ``tracks/index.csv``.
register_reconcilable_index("labels", labels_index)
