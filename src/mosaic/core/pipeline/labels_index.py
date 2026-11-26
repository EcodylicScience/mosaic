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
overwrite.

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

import numpy as np
import pandas as pd

from mosaic.core.entry import Entry
from mosaic.core.helpers import detect_label_format, text_cell, to_safe_name
from mosaic.core.label_intervals import refuse_segments_only
from mosaic.core.pipeline.entry_keys import append_refusing_key_collision
from mosaic.core.pipeline.dataset_indexes import (
    label_kinds,
    register_reconcilable_index,
)
from mosaic.core.pipeline.index_csv import (
    IndexCSV,
    RunIndexRowBase,
    project_to_schema,
)
from mosaic.core.pipeline.labels_identity import (
    labels_variant_root,
    read_labels_variant,
)
from mosaic.core.pipeline.linked_rows import LINK_ID_COLUMN, merge_linked_rows
from mosaic.core.pipeline.tracks_index import (
    consumed_compositions_for,
    encode_source_roots,
)
from mosaic.core.pipeline.variant_rows import (
    check_variant_sidecar,
    keep_last_of_each_key,
    parse_variant_run_id,
    refuse_rows_with_inputs,
    refuse_unnamed_entry,
    validated_entries,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "DROPPED_LEGACY_COLUMNS",
    "LABELS_INDEX_COLUMNS",
    "LABELS_INDEX_PATH_COLUMNS",
    "LABELS_ROWS_PATH_COLUMNS",
    "LabelsIndexRow",
    "LabelsRowDraft",
    "adopt_legacy_label_columns",
    "empty_labels_frame",
    "index_labels_variant",
    "labels_index",
    "labels_index_path",
    "labels_rows",
    "legacy_labels_view",
    "read_labels_index",
    "readable_label_kinds",
    "select_label_variant_rows",
    "write_labels_row",
    "write_labels_rows",
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

LABELS_ROWS_PATH_COLUMNS: Final[tuple[str, ...]] = (
    "abs_path",
    *LABELS_INDEX_PATH_COLUMNS,
)
"""The path columns of a linked row that :func:`labels_rows` anchors at its dataset.

``abs_path`` and :data:`LABELS_INDEX_PATH_COLUMNS`, the columns the portability
passes rewrite, so the two cannot name different columns.
"""

_LABELS_ROW_KEYS: Final = ("run_id", "group", "sequence")
"""The columns that identify a labels row across the datasets of a lookup."""

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


def labels_rows(ds: Dataset, label_kind: str) -> pd.DataFrame:
    """The labels index of *label_kind* in *ds* merged with those of its links.

    Reads each dataset of ``ds.lookup_order("labels")`` with
    :func:`read_labels_index` and merges the frames with
    :func:`~mosaic.core.pipeline.linked_rows.merge_linked_rows`, keyed on
    ``(run_id, group, sequence)``: an own row shadows a linked row of the same
    variant and entry, and an own row and a linked row of one entry under two
    variants both stay, for :func:`select_label_variant_rows` to choose between.
    The cells of :data:`LABELS_ROWS_PATH_COLUMNS` in a linked row are made
    absolute against its own dataset, so they name its files from *ds*.

    Writes nothing.

    Returns:
        The merged rows in lookup order, with a ``link_id`` column that names
        the link of each row and is ``""`` for an own row. For a dataset that
        links nothing, the frame :func:`read_labels_index` returns plus that
        column.
    """
    return merge_linked_rows(
        [
            (
                linked.link_id,
                linked.dataset.base_dir,
                read_labels_index(linked.dataset, label_kind),
            )
            for linked in ds.lookup_order("labels")
        ],
        _LABELS_ROW_KEYS,
        LABELS_ROWS_PATH_COLUMNS,
    )


def readable_label_kinds(ds: Dataset) -> list[str]:
    """The converted label kinds of *ds* and of the datasets it links, each once.

    Returns:
        The kinds of each dataset of ``ds.lookup_order("labels")`` in that
        order, sorted within a dataset, the first occurrence of a kind kept.
    """
    seen: dict[str, None] = {}
    for linked in ds.lookup_order("labels"):
        seen.update(dict.fromkeys(label_kinds(linked.dataset)))
    return list(seen)


def select_label_variant_rows(
    df: pd.DataFrame, run_id: str | None = None
) -> pd.DataFrame:
    """Reduce a labels frame to one row per ``(group, sequence)``.

    The single place that decides which label variant an entry resolves to, so a
    reader and a consumer cannot answer it two ways. Order-preserving.

    ``run_id`` given selects that variant exactly. ``""`` names the *unlabelled*
    rows -- a flat ``labels/<kind>/`` written before variants existed -- which is
    how a layout from before variants stays addressable.

    ``run_id`` of ``None`` means "whichever variant this entry has". On rows that
    :func:`labels_rows` merged, which carry a ``link_id`` and come in lookup
    order, an entry first keeps the rows of the nearest dataset that holds a row
    of it, so an entry converted in the reading dataset takes that variant over a
    linked one. The nearest dataset wins even when its row is unlabelled and a
    linked row is labelled. Within that dataset the rule is the one tracks
    follow: an empty ``run_id`` is unknown, never a peer variant, so a labelled
    row supersedes an unlabelled one and the first re-conversion of a migrated
    dataset does not fire the ambiguity below on every entry. Two genuinely
    different recipes for one entry in one dataset have no defensible default and
    raise.
    """
    if df.empty:
        return df.reset_index(drop=True)

    run_ids = [str(value) for value in df["run_id"]]
    groups = [str(value) for value in df["group"]]
    sequences = [str(value) for value in df["sequence"]]
    link_ids = (
        [str(value) for value in df[LINK_ID_COLUMN]]
        if LINK_ID_COLUMN in df.columns
        else [""] * len(df)
    )

    if run_id is not None:
        named = [i for i, value in enumerate(run_ids) if value == run_id]
        return df.iloc[named].reset_index(drop=True)

    positions_by_entry: dict[tuple[str, str], list[int]] = {}
    for position, entry in enumerate(zip(groups, sequences, strict=True)):
        positions_by_entry.setdefault(entry, []).append(position)

    keep: list[int] = []
    for entry, positions in positions_by_entry.items():
        nearest = link_ids[positions[0]]
        held = [p for p in positions if link_ids[p] == nearest]
        labelled = [p for p in held if run_ids[p]]
        variants = sorted({run_ids[p] for p in labelled})
        if len(variants) > 1:
            raise ValueError(_ambiguous_label_variant_message(entry, variants))
        keep.append((labelled or held)[-1])
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
        f"read: pass labels_run_id=<variant> to run_feature, Dataset.run_feature or "
        f"load_values (or --labels-run-id on `mosaic run`)."
    )


def legacy_labels_view(df: pd.DataFrame) -> pd.DataFrame:
    """Add back the columns the untyped labels writer emitted, for the reverse path.

    ``group_safe``/``sequence_safe`` are re-derived from ``group``/``sequence``,
    and ``kind`` from ``label_kind`` -- so a typed index reads back as the old
    twelve-column CSV when the typed index is rolled back. Never written by
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


@dataclass(frozen=True, slots=True, kw_only=True)
class LabelsRowDraft:
    """The arguments of one :func:`write_labels_row` call, held for a later write.

    The fields are those of :func:`write_labels_row` after ``ds``, which documents
    each of them.
    """

    run_id: str
    group: object
    sequence: object
    out_path: Path
    producer: str
    label_kind: str
    label_format: str = ""
    n_frames: int
    label_ids: str = ""
    label_names: str = ""
    producer_run_id: str = ""
    source: Path | str = ""
    source_md5: str = ""
    consumed_source_roots: Sequence[str] = ()


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
    """Record one converted-labels table. The only way to write one row of this index.

    Keyword-only, and ``group``/``sequence`` are read through
    :func:`~mosaic.core.helpers.text_cell` for the same reason the tracks writer
    does it: a caller reading them off a pandas ``Series`` gets ``numpy``
    scalars, which would defeat the dedup that holds this index to one row per
    entry, and a blank cell arrives there as a float NaN that ``str()`` spells
    as the word ``nan`` -- a group no composition was ever recorded under.
    ``consumed_composition`` is computed here from the declared
    ``consumed_source_roots`` via the shared :func:`consumed_compositions_for`,
    which covers ``labels_raw`` too.

    Call this *after* writing the ``.npz``: ``IndexRowBase`` existence-checks an
    absolute ``abs_path``.
    """
    write_labels_rows(
        ds,
        [
            LabelsRowDraft(
                run_id=run_id,
                group=group,
                sequence=sequence,
                out_path=out_path,
                producer=producer,
                label_kind=label_kind,
                label_format=label_format,
                n_frames=n_frames,
                label_ids=label_ids,
                label_names=label_names,
                producer_run_id=producer_run_id,
                source=source,
                source_md5=source_md5,
                consumed_source_roots=consumed_source_roots,
            )
        ],
    )


def write_labels_rows(ds: Dataset, drafts: Sequence[LabelsRowDraft]) -> None:
    """Record several converted-labels tables of one kind in one write of the index.

    What :func:`write_labels_row` does for one table, for a writer of many: the
    rows are built, checked for key collisions and appended in one
    read-modify-write of the kind's index under its lock, and the compositions
    they record are read once.

    The result is the index that writing the drafts one after another gives. A
    draft that names an entry (and variant) another draft of the batch names
    replaces it, so the last of them keeps its row, at its place in the order. A
    collision between any two entries of the batch, or with the index, refuses
    the whole batch and writes none of it.

    Raises:
        ValueError: If the drafts name more than one ``label_kind``.
        EntryKeyCollisionError: If a new entry shares its key with an entry of
            the index, of the media index or of the batch, or takes a continuous
            group's name.
    """
    if not drafts:
        return
    kinds = sorted({draft.label_kind for draft in drafts})
    if len(kinds) > 1:
        raise ValueError(
            f"A batch of labels rows is of one kind, and these name {kinds}."
        )
    entries = validated_entries((draft.group, draft.sequence) for draft in drafts)
    compositions = consumed_compositions_for(
        ds,
        [
            (entry, draft.consumed_source_roots)
            for entry, draft in zip(entries, drafts, strict=True)
        ],
    )
    rows = [
        LabelsIndexRow(
            abs_path=Path(ds.relative_to_root(draft.out_path)),
            run_id=draft.run_id,
            group=group,
            sequence=sequence,
            producer=draft.producer,
            label_kind=draft.label_kind,
            label_format=draft.label_format,
            producer_run_id=draft.producer_run_id,
            source_abs_path=ds.relative_to_root(draft.source) if draft.source else "",
            source_md5=draft.source_md5,
            consumed_source_roots=encode_source_roots(draft.consumed_source_roots),
            n_frames=int(draft.n_frames),
            label_ids=draft.label_ids,
            label_names=draft.label_names,
            consumed_composition=composition,
        )
        for (group, sequence), draft, composition in zip(
            entries, drafts, compositions, strict=True
        )
    ]
    append_refusing_key_collision(
        ds,
        labels_index(labels_index_path(ds, kinds[0])),
        keep_last_of_each_key(rows),
        set(entries),
    )


_LABELS_CONVERSION_CELLS: Final = (
    "source_abs_path",
    "source_md5",
    "consumed_source_roots",
    "consumed_composition",
)
"""The cells of a row that only a conversion or a derivation fills."""


def index_labels_variant(ds: Dataset, label_kind: str, run_id: str) -> int:
    """Record the files already under ``labels/<label_kind>/<run_id>/`` in the index.

    The call reads the dataset's label files and records them without
    converting: no file is written, moved or renamed. It writes one row per
    ``<entry key>.npz`` of the variant directory through
    :func:`write_labels_rows`, in one write. A name that starts with ``.`` (a
    file being staged) and every file that is not an ``.npz`` are skipped.

    A row records ``producer`` as the kind of *run_id*, and from the file its
    ``label_format``, ``n_frames`` as one past its last frame (0 for a file
    without frames, counted on the file's frame axis), and ``label_ids`` and
    ``label_names`` joined by commas as a conversion joins them. The source,
    its md5 and the consumed source roots and composition stay blank. A rerun
    writes the same rows apart from ``started_at`` and replaces the earlier
    ones by key.

    Args:
        ds: The dataset that holds the variant.
        label_kind: The kind's directory name under ``labels/``, which the
            variant's sidecar records as its ``kind``.
        run_id: The variant's run id, the name of its directory under the kind.

    Returns:
        The number of rows written.

    Raises:
        ValueError: Before any row is written, if *run_id* is not an op run id,
            the variant directory has no readable ``params.json``, the
            sidecar's op is not the kind of *run_id* or its kind is not
            *label_kind*, a file lacks ``label_format``, ``frames``,
            ``label_ids``, ``label_names``, ``group`` or ``sequence``, a file's
            ``group`` and ``sequence`` are not the entry that its name gives, or
            the kind's index holds a row of *run_id* that records a source, its
            md5, consumed source roots or a consumed composition. Such a row
            comes from a conversion, and replacing it would drop the inputs it
            records.
        EntryKeyCollisionError: If an entry shares its key with another entry
            of the index or of the media index. No row is written.
    """
    parsed = parse_variant_run_id(run_id)
    root = labels_variant_root(ds.get_root("labels") / label_kind, run_id)
    sidecar = check_variant_sidecar(root, parsed, read_labels_variant(root))
    if sidecar.kind != label_kind:
        raise ValueError(
            f"The sidecar of {root} records kind {sidecar.kind!r}, "
            f"and the kind to index is {label_kind!r}."
        )
    held = read_labels_index(ds, label_kind)
    refuse_rows_with_inputs(
        held[held["run_id"] == run_id],
        run_id,
        _LABELS_CONVERSION_CELLS,
        f"labels index of {label_kind!r}",
    )
    drafts = [
        _label_draft(path, label_kind, run_id, parsed.kind)
        for path in sorted(root.glob("*.npz"))
        if not path.name.startswith(".") and path.is_file()
    ]
    write_labels_rows(ds, drafts)
    return len(drafts)


def _label_draft(
    path: Path, label_kind: str, run_id: str, producer: str
) -> LabelsRowDraft:
    """The row of the label file at *path*, as its contents and its name give it.

    Raises:
        SegmentsOnlyLabels: If the file is ``segments_v1``, which holds no frames.
        ValueError: If the file lacks a key that the row records, or its ``group``
            and ``sequence`` are not the entry whose key is the file's stem.
    """
    with np.load(path, allow_pickle=True) as loaded:
        refuse_segments_only(detect_label_format(loaded), path, "index_labels_variant")
        for key in ("label_format", "frames", "label_ids", "label_names"):
            if key not in loaded.files:
                raise ValueError(f"{path} has no {key!r} array.")
        entry = _file_entry(loaded, path)
        frames = np.asarray(loaded["frames"])
        return LabelsRowDraft(
            run_id=run_id,
            group=entry[0],
            sequence=entry[1],
            out_path=path,
            producer=producer,
            label_kind=label_kind,
            label_format=str(loaded["label_format"]),
            n_frames=int(frames.max()) + 1 if frames.size else 0,
            label_ids=",".join(str(value) for value in loaded["label_ids"]),
            label_names=",".join(str(value) for value in loaded["label_names"]),
        )


def _file_entry(loaded: np.lib.npyio.NpzFile, path: Path) -> Entry:
    """The ``(group, sequence)`` scalars of a label file, as its stem names them."""
    values: list[str] = []
    for key in ("group", "sequence"):
        if key not in loaded.files:
            raise ValueError(f"{path} has no {key!r} scalar.")
        values.append(text_cell(loaded[key].item()))
    group, sequence = values
    refuse_unnamed_entry(path, group, sequence)
    return group, sequence


# Reconciled through the shared registry, so ``reindex`` and the sweep
# cover every ``labels/<kind>/index.csv`` the way they cover ``tracks/index.csv``.
register_reconcilable_index("labels", labels_index)
