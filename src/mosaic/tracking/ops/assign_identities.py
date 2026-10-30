"""``assign-identities``: relabel a tracks variant with named identities by appearance.

The op reads a tracks variant whose rows record appearance tables, builds one
prototype per identity from named reference tracks, and writes a variant whose
tracklets take the identity whose prototype they resemble. The arithmetic lives
in :mod:`mosaic.core.pipeline.appearance_assignment`.

The params name the source variant. The identity is therefore a function of the
params alone and is minted without reading the dataset. The source is an ``upstream``
term of both identifiers, as for ``resample-tracks``. The op reads tables and
opens no video. Its rows are written with ``records_media=False``.

The relabeled table keeps every source row. ``id`` is the assigned identity's
position among the sorted identity names, or a distinct negative id from -2 down
for an unassigned tracklet, and ``identity`` names the identity across entries.

A table depends on its source tables and on the prototypes. A source row's
composition covers only the media, so a table's row folds into that composition
a digest of the entry's own tracks and appearance tables and of every reference
entry's composition, first frame and tables. A written table is reused only
while that composition and its source's first frame still match.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Final, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, field_validator, model_validator

from mosaic.core.entry import Entry, make_entry_key
from mosaic.core.helpers import text_cell
from mosaic.core.individuals import refuse_repeated_rows
from mosaic.core.params import Declared, DeclaredModel, Params
from mosaic.core.pipeline.appearance import (
    AppearanceTable,
    AppearanceTableError,
    read_appearance_dim,
    read_appearance_file,
    recorded_appearance_path,
)
from mosaic.core.pipeline.appearance_assignment import (
    Prototypes,
    assign,
    prototypes,
    score,
    tracklets,
)
from mosaic.core.pipeline.error_kind import ClassifiedError, ErrorKind
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.op_identity import op_run_id
from mosaic.core.pipeline.ops import Op, OpIdentity, register_op
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.pipeline.tracks_identity import (
    chained_variant_payload,
    tracks_run_id,
    tracks_variant_root,
    write_tracks_variant,
)
from mosaic.core.pipeline.tracks_index import (
    read_first_frame,
    read_frames_read,
    read_known_tail_loss,
    read_media_frames,
    read_tracks_index,
    rows_in_scope,
    select_variant_rows,
    write_tracks_row,
)
from mosaic.core.pipeline.writers import (
    read_parquet_table,
    read_parquet_table_columns,
    write_parquet_atomic,
)
from mosaic.core.schema import ensure_track_schema
from mosaic.tracking.common.entry_ref import EntryRef
from mosaic.tracking.ops.chained_tracks import (
    WrittenTable,
    folded_composition,
    incomplete_variant_message,
    refuse_incomplete_variant,
    written_tables,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope

__all__ = [
    "ASSIGN_IDENTITIES_KIND",
    "AssignIdentitiesConflict",
    "AssignIdentitiesNotFound",
    "AssignIdentitiesOp",
    "AssignIdentitiesParams",
    "IdentityReference",
    "assign_identities_identity",
]

ASSIGN_IDENTITIES_KIND: Final = "assign-identities"
ASSIGN_IDENTITIES_VERSION: Final = "0.1"

_SOURCES_TERM: Final = "sources"

_MIN_IDENTITIES: Final = 2
"""The fewest identities a run assigns. One identity has no second best."""

_FIRST_UNASSIGNED_ID: Final = -2
"""The id of an entry's first unassigned tracklet. -1 is the label wildcard."""

_IDENTITY_DESCRIPTION = "The name of the individual seen in this reference."

_ENTRY_DESCRIPTION = "The entry the individual is seen in."

_ID_DESCRIPTION = (
    "The source track id of the individual in that entry. Unset means the "
    "individual is alone in the entry, and every row of the entry is that identity."
)

_SOURCE_TRACKS_RUN_ID_DESCRIPTION = (
    "The tracks variant to relabel, e.g. 'sleap.1.6-abc123def0'. Each scoped "
    "entry's row must record an appearance table."
)

_REFERENCES_DESCRIPTION = (
    "Where each identity is seen. At least two identities are named."
)

_MIN_SIMILARITY_DESCRIPTION = (
    "A tracklet whose best prototype cosine is below this is not assigned."
)

_MIN_MARGIN_DESCRIPTION = (
    "A tracklet whose best cosine exceeds its second by less than this is not assigned."
)

_MAX_GAP_DESCRIPTION = (
    "A source track whose consecutive rows are further apart than this "
    "is split into tracklets there."
)


class IdentityReference(DeclaredModel):
    """Names one place where an identity is seen."""

    identity: Annotated[str, Field(min_length=1), Declared(_IDENTITY_DESCRIPTION)]
    entry: Annotated[EntryRef, Declared(_ENTRY_DESCRIPTION)]
    id: Annotated[int | None, Declared(_ID_DESCRIPTION)] = None

    @property
    def track(self) -> tuple[Entry, int | None]:
        """The entry and source id this reference names."""
        return (self.entry.entry, self.id)


def _reference_order(ref: IdentityReference) -> tuple[str, str, str, bool, int]:
    """Sort by identity, group, sequence and id, with an unset id first."""
    return (
        ref.identity,
        ref.entry.group,
        ref.entry.sequence,
        ref.id is not None,
        0 if ref.id is None else ref.id,
    )


class AssignIdentitiesParams(Params):
    """Declares the source variant, the references and the assignment thresholds.

    The references are stored sorted. The order they are given in moves no
    identifier.
    """

    source_tracks_run_id: Annotated[
        str, Field(min_length=1), Declared(_SOURCE_TRACKS_RUN_ID_DESCRIPTION)
    ]
    references: Annotated[list[IdentityReference], Declared(_REFERENCES_DESCRIPTION)]
    min_similarity: Annotated[float, Declared(_MIN_SIMILARITY_DESCRIPTION)] = 0.5
    min_margin: Annotated[float, Declared(_MIN_MARGIN_DESCRIPTION)] = 0.05
    max_gap: Annotated[
        int, Field(ge=0), Declared(_MAX_GAP_DESCRIPTION, unit="frames")
    ] = 15

    @field_validator("references")
    @classmethod
    def _sorted_without_repeats(
        cls, value: list[IdentityReference]
    ) -> list[IdentityReference]:
        """Sort the references, and refuse one given twice or one track named twice.

        A reference with an unset id names every track of its entry. A track of
        that entry named for another identity is then named twice.
        """
        ordered = sorted(value, key=_reference_order)
        named: dict[tuple[Entry, int | None], str] = {}
        for ref in ordered:
            other = named.get(ref.track)
            if other == ref.identity:
                msg = (
                    f"references name {ref.identity!r} at {ref.entry.entry!r} id "
                    f"{ref.id} more than once"
                )
                raise ValueError(msg)
            if other is not None:
                msg = (
                    f"references: {ref.entry.entry!r} id {ref.id} is named for both "
                    f"{other!r} and {ref.identity!r}"
                )
                raise ValueError(msg)
            for (entry, source_id), identity in named.items():
                if (
                    entry == ref.entry.entry
                    and None in (source_id, ref.id)
                    and identity != ref.identity
                ):
                    msg = (
                        f"references: {entry!r} is named whole and by id for both "
                        f"{identity!r} and {ref.identity!r}"
                    )
                    raise ValueError(msg)
            named[ref.track] = ref.identity
        return ordered

    @model_validator(mode="after")
    def _two_identities(self) -> Self:
        """Refuse references that name fewer than two identities."""
        identities = sorted({ref.identity for ref in self.references})
        if len(identities) < _MIN_IDENTITIES:
            msg = (
                f"references name {identities}, and a run needs at least two identities"
            )
            raise ValueError(msg)
        return self


class AssignIdentitiesNotFound(ClassifiedError, LookupError):
    """Reports a source variant, reference entry or reference track that is absent.

    Raised for a source variant with no row in the scope, a reference entry
    without a row of the source variant, and a reference without a row or
    without an appearance vector in its entry.
    """

    error_kind: ClassVar[ErrorKind] = "not_found"


class AssignIdentitiesConflict(ClassifiedError, ValueError):
    """Refuses a source or references that cannot be assigned from.

    Raised for an entry whose row records no appearance table, appearance
    tables of different dimensions, a whole-entry reference whose entry has two
    rows in one frame, and two references of one identity whose rows share a
    frame.
    """

    error_kind: ClassVar[ErrorKind] = "conflict"


@dataclass(frozen=True, slots=True)
class _SourceEntry:
    """The tables of one entry of the source variant.

    Attributes:
        entry: The ``(group, sequence)`` entry.
        row: The entry's tracks-index row.
        tracks: The tracks table.
        appearance: The appearance table.
    """

    entry: Entry
    row: pd.Series
    tracks: Path
    appearance: Path


def _source_entry(ds: Dataset, row: pd.Series, source: str) -> _SourceEntry:
    """Read the paths of one source row, refusing a row without an appearance table."""
    entry = (text_cell(row.get("group")), text_cell(row.get("sequence")))
    appearance = recorded_appearance_path(ds, row)
    if appearance is None:
        producer = text_cell(row.get("producer")) or "an unnamed producer"
        msg = (
            f"{ASSIGN_IDENTITIES_KIND}: {entry!r} has no appearance table in "
            f"{source}, which {producer} wrote. A sleap run whose last model is an "
            "embedding model writes appearance tables."
        )
        raise AssignIdentitiesConflict(msg)
    return _SourceEntry(
        entry=entry,
        row=row,
        tracks=ds.resolve_path(text_cell(row.get("abs_path"))),
        appearance=appearance,
    )


def _table_digests(each: _SourceEntry) -> list[str]:
    """Return the content digests of *each*'s tracks and appearance tables."""
    return [file_digest(each.tracks), file_digest(each.appearance)]


def _canonical_digest(recorded: Sequence[object]) -> str:
    """Return the SHA-256 of *recorded* as canonical JSON, in hex."""
    canonical = json.dumps(recorded, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _references_digest(references: Sequence[_SourceEntry]) -> str:
    """Return a digest of what the prototypes are built from, in entry order."""
    return _canonical_digest(
        [
            [
                *each.entry,
                text_cell(each.row.get("consumed_composition")),
                read_first_frame(each.row),
                *_table_digests(each),
            ]
            for each in sorted(references, key=lambda each: each.entry)
        ]
    )


def _built_from(each: _SourceEntry, references_digest: str) -> WrittenTable:
    """Return what a table relabeled from *each* records it was built from."""
    return WrittenTable(
        consumed_composition=folded_composition(
            text_cell(each.row.get("consumed_composition")),
            _SOURCES_TERM,
            _canonical_digest([*_table_digests(each), references_digest]),
        ),
        first_frame=read_first_frame(each.row),
    )


def _appearance_dim(each: _SourceEntry) -> int:
    """Read the dimension of *each*'s appearance table, refusing an unreadable one.

    Raises:
        AssignIdentitiesNotFound: The recorded file is missing.
        AssignIdentitiesConflict: The file is not an appearance table.
    """
    try:
        return read_appearance_dim(each.appearance)
    except FileNotFoundError as exc:
        msg = f"{ASSIGN_IDENTITIES_KIND}: {each.entry!r}: {exc}"
        raise AssignIdentitiesNotFound(msg) from exc
    except AppearanceTableError as exc:
        msg = f"{ASSIGN_IDENTITIES_KIND}: {each.entry!r}: {exc}"
        raise AssignIdentitiesConflict(msg) from exc


def _one_dimension(sources: Sequence[_SourceEntry], source: str) -> None:
    """Refuse appearance tables of different dimensions among *sources*.

    Raises:
        AssignIdentitiesNotFound: A recorded appearance table is missing.
        AssignIdentitiesConflict: A table is not an appearance table, or two
            tables differ in dimension.
    """
    first: tuple[Entry, int] | None = None
    for each in sources:
        dim = _appearance_dim(each)
        if first is None:
            first = (each.entry, dim)
        elif dim != first[1]:
            msg = (
                f"{ASSIGN_IDENTITIES_KIND}: the appearance tables of {source} differ "
                f"in dimension: {first[0]!r} has {first[1]} and {each.entry!r} "
                f"has {dim}"
            )
            raise AssignIdentitiesConflict(msg)


def _reference_frames(
    ref: IdentityReference, tracks: pd.DataFrame
) -> npt.NDArray[np.int64]:
    """Return the frames of the rows *ref* names, refusing a reference without one."""
    frames = tracks["frame"].to_numpy(dtype=np.int64)
    if ref.id is None:
        repeated = frames[pd.Series(frames).duplicated().to_numpy()]
        if len(repeated):
            msg = (
                f"{ASSIGN_IDENTITIES_KIND}: the reference {ref.identity!r} names all "
                f"of {ref.entry.entry!r}, which has more than one row in frame "
                f"{int(repeated.min())}"
            )
            raise AssignIdentitiesConflict(msg)
        return frames
    named = frames[tracks["id"].to_numpy(dtype=np.int64) == ref.id]
    if len(named) == 0:
        msg = (
            f"{ASSIGN_IDENTITIES_KIND}: the reference {ref.identity!r} names id "
            f"{ref.id} of {ref.entry.entry!r}, which has no row of that id"
        )
        raise AssignIdentitiesNotFound(msg)
    return named


def _refuse_shared_frames(
    refs: Sequence[IdentityReference], frames: Sequence[npt.NDArray[np.int64]]
) -> None:
    """Refuse two references of one identity in one entry whose rows share a frame."""
    for (first, first_frames), (second, second_frames) in itertools.combinations(
        zip(refs, frames, strict=True), 2
    ):
        if first.identity != second.identity:
            continue
        shared = np.intersect1d(first_frames, second_frames)
        if len(shared):
            msg = (
                f"{ASSIGN_IDENTITIES_KIND}: two references of {first.identity!r} in "
                f"{first.entry.entry!r} (ids {first.id} and {second.id}) share "
                f"frame {int(shared[0])}"
            )
            raise AssignIdentitiesConflict(msg)


def _reference_prototypes(
    references: Sequence[IdentityReference], sources: Mapping[Entry, _SourceEntry]
) -> Prototypes:
    """Build one prototype per identity from the vectors its references name.

    Every reference is checked before a prototype is built.
    """
    vectors: dict[str, list[npt.NDArray[np.float32]]] = {}
    for entry, refs in itertools.groupby(
        sorted(references, key=lambda ref: ref.entry.entry),
        key=lambda ref: ref.entry.entry,
    ):
        entry_refs = list(refs)
        source = sources[entry]
        tracks = read_parquet_table_columns(source.tracks, ["frame", "id"])
        _refuse_shared_frames(
            entry_refs, [_reference_frames(ref, tracks) for ref in entry_refs]
        )
        try:
            table = read_appearance_file(source.appearance, what=repr(entry))
        except FileNotFoundError as exc:
            msg = f"{ASSIGN_IDENTITIES_KIND}: {exc}"
            raise AssignIdentitiesNotFound(msg) from exc
        except AppearanceTableError as exc:
            msg = f"{ASSIGN_IDENTITIES_KIND}: {entry!r}: {exc}"
            raise AssignIdentitiesConflict(msg) from exc
        for ref in entry_refs:
            named = (
                table.vectors if ref.id is None else table.vectors[table.id == ref.id]
            )
            if len(named) == 0:
                msg = (
                    f"{ASSIGN_IDENTITIES_KIND}: the reference {ref.identity!r} at "
                    f"{entry!r} id {ref.id} has no appearance vector"
                )
                raise AssignIdentitiesNotFound(msg)
            vectors.setdefault(ref.identity, []).append(named)
    return prototypes(
        {identity: np.concatenate(parts) for identity, parts in vectors.items()}
    )


def _relabel_entry(
    tracks: pd.DataFrame,
    appearance: AppearanceTable,
    found: Prototypes,
    references: Sequence[IdentityReference],
    params: AssignIdentitiesParams,
) -> pd.DataFrame:
    """Return *tracks* with each tracklet's assigned identity.

    Each row keeps its columns. ``id`` becomes the identity's position in
    ``found.names``, or a distinct id counting down from -2 per unassigned
    tracklet in order of ``(first frame, source id)``. ``identity`` is the
    identity name or null, ``identity_score`` and ``identity_margin`` the
    tracklet's best score and margin, and ``identity_source_id`` the source id.

    Args:
        tracks: The tracks table of one entry.
        appearance: The appearance table of the same entry.
        found: The identity prototypes.
        references: The references that name this entry. Their tracklets take
            the declared identity.
        params: The thresholds and the gap that splits a track.

    Returns:
        The relabeled table, row for row.
    """
    frame = tracks["frame"].to_numpy(dtype=np.int64)
    source_id = tracks["id"].to_numpy(dtype=np.int64)
    pieces = tracklets(frame, source_id, max_gap=params.max_gap)
    keyed = pd.MultiIndex.from_arrays([appearance.frame, appearance.id])
    vector_row = keyed.get_indexer(pd.MultiIndex.from_arrays([frame, source_id]))
    scores = np.full((len(pieces), len(found.names)), np.nan)
    for position, piece in enumerate(pieces):
        rows = vector_row[piece.rows]
        rows = rows[rows >= 0]
        if len(rows):
            scores[position] = score(appearance.vectors[rows], found)
    declared = {
        position: found.names.index(ref.identity)
        for position, piece in enumerate(pieces)
        for ref in references
        if ref.id is None or ref.id == piece.source_id
    }
    outcomes = assign(
        pieces,
        scores,
        min_similarity=params.min_similarity,
        min_margin=params.min_margin,
        declared=declared,
    )
    abstained = sorted(
        (outcome.tracklet for outcome in outcomes if outcome.identity is None),
        key=lambda piece: (piece.first_frame, piece.source_id),
    )
    unassigned_id = {
        (piece.source_id, piece.first_frame): _FIRST_UNASSIGNED_ID - position
        for position, piece in enumerate(abstained)
    }
    new_id = np.empty(len(tracks), dtype=np.int64)
    name: list[str | None] = [None] * len(tracks)
    best = np.empty(len(tracks), dtype=np.float64)
    margin = np.empty(len(tracks), dtype=np.float64)
    for outcome in outcomes:
        piece = outcome.tracklet
        if outcome.identity is None:
            new_id[piece.rows] = unassigned_id[(piece.source_id, piece.first_frame)]
        else:
            new_id[piece.rows] = outcome.identity
            for row in piece.rows.tolist():
                name[row] = found.names[outcome.identity]
        best[piece.rows] = outcome.score
        margin[piece.rows] = outcome.margin
    relabeled = tracks.copy()
    relabeled["id"] = new_id
    relabeled["identity"] = pd.array(name, dtype="string")
    relabeled["identity_score"] = best
    relabeled["identity_margin"] = margin
    relabeled["identity_source_id"] = source_id
    return relabeled


def assign_identities_identity(params: AssignIdentitiesParams) -> OpIdentity:
    """Name a run and its tracks variant by *params* and the source they name."""
    upstream = params.source_tracks_run_id
    return OpIdentity(
        run_id=op_run_id(
            ASSIGN_IDENTITIES_KIND,
            ASSIGN_IDENTITIES_VERSION,
            {"params": params.identity_dump(), "upstream": upstream},
        ),
        tracks_variant=tracks_run_id(
            ASSIGN_IDENTITIES_KIND,
            ASSIGN_IDENTITIES_VERSION,
            chained_variant_payload(params.identity_dump()),
            upstream=upstream,
        ),
    )


@register_op
class AssignIdentitiesOp(Op[AssignIdentitiesParams]):
    """Relabel a tracks variant with the identities its references name."""

    kind = ASSIGN_IDENTITIES_KIND
    category = "convert"
    domain = "tracking"
    version = ASSIGN_IDENTITIES_VERSION
    writes_tracks = True
    scope_takes = "any"
    scope_dependent = False
    Params = AssignIdentitiesParams

    def plan_identity(
        self,
        ds: Dataset,
        params: AssignIdentitiesParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """Name the run and its variant without reading the dataset."""
        _ = (ds, scope, require_data)
        return assign_identities_identity(params)

    def run(
        self,
        ds: Dataset,
        params: AssignIdentitiesParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        """Relabel the source variant's rows in *scope* and return the run id.

        Every refusal is raised before a table is written. The prototypes are
        built from every reference, whatever the scope. A failed entry is
        recorded and the run continues, then raises naming the failed entries.
        A re-run skips a table already written from the same source tables,
        composition and first frame, and writes it again when any of them moved.
        A run that skips every table reports a cache hit.

        Raises:
            AssignIdentitiesNotFound: The source has no row in the scope, or a
                reference names an entry, a track or a vector that is absent.
            AssignIdentitiesConflict: An entry records no appearance table, the
                tables differ in dimension, or the references conflict.
            AllEntriesFailed: Every scoped entry failed.
            RuntimeError: Some scoped entries failed.
        """
        identity = self.plan_identity(ds, params, scope)
        ctx.set_run_id(identity.run_id)
        source = params.source_tracks_run_id
        variant_rows = select_variant_rows(read_tracks_index(ds), source)
        rows = rows_in_scope(variant_rows, scope.op_entries)
        ctx.tracks_variant([source])
        written = 0
        reused = 0
        failed: list[str] = []
        try:
            if rows.empty:
                msg = (
                    f"{self.kind}: the tracks variant {source} has no row in this "
                    "scope. 'mosaic inventory' lists the tracks variants of each "
                    "entry."
                )
                raise AssignIdentitiesNotFound(msg)
            by_entry = {
                (text_cell(row.get("group")), text_cell(row.get("sequence"))): row
                for _, row in variant_rows.iterrows()
            }
            scoped = [_source_entry(ds, row, source) for _, row in rows.iterrows()]
            referenced = sorted({ref.entry.entry for ref in params.references})
            absent = [entry for entry in referenced if entry not in by_entry]
            if absent:
                msg = (
                    f"{self.kind}: the references name {absent}, which have no row "
                    f"of {source}"
                )
                raise AssignIdentitiesNotFound(msg)
            sources = {each.entry: each for each in scoped}
            for entry in referenced:
                if entry not in sources:
                    sources[entry] = _source_entry(ds, by_entry[entry], source)
            _one_dimension([sources[entry] for entry in sorted(sources)], source)
            # Before the tables are read, so a table replaced in between is
            # recomputed by the next run.
            references_digest = _references_digest(
                [sources[entry] for entry in referenced]
            )
            found = _reference_prototypes(params.references, sources)

            ctx.set_total(len(scoped))
            tracks_root = ds.get_root("tracks")
            variant = identity.tracks_variant
            _ = write_tracks_variant(
                tracks_root, variant, self.kind, self.version, params.identity_dump()
            )
            variant_root = tracks_variant_root(tracks_root, variant)
            recorded = written_tables(ds, variant)
            for done, each in enumerate(scoped, start=1):
                ctx.check_cancel()
                group, sequence = each.entry
                key = make_entry_key(group, sequence)
                out_path = variant_root / f"{key}.parquet"
                try:
                    built_from = _built_from(each, references_digest)
                    if (
                        out_path.exists()
                        and not overwrite
                        and recorded.get(key) == built_from
                    ):
                        reused += 1
                    else:
                        self._write_entry(
                            ds,
                            each,
                            out_path,
                            found,
                            params,
                            built_from=built_from,
                            variant=variant,
                            producer_run_id=identity.run_id,
                        )
                    written += 1
                except Exception as exc:
                    ctx.entry_failed(key, exc)
                    failed.append(key)
                ctx.heartbeat(done)
        finally:
            ctx.entries_written(written)

        if reused == len(rows):
            ctx.cache_hit()
        if failed and len(failed) == len(rows):
            raise AllEntriesFailed(
                incomplete_variant_message(
                    self.kind, identity.tracks_variant, failed, len(rows)
                )
            )
        refuse_incomplete_variant(self.kind, identity.tracks_variant, failed, len(rows))
        return identity.run_id

    def _write_entry(
        self,
        ds: Dataset,
        each: _SourceEntry,
        out_path: Path,
        found: Prototypes,
        params: AssignIdentitiesParams,
        *,
        built_from: WrittenTable,
        variant: str,
        producer_run_id: str,
    ) -> None:
        """Relabel one entry's table, write it to *out_path* and record its row."""
        group, sequence = each.entry
        who = f"{self.kind} {group}/{sequence}"
        relabeled = _relabel_entry(
            read_parquet_table(each.tracks),
            read_appearance_file(each.appearance, what=repr(each.entry)),
            found,
            [ref for ref in params.references if ref.entry.entry == each.entry],
            params,
        )
        refuse_repeated_rows(relabeled, who)
        std_format = text_cell(each.row.get("std_format")) or "trex_v1"
        _ = ensure_track_schema(relabeled, std_format, strict=False, source=who)
        _ = write_parquet_atomic(relabeled, out_path)
        write_tracks_row(
            ds,
            run_id=variant,
            group=group,
            sequence=sequence,
            out_path=out_path,
            producer=self.kind,
            std_format=std_format,
            n_rows=len(relabeled),
            producer_run_id=producer_run_id,
            source=each.tracks,
            consumed_source_roots=("tracks",),
            records_media=False,
            media_frames=read_media_frames(each.row),
            frames_read=read_frames_read(each.row),
            known_tail_loss=read_known_tail_loss(each.row),
            first_frame=built_from.first_frame,
            consumed_composition=built_from.consumed_composition,
        )
