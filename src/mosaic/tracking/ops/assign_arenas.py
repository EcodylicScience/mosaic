"""``assign-arenas``: place each row of a tracks variant in its arena.

The op reads a tracks variant and writes a new one, as ``resample-tracks`` does.
Each entry's table gains ``arena``, the arena of its recording's layout
(``Dataset.arena_layout``) that contains the row's body center, and its ids are
renumbered so that each id stays in one arena. The arithmetic is
:func:`~mosaic.core.pipeline.arena_assignment.assign_arenas`.

The variant identifier names the recipe: the op's params and the variant it
chains from. A layout is data the op read, as a source file is to a conversion,
so each row records its layout's assignment digest beside the source row's
composition. The assignment digest leaves out a boundary and the regions of
interest, which the assignment does not read, so adding one recomputes nothing.
A changed region or exclusion recomputes its recording's tables under the same
variant, and the features that read them see a new tracks composition. The op's
run identifier covers the assignment digests of the layouts its scope read.
Frames are unchanged, and each row records its source row's first frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final

import pandas as pd

from mosaic.core.arenas import ArenaKey
from mosaic.core.helpers import make_entry_key, text_cell
from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.arena_index import (
    ArenaIndexError,
    DeclaredLayout,
    IndexedLayout,
    entry_arena_row,
    media_cameras,
    read_arena_index_rows,
    read_declared_layout,
)
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.op_identity import op_run_id
from mosaic.core.pipeline.ops import IdentityDeferred, Op, OpIdentity, register_op
from mosaic.core.pipeline.tracks_identity import (
    chained_variant_payload,
    tracks_run_id,
    tracks_variant_root,
    write_tracks_variant,
)
from mosaic.core.pipeline.tracks_index import (
    read_carries_tracklets,
    read_first_frame,
    write_tracks_row,
)
from mosaic.tracking.ops.chained_tracks import (
    SOURCE_TRACKS_RUN_ID_DESCRIPTION,
    chained_source_rows,
    folded_composition,
    one_source_variant,
    refuse_incomplete_variant,
    written_tables,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope

__all__ = [
    "AssignArenasOp",
    "AssignArenasParams",
    "assign_arenas_run_id",
]

_ARENAS_TERM: Final = "arenas"


class AssignArenasParams(Params):
    """Parameters for ``assign-arenas``.

    The layouts are not params. They are declared on the dataset per recording
    (``Dataset.set_arena_layout``) and recorded on each row the op writes.
    """

    source_tracks_run_id: Annotated[
        str | None, Declared(SOURCE_TRACKS_RUN_ID_DESCRIPTION)
    ] = None


def assign_arenas_run_id(
    kind: str,
    version: str,
    params: Params,
    upstream: str,
    layouts: Mapping[str, str],
) -> str:
    """Mint an arena assignment's run identifier.

    Args:
        kind: The op's kind.
        version: The op's version.
        params: The run's params.
        upstream: The tracks variant it reads.
        layouts: The assignment digest of each entry's layout, by entry key.
    """
    payload = {
        "params": params.identity_dump(),
        "upstream": upstream,
        "layouts": dict(sorted(layouts.items())),
    }
    return op_run_id(kind, version, payload)


@dataclass(frozen=True, slots=True)
class _EntryLayout:
    """The layout one entry assigns by, and the digest the op records for it.

    Attributes:
        declared: The layout the entry's index row names.
        digest: Its assignment digest, which a boundary or a region of interest
            does not move.
    """

    declared: DeclaredLayout
    digest: str


def _layout_digests(
    ds: Dataset, rows: pd.DataFrame
) -> tuple[dict[str, _EntryLayout | ValueError | None], dict[str, str]]:
    """Return each source row's layout and the digests found, by entry key.

    Each entry's index row is looked up in the rows of one read, and each
    layout file the entries name is read once. An entry whose cameras find
    different layouts, or whose file fails its check, maps to the refusal, which
    the run records as that entry's failure. A file no entry needs is not read.

    Raises:
        ArenaIndexError: If the index cannot be read or declares a key twice.
    """
    index_rows = read_arena_index_rows(ds)
    cameras = media_cameras(ds)
    read: dict[ArenaKey, _EntryLayout | ValueError] = {}
    layouts: dict[str, _EntryLayout | ValueError | None] = {}
    for _, row in rows.iterrows():
        entry = (text_cell(row.get("group", "")), text_cell(row.get("sequence", "")))
        found: _EntryLayout | ValueError | None
        try:
            index_row = entry_arena_row(
                ds, entry, cameras.get(entry, set()), index_rows
            )
            found = None
            if index_row is not None:
                if index_row.key not in read:
                    read[index_row.key] = _read_entry_layout(ds, index_row)
                found = read[index_row.key]
        except ValueError as exc:
            found = exc
        layouts[make_entry_key(*entry)] = found
    digests = {
        key: layout.digest
        for key, layout in layouts.items()
        if isinstance(layout, _EntryLayout)
    }
    return layouts, digests


def _read_entry_layout(ds: Dataset, row: IndexedLayout) -> _EntryLayout | ValueError:
    """Read *row*'s file, or return the refusal of a file that fails its check."""
    try:
        declared = read_declared_layout(ds, row)
    except ArenaIndexError as exc:
        return exc
    return _EntryLayout(declared, declared.layout.assignment_digest())


@register_op
class AssignArenasOp(Op[AssignArenasParams]):
    """Name each tracks row's arena and keep each id in one arena."""

    kind = "assign-arenas"
    category = "convert"
    domain = "tracking"
    version = "0.1"
    writes_tracks = True
    scope_takes = "any"
    scope_dependent = True
    Params = AssignArenasParams

    def plan_identity(
        self,
        ds: Dataset,
        params: AssignArenasParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """Name this run and the variant it writes.

        Deferred when the source is not on disk, because the identity covers
        the variant it chains from.
        """
        rows = chained_source_rows(
            ds, params.source_tracks_run_id, scope.op_entries, producer=self.kind
        )
        if rows.empty:
            raise IdentityDeferred(
                self.kind,
                "the tracks variant it assigns has no rows in this scope yet, "
                "and this run's identity carries the variant it chains from",
            )
        upstream = one_source_variant(rows, self.kind)
        _, digests = _layout_digests(ds, rows)
        return OpIdentity(
            run_id=assign_arenas_run_id(
                self.kind, self.version, params, upstream, digests
            ),
            tracks_variant=tracks_run_id(
                self.kind,
                self.version,
                chained_variant_payload(params.identity_dump()),
                upstream=upstream,
            ),
        )

    def run(
        self,
        ds: Dataset,
        params: AssignArenasParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        from mosaic.core.pipeline.arena_assignment import assign_arenas
        from mosaic.core.pipeline.writers import (
            read_parquet_table,
            write_parquet_atomic,
        )
        from mosaic.core.schema import ensure_track_schema

        rows = chained_source_rows(
            ds, params.source_tracks_run_id, scope.op_entries, producer=self.kind
        )
        identity = self.plan_identity(ds, params, scope)
        ctx.set_run_id(identity.run_id)
        ctx.set_total(int(len(rows)))
        layouts, _ = _layout_digests(ds, rows)

        variant = identity.tracks_variant
        tracks_root = ds.get_root("tracks")
        _ = write_tracks_variant(
            tracks_root, variant, self.kind, self.version, params.identity_dump()
        )
        variant_root = tracks_variant_root(tracks_root, variant)
        written = written_tables(ds, variant)

        failed: list[str] = []
        for done, (_, row) in enumerate(rows.iterrows(), start=1):
            ctx.check_cancel()
            group = text_cell(row.get("group", ""))
            sequence = text_cell(row.get("sequence", ""))
            key = make_entry_key(group, sequence)
            out_path = variant_root / f"{key}.parquet"
            try:
                layout = layouts[key]
                if isinstance(layout, ValueError):
                    raise layout
                if layout is None:
                    message = (
                        f"no arena layout is declared for the recording of "
                        f"{(group, sequence)!r}. Declare one with "
                        "Dataset.set_arena_layout or 'mosaic arenas set'."
                    )
                    raise LookupError(message)
                composition = folded_composition(
                    text_cell(row.get("consumed_composition", "")),
                    _ARENAS_TERM,
                    layout.digest,
                )
                recorded = written.get(key)
                if (
                    out_path.exists()
                    and not overwrite
                    and recorded is not None
                    and recorded.consumed_composition == composition
                ):
                    ctx.heartbeat(done)
                    continue
                source = Path(ds.resolve_path(text_cell(row.get("abs_path", ""))))
                table, report = assign_arenas(
                    read_parquet_table(source), layout.declared.layout
                )
                std_format = text_cell(row.get("std_format", "")) or "trex_v1"
                _ = ensure_track_schema(
                    table, std_format, strict=False, source=f"{group}/{sequence}"
                )
                _ = write_parquet_atomic(table, out_path)
                write_tracks_row(
                    ds,
                    run_id=variant,
                    group=group,
                    sequence=sequence,
                    out_path=out_path,
                    producer=self.kind,
                    std_format=std_format,
                    n_rows=len(table),
                    producer_run_id=identity.run_id,
                    source=source,
                    consumed_source_roots=("tracks",),
                    records_media=False,
                    first_frame=read_first_frame(row),
                    consumed_composition=composition,
                    carries_tracklets=read_carries_tracklets(row),
                )
                if any(report.dropped.values()):
                    ctx.rows_dropped(
                        key, {str(reason): n for reason, n in report.dropped.items()}
                    )
            except Exception as exc:  # recorded per entry, then refused below
                ctx.entry_failed(key, exc)
                failed.append(key)
            ctx.heartbeat(done)

        refuse_incomplete_variant(self.kind, variant, failed, len(rows))
        return identity.run_id
