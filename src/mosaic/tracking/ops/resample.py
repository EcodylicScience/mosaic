"""``resample-tracks``: re-grid a tracks variant onto a uniform frame rate.

A dataset whose recordings disagree on frame rate makes every constant expressed
in frames mean a different duration in each of them -- a response lag, a
smoothing window, a minimum event duration, and a threshold stated in cm/s that
is divided by the rate to get cm/frame. Placing the tables on one grid makes all
of those correct by construction and leaves each of them a single number again.
The arithmetic lives in :mod:`mosaic.core.pipeline.resample_tracks`; this is the
op that addresses a run of it.

**It chains.** Its input is another tracks variant and its output is a new one,
so the source's identity is an ``upstream`` term in both identifiers -- the term
:func:`~mosaic.core.pipeline.tracks_identity.tracks_run_id` has always carried
for a producer that chains, and this is its first caller. That is what makes one
resampling recipe recognisable as the same recipe over two different sources,
and what stops a re-conversion upstream from silently reusing this variant.

**It reads a table and never opens video**, which is why its rows are written
with ``records_media=False``. Claiming a media composition it never looked at
would put this run into the drift reports on the next re-transcode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import pandas as pd

from mosaic.core.helpers import make_entry_key, text_cell
from mosaic.core.pipeline.entry_pool import (
    EntryFailure,
    EntryPool,
    OrderedResults,
    dataset_ref,
    worker_dataset,
)
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.staged_tables import (
    StagedTrackTable,
    TableCommitter,
    stage_table,
    sweep_staged,
)
from mosaic.core.pipeline.op_identity import op_run_id
from mosaic.core.pipeline.ops import IdentityDeferred, Op, OpIdentity, register_op
from mosaic.core.pipeline.tracks_identity import (
    chained_variant_payload,
    tracks_run_id,
    tracks_variant_root,
    write_tracks_variant,
)
from mosaic.core.pipeline.tracks_index import (
    TracksRowDraft,
    read_carries_tracklets,
    read_tracks_index,
)
from mosaic.core.params import (
    Declared,
    Params,
)
from mosaic.tracking.ops.chained_tracks import (
    SOURCE_TRACKS_RUN_ID_DESCRIPTION,
    chained_source_rows,
    one_source_variant,
    refuse_incomplete_variant,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope

__all__ = ["ResampleTracksOp", "ResampleTracksParams", "resample_tracks_run_id"]


def resample_tracks_run_id(
    kind: str, version: str, params: Params, upstream: str
) -> str:
    """Mint a resampling run identifier.

    The source variant is a term because re-gridding two different recipes is two
    different runs even under identical settings -- the same reason it is a term
    in the tracks variant this produces.
    """
    return op_run_id(
        kind, version, {"params": params.identity_dump(), "upstream": upstream}
    )


_TARGET_FPS_DESCRIPTION = (
    "The uniform frame rate every table is re-gridded onto, which is what "
    "makes one constant expressed in frames mean one duration dataset-wide."
)

_PREFILTER_DESCRIPTION = (
    "Displacement from the predecessor above which a native sample is rejected "
    "before interpolating. A rejected mis-detection is dropped instead of "
    "blended into its neighbors. Unset interpolates the samples as they are."
)


class ResampleTracksParams(Params):
    """Parameters for ``resample-tracks``.

    ``Params`` rather than ``TrackerOpParams``: ``convert_to_tracks``,
    ``idle_timeout`` and ``max_runtime`` describe driving an external tool and
    mean nothing to a table rewrite. Each of the three is ``HASH_EXCLUDE``, so
    inheriting them would leave this op's identity where it is -- and they would
    still appear in ``model_dump()``, in the recorded ``params.json`` and in the
    discovery schema a client draws a form from, each one naming a knob this op
    cannot act on.

    Two defaults carry an argument the field's own prose has no room for.
    ``source_tracks_run_id`` unset applies the rule ``select_variant_rows``
    applies everywhere else: a chained producer with two possible upstreams has
    no defensible default, so it refuses rather than picking. ``prefilter``
    unset is the default because the blend attenuates an outlier in the output
    by exactly the factor it attenuates it in a downstream detector -- whatever
    a bad-frame gate then misses contributes less than that gate's own
    threshold.
    """

    target_fps: Annotated[float, Declared(_TARGET_FPS_DESCRIPTION, unit="fps")]
    source_tracks_run_id: Annotated[
        str | None, Declared(SOURCE_TRACKS_RUN_ID_DESCRIPTION)
    ] = None
    prefilter: Annotated[
        float | None, Declared(_PREFILTER_DESCRIPTION, unit="units/s")
    ] = None


@register_op
class ResampleTracksOp(Op[ResampleTracksParams]):
    """Place a dataset's tracks tables on one uniform frame rate."""

    kind = "resample-tracks"
    category = "convert"
    domain = "tracking"
    version = "0.1"
    writes_tracks = True
    scope_takes = "any"
    scope_dependent = True
    parallelizable = True
    Params = ResampleTracksParams

    def target(self, params: ResampleTracksParams, scope: ResolvedScope) -> str:
        return f"{params.target_fps:g}fps"

    def plan_identity(
        self,
        ds: Dataset,
        params: ResampleTracksParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """What this run and the variant it writes will be called.

        Deferred rather than guessed when the source is not on disk: this op's
        identity covers the variant it chains from, and a graph step whose input
        is an earlier step's conversion has nothing to name yet.
        """
        rows = chained_source_rows(
            ds, params.source_tracks_run_id, scope.op_entries, producer=self.kind
        )
        if rows.empty:
            raise IdentityDeferred(
                self.kind,
                "the tracks variant it re-grids has no rows in this scope yet, "
                "and this run's identity carries the variant it chains from",
            )
        upstream = one_source_variant(rows, self.kind)
        return OpIdentity(
            run_id=resample_tracks_run_id(self.kind, self.version, params, upstream),
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
        params: ResampleTracksParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        rows = chained_source_rows(
            ds, params.source_tracks_run_id, scope.op_entries, producer=self.kind
        )
        identity = self.plan_identity(ds, params, scope)
        ctx.set_run_id(identity.run_id)
        ctx.set_total(int(len(rows)))

        variant = identity.tracks_variant
        tracks_root = ds.get_root("tracks")
        _ = write_tracks_variant(
            tracks_root,
            variant,
            self.kind,
            self.version,
            params.identity_dump(),
        )
        variant_root = tracks_variant_root(tracks_root, variant)
        # Files that a run killed before this one left staged are no table of any
        # row, and a worker of this run stages under names of its own.
        _ = sweep_staged(variant_root)
        tracks = read_tracks_index(ds)
        indexed = set(
            zip(
                tracks.loc[tracks["run_id"] == variant, "group"],
                tracks.loc[tracks["run_id"] == variant, "sequence"],
                strict=True,
            )
        )
        committer = TableCommitter(ds, overwrite=overwrite)

        failed: list[str] = []
        done = 0

        def emit(item: ResampledEntry | Exception) -> None:
            nonlocal done
            if isinstance(item, Exception):
                raise item
            if item.failure is not None:
                failed.append(item.failure.key)
                item.failure.record(ctx.entry_failed)
            if item.table is not None:
                for lost in committer.add([item.table]):
                    failed.append(lost.key)
                    lost.record(ctx.entry_failed)
            done += 1
            ctx.heartbeat(done)

        ordered = OrderedResults[ResampledEntry | Exception](emit)

        def on_result(index: int, outcome: ResampledEntry) -> None:
            ordered.put(index, outcome)

        def on_error(index: int, exc: Exception) -> None:
            ordered.put(index, exc)

        pool = EntryPool[int, ResampledEntry].for_settings(
            ctx.parallel,
            on_result=on_result,
            on_error=on_error,
            on_resources=ctx.resources,
        )
        ref = dataset_ref(ds, ctx.parallel.mode)
        try:
            for index, (_, row) in enumerate(rows.iterrows()):
                ctx.check_cancel()
                entry = (
                    text_cell(row.get("group", "")),
                    text_cell(row.get("sequence", "")),
                )
                pool.submit(
                    index,
                    resample_one_entry,
                    ref,
                    row,
                    variant_root,
                    params,
                    variant,
                    identity.run_id,
                    self.kind,
                    overwrite,
                    entry in indexed,
                )
            pool.finish()
            for lost in committer.flush():
                failed.append(lost.key)
                lost.record(ctx.entry_failed)
        except BaseException:
            # A table a worker finished is published with its row.
            pool.settle()
            for item in ordered.held():
                if not isinstance(item, Exception) and item.table is not None:
                    for lost in committer.add([item.table]):
                        lost.record(ctx.entry_failed)
            for lost in committer.flush():
                lost.record(ctx.entry_failed)
            raise
        finally:
            pool.close()

        refuse_incomplete_variant(self.kind, variant, failed, len(rows))
        return identity.run_id


@dataclass(slots=True)
class ResampledEntry:
    """What re-gridding one entry produced, for the process that owns the index.

    Attributes:
        table: The table written under a staging name and the index row that
            describes it. ``None`` for an entry that was already written or that
            failed.
        failure: Why the entry failed, when it did.
    """

    table: StagedTrackTable | None = None
    failure: EntryFailure | None = None


def resample_one_entry(
    ref: Dataset | str,
    row: pd.Series,
    variant_root: Path,
    params: ResampleTracksParams,
    variant: str,
    run_id: str,
    kind: str,
    overwrite: bool,
    indexed: bool,
) -> ResampledEntry:
    """Re-grid one source table and stage it, leaving its publication to the caller.

    The unit of work of :class:`ResampleTracksOp`, run by a worker. *indexed* says
    whether the index already has a row for the entry. A table that exists and has a
    row is kept. A table with no row is made again. A table that cannot be read,
    re-gridded or validated comes back as the outcome's ``failure``.
    """
    from mosaic.core.pipeline.resample_tracks import resample_entry_table
    from mosaic.core.pipeline.writers import read_parquet_table
    from mosaic.core.schema import ensure_track_schema

    ds = worker_dataset(ref)
    group = text_cell(row.get("group", ""))
    sequence = text_cell(row.get("sequence", ""))
    key = make_entry_key(group, sequence)
    out_path = variant_root / f"{key}.parquet"
    outcome = ResampledEntry()
    if out_path.exists() and indexed and not overwrite:
        return outcome
    try:
        source = Path(ds.resolve_path(str(row.get("abs_path", ""))))
        table = read_parquet_table(source)
        resampled = resample_entry_table(
            table,
            params.target_fps,
            prefilter=params.prefilter,
            source=f"{group}/{sequence}",
        )
        std_format = text_cell(row.get("std_format", "")) or "trex_v1"
        # Resampling changes no column and no unit, so the source's own
        # schema is what the output answers to -- read off the row rather
        # than from the tracking-roots table, which this op is
        # deliberately not in.
        _ = ensure_track_schema(
            resampled,
            std_format,
            strict=False,
            source=f"{group}/{sequence} (resampled)",
        )
        staged = stage_table(resampled, out_path)
        outcome.table = StagedTrackTable(
            staged=staged,
            row=TracksRowDraft(
                run_id=variant,
                group=group,
                sequence=sequence,
                out_path=out_path,
                producer=kind,
                std_format=std_format,
                n_rows=int(len(resampled)),
                producer_run_id=run_id,
                source=source,
                consumed_source_roots=("tracks",),
                records_media=False,
                carries_tracklets=read_carries_tracklets(row),
            ),
        )
    except Exception as exc:  # noqa: BLE001 -- recorded by the caller
        outcome.failure = EntryFailure.capture(key, exc)
    return outcome
