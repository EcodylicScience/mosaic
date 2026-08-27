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

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import pandas as pd

from mosaic.core.helpers import make_entry_key, text_cell
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.op_identity import op_run_id
from mosaic.core.pipeline.ops import IdentityDeferred, Op, OpIdentity, register_op
from mosaic.core.pipeline.tracks_identity import (
    resample_variant_payload,
    tracks_run_id,
    tracks_variant_root,
    write_tracks_variant,
)
from mosaic.core.pipeline.tracks_index import (
    read_tracks_index,
    select_variant_rows,
    write_tracks_row,
)
from mosaic.core.pipeline.types import Declared, OpParams, Params

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

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

_SOURCE_TRACKS_RUN_ID_DESCRIPTION = (
    "Which tracks variant to read, e.g. 'trex.0.1-abc123def0'. Unset resolves "
    "whichever variant the scope's entries hold, and refuses when they hold two."
)

_PREFILTER_DESCRIPTION = (
    "Reject a native sample whose displacement from its predecessor exceeds "
    "this many table units per second before interpolating, so a mis-detection "
    "is dropped instead of blended into its neighbors. Unset interpolates the "
    "samples as they are."
)


class ResampleTracksParams(OpParams):
    """Parameters for ``resample-tracks``.

    ``OpParams`` rather than ``TrackerOpParams``: ``convert_to_tracks``,
    ``idle_timeout`` and ``max_runtime`` describe driving an external tool and
    mean nothing to a table rewrite. Each of the three is ``HASH_EXCLUDE``, so
    inheriting them would leave this op's identity where it is -- and they would
    still appear in ``model_dump()``, in the recorded ``params.json`` and in the
    discovery schema a client draws a form from, each one naming a knob this op
    cannot act on.

    Attributes:
        target_fps: The rate to place every table on.
        source_tracks_run_id: Which variant to read. ``None`` resolves whichever
            variant the scope's entries carry, and refuses when they carry two --
            the same rule ``select_variant_rows`` applies everywhere else, because
            a chained producer with two possible upstreams has no defensible
            default.
        prefilter: Reject native samples whose displacement from their
            predecessor exceeds this many table units per second, before
            interpolating -- so a mis-detection is dropped rather than blended
            into its neighbours. ``None`` interpolates the samples as they are,
            which is the default because the blend attenuates an outlier in the
            output by exactly the factor it attenuates it in a downstream
            detector: whatever a bad-frame gate then misses contributes less than
            that gate's own threshold.
    """

    target_fps: Annotated[float, Declared(_TARGET_FPS_DESCRIPTION, unit="fps")]
    source_tracks_run_id: Annotated[
        str | None, Declared(_SOURCE_TRACKS_RUN_ID_DESCRIPTION)
    ] = None
    prefilter: Annotated[
        float | None, Declared(_PREFILTER_DESCRIPTION, unit="units/s")
    ] = None


def _in_scope(row: pd.Series, params: ResampleTracksParams) -> bool:
    """Whether one tracks-index row is one this run covers."""
    if params.entries is None:
        return True
    group = text_cell(row.get("group", ""))
    sequence = text_cell(row.get("sequence", ""))
    return (group, sequence) in set(params.entries)


def _source_rows(
    ds: Dataset, params: ResampleTracksParams, *, producer: str
) -> pd.DataFrame:
    """The source tables this run reads, one row per entry.

    Rows this op produced are excluded before the variant is resolved, unless the
    caller named one explicitly. Two reasons, and the second is the one with
    teeth:

    - A resampler does not read its own output. Once a run has written its
      variant, an unpinned re-run would see two variants per entry and refuse --
      so the op would be runnable exactly once per dataset, and a resumed run
      after a partial failure could never resolve its own source.
    - Excluding by *producer* rather than by the variant this run would mint also
      covers a second resampling at a different target rate, which is a different
      variant and equally not an input here.

    Chaining one resample onto another is still expressible; it just has to be
    said out loud, with ``source_tracks_run_id=``.
    """
    index = read_tracks_index(ds)
    if index.empty:
        return index
    if params.source_tracks_run_id is None and "producer" in index.columns:
        index = index[
            [text_cell(value) != producer for value in index["producer"]]
        ].reset_index(drop=True)
    resolved = select_variant_rows(index, params.source_tracks_run_id)
    if resolved.empty:
        return resolved
    keep = [
        i for i, (_, row) in enumerate(resolved.iterrows()) if _in_scope(row, params)
    ]
    return resolved.iloc[keep].reset_index(drop=True)


def _one_source_variant(rows: pd.DataFrame) -> str:
    """The single variant *rows* were produced by.

    Refused rather than resolved per entry. A chained producer's identity carries
    one ``upstream``, so a scope spanning two would mint one identifier for two
    different recipes -- the mixed-variant dataset that is legal everywhere else
    is exactly what cannot be re-gridded under one name.
    """
    variants = sorted({text_cell(value) for value in rows["run_id"]})
    if len(variants) > 1:
        raise ValueError(
            f"resample-tracks reads one tracks variant and this scope resolves "
            f"{len(variants)} ({variants}). Its identity carries the variant it "
            "chains from, so re-gridding two under one name would claim they were "
            "one recipe. Name one with source_tracks_run_id=, or narrow the scope."
        )
    return variants[0]


@register_op
class ResampleTracksOp(Op[ResampleTracksParams]):
    """Place a dataset's tracks tables on one uniform frame rate."""

    kind = "resample-tracks"
    category = "convert"
    domain = "tracking"
    version = "0.1"
    writes_tracks = True
    Params = ResampleTracksParams

    def target(self, params: ResampleTracksParams) -> str:
        return f"{params.target_fps:g}fps"

    def plan_identity(self, ds: Dataset, params: ResampleTracksParams) -> OpIdentity:
        """What this run and the variant it writes will be called.

        Deferred rather than guessed when the source is not on disk: this op's
        identity covers the variant it chains from, and a graph step whose input
        is an earlier step's conversion has nothing to name yet.
        """
        rows = _source_rows(ds, params, producer=self.kind)
        if rows.empty:
            raise IdentityDeferred(
                self.kind,
                "the tracks variant it re-grids has no rows in this scope yet, "
                "and this run's identity carries the variant it chains from",
            )
        upstream = _one_source_variant(rows)
        return OpIdentity(
            run_id=resample_tracks_run_id(self.kind, self.version, params, upstream),
            tracks_variant=tracks_run_id(
                self.kind,
                self.version,
                resample_variant_payload(params.identity_dump()),
                upstream=upstream,
            ),
        )

    def run(self, ds: Dataset, params: ResampleTracksParams, ctx: JobContext) -> str:
        from mosaic.core.pipeline.resample_tracks import resample_entry_table
        from mosaic.core.pipeline.writers import (
            read_parquet_table,
            write_parquet_atomic,
        )
        from mosaic.core.schema import ensure_track_schema

        rows = _source_rows(ds, params, producer=self.kind)
        identity = self.plan_identity(ds, params)
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

        failed: list[str] = []
        for done, (_, row) in enumerate(rows.iterrows(), start=1):
            ctx.check_cancel()
            group = text_cell(row.get("group", ""))
            sequence = text_cell(row.get("sequence", ""))
            key = make_entry_key(group, sequence)
            out_path = variant_root / f"{key}.parquet"
            if out_path.exists() and not params.overwrite:
                ctx.heartbeat(done)
                continue
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
                _ = write_parquet_atomic(resampled, out_path)
                write_tracks_row(
                    ds,
                    run_id=variant,
                    group=group,
                    sequence=sequence,
                    out_path=out_path,
                    producer=self.kind,
                    std_format=std_format,
                    n_rows=int(len(resampled)),
                    producer_run_id=identity.run_id,
                    source=source,
                    consumed_source_roots=("tracks",),
                    records_media=False,
                )
            except Exception as exc:  # noqa: BLE001 -- recorded, then re-raised below
                ctx.entry_failed(key, exc)
                failed.append(key)
            ctx.heartbeat(done)

        if failed:
            # A partial variant is not a safe resting state here, the way a
            # partial feature run is. The point of this variant is to *replace*
            # the source as what every entry resolves to, so an entry missing from
            # it disappears the moment the source is retired -- silently, and from
            # the middle of a dataset. Re-running resumes: what was written is
            # skipped by the presence check above.
            raise RuntimeError(
                f"resample-tracks left {len(failed)} of {len(rows)} entries "
                f"unwritten ({failed[:5]}{'...' if len(failed) > 5 else ''}). "
                "Variant "
                f"{variant} is incomplete; fix the named entries and re-run to "
                "resume."
            )
        return identity.run_id
