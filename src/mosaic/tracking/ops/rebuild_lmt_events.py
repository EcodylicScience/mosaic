"""Rebuild the events of Live Mouse Tracker databases on copies of them.

lmt-analysis rebuilds a database's ``EVENT`` table in place, and a rebuild can
rewrite ``DETECTION`` as well. A raw file is only ever read, so ``rebuild-lmt-events``
copies each ``lmt_sqlite`` file of its scope into the dataset and runs the
rebuild on the copy. The program runs in the environment that
:mod:`mosaic.tracking.common.lmt_env` locates, and mosaic never imports it.
``convert-tracks`` and ``convert-labels`` read a copy through their
``rebuild_run_id`` param.

**Layout.** One run is one directory under the ``rebuild-lmt-events`` root.
:mod:`mosaic.core.track_library.lmt_rebuild` draws the tree. The root's ``index.csv``
holds one row per ``(run_id, entry)``, written last, so a row names only a finished
directory. The sweeper reclaims rowed directories alone.

**The run id** is ``rebuild-lmt-events.<version>-<digest>`` over the lmt-analysis
commit, the builders and the sorted md5 of each covered source
(:func:`rebuild_lmt_events_run_id`). The commit is 40 lowercase hex digits, because
an abbreviated or uppercase spelling of one commit would mint a second id for it.
The entry names, the paths, the location of the checkout and the environment's
Python and package versions stay out. The manifest records the versions, and the
committed lock of the environment pins them.

**A finished entry** is skipped. An entry whose ``rebuild`` marker is present, whose
manifest names the run's commit, the builders the probe resolved and the source's md5,
and whose copy still has the size and modification time the manifest recorded, is
current: it is not copied, and its index row is appended when the index lacks one.
Any mismatch redoes the entry, and ``overwrite`` redoes every entry. A run whose every
entry was current records a cache hit and runs no rebuild. The run's
``entries_written`` counts the entries that hold a finished copy and row, rebuilt or
current, so a resumed run and a fresh one report the same coverage.

**The entry steps**, under the entry's claim, which a timer and every line of the
runner's output keep alive:

1. drop the entry's index row, so that a row always names a finished copy, clear what
   the phase owns, and refuse a ``-journal`` or ``-wal`` beside the source;
2. check that the disk holds twice the source's size;
3. copy the source to ``.incoming-<execution_id>-<name>`` in 64 MiB chunks while
   hashing it, and refuse a file whose md5 is not the index's;
4. run the runner's ``rebuild`` on the temporary file;
5. refuse a journal beside the temporary file, rename it to the source's name, write
   the manifest and the ``rebuild`` marker, and append the index row.

**The failure path.** An entry that fails is reported through
``ctx.entry_failed_unless_cancelled`` and the run goes on. Its ``finally`` removes
the temporary file, a published copy that has no marker, and the journals beside
either, and keeps ``rebuild.log`` and the report, which say why the entry failed.
Then the claim is released and a directory that holds nothing is removed. A run
that attempted entries and lost all of them raises ``AllEntriesFailed``. A runner
that broke its contract (:class:`~mosaic.tracking.common.lmt_env.LmtRunnerDefect`)
fails the run, and so does a cancel.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, override

from pydantic import field_validator

from mosaic.core.entry import Entry
from mosaic.core.helpers import make_entry_key
from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.dataset_indexes import register_reconcilable_index
from mosaic.core.pipeline.entry_claim import (
    discard_if_empty,
    kept_claimed,
    open_entry,
    phase_activity,
    release_entry,
)
from mosaic.core.pipeline.identity_scheme import write_identity_scheme
from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.markers import (
    InflightMarker,
    PhaseMarker,
    clear_phase_marker,
    read_phase_marker,
    write_phase_marker,
)
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME, op_run_id
from mosaic.core.pipeline.ops import Op, OpIdentity, register_op
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.scope import Scope
from mosaic.core.track_library.lmt import journals_beside
from mosaic.core.track_library.lmt_rebuild import (
    REBUILD_KIND,
    RebuildManifest,
    copy_unchanged,
    read_rebuild_manifest,
)
from mosaic.runlog import now_iso
from mosaic.tracking.common.entry import clear_outputs
from mosaic.tracking.common.lmt_env import (
    LmtRebuildError,
    LmtRunnerReport,
    probe_lmt_analysis,
    run_lmt_rebuild,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope

__all__ = [
    "REBUILD_IDLE_SECONDS",
    "REBUILD_OP_VERSION",
    "LmtRebuildIndexRow",
    "RebuildLmtEventsOp",
    "RebuildLmtEventsParams",
    "rebuild_index",
    "rebuild_lmt_events_run_id",
]

REBUILD_OP_VERSION: Final = "0.1"
"""The op's version, a visible segment of its run id."""

REBUILD_IDLE_SECONDS: Final = 3600.0
"""How long an entry's claim lasts without a refresh.

A timer refreshes the claim every :data:`~mosaic.core.pipeline.entry_claim.INFLIGHT_REFRESH_SECONDS`
while the runner is silent, so the window bounds only how long the claim of a killed
process holds the entry before the next run may take it.
"""

_COPY_CHUNK_BYTES: Final = 64 * 1024 * 1024
"""The size of one read of a source while it is copied and hashed."""

_SOURCE_FORMAT: Final = "lmt_sqlite"
_INCOMING_PREFIX: Final = ".incoming-"
_LOG_NAME: Final = "rebuild.log"
_REPORT_NAME: Final = "rebuild-report.json"
_COMMIT_LENGTH: Final = 40
_HEX_DIGITS: Final = frozenset("0123456789abcdef")


class RebuildLmtEventsParams(Params):
    """Settings of ``rebuild-lmt-events``."""

    lmt_analysis_commit: Annotated[
        str,
        Declared(
            "The commit of the lmt-analysis checkout that rebuilds the events, as "
            "40 lowercase hex digits. The checkout has to be at this commit."
        ),
    ]
    builders: Annotated[
        list[str] | None,
        Declared(
            "The lmtanalysis modules that rebuild events, in order, such as "
            "BuildEventStop. A builder reads the events of those before it. Unset "
            "runs the builders of the checkout's rebuild script. A partial list "
            "rebuilds only the listed builders' events, and the copy keeps the "
            "source's other events."
        ),
    ] = None

    @field_validator("lmt_analysis_commit", check_fields=False)
    @classmethod
    def _full_lowercase_hash(cls, value: str) -> str:
        """Refuse a commit that is not 40 lowercase hex digits."""
        if len(value) != _COMMIT_LENGTH or not set(value) <= _HEX_DIGITS:
            raise ValueError(
                f"lmt_analysis_commit {value!r} is not 40 lowercase hex digits. "
                "An abbreviated or uppercase spelling of one commit would mint a "
                "second run id for it; give the full hash that 'git rev-parse HEAD' "
                "prints."
            )
        return value

    @field_validator("builders", check_fields=False)
    @classmethod
    def _named_builders(cls, value: list[str] | None) -> list[str] | None:
        """Refuse an empty list, a repeated name and a name that is not an identifier."""
        if value is None:
            return None
        if not value:
            raise ValueError(
                "builders is empty. Name at least one builder, or leave it unset "
                "to run the builders of the checkout's rebuild script."
            )
        repeated = sorted({name for name in value if value.count(name) > 1})
        if repeated:
            raise ValueError(
                f"builders repeats {', '.join(repeated)}. Each builder runs once."
            )
        invalid = [name for name in value if not name.isidentifier()]
        if invalid:
            raise ValueError(
                f"builders names {', '.join(repr(name) for name in invalid)}, which "
                "is not the name of an lmtanalysis module."
            )
        return value


def rebuild_lmt_events_run_id(
    params: RebuildLmtEventsParams, source_md5s: Collection[str]
) -> str:
    """The run id of a rebuild of the sources with these md5s.

    The md5s are sorted, so the id does not depend on the order of the index.
    """
    return op_run_id(
        REBUILD_KIND,
        REBUILD_OP_VERSION,
        {"params": params.identity_dump(), "sources": sorted(source_md5s)},
    )


@dataclass(frozen=True, slots=True)
class LmtRebuildIndexRow(RunIndexRowBase):
    """One rebuilt copy.

    ``abs_path`` is the entry directory, because the sweeper keys a row on the
    directory's name.
    """

    group: str
    sequence: str
    source_md5: str


def rebuild_index(path: Path) -> IndexCSV[LmtRebuildIndexRow]:
    """The index of the rebuilt copies, one row per ``(run_id, entry)``."""
    return IndexCSV(
        path, LmtRebuildIndexRow, dedup_keys=["run_id", "group", "sequence"]
    )


register_reconcilable_index(REBUILD_KIND, rebuild_index)


@dataclass(frozen=True, slots=True)
class _Source:
    """One covered raw file, as its index row names it."""

    group: str
    sequence: str
    path: Path
    md5: str

    @property
    def key(self) -> str:
        """The name of the entry's directory."""
        return make_entry_key(self.group, self.sequence)


class _EntryRefused(Exception):
    """An entry that this op refuses or cannot finish, which fails that entry alone."""


def _covered_rows(ds: Dataset, scope: ResolvedScope) -> list[_Source]:
    """The ``lmt_sqlite`` raw files that *scope* covers, sorted by entry key.

    Raises:
        ValueError: An entry named by the scope has no ``lmt_sqlite`` row or has
            two, the scope covers none, or a covered row has no md5.
    """
    by_entry: dict[Entry, list[dict[str, str]]] = defaultdict(list)
    for row in ds.read_tracks_raw_index():
        if row.get("src_format") == _SOURCE_FORMAT:
            by_entry[(row.get("group", ""), row.get("sequence", ""))].append(row)
    wanted = scope.op_entries
    if wanted is None:
        entries = sorted(by_entry)
    else:
        absent = [entry for entry in wanted if entry not in by_entry]
        if absent:
            named = ", ".join(make_entry_key(*entry) for entry in absent)
            raise ValueError(
                f"rebuild-lmt-events rebuilds {_SOURCE_FORMAT} files, and the "
                f"scope names entries without one: {named}."
            )
        entries = list(wanted)
    if not entries:
        raise ValueError(
            f"rebuild-lmt-events rebuilds {_SOURCE_FORMAT} files, and the scope "
            "covers none."
        )
    doubled = [entry for entry in entries if len(by_entry[entry]) > 1]
    if doubled:
        named = ", ".join(make_entry_key(*entry) for entry in doubled)
        raise ValueError(
            f"These entries have more than one {_SOURCE_FORMAT} file, and a "
            f"rebuild makes one copy of an entry: {named}."
        )
    sources = [
        _Source(
            group=entry[0],
            sequence=entry[1],
            path=ds.resolve_path(by_entry[entry][0]["abs_path"]),
            md5=by_entry[entry][0].get("md5", ""),
        )
        for entry in entries
    ]
    unhashed = [source.key for source in sources if not source.md5]
    if unhashed:
        raise ValueError(
            "The index holds no md5 for the file of "
            f"{', '.join(unhashed)}. Run 'mosaic scan' to hash it."
        )
    return sorted(sources, key=lambda source: source.key)


def _refuse_journal(path: Path, *, what: str) -> None:
    """Refuse *path* when a journal sits beside it, because a copy would lack its writes."""
    journals = journals_beside(path)
    if journals:
        raise _EntryRefused(
            f"{what} has {journals[0].name} beside it, so it holds writes that a "
            "copy of the database file alone would lack."
        )


def _refuse_little_space(entry_dir: Path, size: int) -> None:
    """Refuse an entry whose disk lacks twice the source's size."""
    free = shutil.disk_usage(entry_dir).free
    if free < 2 * size:
        raise _EntryRefused(
            f"The disk holds {free} free bytes, and a copy of a {size} byte file "
            f"needs {2 * size}: the copy, and the indexes and the journal that the "
            "rebuild adds."
        )


def _copy_and_hash(ctx: JobContext, source: Path, target: Path) -> str:
    """Copy *source* to *target* in chunks and return the md5 of what was read."""
    digest = hashlib.md5(usedforsecurity=False)
    with source.open("rb") as reading, target.open("wb") as writing:
        while chunk := reading.read(_COPY_CHUNK_BYTES):
            ctx.check_cancel()
            digest.update(chunk)
            _ = writing.write(chunk)
            ctx.heartbeat()
    return digest.hexdigest()


def _manifest(
    *,
    run_id: str,
    commit: str,
    source: _Source,
    source_size: int,
    copy: Path,
    report: LmtRunnerReport,
) -> RebuildManifest:
    """The manifest of the published *copy*, from the runner's report of its rebuild."""
    if (
        report.builders is None
        or report.max_frame is None
        or report.script_max_frame is None
        or report.python is None
        or report.packages is None
    ):
        raise _EntryRefused("The runner's report of a finished rebuild is incomplete.")
    stat = copy.stat()
    return RebuildManifest(
        manifest_format=1,
        run_id=run_id,
        op_version=REBUILD_OP_VERSION,
        lmt_analysis_commit=commit,
        builders=report.builders,
        source_md5=source.md5,
        source_name=source.path.name,
        source_size=source_size,
        copy_name=copy.name,
        copy_size=stat.st_size,
        copy_mtime_ns=stat.st_mtime_ns,
        max_frame=report.max_frame,
        script_max_frame=report.script_max_frame,
        python=report.python,
        packages=report.packages,
        completed_at=now_iso(),
    )


def _is_current(
    entry_dir: Path, source: _Source, commit: str, builders: tuple[str, ...] | None
) -> bool:
    """Whether *entry_dir* holds a finished rebuild of *source* under these inputs.

    The ``rebuild`` marker is present, the manifest names *commit*, *builders* and
    the source's md5, and the copy has the size and modification time that the
    manifest recorded. A manifest that cannot be read, or a missing copy, is not
    current.
    """
    if read_phase_marker(entry_dir, "rebuild") is None or builders is None:
        return False
    copy = entry_dir / source.path.name
    try:
        manifest = read_rebuild_manifest(copy)
        return (
            manifest.lmt_analysis_commit == commit
            and manifest.builders == builders
            and manifest.source_md5 == source.md5
            and manifest.copy_name == copy.name
            and copy.is_file()
            and copy_unchanged(copy, manifest)
        )
    except (ValueError, OSError):
        return False


def _has_index_row(ds: Dataset, run_id: str, source: _Source) -> bool:
    """Whether the rebuild index holds a row for *source*'s entry under *run_id*."""
    path = ds.get_root(REBUILD_KIND) / "index.csv"
    if not path.is_file():
        return False
    rows = rebuild_index(path).read(run_id=run_id)
    return bool(
        ((rows["group"] == source.group) & (rows["sequence"] == source.sequence)).any()
    )


def _append_index_row(
    ds: Dataset, run_id: str, source: _Source, entry_dir: Path
) -> None:
    """Append the row that names the finished *entry_dir*, under the index's lock."""
    rebuild_index(ds.get_root(REBUILD_KIND) / "index.csv").append(
        [
            LmtRebuildIndexRow(
                abs_path=Path(ds.relative_to_root(entry_dir)),
                run_id=run_id,
                group=source.group,
                sequence=source.sequence,
                source_md5=source.md5,
            )
        ]
    )


def _drop_index_row(ds: Dataset, run_id: str, source: _Source) -> None:
    """Drop the row of *source*'s entry under *run_id*, under the index's lock.

    A row implies a finished copy, so it goes before the entry is redone: a redo
    that fails leaves logs and no row, as the first attempt at the entry does.
    """
    _ = rebuild_index(ds.get_root(REBUILD_KIND) / "index.csv").drop_entries(
        scope=Scope(entries=[(source.group, source.sequence)]), run_id=run_id
    )


def _remove_with_journals(path: Path) -> None:
    """Remove the file *path* and the journals beside it, whichever exist."""
    journals = journals_beside(path)
    path.unlink(missing_ok=True)
    for journal in journals:
        journal.unlink(missing_ok=True)


@register_op
class RebuildLmtEventsOp(Op[RebuildLmtEventsParams]):
    """Copy each Live Mouse Tracker database of the scope and rebuild its events.

    Reads the raw files and writes only under the ``rebuild-lmt-events`` root.
    Entries run one at a time, because one rebuild loads a day of detections of
    every animal.
    """

    kind = REBUILD_KIND
    category = "convert"
    domain = "tracking"
    version = REBUILD_OP_VERSION
    scope_takes = "any"
    scope_dependent = True
    writes_tracks = False
    parallelizable = False
    keywords = ("live mouse tracker", "lmt", "flickering", "rebuild events")
    Params = RebuildLmtEventsParams

    @override
    def plan_identity(
        self,
        ds: Dataset,
        params: RebuildLmtEventsParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """The run id over the covered files' md5s, from the raw index alone."""
        sources = _covered_rows(ds, scope)
        return OpIdentity(
            run_id=rebuild_lmt_events_run_id(params, [s.md5 for s in sources])
        )

    @override
    def run(
        self,
        ds: Dataset,
        params: RebuildLmtEventsParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        run_id = self.plan_identity(ds, params, scope).run_id
        ctx.set_run_id(run_id)
        sources = _covered_rows(ds, scope)

        ctx.phase("probe")
        with tempfile.TemporaryDirectory(prefix="mosaic-lmt-probe-") as scratch:
            probed = probe_lmt_analysis(
                params.lmt_analysis_commit,
                params.builders,
                Path(scratch) / _REPORT_NAME,
                cancel_check=ctx.cancel_token.is_cancelled,
            )
        resolved = probed.builders

        ctx.set_total(len(sources))
        run_root = ds.get_root(REBUILD_KIND) / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        write_identity_scheme(run_root, OP_IDENTITY_SCHEME)

        attempted: set[str] = set()
        lost: set[str] = set()
        rebuilt = 0
        current = 0
        try:
            for position, source in enumerate(sources):
                ctx.check_cancel()
                key = source.key
                ctx.progress.on_entry_start(position, len(sources), key)
                opened = open_entry(
                    ds,
                    ctx,
                    run_root,
                    key,
                    kind=REBUILD_KIND,
                    overwrite=overwrite,
                    idle_seconds=REBUILD_IDLE_SECONDS,
                )
                if opened is None:
                    ctx.progress.on_entry_end(position + 1, len(sources), key)
                    ctx.heartbeat(position + 1)
                    continue
                entry_dir, claim = opened
                attempted.add(key)
                try:
                    if _is_current(
                        entry_dir, source, params.lmt_analysis_commit, resolved
                    ):
                        if not _has_index_row(ds, run_id, source):
                            _append_index_row(ds, run_id, source, entry_dir)
                        current += 1
                    else:
                        self._rebuild_entry(
                            ds, ctx, params, run_id, source, entry_dir, claim
                        )
                        rebuilt += 1
                except (_EntryRefused, LmtRebuildError, OSError) as failure:
                    lost.add(key)
                    ctx.entry_failed_unless_cancelled(key, failure)
                finally:
                    release_entry(entry_dir, ctx.execution_id)
                    discard_if_empty(entry_dir)
                ctx.progress.on_entry_end(position + 1, len(sources), key)
                ctx.heartbeat(position + 1)
        finally:
            ctx.entries_written(rebuilt + current)

        if sources and current == len(sources):
            ctx.cache_hit()
        if attempted and lost == attempted:
            raise AllEntriesFailed(
                f"[{REBUILD_KIND}] every one of {len(attempted)} attempted entries "
                f"failed: {', '.join(sorted(lost))}. Run {run_id} holds no copy "
                f"under {run_root}. The run-log names each failure."
            )
        print(
            f"[{REBUILD_KIND}] completed run_id={run_id} "
            f"({rebuilt} rebuilt, {current} current, of {len(sources)})"
        )
        return run_id

    def _rebuild_entry(
        self,
        ds: Dataset,
        ctx: JobContext,
        params: RebuildLmtEventsParams,
        run_id: str,
        source: _Source,
        entry_dir: Path,
        claim: InflightMarker,
    ) -> None:
        """Copy *source* into *entry_dir* and rebuild the copy's events.

        Raises:
            _EntryRefused: The entry cannot be rebuilt.
            LmtRebuildError: The runner stopped the rebuild.
            OSError: A file could not be read or written.
        """
        key = source.key
        _drop_index_row(ds, run_id, source)
        clear_phase_marker(entry_dir, "rebuild")
        clear_outputs(entry_dir, REBUILD_KIND, "rebuild")
        _refuse_journal(source.path, what=f"The source {source.path}")
        size = source.path.stat().st_size
        _refuse_little_space(entry_dir, size)

        incoming = (
            entry_dir / f"{_INCOMING_PREFIX}{ctx.execution_id}-{source.path.name}"
        )
        published = entry_dir / source.path.name
        finished = False
        try:
            with kept_claimed(entry_dir, claim, REBUILD_IDLE_SECONDS):
                ctx.progress.on_phase("copy", key)
                copied = _copy_and_hash(ctx, source.path, incoming)
                if copied != source.md5:
                    raise _EntryRefused(
                        f"{source.path} has md5 {copied}, and the index holds "
                        f"{source.md5}: the file changed after the scan. Run "
                        "'mosaic scan' to index it again, which mints another run id."
                    )
                ctx.progress.on_phase("rebuild", key)
                report = run_lmt_rebuild(
                    incoming,
                    params.lmt_analysis_commit,
                    params.builders,
                    report_path=entry_dir / _REPORT_NAME,
                    log_path=entry_dir / _LOG_NAME,
                    cancel_check=ctx.cancel_token.is_cancelled,
                    on_line=phase_activity(ctx, entry_dir, claim, REBUILD_IDLE_SECONDS),
                )
            _refuse_journal(incoming, what="The rebuilt copy")
            _ = os.replace(incoming, published)
            _manifest(
                run_id=run_id,
                commit=params.lmt_analysis_commit,
                source=source,
                source_size=size,
                copy=published,
                report=report,
            ).write(entry_dir)
            write_phase_marker(
                entry_dir,
                PhaseMarker(
                    phase="rebuild",
                    run_id=run_id,
                    execution_id=ctx.execution_id,
                    completed_at=now_iso(),
                    source=ds.relative_to_root(source.path),
                    source_uid=source.md5,
                ),
            )
            finished = True
        finally:
            if not finished:
                _remove_with_journals(incoming)
                _remove_with_journals(published)
        _append_index_row(ds, run_id, source, entry_dir)
