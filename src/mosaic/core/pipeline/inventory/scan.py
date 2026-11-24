"""Reading a dataset's artifacts: the entry universe, coverage, and the scan.

The completeness predicate and the target narrowing live here rather than inside
``Pipeline`` because they answer a question about the *dataset*, not about a
chain: which entries does this run hold, and which were wanted. They were
module-private with one caller each, so nothing else in the toolkit -- not the
CLI, not an API, not a decider -- could ask.

:func:`run_covers` returns the evidence where the predicate it replaces returned
a bool. That is the whole change in substance: a bool cannot say 89 of 90, and
a caller given one has to re-glob the directory to find out, which is exactly
what the chain runner did -- with a second glob that skipped the readability
filter, so its displayed count could exceed what its own cache gate accepted.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.core.entry import Entry
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.dataset_indexes import feature_storages, label_kinds
from mosaic.core.pipeline.identity_scheme import read_identity_scheme
from mosaic.core.pipeline.index import (
    DriftReads,
    FitDrift,
    drifted_entries,
    fit_drift,
    feature_index,
    feature_index_path,
    feature_run_root,
    feature_run_roots,
)
from mosaic.core.pipeline.labels_index import labels_index_path, read_labels_index
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.parquet_reader import ParquetFooterMemo
from mosaic.core.pipeline.tracks_index import (
    frame_axis_findings,
    media_drift,
    read_tracks_index,
    select_variant_rows,
    tracks_index_path,
    tracks_rows,
)
from mosaic.core.scope import Scope

from ..artifact_kinds import LINKED_KIND, LinkedKind
from ._linked import Remeasure, grouped_by_ref, merged_record, nearest_rows_present
from ._read import IndexReader
from ._runs import finish_state, run_records
from .contributors import inventory_contributor, registered_inventory_kinds
from .label_series import label_series_records
from .media import media_derivative_record, media_variant_records
from .model import (
    AnyRecord,
    ArtifactKind,
    ArtifactStatus,
    Coverage,
    DatasetInventory,
    ArtifactRecord,
    FeatureRunRef,
    InventoryScope,
    LabelsVariantRef,
    TracksVariantRef,
    classify,
    merged_coverage,
)
from .params import RunParams, linked_run_root, read_run_params

from collections.abc import Callable, Sequence

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from mosaic.core.dataset import Dataset, LinkMismatchError
    from mosaic.core.manifest import DatasetLink

__all__ = [
    "GLOBAL_ENTRY",
    "GLOBAL_MARKER",
    "CORE_KINDS",
    "entry_universe",
    "existing_tracks_records",
    "feature_run_target",
    "inventory",
    "narrow_target",
    "reportable_universe",
    "run_covered_entries",
    "run_coverage",
    "run_covers",
    "without_mismatched_links",
]

GLOBAL_MARKER = "__global__"
"""The stem a global fit writes instead of one output per entry."""

GLOBAL_ENTRY: Entry = ("", GLOBAL_MARKER)
"""The index row a global fit records beside that file, written by ``run_feature``.

It is not a real entry, and treating it as one made every global fit -- t-SNE,
k-means, Ward, keypoint-MoSeq -- report as damaged: the row named an entry the
per-entry file set could never contain, which reads as a row with no output.
"""

CORE_KINDS: frozenset[ArtifactKind] = frozenset(
    {
        "feature",
        "tracks-variant",
        "labels-variant",
        "label-series",
        "media-derivative",
        "media-variant",
    }
)
"""The kinds ``core`` can report on by itself, with no producer registered."""


def entry_universe(ds: Dataset, tracks_run_id: str | None = None) -> frozenset[Entry]:
    """Every ``(group, sequence)`` this dataset can process, from its tracks.

    Only rows whose table actually exists are kept, matching what
    ``build_manifest`` would resolve, so the universe is what could really be
    computed rather than what the index claims.

    *tracks_run_id* must be the same selector the runs were made under. A run
    pinned to a variant covering part of the index, measured against the whole
    index, reads as permanently incomplete -- forever, on every invocation. That
    is the same shape of wrong answer as measuring transcode against a run
    directory it never had.
    """
    frame = select_variant_rows(tracks_rows(ds), tracks_run_id)
    return _entries_of(existing_tracks_records(ds, _entry_columns(frame)))


def existing_tracks_records(ds: Dataset, frame: pd.DataFrame) -> list[dict[str, str]]:
    """Return the rows of the tracks index *frame* whose table exists on disk.

    The one rule for which rows of an index have a table: a row names a stored
    path, and the file it resolves to is there. The paths are resolved together
    (:meth:`~mosaic.core.dataset.Dataset.resolve_paths`).
    """
    records = [record for record in index_records(frame) if record.get("abs_path")]
    paths = ds.resolve_paths(record["abs_path"] for record in records)
    return [
        record for record, path in zip(records, paths, strict=True) if path.exists()
    ]


def _entry_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """The columns of a tracks index that name an entry and its table."""
    return frame.loc[:, ["group", "sequence", "abs_path"]]


def _entries_of(records: Iterable[dict[str, str]]) -> frozenset[Entry]:
    """Return the ``(group, sequence)`` entries of tracks index *records*."""
    return frozenset(
        (record.get("group", ""), record.get("sequence", "")) for record in records
    )


def narrow_target(
    universe: frozenset[Entry], scope: Scope | None = None
) -> frozenset[Entry]:
    """Narrow the entry universe to what *scope* names.

    The same narrowing ``build_manifest`` applies. What is measured therefore
    matches what would be processed. ``None`` and an unset selector both keep
    the whole universe. A named selector that lists nothing keeps nothing, the
    rule :attr:`ResolvedScope.op_entries` states. An empty list used to read
    here as no restriction at all.
    """
    selector = scope if scope is not None else Scope()
    pairs = selector.entry_pairs
    if pairs is not None:
        return universe & pairs
    selected = selector.selecting()
    return frozenset(entry for entry in universe if selected(entry))


def entries_held(stems: set[str], candidates: frozenset[Entry]) -> set[Entry]:
    """Which of *candidates* the output stems account for.

    An entry's output is usually one file named exactly ``make_entry_key(...)``,
    but not always: a feature that splits an entry by individual writes
    ``<entry key>__id0``, ``__id1`` and so on, which is how ``kpms-apply`` and
    its kin store their results. Matching the key exactly reads those runs as
    holding nothing at all, and their index rows then look like rows with no
    output -- damage, where the outputs are in fact right there under a
    convention the exact match did not know about.

    An entry is held when its key is a stem, or a stem begins ``<key>__``. Each
    stem is cut at every ``__`` once, and the keys are looked up among the cuts,
    so the cost is the stems and the candidates, not their product.
    """
    by_key = {
        make_entry_key(group, sequence): (group, sequence)
        for group, sequence in candidates
    }
    named: set[str] = set()
    for stem in stems:
        named.add(stem)
        cut = stem.find("__")
        while cut != -1:
            named.add(stem[:cut])
            cut = stem.find("__", cut + 1)
    return {entry for key, entry in by_key.items() if key in named}


def run_covers(
    run_root: Path,
    target: frozenset[Entry],
    known: frozenset[Entry] = frozenset(),
    footers: ParquetFooterMemo | None = None,
) -> Coverage[Entry]:
    """Which of *target* this run root holds, as evidence rather than a verdict.

    Files rather than the index's ``finished_at`` flag, deliberately: that flag
    is relative to whatever manifest one invocation saw, so it answers a
    different question than "is this entry here". It also means the filesystem
    may legitimately be ahead of the index, which is the safe direction -- a file
    check resumes correctly where an index check would recompute.

    A ``__global__`` output sets ``covers_all``: a global fit writes one artifact
    rather than one per entry, so counting entries against it would report zero
    of ninety for a run that is complete.

    *known* names entries to recognise **beyond** the target, and it is what
    makes the empty-target rule work. An output file is a ``<group>__<sequence>``
    stem, and that encoding does not invert -- a stem cannot be parsed back into
    an entry unambiguously -- so an entry is only recognisable if something named
    it first. When the target is empty, as it is on a dataset whose tracks index
    points at files that are gone, nothing would name it and every run would read
    as holding nothing at all. Passing the run's own index rows means the
    question stays "which entries are on disk" rather than becoming "which
    entries are on disk *and* still resolvable from tracks".

    *footers* remembers the files whose footer parsed, so that a caller that scans
    one run root again reads only the files that moved. Without one, every footer
    is read.
    """
    if not run_root.exists():
        return Coverage(target=target, present=frozenset())
    memo = ParquetFooterMemo() if footers is None else footers
    stems = memo.whole_stems(run_root)
    held = entries_held(stems, target | known)
    if GLOBAL_MARKER in stems:
        # The per-entry mapping still applies. A global fit may write per-entry
        # outputs *beside* its marker -- an apply step does exactly that -- so
        # short-circuiting here would leave those files unrecognised and their
        # index rows reading as rows with no output.
        return Coverage(
            target=target,
            present=target | frozenset(held) | {GLOBAL_ENTRY},
            covers_all=True,
        )
    return Coverage(target=target, present=frozenset(held))


def run_coverage(
    ds: Dataset,
    storage: str,
    run_id: str,
    target: frozenset[Entry],
    *,
    known: frozenset[Entry] = frozenset(),
    footers: ParquetFooterMemo | None = None,
) -> Coverage[Entry]:
    """Which of *target* a feature run holds across *ds* and the datasets it links.

    Each run root of :func:`~mosaic.core.pipeline.index.feature_run_roots` is
    measured by :func:`run_covers`, and the measures are joined against *target*
    by :func:`~.model.merged_coverage`. An entry is held when any root holds its
    output.

    Args:
        ds: The reading dataset.
        storage: The feature's storage directory name.
        run_id: The run's identifier.
        target: The entries *ds* wants, which the result keeps.
        known: Entries to recognize beyond *target*, as :func:`run_covers`
            takes them.
        footers: The memo of parsed footers that a holder keeps across calls.
            Without one, every footer is read.

    Returns:
        The joined coverage. It holds nothing when no root exists.
    """
    memo = ParquetFooterMemo() if footers is None else footers
    return merged_coverage(
        (
            run_covers(root, target, known, memo)
            for root in feature_run_roots(ds, storage, run_id)
        ),
        target,
    )


def _orphan_rows(
    coverage: Coverage[Entry], rows: frozenset[Entry], files: frozenset[Entry]
) -> frozenset[Entry]:
    """Return the rows that name an entry with no output, given contradicting files.

    Damage needs contradictory evidence. A run whose outputs are all unmatched
    leaves its rows unconfirmed and uncontradicted. A feature that stores its
    results as ``.npz`` beside a ``seq=<name>`` filename is such a run. A row is
    orphaned only when some file matched an entry and the index names more
    entries than the files cover. When every file is unmatched, the status
    reports the coverage shortfall alone.

    Tracks and labels variants follow the same rule, although each of their
    rows names its file. A variant whose tables are all gone therefore reads
    ``absent`` with no orphaned rows, and one that lost some of its tables reads
    ``inconsistent``.

    The record keeps these rows. A status re-derived from the record over
    another target (:func:`~.model.reclassify`) then agrees with this one.
    """
    if not files and not coverage.covers_all:
        return frozenset()
    return rows - files


def _status_for(
    coverage: Coverage[Entry],
    rows: frozenset[Entry],
    files: frozenset[Entry],
    drift: tuple[Entry, ...],
    finished: bool,
) -> ArtifactStatus:
    """Row-and-file agreement plus coverage, through the one status rule.

    Both halves are required, the idiom ``training_is_complete`` already uses:
    the index proves the producer got far enough to record the entry, the file
    proves the output survived. Either alone reads complete over a run the other
    knows is not. Orphaned rows follow :func:`_orphan_rows`.
    """
    return classify(
        satisfied=coverage.is_satisfied,
        any_covered=bool(coverage.covered) or coverage.covers_all,
        orphan_rows=bool(_orphan_rows(coverage, rows, files)),
        orphan_files=bool(files - rows),
        drifted=bool(drift),
        finished=finished,
    )


def reportable_universe(
    ds: Dataset, tracks_run_id: str | None = None
) -> frozenset[Entry]:
    """The entry universe for a *report*, which may not refuse to answer.

    ``entry_universe`` resolves through ``select_variant_rows``, which raises
    when one entry carries two genuine tracks recipes -- correctly, because
    *executing* against an ambiguous entry has no defensible default and a guess
    would silently read the wrong table.

    An inventory is not executing. A dataset holding two recipes for an entry is
    a legitimate state, and the one question a reader most wants answered about
    it is what it holds. Refusing turns a describable dataset into an exception
    and takes every other artifact's report down with it. So the ambiguity falls
    back to the union across variants: the entry is processable, the caller just
    has to say which variant when they come to process it.
    """
    try:
        return entry_universe(ds, tracks_run_id)
    except ValueError:
        return _entries_of(existing_tracks_records(ds, _entry_columns(tracks_rows(ds))))


def feature_run_target(
    asked: frozenset[Entry], params: RunParams | None, rows: frozenset[Entry]
) -> frozenset[Entry]:
    """The entries a feature run is measured against, out of the *asked* ones.

    A run is measured against the entries its invocations asked for
    (:meth:`~.params.RunParams.asked_scope`) and the entries its index *rows*
    name, within *asked*. A request can apply a scope-free classifier to a test
    group, and the run is then complete when it contains that group. Running a
    scope-dependent run over more entries makes another run, because its scope
    is hashed into its run id. An entry that failed was asked for and its file
    is missing. The run then reads ``partial`` and reports the loss, also after
    a later invocation over other entries.

    The rows stay in the basis for a file written before ``asked`` existed,
    whose ``entries`` name only its latest invocation.

    A scope-free run with ``asked_all`` set was asked for every entry, without
    a selector or as an unnarrowed recipe step. Its target is *asked*, and it
    reads ``partial`` once the dataset gains a sequence. A run whose
    ``params.json`` is missing is measured the same way.

    When the basis shares nothing with *asked*, as for a selector naming entries
    the run did not cover, the target stays *asked*. An empty target reads as
    "anything at all" (:attr:`Coverage.is_satisfied`), and the run would then
    read complete over entries it does not contain.

    The one rule for both the scan and the narrowed projection of an unscoped
    scan (``inventory_wire``), so the two cannot disagree.
    """
    if params is None or params.scope.asked_all:
        return asked
    within = asked & (frozenset(params.asked_scope()) | rows)
    return within or asked


def run_covered_entries(
    params: RunParams | None, rows: frozenset[Entry], within: frozenset[Entry]
) -> frozenset[Entry]:
    """The entries a feature run covers, among the *within* ones.

    A run covers the entries its invocations asked for
    (:meth:`~.params.RunParams.asked_scope`) and the entries its index *rows*
    name. It is not measured over *within* the way :func:`feature_run_target`
    measures it, since that returns the whole target for a run recorded as asked
    for every entry, which is the run a growth caution looks for. An entry the
    run covered and *within* no longer holds counts for nothing.
    """
    covered = rows
    if params is not None:
        covered = covered | frozenset(params.asked_scope())
    return covered & within


def _feature_records(
    ds: Dataset, scope: InventoryScope, reader: IndexReader
) -> list[ArtifactRecord[Entry]]:
    """Every feature run under ``features/<name>/<run_id>/``.

    A run is measured over the part of the target it was asked for
    (:func:`feature_run_target`). Its files are matched over the whole target.
    The orphan files and the ``present`` set therefore do not depend on that
    cut.
    """
    target = narrow_target(reportable_universe(ds, scope.tracks_run_id), scope.selector)
    return _feature_runs(ds, reader, target, _drift_reads(ds, reader))


def _drift_reads(ds: Dataset, reader: IndexReader) -> DriftReads:
    """One read of each feature index and one tracks cell per variant set."""
    return DriftReads(
        ds, lambda path: reader.frame(path, lambda: feature_index(path).read())
    )


def _drifted_fit(
    ds: Dataset,
    name: str,
    run_id: str,
    params: RunParams | None,
    drift_reads: DriftReads,
) -> FitDrift | None:
    """What changed under the fit of a run, or ``None`` when its fit has not drifted."""
    drift = fit_drift(ds, name, run_id, params, drift_reads)
    return drift if drift is not None and drift.drifted else None


def _feature_runs(
    ds: Dataset,
    reader: IndexReader,
    target: frozenset[Entry],
    drift_reads: DriftReads,
) -> list[ArtifactRecord[Entry]]:
    """Every feature run of *ds*, measured over *target*, with one *drift_reads*."""
    records: list[ArtifactRecord[Entry]] = []
    for name in feature_storages(ds):
        index_path = feature_index_path(ds, name)
        frame = reader.frame(index_path, lambda p=index_path: feature_index(p).read())
        if frame.empty or "run_id" not in frame.columns:
            continue
        for run_id, run_rows in run_records(frame).items():
            run_root = feature_run_root(ds, name, run_id)
            rows = _entries_of(run_rows)
            read = read_run_params(run_root)
            # The run's own index rows are what make its outputs recognisable
            # when the target cannot name them -- see ``run_covers``.
            on_disk = run_covers(run_root, target, known=rows, footers=reader.footers)
            files = on_disk.present
            coverage = Coverage[Entry](
                target=feature_run_target(target, read.params, rows),
                present=on_disk.present,
                covers_all=on_disk.covers_all,
            )
            started, finished_at, finished = finish_state(run_rows)
            refit = _drifted_fit(ds, name, run_id, read.params, drift_reads)
            drift = drifted_entries(
                ds,
                name,
                run_id,
                read.params,
                drift_reads,
                fit_drifted=refit is not None,
            )
            records.append(
                ArtifactRecord[Entry](
                    ref=FeatureRunRef(name=name, run_id=run_id),
                    name=name,
                    run_id=run_id,
                    coverage=coverage,
                    status=_status_for(coverage, rows, files, drift, finished),
                    run_root=run_root,
                    index_path=index_path,
                    params=read.params,
                    params_state=read.state,
                    rows=rows,
                    orphan_rows=_orphan_rows(coverage, rows, files),
                    orphan_files=files - rows,
                    drift=drift,
                    refit=refit,
                    identity_scheme=read_identity_scheme(run_root),
                    started_at=started,
                    finished_at=finished_at,
                    upstreams=tuple(
                        sorted(read.params.consumed_run_ids()) if read.params else ()
                    ),
                )
            )
    return records


def _variant_records(
    ds: Dataset, scope: InventoryScope, reader: IndexReader
) -> list[ArtifactRecord[Entry]]:
    """Every tracks recipe under ``tracks/<variant>/``.

    A variant's target is the entries its own rows name, not the dataset's
    universe: two recipes legitimately cover different entries, and measuring
    one against the other's rows would report a complete conversion as short.

    A **bridged** variant also carries what the media it read was, so an entry
    whose video has been re-transcoded since reads as drifted rather than as
    current. Nothing in the variant identity notices that on its own -- it is
    params plus the model, with no term for the pixels -- so without the
    comparison a re-encode is served as a cache hit over different frames. A
    converted variant records no media composition and so never drifts here,
    which is right: it opened no video.

    ``extra["frame_axis_mismatch"]`` names the entries whose tool read a
    different number of frames than their media holds, other than by a known
    tail loss. ``extra["frame_tail_short"]`` names those whose tool read short by
    no more than it is allowed to lose at the end of the file it read
    (``tail_allowance``), as TRex's under-count leaves them. Both are recorded in
    ``extra`` rather than in ``status``, because the status set is five closed
    members crossing the CLI and mosaic-api's wire, and each is a measurement
    two cells apart rather than a state of the artifact. The table is there, it
    is complete, and it is internally right.
    """
    index_path = tracks_index_path(ds)
    frame = reader.frame(index_path, lambda: read_tracks_index(ds))
    if frame.empty or "run_id" not in frame.columns:
        return []
    records: list[ArtifactRecord[Entry]] = []
    findings = frame_axis_findings(frame)
    on_disk = _tables_on_disk(ds, frame)
    for run_id, run_rows in run_records(frame).items():
        rows = _entries_of(run_rows)
        wanted = scope.selector.entry_pairs
        target = frozenset(rows if wanted is None else rows & wanted)
        files = rows & on_disk.get(run_id, frozenset())
        coverage = Coverage(target=target, present=files)
        started, finished_at, finished = finish_state(run_rows)
        drift = media_drift(ds, frame[frame["run_id"].astype(str) == run_id])
        mismatched = frozenset(
            make_entry_key(m.group, m.sequence)
            for m in findings.mismatches
            if m.run_id == run_id
        )
        tail_short = frozenset(
            make_entry_key(m.group, m.sequence)
            for m in findings.tail_shortfalls
            if m.run_id == run_id
        )
        records.append(
            ArtifactRecord[Entry](
                ref=TracksVariantRef(run_id=run_id),
                name="tracks",
                run_id=run_id,
                coverage=coverage,
                status=_status_for(coverage, rows, files, drift, finished),
                index_path=index_path,
                rows=rows,
                orphan_rows=_orphan_rows(coverage, rows, files),
                orphan_files=frozenset(),
                drift=drift,
                started_at=started,
                finished_at=finished_at,
                extra={
                    "frame_axis_mismatch": mismatched,
                    "frame_tail_short": tail_short,
                },
            )
        )
    return records


def _tables_on_disk(ds: Dataset, frame: pd.DataFrame) -> dict[str, frozenset[Entry]]:
    """Each run's entries whose index row names a table that is on disk.

    One pass over the rows of *frame*, which holds every run of one index.
    """
    wanted = ("run_id", "group", "sequence", "abs_path")
    named = frame.loc[:, [column for column in wanted if column in frame.columns]]
    present: dict[str, set[Entry]] = {}
    for record in existing_tracks_records(ds, named):
        run_id = record.get("run_id", "")
        entry = (record.get("group", ""), record.get("sequence", ""))
        present.setdefault(run_id, set()).add(entry)
    return {run_id: frozenset(entries) for run_id, entries in present.items()}


def _labels_records(
    ds: Dataset, scope: InventoryScope, reader: IndexReader
) -> list[ArtifactRecord[Entry]]:
    """Every converted-label variant under ``labels/<kind>/<run_id>/``."""
    records: list[ArtifactRecord[Entry]] = []
    for kind in label_kinds(ds):
        index_path = labels_index_path(ds, kind)
        frame = reader.frame(index_path, lambda k=kind: read_labels_index(ds, k))
        if frame.empty or "run_id" not in frame.columns:
            continue
        on_disk = _tables_on_disk(ds, frame)
        for run_id, run_rows in run_records(frame).items():
            rows = _entries_of(run_rows)
            wanted = scope.selector.entry_pairs
            target = frozenset(rows if wanted is None else rows & wanted)
            files = rows & on_disk.get(run_id, frozenset())
            coverage = Coverage(target=target, present=files)
            started, finished_at, finished = finish_state(run_rows)
            records.append(
                ArtifactRecord[Entry](
                    ref=LabelsVariantRef(label_kind=kind, run_id=run_id),
                    name=kind,
                    run_id=run_id,
                    coverage=coverage,
                    status=_status_for(coverage, rows, files, (), finished),
                    index_path=index_path,
                    rows=rows,
                    orphan_rows=_orphan_rows(coverage, rows, files),
                    orphan_files=frozenset(),
                    started_at=started,
                    finished_at=finished_at,
                )
            )
    return records


def inventory(
    ds: Dataset,
    *,
    kinds: Iterable[ArtifactKind] | None = None,
    scope: Scope | None = None,
    tracks_run_id: str | None = None,
    footers: ParquetFooterMemo | None = None,
) -> DatasetInventory:
    """What this dataset holds: every artifact, its identity and its coverage.

    A kind that a link provides, as ``LINKED_KIND`` maps it, is also read from
    each linked dataset of ``ds.lookup_order`` by the same builder or
    contributor, and the records of one artifact are joined by ref and measured
    over the target of *ds*. A joined record names its link in ``link_id`` when
    *ds* lacks the run directory, and the keys only a link holds in
    ``linked_only``. A linked dataset is read for its own data alone. A link
    whose path holds another dataset is skipped: each wanted kind it provides
    names it in ``errors``, and the own records stay.

    Args:
        ds: The dataset to read. Never written to.
        kinds: Which artifact kinds to report. ``None`` means every kind the
            running process can report on, which depends on what has been
            imported -- see ``unavailable_kinds`` on the result.
        scope: Narrow coverage to what this selector names. ``None`` measures
            against everything the dataset can process.
        tracks_run_id: Which tracks variant defines the entry universe. Pass the
            one the runs were made under; measuring a variant-pinned run against
            the whole index reads as permanently incomplete.
        footers: The parquet files already known to be whole, which a holder of
            several scans keeps so that each reads only the files that moved.

    Returns:
        A :class:`DatasetInventory`, which is a cache and never the record.
    """
    wanted = frozenset(kinds) if kinds is not None else _every_kind()
    asked = InventoryScope(
        kinds=wanted,
        selector=scope if scope is not None else Scope(),
        tracks_run_id=tracks_run_id,
    )
    reader = IndexReader(footers)
    records: list[AnyRecord] = []
    errors: list[str] = []
    unavailable: set[ArtifactKind] = set()

    reading = _without_mismatched_links(ds, wanted, errors)
    links = _Links(reading)
    for kind in sorted(wanted):
        if kind == "media-derivative":
            # Both targets, because they are independent verdicts with
            # independent derivatives: a playback transcode never satisfies an
            # analysis read, and reporting one would hide the other.
            for media_target in ("analysis", "playback"):
                records.append(
                    media_derivative_record(reading, media_target, asked, reader)
                )
            continue
        if kind == "media-variant":
            # Media-variant records are keyed by entry and camera, and cannot
            # share the entry-keyed builder table below.
            records.extend(media_variant_records(reading, asked, reader))
            continue
        if kind == "feature":
            records.extend(_merged_feature_records(reading, asked, reader, links))
            continue
        build = _builder(kind)
        if build is None:
            unavailable.add(kind)
            continue
        held = _held_records(
            kind, build, (("", reading), *links.of(kind)), asked, reader, errors
        )
        if held is not None:
            records.extend(
                _merged(
                    held,
                    fills_linked_only=kind in _LINKED_ONLY_KINDS,
                    nearest_rows=kind in _NEAREST_ROW_KINDS,
                )
            )

    return DatasetInventory(
        dataset_root=Path(ds.base_dir),
        scope=asked,
        records=tuple(records),
        unavailable_kinds=frozenset(unavailable),
        errors=tuple(errors),
    )


def _every_kind() -> frozenset[ArtifactKind]:
    """Core's kinds plus whatever registered itself, which depends on imports."""
    return CORE_KINDS | registered_inventory_kinds()


type _CoreBuilder = Callable[
    ["Dataset", InventoryScope, IndexReader], list[ArtifactRecord[Entry]]
]

type _Builder = Callable[["Dataset", InventoryScope, IndexReader], Sequence[AnyRecord]]

_CORE_BUILDERS: dict[ArtifactKind, _CoreBuilder] = {
    "feature": _feature_records,
    "tracks-variant": _variant_records,
    "labels-variant": _labels_records,
}

_LINKED_ONLY_KINDS: frozenset[ArtifactKind] = frozenset(
    {"feature", "tracks-variant", "frame-run"}
)
"""The kinds whose records name the keys only a linked dataset holds."""

_NEAREST_ROW_KINDS: frozenset[ArtifactKind] = frozenset({"labels-variant"})
"""The kinds whose key is present by the nearest dataset that holds a row of it.

A read of a converted labels entry takes the nearest dataset's row, so a file of
that key in a farther dataset does not count.
"""


def _builder(kind: ArtifactKind) -> _Builder | None:
    """The builder of *kind*'s records, or ``None`` when nothing reports it."""
    if kind == "label-series":
        return label_series_records
    core = _CORE_BUILDERS.get(kind)
    return core if core is not None else inventory_contributor(kind)


def _without_mismatched_links(
    ds: Dataset, wanted: frozenset[ArtifactKind], errors: list[str]
) -> Dataset:
    """*ds* less each link whose path holds another dataset.

    A mismatched link costs the linked records of each kind it provides, and one
    error per wanted kind names it in *errors*. The own records and every other
    link stay, so the own builders, whose reads follow the links of their
    inputs too, read *ds* less that link.
    """
    reading, mismatches = without_mismatched_links(ds)
    errors.extend(
        f"{kind}: {exc}"
        for kind in sorted(wanted)
        for link, exc in mismatches
        if (linked_kind := LINKED_KIND.get(kind)) is not None
        and linked_kind in link.provides
    )
    return reading


def without_mismatched_links(
    ds: Dataset,
) -> tuple[Dataset, tuple[tuple[DatasetLink, LinkMismatchError], ...]]:
    """*ds* less each link whose path holds another dataset, and those links.

    A lookup over every link, which *ds* keeps once it opens them all, finds
    whether any link mismatches.

    Args:
        ds: The reading dataset.

    Returns:
        *ds* itself when no link mismatches, else a reader of it that skips
        the mismatched links, with each such link and its error in declaration
        order.
    """
    from mosaic.core.dataset import LinkMismatchError

    try:
        _ = ds.lookup_order()
    except LinkMismatchError:
        mismatches = ds.link_mismatches()
    else:
        return ds, ()
    return ds.without_links({link.id for link, _ in mismatches}), mismatches


class _Links:
    """The linked datasets of each kind an inventory reads, looked up once each."""

    def __init__(self, ds: Dataset) -> None:
        self._ds: Dataset = ds
        self._found: dict[LinkedKind, tuple[tuple[str, Dataset], ...]] = {}

    def of(self, kind: ArtifactKind) -> tuple[tuple[str, Dataset], ...]:
        """Each linked dataset that provides *kind*, with its link id, in order."""
        linked_kind = LINKED_KIND.get(kind)
        if linked_kind is None:
            return ()
        found = self._found.get(linked_kind)
        if found is None:
            found = tuple(
                (linked.link_id, linked.dataset)
                for linked in self._ds.lookup_order(linked_kind)[1:]
            )
            self._found[linked_kind] = found
        return found


def _held_records(
    kind: ArtifactKind,
    build: _Builder,
    datasets: Iterable[tuple[str, Dataset]],
    asked: InventoryScope,
    reader: IndexReader,
    errors: list[str],
) -> list[tuple[str, Sequence[AnyRecord]]] | None:
    """Each dataset's records of *kind*, with its link id, in lookup order.

    A core builder raises. Any other builder is guarded: one producer's failure
    costs its kind, never the whole answer, because an inventory that raises on
    one malformed tracker root tells a user nothing about the rest of their
    dataset, and a library's series index reaches into other datasets. A linked
    dataset's failure costs its records alone.

    Returns:
        The records, or ``None`` when the own dataset's builder failed.
    """
    guarded = kind not in _CORE_BUILDERS
    held: list[tuple[str, Sequence[AnyRecord]]] = []
    for link_id, dataset in datasets:
        if not guarded:
            held.append((link_id, build(dataset, asked, reader)))
            continue
        try:
            held.append((link_id, build(dataset, asked, reader)))
        except Exception as exc:
            if not link_id:
                errors.append(f"{kind}: {exc}")
                return None
            errors.append(f"{kind} (link {link_id!r}): {exc}")
    return held


def _merged(
    held: Sequence[tuple[str, Sequence[AnyRecord]]],
    *,
    fills_linked_only: bool,
    nearest_rows: bool,
) -> list[AnyRecord]:
    """Join each artifact's records across *held*, an own record alone kept as read.

    With *nearest_rows*, a key counts as present by the nearest dataset that
    holds a row of it, as :func:`~._linked.nearest_rows_present` narrows it.
    """
    if len(held) == 1:
        return list(held[0][1])
    return [
        group[0][1]
        if len(group) == 1 and not group[0][0]
        else merged_record(
            nearest_rows_present(group) if nearest_rows else group,
            fills_linked_only=fills_linked_only,
        )
        for group in grouped_by_ref(held)
    ]


def _merged_feature_records(
    ds: Dataset, asked: InventoryScope, reader: IndexReader, links: _Links
) -> Sequence[AnyRecord]:
    """Every feature run of *ds* and of the datasets it links, measured over *ds*.

    A run that a link holds is measured over the target of *ds*
    (:func:`feature_run_target`) with the parameters the lookup reads
    (:func:`~.params.linked_run_root`), and its drift is computed on *ds*, so an
    entry whose tracks *ds* converted again reads as drifted. The own runs share
    that target and one :class:`DriftReads`.
    """
    target = narrow_target(reportable_universe(ds, asked.tracks_run_id), asked.selector)
    drift_reads = _drift_reads(ds, reader)
    held: list[tuple[str, list[ArtifactRecord[Entry]]]] = [
        ("", _feature_runs(ds, reader, target, drift_reads))
    ]
    held.extend(
        (link_id, _feature_records(linked, asked, reader))
        for link_id, linked in links.of("feature")
    )
    if len(held) == 1:
        return held[0][1]
    records: list[AnyRecord] = []
    for group in grouped_by_ref(held):
        link_id, first = group[0]
        if len(group) == 1 and not link_id:
            records.append(first)
            continue
        rows = frozenset[Entry]().union(*(record.rows for _, record in group))
        params_root = linked_run_root(ds, first.name, first.run_id)
        if not link_id and first.run_root == params_root and params_root.exists():
            # The own run directory is kept and holds the params the lookup
            # reads, so its drift over *ds* stands.
            remeasure = Remeasure(
                target=feature_run_target(target, first.params, rows),
                drift=first.drift,
                refit=first.refit,
            )
        else:
            read = read_run_params(params_root)
            refit = _drifted_fit(ds, first.name, first.run_id, read.params, drift_reads)
            remeasure = Remeasure(
                target=feature_run_target(target, read.params, rows),
                drift=drifted_entries(
                    ds,
                    first.name,
                    first.run_id,
                    read.params,
                    drift_reads,
                    fit_drifted=refit is not None,
                ),
                refit=refit,
                params=read,
            )
        records.append(
            merged_record(group, fills_linked_only=True, remeasure=remeasure)
        )
    return records
