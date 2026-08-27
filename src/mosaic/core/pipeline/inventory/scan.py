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
    drifted_entries,
    feature_index,
    feature_index_path,
    feature_run_root,
)
from mosaic.core.pipeline.labels_index import labels_index_path, read_labels_index
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.tracks_index import (
    drifted_media_entries,
    read_tracks_index,
    select_variant_rows,
    tracks_index_path,
)

from ._read import IndexReader
from .contributors import inventory_contributor, registered_inventory_kinds
from .media import media_derivative_record
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
)
from .params import read_run_params

from collections.abc import Callable

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from mosaic.core.dataset import Dataset

__all__ = [
    "GLOBAL_ENTRY",
    "GLOBAL_MARKER",
    "CORE_KINDS",
    "entry_universe",
    "inventory",
    "narrow_target",
    "parquet_is_readable",
    "reportable_universe",
    "run_covers",
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
    {"feature", "tracks-variant", "labels-variant", "media-derivative"}
)
"""The kinds ``core`` can report on by itself, with no producer registered."""


def parquet_is_readable(path: Path) -> bool:
    """Is this parquet whole? Footer-only, so it costs one seek rather than a read.

    A parquet's footer is written last, so a file torn by an interrupted write
    has no readable metadata. That is cheap enough to run per file on a gate
    while still catching the failure presence alone cannot.
    """
    import pyarrow.parquet as pq

    try:
        # pyarrow ships no type stubs, the same gap ``track_universe`` documents
        # at its own ``read_schema`` call. Only whether the footer parsed
        # matters, so the result is discarded rather than described.
        pq.read_metadata(path)  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
    except Exception:
        return False
    return True


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
    frame = select_variant_rows(read_tracks_index(ds), tracks_run_id)
    found: set[Entry] = set()
    for record in index_records(frame):
        stored = str(record.get("abs_path", ""))
        if stored and ds.resolve_path(stored).exists():
            found.add((str(record.get("group", "")), str(record.get("sequence", ""))))
    return frozenset(found)


def narrow_target(
    universe: frozenset[Entry],
    *,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    entries: Iterable[Entry] | None = None,
) -> frozenset[Entry]:
    """Narrow the entry universe by a scope restriction.

    The same intersecting ``groups`` and ``sequences`` and ``entries`` filter
    ``build_manifest`` applies, so what is measured matches what would actually
    be processed. ``None`` or empty means no restriction on that axis.
    """
    wanted_groups = set(groups) if groups else None
    wanted_sequences = set(sequences) if sequences else None
    wanted_entries = set(entries) if entries else None
    return frozenset(
        (group, sequence)
        for group, sequence in universe
        if (wanted_groups is None or group in wanted_groups)
        and (wanted_sequences is None or sequence in wanted_sequences)
        and (wanted_entries is None or (group, sequence) in wanted_entries)
    )


def _entries_held(stems: set[str], candidates: frozenset[Entry]) -> set[Entry]:
    """Which of *candidates* the output stems account for.

    An entry's output is usually one file named exactly ``make_entry_key(...)``,
    but not always: a feature that splits an entry by individual writes
    ``<entry key>__id0``, ``__id1`` and so on, which is how ``kpms-apply`` and
    its kin store their results. Matching the key exactly reads those runs as
    holding nothing at all, and their index rows then look like rows with no
    output -- damage, where the outputs are in fact right there under a
    convention the exact match did not know about.

    Longest key first, so an entry whose key is a prefix of another's cannot
    claim the other's files.
    """
    by_key = {
        make_entry_key(group, sequence): (group, sequence)
        for group, sequence in candidates
    }
    held: set[Entry] = set()
    for key in sorted(by_key, key=len, reverse=True):
        prefix = f"{key}__"
        if any(stem == key or stem.startswith(prefix) for stem in stems):
            held.add(by_key[key])
    return held


def run_covers(
    run_root: Path,
    target: frozenset[Entry],
    known: frozenset[Entry] = frozenset(),
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
    """
    if not run_root.exists():
        return Coverage(target=target, present=frozenset())
    stems = {
        path.stem for path in run_root.glob("*.parquet") if parquet_is_readable(path)
    }
    held = _entries_held(stems, target | known)
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


def _rows_of(frame: pd.DataFrame, run_id: str) -> frozenset[Entry]:
    """The ``(group, sequence)`` pairs one run's index rows name."""
    if frame.empty or "run_id" not in frame.columns:
        return frozenset()
    return frozenset(
        (str(record.get("group", "")), str(record.get("sequence", "")))
        for record in index_records(frame)
        if str(record.get("run_id", "")) == run_id
    )


def _finish_state(frame: pd.DataFrame, run_id: str) -> tuple[str, str, bool]:
    """``(started_at, finished_at, finished)`` for one run, from its index rows."""
    started, finished = "", ""
    for record in index_records(frame):
        if str(record.get("run_id", "")) != run_id:
            continue
        started = started or str(record.get("started_at", ""))
        finished = finished or str(record.get("finished_at", ""))
    return started, finished, bool(finished)


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
    knows is not.

    **Damage needs contradictory evidence, not missing evidence.** A run whose
    outputs this could attribute to no entry at all says nothing about whether
    its rows are honoured -- a feature storing its results as ``.npz`` beside a
    ``seq=<name>`` filename is doing so legitimately, and calling that a row with
    no output would report damage on a perfectly good run. So a row counts as
    orphaned only when *some* file was attributed and the index names more: the
    two then genuinely disagree. Attributing nothing reports the coverage
    shortfall and stops there.
    """
    attributed_something = bool(files) or coverage.covers_all
    return classify(
        satisfied=coverage.is_satisfied,
        any_covered=bool(coverage.covered) or coverage.covers_all,
        orphan_rows=attributed_something and bool(rows - files),
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
        frame = read_tracks_index(ds)
        return frozenset(
            (record.get("group", ""), record.get("sequence", ""))
            for record in index_records(frame)
            if record.get("abs_path", "")
            and ds.resolve_path(record["abs_path"]).exists()
        )


def _feature_records(
    ds: Dataset, scope: InventoryScope, reader: IndexReader
) -> list[ArtifactRecord[Entry]]:
    """Every feature run under ``features/<name>/<run_id>/``."""
    universe = reportable_universe(ds, scope.tracks_run_id)
    target = narrow_target(universe, entries=scope.entries)
    records: list[ArtifactRecord[Entry]] = []
    for name in feature_storages(ds):
        index_path = feature_index_path(ds, name)
        frame = reader.frame(index_path, lambda p=index_path: feature_index(p).read())
        if frame.empty or "run_id" not in frame.columns:
            continue
        for run_id in _run_ids(frame):
            run_root = feature_run_root(ds, name, run_id)
            rows = _rows_of(frame, run_id)
            # The run's own index rows are what make its outputs recognisable
            # when the target cannot name them -- see ``run_covers``.
            coverage = run_covers(run_root, target, known=rows)
            files = coverage.present
            started, finished_at, finished = _finish_state(frame, run_id)
            read = read_run_params(run_root)
            drift = drifted_entries(ds, name, run_id)
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
                    orphan_rows=rows - files,
                    orphan_files=files - rows,
                    drift=drift,
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
    """
    index_path = tracks_index_path(ds)
    frame = reader.frame(index_path, lambda: read_tracks_index(ds))
    if frame.empty or "run_id" not in frame.columns:
        return []
    records: list[ArtifactRecord[Entry]] = []
    for run_id in _run_ids(frame):
        rows = _rows_of(frame, run_id)
        target = frozenset(rows if scope.entries is None else rows & scope.entries)
        files = frozenset(
            entry for entry in rows if _variant_table_exists(ds, frame, run_id, entry)
        )
        coverage = Coverage(target=target, present=files)
        started, finished_at, finished = _finish_state(frame, run_id)
        drift = drifted_media_entries(ds, run_id)
        records.append(
            ArtifactRecord[Entry](
                ref=TracksVariantRef(run_id=run_id),
                name="tracks",
                run_id=run_id,
                coverage=coverage,
                status=_status_for(coverage, rows, files, drift, finished),
                index_path=index_path,
                rows=rows,
                orphan_rows=rows - files,
                orphan_files=frozenset(),
                drift=drift,
                started_at=started,
                finished_at=finished_at,
            )
        )
    return records


def _variant_table_exists(
    ds: Dataset, frame: pd.DataFrame, run_id: str, entry: Entry
) -> bool:
    """Does the table this row names resolve to a file that is there?"""
    for record in index_records(frame):
        if str(record.get("run_id", "")) != run_id:
            continue
        if (str(record.get("group", "")), str(record.get("sequence", ""))) != entry:
            continue
        stored = str(record.get("abs_path", ""))
        return bool(stored) and ds.resolve_path(stored).exists()
    return False


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
        for run_id in _run_ids(frame):
            rows = _rows_of(frame, run_id)
            target = frozenset(rows if scope.entries is None else rows & scope.entries)
            files = frozenset(
                entry
                for entry in rows
                if _variant_table_exists(ds, frame, run_id, entry)
            )
            coverage = Coverage(target=target, present=files)
            started, finished_at, finished = _finish_state(frame, run_id)
            records.append(
                ArtifactRecord[Entry](
                    ref=LabelsVariantRef(label_kind=kind, run_id=run_id),
                    name=kind,
                    run_id=run_id,
                    coverage=coverage,
                    status=_status_for(coverage, rows, files, (), finished),
                    index_path=index_path,
                    rows=rows,
                    orphan_rows=rows - files,
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
    entries: Iterable[Entry] | None = None,
    tracks_run_id: str | None = None,
) -> DatasetInventory:
    """What this dataset holds: every artifact, its identity and its coverage.

    Args:
        ds: The dataset to read. Never written to.
        kinds: Which artifact kinds to report. ``None`` means every kind the
            running process can report on, which depends on what has been
            imported -- see ``unavailable_kinds`` on the result.
        entries: Narrow coverage to these ``(group, sequence)`` pairs. ``None``
            measures against everything the dataset can process.
        tracks_run_id: Which tracks variant defines the entry universe. Pass the
            one the runs were made under; measuring a variant-pinned run against
            the whole index reads as permanently incomplete.

    Returns:
        A :class:`DatasetInventory`, which is a cache and never the record.
    """
    wanted = frozenset(kinds) if kinds is not None else _every_kind()
    scope = InventoryScope(
        kinds=wanted,
        entries=frozenset(entries) if entries is not None else None,
        tracks_run_id=tracks_run_id,
    )
    reader = IndexReader()
    records: list[AnyRecord] = []
    errors: list[str] = []
    unavailable: set[ArtifactKind] = set()

    for kind in sorted(wanted):
        if kind == "media-derivative":
            # Both targets, because they are independent verdicts with
            # independent derivatives: a playback transcode never satisfies an
            # analysis read, and reporting one would hide the other.
            for media_target in ("analysis", "playback"):
                records.append(media_derivative_record(ds, media_target, scope, reader))
            continue
        builder = _CORE_BUILDERS.get(kind)
        if builder is not None:
            records.extend(builder(ds, scope, reader))
            continue
        contributor = inventory_contributor(kind)
        if contributor is None:
            unavailable.add(kind)
            continue
        try:
            records.extend(contributor(ds, scope, reader))
        except Exception as exc:
            # One producer's failure costs its kind, never the whole answer: an
            # inventory that raises because one tracker root is malformed tells
            # a user nothing about the rest of their dataset.
            errors.append(f"{kind}: {exc}")

    return DatasetInventory(
        dataset_root=Path(ds.base_dir),
        scope=scope,
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

_CORE_BUILDERS: dict[ArtifactKind, _CoreBuilder] = {
    "feature": _feature_records,
    "tracks-variant": _variant_records,
    "labels-variant": _labels_records,
}


def _run_ids(frame: pd.DataFrame) -> list[str]:
    """Every run identifier in an index, sorted. Empty when the index has none."""
    if frame.empty or "run_id" not in frame.columns:
        return []
    return sorted({record.get("run_id", "") for record in index_records(frame)})
