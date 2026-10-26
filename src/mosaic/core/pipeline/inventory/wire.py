"""An inventory as bounded JSON: the models and the one function that fills them.

:func:`inventory_wire` projects a :class:`~.model.DatasetInventory` onto
:class:`InventoryWire`. The CLI's ``mosaic inventory --json`` is that model
dumped, and other callers serve it unchanged. A projection narrows an unscoped
inventory to the kinds, entries and tracks variant asked for, re-deriving each
narrowed status through :func:`~.model.reclassify`, so one held view answers
every narrower question.

Entry lists are bounded. A coverage lists up to :data:`MISSING_SAMPLE_WIRE`
missing keys beside its counts, and a listing is paged by
``(kind, name, run_id, ref)`` with an opaque cursor. The ref is part of the key
because label-series records share their name and an empty run id.

The models declare :class:`~mosaic.core.strict_model.StrictModel`, the
configuration of the graph's models, so this package imports nothing from
``graph/``. ``graph/wire.py`` imports :class:`CoverageWire` from here for a plan.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Final

from pydantic import TypeAdapter, ValidationError

from mosaic.core.entry import Entry
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.error_kind import ClassifiedError, ErrorKind
from mosaic.core.scope import Scope
from mosaic.core.strict_model import StrictModel

from .model import (
    ArtifactKind,
    ArtifactRecord,
    ArtifactRef,
    ArtifactStatus,
    Coverage,
    DatasetInventory,
    FeatureRunRef,
    FramePredictionsRef,
    FrameRunRef,
    FrameSelectionRef,
    LabelSeriesRef,
    LabelsVariantRef,
    MediaDerivativeRef,
    MediaVariantRef,
    PredictionScoreRef,
    PreparedDatasetRef,
    TrackerRunRef,
    TracksVariantRef,
    TrainedModelRef,
    reclassify,
)
from .params import (
    ParamsState,
    RunKey,
    RunParams,
    TracksVariantWalk,
    read_run_params,
)
from .scan import feature_run_target, narrow_target, reportable_universe

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "MISSING_SAMPLE_WIRE",
    "ArtifactDetailWire",
    "ArtifactNotFound",
    "ArtifactWire",
    "CoverageWire",
    "InventoryQueryInvalid",
    "InventoryWire",
    "KeySampleWire",
    "ResolvedWire",
    "coverage_wire",
    "inventory_wire",
    "key_sample",
    "ref_address",
]

MISSING_SAMPLE_WIRE: Final = 20
"""How many keys a sample lists before its count stands in for the rest.

A full entry list for a 2,000-sequence dataset under a ten-step graph is
megabytes on every preview. The sample names enough keys to act on, and
``has_more`` says whether it is the whole list.
"""

type AnyKey = str | tuple[str, ...]
"""A coverage key of any kind: an entry, an entry and camera, or one string."""

type _SortKey = tuple[str, str, str, str]
"""A listing's order: ``(kind, name, run_id, ref)``."""

_CURSOR: Final[TypeAdapter[_SortKey]] = TypeAdapter(tuple[str, str, str, str])
_PARAMS: Final[TypeAdapter[dict[str, JsonValue]]] = TypeAdapter(dict[str, JsonValue])
_INPUTS: Final[TypeAdapter[list[JsonValue]]] = TypeAdapter(list[JsonValue])

_ENTRY_KEYED: Final = (
    FeatureRunRef,
    TracksVariantRef,
    LabelsVariantRef,
    TrackerRunRef,
    FrameRunRef,
    FramePredictionsRef,
    FrameSelectionRef,
    MediaVariantRef,
)
"""The refs whose coverage keys begin with an entry, which a scope narrows."""

_SCANNED_IN_SCOPE: Final = (
    FrameRunRef,
    FramePredictionsRef,
    FrameSelectionRef,
    MediaVariantRef,
)
"""The refs whose scan reads only the index rows that a scope's entries name.

Their rows, orphans and drift are measured over those rows, so a projection
narrows them with the target.
"""


class InventoryQueryInvalid(ClassifiedError, ValueError):
    """A listing asked with a limit below one or a cursor this module did not write."""

    error_kind: ClassVar[ErrorKind] = "invalid"

    def __init__(self, message: str, *, field: str) -> None:
        super().__init__(message)
        self.field: str = field

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the argument that is invalid."""
        return {"field": self.field}


class ArtifactNotFound(ClassifiedError, LookupError):
    """A ref that names no artifact of the inventory, or none the filters keep."""

    error_kind: ClassVar[ErrorKind] = "not_found"

    def __init__(self, ref: str) -> None:
        super().__init__(f"no artifact {ref!r} in this inventory")
        self.ref: str = ref

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the ref that was looked up."""
        return {"ref": self.ref}


class KeySampleWire(StrictModel):
    """A set of keys, counted, with a sorted sample.

    Attributes:
        count: How many keys the set holds.
        sample: Up to :data:`MISSING_SAMPLE_WIRE` keys, sorted, each as its parts.
        has_more: Whether the set holds more keys than the sample lists.
    """

    count: int
    sample: tuple[tuple[str, ...], ...]
    has_more: bool


class CoverageWire(StrictModel):
    """Which of its target keys an artifact holds, summarized.

    Attributes:
        covers_all: A single artifact answers for every key, as a global fit or
            a trained model does.
        covered: How many target keys it holds.
        target: How many keys it was measured over.
        missing_sample: Up to :data:`MISSING_SAMPLE_WIRE` missing keys, sorted,
            each as its parts. An entry is ``[group, sequence]``, and a frame
            run's or a media variant's key adds its camera. A revision, a
            ``video_uuid`` or a run id is one string.
        has_more: Whether more keys are missing than the sample lists.
    """

    covers_all: bool
    covered: int
    target: int
    missing_sample: tuple[tuple[str, ...], ...]
    has_more: bool


class ResolvedWire(StrictModel):
    """One upstream a feature run's reference pinned to, from ``_resolved``.

    Attributes:
        where: The reference's site, such as ``inputs[0]`` or ``inputs[tracks]``.
        feature: The upstream's feature storage, or ``tracks`` or ``labels``.
        run_id: The upstream run, ``""`` for the unlabeled tracks tables, or
            ``None`` for a reference that was never pinned.
    """

    where: str
    feature: str
    run_id: str | None


class ArtifactDetailWire(StrictModel):
    """One artifact in depth: its recorded params and the keys behind its counts.

    Attributes:
        params_finding: Why the run's ``params.json`` was not read, ``""`` when
            it was read or the kind records none.
        inputs: The run's recorded ``_inputs``.
        resolved: Which upstream each of its references pinned to.
        frame_range: The recorded ``(frame_start, frame_end)``.
        overlap_frames: The recorded overlap width.
        scope_entries: How many entries its latest invocation's scope named.
        mosaic_version: The toolkit version that wrote it, ``""`` when unknown.
        identity_scheme: The hashing contract its run id was minted under.
        rows: How many keys its index rows name.
        orphan_rows: The index rows whose output is missing, listed only when
            some output matched an entry.
        orphan_files: The outputs no index row names.
        drift: The entries whose recorded sources have moved.
        extra: The per-kind key sets that ``notes`` counts.
    """

    params_finding: str
    inputs: tuple[JsonValue, ...]
    resolved: tuple[ResolvedWire, ...]
    frame_range: tuple[int | None, int | None]
    overlap_frames: int
    scope_entries: int
    mosaic_version: str
    identity_scheme: str
    rows: int
    orphan_rows: KeySampleWire
    orphan_files: KeySampleWire
    drift: KeySampleWire
    extra: dict[str, KeySampleWire]


class ArtifactWire(StrictModel):
    """One artifact of a dataset, its status and its coverage.

    Attributes:
        kind: Which kind of artifact it is.
        name: A storage name, a label kind, an op kind or a series.
        run_id: Its run id, ``""`` for a media derivative or a label series.
        status: What its coverage and consistency add up to, re-derived over
            the entries asked for.
        coverage: Which of its target keys it holds.
        drift: How many entries' recorded sources have moved.
        params_state: Whether its ``params.json`` was read, never written, or
            unreadable. ``absent`` for a kind that records none.
        params: A feature run's recorded ``_params``, else ``None``. ``None``
            also when a listing left params out, and ``params_state`` then
            still says whether they were read.
        tracks_variants: The labeled tracks variants a feature run read,
            directly or through its upstreams, sorted. Empty for a run over the
            unlabeled tables alone. ``None`` for another kind, or when a
            ``params.json`` on the walk is absent, unreadable or older than its
            ``_resolved`` block.
        upstreams: The run ids it consumed.
        execution_id: The attempt that last wrote its ``params.json``, ``""``
            when unrecorded. It may name a cache hit.
        started_at: When it started, ISO 8601 UTC, ``""`` when unrecorded.
        finished_at: When it finished, ``""`` when unrecorded.
        notes: The size of each per-kind key set, such as
            ``frame_axis_mismatch`` on a tracks variant.
        ref: The address that names this artifact alone, for a listing's
            ``ref``: its kind and the parts of its key, joined by ``/``.
        detail: The artifact in depth, when the listing asked for one ``ref``.
    """

    kind: ArtifactKind
    name: str
    run_id: str
    status: ArtifactStatus
    coverage: CoverageWire
    drift: int
    params_state: ParamsState
    params: dict[str, JsonValue] | None
    tracks_variants: tuple[str, ...] | None
    upstreams: tuple[str, ...]
    execution_id: str
    started_at: str
    finished_at: str
    notes: dict[str, int]
    ref: str
    detail: ArtifactDetailWire | None = None


class InventoryWire(StrictModel):
    """What a dataset holds, as one page of artifacts.

    The run roots, the index paths and the dataset root are left out. They are
    paths on the machine that read the dataset.

    Attributes:
        artifacts: The page, ordered by kind, name, run id and ref.
        unavailable_kinds: The kinds asked for that this inventory cannot
            answer, because no producer of theirs was imported.
        errors: The scan's errors, as ``DatasetInventory.errors`` lists them.
        total: How many artifacts match the query, on every page.
        next_cursor: The cursor of the next page, ``None`` on the last.
    """

    artifacts: tuple[ArtifactWire, ...]
    unavailable_kinds: tuple[ArtifactKind, ...]
    errors: tuple[str, ...]
    total: int
    next_cursor: str | None = None


def key_sample(
    keys: Iterable[AnyKey], *, limit: int = MISSING_SAMPLE_WIRE
) -> KeySampleWire:
    """Return *keys* counted, with a sorted sample of at most *limit* keys' parts."""
    ordered = sorted((key,) if isinstance(key, str) else key for key in keys)
    return KeySampleWire(
        count=len(ordered),
        sample=tuple(ordered[:limit]),
        has_more=len(ordered) > limit,
    )


def coverage_wire(
    coverage: Coverage[AnyKey], *, sample: int = MISSING_SAMPLE_WIRE
) -> CoverageWire:
    """Return *coverage* as counts and a sorted sample of what is missing.

    A single artifact answering for every key covers its whole target and
    reports nothing missing. *sample* bounds the missing keys listed, and a
    caller whose reader needs only the counts passes fewer.
    """
    target = len(coverage.target)
    if coverage.covers_all:
        return CoverageWire(
            covers_all=True,
            covered=target,
            target=target,
            missing_sample=(),
            has_more=False,
        )
    missing = key_sample(coverage.missing, limit=sample)
    return CoverageWire(
        covers_all=False,
        covered=len(coverage.covered),
        target=target,
        missing_sample=missing.sample,
        has_more=missing.has_more,
    )


def ref_address(ref: ArtifactRef) -> str:
    """Return the address that names *ref*: its kind and its key's parts, joined by ``/``.

    Every part is one path component on disk, a storage name, a run id or a
    series key, so ``/`` separates them unambiguously. A label series keeps its
    origin dataset's uuid, ``""`` for this one.
    """
    match ref:
        case FeatureRunRef():
            parts = (ref.name, ref.run_id)
        case TracksVariantRef() | MediaVariantRef() | PredictionScoreRef():
            parts = (ref.run_id,)
        case LabelsVariantRef():
            parts = (ref.label_kind, ref.run_id)
        case LabelSeriesRef():
            parts = (ref.series, ref.key, ref.origin_uuid)
        case TrackerRunRef():
            parts = (ref.root_key, ref.run_id)
        case FrameRunRef():
            parts = (ref.method, ref.run_id)
        case FramePredictionsRef():
            parts = (ref.run_id, ref.frames_run_id)
        case FrameSelectionRef():
            parts = (ref.run_id, ref.source_tracks_run_id)
        case TrainedModelRef() | PreparedDatasetRef():
            parts = (ref.op_kind, ref.run_id)
        case MediaDerivativeRef():
            parts = (ref.target,)
    return "/".join((ref.kind, *parts))


@dataclass(frozen=True, slots=True)
class _Matched:
    """One record that a query keeps, with what the query derived for it."""

    record: ArtifactRecord[AnyKey]
    address: str
    coverage: Coverage[AnyKey]
    status: ArtifactStatus

    def sort_key(self) -> _SortKey:
        """Return the listing's order for this record."""
        record = self.record
        return (record.ref.kind, record.name, record.run_id, self.address)

    def wire(
        self,
        walk: TracksVariantWalk,
        *,
        detailed: bool,
        with_params: bool = True,
        missing_sample: int = MISSING_SAMPLE_WIRE,
    ) -> ArtifactWire:
        """Return the record as an :class:`ArtifactWire`, with its detail if asked.

        *with_params* false leaves out the recorded params, and *missing_sample*
        bounds the missing keys of the coverage.
        """
        record = self.record
        params = record.params
        variants = _variants_of(record, walk)
        return ArtifactWire(
            kind=record.ref.kind,
            name=record.name,
            run_id=record.run_id,
            status=self.status,
            coverage=coverage_wire(self.coverage, sample=missing_sample),
            drift=len(record.drift),
            params_state=record.params_state,
            params=(
                _PARAMS.validate_python(params.params)
                if params and with_params
                else None
            ),
            tracks_variants=tuple(sorted(variants)) if variants is not None else None,
            upstreams=record.upstreams,
            execution_id=params.execution_id if params else "",
            started_at=record.started_at,
            finished_at=record.finished_at,
            notes={name: len(keys) for name, keys in sorted(record.extra.items())},
            ref=self.address,
            detail=_detail(record) if detailed else None,
        )


def inventory_wire(
    ds: Dataset,
    inventory: DatasetInventory,
    *,
    kinds: Iterable[ArtifactKind] | None = None,
    scope: Scope | None = None,
    tracks_run_id: str | None = None,
    status_not: Iterable[ArtifactStatus] | None = None,
    limit: int | None = None,
    cursor: str | None = None,
    ref: str | None = None,
    with_params: bool = True,
    missing_sample: int = MISSING_SAMPLE_WIRE,
) -> InventoryWire:
    """Project *inventory* onto the wire, narrowed and paged as asked.

    With every argument at its default, the wire lists every record of
    *inventory* with the status the scan gave it.

    Args:
        ds: The dataset *inventory* was read from. A feature run's tracks
            variants, a tracks variant's entries and a ref's params are read
            from it. Never written to.
        inventory: The inventory to project, usually unscoped and held.
        kinds: Keep these kinds. A kind *inventory* did not read is listed in
            ``unavailable_kinds``.
        scope: Narrow each entry-keyed artifact's target to the entries this
            selector names, as :func:`~.scan.inventory` narrows it under the
            same scope. A feature run is narrowed as ``build_manifest``
            narrows, by groups, sequences and entries. Another entry-keyed
            kind is narrowed by its entries alone, and the rows, orphans and
            drift of a frame run, a predictions run, a frame selection or a
            media variant with it. A kind not keyed by entry is reported whole.
        tracks_run_id: Keep the feature runs whose tracks variants contain it,
            each measured over that variant's entries. ``""`` names the
            unlabeled tables and keeps the runs whose tracks variants are known
            and empty. A run over labeled and unlabeled tables records only the
            labeled variants, so ``""`` does not keep it.
        status_not: Leave out the artifacts whose narrowed status is one of
            these.
        limit: The page size, at least 1. ``None`` returns every artifact.
        cursor: The ``next_cursor`` of the previous page. The page starts after
            the artifact it names, so a listing that changes between pages
            neither repeats nor skips an artifact that stays.
        ref: Return only the artifact with this :func:`ref_address`, with its
            detail. The other filters still apply.
        with_params: Whether each feature run lists its recorded params. A
            listing that leaves them out keeps ``params_state``.
        missing_sample: The most missing keys each coverage lists.

    Returns:
        One page of the artifacts that match.

    Raises:
        InventoryQueryInvalid: *limit* is below 1, or *cursor* is not one this
            function wrote.
        ArtifactNotFound: *ref* names no artifact that the filters keep.
    """
    if limit is not None and limit < 1:
        raise InventoryQueryInvalid(
            f"limit must be at least 1, got {limit}", field="limit"
        )
    after = _cursor_key(cursor) if cursor is not None else None
    wanted = frozenset(kinds) if kinds is not None else None
    excluded = frozenset(status_not or ())
    universe = (
        reportable_universe(ds, tracks_run_id) if tracks_run_id is not None else None
    )
    narrowing = scope is not None or universe is not None
    # What a feature run is re-measured from: *universe* when a tracks variant
    # names it, else every entry, read once on first need. Kept apart from
    # *universe*, which also says whether a tracks variant narrows the other
    # records. A feature record then cannot change how they read.
    asked_universe = universe
    walk = TracksVariantWalk(ds, _held_params(inventory))

    matched: list[_Matched] = []
    for record in inventory.records:
        if wanted is not None and record.ref.kind not in wanted:
            continue
        address = ref_address(record.ref)
        if ref is not None and address != ref:
            continue
        is_feature = isinstance(record.ref, FeatureRunRef)
        if tracks_run_id is not None and is_feature:
            if not _read_variant(_variants_of(record, walk), tracks_run_id):
                continue
        if narrowing and isinstance(record.ref, _ENTRY_KEYED):
            if is_feature:
                # Re-measured from the asked entries rather than narrowed from
                # the scan's target, which was already cut to the entries the
                # run was asked for: narrowing that again can leave it empty,
                # and an empty target reads as complete.
                if asked_universe is None:
                    asked_universe = reportable_universe(ds, None)
                coverage = _declared(record, narrow_target(asked_universe, scope))
            else:
                coverage = _narrowed_by_entries(record.coverage, scope)
            record = _rows_in_scope(record, scope)
            status = reclassify(record, coverage)
        else:
            coverage, status = record.coverage, record.status
        if status in excluded:
            continue
        matched.append(_Matched(record, address, coverage, status))

    if ref is not None and not matched:
        raise ArtifactNotFound(ref)
    matched.sort(key=_Matched.sort_key)
    remaining = [m for m in matched if after is None or m.sort_key() > after]
    page = remaining if limit is None else remaining[:limit]
    more = len(page) < len(remaining)
    return InventoryWire(
        artifacts=tuple(
            m.wire(
                walk,
                detailed=ref is not None,
                with_params=with_params,
                missing_sample=missing_sample,
            )
            for m in page
        ),
        unavailable_kinds=_unavailable(inventory, wanted),
        errors=inventory.errors,
        total=len(matched),
        next_cursor=_cursor_of(page[-1].sort_key()) if more else None,
    )


def _held_params(inventory: DatasetInventory) -> dict[RunKey, RunParams | None]:
    """Return the params of every feature run that *inventory* holds, by run."""
    return {
        (record.name, record.run_id): record.params
        for record in inventory.records
        if isinstance(record.ref, FeatureRunRef)
    }


def _variants_of(
    record: ArtifactRecord[AnyKey], walk: TracksVariantWalk
) -> frozenset[str] | None:
    """Return the tracks variants that a feature run read, ``None`` for another kind."""
    if not isinstance(record.ref, FeatureRunRef):
        return None
    return walk(record.name, record.run_id)


def _read_variant(variants: frozenset[str] | None, tracks_run_id: str) -> bool:
    """Return whether a run whose walk returned *variants* read *tracks_run_id*.

    ``run_feature`` records no tracks edge for the unlabeled tables, so ``""``
    matches a known empty set.
    """
    if variants is None:
        return False
    return tracks_run_id in variants or (tracks_run_id == "" and not variants)


def _entries_of(keys: frozenset[AnyKey]) -> frozenset[Entry]:
    """Return the entries that begin *keys*."""
    return frozenset((key[0], key[1]) for key in keys if not isinstance(key, str))


def _declared(
    record: ArtifactRecord[AnyKey], asked: frozenset[Entry]
) -> Coverage[AnyKey]:
    """Return *record*'s coverage of *asked*, cut as the scan cuts it.

    :func:`~.scan.feature_run_target` is the rule, shared with the scan.
    """
    target = feature_run_target(asked, record.params, _entries_of(record.rows))
    return Coverage[AnyKey](
        target=frozenset[AnyKey](target),
        present=record.coverage.present,
        covers_all=record.coverage.covers_all,
    )


def _narrowed_by_entries(
    coverage: Coverage[AnyKey], scope: Scope | None
) -> Coverage[AnyKey]:
    """Return *coverage* over the target keys whose entry *scope*'s entries name.

    The scan narrows a variant's target by ``entry_pairs`` alone, and this
    matches it.
    """
    pairs = scope.entry_pairs if scope is not None else None
    entries = _entries_of(coverage.target)
    return _kept(coverage, entries if pairs is None else entries & pairs)


def _rows_in_scope(
    record: ArtifactRecord[AnyKey], scope: Scope | None
) -> ArtifactRecord[AnyKey]:
    """Return *record* with its rows, orphans and drift narrowed to *scope*'s entries.

    Only a :data:`_SCANNED_IN_SCOPE` record is narrowed. Every other kind's scan
    measures its orphans and drift over all of its rows, and
    :func:`~.model.reclassify` takes them from the record unchanged.
    """
    pairs = scope.entry_pairs if scope is not None else None
    if pairs is None or not isinstance(record.ref, _SCANNED_IN_SCOPE):
        return record

    def kept(keys: frozenset[AnyKey]) -> frozenset[AnyKey]:
        return frozenset(
            key
            for key in keys
            if not isinstance(key, str) and (key[0], key[1]) in pairs
        )

    return replace(
        record,
        rows=kept(record.rows),
        orphan_rows=kept(record.orphan_rows),
        orphan_files=kept(record.orphan_files),
        drift=tuple(entry for entry in record.drift if entry in pairs),
    )


def _kept(coverage: Coverage[AnyKey], kept: frozenset[Entry]) -> Coverage[AnyKey]:
    """Return *coverage* over the target keys whose entry is in *kept*."""
    return Coverage[AnyKey](
        target=frozenset(
            key
            for key in coverage.target
            if not isinstance(key, str) and (key[0], key[1]) in kept
        ),
        present=coverage.present,
        covers_all=coverage.covers_all,
    )


def _detail(record: ArtifactRecord[AnyKey]) -> ArtifactDetailWire:
    """Return *record* in depth, re-reading a feature run's ``params.json`` for its finding."""
    finding = ""
    params = record.params
    if isinstance(record.ref, FeatureRunRef) and record.run_root is not None:
        read = read_run_params(record.run_root)
        finding, params = read.finding, read.params
    return ArtifactDetailWire(
        params_finding=finding,
        inputs=tuple(_INPUTS.validate_python(params.inputs)) if params else (),
        resolved=tuple(
            ResolvedWire(where=item.where, feature=item.feature, run_id=item.run_id)
            for item in params.resolved
        )
        if params
        else (),
        frame_range=params.frames() if params else (None, None),
        overlap_frames=params.overlap_frames if params else 0,
        scope_entries=len(params.entry_scope()) if params else 0,
        mosaic_version=params.mosaic_version if params else "",
        identity_scheme=record.identity_scheme,
        rows=len(record.rows),
        orphan_rows=key_sample(record.orphan_rows),
        orphan_files=key_sample(record.orphan_files),
        drift=key_sample(record.drift),
        extra={name: key_sample(keys) for name, keys in sorted(record.extra.items())},
    )


def _unavailable(
    inventory: DatasetInventory, wanted: frozenset[ArtifactKind] | None
) -> tuple[ArtifactKind, ...]:
    """Return the kinds asked for that *inventory* does not answer, sorted.

    A kind the inventory did not read is unavailable to this projection, which
    reports it rather than an empty list of its artifacts.
    """
    if wanted is None:
        return tuple(sorted(inventory.unavailable_kinds))
    answered = inventory.scope.kinds - inventory.unavailable_kinds
    return tuple(sorted(wanted - answered))


def _cursor_of(key: _SortKey) -> str:
    """Return the opaque cursor that starts a page after *key*."""
    text = json.dumps(list(key), separators=(",", ":"))
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def _cursor_key(cursor: str) -> _SortKey:
    """Return the sort key *cursor* names, refusing one this module did not write."""
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        return _CURSOR.validate_json(raw)
    except (binascii.Error, ValueError, ValidationError):
        raise InventoryQueryInvalid(
            f"cursor {cursor!r} is not a cursor this inventory wrote", field="cursor"
        ) from None
