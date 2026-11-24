"""Item 5.3: which sequences a run's state was *fitted* over.

An index row records that a sequence was **applied**. Nothing recorded which
sequences the fit itself consumed, and for one shape of feature the two genuinely
differ: a params-level fitter takes its training set from a pinned ``templates``
or ``model`` reference, so it is scope-free, and applying it to further sequences
reuses one ``run_id`` and one run root. Its index rows are apply-scope by
construction -- correctly so, since applying a trained classifier to new
sequences without retraining is the intended workflow -- but nothing said so.

**Why ``params.json`` could not already answer this.** It is written
unconditionally at the top of every run, *before* ``load_state`` and before the
fit gate, and its ``_scope`` is an apply scope. A second invocation at a wider
apply scope overwrites ``_scope.entries`` with the wider set, so that key is
"whichever ran last". ``_scope.asked`` accumulates the entries of every
invocation, which is the apply scope of them all and still not what was fitted.
This file is written only when ``fit()`` actually ran, so it stays what it says
it is.

**No index column, deliberately.** A fit-scope cell would hold the same value on
every row of a run -- a per-run property duplicated per row, which is the shape
item 4.4 rejects for compositions and rejects here for the same reason: no way to
tell a stale copy from a current one. The pairing is already derivable, and
:func:`fit_and_apply_scopes` is the one place that derives it.

**Whose scope is recorded.** A scope-dependent feature fits from its own stream,
so its fit scope is the scope it ran over. A scope-free fitter does not: its
training set is a pinned artifact (``templates``) written by another run, and the
scope it was *invoked* over says nothing about what that artifact holds. Its
record is therefore derived from the fit records of the artifacts it pins
(:func:`derive_fit_scope`), and is "unknown" when no pinned artifact can say.

A scope-dependent feature may also apply an artifact that another run fitted:
``evaluate-labels`` scores the test groups through a ``label_map`` that
``fit-label-map`` fitted on the training groups. Its ``entries`` stay what its
own stream held, the test groups, and ``via`` names the pinned artifact. What
that artifact was fitted over is the record in its own run, one hop away, so it
is not copied here, where it could go stale.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, Literal, TypeGuard, get_args

from mosaic.core.entry import Entry

from ._utils import ResolvedScope, atomic_write, now_iso
from .error_kind import ClassifiedError, ErrorKind
from .file_digest import MODEL_DIGEST_HEX, file_digest
from .identity_scheme import FEATURE_IDENTITY_SCHEME
from .identity_scheme import MARKER_NAME as IDENTITY_SCHEME_NAME
from .index import EntryCells, FitReference
from .inventory.params import RUN_PARAMS_NAME
from .markers import MARKER_PREFIX, is_atomic_temp
from .types import ArtifactSpec, DefinitionModel, Feature

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.params import Params
    from mosaic.core.json_value import JsonValue

__all__ = [
    "FIT_SCOPE_NAME",
    "FitEntryInputs",
    "FitScope",
    "FitSource",
    "FitSourceRef",
    "FitStateUnreadable",
    "derive_fit_scope",
    "fit_and_apply_scopes",
    "fit_references",
    "list_run_root",
    "read_fit_scope",
    "read_fit_state_digest",
    "read_located_fit_scope",
    "read_run_fit_scope",
    "records_fit_inputs",
    "reported_fit_entries",
    "saved_state_digest",
    "write_fit_scope",
]

FIT_SCOPE_NAME = "fit_scope.json"

FitSource = Literal["stream", "artifacts", "unknown", "legacy"]
"""Where a record's entries came from.

- ``stream``: a scope-dependent feature, fitted over the scope it ran on. Its
  ``via`` names the artifacts it applied that other runs fitted, and is empty
  when it pinned none.
- ``artifacts``: a scope-free fitter, whose entries are the union of the fit
  records of the artifacts it pins (``via``).
- ``unknown``: a scope-free fitter with no pinned artifact to ask, such as one
  that reads a directory of videos. Its entries are ``None``.
- ``legacy``: a record written before ``source`` existed.
"""


@dataclass(frozen=True, slots=True)
class FitSourceRef:
    """One pinned artifact a scope-free fit was derived from.

    Attributes:
        field: The params field that pins it.
        feature: Storage name of the upstream feature.
        run_id: The upstream run.
        known: False when that run has no fit record, or one whose entries are
            unknown, which makes the downstream's entries unknown too.
        state_digest: The digest of that fit's state as the fitting run read it,
            and ``""`` when the fit records none.
    """

    field: str
    feature: str
    run_id: str
    known: bool
    state_digest: str = ""


@dataclass(frozen=True, slots=True)
class FitEntryInputs:
    """The cells of one entry as a fit read it.

    Attributes:
        group: The entry's group.
        sequence: The entry's sequence.
        cells: The entry's roots, tracks and upstream-roots cells at fit time,
            the cells a feature row records.
    """

    group: str
    sequence: str
    cells: EntryCells


@dataclass(frozen=True, slots=True)
class FitScope:
    """What a run's saved state was fitted over.

    ``entries`` and ``tracks_variants`` are sorted, so a diff of two runs' files
    shows what changed rather than how the sets were iterated. ``fitted_at`` is
    provenance and never hashed; ``identity_scheme`` is what makes a record
    written under an older hashing contract recognizable as such.

    ``entries`` and the two variant sets are ``None`` when they are unknown --
    never read that as "fitted over nothing". For ``source="artifacts"`` the
    entries are an upper bound: the entries whose data could reach the fit. An
    ``xgboost`` run fits only the ``split == "train"`` rows of its templates, and
    its record still lists every entry the templates were drawn from.

    For ``source="stream"``, ``via`` lists the pinned artifacts the feature
    applied without fitting them. Those entries are not in ``entries``, and the
    artifact's run says what it was fitted over (``known`` is False when it
    cannot).

    ``state_digest`` is the digest of the files ``save_state`` wrote
    (:func:`saved_state_digest`), and ``""`` when unknown. It is recorded, never
    hashed into a run id.

    ``inputs`` lists the cells of each entry the fit read, taken at fit time and
    sorted by group and sequence. It is ``None`` for a record written without
    it, which differs from an empty tuple. ``execution_id`` is the attempt that
    fitted, and ``""`` when the record names none. Both are recorded and stay
    out of every run id.
    """

    scope_dependent: bool
    entries: tuple[tuple[str, str], ...] | None
    tracks_variants: tuple[str, ...] | None
    labels_variants: tuple[str, ...] | None
    fitted_at: str
    identity_scheme: str
    source: FitSource = "stream"
    via: tuple[FitSourceRef, ...] = ()
    state_digest: str = ""
    inputs: tuple[FitEntryInputs, ...] | None = None
    execution_id: str = ""


_RUN_BOOKKEEPING: Final = frozenset(
    {RUN_PARAMS_NAME, IDENTITY_SCHEME_NAME, FIT_SCOPE_NAME}
)
"""The run-level files ``run_feature`` writes in a run root."""

type RunRootListing = dict[str, tuple[int, int, int]]
"""Each file under a run root, by relative POSIX path: its inode, size and mtime."""


def list_run_root(run_root: Path) -> RunRootListing:
    """Each file under *run_root*, recursively, with its inode, size and ``mtime_ns``.

    The files the engine writes there are left out, because they can change while
    a fit saves its state, and they are not that state:

    - a top-level file named with ``MARKER_PREFIX`` (``markers.py``): the fit's
      claim, which its timer rewrites by rename every ``INFLIGHT_REFRESH_SECONDS``;
    - top-level ``params.json``, ``.identity_scheme`` and ``fit_scope.json``,
      which another invocation of the same run writes before it is refused the
      claim;
    - a temporary file of a write still in flight (``is_atomic_temp``), the
      claim's included. The temporary files of the save are renamed or removed
      by the time it returns.

    A file removed between the walk and its ``stat`` is left out. An absent
    *run_root* lists as empty.
    """
    listing: RunRootListing = {}
    if not run_root.is_dir():
        return listing
    for path in run_root.rglob("*"):
        try:
            if not path.is_file():
                continue
            stat = path.stat()
        except FileNotFoundError:
            continue
        relative = path.relative_to(run_root).as_posix()
        if _is_engine_file(relative):
            continue
        listing[relative] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
    return listing


def _is_engine_file(relative: str) -> bool:
    """Is *relative*, a path under a run root, a file the engine writes there?"""
    name = relative.rpartition("/")[2]
    if is_atomic_temp(name):
        return True
    top_level = name == relative
    return top_level and (name.startswith(MARKER_PREFIX) or name in _RUN_BOOKKEEPING)


def saved_state_digest(
    run_root: Path, before: RunRootListing, after: RunRootListing
) -> str:
    """The digest of the files a ``save_state`` created or changed in *run_root*.

    A file of *after* counts when *before* does not hold it or holds another
    inode, size or ``mtime_ns``. Each counted file contributes
    ``[relative POSIX path, file_digest(path)]``, and the digest is a blake2b of
    the JSON of that list sorted by path, ``MODEL_DIGEST_HEX`` hex characters
    wide. It therefore does not depend on listing order, and a save that writes
    nothing gives the digest of the empty list. A counted file removed before it
    is read is left out.

    Args:
        run_root: The run root the listings were taken of.
        before: :func:`list_run_root` just before ``save_state``.
        after: :func:`list_run_root` just after it.

    Returns:
        The digest, ``MODEL_DIGEST_HEX`` hex characters.
    """
    pairs: list[list[str]] = []
    for relative in sorted(after):
        if before.get(relative) == after[relative]:
            continue
        try:
            digest = file_digest(run_root / relative)
        except FileNotFoundError:
            continue
        pairs.append([relative, digest])
    payload = json.dumps(pairs).encode()
    return hashlib.blake2b(payload, digest_size=MODEL_DIGEST_HEX // 2).hexdigest()


class FitStateUnreadable(ClassifiedError):
    """A ``fit_scope.json`` that cannot be read, or whose content is malformed."""

    error_kind: ClassVar[ErrorKind] = "conflict"

    def __init__(self, *, feature: str, run_id: str, error: str) -> None:
        super().__init__(
            f"run {run_id} of {feature} holds a {FIT_SCOPE_NAME} that cannot be "
            f"read ({error}), so whether the fit this run reads has changed "
            "cannot be told: remove the run and fit it again"
        )
        self.feature: str = feature
        self.run_id: str = run_id

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the feature and the run whose record is unreadable."""
        return {"feature": self.feature, "run_id": self.run_id}


def read_fit_state_digest(run_root: Path) -> str:
    """The digest of the state the fit in *run_root* saved.

    Args:
        run_root: A run root, ``features/<feature>/<run_id>``.

    Returns:
        The recorded digest, and ``""`` when the run has no ``fit_scope.json``
        or a record without the ``state_digest`` key.

    Raises:
        FitStateUnreadable: The record cannot be read, is not a JSON object,
            holds a ``state_digest`` that is not a string, or holds a
            ``fit_inputs`` that is not an object with an ``entries`` list of
            objects.
    """
    fitted = read_fit_scope(run_root, strict=True)
    return "" if fitted is None else fitted.state_digest


def records_fit_inputs(feature: Feature) -> bool:
    """Does the run of *feature* save state derived from its stream or its fits?

    True when the instance's ``fits`` or ``scope_dependent`` is set: a fitter, or
    a feature such as ``evaluate-labels`` that computes its report from its
    stream and saves it. A feature given a ``model`` to apply sets neither, and
    records no fit inputs.
    """
    return bool(getattr(feature, "fits", False)) or feature.scope_dependent


def reported_fit_entries(feature: Feature) -> tuple[Entry, ...] | None:
    """The entries a fit pinned to its training set streams, as its feature reports them.

    A feature that declares ``fit_set_pinned_by`` sets the instance attribute
    ``fitted_entries`` in ``load_state`` when the load asks for a fit: every
    entry the fit streams, so ``feral``'s train and test entries both. The
    attribute is ``None`` after a load that found a fitted state, and a feature
    that pins no fit set reports nothing whatever its attributes hold.

    Returns:
        The entries sorted by group and sequence, or ``None`` when the feature
        declares no pinned fit set or reports no entries.
    """
    if getattr(feature, "fit_set_pinned_by", None) is None:
        return None
    reported: tuple[Entry, ...] | None = getattr(feature, "fitted_entries", None)
    return None if reported is None else tuple(sorted(reported))


def fit_references(params: Params) -> tuple[FitReference, ...]:
    """The pinned fits *params* name, each a state the run reads.

    A top-level field counts when it holds an ``ArtifactSpec`` or a
    ``DefinitionModel`` that names a feature and a run. Any other ``Result``
    names a run's per-entry outputs, which a dependency lookup reads, and is
    left out, as are a ``LabelsSource`` and a reference without a run id.

    Returns:
        The references, sorted by field.
    """
    references: list[FitReference] = []
    for field in sorted(type(params).model_fields):
        value: object = getattr(params, field)
        match value:
            case ArtifactSpec(
                feature=str(feature), run_id=str(run_id), pattern=str(pattern)
            ) if feature and run_id:
                references.append(FitReference(field, feature, run_id, pattern))
            case DefinitionModel(feature=str(feature), run_id=str(run_id)) if (
                feature and run_id
            ):
                references.append(FitReference(field, feature, run_id, None))
            case _:
                pass
    return tuple(references)


def derive_fit_scope(
    ds: Dataset,
    feature: Feature,
    resolved: ResolvedScope,
    artifact_paths: Mapping[str, Path],
    *,
    states: Mapping[str, str],
    inputs: tuple[FitEntryInputs, ...] | None,
    execution_id: str,
) -> FitScope:
    """The fit scope of a fit that just ran, from where its data came from.

    A scope-dependent feature fits from its own stream, so its record is the
    scope it resolved, and its ``via`` names the fits it pinned. A scope-free
    fitter fits from the fits its params pin -- the scope it was invoked over
    is only where it is *applied* -- so its record is the union of those fits'
    own records. That is transitive: each upstream recorded its own fit scope
    when it ran.

    Args:
        ds: The dataset the run belongs to.
        feature: The feature that just fitted.
        resolved: The scope the run resolved.
        artifact_paths: The files and roots ``_resolve_dependencies`` found, by
            params field. A fit reference with none fed nothing into the fit.
        states: The state digest of each fit reference as the run read it, by
            params field.
        inputs: The cells of each entry the fit read, or ``None`` when they
            could not be taken.
        execution_id: The attempt that fitted.

    Returns:
        The record to write. Its entries are ``None`` when no pinned fit exists
        (``source="unknown"``) or when some pinned fit has no known fit scope
        (``source="artifacts"`` with that upstream ``known=False``).
    """
    pinned = _pinned_fits(ds, feature, artifact_paths, states)
    via = tuple(ref for ref, _ in pinned)
    if feature.scope_dependent:
        scope = FitScope(
            scope_dependent=True,
            entries=tuple(sorted(resolved.entries)),
            tracks_variants=tuple(sorted(resolved.tracks_variants)),
            labels_variants=tuple(sorted(resolved.labels_variants)),
            fitted_at=now_iso(),
            identity_scheme=FEATURE_IDENTITY_SCHEME,
            source="stream",
            via=via,
        )
    elif not via:
        scope = _unknown(source="unknown", via=())
    elif not all(ref.known for ref in via):
        scope = _unknown(source="artifacts", via=via)
    else:
        entries: set[tuple[str, str]] = set()
        tracks: set[str] = set()
        labels: set[str] = set()
        for _, upstream in pinned:
            if upstream is not None:
                entries.update(upstream.entries or ())
                tracks.update(upstream.tracks_variants or ())
                labels.update(upstream.labels_variants or ())
        scope = FitScope(
            scope_dependent=False,
            entries=tuple(sorted(entries)),
            tracks_variants=tuple(sorted(tracks)),
            labels_variants=tuple(sorted(labels)),
            fitted_at=now_iso(),
            identity_scheme=FEATURE_IDENTITY_SCHEME,
            source="artifacts",
            via=via,
        )
    return replace(scope, inputs=inputs, execution_id=execution_id)


def _pinned_fits(
    ds: Dataset,
    feature: Feature,
    artifact_paths: Mapping[str, Path],
    states: Mapping[str, str],
) -> list[tuple[FitSourceRef, FitScope | None]]:
    """Each fit the params pin that fed the run, and its own fit record.

    The fits are the :func:`fit_references` of the params, sorted by field, that
    ``_resolve_dependencies`` found a file or root for. The record is ``None``
    for a run that has none. The fit is ``known`` only when its record holds
    entries and both variant sets.
    """
    pinned: list[tuple[FitSourceRef, FitScope | None]] = []
    for reference in fit_references(feature.params):
        if reference.field not in artifact_paths:
            continue
        upstream = read_run_fit_scope(ds, reference.feature, reference.run_id)
        known = (
            upstream is not None
            and upstream.entries is not None
            and upstream.tracks_variants is not None
            and upstream.labels_variants is not None
        )
        ref = FitSourceRef(
            field=reference.field,
            feature=reference.feature,
            run_id=reference.run_id,
            known=known,
            state_digest=states[reference.field],
        )
        pinned.append((ref, upstream))
    return pinned


def _unknown(*, source: FitSource, via: tuple[FitSourceRef, ...]) -> FitScope:
    """A scope-free record whose entries and variants cannot be known."""
    return FitScope(
        scope_dependent=False,
        entries=None,
        tracks_variants=None,
        labels_variants=None,
        fitted_at=now_iso(),
        identity_scheme=FEATURE_IDENTITY_SCHEME,
        source=source,
        via=via,
    )


def write_fit_scope(run_root: Path, scope: FitScope) -> None:
    """Record the scope a fit just consumed, in *run_root*.

    Call **after** ``save_state``, inside the branch that actually fitted. Not
    before ``fit()``: a fit that raises halfway would otherwise leave a record of
    a fit that never completed, which is worse than no record. Not beside
    ``write_identity_scheme`` either -- that one is written on every invocation
    including a pure cache hit, and this must not be.

    Overwriting is correct when it happens: a fit or a refit ran here, so the
    state being described is being replaced too.

    *scope* comes from :func:`derive_fit_scope`. Its unknown sets are written as
    JSON ``null``. Its ``inputs`` are written under ``fit_inputs`` sorted by
    group and sequence, and left out when ``None``. ``via`` is written sorted by
    field, so the file does not depend on the order of either.
    """
    payload: dict[str, object] = {
        "scope_dependent": scope.scope_dependent,
        "entries": (
            None
            if scope.entries is None
            else [list(entry) for entry in sorted(scope.entries)]
        ),
        "tracks_variants": (
            None if scope.tracks_variants is None else sorted(scope.tracks_variants)
        ),
        "labels_variants": (
            None if scope.labels_variants is None else sorted(scope.labels_variants)
        ),
        "fitted_at": scope.fitted_at,
        "identity_scheme": scope.identity_scheme,
        "source": scope.source,
        "via": [
            {
                "field": ref.field,
                "feature": ref.feature,
                "run_id": ref.run_id,
                "known": ref.known,
                "state_digest": ref.state_digest,
            }
            for ref in sorted(scope.via, key=lambda ref: ref.field)
        ],
        "state_digest": scope.state_digest,
        "execution_id": scope.execution_id,
    }
    if scope.inputs is not None:
        payload["fit_inputs"] = {
            "entries": [
                {
                    "group": entry.group,
                    "sequence": entry.sequence,
                    "roots": entry.cells.roots,
                    "tracks": entry.cells.tracks,
                    "inputs": entry.cells.inputs,
                }
                for entry in sorted(
                    scope.inputs, key=lambda entry: (entry.group, entry.sequence)
                )
            ]
        }
    atomic_write(
        run_root / FIT_SCOPE_NAME,
        lambda p: p.write_text(json.dumps(payload, indent=2, sort_keys=True)),
    )


def _items(value: object) -> list[object]:
    """*value* as a list of unconstrained items, or empty when it is not one.

    A record on disk is untrusted input: it may have been hand-edited, truncated
    by a full disk, or written by a future version. Every read below narrows
    before it converts, so a malformed file degrades to "unknown" -- which is
    what :func:`read_fit_scope` promises -- rather than raising inside a caller
    that only asked what a run was fitted over.
    """
    return value if _is_list(value) else []


def _is_list(value: object) -> TypeGuard[list[object]]:
    """Is *value* a JSON array? Its items stay unconstrained."""
    return isinstance(value, list)


def _is_object(value: object) -> TypeGuard[dict[object, object]]:
    """Is *value* a JSON object? Its keys and values stay unconstrained."""
    return isinstance(value, dict)


def _entry(value: object) -> tuple[str, str] | None:
    """One ``[group, sequence]`` pair, or ``None`` when it is not one."""
    pair = _items(value)
    if len(pair) != 2:
        return None
    return (str(pair[0]), str(pair[1]))


def read_run_fit_scope(
    ds: Dataset, storage: str, run_id: str, *, strict: bool = False
) -> FitScope | None:
    """The fit record of a run in the first dataset of the lookup that holds one.

    Args:
        ds: The dataset reading, whose ``features`` lookup order decides.
        storage: The run's feature storage directory.
        run_id: The run.
        strict: Raise for a record that cannot be read, as
            :func:`read_fit_scope` does.

    Returns:
        What :func:`read_fit_scope` returns for that run root, and ``None`` when
        no dataset holds a fit record of the run.

    Raises:
        FitStateUnreadable: *strict* is set and the record cannot be read.
    """
    located = read_located_fit_scope(ds, storage, run_id, strict=strict)
    return None if located is None else located[1]


def read_located_fit_scope(
    ds: Dataset, storage: str, run_id: str, *, strict: bool = False
) -> tuple[str, FitScope | None] | None:
    """The fit record of a run, and the link of the dataset that holds it.

    Args:
        ds: The dataset reading, whose ``features`` lookup order decides.
        storage: The run's feature storage directory.
        run_id: The run.
        strict: Raise for a record that cannot be read, as
            :func:`read_fit_scope` does.

    Returns:
        The link id of the first dataset of the lookup that holds a fit record
        of the run, ``""`` for *ds* itself, and what :func:`read_fit_scope`
        returns for that run root. ``None`` when no dataset holds one.

    Raises:
        FitStateUnreadable: *strict* is set and the record cannot be read.
    """
    from .index import feature_run_file_location

    located = feature_run_file_location(ds, storage, run_id, FIT_SCOPE_NAME)
    if located is None:
        return None
    link_id, path = located
    return link_id, read_fit_scope(path.parent, strict=strict)


def _unreadable(run_root: Path, error: str, *, strict: bool) -> None:
    """Refuse the record of *run_root* for *error* when *strict*.

    Otherwise return ``None``, the record read as none.

    Raises:
        FitStateUnreadable: *strict* is set.
    """
    if strict:
        raise FitStateUnreadable(
            feature=run_root.parent.name, run_id=run_root.name, error=error
        )


def read_fit_scope(run_root: Path, *, strict: bool = False) -> FitScope | None:
    """The fit record in *run_root*, or ``None`` when there is none.

    ``None`` means **unknown**, not "fitted over nothing": it covers a run that
    predates this record and a feature that has no fit phase at all. A caller
    must not read it as an empty fit scope -- that would report every run
    written before item 5.3 as having trained on nothing.

    A JSON ``null`` set reads as ``None`` (unknown). A record without a
    ``source`` key reads as ``source="legacy"``. A legacy scope-free record held
    the scope it was *invoked* over, not what its artifacts held, so its
    entries and variants read as ``None``; a legacy scope-dependent record
    keeps them.
    A record without ``state_digest`` reads it as ``""``, and one whose
    ``state_digest`` is not a string reads as ``None``. A record without
    ``fit_inputs`` reads ``inputs=None``, without ``execution_id`` reads ``""``,
    and a ``via`` item without ``state_digest`` reads ``""``. A ``fit_inputs``
    that is not an object with an ``entries`` list reads the record as ``None``,
    and an entry of it that is not an object is left out.
    Nothing is rewritten on disk.

    Args:
        run_root: A run root, ``features/<feature>/<run_id>``.
        strict: Raise for a record that cannot be read or parsed, is not a
            JSON object, holds a ``state_digest`` that is not a string, or
            holds a ``fit_inputs`` that is not an object with an ``entries``
            list, or an entry of it that is not an object, instead of reading
            it as ``None`` or leaving the entry out.

    Raises:
        FitStateUnreadable: *strict* is set and the record is one of those.
    """
    path = run_root / FIT_SCOPE_NAME
    parsed: object
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    except OSError as error:
        return _unreadable(run_root, str(error), strict=strict)
    try:
        parsed = json.loads(text)
    except ValueError as error:
        return _unreadable(run_root, str(error), strict=strict)
    # Narrowed to a fully-known mapping type rather than iterated: ``json.loads``
    # is untyped, so every read off a plain ``isinstance`` narrowing would
    # otherwise be an unknown the strict checker cannot see through.
    if not _is_object(parsed):
        return _unreadable(run_root, "not a JSON object", strict=strict)
    raw = parsed
    state_digest = raw.get("state_digest", "")
    if not isinstance(state_digest, str):
        return _unreadable(run_root, "its state_digest is not a string", strict=strict)
    inputs: tuple[FitEntryInputs, ...] | None = None
    if "fit_inputs" in raw:
        raw_inputs = raw["fit_inputs"]
        raw_items = raw_inputs.get("entries") if _is_object(raw_inputs) else None
        if not _is_list(raw_items):
            return _unreadable(
                run_root,
                "its fit_inputs is not an object with an entries list",
                strict=strict,
            )
        entries_read: list[FitEntryInputs] = []
        for raw_item in raw_items:
            item = _entry_inputs(raw_item)
            if item is None:
                _unreadable(
                    run_root,
                    "an entry of its fit_inputs is not an object",
                    strict=strict,
                )
                continue
            entries_read.append(item)
        inputs = tuple(
            sorted(entries_read, key=lambda item: (item.group, item.sequence))
        )
    scope_dependent = bool(raw.get("scope_dependent", False))
    raw_source = raw.get("source")
    source: FitSource = "legacy"
    for known_source in get_args(FitSource):
        if raw_source == known_source:
            source = known_source
    unknown = source == "legacy" and not scope_dependent
    raw_entries = raw.get("entries")
    entries = (
        None
        if unknown or raw_entries is None
        else tuple(
            entry for entry in map(_entry, _items(raw_entries)) if entry is not None
        )
    )
    return FitScope(
        scope_dependent=scope_dependent,
        entries=entries,
        tracks_variants=None if unknown else _strings(raw.get("tracks_variants")),
        labels_variants=None if unknown else _strings(raw.get("labels_variants")),
        fitted_at=str(raw.get("fitted_at", "")),
        identity_scheme=str(raw.get("identity_scheme", "")),
        source=source,
        via=tuple(ref for ref in map(_source_ref, _items(raw.get("via"))) if ref),
        state_digest=state_digest,
        inputs=inputs,
        execution_id=str(raw.get("execution_id", "")),
    )


def _strings(value: object) -> tuple[str, ...] | None:
    """A list of strings, or ``None`` for JSON ``null``."""
    if value is None:
        return None
    return tuple(str(item) for item in _items(value))


def _source_ref(value: object) -> FitSourceRef | None:
    """One ``via`` item, or ``None`` when it is not one."""
    if not _is_object(value):
        return None
    return FitSourceRef(
        field=str(value.get("field", "")),
        feature=str(value.get("feature", "")),
        run_id=str(value.get("run_id", "")),
        known=bool(value.get("known", False)),
        state_digest=str(value.get("state_digest", "")),
    )


def _entry_inputs(value: object) -> FitEntryInputs | None:
    """One ``fit_inputs`` entry, or ``None`` when it is not an object."""
    if not _is_object(value):
        return None
    return FitEntryInputs(
        group=str(value.get("group", "")),
        sequence=str(value.get("sequence", "")),
        cells=EntryCells(
            roots=str(value.get("roots", "")),
            tracks=str(value.get("tracks", "")),
            inputs=str(value.get("inputs", "")),
        ),
    )


def fit_and_apply_scopes(
    ds: Dataset, feature_name: str, run_id: str
) -> tuple[frozenset[tuple[str, str]] | None, frozenset[tuple[str, str]]]:
    """``(fit, apply)`` for one run. ``None`` for the fit half means unknown.

    The one place the pairing is derived, so nobody derives it twice and gets a
    different answer. The apply half is simply the run's index rows: a row *is*
    the record that a sequence was applied. A row is fit-and-apply exactly when
    its entry appears in both.
    """
    from .index import feature_run_rows, has_feature_index

    fitted = read_run_fit_scope(ds, feature_name, run_id)
    fit = (
        None if fitted is None or fitted.entries is None else frozenset(fitted.entries)
    )
    if not has_feature_index(ds, feature_name):
        return fit, frozenset()
    rows = feature_run_rows(ds, feature_name, run_id)
    applied = frozenset(
        (str(row["group"]), str(row["sequence"])) for _, row in rows.iterrows()
    )
    return fit, applied
