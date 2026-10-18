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

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeGuard, get_args

from ._utils import ResolvedScope, atomic_write, now_iso
from .identity_scheme import FEATURE_IDENTITY_SCHEME
from .types import ArtifactSpec, Feature

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

    from .resolve import Resolution

__all__ = [
    "FIT_SCOPE_NAME",
    "FitScope",
    "FitSource",
    "FitSourceRef",
    "derive_fit_scope",
    "fit_and_apply_scopes",
    "read_fit_scope",
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
    """

    field: str
    feature: str
    run_id: str
    known: bool


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
    """

    scope_dependent: bool
    entries: tuple[tuple[str, str], ...] | None
    tracks_variants: tuple[str, ...] | None
    labels_variants: tuple[str, ...] | None
    fitted_at: str
    identity_scheme: str
    source: FitSource = "stream"
    via: tuple[FitSourceRef, ...] = ()


def derive_fit_scope(
    ds: Dataset,
    feature: Feature,
    resolved: ResolvedScope,
    resolutions: tuple[Resolution, ...],
    artifact_paths: Mapping[str, Path],
) -> FitScope:
    """The fit scope of a fit that just ran, from where its data came from.

    A scope-dependent feature fits from its own stream, so its record is the
    scope it resolved, and its ``via`` names the artifacts it pinned. A
    scope-free fitter fits from the artifacts its params pin -- the scope it was
    invoked over is only where it is *applied* -- so its record is the union of
    those artifacts' own fit records. That is transitive: each upstream recorded
    its own fit scope when it ran.

    Args:
        ds: The dataset the run belongs to.
        feature: The feature that just fitted.
        resolved: The scope the run resolved.
        resolutions: The references ``resolve_references`` pinned for this run,
            reused rather than walking the params a second time.
        artifact_paths: The artifact files ``_resolve_dependencies`` found, by
            params field. A reference with no file fed nothing into the fit.

    Returns:
        The record to write. Its entries are ``None`` when no pinned artifact
        exists (``source="unknown"``) or when some pinned artifact has no known
        fit scope (``source="artifacts"`` with that upstream ``known=False``).
    """
    pinned = _pinned_artifacts(ds, feature, resolutions, artifact_paths)
    if feature.scope_dependent:
        return FitScope(
            scope_dependent=True,
            entries=tuple(sorted(resolved.entries)),
            tracks_variants=tuple(sorted(resolved.tracks_variants)),
            labels_variants=tuple(sorted(resolved.labels_variants)),
            fitted_at=now_iso(),
            identity_scheme=FEATURE_IDENTITY_SCHEME,
            source="stream",
            via=tuple(ref for ref, _ in pinned),
        )

    entries: set[tuple[str, str]] = set()
    tracks: set[str] = set()
    labels: set[str] = set()
    for _, upstream in pinned:
        if upstream is not None:
            entries.update(upstream.entries or ())
            tracks.update(upstream.tracks_variants or ())
            labels.update(upstream.labels_variants or ())

    via = tuple(ref for ref, _ in pinned)
    if not via:
        return _unknown(source="unknown", via=())
    if not all(ref.known for ref in via):
        return _unknown(source="artifacts", via=via)
    return FitScope(
        scope_dependent=False,
        entries=tuple(sorted(entries)),
        tracks_variants=tuple(sorted(tracks)),
        labels_variants=tuple(sorted(labels)),
        fitted_at=now_iso(),
        identity_scheme=FEATURE_IDENTITY_SCHEME,
        source="artifacts",
        via=via,
    )


def _pinned_artifacts(
    ds: Dataset,
    feature: Feature,
    resolutions: tuple[Resolution, ...],
    artifact_paths: Mapping[str, Path],
) -> list[tuple[FitSourceRef, FitScope | None]]:
    """Each artifact the params pin that fed the run, and its own fit record.

    The record is ``None`` for a run that has none. The artifact is ``known``
    only when its record holds entries and both variant sets.
    """
    from .index import feature_run_root

    pinned: list[tuple[FitSourceRef, FitScope | None]] = []
    for resolution in resolutions:
        field = resolution.where.removeprefix("params.")
        if field == resolution.where or field not in artifact_paths:
            continue
        if resolution.run_id is None:
            continue
        if not isinstance(getattr(feature.params, field, None), ArtifactSpec):
            continue
        upstream = read_fit_scope(
            feature_run_root(ds, resolution.feature, resolution.run_id)
        )
        known = (
            upstream is not None
            and upstream.entries is not None
            and upstream.tracks_variants is not None
            and upstream.labels_variants is not None
        )
        ref = FitSourceRef(
            field=field,
            feature=resolution.feature,
            run_id=resolution.run_id,
            known=known,
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

    Overwriting is correct when it happens: reaching here means ``load_state``
    found nothing, so the state being described is being replaced too.

    *scope* comes from :func:`derive_fit_scope`. Its unknown sets are written as
    JSON ``null``.
    """
    payload = {
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
            }
            for ref in scope.via
        ],
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


def read_fit_scope(run_root: Path) -> FitScope | None:
    """The fit record in *run_root*, or ``None`` when there is none.

    ``None`` means **unknown**, not "fitted over nothing": it covers a run that
    predates this record and a feature that has no fit phase at all. A caller
    must not read it as an empty fit scope -- that would report every run written
    before item 5.3 as having trained on nothing.

    A JSON ``null`` set reads as ``None`` (unknown). A record without a
    ``source`` key reads as ``source="legacy"``. A legacy scope-free record held
    the scope it was *invoked* over, not what its artifacts held, so its entries
    and variants read as ``None``; a legacy scope-dependent record keeps them.
    Nothing is rewritten on disk.
    """
    path = run_root / FIT_SCOPE_NAME
    if not path.exists():
        return None
    parsed: object
    try:
        parsed = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    # Narrowed to a fully-known mapping type rather than iterated: ``json.loads``
    # is untyped, so every read off a plain ``isinstance`` narrowing would
    # otherwise be an unknown the strict checker cannot see through.
    if not _is_object(parsed):
        return None
    raw = parsed
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
    from .index import feature_index, feature_index_path, feature_run_root

    fitted = read_fit_scope(feature_run_root(ds, feature_name, run_id))
    fit = (
        None if fitted is None or fitted.entries is None else frozenset(fitted.entries)
    )
    index = feature_index(feature_index_path(ds, feature_name))
    if not index.path.exists():
        return fit, frozenset()
    rows = index.read(run_id=run_id)
    applied = frozenset(
        (str(row["group"]), str(row["sequence"])) for _, row in rows.iterrows()
    )
    return fit, applied
