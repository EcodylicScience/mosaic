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
fit gate, so a second invocation at a wider apply scope overwrites its
``_scope.entries`` with the wider set. The value is "whichever ran last", which
that key's own comment concedes; what misleads is its name. This file is written
only when ``fit()` actually ran, so it stays what it says it is.

**No index column, deliberately.** A fit-scope cell would hold the same value on
every row of a run -- a per-run property duplicated per row, which is the shape
item 4.4 rejects for compositions and rejects here for the same reason: no way to
tell a stale copy from a current one. The pairing is already derivable, and
:func:`fit_and_apply_scopes` is the one place that derives it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ._utils import Scope, atomic_write, now_iso
from .identity_scheme import FEATURE_IDENTITY_SCHEME

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "FIT_SCOPE_NAME",
    "FitScope",
    "fit_and_apply_scopes",
    "read_fit_scope",
    "write_fit_scope",
]

FIT_SCOPE_NAME = "fit_scope.json"


@dataclass(frozen=True, slots=True)
class FitScope:
    """What a run's saved state was fitted over.

    ``entries`` and ``tracks_variants`` are sorted, so a diff of two runs' files
    shows what changed rather than how the sets were iterated. ``fitted_at`` is
    provenance and never hashed; ``identity_scheme`` is what makes a record
    written under an older hashing contract recognizable as such.
    """

    scope_dependent: bool
    entries: tuple[tuple[str, str], ...]
    tracks_variants: tuple[str, ...]
    fitted_at: str
    identity_scheme: str


def write_fit_scope(run_root: Path, scope: Scope, *, scope_dependent: bool) -> None:
    """Record the scope a fit just consumed, in *run_root*.

    Call **after** ``save_state``, inside the branch that actually fitted. Not
    before ``fit()``: a fit that raises halfway would otherwise leave a record of
    a fit that never completed, which is worse than no record. Not beside
    ``write_identity_scheme`` either -- that one is written on every invocation
    including a pure cache hit, and this must not be.

    Overwriting is correct when it happens: reaching here means ``load_state``
    found nothing, so the state being described is being replaced too.
    """
    payload = {
        "scope_dependent": scope_dependent,
        "entries": [list(entry) for entry in sorted(scope.entries)],
        "tracks_variants": sorted(scope.tracks_variants),
        "fitted_at": now_iso(),
        "identity_scheme": FEATURE_IDENTITY_SCHEME,
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
    if not isinstance(value, list):
        return []
    listed: list[object] = value
    return listed


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
    """
    path = run_root / FIT_SCOPE_NAME
    if not path.exists():
        return None
    parsed: object
    try:
        parsed = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    # Widened to a fully-known mapping type rather than iterated: ``json.loads``
    # is untyped, so every read off the narrowed dict would otherwise be an
    # unknown the strict checker cannot see through.
    raw: Mapping[object, object] = parsed
    entries = [
        entry for entry in map(_entry, _items(raw.get("entries"))) if entry is not None
    ]
    return FitScope(
        scope_dependent=bool(raw.get("scope_dependent", False)),
        entries=tuple(entries),
        tracks_variants=tuple(
            str(variant) for variant in _items(raw.get("tracks_variants"))
        ),
        fitted_at=str(raw.get("fitted_at", "")),
        identity_scheme=str(raw.get("identity_scheme", "")),
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
    index = feature_index(feature_index_path(ds, feature_name))
    if not index.path.exists():
        return (None if fitted is None else frozenset(fitted.entries)), frozenset()
    rows = index.read(run_id=run_id)
    applied = frozenset(
        (str(row["group"]), str(row["sequence"])) for _, row in rows.iterrows()
    )
    return (None if fitted is None else frozenset(fitted.entries)), applied
