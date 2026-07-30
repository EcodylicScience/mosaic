"""The forward pass: recompute every artifact's identifier, re-address what moved.

Every other maintenance pass in the toolkit reconciles the *index* against the
*disk* -- ``Dataset.reindex`` drops rows whose file is gone, ``reprobe-media``
re-probes media, ``prune-media``/``sweep-tracking`` delete stranded or expired
derivatives. None of them recomputes a feature/tracks/labels ``run_id`` from the
current code and moves the artifact under its new address. That is the one gap
this module fills, and it is the pass the identity-scheme marker was built to make
possible: the marker records *which hashing contract* minted a run, so a change to
the contract is detectable and a migration over it is idempotent and resumable.

**The mechanism.** For each addressable artifact: read the recorded old
identifier (never recompute it -- once the hash machinery has moved, the code that
produced the old value is gone); reconstruct the identity inputs from recorded
provenance (``params.json``, the index row, ``<root>/sequences.csv``); recompute
the new identifier by handing those inputs to the *current* identity function; and
if it moved, confirm from that provenance that the inputs did not change and
re-address the artifact -- move its directory, rewrite its index row, refresh its
marker. It runs bottom-up over the dependency graph, because each level's new
identifier is a function of the level below's.

**The honest bound.** An identifier moves for two reasons: the inputs changed, or
the identity function changed while the inputs are identical. Only the second is a
safe re-address. When recorded provenance cannot confirm the inputs unchanged --
a run that predates provenance, an upstream that was never pinned, a fit whose
scope was not recorded -- the artifact is reported and left, to be recomputed by
an ordinary run, never re-stamped with the current version. A loud wrong cache
*miss* beats a silent wrong *hit*. This is the ``identity_shift_recompute`` /
``unresolvable_pre_provenance`` half of the taxonomy, and it propagates: a
downstream whose upstream could not be re-addressed cannot be re-addressed either.

The engine here is artifact-kind-agnostic. Each kind (features, tracks, labels,
sources) registers an :class:`IdentityReconciler`; the driver orders them
bottom-up by ``depends_on`` and threads a shared remap of old-id to new-id down
the graph. A future identity change is a bumped scheme constant plus, at most, an
updated reconciler -- the driver, taxonomy and report do not move.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "FindingAction",
    "IdentityReconciler",
    "PassBuilder",
    "ReconcileFinding",
    "ReconcileKey",
    "ReconcilePass",
    "ReconcileReport",
    "ReconcileState",
    "UpstreamRemap",
    "Verdict",
    "backup_index",
    "identity_reconcilers",
    "known_keys",
    "register_identity_reconciler",
    "run_reconcile",
    "substitute_upstreams",
]

ReconcileKey = Literal[
    "media_raw", "tracks_raw", "labels_raw", "tracks", "labels", "features"
]
"""One addressable artifact kind. Source roots first, then derived roots.

The order also states the bottom-up dependency direction the driver relies on: a
source is read by tracks and features, tracks are read by features, and so on.
"""

Verdict = Literal[
    "ok",
    "scheme_stale",
    "identity_shift_relocatable",
    "identity_shift_recompute",
    "drift",
    "unresolvable_pre_provenance",
    "dangling",
    "orphan",
    "non_portable",
]
"""What the pass found about one artifact.

- ``ok`` -- the recomputed identifier equals the recorded one and the marker
  names the current scheme. Nothing to do.
- ``scheme_stale`` -- the identifier is unchanged but the marker names an older
  scheme (the payload's *value* did not move even though its *contract* did).
  Refresh the marker; touch nothing else.
- ``identity_shift_relocatable`` -- the identifier moved and recorded provenance
  confirms the inputs did not, so the move is a pure re-address.
- ``identity_shift_recompute`` -- the identifier moved but provenance cannot
  confirm the inputs unchanged. Decline to re-address; recompute by an ordinary
  run. Cascades to every downstream.
- ``drift`` -- the identifier is unchanged but a consumed source's recorded
  composition or fingerprint no longer matches disk, so the artifact is wrong
  under its own name. Report; a ``--force`` delete is the remedy.
- ``unresolvable_pre_provenance`` -- the recorded inputs are unreadable or
  predate provenance, so no honest recompute is possible.
- ``dangling`` -- an index row whose file is gone.
- ``orphan`` -- an on-disk artifact with no index row and no pipeline keeper.
- ``non_portable`` -- an ``abs_path`` stored absolute-in-tree.
"""

FindingAction = Literal[
    "none",
    "reported",
    "marker_refreshed",
    "relocated",
    "row_pruned",
    "path_rewritten",
    "deleted",
]
"""What ``--apply``/``--force`` actually did about a finding (``none`` on a dry run)."""

# (kind, old_id) -> new_id, for every artifact re-addressed this pass. Threaded
# down the graph so a downstream substitutes its parents' new addresses before it
# recomputes its own. Mutable on purpose: the driver fills it as each level lands.
UpstreamRemap = dict[tuple[ReconcileKey, str], str]


@dataclass
class ReconcileState:
    """What one level of the pass hands the next, threaded bottom-up.

    ``remap`` names every artifact re-addressed this pass, so a downstream
    substitutes its parents' new addresses before recomputing its own. ``blocked``
    names every artifact that *should* have moved but could not be confirmed
    (``identity_shift_recompute`` / ``unresolvable_pre_provenance``): a downstream
    that consumed one cannot itself be re-addressed, because one of its inputs is
    now at an unknown address. The honesty rule propagating up the graph.
    """

    remap: UpstreamRemap = field(default_factory=dict)
    blocked: set[tuple[ReconcileKey, str]] = field(default_factory=set)

    def record_move(self, key: ReconcileKey, old_id: str, new_id: str) -> None:
        self.remap[(key, old_id)] = new_id

    def record_blocked(self, key: ReconcileKey, old_id: str) -> None:
        self.blocked.add((key, old_id))

    def resolved(self, key: ReconcileKey, old_id: str) -> str:
        """The address *old_id* now lives at (itself, when it did not move)."""
        return self.remap.get((key, old_id), old_id)

    def is_blocked(self, key: ReconcileKey, old_id: str) -> bool:
        return (key, old_id) in self.blocked


@dataclass(frozen=True, slots=True)
class ReconcileFinding:
    """One artifact, what the pass decided about it, and what it did."""

    key: ReconcileKey
    verdict: Verdict
    old_address: str
    new_address: str
    index_path: Path
    run_root: str
    reason: str
    action: FindingAction = "none"

    def with_action(self, action: FindingAction) -> "ReconcileFinding":
        """A copy recording what ``apply`` did, leaving the diagnosis intact."""
        return ReconcileFinding(
            key=self.key,
            verdict=self.verdict,
            old_address=self.old_address,
            new_address=self.new_address,
            index_path=self.index_path,
            run_root=self.run_root,
            reason=self.reason,
            action=action,
        )


@dataclass(frozen=True, slots=True)
class ReconcilePass:
    """What one reconciler found and did over its own artifact kind."""

    findings: tuple[ReconcileFinding, ...] = ()
    backups: tuple[Path, ...] = ()
    errors: tuple[str, ...] = ()


class IdentityReconciler(Protocol):
    """The forward pass for one artifact kind.

    Registered rather than imported, so a producer that lives outside ``core``
    (a tracker's ``_tracking`` output) can add itself without ``core`` importing
    it -- the same seam ``FEATURES``/``OPS`` and ``register_reconcilable_index``
    already use.
    """

    key: ReconcileKey
    depends_on: tuple[ReconcileKey, ...]

    def reconcile(
        self, state: ReconcileState, *, apply: bool, force: bool
    ) -> ReconcilePass:
        """Sweep this kind: read, recompute, classify, and (if apply/force) act.

        Reads every artifact's recorded identity, recomputes it through the
        current identity function, records any confirmed re-address into *state*
        (so a downstream reconciler sees its parents' new addresses, and a
        downstream of a blocked parent declines), and returns one finding per
        artifact. Must never raise on a single bad artifact -- catch it, record it
        in ``ReconcilePass.errors`` or as an ``unresolvable_pre_provenance``
        finding, and continue.
        """
        ...


_FACTORIES: dict[ReconcileKey, Callable[[Dataset], IdentityReconciler]] = {}


def register_identity_reconciler(
    key: ReconcileKey, factory: Callable[[Dataset], IdentityReconciler]
) -> None:
    """Declare how to build the forward-pass reconciler for *key*.

    Each built-in reconciler calls this on import; ``Dataset.reconcile`` imports
    the reconciler modules (triggering registration) and reads the registry back.
    """
    _FACTORIES[key] = factory


_builtins_registered = False


def _ensure_registered() -> None:
    """Import the built-in reconcilers so the registry is populated.

    Registration is a module import side effect (the ``FEATURES``/``OPS`` seam), so
    a caller that reads the registry before any reconciler module was imported --
    the CLI validating ``--only`` before ``Dataset.reconcile`` runs -- would see it
    empty. The imports are lazy and here, not at module top, because the reconciler
    modules import *from* this one; deferring to call time breaks that cycle.
    """
    global _builtins_registered
    if _builtins_registered:
        return
    _builtins_registered = True
    from mosaic.core.pipeline import reconcile_features, reconcile_variants

    _ = (reconcile_features, reconcile_variants)


def known_keys() -> tuple[ReconcileKey, ...]:
    """Every artifact kind that has a registered reconciler, in registration order."""
    _ensure_registered()
    return tuple(_FACTORIES)


def identity_reconcilers(
    ds: Dataset, only: tuple[str, ...] = ()
) -> list[IdentityReconciler]:
    """Every registered reconciler for *ds*, optionally filtered to *only*.

    *only* is plain ``str`` rather than ``ReconcileKey`` so a CLI flag can reach it
    without validation gymnastics; an unknown key simply matches nothing. Order is
    not meaningful here -- :func:`run_reconcile` toposorts on ``depends_on`` -- but
    is stable (registration order) for a readable report.
    """
    _ensure_registered()
    wanted = set(only)
    return [
        factory(ds)
        for key, factory in _FACTORIES.items()
        if not wanted or key in wanted
    ]


def _ordered(reconcilers: list[IdentityReconciler]) -> list[IdentityReconciler]:
    """Toposort *reconcilers* so every kind runs after the kinds it depends on.

    A dependency naming a kind not present in this run (filtered out by ``--only``)
    is simply ignored -- the remap for that kind stays empty, which is exactly the
    "nothing moved upstream" case, so a partial run is a narrower correct pass
    rather than an error.
    """
    present = {rec.key: rec for rec in reconcilers}
    ordered: list[IdentityReconciler] = []
    placed: set[ReconcileKey] = set()

    def place(rec: IdentityReconciler) -> None:
        if rec.key in placed:
            return
        placed.add(rec.key)  # mark first, so a dependency cycle terminates
        for dep in rec.depends_on:
            parent = present.get(dep)
            if parent is not None:
                place(parent)
        ordered.append(rec)

    for rec in reconcilers:
        place(rec)
    return ordered


@dataclass(frozen=True)
class ReconcileReport:
    """What the whole pass found across every kind, and what it did about it.

    ``findings`` are the identity forward pass (features, tracks, labels).
    ``pruned`` and ``repathed`` are the composed index-hygiene passes -- dangling
    rows dropped by :meth:`Dataset.reindex` and non-portable ``abs_path`` cells
    rewritten by :meth:`Dataset.make_portable` -- each a ``{path: count}`` map,
    kept apart from the findings because they are index operations rather than
    per-artifact identity verdicts.
    """

    applied: bool
    forced: bool
    findings: tuple[ReconcileFinding, ...]
    backups: tuple[Path, ...]
    errors: tuple[str, ...]
    pruned: dict[str, int] = field(default_factory=dict)
    repathed: dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        """True when anything other than ``ok`` was found across all passes."""
        if any(finding.verdict != "ok" for finding in self.findings):
            return True
        return bool(self.pruned) or bool(self.repathed)

    def counts(self) -> dict[Verdict, int]:
        """How many findings landed on each verdict (zeros omitted)."""
        tally: dict[Verdict, int] = {}
        for finding in self.findings:
            tally[finding.verdict] = tally.get(finding.verdict, 0) + 1
        return tally

    def of_verdict(self, verdict: Verdict) -> tuple[ReconcileFinding, ...]:
        """Every finding with *verdict*, in report order."""
        return tuple(f for f in self.findings if f.verdict == verdict)

    def payload(self) -> dict[str, object]:
        """The ``--json`` document: one flat object, no nested optionals."""
        return {
            "applied": self.applied,
            "forced": self.forced,
            "changed": self.changed,
            "counts": {verdict: n for verdict, n in self.counts().items()},
            "findings": [
                {
                    "key": f.key,
                    "verdict": f.verdict,
                    "action": f.action,
                    "old_address": f.old_address,
                    "new_address": f.new_address,
                    "index": str(f.index_path),
                    "run_root": f.run_root,
                    "reason": f.reason,
                }
                for f in self.findings
            ],
            "backups": [str(path) for path in self.backups],
            "errors": list(self.errors),
            "pruned": dict(self.pruned),
            "repathed": dict(self.repathed),
        }


def run_reconcile(
    reconcilers: list[IdentityReconciler], *, apply: bool, force: bool
) -> ReconcileReport:
    """Drive the forward pass over *reconcilers*, bottom-up.

    Toposorts on ``depends_on`` so a source runs before the tracks that read it
    and tracks before the features that read them, threads one shared *state* down
    the graph, and aggregates every reconciler's findings into one report. The
    state is what makes the pass bottom-up: a re-address (or a blocked artifact)
    recorded by one level is seen by the next.
    """
    state = ReconcileState()
    findings: list[ReconcileFinding] = []
    backups: list[Path] = []
    errors: list[str] = []
    for rec in _ordered(reconcilers):
        result = rec.reconcile(state, apply=apply, force=force)
        findings.extend(result.findings)
        backups.extend(result.backups)
        errors.extend(result.errors)
    return ReconcileReport(
        applied=apply,
        forced=force,
        findings=tuple(findings),
        backups=tuple(backups),
        errors=tuple(errors),
    )


def backup_index(index_path: Path) -> Path:
    """Copy *index_path* aside under a UTC timestamp before it is rewritten.

    The same guard ``reprobe-media`` takes before it changes an index: a
    re-address rewrites run-id and path cells, so the copy is the only record of
    the index in its pre-move shape. The ``.backup`` suffix is not an index or
    media extension, so a backup is never picked up by a later scan.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = index_path.with_name(f"{index_path.name}.{stamp}.backup")
    _ = shutil.copy2(index_path, backup)
    return backup


# A per-reconciler accumulator, so a concrete reconciler builds its ReconcilePass
# without repeating the list plumbing. Kept here beside the pass it produces.
@dataclass
class PassBuilder:
    """Mutable collector a reconciler fills, then freezes into a ReconcilePass."""

    findings: list[ReconcileFinding] = field(default_factory=list)
    backups: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add(self, finding: ReconcileFinding) -> None:
        self.findings.append(finding)

    def backed_up(self, path: Path) -> None:
        self.backups.append(path)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def build(self) -> ReconcilePass:
        return ReconcilePass(
            findings=tuple(self.findings),
            backups=tuple(self.backups),
            errors=tuple(self.errors),
        )


def substitute_upstreams(
    state: ReconcileState, edges: Mapping[ReconcileKey, str]
) -> dict[ReconcileKey, str]:
    """Resolve a mapping of upstream ``(kind -> old_id)`` edges through *state*.

    A small shared helper for the concrete reconcilers: each value becomes the new
    id its upstream landed on this pass, or the old id when that upstream did not
    move.
    """
    return {kind: state.resolved(kind, old) for kind, old in edges.items()}
