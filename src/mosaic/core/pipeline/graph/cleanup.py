"""Which superseded feature runs a recipe's plan may remove, and removing them.

A recipe's feature steps each name a run of a storage under ``features/``. The
other runs of those storages are **superseded**: an earlier version of the same
step, with other params. :func:`plan_cleanup` reports each one as removable or as
kept with its reason, and :func:`run_cleanup` removes the ones a submission
approved, after the request's steps have finished.

Only feature runs of the storages the recipe names are ever candidates. Tracks,
models, media and the storages of other recipes are not, and a run directory with
no index rows is not in the inventory and is left to ``mosaic reindex``.

A candidate is kept, by the first rule that matches:

1. Its storage's step has no run id yet (``keeper_unresolved``), or the whole
   plan is held: a managed dataset (``managed_dataset``) or a request file that
   does not read (``request_unreadable``).
2. The request being cleaned up pins it (``bound``).
3. Its feature declares ``costly_fit`` (``costly_fit``).
4. Its ``params.json`` is absent or unreadable, so it could not be computed
   again (``params_unknown``).
5. A run-log that may still run names it (``attempt_open``).
6. Another request holds it (``held_by_request``): a running or orphaned
   request whatever its age, a running one for every run of the features its
   recipe names, and an ended one for :data:`~.model.REQUEST_HOLD_FLOOR_HOURS`
   or ``min_age_hours``, whichever is longer.
7. It holds entries its replacement lacks (``wider_coverage``), or it is a
   scope-dependent fit over entries its replacement does not cover
   (``other_fit_scope``).
8. It is younger than ``min_age_hours`` (``young``).
9. It is among the ``keep_newest`` newest superseded runs of its storage
   (``newest``).
10. A run that is kept reads it, through ``inputs``, a params reference or its
    fit record, directly or through other runs (``read_by``).

Planning is a read. It takes no lock and writes nothing, and its module imports
the standard library and pydantic alone; the inventory, the run-log readers and
the request store are imported inside the functions that use them. It never
imports the feature registry.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from mosaic.core.entry import Entry

from .model import REQUEST_HOLD_FLOOR_HOURS, CleanupPolicy, FeatureStepSpec

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.scope import Scope

    from ..fit_scope import FitScope
    from ..inventory.cache import InventoryCache
    from ..inventory.model import AnyRecord
    from ..inventory.params import RunParams
    from .compatibility import DeclarationCatalog
    from .model import Recipe, Request
    from .plan import Plan

__all__ = [
    "CleanupPlan",
    "KeptReason",
    "SupersededRun",
    "plan_cleanup",
    "plan_with_cleanup",
]

type KeptReason = Literal[
    "keeper",
    "keeper_unresolved",
    "managed_dataset",
    "request_unreadable",
    "bound",
    "costly_fit",
    "params_unknown",
    "attempt_open",
    "held_by_request",
    "wider_coverage",
    "other_fit_scope",
    "young",
    "newest",
    "read_by",
    "absent",
]
"""Why a superseded run is kept. ``keeper`` and ``absent`` are answered only when
the cleanup runs: an approved run the plan now expects, and one already gone."""

_FROM: Final = "__from__"
"""What joins a feature's slug to the storages it reads in a storage name."""


@dataclass(frozen=True, slots=True)
class SupersededRun:
    """One run of a planned storage that the plan does not expect.

    Attributes:
        step_id: The first step of the plan writing to its storage.
        storage: Its storage name under ``features/``.
        run_id: The run.
        removable: Whether a cleanup may remove it.
        reason: ``superseded`` when removable, otherwise why it is kept.
        detail: What made the reason so, in a sentence; ``""`` when removable.
        bytes: The bytes its directory holds.
        entries: How many entries its index rows name.
        finished_at: When it finished, ``""`` for a run that did not.
        superseded_by: The run or runs the plan expects in its storage.
    """

    step_id: str
    storage: str
    run_id: str
    removable: bool
    reason: KeptReason | Literal["superseded"]
    detail: str
    bytes: int
    entries: int
    finished_at: str
    superseded_by: str


@dataclass(frozen=True, slots=True)
class CleanupPlan:
    """Every superseded run of a plan's storages, and the policy that judged them.

    Attributes:
        policy: The age and recency the runs were judged under.
        runs: One record per superseded run, removable ones first in each
            storage, then the newest.
    """

    policy: CleanupPolicy
    runs: tuple[SupersededRun, ...] = ()

    @property
    def removable(self) -> tuple[SupersededRun, ...]:
        """The runs a cleanup may remove."""
        return tuple(run for run in self.runs if run.removable)

    @property
    def kept(self) -> tuple[SupersededRun, ...]:
        """The runs kept, each with its reason."""
        return tuple(run for run in self.runs if not run.removable)

    @property
    def removable_bytes(self) -> int:
        """The bytes the removable runs hold."""
        return sum(run.bytes for run in self.removable)

    @property
    def kept_bytes(self) -> int:
        """The bytes the kept runs hold."""
        return sum(run.bytes for run in self.kept)


def plan_with_cleanup(
    ds: Dataset,
    recipe: Recipe,
    *,
    cleanup: CleanupPolicy | None,
    scope: Scope | None = None,
    request: Request | None = None,
    inventory: InventoryCache | None = None,
    catalog: DeclarationCatalog | None = None,
    refuse: bool = True,
    step_scopes: Mapping[str, Scope] | None = None,
    now: datetime.datetime | None = None,
) -> Plan:
    """Plan *recipe*, and report its superseded runs when *cleanup* is given.

    The one entry point the CLI's ``pipeline plan``, the MCP ``plan_recipe`` and
    ``submit_pipeline`` and :func:`~.request.submit_request` share, so all four
    judge the same runs the same way.

    Args:
        ds: The dataset.
        recipe: The graph.
        cleanup: The policy to judge superseded runs under, or ``None`` to
            plan without them, which leaves the plan and its fingerprint as
            :func:`~.plan.plan_pipeline` returns them.
        scope: As for :func:`~.plan.plan_pipeline`.
        request: As for :func:`~.plan.plan_pipeline`. Its binds are kept, and
            its own request file holds nothing.
        inventory: As for :func:`~.plan.plan_pipeline`.
        catalog: As for :func:`~.plan.plan_pipeline`.
        refuse: As for :func:`~.plan.plan_pipeline`.
        step_scopes: As for :func:`~.plan.plan_pipeline`.
        now: The instant ages are judged at. ``None`` reads the clock.

    Returns:
        The plan, with ``cleanup`` set when *cleanup* was given.
    """
    from ..inventory.cache import InventoryCache
    from .plan import plan_pipeline
    from .resolve import declaration_catalog

    held = InventoryCache(ds) if inventory is None else inventory
    declarations = declaration_catalog() if catalog is None else catalog
    plan = plan_pipeline(
        ds,
        recipe,
        scope=scope,
        request=request,
        inventory=held,
        catalog=declarations,
        refuse=refuse,
        step_scopes=step_scopes,
    )
    if cleanup is None:
        return plan
    found = plan_cleanup(
        ds,
        plan,
        policy=cleanup,
        inventory=held,
        catalog=declarations,
        own_request=request,
        now=now,
    )
    return replace(plan, cleanup=found)


def plan_cleanup(
    ds: Dataset,
    plan: Plan,
    *,
    policy: CleanupPolicy,
    inventory: InventoryCache,
    catalog: DeclarationCatalog,
    own_request: Request | None = None,
    recorded: Mapping[str, str] | None = None,
    now: datetime.datetime | None = None,
    short_storages: Iterable[str] = (),
) -> CleanupPlan:
    """Judge every superseded run of *plan*'s feature storages. Writes nothing.

    The rules are listed in the module docstring, and the first that matches is
    the reason a run is kept. Every kept run is reported with it.

    Args:
        ds: The planned dataset.
        plan: The plan, whose feature steps' run ids are the runs to keep.
        policy: The age and recency rules.
        inventory: The held inventory the plan read, which lists the runs.
        catalog: The declarations, for ``costly_fit``.
        own_request: The request being cleaned up: its binds are kept, and its
            own file and attempts hold nothing.
        recorded: The run id each step's attempt recorded, which replaces the
            plan's prediction for that step.
        now: The instant ages are judged at. ``None`` reads the clock.
        short_storages: Storages whose run in this request lost entries. Each
            of their candidates is kept as ``wider_coverage``.

    Returns:
        The superseded runs, removable or kept.
    """
    from ..feature_run_removal import run_bytes
    from ..fit_scope import read_fit_scope
    from ..inventory.scan import GLOBAL_ENTRY

    moment = now or datetime.datetime.now(datetime.UTC)
    keepers = _keepers(plan, recorded or {})
    records = inventory.get().of_kind("feature")
    for record in records:
        keeper = keepers.get(record.name)
        if keeper is not None and record.run_id in keeper.run_ids:
            keeper.entries.update(_entries_of(record) - {GLOBAL_ENTRY})
    candidates = [
        record
        for record in records
        if record.name in keepers and record.run_id not in keepers[record.name].run_ids
    ]
    if not candidates:
        return CleanupPlan(policy=policy)

    base = ds.base_dir
    own_id = own_request.request_id if own_request is not None else ""
    bound = _bound_pairs(own_request)
    hold_hours = max(policy.min_age_hours, REQUEST_HOLD_FLOOR_HOURS)
    holds = _request_holds(base, now=moment, hold_hours=hold_hours, exclude=own_id)
    whole = _whole_plan_hold(ds, holds)
    opened = _open_attempts(base, now=moment)
    short = frozenset(short_storages)
    ranks = _newest_ranks(candidates)
    fits: dict[tuple[str, str], FitScope | None] = {}

    decided: dict[tuple[str, str], tuple[str, str]] = {}
    for record in candidates:
        key = (record.name, record.run_id)
        keeper = keepers[record.name]
        fit = read_fit_scope(record.run_root) if record.run_root is not None else None
        fits[key] = fit
        decided[key] = whole or _first_reason(
            record,
            keeper=keeper,
            bound=bound,
            catalog=catalog,
            opened=opened,
            holds=holds,
            short=record.name in short,
            fit=fit,
            policy=policy,
            rank=ranks[key],
            moment=moment,
            global_entry=GLOBAL_ENTRY,
        )

    _keep_what_is_read(ds, records, decided, opened, fits)

    runs: list[SupersededRun] = []
    for record in candidates:
        key = (record.name, record.run_id)
        reason, detail = decided[key]
        keeper = keepers[record.name]
        size = run_bytes(record.run_root)[0] if record.run_root is not None else 0
        runs.append(
            SupersededRun(
                step_id=keeper.step_id,
                storage=record.name,
                run_id=record.run_id,
                removable=reason == "superseded",
                reason=_reason(reason),
                detail=detail,
                bytes=size,
                entries=len(_entries_of(record) - {GLOBAL_ENTRY}),
                finished_at=record.finished_at,
                superseded_by=", ".join(sorted(keeper.run_ids)),
            )
        )
    runs.sort(
        key=lambda run: (
            run.storage,
            not run.removable,
            -_timestamp(run.finished_at),
            run.run_id,
        )
    )
    return CleanupPlan(policy=policy, runs=tuple(runs))


# --- what the plan keeps --------------------------------------------------------


@dataclass(slots=True)
class _Keeper:
    """What one storage's steps expect: their runs and the entries they cover."""

    step_id: str
    run_ids: set[str] = field(default_factory=set[str])
    entries: set[Entry] = field(default_factory=set[Entry])
    unresolved: str = ""


def _keepers(plan: Plan, recorded: Mapping[str, str]) -> dict[str, _Keeper]:
    """Every feature storage of *plan*, with the runs its steps expect.

    Unioned per storage, as two steps of one feature with other params share a
    storage. A step whose run id is unknown keeps its whole storage.
    """
    from .plan import step_target

    keepers: dict[str, _Keeper] = {}
    for planned in plan.steps:
        if planned.kind != "feature" or not planned.storage_name:
            continue
        keeper = keepers.setdefault(
            planned.storage_name, _Keeper(step_id=planned.step_id)
        )
        run_id = recorded.get(planned.step_id) or planned.run_id
        if run_id is None:
            keeper.unresolved = keeper.unresolved or planned.step_id
        else:
            keeper.run_ids.add(run_id)
        keeper.entries.update(step_target(planned, plan))
    return keepers


def _bound_pairs(request: Request | None) -> frozenset[tuple[str, str]]:
    """The ``(storage, run_id)`` pairs that *request* binds."""
    if request is None:
        return frozenset()
    return frozenset((bound.feature, bound.run_id) for bound in request.bind.values())


# --- holds outside the plan ------------------------------------------------------


@dataclass(slots=True)
class _Holds:
    """What other requests hold, and the first request file that does not read."""

    pairs: dict[tuple[str, str], str] = field(
        default_factory=dict[tuple[str, str], str]
    )
    run_ids: dict[str, str] = field(default_factory=dict[str, str])
    slugs: dict[str, str] = field(default_factory=dict[str, str])
    unreadable: str = ""


def _request_holds(
    base_dir: Path, *, now: datetime.datetime, hold_hours: float, exclude: str
) -> _Holds:
    """What the dataset's other requests hold.

    A running or orphaned request holds the run ids its attempts recorded and its
    binds, whatever its age, and a running one also every run of each feature
    its recipe names. A request that has ended holds its recorded runs and binds
    for *hold_hours* after its last attempt wrote a line.
    """
    from ..markers import parse_instant
    from .request import load_recipe_for_request
    from .rollup import request_rollup
    from .status import request_verdict_of
    from .step_preflight import StepRefused
    from .store import RequestUnreadable, load_request, requests_root

    holds = _Holds()
    root = requests_root(base_dir)
    names = sorted(path.stem for path in root.glob("*.json")) if root.is_dir() else []
    for name in names:
        if name == exclude:
            continue
        try:
            request = load_request(base_dir, name)
        except (RequestUnreadable, OSError) as error:
            holds.unreadable = holds.unreadable or str(error)
            continue
        rollup = request_rollup(base_dir, request)
        verdict = request_verdict_of(base_dir, rollup, now=now)
        if verdict not in ("running", "orphaned"):
            instants = [
                parse_instant(stamp)
                for stamp in (
                    request.created_at,
                    *(attempt.heartbeat_at for attempt in rollup.steps),
                )
                if stamp
            ]
            last = max((at for at in instants if at is not None), default=None)
            if last is not None and (now - last).total_seconds() > hold_hours * 3600:
                continue
        label = f"request {name} ({verdict})"
        for attempt in rollup.steps:
            if attempt.run_id:
                _ = holds.run_ids.setdefault(attempt.run_id, label)
        for bound in request.bind.values():
            _ = holds.pairs.setdefault((bound.feature, bound.run_id), label)
        if verdict == "running":
            try:
                recipe = load_recipe_for_request(base_dir, request)
            except StepRefused:
                continue
            for step in recipe.steps:
                if isinstance(step, FeatureStepSpec):
                    _ = holds.slugs.setdefault(step.feature, label)
    return holds


def _whole_plan_hold(ds: Dataset, holds: _Holds) -> tuple[str, str] | None:
    """The reason every candidate is kept, or ``None`` when none holds them all."""
    if ds.manifest.is_managed:
        return (
            "managed_dataset",
            f"the dataset is managed by {ds.manifest.managed_by!r}",
        )
    if holds.unreadable:
        return ("request_unreadable", holds.unreadable)
    return None


def _open_attempts(base_dir: Path, *, now: datetime.datetime) -> dict[str, str]:
    """The run ids of the attempts that may still run, by the attempt naming each.

    An attempt that has not ended is open while its process may still run
    (:func:`~.liveness.snapshot_liveness`): ``alive``, or ``unknown`` on another
    host, which this one cannot check.
    """
    from mosaic.runlog import TERMINAL_STATUSES, list_runs, run_log_dir

    from .liveness import is_local, snapshot_liveness

    opened: dict[str, str] = {}
    for snapshot in list_runs(run_log_dir(base_dir))["items"]:
        if snapshot["status"] in TERMINAL_STATUSES or not snapshot["run_id"]:
            continue
        liveness = snapshot_liveness(snapshot, now=now)
        if liveness == "alive" or (
            liveness == "unknown" and not is_local(snapshot["host"], snapshot["pid_ns"])
        ):
            _ = opened.setdefault(snapshot["run_id"], snapshot["execution_id"])
    return opened


# --- one candidate -----------------------------------------------------------------


def _first_reason(
    record: AnyRecord,
    *,
    keeper: _Keeper,
    bound: frozenset[tuple[str, str]],
    catalog: DeclarationCatalog,
    opened: Mapping[str, str],
    holds: _Holds,
    short: bool,
    fit: FitScope | None,
    policy: CleanupPolicy,
    rank: int,
    moment: datetime.datetime,
    global_entry: Entry,
) -> tuple[str, str]:
    """The first rule that keeps *record*, or ``("superseded", "")``."""
    key = (record.name, record.run_id)
    slug = record.name.split(_FROM, 1)[0]
    if keeper.unresolved:
        return (
            "keeper_unresolved",
            f"step {keeper.unresolved!r} has no run id yet, so its storage is kept",
        )
    if key in bound:
        return ("bound", "the request pins it")
    declaration = catalog.get(slug)
    if declaration is not None and declaration.costly_fit:
        return ("costly_fit", f"{slug} is costly to fit again")
    if record.params is None or record.params_state != "present":
        return (
            "params_unknown",
            f"its params.json is {record.params_state}, so it could not be "
            f"computed again",
        )
    if record.run_id in opened:
        return ("attempt_open", f"attempt {opened[record.run_id]} may still run")
    held = (
        holds.pairs.get(key)
        or holds.run_ids.get(record.run_id)
        or holds.slugs.get(slug)
    )
    if held:
        return ("held_by_request", f"{held} holds it")
    coverage = _coverage_reason(record, keeper, short, fit, global_entry)
    if coverage is not None:
        return coverage
    age = _age_hours(record, moment)
    if age is None or age < policy.min_age_hours:
        shown = "unknown" if age is None else f"{age:.1f} h"
        return ("young", f"its age, {shown}, is under {policy.min_age_hours:g} h")
    if rank < policy.keep_newest:
        return (
            "newest",
            f"it is among the {policy.keep_newest} newest superseded runs",
        )
    return ("superseded", "")


def _coverage_reason(
    record: AnyRecord,
    keeper: _Keeper,
    short: bool,
    fit: FitScope | None,
    global_entry: Entry,
) -> tuple[str, str] | None:
    """Keep a run that holds entries, or a fit, its replacement does not."""
    if short:
        return ("wider_coverage", "its replacement lost entries in this request")
    params = record.params
    dependent = (fit is not None and fit.scope_dependent) or (
        params is not None and params.scope.scope_dependent
    )
    if dependent:
        if fit is None or fit.entries is None:
            return ("other_fit_scope", "the entries it was fitted over are unknown")
        fitted = {(group, sequence) for group, sequence in fit.entries}
        if not fitted <= keeper.entries:
            outside = len(fitted - keeper.entries)
            return (
                "other_fit_scope",
                f"it was fitted over {outside} entries its replacement does not cover",
            )
    held = _entries_of(record) - {global_entry}
    beyond = held - keeper.entries
    if beyond:
        return (
            "wider_coverage",
            f"it holds {len(beyond)} entries its replacement does not",
        )
    return None


def _age_hours(record: AnyRecord, moment: datetime.datetime) -> float | None:
    """Hours since *record* finished, or since it started when it did not."""
    from ..markers import parse_instant

    stamp = parse_instant(record.finished_at or record.started_at)
    if stamp is None:
        return None
    return (moment - stamp).total_seconds() / 3600


def _newest_ranks(candidates: Iterable[AnyRecord]) -> dict[tuple[str, str], int]:
    """Each candidate's rank among its storage's, 0 for the newest."""
    by_storage: dict[str, list[AnyRecord]] = {}
    for record in candidates:
        by_storage.setdefault(record.name, []).append(record)
    ranks: dict[tuple[str, str], int] = {}
    for storage, held in by_storage.items():
        ordered = sorted(
            held,
            key=lambda record: (
                -_timestamp(record.finished_at or record.started_at),
                record.run_id,
            ),
        )
        for rank, record in enumerate(ordered):
            ranks[(storage, record.run_id)] = rank
    return ranks


# --- what the kept runs read -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Reads:
    """What one run reads: named runs, every run of a storage, or anything."""

    pairs: frozenset[tuple[str, str]] = frozenset()
    storages: frozenset[str] = frozenset()
    chain: str = ""
    everything: bool = False

    def covers(self, storage: str, run_id: str) -> bool:
        """Whether this run may read run *run_id* of *storage*."""
        return (
            self.everything
            or (storage, run_id) in self.pairs
            or storage in self.storages
            or (bool(self.chain) and storage in self.chain)
        )


def _keep_what_is_read(
    ds: Dataset,
    records: Iterable[AnyRecord],
    decided: dict[tuple[str, str], tuple[str, str]],
    opened: Mapping[str, str],
    fits: Mapping[tuple[str, str], FitScope | None],
) -> None:
    """Keep every removable run that a kept run reads, until nothing changes.

    The readers are every feature run of the dataset that is not removable, and
    the run of every attempt that may still run, found by its ``params.json``,
    which an attempt writes once it names its run.
    """
    from ..fit_scope import read_fit_scope
    from ..inventory.params import read_run_params

    readers: dict[tuple[str, str], _Reads] = {}
    for record in records:
        key = (record.name, record.run_id)
        fit = (
            fits[key]
            if key in fits
            else read_fit_scope(record.run_root)
            if record.run_root is not None
            else None
        )
        readers[key] = _reads_of(record.name, record.params, fit)
    features = ds.get_root("features")
    for run_id in opened:
        for params_path in features.glob(f"*/{run_id}/params.json"):
            run_root = params_path.parent
            key = (run_root.parent.name, run_id)
            readers[key] = _reads_of(
                key[0], read_run_params(run_root).params, read_fit_scope(run_root)
            )

    changed = True
    while changed:
        changed = False
        removable = [
            key for key, (reason, _) in decided.items() if reason == "superseded"
        ]
        for reader, reads in readers.items():
            if decided.get(reader, ("kept", ""))[0] == "superseded":
                continue
            for key in removable:
                if decided[key][0] == "superseded" and reads.covers(*key):
                    decided[key] = ("read_by", f"run {reader[0]}/{reader[1]} reads it")
                    changed = True


def _reads_of(storage: str, params: RunParams | None, fit: FitScope | None) -> _Reads:
    """What the run of *storage* with *params* and *fit* reads.

    A run whose params are known reads what its ``_resolved`` block names, or,
    for a file older than that block, the runs its ``_inputs`` and top-level
    ``_params`` name. A reference recorded without a run id reads every run of
    its storage. A run whose params are unknown reads every storage its own
    storage name was derived from, and everything when its feature takes a run
    through a params field.
    """
    from ..inventory.params import LABELS_EDGE, TRACKS_EDGE
    from .resolve import run_reference_fields

    via: frozenset[tuple[str, str]] = (
        frozenset((ref.feature, ref.run_id) for ref in fit.via) if fit else frozenset()
    )
    if params is None:
        slug, _, chain = storage.partition(_FROM)
        return _Reads(
            pairs=via,
            chain=chain,
            everything=bool(run_reference_fields(slug)),
        )
    pairs: set[tuple[str, str]] = set(via)
    storages: set[str] = set()
    if params.records_resolutions:
        for ref in params.resolved:
            if ref.where in (TRACKS_EDGE, LABELS_EDGE) or not ref.feature:
                continue
            if ref.run_id:
                pairs.add((ref.feature, ref.run_id))
            else:
                storages.add(ref.feature)
    else:
        for value in (*params.inputs, *params.params.values()):
            named = _named_run(value)
            if named is None:
                continue
            if named[1]:
                pairs.add(named)
            else:
                storages.add(named[0])
    return _Reads(pairs=frozenset(pairs), storages=frozenset(storages))


def _named_run(value: object) -> tuple[str, str] | None:
    """The ``(feature, run_id)`` that a recorded ``{feature, run_id}`` names."""
    if not isinstance(value, dict):
        return None
    named = cast("dict[str, object]", value)
    feature = named.get("feature")
    run_id = named.get("run_id")
    if not isinstance(feature, str) or not feature:
        return None
    return feature, run_id if isinstance(run_id, str) else ""


# --- helpers -------------------------------------------------------------------


def _entries_of(record: AnyRecord) -> set[Entry]:
    """The ``(group, sequence)`` entries a feature run's index rows name."""
    return {
        (row[0], row[1])
        for row in record.rows
        if isinstance(row, tuple) and len(row) == 2
    }


def _timestamp(stamp: str) -> float:
    """*stamp* as seconds since the epoch, 0 when it does not parse."""
    from ..markers import parse_instant

    moment = parse_instant(stamp) if stamp else None
    return moment.timestamp() if moment is not None else 0.0


def _reason(reason: str) -> KeptReason | Literal["superseded"]:
    """*reason*, narrowed to the reasons this module writes."""
    for known in _REASONS:
        if known == reason:
            return known
    raise ValueError(f"unknown cleanup reason {reason!r}")


_REASONS: Final[tuple[KeptReason | Literal["superseded"], ...]] = (
    "superseded",
    "keeper",
    "keeper_unresolved",
    "managed_dataset",
    "request_unreadable",
    "bound",
    "costly_fit",
    "params_unknown",
    "attempt_open",
    "held_by_request",
    "wider_coverage",
    "other_fit_scope",
    "young",
    "newest",
    "read_by",
    "absent",
)
