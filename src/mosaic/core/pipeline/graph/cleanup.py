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
3. Its feature declares ``costly_fit``, or the catalog does not know its
   feature (``costly_fit``).
4. Its ``params.json`` is absent or unreadable, so it could not be computed
   again (``params_unknown``).
5. A run-log that may still run names it (``attempt_open``).
6. Another request holds it (``held_by_request``): a running or orphaned
   request whatever its age, for the runs it recorded, binds or names in its
   steps' params, a running one also for every run of the features its recipe
   names, and an ended one for :data:`~.model.REQUEST_HOLD_FLOOR_HOURS` or
   ``min_age_hours``, whichever is longer.
7. It holds entries its replacement lacks (``wider_coverage``), or it is a
   scope-dependent fit over entries its replacement does not cover
   (``other_fit_scope``).
8. It is younger than ``min_age_hours`` (``young``).
9. It is among the ``keep_newest`` newest superseded runs of its storage
   (``newest``).
10. A run that is kept, or a step of the plan, reads it, through ``inputs``, a
    params reference or its fit record, directly or through other runs
    (``read_by``).

Planning is a read. It takes no lock and writes nothing, and its module imports
the standard library and pydantic alone; the inventory, the run-log readers and
the request store are imported inside the functions that use them. It never
imports the feature registry.
"""

from __future__ import annotations

import datetime
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from mosaic.core.entry import Entry

from .model import REQUEST_HOLD_FLOOR_HOURS, CleanupPolicy, FeatureStepSpec

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.scope import Scope
    from mosaic.runlog import RunLogSnapshot

    from ..feature_run_removal import RemovedRun
    from ..fit_scope import FitScope
    from ..inventory.cache import InventoryCache
    from mosaic.runlog import JsonlRunLog

    from ..job import CancelToken
    from ..inventory.model import AnyRecord
    from ..inventory.params import RunParams
    from .compatibility import DeclarationCatalog
    from .model import Recipe, Request
    from .plan import Plan

__all__ = [
    "CLEANUP_KIND",
    "CleanupOutcome",
    "CleanupPlan",
    "KeptReason",
    "SupersededRun",
    "plan_cleanup",
    "plan_with_cleanup",
    "run_cleanup",
]

CLEANUP_KIND: Final = "feature-cleanup"
"""The ``kind`` a cleanup's run-log records."""

FEATURE_ATTEMPT_KIND: Final = "feature"
"""The ``kind`` a feature run's run-log records, before it names its run."""

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
    opened, unnamed = _open_attempts(base, now=moment)
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

    _keep_what_is_read(ds, records, decided, opened, unnamed, fits, plan)

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


# --- running a cleanup ---------------------------------------------------------------


type CleanupStatus = Literal["finished", "failed", "cancelled"]
"""How a cleanup that ran ended."""


@dataclass(frozen=True, slots=True)
class CleanupOutcome:
    """What one run of a request's cleanup did.

    Attributes:
        execution_id: The cleanup's attempt, whose run-log records each run.
        status: How it ended.
        removed: The runs it removed.
        kept: The approved runs it kept, as ``(storage, run_id, reason)``.
        failed: The approved runs whose removal raised, as ``storage/run_id``.
        error: What failed the cleanup as a whole, ``""`` otherwise.
    """

    execution_id: str
    status: CleanupStatus
    removed: tuple[RemovedRun, ...] = ()
    kept: tuple[tuple[str, str, str], ...] = ()
    failed: tuple[str, ...] = ()
    error: str = ""


def run_cleanup(
    ds: Dataset,
    request: Request,
    *,
    token: CancelToken,
    inventory: InventoryCache | None = None,
    catalog: DeclarationCatalog | None = None,
    now: datetime.datetime | None = None,
) -> CleanupOutcome:
    """Remove the superseded runs that *request* approved, after its steps finished.

    The request is planned again with the run ids its attempts recorded, and only
    the approved runs that the plan still finds removable are removed
    (:func:`~mosaic.core.pipeline.feature_run_removal.remove_feature_run`). Every
    other approved run is recorded kept with its reason, and the set removed never
    grows past the one approved. Before each removal the attempts that may still
    run, or that ended since the cleanup began, are read again, and a run one of
    them names or reads is kept, as is a run that a request submitted since holds.
    A removal that an earlier attempt began is finished only under the same
    checks, and never for a run the plan now expects. The keepers whose
    directories existed before are checked to exist after.

    A run whose removal raises is recorded as a lost entry and the cleanup goes
    on. The cancel token is read between runs. The cleanup runs again from its
    run-log: a run it removed is not removed twice, and a removal it began is
    finished.

    Args:
        ds: The dataset.
        request: The request, whose ``cleanup`` is set.
        token: Fired to stop between runs.
        inventory: A held inventory, revalidated by the plan.
        catalog: The declarations.
        now: The instant ages are judged at. ``None`` reads the clock.

    Returns:
        What it removed and kept, and how it ended. A failure of the cleanup
        as a whole is recorded and returned, not raised, because a failed
        cleanup does not fail the request.

    Raises:
        ValueError: *request* has no cleanup.
    """
    import json
    import socket
    import traceback

    from mosaic.runlog import (
        JsonlRunLog,
        read_removal_records,
        run_log_dir,
        run_log_path,
    )

    from ..markers import current_boot_id, current_pid_namespace

    spec = request.cleanup
    if spec is None:
        raise ValueError(f"request {request.request_id!r} has no cleanup")
    base = ds.base_dir
    with JsonlRunLog(run_log_path(base, spec.execution_id), spec.execution_id) as log:
        # The boot and the pid namespace, as a step records them, so a cancel
        # signals this process and no other that reuses its pid.
        log.started(
            kind=CLEANUP_KIND,
            target=request.request_id,
            owner=request.owner,
            host=socket.gethostname(),
            pid=os.getpid(),
            boot_id=current_boot_id(),
            pid_ns=current_pid_namespace(),
        )
        log.set_total(len(spec.approved))
        prior = {
            (record["feature"], record["run_id"]): record["state"]
            for record in read_removal_records(run_log_dir(base), spec.execution_id)
        }
        try:
            outcome = _remove_approved(
                ds,
                request,
                log=log,
                token=token,
                prior=prior,
                inventory=inventory,
                catalog=catalog,
                now=now,
            )
        except Exception as error:
            blob = json.dumps(
                {
                    "type": type(error).__name__,
                    "message": str(error)[:2000],
                    "traceback": traceback.format_exc()[:8000],
                }
            )
            log.failed(blob)
            return CleanupOutcome(
                execution_id=spec.execution_id, status="failed", error=str(error)
            )
        if outcome.status == "cancelled":
            log.cancelled()
        else:
            log.finished()
        return outcome


def _remove_approved(
    ds: Dataset,
    request: Request,
    *,
    log: JsonlRunLog,
    token: CancelToken,
    prior: Mapping[tuple[str, str], str],
    inventory: InventoryCache | None,
    catalog: DeclarationCatalog | None,
    now: datetime.datetime | None,
) -> CleanupOutcome:
    """Remove each approved run still removable, recording every one in *log*."""
    import json

    from ..feature_run_removal import finish_feature_run_removal, remove_feature_run
    from ..index import feature_run_root
    from ..inventory.cache import InventoryCache
    from ..inventory.params import read_run_params
    from .plan import plan_pipeline
    from .request import load_recipe_for_request
    from .resolve import declaration_catalog
    from .rollup import request_rollup

    spec = request.cleanup
    assert spec is not None
    base = ds.base_dir
    held = InventoryCache(ds) if inventory is None else inventory
    declarations = declaration_catalog() if catalog is None else catalog
    # Before the runs are judged, so an attempt or a request that appears
    # between the judging and a removal is read again before that removal.
    watch = _Watch(base, exclude=request.request_id)
    recipe = load_recipe_for_request(base, request)
    rollup = request_rollup(base, request)
    recorded = {
        attempt.step_id: attempt.run_id for attempt in rollup.steps if attempt.run_id
    }
    plan = plan_pipeline(
        ds, recipe, request=request, inventory=held, catalog=declarations, refuse=False
    )
    short = {
        plan.step(attempt.step_id).storage_name
        for attempt in rollup.steps
        if attempt.entries_failed > 0 and attempt.step_id in plan.run_ids
    }
    found = plan_cleanup(
        ds,
        plan,
        policy=spec.policy,
        inventory=held,
        catalog=declarations,
        own_request=request,
        recorded=recorded,
        now=now,
        short_storages=short,
    )
    judged = {(run.storage, run.run_id): run for run in found.runs}
    keepers = _keepers(plan, recorded)
    kept_roots = [
        feature_run_root(ds, storage, run_id)
        for storage, keeper in keepers.items()
        for run_id in keeper.run_ids
        if feature_run_root(ds, storage, run_id).is_dir()
    ]

    removed: list[RemovedRun] = []
    kept: list[tuple[str, str, str]] = []
    failed: list[str] = []
    status: CleanupStatus = "finished"
    for done, target in enumerate(spec.approved, start=1):
        if token.is_cancelled():
            status = "cancelled"
            break
        key = (target.storage, target.run_id)
        run_root = feature_run_root(ds, target.storage, target.run_id)
        judged_run = judged.get(key)
        reason, detail = "", ""
        if prior.get(key) == "removed":
            pass
        elif judged_run is not None and not judged_run.removable:
            reason, detail = judged_run.reason, judged_run.detail
        elif (
            target.storage in keepers
            and target.run_id in keepers[target.storage].run_ids
        ):
            reason, detail = "keeper", "the request's plan now expects it"
        elif judged_run is None:
            if prior.get(key) in ("interrupted", "failed"):
                # Its rows are gone and its files may not be: finish only what
                # nothing has started to read or write since.
                hold = watch.holding(ds, key) or _read_by_any(
                    held.get().of_kind("feature"), key
                )
                if hold is not None:
                    reason, detail = hold
                else:
                    removed.append(
                        finish_feature_run_removal(
                            ds, target.storage, target.run_id, log=log
                        )
                    )
            else:
                reason, detail = "absent", "it is no longer in the feature index"
        else:
            hold = watch.holding(ds, key)
            params = read_run_params(run_root).params
            if hold is not None:
                reason, detail = hold
            elif params is None:
                reason, detail = "params_unknown", "its params.json no longer reads"
            else:
                try:
                    removed.append(
                        remove_feature_run(
                            ds,
                            target.storage,
                            target.run_id,
                            params=params,
                            log=log,
                            reason="superseded",
                            superseded_by=judged_run.superseded_by,
                        )
                    )
                except (OSError, RuntimeError) as error:
                    failed.append(f"{target.storage}/{target.run_id}")
                    log.entry_failed(
                        f"{target.storage}/{target.run_id}",
                        json.dumps(
                            {"type": type(error).__name__, "message": str(error)}
                        ),
                    )
        if reason:
            kept.append((target.storage, target.run_id, reason))
            log.run_kept(
                feature=target.storage,
                run_id=target.run_id,
                reason=reason,
                detail=detail,
            )
        log.heartbeat(done, len(spec.approved))

    lost = [root for root in kept_roots if not root.is_dir()]
    if lost:
        raise RuntimeError(
            f"the cleanup of request {request.request_id} left kept runs missing: "
            f"{', '.join(str(root) for root in lost)}"
        )
    return CleanupOutcome(
        execution_id=spec.execution_id,
        status=status,
        removed=tuple(removed),
        kept=tuple(kept),
        failed=tuple(failed),
    )


class _Watch:
    """What may read a run, read again before each removal.

    An attempt that may still run, or one that ended after the watch began, is
    read for the run it names and the runs it reads; one already ended when the
    watch began is not read again. A request file that appeared after the watch
    began holds its binds, its recorded runs and every run of the features its
    recipe names, and one that does not read holds every run. The watch begins
    before the runs are judged, so nothing that appears between the judging and
    a removal is missed.
    """

    def __init__(self, base_dir: Path, *, exclude: str) -> None:
        from mosaic.runlog import TERMINAL_STATUSES, reduce_run_log, run_log_dir

        from .store import requests_root

        self._base: Path = base_dir
        self._runs: Path = run_log_dir(base_dir)
        self._requests: Path = requests_root(base_dir)
        self._exclude: str = exclude
        self._ended: set[str] = set()
        for path in sorted(self._runs.glob("*.jsonl")):
            snapshot = reduce_run_log(path)
            if snapshot is not None and snapshot["status"] in TERMINAL_STATUSES:
                self._ended.add(path.stem)
        self._known_requests: frozenset[str] = frozenset(self._request_names())

    def _request_names(self) -> list[str]:
        if not self._requests.is_dir():
            return []
        return sorted(path.stem for path in self._requests.glob("*.json"))

    def holding(self, ds: Dataset, key: tuple[str, str]) -> tuple[str, str] | None:
        """Why run *key* must be kept now, as ``(reason, detail)``, or ``None``."""
        return self._attempt_holding(ds, key) or self._request_holding(key)

    def _attempt_holding(
        self, ds: Dataset, key: tuple[str, str]
    ) -> tuple[str, str] | None:
        from mosaic.runlog import TERMINAL_STATUSES, reduce_run_log

        from ..fit_scope import read_fit_scope
        from ..inventory.params import read_run_params

        features = ds.get_root("features")
        for path in sorted(self._runs.glob("*.jsonl")):
            name = path.stem
            if name in self._ended:
                continue
            snapshot = reduce_run_log(path)
            if snapshot is None:
                continue
            ended = snapshot["status"] in TERMINAL_STATUSES
            if not ended and not _may_still_run(snapshot):
                continue
            run_id = snapshot["run_id"]
            if not run_id:
                if (
                    not ended
                    and snapshot["kind"] == FEATURE_ATTEMPT_KIND
                    and _reads_of(snapshot["target"], None, None).covers(*key)
                ):
                    return (
                        "attempt_open",
                        f"attempt {name} has not named its run and may read it",
                    )
                continue
            if run_id == key[1]:
                return ("attempt_open", f"attempt {name} writes run {run_id}")
            for params_path in features.glob(f"*/{run_id}/params.json"):
                run_root = params_path.parent
                reads = _reads_of(
                    run_root.parent.name,
                    read_run_params(run_root).params,
                    read_fit_scope(run_root),
                )
                if reads.covers(*key):
                    return ("attempt_open", f"attempt {name} reads it")
        return None

    def _request_holding(self, key: tuple[str, str]) -> tuple[str, str] | None:
        from .request import load_recipe_for_request
        from .rollup import request_rollup
        from .step_preflight import StepRefused
        from .store import RequestUnreadable, load_request

        slug = key[0].split(_FROM, 1)[0]
        for name in self._request_names():
            if name in self._known_requests or name == self._exclude:
                continue
            label = f"request {name}, submitted during the cleanup,"
            try:
                request = load_request(self._base, name)
                recipe = load_recipe_for_request(self._base, request)
            except (RequestUnreadable, StepRefused, OSError) as error:
                return ("request_unreadable", f"{label} does not read: {error}")
            if any(
                (bound.feature, bound.run_id) == key for bound in request.bind.values()
            ):
                return ("held_by_request", f"{label} pins it")
            rollup = request_rollup(self._base, request)
            if any(attempt.run_id == key[1] for attempt in rollup.steps):
                return ("held_by_request", f"{label} recorded it")
            for step in recipe.steps:
                if not isinstance(step, FeatureStepSpec):
                    continue
                if step.feature == slug:
                    return ("held_by_request", f"{label} runs {slug}")
                if _literal_reads(step.params).covers(*key):
                    return ("held_by_request", f"{label} names it")
        return None


def _read_by_any(
    records: Iterable[AnyRecord], key: tuple[str, str]
) -> tuple[str, str] | None:
    """The first feature run in *records* that reads run *key*, as a kept reason."""
    from ..fit_scope import read_fit_scope

    for record in records:
        if (record.name, record.run_id) == key:
            continue
        fit = read_fit_scope(record.run_root) if record.run_root is not None else None
        if _reads_of(record.name, record.params, fit).covers(*key):
            return ("read_by", f"run {record.name}/{record.run_id} reads it")
    return None


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
    storages: dict[str, str] = field(default_factory=dict[str, str])
    unreadable: str = ""


def _request_holds(
    base_dir: Path, *, now: datetime.datetime, hold_hours: float, exclude: str
) -> _Holds:
    """What the dataset's other requests hold.

    A running or orphaned request holds the run ids its attempts recorded, its
    binds and the runs its steps' params name outright, whatever its age, and a
    running one also every run of each feature its recipe names. A request that has ended holds its recorded runs and binds
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
        if verdict in ("running", "orphaned"):
            try:
                recipe = load_recipe_for_request(base_dir, request)
            except StepRefused:
                continue
            for step in recipe.steps:
                if not isinstance(step, FeatureStepSpec):
                    continue
                if verdict == "running":
                    _ = holds.slugs.setdefault(step.feature, label)
                # A run its params name outright, which no step of it writes.
                named = _literal_reads(step.params)
                for pair in named.pairs:
                    _ = holds.pairs.setdefault(pair, label)
                for storage in named.storages:
                    _ = holds.storages.setdefault(storage, label)
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


def _open_attempts(
    base_dir: Path, *, now: datetime.datetime
) -> tuple[dict[str, str], dict[str, str]]:
    """The attempts that may still run: by the run each names, and those naming none.

    An attempt that has not ended is open while its process may still run
    (:func:`_may_still_run`). One that names its run is returned by that run id
    with its execution id. A feature attempt that names none yet is returned by
    its execution id with the storage it writes, which tells what its inputs
    are: it is resolving them, and may read any run of theirs.
    """
    from mosaic.runlog import TERMINAL_STATUSES, list_runs, run_log_dir

    opened: dict[str, str] = {}
    unnamed: dict[str, str] = {}
    for snapshot in list_runs(run_log_dir(base_dir))["items"]:
        if snapshot["status"] in TERMINAL_STATUSES:
            continue
        if not _may_still_run(snapshot, now=now):
            continue
        if snapshot["run_id"]:
            _ = opened.setdefault(snapshot["run_id"], snapshot["execution_id"])
        elif snapshot["kind"] == FEATURE_ATTEMPT_KIND and snapshot["target"]:
            unnamed[snapshot["execution_id"]] = snapshot["target"]
    return opened, unnamed


def _may_still_run(
    snapshot: RunLogSnapshot, *, now: datetime.datetime | None = None
) -> bool:
    """Whether an attempt's process may still run: ``alive``, or ``unknown`` elsewhere."""
    from .liveness import is_local, snapshot_liveness

    liveness = snapshot_liveness(snapshot, now=now)
    return liveness == "alive" or (
        liveness == "unknown" and not is_local(snapshot["host"], snapshot["pid_ns"])
    )


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
    if declaration is None:
        return (
            "costly_fit",
            f"{slug} is not in the catalog, so whether it is costly to fit again "
            f"is unknown",
        )
    if declaration.costly_fit:
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
        or holds.storages.get(record.name)
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
    unnamed: Mapping[str, str],
    fits: Mapping[tuple[str, str], FitScope | None],
    plan: Plan,
) -> None:
    """Keep every removable run that a kept run reads, until nothing changes.

    The readers are every feature run of the dataset that is not removable, the
    run of every attempt that may still run, found by its ``params.json``,
    which an attempt writes once it names its run, an attempt that has not
    named its run yet, read by the storage it writes as a run of unknown params
    is, and every feature step of *plan*, which reads what its resolved inputs
    and params name.
    """
    from ..fit_scope import read_fit_scope
    from ..inventory.params import read_run_params

    readers: dict[tuple[str, str], _Reads] = {}
    for planned in plan.steps:
        if planned.kind == "feature":
            readers[(f"step {planned.step_id!r} of the plan", "")] = _literal_reads(
                {
                    **{
                        f"inputs[{i}]": item
                        for i, item in enumerate(planned.spec.inputs)
                    },
                    **planned.spec.params,
                }
            )
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
    for execution_id, storage in unnamed.items():
        readers[(f"attempt {execution_id}, which has not named its run,", "")] = (
            _reads_of(storage, None, None)
        )
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
                    name = f"run {reader[0]}/{reader[1]}" if reader[1] else reader[0]
                    decided[key] = ("read_by", f"{name} reads it")
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
        named = _literal_reads(
            {
                **{f"inputs[{i}]": item for i, item in enumerate(params.inputs)},
                **params.params,
            }
        )
        pairs.update(named.pairs)
        storages.update(named.storages)
    return _Reads(pairs=frozenset(pairs), storages=frozenset(storages))


def _literal_reads(values: Mapping[str, object]) -> _Reads:
    """What the ``{feature, run_id}`` references among *values* read.

    A reference without a run id reads every run of its storage, as it resolves
    to whichever run is current when its step runs.
    """
    pairs: set[tuple[str, str]] = set()
    storages: set[str] = set()
    for value in values.values():
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
