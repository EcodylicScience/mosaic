"""Retiring saved recipes without a submission, and the plan a person reviews first.

Removing older versions of a pipeline, cleaning up a project and deleting a
pipeline with its results submit no recipe. :func:`plan_retirement` judges the
runs that the retired recipes' requests wrote, against the plans of the
pipelines that stay (the **keepers**), by the rules of :mod:`~.cleanup`
(:func:`~.cleanup.judge_runs`). The runs a keeper expects are never candidates,
and neither is a run the dataset reads through a link.

Planning refuses nothing. A digest that cannot be retired is listed in
:attr:`RetirementPlan.not_retired` with its reason and left out:

- ``recipe_unreadable``: the digest is malformed, names no saved recipe, or its
  recipe does not read.
- ``keeper_recipe``: it is the recipe a keeper runs. Retiring it would strip the
  requests of a pipeline that stays of the protection of their runs.

A retired recipe with a request that has not ended is judged as any other, and
the rules that hold such a request's runs keep them.

Planning is a read. It takes no lock, writes nothing and loads no feature
definition, and its fingerprint (:func:`retirement_fingerprint`) names what a
removal may remove, so a confirm can tell that the dataset moved since the plan
was reviewed.

:func:`run_retirement` runs the approved form (:class:`~.model.RetirementSpec`) as
a job attempt. It judges the runs again and removes only the approved runs still
removable, recording each run in the attempt's run-log.
"""

from __future__ import annotations

import datetime
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .model import CleanupPolicy, CleanupTarget, RetirementKeeper, RetirementSpec

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.json_value import JsonValue

    from ..inventory.cache import InventoryCache
    from ..job import JobContext
    from .cleanup import CleanupOutcome, CleanupPlan, SupersededRun
    from .compatibility import DeclarationCatalog
    from .plan import Plan

__all__ = [
    "NotRetiredReason",
    "NotRetiredRecipe",
    "RetirementPlan",
    "plan_retirement",
    "retirement_fingerprint",
    "run_retirement",
]

type NotRetiredReason = Literal["recipe_unreadable", "keeper_recipe"]
"""Why a digest given to a retirement is not retired."""


@dataclass(frozen=True, slots=True)
class NotRetiredRecipe:
    """A digest a retirement was given and did not retire.

    Attributes:
        digest: The digest as given.
        reason: Why it is not retired.
        message: Why, in a sentence.
    """

    digest: str
    reason: NotRetiredReason
    message: str


@dataclass(frozen=True, slots=True)
class RetirementPlan:
    """What retiring recipes would remove and keep, as a person reviews it.

    Attributes:
        dataset_uuid: The ``uuid`` of the dataset's manifest.
        policy: The age and recency rules the runs were judged under.
        retired: The digests retired, in the current canonical form, sorted.
        not_retired: The digests given and not retired, in the order given.
        keepers: The pipelines that stay.
        plans: Each keeper's plan, in the order of *keepers*.
        cleanup: Every run the retired recipes' requests wrote that no keeper
            expects, removable or kept with its reason.
        planned_at: The instant the runs were judged at, as ISO 8601.
        fingerprint: Names what the plan found
            (:func:`retirement_fingerprint`).
    """

    dataset_uuid: str
    policy: CleanupPolicy
    retired: tuple[str, ...]
    not_retired: tuple[NotRetiredRecipe, ...]
    keepers: tuple[RetirementKeeper, ...]
    plans: tuple[Plan, ...]
    cleanup: CleanupPlan
    planned_at: str
    fingerprint: str

    @property
    def removable(self) -> tuple[SupersededRun, ...]:
        """The runs the retirement may remove."""
        return self.cleanup.removable

    @property
    def kept(self) -> tuple[SupersededRun, ...]:
        """The runs kept, each with its reason."""
        return self.cleanup.kept

    @property
    def spec(self) -> RetirementSpec:
        """The plan as it is approved, with the removable runs as its targets."""
        return RetirementSpec(
            retire=list(self.retired),
            keepers=list(self.keepers),
            policy=self.policy,
            approved=[
                CleanupTarget(storage=run.storage, run_id=run.run_id)
                for run in self.removable
            ],
            planned_at=self.planned_at,
            fingerprint=self.fingerprint,
        )


def plan_retirement(
    ds: Dataset,
    retire: Iterable[str],
    keepers: Sequence[RetirementKeeper],
    policy: CleanupPolicy,
    manager: str,
    *,
    inventory: InventoryCache | None = None,
    catalog: DeclarationCatalog | None = None,
    now: datetime.datetime | None = None,
) -> RetirementPlan:
    """Judge the runs the requests of the recipes in *retire* wrote. Writes nothing.

    Each keeper is planned over its scope as a replacement is in a submission's
    cleanup, without refusing. Every request of a recipe that is not retired keeps
    the runs it recorded, binds or pins, at any age. It takes no lock and loads no
    feature definition.

    Args:
        ds: The dataset.
        retire: The digests of the saved recipes to retire, in any canonical
            form the dataset holds.
        keepers: The pipelines that stay, each with the recipe of its current
            version and the scope to plan it over.
        policy: The age and recency rules.
        manager: The manager the plan acts for. It lifts the ``managed_dataset``
            hold when it equals the manifest's ``managed_by``; ``""`` lifts
            nothing.
        inventory: A held inventory, shared by every keeper's plan and the
            judging.
        catalog: The declarations.
        now: The instant the runs are judged at. ``None`` reads the clock.

    Returns:
        The plan, with its fingerprint.
    """
    from ..definitions.scope import definition_scope
    from ..inventory.cache import InventoryCache
    from .digest import recipe_digest
    from .resolve import declaration_catalog
    from .retirement import RetireUnknownRecipe, saved_digest

    moment = now or datetime.datetime.now(datetime.UTC)
    base = ds.base_dir
    kept_digests = {recipe_digest(keeper.recipe) for keeper in keepers}
    retired: set[str] = set()
    not_retired: list[NotRetiredRecipe] = []
    for digest in dict.fromkeys(retire):
        try:
            current = saved_digest(base, digest)
        except RetireUnknownRecipe as error:
            not_retired.append(
                NotRetiredRecipe(digest, "recipe_unreadable", str(error))
            )
            continue
        if current in kept_digests:
            not_retired.append(
                NotRetiredRecipe(
                    digest,
                    "keeper_recipe",
                    f"recipe {current!r} is the recipe of a pipeline that stays, "
                    f"so retiring it would remove the runs that pipeline's "
                    f"requests wrote",
                )
            )
            continue
        retired.add(current)

    held = InventoryCache(ds) if inventory is None else inventory
    with definition_scope(ds, load=False):
        declarations = declaration_catalog() if catalog is None else catalog
        plans, judged = _judge_retirement(
            ds,
            keepers,
            retired,
            policy,
            manager,
            inventory=held,
            catalog=declarations,
            now=moment,
            retired_before="",
        )
    dataset_uuid = ds.uuid or ""
    ordered = tuple(sorted(retired))
    return RetirementPlan(
        dataset_uuid=dataset_uuid,
        policy=policy,
        retired=ordered,
        not_retired=tuple(not_retired),
        keepers=tuple(keepers),
        plans=plans,
        cleanup=judged,
        planned_at=moment.isoformat(),
        fingerprint=retirement_fingerprint(
            dataset_uuid=dataset_uuid,
            policy=policy,
            retired=ordered,
            not_retired=not_retired,
            plans=plans,
            cleanup=judged,
        ),
    )


def run_retirement(
    ds: Dataset,
    plan: RetirementSpec,
    manager: str,
    *,
    ctx: JobContext,
    inventory: InventoryCache | None = None,
    catalog: DeclarationCatalog | None = None,
    now: datetime.datetime | None = None,
) -> CleanupOutcome:
    """Remove each run *plan* approved that is still removable, in the attempt *ctx*.

    Every keeper is planned again and the retired runs are judged again, with
    ``plan.planned_at`` as the cutoff of a retired request, under
    :func:`~.store.submission_lock` so no request is written between the two. The
    lock is released before the removals, and the attempts and requests that
    appear after the judging are read again before each removal
    (:class:`~.cleanup.RemovalWatch`). Only an approved run the judging finds
    removable is removed, with reason ``retired``; every other approved run is
    recorded kept with its reason (:func:`~.cleanup.remove_approved_runs`). It
    judges in the definition scope of *ds* and loads no definition.

    The run-log of *ctx* records each run, a removal that raised as a lost entry,
    and a heartbeat per run. The removals of an earlier attempt with the same
    execution id are read back, so a run is not removed twice and a removal it
    began is finished.

    Args:
        ds: The dataset.
        plan: The approved retirement.
        manager: The manager the retirement acts for. It lifts the
            ``managed_dataset`` hold when it equals the manifest's
            ``managed_by``; ``""`` lifts nothing.
        ctx: The job attempt, whose run-log records the removals.
        inventory: A held inventory, shared by the keepers' plans and the
            judging.
        catalog: The declarations.
        now: The instant ages are judged at. ``None`` reads the clock.

    Returns:
        What it removed, kept and failed to remove.

    Raises:
        ValueError: *ctx* has no run-log, because no removal is made without its
            record.
        Cancelled: The cancel token of *ctx* fired between runs.
        AllEntriesFailed: Every approved run whose removal was attempted raised,
            and no earlier attempt with the same execution id removed an
            approved run.
        RuntimeError: A run a keeper expects, whose directory existed before, is
            gone after.
    """
    from ..definitions.scope import definition_scope
    from ..inventory.cache import InventoryCache
    from ..job import Cancelled
    from ..run import AllEntriesFailed
    from .cleanup import RemovalWatch, read_prior_removals, remove_approved_runs
    from .resolve import declaration_catalog
    from .store import submission_lock

    log = ctx.run_log
    if log is None:
        raise ValueError(
            f"retirement {ctx.execution_id} has no run-log, and a run is never "
            f"removed without its record"
        )
    base = ds.base_dir
    ctx.set_total(len(plan.approved))
    prior = read_prior_removals(base, ctx.execution_id)
    held = InventoryCache(ds) if inventory is None else inventory
    with definition_scope(ds, load=False):
        declarations = declaration_catalog() if catalog is None else catalog
        # Before the runs are judged, so an attempt or a request that appears
        # between the judging and a removal is read again before that removal.
        watch = RemovalWatch(base, exclude="")
        with submission_lock(base):
            plans, judged = _judge_retirement(
                ds,
                plan.keepers,
                plan.retire,
                plan.policy,
                manager,
                inventory=held,
                catalog=declarations,
                now=now,
                retired_before=plan.planned_at,
            )
        outcome = remove_approved_runs(
            ds,
            plan.approved,
            execution_id=ctx.execution_id,
            judged=judged,
            replacements=plans,
            watch=watch,
            inventory=held,
            log=log,
            token=ctx.cancel_token,
            prior=prior,
            progress=lambda done, _total: ctx.heartbeat(done),
            label=f"retirement {ctx.execution_id}",
        )
    if outcome.status == "cancelled":
        raise Cancelled(f"retirement {ctx.execution_id} was cancelled between runs")
    removed_before = sum(
        prior.get((target.storage, target.run_id)) == "removed"
        for target in plan.approved
    )
    if outcome.failed and not outcome.removed and not removed_before:
        raise AllEntriesFailed(
            f"retirement {ctx.execution_id} failed to remove each of the "
            f"{len(outcome.failed)} approved runs it tried: "
            f"{', '.join(outcome.failed)}"
        )
    return outcome


def _judge_retirement(
    ds: Dataset,
    keepers: Sequence[RetirementKeeper],
    retire: Collection[str],
    policy: CleanupPolicy,
    manager: str,
    *,
    inventory: InventoryCache,
    catalog: DeclarationCatalog,
    now: datetime.datetime | None,
    retired_before: str,
) -> tuple[tuple[Plan, ...], CleanupPlan]:
    """Plan every keeper and judge the runs of the retired recipes against them.

    A plan and a removal judge through this one function, so the fingerprint a
    person reviews and the judging a removal repeats cannot differ in what they
    plan or judge. The caller must call it within a definition scope of *ds*.

    Args:
        ds: The dataset.
        keepers: The pipelines that stay.
        retire: The retired recipes, as their current digests.
        policy: The age and recency rules.
        manager: The manager the judging acts for.
        inventory: The held inventory the plans and the judging read.
        catalog: The declarations.
        now: The instant ages are judged at. ``None`` reads the clock.
        retired_before: A request of a retired recipe created at or after this
            instant is not retired. ``""`` for none.

    Returns:
        The keepers' plans, in order, and the judged runs.
    """
    from .cleanup import judge_runs
    from .plan import plan_pipeline

    plans = tuple(
        plan_pipeline(
            ds,
            keeper.recipe,
            scope=keeper.scope,
            inventory=inventory,
            catalog=catalog,
            refuse=False,
        )
        for keeper in keepers
    )
    judged = judge_runs(
        ds,
        plans,
        policy=policy,
        inventory=inventory,
        catalog=catalog,
        candidates="retired",
        planned_digest="",
        planned_name="",
        now=now,
        retire=retire,
        retired_before=retired_before,
        manager=manager,
    )
    return plans, judged


def retirement_fingerprint(
    *,
    dataset_uuid: str,
    policy: CleanupPolicy,
    retired: Iterable[str],
    not_retired: Iterable[NotRetiredRecipe],
    plans: Iterable[Plan],
    cleanup: CleanupPlan,
) -> str:
    """Name what a retirement's plan found, for a confirm to compare against.

    The digest covers the dataset's ``uuid``, the policy, the retired digests, the
    digests not retired with their reasons, each keeper plan's
    :func:`~.serialize.plan_fingerprint` and the removable runs, each set sorted.
    The instant of the plan, the sizes and the kept runs are left out, so a plan
    made again differs only when the runs it may remove or a keeper's plan
    differ.

    Args:
        dataset_uuid: The ``uuid`` of the dataset's manifest.
        policy: The age and recency rules.
        retired: The digests retired.
        not_retired: The digests not retired.
        plans: The keepers' plans.
        cleanup: The judged runs.

    Returns:
        The fingerprint.
    """
    from .digest import document_digest
    from .serialize import cleanup_fingerprint_parts, plan_fingerprint

    refused: list[JsonValue] = [
        [digest, reason]
        for digest, reason in sorted((each.digest, each.reason) for each in not_retired)
    ]
    retire: list[JsonValue] = [*sorted(retired)]
    keepers: list[JsonValue] = [*sorted(plan_fingerprint(plan) for plan in plans)]
    document: dict[str, JsonValue] = {
        "dataset": dataset_uuid,
        **cleanup_fingerprint_parts(policy, cleanup.removable),
        "retire": retire,
        "not_retired": refused,
        "keepers": keepers,
    }
    return document_digest(document)
