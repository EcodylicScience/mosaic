"""Keeping held-out groups out of a fit, and saying what a new group would refit.

A dataset's manifest may give a group the role ``test`` or ``sealed``
(:mod:`mosaic.core.splits`). A step that learns from the entries it covers must
not cover such a group, or the held-out score measures what the model was fitted
on. This module holds the two questions a planner asks of that rule. Both are
pure: they read the facts a plan already resolved and touch no dataset.

**The guard** (:func:`held_out_fit_problem`) refuses a step whose identity covers
its entries (``scope_dependent``) and that learns from them (``fits``) when its
target holds a held-out group. Both flags are read off the *built* feature,
because several features narrow them per instance: a step given a ``model`` to
apply fits nothing and may cover the test group. A scope-free fitter (a
classifier fitted from a templates run) is not flagged here, because its fit set
is the templates step's target and that step is.

**The refit set** (:func:`refit_set`) answers "if these groups joined the
dataset, which steps would get a new run?". A ``scope_dependent`` step whose
target would take the group hashes a new entry set, and so does every step
downstream of it through an edge that enters an identity. Each moved step is a
*refit* when it fits and a *re-score* when it only moves because an input moved
or it only scores. It is the basis of the ``assume_added`` check of
``plan_recipe`` and ``mosaic pipeline plan``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

from mosaic.core.splits import groups_with_role, held_out_groups, role_of

from .topo import edges, topological_order
from .wire import Problem

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from mosaic.core.entry import Entry
    from mosaic.core.manifest import SplitRole
    from mosaic.core.scope import Scope

    from .model import Recipe
    from .plan import Plan, PlannedStep

__all__ = [
    "MovedStep",
    "StepFacts",
    "assume_added_cautions",
    "deferred_fit_cautions",
    "deferred_scope_problem",
    "held_out_fit_problem",
    "listed",
    "refit_set",
    "step_facts",
]

_PLACEHOLDER_SEQUENCE: Final = ""
"""The sequence of the entry a group that has not joined yet is asked about."""

_LISTED_GROUPS: Final = 8
"""How many groups a message names before it counts the rest."""

type Joins = Literal["yes", "maybe", "no"]


# --- the guard -----------------------------------------------------------------


def held_out_fit_problem(
    planned: PlannedStep,
    target: Iterable[Entry],
    splits: Mapping[str, SplitRole],
    *,
    narrowed: bool,
) -> Problem | None:
    """The problem of a fitting step whose *target* holds a held-out group.

    Args:
        planned: The planned step, whose ``scope_dependent`` and ``fits`` were
            read off the built feature.
        target: The entries the step is planned over, its step scope applied.
        splits: The role of each group the manifest names.
        narrowed: Whether the step has a step scope of its own, which decides
            where the problem points.

    Returns:
        ``None`` unless the step is ``scope_dependent`` and ``fits`` and its
        target holds a group of role ``test`` or ``sealed``. The problem names
        the step, the groups with their roles and the two ways out.
    """
    if not (planned.scope_dependent and planned.fits) or not splits:
        return None
    held = set(held_out_groups(splits))
    inside = sorted({group for group, _ in target} & held)
    if not inside:
        return None
    step = planned.step_id
    where = f"step_scopes.{step}" if narrowed else "scope"
    return Problem(step, where, _leak_message(planned, inside, splits))


def deferred_scope_problem(
    planned: PlannedStep,
    splits: Mapping[str, SplitRole],
    *,
    request_scope: Scope | None,
    step_scope: Scope | None,
) -> Problem | None:
    """The problem of a fitting step whose scope names a held-out group outright.

    A graph that converts first has no entries to ask :func:`held_out_fit_problem`
    about, but a step scope or a request scope that *names* a group (``groups``,
    or the groups of ``entries``) says where the step reads whatever the
    conversion writes. A selector of ``sequences`` alone names no group and is
    left to :func:`deferred_fit_cautions`.

    Args:
        planned: The planned step.
        splits: The role of each group the manifest names.
        request_scope: The selector the plan was asked for, ``None`` for none.
        step_scope: The step's own scope, ``None`` when it has none.

    Returns:
        ``None`` unless the step is ``scope_dependent`` and ``fits`` and a scope
        names a group of role ``test`` or ``sealed``. The step scope is read first,
        and the problem points at it.
    """
    if not (planned.scope_dependent and planned.fits) or not splits:
        return None
    step = planned.step_id
    for selector, where in (
        (step_scope, f"step_scopes.{step}"),
        (request_scope, "scope"),
    ):
        inside = sorted(_named_held_out(selector, splits))
        if inside:
            return Problem(step, where, _leak_message(planned, inside, splits))
    return None


def _named_held_out(scope: Scope | None, splits: Mapping[str, SplitRole]) -> set[str]:
    """The held-out groups that *scope* names by group or by an entry."""
    if scope is None or scope.is_unset:
        return set()
    pairs = scope.entry_pairs
    named = scope.groups if pairs is None else [group for group, _ in pairs]
    return set(named or ()) & set(held_out_groups(splits))


def _leak_message(
    planned: PlannedStep, inside: Sequence[str], splits: Mapping[str, SplitRole]
) -> str:
    """The sentence of a fit that covers the held-out groups *inside*."""
    named = listed([f"{group!r} ({role_of(splits, group)})" for group in inside])
    step = planned.step_id
    reference = json.dumps({"step": step})
    return (
        f"{planned.runs} fits on the entries it covers, and those include the "
        f"held-out group(s) {named}. Scope {step!r} to the training groups "
        f"({_train_groups(splits)}) with a step scope, and use what it fits on the "
        f"other groups in a step that reads it as {reference} in its model, "
        "templates or label_map param"
    )


def _train_groups(splits: Mapping[str, SplitRole]) -> str:
    """The training groups of *splits*, for a message that scopes a step to them."""
    train = groups_with_role(splits, "train")
    return listed([repr(group) for group in train]) or "the training groups"


def deferred_fit_cautions(plan: Plan) -> tuple[Problem, ...]:
    """What the guard cannot yet see in a plan that waits on a conversion.

    A plan for a dataset with no tracks, whose graph converts raw files, has no
    entries to ask the guard about, so a fitting step over every entry passes it.
    The step is refused when it runs, after the steps above it have, if the
    conversion writes a group of role ``test`` or ``sealed``. This names each such
    step up front. It is a caution because nothing can enumerate the groups the
    conversion will write. A step bound to a run is not executed and is left out,
    as are the steps whose scope names a held-out group, which
    :func:`deferred_scope_problem` refuses.

    Args:
        plan: The plan. Its ``deferred_by``, ``splits``, ``bound`` and
            ``request_scope`` are read.

    Returns:
        One caution for each step that is ``scope_dependent``, ``fits``, is not
        bound and has no step scope. Empty unless the plan is deferred and the
        manifest names a held-out group.
    """
    held = held_out_groups(plan.splits)
    if plan.deferred_by is None or not held:
        return ()
    if _named_held_out(plan.request_scope, plan.splits):
        return ()
    named = listed([f"{group!r} ({role_of(plan.splits, group)})" for group in held])
    return tuple(
        Problem(
            planned.step_id,
            "scope",
            f"{planned.runs} fits on the entries it covers, which are the ones "
            f"{plan.deferred_by!r} converts, and the dataset holds no tracks to "
            f"check them against. If that includes the held-out group(s) {named} "
            f"this step is refused when it runs. Scope {planned.step_id!r} to "
            f"the training groups ({_train_groups(plan.splits)}) with a step scope",
        )
        for planned in plan.steps
        if planned.scope_dependent
        and planned.fits
        and planned.step_id not in plan.step_scopes
        and planned.step_id not in plan.bound
    )


# --- the refit set --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StepFacts:
    """What :func:`refit_set` needs to know of one step beyond the recipe.

    Attributes:
        step_id: The step's id.
        scope_dependent: Whether its identity covers the set of entries it is
            asked for.
        fits: Whether it learns from them.
        scope: The step scope narrowing it, ``None`` when it has none.
    """

    step_id: str
    scope_dependent: bool = False
    fits: bool = False
    scope: Scope | None = None


@dataclass(frozen=True, slots=True)
class MovedStep:
    """One step whose run would change if the assumed groups joined.

    Attributes:
        step_id: The step's id.
        action: ``"refit"`` when the step fits, and ``"re-score"`` when it only
            moves because an input moved or it only scores.
        joins: The assumed groups its own target would take, empty when it moves
            only because a step above it did.
        via: The steps above it that moved, empty when it moves on its own.
        uncertain: Whether the answer rests on a ``sequences`` selector, which
            cannot say whether a group that is not indexed yet holds a named
            sequence. The answer then errs towards moving.
    """

    step_id: str
    action: Literal["refit", "re-score"]
    joins: tuple[str, ...] = ()
    via: tuple[str, ...] = ()
    uncertain: bool = field(default=False)


def _joins(scope: Scope | None, group: str) -> Joins:
    """Whether *scope* would take a group that is not in the dataset yet.

    ``Scope.selects`` is asked about a placeholder entry of the group. An
    ``entries`` selector names exact entries, so it may take the group when it
    names an entry of it, which nothing can tell before the group is indexed. A
    ``sequences`` selector may match a sequence the group holds, likewise.
    """
    if scope is None or scope.is_unset:
        return "yes"
    pairs = scope.entry_pairs
    if pairs is not None:
        return "maybe" if any(named == group for named, _ in pairs) else "no"
    placeholder = (group, _PLACEHOLDER_SEQUENCE)
    if scope.sequences is None:
        return "yes" if scope.selects(placeholder) else "no"
    unsequenced = scope.model_copy(update={"sequences": None})
    return "maybe" if unsequenced.selects(placeholder) else "no"


def _both(first: Joins, second: Joins) -> Joins:
    if "no" in (first, second):
        return "no"
    return "maybe" if "maybe" in (first, second) else "yes"


def refit_set(
    recipe: Recipe,
    facts: Mapping[str, StepFacts],
    groups: Iterable[str],
    *,
    scope: Scope | None = None,
) -> tuple[MovedStep, ...]:
    """The steps whose run would change if *groups* joined the dataset.

    A ``scope_dependent`` step moves when both the request's *scope* and its own
    step scope would take a group. A step moves when a step it reads through an
    edge that enters its identity moved, which is every edge but ``after``.

    Args:
        recipe: The recipe, read for its edges and its order.
        facts: What each step declares, keyed by step id. A step missing from it
            is neither ``scope_dependent`` nor fitting, so it moves only with a
            parent.
        groups: The groups assumed to join.
        scope: The request's selector. ``None`` and an unset one cover the
            dataset, so any new group joins.

    Returns:
        The moved steps in topological order. Pure: the same inputs always give
        the same answer.
    """
    assumed = tuple(dict.fromkeys(groups))
    parents: dict[str, list[str]] = {}
    for edge in edges(recipe):
        if edge.site != "after" and edge.producer not in parents.setdefault(
            edge.consumer, []
        ):
            parents[edge.consumer].append(edge.producer)
    moved: dict[str, MovedStep] = {}
    for step in topological_order(recipe):
        fact = facts.get(step.id, StepFacts(step.id))
        taken: list[tuple[str, Joins]] = []
        if fact.scope_dependent:
            for group in assumed:
                answer = _both(_joins(scope, group), _joins(fact.scope, group))
                if answer != "no":
                    taken.append((group, answer))
        above = tuple(parent for parent in parents.get(step.id, ()) if parent in moved)
        if not taken and not above:
            continue
        moved[step.id] = MovedStep(
            step_id=step.id,
            action="refit" if fact.fits else "re-score",
            joins=tuple(group for group, _ in taken),
            via=above,
            uncertain=any(answer == "maybe" for _, answer in taken)
            or any(moved[parent].uncertain for parent in above),
        )
    return tuple(moved.values())


def step_facts(plan: Plan) -> dict[str, StepFacts]:
    """The :class:`StepFacts` of every step of *plan*, keyed by step id."""
    return {
        planned.step_id: StepFacts(
            step_id=planned.step_id,
            scope_dependent=planned.scope_dependent,
            fits=planned.fits,
            scope=plan.step_scopes.get(planned.step_id),
        )
        for planned in plan.steps
    }


def listed(items: Sequence[str], limit: int = _LISTED_GROUPS) -> str:
    """The first *limit* of *items* joined by commas, then a count of the rest."""
    shown = ", ".join(items[:limit])
    rest = len(items) - limit
    return shown + (f" and {rest} more" if rest > 0 else "")


def _listed(names: Sequence[str]) -> str:
    return listed([repr(name) for name in names])


def assume_added_cautions(
    recipe: Recipe,
    plan: Plan,
    groups: Iterable[str],
    *,
    scope: Scope | None = None,
) -> tuple[Problem, ...]:
    """What *plan* would refit if *groups* joined the dataset, as cautions.

    A group already in the plan's scope is reported as such and left out of the
    answer: planning it shows what it does. For the others the caution lists the
    steps that would refit and the ones that would only re-score. A refit step
    that would take a group whose role is ``test`` or ``sealed`` is marked, since
    the planner refuses that fit. A step marked with ``?`` moves only if a
    ``sequences`` selector matches a sequence the new group holds.

    Args:
        recipe: The recipe that was planned.
        plan: Its plan, read for the steps' flags and step scopes.
        groups: The groups assumed to join.
        scope: The selector the plan was asked for, ``None`` for none.

    Returns:
        Cautions at ``assume_added``, in the order the groups were given. None
        refuses a plan.
    """
    present = {group for group, _ in plan.scope}
    named = tuple(dict.fromkeys(groups))
    found: list[Problem] = [
        Problem(
            "",
            "assume_added",
            f"group {group!r} is already in this plan's scope, so there is "
            "nothing to assume; the plan shows what it covers",
        )
        for group in named
        if group in present
    ]
    new = tuple(group for group in named if group not in present)
    if not new:
        return tuple(found)
    moved = refit_set(recipe, step_facts(plan), new, scope=scope)
    held = set(held_out_groups(plan.splits))
    subject = f"group {new[0]!r}" if len(new) == 1 else f"groups {_listed(new)}"
    roles = ", ".join(
        f"{group!r} is {role}" for group in new if (role := role_of(plan.splits, group))
    )
    note = f" ({roles})" if roles else ""
    if not moved:
        found.append(
            Problem(
                "",
                "assume_added",
                f"if {subject} joined{note}, no step's run would change",
            )
        )
        return tuple(found)

    def label(step: MovedStep) -> str:
        leaked = sorted(held & set(step.joins))
        text = step.step_id + ("?" if step.uncertain else "")
        if step.action == "refit" and leaked:
            text += f" (refused: its fit would include {_listed(leaked)})"
        return text

    refits = [label(step) for step in moved if step.action == "refit"]
    rescores = [label(step) for step in moved if step.action == "re-score"]
    parts: list[str] = []
    if refits:
        parts.append("these steps would refit: " + ", ".join(refits))
    if rescores:
        parts.append("these would only re-score: " + ", ".join(rescores))
    if any(step.uncertain for step in moved):
        parts.append(
            "? marks a step that moves only if its sequences or entries selector "
            "matches something the new group holds"
        )
    found.append(
        Problem("", "assume_added", f"if {subject} joined{note}, " + "; ".join(parts))
    )
    return tuple(found)
