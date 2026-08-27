"""What a step checks before it does any work, and how it says no.

A step that is about to compute something is the last party in a position to
notice that what it is about to read is not what was intended. Nothing is
re-planning the graph while it runs -- that is the price of deciding the whole
thing at submit -- so the observation moves to the party that cares, which is the
consumer, at its own start.

**Coverage counts cannot detect wrongness.** A hundred and twenty parquets of NaN
is a hundred and twenty of a hundred and twenty, and no count will ever notice.
That is the single weakness of deriving status from the record, and it is why
these checks are here: each is a predicate that already exists somewhere, run
where a refusal is still free. A general check registry, and any check that reads
the data itself, wait for a third caller.

**A refusal is not a new terminal status.** It is an ordinary failure carrying a
reason: the exit code is :data:`REFUSED_EXIT_CODE`, the run-log status stays
``failed``, and the reason travels in ``error_json``. Adding a status would mean
adding a member to ``runlog.TERMINAL_STATUSES``, which three repositories read
and mosaic-api's sweeper reaps -- the same reason ``partial`` was kept out of it.

**``allow_partial`` answers exactly one of these.** A shortfall is a question
about *how much*, and a person may decide to proceed over less. A digest that
does not match, a version that moved, a variant that disagrees and an upstream
that produced nothing are not questions about how much, and no flag here unlocks
them.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final, Literal

from ..manifest import refuse_mixed_track_schemas
from .plan import COMPLETE_STATUSES, MISSING_SAMPLE, Plan, PlannedStep
from .resolve import build_step_feature

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.entry import Entry
    from mosaic.core.json_value import JsonValue


__all__ = [
    "REFUSED_EXIT_CODE",
    "CoverageShortfall",
    "RefusalReason",
    "StepRefused",
    "preflight",
    "refuse_mixed_schemas",
]

REFUSED_EXIT_CODE: Final = 65
"""The exit code a step uses when it refuses before doing any work.

Reserved so a driver can tell a refusal from a crash without parsing anything,
and chosen to land where ``terminal_status_for_exit`` already maps it: not zero,
not the cooperative-cancel code, not negative. The ledger row therefore reads
``failed``, which is what it is, with the reason beside it in ``error_json``.
"""

type RefusalReason = Literal[
    "coverage_shortfall",
    "upstream_empty",
    "schema_family_mismatch",
    "variant_mismatch",
    "version_moved",
    "parent_unrecorded",
    "recipe_missing",
    "digest_mismatch",
]
"""Every way a step can refuse before running. A closed set on purpose.

It crosses two repository boundaries as the ``reason`` field of ``error_json``,
so a new member is a wire addition rather than a local choice, and a free-text
reason would be one nobody downstream can branch on.
"""


class StepRefused(RuntimeError):
    """A step declined to run, and said why.

    Attributes:
        reason: Which refusal this is.
        step_id: The step that refused.
        detail: The numbers or names that make the reason actionable. JSON, so it
            travels in the run-log unchanged.
    """

    def __init__(
        self,
        reason: RefusalReason,
        step_id: str,
        message: str,
        detail: dict[str, JsonValue] | None = None,
    ) -> None:
        self.reason: RefusalReason = reason
        self.step_id: str = step_id
        self.detail: dict[str, JsonValue] = detail or {}
        super().__init__(message)

    def error_json(self) -> str:
        """The refusal as the ``error_json`` blob a ledger row carries."""
        payload: dict[str, JsonValue] = {
            "reason": self.reason,
            "step": self.step_id,
            "message": str(self),
            **self.detail,
        }
        return json.dumps(payload)


class CoverageShortfall(StepRefused):
    """A step cannot be run over everything it was planned for.

    Raised rather than proceeded through, because for a ``scope_dependent`` step
    it is a scientific question rather than a maintenance one: a model fitted on
    89 sequences is a different model from one fitted on 90, and mosaic says so
    by giving it a different name. Proceeding is a decision, and
    ``allow_partial`` is where that decision is recorded.
    """

    def __init__(self, step_id: str, covered: int, target: int, detail: str) -> None:
        self.covered: int = covered
        self.target: int = target
        super().__init__(
            "coverage_shortfall",
            step_id,
            f"step {step_id!r} would run over {covered} of {target} entries: "
            f"{detail}. Its identity covers the set it was fitted on, so this is "
            f"a different run from the one planned. Pass allow_partial to proceed "
            f"deliberately, or complete the steps above it first.",
            {"covered": covered, "target": target, "detail": detail},
        )


def preflight(
    ds: Dataset, plan: Plan, step_id: str, *, allow_partial: bool = False
) -> None:
    """Refuse *step_id* if what it is about to read is not what was intended.

    Args:
        ds: The dataset, for the checks that read the tracks index.
        plan: This step's graph, resolved against *ds* moments ago.
        step_id: The step about to run.
        allow_partial: Whether a shortfall has been answered.

    Raises:
        StepRefused: Carrying the reason and the numbers behind it.
    """
    planned = plan.step(step_id)
    _refuse_empty_upstream(plan, planned)
    _refuse_variant_disagreement(ds, plan, planned)
    if not allow_partial:
        refuse_shortfall(plan, planned)


def refuse_mixed_schemas(ds: Dataset, step_id: str) -> None:
    """Refuse a dataset whose tracks tables are of incompatible schemas.

    ``trex_v1``'s spatial columns are centimetres and its ``X`` is a head
    position, so mixing one with a ``mosaic_v1`` table is mixing units and
    landmarks. The predicate is the loader's own, called here so the refusal
    arrives before the work rather than partway through it.

    **Asked of the whole dataset, and before planning rather than after.** The
    term a feature identifier carries for the tables it reads is deliberately
    scope-free -- a scope-free feature must get one identifier for every scope --
    so resolving *any* feature's identity against a mixed dataset already fails.
    Running the check first is what turns that into a refusal naming the schemas,
    instead of an exception out of the middle of a hash.
    """
    try:
        refuse_mixed_track_schemas(ds)
    except ValueError as exc:
        raise StepRefused("schema_family_mismatch", step_id, str(exc)) from exc


def refuse_shortfall(plan: Plan, planned: PlannedStep) -> None:
    """Refuse a step whose inputs do not cover what it was planned over.

    Only for a ``scope_dependent`` step, and the asymmetry is the point: a
    scope-free step running over 89 of 90 entries produces 89 correct outputs
    under the identifier they belong to, and the ninetieth arrives later under
    the same one. A scope-dependent step running over 89 produces *one* artifact
    that is not the artifact anyone asked for, under a name saying it is.
    """
    if planned.kind != "feature":
        return
    if not build_step_feature(planned.spec).scope_dependent:
        return
    short = [
        parent
        for parent in planned.parents
        if not _covers_scope(plan, parent, plan.scope)
    ]
    if not short:
        return
    covered = min(
        (len(plan.step(parent).coverage.covered) for parent in short), default=0
    )
    missing = sorted(
        str(entry) for parent in short for entry in plan.step(parent).coverage.missing
    )
    raise CoverageShortfall(
        planned.step_id,
        covered,
        len(plan.scope),
        "missing " + ", ".join(missing[:MISSING_SAMPLE]),
    )


def _covers_scope(plan: Plan, step_id: str, scope: frozenset[Entry]) -> bool:
    """Does this parent hold everything the graph is being run over?"""
    try:
        parent = plan.step(step_id)
    except KeyError:  # pragma: no cover - parents come from the same recipe
        return True
    return parent.status in COMPLETE_STATUSES or not (scope - parent.coverage.covered)


def _refuse_empty_upstream(plan: Plan, planned: PlannedStep) -> None:
    """Refuse a step above which something ran and wrote nothing.

    A step whose manifest resolves empty finishes cleanly with zero outputs, and
    a consumer would happily chain onto the empty directory. Never-ran and
    ran-and-produced-nothing both read as ``absent``, so the finish is what tells
    them apart, and it is the one wrongness a count *can* see.
    """
    for parent_id in planned.parents:
        try:
            parent = plan.step(parent_id)
        except KeyError:  # pragma: no cover - parents come from the same recipe
            continue
        if not parent.finished or parent.coverage.covers_all:
            continue
        if parent.coverage.target and not parent.coverage.covered:
            raise StepRefused(
                "upstream_empty",
                planned.step_id,
                f"step {parent_id!r} finished having written no entries, so "
                f"step {planned.step_id!r} would read an empty artifact. Check "
                f"what {parent_id!r} was asked to compute before running this.",
                {"parent": parent_id, "target": len(parent.coverage.target)},
            )


def _refuse_variant_disagreement(ds: Dataset, plan: Plan, planned: PlannedStep) -> None:
    """Refuse when an op parent produced a tracks variant other than the pinned one.

    The variant an op step will write is *minted* from the recipe's settings at
    planning time, because a params-only payload is knowable before the op runs.
    That makes it a prediction, and this is where the prediction meets the index
    the op actually wrote. Reading it back is a verification, never the mechanism
    by which the edge resolves -- an index with no rows in it yet would otherwise
    hash an empty term where execution hashes a real one.
    """
    if planned.spec.tracks_run_id is None:
        return
    from ..tracks_index import read_tracks_index, variant_for_producer_run

    for parent_id in planned.parents:
        try:
            parent = plan.step(parent_id)
        except KeyError:  # pragma: no cover - parents come from the same recipe
            continue
        if parent.kind != "op" or not parent.tracks_variant or not parent.run_id:
            continue
        produced = variant_for_producer_run(read_tracks_index(ds), parent.run_id)
        if produced is None or produced == parent.tracks_variant:
            # ``None`` is not a disagreement: the op has written no rows this
            # step can check against, which the coverage check already reports.
            continue
        raise StepRefused(
            "variant_mismatch",
            planned.step_id,
            f"step {planned.step_id!r} is pinned to tracks variant "
            f"{parent.tracks_variant!r}, but run {parent.run_id!r} of step "
            f"{parent_id!r} produced {produced!r}. Re-plan rather than reading "
            f"tables under a name that does not describe them.",
            {
                "parent": parent_id,
                "pinned": parent.tracks_variant,
                "produced": produced,
                "producer_run_id": parent.run_id,
            },
        )
