"""Turning a recipe into a submission: two files, and one attempt id per step.

A recipe says what the pipeline is; a request says what one person asked for, of
one dataset, on one occasion. Submitting writes both into the dataset -- the
recipe addressed by its digest, the request by its own id -- and assigns every
step its execution id **before anything runs**, which is what makes the request
document complete at submit rather than filled in as work lands.

**The execution-id map is the load-bearing part.** A step that re-resolved its
input by *feature name* at its own start would fall through to the latest-run
rule, which is wall clock: two requests on one dataset running one feature with
different params would cross-bind, the second step of one picking up the other's
output because its index row landed a second later. Naming the parent's attempt
removes the ambiguity, because a step then reads *its own parent's* run-log for
the identity to pin.

**Nothing here submits anything.** It resolves, writes two files and returns the
plan. What consumes the plan differs -- a shell loop, a job array, a queue, or
the sequential runner in this process -- and that difference is a calling policy
rather than an architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from mosaic.runlog import new_execution_id, now_iso
from mosaic.core.scope import Scope

from ..error_kind import ClassifiedError, ErrorKind
from .digest import recipe_digest
from .model import BoundRef, FeatureStepSpec, Recipe, Request
from .plan import Plan, plan_pipeline
from .serialize import plan_fingerprint
from .step_preflight import StepRefused, refuse_mixed_schemas
from .resolve import declared_version
from .store import (
    load_recipe,
    open_requests_matching,
    recipe_path,
    save_recipe,
    save_request,
    submission_lock,
)
from .topo import topological_order

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from mosaic.core.dataset import Dataset
    from mosaic.core.json_value import JsonValue

    from ..inventory.cache import InventoryCache
    from .compatibility import DeclarationCatalog

__all__ = [
    "DuplicateRequest",
    "PlanChanged",
    "SubmittedRequest",
    "load_recipe_for_request",
    "step_argv",
    "submit_request",
]


@dataclass(frozen=True, slots=True)
class SubmittedRequest:
    """One submission: the request document, where it landed, and its plan.

    Attributes:
        request: What was asked for, with every step's attempt id assigned.
        plan: What the recipe resolves to against this dataset, right now. A
            preview: every step resolves its own identity again at its own start,
            so nothing downstream may treat these identifiers as authoritative.
        recipe_path: Where the recipe was copied to inside the dataset.
        request_path: Where the request document was written.
        duplicates: The ids of the requests of the same recipe over an
            overlapping scope that were open at submit
            (:func:`~.store.open_requests_matching`), in descending id order.
    """

    request: Request
    plan: Plan
    recipe_path: Path
    request_path: Path
    duplicates: tuple[str, ...] = ()


class PlanChanged(ClassifiedError):
    """Refuse a submission whose plan has changed since its caller reviewed it.

    Attributes:
        expected: The fingerprint that the submission was given.
        found: The fingerprint of the plan resolved at submit.
    """

    error_kind: ClassVar[ErrorKind] = "conflict"

    def __init__(self, *, expected: str, found: str) -> None:
        message = (
            f"the plan of this recipe has fingerprint {found!r}, and the "
            f"submission expected {expected!r}. The dataset changed after it was "
            f"planned. Plan the recipe again, review it, and submit with the new "
            f"fingerprint."
        )
        super().__init__(message)
        self.expected: str = expected
        self.found: str = found

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the expected and the found fingerprints."""
        return {"expected": self.expected, "found": self.found}


class DuplicateRequest(ClassifiedError):
    """Refuse a submission while a request of its recipe over its entries is open.

    Attributes:
        recipe_digest: The recipe of the submission.
        request_ids: The open requests (:func:`~.store.open_requests_matching`).
    """

    error_kind: ClassVar[ErrorKind] = "conflict"

    def __init__(self, *, recipe_digest: str, request_ids: tuple[str, ...]) -> None:
        listed = ", ".join(request_ids)
        message = (
            f"request(s) {listed} of recipe {recipe_digest!r} cover entries of "
            f"this scope and have not ended. Both requests would compute the same "
            f"runs at once. Run the steps of the open request, or delete its file "
            f"under .mosaic/pipelines/requests/ if it will not run, then submit "
            f"again."
        )
        super().__init__(message)
        self.recipe_digest: str = recipe_digest
        self.request_ids: tuple[str, ...] = request_ids

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the recipe digest and the ids of the open requests."""
        return {
            "recipe_digest": self.recipe_digest,
            "request_ids": list(self.request_ids),
        }


def submit_request(
    ds: Dataset,
    recipe: Recipe,
    *,
    scope: Scope | None = None,
    bind: Mapping[str, BoundRef] | None = None,
    allow_partial: bool = False,
    max_concurrent_steps: int | None = None,
    owner: str = "",
    request_id: str = "",
    inventory: InventoryCache | None = None,
    catalog: DeclarationCatalog | None = None,
    expect_plan: str | None = None,
    refuse_duplicates: bool = False,
) -> SubmittedRequest:
    """Record one submission of *recipe* against *ds*, and resolve its plan.

    Writes the recipe into the dataset and a request document beside it, assigns
    one execution id per step, and pins the version every step's producer
    declares. Nothing is executed and nothing is enqueued.

    Args:
        ds: The dataset this submission is against.
        recipe: The graph. Refused before the dataset is touched if malformed.
        scope: Narrow the graph to what this selector names, or ``None`` for
            the dataset's own scope.
        bind: Out-of-graph artifacts this submission pins, by step id.
        allow_partial: Whether a coverage shortfall may proceed.
        max_concurrent_steps: How many of this request's steps may run at once.
            Recorded for whatever schedules the work; nothing here enforces it.
        owner: Free-form attribution recorded on the request.
        request_id: Reuse an externally minted id. Empty mints one.
        inventory: A held inventory to plan against.
        catalog: The declarations to resolve against.
        expect_plan: The :func:`~.serialize.plan_fingerprint` of the plan that
            the caller reviewed, or ``None`` to submit whatever the plan is now.
        refuse_duplicates: Whether an open request of *recipe* over an
            overlapping scope refuses the submission. Otherwise the submission
            proceeds and reports the open requests in ``duplicates``. The check
            runs before planning, and again under
            :func:`~.store.submission_lock` with the write of the request, so two
            submissions on one mount cannot both pass it. A submitted request
            whose steps never run stays open.

    Returns:
        The request, the plan, the paths of both documents, and the open
        requests of the same recipe over an overlapping scope.

    Raises:
        RecipeInvalid: The recipe is malformed.
        PlanChanged: *expect_plan* differs from the fingerprint of the plan
            resolved here. The dataset is left unwritten.
        DuplicateRequest: *refuse_duplicates* is set and a request of *recipe*
            over an overlapping scope is open. Neither the recipe nor the
            request is written.
        StepRefused: The dataset's tracks tables are of incompatible schemas, so
            no feature identity resolves against it at all.
    """
    # Before planning, and again at each step's own start. Resolving any feature
    # identity against a dataset of mixed schemas already fails; checking first
    # is what makes that a refusal naming the schemas rather than an exception
    # out of the middle of a hash.
    refuse_mixed_schemas(ds, "")
    narrowing = scope if scope is not None else Scope()
    digest = recipe_digest(recipe)
    if refuse_duplicates:
        # Refuses before planning, which can take seconds.
        _ = _open_duplicates(ds.base_dir, digest, narrowing, refuse=True)
    plan = plan_pipeline(
        ds,
        recipe,
        scope=narrowing,
        inventory=inventory,
        catalog=catalog,
    )
    if expect_plan is not None:
        found = plan_fingerprint(plan)
        if found != expect_plan:
            raise PlanChanged(expected=expect_plan, found=found)
    ordered = topological_order(recipe)
    request = Request(
        request_id=request_id or new_execution_id(),
        recipe_digest=digest,
        owner=owner,
        created_at=now_iso(),
        scope=narrowing,
        bind=dict(bind or {}),
        allow_partial=allow_partial,
        max_concurrent_steps=max_concurrent_steps,
        step_executions={step.id: new_execution_id() for step in ordered},
        step_versions={
            step.id: declared_version(
                "feature" if isinstance(step, FeatureStepSpec) else "op",
                step.feature if isinstance(step, FeatureStepSpec) else step.kind,
            )
            for step in ordered
        },
    )
    with submission_lock(ds.base_dir):
        duplicates = _open_duplicates(
            ds.base_dir, digest, narrowing, refuse=refuse_duplicates
        )
        return SubmittedRequest(
            request=request,
            plan=plan,
            recipe_path=save_recipe(ds.base_dir, recipe),
            request_path=save_request(ds.base_dir, request),
            duplicates=duplicates,
        )


def _open_duplicates(
    base_dir: Path | str, digest: str, scope: Scope, *, refuse: bool
) -> tuple[str, ...]:
    """Return the open requests of *digest* over *scope*, or refuse on any.

    Raises:
        DuplicateRequest: *refuse* is set and a request is open.
    """
    duplicates = open_requests_matching(base_dir, digest, scope)
    if duplicates and refuse:
        raise DuplicateRequest(recipe_digest=digest, request_ids=duplicates)
    return duplicates


def load_recipe_for_request(base_dir: Path | str, request: Request) -> Recipe:
    """The recipe this request names, checked against the digest that names it.

    Both faults refuse by name rather than proceeding. A missing file means the
    dataset no longer holds the pipeline it recorded, and a digest that does not
    match its contents means the file was edited underneath an open request --
    which would make every step below the edit resolve under a graph its
    predecessors never ran.

    Raises:
        StepRefused: ``recipe_missing`` or ``digest_mismatch``.
    """
    path = recipe_path(base_dir, request.recipe_digest)
    try:
        recipe = load_recipe(path)
    except (FileNotFoundError, ValueError) as exc:
        raise StepRefused(
            "recipe_missing",
            "",
            f"request {request.request_id!r} names recipe "
            f"{request.recipe_digest!r}, which cannot be read at {path}: {exc}",
            {"recipe_digest": request.recipe_digest, "path": str(path)},
        ) from exc
    found = recipe_digest(recipe)
    if found != request.recipe_digest:
        raise StepRefused(
            "digest_mismatch",
            "",
            f"the recipe at {path} digests to {found!r}, but request "
            f"{request.request_id!r} names {request.recipe_digest!r}. The file "
            f"was changed underneath an open request; submit a new one rather "
            f"than running half a graph against each version.",
            {"expected": request.recipe_digest, "found": found, "path": str(path)},
        )
    return recipe


def step_argv(
    manifest_path: Path | str,
    request: Request,
    step_id: str,
    *,
    overwrite: bool = False,
    program: str = "mosaic",
) -> list[str]:
    """The command that runs one step of one request.

    Step-addressed rather than fully specified, and strictly more expressive for
    it: a spelled-out invocation has no flag for several of the arguments that
    reach a feature's identity, while a step that re-plans itself reads all of
    them out of the recipe.

    **The request is found from the manifest's parent, and there is deliberately
    no second path flag.** A path a queue does not know about is one it cannot
    translate for a substrate that mounts the dataset somewhere else, and that
    would break precisely where it is hardest to see.

    Args:
        manifest_path: The ``dataset.yaml`` the job will be given.
        request: The submission, for its id and this step's execution id.
        step_id: Which step to run.
        overwrite: Recompute even where a cached run exists. An argv flag rather
            than a recipe field, because overwriting is a property of an attempt.
        program: argv[0]. A driver that knows where mosaic lives substitutes it.

    Returns:
        The argv, ready to run.
    """
    argv = [
        program,
        "run",
        "--json",
        "--manifest",
        str(manifest_path),
        "--graph-request",
        request.request_id,
        "--step",
        step_id,
        "--execution-id",
        request.execution_of(step_id),
    ]
    if request.owner:
        argv += ["--owner", request.owner]
    if overwrite:
        argv += ["--overwrite"]
    return argv
