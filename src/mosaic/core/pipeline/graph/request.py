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

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar

from mosaic.runlog import new_execution_id, now_iso
from mosaic.core.scope import Scope

from ..error_kind import ClassifiedError, ErrorKind
from .digest import document_digest, recipe_digest
from .cleanup import plan_cleanup
from .model import (
    BoundRef,
    CleanupPolicy,
    CleanupSpec,
    CleanupTarget,
    FeatureStepSpec,
    Recipe,
    Request,
)
from .plan import Plan, plan_pipeline
from .serialize import plan_fingerprint
from .step_preflight import StepRefused, refuse_mixed_schemas
from .resolve import declared_version
from .retirement import validate_retirement
from .store import (
    open_requests_matching,
    read_recipe_document,
    recipe_path,
    save_recipe,
    save_request,
    submission_lock,
)
from .topo import topological_order

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from mosaic.core.dataset import Dataset
    from mosaic.core.json_value import JsonValue

    from ..inventory.cache import InventoryCache
    from .compatibility import DeclarationCatalog

__all__ = [
    "CleanupNeedsPlan",
    "CleanupOnManagedDataset",
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
            f"runs at once. Run the open request with 'mosaic pipeline run "
            f"--request <id>', or delete its file under .mosaic/pipelines/requests/ "
            f"if it will not run, then submit again."
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


class CleanupNeedsPlan(ClassifiedError):
    """Refuse a cleanup submitted without the fingerprint of a reviewed plan.

    A cleanup removes the runs that the reviewed plan found removable, so a
    submission asking for one names that plan.
    """

    error_kind: ClassVar[ErrorKind] = "invalid"

    def __init__(self) -> None:
        super().__init__(
            "a cleanup removes the superseded runs of a reviewed plan, so it "
            "needs that plan's fingerprint: plan the recipe with the cleanup, "
            "review it, and submit with --expect-plan <fingerprint>"
        )

    def error_reason(self) -> str:
        """Return ``cleanup_needs_plan``."""
        return "cleanup_needs_plan"


class CleanupOnManagedDataset(ClassifiedError):
    """Refuse a cleanup on a dataset that a service manages.

    The service removes what it manages, and a request file carrying a cleanup
    is one that an older mosaic, such as a service's worker, cannot read.

    Attributes:
        managed_by: The service the manifest names.
    """

    error_kind: ClassVar[ErrorKind] = "unavailable"

    def __init__(self, managed_by: JsonValue) -> None:
        super().__init__(
            f"the dataset is managed by {managed_by!r}, which removes its own "
            f"runs; submit without a cleanup"
        )
        self.managed_by: JsonValue = managed_by

    def error_reason(self) -> str:
        """Return ``cleanup_on_managed_dataset``."""
        return "cleanup_on_managed_dataset"

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the managing service."""
        return {"managed_by": self.managed_by}


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
    step_scopes: Mapping[str, Scope] | None = None,
    cleanup: CleanupPolicy | None = None,
    retire: Iterable[str] = (),
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
        step_scopes: Narrow single steps, keyed by step id, each intersected
            with *scope*. Recorded on the request, which every step re-plans
            from.
        cleanup: Remove the plan's superseded runs after every step has
            finished, judged under this policy, or ``None`` for no cleanup.
            The plan is resolved with them, as :func:`~.cleanup.plan_with_cleanup`
            resolves it, so *expect_plan* must be the fingerprint of a plan reviewed with
            the same policy. The request records the runs that plan found
            removable, less any run *bind* pins, and removes no other.
        retire: The digests of saved recipes to retire, which needs *cleanup*:
            the runs their requests wrote are judged wherever they sit
            (:func:`~.retirement.validate_retirement`). *expect_plan* must be the
            fingerprint of a plan reviewed with the same recipes. The request
            records them in ``cleanup.retired``, for this submission alone.

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
        CleanupNeedsPlan: *cleanup* is given without *expect_plan*.
        CleanupOnManagedDataset: *cleanup* is given and a service manages the
            dataset.
        RetireNeedsCleanup: *retire* is given without *cleanup*.
        RetireUnknownRecipe: A digest is not that of a saved recipe that reads.
        RetireOwnRecipe: A digest is *recipe*'s own.
        RetireOpenRecipe: A retired recipe has a request that has not ended.
    """
    # Before planning, and again at each step's own start. Resolving any feature
    # identity against a dataset of mixed schemas already fails; checking first
    # is what makes that a refusal naming the schemas rather than an exception
    # out of the middle of a hash.
    if cleanup is not None:
        if expect_plan is None:
            raise CleanupNeedsPlan()
        if ds.manifest.is_managed:
            raise CleanupOnManagedDataset(ds.manifest.managed_by)
    refuse_mixed_schemas(ds, "")
    narrowing = scope if scope is not None else Scope()
    digest = recipe_digest(recipe)
    retired = validate_retirement(
        ds.base_dir, digest, retire, cleanup=cleanup is not None
    )
    if refuse_duplicates:
        # Refuses before planning, which can take seconds.
        _ = _open_duplicates(ds.base_dir, digest, narrowing, refuse=True)
    narrowed_steps = dict(step_scopes or {})
    if cleanup is not None:
        from ..inventory.cache import InventoryCache
        from .resolve import declaration_catalog

        inventory = InventoryCache(ds) if inventory is None else inventory
        catalog = declaration_catalog() if catalog is None else catalog
    plan = plan_pipeline(
        ds,
        recipe,
        scope=narrowing,
        inventory=inventory,
        catalog=catalog,
        step_scopes=narrowed_steps,
    )
    if cleanup is not None and inventory is not None and catalog is not None:
        # What plan_with_cleanup does, with the plan_pipeline of this module.
        found_runs = plan_cleanup(
            ds,
            plan,
            policy=cleanup,
            inventory=inventory,
            catalog=catalog,
            recipe_name=recipe.name,
            retire=retired,
        )
        plan = replace(plan, cleanup=found_runs)
    found = plan_fingerprint(plan)
    if expect_plan is not None and found != expect_plan:
        raise PlanChanged(expected=expect_plan, found=found)
    pinned = {(bound.feature, bound.run_id) for bound in (bind or {}).values()}
    removal = (
        None
        if cleanup is None or plan.cleanup is None
        else CleanupSpec(
            policy=cleanup,
            execution_id=new_execution_id(),
            approved=[
                CleanupTarget(storage=run.storage, run_id=run.run_id)
                for run in plan.cleanup.removable
                if (run.storage, run.run_id) not in pinned
            ],
            fingerprint=found,
            retired=list(retired),
        )
    )
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
        step_scopes=narrowed_steps,
        step_executions={step.id: new_execution_id() for step in ordered},
        step_versions={
            step.id: declared_version(
                "feature" if isinstance(step, FeatureStepSpec) else "op",
                step.feature if isinstance(step, FeatureStepSpec) else step.kind,
            )
            for step in ordered
        },
        cleanup=removal,
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

    The file matches when the recipe it holds digests to the name, or when the
    document as written does. The second reading covers a file that an earlier
    mosaic saved before a change to the canonical form, such as a new defaulted
    field or the normalization of numbers in ``params``: its bytes are still the
    bytes its name digests. An edit changes both digests and still refuses.

    Raises:
        StepRefused: ``recipe_missing`` or ``digest_mismatch``.
    """
    path = recipe_path(base_dir, request.recipe_digest)
    try:
        document = read_recipe_document(path)
        recipe = Recipe.model_validate(document)
    except (FileNotFoundError, ValueError) as exc:
        raise StepRefused(
            "recipe_missing",
            "",
            f"request {request.request_id!r} names recipe "
            f"{request.recipe_digest!r}, which cannot be read at {path}: {exc}",
            {"recipe_digest": request.recipe_digest, "path": str(path)},
        ) from exc
    found = recipe_digest(recipe)
    saved = document_digest(document)
    if request.recipe_digest not in {found, saved}:
        raise StepRefused(
            "digest_mismatch",
            "",
            f"the recipe at {path} digests to {found!r}, but request "
            f"{request.request_id!r} names {request.recipe_digest!r}. The file "
            f"was changed underneath an open request; submit a new one rather "
            f"than running half a graph against each version.",
            {
                "expected": request.recipe_digest,
                "found": found,
                "saved": saved,
                "path": str(path),
            },
        )
    return recipe


def step_argv(
    manifest_path: Path | str,
    request: Request,
    step_id: str,
    *,
    overwrite: bool = False,
    program: str | Sequence[str] = "mosaic",
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
        program: The command that runs mosaic: argv[0], or the words that the
            argv starts with, as ``[sys.executable, "-m", "mosaic"]``. A driver
            that knows where mosaic lives substitutes it.

    Returns:
        The argv, ready to run.
    """
    argv = [
        *([program] if isinstance(program, str) else program),
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
