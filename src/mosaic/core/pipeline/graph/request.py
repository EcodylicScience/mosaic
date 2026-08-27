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
from typing import TYPE_CHECKING

from mosaic.runlog import new_execution_id, now_iso

from .digest import recipe_digest
from .model import BoundRef, FeatureStepSpec, Recipe, Request
from .plan import Plan, plan_pipeline
from .preflight import StepRefused, refuse_mixed_schemas
from .resolve import declared_version
from .store import load_recipe, recipe_path, save_recipe, save_request
from .topo import topological_order

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from mosaic.core.dataset import Dataset
    from mosaic.core.entry import Entry

    from ..inventory.cache import InventoryCache
    from .compatibility import DeclarationCatalog

__all__ = [
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
    """

    request: Request
    plan: Plan
    recipe_path: Path
    request_path: Path


def submit_request(
    ds: Dataset,
    recipe: Recipe,
    *,
    entries: Iterable[Entry] | None = None,
    bind: Mapping[str, BoundRef] | None = None,
    allow_partial: bool = False,
    max_concurrent_steps: int | None = None,
    owner: str = "",
    request_id: str = "",
    inventory: InventoryCache | None = None,
    catalog: DeclarationCatalog | None = None,
) -> SubmittedRequest:
    """Record one submission of *recipe* against *ds*, and resolve its plan.

    Writes the recipe into the dataset and a request document beside it, assigns
    one execution id per step, and pins the version every step's producer
    declares. Nothing is executed and nothing is enqueued.

    Args:
        ds: The dataset this submission is against.
        recipe: The graph. Refused before the dataset is touched if malformed.
        entries: Narrow the graph to these entries, or ``None`` for the dataset's
            own scope.
        bind: Out-of-graph artifacts this submission pins, by step id.
        allow_partial: Whether a coverage shortfall may proceed.
        max_concurrent_steps: How many of this request's steps may run at once.
            Recorded for whatever schedules the work; nothing here enforces it.
        owner: Free-form attribution recorded on the request.
        request_id: Reuse an externally minted id. Empty mints one.
        inventory: A held inventory to plan against.
        catalog: The declarations to resolve against.

    Returns:
        The request, the plan, and where both documents landed.

    Raises:
        RecipeInvalid: The recipe is malformed.
        StepRefused: The dataset's tracks tables are of incompatible schemas, so
            no feature identity resolves against it at all.
    """
    # Before planning, and again at each step's own start. Resolving any feature
    # identity against a dataset of mixed schemas already fails; checking first
    # is what makes that a refusal naming the schemas rather than an exception
    # out of the middle of a hash.
    refuse_mixed_schemas(ds, "")
    narrowing = None if entries is None else [(str(g), str(s)) for g, s in entries]
    plan = plan_pipeline(
        ds,
        recipe,
        intended_entries=narrowing,
        inventory=inventory,
        catalog=catalog,
    )
    ordered = topological_order(recipe)
    request = Request(
        request_id=request_id or new_execution_id(),
        recipe_digest=recipe_digest(recipe),
        owner=owner,
        created_at=now_iso(),
        entries=narrowing,
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
    return SubmittedRequest(
        request=request,
        plan=plan,
        recipe_path=save_recipe(ds.base_dir, recipe),
        request_path=save_request(ds.base_dir, request),
    )


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
