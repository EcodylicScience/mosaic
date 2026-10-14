"""``mosaic pipeline``: read a graph, say what it would do, and do it.

Eight verbs, and the split between them is the split the library makes.
``catalog`` lists the step types a recipe may run. ``validate`` asks only about
the document and opens no dataset.
``plan`` and ``show`` resolve it against a dataset and write nothing. ``run``
executes it here, in this process, one step after another -- which is what a
notebook or a bare compute node has, and is not a lesser path than a queue.

``submit`` and ``status`` are the other half of that last point: ``submit``
records the pipeline against the dataset, assigns every step an attempt id, and
prints the command each step is run by, so a graph can be driven by anything that
starts processes in order -- a shell loop, a job array, a scheduler. ``status``
then says how far that submission got, reading only the attempts' own logs.
``list`` reports a page of the dataset's submissions the same way.

``--recipe`` takes the same ``@file.json`` / inline-JSON argument every other
JSON option on this CLI takes, so a recipe can be piped in with ``@-``.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import (
    EntriesOption,
    emit_json,
    entries_scope,
    exit_refused,
    fail,
    fail_classified,
    load_json_arg,
    log,
    stdout_to_stderr,
    with_command_line_scope,
)
from mosaic.core.pipeline.error_kind import ClassifiedError
from mosaic.core.pipeline.ops import ScopeRefused
from mosaic.core.pipeline.refusal import Refusal
from mosaic.cli._render import render_table

if TYPE_CHECKING:
    from mosaic.core.pipeline.graph import Plan, PlannedStep, Recipe, Request

pipeline_app = typer.Typer(
    name="pipeline",
    help="Validate, plan and run a pipeline recipe.",
    no_args_is_help=True,
    add_completion=False,
)

RecipeOption = Annotated[
    str,
    typer.Option(
        "--recipe",
        "-r",
        help="Recipe as @file.json, @- for stdin, or an inline JSON object.",
    ),
]
ManifestOption = Annotated[
    Path,
    typer.Option(
        "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
    ),
]
JsonOption = Annotated[bool, typer.Option("--json", help="Emit as a JSON object.")]

SCOPE_FLAGS_REMEDY = (
    "--entry group:sequence (repeatable). Omit it to cover the whole dataset."
)
"""How to name a scope on this command, appended to a refusal the library wrote.

``check_scope_takes`` answers in ``Scope(...)``, which is right for the library
and for mosaic-api. This command offers one flag and it is spelled ``--entry``.
"""


def _recipe(argument: str, *, as_json: bool) -> "Recipe":
    """Parse the ``--recipe`` argument, failing cleanly on a bad document."""
    from mosaic.core.pipeline.graph import RecipeInvalid, read_recipe

    try:
        return read_recipe(load_json_arg(argument))
    except RecipeInvalid as exc:
        fail_classified(exc, as_json=as_json)


def _fail_unrunnable(
    error: ClassifiedError, *, as_json: bool, step: str = ""
) -> NoReturn:
    """Report a classified error that stopped a recipe, and exit 1.

    A scope refusal is reported with the flag that answers it.

    Args:
        error: The error, raised while planning, submitting or running a step.
        as_json: Whether ``--json`` was given.
        step: The step that failed, or ``""`` before any step started.
    """
    message = (
        with_command_line_scope(str(error), SCOPE_FLAGS_REMEDY)
        if isinstance(error, ScopeRefused)
        else None
    )
    fail_classified(error, as_json=as_json, message=message, step=step)


def validate_command(recipe: RecipeOption, as_json: JsonOption = False) -> None:
    """Check a recipe against the registries. Opens no dataset."""
    from mosaic.core.pipeline.graph import validation_report

    report = validation_report(load_json_arg(recipe))
    if as_json:
        emit_json(report.model_dump(mode="json"))
        if not report.valid:
            raise typer.Exit(code=1)
        return
    for caution in report.cautions:
        typer.echo(f"caution: {caution}", err=True)
    if report.unchecked:
        steps = ", ".join(report.unchecked)
        typer.echo(f"Columns not checked for {steps}", err=True)
    if report.valid:
        typer.echo("This recipe is valid.")
        return
    for problem in report.problems:
        typer.echo(str(problem), err=True)
    raise typer.Exit(code=1)


def catalog_command(as_json: JsonOption = False) -> None:
    """List the step types a recipe may run, with their params schemas under --json."""
    from mosaic.core.pipeline.graph import editor_catalog

    catalog = editor_catalog()
    if as_json:
        emit_json(catalog.model_dump(mode="json"))
        return
    render_table(
        [
            {
                "name": entry.name,
                "type": entry.type,
                "category": entry.category,
                "resource_class": entry.declaration.resource_class,
                "available": "no" if entry.unavailable else "yes",
            }
            for entry in catalog.entries
        ],
        ["name", "type", "category", "resource_class", "available"],
    )


def plan_command(
    recipe: RecipeOption,
    manifest: ManifestOption,
    entry: EntriesOption = None,
    as_json: JsonOption = False,
) -> None:
    """Resolve a recipe against a dataset: identities, coverage, what is left.

    Exits 1 when the dataset refuses the recipe. Under ``--json`` the plan is
    printed with its problems.
    """
    from mosaic.core.pipeline.graph import RecipeRefused, plan_pipeline, plan_wire

    ds = load_dataset(manifest, read_only=True)
    parsed = _recipe(recipe, as_json=as_json)
    try:
        with stdout_to_stderr():
            plan = plan_pipeline(ds, parsed, scope=entries_scope(entry))
    except RecipeRefused as refusal:
        if as_json:
            emit_json(plan_wire(refusal.plan).model_dump(mode="json"))
            raise typer.Exit(code=1) from refusal
        fail(str(refusal))
    except ClassifiedError as error:
        _fail_unrunnable(error, as_json=as_json)
    _emit_plan(plan, as_json=as_json)


def show_command(recipe: RecipeOption, as_json: JsonOption = False) -> None:
    """Print a recipe's steps and the references between them. No dataset."""
    from mosaic.core.pipeline.graph import edges, recipe_digest, topological_order

    parsed = _recipe(recipe, as_json=as_json)
    ordered = topological_order(parsed)
    wires = edges(parsed)
    if as_json:
        emit_json(
            {
                "name": parsed.name,
                "digest": recipe_digest(parsed),
                "steps": [
                    {"id": step.id, "type": step.type, "runs": _runs(step)}
                    for step in ordered
                ],
                "edges": [
                    {
                        "producer": wire.producer,
                        "consumer": wire.consumer,
                        "site": wire.site,
                        "where": wire.where,
                    }
                    for wire in wires
                ],
            }
        )
        return
    typer.echo(f"{parsed.name or '(unnamed)'}  {recipe_digest(parsed)}")
    render_table(
        [
            {
                "step": step.id,
                "type": step.type,
                "runs": _runs(step),
                "reads": ", ".join(
                    wire.producer for wire in wires if wire.consumer == step.id
                ),
            }
            for step in ordered
        ],
        ["step", "type", "runs", "reads"],
    )


def _failed_attempt(
    base_dir: Path, request: Request | None, error: ClassifiedError
) -> tuple[str, str]:
    """Return the step of *request* that *error* ended the run at, and its attempt.

    An error that names its step names the attempt too, because the request
    assigned every step one. A refused step can be refused after its attempt
    finished, when the run finds coverage did not move. An error raised inside an
    attempt, such as a tracker's refusal of a codec or a missing optional
    dependency, does not name its step. A step that raises ends the run. Its
    attempt is therefore the one that the run-logs record as failed: every step
    before it finished, and none after it started. Both are empty when neither answers, as for an
    error raised before any step started.
    """
    from mosaic.core.pipeline.graph import request_rollup

    named = error.error_step() or ""
    if request is None:
        return named, ""
    if named:
        return named, request.step_executions.get(named, "")
    for attempt in request_rollup(base_dir, request).steps:
        if attempt.status == "failed":
            return attempt.step_id, attempt.execution_id
    return "", ""


def run_command(
    recipe: RecipeOption,
    manifest: ManifestOption,
    entry: EntriesOption = None,
    allow_partial: Annotated[
        bool,
        typer.Option(
            "--allow-partial",
            help=(
                "Proceed when a step would run over less than it was planned "
                "for, or when it produces nothing new."
            ),
        ),
    ] = False,
    owner: Annotated[
        str, typer.Option("--owner", help="Recorded on each attempt.")
    ] = "",
    as_json: JsonOption = False,
) -> None:
    """Run every step of a recipe here, in order, skipping what is already done."""
    from mosaic.core.pipeline.graph import run_pipeline, submit_request

    ds = load_dataset(manifest)
    parsed = _recipe(recipe, as_json=as_json)
    submitted: Request | None = None
    try:
        with stdout_to_stderr():
            # Submitted here rather than inside run_pipeline, so that a refusal
            # raised by one step can be traced to that step's attempt.
            submitted = submit_request(
                ds,
                parsed,
                scope=entries_scope(entry),
                allow_partial=allow_partial,
                owner=owner,
            ).request
            done = run_pipeline(ds, parsed, request=submitted, owner=owner)
    except Refusal as refusal:
        # A step's own refusal, or one raised inside a step's attempt, such as a
        # tracker's refusal of a codec. Either ends the run at that step.
        step_id, execution_id = _failed_attempt(ds.base_dir, submitted, refusal)
        exit_refused(execution_id, refusal, step_id=step_id, as_json=as_json)
    except ClassifiedError as error:
        step_id, _ = _failed_attempt(ds.base_dir, submitted, error)
        _fail_unrunnable(error, as_json=as_json, step=step_id)

    rows = [
        {
            "step": outcome.step_id,
            "type": outcome.kind,
            "run_id": outcome.run_id,
            "state": outcome.state,
        }
        for outcome in done.outcomes
    ]
    if as_json:
        emit_json(
            {
                "request_id": done.request_id,
                "recipe_digest": done.recipe_digest,
                "scope": sorted(list(pair) for pair in done.scope),
                "steps": [
                    {
                        **row,
                        "planned_run_id": outcome.planned_run_id,
                        "execution_id": outcome.execution_id,
                        "covered": outcome.covered,
                        "target": outcome.target,
                    }
                    for row, outcome in zip(rows, done.outcomes, strict=True)
                ],
            }
        )
        return
    render_table(rows, ["step", "type", "run_id", "state"])
    for outcome in done.outcomes:
        if outcome.diverged:
            # Worth saying out loud: the plan is a preview, and a step that
            # recorded a different identifier means the preview was wrong about
            # something -- most often a scope that changed underneath it.
            log(
                f"note: step {outcome.step_id!r} recorded {outcome.run_id}, "
                f"where the plan resolved {outcome.planned_run_id}"
            )


def submit_command(
    recipe: RecipeOption,
    manifest: ManifestOption,
    entry: EntriesOption = None,
    allow_partial: Annotated[
        bool,
        typer.Option(
            "--allow-partial",
            help="Let a step proceed when it would run over less than planned.",
        ),
    ] = False,
    max_concurrent_steps: Annotated[
        int | None,
        typer.Option(
            "--max-concurrent-steps",
            help="How many of this request's steps may run at once. Advisory here.",
        ),
    ] = None,
    owner: Annotated[
        str, typer.Option("--owner", help="Recorded on the request and its attempts.")
    ] = "",
    expect_plan: Annotated[
        str | None,
        typer.Option(
            "--expect-plan",
            help=(
                "The fingerprint that 'mosaic pipeline plan' printed. Refuses the "
                "submission, writing nothing, when the plan has changed since."
            ),
        ),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Record a submission of a recipe, and print the command for every step.

    Writes the recipe and a request beside it into the dataset, assigns one
    attempt id per step, and pins the version every step's producer declares.
    Nothing is executed: the printed commands are, and any driver that can run
    them in dependency order will run the graph correctly.
    """
    from mosaic.core.pipeline.graph import plan_fingerprint, step_argv, submit_request

    ds = load_dataset(manifest)
    parsed = _recipe(recipe, as_json=as_json)
    try:
        with stdout_to_stderr():
            submitted = submit_request(
                ds,
                parsed,
                scope=entries_scope(entry),
                allow_partial=allow_partial,
                max_concurrent_steps=max_concurrent_steps,
                owner=owner,
                expect_plan=expect_plan,
            )
    except ClassifiedError as error:
        _fail_unrunnable(error, as_json=as_json)
    request = submitted.request
    steps = [
        {
            "step": planned.step_id,
            "type": planned.kind,
            "runs": planned.runs,
            "execution_id": request.execution_of(planned.step_id),
            "lane": planned.lane,
            "run_id": planned.run_id or "-",
            "status": planned.status,
            "parents": list(planned.parents),
            "depends_on": [request.execution_of(parent) for parent in planned.parents],
            "argv": step_argv(ds.manifest_path, request, planned.step_id),
        }
        for planned in submitted.plan.steps
    ]
    if as_json:
        emit_json(
            {
                "request_id": request.request_id,
                "recipe_digest": request.recipe_digest,
                "plan_fingerprint": plan_fingerprint(submitted.plan),
                "recipe_path": str(submitted.recipe_path),
                "request_path": str(submitted.request_path),
                "allow_partial": request.allow_partial,
                "max_concurrent_steps": request.max_concurrent_steps,
                "scope": sorted(list(pair) for pair in submitted.plan.scope),
                "steps": steps,
            }
        )
        return
    typer.echo(f"request {request.request_id}  recipe {request.recipe_digest}")
    render_table(
        [
            {
                "step": row["step"],
                "runs": row["runs"],
                "execution_id": row["execution_id"],
                "lane": row["lane"],
                "status": row["status"],
            }
            for row in steps
        ],
        ["step", "runs", "execution_id", "lane", "status"],
    )
    for row in steps:
        argv = row["argv"]
        assert isinstance(argv, list)
        typer.echo(" ".join(shlex.quote(str(word)) for word in argv))


def status_command(
    manifest: ManifestOption,
    request_id: Annotated[
        str, typer.Option("--request", help="Which submission to report on.")
    ],
    as_json: JsonOption = False,
) -> None:
    """Say how far one submission got, from its steps' own attempt logs.

    Deliberately not a coverage report: whether every step's *work* is done is a
    question about artifacts, which ``plan`` answers. This one is about attempts,
    so it stays cheap enough to poll. Each step reports its outcome, progress,
    recorded error and first 20 failed entries, and the request a ``verdict``
    that names a finished request with lost entries ``partial``.
    """
    from mosaic.core.pipeline.graph import load_request, request_status_wire

    ds = load_dataset(manifest, read_only=True)
    try:
        request = load_request(ds.base_dir, request_id)
    except FileNotFoundError as exc:
        fail(str(exc))
    report = request_status_wire(ds.base_dir, request)
    if as_json:
        emit_json(report.model_dump(mode="json"))
        return
    typer.echo(f"request {report.request_id}: {report.verdict}")
    render_table(
        [
            {
                "step": step.step,
                "execution_id": step.execution_id,
                "status": step.outcome,
                "run_id": step.run_id or "-",
                "entries_failed": step.entries_failed,
            }
            for step in report.steps
        ],
        ["step", "execution_id", "status", "run_id", "entries_failed"],
    )


def list_command(
    manifest: ManifestOption,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="List at most this many requests.", min=1),
    ] = None,
    after: Annotated[
        str | None,
        typer.Option(
            "--after", help="Start after the page that printed this next_after."
        ),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """List the dataset's pipeline requests newest first, each with its verdict.

    Only the requests on the page are read, each from its steps' attempt logs.
    """
    from mosaic.core.pipeline.graph import list_requests

    ds = load_dataset(manifest, read_only=True)
    try:
        page = list_requests(ds.base_dir, limit=limit, after=after)
    except ValueError as exc:
        fail(str(exc))
    if as_json:
        emit_json(page.model_dump(mode="json"))
        return
    if not page.items:
        typer.echo("No pipeline requests recorded in this dataset.")
    else:
        render_table(
            [
                {
                    "request_id": item.request_id,
                    "created_at": item.created_at or "-",
                    "recipe": item.recipe_digest,
                    "steps": item.step_count,
                    "verdict": item.verdict,
                }
                for item in page.items
            ],
            ["request_id", "created_at", "recipe", "steps", "verdict"],
        )
    if page.next_after is not None:
        log(f"note: --after {page.next_after} lists the next page")


def _runs(step: object) -> str:
    """What a step runs: its feature slug or its op kind."""
    return str(getattr(step, "feature", "") or getattr(step, "kind", ""))


def _emit_plan(plan: "Plan", *, as_json: bool) -> None:
    """Render a plan as its fingerprint and a table, or as the JSON an API serves."""
    from mosaic.core.pipeline.graph import plan_fingerprint, plan_wire

    if as_json:
        emit_json(plan_wire(plan).model_dump(mode="json"))
        return
    typer.echo(f"plan {plan_fingerprint(plan)}")
    rows = [
        {
            "step": planned.step_id,
            "type": planned.kind,
            "runs": planned.runs,
            "run_id": planned.run_id or "-",
            "status": planned.status,
            "coverage": _coverage_cell(planned),
            "lane": planned.lane,
            "reason": _reason_cell(planned),
        }
        for planned in plan.steps
    ]
    render_table(
        rows, ["step", "type", "runs", "run_id", "status", "coverage", "lane", "reason"]
    )


def _coverage_cell(planned: "PlannedStep") -> str:
    """``covered/target``, or ``all`` for an artifact answering for every key."""
    if planned.coverage.covers_all:
        return "all"
    return f"{len(planned.coverage.covered)}/{len(planned.coverage.target)}"


def _reason_cell(planned: "PlannedStep") -> str:
    """Why this step is not simply running, in one short phrase."""
    from mosaic.core.pipeline.graph import (
        CoverageShort,
        DepsIncomplete,
        HeldOnParents,
        IdentityUnresolved,
        Stalled,
        WaitingOnResource,
    )

    reason = planned.reason
    match reason:
        case None:
            return ""
        case DepsIncomplete():
            return f"waiting on {', '.join(reason.blocking)}"
        case CoverageShort():
            return f"has {reason.covered} of {reason.target}"
        case HeldOnParents():
            return f"held on {', '.join(reason.parents)}"
        case WaitingOnResource():
            return f"{reason.resource_class} at {reason.in_use}/{reason.capacity}"
        case Stalled():
            return f"stalled at {reason.covered} of {reason.target}"
        case IdentityUnresolved():
            return f"unresolvable: {reason.because}"


_ = pipeline_app.command(name="catalog")(catalog_command)
_ = pipeline_app.command(name="validate")(validate_command)
_ = pipeline_app.command(name="plan")(plan_command)
_ = pipeline_app.command(name="show")(show_command)
_ = pipeline_app.command(name="submit")(submit_command)
_ = pipeline_app.command(name="run")(run_command)
_ = pipeline_app.command(name="status")(status_command)
_ = pipeline_app.command(name="list")(list_command)
