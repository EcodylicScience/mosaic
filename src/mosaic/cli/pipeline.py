"""``mosaic pipeline``: read a graph, say what it would do, and do it.

Six verbs over one recipe file, and the split between them is the split the
library makes. ``validate`` asks only about the document and opens no dataset.
``plan`` and ``show`` resolve it against a dataset and write nothing. ``run``
executes it here, in this process, one step after another -- which is what a
notebook or a bare compute node has, and is not a lesser path than a queue.

``submit`` and ``status`` are the other half of that last point: ``submit``
records the pipeline against the dataset, assigns every step an attempt id, and
prints the command each step is run by, so a graph can be driven by anything that
starts processes in order -- a shell loop, a job array, a scheduler. ``status``
then says how far that submission got, reading only the attempts' own logs.

``--recipe`` takes the same ``@file.json`` / inline-JSON argument every other
JSON option on this CLI takes, so a recipe can be piped in with ``@-``.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail, load_json_arg, log, stdout_to_stderr
from mosaic.cli._render import render_table
from mosaic.core.scope import Scope

if TYPE_CHECKING:
    from mosaic.core.pipeline.graph import Plan, PlannedStep, Recipe

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
EntriesOption = Annotated[
    list[str] | None,
    typer.Option(
        "--entry",
        help="Narrow to a group:sequence entry (repeatable). Default: the whole dataset.",
    ),
]
JsonOption = Annotated[bool, typer.Option("--json", help="Emit as a JSON object.")]


def _recipe(argument: str) -> "Recipe":
    """Parse the ``--recipe`` argument, failing cleanly on a bad document."""
    from pydantic import ValidationError

    from mosaic.core.pipeline.graph import Recipe

    document = load_json_arg(argument)
    try:
        return Recipe.model_validate(document)
    except ValidationError as exc:
        fail(f"Not a valid recipe: {exc}")


def _scope(entries: list[str] | None) -> Scope | None:
    """The narrowing as a selector, or ``None`` for the dataset's own scope."""
    from mosaic.cli._io import parse_entries

    pairs = parse_entries(entries)
    return Scope(entries=pairs) if pairs else None


def validate_command(recipe: RecipeOption, as_json: JsonOption = False) -> None:
    """Check a recipe against the registries. Opens no dataset."""
    from mosaic.core.pipeline.graph import check_recipe

    problems = check_recipe(_recipe(recipe))
    if as_json:
        emit_json(
            {
                "valid": not problems,
                "problems": [
                    {
                        "step": problem.step,
                        "where": problem.where,
                        "message": problem.message,
                    }
                    for problem in problems
                ],
            }
        )
        if problems:
            raise typer.Exit(code=1)
        return
    if not problems:
        typer.echo("This recipe is valid.")
        return
    for problem in problems:
        typer.echo(str(problem), err=True)
    raise typer.Exit(code=1)


def plan_command(
    recipe: RecipeOption,
    manifest: ManifestOption,
    entry: EntriesOption = None,
    as_json: JsonOption = False,
) -> None:
    """Resolve a recipe against a dataset: identities, coverage, what is left."""
    from mosaic.core.pipeline.graph import plan_pipeline

    ds = load_dataset(manifest)
    with stdout_to_stderr():
        plan = plan_pipeline(ds, _recipe(recipe), scope=_scope(entry))
    _emit_plan(plan, as_json=as_json)


def show_command(recipe: RecipeOption, as_json: JsonOption = False) -> None:
    """Print a recipe's steps and the references between them. No dataset."""
    from mosaic.core.pipeline.graph import edges, recipe_digest, topological_order

    parsed = _recipe(recipe)
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
    from mosaic.core.pipeline.graph import StepRefused, run_pipeline

    ds = load_dataset(manifest)
    parsed = _recipe(recipe)
    try:
        with stdout_to_stderr():
            done = run_pipeline(
                ds,
                parsed,
                scope=_scope(entry),
                allow_partial=allow_partial,
                owner=owner,
            )
    except StepRefused as refusal:
        fail(str(refusal))

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
    as_json: JsonOption = False,
) -> None:
    """Record a submission of a recipe, and print the command for every step.

    Writes the recipe and a request beside it into the dataset, assigns one
    attempt id per step, and pins the version every step's producer declares.
    Nothing is executed: the printed commands are, and any driver that can run
    them in dependency order will run the graph correctly.
    """
    from mosaic.core.pipeline.graph import step_argv, submit_request

    ds = load_dataset(manifest)
    parsed = _recipe(recipe)
    with stdout_to_stderr():
        submitted = submit_request(
            ds,
            parsed,
            scope=_scope(entry),
            allow_partial=allow_partial,
            max_concurrent_steps=max_concurrent_steps,
            owner=owner,
        )
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
    so it stays cheap enough to poll.
    """
    from mosaic.core.pipeline.graph import load_request, request_rollup

    ds = load_dataset(manifest)
    try:
        request = load_request(ds.base_dir, request_id)
    except FileNotFoundError as exc:
        fail(str(exc))
    rollup = request_rollup(ds.base_dir, request)
    rows = [
        {
            "step": attempt.step_id,
            "execution_id": attempt.execution_id,
            "status": attempt.status or "-",
            "run_id": attempt.run_id or "-",
            "entries_failed": attempt.entries_failed,
        }
        for attempt in rollup.steps
    ]
    if as_json:
        emit_json(
            {
                "request_id": rollup.request_id,
                "status": rollup.status,
                "terminal": rollup.is_terminal,
                "steps": [
                    {
                        **row,
                        "error_json": attempt.error_json,
                        "entries_written": attempt.entries_written,
                        "cache_hit": attempt.cache_hit,
                    }
                    for row, attempt in zip(rows, rollup.steps, strict=True)
                ],
            }
        )
        return
    typer.echo(f"request {rollup.request_id}: {rollup.status}")
    render_table(rows, ["step", "execution_id", "status", "run_id", "entries_failed"])


def _runs(step: object) -> str:
    """What a step runs: its feature slug or its op kind."""
    return str(getattr(step, "feature", "") or getattr(step, "kind", ""))


def _emit_plan(plan: "Plan", *, as_json: bool) -> None:
    """Render a plan as a table, or as the JSON an API would serve."""
    from mosaic.core.pipeline.graph import asked_of

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
    if as_json:
        emit_json(
            {
                "recipe_digest": plan.recipe_digest,
                "scope": sorted(list(pair) for pair in plan.scope),
                "complete": plan.is_complete,
                "steps": [
                    {
                        **row,
                        "storage_name": planned.storage_name,
                        "tracks_variant": planned.tracks_variant,
                        "parents": list(planned.parents),
                        "entries": [list(pair) for pair in asked_of(planned, plan)],
                        "drift": len(planned.drift),
                    }
                    for row, planned in zip(rows, plan.steps, strict=True)
                ],
            }
        )
        return
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


_ = pipeline_app.command(name="validate")(validate_command)
_ = pipeline_app.command(name="plan")(plan_command)
_ = pipeline_app.command(name="show")(show_command)
_ = pipeline_app.command(name="submit")(submit_command)
_ = pipeline_app.command(name="run")(run_command)
_ = pipeline_app.command(name="status")(status_command)
