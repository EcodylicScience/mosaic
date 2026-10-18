"""``mosaic pipeline``: read a graph, say what it would do, and do it.

Eight verbs, and the split between them is the split the library makes.
``catalog`` lists the step types a recipe may run. ``validate`` asks only about
the document and opens no dataset.
``plan`` and ``show`` resolve it against a dataset and write nothing. ``run``
executes it here, in this process, one step after another -- which is what a
notebook or a bare compute node has, and is not a lesser path than a queue.
``run --request`` runs the unfinished steps of a submitted request, one process
per step, and ``--detach`` starts that driver in a new session and returns.

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

import json
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, NoReturn

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import (
    EntriesOption,
    GroupOption,
    MinFreeMemoryOption,
    ParallelModeOption,
    StepScopeOption,
    WorkersOption,
    parallel_settings,
    emit_json,
    entries_scope,
    exit_refused,
    fail,
    fail_classified,
    load_json_arg,
    log,
    parse_step_scopes,
    stdout_to_stderr,
    with_command_line_scope,
)
from mosaic.core.pipeline.error_kind import ClassifiedError
from mosaic.core.pipeline.graph.status import MAX_WAIT_SECONDS
from mosaic.core.pipeline.ops import ScopeRefused
from mosaic.core.pipeline.refusal import Refusal
from mosaic.cli._render import render_step_type_detail, render_table
from mosaic.cli.features.describe import DetailOption

if TYPE_CHECKING:
    from mosaic.core.pipeline.parallelism import ParallelSettings
    from mosaic.core.pipeline.graph import (
        CleanupPolicy,
        CleanupStatusWire,
        Plan,
        PlannedStep,
        Recipe,
        Request,
        RequestStatusWire,
        RequestSummaryWire,
        RetirableRecipeWire,
    )

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

CleanupChoice = Literal["none", "superseded"]

CleanupOption = Annotated[
    CleanupChoice,
    typer.Option(
        "--cleanup",
        help=(
            "'superseded' reports the other runs of the recipe's feature storages, "
            "each removable or kept with its reason; on submit and run they are "
            "removed after every step finished. Needs --expect-plan there."
        ),
    ),
]
KeepNewestOption = Annotated[
    int,
    typer.Option(
        "--keep-newest",
        min=0,
        help="With --cleanup, keep this many superseded runs per storage, newest first.",
    ),
]
MinAgeOption = Annotated[
    float,
    typer.Option(
        "--min-age-hours",
        min=0,
        help="With --cleanup, keep every run younger than this many hours.",
    ),
]

RetireOption = Annotated[
    list[str] | None,
    typer.Option(
        "--retire",
        help=(
            "With --cleanup superseded, also judge the runs that this saved "
            "recipe's requests wrote, in any storage (repeatable; a recipe "
            "digest, which 'pipeline plan --cleanup superseded' lists). It "
            "applies to this plan or submission only: give it to the plan and "
            "again to the submission."
        ),
    ),
]

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


def _cleanup_policy(
    cleanup: CleanupChoice, keep_newest: int, min_age_hours: float
) -> CleanupPolicy | None:
    """The policy that ``--cleanup`` and its flags name, ``None`` for no cleanup."""
    from mosaic.core.pipeline.graph import CleanupPolicy

    if cleanup == "none":
        return None
    return CleanupPolicy(keep_newest=keep_newest, min_age_hours=min_age_hours)


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


def catalog_command(
    name: Annotated[
        str | None,
        typer.Argument(
            metavar="NAME",
            help="A feature slug, an op kind or 'tracks', to describe that one.",
        ),
    ] = None,
    detail: DetailOption = "fields",
    as_json: JsonOption = False,
) -> None:
    """List the step types a recipe may run, or describe the one NAME names.

    Without NAME, ``--json`` prints the whole catalog with its params schemas.
    With NAME, the step type's facts and a row per params field are printed,
    and ``--detail schema`` adds its params JSON Schema.
    """
    from mosaic.core.pipeline.graph import (
        StepTypeNotFound,
        editor_catalog,
        step_type_detail,
    )

    if name is not None:
        try:
            described = step_type_detail(name, detail=detail)
        except StepTypeNotFound as error:
            fail_classified(error, as_json=as_json)
        if as_json:
            emit_json(described.model_dump(mode="json"))
        else:
            render_step_type_detail(described)
        return
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
    group: GroupOption = None,
    step_scope: StepScopeOption = None,
    cleanup: CleanupOption = "none",
    keep_newest: KeepNewestOption = 1,
    min_age_hours: MinAgeOption = 24.0,
    retire: RetireOption = None,
    as_json: JsonOption = False,
) -> None:
    """Resolve a recipe against a dataset: identities, coverage, what is left.

    Exits 1 when the dataset refuses the recipe. Under ``--json`` the plan is
    printed with its problems. The requests of the recipe over an overlapping
    scope that have not ended, running or orphaned, are listed with it.
    ``--step-scope`` narrows one step, as a train/test split narrows the fit
    to the training groups and the scoring to the test groups.
    ``--cleanup superseded`` also reports the superseded runs of the recipe's
    feature storages, and the fingerprint then covers the ones removable. It
    also lists the saved recipes whose requests wrote runs, and ``--retire
    <digest>`` judges the runs of one of them too, in any storage. It writes
    nothing.
    """
    from mosaic.core.pipeline.graph import (
        RecipeRefused,
        plan_wire,
        plan_with_cleanup,
        requests_matching,
        retirable_recipes,
    )
    from mosaic.core.pipeline.index import feature_run_root
    from mosaic.core.scope import Scope

    ds = load_dataset(manifest, read_only=True)
    parsed = _recipe(recipe, as_json=as_json)
    scope = entries_scope(entry, group)
    step_scopes = parse_step_scopes(step_scope)
    try:
        with stdout_to_stderr():
            plan = plan_with_cleanup(
                ds,
                parsed,
                cleanup=_cleanup_policy(cleanup, keep_newest, min_age_hours),
                scope=scope,
                step_scopes=step_scopes,
                retire=retire or (),
            )
    except RecipeRefused as refusal:
        if as_json:
            emit_json(plan_wire(refusal.plan).model_dump(mode="json"))
            raise typer.Exit(code=1) from refusal
        fail(str(refusal))
    except ClassifiedError as error:
        _fail_unrunnable(error, as_json=as_json)
    unended = requests_matching(
        ds.base_dir, plan.recipe_digest, Scope() if scope is None else scope
    )
    retirable, omitted = (
        retirable_recipes(
            ds.base_dir,
            exclude_digest=plan.recipe_digest,
            present=lambda storage, run_id: feature_run_root(
                ds, storage, run_id
            ).is_dir(),
        )
        if plan.cleanup is not None
        else ((), 0)
    )
    _emit_plan(
        plan,
        open_requests=unended,
        retirable=retirable,
        retirable_omitted=omitted,
        as_json=as_json,
    )


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
    """Return the step of *request* that raised *error*, and its attempt.

    An error that names its step names the attempt too, because the request
    assigned every step one. A refused step can be refused after its attempt
    finished, when the run finds coverage did not move. An error raised inside an
    attempt, such as a tracker's refusal of a codec or a missing optional
    dependency, does not name its step. The run raises the first error of its
    steps, and they run in the request's order. Its attempt is therefore the
    first failed one in the run-logs, leaving out the steps recorded failed for
    a failed parent (``PARENT_FAILED_TYPE``). Both are empty when neither
    answers, as for an error raised before any step started.
    """
    from mosaic.core.pipeline.graph import PARENT_FAILED_TYPE, request_rollup
    from mosaic.runlog import error_summary

    named = error.error_step() or ""
    if request is None:
        return named, ""
    if named:
        return named, request.step_executions.get(named, "")
    for attempt in request_rollup(base_dir, request).steps:
        if (
            attempt.status == "failed"
            and error_summary(attempt.error_json)["type"] != PARENT_FAILED_TYPE
        ):
            return attempt.step_id, attempt.execution_id
    return "", ""


def _warn_duplicates(duplicates: tuple[str, ...]) -> None:
    """Warn on stderr about the open requests of a submitted recipe."""
    if not duplicates:
        return
    listed = ", ".join(duplicates)
    warning = (
        f"warning: request(s) {listed} of this recipe over an overlapping scope "
        f"have not ended. Running both computes the same runs twice at once."
    )
    log(warning)


def run_command(
    manifest: ManifestOption,
    recipe: Annotated[
        str | None,
        typer.Option(
            "--recipe",
            "-r",
            help=(
                "Recipe to submit and run, as @file.json, @- for stdin, or an "
                "inline JSON object."
            ),
        ),
    ] = None,
    request_id: Annotated[
        str | None,
        typer.Option(
            "--request",
            help=(
                "Run the unfinished steps of this submitted request, each as its "
                "own process, instead of submitting --recipe. Running it again "
                "resumes it."
            ),
        ),
    ] = None,
    detach: Annotated[
        bool,
        typer.Option(
            "--detach",
            help=(
                "With --request, start the driver in a new session, print its pid "
                "and return. The driver and its steps append their output to the "
                "request's log under .mosaic/drivers/."
            ),
        ),
    ] = False,
    entry: EntriesOption = None,
    group: GroupOption = None,
    step_scope: StepScopeOption = None,
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
    cleanup: CleanupOption = "none",
    keep_newest: KeepNewestOption = 1,
    min_age_hours: MinAgeOption = 24.0,
    retire: RetireOption = None,
    expect_plan: Annotated[
        str | None,
        typer.Option(
            "--expect-plan",
            help=(
                "With --recipe, the fingerprint that 'mosaic pipeline plan' "
                "printed. Refuses the run, writing nothing, when the plan has "
                "changed since. Required with --cleanup."
            ),
        ),
    ] = None,
    workers: WorkersOption = None,
    parallel_mode: ParallelModeOption = None,
    min_free_memory_fraction: MinFreeMemoryOption = 0.0,
    as_json: JsonOption = False,
) -> None:
    """Run every step of a recipe here, in order, skipping what is already done.

    An open request of the same recipe over an overlapping scope is reported on
    stderr. With ``--request`` the steps of a submitted request run instead, and
    the command exits 1 when the request failed. Either way it exits 130 when the
    run was cancelled, by SIGTERM, SIGINT or ``mosaic cancel``.

    With ``--cleanup superseded --expect-plan`` the runs that plan found
    removable are removed once every step has finished, and the command exits 1
    when that cleanup failed. ``--retire <digest>`` adds the runs that a saved
    recipe's requests wrote, in any storage, to what that plan reviewed.
    ``--request`` runs the cleanup its request recorded.

    ``--workers`` computes several entries of each parallelizable feature at once.
    It belongs to the machine and changes neither a run's identity nor its
    outputs, so ``--request`` accepts it where it rejects the options above.
    Memory grows with the worker count, most steeply with ``--parallel-mode
    process``.
    """
    from mosaic.core.pipeline.graph import (
        request_status_wire,
        run_pipeline,
        submit_request,
    )
    from mosaic.core.pipeline.job import Cancelled, CancelToken, install_signal_handler

    parallel = parallel_settings(workers, parallel_mode, min_free_memory_fraction)
    if request_id is not None:
        if recipe is not None:
            fail("Provide exactly one of --recipe or --request.")
        if (
            entry
            or group
            or step_scope
            or allow_partial
            or owner
            or cleanup != "none"
            or retire
            or expect_plan is not None
        ):
            fail(
                "--entry, --group, --step-scope, --allow-partial, --owner, "
                "--cleanup, --retire and --expect-plan are recorded on a request "
                "at submit, and are not used with --request."
            )
        _run_request(
            manifest, request_id, detach=detach, as_json=as_json, parallel=parallel
        )
        return
    if recipe is None:
        fail("Provide exactly one of --recipe or --request.")
    if detach:
        fail("--detach is used with --request. Submit the recipe first.")
    ds = load_dataset(manifest)
    parsed = _recipe(recipe, as_json=as_json)
    scope = entries_scope(entry, group)
    step_scopes = parse_step_scopes(step_scope)
    submitted: Request | None = None
    token = CancelToken()
    install_signal_handler(token)
    try:
        with stdout_to_stderr():
            # Submitted here rather than inside run_pipeline, so that a refusal
            # raised by one step can be traced to that step's attempt.
            recorded = submit_request(
                ds,
                parsed,
                scope=scope,
                allow_partial=allow_partial,
                owner=owner,
                step_scopes=step_scopes,
                expect_plan=expect_plan,
                cleanup=_cleanup_policy(cleanup, keep_newest, min_age_hours),
                retire=retire or (),
            )
            submitted = recorded.request
            _warn_duplicates(recorded.duplicates)
            done = run_pipeline(
                ds,
                parsed,
                request=submitted,
                owner=owner,
                cancel_token=token,
                parallel=parallel,
            )
    except Cancelled:
        if submitted is not None:
            report = request_status_wire(ds.base_dir, submitted)
            if as_json:
                emit_json(report.model_dump(mode="json"))
            else:
                _render_request_status(report)
        raise typer.Exit(code=130) from None
    except Refusal as refusal:
        # A step's own refusal, or one raised inside a step's attempt, such as a
        # tracker's refusal of a codec. The run raises the first one after the
        # steps on other branches have run.
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
    removal = (
        request_status_wire(ds.base_dir, submitted).cleanup
        if submitted.cleanup is not None
        else None
    )
    if as_json:
        emit_json(
            {
                "request_id": done.request_id,
                "cleanup": None if removal is None else removal.model_dump(mode="json"),
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
        _exit_on_failed_cleanup(done.cleanup_status)
        return
    render_table(rows, ["step", "type", "run_id", "state"])
    if removal is not None:
        _render_cleanup_status(removal)
    for outcome in done.outcomes:
        if outcome.diverged:
            # Worth saying out loud: the plan is a preview, and a step that
            # recorded a different identifier means the preview was wrong about
            # something -- most often a scope that changed underneath it.
            log(
                f"note: step {outcome.step_id!r} recorded {outcome.run_id}, "
                f"where the plan resolved {outcome.planned_run_id}"
            )
    _exit_on_failed_cleanup(done.cleanup_status)


def _exit_on_failed_cleanup(status: str) -> None:
    """Exit 1 when the request's cleanup failed, after its report was printed."""
    if status == "failed":
        raise typer.Exit(code=1)


def _run_request(
    manifest: Path,
    request_id: str,
    *,
    detach: bool,
    as_json: bool,
    parallel: ParallelSettings | None,
) -> None:
    """Drive one submitted request here, or start its driver in a new session."""
    from mosaic.core.pipeline.graph import (
        load_request,
        request_status_wire,
        run_request,
        start_detached_driver,
    )
    from mosaic.core.pipeline.job import CancelToken, install_signal_handler

    ds = load_dataset(manifest)
    try:
        request = load_request(ds.base_dir, request_id)
    except FileNotFoundError as exc:
        fail(str(exc))
    if detach:
        try:
            driver = start_detached_driver(ds, request, parallel=parallel)
        except Refusal as refusal:
            exit_refused("", refusal, as_json=as_json)
        except ClassifiedError as error:
            _fail_unrunnable(error, as_json=as_json, step=error.error_step() or "")
        if as_json:
            emit_json(
                {
                    "request_id": driver.request_id,
                    "driver_pid": driver.pid,
                    "log_path": str(driver.log_path),
                }
            )
        else:
            log(
                f"[mosaic] request {driver.request_id} is driven by pid "
                f"{driver.pid}; its output is appended to {driver.log_path}"
            )
        return
    token = CancelToken()
    install_signal_handler(token)
    try:
        with stdout_to_stderr():
            done = run_request(ds, request, cancel_token=token, parallel=parallel)
    except Refusal as refusal:
        exit_refused("", refusal, as_json=as_json)
    except ClassifiedError as error:
        _fail_unrunnable(error, as_json=as_json, step=error.error_step() or "")
    report = request_status_wire(ds.base_dir, request)
    if as_json:
        emit_json(
            {
                "request_id": done.request_id,
                "status": report.status,
                "verdict": report.verdict,
                "ran": list(done.ran),
                "skipped": list(done.skipped),
                "cancelled": list(done.cancelled),
                "blocked": list(done.blocked),
                "log_path": str(done.log_path),
                "cleanup": None
                if report.cleanup is None
                else report.cleanup.model_dump(mode="json"),
            }
        )
    else:
        _render_request_status(report)
        log(f"note: the steps' output is in {done.log_path}")
    if done.status == "failed" or done.cleanup_status == "failed":
        raise typer.Exit(code=1)
    if done.status == "cancelled":
        raise typer.Exit(code=130)


def submit_command(
    recipe: RecipeOption,
    manifest: ManifestOption,
    entry: EntriesOption = None,
    group: GroupOption = None,
    step_scope: StepScopeOption = None,
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
    refuse_duplicates: Annotated[
        bool,
        typer.Option(
            "--refuse-duplicates",
            help=(
                "Refuse the submission, writing nothing, while a request of the "
                "same recipe over an overlapping scope has not ended. A submitted "
                "request whose steps were never run has not ended."
            ),
        ),
    ] = False,
    cleanup: CleanupOption = "none",
    keep_newest: KeepNewestOption = 1,
    min_age_hours: MinAgeOption = 24.0,
    retire: RetireOption = None,
    as_json: JsonOption = False,
) -> None:
    """Record a submission of a recipe, and print the command for every step.

    Writes the recipe and a request beside it into the dataset, assigns one
    attempt id per step, and pins the version every step's producer declares.
    Nothing is executed: the printed commands are, and any driver that can run
    them in dependency order will run the graph correctly.

    An open request of the same recipe over an overlapping scope is reported on
    stderr and under ``duplicates``, or refuses the submission with
    ``--refuse-duplicates``.

    ``--cleanup superseded`` records the runs that the plan reviewed under
    ``--expect-plan`` found removable, and whatever runs the request removes
    them after every step has finished. ``--retire <digest>`` adds the runs that
    a saved recipe's requests wrote, in any storage, and the plan reviewed must
    have been planned with the same digests. A queue that dispatches the steps
    itself runs no cleanup: ``pipeline run --request`` does.
    """
    from mosaic.core.pipeline.graph import plan_fingerprint, step_argv, submit_request

    ds = load_dataset(manifest)
    parsed = _recipe(recipe, as_json=as_json)
    scope = entries_scope(entry, group)
    step_scopes = parse_step_scopes(step_scope)
    try:
        with stdout_to_stderr():
            submitted = submit_request(
                ds,
                parsed,
                scope=scope,
                allow_partial=allow_partial,
                max_concurrent_steps=max_concurrent_steps,
                owner=owner,
                expect_plan=expect_plan,
                refuse_duplicates=refuse_duplicates,
                step_scopes=step_scopes,
                cleanup=_cleanup_policy(cleanup, keep_newest, min_age_hours),
                retire=retire or (),
            )
    except ClassifiedError as error:
        _fail_unrunnable(error, as_json=as_json)
    request = submitted.request
    _warn_duplicates(submitted.duplicates)
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
                "step_scopes": {
                    step_id: narrowed.model_dump(mode="json")
                    for step_id, narrowed in sorted(request.step_scopes.items())
                },
                "steps": steps,
                "duplicates": list(submitted.duplicates),
                "cleanup": None
                if request.cleanup is None
                else request.cleanup.model_dump(mode="json"),
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
    if request.cleanup is not None:
        retiring = (
            f", {len(request.cleanup.retired)} recipe(s) retired"
            if request.cleanup.retired
            else ""
        )
        log(
            f"note: {len(request.cleanup.approved)} superseded run(s){retiring} are "
            f"removed once every step has finished, by 'mosaic pipeline run "
            f"--request {request.request_id}'"
        )


def status_command(
    manifest: ManifestOption,
    request_id: Annotated[
        str, typer.Option("--request", help="Which submission to report on.")
    ],
    wait_seconds: Annotated[
        int,
        typer.Option(
            "--wait-seconds",
            help=(
                "Wait up to this many seconds for a step to end, and report as "
                "soon as one has, or the request has. 0 reports at once."
            ),
            min=0,
            max=MAX_WAIT_SECONDS,
        ),
    ] = 0,
    brief: Annotated[
        bool,
        typer.Option(
            "--brief",
            help=(
                "List only the steps that are running, failed, cancelled or "
                "partial, and those that ended during the wait. Count the rest."
            ),
        ),
    ] = False,
    as_json: JsonOption = False,
) -> None:
    """Say how far one submission got, from its steps' own attempt logs.

    Deliberately not a coverage report: whether every step's *work* is done is a
    question about artifacts, which ``plan`` answers. This one is about attempts,
    so it stays cheap enough to poll. Each step reports its outcome, progress,
    recorded error and first 20 failed entries, its phase and its liveness, and
    the request a ``verdict`` that names a finished request with lost entries
    ``partial``. A running request reads ``orphaned`` when nothing drives it:
    its driver lease is stale and none of its step processes may still run, or
    it has no lease and each of its running steps is provably dead (its pid
    exited, its host rebooted, or it wrote nothing for ten heartbeat intervals).
    ``pipeline run --request`` resumes it. A step of ``unknown`` liveness runs on
    another host whose run-log declares no heartbeat, and ``mosaic cancel
    --assume-exited`` records it once its process is known to have exited.

    ``--wait-seconds`` waits instead of polling: the report comes as soon as a
    step has ended since the command began, or the request has ended or reads
    ``orphaned``, and otherwise when the time is up. ``--brief`` lists the steps
    that are running or did not finish cleanly, and counts the others.
    """
    from mosaic.core.pipeline.graph import load_request, request_status_wire

    ds = load_dataset(manifest, read_only=True)
    try:
        request = load_request(ds.base_dir, request_id)
    except FileNotFoundError as exc:
        fail(str(exc))
    report = request_status_wire(
        ds.base_dir, request, brief=brief, wait_seconds=wait_seconds
    )
    if as_json:
        emit_json(report.model_dump(mode="json"))
        return
    _render_request_status(report)


def _render_request_status(report: RequestStatusWire) -> None:
    """Print a request's verdict and a table of its steps."""
    typer.echo(f"request {report.request_id}: {report.verdict}")
    render_table(
        [
            {
                "step": step.step,
                "execution_id": step.execution_id,
                "status": step.outcome,
                "run_id": step.run_id or "-",
                "entries_failed": step.entries_failed,
                "phase": step.phase or "-",
                "liveness": step.liveness or "-",
            }
            for step in report.steps
        ],
        [
            "step",
            "execution_id",
            "status",
            "run_id",
            "entries_failed",
            "phase",
            "liveness",
        ],
    )
    if report.omitted_finished or report.omitted_pending:
        log(
            f"note: {report.omitted_finished} finished and {report.omitted_pending} "
            "not started step(s) are not listed"
        )
    if report.cleanup is not None:
        _render_cleanup_status(report.cleanup)
    if report.poll_after_seconds is not None:
        log(f"note: poll again in {report.poll_after_seconds} s")


def _render_cleanup_status(cleanup: CleanupStatusWire) -> None:
    """Print a request's cleanup: its status, its totals and the runs it kept."""
    typer.echo(
        f"cleanup {cleanup.execution_id}: {cleanup.status}, removed "
        f"{cleanup.removed_runs} of {cleanup.approved_runs} approved run(s) "
        f"({_size(cleanup.removed_bytes)}), kept {cleanup.kept_runs}, failed "
        f"{cleanup.failed_runs}, interrupted {cleanup.interrupted_runs}"
    )
    if cleanup.kept:
        render_table(
            [
                {
                    "storage": run.storage,
                    "run_id": run.run_id,
                    "reason": run.reason,
                    "detail": run.detail,
                }
                for run in cleanup.kept
            ],
            ["storage", "run_id", "reason", "detail"],
        )


def _size(size: int) -> str:
    """*size* bytes in the largest unit that keeps it at least 1."""
    amount = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if amount < 1024 or unit == "GB":
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} GB"


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


def _emit_plan(
    plan: "Plan",
    *,
    open_requests: "tuple[RequestSummaryWire, ...]" = (),
    retirable: "tuple[RetirableRecipeWire, ...]" = (),
    retirable_omitted: int = 0,
    as_json: bool,
) -> None:
    """Render a plan as its fingerprint and a table, or as the JSON an API serves.

    Each of the plan's cautions and of *open_requests* is noted below the table,
    followed by the *retirable* recipes of a plan asked for a cleanup.
    """
    from mosaic.core.pipeline.graph import plan_wire

    wire = plan_wire(
        plan,
        open_requests=open_requests,
        retirable=retirable,
        retirable_omitted=retirable_omitted,
    )
    if as_json:
        emit_json(wire.model_dump(mode="json"))
        return
    typer.echo(f"plan {wire.fingerprint}")
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
    for caution in wire.cautions:
        log(f"caution: {caution}")
    for planned in plan.steps:
        narrowed = plan.step_scopes.get(planned.step_id)
        if narrowed is not None:
            selector = json.dumps(narrowed.model_dump(mode="json", exclude_none=True))
            log(
                f"note: step {planned.step_id!r} is narrowed to {selector}, "
                f"{len(planned.coverage.target)} of {len(plan.scope)} entries"
            )
    if wire.cleanup is not None:
        superseded = [run for step in wire.steps for run in step.superseded]
        if superseded:
            typer.echo("superseded runs")
            retiring = any(run.retired for run in superseded)
            render_table(
                [
                    {
                        "step": step.step,
                        "run_id": run.run_id,
                        "removable": "yes" if run.removable else "kept",
                        "reason": run.reason,
                        "size": _size(run.bytes),
                        "recipe": run.retired or "-",
                    }
                    for step in wire.steps
                    for run in step.superseded
                ],
                ["step", "run_id", "removable", "reason", "size"]
                + (["recipe"] if retiring else []),
            )
        if wire.cleanup.retired_runs:
            typer.echo("retired runs")
            render_table(
                [
                    {
                        "recipe": run.retired,
                        "storage": run.storage,
                        "run_id": run.run_id,
                        "removable": "yes" if run.removable else "kept",
                        "reason": run.reason,
                        "size": _size(run.bytes),
                    }
                    for run in wire.cleanup.retired_runs
                ],
                ["recipe", "storage", "run_id", "removable", "reason", "size"],
            )
        omitted = (
            sum(step.superseded_omitted for step in wire.steps)
            + wire.cleanup.retired_omitted
        )
        log(
            f"cleanup: {wire.cleanup.removable_runs} superseded run(s) removable "
            f"({_size(wire.cleanup.removable_bytes)}), {wire.cleanup.kept_runs} "
            f"kept ({_size(wire.cleanup.kept_bytes)})"
            + (f"; {omitted} not listed" if omitted else "")
        )
        if wire.retirable:
            typer.echo("retirable recipes (pass a digest to --retire)")
            render_table(
                [
                    {
                        "recipe": row.digest,
                        "name": row.name or "-",
                        "requests": row.requests,
                        "runs": row.runs,
                        "open": "yes" if row.open else "no",
                    }
                    for row in wire.retirable
                ],
                ["recipe", "name", "requests", "runs", "open"],
            )
            if wire.retirable_omitted:
                log(f"note: {wire.retirable_omitted} more recipe(s) not listed")
    for summary in open_requests:
        log(
            f"note: request {summary.request_id} of this recipe is "
            f"{summary.verdict}"
            + (
                "; 'mosaic pipeline run --request' resumes it"
                if summary.verdict == "orphaned"
                else ""
            )
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
