"""``mosaic pipeline retire``: remove the runs that saved recipes' requests wrote.

Removing older versions of a pipeline, or a pipeline with its results, submits no
recipe. This verb plans the retirement of the recipes ``--retire`` names against
the pipelines that stay (``--keeper``), which
:func:`~mosaic.core.pipeline.graph.retire.plan_retirement` judges by the rules a
submission's cleanup follows.

The preview is the default. It opens the dataset read-only, takes no lock and
prints the plan. ``--apply`` plans the same way, then runs the approved plan as
the ``retire-recipes`` op in this process, the path a queue runs it by, and
prints what the op removed and kept. The op judges every run again, so a run that
stopped being removable since the plan is kept. The verb acts for no manager, so
a dataset another manager manages keeps every run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer

from mosaic.byte_sizes import format_size
from mosaic.cli._context import attempt_facts, load_dataset
from mosaic.cli._io import (
    EntriesOption,
    GroupOption,
    TagOption,
    emit_json,
    entries_scope,
    log,
    resolve_choices,
    stdout_to_stderr,
)
from mosaic.cli._render import render_table
from mosaic.cli.pipeline import (
    KEEP_NEWEST,
    MIN_AGE_HOURS,
    JsonOption,
    ManifestOption,
    fail_unrunnable,
    keep_newest_option,
    min_age_option,
    recipe_argument,
    render_cleanup_status,
)
from mosaic.core.pipeline.definitions.scope import definition_scope
from mosaic.core.pipeline.error_kind import ClassifiedError

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.graph import (
        CleanupStatusWire,
        RetirementPlan,
        RetirementPlanWire,
    )


def _plan(
    ds: Dataset,
    retire: list[str],
    keeper: list[str] | None,
    entry: list[str] | None,
    group: list[str] | None,
    tag: list[str] | None,
    keep_newest: int,
    min_age_hours: float,
    *,
    as_json: bool,
) -> RetirementPlan:
    """Plan retiring *retire* with each keeper over the scope the flags name."""
    from mosaic.core.pipeline.graph import (
        CleanupPolicy,
        RetirementKeeper,
        plan_retirement,
    )
    from mosaic.core.scope import Scope

    chosen = resolve_choices(
        ds, entries_scope(entry, group, tags=tag), None, as_json=as_json
    )
    scope = Scope() if chosen.scope is None else chosen.scope
    keepers = [
        RetirementKeeper(recipe=recipe_argument(each, as_json=as_json), scope=scope)
        for each in keeper or ()
    ]
    policy = CleanupPolicy(keep_newest=keep_newest, min_age_hours=min_age_hours)
    with definition_scope(ds, load=False):
        try:
            with stdout_to_stderr():
                return plan_retirement(ds, retire, keepers, policy, "")
        except ClassifiedError as error:
            fail_unrunnable(error, as_json=as_json)


def _render_plan(wire: RetirementPlanWire) -> None:
    """Print a retirement's plan: its fingerprint, the runs judged, the digests left out."""
    typer.echo(f"retirement {wire.fingerprint}")
    if wire.runs:
        render_table(
            [
                {
                    "recipe": run.retired or "-",
                    "storage": run.storage,
                    "run_id": run.run_id,
                    "removable": "yes" if run.removable else "kept",
                    "reason": run.reason,
                    "size": format_size(run.bytes),
                }
                for run in wire.runs
            ],
            ["recipe", "storage", "run_id", "removable", "reason", "size"],
        )
    if wire.not_retired:
        typer.echo("not retired")
        render_table(
            [
                {"recipe": each.digest, "reason": each.reason, "why": each.message}
                for each in wire.not_retired
            ],
            ["recipe", "reason", "why"],
        )
    log(
        f"retirement: {wire.removable_runs} run(s) removable "
        f"({format_size(wire.removable_bytes)}), {wire.kept_runs} kept "
        f"({format_size(wire.kept_bytes)}); dry run, nothing removed: --apply "
        "removes the removable runs"
    )


def _emit_report(
    report: CleanupStatusWire, extra: dict[str, object], *, as_json: bool
) -> None:
    """Print a retirement's report, with *extra* keys added under ``--json``."""
    if as_json:
        emit_json({**report.model_dump(mode="json"), **extra})
    else:
        render_cleanup_status(report)


def retire_command(
    manifest: ManifestOption,
    retire: Annotated[
        list[str],
        typer.Option(
            "--retire",
            metavar="DIGEST",
            help=(
                "A saved recipe to retire, by its digest (repeatable, at least "
                "one). The runs its requests wrote are judged for removal. A "
                "digest that names no readable recipe, or the recipe a keeper "
                "runs, is listed and left out."
            ),
        ),
    ],
    keeper: Annotated[
        list[str] | None,
        typer.Option(
            "--keeper",
            metavar="RECIPE",
            help=(
                "A pipeline that stays (repeatable), as a path to a JSON recipe, "
                "@file.json, @- or inline JSON. The runs its plan expects are "
                "never removed."
            ),
        ),
    ] = None,
    entry: EntriesOption = None,
    group: GroupOption = None,
    tag: TagOption = None,
    keep_newest: Annotated[
        int,
        keep_newest_option(
            "Keep this many of the runs the retirement judges per storage, newest "
            "first."
        ),
    ] = KEEP_NEWEST,
    min_age_hours: Annotated[
        float,
        min_age_option(
            "Keep every run the retirement judges that is younger than this many hours."
        ),
    ] = MIN_AGE_HOURS,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply/--dry-run",
            help="Remove the removable runs. Default is a dry-run report.",
        ),
    ] = False,
    as_json: JsonOption = False,
) -> None:
    """Retire saved recipes: remove the runs their requests wrote that no rule keeps.

    The keepers are the pipelines that stay, each planned over the scope that
    ``--entry``, ``--group`` or ``--tag`` names, all entries when none is given.
    A run a keeper expects, reads or that another request recorded, binds or
    pins is kept, as is a run younger than ``--min-age-hours`` and the
    ``--keep-newest`` newest of each storage.

    The dry run prints the plan's fingerprint, each run judged with its reason,
    and each digest not retired; ``--json`` prints the plan's wire. It writes
    nothing. ``--apply`` runs the approved plan as the ``retire-recipes`` op,
    which judges every run again and removes only an approved run still
    removable, and prints what it removed and kept. ``--json`` adds the op's
    ``run_id``, the attempt's facts, with its status ``partial`` when some
    removal raised, and ``not_retired``, the digests left out with their
    reasons. It exits 1 when every approved removal raised, reporting the
    attempt as ``failed`` without ``run_id``, which the op returns only when it
    ends, and 130 when cancelled.
    """
    ds = load_dataset(manifest, read_only=not apply)
    plan = _plan(
        ds,
        retire,
        keeper,
        entry,
        group,
        tag,
        keep_newest,
        min_age_hours,
        as_json=as_json,
    )
    from mosaic.core.pipeline.graph import retirement_wire

    wire = retirement_wire(plan)
    if not apply:
        if as_json:
            emit_json(wire.model_dump(mode="json"))
        else:
            _render_plan(wire)
        return

    from mosaic.runlog import new_execution_id
    from mosaic.core.pipeline.graph import retirement_status_wire
    from mosaic.core.pipeline.job import CancelToken, Cancelled, install_signal_handler
    from mosaic.core.pipeline.ops import run_op
    from mosaic.core.pipeline.retire_recipes import RetireRecipesParams
    from mosaic.core.pipeline.run import AllEntriesFailed

    for each in wire.not_retired:
        log(f"not retired: {each.digest} ({each.reason}): {each.message}")
    refused: dict[str, object] = {
        "not_retired": [each.model_dump(mode="json") for each in wire.not_retired]
    }
    spec = plan.spec
    execution_id = new_execution_id()
    token = CancelToken()
    install_signal_handler(token)
    log(f"[mosaic] execution_id={execution_id} running retire-recipes")
    try:
        with stdout_to_stderr():
            run_id = run_op(
                ds,
                "retire-recipes",
                RetireRecipesParams(retirement=spec),
                execution_id=execution_id,
                cancel_token=token,
            )
    except Cancelled:
        if as_json:
            emit_json({"execution_id": execution_id, "status": "cancelled"})
        else:
            log(f"[mosaic] cancelled {execution_id}")
        raise typer.Exit(code=130) from None
    except AllEntriesFailed:
        report = retirement_status_wire(ds.base_dir, execution_id, len(spec.approved))
        facts = attempt_facts(ds, execution_id)
        del facts["status"]
        _emit_report(report, {**facts, **refused}, as_json=as_json)
        raise typer.Exit(code=1) from None
    except ClassifiedError as error:
        fail_unrunnable(error, as_json=as_json)
    report = retirement_status_wire(ds.base_dir, execution_id, len(spec.approved))
    _emit_report(
        report,
        {"run_id": run_id, **attempt_facts(ds, execution_id), **refused},
        as_json=as_json,
    )
