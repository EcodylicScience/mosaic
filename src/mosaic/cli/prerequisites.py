"""``mosaic prerequisites``: the runs that an op needs before it reads its entries.

The verb takes the op flags of ``mosaic run``. A user swaps the verb and keeps the
rest of the line. Its output lists each run in dependency order with the
``mosaic run`` command of the run. It lists each refused entry with its cause and
reason, and then the command of the op. Nothing runs, and the dataset is left
unchanged. ``--json`` prints the same as one object.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import (
    RUN_SCOPE_FLAGS_REMEDY,
    RunEntriesOption,
    RunGroupsOption,
    RunKindOption,
    RunParamsOption,
    RunScopeOption,
    RunSequencesOption,
    command_line_params,
    command_line_scope,
    emit_json,
    fail,
    fail_classified,
    refuse_scope_named_twice,
    stdout_to_stderr,
    with_command_line_scope,
)
from mosaic.core.pipeline.run_commands import mosaic_run_command

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.entry import Entry
    from mosaic.tracking import PrerequisitePlan

type NoCommandCause = Literal[
    "every_entry_refused", "no_indexed_media", "empty_scope", "nameless_cameras"
]
"""The cause that withholds the command of the op."""

_NAMELESS_CAMERA: Final = (
    "a camera of a nameless entry that the plan keeps only in part, and the scope "
    "of a run covers every camera of an entry"
)
"""Why a run's scope cannot name an entry of an unscoped plan."""

_BECAUSE: Final[dict[NoCommandCause, str]] = {
    "every_entry_refused": "every entry is refused",
    "no_indexed_media": "the media index of the dataset is absent or empty",
    "empty_scope": "the scope selects an empty set of entries",
    "nameless_cameras": f"each entry left is {_NAMELESS_CAMERA}",
}


@dataclass(frozen=True, slots=True)
class _Answer:
    """A plan, with the commands that it prints.

    Attributes:
        plan: The plan.
        run_commands: The command of each of the plan's runs, in order.
        command: The op's command, or ``""`` when it is withheld.
        no_command: The cause that withholds the op's command, or ``None``.
        omitted: The entries of the plan that the op's command omits.
    """

    plan: PrerequisitePlan
    run_commands: tuple[str, ...]
    command: str
    no_command: NoCommandCause | None
    omitted: tuple[Entry, ...]

    @property
    def no_command_reason(self) -> str:
        """Return the sentence that says why the op's command is withheld.

        It is ``""`` when the command is printed.
        """
        if self.no_command is None:
            return ""
        because = _BECAUSE[self.no_command]
        return f"The {self.plan.kind} command is withheld, because {because}."

    @property
    def omitted_reason(self) -> str:
        """Return the sentence that says why the op's command omits entries.

        It is ``""`` when the command omits no entry.
        """
        if not self.omitted:
            return ""
        return (
            f"The {self.plan.kind} command omits each of these entries, "
            f"because each is {_NAMELESS_CAMERA}."
        )


def prerequisites_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    kind: RunKindOption,
    params: RunParamsOption = None,
    groups: RunGroupsOption = None,
    sequences: RunSequencesOption = None,
    entries: RunEntriesOption = None,
    scope_json: RunScopeOption = None,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json", help="Emit one JSON object on stdout; logs go to stderr."
        ),
    ] = False,
) -> None:
    """Print the runs that an op needs before it reads its entries, and run none."""
    # Typer refuses a missing --kind, which this command declares without a default.
    assert kind is not None
    refuse_scope_named_twice(entries, groups, sequences, scope_json)
    # Imported in the body, so --help and the other verbs stay import-light.
    from pydantic import ValidationError

    from mosaic.core.pipeline.ops import ScopeRefused, check_scope_takes, registered_op
    from mosaic.core.strict_model import terse
    from mosaic.tracking import plan_prerequisites, register_ops

    ds = load_dataset(manifest, read_only=True)
    params_dict = command_line_params(params)
    scope = command_line_scope(entries, groups, sequences, scope_json)
    register_ops()
    try:
        op = registered_op(kind)
        resolved = ds.resolve_scope(scope)
        check_scope_takes(kind, op.scope_takes, resolved)
        with stdout_to_stderr():
            plan = plan_prerequisites(
                ds, kind, params_dict, entries=resolved.op_entries
            )
    except ScopeRefused as refusal:
        message = with_command_line_scope(str(refusal), RUN_SCOPE_FLAGS_REMEDY)
        fail_classified(refusal, as_json=as_json, message=message)
    except (KeyError, FileNotFoundError) as exc:
        fail(str(exc))
    except ValidationError as exc:
        fail(f"Invalid params: {terse(exc)}")
    except ValueError as exc:
        fail(str(exc))

    answer = _answer(
        ds, plan, params_dict or {}, scoped=resolved.op_entries is not None
    )
    if as_json:
        emit_json(_payload(answer))
    else:
        _render(answer)


def _answer(
    ds: Dataset, plan: PrerequisitePlan, params: Mapping[str, object], *, scoped: bool
) -> _Answer:
    """Return *plan* with the command of each run and of the op.

    The op's command reads the given *params*, with the plan's ``media`` set, over
    the entries of :func:`_op_scope`.

    Args:
        ds: The dataset that the plan was made for.
        plan: The plan.
        params: The ``--params`` mapping.
        scoped: Whether the scope flags named entries.
    """
    manifest = str(ds.manifest_path.absolute())
    run_commands = tuple(
        mosaic_run_command(
            run.kind,
            run.entries,
            params=_params_text(run.params.model_dump(mode="json")),
            manifest=manifest,
        )
        for run in plan.runs
    )
    op_scope = _op_scope(ds, plan, scoped=scoped)
    if isinstance(op_scope, str):
        return _Answer(plan, run_commands, "", op_scope, ())
    named, omitted = op_scope
    op_params = {**params, "media": plan.media} if plan.media else dict(params)
    command = mosaic_run_command(
        plan.kind, named, params=_params_text(op_params), manifest=manifest
    )
    return _Answer(plan, run_commands, command, None, omitted)


def _op_scope(
    ds: Dataset, plan: PrerequisitePlan, *, scoped: bool
) -> tuple[tuple[Entry, ...], tuple[Entry, ...]] | NoCommandCause:
    """Return the entries that the op's command names and omits, or why it is withheld.

    Without scope flags and without a refusal, the command omits the scope flags.
    It then covers the entries of the plan as an unscoped run reads them.
    Otherwise it names each entry of the plan by its indexed key. A run's scope
    names a nameless entry with every camera. The command therefore omits the
    label of a camera when the plan leaves out another camera of its entry. The
    command is withheld when it would name no entry, because a command without a
    scope covers every entry.

    Args:
        ds: The dataset that the plan was made for.
        plan: The plan.
        scoped: Whether the scope flags named entries.

    Returns:
        The entries that the command names, empty to omit the scope flags, and
        the entries of the plan that it omits. Otherwise, why the command is
        withheld.
    """
    from mosaic.tracking import indexed_entries

    if not plan.entries:
        if plan.refusals:
            return "every_entry_refused"
        return "empty_scope" if scoped else "no_indexed_media"
    if scoped:
        return plan.entries, ()
    if not plan.refusals:
        return (), ()
    named, omitted = indexed_entries(ds, plan.entries)
    return (named, omitted) if named else "nameless_cameras"


def _params_text(params: Mapping[str, object]) -> str:
    """Return *params* as compact JSON, or ``""`` when there are none."""
    return json.dumps(params, separators=(",", ":")) if params else ""


def _payload(answer: _Answer) -> dict[str, object]:
    """Return *answer* as the object that ``--json`` prints.

    ``after`` lists the positions in ``runs``, from 0, of the runs that must
    finish first.
    """
    plan = answer.plan
    return {
        "kind": plan.kind,
        "media": plan.media,
        "runs": [
            {
                "kind": run.kind,
                "run_id": run.run_id,
                "params": run.params.model_dump(mode="json"),
                "entries": [list(entry) for entry in run.entries],
                "after": list(run.after),
                "command": command,
            }
            for run, command in zip(plan.runs, answer.run_commands, strict=True)
        ],
        "refusals": [
            {
                "group": refusal.group,
                "sequence": refusal.sequence,
                "camera": refusal.camera,
                "cause": refusal.cause,
                "reason": refusal.reason,
            }
            for refusal in plan.refusals
        ],
        "entries": [list(entry) for entry in plan.entries],
        "command": answer.command or None,
        "no_command": None
        if answer.no_command is None
        else {"cause": answer.no_command, "reason": answer.no_command_reason},
        "omitted": None
        if not answer.omitted
        else {
            "entries": [list(entry) for entry in answer.omitted],
            "reason": answer.omitted_reason,
        },
    }


def _render(answer: _Answer) -> None:
    """Print *answer*: the runs, numbered from 1, the refusals and the op's command.

    The entries of the plan that the command omits follow the command.
    """
    plan = answer.plan
    typer.echo(f"kind: {plan.kind}")
    typer.echo(f"media: {plan.media or 'the entry media'}")
    typer.echo(f"runs: {len(plan.runs)}")
    for position, (run, command) in enumerate(
        zip(plan.runs, answer.run_commands, strict=True), start=1
    ):
        after = ", ".join(str(index + 1) for index in run.after)
        suffix = f" after {after}" if after else ""
        typer.echo(f"  {position}. {run.kind} {run.run_id}{suffix}")
        typer.echo(f"     {command}")
    typer.echo(f"refusals: {len(plan.refusals)}")
    for refusal in plan.refusals:
        camera = f" camera {refusal.camera}" if refusal.camera else ""
        typer.echo(f"  {refusal.group}:{refusal.sequence}{camera} ({refusal.cause})")
        typer.echo(f"     {refusal.reason}")
    typer.echo("command:")
    typer.echo(f"  {answer.command or answer.no_command_reason}")
    if answer.omitted:
        typer.echo(f"omitted: {len(answer.omitted)}")
        for group, sequence in answer.omitted:
            typer.echo(f"  {group}:{sequence}")
        typer.echo(f"     {answer.omitted_reason}")
