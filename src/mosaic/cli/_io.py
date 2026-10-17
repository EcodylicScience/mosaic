"""Shared CLI I/O helpers: JSON args, stream-separated output, clean failures.

The ``--json`` contract for every command: stdout carries **exactly one** JSON
value (via :func:`emit_json`); all human/log breadcrumbs go to **stderr** (via
:func:`log`). Machine consumers (the Layer-2 executor, MCP) therefore get a
pristine stdout to parse regardless of any progress chatter.
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Generator
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer

from mosaic.core.entry import parse_entry_tokens
from mosaic.user_paths import user_path

if TYPE_CHECKING:
    from mosaic.core.pipeline.error_kind import ClassifiedError
    from mosaic.core.pipeline.refusal import Refusal
    from mosaic.core.scope import Scope

EntriesOption = Annotated[
    list[str] | None,
    typer.Option(
        "--entry",
        help="Narrow to a group:sequence entry (repeatable). Default: the whole dataset.",
    ),
]

GroupOption = Annotated[
    list[str] | None,
    typer.Option(
        "--group",
        help="Narrow to every entry of a group (repeatable). Not with --entry.",
    ),
]

StepScopeOption = Annotated[
    list[str] | None,
    typer.Option(
        "--step-scope",
        help=(
            "Narrow one step to part of the request's scope, as STEP=SCOPE "
            'where SCOPE is a JSON selector such as {"groups": ["train"]} or '
            '{"entries": [["g", "s"]]}, or @file.json (repeatable).'
        ),
    ),
]


@contextlib.contextmanager
def stdout_to_stderr() -> Generator[None]:
    """Redirect library stdout chatter (progress/completion prints) to stderr.

    Keeps the ``--json`` contract (one clean JSON value on stdout) intact no
    matter what the compute path prints. Wrap library calls, not the final
    :func:`emit_json`.
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield


def fail(message: str, code: int = 1) -> NoReturn:
    """Print *message* to stderr and exit with *code* (default 1)."""
    typer.echo(message, err=True)
    raise typer.Exit(code=code)


def log(message: str) -> None:
    """Emit a human/log breadcrumb to stderr (keeps stdout clean for ``--json``)."""
    typer.echo(message, err=True)


def emit_json(payload: object) -> None:
    """Emit one JSON value to stdout (the machine-readable ``--json`` output)."""
    typer.echo(json.dumps(payload, indent=2, default=str))


def fail_classified(
    error: ClassifiedError,
    *,
    as_json: bool,
    message: str | None = None,
    step: str = "",
) -> NoReturn:
    """Report a classified error, and exit 1.

    The message goes to stderr. Under ``--json`` the error is also the one JSON
    value on stdout, as its :class:`~mosaic.core.pipeline.error_envelope.ErrorEnvelope`.

    Args:
        error: The error.
        as_json: Whether ``--json`` was given.
        message: The sentence to report in place of ``str(error)``, such as one
            naming this command's flags.
        step: The step to name when *error* names none, or ``""`` when the
            step is unknown.

    Raises:
        typer.Exit: Always, with code 1.
    """
    from mosaic.core.pipeline.error_envelope import error_envelope

    envelope = error_envelope(error)
    if message is not None:
        envelope = envelope.model_copy(update={"message": message})
    if envelope.step is None and step:
        envelope = envelope.model_copy(update={"step": step})
    log(envelope.message)
    if as_json:
        emit_json(envelope.model_dump(mode="json"))
    raise typer.Exit(code=1) from None


def exit_refused(
    execution_id: str, refusal: Refusal, *, step_id: str = "", as_json: bool
) -> NoReturn:
    """Report a refusal, and exit with the code reserved for one.

    The message goes to stderr. Under ``--json`` the refusal is also the one JSON
    value on stdout: the refusal's
    :class:`~mosaic.core.pipeline.error_envelope.ErrorEnvelope`, with the
    execution, ``"status": "refused"``, the step and the ``error_json`` blob that
    a ledger row records beside it. ``step`` is ``""`` when the step is unknown.

    Args:
        execution_id: The attempt that refused.
        refusal: The refusal.
        step_id: The step to name when *refusal* does not name its own, or
            ``""`` for a run outside a pipeline.
        as_json: Whether ``--json`` was given.

    Raises:
        typer.Exit: Always, with ``REFUSED_EXIT_CODE``.
    """
    from mosaic.core.pipeline.error_envelope import error_envelope
    from mosaic.core.pipeline.refusal import REFUSED_EXIT_CODE

    log(f"[mosaic] refused ({refusal.reason}): {refusal}")
    if as_json:
        emit_json(
            {
                **error_envelope(refusal).model_dump(mode="json"),
                "execution_id": execution_id,
                "status": "refused",
                "step": refusal.step_id or step_id,
                "error_json": refusal.error_json(step_id),
            }
        )
    raise typer.Exit(code=REFUSED_EXIT_CODE) from None


def load_json_arg(value: str | None) -> object | None:
    """Resolve a ``--params``/``--inputs``-style argument to a JSON value.

    Accepts ``@path.json`` (read a file), ``@-`` (read stdin), or an inline
    JSON string. Returns ``None`` when *value* is ``None``. JSON / file errors
    exit cleanly via :func:`fail`.
    """
    if value is None:
        return None
    if value == "@-":
        raw = sys.stdin.read()
        source = "<stdin>"
    elif value.startswith("@"):
        # No shell expands the tilde in `@~/params.json` -- expansion applies to
        # the start of a word, and here the word starts with `@` -- so this form
        # only works if it is expanded here.
        path = user_path(value[1:])
        if not path.exists():
            fail(f"JSON file not found: {path}")
        raw = path.read_text()
        source = str(path)
    else:
        raw = value
        source = "inline JSON"
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        fail(f"Invalid JSON ({source}): {exc}")


def with_command_line_scope(refusal: str, remedy: str) -> str:
    """*refusal* with the flag spelling of its remedy appended.

    ``check_scope_takes`` is the only place a scope refusal is written, and it
    answers in ``Scope(...)`` because its callers are the library, the graph
    planner and mosaic-api, which all construct one. A person at a terminal
    types flags instead, and each command offers its own spelling of them.

    Appended rather than substituted. The sentence the checker wrote states
    what an unscoped run would cover. That is what decides whether to narrow or
    to proceed, and no flag list replaces it.

    Args:
        refusal: what :func:`~mosaic.core.pipeline.ops.check_scope_takes` said.
        remedy: the flags this command offers, as a phrase.

    Returns:
        The message to print.
    """
    return f"{refusal} At the command line: {remedy}"


def parse_entries(entries: list[str] | None) -> list[tuple[str, str]]:
    """Parse repeated ``group:sequence`` tokens into pairs.

    One grammar, in :func:`mosaic.core.entry.parse_entry_tokens`. This used to
    reject a token with no ``:`` while the ops reading the same values out of
    ``--params`` accepted it as a bare sequence in the empty group, so the two
    ways of naming one entry on one command line disagreed.
    """
    return parse_entry_tokens(entries)


def entries_scope(
    entries: list[str] | None, groups: list[str] | None = None
) -> Scope | None:
    """Return repeated ``--entry`` or ``--group`` tokens as a selector.

    ``None`` when neither is given. The two are refused together, because a
    selector names entries or groups and not both.
    """
    from mosaic.core.scope import Scope

    pairs = parse_entries(entries)
    if pairs and groups:
        fail("Give --entry or --group, not both.")
    if groups:
        return Scope(groups=list(groups))
    return Scope(entries=pairs) if pairs else None


def parse_step_scopes(values: list[str] | None) -> dict[str, Scope]:
    """Parse repeated ``--step-scope STEP=SCOPE`` tokens into step scopes.

    SCOPE is a JSON selector, inline or as ``@file.json``. A token without a
    step, a repeated step, or a value that is not a selector fails, naming
    the token.
    """
    from pydantic import ValidationError

    from mosaic.core.scope import Scope

    found: dict[str, Scope] = {}
    for token in values or ():
        step_id, separator, value = token.partition("=")
        step_id = step_id.strip()
        if not separator or not step_id or not value:
            fail(f"--step-scope {token!r} is not STEP=SCOPE.")
        if step_id in found:
            fail(f"--step-scope names step {step_id!r} twice.")
        selector = load_json_arg(value)
        if not isinstance(selector, dict):
            fail(f"--step-scope {token!r}: SCOPE must be a JSON object.")
        try:
            found[step_id] = Scope.model_validate(selector)
        except ValidationError as exc:
            fail(f"--step-scope {token!r} is not a scope: {exc}")
    return found
