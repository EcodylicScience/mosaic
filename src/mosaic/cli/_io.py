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
from typing import NoReturn

import typer

from mosaic.core.entry import parse_entry_tokens
from mosaic.user_paths import user_path


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
