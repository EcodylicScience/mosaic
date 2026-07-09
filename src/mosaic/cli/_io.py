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
from pathlib import Path
from typing import NoReturn

import typer


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
        path = Path(value[1:])
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


def parse_entries(entries: list[str] | None) -> list[tuple[str, str]]:
    """Parse repeated ``group:sequence`` tokens into pairs (split on first ``:``).

    An empty group is allowed (``:seq`` -> ``("", "seq")``).
    """
    pairs: list[tuple[str, str]] = []
    for token in entries or []:
        if ":" not in token:
            fail(f"Invalid --entries value {token!r}; expected 'group:sequence'.")
        group, sequence = token.split(":", 1)
        pairs.append((group, sequence))
    return pairs
