"""Run a ``mosaic`` command the way a test asserts on its ``--json`` output."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from pydantic import TypeAdapter
from typer.testing import CliRunner

from mosaic.cli import app

_PAYLOAD: Final = TypeAdapter(dict[str, object])


def invoke_json(args: Sequence[str]) -> dict[str, object]:
    """Run ``mosaic`` with *args*, assert it exited 0, and return its JSON stdout.

    A failure names the exit code and both streams, because a command says why it
    failed on stderr.
    """
    result = CliRunner().invoke(app, list(args))
    assert result.exit_code == 0, (
        f"exit={result.exit_code}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    return _PAYLOAD.validate_json(result.stdout)
