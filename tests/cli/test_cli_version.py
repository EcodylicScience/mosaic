"""Test ``mosaic --version``, the root option that prints the installed version."""

from __future__ import annotations

from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.version import installed_version

runner = CliRunner()


def test_version_prints_the_installed_version_alone() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout == f"{installed_version()}\n"


def test_no_arguments_still_print_the_help() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 2
    assert "Usage: mosaic" in result.output
    assert "--version" in result.output


def test_version_runs_no_subcommand() -> None:
    result = runner.invoke(app, ["--version", "pipeline", "status"])

    assert result.exit_code == 0
    assert result.stdout == f"{installed_version()}\n"
