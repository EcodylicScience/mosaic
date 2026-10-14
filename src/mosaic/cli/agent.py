"""``mosaic agent``: the guide for an AI coding agent that uses mosaic.

``guide`` prints the guide shipped with the installed mosaic, then the paths of
this installation. ``install`` points a project's agents at that command, which
keeps the guide they read matched to the mosaic they run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mosaic.agent import (
    AGENT_FILES,
    BLOCK_BEGIN,
    BLOCK_END,
    SITE_URL,
    SKILL_DIRECTORY,
    Installation,
    find_installation,
    guide_text,
    install_agent_files,
)
from mosaic.cli._io import fail
from mosaic.user_paths import user_path

agent_app = typer.Typer(
    name="agent",
    help="Print or install the guide for an AI coding agent that uses mosaic.",
    no_args_is_help=True,
    add_completion=False,
)


@agent_app.command("guide")
def guide_command() -> None:
    """Print the agent guide for this version, then this installation's paths."""
    try:
        guide = guide_text()
    except FileNotFoundError as exc:
        fail(f"The agent guide is missing from this installation: {exc}")
    installation = find_installation(Path.cwd())
    typer.echo(guide.rstrip())
    typer.echo("")
    typer.echo("\n".join(_installation_lines(installation)))


def _installation_lines(installation: Installation) -> list[str]:
    executable = (
        str(installation.executable)
        if installation.executable is not None
        else "not found; use the interpreter with -m mosaic"
    )
    lines = [
        "## This installation",
        "",
        f"- mosaic executable: {executable}",
        f"- Python interpreter: {installation.interpreter}",
    ]
    if installation.docs is not None:
        lines += [
            f"- Documentation: {installation.docs}",
            f"  A page {SITE_URL}<page>/ is {installation.docs}/<page>.md here, or",
            f"  {installation.docs}/<page>/index.md for a section.",
        ]
    else:
        lines.append(f"- Documentation: {SITE_URL}")
    if installation.manifest is not None:
        lines.append(
            f"- Dataset: {installation.manifest.parent} "
            f"(manifest {installation.manifest})"
        )
    else:
        lines.append("- Dataset: the working directory is not inside one")
    return lines


@agent_app.command("install")
def install_command(
    directory: Annotated[
        Path,
        typer.Argument(help="Project directory. Defaults to the working directory."),
    ] = Path(),
    no_memory: Annotated[
        bool,
        typer.Option(
            "--no-memory",
            help="Write the skill only, and leave CLAUDE.md and AGENTS.md alone.",
        ),
    ] = False,
) -> None:
    """Point a project's AI coding agents at `mosaic agent guide`.

    Writes .claude/skills/mosaic/SKILL.md, and a marked block in CLAUDE.md and
    AGENTS.md (creating CLAUDE.md when neither exists). Running it again replaces
    both.
    """
    target = user_path(directory).resolve()
    if not target.is_dir():
        fail(f"Directory not found: {target}")
    installation = find_installation(Path.cwd())
    try:
        written = install_agent_files(target, installation, memory=not no_memory)
    except ValueError as exc:
        fail(f"install failed: {exc}")

    for entry in written:
        typer.echo(f"{entry.action}: {entry.path}")
    removal = f"To remove, delete {target / SKILL_DIRECTORY}"
    blocks = _files_with_block(target)
    if blocks:
        names = ", ".join(str(path) for path in blocks)
        removal += f" and the lines from {BLOCK_BEGIN} to {BLOCK_END} in {names}"
    typer.echo("")
    typer.echo(f"{removal}.")


def _files_with_block(directory: Path) -> list[Path]:
    """The agent memory files in *directory* that hold a pointer block.

    An earlier install may have written one that this run left alone.
    """
    paths = [directory / name for name in AGENT_FILES]
    return [
        path
        for path in paths
        if path.is_file() and BLOCK_BEGIN in path.read_text(encoding="utf-8")
    ]
