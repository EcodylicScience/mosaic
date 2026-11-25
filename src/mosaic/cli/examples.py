"""``mosaic examples``: list the example datasets and download one.

``list`` prints the catalog without touching the network. ``fetch`` downloads an
example, unpacks it and prints the path of its manifest alone on standard output, so
that ``mosaic agent install --desktop --dataset "$(mosaic examples fetch calms21)"``
works. Progress and the closing hint go to standard error, and a failure leaves
standard output empty.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from mosaic.byte_sizes import format_size
from mosaic.cli._io import emit_json, fail, log
from mosaic.examples.catalog import CATALOG, Example
from mosaic.examples.fetch import (
    ExampleFetchError,
    FetchedExample,
    UnknownExample,
    default_directory,
    fetch_example,
    find_example,
)

examples_app = typer.Typer(
    name="examples",
    help="Download example datasets to try mosaic on.",
    no_args_is_help=True,
    add_completion=False,
)

_PICKLE_REASON = (
    "Its raw files load only by unpickling, which can run code in the file; "
    "use these settings only for datasets from sources you trust."
)

_INTERRUPTED = 130


def stdin_is_terminal() -> bool:
    """Whether standard input is a terminal that can answer a prompt."""
    return sys.stdin.isatty()


def is_windows() -> bool:
    """Whether the hint is for ``cmd.exe``, which the Miniforge prompts run."""
    return sys.platform == "win32"


def _example_json(example: Example) -> dict[str, object]:
    return {
        "name": example.name,
        "repository": example.repository,
        "revision": example.revision,
        "license": example.license,
        "summary": example.summary,
        "notebook": example.notebook,
        "raw_needs_unpickling": example.raw_needs_unpickling,
        "download_bytes": example.download_bytes,
        "archives": [
            {
                "file": archive.file,
                "unpacks_to": archive.unpacks_to,
                "download_bytes": archive.download_bytes,
                "dataset": archive.dataset,
            }
            for archive in example.archives
        ],
    }


@examples_app.command("list")
def list_command(
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the catalog as a JSON list.")
    ] = False,
) -> None:
    """List the example datasets that `mosaic examples fetch` downloads."""
    if as_json:
        emit_json([_example_json(example) for example in CATALOG])
        return
    blocks = [
        "\n".join(
            [
                example.name,
                f"  {format_size(example.download_bytes)}, {example.license}",
                f"  {example.summary}",
                f"  Notebook: {example.notebook}",
            ]
        )
        for example in CATALOG
    ]
    typer.echo("\n\n".join(blocks))


def _find(name: str) -> Example:
    """The catalog entry called *name*, or exit 2 naming the known names."""
    try:
        return find_example(name, CATALOG)
    except UnknownExample as error:
        fail(str(error), 2)


def _ask_which() -> str:
    """Print the numbered catalog on standard error and ask for a number or a name."""
    for number, example in enumerate(CATALOG, start=1):
        log(
            f"{number}. {example.name} ({format_size(example.download_bytes)}): "
            f"{example.summary}"
        )
    try:
        answer: str = typer.prompt("Which example (number or name)", err=True)
    except typer.Abort:
        raise typer.Exit(_INTERRUPTED) from None
    answer = answer.strip()
    if answer.isdecimal() and 1 <= int(answer) <= len(CATALOG):
        return CATALOG[int(answer) - 1].name
    return answer


def _hint_lines(example: Example, manifest: Path) -> list[str]:
    """The two registration commands for *manifest*, with what conversion needs."""
    windows = is_windows()
    path = (
        subprocess.list2cmdline([str(manifest)])
        if windows
        else shlex.quote(str(manifest))
    )
    desktop = "mosaic agent install --desktop"
    if example.raw_needs_unpickling:
        desktop += " --trust-pickled-files"
    lines = [
        "",
        "To try it with Claude, register the mosaic server with this dataset.",
        "In Claude Desktop (quit it first):",
        f"  {desktop} --dataset {path}",
        "In Claude Code, from a project directory:",
    ]
    if example.raw_needs_unpickling:
        verb = "set" if windows else "export"
        lines.append(f"  {verb} MOSAIC_TRUST_PICKLED_FILES=1")
    lines.append(f"  mosaic agent install --mcp --dataset {path}")
    if example.raw_needs_unpickling:
        lines += ["", _PICKLE_REASON]
    return lines


def _fetched_json(example: Example, fetched: FetchedExample) -> dict[str, object]:
    return {
        "name": fetched.name,
        "directory": str(fetched.directory),
        "dataset": str(fetched.dataset),
        "manifest": str(fetched.manifest),
        "archives": [
            {
                "file": archive.file,
                "path": str(fetched.paths[archive.unpacks_to]),
                "action": fetched.actions[archive.file],
            }
            for archive in example.archives
        ],
    }


@examples_app.command("fetch")
def fetch_command(
    name: Annotated[
        str | None,
        typer.Argument(
            metavar="[NAME]", help="A name that `mosaic examples list` shows."
        ),
    ] = None,
    directory: Annotated[
        Path | None,
        typer.Option(
            "--dir",
            help="The directory the archives unpack into. Default: "
            "~/mosaic-examples/<NAME>/.",
        ),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print the directory, the manifest and what was done to each "
            "archive as JSON, instead of the manifest path.",
        ),
    ] = False,
) -> None:
    """Download an example dataset and unpack it. Prints the path of its manifest.

    Without NAME on a terminal, asks which example. Running it again reuses a
    complete download.
    """
    if name is None:
        if as_json or not stdin_is_terminal():
            fail("Name an example. `mosaic examples list` lists them.", 2)
        name = _ask_which()
    example = _find(name)
    try:
        fetched = fetch_example(example.name, directory)
    except KeyboardInterrupt:
        raise typer.Exit(_INTERRUPTED) from None
    except ExampleFetchError as error:
        fail(str(error))
    except OSError as error:
        target = directory if directory is not None else default_directory(name)
        fail(f"Cannot write {target}: {error}")
    if as_json:
        emit_json(_fetched_json(example, fetched))
    else:
        typer.echo(str(fetched.manifest))
    log("\n".join(_hint_lines(example, fetched.manifest)))
