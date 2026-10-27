"""``mosaic arenas``: the arena layout of each recording and camera.

A layout names the tanks of a camera's view by rectangles in source pixels. It
is declared once per recording, and ``assign-arenas`` reads it to name each
tracks row's arena.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail
from mosaic.cli.tags import ManifestOption

if TYPE_CHECKING:
    from mosaic.core.arenas import Rect

arenas_app = typer.Typer(
    name="arenas",
    help="Declare and show the arena layout of each recording.",
    no_args_is_help=True,
    add_completion=False,
)

RecordingOption = Annotated[
    str,
    typer.Option(
        "--recording",
        "-r",
        help="The recording, named as 'mosaic describe' lists it.",
    ),
]
CameraOption = Annotated[
    str,
    typer.Option(
        "--camera", help="The camera the layout applies to; '' for every one."
    ),
]


def _edges(text: str, flag: str) -> list[int]:
    """Parse comma-separated pixel edges."""
    try:
        return [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        fail(f"{flag} takes comma-separated whole pixels, got {text!r}.")


def _strip(text: str, flag: str) -> tuple[int, int]:
    """Parse a strip's two edges, the lower first."""
    edges = _edges(text, flag)
    if len(edges) != 2:
        fail(f"{flag} takes two edges, e.g. 95,105, got {text!r}.")
    low, high = edges
    if low >= high:
        fail(f"{flag} takes a lower edge and then a higher one, got {text!r}.")
    return low, high


def _strip_exclusions(
    xs: list[int],
    ys: list[int],
    exclude_x: list[str],
    exclude_y: list[str],
) -> dict[str, list[Rect]]:
    """Return what each grid arena excludes: the part of each strip inside it.

    A strip of ``--exclude-x`` spans every row and one of ``--exclude-y`` every
    column. Arenas are named as :meth:`ArenaLayout.grid` names them. A strip
    that crosses no arena is refused.
    """
    from mosaic.core.arenas import Rect

    cells = {
        f"{chr(ord('A') + column)}{row}": (x0, y0, x1, y1)
        for column, (x0, x1) in enumerate(pairwise(xs))
        for row, (y0, y1) in enumerate(pairwise(ys), start=1)
    }
    strips = [("--exclude-x", text) for text in exclude_x] + [
        ("--exclude-y", text) for text in exclude_y
    ]
    excluded: dict[str, list[Rect]] = {}
    for flag, text in strips:
        low, high = _strip(text, flag)
        crossed = False
        for name, (x0, y0, x1, y1) in cells.items():
            if flag == "--exclude-x":
                left, top, right, bottom = max(x0, low), y0, min(x1, high), y1
            else:
                left, top, right, bottom = x0, max(y0, low), x1, min(y1, high)
            if left < right and top < bottom:
                rect = Rect(x0=left, y0=top, x1=right, y1=bottom)
                excluded.setdefault(name, []).append(rect)
                crossed = True
        if not crossed:
            fail(f"{flag} {text} crosses no arena of the grid.")
    return excluded


@arenas_app.command("set")
def set_arenas(
    manifest: ManifestOption,
    recording: RecordingOption,
    layout: Annotated[
        Path | None,
        typer.Option("--layout", help="A JSON file holding an ArenaLayout."),
    ] = None,
    xs: Annotated[
        str | None,
        typer.Option("--xs", help="Grid column edges in pixels, e.g. 0,506,1012."),
    ] = None,
    ys: Annotated[
        str | None,
        typer.Option("--ys", help="Grid row edges in pixels, e.g. 0,170,346."),
    ] = None,
    capacity: Annotated[
        int,
        typer.Option("--capacity", help="Each grid arena's capacity; 0 for no limit."),
    ] = 1,
    exclude_x: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude-x",
            help=(
                "A vertical strip x0,x1 that every grid arena it crosses "
                "excludes, e.g. 95,105. Repeatable."
            ),
        ),
    ] = None,
    exclude_y: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude-y",
            help=(
                "A horizontal strip y0,y1 that every grid arena it crosses "
                "excludes. Repeatable."
            ),
        ),
    ] = None,
    camera: CameraOption = "",
) -> None:
    """Declare a recording's arenas from a layout file or a grid of edges."""
    from mosaic.core.arenas import ArenaLayout
    from mosaic.core.strict_model import terse

    if (layout is None) == (xs is None or ys is None):
        fail("give --layout, or --xs and --ys for a grid.")
    if layout is not None and (exclude_x or exclude_y):
        fail(
            "--exclude-x and --exclude-y apply to a grid; a layout file names its own."
        )
    try:
        if layout is not None:
            declared = ArenaLayout.model_validate_json(layout.read_text("utf-8"))
        else:
            column_edges = _edges(xs or "", "--xs")
            row_edges = _edges(ys or "", "--ys")
            declared = ArenaLayout.grid(
                xs=column_edges,
                ys=row_edges,
                capacity=capacity or None,
                exclude=_strip_exclusions(
                    column_edges, row_edges, exclude_x or [], exclude_y or []
                ),
            )
    except (OSError, ValueError) as exc:
        fail(f"arenas set failed: {terse(exc)}")
    dataset = load_dataset(manifest)
    try:
        key = dataset.recording_named(recording)
        path = dataset.set_arena_layout(
            declared, group=key.group, sequence=key.sequence, camera=camera
        )
    except ValueError as exc:
        fail(f"arenas set failed: {terse(exc)}")
    names = ", ".join(declared.names)
    typer.echo(f"Set {len(declared.arenas)} arenas of {recording!r} ({names}): {path}")


@arenas_app.command("show")
def show_arenas(
    manifest: ManifestOption,
    recording: Annotated[
        str | None,
        typer.Option(
            "--recording",
            "-r",
            help="Show the layout this recording finds; unset lists every layout.",
        ),
    ] = None,
    camera: CameraOption = "",
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """List the declared layouts, or show the one a recording and camera find."""
    from mosaic.core.dataset_summary import arena_layout_summaries
    from mosaic.core.strict_model import terse

    dataset = load_dataset(manifest, read_only=True)
    try:
        if recording is None:
            listed = arena_layout_summaries(dataset)
            if as_json:
                emit_json({"arena_layouts": [item.model_dump() for item in listed]})
                return
            if not listed:
                typer.echo(
                    "No arena layout is declared. Set one with 'mosaic arenas set'."
                )
                return
            for item in listed:
                where = (
                    f"{item.recording} [{item.camera}]"
                    if item.camera
                    else item.recording
                )
                typer.echo(
                    f"{where:<32} {len(item.arenas):>3} arenas  {item.digest[:12]}"
                )
            return
        key = dataset.recording_named(recording)
        found = dataset.arena_layout(key.group, key.sequence, camera)
    except ValueError as exc:
        fail(f"arenas show failed: {terse(exc)}")
    if found is None:
        fail(f"no arena layout applies to {recording!r}.")
    if as_json:
        emit_json(
            {
                "recording": recording,
                "camera": found.key.camera,
                "digest": found.digest,
                "layout": found.layout.model_dump(mode="json"),
            }
        )
        return
    typer.echo(f"{recording} ({found.layout.width}x{found.layout.height} px)")
    for ordinal, arena in enumerate(found.layout.arenas):
        region = arena.region
        rect = f"x {region.x0}-{region.x1}, y {region.y0}-{region.y1}"
        limit = "" if arena.capacity is None else f", capacity {arena.capacity}"
        excluded = f", {len(arena.exclude)} excluded" if arena.exclude else ""
        typer.echo(f"  {ordinal:>3} {arena.name:<12} {rect}{limit}{excluded}")
