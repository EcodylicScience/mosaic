"""``mosaic tracking`` sub-app: list | describe (schema-driven tracking-op discovery)."""

from __future__ import annotations

import typer

from mosaic.cli.tracking.describe import describe_command
from mosaic.cli.tracking.list import list_command

tracking_app = typer.Typer(
    name="tracking",
    help="Discover tracking ops and their parameter schemas.",
    no_args_is_help=True,
    add_completion=False,
)

_ = tracking_app.command(name="list")(list_command)
_ = tracking_app.command(name="describe")(describe_command)
