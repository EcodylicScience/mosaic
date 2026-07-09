"""``mosaic features`` sub-app: list | describe (schema-driven feature discovery)."""

from __future__ import annotations

import typer

from mosaic.cli.features.describe import describe_command
from mosaic.cli.features.list import list_command

features_app = typer.Typer(
    name="features",
    help="Discover features and their parameter schemas.",
    no_args_is_help=True,
    add_completion=False,
)

_ = features_app.command(name="list")(list_command)
_ = features_app.command(name="describe")(describe_command)
