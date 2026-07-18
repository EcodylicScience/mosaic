"""The ``mosaic`` CLI: a thin Typer adapter over the library's Job Contract.

Bound to the ``mosaic`` console script via ``[project.scripts]`` in
``pyproject.toml`` (``mosaic = "mosaic.cli:app"``). This CLI is the executor's
unit of work (Layer 2 shells out to ``mosaic run --json``) and the future
MCP-stdio payload, so it is never throwaway. Command modules keep their heavy
mosaic imports lazy (inside the command bodies) so ``--help`` and the read-only
commands stay fast and import-light.
"""

from __future__ import annotations

import typer

from mosaic.cli.cancel import cancel_command
from mosaic.cli.convert_labels import convert_labels_command
from mosaic.cli.convert_tracks import convert_tracks_command
from mosaic.cli.features import features_app
from mosaic.cli.index_media import index_media_command
from mosaic.cli.index_tracks import index_tracks_command
from mosaic.cli.reindex import reindex_command
from mosaic.cli.run import run_command
from mosaic.cli.runs import runs_command
from mosaic.cli.sequences import sequences_command
from mosaic.cli.status import status_command
from mosaic.cli.tracking import tracking_app
from mosaic.cli.trex import trex_command
from mosaic_media.cli import media_app

app = typer.Typer(
    name="mosaic",
    help="Mosaic execution CLI: run features and tracking ops under the Job Contract.",
    no_args_is_help=True,
    add_completion=False,
)

# Compute + observe (the critical path that unblocks the Layer-2 executor).
_ = app.command(name="run")(run_command)
_ = app.command(name="status")(status_command)
_ = app.command(name="runs")(runs_command)
_ = app.command(name="cancel")(cancel_command)
_ = app.command(name="trex")(trex_command)

# Discover.
app.add_typer(features_app, name="features")
app.add_typer(tracking_app, name="tracking")
app.add_typer(media_app, name="media")
_ = app.command(name="sequences")(sequences_command)

# Dataset prep.
_ = app.command(name="index-media")(index_media_command)
_ = app.command(name="index-tracks")(index_tracks_command)
_ = app.command(name="reindex")(reindex_command)
_ = app.command(name="convert-tracks")(convert_tracks_command)
_ = app.command(name="convert-labels")(convert_labels_command)

__all__ = ["app"]
