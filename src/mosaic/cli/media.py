"""``mosaic media``: mosaic-media's commands, with a ``probe`` that also reads imgstores.

mosaic-media probes a file through ffprobe and knows no imgstore, a directory of
chunk files. This app mounts every command of ``mosaic_media.cli.media_app``
except ``probe``, and its own ``probe`` reads a store through mosaic's native
reader. Any other path is probed by mosaic-media's command, so its output there
is the same.
"""

from __future__ import annotations

import dataclasses
import enum
import json
from pathlib import Path
from typing import Annotated

import typer
from mosaic_media.cli.application import media_app as upstream_media_app
from mosaic_media.cli.application import probe as upstream_probe
from typer.main import get_command_name
from typer.models import CommandInfo

from mosaic.cli._io import fail
from mosaic.core.narrowing import is_object_set

media_app = typer.Typer(
    name="media",
    help=(
        "Probe a video or an imgstore, compare two videos for duplication, and "
        "run the minimum ffmpeg transcode a verdict calls for."
    ),
    no_args_is_help=True,
    add_completion=False,
)


def _command_name(command: CommandInfo) -> str:
    """The name that Typer gives *command* on the command line."""
    if command.name:
        return command.name
    if command.callback is None:
        return ""
    return get_command_name(command.callback.__name__)


def _json_default(value: object) -> object:
    """A set as a sorted list and an enum as its value, as mosaic-media writes them."""
    if is_object_set(value):
        return sorted(value, key=str)
    if isinstance(value, enum.Enum):
        return value.value
    message = f"cannot serialize {type(value).__name__} to JSON"
    raise TypeError(message)


def _store_document(path: Path) -> dict[str, object]:
    """The facts, the stored verdict, and the identity and chunk layout of a store."""
    from mosaic.core.media.imgstore_io import (
        STORE_VERDICT,
        imgstore_facts,
        imgstore_store_identity,
    )
    from mosaic.core.media.imgstore_native import NativeStore
    from mosaic.core.pipeline.stream_copy import header_codec

    facts = imgstore_facts(path)
    with NativeStore(path) as store:
        chunks = store.chunk_paths()
        layout: dict[str, object] = {
            "chunksize": store.chunksize,
            "chunk_count": store.chunk_count,
            "header_codec": header_codec(chunks[0]) if chunks else "",
        }
    identity = dataclasses.asdict(imgstore_store_identity(path))
    return {
        "facts": dataclasses.asdict(facts),
        "verdict": dataclasses.asdict(STORE_VERDICT),
        "store": {**identity, **layout},
    }


def probe(
    path: Annotated[
        Path, typer.Argument(help="A video file, or an imgstore directory.")
    ],
) -> None:
    """Probe PATH and print its MediaFacts and both verdicts as JSON on stdout.

    For an imgstore, the verdict is the one the media index records for a store,
    and ``store`` adds the Motif identity fields of ``metadata.yaml``, the
    declared ``chunksize``, the number of indexed chunks, and the codec named in
    the header of the first chunk (``""`` for a store of image files).
    """
    from mosaic_media import MediaProbeError

    from mosaic.core.media.imgstore_io import is_imgstore

    if not is_imgstore(path):
        upstream_probe(path)
        return
    try:
        document = _store_document(path)
    except MediaProbeError as exc:
        fail(f"probe failed for {path}: {exc}")
    typer.echo(json.dumps(document, default=_json_default, indent=2, sort_keys=True))


# mosaic-media's commands in their order, with this module's probe in place of its own.
for command in upstream_media_app.registered_commands:
    if _command_name(command) == "probe":
        _ = media_app.command(name="probe")(probe)
    else:
        media_app.registered_commands.append(command)


__all__ = ["media_app"]
