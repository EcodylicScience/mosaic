"""``mosaic notes`` and ``mosaic tags``: what a dataset says about itself.

Both live in the manifest, so they travel with the dataset when it is copied,
archived or synced. Tags carry the same type / constraints / value shape as the
sequence and individual tags in mosaic-api, so a tag means one thing wherever it
is written -- but these describe the *dataset*, not a sequence within it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Final

import typer
from pydantic import TypeAdapter, ValidationError

from mosaic.core.json_value import JsonValue

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail, terse
from mosaic.user_paths import user_path

notes_app = typer.Typer(
    name="notes",
    help="Read and write the dataset's free-text notes.",
    no_args_is_help=True,
    add_completion=False,
)
tags_app = typer.Typer(
    name="tags",
    help="Declare and set the dataset's typed tags.",
    no_args_is_help=True,
    add_completion=False,
)

_CONSTRAINTS: Final = TypeAdapter(dict[str, JsonValue])
"""Parses ``--constraints``. A JSON parser returns an untyped object, and a
constraint blob that is a list or a bare string is a real thing to be handed."""

ManifestOption = Annotated[
    Path,
    typer.Option(
        "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
    ),
]


@notes_app.command("show")
def show_notes(
    manifest: ManifestOption,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Print the dataset's notes."""
    dataset = load_dataset(manifest)
    if as_json:
        emit_json({"notes": dataset.notes})
    elif dataset.notes:
        typer.echo(dataset.notes)


@notes_app.command("set")
def set_notes(
    manifest: ManifestOption,
    text: Annotated[
        str | None,
        typer.Argument(help="The notes. Use '-' to read them from standard input."),
    ] = None,
    from_file: Annotated[
        Path | None, typer.Option("--from-file", help="Read the notes from this file.")
    ] = None,
) -> None:
    """Replace the dataset's notes."""
    dataset = load_dataset(manifest)
    if (text is None) == (from_file is None):
        fail("Pass either the text (or '-') or --from-file, not both and not neither.")
    if from_file is not None:
        notes_path = user_path(from_file)
        if not notes_path.exists():
            fail(f"Notes file not found: {notes_path}")
        body = notes_path.read_text(encoding="utf-8")
    else:
        body = sys.stdin.read() if text == "-" else str(text)
    try:
        dataset.set_notes(body)
    except ValueError as exc:
        fail(f"notes set failed: {terse(exc)}")
    typer.echo(f"Wrote {len(body)} characters of notes.")


@notes_app.command("clear")
def clear_notes(manifest: ManifestOption) -> None:
    """Remove the dataset's notes."""
    dataset = load_dataset(manifest)
    dataset.set_notes("")
    typer.echo("Cleared the notes.")


@tags_app.command("list")
def list_tags(
    manifest: ManifestOption,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Show every tag with its type, constraints and value."""
    dataset = load_dataset(manifest)
    listed = [
        {
            "name": tag.name,
            "type": tag.type,
            "type_constraints": tag.type_constraints,
            "value": tag.value,
            "description": tag.description,
            "display_order": tag.display_order,
        }
        for tag in dataset.tags
    ]
    if as_json:
        emit_json({"tags": listed})
        return
    if not listed:
        typer.echo("No tags. Define one with 'mosaic tags define'.")
        return
    for tag in dataset.tags:
        shown = "" if tag.type == "label" else f" = {tag.value!r}"
        typer.echo(f"{tag.name:<24} {tag.type:<12}{shown}")


@tags_app.command("define")
def define_tag(
    manifest: ManifestOption,
    name: Annotated[str, typer.Argument(help="The tag name.")],
    type_: Annotated[
        str,
        typer.Option(
            "--type",
            help="label, text, int, float, bool or categorical.",
        ),
    ],
    constraints: Annotated[
        str | None,
        typer.Option("--constraints", help="Constraints as a JSON object."),
    ] = None,
    options: Annotated[
        str | None,
        typer.Option("--options", help="categorical: comma-separated allowed values."),
    ] = None,
    minimum: Annotated[
        float | None, typer.Option("--min", help="int/float: lowest allowed value.")
    ] = None,
    maximum: Annotated[
        float | None, typer.Option("--max", help="int/float: highest allowed value.")
    ] = None,
    max_length: Annotated[
        int | None, typer.Option("--max-length", help="text: longest allowed value.")
    ] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="What the tag means.")
    ] = None,
    order: Annotated[
        int, typer.Option("--order", help="Display order. Ties break on name.")
    ] = 0,
) -> None:
    """Declare a tag, or redeclare one, keeping any value that still fits."""
    from mosaic.core.manifest import DatasetTag

    dataset = load_dataset(manifest)

    # Either spelling assembles the same object and goes through the same
    # validator, so a shorthand cannot express something the JSON form could not.
    built: dict[str, JsonValue] = {}
    if constraints is not None:
        if options or minimum is not None or maximum is not None or max_length:
            fail("Pass --constraints or the typed shorthands, not both.")
        try:
            built = _CONSTRAINTS.validate_json(constraints)
        except ValidationError:
            fail("--constraints must be a JSON object mapping names to values.")
    else:
        if options:
            built["options"] = [
                part.strip() for part in options.split(",") if part.strip()
            ]
        if minimum is not None:
            built["min"] = int(minimum) if type_ == "int" else minimum
        if maximum is not None:
            built["max"] = int(maximum) if type_ == "int" else maximum
        if max_length is not None:
            built["max_length"] = max_length

    existing = dataset.tag(name)
    try:
        tag = DatasetTag.model_validate(
            {
                "name": name,
                "type": type_,
                "type_constraints": built,
                # A redefinition keeps the value it had, which is what makes the
                # re-validation meaningful: narrowing a constraint under a value
                # that no longer fits has to be refused, not silently applied.
                "value": existing.value if existing is not None else None,
                "description": description,
                "display_order": order,
            }
        )
        dataset.define_tag(tag)
    except ValueError as exc:
        fail(f"tags define failed: {terse(exc)}")
    typer.echo(f"Defined tag {name!r} ({type_}).")


@tags_app.command("set")
def set_tag(
    manifest: ManifestOption,
    name: Annotated[str, typer.Argument(help="The tag to set.")],
    value: Annotated[
        str | None,
        typer.Argument(help="The value. Omit only for a label tag, which has none."),
    ] = None,
) -> None:
    """Set an already-defined tag's value, parsed as its declared type."""
    dataset = load_dataset(manifest)
    tag = dataset.tag(name)
    if tag is None:
        declared = sorted(existing.name for existing in dataset.tags)
        fail(f"no tag named {name!r}; defined: {declared or 'none'}. Define it first.")

    if tag.type == "label":
        if value is not None:
            fail(
                f"tag {name!r} is a label: it is either attached or not, and has no value."
            )
        typed: str | int | float | bool | None = None
    elif value is None:
        fail(f"tag {name!r} is a {tag.type}, so it needs a value.")
    elif tag.type == "int":
        try:
            typed = int(value)
        except ValueError:
            fail(f"tag {name!r} is an int; {value!r} is not one.")
    elif tag.type == "float":
        try:
            typed = float(value)
        except ValueError:
            fail(f"tag {name!r} is a float; {value!r} is not a number.")
    elif tag.type == "bool":
        lowered = value.strip().lower()
        if lowered not in ("true", "false"):
            fail(f"tag {name!r} is a bool; pass 'true' or 'false', not {value!r}.")
        typed = lowered == "true"
    else:
        typed = value

    try:
        dataset.set_tag_value(name, typed)
    except (KeyError, ValueError) as exc:
        fail(f"tags set failed: {terse(exc)}")
    typer.echo(f"Set {name!r}.")


@tags_app.command("remove")
def remove_tag(
    manifest: ManifestOption,
    name: Annotated[str, typer.Argument(help="The tag to drop.")],
) -> None:
    """Drop a tag."""
    dataset = load_dataset(manifest)
    if not dataset.remove_tag(name):
        fail(f"no tag named {name!r}.")
    typer.echo(f"Removed {name!r}.")
