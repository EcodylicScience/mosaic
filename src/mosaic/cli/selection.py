"""``mosaic selection``: a dataset's saved selections.

A selection names the entries it was saved with and does not grow when entries
are added, so a recipe or a script that names it computes the same thing later.
It is a file under ``<dataset>/.mosaic/selections/``. ``list`` and ``show`` read
and write nothing. ``save`` and ``delete`` refuse a dataset that a service
manages. A scope flag refers to a saved selection as ``@name`` (``--entries
@name``, ``--entry @name``) or as ``{"selection": "name"}``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import (
    command_line_scope,
    emit_json,
    entries_scope,
    fail,
    fail_classified,
    resolve_choices,
)
from mosaic.cli._render import render_kv, render_table
from mosaic.cli.tags import ManifestOption
from mosaic.core.entry import Entry, entry_token, parse_entry_tokens
from mosaic.user_paths import user_path

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.dataset_summary import SelectionSummary
    from mosaic.core.scope import Scope
    from mosaic.core.selections import UnreadableSelection

selection_app = typer.Typer(
    name="selection",
    help="Save, list, show and delete named entry sets that do not grow.",
    no_args_is_help=True,
    add_completion=False,
)

NameArgument = Annotated[str, typer.Argument(help="The selection's name.")]
JsonOption = Annotated[bool, typer.Option("--json", help="Emit JSON.")]

_SOURCES = "--from-scope, --entries-file, --entry or --group"


def _refuse_managed(dataset: Dataset, verb: str) -> None:
    """Fail when a service manages *dataset*, which then owns what is saved in it."""
    if dataset.manifest.is_managed:
        fail(
            f"The dataset is managed by {dataset.manifest.managed_by!r}, so "
            f"'mosaic selection {verb}' is refused."
        )


def _entries_from_file(path: Path) -> list[Entry]:
    """Read the entries in the file *path*.

    A file that starts with ``[`` is a JSON list of ``[group, sequence]`` pairs.
    Any other holds one ``group:sequence`` token on each line, and blank lines are
    skipped.
    """
    from pydantic import TypeAdapter, ValidationError

    from mosaic.core.strict_model import terse

    path = user_path(path)
    if not path.is_file():
        fail(f"Entries file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        fail(f"Entries file {path} cannot be read: {exc}")
    if text.lstrip().startswith("["):
        try:
            pairs = TypeAdapter(list[tuple[str, str]]).validate_json(text)
        except ValidationError as exc:
            fail(
                f"Entries file {path} is not a JSON list of [group, sequence] "
                f"pairs: {terse(exc)}"
            )
        return list(pairs)
    return parse_entry_tokens(
        line.strip() for line in text.splitlines() if line.strip()
    )


def _scope_entries(dataset: Dataset, scope: Scope) -> list[Entry]:
    """Return the entries that *scope* names, the dataset's own for a selector.

    Entries a scope names are returned as named, so a save refuses one that the
    dataset does not hold. A group or sequence selector gives the entries of the
    dataset that it selects.
    """
    from mosaic.core.pipeline.inventory.scan import reportable_universe

    pairs = scope.entry_pairs
    if pairs is not None:
        return sorted(pairs)
    selected = scope.selecting()
    return [entry for entry in sorted(reportable_universe(dataset)) if selected(entry)]


@selection_app.command("save")
def save_command(
    manifest: ManifestOption,
    name: NameArgument,
    from_scope: Annotated[
        str | None,
        typer.Option(
            "--from-scope",
            help="Save the entries of a scope given as JSON: an inline object, "
            '@path.json or @- for stdin, such as {"groups": ["g"]}.',
        ),
    ] = None,
    entries_file: Annotated[
        Path | None,
        typer.Option(
            "--entries-file",
            help="Save the entries in a file: one group:sequence token on each "
            "line, or a JSON list of [group, sequence] pairs.",
        ),
    ] = None,
    entry: Annotated[
        list[str] | None,
        typer.Option(
            "--entry",
            help="Save a group:sequence entry (repeatable), or the entries of a "
            "saved selection as @name, alone.",
        ),
    ] = None,
    group: Annotated[
        list[str] | None,
        typer.Option("--group", help="Save every entry of a group (repeatable)."),
    ] = None,
    note: Annotated[
        str, typer.Option("--note", help="Free text to keep with the selection.")
    ] = "",
    as_json: JsonOption = False,
) -> None:
    """Save a named entry set that does not grow. Give one source of entries."""
    from mosaic.core.pipeline.error_kind import ClassifiedError
    from mosaic.core.selections import save_selection

    given = [from_scope is not None, entries_file is not None, bool(entry), bool(group)]
    if sum(given) != 1:
        fail(f"Name the entries with exactly one of {_SOURCES}.")
    dataset = load_dataset(manifest)
    _refuse_managed(dataset, "save")
    if entries_file is not None:
        entries = _entries_from_file(entries_file)
    else:
        choice = (
            command_line_scope(None, None, None, from_scope)
            if from_scope is not None
            else entries_scope(entry, group)
        )
        if choice is None:
            fail(f"Name the entries with exactly one of {_SOURCES}.")
        scope = resolve_choices(dataset, choice, None, as_json=as_json).scope
        assert scope is not None
        entries = _scope_entries(dataset, scope)
    try:
        saved = save_selection(dataset, name, entries, note)
    except ClassifiedError as error:
        fail_classified(error, as_json=as_json)
    except ValueError as exc:
        fail(f"selection save failed: {exc}")
    if as_json:
        emit_json(saved.model_dump(mode="json"))
        return
    count = len(saved.entries)
    typer.echo(
        f"Saved selection {saved.name} ({count} {'entry' if count == 1 else 'entries'})."
    )


@selection_app.command("list")
def list_command(manifest: ManifestOption, as_json: JsonOption = False) -> None:
    """List the saved selections, and the files that hold none."""
    from mosaic.core.dataset_summary import selection_summaries

    dataset = load_dataset(manifest, read_only=True)
    saved, unreadable = selection_summaries(dataset)
    if as_json:
        emit_json(
            {
                "selections": [item.model_dump(mode="json") for item in saved],
                "unreadable": [item.model_dump(mode="json") for item in unreadable],
            }
        )
        return
    if saved:
        render_selections(saved)
    elif not unreadable:
        typer.echo("No selection is saved. Save one with 'mosaic selection save'.")
    for item in unreadable:
        typer.echo(_unreadable_line(dataset, item))


def render_selections(items: Sequence[SelectionSummary]) -> None:
    """Print saved selections as a table of name, entry count, date and note."""
    render_table(
        [
            {
                "name": item.name,
                "entries": item.n_entries,
                "created_at": item.created_at,
                "note": item.note,
            }
            for item in items
        ],
        ["name", "entries", "created_at", "note"],
    )


def _unreadable_line(dataset: Dataset, item: UnreadableSelection) -> str:
    """Say what is wrong with a selection file, and how to delete it.

    A file whose stem is a valid selection name goes with ``mosaic selection
    delete``. One whose stem is not, such as a sync-conflict copy ``day1 (1).json``,
    is named by no selection, so its path is given to delete by hand.
    """
    from mosaic.core.selections import selections_directory, validate_selection_name

    stem = item.file.removesuffix(".json")
    head = f"unreadable: {item.file}: {item.reason}."
    try:
        _ = validate_selection_name(stem)
    except ValueError:
        path = selections_directory(dataset) / item.file
        return f"{head} No selection name reaches it. Delete the file {path} by hand."
    return f"{head} Delete it with 'mosaic selection delete {stem}'."


@selection_app.command("show")
def show_command(
    manifest: ManifestOption, name: NameArgument, as_json: JsonOption = False
) -> None:
    """Show a selection's fields and its entries."""
    from mosaic.core.pipeline.error_kind import ClassifiedError
    from mosaic.core.selections import read_selection

    dataset = load_dataset(manifest, read_only=True)
    try:
        saved = read_selection(dataset, name)
    except ClassifiedError as error:
        fail_classified(error, as_json=as_json)
    if as_json:
        emit_json(saved.model_dump(mode="json"))
        return
    render_kv(
        {
            "name": saved.name,
            "format": saved.format,
            "created_at": saved.created_at,
            "note": saved.note,
            "digest": saved.digest,
            "entries": len(saved.entries),
        }
    )
    for group_name, sequence in saved.entries:
        typer.echo(entry_token(group_name, sequence))


@selection_app.command("delete")
def delete_command(manifest: ManifestOption, name: NameArgument) -> None:
    """Delete a selection. Naming it again later means saving it again."""
    from mosaic.core.pipeline.error_kind import ClassifiedError
    from mosaic.core.selections import delete_selection

    dataset = load_dataset(manifest)
    _refuse_managed(dataset, "delete")
    try:
        delete_selection(dataset, name)
    except ClassifiedError as error:
        fail_classified(error, as_json=False)
    typer.echo(f"Deleted selection {name}.")
