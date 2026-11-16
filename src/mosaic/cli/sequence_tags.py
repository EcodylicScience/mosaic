"""``mosaic sequence-tags``: import a CSV of sequence tags and list the table's tags.

The tags live in the dataset's ``tags`` series, so they travel with the dataset
and a run reads them as it reads any other input. ``mosaic tags`` keeps the
dataset's own tags, which describe the dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail
from mosaic.core.strict_model import terse
from mosaic.user_paths import user_path

if TYPE_CHECKING:
    from mosaic.core.annotations.sequence_tag_wire import (
        SequenceTagImport,
        SequenceTagListing,
    )
    from mosaic.core.typed_attribute import TypedAttributeType

sequence_tags_app = typer.Typer(
    name="sequence-tags",
    help="Sequence tags: import a CSV, list them.",
    no_args_is_help=True,
    add_completion=False,
)

ManifestOption = Annotated[
    Path,
    typer.Option(
        "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
    ),
]


def _types(options: list[str] | None) -> dict[str, "TypedAttributeType"]:
    """The ``--type NAME=TYPE`` options as a mapping, failing on a malformed one."""
    from mosaic.core.typed_attribute import as_typed_attribute_type

    declared: dict[str, TypedAttributeType] = {}
    for option in options or []:
        name, separator, type_text = option.rpartition("=")
        name = name.strip()
        if not (separator and name):
            fail(f"--type {option!r} is not NAME=TYPE.", code=2)
        type_ = as_typed_attribute_type(type_text.strip())
        if type_ is None:
            fail(
                f"--type {option!r} names no type. TYPE is one of label, text, "
                "int, float, bool or categorical.",
                code=2,
            )
        declared[name] = type_
    return declared


def _print_import(report: "SequenceTagImport") -> None:
    changed = [change for change in report.changes if change.change != "unchanged"]
    if report.dry_run:
        typer.echo(
            f"Dry run: nothing written. {report.rows_matched} rows name entries."
        )
    elif report.written:
        typer.echo(
            f"Wrote revision {report.revision}. {report.rows_matched} rows name entries."
        )
    else:
        typer.echo(
            f"Nothing changed. The tag table stays at revision {report.revision}."
        )
    for change in changed:
        typer.echo(
            f"{change.change:<8}{change.name:<24} {change.type:<12} "
            f"{change.entries_set} set, {change.entries_cleared} cleared"
        )
    if report.dry_run and not changed:
        typer.echo("No tag changes.")


@sequence_tags_app.command("import")
def import_command(
    csv: Annotated[Path, typer.Argument(help="The CSV of sequence tags.")],
    manifest: ManifestOption,
    key: Annotated[
        str | None,
        typer.Option(
            "--key",
            help=(
                "NAME=FORM: the column that names each row's entries and how its "
                "values read, where FORM is entry, sequence or recording. Without "
                "it the header holds exactly one of entry, group with sequence, "
                "sequence, or recording."
            ),
        ),
    ] = None,
    type_: Annotated[
        list[str] | None,
        typer.Option(
            "--type",
            help=(
                "NAME=TYPE (repeatable): declare a tag's type, where TYPE is "
                "label, text, int, float, bool or categorical. The one way to "
                "change a defined tag's type."
            ),
        ),
    ] = None,
    replace: Annotated[
        bool,
        typer.Option(
            "--replace",
            help="Remove the sequence tags that the CSV does not name.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print the changes, write nothing and take no lock.",
        ),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the import report as JSON."),
    ] = False,
) -> None:
    """Write a CSV of sequence tags as the next revision of the tag table.

    One column names each row's entries. Every other column is a tag, and a
    tag that the CSV names takes the CSV's values on every entry. A tag it does
    not name keeps its values, unless --replace removes it. A row that names no
    entry is warned of on stderr, and the matched rows are written. A value that
    its tag's type refuses fails the import before anything is written. A
    dataset that a service manages is refused, since the service's next save
    replaces the table.
    """
    from mosaic.core.annotations.sequence_tags import (
        SequenceTagCsvInvalid,
        import_sequence_tags,
        parse_key_option,
    )

    csv_path = user_path(csv)
    if not csv_path.is_file():
        fail(f"CSV not found: {csv_path}")
    try:
        key_column = None if key is None else parse_key_option(key)
    except SequenceTagCsvInvalid as exc:
        fail(f"--key {terse(exc)}", code=2)
    declared = _types(type_)
    dataset = load_dataset(manifest, read_only=dry_run)
    try:
        report = import_sequence_tags(
            dataset,
            csv_path,
            key=key_column,
            types=declared,
            replace=replace,
            dry_run=dry_run,
        )
    except ValueError as exc:
        fail(f"sequence-tags import failed: {terse(exc)}")
    if as_json:
        emit_json(report.model_dump(mode="json"))
    else:
        _print_import(report)


def _print_listing(listing: "SequenceTagListing") -> None:
    if not listing.tags:
        typer.echo("No sequence tags. Import a CSV with 'mosaic sequence-tags import'.")
        return
    typer.echo(f"Revision {listing.revision}: {listing.entries} entries.")
    for tag in listing.tags:
        typer.echo(f"{tag.name:<24} {tag.type:<12} {tag.entries} entries")
        if tag.type != "label":
            for counted in tag.values:
                typer.echo(f"    {counted.value!r}: {counted.entries}")


@sequence_tags_app.command("list")
def list_command(
    manifest: ManifestOption,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the listing as JSON.")
    ] = False,
) -> None:
    """Show each sequence tag of the latest revision with its values and counts."""
    from mosaic.core.annotations.sequence_tags import list_sequence_tags

    dataset = load_dataset(manifest, read_only=True)
    try:
        listing = list_sequence_tags(dataset)
    except ValueError as exc:
        fail(f"sequence-tags list failed: {terse(exc)}")
    if as_json:
        emit_json(listing.model_dump(mode="json"))
    else:
        _print_listing(listing)
