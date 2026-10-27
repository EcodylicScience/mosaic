"""``mosaic sources``: declare where a dataset draws its raw files from.

A source is the durable form of a search directory. Before it existed, a scan's
directories and its recipe -- its extensions, its source format, its grouping
rule -- were arguments typed at the command line and remembered nowhere, so a
rescan was only reproducible if the operator retyped them identically.
"""

from __future__ import annotations

import re

from pathlib import Path
from typing import Annotated, Literal

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail
from mosaic.core.media.store_division import RecordingIdentity
from mosaic.core.strict_model import terse
from mosaic.user_paths import user_path

sources_app = typer.Typer(
    name="sources",
    help="Declare and inspect the directories and files a dataset scans.",
    no_args_is_help=True,
    add_completion=False,
)

ManifestOption = Annotated[
    Path,
    typer.Option(
        "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
    ),
]
KindOption = Annotated[
    str, typer.Option("--kind", help="Which raw root: media, tracks or labels.")
]


def _csv(value: str | None) -> tuple[str, ...]:
    return tuple(part.strip() for part in (value or "").split(",") if part.strip())


SourceKindArg = Literal["media", "tracks", "labels"]


_FIELD_FLAGS = re.compile(
    r"(?<![\w-])(?<!a )(multi_sequences_per_file|exclude_patterns|group_pattern|group_from"
    r"|match_mode|src_format|extensions|patterns|layout|series|divide_frames"
    r"|recording_name)(?![\w-])"
)
"""A source field named in a refusal, to spell as the flag that sets it.

"a series" is the thing, not the field.
"""


def _flagged(message: str) -> str:
    """*message* with each field it names spelled as the flag that sets it."""
    named = message.replace("recording_identity", "--recording-by")
    return _FIELD_FLAGS.sub(
        lambda match: "--" + match.group(1).replace("_", "-"), named
    )


def _check_kind(kind: str) -> SourceKindArg:
    """Narrow a free-text option to the closed set the library takes.

    Spelled out rather than cast: the three branches are what actually proves to
    the type checker that the string is one of the three, and a cast would only
    assert it.
    """
    if kind == "media":
        return "media"
    if kind == "tracks":
        return "tracks"
    if kind == "labels":
        return "labels"
    fail(f"--kind must be media, tracks or labels, got {kind!r}")


@sources_app.command("list")
def list_sources(
    manifest: ManifestOption,
    kind: Annotated[
        str | None,
        typer.Option("--kind", help="Restrict to one kind. Default: all three."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Show every declared source, and whether its files are currently there."""
    dataset = load_dataset(manifest, read_only=True)
    kinds: list[SourceKindArg] = (
        [_check_kind(kind)] if kind else ["media", "tracks", "labels"]
    )

    listed: list[dict[str, object]] = []
    for one in kinds:
        for source in dataset.scan_sources(one):
            resolved = dataset.resolve_source_path(source)
            missing = (
                sum(1 for entry in source.files if not (resolved / entry).exists())
                if source.mode == "files"
                else 0
            )
            listed.append(
                {
                    "kind": one,
                    "id": source.id,
                    "mode": source.mode,
                    "path": str(resolved),
                    "exists": resolved.exists(),
                    "files": len(source.files),
                    "missing": missing,
                }
            )

    if as_json:
        emit_json({"sources": listed})
        return
    if not listed:
        typer.echo("No scan sources declared. Add one with 'mosaic sources add'.")
        return
    for row in listed:
        detail = f"{row['files']} file(s)" if row["mode"] == "files" else "directory"
        state = "" if row["exists"] else "  [path not present]"
        gone = f"  [{row['missing']} listed file(s) missing]" if row["missing"] else ""
        typer.echo(
            f"{row['kind']:<7} {row['id']:<24} {detail:<14} {row['path']}{state}{gone}"
        )


@sources_app.command("add")
def add_source(
    manifest: ManifestOption,
    kind: KindOption,
    path: Annotated[
        str,
        typer.Option("--path", help="Where the files are. May be outside the dataset."),
    ],
    source_id: Annotated[
        str | None,
        typer.Option("--id", help="Stable handle. Defaults to the directory name."),
    ] = None,
    file: Annotated[
        list[str] | None,
        typer.Option(
            "--file",
            help="Claim only this file, relative to --path (repeatable). "
            "Makes this a file source, which claims nothing else beside it.",
        ),
    ] = None,
    files_from: Annotated[
        Path | None,
        typer.Option(
            "--files-from", help="Read the file list from this file, one per line."
        ),
    ] = None,
    recursive: Annotated[
        bool, typer.Option("--recursive/--no-recursive", help="Walk subdirectories.")
    ] = True,
    extensions: Annotated[
        str | None,
        typer.Option("--extensions", help="Media only: comma-separated suffixes."),
    ] = None,
    layout: Annotated[
        str | None,
        typer.Option("--layout", help="Media only: 'stem' or 'per_sequence'."),
    ] = None,
    match_mode: Annotated[
        str | None,
        typer.Option("--match-mode", help="Media only: 'exact' or 'prefix'."),
    ] = None,
    patterns: Annotated[
        str | None,
        typer.Option("--patterns", help="Tracks/labels only: comma-separated globs."),
    ] = None,
    src_format: Annotated[
        str | None,
        typer.Option(
            "--src-format", help="Tracks/labels only: which converter reads these."
        ),
    ] = None,
    exclude_patterns: Annotated[
        str | None,
        typer.Option(
            "--exclude-patterns", help="Tracks/labels only: basename globs to skip."
        ),
    ] = None,
    multi_sequences_per_file: Annotated[
        bool,
        typer.Option(
            "--multi-sequences-per-file", help="One file holds several sequences."
        ),
    ] = False,
    group_from: Annotated[
        str | None,
        typer.Option(
            "--group-from", help="'filename' or 'parent'. Multi-sequence files only."
        ),
    ] = None,
    group_pattern: Annotated[
        str | None,
        typer.Option("--group-pattern", help="Regex extracting the group from a path."),
    ] = None,
    md5: Annotated[
        bool,
        typer.Option(
            "--md5/--no-md5",
            help="Checksum each file. On by default: the composition hash is over these.",
        ),
    ] = True,
    series: Annotated[
        str | None,
        typer.Option(
            "--series",
            help=(
                "Labels only: claim revisions of this versioned label series "
                "(e.g. 'keypoints') instead of uploaded label files. Usually "
                "with --file, naming exactly the revisions wanted."
            ),
        ),
    ] = None,
    divide_frames: Annotated[
        int | None,
        typer.Option(
            "--divide-frames",
            help=(
                "Media only: divide each imgstore recording into sequences of "
                "this many frames, a whole multiple of the store's chunksize."
            ),
        ),
    ] = None,
    recording_by: Annotated[
        RecordingIdentity | None,
        typer.Option(
            "--recording-by",
            help=(
                "Media only, with --divide-frames: 'store' (default) makes each "
                "store a recording; 'camera' joins every store of a camera serial."
            ),
        ),
    ] = None,
    recording_name: Annotated[
        str | None,
        typer.Option(
            "--recording-name",
            help=(
                "Media only, with --divide-frames: the recording's name, in place "
                "of <camera_serial>_<directory name>."
            ),
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Declare a source. A source may point outside the dataset, unlike a root."""
    from mosaic.core.source_fields import (
        SourceFields,
        SourceFieldsInvalid,
        scan_source_from_fields,
    )

    dataset = load_dataset(manifest)
    kind = _check_kind(kind)

    listed = list(file or [])
    if files_from is not None:
        list_path = user_path(files_from)
        if not list_path.exists():
            fail(f"File list not found: {list_path}")
        listed.extend(
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    fields = SourceFields(
        kind=kind,
        path=path,
        id=source_id,
        files=tuple(listed),
        recursive=recursive,
        extensions=None if extensions is None else _csv(extensions),
        layout=layout,
        match_mode=match_mode,
        patterns=None if patterns is None else _csv(patterns),
        src_format=src_format,
        exclude_patterns=None if exclude_patterns is None else _csv(exclude_patterns),
        multi_sequences_per_file=multi_sequences_per_file,
        group_from=group_from,
        group_pattern=group_pattern,
        md5=md5,
        series=series,
        divide_frames=divide_frames,
        recording_identity=recording_by,
        recording_name=recording_name,
    )
    try:
        source = scan_source_from_fields(fields)
        dataset.add_scan_source(source)
    except SourceFieldsInvalid as exc:
        fail(f"sources add failed: {_flagged(str(exc))}")
    except ValueError as exc:
        fail(f"sources add failed: {terse(exc)}")
    identifier = source.id

    if as_json:
        emit_json({"kind": kind, "id": identifier, "mode": source.mode})
    else:
        typer.echo(f"Declared {kind} source {identifier!r} ({source.mode}) -> {path}")


@sources_app.command("add-files")
def add_source_files(
    manifest: ManifestOption,
    kind: KindOption,
    source_id: Annotated[str, typer.Option("--id", help="Which source to extend.")],
    file: Annotated[
        list[str],
        typer.Option("--file", help="Path relative to the source (repeatable)."),
    ],
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Add files to a file source: a second import into the same folder."""
    dataset = load_dataset(manifest)
    try:
        added = dataset.add_source_files(_check_kind(kind), source_id, file)
    except (KeyError, ValueError) as exc:
        fail(f"sources add-files failed: {terse(exc)}")
    if as_json:
        emit_json({"id": source_id, "added": added})
    else:
        typer.echo(f"Added {added} file(s) to {source_id!r}.")


@sources_app.command("remove-files")
def remove_source_files(
    manifest: ManifestOption,
    kind: KindOption,
    source_id: Annotated[str, typer.Option("--id", help="Which source to shrink.")],
    file: Annotated[
        list[str],
        typer.Option("--file", help="Path relative to the source (repeatable)."),
    ],
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Drop files from a file source, and their index rows with them."""
    dataset = load_dataset(manifest)
    try:
        removed = dataset.remove_source_files(_check_kind(kind), source_id, file)
    except (KeyError, ValueError) as exc:
        fail(f"sources remove-files failed: {terse(exc)}")
    if as_json:
        emit_json({"id": source_id, "removed": removed})
    else:
        typer.echo(f"Removed {removed} file(s) from {source_id!r}.")


@sources_app.command("remove")
def remove_source(
    manifest: ManifestOption,
    kind: KindOption,
    source_id: Annotated[str, typer.Option("--id", help="Which source to undeclare.")],
    drop_rows: Annotated[
        bool,
        typer.Option(
            "--drop-rows/--keep-rows",
            help="Also delete the index rows this source was claiming.",
        ),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Undeclare a source. Its rows stay unless you ask for them to go."""
    dataset = load_dataset(manifest)
    checked = _check_kind(kind)
    try:
        matched = next(s for s in dataset.scan_sources(checked) if s.id == source_id)
        claim = dataset.source_claim(matched)
    except StopIteration:
        declared = sorted(s.id for s in dataset.scan_sources(checked))
        fail(f"no {checked} source named {source_id!r}; declared: {declared or 'none'}")
    orphaned = dataset.remove_scan_source(checked, source_id)
    # A series source's rows are in that series' own index, not the root's.
    from mosaic.core.manifest import LabelsScanSource

    claimed_series = matched.series if isinstance(matched, LabelsScanSource) else None
    dropped = (
        dataset.drop_claimed_rows(checked, claim, series=claimed_series)
        if drop_rows
        else 0
    )

    if as_json:
        emit_json(
            {"id": source_id, "unclaimed_rows": orphaned, "dropped_rows": dropped}
        )
        return
    typer.echo(f"Undeclared {checked} source {source_id!r}.")
    if drop_rows:
        typer.echo(f"Dropped {dropped} index row(s).")
    elif orphaned:
        typer.echo(
            f"{orphaned} index row(s) are now claimed by no source. They are kept; "
            "'mosaic reindex' and 'mosaic prune-media' are the passes that clean up."
        )
