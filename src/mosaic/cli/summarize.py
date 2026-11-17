"""``mosaic summarize``: one feature run's rows counted and described by group.

The verb reads the run's output tables, keeps the rows that ``--where`` allows,
groups them by the ``--by`` keys and prints each group's rows and, for each value
column, its statistics and the tests that ``--test`` asks for. ``--json`` prints the
:class:`~mosaic.core.pipeline.run_summary.RunSummaryWire` that
:func:`~mosaic.core.pipeline.run_summary.run_summary_wire` returns. The verb writes
nothing to the dataset.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import (
    EntriesOption,
    GroupOption,
    TagOption,
    emit_json,
    entries_scope,
    fail_classified,
    resolve_choices,
)
from mosaic.cli._render import render_kv, render_table

if TYPE_CHECKING:
    from mosaic.core.pipeline.run_summary import RunSummaryWire
    from mosaic.core.pipeline.summary_arguments import KeyValue, SummaryTest


def summarize_command(
    ref: Annotated[
        str,
        typer.Argument(
            help=(
                "The feature run's ref, as 'mosaic inventory' prints it: "
                "feature/<storage>/<run_id>."
            )
        ),
    ],
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    by: Annotated[
        list[str] | None,
        typer.Option(
            "--by",
            help=(
                "A group key (repeatable): 'group', 'sequence', a column the run "
                "writes, such as a label kept with keep_columns, or a sequence "
                "tag. Default: one group of every row."
            ),
        ),
    ] = None,
    column: Annotated[
        list[str] | None,
        typer.Option(
            "--column",
            help=(
                "A value column or pattern (repeatable). A name with *, ? or [ is "
                "a pattern, such as 'speed_*'. Default: every numeric column."
            ),
        ),
    ] = None,
    test: Annotated[
        list[str] | None,
        typer.Option(
            "--test",
            help=(
                "A test between groups (repeatable): kruskal over all groups, or "
                "mannwhitney for every pair, adjusted by Holm within a column."
            ),
        ),
    ] = None,
    where: Annotated[
        list[str] | None,
        typer.Option(
            "--where",
            help=(
                "Keep the rows whose KEY is VALUE (repeatable). Values of one key "
                "are alternatives. A value is read as text, and as a number or a "
                "boolean where the key holds one: --where n_inds=1."
            ),
        ),
    ] = None,
    entry: EntriesOption = None,
    group: GroupOption = None,
    tag: TagOption = None,
    tag_revision: Annotated[
        int | None,
        typer.Option(
            "--tag-revision",
            min=1,
            help=(
                "The revision of the sequence tags to read, for --tag and for a "
                "--by or --where key that names a sequence tag. Default: the "
                "latest."
            ),
        ),
    ] = None,
    allow_partial: Annotated[
        bool,
        typer.Option(
            "--allow-partial",
            help="Read a run that lacks entries of the scope or is not finished.",
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit as a JSON object.")
    ] = False,
) -> None:
    """Summarize one feature run: rows, statistics and tests by group.

    A --by or --where key that the run does not write and that names a sequence
    tag groups or filters the entries by the tag's value. --tag selects the entries
    that hold a sequence tag, as 'mosaic pipeline plan --tag' does. A --tag and a
    tag key read one revision of the tags, which the output records.
    """
    # Imported in the body, so --help and the other verbs stay import-light.
    from mosaic.core.pipeline.error_kind import ClassifiedError
    from mosaic.core.pipeline.run_summary import summarize_ref

    ds = load_dataset(manifest, read_only=True)
    chosen = resolve_choices(
        ds,
        entries_scope(entry, group, tags=tag),
        None,
        as_json=as_json,
        revision=tag_revision,
    )
    if chosen.tags is not None:
        tag_revision = None
    try:
        tests = _tests(test or [])
        wire = summarize_ref(
            ds,
            ref,
            by=by or (),
            columns=column,
            tests=tests,
            where=_where(where or []),
            scope=chosen.scope,
            allow_partial=allow_partial,
            tag_revision=tag_revision,
            tag_table=chosen.tags,
        )
    except ClassifiedError as error:
        fail_classified(error, as_json=as_json)
    if as_json:
        emit_json(wire.model_dump(mode="json"))
        return
    _render(wire, tests)


def _tests(names: list[str]) -> tuple[SummaryTest, ...]:
    """Return the ``--test`` names as tests, each once.

    Raises:
        PreviewInvalid: A name other than ``kruskal`` and ``mannwhitney``.
    """
    from mosaic.core.pipeline.preview import PreviewInvalid
    from mosaic.core.pipeline.summary_arguments import SUMMARY_TESTS

    known: dict[str, SummaryTest] = {test: test for test in SUMMARY_TESTS}
    tests: list[SummaryTest] = []
    for name in dict.fromkeys(names):
        test = known.get(name)
        if test is None:
            msg = (
                f"--test {name!r} is not a test. The tests are "
                f"{', '.join(SUMMARY_TESTS)}."
            )
            raise PreviewInvalid(msg, field="test")
        tests.append(test)
    return tuple(tests)


def _where(tokens: list[str]) -> dict[str, list[KeyValue]]:
    """Return the ``KEY=VALUE`` tokens as the values to keep, by key.

    A value is every reading of its text that a key could hold
    (:func:`~mosaic.core.pipeline.summary_arguments.readings`), so ``n_inds=1``
    keeps the integer 1 and ``code=1`` keeps the text ``"1"``.

    Raises:
        PreviewInvalid: A token without ``=`` or with an empty key.
    """
    from mosaic.core.pipeline.preview import PreviewInvalid
    from mosaic.core.pipeline.summary_arguments import read_where

    texts: dict[str, list[KeyValue]] = {}
    for token in tokens:
        name, separator, text = token.partition("=")
        if not separator or not name:
            msg = f"--where {token!r} is not KEY=VALUE."
            raise PreviewInvalid(msg, field="where")
        texts.setdefault(name, []).append(text)
    return read_where(texts)


def _render(wire: RunSummaryWire, tests: tuple[SummaryTest, ...]) -> None:
    """Print the run, the groups, the statistics and each test asked for."""
    header: dict[str, object] = {
        "ref": wire.ref,
        "by": ", ".join(wire.by) or "(none)",
        "columns": ", ".join(wire.columns),
        "entries": wire.n_entries,
    }
    if wire.tag_revision is not None:
        if wire.tag_keys:
            header["tag_keys"] = (
                f"{', '.join(wire.tag_keys)} (revision {wire.tag_revision.revision})"
            )
        else:
            header["tag_revision"] = wire.tag_revision.revision
    render_kv(header)
    typer.echo()
    render_table(
        [
            {**group.key, **_cells(group.model_dump(exclude={"key"}))}
            for group in wire.groups
        ],
        [*wire.by, "n_rows", "n_entries"],
    )
    typer.echo()
    render_table(
        [
            {**row.key, **_cells(row.model_dump(exclude={"key"}))}
            for row in wire.statistics
        ],
        [*wire.by, "column", "n", "n_missing", "mean", "sd", "sem", "median"],
    )
    if "kruskal" in tests:
        typer.echo()
        render_table(
            [_cells(row.model_dump()) for row in wire.kruskal],
            ["column", "k", "n", "statistic", "df", "p", "reason"],
        )
    if "mannwhitney" in tests:
        typer.echo()
        render_table(
            [
                {
                    **_cells(row.model_dump(exclude={"a", "b"})),
                    **{f"{name}_a": value for name, value in row.a.items()},
                    **{f"{name}_b": value for name, value in row.b.items()},
                }
                for row in wire.mann_whitney
            ],
            [
                "column",
                *(f"{name}_{side}" for name in wire.by for side in "ab"),
                "n_a",
                "n_b",
                "statistic",
                "p",
                "p_holm",
                "comparisons",
                "reason",
            ],
        )
    _note_what_is_left_out(wire)


def _cells(row: Mapping[str, object]) -> dict[str, object]:
    """Return *row* with a missing value blank, so that a table cell is empty."""
    return {name: "" if value is None else value for name, value in row.items()}


def _note_what_is_left_out(wire: RunSummaryWire) -> None:
    """Report on stderr what the summary does not hold."""
    if wire.rows_without_key:
        typer.echo(
            f"note: rows left out because a key is null: {wire.rows_without_key}",
            err=True,
        )
    for label, entries in (
        ("entries with no row", wire.entries_without_rows),
        ("entries missing from the run", wire.missing),
    ):
        if entries:
            named = ", ".join(f"{group}:{sequence}" for group, sequence in entries)
            typer.echo(f"note: {label}: {named}", err=True)
    if not wire.finished:
        typer.echo("note: the run is not finished", err=True)
