"""Reading a CSV of sequence tags: which entries each row names, and what each cell holds.

A row names entries by one key column, or by the pair of columns ``group`` and
``sequence``, and every other column is a tag. The cells are text. A tag's type is
the one the caller declares, else the one the dataset already defines, else the
first of ``bool``, ``int``, ``float``, ``categorical`` and ``text`` that every cell
of the column reads as. Each value is checked with
:func:`~mosaic.core.annotations.tag_table.check_tag_value` before anything is
written, so a value that its type refuses fails the import and names its line and
column.

A row that names no entry is reported as a finding and skipped. Two rows that name
one entry, a header that names its entries two ways or none, and a cell that its
type refuses fail the read.

:func:`import_sequence_tags` writes the file as the next revision of the dataset's
tag table, and :func:`list_sequence_tags` reports what the latest revision holds.
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from mosaic.core.annotations.projection import write_tag_table_revision
from mosaic.core.annotations.sequence_tag_wire import (
    SequenceTagChange,
    SequenceTagImport,
    SequenceTagListed,
    SequenceTagListing,
    SequenceTagValueCount,
)
from mosaic.core.annotations.tag_table import (
    SequenceTags,
    TagDefinition,
    TagDefinitions,
    TagTable,
    check_tag_value,
    read_tag_table,
    reserved_name_message,
)
from mosaic.core.entry import Entry, entry_token, parse_entry_tokens
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.index_lock import index_lock_unless_dry_run
from mosaic.core.pipeline.label_series import (
    TAG_TABLE_KEY,
    TAGS_SERIES,
    LabelRevisionRef,
)
from mosaic.core.pipeline.label_series_index import (
    ResolvedRevision,
    SeveralOrigins,
    read_label_series,
    read_revision_manifest,
    resolve_series_refs,
    series_index_path,
)
from mosaic.core.pipeline.ops import IdentityDeferred
from mosaic.core.pipeline.summary_arguments import readings
from mosaic.core.scan_findings import FindingLog
from mosaic.core.typed_attribute import (
    OPTION_MAX_LENGTH,
    OPTIONS_MAX_COUNT,
    TypedAttributeType,
    TypedAttributeValue,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "KeyForm",
    "SequenceTagCsv",
    "SequenceTagCsvInvalid",
    "SequenceTagImportRefused",
    "SequenceTagTableUnavailable",
    "SequenceTagsMoved",
    "import_sequence_tags",
    "list_sequence_tags",
    "parse_key_option",
    "read_sequence_tag_csv",
]

KeyForm = Literal["entry", "sequence", "recording"]
"""How a key cell names entries.

``entry`` is a token as ``entry_token`` writes it, ``group:sequence``. ``sequence`` is
a sequence name, alone or beside a ``group`` column. ``recording`` is a continuous
group or an undivided recording's entry token, and names every sequence of it.
"""

_RESERVED_COLUMNS: Final[frozenset[str]] = frozenset(
    {"group", "sequence", "id", "frame", "time", "individual", "uid"}
)
"""The names of a column that a tag may not have, since a step that reads tags writes it."""

_KEY_FORMS: Final[tuple[KeyForm, ...]] = ("entry", "sequence", "recording")
_EXPECTED: Final[Mapping[str, str]] = {
    "bool": "true or false",
    "int": "a whole number",
    "float": "a number",
}


class SequenceTagCsvInvalid(ValueError):
    """The CSV cannot be read as sequence tags. The message says what to change."""


class SequenceTagImportRefused(ValueError):
    """The dataset's tags are edited elsewhere, so a CSV import is refused."""


class SequenceTagTableUnavailable(ValueError):
    """The tag table cannot be read as one revision of one dataset."""


class SequenceTagsMoved(ValueError):
    """A revision of the tag table was written between the import's read and write."""


@dataclass(frozen=True, slots=True)
class SequenceTagCsv:
    """What a CSV of sequence tags holds.

    Attributes:
        key_form: How the rows name entries.
        columns: Each tag column of the file and its type, in the header's order.
        values: For every entry that a row names, the values of its non-empty
            cells by tag name. An entry that no row names is absent. A ``label``
            that is assigned maps to ``None``.
        file_name: The file's name.
        sha256: The SHA-256 of the file's bytes, in hex.
        matched_rows: The rows that named at least one entry.
    """

    key_form: KeyForm
    columns: tuple[TagDefinition, ...]
    values: Mapping[Entry, Mapping[str, TypedAttributeValue]]
    file_name: str
    sha256: str
    matched_rows: int


def parse_key_option(text: str) -> tuple[str, KeyForm]:
    """The column and form that ``NAME=FORM`` names.

    The form follows the last ``=``, so a column name may hold one.

    Raises:
        SequenceTagCsvInvalid: *text* is not ``NAME=FORM`` with a non-empty name
            and a form of ``entry``, ``sequence`` or ``recording``.
    """
    name, separator, form_text = text.rpartition("=")
    name = name.strip()
    if separator and name:
        for form in _KEY_FORMS:
            if form == form_text.strip():
                return name, form
    msg = f"{text!r} is not NAME=FORM, where FORM is one of {', '.join(_KEY_FORMS)}"
    raise SequenceTagCsvInvalid(msg)


def read_sequence_tag_csv(
    ds: Dataset,
    path: Path,
    *,
    key: tuple[str, KeyForm] | None,
    types: Mapping[str, TypedAttributeType],
    kept: Mapping[str, TypedAttributeType],
    findings: FindingLog,
) -> SequenceTagCsv:
    """Read the CSV at *path* as sequence tags of the entries of *ds*.

    Args:
        ds: The dataset whose entries the rows name.
        path: The CSV. It is decoded as UTF-8, with or without a byte order mark,
            and each cell is stripped of surrounding whitespace.
        key: The key column and how its cells read, or ``None`` when the header
            holds one of the key forms.
        types: The types the caller declares, by column name. Each names a column
            of the file other than the key.
        kept: The types the dataset already defines for sequence tags. A declared
            type takes precedence.
        findings: Receives one ``row_unmatched`` finding for each row that names
            no entry.

    Returns:
        The key form, the typed columns and each matched entry's values.

    Raises:
        SequenceTagCsvInvalid: The file cannot be read, its header names the
            entries two ways or none, a column is named after a reserved name, a
            declared type names no column, two rows name one entry, a column holds
            no value and has no declared type, or a value is not one its type
            reads.
    """
    try:
        data = path.read_bytes()
        text = data.decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        msg = f"cannot read {path}: {exc}"
        raise SequenceTagCsvInvalid(msg) from exc
    header, rows = _parse_rows(path.name, text)
    key_form, key_indexes = _key(header, key)
    tag_indexes = [index for index in range(len(header)) if index not in key_indexes]
    for index in tag_indexes:
        if header[index] in _RESERVED_COLUMNS:
            raise SequenceTagCsvInvalid(reserved_name_message(header[index]))
    tag_names = {header[index] for index in tag_indexes}
    for name in types:
        if name not in tag_names:
            reason = (
                "is the key column"
                if name in {header[index] for index in key_indexes}
                else "is not a column of the file"
            )
            msg = f"a type is declared for {name!r}, which {reason}"
            raise SequenceTagCsvInvalid(msg)
    resolve = _resolver(ds, key_form)
    matched: list[tuple[int, tuple[Entry, ...], list[str]]] = []
    owner: dict[Entry, int] = {}
    for line, cells in rows:
        key_cells = tuple(cells[index] for index in key_indexes)
        entries = resolve(key_cells, line)
        if not entries:
            findings.report(
                "row_unmatched",
                _unmatched_message(key_cells, line),
                path=path,
                line=line,
            )
            continue
        for entry in entries:
            if entry in owner:
                msg = (
                    f"lines {owner[entry]} and {line} name the same entry "
                    f"{entry_token(*entry)!r}"
                )
                raise SequenceTagCsvInvalid(msg)
            owner[entry] = line
        matched.append((line, entries, cells))
    columns: list[TagDefinition] = []
    values: dict[Entry, dict[str, TypedAttributeValue]] = {entry: {} for entry in owner}
    for index in tag_indexes:
        name = header[index]
        cells = [(line, row[index]) for line, _, row in matched if row[index]]
        declared = types.get(name)
        if declared is None and not cells:
            msg = (
                f"the column {name!r} has no value on a row that names an entry. "
                "Declare its type with --type to define it with no value"
            )
            raise SequenceTagCsvInvalid(msg)
        type_ = declared or kept.get(name) or _infer([text for _, text in cells])
        columns.append(TagDefinition(name, type_))
        by_line = {line: _read_cell(type_, name, line, text) for line, text in cells}
        for line, entries, row in matched:
            if row[index]:
                for entry in entries:
                    values[entry][name] = by_line[line]
    return SequenceTagCsv(
        key_form=key_form,
        columns=tuple(columns),
        values=values,
        file_name=path.name,
        sha256=hashlib.sha256(data).hexdigest(),
        matched_rows=len(matched),
    )


def _parse_rows(
    file_name: str, text: str
) -> tuple[list[str], list[tuple[int, list[str]]]]:
    """The header and each later row with the line it starts on, blank lines skipped."""
    reader = csv.reader(io.StringIO(text, newline=""))
    header: list[str] | None = None
    rows: list[tuple[int, list[str]]] = []
    previous = 0
    try:
        for record in reader:
            line = previous + 1
            previous = reader.line_num
            if not record:
                continue
            cells = [cell.strip() for cell in record]
            if header is None:
                header = cells
                _check_header(header, line)
                continue
            if len(cells) > len(header):
                if any(cells[len(header) :]):
                    msg = (
                        f"line {line} has {len(cells)} cells and the header "
                        f"{len(header)}"
                    )
                    raise SequenceTagCsvInvalid(msg)
                cells = cells[: len(header)]
            rows.append((line, cells + [""] * (len(header) - len(cells))))
    except csv.Error as exc:
        msg = f"{file_name} is not a CSV file: line {reader.line_num}: {exc}"
        raise SequenceTagCsvInvalid(msg) from exc
    if header is None:
        msg = f"{file_name} has no header"
        raise SequenceTagCsvInvalid(msg)
    return header, rows


def _check_header(header: Sequence[str], line: int) -> None:
    if not all(header):
        msg = f"line {line}: a column has no name"
        raise SequenceTagCsvInvalid(msg)
    for name in header:
        if header.count(name) > 1:
            msg = f"the column {name!r} appears more than once"
            raise SequenceTagCsvInvalid(msg)


def _forms_in(header: Sequence[str]) -> list[tuple[KeyForm, tuple[str, ...]]]:
    """The key forms that *header* holds, each with the columns that carry it."""
    found: list[tuple[KeyForm, tuple[str, ...]]] = []
    if "entry" in header:
        found.append(("entry", ("entry",)))
    if "sequence" in header:
        pair = ("group", "sequence") if "group" in header else ("sequence",)
        found.append(("sequence", pair))
    if "recording" in header:
        found.append(("recording", ("recording",)))
    return found


def _spelled(columns: tuple[str, ...]) -> str:
    return " and ".join(columns)


def _key(
    header: Sequence[str], key: tuple[str, KeyForm] | None
) -> tuple[KeyForm, tuple[int, ...]]:
    """The form and the indexes of the columns that name each row's entries."""
    present = _forms_in(header)
    if key is not None:
        name, form = key
        if present:
            held = ", ".join(_spelled(columns) for _, columns in present)
            msg = (
                f"--key names the column {name!r}, but the header also holds "
                f"{held}. Remove that column or the option"
            )
            raise SequenceTagCsvInvalid(msg)
        if name not in header:
            msg = f"--key names the column {name!r}, which the header lacks"
            raise SequenceTagCsvInvalid(msg)
        return form, (header.index(name),)
    if not present:
        msg = (
            "the header has no key column. Add one of entry, group and sequence, "
            "sequence or recording, or name the column with --key NAME=FORM"
        )
        raise SequenceTagCsvInvalid(msg)
    if len(present) > 1:
        held = ", ".join(_spelled(columns) for _, columns in present)
        msg = (
            f"the header names its entries in more than one way: {held}. Keep one, "
            "or name the key column with --key NAME=FORM"
        )
        raise SequenceTagCsvInvalid(msg)
    ((form, columns),) = present
    return form, tuple(header.index(column) for column in columns)


type _Resolve = Callable[[tuple[str, ...], int], tuple[Entry, ...]]


def _resolver(ds: Dataset, form: KeyForm) -> _Resolve:
    """A function from a row's key cells and line to the entries the row names."""
    universe = ds.recording_entries()
    if form == "recording":
        by_name = {
            recording.name: recording.members for recording in ds.recordings(universe)
        }
        return lambda cells, _line: by_name.get(cells[0], ())
    if form == "entry":

        def by_token(cells: tuple[str, ...], _line: int) -> tuple[Entry, ...]:
            if not cells[0]:
                return ()
            (entry,) = parse_entry_tokens([cells[0]])
            return (entry,) if entry in universe else ()

        return by_token
    groups: dict[str, list[str]] = {}
    for group, sequence in sorted(universe):
        groups.setdefault(sequence, []).append(group)

    def by_sequence(cells: tuple[str, ...], line: int) -> tuple[Entry, ...]:
        sequence = cells[-1]
        held = groups.get(sequence, [])
        if len(cells) == 2:
            return ((cells[0], sequence),) if cells[0] in held else ()
        if len(held) > 1:
            listed = ", ".join(repr(group) for group in held)
            msg = (
                f"line {line}: the sequence {sequence!r} names entries in more than "
                f"one group: {listed}. Name the group too"
            )
            raise SequenceTagCsvInvalid(msg)
        return ((held[0], sequence),) if held else ()

    return by_sequence


def _unmatched_message(key_cells: tuple[str, ...], line: int) -> str:
    if len(key_cells) == 2:
        named = entry_token(*key_cells) if key_cells[1] else ""
    else:
        named = key_cells[0]
    if not named:
        return f"line {line}: the row's key cell is empty, so it names no entry"
    return f"line {line}: the row names no entry: {named!r}"


def _numeric(text: str) -> list[int | float]:
    """The integer or float readings of *text*."""
    return [
        reading
        for reading in readings(text)
        if isinstance(reading, (int, float)) and not isinstance(reading, bool)
    ]


def _is_code(text: str) -> bool:
    """Whether *text* is a whole number written with a leading zero, as ``03``."""
    digits = text.lstrip("+-")
    return (
        len(digits) > 1
        and digits.startswith("0")
        and any(isinstance(reading, int) for reading in _numeric(text))
    )


def _infer(texts: Sequence[str]) -> TypedAttributeType:
    """The first type that every one of *texts* reads as."""
    if all(text in ("true", "false") for text in texts):
        return "bool"
    if not any(_is_code(text) for text in texts):
        if all(
            any(isinstance(reading, int) for reading in _numeric(text))
            for text in texts
        ):
            return "int"
        if all(_numeric(text) for text in texts):
            return "float"
    if len(set(texts)) <= OPTIONS_MAX_COUNT and all(
        len(text) <= OPTION_MAX_LENGTH for text in texts
    ):
        return "categorical"
    return "text"


def _read_cell(
    type_: TypedAttributeType, column: str, line: int, text: str
) -> TypedAttributeValue:
    """The value of *text* as a *type_*, or a refusal naming its line and column."""
    try:
        value = _read_value(type_, text)
        check_tag_value(type_, value)
    except ValueError as exc:
        msg = f"line {line}, column {column!r}: {exc}"
        raise SequenceTagCsvInvalid(msg) from exc
    return value


def _read_value(type_: TypedAttributeType, text: str) -> TypedAttributeValue:
    if type_ == "label":
        return None
    if type_ in ("categorical", "text"):
        return text
    if type_ == "bool":
        for reading in readings(text):
            if isinstance(reading, bool):
                return reading
    else:
        numbers = _numeric(text)
        if type_ == "int":
            for number in numbers:
                if isinstance(number, int):
                    return number
        elif numbers:
            return float(numbers[0])
    msg = f"{text!r} is not {_EXPECTED[type_]}"
    raise ValueError(msg)


def _latest_revision(ds: Dataset, *, kind: str) -> ResolvedRevision | None:
    """The revision of the tag table that a run reads now, or ``None`` for none.

    The latest indexed revision of any dataset, resolved as a run resolves it, so
    its digest is measured off disk.

    Raises:
        SequenceTagTableUnavailable: Several datasets' revisions are indexed, or
            the latest indexed revision is not on disk.
    """
    indexed = [
        record
        for record in index_records(read_label_series(ds, TAGS_SERIES))
        if record.get("key", "") == TAG_TABLE_KEY
    ]
    if not indexed:
        return None
    try:
        (found,) = resolve_series_refs(
            ds, TAGS_SERIES, (LabelRevisionRef(key=TAG_TABLE_KEY),), kind=kind
        )
    except SeveralOrigins as exc:
        msg = (
            f"the {exc.series} key {exc.key!r} is indexed from "
            f"{len(exc.origins)} datasets ({', '.join(exc.origins)}), so a run "
            "cannot tell which table to read. Keep the revisions of one dataset "
            "in labels_raw/tags"
        )
        raise SequenceTagTableUnavailable(msg) from exc
    except IdentityDeferred as exc:
        raise SequenceTagTableUnavailable(exc.because) from exc
    return found


def _position(found: ResolvedRevision | None) -> tuple[str, int] | None:
    return None if found is None else (found.origin_uuid, found.revision)


def _named(position: tuple[str, int] | None) -> str:
    if position is None:
        return "no revision"
    origin, revision = position
    return f"revision {revision} of dataset {origin!r}"


def import_sequence_tags(
    ds: Dataset,
    path: Path,
    *,
    key: tuple[str, KeyForm] | None = None,
    types: Mapping[str, TypedAttributeType] | None = None,
    replace: bool = False,
    dry_run: bool = False,
) -> SequenceTagImport:
    """Write the CSV at *path* as the next revision of the dataset's tag table.

    The next revision lists the entries of the latest revision and every recording
    entry of *ds*. A tag that the CSV names takes the CSV's values on every entry,
    so an entry that no row names holds none of them. A tag that it does not name
    keeps its values, unless *replace* removes it. The individual and arena rows
    are kept as they are. A CSV that changes nothing writes nothing.

    Args:
        ds: The dataset whose entries the rows name.
        path: The CSV, as :func:`read_sequence_tag_csv` reads it.
        key: The key column and how its cells read, or ``None`` when the header
            holds one of the key forms.
        types: The types to declare, by column name. A tag the table defines keeps
            its type without one.
        replace: Remove the sequence tags that the CSV does not name.
        dry_run: Report the changes and write nothing. A dry run takes no lock and
            creates no file.

    Returns:
        The revision, the changes to each tag and the rows that named no entry.

    Raises:
        SequenceTagImportRefused: The dataset's manifest is managed, the latest
            revision names a ``project_id`` or was saved by another dataset, or
            several datasets' revisions are indexed, so its tags are edited
            elsewhere or a second table would result.
        SequenceTagCsvInvalid: The CSV cannot be read. Nothing is written.
        SequenceTagsMoved: A revision was written after the one this import read.
            Nothing is written, so run the import again.
        SequenceTagTableUnavailable: Under the lock, a second dataset's revision is
            indexed or the latest revision is not on disk. Nothing is written. The
            same states at the first read raise ``SequenceTagImportRefused``.
    """
    if ds.manifest.is_managed:
        msg = (
            "the dataset is managed by a service, whose next save replaces the tag "
            "table. Edit its tags there"
        )
        raise SequenceTagImportRefused(msg)
    try:
        started = _latest_revision(ds, kind="sequence-tags import")
    except SequenceTagTableUnavailable as exc:
        raise SequenceTagImportRefused(str(exc)) from exc
    if started is not None:
        if started.origin_uuid != (ds.uuid or ""):
            msg = (
                f"revision {started.revision} of the tag table was saved by another "
                f"dataset ({started.origin_uuid!r}). Importing here would start a "
                "second table. Import into that dataset"
            )
            raise SequenceTagImportRefused(msg)
        if "project_id" in read_revision_manifest(started.payload).origin:
            msg = (
                f"revision {started.revision} of the tag table was saved from a "
                "project (its origin names a project_id), and the project's next "
                "save replaces the table. Edit its tags there"
            )
            raise SequenceTagImportRefused(msg)
    old = read_tag_table(started.payload) if started is not None else None
    findings = FindingLog(kind="labels", source=path.name)
    csv_tags = read_sequence_tag_csv(
        ds,
        path,
        key=key,
        types=types or {},
        kept={
            definition.name: definition.type
            for definition in (old.definitions.sequence if old is not None else ())
        },
        findings=findings,
    )
    table = _next_table(old, csv_tags, ds.recording_entries(), replace=replace)
    started_position = _position(started)
    revision = None if started is None else started.revision
    written = False
    with index_lock_unless_dry_run(series_index_path(ds, TAGS_SERIES), dry_run=dry_run):
        if not dry_run:
            found = _position(_latest_revision(ds, kind="sequence-tags import"))
            if found != started_position:
                msg = (
                    f"the tag table was at {_named(started_position)} when the "
                    f"import read it and is at {_named(found)} now. Run the import "
                    "again"
                )
                raise SequenceTagsMoved(msg)
            saved = write_tag_table_revision(
                ds,
                table=table,
                origin={
                    "system": "mosaic-csv",
                    "file": csv_tags.file_name,
                    "sha256": csv_tags.sha256,
                },
                origin_ref=f"csv:{csv_tags.sha256[:10]}",
            )
            revision = saved.revision
            written = saved.written
    return SequenceTagImport(
        revision=revision,
        written=written,
        dry_run=dry_run,
        rows_matched=csv_tags.matched_rows,
        changes=_changes(old, table),
        findings=tuple(findings.findings),
    )


def _next_table(
    old: TagTable | None,
    csv_tags: SequenceTagCsv,
    recording_entries: set[Entry],
    *,
    replace: bool,
) -> TagTable:
    """The table that follows *old* once *csv_tags* is imported."""
    named = {column.name for column in csv_tags.columns}
    kept = [
        definition
        for definition in (old.definitions.sequence if old is not None else ())
        if definition.name not in named and not replace
    ]
    definitions = TagDefinitions(
        sequence=tuple(sorted((*kept, *csv_tags.columns), key=lambda d: d.name)),
        individual=old.definitions.individual if old is not None else (),
    )
    before = {
        (sequence.group, sequence.sequence): sequence
        for sequence in (old.sequences if old is not None else ())
    }
    kept_names = {definition.name for definition in kept}
    sequences: list[SequenceTags] = []
    for entry in sorted(recording_entries | set(before)):
        previous = before.get(entry)
        tags: dict[str, TypedAttributeValue] = {}
        if previous is not None:
            tags.update(
                (name, value)
                for name, value in previous.tags.items()
                if name in kept_names
            )
        tags.update(csv_tags.values.get(entry, {}))
        sequences.append(
            SequenceTags(
                group=entry[0],
                sequence=entry[1],
                tags=tags,
                individuals=previous.individuals if previous is not None else {},
                arenas=previous.arenas if previous is not None else (),
            )
        )
    return TagTable(definitions=definitions, sequences=tuple(sequences))


def _held(table: TagTable | None) -> dict[str, dict[Entry, TypedAttributeValue]]:
    """Each sequence tag's values by entry, for the entries that hold one."""
    held: dict[str, dict[Entry, TypedAttributeValue]] = {}
    if table is None:
        return held
    for definition in table.definitions.sequence:
        held[definition.name] = {}
    for sequence in table.sequences:
        for name, value in sequence.tags.items():
            held[name][(sequence.group, sequence.sequence)] = value
    return held


def _types_of(table: TagTable | None) -> dict[str, TypedAttributeType]:
    if table is None:
        return {}
    return {
        definition.name: definition.type for definition in table.definitions.sequence
    }


def _changes(old: TagTable | None, new: TagTable) -> tuple[SequenceTagChange, ...]:
    """What going from *old* to *new* does to each sequence tag."""
    old_types, new_types = _types_of(old), _types_of(new)
    old_held, new_held = _held(old), _held(new)
    changes: list[SequenceTagChange] = []
    for name in sorted(old_types.keys() | new_types.keys()):
        before, after = old_held.get(name, {}), new_held.get(name, {})
        retyped = (
            name in old_types
            and name in new_types
            and old_types[name] != new_types[name]
        )
        entries_set = sum(
            1
            for entry, value in after.items()
            if retyped or entry not in before or before[entry] != value
        )
        entries_cleared = sum(1 for entry in before if entry not in after)
        change: Literal["added", "changed", "unchanged", "removed"]
        if name not in old_types:
            change = "added"
        elif name not in new_types:
            change = "removed"
        elif retyped or entries_set or entries_cleared:
            change = "changed"
        else:
            change = "unchanged"
        changes.append(
            SequenceTagChange(
                name=name,
                type=new_types.get(name) or old_types[name],
                change=change,
                entries_set=entries_set,
                entries_cleared=entries_cleared,
            )
        )
    return tuple(changes)


def list_sequence_tags(ds: Dataset) -> SequenceTagListing:
    """Each sequence tag of the latest revision, with its values and their counts.

    Reads only: no lock is taken and no file or directory is created. A dataset
    without a revision lists nothing.

    The revision listed is the one a run reads: the latest indexed.

    Raises:
        SequenceTagTableUnavailable: Several datasets' revisions are indexed, or
            the latest indexed revision is not on disk.
        ValueError: The latest revision's payload is not a valid tag table.
    """
    latest = _latest_revision(ds, kind="sequence-tags list")
    if latest is None:
        return SequenceTagListing(revision=None, digest=None, entries=0, tags=())
    table = read_tag_table(latest.payload)
    held = _held(table)
    listed: list[SequenceTagListed] = []
    for definition in sorted(table.definitions.sequence, key=lambda d: d.name):
        counts: Counter[str | int | float | bool] = Counter()
        for value in held[definition.name].values():
            if definition.type == "label":
                counts[True] += 1
            elif value is not None:
                counts[value] += 1
        listed.append(
            SequenceTagListed(
                name=definition.name,
                type=definition.type,
                entries=len(held[definition.name]),
                values=tuple(
                    SequenceTagValueCount(value=value, entries=entries)
                    for value, entries in sorted(
                        counts.items(), key=lambda item: _value_order(item[0])
                    )
                ),
            )
        )
    return SequenceTagListing(
        revision=latest.revision,
        digest=latest.digest,
        entries=len(table.sequences),
        tags=tuple(listed),
    )


def _value_order(value: str | int | float | bool) -> tuple[str, float]:
    """A sort key for the values of one tag, which are all strings or all numbers."""
    if isinstance(value, str):
        return (value, 0.0)
    return ("", float(value))
