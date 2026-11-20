"""Saved selections: a named entry set that does not grow.

A read or plan with no scope covers the dataset's entries at the time it runs, so
an earlier recipe computes something else once entries are added. A selection
names the entries it was saved with. It is a file at
``<dataset>/.mosaic/selections/<name>.json`` and is frozen: a save refuses a name
that exists, and replacing a selection is a delete and a save.

A save and a delete run under :func:`~mosaic.core.pipeline.index_lock.index_lock`
on ``.guard`` in that directory, so two writers neither pick one name nor
disagree about which names exist. A reader takes no lock, because a save replaces
the file by rename. A read-only open of a dataset lists and reads without creating
the directory.

The module imports the standard library, pydantic, ``entry``, ``scope``,
``strict_model``, ``error_kind`` and ``index_lock`` at module scope, and loads no
pandas. The entry universe and ``atomic_write`` are imported inside the functions
that need them, because their modules load pandas.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, Literal

from pydantic import ConfigDict, ValidationError

from mosaic.core.entry import Entry, validate_entry_name
from mosaic.core.pipeline.error_kind import ClassifiedError, ErrorKind
from mosaic.core.pipeline.index_lock import index_lock
from mosaic.core.scope import Scope
from mosaic.core.strict_model import StrictModel, located

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.json_value import JsonValue

__all__ = [
    "SELECTION_FORMAT",
    "Selection",
    "SelectionExists",
    "SelectionNotFound",
    "SelectionSummary",
    "SelectionUnreadable",
    "UnreadableSelection",
    "delete_selection",
    "list_selections",
    "read_selection",
    "save_selection",
    "selection_scope",
    "selections_directory",
    "unreadable_selections",
    "validate_selection_name",
]

type SelectionFormat = Literal["mosaic-selection/1"]

SELECTION_FORMAT: Final[SelectionFormat] = "mosaic-selection/1"
"""The ``format`` key of a selection file."""

MAX_NAME_BYTES: Final = 128
"""The longest selection name, in UTF-8 bytes."""

_GUARD: Final = ".guard"
_SUFFIX: Final = ".json"
_SHOWN_MISSING: Final = 5


class SelectionSummary(StrictModel):
    """One saved selection, without its entries.

    Attributes:
        name: The name it is saved under.
        n_entries: How many entries it holds.
        created_at: When it was saved, as UTC ISO 8601 text.
        note: Free text kept with it, empty when none was given.
    """

    name: str
    n_entries: int
    created_at: str
    note: str


class Selection(StrictModel):
    """A saved selection, as its file holds it.

    Attributes:
        format: The file format, :data:`SELECTION_FORMAT`.
        name: The name the selection is saved under.
        created_at: When it was saved, as UTC ISO 8601 text ending in ``Z``.
        note: Free text about the selection, empty when none was given.
        entries: The ``(group, sequence)`` pairs, sorted and unique.
        digest: ``sha256:`` and the hex digest of the canonical JSON of
            *entries*. Two selections of the same entries share it.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    format: SelectionFormat = SELECTION_FORMAT
    name: str
    created_at: str
    note: str = ""
    entries: list[Entry]
    digest: str

    def summary(self) -> SelectionSummary:
        """Return the selection without its entries."""
        return SelectionSummary(
            name=self.name,
            n_entries=len(self.entries),
            created_at=self.created_at,
            note=self.note,
        )


class SelectionNotFound(ClassifiedError, KeyError):
    """A selection that the dataset does not hold."""

    error_kind: ClassVar[ErrorKind] = "not_found"

    def __init__(self, name: str, present: list[str]) -> None:
        held = ", ".join(present) if present else "none"
        super().__init__(f"No selection named {name!r}. Selections present: {held}.")
        self.name: str = name
        self.present: list[str] = present

    def __str__(self) -> str:
        """Return the message without the quoting that ``KeyError`` adds."""
        return str(self.args[0])

    def error_reason(self) -> str:
        """Return ``selection_not_found``."""
        return "selection_not_found"

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the name asked for and the names the dataset holds."""
        held: list[JsonValue] = list(self.present)
        return {"name": self.name, "selections": held}


class SelectionExists(ClassifiedError, ValueError):
    """A save under a name that an existing selection has, ignoring case."""

    error_kind: ClassVar[ErrorKind] = "conflict"

    def __init__(self, name: str, existing: str) -> None:
        if existing == name:
            message = (
                f"A selection named {name!r} exists. A selection is frozen: "
                "delete it, then save it again."
            )
        else:
            message = (
                f"The selection {existing!r} has the name {name!r} ignoring "
                "case, and two such names share a file on a case-insensitive "
                "filesystem. Choose another name."
            )
        super().__init__(message)
        self.name: str = name
        self.existing: str = existing

    def error_reason(self) -> str:
        """Return ``selection_exists``."""
        return "selection_exists"

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the name asked for and the name that holds it."""
        return {"name": self.name, "existing": self.existing}


class SelectionUnreadable(ClassifiedError, ValueError):
    """A selection file that exists and cannot be read as the selection it names.

    A file that does not parse, one of another format version, and one whose stored
    name is not its file name, as a sync-conflict copy is.
    """

    error_kind: ClassVar[ErrorKind] = "invalid"

    def __init__(self, file: str, reason: str) -> None:
        super().__init__(f"The selection file {file} cannot be read: {reason}")
        self.file: str = file
        self.reason: str = reason

    def error_reason(self) -> str:
        """Return ``selection_unreadable``."""
        return "selection_unreadable"

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the file name and why it cannot be read."""
        return {"file": self.file, "reason": self.reason}


class UnreadableSelection(StrictModel):
    """A file of the selections directory that holds no readable selection.

    Attributes:
        file: The file's name in the directory.
        reason: Why it cannot be read.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    file: str
    reason: str


def validate_selection_name(name: str) -> str:
    """Return *name* when it can name a selection.

    The rules of :func:`~mosaic.core.entry.validate_entry_name`, and a selection
    name is also not empty, does not start with ``.``, holds no whitespace, ``:``
    or ``@``, and is at most :data:`MAX_NAME_BYTES` bytes. ``:`` and ``@`` keep a
    name distinct from an entry token and from the ``@name`` form that refers to
    a selection.

    Raises:
        ValueError: *name* breaks one of the rules, and the message names it.
    """
    if not name:
        raise ValueError("A selection name may not be empty.")
    _ = validate_entry_name(name, "A selection name")
    if name.startswith("."):
        raise ValueError(f"A selection name may not start with '.': {name!r}.")
    if any(char.isspace() for char in name):
        raise ValueError(f"A selection name may not contain whitespace: {name!r}.")
    for char in (":", "@"):
        if char in name:
            raise ValueError(f"A selection name may not contain {char!r}: {name!r}.")
    size = len(name.encode("utf-8"))
    if size > MAX_NAME_BYTES:
        raise ValueError(
            f"A selection name is at most {MAX_NAME_BYTES} bytes, and this one "
            f"is {size}."
        )
    return name


def selections_directory(ds: Dataset) -> Path:
    """Return the directory that holds the selection files of *ds*.

    The directory is not created, and a dataset that has saved nothing lacks it.
    """
    return ds.base_dir / ".mosaic" / "selections"


def _digest(entries: list[Entry]) -> str:
    canonical = json.dumps(
        [list(entry) for entry in entries], sort_keys=True, separators=(",", ":")
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _present(directory: Path) -> list[Path]:
    """The selection files of *directory*, by name, and none where it is absent."""
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.glob(f"*{_SUFFIX}") if path.is_file())


def _names(directory: Path) -> list[str]:
    return [path.name.removesuffix(_SUFFIX) for path in _present(directory)]


def _is_valid(name: str) -> bool:
    try:
        _ = validate_selection_name(name)
    except ValueError:
        return False
    return True


def _not_found(directory: Path, name: str) -> SelectionNotFound:
    """The error for *name*, naming the readable selections of *directory*."""
    return SelectionNotFound(
        name, [selection.name for selection in _scan(directory)[0]]
    )


def _load(path: Path) -> Selection:
    """Read the selection in *path*, which a file named for its selection holds.

    Raises:
        FileNotFoundError: *path* does not exist.
        SelectionUnreadable: *path* does not parse as a selection, or the name it
            holds is not its file name.
    """
    try:
        selection = Selection.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (ValidationError, UnicodeDecodeError, OSError) as exc:
        raise SelectionUnreadable(path.name, located(exc)) from exc
    stem = path.name.removesuffix(_SUFFIX)
    if selection.name != stem:
        raise SelectionUnreadable(
            path.name,
            f"it holds the selection {selection.name!r}, not {stem!r}",
        )
    return selection


def _scan(directory: Path) -> tuple[list[Selection], list[UnreadableSelection]]:
    """Read every file of *directory*, and split the readable from the others."""
    readable: list[Selection] = []
    unreadable: list[UnreadableSelection] = []
    for path in _present(directory):
        try:
            readable.append(_load(path))
        except FileNotFoundError:
            continue
        except SelectionUnreadable as exc:
            unreadable.append(UnreadableSelection(file=exc.file, reason=exc.reason))
    return readable, unreadable


def save_selection(
    ds: Dataset, name: str, entries: Iterable[Entry], note: str = ""
) -> Selection:
    """Save *entries* as the selection *name*, and return what was written.

    The entries are stored sorted and without repeats. The write is atomic and
    runs under the lock of the selections directory.

    Args:
        ds: The dataset to save into.
        name: The selection's name, per :func:`validate_selection_name`.
        entries: The ``(group, sequence)`` pairs to keep.
        note: Free text kept with the selection.

    Raises:
        ValueError: *name* is not a valid name, *entries* is empty, or an entry
            is not one the dataset holds. The last message names up to five.
        SelectionExists: A selection of that name exists, ignoring case.
    """
    from mosaic.core.pipeline._utils import atomic_write
    from mosaic.core.pipeline.inventory.scan import reportable_universe

    _ = validate_selection_name(name)
    chosen = sorted({(group, sequence) for group, sequence in entries})
    if not chosen:
        raise ValueError(f"Selection {name!r} would hold no entries.")
    universe = reportable_universe(ds)
    outside = [entry for entry in chosen if entry not in universe]
    if outside:
        shown = ", ".join(
            f"{group}:{sequence}" for group, sequence in outside[:_SHOWN_MISSING]
        )
        left = len(outside) - _SHOWN_MISSING
        more = f" and {left} more" if left > 0 else ""
        raise ValueError(
            f"{len(outside)} of the entries are not in the dataset: {shown}{more}."
        )
    directory = selections_directory(ds)
    with index_lock(directory / _GUARD):
        for existing in _names(directory):
            if existing.casefold() == name.casefold():
                raise SelectionExists(name, existing)
        selection = Selection(
            name=name,
            created_at=datetime.datetime.now(datetime.UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            note=note,
            entries=chosen,
            digest=_digest(chosen),
        )
        text = json.dumps(selection.model_dump(mode="json"), indent=2) + "\n"
        atomic_write(
            directory / f"{name}{_SUFFIX}",
            lambda temp: temp.write_text(text, encoding="utf-8"),
        )
    return selection


def read_selection(ds: Dataset, name: str) -> Selection:
    """Return the selection *name*.

    Raises:
        SelectionNotFound: The dataset holds no file for that name, or the name
            cannot name one. The error names the readable selections.
        SelectionUnreadable: The file exists and does not read as the selection
            *name*.
    """
    directory = selections_directory(ds)
    if not _is_valid(name):
        raise _not_found(directory, name)
    try:
        return _load(directory / f"{name}{_SUFFIX}")
    except FileNotFoundError:
        raise _not_found(directory, name) from None


def list_selections(ds: Dataset) -> list[Selection]:
    """Return every readable selection of the dataset, sorted by name.

    Only ``*.json`` files are read, so the guard file and its lock are skipped. A
    file that cannot be read is left out, and :func:`unreadable_selections` lists it.
    """
    return _scan(selections_directory(ds))[0]


def unreadable_selections(ds: Dataset) -> list[UnreadableSelection]:
    """Return the files of the selections directory that hold no readable selection.

    Sorted by file name. A file that does not parse, one of another format version
    and one whose stored name is not its file name are listed, each with its reason.
    """
    return _scan(selections_directory(ds))[1]


def delete_selection(ds: Dataset, name: str) -> None:
    """Delete the selection file *name*, under the lock of the selections directory.

    The file is removed whether or not it reads, which is how an unreadable
    selection is cleared.

    Raises:
        SelectionNotFound: The dataset holds no file for that name.
    """
    directory = selections_directory(ds)
    with index_lock(directory / _GUARD):
        if not _is_valid(name):
            raise _not_found(directory, name)
        try:
            (directory / f"{name}{_SUFFIX}").unlink()
        except FileNotFoundError:
            raise _not_found(directory, name) from None


def selection_scope(ds: Dataset, name: str) -> Scope:
    """Return the scope of the entries of the selection *name*.

    Raises:
        SelectionNotFound: The dataset holds no selection of that name.
    """
    return Scope(entries=read_selection(ds, name).entries)
