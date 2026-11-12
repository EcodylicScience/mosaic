"""What a refusal to read or summarize one feature run says beyond its message.

:class:`RunError` carries the feature, the run and the entries or columns that a
message lists, and returns them as the error's detail. The classified errors of
:mod:`~mosaic.core.pipeline.run_matrix` and
:mod:`~mosaic.core.pipeline.run_summary` subclass it beside
:class:`~mosaic.core.pipeline.error_kind.ClassifiedError`.

The module is a leaf. It imports the standard library only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mosaic.core.entry import Entry
    from mosaic.core.json_value import JsonValue

__all__ = ["RunError", "entry_label"]


def entry_label(entry: Entry) -> str:
    """Return the ``group/sequence`` that names an entry in a message."""
    return f"{entry[0]}/{entry[1]}"


class RunError(Exception):
    """A refusal about one feature run, with the names its message lists.

    Attributes:
        feature: The storage name of the feature whose run was asked for.
        run_id: The run, or ``None`` where the request had not chosen one.
        entries: The entries that the message lists.
        columns: The columns that the message lists.
    """

    def __init__(
        self,
        message: str,
        *,
        feature: str,
        run_id: str | None,
        entries: Sequence[Entry] = (),
        columns: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.feature: str = feature
        self.run_id: str | None = run_id
        self.entries: tuple[Entry, ...] = tuple(entries)
        self.columns: tuple[str, ...] = tuple(columns)

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the feature and run, and the entries and columns the message lists."""
        detail: dict[str, JsonValue] = {"feature": self.feature, "run_id": self.run_id}
        if self.entries:
            detail["entries"] = [entry_label(entry) for entry in self.entries]
        if self.columns:
            detail["columns"] = list(self.columns)
        return detail
