"""Declares the window file and reads it.

One window file names a source dataset, the tracks variant of that dataset to
cut from, and a list of windows. A window is a span of frames of one source
entry, and it is written as an entry of its own.

Imports pydantic, ``tomllib``, the standard library, :mod:`mosaic.core.entry`,
:mod:`mosaic.core.strict_model` and :mod:`mosaic.user_paths` at module level.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import Field, ValidationError, field_validator, model_validator

from mosaic.core.entry import validate_entry_name
from mosaic.core.strict_model import StrictModel, located
from mosaic.user_paths import user_path

__all__ = ["WINDOW_FORMAT", "WindowEntry", "WindowSpec", "read_window_spec"]

WINDOW_FORMAT = "mosaic-window/1"


class WindowEntry(StrictModel):
    """One window: the frames ``start <= frame < stop`` of one source entry.

    An absent bound leaves that side of the entry open. ``as_group`` and
    ``as_sequence`` name the written entry, and :meth:`output_entry` fills the
    ones left out.
    """

    group: str = ""
    sequence: str
    start: int | None = None
    stop: int | None = None
    as_group: str | None = None
    as_sequence: str | None = None

    def output_entry(self) -> tuple[str, str]:
        """The ``(group, sequence)`` this window is written under.

        The default group is the source group. The default sequence is the source
        sequence for a window without bounds, and otherwise
        ``<sequence>_f<start>-<stop>``, with an absent bound written as ``first``
        or ``end``.
        """
        group = self.group if self.as_group is None else self.as_group
        if self.as_sequence is not None:
            return group, self.as_sequence
        if self.start is None and self.stop is None:
            return group, self.sequence
        start = "first" if self.start is None else str(self.start)
        stop = "end" if self.stop is None else str(self.stop)
        return group, f"{self.sequence}_f{start}-{stop}"

    @model_validator(mode="after")
    def _check_window(self) -> WindowEntry:
        if self.start is not None and self.stop is not None and self.start >= self.stop:
            msg = (
                f"start must be below stop, got start {self.start} and stop {self.stop}"
            )
            raise ValueError(msg)
        group, sequence = self.output_entry()
        _ = validate_entry_name(
            group, "as_group" if self.as_group is not None else "the default as_group"
        )
        _ = validate_entry_name(
            sequence,
            "as_sequence"
            if self.as_sequence is not None
            else "the default as_sequence",
        )
        return self


class WindowSpec(StrictModel):
    """The window file: a source dataset, its tracks variant and the windows."""

    format: str
    source: Path
    tracks_run_id: str = Field(min_length=1)
    windows: list[WindowEntry] = Field(min_length=1)

    @field_validator("format")
    @classmethod
    def _check_format(cls, value: str) -> str:
        if value != WINDOW_FORMAT:
            msg = f"format must be {WINDOW_FORMAT!r}, got {value!r}"
            raise ValueError(msg)
        return value

    @field_validator("source", mode="before")
    @classmethod
    def _check_source(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            msg = "source must name a manifest or its directory"
            raise ValueError(msg)
        return value

    @field_validator("windows")
    @classmethod
    def _check_outputs_differ(cls, value: list[WindowEntry]) -> list[WindowEntry]:
        first_writer: dict[tuple[str, str], int] = {}
        for index, entry in enumerate(value):
            output = entry.output_entry()
            earlier = first_writer.setdefault(output, index)
            if earlier != index:
                msg = (
                    f"windows[{index}] and windows[{earlier}] both write the entry "
                    f"group={output[0]!r} sequence={output[1]!r}. Give one of them "
                    "an as_sequence or an as_group"
                )
                raise ValueError(msg)
        return value


def read_window_spec(path: Path) -> WindowSpec:
    """Read and validate the window file at *path*.

    A leading ``~`` of ``source`` expands to the home directory, and a relative
    ``source`` then resolves against the directory of *path*. The source is not
    opened.

    Raises:
        ValueError: The file is not valid TOML, or a model refuses it. The message
            names *path* and each refused field.
    """
    with path.open("rb") as handle:
        try:
            document = tomllib.load(handle)
        except tomllib.TOMLDecodeError as error:
            msg = f"{path} is not valid TOML: {error}"
            raise ValueError(msg) from error
    try:
        spec = WindowSpec.model_validate(document)
    except ValidationError as error:
        msg = f"{path} is not a window file: {located(error)}"
        raise ValueError(msg) from error
    return spec.model_copy(update={"source": path.parent / user_path(spec.source)})
