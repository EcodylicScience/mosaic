"""Window files -> tables cut from another dataset's tracks, one format per schema.

One window file names a source dataset, the tracks variant of that dataset to
cut from, and a list of windows. A window is a span of frames of one source
entry, and it is written as one entry. The window converters are enumerable.
Each window of a file is one ``(group, sequence)``, and a tracks source declares
``multi_sequences_per_file=True`` to read it.

**A window keeps its source's frames and times.** Its rows are every source row
with ``start <= frame < stop``, in their stored order, with every column the
source table has. Only ``group`` and ``sequence`` are rewritten, to the window's
entry. The written row measures the window's extent.

**The source is read and not written.** It is opened without creating its roots,
and only its tracks index and the window's table are read. A window's conversion
facts are its file's. A window of a source reconverted under the same variant
stays current until ``convert-tracks --overwrite`` rewrites it.

**Each schema is its own format.** A converter declares one ``output_schema``
for all of its files (``trex.py``). ``window_mosaic_v1``, ``window_trex_v2``,
``window_mosaic_cm_v1`` and ``window_mosaic_world_v1`` each read a source of
their schema and refuse another, naming the format that reads it. No format reads
the legacy ``trex_v1``, which ``mosaic upgrade-tracks`` rescales into pixels.

**Tracklets are declared.** ``WindowParams.carries_tracklets`` is what a written
row records, and a conversion refuses a source whose row answers otherwise.

Imports pandas, pydantic, the standard library, ``mosaic.user_paths`` and the
:mod:`mosaic.core` modules ``entry``, ``params``, ``schema``, ``strict_model`` and
``track_converter`` at module level. The dataset and the tracks index are imported
inside :func:`read_window_table`. Importing the track library loads neither.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, override

import pandas as pd
from pydantic import Field, field_validator, model_validator

from mosaic.core.entry import validate_entry_name
from mosaic.core.params import Declared
from mosaic.core.schema import LEGACY_SCHEMA
from mosaic.core.strict_model import StrictModel, read_toml_model
from mosaic.core.track_converter import (
    TRACK_CONVERTERS,
    EntryHints,
    TrackConverter,
    TrackConvertParams,
    register_track_converter,
)
from mosaic.user_paths import user_path

__all__ = [
    "WINDOW_FORMAT",
    "TracksWindowConverter",
    "WindowEntry",
    "WindowMosaicCmV1Converter",
    "WindowMosaicV1Converter",
    "WindowMosaicWorldV1Converter",
    "WindowParams",
    "WindowSpec",
    "WindowTrexV2Converter",
    "read_window_spec",
    "read_window_table",
]

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
    spec = read_toml_model(path, WindowSpec, "a window file")
    return spec.model_copy(update={"source": path.parent / user_path(spec.source)})


_CARRIES_TRACKLETS_DESCRIPTION = (
    "Whether the source's ids follow animals, as its tracks index records. The "
    "written rows record it, and a window whose source records another answer is "
    "refused. Leave unset for a source that records none."
)


class WindowParams(TrackConvertParams):
    """The tracklets answer of a window conversion.

    The source, its variant and the windows are in the file.
    """

    carries_tracklets: Annotated[
        bool | None, Declared(_CARRIES_TRACKLETS_DESCRIPTION)
    ] = None


def _spelled(answer: bool | None) -> str:
    """The params spelling of a tracklets answer."""
    if answer is None:
        return "carries_tracklets unset"
    return f"carries_tracklets = {str(answer).lower()}"


def _window_format_for(schema: str) -> str | None:
    """The registered window format that reads a source of *schema*, if any."""
    for src_format, cls in TRACK_CONVERTERS.items():
        if issubclass(cls, TracksWindowConverter) and cls.output_schema == schema:
            return src_format
    return None


def read_window_table(
    spec: WindowSpec,
    window: WindowEntry,
    *,
    schema: str,
    carries_tracklets: bool | None,
) -> pd.DataFrame:
    """Read the source rows of *window*, as stored, from the source of *spec*.

    The source dataset is opened without creating its roots, and only its tracks
    index and the entry's table are read.

    Args:
        spec: The window file, its ``source`` resolved.
        window: One of its windows.
        schema: The schema the source row must record.
        carries_tracklets: The tracklets answer the source row must give.

    Returns:
        Every source row with ``start <= frame < stop``, in stored order, with
        every column of the source table.

    Raises:
        ValueError: The source entry has no row in the variant, its row records
            another schema or another tracklets answer, or no row falls in the
            window. The message names the source entry.
    """
    from mosaic.core.dataset import open_dataset
    from mosaic.core.pipeline.tracks_index import (
        read_carries_tracklets,
        variant_entry_rows,
    )

    source = open_dataset(spec.source, ensure_roots=False)
    named = (
        f"source entry group={window.group!r} sequence={window.sequence!r} of "
        f"{spec.source}"
    )
    rows = variant_entry_rows(
        source, spec.tracks_run_id, (window.group, window.sequence)
    )
    if rows.empty:
        msg = f"The {named} has no table in the tracks variant {spec.tracks_run_id!r}."
        raise ValueError(msg)
    row = rows.iloc[0]
    recorded = str(row["std_format"]).strip() or LEGACY_SCHEMA
    if recorded == LEGACY_SCHEMA:
        msg = (
            f"The table of the {named} records the legacy schema {LEGACY_SCHEMA}, "
            "in centimeters with X at the head, which no window format reads. Run "
            "`mosaic upgrade-tracks --apply` on the source dataset, then convert "
            "the window."
        )
        raise ValueError(msg)
    if recorded != schema:
        reader = _window_format_for(recorded)
        remedy = (
            f"Convert it as {reader}."
            if reader is not None
            else "No window format reads it."
        )
        msg = f"The table of the {named} records the schema {recorded}. {remedy}"
        raise ValueError(msg)
    answer = read_carries_tracklets(row)
    if answer != carries_tracklets:
        msg = (
            f"The tracks index of the {named} records {_spelled(answer)}, and the "
            f"conversion has {_spelled(carries_tracklets)}. Convert with "
            f"{_spelled(answer)}."
        )
        raise ValueError(msg)
    filters: list[tuple[str, str, int]] = []
    if window.start is not None:
        filters.append(("frame", ">=", window.start))
    if window.stop is not None:
        filters.append(("frame", "<", window.stop))
    table_path = source.resolve_path(str(row["abs_path"]))
    table = pd.read_parquet(table_path, filters=filters or None)
    if table.empty:
        start = "" if window.start is None else f"{window.start} <= "
        stop = "" if window.stop is None else f" < {window.stop}"
        msg = f"The {named} has no row with {start}frame{stop} in {table_path}."
        raise ValueError(msg)
    return table.reset_index(drop=True)


class TracksWindowConverter(TrackConverter[WindowParams]):
    """A window file -> one table per window, cut from a source of one schema."""

    Params: ClassVar[type[TrackConvertParams]] = WindowParams
    version = "0.1"
    enumerable = True
    carries_tracklets = None

    @override
    def tracklets(self, params: WindowParams) -> bool | None:
        return params.carries_tracklets

    @override
    def enumerate_sequences(self, path: Path) -> list[tuple[str, str]]:
        return [window.output_entry() for window in read_window_spec(path).windows]

    @override
    def convert(
        self, path: Path, params: WindowParams, hints: EntryHints
    ) -> pd.DataFrame:
        spec = read_window_spec(path)
        output = (hints.group, hints.sequence)
        window = next(
            (each for each in spec.windows if each.output_entry() == output), None
        )
        if window is None:
            msg = (
                f"{path} lists no window written as group={output[0]!r} "
                f"sequence={output[1]!r}."
            )
            if output[1] == path.stem:
                msg += (
                    " The file was indexed as one sequence named for its stem. Declare "
                    "its tracks source with --multi-sequences-per-file to make each "
                    "window an entry."
                )
            raise ValueError(msg)
        table = read_window_table(
            spec,
            window,
            schema=self.output_schema,
            carries_tracklets=params.carries_tracklets,
        )
        table["group"] = output[0]
        table["sequence"] = output[1]
        return table


@register_track_converter
class WindowMosaicV1Converter(TracksWindowConverter):
    """A window file of a ``mosaic_v1`` source -> ``mosaic_v1`` tables."""

    src_format = "window_mosaic_v1"
    output_schema = "mosaic_v1"


@register_track_converter
class WindowTrexV2Converter(TracksWindowConverter):
    """A window file of a ``trex_v2`` source -> ``trex_v2`` tables."""

    src_format = "window_trex_v2"
    output_schema = "trex_v2"


@register_track_converter
class WindowMosaicCmV1Converter(TracksWindowConverter):
    """A window file of a ``mosaic_cm_v1`` source -> ``mosaic_cm_v1`` tables."""

    src_format = "window_mosaic_cm_v1"
    output_schema = "mosaic_cm_v1"


@register_track_converter
class WindowMosaicWorldV1Converter(TracksWindowConverter):
    """A window file of a ``mosaic_world_v1`` source -> ``mosaic_world_v1`` tables."""

    src_format = "window_mosaic_world_v1"
    output_schema = "mosaic_world_v1"
