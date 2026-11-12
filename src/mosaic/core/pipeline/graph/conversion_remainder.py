"""What a ``convert-tracks`` step has left to convert, read from the raw index.

``convert-tracks`` takes no scope, so a plan measuring it against the tables the
dataset holds would read a conversion that stopped part way as complete. Its
work is the raw rows of its format, so the plan measures it against them: each
entry that the rows name, and each raw file that names no sequence, is held when
a table of the step's variant answers for it.

A file that names no sequence is expanded into entries of its own that only
reading it can list. Until a table of the variant records it as its source, it
stands in the target as an entry of group ``""`` named by its path. Once one
does, the stand-in leaves the target and the file's own entries are the
variant's, so a file converted in part reads converted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.core.entry import Entry
from mosaic.core.pipeline.inventory.scan import existing_tracks_records
from mosaic.core.pipeline.tracks_index import read_tracks_index

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.convert_ops import ConvertTracksParams

__all__ = ["ConversionRemainder", "conversion_remainder", "stand_ins"]


@dataclass(frozen=True, slots=True)
class ConversionRemainder:
    """The work of one ``convert-tracks`` step, and what of it is not yet done.

    Attributes:
        target: Every entry that the raw rows of the step's format name, and a
            stand-in for each file that names no sequence and has no table yet.
        missing: The part of *target* that no table of the variant answers for.
    """

    target: frozenset[Entry]
    missing: frozenset[Entry]


def conversion_remainder(
    ds: Dataset, params: ConvertTracksParams, variant: str
) -> ConversionRemainder:
    """Measure the raw rows of *params*' format against the tables of *variant*.

    Reads the raw index and the tracks index and writes nothing.
    """
    sources = ds.track_conversion_sources(params.source_format)
    frame = read_tracks_index(ds)
    frame = frame.loc[
        frame["run_id"] == variant, ["group", "sequence", "abs_path", "source_abs_path"]
    ]
    records = existing_tracks_records(ds, frame)
    held_entries = {
        (record.get("group", ""), record.get("sequence", "")) for record in records
    }
    held_sources = {
        ds.resolve_path(record["source_abs_path"])
        for record in records
        if record.get("source_abs_path")
    }
    pending = [file for file in sources.files if file not in held_sources]
    unconverted = stand_ins(ds, pending)
    entries = sources.entries
    return ConversionRemainder(
        target=entries | unconverted,
        missing=(entries - held_entries) | unconverted,
    )


def stand_ins(ds: Dataset, files: list[Path]) -> frozenset[Entry]:
    """The stand-in entries of *files*: group ``""`` and the file as a label.

    The label is the path relative to the dataset root, or the file name for a
    file outside the dataset tree, so that no machine's absolute path reaches a
    plan. Two files that would share a label keep their full paths.
    """
    labels = {file: _label(ds, file) for file in files}
    counts = Counter(labels.values())
    return frozenset(
        ("", label if counts[label] == 1 else str(file))
        for file, label in labels.items()
    )


def _label(ds: Dataset, file: Path) -> str:
    """The path of *file* relative to the dataset root, else its name."""
    stored = ds.relative_to_root(file)
    return file.name if Path(stored).is_absolute() else stored
