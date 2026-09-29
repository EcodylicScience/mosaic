"""Resolve the media variant file that an entry is read from when ``media`` names one.

A consumer that names a variant reads the variant's file for each entry instead of
the entry's own media, and publishes its results in the entry's source space.
:meth:`VariantLookup.resolve` returns, for one entry, the file to read, its
recorded facts, and its placement in the entry media. The placement maps a table
tracked on the file back to source space. A consumer reads one lookup for its
whole scope and resolves each entry against it.

The lookup reads the variant index row and does not probe the file. The row
stores the file's probed facts and its placement. A reader is handed the facts,
and map-back reads one row. The row is refused when the entry lacks one, when its
file is gone, and when the entry's media is no longer the media that the file was
written from.

The trackers, the inference ops and a chained ``preprocess`` run all resolve an
entry's variant here, and apply one rule for a missing or drifted variant.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

from mosaic_media import MediaFacts

from mosaic.core.helpers import make_entry_key
from mosaic.core.media.preprocess.geometry import Placement
from mosaic.core.media.timeline import concatenated_timeline
from mosaic.core.pipeline.composition import compositions_disagree
from mosaic.core.pipeline.placement import SourceMapping
from mosaic.core.pipeline.preprocess_index import (
    MediaVariantDriftedError,
    MediaVariantMissingError,
    media_variant_facts,
    media_variant_placement,
    media_variant_rows,
)
from mosaic.core.pipeline.preprocess_layout import media_variant_recipe_path
from mosaic.core.pipeline.sequence_index import media_compositions_for

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset, ResolvedScopeEntry
    from mosaic.core.entry import CameraEntry, Entry

__all__ = [
    "VariantLookup",
    "VariantSource",
    "preprocess_command",
    "unreadable_variant_refusal",
]


@dataclass(frozen=True, slots=True)
class VariantSource:
    """Record one entry's media variant file and its placement in the entry media.

    Attributes:
        run_id: The variant's run identifier, the value that ``media`` named.
        path: The variant file.
        facts: The file's probed facts, as its index row stores them.
        placement: The placement of the file's pixels and frames in the entry
            media.
        entry_paths: The entry media's clips, in ``video_order``.
        entry_facts: The entry media's probed facts, parallel to *entry_paths*.
    """

    run_id: str
    path: Path
    facts: MediaFacts
    placement: Placement
    entry_paths: tuple[Path, ...]
    entry_facts: tuple[MediaFacts, ...]

    def __post_init__(self) -> None:
        """Refuse entry media that the placement does not map into.

        Raises:
            ValueError: If the entry media is not one frame axis that the
                placement maps into.
        """
        _ = self.mapping()

    def mapping(self) -> SourceMapping:
        """Return the file's placement in the entry media, with that media's timing."""
        return SourceMapping(self.placement, concatenated_timeline(self.entry_facts))

    @property
    def consumed_media(self) -> tuple[Path, ...]:
        """Every media file that a table tracked on the variant derives from.

        They are the variant file and the entry media that it was made from. A
        table's consumed roots then name the entry media's root, and a change to
        that media is visible from the table.
        """
        return (self.path, *self.entry_paths)


def unreadable_variant_refusal(
    ds: Dataset,
    kind: str,
    media: str,
    run_id: str,
    *,
    lost: AbstractSet[str],
    unresolved: Mapping[str, Entry],
) -> str | None:
    """Return the refusal for a run of *kind* that lost every entry it attempted.

    A refusal is returned when every lost entry is one whose file of *media* could not
    be read. The tool did not run for any of them and did not write output to keep. The
    refusal names the variant and the command that writes it. The function returns
    ``None`` when the run did not lose an entry, and when a tool ran for some lost
    entry. The caller's refusal then locates that entry's tool output. An entry held by
    another execution was not attempted and is in neither set.

    The tracker driver and the inference ops both call this function, and refuse
    the same runs in the same words.

    Args:
        ds: The dataset, whose record of *media*'s recipe the command names.
        kind: The op kind of the run.
        media: The media variant that the run names.
        run_id: The run's identifier.
        lost: The keys of every entry that the run attempted and lost.
        unresolved: The entries whose file of *media* could not be read, by key.
    """
    if not lost or not set(lost) <= set(unresolved):
        return None
    entries = sorted(unresolved[key] for key in lost)
    keys = ", ".join(make_entry_key(group, sequence) for group, sequence in entries)
    return (
        f"[{kind}] run_id={run_id} did not produce tracks, because every entry "
        f"that it attempted lacks a readable file of the media variant {media}: "
        f"{keys}. Run the preprocess step that made {media} over these entries:\n"
        f"{preprocess_command(ds, media, entries)}"
    )


def preprocess_command(ds: Dataset, run_id: str, entries: Iterable[Entry]) -> str:
    """Return the command that writes variant *run_id* for *entries*, indented.

    ``--params`` names the recipe that the variant's runs recorded, by absolute
    path, because ``@file`` is read relative to the working directory. A variant
    without a recorded recipe is named instead, with a second line that asks for
    the recipe.
    """
    selected = " ".join(
        f"--entries {shlex.quote(f'{group}:{sequence}')}"
        for group, sequence in sorted(set(entries))
    )
    command = f"    mosaic run -m <manifest> --kind preprocess {selected} --params"
    recipe = media_variant_recipe_path(ds, run_id).absolute()
    if recipe.is_file():
        return f"{command} {shlex.quote(f'@{recipe}')}"
    return (
        f"{command} '<the parameters of {run_id}>'\n"
        f"    {run_id} does not record a recipe at {recipe}. Give the parameters "
        f"that made it."
    )


@dataclass(frozen=True, slots=True)
class VariantLookup:
    """One variant's index rows and its entries' current media compositions.

    Resolving the variant for an entry reads the variant index and the sequence
    projection besides the entry itself. A caller that resolves it for many entries
    reads both once for the scope, with :meth:`read`, and resolves each entry
    against them.

    Attributes:
        run_id: The variant, as a ``media`` parameter names it.
        rows: The variant's rows, keyed by entry and camera.
        compositions: The current media composition of each entry looked up.
    """

    run_id: str
    rows: Mapping[CameraEntry, Mapping[str, str]]
    compositions: Mapping[Entry, str]

    @classmethod
    def read(cls, ds: Dataset, run_id: str, entries: Iterable[Entry]) -> Self:
        """Read variant *run_id*'s rows and the current compositions of *entries*."""
        return cls(
            run_id=run_id,
            rows=media_variant_rows(ds, run_id),
            compositions=media_compositions_for(ds, entries),
        )

    def resolve(self, ds: Dataset, entry: ResolvedScopeEntry) -> VariantSource:
        """Return the variant's file for *entry*, checked against its current media.

        *entry* is the entry's media as ``Dataset.resolve_media_scope`` resolved
        it, reduced to the camera that a consumer reads. The variant row is looked up
        for that camera. An entry missing from :attr:`compositions` has an
        unknown composition, which is not drift.

        Args:
            ds: The dataset.
            entry: The entry and camera, with the media it resolves to now.

        Returns:
            The variant file, its stored facts and placement, and the entry media
            that the placement maps into.

        Raises:
            MediaVariantMissingError: If the variant lacks a row for the entry and
                camera, or the row's file is gone.
            MediaVariantDriftedError: If the entry's media changed after the file
                was written from it, or no longer has the frames that the file's
                placement maps into.
        """
        run_id = self.run_id
        group, sequence, camera = entry.group, entry.sequence, entry.camera
        key = make_entry_key(group, sequence)
        where = f"{key} (camera {camera})" if camera else key

        def command() -> str:
            return preprocess_command(ds, run_id, [(group, sequence)])

        row = self.rows.get((group, sequence, camera))
        if row is None:
            message = (
                f"{where}: the media variant {run_id} lacks a file for this entry. "
                f"Run the preprocess step that made {run_id} over this entry "
                f"first:\n{command()}"
            )
            raise MediaVariantMissingError(message)
        path = ds.resolve_path(row["abs_path"])
        if not path.is_file():
            message = (
                f"{where}: the file of the media variant {run_id} is missing at "
                f"{path}. Run the preprocess step that made {run_id} over this "
                f"entry again:\n{command()}"
            )
            raise MediaVariantMissingError(message)
        current = self.compositions.get((group, sequence), "")
        if compositions_disagree(row["consumed_media_composition"], current):
            message = (
                f"{where}: the entry's media changed after the media variant "
                f"{run_id} was written from it. The variant depicts media that the "
                f"entry no longer has. Run the preprocess step that made {run_id} "
                f"over this entry again, which rewrites a variant whose media "
                f"changed:\n{command()}"
            )
            raise MediaVariantDriftedError(message)
        try:
            return VariantSource(
                run_id=run_id,
                path=path,
                facts=media_variant_facts(row),
                placement=media_variant_placement(row),
                entry_paths=tuple(entry.resolved.paths),
                entry_facts=tuple(entry.resolved.facts),
            )
        except ValueError as exc:
            message = (
                f"{where}: the media variant {run_id} was made from media that the "
                f"entry no longer resolves to ({exc}). Run the preprocess step that "
                f"made {run_id} over this entry again, which rewrites a variant whose "
                f"placement no longer fits the entry's media:\n{command()}"
            )
            raise MediaVariantDriftedError(message) from exc
