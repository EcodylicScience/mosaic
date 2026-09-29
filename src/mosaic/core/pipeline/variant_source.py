"""The media variant file an entry is read from, when a step names one in ``media``.

A consumer that names a variant reads the variant's file for each entry instead of
the entry's own media, and publishes what it finds in the entry's source space.
:func:`resolve_variant_source` answers both halves for one entry: which file to
read and what is known about it, and where that file sits in the entry media, so
a table tracked on it can be mapped back.

The answer comes from the variant index row, never from the file: the row stores
the file's probed facts, so a reader is handed them rather than probing again,
and its placement, so map-back reads one row. The row is refused when the entry
has none, when its file is gone, and when the entry's media is no longer the
media the file was written from.

The trackers, the inference ops and a chained ``preprocess`` run all resolve an
entry's variant here, so the three agree on what counts as missing or drifted.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mosaic_media import MediaFacts

from mosaic.core.helpers import make_entry_key
from mosaic.core.media.preprocess.geometry import Placement
from mosaic.core.media.timeline import concatenated_timeline
from mosaic.core.pipeline.composition import compositions_disagree
from mosaic.core.pipeline.placement import SourceMapping
from mosaic.core.pipeline.preprocess_index import (
    MediaVariantDriftedError,
    MediaVariantMissingError,
    variant_facts,
    variant_placement,
    variant_row,
)
from mosaic.core.pipeline.tracks_index import media_composition_for

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset, ResolvedScopeEntry

__all__ = ["VariantSource", "no_readable_variant_message", "resolve_variant_source"]


@dataclass(frozen=True, slots=True)
class VariantSource:
    """One entry's media variant file, and where it sits in the entry media.

    Attributes:
        run_id: The variant's run identifier, the value ``media`` named.
        path: The variant file.
        facts: The file's probed facts, as its index row stores them.
        placement: Where the file's pixels and frames sit in the entry media.
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
        """Refuse entry media the placement does not map into.

        Raises:
            ValueError: If the entry media is not one frame axis the placement
                maps into.
        """
        _ = self.mapping()

    def mapping(self) -> SourceMapping:
        """Where the file sits in the entry media, and how that media is timed."""
        return SourceMapping(self.placement, concatenated_timeline(self.entry_facts))

    @property
    def consumed_paths(self) -> tuple[Path, ...]:
        """Every media file a table tracked on the variant derives from.

        The variant file and the entry media it was made from. A table's
        consumed roots then name the entry media's root, so a change to that
        media reaches the table.
        """
        return (self.path, *self.entry_paths)


def no_readable_variant_message(
    kind: str, media: str, run_id: str, keys: Iterable[str]
) -> str:
    """Why a run of *kind* produced nothing: no entry's file of *media* is readable.

    Every entry was lost before a tool ran, so there is no tool output to keep.

    Args:
        kind: The op kind of the run.
        media: The media variant the run names.
        run_id: The run's identifier.
        keys: The entry keys lost.
    """
    return (
        f"[{kind}] no entry in scope has a readable file of the media variant "
        f"{media}, so run_id={run_id} produced no tracks: "
        f"{', '.join(sorted(keys))}. The per-entry errors are in this "
        f"attempt's run-log."
    )


def _preprocess_command(group: str, sequence: str, run_id: str) -> str:
    """The command that writes variant *run_id* for one entry."""
    return (
        f"    mosaic run -m <manifest> --kind preprocess "
        f"--entries \"{group}:{sequence}\" --params '<the recipe of {run_id}>'"
    )


def resolve_variant_source(
    ds: Dataset, run_id: str, entry: ResolvedScopeEntry
) -> VariantSource:
    """The file of variant *run_id* for *entry*, checked against its current media.

    *entry* is the entry's media as ``Dataset.resolve_media_scope`` resolved it,
    reduced to the camera a consumer reads. The variant row is looked up for that
    camera.

    Args:
        ds: The dataset.
        run_id: The variant to read, as a ``media`` parameter names it.
        entry: The entry and camera, with the media it resolves to now.

    Returns:
        The variant file, its stored facts and placement, and the entry media
        the placement maps into.

    Raises:
        MediaVariantMissingError: If the variant has no row for the entry and
            camera, or the row's file is gone.
        MediaVariantDriftedError: If the entry's media changed after the file
            was written from it, or no longer holds the frames the file's
            placement maps into.
    """
    group, sequence, camera = entry.group, entry.sequence, entry.camera
    key = make_entry_key(group, sequence)
    where = f"{key} (camera {camera})" if camera else key
    command = _preprocess_command(group, sequence, run_id)
    row = variant_row(ds, run_id, group, sequence, camera)
    if row is None:
        message = (
            f"{where}: the media variant {run_id} holds no file for this entry. Run "
            f"the preprocess step that made {run_id} over this entry first:\n"
            f"{command}"
        )
        raise MediaVariantMissingError(message)
    path = ds.resolve_path(row["abs_path"])
    if not path.is_file():
        message = (
            f"{where}: the file of the media variant {run_id} is missing at {path}. "
            f"Run the preprocess step that made {run_id} over this entry again:\n"
            f"{command}"
        )
        raise MediaVariantMissingError(message)
    current = media_composition_for(ds, group, sequence)
    if compositions_disagree(row["consumed_media_composition"], current):
        message = (
            f"{where}: the entry's media changed after the media variant {run_id} "
            f"was written from it, so the variant depicts media the entry no "
            f"longer holds. Run the preprocess step that made {run_id} over this "
            f"entry again, which rewrites a variant whose media changed:\n"
            f"{command}"
        )
        raise MediaVariantDriftedError(message)
    facts = variant_facts(row)
    placement = variant_placement(row)
    try:
        return VariantSource(
            run_id=run_id,
            path=path,
            facts=facts,
            placement=placement,
            entry_paths=tuple(entry.resolved.paths),
            entry_facts=tuple(entry.resolved.facts),
        )
    except ValueError as exc:
        message = (
            f"{where}: the media variant {run_id} was made from media the entry "
            f"no longer resolves to ({exc}). Run the preprocess step that made "
            f"{run_id} over this entry again with overwrite:\n{command}"
        )
        raise MediaVariantDriftedError(message) from exc
