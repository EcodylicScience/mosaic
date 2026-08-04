"""Promoting a manual correction into ``tracks_raw`` -- item 8.6.

A corrected track set cannot be recomputed. That makes it *source*, not a derived
variant, and rule P1 says source lives under a source root -- so promotion is a
copy from a tracker working directory into ``tracks_raw/<sequence>/``, after
which the ordinary machinery does the rest: item 4.5's checksums are already on
by default, item 4.4's composition already covers that sequence, and every
artifact built from ``tracks_raw`` is invalidated by the composition moving.
There is no new identity machinery here, which is what the item promised.

**Open item O1, resolved: an append-only revision series.** A second correction of
the same sequence is not a conflict requiring a force every time; it is the next
revision of a source file. This is the one place a revision counter earns its
place, and it earns it where O1 said it might -- on the source file, not on the
recipe. Nothing is overwritten, every earlier correction stays addressable, and
each one moves the composition, which is what invalidates downstream artifacts
correctly rather than silently.

**What blocks is derivatives, not history.** P4's rule is that a source change
blocks while derivatives exist and forcing it proceeds; the preview is
``reached_by``, which is documented as answering *membership* when run before the
change. So the block is "these artifacts were built from this sequence's
``tracks_raw`` and will be stale", and forcing does not delete them -- deleting is
``delete_set``'s gesture, behind its own force. Promotion refuses to be two
destructive operations wearing one name.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd

from mosaic.core.helpers import entry_directory, validate_entry_name
from mosaic.core.pipeline.provenance import reached_by
from mosaic.core.pipeline.sequence_index import (
    SequenceLabelRow,
    read_sequence_labels,
    sequence_label_path,
    sequence_labels,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "PromotionReport",
    "next_revision",
    "promote_correction",
]

REVISION_STEM: Final = "corrected"
"""Stem every promoted file carries, before its revision number.

Fixed rather than derived from the source filename, so the revision series is
readable as one: ``corrected.rev1.npz`` beside ``corrected.rev2.npz`` says what
it is, where ``vid1_fish0.npz`` beside ``vid1_fish0.rev2.npz`` does not.
"""


@dataclass(frozen=True)
class PromotionReport:
    """What a promotion would do, or did."""

    applied: bool
    group: str
    sequence: str
    revision: int
    promoted: tuple[Path, ...] = ()
    derived_from: str = ""
    blocked: tuple[str, ...] = ()
    reached: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    @property
    def would_proceed(self) -> bool:
        """Is this promotion unblocked, or forced past its blocks?"""
        return not self.blocked


def next_revision(destination: Path) -> int:
    """The next revision number for *destination*, counting from 1.

    Reads the directory rather than a stored counter: the files *are* the series,
    so a counter could disagree with them, and the disagreement would be silent.
    """
    highest = 0
    for existing in destination.glob(f"{REVISION_STEM}.rev*"):
        marker = existing.name.split(".")[1]
        if not marker.startswith("rev"):
            continue
        try:
            highest = max(highest, int(marker[3:]))
        except ValueError:
            continue
    return highest + 1


def promote_correction(
    ds: Dataset,
    group: str,
    sequence: str,
    source: Path | Iterable[Path],
    *,
    src_format: str,
    derived_from: str = "",
    apply: bool = False,
    force: bool = False,
) -> PromotionReport:
    """Copy a corrected track set into ``tracks_raw`` as the next revision.

    Dry-run unless *apply*, and blocked while derivatives built from this
    sequence's ``tracks_raw`` exist -- pass *force* to promote anyway, leaving
    them to be reported or removed by ``delete_set``.

    Args:
        ds: The dataset.
        group: Entry group; may be empty.
        sequence: Entry sequence.
        source: The corrected file, or files, to promote. Typically the contents
            of a tracker working directory under ``_tracking``.
        src_format: The converter that reads the promoted files. Required, with
            no default: this lands in ``tracks_raw/index.csv`` and is the only
            thing that says how the file converts, while ``_tracking`` holds
            three trackers' working directories -- a default would quietly index
            a SLEAP or Lightning Pose correction as TRex, and the error would
            surface at conversion rather than here. It matches
            ``TracksRawIndexScope.src_format``, which the scope below is built
            from and which has never had a default either.
        derived_from: The producer run this was corrected from, recorded on the
            dataset-level ``sequences.csv`` so the correction's lineage survives
            the working directory being swept.
        apply: Perform the copy and re-index.
        force: Promote despite existing derivatives.

    Returns:
        A :class:`PromotionReport`.
    """
    from mosaic.core.pipeline.tracks_raw_index import TracksRawIndexScope

    group = validate_entry_name(group, "group")
    sequence = validate_entry_name(sequence, "sequence")
    sources = (
        [Path(source)] if isinstance(source, (str, Path)) else [Path(p) for p in source]
    )
    missing = [p for p in sources if not p.is_file()]
    if missing:
        named = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"nothing to promote at: {named}")

    # O3 decided per-sequence subdirectories for tracks_raw, and this is the
    # gesture that needs them: a flat root cannot hold a revision series without
    # the sequence name being part of every filename.
    destination = entry_directory(ds.get_root("tracks_raw"), group, sequence)
    revision = next_revision(destination) if destination.exists() else 1

    # Run *before* the change, where every row reads `current` and the answer is
    # membership: what this promotion would make stale.
    reached = reached_by(ds, [(group, sequence)], "tracks_raw")
    blocked: tuple[str, ...] = ()
    if not reached.empty and not force:
        blocked = (
            f"{len(reached)} derived artifact(s) were built from this sequence's "
            "tracks_raw and would become stale. Re-run them, or pass force=True "
            "to promote anyway and use delete_set to remove what became wrong.",
        )

    report = PromotionReport(
        applied=False,
        group=group,
        sequence=sequence,
        revision=revision,
        derived_from=derived_from,
        blocked=blocked,
        reached=reached,
    )
    if blocked or not apply:
        return report

    destination.mkdir(parents=True, exist_ok=True)
    promoted: list[Path] = []
    ordered = sorted(sources)
    for original in ordered:
        # One revision, one file per member. A correction of a sequence whose
        # tracker wrote a file per individual is still *one* event, so the
        # revision number is shared -- but the members need distinct names or
        # each copy lands on the last one's path and only the final file
        # survives, with the index recording that survivor's checksum so nothing
        # downstream can tell the rest are gone. The token sits after the
        # revision marker, which the series reader takes as the second
        # dot-separated field, so it stays readable.
        member = f".{_member_token(original)}" if len(ordered) > 1 else ""
        landed = (
            destination / f"{REVISION_STEM}.rev{revision}{member}{_suffixes(original)}"
        )
        _ = shutil.copy2(original, landed)
        promoted.append(landed)

    # The scope form, not `index_tracks_raw`: the caller knows this entry's
    # identity and must not have it re-derived from a filename that now reads
    # `corrected.rev2` rather than the sequence name.
    _ = ds.write_tracks_raw_index(
        [
            TracksRawIndexScope(
                directory=destination,
                group=group,
                sequence=sequence,
                src_format=src_format,
            )
        ],
        patterns=[f"{REVISION_STEM}.rev*"],
    )
    if derived_from:
        _record_lineage(ds, group, sequence, derived_from)

    return PromotionReport(
        applied=True,
        group=group,
        sequence=sequence,
        revision=revision,
        promoted=tuple(promoted),
        derived_from=derived_from,
        reached=reached,
    )


def _suffixes(path: Path) -> str:
    """``.npz``, or ``.tar.gz`` -- every suffix, so a double extension survives."""
    return "".join(path.suffixes)


def _member_token(path: Path) -> str:
    """What distinguishes one file of a multi-file correction from its siblings.

    The original name without its suffixes, which for a tracker writing one file
    per individual is what names the individual. Dots become hyphens so the
    revision series stays readable as dot-separated fields.
    """
    stem = path.name[: len(path.name) - len(_suffixes(path))] or path.stem
    return stem.replace(".", "-")


def _record_lineage(ds: Dataset, group: str, sequence: str, derived_from: str) -> None:
    """Write ``derived_from`` on the dataset-level sequence row.

    Its own writer rather than a keyword on ``set_display_name``: that method
    means "call this sequence X", and promotion is not a relabel. Folding them
    would make one gesture mean two things and give each the other's failure
    modes -- a promotion that clears a display name, a rename that claims a
    lineage.

    The column was declared in M3 and left unused for exactly this.
    """
    path = sequence_label_path(ds)
    labels = sequence_labels(path)
    labels.append(
        [
            SequenceLabelRow(
                group=group,
                sequence=sequence,
                # Carried, not cleared: a promotion must not silently un-name a
                # sequence somebody labelled.
                display_group=_prior_cell(ds, group, sequence, "display_group"),
                display_name=_prior_cell(ds, group, sequence, "display_name"),
                derived_from=derived_from,
            )
        ]
    )


def _prior_cell(ds: Dataset, group: str, sequence: str, column: str) -> str:
    """One cell of this entry's existing label row, or ``""``.

    Through ``read_sequence_labels``, which its own docstring calls *the single
    reader*: it projects onto the current schema and answers an absent file with
    the full-schema empty frame. That last part is what this needed -- a dataset
    that has never named a sequence has no label file, and the first draft here
    read it directly and died on the first promotion to such a dataset, *after*
    having already copied the file and rewritten the index. Adding a second
    reader to fix that would have been the wrong repair twice over.
    """
    frame = read_sequence_labels(ds)
    matching = frame[(frame["group"] == group) & (frame["sequence"] == sequence)]
    if matching.empty:
        return ""
    return str(matching.iloc[-1][column])
