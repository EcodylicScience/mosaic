"""Rearranging a sequence's clips -- item 6.3, and the gestures that guard it.

A reorder is metadata-only on the filesystem: ``video_order`` is a column, and
after Stage 7 every derivative is named for the video it came from, so nothing is
renamed and nothing re-encodes. It is emphatically not *inert*, because
``video_order`` is a term of the media composition and the composition is what
says a source moved. That is the point of the gesture rather than a side effect.

**Preview by default.** ``apply`` is off, so calling this answers "what would
happen" and touches nothing. The preview and the change run the same enumeration
over the same arrangement, which is what P4's enumerate-twice asks for: the
second pass happens inside the lock ``write_media_index`` already takes.

**Both endpoints are constructed here, never accepted from a caller.**
``MediaIndexScope`` names a *directory*, and ``write_media_index`` preserves
verbatim every row not under a passed scope -- so a cross-sequence move that
passed only the destination would leave the source holding a row pointing at a
file that is gone and a hole in its order. The one production caller builds its
scopes from imported paths, which is right for an upload and silently wrong for a
move. Building them here is how the requirement is enforced rather than
documented.

**What blocks, and what does not.** A block is a refusal a caller can override
with ``force``; the two that exist are the ones where proceeding silently
destroys something no recipe can rebuild:

- *Converted labels.* Their frame indices are sequence-global, so a reorder
  shifts every index past the change point. The remap is not built (item 9.3
  gives it a source side to be checked against first), so M4 refuses rather than
  shipping an inexact rewrite over human-authored scoring.
- *A readability regression.* If the proposed order would make a sequence one
  the reader refuses **and the committed order is one it accepts**, that is a
  break the gesture is introducing. A sequence already unreadable does not block:
  refusing there would strand it, and reordering may be exactly how a user fixes
  it.

A scope directory holding another sequence's files is a hard refusal rather than
a block, because ``write_media_index`` would assign *every* file under it to the
scope's identity -- merging two sequences into one, which no force flag should be
able to ask for by accident.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.media.uniformity import UniformityVerdict
from mosaic.core.pipeline.media_index import MediaIndexScope

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "Arrangement",
    "RearrangeReport",
    "rearrange_media",
]


@dataclass(frozen=True, slots=True)
class Arrangement:
    """One sequence's proposed clip order.

    ``order_by_name`` maps a clip's basename to its position, the same shape
    :class:`~mosaic.core.pipeline.media_index.MediaIndexScope` carries, so a
    caller that has an arrangement has it in one spelling rather than two.
    """

    group: str
    sequence: str
    order_by_name: Mapping[str, int]

    @property
    def entry(self) -> tuple[str, str]:
        return (self.group, self.sequence)


@dataclass(frozen=True, slots=True)
class RearrangeReport:
    """What a rearrangement would do, or did.

    ``blocked`` is empty when the gesture may proceed. A non-empty ``blocked``
    with ``applied`` false is the ordinary refusal; with ``applied`` true it
    records what was overridden, so a forced run still says what it ran over.
    """

    applied: bool
    forced: bool
    arrangements: tuple[Arrangement, ...]
    reached: pd.DataFrame
    uniformity: dict[tuple[str, str, str], UniformityVerdict] = field(
        default_factory=dict
    )
    blocked: tuple[str, ...] = ()

    @property
    def would_proceed(self) -> bool:
        """Whether an ``apply=True`` call would go through without ``force``."""
        return not self.blocked


def rearrange_media(
    ds: Dataset,
    arrangements: Iterable[Arrangement],
    *,
    apply: bool = False,
    force: bool = False,
    index_filename: str = "index.csv",
) -> RearrangeReport:
    """Reorder clips within sequences, previewing by default.

    Returns what the change reaches, which cameras it would make unreadable, and
    why it refuses. Nothing is written unless *apply*; nothing is written over a
    block unless *force*.

    Raises:
        ValueError: When a sequence's clips do not share a directory, or share
            one with another sequence's. ``write_media_index`` assigns every file
            under a scope to that scope's identity, so proceeding would merge two
            sequences -- a hard refusal rather than a block, because no caller
            should be able to ask for it by accident.
    """
    proposed = tuple(arrangements)
    if not proposed:
        return RearrangeReport(
            applied=False, forced=force, arrangements=(), reached=_no_rows()
        )

    rows = ds.read_media_index(index_filename)
    scopes = [_scope_for(ds, rows, item) for item in proposed]

    uniformity = _readability_regressions(ds, proposed, index_filename)
    reached = _reached(ds, proposed)
    blocked = _blocks(ds, proposed, uniformity)

    if not apply or (blocked and not force):
        return RearrangeReport(
            applied=False,
            forced=force,
            arrangements=proposed,
            reached=reached,
            uniformity=uniformity,
            blocked=blocked,
        )

    _ = ds.write_media_index(scopes, index_filename=index_filename)
    return RearrangeReport(
        applied=True,
        forced=force,
        arrangements=proposed,
        # Recomputed after the write, so a caller reading the report sees what
        # the change produced rather than what it predicted. The two agree when
        # nothing raced; when something did, the second answer is the true one.
        reached=_reached(ds, proposed),
        uniformity=uniformity,
        blocked=blocked,
    )


def _no_rows() -> pd.DataFrame:
    from mosaic.core.pipeline.provenance import PROVENANCE_COLUMNS

    return pd.DataFrame(columns=pd.Index(PROVENANCE_COLUMNS))


def _reached(ds: Dataset, proposed: Sequence[Arrangement]) -> pd.DataFrame:
    """The blast radius of a media change over these sequences."""
    from mosaic.core.pipeline.provenance import reached_by

    return reached_by(ds, [item.entry for item in proposed], "media_raw")


def _scope_for(
    ds: Dataset, rows: Sequence[Mapping[str, str]], item: Arrangement
) -> MediaIndexScope:
    """The scope this sequence's rearrangement needs, built rather than accepted."""
    directory = _sequence_directory(ds, rows, item)
    return MediaIndexScope(
        directory=directory,
        group=item.group,
        sequence=item.sequence,
        order_by_name=dict(item.order_by_name),
    )


def _sequence_directory(
    ds: Dataset, rows: Sequence[Mapping[str, str]], item: Arrangement
) -> Path:
    """Where this sequence's clips live, refusing anything a scope would merge."""
    mine = [
        row
        for row in rows
        if (str(row.get("group", "")), str(row.get("sequence", ""))) == item.entry
    ]
    if not mine:
        message = f"no media rows for {item.entry!r}; nothing to rearrange"
        raise ValueError(message)

    parents = {ds.resolve_path(str(row["abs_path"])).parent for row in mine}
    if len(parents) != 1:
        listed = ", ".join(sorted(str(parent) for parent in parents))
        message = (
            f"{item.entry!r} spans {len(parents)} directories ({listed}); a scope "
            f"names one directory and would assign every file under it to this "
            f"sequence"
        )
        raise ValueError(message)
    directory = parents.pop()

    intruders = sorted(
        {
            (str(row.get("group", "")), str(row.get("sequence", "")))
            for row in rows
            if ds.resolve_path(str(row["abs_path"])).parent == directory
            and (str(row.get("group", "")), str(row.get("sequence", ""))) != item.entry
        }
    )
    if intruders:
        message = (
            f"{directory} also holds media for {intruders!r}; rearranging "
            f"{item.entry!r} through a scope on it would reassign theirs too"
        )
        raise ValueError(message)
    return directory


def _readability_regressions(
    ds: Dataset, proposed: Sequence[Arrangement], index_filename: str
) -> dict[tuple[str, str, str], UniformityVerdict]:
    """Cameras the proposed order would make unreadable that are readable now.

    A sequence already unreadable is left out deliberately: refusing there would
    strand it, and rearranging may be exactly how it gets fixed.
    """
    regressions: dict[tuple[str, str, str], UniformityVerdict] = {}
    for item in proposed:
        before = ds.sequence_uniformity(
            item.group, item.sequence, index_filename=index_filename
        )
        after = ds.sequence_uniformity(
            item.group,
            item.sequence,
            order_by_name=item.order_by_name,
            index_filename=index_filename,
        )
        for camera, verdict in after.items():
            was_readable = camera not in before or before[camera].readable
            if was_readable and not verdict.readable:
                regressions[(item.group, item.sequence, camera)] = verdict
    return regressions


def _blocks(
    ds: Dataset,
    proposed: Sequence[Arrangement],
    regressions: Mapping[tuple[str, str, str], UniformityVerdict],
) -> tuple[str, ...]:
    """Why this rearrangement refuses, in the words a user can act on."""
    reasons: list[str] = []
    for item in proposed:
        kinds = _label_kinds(ds, item.entry)
        if kinds:
            listed = ", ".join(kinds)
            reasons.append(
                f"{item.entry!r} has converted labels ({listed}) whose frame "
                f"indices are sequence-global, so this reorder shifts every index "
                f"past the change point. The remap is not built; forcing leaves "
                f"the labels describing different frames"
            )
    for (group, sequence, camera), verdict in sorted(regressions.items()):
        mismatch = verdict.mismatch
        detail = (
            f"{mismatch.field} {mismatch.first} vs {mismatch.other}"
            if mismatch is not None
            else "a property mismatch"
        )
        where = f"({group!r}, {sequence!r})" + (f" camera {camera!r}" if camera else "")
        reasons.append(
            f"{where} reads today and would not after this order: {detail}. "
            f"Position 0 sets the reference every other clip is compared against"
        )
    return tuple(reasons)


def _label_kinds(ds: Dataset, entry: tuple[str, str]) -> tuple[str, ...]:
    """Converted label kinds for *entry*, sorted. Empty when the root has none."""
    try:
        labels_root = ds.get_root("labels")
    except KeyError:
        return ()
    if not labels_root.exists():
        return ()
    found: set[str] = set()
    for index_path in sorted(labels_root.glob("*/index.csv")):
        # Read as plain records rather than a frame: the labels index is the one
        # index in the dataset with no typed row and no dtype map (item 6.1's
        # reconciler is what gives it one), so pandas would infer a numeric
        # sequence name back into a number here.
        try:
            with index_path.open(newline="") as handle:
                records = list(csv.DictReader(handle))
        except OSError:
            continue
        for record in records:
            found_entry = (
                str(record.get("group", "")),
                str(record.get("sequence", "")),
            )
            if found_entry == entry:
                found.add(index_path.parent.name)
                break
    return tuple(sorted(found))
