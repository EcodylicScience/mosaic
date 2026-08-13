"""Transcode coverage: the kind with no run directory to look in.

Every other artifact is addressed by a run identifier naming a directory, and
its coverage is which outputs that directory holds. Transcode is not.
``transcode_run_id`` says so of itself -- "This value addresses nothing. It names
no directory and gates no reuse" -- because the output is named by its recipe and
reuse is gated by that filename plus the forward link on the source row.

So its coverage is a property of the **media index**: which in-scope rows can be
read for this target, either because they need no derivative or because the
derivative they need is registered and present. Nothing else needs to exist.

**This is the case a single coverage signature gets wrong**, and it gets it wrong
in the worst direction. Asked for a run directory that was never supposed to
exist, a directory-shaped check reports zero of N -- so a corpus that is entirely
clean, with nothing to transcode and nothing missing, reads as permanently
incomplete. Anything acting on that resubmits the same work every tick, forever.

**Built in ``core`` rather than through the contributor registry**, unlike the
ops kinds. The registry exists to carry what lives above the layering line;
the media index and its verdict columns are core's own, so routing this through
a registration would create a seam with no boundary behind it and make the kind
unavailable to a core-only caller for no reason.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mosaic.core.media.facts_columns import (
    derivative_path_for_target,
    media_row_uuid,
    read_link_cell,
    transcode_required,
)
from mosaic.core.pipeline.media_index import read_media_index

from .model import ArtifactRecord, Coverage, InventoryScope, MediaDerivativeRef, Target
from .model import classify

if TYPE_CHECKING:
    from ._read import IndexReader
    from mosaic.core.dataset import Dataset

__all__ = ["media_derivative_record"]


def _nothing_to_cover(target: Target) -> ArtifactRecord[str]:
    """The record for a dataset holding no media at all: nothing is missing."""
    coverage = Coverage[str](target=frozenset(), present=frozenset())
    return ArtifactRecord[str](
        ref=MediaDerivativeRef(target=target),
        name=f"transcode:{target}",
        run_id="",
        coverage=coverage,
        status="absent",
        extra={"needs_transcode": frozenset(), "needs_probe": frozenset()},
    )


def media_derivative_record(
    ds: Dataset, target: Target, scope: InventoryScope, reader: IndexReader
) -> ArtifactRecord[str]:
    """Whether every in-scope media row can be read for *target*.

    Keyed on ``video_uuid``, which is what a derivative links back by and what
    survives a rename. A row carrying none -- an imgstore, or a row not yet
    probed -- falls back to its stored path so it is still nameable rather than
    silently dropped from the target.

    Two remedies are reported separately rather than as one "incomplete" count,
    mirroring the two textually distinct errors the read path already raises: a
    row needing a transcode wants ``mosaic run --kind transcode``, and a row with
    no reconstructable measurement wants ``mosaic reprobe-media``. Collapsing
    them would tell a user their corpus is short without saying what to do.
    """
    # ``resolve_media_root`` falls back to ``"media"`` when ``media_raw`` is
    # unset, and returns that name whether or not ``media`` is set either -- so a
    # tracks-only dataset, which declares both roots and fills neither, names a
    # root ``get_root`` then refuses. There is no media to be short of in that
    # case, and empty coverage is the honest answer rather than an exception out
    # of a read.
    root_key = ds.resolve_media_root()
    if not ds.has_root(root_key):
        return _nothing_to_cover(target)
    index_path = ds.get_root(root_key) / "index.csv"
    reader.note(index_path)
    # Derivatives are anchored under the ``media`` root. Without one, nothing can
    # be registered, so a row needing a transcode reads as needing it still.
    media_root = ds.get_root("media") if ds.has_root("media") else None

    covered: set[str] = set()
    target_keys: set[str] = set()
    needs_transcode: set[str] = set()
    needs_probe: set[str] = set()

    for row in read_media_index(index_path):
        if read_link_cell(row, "media_type") == "imgstore":
            # A store is read natively and has no elementary stream to transcode,
            # so it is not a row this coverage can be short of.
            continue
        entry = (read_link_cell(row, "group"), read_link_cell(row, "sequence"))
        if scope.entries is not None and entry not in scope.entries:
            continue
        key = media_row_uuid(row) or read_link_cell(row, "abs_path")
        if not key:
            continue
        target_keys.add(key)

        if transcode_required(row, target):
            linked = (
                derivative_path_for_target(row, target, media_root)
                if media_root is not None
                else None
            )
            # Both halves, matching the reuse gate the transcode op itself
            # applies: the link records the registration and the file is the
            # output. Registration writes the back-link row first and the
            # forward link last, so an unlinked file is the recoverable state
            # and reads here as still needing the work.
            if linked is not None and linked.exists():
                covered.add(key)
            else:
                needs_transcode.add(key)
            continue

        if read_link_cell(row, "media_facts"):
            covered.add(key)
        else:
            needs_probe.add(key)

    coverage = Coverage(target=frozenset(target_keys), present=frozenset(covered))
    return ArtifactRecord[str](
        ref=MediaDerivativeRef(target=target),
        name=f"transcode:{target}",
        run_id="",
        coverage=coverage,
        status=classify(
            satisfied=coverage.is_satisfied,
            any_covered=bool(coverage.covered),
            orphan_rows=False,
            orphan_files=False,
            drifted=False,
            finished=True,
        ),
        index_path=index_path,
        rows=frozenset(target_keys),
        extra={
            "needs_transcode": frozenset(needs_transcode),
            "needs_probe": frozenset(needs_probe),
        },
    )
