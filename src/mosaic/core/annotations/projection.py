"""Saving an annotation set's state as a revision of the ``keypoints`` label series.

The annotation tool keeps its state somewhere editing is cheap -- a database with
history. What a training run reads has to be in the dataset, and it has to be a
*specific version*, so a model can be tied to exactly what it saw. This is the
seam between the two: the caller hands over the current state, and mosaic writes
it as the next revision, or recognizes that nothing changed and writes nothing.

A revision holds the full state (:mod:`mosaic.core.annotations.pose_annotations`)
and a COCO Keypoints export generated from it
(:mod:`mosaic.core.annotations.writers.coco`) for the tools that read COCO. Only
the first is the revision's identity.

mosaic owns the layout under ``labels_raw/keypoints/``. The caller chooses no
filename and no revision number. See :mod:`mosaic.core.pipeline.label_series` for
the rule a series follows and why it is not the rule the rest of ``labels_raw``
follows.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mosaic.core.annotations.pose_annotations import (
    PoseAnnotationSet,
    pose_annotations_payload,
)
from mosaic.core.annotations.writers.coco import coco_keypoints_payload
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.label_series import series_spec
from mosaic.core.pipeline.label_series_index import (
    SeriesRevision,
    write_series_revision,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "COCO_EXPORT_FILENAME",
    "KEYPOINTS_SERIES",
    "anchored_keypoint_set",
    "write_keypoint_set_revision",
]

KEYPOINTS_SERIES: Final = "keypoints"
"""The series a keypoint annotation set is saved into."""

COCO_EXPORT_FILENAME: Final = series_spec(KEYPOINTS_SERIES).exports[0]
"""The COCO Keypoints export every revision holds beside its saved state."""


def anchored_keypoint_set(
    ds: Dataset, annotations: PoseAnnotationSet
) -> PoseAnnotationSet:
    """*annotations* with every image named relative to *ds*, frames in path order.

    Root-relative is what makes a revision portable. It names the same pinned
    image after the dataset moves, and a reader in another dataset finds it from
    the ``image_root`` the revision's manifest records. An image outside the
    dataset keeps its absolute path, the rule every index follows for a file the
    dataset does not own. Anchoring comes first because a relative path and an
    absolute one can name the same image.
    """
    frames = sorted(
        (
            replace(
                frame,
                image_path=Path(ds.relative_to_root(annotations.resolve(frame))),
            )
            for frame in annotations.frames
        ),
        key=lambda frame: frame.image_path.as_posix(),
    )
    return replace(annotations, frames=tuple(frames), image_root=ds.base_dir)


def write_keypoint_set_revision(
    ds: Dataset,
    *,
    set_key: str,
    annotations: PoseAnnotationSet,
    origin: Mapping[str, JsonValue],
    origin_ref: str = "",
) -> SeriesRevision:
    """Save *annotations* as the next revision of *set_key*, unless nothing changed.

    Pass the set's whole state: every frame, finished or not, with its
    ``usable`` sign-off, and every object. What a training run uses of it is
    chosen when the run is prepared, so one saved state serves every run.

    Safe to call whenever a pinned copy is needed -- when a training run is
    submitted, or when the annotations are exported. An unchanged state returns
    the revision already there and writes nothing. The returned revision is what
    a training run should name, so the model is tied to exactly this state.

    Args:
        ds: The dataset the annotated images belong to.
        set_key: What the set is called on disk. One path component, and stable
            across saves: a key that changes starts a new series of revisions.
        annotations: The state to save. Nothing time-dependent may reach it: a
            timestamp would make every save a new revision.
        origin: Provenance from the authoring store, recorded verbatim beside the
            payload. A database commit belongs here, and so does any timestamp.
        origin_ref: A short pointer into *origin* for the index row, such as
            ``"dolt:1a2b3c4d5e"``.

    Returns:
        The revision, and whether this call wrote it. Its ``path`` is the saved
        state; the COCO export sits beside it as :data:`COCO_EXPORT_FILENAME`.

    Raises:
        ValueError: Two frames name one image once anchored.
    """
    spec = series_spec(KEYPOINTS_SERIES)
    anchored = anchored_keypoint_set(ds, annotations)
    return write_series_revision(
        ds,
        series=spec.name,
        key=set_key,
        payload=pose_annotations_payload(anchored),
        exports={COCO_EXPORT_FILENAME: coco_keypoints_payload(anchored)},
        origin=origin,
        n_records=len(anchored.frames),
        origin_ref=origin_ref,
    )
