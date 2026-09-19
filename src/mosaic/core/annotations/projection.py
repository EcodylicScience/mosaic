"""Saving an annotator's state as a revision of the ``keypoints`` label series.

The annotation tool keeps its state somewhere editing is cheap -- a database with
history. What a training run reads has to be in the dataset, and it has to be a
*specific version*, so a model can be tied to exactly what it saw. This is the
seam between the two: the caller hands over the current state, and mosaic writes
it as the next revision, or recognizes that nothing changed and writes nothing.

mosaic owns the layout under ``labels_raw/keypoints/``. The caller chooses no
filename and no revision number. See :mod:`mosaic.core.pipeline.label_series` for
the rule a series follows and why it is not the rule the rest of ``labels_raw``
follows.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.core.annotations.model import AnnotationFrame, AnnotationSet
from mosaic.core.annotations.writers.coco import coco_keypoints_document
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.label_series_index import (
    SeriesRevision,
    write_series_revision,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = ["KEYPOINTS_SERIES", "keypoint_set_payload", "write_keypoint_set_revision"]

KEYPOINTS_SERIES = "keypoints"
"""The series a keypoint annotation set is saved into."""


def _anchored(ds: Dataset, annotations: AnnotationSet, frame: AnnotationFrame) -> Path:
    """*frame*'s image as the payload names it: relative to the dataset root.

    Root-relative is what makes a revision portable. It names the same pinned
    image after the dataset moves, and a reader in another dataset finds it from
    the ``image_root`` the revision's manifest records. An image outside the
    dataset keeps its absolute path, the rule every index follows for a file the
    dataset does not own.
    """
    return Path(ds.relative_to_root(annotations.resolve(frame)))


def keypoint_set_payload(ds: Dataset, annotations: AnnotationSet) -> bytes:
    """The exact bytes a revision of *annotations* would hold.

    **Deterministic for a given state**, which is what lets an unchanged save be
    recognized. Frames are sorted by image path, keys are sorted, and the form is
    compact, so neither the order the caller happened to collect frames in nor
    any whitespace is a change. Objects keep the order they were given in: that
    order is the instance axis, and the caller is the one who knows it.

    Nothing time-dependent may reach *annotations*. A timestamp in the payload
    would make every save a new revision.
    """
    anchored = sorted(
        (
            AnnotationFrame(
                image_path=_anchored(ds, annotations, frame),
                width=frame.width,
                height=frame.height,
                objects=frame.objects,
                video=frame.video,
                frame_index=frame.frame_index,
                split=frame.split,
            )
            for frame in annotations.frames
        ),
        key=lambda frame: frame.image_path.as_posix(),
    )
    document = coco_keypoints_document(annotations.with_frames(anchored))
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_keypoint_set_revision(
    ds: Dataset,
    *,
    set_key: str,
    annotations: AnnotationSet,
    origin: Mapping[str, JsonValue],
    origin_ref: str = "",
) -> SeriesRevision:
    """Save *annotations* as the next revision of *set_key*, unless nothing changed.

    Safe to call every time the annotator is closed, and again when a training
    run is submitted: an unchanged state returns the revision already there and
    writes nothing. The returned revision is what a training run should name, so
    the model is tied to exactly this state.

    Args:
        ds: The dataset the annotated images belong to.
        set_key: What the set is called on disk. One path component, and stable
            across saves: a key that changes starts a new series of revisions.
        annotations: The state to save. Pass only what should be trained on --
            mosaic does not know which frames a person has signed off.
        origin: Provenance from the authoring store, recorded verbatim beside the
            payload. A database commit belongs here, and so does any timestamp.
        origin_ref: A short pointer into *origin* for the index row, such as
            ``"dolt:1a2b3c4d5e"``.

    Returns:
        The revision, and whether this call wrote it.
    """
    return write_series_revision(
        ds,
        series=KEYPOINTS_SERIES,
        key=set_key,
        payload=keypoint_set_payload(ds, annotations),
        origin=origin,
        n_records=len(annotations.frames),
        origin_ref=origin_ref,
    )
