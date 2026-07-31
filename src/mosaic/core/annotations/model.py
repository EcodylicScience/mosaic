"""What a set of labelled frames is, independent of who wrote it or reads it.

Every converter under ``tracking/pose_training/converters/`` went straight from
one source format to one training-dataset layout, so seven readers and four
writers were braided into seven functions. Nothing in the middle named a frame,
an instance, or a keypoint, which is why the usable-images filter exists five
times and why re-deciding a bounding box meant parsing emitted label text back
into arrays.

These four levels are that middle. They deliberately match the shape the control
plane already stores -- an annotation set holds frames, a frame holds objects, an
object holds keypoints -- so that reading labels out of the database and reading
them out of a COCO file can arrive at the same thing.

Two conventions are worth stating because they are choices, not consequences.

**Keypoints are positional and dense.** An object's ``keypoints`` is exactly as
long as the schema's ``names``, aligned index for index, and a keypoint that was
never placed is present with ``visibility=0`` and ``NaN`` coordinates. The
control plane instead omits the row -- an unplaced keypoint is the absence of a
record -- and a reader materializes that gap. Dense costs a little memory and
buys every emitter a straight ``zip``, removing the class of defect where one
index slips against another.

**Visibility is decided by the reader and never re-decided by an emitter.** The
canonical scale is COCO's: 0 unlabelled, 1 labelled but occluded, 2 visible. A
source that cannot express occlusion says so once, in its reader, rather than
leaving every writer to guess -- which is how the same annotation currently
becomes ``2`` through CVAT, a confidence threshold through Lightning Pose, and a
passthrough through COCO.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

__all__ = [
    "AnnotationFrame",
    "AnnotationObject",
    "AnnotationSet",
    "Bbox",
    "Keypoint",
    "KeypointSchema",
    "Split",
    "Visibility",
]

Visibility = Literal[0, 1, 2]
"""COCO's scale: 0 unlabelled, 1 labelled but occluded, 2 labelled and visible.

Chosen over a bare bool because YOLO writes the flag through unchanged and COCO
round-trips it, so collapsing the middle value would lose information mosaic was
handed. A source with no notion of occlusion simply never produces ``1``.
"""

Split = Literal["train", "valid", "test", "unassigned"]
"""Which subset a frame belongs to.

``unassigned`` is the state before a splitter has run, and it is a real state
rather than a ``None``: a set read from disk and never split is not the same as
one whose split is empty, and an emitter should be able to tell.
"""


@dataclass(frozen=True, slots=True)
class KeypointSchema:
    """The keypoint layout every object in a set is aligned to.

    Attributes:
        names: Keypoint names, in the order coordinates are stored.
        skeleton: Edges as pairs of 0-based indices into *names*.
    """

    names: tuple[str, ...]
    skeleton: tuple[tuple[int, int], ...] = ()

    @property
    def num_keypoints(self) -> int:
        return len(self.names)

    @property
    def kpt_shape(self) -> list[int]:
        """``[num_keypoints, 3]``, the shape a YOLO ``data.yaml`` declares."""
        return [self.num_keypoints, 3]

    def subset(self, indices: Sequence[int]) -> KeypointSchema:
        """The schema restricted to *indices*, with the skeleton remapped.

        Dropping keypoints without remapping would leave edges naming positions
        that no longer exist, so an edge survives only when both of its endpoints
        do, and its indices are rewritten to the new positions.
        """
        kept = list(indices)
        position = {old: new for new, old in enumerate(kept)}
        return KeypointSchema(
            names=tuple(self.names[i] for i in kept),
            skeleton=tuple(
                (position[a], position[b])
                for a, b in self.skeleton
                if a in position and b in position
            ),
        )


@dataclass(frozen=True, slots=True)
class Bbox:
    """An axis-aligned box in native image pixels, top-left origin."""

    x: float
    y: float
    width: float
    height: float

    @property
    def is_degenerate(self) -> bool:
        """Whether the box has no area, and so cannot describe an instance."""
        return self.width <= 0.0 or self.height <= 0.0


@dataclass(frozen=True, slots=True)
class Keypoint:
    """One placed or unplaced point, in native image pixels.

    Attributes:
        x: Column, ``NaN`` when the point was never placed.
        y: Row, ``NaN`` when the point was never placed.
        visibility: See :data:`Visibility`.
        score: The producer's confidence, when the source is a prediction rather
            than a human annotation. Kept separate from *visibility* because
            thresholding a score is a policy an emitter should not have to
            re-invent, and because a thresholded score cannot be un-thresholded.
    """

    x: float
    y: float
    visibility: Visibility
    score: float | None = None

    @property
    def is_placed(self) -> bool:
        """Whether this point has usable coordinates."""
        return (
            self.visibility != 0 and not math.isnan(self.x) and not math.isnan(self.y)
        )

    @classmethod
    def absent(cls) -> Keypoint:
        """The dense representation of a point that was never placed."""
        return cls(x=math.nan, y=math.nan, visibility=0)


@dataclass(frozen=True, slots=True)
class AnnotationObject:
    """One annotated instance within a frame.

    Attributes:
        keypoints: Exactly ``len(schema.names)`` points, aligned to the schema.
        category: The class this instance belongs to, by name.
        track_id: Identity across frames, when the source carries one. This is
            the axis every existing converter drops: instances survive as extra
            lines in a label file, but which line is which animal does not.
        source_id: The identifier the source gave this instance, kept so a
            round trip can be traced back to the record it came from.
        bbox: The source's own box, when it supplied one. ``None`` means derive
            it, which is what a :class:`~mosaic.core.annotations.bbox.BboxPolicy`
            is for.
    """

    keypoints: tuple[Keypoint, ...]
    category: str = "animal"
    track_id: str = ""
    source_id: str = ""
    bbox: Bbox | None = None

    @property
    def placed_keypoints(self) -> tuple[Keypoint, ...]:
        """The points that carry usable coordinates."""
        return tuple(point for point in self.keypoints if point.is_placed)


@dataclass(frozen=True, slots=True)
class AnnotationFrame:
    """One image, and every instance annotated in it.

    Attributes:
        image_path: Where the image is. Absolute, or relative to the set's
            ``image_root``.
        width: Image width in pixels, needed by every emitter that normalizes.
        height: Image height in pixels.
        objects: The instances. Empty is legal and means "annotated, nothing
            present", which is not the same as never having been annotated.
        video: Which recording the frame came from, and the key a group-aware
            split keeps together so frames of one video cannot leak between
            train and validation.
        frame_index: Position within *video*, or ``-1`` when the source is a
            loose image with no sequence.
        split: See :data:`Split`.
    """

    image_path: Path
    width: int
    height: int
    objects: tuple[AnnotationObject, ...] = ()
    video: str = ""
    frame_index: int = -1
    split: Split = "unassigned"

    @property
    def is_annotated(self) -> bool:
        """Whether anything was labelled here."""
        return bool(self.objects)


@dataclass(frozen=True, slots=True)
class AnnotationSet:
    """A schema, a class list, and the frames labelled against them.

    Attributes:
        schema: The keypoint layout every object is aligned to.
        frames: The labelled frames.
        categories: Class names, in the order their ids are assigned.
        image_root: What relative ``image_path`` values are relative to.
        source_format: Which reader produced this, for provenance.
    """

    schema: KeypointSchema
    frames: tuple[AnnotationFrame, ...] = ()
    categories: tuple[str, ...] = ("animal",)
    image_root: Path | None = None
    source_format: str = ""

    def __post_init__(self) -> None:
        """Reject a set whose objects are not aligned to its schema.

        Checked on construction rather than trusted, because a misaligned object
        is not detectable downstream: every emitter zips coordinates against
        names, so a short tuple silently drops the tail keypoint and a long one
        silently invents a name for it.
        """
        expected = self.schema.num_keypoints
        for frame in self.frames:
            for index, obj in enumerate(frame.objects):
                if len(obj.keypoints) != expected:
                    raise ValueError(
                        f"{frame.image_path.name} object {index} carries "
                        f"{len(obj.keypoints)} keypoints, but the schema declares "
                        f"{expected}; a set is dense and positional"
                    )

    @property
    def annotated_frames(self) -> tuple[AnnotationFrame, ...]:
        """Frames carrying at least one instance.

        Every emitter wants this and each used to compute it, differing only in
        whether it also checked the image existed on disk -- a filter written
        five times over five slightly different record shapes.
        """
        return tuple(frame for frame in self.frames if frame.is_annotated)

    @property
    def max_objects_per_frame(self) -> int:
        """The widest frame, which is what tells a single-instance target to refuse."""
        return max((len(frame.objects) for frame in self.frames), default=0)

    def category_ids(self) -> dict[str, int]:
        """Class name to contiguous 0-based id, in declaration order."""
        return {name: index for index, name in enumerate(self.categories)}

    def resolve(self, frame: AnnotationFrame) -> Path:
        """*frame*'s image path, anchored against ``image_root`` when relative."""
        if frame.image_path.is_absolute() or self.image_root is None:
            return frame.image_path
        return self.image_root / frame.image_path

    def with_frames(self, frames: Sequence[AnnotationFrame]) -> AnnotationSet:
        """A copy carrying *frames*, for a pass that rewrites them wholesale."""
        return AnnotationSet(
            schema=self.schema,
            frames=tuple(frames),
            categories=self.categories,
            image_root=self.image_root,
            source_format=self.source_format,
        )

    def __iter__(self) -> Iterator[AnnotationFrame]:
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)
