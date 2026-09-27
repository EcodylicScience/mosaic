"""An annotation set's full state, and the file mosaic saves it as.

The annotator's state lives in the control plane's database. What a training run
reads, and what a trained model names as what it saw, has to be on disk and pinned
to a version: a revision of the ``keypoints`` label series. This module is the
format of that revision. It holds every frame of the set, signed off or not; every
pose its objects use, with the aliases and mirror pairs the pose declares; every
object, with the box the annotator drew and where the object came from; and the
padding the annotator's canvas derived the other boxes with.

**A format of mosaic's own, with COCO written beside it.** Every annotation tool
surveyed (CVAT, Label Studio, Roboflow, Ultralytics, V7, SLEAP) keeps its full
state in a format of its own and generates COCO or YOLO from that. COCO has no
place for an unfinished frame, an alias with a stable id, a mirror pair, or whether
a box was drawn, and a file that tried to carry them would either mislead the tools
that read COCO or lose the information. So a revision holds this file, which is
what training reads and what its identity is taken over, and a COCO Keypoints
export generated from it (:mod:`mosaic.core.annotations.writers.coco`), which is
what another tool reads.

**What was annotated, not what one model trains on.** Choosing a pose, keeping the
signed-off frames and turning aliases into classes happen when a training dataset
is prepared (:mod:`mosaic.core.annotations.narrow`). A saved state therefore serves
every training run, rather than fixing one run's choices into the record.

**Deterministic by construction.** The payload's digest decides whether a save
changed anything, so one state always serializes to the same bytes: poses and
aliases are sorted by id, frames by image path, edges and pairs ascending, keys
sorted, the form compact, and every coordinate a float. Objects keep the order
they were given in, because that order is the instance axis and only the caller
knows it. Nothing time-dependent, no styling and no database row id belongs in
the payload; provenance goes in the revision's ``origin``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, ValidationError

from mosaic.core.annotations.bbox import BboxPolicy
from mosaic.core.annotations.model import (
    AnnotationSet,
    Bbox,
    Keypoint,
    KeypointSchema,
)
from mosaic.core.json_value import JsonValue
from mosaic.core.strict_model import StrictModel, terse

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "Alias",
    "ObjectOrigin",
    "PoseAnnotationSet",
    "PoseDefinition",
    "PoseFrame",
    "PoseObject",
    "pose_annotations_document",
    "pose_annotations_json_schema",
    "pose_annotations_payload",
    "read_pose_annotations",
    "widen",
]

FORMAT_NAME: Final = "mosaic-pose-annotations"
"""The ``format`` every file of this kind declares."""

FORMAT_VERSION: Final = 1
"""The version this module writes, and the newest it reads."""

ObjectOrigin = Literal["human", "model", "heuristic"]
"""Who placed an object: a person, a model's prediction, or a rule."""


@dataclass(frozen=True, slots=True)
class Alias:
    """A named identity within a pose, such as ``resident`` or ``intruder``.

    Attributes:
        id: The authoring store's id, which a rename leaves unchanged.
        name: What it is called now.
    """

    id: int
    name: str


@dataclass(frozen=True, slots=True)
class PoseDefinition:
    """One pose: its keypoint layout, its name, and the aliases it declares.

    Attributes:
        id: The authoring store's id, which a rename leaves unchanged.
        name: What it is called now.
        schema: Keypoint names, skeleton edges and mirror pairs.
        aliases: The aliases objects of this pose may carry.
    """

    id: int
    name: str
    schema: KeypointSchema
    aliases: tuple[Alias, ...] = ()

    def __post_init__(self) -> None:
        ids = [alias.id for alias in self.aliases]
        if len(set(ids)) != len(ids):
            msg = f"pose {self.name!r} declares one alias id twice: {sorted(ids)}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PoseObject:
    """One annotated instance of one pose.

    Attributes:
        pose_id: Which of the set's poses it is.
        keypoints: Exactly one per keypoint of that pose, in the pose's order. An
            unplaced keypoint is :meth:`Keypoint.absent`. A keypoint's ``score``
            is not saved: annotations are not predictions with a confidence.
        alias_id: Which of the pose's aliases it carries, if any.
        origin: Who placed it, or ``None`` when the source did not say.
        source_ref: What produced it -- a model run, a rule -- when not a person.
        bbox: The box the annotator drew. ``None`` means the box is derived from
            the keypoints under the set's :attr:`PoseAnnotationSet.bbox_policy`.
        track_id: Identity across frames, when the source carries one.
    """

    pose_id: int
    keypoints: tuple[Keypoint, ...]
    alias_id: int | None = None
    origin: ObjectOrigin | None = None
    source_ref: str | None = None
    bbox: Bbox | None = None
    track_id: str = ""


@dataclass(frozen=True, slots=True)
class PoseFrame:
    """One image, the objects annotated in it, and whether it is finished.

    Attributes:
        image_path: Absolute, or relative to the set's ``image_root``.
        width: Image width in pixels.
        height: Image height in pixels.
        usable: The annotator's sign-off that the frame is complete. Only a
            usable frame is trained on or exported, and only on a usable frame
            does "no objects" mean "looked at, nothing there".
        objects: The instances, of any of the set's poses.
        sequence: The recording the frame came from, or ``""`` when unknown.
        frame_index: Its position in that recording, or ``-1`` when unknown.
    """

    image_path: Path
    width: int
    height: int
    usable: bool
    objects: tuple[PoseObject, ...] = ()
    sequence: str = ""
    frame_index: int = -1


@dataclass(frozen=True, slots=True)
class PoseAnnotationSet:
    """The full state of one annotation set.

    Attributes:
        poses: Every pose the set's objects use.
        frames: Every frame of the set, finished or not.
        bbox_policy: How the annotator's canvas derived a box the annotator did
            not draw. Part of the state: changing it changes every derived box.
        image_root: What relative ``image_path`` values are relative to.
    """

    poses: tuple[PoseDefinition, ...]
    frames: tuple[PoseFrame, ...] = ()
    bbox_policy: BboxPolicy = field(default_factory=BboxPolicy)
    image_root: Path | None = None

    def __post_init__(self) -> None:
        """Refuse a state no reader could take back as the same state.

        Checked on construction, because a file that saved a malformed state
        would be an immutable revision a model may name.
        """
        ids = [pose.id for pose in self.poses]
        if len(set(ids)) != len(ids):
            msg = f"the set declares one pose id twice: {sorted(ids)}"
            raise ValueError(msg)
        alias_ids = [alias.id for pose in self.poses for alias in pose.aliases]
        if len(set(alias_ids)) != len(alias_ids):
            msg = f"two poses declare one alias id: {sorted(alias_ids)}"
            raise ValueError(msg)
        poses = {pose.id: pose for pose in self.poses}
        for frame in self.frames:
            for index, obj in enumerate(frame.objects):
                where = f"{frame.image_path.as_posix()} object {index}"
                pose = poses.get(obj.pose_id)
                if pose is None:
                    msg = f"{where} is of pose {obj.pose_id}, which the set does not declare"
                    raise ValueError(msg)
                _check_object(obj, pose, where)

    def pose(self, pose_id: int) -> PoseDefinition:
        """The pose with id *pose_id*."""
        for pose in self.poses:
            if pose.id == pose_id:
                return pose
        msg = f"the set declares no pose {pose_id}"
        raise KeyError(msg)

    def resolve(self, frame: PoseFrame) -> Path:
        """*frame*'s image path, anchored against ``image_root`` when relative."""
        if frame.image_path.is_absolute() or self.image_root is None:
            return frame.image_path
        return self.image_root / frame.image_path


def _check_object(obj: PoseObject, pose: PoseDefinition, where: str) -> None:
    expected = pose.schema.num_keypoints
    if len(obj.keypoints) != expected:
        msg = (
            f"{where} carries {len(obj.keypoints)} keypoints, but pose "
            f"{pose.name!r} declares {expected}"
        )
        raise ValueError(msg)
    if obj.alias_id is not None and obj.alias_id not in {a.id for a in pose.aliases}:
        msg = f"{where} carries alias {obj.alias_id}, which pose {pose.name!r} does not declare"
        raise ValueError(msg)
    for position, point in enumerate(obj.keypoints):
        if point.visibility != 0 and not (
            math.isfinite(point.x) and math.isfinite(point.y)
        ):
            msg = f"{where} keypoint {pose.schema.names[position]!r} is placed at a non-finite position"
            raise ValueError(msg)
    box = obj.bbox
    if box is not None and not all(
        math.isfinite(value) for value in (box.x, box.y, box.width, box.height)
    ):
        msg = f"{where} has a box with a non-finite side"
        raise ValueError(msg)


# --- Writing ---------------------------------------------------------------------


def pose_annotations_document(annotations: PoseAnnotationSet) -> dict[str, JsonValue]:
    """*annotations* as the mapping the file holds, in its canonical order.

    Raises:
        ValueError: Two frames name one image. One path has to address one image
            for an image and its annotations to stay matched.
    """
    frames = sorted(annotations.frames, key=lambda frame: frame.image_path.as_posix())
    for earlier, later in zip(frames, frames[1:], strict=False):
        if earlier.image_path == later.image_path:
            msg = f"two frames name the image {later.image_path.as_posix()}"
            raise ValueError(msg)
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "bbox_policy": annotations.bbox_policy.model_dump(mode="json"),
        "poses": [
            _pose_document(pose)
            for pose in sorted(annotations.poses, key=lambda pose: pose.id)
        ],
        "frames": [_frame_document(frame) for frame in frames],
    }


def pose_annotations_payload(annotations: PoseAnnotationSet) -> bytes:
    """The exact bytes a revision of *annotations* holds: compact, keys sorted."""
    document = pose_annotations_document(annotations)
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _pairs(pairs: Sequence[tuple[int, int]]) -> list[JsonValue]:
    return [[a, b] for a, b in sorted(pairs)]


def _pose_document(pose: PoseDefinition) -> dict[str, JsonValue]:
    return {
        "id": pose.id,
        "name": pose.name,
        "keypoints": list(pose.schema.names),
        "skeleton": _pairs(pose.schema.skeleton),
        "symmetries": _pairs(pose.schema.symmetries),
        "aliases": [
            {"id": alias.id, "name": alias.name}
            for alias in sorted(pose.aliases, key=lambda alias: alias.id)
        ],
    }


def _frame_document(frame: PoseFrame) -> dict[str, JsonValue]:
    return {
        "image": frame.image_path.as_posix(),
        "width": frame.width,
        "height": frame.height,
        "sequence": frame.sequence or None,
        "frame_index": frame.frame_index if frame.frame_index >= 0 else None,
        "usable": frame.usable,
        "objects": [_object_document(obj) for obj in frame.objects],
    }


def _object_document(obj: PoseObject) -> dict[str, JsonValue]:
    box = obj.bbox
    return {
        "pose": obj.pose_id,
        "alias": obj.alias_id,
        "origin": obj.origin,
        "source_ref": obj.source_ref,
        "track_id": obj.track_id or None,
        "bbox": (
            None
            if box is None
            else [float(box.x), float(box.y), float(box.width), float(box.height)]
        ),
        "keypoints": [
            None
            if point.visibility == 0
            else [float(point.x), float(point.y), int(point.visibility)]
            for point in obj.keypoints
        ],
    }


# --- Reading ---------------------------------------------------------------------


# The file is mosaic's own, so a key these models do not know is a mistake rather
# than a newer writer's addition: that is what the version field is for. Their
# descriptions are the format's documentation, rendered into the reference by
# pose_annotations_json_schema.


class AliasRecord(StrictModel):
    """An alias as the file holds it."""

    id: int = Field(description="The authoring store's id; a rename keeps it.")
    name: str = Field(description="What the alias is called when saved.")


class PoseRecord(StrictModel):
    """A pose as the file holds it."""

    id: int = Field(description="The authoring store's id; a rename keeps it.")
    name: str = Field(description="What the pose is called when saved.")
    keypoints: list[str] = Field(
        description="Keypoint names, in the order every object stores its points."
    )
    skeleton: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Edges as pairs of 0-based keypoint positions, ascending.",
    )
    symmetries: list[tuple[int, int]] = Field(
        default_factory=list,
        description=(
            "Left-right mirror pairs as 0-based positions, lower first. What a "
            "trainer that flips images swaps."
        ),
    )
    aliases: list[AliasRecord] = Field(
        default_factory=list, description="The aliases this pose declares, by id."
    )


class ObjectRecord(StrictModel):
    """An annotated object as the file holds it."""

    pose: int = Field(description="The id of the object's pose.")
    keypoints: list[tuple[float, float, Literal[1, 2]] | None] = Field(
        description=(
            "One entry per keypoint of the pose: [x, y, visibility] in image "
            "pixels, where visibility 1 is occluded and 2 visible, or null when "
            "the keypoint was not placed."
        )
    )
    alias: int | None = Field(
        default=None, description="The id of the object's alias, or null."
    )
    origin: ObjectOrigin | None = Field(
        default=None, description="Who placed it, or null when the source did not say."
    )
    source_ref: str | None = Field(
        default=None,
        description="What produced it when not a person, such as a model run.",
    )
    track_id: str | None = Field(
        default=None, description="Identity across frames, when the source has one."
    )
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description=(
            "The box the annotator drew, [x, y, width, height] in pixels from the "
            "top-left. Null means derived from the keypoints under bbox_policy."
        ),
    )


class FrameRecord(StrictModel):
    """A frame as the file holds it."""

    image: str = Field(
        description=(
            "The image, relative to the dataset the set was saved in; absolute "
            "when it lies outside it."
        )
    )
    width: int = Field(description="Image width in pixels.")
    height: int = Field(description="Image height in pixels.")
    usable: bool = Field(
        description=(
            "The annotator's sign-off that the frame is complete. Only a usable "
            "frame is trained on or exported."
        )
    )
    sequence: str | None = Field(
        default=None, description="The recording the frame came from."
    )
    frame_index: int | None = Field(
        default=None, description="The frame's position in that recording."
    )
    objects: list[ObjectRecord] = Field(
        default_factory=list,
        description="The objects, in the order the annotator's tool holds them.",
    )


class PoseAnnotationsFile(StrictModel):
    """A ``mosaic-pose-annotations`` file."""

    format: Literal["mosaic-pose-annotations"] = Field(
        description="Always mosaic-pose-annotations."
    )
    version: int = Field(
        description="The format version the file was written in. This is version 1."
    )
    bbox_policy: BboxPolicy = Field(
        description="How the annotator's tool derived a box the annotator did not draw."
    )
    poses: list[PoseRecord] = Field(
        description="Every pose an object uses, ascending by id."
    )
    frames: list[FrameRecord] = Field(
        description="Every frame of the set, finished or not, ascending by image."
    )


def pose_annotations_json_schema() -> dict[str, JsonValue]:
    """The file's JSON Schema, which is also its documentation."""
    schema: dict[str, JsonValue] = PoseAnnotationsFile.model_json_schema()
    return schema


def read_pose_annotations(
    path: str | Path, image_root: str | Path | None = None
) -> PoseAnnotationSet:
    """Read a file :func:`pose_annotations_payload` wrote.

    Args:
        path: The file.
        image_root: What its relative image paths are relative to. A revision
            records it in its manifest, relative to the revision directory.

    Raises:
        ValueError: The file is not this format, is a newer version of it than
            this mosaic reads, or holds a state the model refuses.
    """
    path = Path(path)
    try:
        parsed = PoseAnnotationsFile.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        msg = f"{path} is not a {FORMAT_NAME} file: {terse(exc)}"
        raise ValueError(msg) from exc
    if parsed.version > FORMAT_VERSION:
        msg = (
            f"{path} is {FORMAT_NAME} version {parsed.version}; this mosaic reads "
            f"up to version {FORMAT_VERSION}. Upgrade mosaic to read it."
        )
        raise ValueError(msg)
    return PoseAnnotationSet(
        poses=tuple(
            PoseDefinition(
                id=pose.id,
                name=pose.name,
                schema=KeypointSchema(
                    names=tuple(pose.keypoints),
                    skeleton=tuple(pose.skeleton),
                    symmetries=tuple(pose.symmetries),
                ),
                aliases=tuple(Alias(id=a.id, name=a.name) for a in pose.aliases),
            )
            for pose in parsed.poses
        ),
        frames=tuple(
            PoseFrame(
                image_path=Path(frame.image),
                width=frame.width,
                height=frame.height,
                usable=frame.usable,
                objects=tuple(_read_object(obj) for obj in frame.objects),
                sequence=frame.sequence or "",
                frame_index=-1 if frame.frame_index is None else frame.frame_index,
            )
            for frame in parsed.frames
        ),
        bbox_policy=parsed.bbox_policy,
        image_root=None if image_root is None else Path(image_root),
    )


def _read_object(entry: ObjectRecord) -> PoseObject:
    box = entry.bbox
    return PoseObject(
        pose_id=entry.pose,
        keypoints=tuple(
            Keypoint.absent()
            if point is None
            else Keypoint(x=point[0], y=point[1], visibility=point[2])
            for point in entry.keypoints
        ),
        alias_id=entry.alias,
        origin=entry.origin,
        source_ref=entry.source_ref,
        track_id=entry.track_id or "",
        bbox=None
        if box is None
        else Bbox(x=box[0], y=box[1], width=box[2], height=box[3]),
    )


# --- From the narrow model -------------------------------------------------------


def widen(
    annotations: AnnotationSet, *, bbox_policy: BboxPolicy | None = None
) -> PoseAnnotationSet:
    """A single-schema set as a full state: one pose, every frame finished.

    For a caller holding the narrow model -- a set read from a foreign file, or
    a merged training set -- that needs the writers built on this one. The pose
    takes id 1 and the set's first class name. Classes, splits and per-object
    source ids are the narrow model's and have no place here.

    Args:
        annotations: The set.
        bbox_policy: What a missing box is derived under. Defaults to the bare
            hull of the placed keypoints.
    """
    pose = PoseDefinition(
        id=1,
        name=annotations.categories[0] if annotations.categories else "animal",
        schema=annotations.schema,
    )
    return PoseAnnotationSet(
        poses=(pose,),
        frames=tuple(
            PoseFrame(
                image_path=frame.image_path,
                width=frame.width,
                height=frame.height,
                usable=True,
                objects=tuple(
                    PoseObject(
                        pose_id=pose.id,
                        keypoints=obj.keypoints,
                        bbox=obj.bbox,
                        track_id=obj.track_id,
                    )
                    for obj in frame.objects
                ),
                sequence=frame.video,
                frame_index=frame.frame_index,
            )
            for frame in annotations.frames
        ),
        bbox_policy=bbox_policy or BboxPolicy(method="tight", margin=0.0),
        image_root=annotations.image_root,
    )
