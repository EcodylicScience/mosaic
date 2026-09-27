"""From what was annotated to what one training run reads.

A saved state (:mod:`mosaic.core.annotations.pose_annotations`) holds every pose
a set uses, every frame whether finished or not, and aliases as the annotator
assigned them. A training dataset holds one keypoint layout, finished frames only,
and classes. This is the step between, and the only place those choices are made,
so the saved state never has to anticipate them.

It works over the union of the sets a dataset is built from, because class ids
and class names are decided over all of them.

Three rules are what keep the result honest.

- **Finished frames only, always.** Only a frame the annotator signed off is
  known to be complete. There is no switch to include the rest: training an
  unfinished frame teaches every animal not yet annotated on it as background.
- **Objects are never filtered out of a frame they belong to.** Leaving one
  animal of the chosen pose unlabelled on a frame trains it as background, so a
  kept frame keeps every object of that pose. Objects of other poses leave, which
  is correct: a cricket is background to a mouse model.
- **A frame left with no object of the chosen pose stays**, as a negative. It is
  finished, so it is known to hold none.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from mosaic.core.annotations.bbox import BboxPolicy, derived_bbox
from mosaic.core.annotations.model import (
    AnnotationFrame,
    AnnotationObject,
    AnnotationSet,
    KeypointSchema,
)
from mosaic.core.annotations.pose_annotations import (
    FORMAT_NAME,
    PoseAnnotationSet,
    PoseDefinition,
    PoseFrame,
)

__all__ = ["AliasRole", "narrow_pose_sets"]

AliasRole = Literal["alias"]
"""What an object's alias can become: today only itself, as a class or a track."""


def narrow_pose_sets(
    sets: Mapping[str, PoseAnnotationSet],
    *,
    pose: int | str | None = None,
    class_by: AliasRole | None = None,
    track_by: AliasRole | None = None,
    bbox: BboxPolicy | None = None,
) -> dict[str, AnnotationSet]:
    """Each set narrowed to one pose's finished frames, with classes decided jointly.

    Args:
        sets: The sets, keyed by a label that names each in an error. The first
            in iteration order is the reference: its keypoint layout is the one
            every set must match, and an alias takes its name from the first
            set that declares it.
        pose: Which pose: its id, its name, or ``None`` when every set declares
            exactly one pose.
        class_by: ``"alias"`` makes each alias a class, and refuses an object
            of the pose that carries none. ``None`` is one class, named by the
            pose.
        track_by: ``"alias"`` gives each object its alias's name as its track
            identity, which is what a SLEAP identity model learns. ``None``
            keeps the track the source carried.
        bbox: What a box the annotator did not draw is derived under. ``None``
            uses each set's own policy: the box the annotator saw.

    Returns:
        One single-schema set per label, holding the finished frames and their
        objects of the chosen pose, with every box explicit.

    Raises:
        ValueError: A set does not declare the pose, or declares it ambiguously;
            the sets' layouts differ; ``class_by`` meets an object without an
            alias, or two aliases share a name; or no set holds a finished
            frame.
    """
    if not sets:
        msg = "there are no annotation sets to narrow"
        raise ValueError(msg)
    chosen = {label: _select(label, state, pose) for label, state in sets.items()}
    reference_label = next(iter(chosen))
    reference = chosen[reference_label]
    for label, definition in chosen.items():
        _check_layout(label, definition.schema, reference_label, reference.schema)

    alias_names: dict[int, str] = {}
    for definition in chosen.values():
        for alias in definition.aliases:
            _ = alias_names.setdefault(alias.id, alias.name)

    kept = {
        label: tuple(frame for frame in state.frames if frame.usable)
        for label, state in sets.items()
    }
    if not any(kept.values()):
        msg = f"none of the sets {sorted(sets)} holds a finished (usable) frame"
        raise ValueError(msg)

    classes = (
        _alias_classes(kept, chosen, alias_names)
        if class_by == "alias"
        else (reference.name,)
    )

    narrowed: dict[str, AnnotationSet] = {}
    for label, state in sets.items():
        definition = chosen[label]
        policy = bbox or state.bbox_policy
        frames = tuple(
            _narrow_frame(
                frame,
                definition,
                policy,
                alias_names,
                class_name=classes[0] if class_by is None else None,
                track_by=track_by,
            )
            for frame in kept[label]
        )
        narrowed[label] = AnnotationSet(
            schema=reference.schema,
            frames=frames,
            categories=classes,
            image_root=state.image_root,
            source_format=FORMAT_NAME,
        )
    return narrowed


def _select(
    label: str, state: PoseAnnotationSet, pose: int | str | None
) -> PoseDefinition:
    if pose is None:
        if len(state.poses) != 1:
            names = sorted(definition.name for definition in state.poses)
            msg = (
                f"set {label} holds {len(state.poses)} poses ({names}); name the one "
                "to train"
            )
            raise ValueError(msg)
        return state.poses[0]
    matching = [
        definition
        for definition in state.poses
        if (definition.id == pose if isinstance(pose, int) else definition.name == pose)
    ]
    if len(matching) == 1:
        return matching[0]
    held = sorted(f"{definition.id}:{definition.name}" for definition in state.poses)
    if not matching:
        msg = f"set {label} holds no pose {pose!r} (it holds {held})"
    else:
        msg = f"set {label} holds {len(matching)} poses called {pose!r}; name one by id ({held})"
    raise ValueError(msg)


def _check_layout(
    label: str,
    schema: KeypointSchema,
    reference_label: str,
    reference: KeypointSchema,
) -> None:
    """Refuse a set whose keypoints, edges or mirror pairs differ from the first.

    One model has one layout. Names are compared position by position, because
    the position is what a label line stores; edges and pairs as sets of
    unordered pairs, because their order carries no meaning.
    """
    for aspect, mine, theirs in (
        ("keypoints", list(schema.names), list(reference.names)),
        ("skeleton", _unordered(schema.skeleton), _unordered(reference.skeleton)),
        (
            "symmetry pairs",
            _unordered(schema.symmetries),
            _unordered(reference.symmetries),
        ),
    ):
        if mine != theirs:
            msg = (
                f"set {label} has {aspect} {mine}, but set {reference_label} has "
                f"{theirs}. One model has one keypoint layout."
            )
            raise ValueError(msg)


def _unordered(pairs: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    return sorted((min(a, b), max(a, b)) for a, b in pairs)


def _alias_classes(
    kept: Mapping[str, tuple[PoseFrame, ...]],
    chosen: Mapping[str, PoseDefinition],
    alias_names: Mapping[int, str],
) -> tuple[str, ...]:
    """The class names aliases make, sorted, refusing what would mislabel.

    An object of the pose without an alias would have no class, and dropping it
    would leave that animal unlabelled on its frame, which trains it as
    background. So it is refused, naming where. Two aliases sharing a name would
    merge two identities into one class without anyone having chosen to.
    """
    used: set[int] = set()
    missing: dict[str, int] = {}
    for label, frames in kept.items():
        pose_id = chosen[label].id
        for frame in frames:
            for obj in frame.objects:
                if obj.pose_id != pose_id:
                    continue
                if obj.alias_id is None:
                    missing[label] = missing.get(label, 0) + 1
                else:
                    used.add(obj.alias_id)
    if missing:
        where = ", ".join(f"{count} in {label}" for label, count in missing.items())
        msg = (
            f"aliases are the classes, but objects carry none: {where}. Assign "
            "each an alias, or train without classes by alias."
        )
        raise ValueError(msg)
    by_name: dict[str, list[int]] = {}
    for alias_id in sorted(used):
        by_name.setdefault(alias_names[alias_id], []).append(alias_id)
    shared = {name: ids for name, ids in by_name.items() if len(ids) > 1}
    if shared:
        msg = f"different aliases share a name, so they cannot be told apart as classes: {shared}"
        raise ValueError(msg)
    return tuple(sorted(by_name))


def _narrow_frame(
    frame: PoseFrame,
    definition: PoseDefinition,
    policy: BboxPolicy,
    alias_names: Mapping[int, str],
    *,
    class_name: str | None,
    track_by: AliasRole | None,
) -> AnnotationFrame:
    objects: list[AnnotationObject] = []
    for obj in frame.objects:
        if obj.pose_id != definition.id:
            continue
        alias = "" if obj.alias_id is None else alias_names[obj.alias_id]
        objects.append(
            AnnotationObject(
                keypoints=obj.keypoints,
                category=alias if class_name is None else class_name,
                track_id=alias if track_by == "alias" else obj.track_id,
                bbox=obj.bbox
                or derived_bbox(obj.keypoints, frame.width, frame.height, policy),
            )
        )
    return AnnotationFrame(
        image_path=frame.image_path,
        width=frame.width,
        height=frame.height,
        objects=tuple(objects),
        video=frame.sequence,
        frame_index=frame.frame_index,
    )
