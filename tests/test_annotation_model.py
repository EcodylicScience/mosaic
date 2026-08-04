"""The invariants the canonical annotation representation is worth having for.

Mostly one invariant, stated four ways: an object's keypoints are positional and
dense, aligned index for index to the schema. Everything downstream zips
coordinates against names, so a misaligned object is not a wrong answer that
surfaces -- it is a silently dropped tail keypoint, or a coordinate attributed to
the wrong name.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from mosaic.core.annotations import (
    AnnotationFrame,
    AnnotationObject,
    AnnotationSet,
    Bbox,
    Keypoint,
    KeypointSchema,
)

SCHEMA = KeypointSchema(names=("nose", "thorax", "abdomen"), skeleton=((0, 1), (1, 2)))


def _frame(*objects: AnnotationObject, name: str = "f.png") -> AnnotationFrame:
    return AnnotationFrame(image_path=Path(name), width=100, height=80, objects=objects)


def _placed(x: float, y: float) -> Keypoint:
    return Keypoint(x=x, y=y, visibility=2)


# --- density ----------------------------------------------------------------


def test_an_object_must_carry_one_keypoint_per_schema_name() -> None:
    short = AnnotationObject(keypoints=(_placed(1.0, 2.0),))
    with pytest.raises(ValueError, match="carries 1 keypoints"):
        _ = AnnotationSet(schema=SCHEMA, frames=(_frame(short),))


def test_a_longer_object_is_refused_too() -> None:
    """The symmetric error, which would otherwise invent a name for the extra."""
    long = AnnotationObject(keypoints=tuple(_placed(float(i), 0.0) for i in range(4)))
    with pytest.raises(ValueError, match="carries 4 keypoints"):
        _ = AnnotationSet(schema=SCHEMA, frames=(_frame(long),))


def test_an_unplaced_keypoint_is_present_and_not_omitted() -> None:
    """Absence is a value, so the tuple stays aligned to the schema."""
    obj = AnnotationObject(
        keypoints=(_placed(10.0, 20.0), Keypoint.absent(), _placed(30.0, 40.0))
    )
    annotations = AnnotationSet(schema=SCHEMA, frames=(_frame(obj),))

    assert len(obj.keypoints) == SCHEMA.num_keypoints
    assert math.isnan(obj.keypoints[1].x)
    assert obj.keypoints[1].visibility == 0
    assert len(obj.placed_keypoints) == 2
    assert annotations.frames[0].objects[0].keypoints[2].x == 30.0


def test_zero_zero_is_a_position_not_an_absence() -> None:
    """The distinction NaN exists to preserve.

    COCO writes an unlabelled point as literal ``(0, 0, 0)``, so a reader that
    kept the coordinates would put an instance's box corner at the origin.
    """
    origin = Keypoint(x=0.0, y=0.0, visibility=2)
    assert origin.is_placed
    assert not Keypoint.absent().is_placed
    assert not Keypoint(x=0.0, y=0.0, visibility=0).is_placed


# --- schema subsetting ------------------------------------------------------


def test_subsetting_remaps_the_skeleton() -> None:
    """An edge survives only when both endpoints do, renumbered to new positions."""
    subset = SCHEMA.subset([0, 2])
    assert subset.names == ("nose", "abdomen")
    assert subset.skeleton == (), "both edges lost an endpoint"


def test_subsetting_keeps_an_edge_whose_endpoints_both_survive() -> None:
    schema = KeypointSchema(
        names=("a", "b", "c", "d"), skeleton=((0, 1), (1, 2), (2, 3))
    )
    subset = schema.subset([1, 2, 3])
    assert subset.names == ("b", "c", "d")
    assert subset.skeleton == ((0, 1), (1, 2)), "renumbered, not carried over"


def test_a_subset_skeleton_never_names_a_dropped_keypoint() -> None:
    """The defect the remap exists to prevent, asserted directly."""
    subset = KeypointSchema(names=("a", "b", "c"), skeleton=((0, 2),)).subset([0, 1])
    assert all(
        a < len(subset.names) and b < len(subset.names) for a, b in subset.skeleton
    )


# --- frames and sets --------------------------------------------------------


def test_an_annotated_frame_is_one_with_objects() -> None:
    """Empty means "looked at, nothing there", which is not unannotated."""
    obj = AnnotationObject(keypoints=tuple(Keypoint.absent() for _ in range(3)))
    annotations = AnnotationSet(
        schema=SCHEMA,
        frames=(_frame(obj, name="has.png"), _frame(name="empty.png")),
    )
    assert [f.image_path.name for f in annotations.annotated_frames] == ["has.png"]
    assert len(annotations) == 2


def test_the_widest_frame_is_what_a_single_instance_target_checks() -> None:
    one = AnnotationObject(keypoints=tuple(Keypoint.absent() for _ in range(3)))
    annotations = AnnotationSet(
        schema=SCHEMA,
        frames=(_frame(one, name="a.png"), _frame(one, one, name="b.png")),
    )
    assert annotations.max_objects_per_frame == 2


def test_an_empty_set_has_no_widest_frame_rather_than_raising() -> None:
    assert AnnotationSet(schema=SCHEMA).max_objects_per_frame == 0


def test_categories_map_to_contiguous_ids_in_declaration_order() -> None:
    annotations = AnnotationSet(schema=SCHEMA, categories=("worker", "queen"))
    assert annotations.category_ids() == {"worker": 0, "queen": 1}


def test_a_relative_image_path_resolves_against_the_root() -> None:
    annotations = AnnotationSet(
        schema=SCHEMA,
        frames=(_frame(name="clip/f.png"),),
        image_root=Path("/data/images"),
    )
    assert annotations.resolve(annotations.frames[0]) == Path("/data/images/clip/f.png")


def test_an_absolute_image_path_ignores_the_root() -> None:
    frame = AnnotationFrame(image_path=Path("/elsewhere/f.png"), width=10, height=10)
    annotations = AnnotationSet(
        schema=SCHEMA, frames=(frame,), image_root=Path("/data/images")
    )
    assert annotations.resolve(frame) == Path("/elsewhere/f.png")


def test_replacing_frames_keeps_everything_else() -> None:
    """The split pass rewrites frames wholesale and must not drop the schema."""
    annotations = AnnotationSet(
        schema=SCHEMA,
        frames=(_frame(name="a.png"),),
        categories=("worker",),
        image_root=Path("/data"),
        source_format="coco",
    )
    replaced = annotations.with_frames([_frame(name="b.png")])
    assert replaced.schema is annotations.schema
    assert replaced.categories == ("worker",)
    assert replaced.image_root == Path("/data")
    assert replaced.source_format == "coco"
    assert [f.image_path.name for f in replaced] == ["b.png"]


# --- boxes ------------------------------------------------------------------


def test_a_box_with_no_area_is_degenerate() -> None:
    """What every emitter checks before writing a line it cannot train on."""
    assert Bbox(x=0.0, y=0.0, width=0.0, height=5.0).is_degenerate
    assert Bbox(x=0.0, y=0.0, width=5.0, height=0.0).is_degenerate
    assert not Bbox(x=0.0, y=0.0, width=5.0, height=5.0).is_degenerate


def test_a_source_box_is_optional_and_distinguishable_from_a_derived_one() -> None:
    """``None`` means derive; a box means the source supplied one."""
    points = tuple(Keypoint.absent() for _ in range(3))
    assert AnnotationObject(keypoints=points).bbox is None
    supplied = AnnotationObject(keypoints=points, bbox=Bbox(1.0, 2.0, 3.0, 4.0))
    assert supplied.bbox is not None


# --- the Lightning Pose default skeleton ------------------------------------


def test_the_mouse_skeleton_survives_being_subset() -> None:
    """The converter's default layout, checked where the goldens cannot reach.

    The Lightning Pose converter tested ``keypoint_indices is None`` after the
    branch above had already assigned it, so the condition was never true and
    every schema it produced carried an empty skeleton. The characterization
    fixture is a three-keypoint layout rather than this one, so the fix is
    invisible there -- which is exactly why it needs asserting here.
    """
    from mosaic.tracking.pose_training.converters.lightning_pose import MOUSE_LP_27

    assert MOUSE_LP_27.num_keypoints == 27
    assert len(MOUSE_LP_27.skeleton) == 24, "the full layout has edges"

    # nose -> neck -> mid_back -> mouse_center is a contiguous run of the spine.
    spine = MOUSE_LP_27.subset([0, 7, 8, 9])
    assert spine.names == ("nose", "neck", "mid_back", "mouse_center")
    assert spine.skeleton == ((0, 1), (1, 2), (2, 3)), "renumbered to the subset"

    # Every surviving edge names a position that exists.
    for a, b in spine.skeleton:
        assert a < len(spine.names) and b < len(spine.names)
