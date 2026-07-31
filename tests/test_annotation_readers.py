"""What a reader is responsible for deciding, asserted per source.

A reader answers, for one format, the questions the representation asks: where
the image is, how big, which instances, and per keypoint whether it was placed.
The interesting cases are all where a source says something the representation
does not say the same way -- and those decisions belong to the reader, so this
is where they are pinned rather than in whichever emitter meets them first.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from mosaic.core.annotations.readers import read_coco_keypoints


def _coco(tmp_path: Path, **overrides: Any) -> Path:
    """A minimal single-category COCO file, overridable per test."""
    document: dict[str, Any] = {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 100, "height": 80},
            {"id": 2, "file_name": "b.png", "width": 100, "height": 80},
        ],
        "annotations": [
            {
                "id": 11,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "keypoints": [10.0, 20.0, 2, 15.0, 25.0, 1, 0.0, 0.0, 0],
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "mouse",
                "keypoints": ["nose", "thorax", "tail"],
                "skeleton": [[0, 1], [1, 2]],
            }
        ],
    }
    document.update(overrides)
    path = tmp_path / "annotations.json"
    _ = path.write_text(json.dumps(document))
    return path


# --- COCO -------------------------------------------------------------------


def test_the_schema_comes_from_the_category(tmp_path: Path) -> None:
    annotations = read_coco_keypoints(_coco(tmp_path), tmp_path / "images")
    assert annotations.schema.names == ("nose", "thorax", "tail")
    assert annotations.schema.skeleton == ((0, 1), (1, 2))
    assert annotations.categories == ("mouse",)
    assert annotations.source_format == "coco"


def test_an_unlabelled_keypoint_does_not_keep_its_filler_coordinates(
    tmp_path: Path,
) -> None:
    """COCO writes ``(0, 0, 0)``; those are not a position at the origin.

    Kept, they drag an instance's box corner to the top-left of the image --
    which is the defect the NaN convention exists to prevent, so the reader is
    where it has to be stopped.
    """
    annotations = read_coco_keypoints(_coco(tmp_path), tmp_path / "images")
    tail = annotations.frames[0].objects[0].keypoints[2]
    assert tail.visibility == 0
    assert math.isnan(tail.x) and math.isnan(tail.y)
    assert not tail.is_placed


def test_occlusion_survives_as_the_middle_value(tmp_path: Path) -> None:
    """A reader that collapsed 1 into 2 or 0 would lose what COCO recorded."""
    annotations = read_coco_keypoints(_coco(tmp_path), tmp_path / "images")
    assert [k.visibility for k in annotations.frames[0].objects[0].keypoints] == [
        2,
        1,
        0,
    ]


def test_a_source_box_is_carried_rather_than_recomputed(tmp_path: Path) -> None:
    annotations = read_coco_keypoints(_coco(tmp_path), tmp_path / "images")
    bbox = annotations.frames[0].objects[0].bbox
    assert bbox is not None
    assert (bbox.x, bbox.y, bbox.width, bbox.height) == (10.0, 20.0, 30.0, 40.0)


def test_an_annotation_without_a_box_leaves_it_to_be_derived(tmp_path: Path) -> None:
    path = _coco(
        tmp_path,
        annotations=[
            {
                "id": 11,
                "image_id": 1,
                "category_id": 1,
                "keypoints": [10.0, 20.0, 2, 15.0, 25.0, 2, 12.0, 30.0, 2],
            }
        ],
    )
    assert read_coco_keypoints(path, tmp_path).frames[0].objects[0].bbox is None


def test_an_image_with_no_annotations_is_still_a_frame(tmp_path: Path) -> None:
    """Looked at and empty is not the same as never looked at."""
    annotations = read_coco_keypoints(_coco(tmp_path), tmp_path / "images")
    assert len(annotations.frames) == 2
    assert len(annotations.annotated_frames) == 1
    assert annotations.frames[1].objects == ()


def test_several_instances_in_one_image_stay_separable(tmp_path: Path) -> None:
    """The axis every existing converter flattens into extra label lines."""
    path = _coco(
        tmp_path,
        annotations=[
            {
                "id": 11,
                "image_id": 1,
                "category_id": 1,
                "keypoints": [1.0, 1.0, 2, 2.0, 2.0, 2, 3.0, 3.0, 2],
            },
            {
                "id": 12,
                "image_id": 1,
                "category_id": 1,
                "keypoints": [4.0, 4.0, 2, 5.0, 5.0, 2, 6.0, 6.0, 2],
            },
        ],
    )
    objects = read_coco_keypoints(path, tmp_path).frames[0].objects
    assert len(objects) == 2
    assert [o.source_id for o in objects] == ["11", "12"]


def test_another_category_is_not_merged_in(tmp_path: Path) -> None:
    """One set has one schema, so reading is a filter and not a union."""
    path = _coco(
        tmp_path,
        annotations=[
            {"id": 11, "image_id": 1, "category_id": 1, "keypoints": [1.0, 1.0, 2] * 3},
            {"id": 99, "image_id": 1, "category_id": 2, "keypoints": [9.0, 9.0, 2] * 2},
        ],
        categories=[
            {"id": 1, "name": "mouse", "keypoints": ["nose", "thorax", "tail"]},
            {"id": 2, "name": "beetle", "keypoints": ["head", "abdomen"]},
        ],
    )
    annotations = read_coco_keypoints(path, tmp_path, category_name="beetle")
    assert annotations.schema.names == ("head", "abdomen")
    assert [o.source_id for o in annotations.frames[0].objects] == ["99"]


def test_an_absent_category_says_which_ones_exist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"Available: \['mouse'\]"):
        _ = read_coco_keypoints(_coco(tmp_path), tmp_path, category_name="beetle")


def test_a_file_with_no_categories_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="declares no categories"):
        _ = read_coco_keypoints(_coco(tmp_path, categories=[]), tmp_path)


def test_a_keypoint_subset_renumbers_the_skeleton(tmp_path: Path) -> None:
    """Dropping names without remapping leaves edges naming absent positions."""
    annotations = read_coco_keypoints(
        _coco(tmp_path), tmp_path, keypoint_indices=[0, 1]
    )
    assert annotations.schema.names == ("nose", "thorax")
    assert annotations.schema.skeleton == ((0, 1),), "the 1-2 edge lost an endpoint"
    assert len(annotations.frames[0].objects[0].keypoints) == 2


def test_a_short_keypoint_list_is_padded_rather_than_misaligned(
    tmp_path: Path,
) -> None:
    """A malformed annotation must not shift every later keypoint by one.

    Density is what the representation enforces, so the reader supplies the
    missing points as absent rather than handing back a short tuple.
    """
    path = _coco(
        tmp_path,
        annotations=[
            {"id": 11, "image_id": 1, "category_id": 1, "keypoints": [1.0, 2.0, 2]}
        ],
    )
    keypoints = read_coco_keypoints(path, tmp_path).frames[0].objects[0].keypoints
    assert len(keypoints) == 3
    assert keypoints[0].is_placed
    assert not keypoints[1].is_placed and not keypoints[2].is_placed


# --- round trip -------------------------------------------------------------


def test_writing_then_reading_gives_back_the_same_set(tmp_path: Path) -> None:
    """The contract that makes the pair usable as a boundary.

    Anything lost here is lost when mosaic hands an annotation set to a tool
    that speaks COCO -- which is how a ``.slp`` gets built, so this is load
    bearing rather than tidy.
    """
    from mosaic.core.annotations.writers import write_coco_keypoints

    original = read_coco_keypoints(_coco(tmp_path), tmp_path / "images")
    path = write_coco_keypoints(original, tmp_path / "round" / "ann.json")
    again = read_coco_keypoints(path, tmp_path / "images")

    assert again.schema == original.schema
    assert again.categories == original.categories
    assert [f.image_path for f in again] == [f.image_path for f in original]
    assert [f.width for f in again] == [f.width for f in original]
    for before, after in zip(original.frames, again.frames, strict=True):
        assert len(after.objects) == len(before.objects)
        for a, b in zip(before.objects, after.objects, strict=True):
            assert [k.visibility for k in a.keypoints] == [
                k.visibility for k in b.keypoints
            ]
            assert [k.is_placed for k in a.keypoints] == [
                k.is_placed for k in b.keypoints
            ]
            placed_before = [(k.x, k.y) for k in a.placed_keypoints]
            placed_after = [(k.x, k.y) for k in b.placed_keypoints]
            assert placed_before == placed_after


def test_an_unplaced_keypoint_survives_the_round_trip_as_unplaced(
    tmp_path: Path,
) -> None:
    """COCO has no NaN, so this is the one asymmetry, and it must be undone."""
    from mosaic.core.annotations.writers import write_coco_keypoints

    original = read_coco_keypoints(_coco(tmp_path), tmp_path / "images")
    path = write_coco_keypoints(original, tmp_path / "round" / "ann.json")
    tail = read_coco_keypoints(path, tmp_path).frames[0].objects[0].keypoints[2]

    assert tail.visibility == 0
    assert math.isnan(tail.x), "written as the origin, read back as absent"
