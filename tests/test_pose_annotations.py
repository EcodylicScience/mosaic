"""An annotation set's saved state keeps everything, and says the same thing twice.

A revision of the ``keypoints`` series is the annotator's whole state -- every
pose, alias, mirror pair and frame, finished or not -- in mosaic's own format,
with a COCO Keypoints export beside it for other tools. These tests pin the three
promises that makes: one state is one sequence of bytes, what is written reads
back, and the export is correct for a reader that knows nothing about mosaic.
Narrowing a state into what one training run reads is tested here too, because
the rules that keep a training set honest live there.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path

import pytest

from mosaic.core.annotations.bbox import BboxPolicy, derived_bbox
from mosaic.core.annotations.model import Bbox, Keypoint, KeypointSchema
from mosaic.core.annotations.narrow import narrow_pose_sets
from mosaic.core.annotations.pose_annotations import (
    FORMAT_VERSION,
    Alias,
    PoseAnnotationSet,
    PoseDefinition,
    PoseObject,
    pose_annotations_payload,
    read_pose_annotations,
)
from mosaic.core.annotations.readers.coco import read_coco_keypoints
from mosaic.core.json_value import JsonValue
from mosaic.core.annotations.writers.coco import (
    coco_keypoints_document,
    coco_keypoints_payload,
)
from tests.helpers import MOUSE, pose_frame, pose_object, pose_set

RESIDENT = Alias(id=12, name="resident")
INTRUDER = Alias(id=13, name="intruder")
MOUSE_WITH_ALIASES = replace(
    MOUSE,
    schema=KeypointSchema(
        names=("nose", "left_ear", "right_ear", "tail"),
        skeleton=((0, 3), (0, 1), (0, 2)),
        symmetries=((1, 2),),
    ),
    aliases=(INTRUDER, RESIDENT),
)
CRICKET = PoseDefinition(
    id=8,
    name="cricket",
    schema=KeypointSchema(names=("head", "tail"), skeleton=((0, 1),)),
)
PADDING = BboxPolicy(method="isotropic", pad_frac_of_body=0.3, min_pad_px=20.0)


DRAWN = Bbox(10.0, 10.0, 20.0, 35.0)


def _mouse(
    *, alias: Alias | None, bbox: Bbox | None = None, x: float = 20.0
) -> PoseObject:
    """A mouse whose keypoint hull is 10 by 25 pixels, inside a 64 by 48 frame."""
    return pose_object(
        (x, 20.0), (x - 5.0, 15.0), (x + 5.0, 15.0), (x, 40.0),
        alias_id=None if alias is None else alias.id,
        bbox=bbox,
    )  # fmt: skip


def _cricket(x: float = 40.0) -> PoseObject:
    return pose_object((x, 10.0), (x + 12.0, 16.0), pose_id=CRICKET.id, origin="model")


def _state() -> PoseAnnotationSet:
    """Two poses, both aliases, a drawn box, an unfinished frame and an empty one."""
    frames = [
        pose_frame(
            "frames/a/frame_000001.png",
            _mouse(alias=RESIDENT, bbox=DRAWN),
            _cricket(),
            frame_index=1,
        ),
        pose_frame("frames/a/frame_000002.png", _mouse(alias=INTRUDER), frame_index=2),
        pose_frame("frames/a/frame_000003.png", frame_index=3),
        pose_frame(
            "frames/a/frame_000004.png",
            _mouse(alias=None, x=30.0),
            frame_index=4,
            usable=False,
        ),
    ]
    return pose_set(frames, poses=(MOUSE_WITH_ALIASES, CRICKET), bbox_policy=PADDING)


def _document(state: PoseAnnotationSet) -> dict[str, object]:
    loaded: dict[str, object] = json.loads(pose_annotations_payload(state))
    return loaded


# ---------------------------------------------------------------- one state, one bytes

GOLDEN = Path(__file__).parent / "data"
UPDATE_ENV = "MOSAIC_UPDATE_GOLDEN"


@pytest.mark.parametrize(
    ("golden", "encode"),
    [
        ("pose_annotations_golden.json", pose_annotations_payload),
        ("coco_export_golden.json", coco_keypoints_payload),
    ],
    ids=["saved-state", "coco-export"],
)
def test_the_files_are_exactly_what_they_were(golden: str, encode: object) -> None:
    """Both formats are read outside mosaic, so an unplanned change is a defect.

    The golden is kept indented so a diff reads; the bytes a revision holds are
    its canonical form, and that is what is compared. Regenerate after a
    deliberate change with ``MOSAIC_UPDATE_GOLDEN=1`` and read the diff.
    """
    assert callable(encode)
    written = encode(_state())
    assert isinstance(written, bytes)
    path = GOLDEN / golden
    if os.environ.get(UPDATE_ENV) == "1":
        document = json.loads(written)
        _ = path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    canonical = json.dumps(
        json.loads(path.read_text()), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    assert written == canonical


def test_the_order_a_state_was_collected_in_is_not_a_change() -> None:
    state = _state()
    shuffled = PoseAnnotationSet(
        poses=tuple(
            replace(
                pose,
                aliases=tuple(reversed(pose.aliases)),
                schema=KeypointSchema(
                    names=pose.schema.names,
                    skeleton=tuple(reversed(pose.schema.skeleton)),
                    symmetries=pose.schema.symmetries,
                ),
            )
            for pose in reversed(state.poses)
        ),
        frames=tuple(reversed(state.frames)),
        bbox_policy=state.bbox_policy,
    )

    assert pose_annotations_payload(shuffled) == pose_annotations_payload(state)


def test_the_order_of_objects_in_a_frame_is_the_instance_axis() -> None:
    state = _state()
    first = state.frames[0]
    swapped = replace(first, objects=tuple(reversed(first.objects)))

    changed = replace(state, frames=(swapped, *state.frames[1:]))

    assert pose_annotations_payload(changed) != pose_annotations_payload(state)


def test_an_integer_coordinate_is_written_as_the_float_it_is() -> None:
    whole = pose_set([pose_frame("f.png", pose_object((10, 20), (30, 40)))])
    as_float = pose_set([pose_frame("f.png", pose_object((10.0, 20.0), (30.0, 40.0)))])

    assert pose_annotations_payload(whole) == pose_annotations_payload(as_float)
    assert b"10.0" in pose_annotations_payload(whole)


def test_the_padding_is_part_of_the_state() -> None:
    """The annotator saw boxes under it, so a change to it changes what was annotated."""
    state = _state()
    wider = replace(state, bbox_policy=PADDING.model_copy(update={"min_pad_px": 40.0}))

    assert pose_annotations_payload(wider) != pose_annotations_payload(state)


# ------------------------------------------------------------------- written, read back


def test_what_is_written_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "annotations.mosaic.json"
    _ = path.write_bytes(pose_annotations_payload(_state()))

    reread = read_pose_annotations(path, tmp_path)

    assert pose_annotations_payload(reread) == path.read_bytes()
    mouse = reread.pose(MOUSE.id)
    assert mouse.schema.symmetries == ((1, 2),)
    assert mouse.aliases == (RESIDENT, INTRUDER), "sorted by id"
    frames = {frame.frame_index: frame for frame in reread.frames}
    assert frames[1].objects[0].bbox == DRAWN
    assert frames[2].objects[0].bbox is None, "derived stays derived"
    assert frames[1].objects[1].origin == "model"
    assert frames[3].objects == () and frames[3].usable
    assert not frames[4].usable and frames[4].objects[0].alias_id is None
    assert reread.bbox_policy == PADDING
    assert reread.resolve(frames[1]) == tmp_path / "frames/a/frame_000001.png"


def test_an_unplaced_keypoint_reads_back_unplaced(tmp_path: Path) -> None:
    obj = PoseObject(
        pose_id=MOUSE.id,
        keypoints=(Keypoint(x=1.0, y=2.0, visibility=1), Keypoint.absent()),
    )
    path = tmp_path / "state.json"
    _ = path.write_bytes(pose_annotations_payload(pose_set([pose_frame("f.png", obj)])))

    first, second = read_pose_annotations(path).frames[0].objects[0].keypoints

    assert (first.x, first.y, first.visibility) == (1.0, 2.0, 1)
    assert second.visibility == 0 and math.isnan(second.x)


def test_a_newer_version_is_refused_rather_than_misread(tmp_path: Path) -> None:
    document = _document(_state())
    document["version"] = FORMAT_VERSION + 1
    path = tmp_path / "state.json"
    _ = path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="Upgrade mosaic"):
        _ = read_pose_annotations(path)


@pytest.mark.parametrize(
    "change",
    [{"format": "coco"}, {"extra": 1}],
    ids=["another-format", "an-unknown-key"],
)
def test_a_file_that_is_not_this_format_is_refused(
    tmp_path: Path, change: dict[str, object]
) -> None:
    document = _document(_state())
    document.update(change)
    path = tmp_path / "state.json"
    _ = path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="is not a mosaic-pose-annotations file"):
        _ = read_pose_annotations(path)


# ------------------------------------------------------------ a state the model refuses


def test_an_object_of_an_undeclared_pose_is_refused() -> None:
    with pytest.raises(ValueError, match="does not declare"):
        _ = pose_set([pose_frame("f.png", _cricket())])


def test_an_object_with_the_wrong_number_of_keypoints_is_refused() -> None:
    with pytest.raises(ValueError, match="declares 2"):
        _ = pose_set([pose_frame("f.png", pose_object((1.0, 1.0)))])


def test_an_alias_of_another_pose_is_refused() -> None:
    stray = pose_object((1.0, 1.0), (2.0, 2.0), alias_id=RESIDENT.id)
    with pytest.raises(ValueError, match="alias 12"):
        _ = pose_set([pose_frame("f.png", stray)])


@pytest.mark.parametrize(
    "poses",
    [
        (MOUSE, replace(CRICKET, id=MOUSE.id)),
        (MOUSE_WITH_ALIASES, replace(CRICKET, aliases=(RESIDENT,))),
    ],
    ids=["pose-id", "alias-id"],
)
def test_one_id_naming_two_things_is_refused(poses: tuple[PoseDefinition, ...]) -> None:
    with pytest.raises(ValueError, match="twice|two poses"):
        _ = pose_set([], poses=poses)


def test_a_placed_keypoint_off_the_number_line_is_refused() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _ = pose_set([pose_frame("f.png", pose_object((math.nan, 1.0), (2.0, 2.0)))])


def test_two_frames_naming_one_image_are_refused() -> None:
    frame = pose_frame("f.png")
    with pytest.raises(ValueError, match="two frames name"):
        _ = pose_annotations_payload(pose_set([frame, frame]))


# ----------------------------------------------------------------------- mirror pairs


def test_a_mirror_pair_becomes_the_flip_permutation() -> None:
    assert MOUSE_WITH_ALIASES.schema.flip_idx == [0, 2, 1, 3]


@pytest.mark.parametrize(
    "pairs",
    [((1, 4),), ((2, 1),), ((1, 1),), ((0, 1), (1, 2))],
    ids=["out-of-range", "higher-first", "self", "two-mirrors"],
)
def test_a_pair_that_cannot_be_a_mirror_is_refused(
    pairs: tuple[tuple[int, int], ...],
) -> None:
    with pytest.raises(ValueError):
        _ = KeypointSchema(names=("a", "b", "c", "d"), symmetries=pairs)


def test_a_subset_keeps_a_pair_only_when_both_ends_survive() -> None:
    schema = MOUSE_WITH_ALIASES.schema

    assert schema.subset([0, 1, 3]).symmetries == ()
    assert schema.subset([2, 1, 0]).symmetries == ((0, 1),), "rewritten lower first"


# --------------------------------------------------------------------- the COCO export


def test_the_export_holds_finished_frames_only() -> None:
    document = coco_keypoints_document(_state())

    frames = [image["frame_num"] for image in _records(document, "images")]
    assert frames == [1, 2, 3], "the unfinished frame 4 is not something to train on"
    empty = [image for image in _records(document, "images") if image["frame_num"] == 3]
    assert not [
        ann
        for ann in _records(document, "annotations")
        if ann["image_id"] == empty[0]["id"]
    ], "a finished empty frame is a true negative"


def test_the_export_names_poses_as_categories_with_positional_ids() -> None:
    categories = _records(coco_keypoints_document(_state()), "categories")

    assert [(c["id"], c["name"]) for c in categories] == [(1, "mouse"), (2, "cricket")]
    assert [c["attributes"] for c in categories] == [{"pose_id": 3}, {"pose_id": 8}]
    assert categories[0]["skeleton"] == [[1, 2], [1, 3], [1, 4]], "one-based"


def test_the_export_carries_what_mosaic_knows_as_scalar_attributes() -> None:
    records = _records(coco_keypoints_document(_state()), "annotations")
    drawn, cricket, derived = records

    assert drawn["attributes"] == {
        "alias": "resident", "alias_id": 12, "bbox_source": "drawn", "origin": "human",
    }  # fmt: skip
    assert drawn["bbox"] == [DRAWN.x, DRAWN.y, DRAWN.width, DRAWN.height]
    assert cricket["category_id"] == 2 and _field(cricket, "attributes") == {
        "bbox_source": "derived",
        "origin": "model",
    }
    assert _field(derived, "attributes")["bbox_source"] == "derived"


def test_a_derived_box_is_the_padded_one_the_annotator_saw() -> None:
    state = _state()
    records = _records(coco_keypoints_document(state), "annotations")
    obj = state.frames[1].objects[0]

    expected = derived_bbox(obj.keypoints, 64, 48, PADDING)
    hull_width = 10.0
    box = records[2]["bbox"]
    assert box == [expected.x, expected.y, expected.width, expected.height]
    assert expected.width > hull_width, "padded, not the bare hull"


def test_a_coco_reader_takes_the_export_as_it_is(tmp_path: Path) -> None:
    path = tmp_path / "annotations.coco.json"
    _ = path.write_text(json.dumps(coco_keypoints_document(_state())))

    mice = read_coco_keypoints(path, tmp_path, category_name="mouse")
    crickets = read_coco_keypoints(path, tmp_path, category_name="cricket")

    assert mice.schema.names == MOUSE_WITH_ALIASES.schema.names
    assert [frame.frame_index for frame in mice.frames] == [1, 2, 3]
    assert sum(len(frame.objects) for frame in mice.frames) == 2
    assert sum(len(frame.objects) for frame in crickets.frames) == 1


def _records(document: dict[str, JsonValue], key: str) -> list[dict[str, JsonValue]]:
    value = document[key]
    assert isinstance(value, list)
    return [record for record in value if isinstance(record, dict)]


def _field(record: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    value = record[key]
    assert isinstance(value, dict)
    return value


# ---------------------------------------------------------------------------- narrowing


def test_a_pose_is_chosen_by_id_or_by_name() -> None:
    by_id = narrow_pose_sets({"s": _state()}, pose=CRICKET.id).sets["s"]
    by_name = narrow_pose_sets({"s": _state()}, pose="cricket").sets["s"]

    assert by_id.schema == by_name.schema == CRICKET.schema
    assert by_id.categories == ("cricket",)


@pytest.mark.parametrize("pose", [CRICKET.id, "cricket"], ids=["by-id", "by-name"])
def test_the_narrowing_names_the_pose_it_chose(pose: int | str) -> None:
    assert narrow_pose_sets({"s": _state()}, pose=pose).pose == CRICKET


def test_a_set_of_one_pose_is_narrowed_to_it_unnamed() -> None:
    finished = pose_set([pose_frame("f.png", pose_object((1.0, 2.0), (3.0, 4.0)))])

    assert narrow_pose_sets({"s": finished}).pose == MOUSE


@pytest.mark.parametrize(
    "other",
    [replace(MOUSE, id=MOUSE.id + 1, name="rat"), CRICKET],
    ids=["same-layout", "other-layout"],
)
def test_sets_that_choose_different_poses_are_refused(other: PoseDefinition) -> None:
    """The different pose is named, whether or not its layout differs too.

    Two poses sharing a layout would otherwise train as one class. Two poses of
    different layouts differ in layout because they are different poses, so the
    pose is the cause the refusal names. Each set holds one pose, so each chooses
    its own.
    """
    mice = pose_set([pose_frame("m.png", pose_object((1.0, 2.0), (3.0, 4.0)))])
    others = pose_set(
        [pose_frame("o.png", pose_object((1.0, 2.0), (3.0, 4.0), pose_id=other.id))],
        poses=(other,),
    )

    with pytest.raises(ValueError, match="One model trains one pose") as caught:
        _ = narrow_pose_sets({"others": others, "mice": mice})

    message = str(caught.value)
    assert f"set mice chose pose {MOUSE.id} ({MOUSE.name})" in message
    assert f"set others chose pose {other.id} ({other.name})" in message


def test_a_state_with_two_poses_says_which_to_name() -> None:
    with pytest.raises(ValueError, match="name the one to train"):
        _ = narrow_pose_sets({"s": _state()})


def test_a_set_without_the_pose_is_refused() -> None:
    with pytest.raises(ValueError, match="holds no pose 'rat'"):
        _ = narrow_pose_sets({"s": _state()}, pose="rat")


def test_only_finished_frames_are_kept_and_a_frame_without_the_pose_stays() -> None:
    narrowed = narrow_pose_sets({"s": _state()}, pose=CRICKET.id).sets["s"]

    assert [frame.frame_index for frame in narrowed.frames] == [1, 2, 3]
    assert [len(frame.objects) for frame in narrowed.frames] == [1, 0, 0], (
        "frames 2 and 3 are finished and hold no cricket: negatives"
    )


def test_aliases_become_sorted_classes() -> None:
    finished = replace(_state(), frames=_state().frames[:3])
    narrowed = narrow_pose_sets({"s": finished}, pose="mouse", class_by="alias").sets[
        "s"
    ]

    assert narrowed.categories == ("intruder", "resident")
    assert [obj.category for frame in narrowed.frames for obj in frame.objects] == [
        "resident",
        "intruder",
    ]


def test_an_object_without_an_alias_is_refused_rather_than_left_unlabelled() -> None:
    no_alias = pose_frame("x.png", _mouse(alias=None))
    state = replace(_state(), frames=(*_state().frames, no_alias))

    with pytest.raises(ValueError, match="carry none: 1 in s"):
        _ = narrow_pose_sets({"s": state}, pose="mouse", class_by="alias")


def test_an_alias_renamed_between_saves_is_one_class_named_by_the_first() -> None:
    finished = replace(_state(), frames=_state().frames[:3])
    renamed = replace(
        finished,
        poses=(
            replace(MOUSE_WITH_ALIASES, aliases=(INTRUDER, Alias(12, "host"))),
            CRICKET,
        ),
    )

    narrowed = narrow_pose_sets(
        {"first": finished, "second": renamed}, pose="mouse", class_by="alias"
    )

    assert narrowed.sets["second"].categories == ("intruder", "resident")


def test_two_aliases_sharing_a_name_cannot_be_two_classes() -> None:
    finished = replace(_state(), frames=_state().frames[:3])
    clash = replace(
        finished,
        poses=(
            replace(MOUSE_WITH_ALIASES, aliases=(Alias(13, "resident"), RESIDENT)),
            CRICKET,
        ),
    )

    with pytest.raises(ValueError, match="share a name"):
        _ = narrow_pose_sets({"s": clash}, pose="mouse", class_by="alias")


def test_an_alias_can_be_the_track_a_sleap_identity_model_learns() -> None:
    narrowed = narrow_pose_sets({"s": _state()}, pose="mouse", track_by="alias").sets[
        "s"
    ]

    assert [obj.track_id for frame in narrowed.frames for obj in frame.objects] == [
        "resident",
        "intruder",
    ]


def test_every_box_is_explicit_and_a_drawn_one_is_kept() -> None:
    state = _state()
    own = narrow_pose_sets({"s": state}, pose="mouse").sets["s"]
    tight = narrow_pose_sets(
        {"s": state}, pose="mouse", bbox=BboxPolicy(method="tight", margin=0.0)
    ).sets["s"]

    assert own.frames[0].objects[0].bbox == tight.frames[0].objects[0].bbox == DRAWN
    keypoints = state.frames[1].objects[0].keypoints
    assert own.frames[1].objects[0].bbox == derived_bbox(keypoints, 64, 48, PADDING)
    hull = tight.frames[1].objects[0].bbox
    assert hull is not None
    assert (hull.x, hull.y, hull.width, hull.height) == pytest.approx((15, 15, 10, 25))


def test_a_state_with_nothing_finished_is_refused() -> None:
    unfinished = pose_set([pose_frame("f.png", usable=False)])

    with pytest.raises(ValueError, match="finished"):
        _ = narrow_pose_sets({"s": unfinished})
