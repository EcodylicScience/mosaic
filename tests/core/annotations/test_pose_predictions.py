"""Pins the pose predictions format and the objects that a set accepts from it.

A ``mosaic-pose-predictions`` file contains one model's predictions on the frames of
one entry camera. These tests pin its canonical bytes, the reader's refusals, the
source refs that an accepted object records, and how a predicted object becomes an
annotation object of a set's pose.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path

import pytest

from mosaic.core.annotations.keypoint_map import KeypointMap, resolve_keypoint_map
from mosaic.core.annotations.model import Bbox, Keypoint, KeypointSchema
from mosaic.core.annotations.pose_annotations import PoseObject
from mosaic.core.annotations.pose_predictions import (
    FORMAT_VERSION,
    FramePredictions,
    PredictedFrame,
    PredictedObject,
    PredictedPose,
    accepted_object,
    parse_prediction_source_ref,
    pose_predictions_json_schema,
    pose_predictions_payload,
    prediction_source_ref,
    read_pose_predictions,
)
from mosaic.core.pipeline._utils import atomic_write

FRAMES_RUN = "kmeans-0123456789"
IMAGE_100 = f"media/frames/kmeans/{FRAMES_RUN}/s1/frame_000100.png"
IMAGE_200 = f"media/frames/kmeans/{FRAMES_RUN}/s1/frame_000200.png"
PREDICTION_RUN = "predict-frames.0.1-abcdef0123"
MODEL_POSE = PredictedPose(
    id=3,
    name="mouse",
    schema=KeypointSchema(
        names=("nose", "left_ear", "right_ear", "tail"),
        skeleton=((3, 1), (0, 1)),
        symmetries=((1, 2),),
    ),
)
MODEL_BOX = Bbox(10, 20, 50, 40)


def _point(x: float, y: float, score: float | None = None) -> Keypoint:
    return Keypoint(x=x, y=y, visibility=2, score=score)


def _predictions(
    *frames: PredictedFrame, pose: PredictedPose = MODEL_POSE
) -> FramePredictions:
    return FramePredictions(
        pose=pose,
        frames_run_id=FRAMES_RUN,
        group="",
        sequence="s1",
        camera="",
        frames=frames,
    )


def _frame(image: str, frame_index: int, *objects: PredictedObject) -> PredictedFrame:
    return PredictedFrame(
        image=image, frame_index=frame_index, width=1280, height=720, objects=objects
    )


def _collected() -> FramePredictions:
    """Two frames, given out of order: one with two objects, one with none."""
    found = _frame(
        IMAGE_100,
        100,
        PredictedObject(
            keypoints=(
                _point(12, 30, 0.88),
                Keypoint.absent(),
                _point(math.nan, 5.0, 0.5),
                _point(40.0, 41.0, math.inf),
            ),
            score=0.91,
            bbox=MODEL_BOX,
        ),
        PredictedObject(
            keypoints=(
                _point(1.5, 2.5),
                _point(3.5, 4.5, 0.25),
                _point(5.5, 6.5, 1),
                Keypoint.absent(),
            ),
            score=math.nan,
            track_id="resident",
        ),
    )
    return _predictions(_frame(IMAGE_200, 200), found)


def _document(predictions: FramePredictions) -> dict[str, object]:
    loaded: dict[str, object] = json.loads(pose_predictions_payload(predictions))
    return loaded


def _write(path: Path, payload: bytes) -> None:
    atomic_write(path, lambda temporary: temporary.write_bytes(payload))


# ---------------------------------------------------------------------- canonical bytes


def test_the_payload_is_canonical() -> None:
    """Frames by image, keys sorted, compact, floats, and non-finite values as null."""
    expected = (
        b'{"format":"mosaic-pose-predictions",'
        b'"frames":[{"frame_index":100,"height":720,'
        b'"image":"media/frames/kmeans/kmeans-0123456789/s1/frame_000100.png",'
        b'"objects":['
        b'{"bbox":{"height":40.0,"width":50.0,"x":10.0,"y":20.0},'
        b'"keypoints":[[12.0,30.0,0.88],null,null,[40.0,41.0,null]],'
        b'"score":0.91,"track_id":null},'
        b'{"bbox":null,'
        b'"keypoints":[[1.5,2.5,null],[3.5,4.5,0.25],[5.5,6.5,1.0],null],'
        b'"score":null,"track_id":"resident"}],'
        b'"width":1280},'
        b'{"frame_index":200,"height":720,'
        b'"image":"media/frames/kmeans/kmeans-0123456789/s1/frame_000200.png",'
        b'"objects":[],"width":1280}],'
        b'"pose":{"id":3,"keypoints":["nose","left_ear","right_ear","tail"],'
        b'"name":"mouse","skeleton":[[0,1],[1,3]],"symmetries":[[1,2]]},'
        b'"source":{"camera":"","frames_run_id":"kmeans-0123456789","group":"",'
        b'"sequence":"s1"},'
        b'"version":1}'
    )

    assert pose_predictions_payload(_collected()) == expected


def test_the_same_predictions_give_the_same_bytes() -> None:
    """The order a tool gives frames in and the direction of its edges do not count."""
    collected = _collected()
    flipped = replace(
        MODEL_POSE,
        schema=KeypointSchema(
            names=MODEL_POSE.schema.names,
            skeleton=((1, 0), (1, 3)),
            symmetries=MODEL_POSE.schema.symmetries,
        ),
    )
    reordered = _predictions(*reversed(collected.frames), pose=flipped)

    assert pose_predictions_payload(reordered) == pose_predictions_payload(collected)
    assert pose_predictions_payload(_collected()) == pose_predictions_payload(collected)


def test_the_order_of_objects_in_a_frame_is_the_instance_axis() -> None:
    collected = _collected()
    found = collected.frames[1]
    swapped = replace(found, objects=tuple(reversed(found.objects)))

    changed = replace(collected, frames=(collected.frames[0], swapped))

    assert pose_predictions_payload(changed) != pose_predictions_payload(collected)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_coordinate_is_an_unplaced_keypoint(value: float) -> None:
    obj = PredictedObject(
        keypoints=(
            _point(value, 1.0, 0.9),
            _point(1.0, value, 0.9),
            _point(1.0, 2.0, value),
            _point(3.0, 4.0, 0.5),
        ),
        score=value,
    )
    document = _document(_predictions(_frame(IMAGE_100, 100, obj)))

    assert document["frames"] == [
        {
            "frame_index": 100,
            "height": 720,
            "image": IMAGE_100,
            "width": 1280,
            "objects": [
                {
                    "bbox": None,
                    "keypoints": [None, None, [1.0, 2.0, None], [3.0, 4.0, 0.5]],
                    "score": None,
                    "track_id": None,
                }
            ],
        }
    ]


def test_a_box_with_a_non_finite_side_is_written_as_no_box() -> None:
    obj = PredictedObject(
        keypoints=(_point(1.0, 2.0),) * 4, bbox=Bbox(1.0, 2.0, math.inf, 4.0)
    )
    payload = pose_predictions_payload(_predictions(_frame(IMAGE_100, 100, obj)))

    assert b"Infinity" not in payload
    assert b'"bbox":null' in payload


# ------------------------------------------------------------------- written, read back


def test_what_is_written_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    _write(path, pose_predictions_payload(_collected()))

    reread = read_pose_predictions(path)

    assert pose_predictions_payload(reread) == path.read_bytes()
    assert reread.pose.id == 3 and reread.pose.name == "mouse"
    assert reread.pose.schema.skeleton == ((0, 1), (1, 3))
    assert reread.pose.schema.symmetries == ((1, 2),)
    assert (reread.frames_run_id, reread.group, reread.sequence, reread.camera) == (
        FRAMES_RUN,
        "",
        "s1",
        "",
    )
    found, empty = reread.frames
    assert (found.image, found.frame_index, found.width, found.height) == (
        IMAGE_100,
        100,
        1280,
        720,
    )
    first, second = found.objects
    assert first.score == 0.91 and first.track_id is None and first.bbox == MODEL_BOX
    nose, left_ear, right_ear, tail = first.keypoints
    assert (nose.x, nose.y, nose.visibility, nose.score) == (12.0, 30.0, 2, 0.88)
    assert left_ear.visibility == 0 and math.isnan(left_ear.x)
    assert right_ear.visibility == 0, "a non-finite coordinate reads back unplaced"
    assert (tail.visibility, tail.score) == (2, None)
    assert (
        second.score is None and second.track_id == "resident" and second.bbox is None
    )
    assert empty.image == IMAGE_200 and empty.objects == ()


def test_a_frame_where_nothing_was_found_survives(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    _write(path, pose_predictions_payload(_predictions(_frame(IMAGE_100, 100))))

    (frame,) = read_pose_predictions(path).frames

    assert frame.objects == ()
    assert _document(read_pose_predictions(path))["frames"] == [
        {
            "frame_index": 100,
            "height": 720,
            "image": IMAGE_100,
            "objects": [],
            "width": 1280,
        }
    ]


def test_a_pose_without_provenance_reads_back_unnamed(tmp_path: Path) -> None:
    unnamed = PredictedPose(
        id=None, name="", schema=KeypointSchema(names=("kp0", "kp1", "kp2", "kp3"))
    )
    path = tmp_path / "predictions.json"
    _write(path, pose_predictions_payload(_predictions(pose=unnamed)))

    pose = read_pose_predictions(path).pose

    assert pose == unnamed
    assert _document(_predictions(pose=unnamed))["pose"] == {
        "id": None,
        "keypoints": ["kp0", "kp1", "kp2", "kp3"],
        "name": "",
        "skeleton": [],
        "symmetries": [],
    }


@pytest.mark.parametrize(
    "addition",
    [{}, {"colors": [1]}],
    ids=["same-keys", "a-key-of-the-newer-version"],
)
def test_a_newer_version_is_refused(
    tmp_path: Path, addition: dict[str, object]
) -> None:
    document = _document(_collected())
    document["version"] = FORMAT_VERSION + 1
    document.update(addition)
    path = tmp_path / "predictions.json"
    _ = path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="Upgrade mosaic"):
        _ = read_pose_predictions(path)


@pytest.mark.parametrize(
    "change",
    [
        {"format": "mosaic-pose-annotations"},
        {"extra": 1},
        {"frames": [{"image": IMAGE_100}]},
    ],
    ids=["another-format", "an-unknown-key", "a-frame-missing-its-fields"],
)
def test_a_file_that_is_not_this_format_is_refused(
    tmp_path: Path, change: dict[str, object]
) -> None:
    document = _document(_collected())
    document.update(change)
    path = tmp_path / "predictions.json"
    _ = path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="is not a mosaic-pose-predictions file"):
        _ = read_pose_predictions(path)


@pytest.mark.parametrize(
    "edge",
    [[3, 1], [2, 2], [0, 4], [-1, 0]],
    ids=["higher-first", "one-keypoint", "past-the-last", "negative"],
)
def test_a_skeleton_edge_off_the_keypoints_is_refused(
    tmp_path: Path, edge: list[int]
) -> None:
    document = _document(_collected())
    pose = document["pose"]
    assert isinstance(pose, dict)
    pose["skeleton"] = [[0, 1], edge]
    path = tmp_path / "predictions.json"
    _ = path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match=re.escape(f"skeleton edge {tuple(edge)}")):
        _ = read_pose_predictions(path)


@pytest.mark.parametrize(
    "edge",
    [(2, 2), (0, 4), (-1, 0)],
    ids=["one-keypoint", "past-the-last", "negative"],
)
def test_predictions_with_a_skeleton_edge_off_the_keypoints_are_refused(
    edge: tuple[int, int],
) -> None:
    """The writer refuses every edge that the reader would refuse once oriented."""
    schema = KeypointSchema(names=MODEL_POSE.schema.names, skeleton=((0, 1), edge))

    with pytest.raises(ValueError, match=re.escape(f"skeleton edge {edge}")):
        _ = _predictions(pose=replace(MODEL_POSE, schema=schema))


def test_an_object_with_another_number_of_keypoints_is_refused() -> None:
    short = PredictedObject(keypoints=(_point(1.0, 2.0),))

    with pytest.raises(ValueError, match="1 keypoints"):
        _ = _predictions(_frame(IMAGE_100, 100, short))


def test_two_frames_naming_one_image_are_refused() -> None:
    with pytest.raises(ValueError, match="two frames name the image"):
        _ = _predictions(_frame(IMAGE_100, 100), _frame(IMAGE_100, 101))


def test_the_schema_documents_the_file() -> None:
    schema = pose_predictions_json_schema()

    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert sorted(properties) == sorted(_document(_collected()))
    definitions = schema["$defs"]
    assert isinstance(definitions, dict)
    assert "PredictedPoseRecord" in definitions


# -------------------------------------------------------------------------- source refs


def test_a_source_ref_names_the_run_and_the_instance() -> None:
    ref = prediction_source_ref(PREDICTION_RUN, 7)

    assert ref == f"{PREDICTION_RUN}#7"
    assert parse_prediction_source_ref(ref) == (PREDICTION_RUN, 7)
    assert parse_prediction_source_ref(prediction_source_ref("run", 0)) == ("run", 0)


@pytest.mark.parametrize(
    "ref",
    [
        "",
        PREDICTION_RUN,
        f"{PREDICTION_RUN}#",
        "#3",
        "run#x",
        "run#-1",
        "run#1.5",
        "run#1#2",
        "run# 3",
        "run#03",
        "run#\u0663",
    ],
)
def test_a_malformed_source_ref_names_nothing(ref: str) -> None:
    assert parse_prediction_source_ref(ref) is None


@pytest.mark.parametrize(("run_id", "instance"), [("", 0), ("run#1", 0), ("run", -1)])
def test_a_source_ref_that_would_not_parse_back_is_refused(
    run_id: str, instance: int
) -> None:
    with pytest.raises(ValueError, match="source ref"):
        _ = prediction_source_ref(run_id, instance)


# --------------------------------------------------------------------- accepted objects


ACCEPTED_REF = prediction_source_ref(PREDICTION_RUN, 0)
PREDICTED = PredictedObject(
    keypoints=(
        _point(12.0, 30.0, 0.88),
        _point(14.0, 28.0, 0.4),
        Keypoint.absent(),
        _point(40.0, 60.0),
    ),
    score=0.91,
    bbox=MODEL_BOX,
)


def _identity() -> KeypointMap:
    return resolve_keypoint_map(MODEL_POSE.schema, MODEL_POSE.schema.names)


def _points(obj: PoseObject) -> list[tuple[float, float, int, float | None] | None]:
    """Each keypoint as ``(x, y, visibility, score)``, or ``None`` when absent."""
    return [
        (point.x, point.y, point.visibility, point.score) if point.visibility else None
        for point in obj.keypoints
    ]


def test_an_identity_map_accepts_every_placed_keypoint_without_its_score() -> None:
    accepted = accepted_object(
        PREDICTED, _identity(), pose_id=21, source_ref=ACCEPTED_REF
    )

    assert _points(accepted) == [
        (12.0, 30.0, 2, None),
        (14.0, 28.0, 2, None),
        None,
        (40.0, 60.0, 2, None),
    ]
    assert accepted.pose_id == 21
    assert accepted.origin == "model"
    assert accepted.source_ref == ACCEPTED_REF
    assert accepted.alias_id is None
    assert accepted.track_id == ""
    assert accepted.bbox is None, "the set derives the box from the keypoints"


def test_another_map_places_each_source_keypoint_at_its_target() -> None:
    target = KeypointSchema(names=("snout", "tail", "left_ear", "whisker"))
    keypoint_map = resolve_keypoint_map(
        target, MODEL_POSE.schema.names, overrides={"snout": "nose"}
    )

    accepted = accepted_object(
        PREDICTED, keypoint_map, pose_id=4, source_ref=ACCEPTED_REF
    )

    assert _points(accepted) == [
        (12.0, 30.0, 2, None),
        (40.0, 60.0, 2, None),
        (14.0, 28.0, 2, None),
        None,
    ]
    assert accepted.bbox is None


def test_a_keypoint_below_the_threshold_is_absent_and_an_unscored_one_passes() -> None:
    accepted = accepted_object(
        PREDICTED,
        _identity(),
        pose_id=21,
        source_ref=ACCEPTED_REF,
        min_keypoint_score=0.5,
    )

    assert _points(accepted) == [
        (12.0, 30.0, 2, None),
        None,
        None,
        (40.0, 60.0, 2, None),
    ]


def test_a_keypoint_at_the_threshold_passes() -> None:
    accepted = accepted_object(
        PREDICTED,
        _identity(),
        pose_id=21,
        source_ref=ACCEPTED_REF,
        min_keypoint_score=0.88,
    )

    assert _points(accepted)[0] == (12.0, 30.0, 2, None)
    assert _points(accepted)[1] is None


def test_a_track_name_becomes_the_object_track() -> None:
    tracked = replace(PREDICTED, track_id="resident")

    accepted = accepted_object(
        tracked, _identity(), pose_id=21, source_ref=ACCEPTED_REF
    )

    assert accepted.track_id == "resident"


def test_an_object_with_no_mapped_placed_keypoint_is_refused() -> None:
    keypoint_map = resolve_keypoint_map(
        KeypointSchema(names=("nose", "tail")), MODEL_POSE.schema.names
    )
    ears_only = PredictedObject(
        keypoints=(
            Keypoint.absent(),
            _point(1.0, 1.0, 0.9),
            _point(2.0, 1.0, 0.9),
            Keypoint.absent(),
        )
    )

    with pytest.raises(ValueError, match=re.escape(ACCEPTED_REF)) as refused:
        _ = accepted_object(
            ears_only, keypoint_map, pose_id=21, source_ref=ACCEPTED_REF
        )

    assert "no placed keypoint" in str(refused.value)


def test_an_object_with_every_mapped_keypoint_below_the_threshold_is_refused() -> None:
    faint = PredictedObject(
        keypoints=(
            _point(1.0, 1.0, 0.2),
            Keypoint.absent(),
            Keypoint.absent(),
            _point(2.0, 2.0, 0.3),
        )
    )

    with pytest.raises(ValueError, match=re.escape(ACCEPTED_REF)) as refused:
        _ = accepted_object(
            faint,
            _identity(),
            pose_id=21,
            source_ref=ACCEPTED_REF,
            min_keypoint_score=0.5,
        )

    assert "below min_keypoint_score" in str(refused.value)
