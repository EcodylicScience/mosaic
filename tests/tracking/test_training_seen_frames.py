"""The frames that a trained model trained or validated on.

A preparation records where it placed each frame in ``split.csv``. The seen
frames of a model are the rows that its target trains or validates on. A
preparation without that record counts every usable frame of the revisions it
consumed. A fine-tune adds the frames of its base model, and a model of two
trainings unites theirs. A link that the walk cannot read leaves the answer
incomplete, with the reason.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from mosaic.core.annotations.model import KeypointSchema
from mosaic.core.annotations.pose_annotations import (
    PoseDefinition,
    PoseFrame,
    PoseObject,
)
from mosaic.core.annotations.projection import write_keypoint_set_revision
from mosaic.core.dataset import Dataset, open_dataset
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.models import (
    SPLIT_RECORD_FILENAME,
    model_index_path,
    model_run_root,
)
from mosaic.core.pipeline.ops import run_op
from mosaic.tracking import register_ops
from mosaic.tracking.ops._common import resolve_training_data
from mosaic.tracking.training_provenance import (
    SeenFrames,
    seen_frames,
    seen_frames_of_model,
    training_provenance,
)
from tests.helpers import (
    MOUSE,
    assert_tree_unchanged,
    extract_entry_camera,
    make_dataset,
    pose_frame,
    pose_object,
    pose_set,
    record_slp_writes,
    register_trained_model,
    revision_file,
    tree_snapshot,
)

register_ops()

PREPARE: Final = "prepare-training-data"
POSE: Final = "train-pose"
SLEAP: Final = "train-sleap"

TRAINED: Final = "frames"
"""The set that most preparations train on."""
HELD: Final = "held"
"""The set that most preparations hold out."""
OTHER: Final = "other"
"""A set of a recording that the other sets do not contain."""

EARS: Final = PoseDefinition(
    id=5,
    name="eared",
    schema=KeypointSchema(names=("nose", "left_ear", "right_ear")),
)
INSTANTS: Final = range(100, 116)
"""The frame indices that the frames run extracted from every entry camera."""
SPLIT: Final = (0.1, 0.1, 0.8)
"""Fractions that place frames in every split.

The small-set guard fills the training and the validation split, and the test
split takes most of the rest.
"""

MODEL: Final = "train-pose.0.2-feedfacade"
BASE: Final = "train-pose.0.2-ba5eba5e00"
MIDDLE: Final = "train-pose.0.2-0dd0dd0dd0"
TUNED: Final = "train-pose.0.2-7a11ed7a11"
MISSING: Final = "train-pose.0.2-dead0dead0"
CENTROID: Final = "train-sleap.0.3-cec0cec0ce"
INSTANCE: Final = "train-sleap.0.3-1ce01ce01c"


def _extract(
    ds: Dataset, group: str, sequence: str, camera: str = ""
) -> dict[int, Path]:
    return extract_entry_camera(
        ds,
        run_id="uniform-1",
        group=group,
        sequence=sequence,
        camera=camera,
        sizes={index: (64, 48) for index in INSTANTS},
    )


def _mouse(index: int) -> PoseObject:
    return pose_object((10.0 + index - INSTANTS[0], 12.0), (30.0, 20.0))


def _save(ds: Dataset, set_key: str, frames: list[PoseFrame]) -> None:
    state = pose_set(frames, poses=(MOUSE, EARS), image_root=ds.base_dir)
    _ = write_keypoint_set_revision(ds, set_key=set_key, annotations=state, origin={})


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    """Return a dataset of three extracted recordings and three saved sets.

    ``frames`` holds frames 100 to 115 of ``A/s1`` and of camera ``left`` of
    ``B/s2``, a mouse on every frame and an eared animal on those of ``B/s2``.
    Frame 115 of ``B/s2`` is unfinished. ``held`` holds frames 100 and 101 of
    ``A/s1``, and ``other`` holds frames 100 to 115 of ``C/s3``.
    """
    dataset = make_dataset(tmp_path / "ds", name="seen")
    first = _extract(dataset, "A", "s1")
    second = _extract(dataset, "B", "s2", "left")
    third = _extract(dataset, "C", "s3")
    eared = pose_object((40.0, 20.0), (36.0, 16.0), (44.0, 16.0), pose_id=EARS.id)
    trained = [
        pose_frame(image, _mouse(index), sequence="s1", frame_index=index)
        for index, image in first.items()
    ]
    trained.extend(
        pose_frame(
            image,
            _mouse(index),
            eared,
            sequence="s2",
            frame_index=index,
            usable=index != INSTANTS[-1],
        )
        for index, image in second.items()
    )
    _save(dataset, TRAINED, trained)
    _save(
        dataset,
        HELD,
        [
            pose_frame(first[index], _mouse(index), sequence="s1", frame_index=index)
            for index in (100, 101)
        ],
    )
    _save(
        dataset,
        OTHER,
        [
            pose_frame(image, _mouse(index), sequence="s3", frame_index=index)
            for index, image in third.items()
        ],
    )
    return dataset


def _prepare(ds: Dataset, target: str = "yolo-pose", **overrides: object) -> str:
    """A preparation of ``frames`` that holds out ``held``, unless *overrides* say."""
    params: dict[str, object] = {
        "sets": [{"set_key": TRAINED}],
        "holdout": [{"set_key": HELD}],
        "pose": MOUSE.name,
        "target": target,
        "split_by": "frame",
        "split": SPLIT,
        **overrides,
    }
    return run_op(ds, PREPARE, params)


def _prepare_other(ds: Dataset, target: str = "yolo-pose") -> str:
    """A preparation of ``other``, which shares no frame with ``frames``."""
    return _prepare(ds, target, sets=[{"set_key": OTHER}], holdout=[])


def _data(ds: Dataset, prepared: str) -> Path:
    """What a trainer is handed from the preparation *prepared*."""
    return resolve_training_data(ds, prepared)


def _train(
    ds: Dataset, kind: str, run_id: str, data: Path | None, *, base_run_id: str = ""
) -> str:
    """Register *run_id* as a model of *kind* trained on *data*, and return it."""
    weights = model_run_root(ds, kind, run_id) / "best.pt"
    weights.parent.mkdir(parents=True)
    _ = weights.write_bytes(b"weights")
    register_trained_model(
        ds, kind, run_id, weights, data_path=data, base_run_id=base_run_id
    )
    return run_id


def _record(ds: Dataset, prepared: str) -> Path:
    return model_run_root(ds, PREPARE, prepared) / SPLIT_RECORD_FILENAME


def _rows(ds: Dataset, prepared: str) -> list[dict[str, str]]:
    return index_records(
        pd.read_csv(_record(ds, prepared), dtype="string", keep_default_na=False)
    )


def _keys(ds: Dataset, prepared: str, *splits: str) -> frozenset[str]:
    """The frame keys of the record's rows in *splits*, or of every row without any."""
    return frozenset(
        row["frame_key"]
        for row in _rows(ds, prepared)
        if not splits or row["split"] in splits
    )


def _revision(ds: Dataset, set_key: str) -> Path:
    """The file of the first revision of *set_key*."""
    return ds.get_root("labels_raw") / "keypoints" / set_key / revision_file(1)


def _rewrite_preparations(ds: Dataset, column: str, value: str) -> None:
    """Set the cell *column* of every preparation's index row to *value*."""
    path = model_index_path(ds, PREPARE)
    written = pd.read_csv(path, dtype=str, keep_default_na=False)
    written[column] = value
    written.to_csv(path, index=False)


# ------------------------------------------------------------- the split record


@pytest.mark.parametrize(
    ("kind", "target", "seen"),
    [
        pytest.param(POSE, "yolo-pose", ("train", "valid"), id="yolo-pose"),
        pytest.param("train-points", "polo", ("train", "valid"), id="polo"),
        pytest.param(SLEAP, "sleap", ("train", "valid"), id="sleap"),
        pytest.param(
            "train-litpose", "litpose", ("train", "valid", "test"), id="litpose"
        ),
    ],
)
def test_a_model_saw_the_rows_that_its_target_trains_or_validates_on(
    ds: Dataset,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    target: str,
    seen: tuple[str, ...],
) -> None:
    """Validation picks the weights of a tree target, and SLEAP reads two files.

    Lightning Pose redraws its split from every frame that the hold-out leaves.
    """
    _ = record_slp_writes(monkeypatch)
    prepared = _prepare(ds, target)
    model = _train(ds, kind, f"{kind}.0.1-feedfacade", _data(ds, prepared))

    found = seen_frames(ds, kind, model)

    assert {row["split"] for row in _rows(ds, prepared)} == {
        "train",
        "valid",
        "test",
        "excluded",
    }, "the record places frames in every split"
    assert found == SeenFrames(
        keys=_keys(ds, prepared, *seen), complete=True, reason=""
    )


def test_a_split_record_is_read_without_the_revisions(ds: Dataset) -> None:
    """The record names every frame, so a revision that is gone is not read."""
    prepared = _prepare(ds)
    model = _train(ds, POSE, MODEL, _data(ds, prepared))
    _revision(ds, TRAINED).unlink()

    found = seen_frames(ds, POSE, model)

    assert training_provenance(ds, POSE, model).stopped_at, "the revision is gone"
    assert found == SeenFrames(
        keys=_keys(ds, prepared, "train", "valid"), complete=True, reason=""
    )


@pytest.mark.parametrize(
    "pose_recorded", [True, False], ids=["pose-recorded", "pose-unknown"]
)
def test_a_preparation_without_a_split_record_counts_every_consumed_usable_frame(
    ds: Dataset, pose_recorded: bool
) -> None:
    """A preparation older than 0.3 wrote no split record.

    Every usable frame of the revisions that it consumed counts, the test frames
    and the frames that a hold-out removes among them. A row without its pose
    counts the usable frames of every pose.
    """
    prepared = _prepare(ds)
    model = _train(ds, POSE, MODEL, _data(ds, prepared))
    every_row = _keys(ds, prepared)
    _record(ds, prepared).unlink()
    if not pose_recorded:
        _rewrite_preparations(ds, "pose_id", "")

    found = seen_frames(ds, POSE, model)

    assert len(every_row) == 2 * len(INSTANTS) - 1, "the unfinished frame has no row"
    assert found == SeenFrames(keys=every_row, complete=True, reason="")


# ------------------------------------------------------------------ base models


def test_a_fine_tune_unites_the_frames_of_its_base_models(ds: Dataset) -> None:
    """The walk follows each base model in turn, two links back here."""
    first = _prepare(ds)
    second = _prepare_other(ds)
    _ = _train(ds, POSE, BASE, _data(ds, first))
    _ = _train(ds, POSE, MIDDLE, _data(ds, second), base_run_id=BASE)
    tuned = _train(ds, POSE, TUNED, _data(ds, second), base_run_id=MIDDLE)
    from_base = _keys(ds, first, "train", "valid")
    from_others = _keys(ds, second, "train", "valid")

    found = seen_frames(ds, POSE, tuned)

    assert not from_base & from_others, "only the first base saw these frames"
    assert found == SeenFrames(keys=from_base | from_others, complete=True, reason="")


def test_a_chain_of_base_models_that_returns_to_a_model_is_refused(
    ds: Dataset,
) -> None:
    data = _data(ds, _prepare(ds))
    _ = _train(ds, POSE, BASE, data, base_run_id=TUNED)
    _ = _train(ds, POSE, TUNED, data, base_run_id=BASE)

    with pytest.raises(ValueError, match=re.escape(f"{TUNED} -> {BASE} -> {TUNED}")):
        _ = seen_frames(ds, POSE, TUNED)


def test_a_base_model_that_cannot_be_walked_leaves_the_frames_incomplete(
    ds: Dataset,
) -> None:
    prepared = _prepare(ds)
    tuned = _train(ds, POSE, TUNED, _data(ds, prepared), base_run_id=MISSING)

    found = seen_frames(ds, POSE, tuned)

    assert found == SeenFrames(
        keys=_keys(ds, prepared, "train", "valid"),
        complete=False,
        reason=f"base model {MISSING}: {seen_frames(ds, POSE, MISSING).reason}",
    )


# ------------------------------------------------------------ incomplete chains


def _without_data_path(ds: Dataset) -> str:
    return _train(ds, POSE, MODEL, None)


def _on_data_outside_a_preparation(ds: Dataset) -> str:
    data = ds.base_dir.parent / "elsewhere" / "data.yaml"
    data.parent.mkdir(parents=True)
    _ = data.write_text("names: [mouse]\n")
    return _train(ds, POSE, MODEL, data)


def _on_a_preparation_without_a_row(ds: Dataset) -> str:
    data = model_run_root(ds, PREPARE, f"{PREPARE}.0.3-0000000000") / "data.yaml"
    data.parent.mkdir(parents=True)
    _ = data.write_text("names: [mouse]\n")
    return _train(ds, POSE, MODEL, data)


def _on_a_lost_revision_without_a_split_record(ds: Dataset) -> str:
    prepared = _prepare(ds)
    model = _train(ds, POSE, MODEL, _data(ds, prepared))
    _record(ds, prepared).unlink()
    _revision(ds, TRAINED).unlink()
    return model


@pytest.mark.parametrize(
    "registered",
    [
        pytest.param(_without_data_path, id="no-data-path"),
        pytest.param(_on_data_outside_a_preparation, id="not-a-preparation"),
        pytest.param(_on_a_preparation_without_a_row, id="no-preparation-row"),
        pytest.param(_on_a_lost_revision_without_a_split_record, id="lost-revision"),
    ],
)
def test_a_chain_that_stops_gives_the_reason_that_the_walk_stopped(
    ds: Dataset, registered: Callable[[Dataset], str]
) -> None:
    model = registered(ds)
    stopped_at = training_provenance(ds, POSE, model).stopped_at

    found = seen_frames(ds, POSE, model)

    assert stopped_at, "the walk stops"
    assert found == SeenFrames(keys=frozenset(), complete=False, reason=stopped_at)


def _of_an_unknown_target(ds: Dataset, prepared: str) -> str:
    _rewrite_preparations(ds, "target", "coco")
    return re.escape(f"{PREPARE} run {prepared} records an unknown target 'coco'")


def _empty(ds: Dataset, prepared: str) -> str:
    _ = _record(ds, prepared).write_text("")
    cannot_read = f"the split record of {PREPARE} run {prepared} cannot be read: "
    return re.escape(cannot_read) + ".+"


def _without_a_split_column(ds: Dataset, prepared: str) -> str:
    path = _record(ds, prepared)
    written = pd.read_csv(path, dtype=str, keep_default_na=False)
    written.drop(columns="split").to_csv(path, index=False)
    return _lacks_a_column(prepared)


def _of_a_header_without_a_split_column(ds: Dataset, prepared: str) -> str:
    _ = _record(ds, prepared).write_text("frame_key\n")
    return _lacks_a_column(prepared)


def _lacks_a_column(prepared: str) -> str:
    lacks = (
        f"the split record of {PREPARE} run {prepared} lacks a frame_key or "
        "split column"
    )
    return re.escape(lacks)


@pytest.mark.parametrize(
    "broken",
    [
        pytest.param(_of_an_unknown_target, id="unknown-target"),
        pytest.param(_empty, id="unreadable"),
        pytest.param(_without_a_split_column, id="no-split-column"),
        pytest.param(_of_a_header_without_a_split_column, id="header-only"),
    ],
)
def test_a_split_record_that_cannot_be_read_leaves_the_frames_unknown(
    ds: Dataset, broken: Callable[[Dataset, str], str]
) -> None:
    """*broken* breaks one preparation and returns a pattern of the reason.

    The check reads the columns of the record. A record of a header alone fails
    it too.
    """
    prepared = _prepare(ds)
    model = _train(ds, POSE, MODEL, _data(ds, prepared))
    reason = broken(ds, prepared)

    found = seen_frames(ds, POSE, model)

    assert not found.complete
    assert found.keys == frozenset()
    assert re.fullmatch(reason, found.reason), found.reason


def test_a_revision_that_cannot_be_read_leaves_the_frames_incomplete(
    ds: Dataset,
) -> None:
    prepared = _prepare(ds)
    model = _train(ds, POSE, MODEL, _data(ds, prepared))
    _record(ds, prepared).unlink()
    _ = _revision(ds, TRAINED).write_text("not a revision")

    found = seen_frames(ds, POSE, model)

    assert not found.complete
    assert found.keys == frozenset()
    assert f"{TRAINED} rev1" in found.reason


def test_a_model_that_no_index_registers_has_unknown_frames(ds: Dataset) -> None:
    found = seen_frames(ds, POSE, MISSING)

    assert not found.complete
    assert found.keys == frozenset()
    assert MISSING in found.reason


# ------------------------------------------------------- a model of two trainings


def test_a_model_of_two_trainings_saw_the_frames_of_both(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SLEAP top-down model is a centroid and a centered-instance training."""
    _ = record_slp_writes(monkeypatch)
    first = _prepare(ds, "sleap")
    second = _prepare_other(ds, "sleap")
    centroid = _train(ds, SLEAP, CENTROID, _data(ds, first))
    instance = _train(ds, SLEAP, INSTANCE, _data(ds, second))

    found = seen_frames_of_model(ds, SLEAP, [centroid, instance])

    assert found == SeenFrames(
        keys=_keys(ds, first, "train", "valid") | _keys(ds, second, "train", "valid"),
        complete=True,
        reason="",
    )


def test_a_model_of_two_trainings_is_incomplete_when_one_of_them_is(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = record_slp_writes(monkeypatch)
    prepared = _prepare(ds, "sleap")
    centroid = _train(ds, SLEAP, CENTROID, _data(ds, prepared))
    instance = _train(ds, SLEAP, INSTANCE, None)

    found = seen_frames_of_model(ds, SLEAP, [centroid, instance])

    assert found == SeenFrames(
        keys=_keys(ds, prepared, "train", "valid"),
        complete=False,
        reason=f"{INSTANCE}: {seen_frames(ds, SLEAP, INSTANCE).reason}",
    )


def test_a_model_of_one_training_has_the_frames_of_that_training(
    ds: Dataset,
) -> None:
    model = _train(ds, POSE, MODEL, None)

    assert seen_frames_of_model(ds, POSE, [model]) == seen_frames(ds, POSE, model)


def test_a_model_of_no_training_is_refused(ds: Dataset) -> None:
    with pytest.raises(ValueError, match="at least one training run"):
        _ = seen_frames_of_model(ds, POSE, [])


# --------------------------------------------------------------------- read only


def test_a_read_only_dataset_gains_no_directory(ds: Dataset) -> None:
    """The walk reads a split record and, for a base without one, its revisions."""
    recorded = _prepare(ds)
    unrecorded = _prepare_other(ds)
    _ = _train(ds, POSE, BASE, _data(ds, unrecorded))
    tuned = _train(ds, POSE, TUNED, _data(ds, recorded), base_run_id=BASE)
    expected = _keys(ds, recorded, "train", "valid") | _keys(ds, unrecorded)
    _record(ds, unrecorded).unlink()
    reopened = open_dataset(ds.base_dir, ensure_roots=False)
    before = tree_snapshot(ds.base_dir)

    found = seen_frames(reopened, POSE, tuned)

    assert found == SeenFrames(keys=expected, complete=True, reason="")
    assert_tree_unchanged(ds.base_dir, before)


def test_a_read_only_dataset_without_its_roots_gains_no_directory(
    tmp_path: Path,
) -> None:
    base = tmp_path / "ro"
    _ = make_dataset(base, ensure_roots=False)
    reopened = open_dataset(base, ensure_roots=False)
    before = tree_snapshot(base)

    found = seen_frames_of_model(reopened, POSE, [MODEL])

    assert not found.complete
    assert_tree_unchanged(base, before)
