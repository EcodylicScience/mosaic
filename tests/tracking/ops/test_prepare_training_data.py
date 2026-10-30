"""Annotation revisions from several projects become one training dataset.

The workflow this pins end to end: two projects each save a keypoint annotation
set, a library claims one exact revision of each, prepares a union, and a model
trained from it can say which annotation states it saw -- from the library and
from any project that links it.
"""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Final

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from mosaic.core.annotations.bbox import BboxPolicy, derived_bbox
from mosaic.core.annotations.model import KeypointSchema
from mosaic.core.annotations.pose_annotations import (
    Alias,
    PoseAnnotationSet,
    PoseDefinition,
    PoseFrame,
    FrameRecording,
    PoseObject,
    pose_annotations_payload,
    read_pose_annotations,
)
from mosaic.core.annotations.projection import (
    COCO_EXPORT_FILENAME,
    KEYPOINTS_SERIES,
    write_keypoint_set_revision,
)
from mosaic.core.annotations.writers.coco import coco_keypoints_payload
from mosaic.core.dataset import Dataset
from mosaic.core.entry import make_entry_key
from mosaic.core.manifest import LabelsScanSource, LibraryLink
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.label_series_index import write_series_revision
from mosaic.core.pipeline.inventory import inventory
from mosaic.core.pipeline.graph import IdentityUnresolved, Recipe, plan_pipeline
from mosaic.core.pipeline.models import (
    SPLIT_RECORD_FILENAME,
    model_index_path,
    model_run_root,
)
from mosaic.core.pipeline.op_identity import op_run_id
from mosaic.core.pipeline.ops import OPS, IdentityDeferred, run_op
from mosaic.core.recording import RecordingKey
from mosaic.tracking import register_ops
from mosaic.tracking.keypoint_sets import ResolvedSet, resolve_keypoint_sets
from mosaic.tracking.ops import prepare
from mosaic.tracking.ops._common import (
    fingerprint_dataset,
    fingerprint_yolo_dataset,
    resolve_training_data,
)
from mosaic.tracking.ops.prepare import (
    PreparedDatasetIndexRow,
    PrepareTrainingDataParams,
    SplitPlan,
    SplitRow,
    check_preparation,
    prepare_training_data_run_id,
    prepared_dataset_index,
)
from mosaic.tracking.ops.train import PoseTrainParams, finalize_training
from mosaic.tracking.ops.trained_models import trained_model_index
from mosaic.tracking.training_provenance import training_provenance
from tests.helpers import (
    MOUSE,
    add_track_sequences,
    assert_tree_unchanged,
    extract_entry_camera,
    make_dataset,
    pose_frame,
    pose_object,
    pose_set,
    record_slp_writes,
    revision_file,
    rewrite_as_older_prepared_index,
    tree_snapshot,
)

register_ops()

KIND = "prepare-training-data"


def _image(ds: Dataset, sequence: str, index: int) -> Path:
    """An extract-shaped frame image on disk, by its path relative to *ds*.

    Every sequence numbers its frames from zero, the way mosaic's frame
    extraction does, so two sequences share every basename. The bytes name the
    frame, so two images that should differ do.
    """
    relative = Path(
        f"media/frames/kmeans/kmeans-d7968c97b0/{sequence}/frame_{index:06d}.png"
    )
    image = ds.base_dir / relative
    image.parent.mkdir(parents=True, exist_ok=True)
    _ = image.write_bytes(f"{ds.name}/{sequence}/{index}".encode())
    return relative


def _save(ds: Dataset, key: str, state: PoseAnnotationSet, commit: str = "c1") -> int:
    return write_keypoint_set_revision(
        ds, set_key=key, annotations=state, origin={"dolt_commit": commit}
    ).revision


def _save_set(
    ds: Dataset,
    key: str,
    sequences: dict[str, int],
    *,
    shift: float = 0.0,
    commit: str = "c1",
) -> int:
    """One mouse on every frame of *sequences*, all finished, saved as one revision."""
    frames = [
        pose_frame(
            _image(ds, sequence, index),
            pose_object((10.0 + index + shift, 12.0), (30.0 + index, 20.0)),
            sequence=sequence,
            frame_index=index,
        )
        for sequence, count in sequences.items()
        for index in range(count)
    ]
    return _save(ds, key, pose_set(frames, image_root=ds.base_dir), commit)


def _claim(library: Dataset, project: Dataset, key: str, revision: int) -> None:
    library.add_scan_source(
        LabelsScanSource(
            id=f"{project.name}-{key}",
            path=str(project.get_root("labels_raw") / "keypoints" / key),
            files=(revision_file(revision),),
            series="keypoints",
        )
    )
    _ = library.scan_labels()


WORLD_UUIDS: Final = {
    "mice": "52000000-0000-4000-8000-000000000000",
    "rats": "53000000-0000-4000-8000-000000000000",
    "library": "07000000-0000-4000-8000-000000000000",
}
"""The manifest ``uuid`` of each dataset of the ``world`` fixture.

Their first eight characters differ, because a placed name begins with them.
"""


@pytest.fixture
def world(tmp_path: Path) -> tuple[Dataset, Dataset, Dataset]:
    """Two projects with a saved set each, and a library that has claimed both.

    The ``uuid`` of each project enters the key of every unit it holds, so the
    split of every frame is fixed. ``SLEAP_SPLIT`` places frames in all three splits.
    """
    mice = make_dataset(tmp_path / "52", name="mice", uuid=WORLD_UUIDS["mice"])
    rats = make_dataset(tmp_path / "53", name="rats", uuid=WORLD_UUIDS["rats"])
    library = make_dataset(
        tmp_path / "libraries" / "7", name="library", uuid=WORLD_UUIDS["library"]
    )
    _claim(
        library,
        mice,
        "17-openfield",
        _save_set(mice, "17-openfield", {"m01": 4, "m02": 4}),
    )
    _claim(library, rats, "21-arena", _save_set(rats, "21-arena", {"m01": 4, "r09": 4}))
    return mice, rats, library


def _prepare(library: Dataset, **overrides: object) -> str:
    params: dict[str, object] = {
        "sets": [{"set_key": "17-openfield"}, {"set_key": "21-arena"}],
        "split": (0.5, 0.5, 0.0),
    }
    params.update(overrides)
    return run_op(library, KIND, params)


def test_a_union_across_projects_loses_no_image(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    """Sixteen frames, eight of them sharing basenames pairwise, all sixteen kept."""
    _mice, _rats, library = world

    run_id = _prepare(library)

    out = model_run_root(library, KIND, run_id)
    images = sorted(p.name for p in out.rglob("images/*.png"))
    labels = sorted(p.name for p in out.rglob("labels/*.txt"))
    assert len(images) == 16 and len(set(images)) == 16
    assert [Path(n).stem for n in images] == [Path(n).stem for n in labels]
    assert all(not p.is_symlink() for p in out.rglob("images/*.png")), (
        "copied, not linked"
    )
    contents = {p.read_bytes() for p in out.rglob("images/*.png")}
    assert len(contents) == 16, "sixteen distinct images, none written over another"


def test_the_tree_is_portable_and_is_a_yolo_pose_dataset(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    _mice, _rats, library = world
    run_id = _prepare(library)

    declared = yaml.safe_load(
        (model_run_root(library, KIND, run_id) / "data.yaml").read_text()
    )

    assert "path" not in declared, "left out, so the trainer roots the tree at the YAML"
    assert declared["kpt_shape"] == [2, 3]
    assert "flip_idx" not in declared, "no mirror pairs, so flips stay off"
    assert declared["train"] == "train/images" and declared["val"] == "valid/images"


def test_a_sequence_never_straddles_the_split(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    """The default keeps a recording together, and tells two projects' ``m01`` apart."""
    _mice, _rats, library = world
    out = model_run_root(library, KIND, _prepare(library))

    where: dict[str, set[str]] = {}
    for image in out.rglob("images/*.png"):
        origin, _set, sequence, _frame = image.name.split("__", 3)
        where.setdefault(f"{origin}:{sequence}", set()).add(image.parts[-3])

    assert len(where) == 4, "m01 of one project is not m01 of the other"
    assert all(len(splits) == 1 for splits in where.values())


def test_the_row_records_exactly_which_revisions_were_read(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    mice, rats, library = world
    run_id = _prepare(library)

    row = (
        prepared_dataset_index(model_index_path(library, KIND))
        .read(run_id=run_id)
        .iloc[0]
    )
    consumed = json.loads(str(row["consumed_sets"]))

    assert {(c["set_key"], c["revision"]) for c in consumed} == {
        ("17-openfield", 1),
        ("21-arena", 1),
    }
    assert {c["origin_uuid"] for c in consumed} == {mice.uuid, rats.uuid}
    assert row["artifact_path"] == f"models/{KIND}/{run_id}/data.yaml"


def _recorded_pose(library: Dataset, run_id: str) -> tuple[str, str]:
    """The pose a preparation's index row names, as its ``(id, name)`` cells."""
    rows = index_records(
        prepared_dataset_index(model_index_path(library, KIND)).read(run_id=run_id)
    )
    return rows[-1]["pose_id"], rows[-1]["pose_name"]


def test_a_preparation_records_the_one_pose_its_sets_hold(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    _mice, _rats, library = world
    run_id = _prepare(library)

    assert _recorded_pose(library, run_id) == (str(MOUSE.id), MOUSE.name)
    params = PrepareTrainingDataParams.model_validate(
        {
            "sets": [{"set_key": "17-openfield"}, {"set_key": "21-arena"}],
            "split": (0.5, 0.5, 0.0),
        }
    )
    planned = OPS[KIND]().plan_identity(library, params, ResolvedScope()).run_id
    assert run_id == planned, (
        "named before the sets are narrowed, so the pose it records is not part of it"
    )


def test_a_selector_is_not_content(world: tuple[Dataset, Dataset, Dataset]) -> None:
    """Naming the revision, or the sets in another order, is the same run."""
    _mice, _rats, library = world
    op = OPS[KIND]()

    def plan(sets: list[dict[str, object]]) -> str:
        params = PrepareTrainingDataParams.model_validate({"sets": sets})
        return op.plan_identity(library, params, ResolvedScope()).run_id

    latest = plan([{"set_key": "17-openfield"}, {"set_key": "21-arena"}])
    named = plan(
        [
            {"set_key": "21-arena", "revision": 1},
            {"set_key": "17-openfield", "revision": 1},
        ]
    )

    assert latest == named


def test_a_changed_annotation_state_is_a_new_dataset(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    mice, _rats, library = world
    before = _prepare(library)

    revision = _save_set(
        mice, "17-openfield", {"m01": 4, "m02": 4}, shift=0.5, commit="c2"
    )
    _ = library.add_source_files(
        "labels", "mice-17-openfield", [revision_file(revision)]
    )
    _ = library.scan_labels()

    assert revision == 2
    assert _prepare(library) != before, "the latest revision moved, so the run did"
    pinned = _prepare(
        library,
        sets=[{"set_key": "17-openfield", "revision": 1}, {"set_key": "21-arena"}],
    )
    assert pinned == before, "and the earlier state is still nameable"


def test_a_second_identical_run_is_a_cache_hit(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    _mice, _rats, library = world
    first = _prepare(library)
    marker = model_run_root(library, KIND, first) / "data.yaml"
    stamp = marker.stat().st_mtime_ns

    assert _prepare(library) == first
    assert marker.stat().st_mtime_ns == stamp, "nothing was rewritten"


def test_a_set_nobody_claimed_defers_with_the_repair(tmp_path: Path) -> None:
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    params = PrepareTrainingDataParams.model_validate(
        {"sets": [{"set_key": "17-openfield"}]}
    )

    with pytest.raises(IdentityDeferred) as refused:
        _ = OPS[KIND]().plan_identity(library, params, ResolvedScope())
    assert "declare a labels source" in refused.value.because


def _one_pose_set(library: Dataset, key: str, pose: PoseDefinition) -> None:
    """One finished frame of *pose*, saved under *key* in *library*."""
    points = tuple((float(i), 1.0) for i in range(pose.schema.num_keypoints))
    frame = pose_frame(
        _image(library, "m07", 0),
        pose_object(*points, pose_id=pose.id),
        sequence="m07",
    )
    _ = _save(
        library, key, pose_set([frame], poses=(pose,), image_root=library.base_dir)
    )


@pytest.mark.parametrize(
    "schema",
    [
        KeypointSchema(names=("nose", "ear", "tail")),
        KeypointSchema(names=("nose", "tail")),
        KeypointSchema(
            names=("nose", "tail"), skeleton=((0, 1),), symmetries=((0, 1),)
        ),
    ],
    ids=["keypoints", "skeleton", "symmetry"],
)
def test_two_layouts_cannot_be_one_model(
    tmp_path: Path, schema: KeypointSchema
) -> None:
    """Names, edges and mirror pairs all have to agree; each alone differs here.

    Both sets hold the one pose, edited between them. A pose keeps its id when its
    layout changes, so the pose check passes and the layout check refuses.
    """
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    _ = _save_set(library, "two-points", {"m01": 2})
    _one_pose_set(library, "other", replace(MOUSE, schema=schema))

    with pytest.raises(ValueError, match="One model has one keypoint layout"):
        _ = run_op(
            library, KIND, {"sets": [{"set_key": "two-points"}, {"set_key": "other"}]}
        )


def test_a_pose_renamed_between_sets_is_named_as_the_first_set_names_it(
    tmp_path: Path,
) -> None:
    """One pose id under two definitions, as a pose renamed across revisions.

    The sets are read in identity order, whatever order they are named in, and
    the first set's definition names the prepared data.
    """
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    _ = _save_set(library, "a-before", {"m01": 2, "m02": 2})
    renamed = replace(MOUSE, name="rodent")
    frames = [
        pose_frame(
            _image(library, sequence, index),
            pose_object((10.0 + index, 12.0), (30.0 + index, 20.0)),
            sequence=sequence,
            frame_index=index,
        )
        for sequence in ("m03", "m04")
        for index in range(2)
    ]
    _ = _save(
        library,
        "b-renamed",
        pose_set(frames, poses=(renamed,), image_root=library.base_dir),
    )

    run_id = run_op(
        library,
        KIND,
        {
            "sets": [{"set_key": "b-renamed"}, {"set_key": "a-before"}],
            "split_by": "frame",
        },
    )

    assert _recorded_pose(library, run_id) == (str(MOUSE.id), MOUSE.name)


def test_a_missing_annotated_image_is_refused(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    mice, _rats, library = world
    (
        mice.base_dir / "media/frames/kmeans/kmeans-d7968c97b0/m01/frame_000002.png"
    ).unlink()

    with pytest.raises(Exception, match="not on disk"):
        _ = _prepare(library)


def test_polo_writes_one_point_per_instance(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    _mice, _rats, library = world
    out = model_run_root(library, KIND, _prepare(library, target="polo", radius=25.0))

    declared = yaml.safe_load((out / "data.yaml").read_text())
    line = next(out.rglob("labels/*.txt")).read_text().split()

    assert declared["radii"] == {0: 25.0} and "path" not in declared
    assert len(line) == 4 and line[1] == "25.0"


def test_a_lightning_pose_project_holds_every_frame_under_its_own_name(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    _mice, _rats, library = world
    run_id = _prepare(library, target="litpose")
    out = model_run_root(library, KIND, run_id)

    row = (
        prepared_dataset_index(model_index_path(library, KIND))
        .read(run_id=run_id)
        .iloc[0]
    )
    project = library.resolve_path(str(row["artifact_path"]))

    assert project == out / "project" and (project / "CollectedData.csv").is_file()
    labelled = list((project / "labeled-data").rglob("*.png"))
    assert len(labelled) == 16 and len({p.read_bytes() for p in labelled}) == 16
    assert resolve_training_data(library, run_id) == project, (
        "what train-litpose is handed"
    )


SLEAP_SPLIT: Final[dict[str, object]] = {
    "target": "sleap",
    "split_by": "frame",
    "split": (0.5, 0.25, 0.25),
}
"""A sleap preparation of the ``world`` sets.

Under the ``uuid`` values of ``WORLD_UUIDS``, it places frames in all three splits.
"""


def _names_by_split(out: Path) -> dict[str, list[str]]:
    """The placed names that ``split.csv`` records in each split, sorted."""
    record = _split_record(out)
    split_of = dict(zip(record["placed_name"], record["split"], strict=True))
    return {
        split: sorted(name for name, placed in split_of.items() if placed == split)
        for split in ("train", "valid", "test")
    }


def test_the_sleap_writer_is_handed_two_sets_with_collision_free_images(
    world: tuple[Dataset, Dataset, Dataset], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write the training frames to ``labels.slp`` and the validation frames beside it.

    The ``.slp`` itself needs the SLEAP environment; what it is given does not. A
    test frame is written to neither file.
    """
    _mice, _rats, library = world
    calls = record_slp_writes(monkeypatch)

    run_id = _prepare(library, **SLEAP_SPLIT)

    out = model_run_root(library, KIND, run_id)
    names_in = _names_by_split(out)
    assert names_in["test"], "the split under test places a test frame"
    assert [path for path, _set, _images in calls] == [
        out / "labels.slp",
        out / "labels.val.slp",
    ]
    for (_path, handed, images_dir), split in zip(
        calls, ("train", "valid"), strict=True
    ):
        names = sorted(frame.image_path.name for frame in handed.frames)
        assert names == names_in[split], split
        assert handed.image_root == out / "images" == images_dir
    written = [
        frame.image_path.name
        for _path, handed, _images in calls
        for frame in handed.frames
    ]
    assert len(set(written)) == len(written) == 16 - len(names_in["test"])
    assert all((out / "images" / name).is_file() for name in written)
    assert resolve_training_data(library, run_id) == out / "labels.slp"


def test_a_sleap_preparation_counts_the_frames_of_its_two_files(
    world: tuple[Dataset, Dataset, Dataset], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Count the training and validation frames on the row, and no test frame."""
    _mice, _rats, library = world
    calls = record_slp_writes(monkeypatch)

    run_id = _prepare(library, **SLEAP_SPLIT)

    names_in = _names_by_split(model_run_root(library, KIND, run_id))
    written = sum(len(handed.frames) for _path, handed, _images in calls)
    assert written == len(names_in["train"]) + len(names_in["valid"]) < 16
    row = _row_of(library, run_id)
    assert row["n_frames"] == str(written)
    assert row["n_train"] == str(len(names_in["train"]))
    assert row["n_valid"] == str(len(names_in["valid"]))
    assert row["n_test"] == "0"


def test_a_lightning_pose_preparation_counts_every_frame_it_writes(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    """Lightning Pose draws its own split from every frame it is given."""
    _mice, _rats, library = world

    run_id = _prepare(library, target="litpose", split=(0.5, 0.25, 0.25))

    names_in = _names_by_split(model_run_root(library, KIND, run_id))
    row = _row_of(library, run_id)
    assert row["n_frames"] == "16"
    assert [row[f"n_{split}"] for split in ("train", "valid", "test")] == [
        str(len(names_in[split])) for split in ("train", "valid", "test")
    ]


def test_a_sleap_preparation_without_its_validation_file_is_prepared_again(
    world: tuple[Dataset, Dataset, Dataset], monkeypatch: pytest.MonkeyPatch
) -> None:
    _mice, _rats, library = world
    calls = record_slp_writes(monkeypatch)
    first = _prepare(library, target="sleap")
    assert _prepare(library, target="sleap") == first
    assert len(calls) == 2, "a whole preparation is reused"

    validation = model_run_root(library, KIND, first) / "labels.val.slp"
    validation.unlink()

    assert _prepare(library, target="sleap") == first
    assert len(calls) == 4, "both files are written again"
    assert validation.is_file()


# ---------------------------------------------- choosing from the full saved state

EARS = PoseDefinition(
    id=5,
    name="mouse",
    schema=KeypointSchema(
        names=("nose", "left_ear", "right_ear"),
        skeleton=((0, 1), (0, 2)),
        symmetries=((1, 2),),
    ),
    aliases=(Alias(12, "resident"), Alias(13, "intruder")),
)
CRICKET = PoseDefinition(
    id=8,
    name="cricket",
    schema=KeypointSchema(names=("head", "tail"), skeleton=((0, 1),)),
)
PADDING = BboxPolicy(method="isotropic", pad_frac_of_body=0.3, min_pad_px=4.0)


def _ears(x: float, alias: Alias) -> PoseObject:
    return pose_object(
        (x, 20.0), (x - 4.0, 16.0), (x + 4.0, 16.0), pose_id=EARS.id, alias_id=alias.id
    )


@pytest.fixture
def full_state(tmp_path: Path) -> Dataset:
    """A library holding one set in its whole saved state.

    Two recordings of four frames. Every frame holds a resident and an intruder
    mouse and a cricket; the last frame of ``m02`` is unfinished and its image
    was never kept.
    """
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    resident, intruder = EARS.aliases
    frames: list[PoseFrame] = []
    for sequence in ("m01", "m02"):
        for index in range(4):
            unfinished = (sequence, index) == ("m02", 3)
            image = _image(library, sequence, index)
            if unfinished:
                (library.base_dir / image).unlink()
            frames.append(
                pose_frame(
                    image,
                    _ears(10.0 + index, resident),
                    _ears(40.0, intruder),
                    pose_object((30.0, 40.0), (36.0, 42.0), pose_id=CRICKET.id),
                    sequence=sequence,
                    frame_index=index,
                    usable=not unfinished,
                )
            )
    state = pose_set(
        frames, poses=(EARS, CRICKET), image_root=library.base_dir, bbox_policy=PADDING
    )
    _ = _save(library, "full", state)
    return library


FULL: dict[str, object] = {
    "sets": [{"set_key": "full"}],
    "pose": "mouse",
    # Two recordings are too few to draw whole ones into training and validation.
    "split_by": "frame",
}
"""The full state's preparation, split so that every split of the tree holds frames."""


def _prepare_full(library: Dataset, **overrides: object) -> Path:
    params = {**FULL, **overrides}
    return model_run_root(library, KIND, run_op(library, KIND, params))


def _label(out: Path, sequence: str, index: int) -> list[list[str]]:
    """The label lines of one frame, however the split placed it."""
    (found,) = out.rglob(f"labels/*__{sequence}__frame_{index:06d}.txt")
    return [line.split() for line in found.read_text().splitlines()]


def test_only_finished_frames_are_trained_on(full_state: Dataset) -> None:
    """The unfinished frame is left out, so its missing image is not a failure."""
    out = _prepare_full(full_state)

    assert len(list(out.rglob("images/*.png"))) == 7
    assert not list(out.rglob("labels/*__m02__frame_000003.txt"))


def test_one_pose_is_trained_and_the_others_are_background(full_state: Dataset) -> None:
    mice = _prepare_full(full_state)
    crickets = _prepare_full(full_state, pose="cricket")

    assert len(_label(mice, "m01", 0)) == 2, "both mice, no cricket"
    assert len(_label(crickets, "m01", 0)) == 1
    declared = yaml.safe_load((crickets / "data.yaml").read_text())
    assert declared["names"] == ["cricket"] and declared["kpt_shape"] == [2, 3]


@pytest.mark.parametrize(
    ("pose", "target", "chosen"),
    [
        (EARS.id, "yolo-pose", EARS),
        ("mouse", "yolo-pose", EARS),
        (CRICKET.id, "yolo-pose", CRICKET),
        ("cricket", "yolo-pose", CRICKET),
        ("cricket", "sleap", CRICKET),
    ],
    ids=[
        "mouse-by-id",
        "mouse-by-name",
        "cricket-by-id",
        "cricket-by-name",
        "cricket-for-sleap",
    ],
)
def test_a_preparation_records_the_pose_it_chose(
    full_state: Dataset,
    pose: int | str,
    target: str,
    chosen: PoseDefinition,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recorded on both write paths: the split tree, and a tool's own format."""
    _ = record_slp_writes(monkeypatch)

    run_id = run_op(full_state, KIND, {**FULL, "pose": pose, "target": target})

    assert _recorded_pose(full_state, run_id) == (str(chosen.id), chosen.name)


def test_a_sleap_preparation_by_alias_hands_each_object_its_alias_as_its_track(
    full_state: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writer names each track by its ``track_id``, so a model reports the alias."""
    calls = record_slp_writes(monkeypatch)

    _ = run_op(full_state, KIND, {**FULL, "target": "sleap", "track_by": "alias"})

    tracks = [
        sorted(obj.track_id for obj in frame.objects)
        for _path, handed, _images in calls
        for frame in handed.frames
    ]
    assert tracks and all(found == ["intruder", "resident"] for found in tracks)


def _recorded_track_by(library: Dataset, run_id: str) -> str:
    rows = index_records(
        pd.read_csv(model_index_path(library, KIND), dtype=str, keep_default_na=False)
    )
    (row,) = [row for row in rows if row["run_id"] == run_id]
    return row["track_by"]


def test_a_sleap_preparation_records_how_it_named_its_tracks(
    full_state: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = record_slp_writes(monkeypatch)

    by_alias = run_op(
        full_state, KIND, {**FULL, "target": "sleap", "track_by": "alias"}
    )
    plain = run_op(full_state, KIND, {**FULL, "target": "sleap"})
    other_target = run_op(full_state, KIND, FULL)

    assert _recorded_track_by(full_state, by_alias) == "alias"
    assert _recorded_track_by(full_state, plain) == ""
    assert _recorded_track_by(full_state, other_target) == ""


def test_an_index_without_the_track_by_column_reads_blank_once_adopted(
    full_state: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = record_slp_writes(monkeypatch)
    earlier = run_op(full_state, KIND, {**FULL, "target": "sleap", "track_by": "alias"})
    rewrite_as_older_prepared_index(full_state)

    _ = run_op(full_state, KIND, {**FULL, "target": "sleap"})

    assert _recorded_track_by(full_state, earlier) == ""


def test_a_reused_preparation_fills_the_blank_track_by_cell(
    full_state: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = record_slp_writes(monkeypatch)
    params = {**FULL, "target": "sleap", "track_by": "alias"}
    first = run_op(full_state, KIND, params)
    written = len(calls)
    rewrite_as_older_prepared_index(full_state)

    assert run_op(full_state, KIND, params) == first

    assert len(calls) == written, "the preparation was reused, not written again"
    assert _recorded_track_by(full_state, first) == "alias"


def test_a_recipe_over_an_older_preparation_defers_the_global_id_check(
    full_state: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan cannot read the blank cell that the preparation's run fills.

    The train step is deferred rather than refused as trained on a
    preparation that did not name its tracks by alias.
    """
    _ = record_slp_writes(monkeypatch)
    params = {**FULL, "target": "sleap", "track_by": "alias"}
    _ = run_op(full_state, KIND, params)
    rewrite_as_older_prepared_index(full_state)
    recipe = Recipe.model_validate(
        {
            "steps": [
                {"id": "prepare", "type": "op", "kind": KIND, "params": params},
                {
                    "id": "train",
                    "type": "op",
                    "kind": "train-sleap",
                    "params": {
                        "labels": {"step": "prepare"},
                        "head": "embedding",
                        "sleap_overrides": {
                            "data_config.identity.track_names_are_global": True
                        },
                    },
                },
            ]
        }
    )

    plan = plan_pipeline(full_state, recipe)

    train = plan.step("train")
    assert train.run_id is None
    assert isinstance(train.reason, IdentityUnresolved)
    assert "track_by" in train.reason.because


def test_a_reused_preparation_leaves_other_rows_and_a_recorded_cell_alone(
    full_state: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = record_slp_writes(monkeypatch)
    by_alias = {**FULL, "target": "sleap", "track_by": "alias"}
    plain = {**FULL, "target": "sleap"}
    first = run_op(full_state, KIND, by_alias)
    second = run_op(full_state, KIND, plain)
    path = model_index_path(full_state, KIND)
    before = path.read_bytes()

    assert run_op(full_state, KIND, by_alias) == first
    assert run_op(full_state, KIND, plain) == second

    assert path.read_bytes() == before, "a recorded or unset cell is not rewritten"


def test_a_state_holding_two_poses_needs_one_named(full_state: Dataset) -> None:
    with pytest.raises(ValueError, match="name the one to train"):
        _ = _prepare_full(full_state, pose=None)


def test_mirror_pairs_become_the_flip_permutation(full_state: Dataset) -> None:
    declared = yaml.safe_load((_prepare_full(full_state) / "data.yaml").read_text())

    assert declared["flip_idx"] == [0, 2, 1]


def test_aliases_become_classes_in_the_labels_and_the_yaml(full_state: Dataset) -> None:
    out = _prepare_full(full_state, class_by="alias")

    declared = yaml.safe_load((out / "data.yaml").read_text())
    assert declared["names"] == ["intruder", "resident"]
    by_x = {round(float(row[1]) * 64): row[0] for row in _label(out, "m01", 0)}
    assert by_x == {10: "1", 40: "0"}, "the resident at x=10, the intruder at x=40"


def test_polo_gives_every_class_its_radius(full_state: Dataset) -> None:
    out = _prepare_full(full_state, class_by="alias", target="polo", radius=25.0)

    declared = yaml.safe_load((out / "data.yaml").read_text())
    assert declared["names"] == {0: "intruder", 1: "resident"}
    assert declared["radii"] == {0: 25.0, 1: 25.0}


@pytest.mark.parametrize("target", ["sleap", "litpose"])
def test_a_one_class_trainer_refuses_classes_by_alias(
    full_state: Dataset, target: str
) -> None:
    with pytest.raises(ValueError, match="trains one class"):
        _ = _prepare_full(full_state, class_by="alias", target=target)


@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        ({"pose": None}, "name the one to train"),
        ({"class_by": "alias", "target": "sleap"}, "trains one class"),
        ({"split": (1.0, 0.0, 0.0)}, "the validation share is 0"),
    ],
)
def test_the_check_refuses_what_a_run_would_and_writes_nothing(
    full_state: Dataset, overrides: dict[str, object], refusal: str
) -> None:
    """A caller queueing a preparation hears the refusal before anything is queued."""
    params = {**FULL, **overrides}
    with pytest.raises(ValueError, match=refusal):
        check_preparation(full_state, PrepareTrainingDataParams.model_validate(params))

    assert not (full_state.base_dir / "models" / KIND).exists()


def test_the_check_passes_what_a_run_accepts(full_state: Dataset) -> None:
    params = {**FULL, "pose": "cricket"}

    check_preparation(full_state, PrepareTrainingDataParams.model_validate(params))


@pytest.fixture
def small_sets(tmp_path: Path) -> Dataset:
    """Return a library with a set of four frames of one recording and a set of one frame."""
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    _ = _save_set(library, "one-recording", {"m01": 4})
    _ = _save_set(library, "one-frame", {"m02": 1})
    return library


def _refusal_of_both(library: Dataset, params: dict[str, object]) -> list[str]:
    """Return the messages that check_preparation and a run refuse *params* with."""
    messages: list[str] = []
    with pytest.raises(ValueError) as checked:
        check_preparation(library, PrepareTrainingDataParams.model_validate(params))
    messages.append(str(checked.value))
    with pytest.raises(ValueError) as ran:
        _ = run_op(library, KIND, params)
    messages.append(str(ran.value))
    return messages


@pytest.mark.parametrize("target", ["yolo-pose", "polo", "sleap"])
@pytest.mark.parametrize(
    ("set_key", "split_by", "refusal"),
    [
        (
            "one-recording",
            "recording",
            "the training and validation splits need at least 2 recordings and "
            "split_by='recording' draws 1. Annotate frames from another recording, "
            "or split by frame.",
        ),
        (
            "one-recording",
            "sequence",
            "the training and validation splits need at least 2 sequences and "
            "split_by='sequence' draws 1. Annotate frames from another sequence, "
            "or split by frame.",
        ),
        (
            "one-recording",
            "group",
            "the training and validation splits need at least 2 groups and "
            "split_by='group' draws 1. Annotate frames from another group, "
            "or split by frame.",
        ),
        (
            "one-frame",
            "frame",
            "the training and validation splits need at least 2 frames and "
            "split_by='frame' draws 1.",
        ),
    ],
    ids=["recording", "sequence", "group", "frame"],
)
def test_a_split_of_fewer_than_two_units_is_refused(
    small_sets: Dataset, target: str, set_key: str, split_by: str, refusal: str
) -> None:
    """One unit cannot fill both a training and a validation split.

    The trainer would otherwise fail on the empty directory, after the preparation
    reported success.
    """
    params: dict[str, object] = {
        "sets": [{"set_key": set_key}],
        "split_by": split_by,
        "target": target,
    }

    assert _refusal_of_both(small_sets, params) == [refusal, refusal]


@pytest.mark.parametrize("target", ["yolo-pose", "polo", "sleap"])
def test_a_split_without_a_validation_share_is_refused(
    world: tuple[Dataset, Dataset, Dataset], target: str
) -> None:
    _mice, _rats, library = world
    params: dict[str, object] = {
        "sets": [{"set_key": "17-openfield"}, {"set_key": "21-arena"}],
        "split": (1.0, 0.0, 0.0),
        "target": target,
    }
    refusal = (
        f"the validation share is 0 and {target} needs a validation split. "
        "Give split a validation share above 0."
    )

    assert _refusal_of_both(library, params) == [refusal, refusal]


@pytest.mark.parametrize("target", ["yolo-pose", "polo", "sleap"])
def test_a_split_without_a_training_share_is_refused(
    world: tuple[Dataset, Dataset, Dataset], target: str
) -> None:
    _mice, _rats, library = world
    params: dict[str, object] = {
        "sets": [{"set_key": "17-openfield"}, {"set_key": "21-arena"}],
        "split": (0.0, 0.5, 0.5),
        "target": target,
    }
    refusal = "the training share is 0. Give split a training share above 0."

    assert _refusal_of_both(library, params) == [refusal, refusal]


def test_two_recordings_fill_a_training_and_a_validation_split(
    full_state: Dataset,
) -> None:
    """Whole recordings are drawn, and two fill both splits a tree trainer reads."""
    out = _prepare_full(full_state, split_by="sequence")

    trained = {image.name.split("__")[2] for image in out.glob("train/images/*.png")}
    validated = {image.name.split("__")[2] for image in out.glob("valid/images/*.png")}
    assert len(trained) == len(validated) == 1 and trained != validated


def test_a_trainer_that_splits_for_itself_takes_one_recording(
    small_sets: Dataset,
) -> None:
    """Lightning Pose draws the validation split from the frames it is given."""
    params = {
        "sets": [{"set_key": "one-recording"}],
        "split_by": "sequence",
        "target": "litpose",
    }

    check_preparation(small_sets, PrepareTrainingDataParams.model_validate(params))


def test_sleap_is_refused_one_recording(small_sets: Dataset) -> None:
    """SLEAP is refused one recording, which cannot fill a validation split."""
    params = PrepareTrainingDataParams.model_validate(
        {"sets": [{"set_key": "one-recording"}], "target": "sleap"}
    )

    with pytest.raises(ValueError, match="need at least 2 recordings"):
        check_preparation(small_sets, params)


# --------------------------------------------------- split units and placed names

INSTANTS = range(100, 108)
"""The frame indices that the frames run extracted from every entry camera."""
EXTRACTED_ENTRIES = ("A__s1", "B__s1", "C__s2__left", "C__s2__right")
"""How a placed name spells each entry camera of the ``extracted`` set."""


@pytest.fixture
def extracted(tmp_path: Path) -> Dataset:
    """Return a library with one set of frames that a frames run extracted.

    ``A/s1`` and ``B/s1`` share a sequence name across two groups, and two cameras
    record ``C/s2``. A frames run extracted frames 100 to 107 of every entry
    camera, and the set contains each of them, frame 100 of ``A/s1``, of ``B/s1``
    and of both cameras of ``C/s2`` among them. The revision records each frame's
    bare sequence name.
    """
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    sizes = {index: (64, 48) for index in INSTANTS}
    frames: list[PoseFrame] = []
    for group, sequence, camera in (
        ("A", "s1", ""),
        ("B", "s1", ""),
        ("C", "s2", "left"),
        ("C", "s2", "right"),
    ):
        images = extract_entry_camera(
            library,
            run_id="uniform-1",
            group=group,
            sequence=sequence,
            camera=camera,
            sizes=sizes,
        )
        frames.extend(
            pose_frame(
                image,
                pose_object((10.0 + index - 100, 12.0), (30.0, 20.0)),
                sequence=sequence,
                frame_index=index,
            )
            for index, image in images.items()
        )
    _ = _save(library, "extracted", pose_set(frames, image_root=library.base_dir))
    return library


def _placed(library: Dataset, set_key: str, entry: str, index: int) -> str:
    return f"{(library.uuid or '')[:8]}__{set_key}__{entry}__frame_{index:06d}.png"


def _splits_by_name(out: Path) -> dict[str, str]:
    """Each image of a prepared tree, by file name, and the split it was placed in."""
    return {image.name: image.parts[-3] for image in out.rglob("images/*.png")}


def test_two_groups_and_two_cameras_prepare_under_distinct_names(
    extracted: Dataset,
) -> None:
    """Frame 100 of ``A/s1``, ``B/s1`` and both cameras of ``C/s2`` get four names."""
    run_id = run_op(
        extracted, KIND, {"sets": [{"set_key": "extracted"}], "split": (0.5, 0.5, 0.0)}
    )
    out = model_run_root(extracted, KIND, run_id)

    expected = sorted(
        _placed(extracted, "extracted", entry, index)
        for entry in EXTRACTED_ENTRIES
        for index in INSTANTS
    )
    assert sorted(_splits_by_name(out)) == expected
    contents = {image.read_bytes() for image in out.rglob("images/*.png")}
    assert len(contents) == len(expected)


def test_under_split_by_frame_the_cameras_of_one_instant_fall_in_one_split(
    extracted: Dataset,
) -> None:
    params: dict[str, object] = {
        "sets": [{"set_key": "extracted"}],
        "split_by": "frame",
        "split": (0.5, 0.5, 0.0),
    }
    out = model_run_root(extracted, KIND, run_op(extracted, KIND, params))

    split_of = _splits_by_name(out)
    pairs = [
        (
            split_of[_placed(extracted, "extracted", "C__s2__left", index)],
            split_of[_placed(extracted, "extracted", "C__s2__right", index)],
        )
        for index in INSTANTS
    ]
    assert all(left == right for left, right in pairs)
    assert {left for left, _right in pairs} == {"train", "valid"}, (
        "the instants are drawn apart"
    )


def test_one_frame_saved_in_two_sets_falls_in_one_split(tmp_path: Path) -> None:
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    recordings = {"m01": 8, "m02": 8}
    _ = _save_set(library, "first", recordings)
    _ = _save_set(library, "second", recordings, shift=1.0)
    params: dict[str, object] = {
        "sets": [{"set_key": "first"}, {"set_key": "second"}],
        "split_by": "frame",
        "split": (0.5, 0.5, 0.0),
    }

    split_of = _splits_by_name(
        model_run_root(library, KIND, run_op(library, KIND, params))
    )

    pairs = [
        (
            split_of[_placed(library, "first", sequence, index)],
            split_of[_placed(library, "second", sequence, index)],
        )
        for sequence, count in recordings.items()
        for index in range(count)
    ]
    assert all(first == second for first, second in pairs)
    assert {first for first, _second in pairs} == {"train", "valid"}, (
        "the frames are drawn apart"
    )


SPLIT_COLUMNS = [
    "origin_uuid",
    "set_key",
    "revision",
    "image",
    "group",
    "sequence",
    "camera",
    "frame_index",
    "frame_key",
    "unit_key",
    "placed_name",
    "split",
]


def _split_record(out: Path) -> pd.DataFrame:
    return pd.read_csv(out / SPLIT_RECORD_FILENAME, dtype=str, keep_default_na=False)


def test_split_csv_has_one_sorted_row_per_frame_with_a_unique_name(
    extracted: Dataset,
) -> None:
    run_id = run_op(
        extracted, KIND, {"sets": [{"set_key": "extracted"}], "split": (0.5, 0.5, 0.0)}
    )
    record = _split_record(model_run_root(extracted, KIND, run_id))

    assert list(record.columns) == SPLIT_COLUMNS
    assert len(record) == len(EXTRACTED_ENTRIES) * len(INSTANTS)
    assert list(record["placed_name"]) == sorted(record["placed_name"])
    assert record["placed_name"].is_unique
    assert set(record["set_key"]) == {"extracted"}
    assert set(record["revision"]) == {"1"}
    assert set(record["frame_index"]) == {str(index) for index in INSTANTS}


def test_split_csv_agrees_with_the_tree(extracted: Dataset) -> None:
    run_id = run_op(
        extracted, KIND, {"sets": [{"set_key": "extracted"}], "split": (0.5, 0.5, 0.0)}
    )
    out = model_run_root(extracted, KIND, run_id)
    record = _split_record(out)

    assert dict(zip(record["placed_name"], record["split"], strict=True)) == (
        _splits_by_name(out)
    )


def test_split_csv_does_not_change_the_dataset_fingerprint(
    extracted: Dataset,
) -> None:
    run_id = run_op(
        extracted, KIND, {"sets": [{"set_key": "extracted"}], "split": (0.5, 0.5, 0.0)}
    )
    out = model_run_root(extracted, KIND, run_id)
    with_file = fingerprint_yolo_dataset(out / "data.yaml")
    (out / SPLIT_RECORD_FILENAME).unlink()

    assert fingerprint_yolo_dataset(out / "data.yaml") == with_file


def test_split_csv_sits_beside_a_lightning_pose_project(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    _mice, _rats, library = world
    run_id = _prepare(library, target="litpose")
    out = model_run_root(library, KIND, run_id)
    record = _split_record(out)

    assert (out / "project").is_dir()
    assert sorted(record["placed_name"]) == sorted(
        path.name for path in (out / "images").iterdir()
    )
    with_file = fingerprint_dataset(out / "project")
    (out / SPLIT_RECORD_FILENAME).unlink()
    assert fingerprint_dataset(out / "project") == with_file


@pytest.fixture
def grouped(tmp_path: Path) -> Dataset:
    """Return a library with one set of four recordings.

    ``m01`` and ``m02`` contain a mouse on two finished frames each, ``m03``
    contains only a cricket, and the one frame of ``m04`` is unfinished.
    """
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    mouse = pose_object((10.0, 12.0), (30.0, 20.0))
    frames = [
        pose_frame(
            _image(library, sequence, index),
            mouse,
            sequence=sequence,
            frame_index=index,
        )
        for sequence in ("m01", "m02")
        for index in range(2)
    ]
    cricket = pose_object((30.0, 40.0), (36.0, 42.0), pose_id=CRICKET.id)
    frames.append(pose_frame(_image(library, "m03", 0), cricket, sequence="m03"))
    frames.append(
        pose_frame(_image(library, "m04", 0), mouse, sequence="m04", usable=False)
    )
    state = pose_set(frames, poses=(MOUSE, CRICKET), image_root=library.base_dir)
    _ = _save(library, "grouped", state)
    return library


def _grouped(sequence_groups: dict[str, str]) -> dict[str, object]:
    return {
        "sets": [{"set_key": "grouped"}],
        "pose": "mouse",
        "split_by": "group",
        "split": (0.5, 0.5, 0.0),
        "sequence_groups": sequence_groups,
    }


def _unmatched_refusal(*keys: str) -> str:
    listed = ", ".join(repr(key) for key in keys)
    return (
        f"these sequence_groups keys match no sequence of sets or holdout: {listed}. "
        "Key a sequence by its name, or by <origin_uuid>:<group>:<sequence> with "
        "each part percent-encoded where names repeat."
    )


def test_a_group_key_for_any_recording_of_the_sets_is_accepted(
    grouped: Dataset,
) -> None:
    """Keys name recordings of the whole saved state, whatever the pose or finish.

    ``m03`` contains only a cricket and ``m04`` only an unfinished frame, and their
    keys are accepted and unused.
    """
    uuid = grouped.uuid or ""
    params = _grouped({"m01": "a", f"{uuid}::m02": "b", "m03": "a", "m04": "b"})

    check_preparation(grouped, PrepareTrainingDataParams.model_validate(params))
    _ = run_op(grouped, KIND, params)


@pytest.mark.parametrize(
    "key", ["m09", "{uuid}::m0l"], ids=["unknown-name", "mistyped-key"]
)
def test_a_group_key_that_names_no_recording_is_refused(
    grouped: Dataset, key: str
) -> None:
    """A group split does not degrade to a sequence split without a message."""
    unmatched = key.format(uuid=grouped.uuid or "")
    params = _grouped({unmatched: "a", "m01": "a", "m02": "b"})
    refusal = _unmatched_refusal(unmatched)

    assert _refusal_of_both(grouped, params) == [refusal, refusal]


def test_a_group_key_in_the_older_spelling_is_refused_with_the_newer_one(
    grouped: Dataset,
) -> None:
    uuid = grouped.uuid or ""
    older = f"{uuid}:m01"
    refusal = (
        f"sequence_groups key {older!r} uses the <origin_uuid>:<sequence> spelling, "
        "which prepare-training-data 0.3 no longer reads. "
        f"Key it as {f'{uuid}::m01'!r}."
    )

    assert _refusal_of_both(grouped, _grouped({older: "a", "m02": "b"})) == [
        refusal,
        refusal,
    ]


def test_the_older_spelling_of_a_name_in_two_groups_is_given_both_keys(
    extracted: Dataset,
) -> None:
    uuid = extracted.uuid or ""
    params = PrepareTrainingDataParams.model_validate(
        {
            "sets": [{"set_key": "extracted"}],
            "split_by": "group",
            "sequence_groups": {f"{uuid}:s1": "a"},
        }
    )

    with pytest.raises(ValueError) as refused:
        check_preparation(extracted, params)

    assert str(refused.value).endswith(
        f"Key it as {f'{uuid}:A:s1'!r} or {f'{uuid}:B:s1'!r}."
    )


@pytest.mark.parametrize("key", ["m09", "{uuid}:m01"], ids=["unmatched", "older"])
def test_the_check_refuses_a_group_key_and_writes_nothing(
    grouped: Dataset, key: str
) -> None:
    params = PrepareTrainingDataParams.model_validate(
        _grouped({key.format(uuid=grouped.uuid or ""): "a"})
    )
    before = tree_snapshot(grouped.base_dir)

    with pytest.raises(ValueError, match="sequence_groups key"):
        check_preparation(grouped, params)

    assert_tree_unchanged(grouped.base_dir, before)


# ------------------------------------------------------------------------ hold-out


@pytest.fixture
def held_out(tmp_path: Path) -> Dataset:
    """Return a library with a set to train on and sets to hold out of it.

    ``train`` contains frames 0 to 3 of each of ``m01`` to ``m04``. ``m01-start``
    contains frames 0 and 1 of ``m01``, which ``train`` contains too, ``h01`` two
    frames of a recording that ``train`` lacks, ``all-four`` frame 0 of every
    recording of ``train``, and ``every-frame`` every frame of ``train``.
    ``unfinished`` contains one unfinished frame, and ``other-layout`` the mouse
    pose with a third keypoint.
    """
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    every_recording = {"m01": 4, "m02": 4, "m03": 4, "m04": 4}
    _ = _save_set(library, "train", every_recording)
    _ = _save_set(library, "every-frame", every_recording, shift=1.0)
    _ = _save_set(library, "m01-start", {"m01": 2})
    _ = _save_set(library, "h01", {"h01": 2})
    _ = _save_set(library, "all-four", {"m01": 1, "m02": 1, "m03": 1, "m04": 1})
    unfinished = pose_frame(
        _image(library, "m05", 0),
        pose_object((10.0, 12.0), (30.0, 20.0)),
        sequence="m05",
        usable=False,
    )
    _ = _save(
        library, "unfinished", pose_set([unfinished], image_root=library.base_dir)
    )
    three_points = KeypointSchema(names=("nose", "ear", "tail"))
    _one_pose_set(library, "other-layout", replace(MOUSE, schema=three_points))
    return library


def _with_holdout(*holdout: str, **overrides: object) -> dict[str, object]:
    """Return the parameters that train on ``train`` and hold out *holdout*."""
    return {
        "sets": [{"set_key": "train"}],
        "holdout": [{"set_key": key} for key in holdout],
        "split": (0.5, 0.5, 0.0),
        **overrides,
    }


def _record_plans(monkeypatch: pytest.MonkeyPatch) -> list[SplitPlan]:
    """Record every split plan that a preparation builds from now on."""
    plans: list[SplitPlan] = []

    def record(
        *,
        name_by_frame: dict[int, str],
        split_of: dict[str, str],
        rows: tuple[SplitRow, ...],
        n_held_out: int,
    ) -> SplitPlan:
        plan = SplitPlan(
            name_by_frame=name_by_frame,
            split_of=split_of,
            rows=rows,
            n_held_out=n_held_out,
        )
        plans.append(plan)
        return plan

    monkeypatch.setattr(prepare, "SplitPlan", record)
    return plans


def _excluded(record: pd.DataFrame) -> list[str]:
    """Return the placed names that a split record leaves out, sorted."""
    return sorted(record.loc[record["split"] == "excluded", "placed_name"])


@pytest.mark.parametrize(
    ("split_by", "sequence_groups", "removed"),
    [
        ("frame", {}, [("m01", 0), ("m01", 1)]),
        ("sequence", {}, [("m01", index) for index in range(4)]),
        (
            "group",
            {"m01": "g", "m02": "g"},
            [(sequence, index) for sequence in ("m01", "m02") for index in range(4)],
        ),
    ],
    ids=["frame", "sequence", "group"],
)
def test_a_hold_out_removes_every_training_frame_of_its_units(
    held_out: Dataset,
    monkeypatch: pytest.MonkeyPatch,
    split_by: str,
    sequence_groups: dict[str, str],
    removed: list[tuple[str, int]],
) -> None:
    """``m01-start`` contains frames 0 and 1 of ``m01``, as ``train`` does."""
    plans = _record_plans(monkeypatch)
    params = _with_holdout(
        "m01-start", split_by=split_by, sequence_groups=sequence_groups
    )
    out = model_run_root(held_out, KIND, run_op(held_out, KIND, params))

    record = _split_record(out)
    excluded = sorted(
        _placed(held_out, "train", sequence, index) for sequence, index in removed
    )
    assert _excluded(record) == excluded
    assert set(record["set_key"]) == {"train"}, "only training frames are rows"
    assert sorted(_splits_by_name(out)) == sorted(
        set(record["placed_name"]) - set(excluded)
    )
    assert {plan.n_held_out for plan in plans} == {len(excluded)}


def test_a_hold_out_recording_keyed_into_a_group_holds_out_the_group(
    held_out: Dataset,
) -> None:
    """``h01`` is a recording of the hold-out alone, and its key is accepted."""
    params = _with_holdout(
        "h01", split_by="group", sequence_groups={"h01": "g", "m02": "g"}
    )

    check_preparation(held_out, PrepareTrainingDataParams.model_validate(params))
    out = model_run_root(held_out, KIND, run_op(held_out, KIND, params))

    assert _excluded(_split_record(out)) == sorted(
        _placed(held_out, "train", "m02", index) for index in range(4)
    )
    without = PrepareTrainingDataParams.model_validate({**params, "holdout": []})
    with pytest.raises(
        ValueError, match=r"match no sequence of sets or holdout: 'h01'"
    ):
        check_preparation(held_out, without)


def test_under_split_by_frame_a_hold_out_of_one_camera_holds_out_the_other(
    extracted: Dataset,
) -> None:
    entry = extracted.get_root("frames") / "uniform" / "uniform-1"
    left = entry / make_entry_key("C", "s2") / "left" / "frame_000100.png"
    frame = pose_frame(
        extracted.relative_to_root(left),
        pose_object((10.0, 12.0), (30.0, 20.0)),
        sequence="s2",
        frame_index=100,
    )
    _ = _save(extracted, "left-100", pose_set([frame], image_root=extracted.base_dir))
    params: dict[str, object] = {
        "sets": [{"set_key": "extracted"}],
        "holdout": [{"set_key": "left-100"}],
        "split_by": "frame",
        "split": (0.5, 0.5, 0.0),
    }

    out = model_run_root(extracted, KIND, run_op(extracted, KIND, params))

    assert _excluded(_split_record(out)) == sorted(
        _placed(extracted, "extracted", camera, 100)
        for camera in ("C__s2__left", "C__s2__right")
    )


EVERY_GROUP: Final = {sequence: "g" for sequence in ("h01", "m01", "m02", "m03", "m04")}
"""Groups that put ``h01`` in one group with every recording of ``train``."""

HOLD_OUT_REFUSALS: Final[dict[str, tuple[tuple[str, ...], dict[str, object], str]]] = {
    "named-twice": (
        ("train",),
        {},
        "keypoint set 'train' is named in both sets and holdout. Train on it or "
        "hold it out.",
    ),
    "no-usable-frame": (
        ("unfinished",),
        {},
        "hold-out set 'unfinished' has no usable frames for pose 'mouse'",
    ),
    "every-recording": (
        ("all-four",),
        {},
        "the hold-out shares a recording with every training frame. Choose a "
        "hold-out from other recordings, or split by frame.",
    ),
    "every-sequence": (
        ("all-four",),
        {"split_by": "sequence"},
        "the hold-out shares a sequence with every training frame. Choose a "
        "hold-out from other sequences, or split by frame.",
    ),
    "every-group": (
        ("h01",),
        {"split_by": "group", "sequence_groups": EVERY_GROUP},
        "the hold-out shares a group with every training frame. Choose a "
        "hold-out from other groups, or split by frame.",
    ),
    "every-frame": (
        ("every-frame",),
        {"split_by": "frame"},
        "the hold-out shares a frame with every training frame. Choose a "
        "hold-out from other frames.",
    ),
}
"""Each hold-out refusal with a fixed message: hold-out, other parameters, message."""


@pytest.mark.parametrize(
    ("holdout", "overrides", "refusal"),
    list(HOLD_OUT_REFUSALS.values()),
    ids=list(HOLD_OUT_REFUSALS),
)
def test_a_hold_out_is_refused_by_the_check_and_by_a_run(
    held_out: Dataset,
    holdout: tuple[str, ...],
    overrides: dict[str, object],
    refusal: str,
) -> None:
    params = _with_holdout(*holdout, **overrides)

    assert _refusal_of_both(held_out, params) == [refusal, refusal]


@pytest.mark.parametrize("pose", [None, "mouse"], ids=["only-pose", "named"])
def test_a_hold_out_without_the_pose_is_refused_for_the_pose(
    held_out: Dataset, pose: str | None
) -> None:
    """The pose that a hold-out lacks is named before its lack of a finished frame."""
    uuid = held_out.uuid or ""
    unfinished = pose_frame(
        _image(held_out, "m06", 0),
        pose_object((30.0, 40.0), (36.0, 42.0), pose_id=CRICKET.id),
        sequence="m06",
        usable=False,
    )
    crickets = pose_set([unfinished], poses=(CRICKET,), image_root=held_out.base_dir)
    _ = _save(held_out, "crickets", crickets)
    refusal = {
        None: (
            f"set {uuid}:crickets@rev1 chose pose 8 (cricket), but set "
            f"{uuid}:train@rev1 chose pose 3 (mouse). One model trains one pose."
        ),
        "mouse": (
            f"set {uuid}:crickets@rev1 holds no pose 'mouse' (it holds ['8:cricket'])"
        ),
    }[pose]

    assert _refusal_of_both(held_out, _with_holdout("crickets", pose=pose)) == [
        refusal,
        refusal,
    ]


def test_a_hold_out_of_another_layout_is_refused(held_out: Dataset) -> None:
    """The hold-out is compared with the layout of the sets it is held out of."""
    uuid = held_out.uuid or ""
    refusal = (
        f"set {uuid}:other-layout@rev1 has keypoints ['nose', 'ear', 'tail'], but "
        f"set {uuid}:train@rev1 has ['nose', 'tail']. One model has one keypoint "
        "layout."
    )

    assert _refusal_of_both(held_out, _with_holdout("other-layout")) == [
        refusal,
        refusal,
    ]


@pytest.mark.parametrize(
    ("holdout", "overrides"),
    [
        *((holdout, overrides) for holdout, overrides, _ in HOLD_OUT_REFUSALS.values()),
        (("other-layout",), {}),
    ],
    ids=[*HOLD_OUT_REFUSALS, "other-layout"],
)
def test_the_check_refuses_a_hold_out_and_writes_nothing(
    held_out: Dataset, holdout: tuple[str, ...], overrides: dict[str, object]
) -> None:
    params = PrepareTrainingDataParams.model_validate(
        _with_holdout(*holdout, **overrides)
    )
    before = tree_snapshot(held_out.base_dir)

    with pytest.raises(ValueError):
        check_preparation(held_out, params)

    assert_tree_unchanged(held_out.base_dir, before)


@pytest.mark.parametrize("target", ["yolo-pose", "litpose"])
def test_a_held_out_frame_whose_image_is_gone_is_left_out(
    held_out: Dataset, target: str
) -> None:
    """Frame 3 of ``m01`` is held out with its recording. The writers skip its image."""
    gone = "media/frames/kmeans/kmeans-d7968c97b0/m01/frame_000003.png"
    (held_out.base_dir / gone).unlink()

    run_id = run_op(held_out, KIND, _with_holdout("m01-start", target=target))
    out = model_run_root(held_out, KIND, run_id)

    record = _split_record(out)
    placed = _placed(held_out, "train", "m01", 3)
    assert placed in _excluded(record)
    written = sorted(image.name for image in out.rglob("*.png"))
    assert written and not [name for name in written if "__m01__" in name]


def test_a_hold_out_names_the_preparation_by_its_units(held_out: Dataset) -> None:
    """A correction inside the hold-out keeps the name, and a new unit moves it."""
    op = OPS[KIND]()

    def plan(*holdout: str) -> str:
        params = PrepareTrainingDataParams.model_validate(_with_holdout(*holdout))
        return op.plan_identity(held_out, params, ResolvedScope()).run_id

    without, held = plan(), plan("m01-start")
    _ = _save_set(held_out, "m01-start", {"m01": 2}, shift=1.0, commit="c2")
    corrected = plan("m01-start")
    _ = _save_set(held_out, "m01-start", {"m01": 2, "m02": 1}, commit="c3")
    widened = plan("m01-start")

    assert held != without
    assert corrected == held
    assert widened not in {without, held}


def test_the_hold_out_enters_the_identity_only_by_its_units() -> None:
    """Without units the payload is the one minted before the hold-out existed."""
    sets = (
        ResolvedSet(
            origin_uuid="uuid-a",
            set_key="train",
            revision=1,
            digest="aaaaaaaaaaaaaaaa",
            payload=Path("train.json"),
        ),
    )
    plain = PrepareTrainingDataParams.model_validate({"sets": [{"set_key": "train"}]})
    with_sets_held_out = PrepareTrainingDataParams.model_validate(
        {"sets": [{"set_key": "train"}], "holdout": [{"set_key": "h01"}]}
    )
    dumped = plain.identity_dump()
    for name in ("sets", "holdout"):
        _ = dumped.pop(name)
    version = OPS[KIND].version
    payload: dict[str, object] = {
        "params": dumped,
        "sets": [["uuid-a", "train", "aaaaaaaaaaaaaaaa"]],
    }

    assert prepare_training_data_run_id(KIND, version, plain, sets) == op_run_id(
        KIND, version, payload
    )
    assert prepare_training_data_run_id(
        KIND, version, with_sets_held_out, sets, holdout_units=()
    ) == op_run_id(KIND, version, payload)
    assert prepare_training_data_run_id(
        KIND,
        version,
        with_sets_held_out,
        sets,
        holdout_units=("uuid-a::m02", "uuid-a::m01"),
    ) == op_run_id(
        KIND, version, {**payload, "holdout": ["uuid-a::m01", "uuid-a::m02"]}
    )


# ------------------------------------------------------------- divided recordings

DIVIDED_ENTRIES: Final = (
    ("night", "h01"),
    ("night", "h02"),
    ("night", "h03"),
    ("day", "h01"),
    ("day", "h02"),
    ("dawn", "h01"),
    ("lone", "s1"),
)
"""The entries of the ``divided`` set. Hourly tokens repeat across recordings."""

DIVIDED_UUID: Final = "08000000-0000-4000-8000-000000000000"


def _divided_library(tmp_path: Path, declared: tuple[str, ...]) -> Dataset:
    """Return a library with frames 0 and 1 of every entry, and *declared* divided.

    A frames run extracted the frames of every entry of ``DIVIDED_ENTRIES``, and
    the set ``hourly`` contains each of them. ``night-h02`` contains frame 0 of
    ``night/h02``, which ``hourly`` contains too. Both sets are saved after
    *declared* is declared, so their frames record its recordings.
    """
    library = make_dataset(
        tmp_path / "libraries" / "8", name="library", uuid=DIVIDED_UUID
    )
    add_track_sequences(library, *DIVIDED_ENTRIES)
    library.set_continuous_groups(declared)
    frames: list[PoseFrame] = []
    for group, sequence in DIVIDED_ENTRIES:
        images = extract_entry_camera(
            library,
            run_id="uniform-1",
            group=group,
            sequence=sequence,
            camera="",
            sizes={index: (64, 48) for index in range(2)},
        )
        frames.extend(
            pose_frame(
                image,
                pose_object((10.0 + index, 12.0), (30.0, 20.0)),
                sequence=sequence,
                group=group,
                frame_index=index,
            )
            for index, image in images.items()
        )
    _ = _save(library, "hourly", pose_set(frames, image_root=library.base_dir))
    (held,) = [
        frame
        for frame in frames
        if frame.group == "night" and frame.sequence == "h02" and frame.frame_index == 0
    ]
    _ = _save(
        library,
        "night-h02",
        pose_set([replace(held, objects=(pose_object((11.0, 13.0), (31.0, 21.0)),))],
                 image_root=library.base_dir),
    )  # fmt: skip
    return library


def _save_again(ds: Dataset, key: str, revision: int) -> int:
    """Save revision *revision* of *key* again, unchanged, and return the new number.

    The save records each frame's recording as *ds* declares it now.
    """
    saved = ds.get_root("labels_raw") / "keypoints" / key / revision_file(revision)
    return _save(ds, key, read_pose_annotations(saved, ds.base_dir))


@pytest.fixture
def divided(tmp_path: Path) -> Dataset:
    """Return the divided library with ``day`` and ``night`` declared continuous."""
    return _divided_library(tmp_path, ("day", "night"))


def _hourly(**overrides: object) -> dict[str, object]:
    return {"sets": [{"set_key": "hourly"}], "split": (0.5, 0.5, 0.0), **overrides}


HOURLY_UNITS: Final = {
    ("dawn", "h01"): f"{DIVIDED_UUID}:dawn:h01",
    ("day", "h01"): f"{DIVIDED_UUID}:day:",
    ("day", "h02"): f"{DIVIDED_UUID}:day:",
    ("lone", "s1"): f"{DIVIDED_UUID}:lone:s1",
    ("night", "h01"): f"{DIVIDED_UUID}:night:",
    ("night", "h02"): f"{DIVIDED_UUID}:night:",
    ("night", "h03"): f"{DIVIDED_UUID}:night:",
}
"""The unit of each entry of ``hourly`` saved with ``day`` and ``night`` divided."""


def _units(record: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Each entry of a split record, and the unit that its frames were drawn in."""
    return {
        (str(group), str(sequence)): str(unit)
        for group, sequence, unit in zip(
            record["group"], record["sequence"], record["unit_key"], strict=True
        )
    }


def _plan(ds: Dataset, params: dict[str, object]) -> str:
    validated = PrepareTrainingDataParams.model_validate(params)
    return OPS[KIND]().plan_identity(ds, validated, ResolvedScope()).run_id


def test_the_sequences_of_a_divided_recording_never_straddle_the_split(
    divided: Dataset,
) -> None:
    record = _split_record(
        model_run_root(divided, KIND, run_op(divided, KIND, _hourly()))
    )

    assert _units(record) == HOURLY_UNITS
    assert record.groupby("unit_key")["split"].nunique().eq(1).all()
    assert set(record["split"]) == {"train", "valid"}


def test_undivided_data_splits_as_a_sequence_split_does(extracted: Dataset) -> None:
    def record(**overrides: object) -> pd.DataFrame:
        params: dict[str, object] = {
            "sets": [{"set_key": "extracted"}],
            "split": (0.5, 0.5, 0.0),
            **overrides,
        }
        run_id = run_op(extracted, KIND, params)
        return _split_record(model_run_root(extracted, KIND, run_id))

    by_recording, by_sequence = record(), record(split_by="sequence")

    columns = ["placed_name", "unit_key", "split"]
    assert by_recording[columns].equals(by_sequence[columns])


def test_a_saved_undivided_recording_draws_its_sequence_on_its_own(
    divided: Dataset,
) -> None:
    """A revision written outside the toolkit records each frame's own recording."""
    saved = read_pose_annotations(
        divided.get_root("labels_raw") / "keypoints" / "hourly" / revision_file(1),
        divided.base_dir,
    )
    own = replace(
        saved,
        frames=tuple(
            replace(
                frame,
                recording=FrameRecording(
                    kind="undivided", key=RecordingKey(frame.group, frame.sequence)
                ),
            )
            for frame in saved.frames
        ),
    )
    _ = write_series_revision(
        divided,
        series=KEYPOINTS_SERIES,
        key="own",
        payload=pose_annotations_payload(own),
        origin={"dolt_commit": "c1"},
        n_records=len(own.frames),
        exports={COCO_EXPORT_FILENAME: coco_keypoints_payload(own)},
    )

    params = {"sets": [{"set_key": "own"}], "split": (0.5, 0.5, 0.0)}
    record = _split_record(model_run_root(divided, KIND, run_op(divided, KIND, params)))

    assert _units(record) == {
        entry: f"{DIVIDED_UUID}:{entry[0]}:{entry[1]}" for entry in HOURLY_UNITS
    }


@pytest.mark.parametrize(
    ("split_by", "removed"),
    [
        ("recording", [("night", sequence) for sequence in ("h01", "h02", "h03")]),
        ("sequence", [("night", "h02")]),
    ],
    ids=["recording", "sequence"],
)
def test_a_hold_out_is_keyed_by_its_recording(
    divided: Dataset, split_by: str, removed: list[tuple[str, str]]
) -> None:
    params = _hourly(holdout=[{"set_key": "night-h02"}], split_by=split_by)
    record = _split_record(model_run_root(divided, KIND, run_op(divided, KIND, params)))

    excluded = record.loc[record["split"] == "excluded"]
    assert sorted(set(zip(excluded["group"], excluded["sequence"], strict=True))) == (
        sorted(removed)
    )


def test_a_recording_split_is_named_by_its_parameters_and_revisions_alone(
    divided: Dataset,
) -> None:
    params = PrepareTrainingDataParams.model_validate(_hourly())
    sets = resolve_keypoint_sets(divided, params.sets, kind=KIND)

    assert _plan(divided, _hourly()) == prepare_training_data_run_id(
        KIND, OPS[KIND].version, params, sets
    )


def test_declaring_a_continuous_group_moves_the_identity_at_the_next_save(
    tmp_path: Path,
) -> None:
    """Each declared group that holds a sequence of the set moves it once saved."""
    library = _divided_library(tmp_path, ())
    identities = [_plan(library, _hourly())]

    for declared in (("night",), ("day", "night"), ("dawn", "day", "night")):
        library.set_continuous_groups(declared)
        assert _plan(library, _hourly()) == identities[-1], "the revision is unchanged"
        _ = _save_again(library, "hourly", len(identities))
        identities.append(_plan(library, _hourly()))
    library.set_continuous_groups(())
    _ = _save_again(library, "hourly", len(identities))

    assert len(set(identities)) == 4, "dawn holds one sequence"
    assert _plan(library, _hourly()) == identities[0]


def test_declaring_a_recording_of_one_annotated_sequence_moves_the_run(
    tmp_path: Path,
) -> None:
    """Its sequence is drawn as the recording, and the run id follows that draw."""
    library = _divided_library(tmp_path, ())

    def run() -> tuple[str, str]:
        run_id = run_op(library, KIND, _hourly())
        record = _split_record(model_run_root(library, KIND, run_id))
        return run_id, _units(record)[("dawn", "h01")]

    undeclared, undeclared_unit = run()
    library.set_continuous_groups(("dawn",))
    _ = _save_again(library, "hourly", 1)
    declared, declared_unit = run()

    assert (undeclared_unit, declared_unit) == (
        f"{DIVIDED_UUID}:dawn:h01",
        f"{DIVIDED_UUID}:dawn:",
    )
    assert declared != undeclared


def test_sequence_groups_is_refused_under_a_recording_split() -> None:
    with pytest.raises(ValidationError, match="sequence_groups applies only"):
        _ = PrepareTrainingDataParams.model_validate(
            {"sets": [{"set_key": "hourly"}], "sequence_groups": {"h01": "a"}}
        )
    _ = PrepareTrainingDataParams.model_validate(
        {
            "sets": [{"set_key": "hourly"}],
            "split_by": "group",
            "sequence_groups": {"h01": "a"},
        }
    )


def _grouped_frames(ds: Dataset, groups: tuple[str, ...]) -> PoseAnnotationSet:
    """Frame 0 of sequence ``s1`` of each of *groups*, under no frames run."""
    frames: list[PoseFrame] = []
    for group in groups:
        relative = Path(f"media/frames/annotated/{group}/s1/frame_000000.png")
        image = ds.base_dir / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        _ = image.write_bytes(f"{group}/s1/0".encode())
        frames.append(
            pose_frame(
                relative,
                pose_object((10.0, 12.0), (30.0, 20.0)),
                sequence="s1",
                group=group,
            )
        )
    return pose_set(frames, image_root=ds.base_dir)


def test_frames_of_one_sequence_name_in_two_groups_are_two_units(
    tmp_path: Path,
) -> None:
    """A frame that no frames run extracted is named by the group it records."""
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    _ = _save(library, "grouped", _grouped_frames(library, ("A", "B")))
    params = {"sets": [{"set_key": "grouped"}], "split": (0.5, 0.5, 0.0)}

    record = _split_record(model_run_root(library, KIND, run_op(library, KIND, params)))

    uuid = library.uuid or ""
    assert sorted(record["unit_key"]) == [f"{uuid}:A:s1", f"{uuid}:B:s1"]
    assert sorted(record["group"]) == ["A", "B"]
    assert record["placed_name"].nunique() == 2


def _claiming_library(tmp_path: Path, project: Dataset) -> Dataset:
    """Return a library that claims revision 1 of *project*'s ``hourly`` and links none."""
    library = make_dataset(
        tmp_path / "libraries" / "7", name="claims", uuid=WORLD_UUIDS["library"]
    )
    _claim(library, project, "hourly", 1)
    return library


def test_a_library_draws_the_recordings_of_a_project_that_it_does_not_link(
    tmp_path: Path,
) -> None:
    library = _claiming_library(tmp_path, _divided_library(tmp_path, ("day", "night")))

    record = _split_record(
        model_run_root(library, KIND, run_op(library, KIND, _hourly()))
    )

    assert library.manifest.libraries == ()
    assert _units(record) == HOURLY_UNITS
    assert record.groupby("unit_key")["split"].nunique().eq(1).all()


def test_a_declaration_changed_after_the_save_leaves_the_split_unchanged(
    divided: Dataset,
) -> None:
    """``night`` is withdrawn and ``dawn`` declared, and nothing is saved again."""
    divided.set_continuous_groups(("dawn", "day"))

    record = _split_record(
        model_run_root(divided, KIND, run_op(divided, KIND, _hourly()))
    )

    assert _units(record) == HOURLY_UNITS


def test_a_preparation_follows_the_declaration_that_its_revision_records(
    tmp_path: Path,
) -> None:
    """A library that links the project claims it, then ``night`` is withdrawn."""
    project = _divided_library(tmp_path, ("day", "night"))
    library = _claiming_library(tmp_path, project)
    _ = library.add_library(LibraryLink(id="project", path="../8"))
    claimed = _plan(library, _hourly())

    project.set_continuous_groups(("day",))
    run_id = run_op(library, KIND, _hourly())
    later = make_dataset(tmp_path / "libraries" / "9", name="later")
    _claim(later, project, "hourly", _save_again(project, "hourly", 1))
    resaved = _split_record(model_run_root(later, KIND, run_op(later, KIND, _hourly())))

    assert run_id == claimed
    assert _units(_split_record(model_run_root(library, KIND, run_id))) == (
        HOURLY_UNITS
    )
    assert _units(resaved) == {
        **HOURLY_UNITS,
        **{
            ("night", sequence): f"{DIVIDED_UUID}:night:{sequence}"
            for sequence in ("h01", "h02", "h03")
        },
    }


def test_a_frame_is_drawn_by_the_declaration_of_the_dataset_it_came_from(
    tmp_path: Path,
) -> None:
    """The library's revision records ``day`` and ``night``, and the project ``dawn``."""
    library = _divided_library(tmp_path, ("day", "night"))
    project = make_dataset(tmp_path / "52", name="mice", uuid=WORLD_UUIDS["mice"])
    add_track_sequences(project, ("dawn", "h01"), ("dawn", "h02"))
    project.set_continuous_groups(("dawn",))
    _ = project.add_library(LibraryLink(id="group", path="../libraries/8"))
    _claim(project, library, "hourly", 1)

    run_id = run_op(project, KIND, _hourly())

    record = _split_record(model_run_root(project, KIND, run_id))
    assert _units(record) == HOURLY_UNITS


def test_a_link_that_is_not_reachable_leaves_the_preparation_unchanged(
    tmp_path: Path,
) -> None:
    """The library links the project at a path that is gone, and claims in place."""
    project = _divided_library(tmp_path, ("day", "night"))
    library = _claiming_library(tmp_path, project)
    mounted = tmp_path / "mounted"
    mounted.symlink_to(project.base_dir, target_is_directory=True)
    _ = library.add_library(LibraryLink(id="project", path=str(mounted)))
    linked = _plan(library, _hourly())

    mounted.unlink()
    check_preparation(library, PrepareTrainingDataParams.model_validate(_hourly()))
    run_id = run_op(library, KIND, _hourly())

    assert run_id == linked
    assert _units(_split_record(model_run_root(library, KIND, run_id))) == HOURLY_UNITS


def test_a_derived_box_trains_as_the_annotator_saw_it(full_state: Dataset) -> None:
    """Unset, the set's own padding applies; a policy given to the run replaces it."""
    resident = _ears(10.0, EARS.aliases[0])
    seen = derived_bbox(resident.keypoints, 64, 48, PADDING)

    own = _label(_prepare_full(full_state), "m01", 0)
    tight = _label(
        _prepare_full(full_state, bbox={"method": "tight", "margin": 0.0}), "m01", 0
    )

    widths = sorted(float(row[3]) * 64 for row in own)
    assert widths[0] == pytest.approx(seen.width, abs=1e-3)
    assert sorted(float(row[3]) * 64 for row in tight) == pytest.approx([8.0, 8.0])


def test_one_set_named_twice_is_refused(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    """Two revisions of a set show the same images, which one tree cannot hold."""
    _mice, _rats, library = world
    params = PrepareTrainingDataParams.model_validate(
        {
            "sets": [
                {"set_key": "17-openfield"},
                {"set_key": "17-openfield", "revision": 1},
            ]
        }
    )

    with pytest.raises(ValueError, match="named twice"):
        _ = OPS[KIND]().plan_identity(library, params, ResolvedScope())


def test_how_a_set_is_narrowed_is_part_of_the_dataset_it_names(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    _mice, _rats, library = world
    op = OPS[KIND]()

    def plan(**choice: object) -> str:
        params = PrepareTrainingDataParams.model_validate(
            {"sets": [{"set_key": "17-openfield"}], **choice}
        )
        return op.plan_identity(library, params, ResolvedScope()).run_id

    names = {plan(), plan(pose="mouse"), plan(class_by="alias"), plan(bbox={})}
    assert len(names) == 4


# ------------------------------------------------------- training on a preparation


def test_training_data_may_be_named_by_the_run_that_prepared_it(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    _mice, _rats, library = world
    run_id = _prepare(library)
    data_yaml = model_run_root(library, KIND, run_id) / "data.yaml"

    assert resolve_training_data(library, run_id) == data_yaml
    assert resolve_training_data(library, str(data_yaml)) == data_yaml, (
        "a path still works"
    )
    unknown = "prepare-training-data.0.1-0000000000"
    assert resolve_training_data(library, unknown).name == unknown, (
        "the caller reports it"
    )


def test_a_train_identity_over_a_run_id_carries_no_location(
    world: tuple[Dataset, Dataset, Dataset], tmp_path: Path
) -> None:
    """The reference string is hashed, so it has to be content and not a place."""
    _mice, _rats, library = world
    run_id = _prepare(library)
    op = OPS["train-pose"]()
    by_run = op.plan_identity(
        library, PoseTrainParams(data=run_id, epochs=1), ResolvedScope()
    ).run_id

    moved = tmp_path / "moved-library"
    _ = library.base_dir.rename(moved)
    relocated = Dataset(manifest_path=moved / "dataset.yaml").load()
    after_move = op.plan_identity(
        relocated, PoseTrainParams(data=run_id, epochs=1), ResolvedScope()
    ).run_id

    assert after_move == by_run, "moving the library re-mints nothing"


# ------------------------------------------------------------------- provenance


def _train(library: Dataset, prepared_run_id: str) -> str:
    """A finished ``train-pose`` row over the prepared data, without a trainer."""
    kind, run_id = "train-pose", "train-pose.0.2-feedfacade"
    data_yaml = resolve_training_data(library, prepared_run_id)
    run_root = model_run_root(library, kind, run_id)
    weights = run_root / "train" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    _ = weights.write_bytes(b"weights")
    finalize_training(
        library, kind, run_id, run_root, PoseTrainParams(data=prepared_run_id, epochs=1),
        "", "", "", weights, run_root / "train" / "results.csv", 1,
        data_path=data_yaml, data_fingerprint=fingerprint_yolo_dataset(data_yaml),
    )  # fmt: skip
    return run_id


def test_a_model_names_the_annotation_revisions_it_saw(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    mice, rats, library = world
    model = _train(library, _prepare(library))

    found = training_provenance(library, "train-pose", model)

    assert found.prepared_kind == KIND and found.stopped_at == ""
    assert {(s.origin_uuid, s.set_key, s.revision) for s in found.sets} == {
        (mice.uuid, "17-openfield", 1),
        (rats.uuid, "21-arena", 1),
    }
    assert all(s.reachable and s.origin == {"dolt_commit": "c1"} for s in found.sets)


def test_a_model_names_the_pose_it_was_trained_on(full_state: Dataset) -> None:
    model = _train(full_state, run_op(full_state, KIND, {**FULL, "pose": "cricket"}))

    found = training_provenance(full_state, "train-pose", model)

    assert (found.pose_id, found.pose_name) == (CRICKET.id, CRICKET.name)
    document = found.as_json()
    assert (document["pose_id"], document["pose_name"]) == (CRICKET.id, CRICKET.name)


def test_a_pose_model_names_how_many_keypoints_it_predicts(full_state: Dataset) -> None:
    """Read from the prepared data's kpt_shape, which every pose preparation wrote,
    so a model prepared before any other record of it still answers."""
    model = _train(full_state, run_op(full_state, KIND, {**FULL, "pose": "cricket"}))

    found = training_provenance(full_state, "train-pose", model)

    assert found.keypoint_count == len(CRICKET.schema.names)
    assert found.as_json()["keypoint_count"] == len(CRICKET.schema.names)


def test_an_index_older_than_the_pose_reads_blank_and_is_adopted_on_the_next_write(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    """A row from before the pose was recorded names none, which reads as unknown.

    The older file holds its columns out of schema order and one the schema has
    retired, so an appended row alone does not bring it to the schema. The
    adoption does.
    """
    _mice, _rats, library = world
    earlier = _prepare(library)
    model = _train(library, earlier)
    path = model_index_path(library, KIND)
    written = pd.read_csv(path, dtype=str, keep_default_na=False)
    legacy = written.drop(columns=["pose_id", "pose_name"])
    legacy = legacy[list(reversed(legacy.columns))].assign(retired="x")
    legacy.to_csv(path, index=False)

    found = training_provenance(library, "train-pose", model)
    assert (found.pose_id, found.pose_name) == (None, "")
    assert found.keypoint_count is not None
    assert found.stopped_at == "", "the rest of the chain still reads"

    later = _prepare(library, target="polo")

    adopted = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(adopted.columns) == [
        field.name for field in fields(PreparedDatasetIndexRow)
    ]
    assert "retired" not in adopted.columns
    rows = {row["run_id"]: row for row in index_records(adopted)}
    assert (rows[earlier]["pose_id"], rows[earlier]["pose_name"]) == ("", "")
    assert (rows[later]["pose_id"], rows[later]["pose_name"]) == (
        str(MOUSE.id),
        MOUSE.name,
    )
    assert rows[earlier]["n_frames"] == "16", "an integer cell is not widened"


def test_the_same_answer_from_a_project_that_links_the_library(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    mice, _rats, library = world
    model = _train(library, _prepare(library))
    _ = mice.add_library(LibraryLink(id="group", path="../libraries/7"))

    found = training_provenance(mice, "train-pose", model)

    assert found.served_by == "group"
    assert len(found.sets) == 2


def test_an_archived_project_is_named_as_where_the_chain_stops(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    """The model survives: its prepared tree holds a copy. The chain says what is gone."""
    _mice, rats, library = world
    prepared = _prepare(library)
    model = _train(library, prepared)

    (rats.get_root("labels_raw") / "keypoints/21-arena" / revision_file(1)).unlink()

    found = training_provenance(library, "train-pose", model)
    assert "21-arena rev1" in found.stopped_at
    assert [s.reachable for s in found.sets if s.set_key == "21-arena"] == [False]
    assert (
        len(list(model_run_root(library, KIND, prepared).rglob("labels/*.txt"))) == 16
    )


def test_a_model_trained_from_a_bare_folder_says_so(tmp_path: Path) -> None:
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    exported = tmp_path / "exported" / "data.yaml"
    exported.parent.mkdir(parents=True)
    _ = exported.write_text(
        "train: train/images\nval: valid/images\nnc: 1\nnames: [a]\n"
    )
    kind, run_id = "train-pose", "train-pose.0.2-0ddba11000"
    run_root = model_run_root(library, kind, run_id)
    weights = run_root / "best.pt"
    weights.parent.mkdir(parents=True)
    _ = weights.write_bytes(b"w")
    finalize_training(
        library, kind, run_id, run_root, PoseTrainParams(data=str(exported), epochs=1),
        "", "", "", weights, run_root / "results.csv", 1, data_path=exported, data_fingerprint="x",
    )  # fmt: skip

    found = training_provenance(library, kind, run_id)

    assert found.sets == () and "not written by a preparation run" in found.stopped_at
    assert trained_model_index(model_index_path(library, kind)).read(
        run_id=run_id
    ).iloc[0]["data_path"] == str(exported)


def test_the_inventory_tells_a_model_from_the_data_behind_it(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    _mice, _rats, library = world
    prepared = _prepare(library)
    model = _train(library, prepared)

    found = inventory(
        library, kinds=["trained-model", "prepared-dataset", "label-series"]
    )
    by_kind = {record.ref.kind: record for record in found.records}

    assert by_kind["trained-model"].run_id == model
    assert by_kind["prepared-dataset"].run_id == prepared
    assert by_kind["prepared-dataset"].status == "complete"
    assert sum(1 for r in found.records if r.ref.kind == "label-series") == 2


def test_the_provenance_command_prints_the_chain(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    from typer.testing import CliRunner

    from mosaic.cli import app

    mice, _rats, library = world
    model = _train(library, _prepare(library))
    _ = mice.add_library(LibraryLink(id="group", path="../libraries/7"))

    shown = CliRunner().invoke(
        app, ["models", "provenance", "-m", str(mice.manifest_path), model, "--json"]
    )

    assert shown.exit_code == 0, shown.output
    document = json.loads(shown.output)
    assert document["served_by"] == "group"
    assert {entry["set_key"] for entry in document["sets"]} == {
        "17-openfield",
        "21-arena",
    }
    assert all(entry["origin"] == {"dolt_commit": "c1"} for entry in document["sets"])
    assert (document["pose_id"], document["pose_name"]) == (MOUSE.id, MOUSE.name)
    printed = CliRunner().invoke(
        app, ["models", "provenance", "-m", str(mice.manifest_path), model]
    )
    assert printed.exit_code == 0 and f"mouse (id {MOUSE.id})" in printed.output

    missing = CliRunner().invoke(
        app,
        [
            "models",
            "provenance",
            "-m",
            str(mice.manifest_path),
            "train-pose.0.2-0000000000",
        ],
    )
    assert missing.exit_code != 0 and "is registered" in missing.output


# ------------------------------------------------------------ provenance columns


_CONSUMED_KEYS: Final = {"origin_uuid", "set_key", "revision", "digest"}


def _row_of(library: Dataset, run_id: str) -> dict[str, str]:
    frame = pd.read_csv(
        model_index_path(library, KIND), dtype=str, keep_default_na=False
    )
    return next(row for row in index_records(frame) if row["run_id"] == run_id)


def test_a_preparation_without_a_hold_out_records_an_empty_list(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    _mice, _rats, library = world

    row = _row_of(library, _prepare(library))

    assert row["held_out_sets"] == "[]"
    assert row["n_held_out"] == "0"


def test_a_hold_out_is_recorded_with_its_revisions_and_its_frames(
    held_out: Dataset,
) -> None:
    run_id = run_op(held_out, KIND, _with_holdout("m01-start", "h01"))

    row = _row_of(held_out, run_id)

    uuid = held_out.uuid or ""
    recorded = json.loads(row["held_out_sets"])
    assert [(entry["origin_uuid"], entry["set_key"]) for entry in recorded] == [
        (uuid, "h01"),
        (uuid, "m01-start"),
    ]
    assert all(
        entry["revision"] == 1 and entry["digest"] and set(entry) == _CONSUMED_KEYS
        for entry in recorded
    )
    assert row["n_held_out"] == "4", "every frame of recording m01, the unit held out"


@pytest.mark.parametrize("target", ["yolo-pose", "sleap", "litpose"])
def test_a_pose_target_records_the_narrowed_pose(
    world: tuple[Dataset, Dataset, Dataset],
    target: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mice, _rats, library = world
    _ = record_slp_writes(monkeypatch)

    row = _row_of(library, _prepare(library, target=target))

    assert json.loads(row["keypoint_schema"]) == {
        "keypoints": ["nose", "tail"],
        "skeleton": [[0, 1]],
        "symmetries": [],
    }


def test_a_wider_pose_records_its_skeleton_and_mirror_pairs(
    full_state: Dataset,
) -> None:
    """The three-keypoint mouse, unlike the two-keypoint worlds, keeps its pairs."""
    run_id = run_op(full_state, KIND, FULL)

    row = _row_of(full_state, run_id)

    assert json.loads(row["keypoint_schema"]) == {
        "keypoints": ["nose", "left_ear", "right_ear"],
        "skeleton": [[0, 1], [0, 2]],
        "symmetries": [[1, 2]],
    }


def test_polo_records_the_one_keypoint_it_trains_on(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    _mice, _rats, library = world

    row = _row_of(library, _prepare(library, target="polo", point_index=1))

    assert json.loads(row["keypoint_schema"]) == {
        "keypoints": ["tail"],
        "skeleton": [],
        "symmetries": [],
    }


def test_an_index_older_than_the_hold_out_columns_adopts_them_blank(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    _mice, _rats, library = world
    earlier = _prepare(library)
    rewrite_as_older_prepared_index(library)

    later = _prepare(library, target="polo")

    assert _row_of(library, earlier)["held_out_sets"] == ""
    assert _row_of(library, earlier)["n_held_out"] == ""
    assert _row_of(library, earlier)["keypoint_schema"] == ""
    assert _row_of(library, later)["held_out_sets"] == "[]"


def test_a_model_names_the_hold_out_it_never_saw(held_out: Dataset) -> None:
    prepared = run_op(held_out, KIND, _with_holdout("m01-start", "h01"))
    model = _train(held_out, prepared)

    found = training_provenance(held_out, "train-pose", model)

    assert [(item.set_key, item.revision) for item in found.held_out_sets] == [
        ("h01", 1),
        ("m01-start", 1),
    ]
    assert all(item.reachable for item in found.held_out_sets)
    reported = found.as_json()["held_out_sets"]
    assert isinstance(reported, list)
    assert [entry["set_key"] for entry in reported if isinstance(entry, dict)] == [
        "h01",
        "m01-start",
    ]
    assert found.sets and {item.set_key for item in found.sets} == {"train"}


def test_a_model_prepared_without_a_hold_out_reports_none(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    _mice, _rats, library = world

    found = training_provenance(
        library, "train-pose", _train(library, _prepare(library))
    )

    assert found.held_out_sets == ()
    assert found.as_json()["held_out_sets"] == []
