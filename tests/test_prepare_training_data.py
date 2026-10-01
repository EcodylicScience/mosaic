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

import pandas as pd
import pytest
import yaml

from mosaic.core.annotations.bbox import BboxPolicy, derived_bbox
from mosaic.core.annotations.model import AnnotationSet, KeypointSchema
from mosaic.core.annotations.pose_annotations import (
    Alias,
    PoseAnnotationSet,
    PoseDefinition,
    PoseFrame,
    PoseObject,
)
from mosaic.core.annotations.projection import write_keypoint_set_revision
from mosaic.core.dataset import Dataset
from mosaic.core.manifest import LabelsScanSource, LibraryLink
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.inventory import inventory
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.core.pipeline.ops import OPS, IdentityDeferred, run_op
from mosaic.tracking import register_ops
from mosaic.tracking.ops._common import (
    fingerprint_yolo_dataset,
    resolve_training_data,
)
from mosaic.tracking.ops.prepare import (
    PreparedDatasetIndexRow,
    PrepareTrainingDataParams,
    check_preparation,
    prepared_dataset_index,
)
from mosaic.tracking.ops.train import (
    PoseTrainParams,
    finalize_training,
    trained_model_index,
)
from mosaic.tracking.training_provenance import training_provenance
from tests.helpers import (
    MOUSE,
    make_dataset,
    pose_frame,
    pose_object,
    pose_set,
    revision_file,
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


@pytest.fixture
def world(tmp_path: Path) -> tuple[Dataset, Dataset, Dataset]:
    """Two projects with a saved set each, and a library that has claimed both."""
    mice = make_dataset(tmp_path / "52", name="mice")
    rats = make_dataset(tmp_path / "53", name="rats")
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
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

    Both sets hold the one pose, edited between them: a pose keeps its id when its
    layout changes, so the pose check passes and the layout check refuses.
    """
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    _ = _save_set(library, "two-points", {"m01": 2})
    _one_pose_set(library, "other", replace(MOUSE, schema=schema))

    with pytest.raises(ValueError, match="One model has one keypoint layout"):
        _ = run_op(
            library, KIND, {"sets": [{"set_key": "two-points"}, {"set_key": "other"}]}
        )


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


def test_the_sleap_writer_is_handed_one_set_with_collision_free_images(
    world: tuple[Dataset, Dataset, Dataset], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``.slp`` itself needs the SLEAP environment; what it is given does not."""
    _mice, _rats, library = world
    seen: dict[str, object] = {}

    def fake_write_slp(
        annotations: AnnotationSet, out_path: Path, **kwargs: object
    ) -> Path:
        seen["names"] = [frame.image_path.name for frame in annotations.frames]
        seen["root"] = annotations.image_root
        seen["images_dir"] = kwargs.get("images_dir")
        _ = Path(out_path).write_bytes(b"slp")
        return Path(out_path)

    monkeypatch.setattr("mosaic.tracking.sleap.labels.write_slp", fake_write_slp)

    run_id = _prepare(library, target="sleap")

    out = model_run_root(library, KIND, run_id)
    names = seen["names"]
    assert isinstance(names, list) and len(names) == 16 and len(set(names)) == 16
    assert seen["root"] == out / "images" == seen["images_dir"]
    assert all((out / "images" / str(name)).is_file() for name in names)
    assert resolve_training_data(library, run_id) == out / "labels.slp"


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

    def fake_write_slp(
        annotations: AnnotationSet, out_path: Path, **kwargs: object
    ) -> Path:
        _ = out_path.write_bytes(b"slp")
        return out_path

    monkeypatch.setattr("mosaic.tracking.sleap.labels.write_slp", fake_write_slp)

    run_id = run_op(full_state, KIND, {**FULL, "pose": pose, "target": target})

    assert _recorded_pose(full_state, run_id) == (str(chosen.id), chosen.name)


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


@pytest.mark.parametrize(
    ("overrides", "empty"),
    [
        ({"split_by": "sequence"}, "validation"),
        ({"split_by": "sequence", "split": (0.3, 0.3, 0.4)}, "training"),
    ],
)
def test_a_split_leaving_a_tree_without_training_or_validation_images_is_refused(
    full_state: Dataset, overrides: dict[str, object], empty: str
) -> None:
    """Whole recordings are drawn, and two cannot fill three splits.

    The trainer would otherwise fail on the empty directory, after the preparation
    reported success.
    """
    params = PrepareTrainingDataParams.model_validate({**FULL, **overrides})

    with pytest.raises(ValueError, match=f"the {empty} split would be empty"):
        check_preparation(full_state, params)
    with pytest.raises(ValueError, match=f"the {empty} split would be empty"):
        _ = _prepare_full(full_state, **overrides)


def test_a_trainer_that_splits_for_itself_takes_one_recording(
    full_state: Dataset,
) -> None:
    """SLEAP and Lightning Pose draw their own validation split from what they are given."""
    params = {**FULL, "split_by": "sequence", "target": "sleap"}

    check_preparation(full_state, PrepareTrainingDataParams.model_validate(params))


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


def test_an_index_older_than_the_pose_reads_blank_and_is_adopted_on_the_next_write(
    world: tuple[Dataset, Dataset, Dataset],
) -> None:
    """A row from before the pose was recorded names none; blank means unknown.

    The older file holds its columns out of schema order and one the schema has
    retired, so an appended row alone does not bring it to the schema: only the
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
