"""Annotation revisions from several projects become one training dataset.

The workflow this pins end to end: two projects each save a keypoint annotation
set, a library claims one exact revision of each, prepares a union, and a model
trained from it can say which annotation states it saw -- from the library and
from any project that links it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mosaic.core.annotations.model import (
    AnnotationFrame,
    AnnotationObject,
    AnnotationSet,
    Keypoint,
    KeypointSchema,
)
from mosaic.core.annotations.projection import write_keypoint_set_revision
from mosaic.core.dataset import Dataset
from mosaic.core.manifest import LabelsScanSource, LibraryLink
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.inventory import inventory
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.core.pipeline.ops import OPS, IdentityDeferred, run_op
from mosaic.tracking import register_ops
from mosaic.tracking.ops._common import (
    fingerprint_yolo_dataset,
    resolve_training_data,
)
from mosaic.tracking.ops.prepare import (
    PrepareTrainingDataParams,
    prepared_dataset_index,
)
from mosaic.tracking.ops.train import (
    PoseTrainParams,
    finalize_training,
    trained_model_index,
)
from mosaic.tracking.training_provenance import training_provenance
from tests.helpers import make_dataset

register_ops()

SCHEMA = KeypointSchema(names=("nose", "tail"), skeleton=((0, 1),))
KIND = "prepare-training-data"


def _save_set(
    ds: Dataset,
    key: str,
    sequences: dict[str, int],
    *,
    shift: float = 0.0,
    commit: str = "c1",
) -> int:
    """Extract-shaped frames for *sequences*, annotated, saved as one revision.

    Every sequence numbers its frames from zero, the way mosaic's frame
    extraction does, so two sequences share every basename.
    """
    frames: list[AnnotationFrame] = []
    for sequence, count in sequences.items():
        for index in range(count):
            relative = Path(
                f"media/frames/kmeans/kmeans-d7968c97b0/{sequence}/frame_{index:06d}.png"
            )
            image = ds.base_dir / relative
            image.parent.mkdir(parents=True, exist_ok=True)
            _ = image.write_bytes(f"{ds.name}/{sequence}/{index}".encode())
            obj = AnnotationObject(
                keypoints=(
                    Keypoint(x=10.0 + index + shift, y=12.0, visibility=2),
                    Keypoint(x=30.0 + index, y=20.0, visibility=2),
                )
            )
            frames.append(
                AnnotationFrame(
                    image_path=relative, width=64, height=48, objects=(obj,),
                    video=sequence, frame_index=index,
                )
            )  # fmt: skip
    annotations = AnnotationSet(
        schema=SCHEMA, frames=tuple(frames), image_root=ds.base_dir
    )
    return write_keypoint_set_revision(
        ds, set_key=key, annotations=annotations, origin={"dolt_commit": commit}
    ).revision


def _claim(library: Dataset, project: Dataset, key: str, revision: int) -> None:
    library.add_scan_source(
        LabelsScanSource(
            id=f"{project.name}-{key}",
            path=str(project.get_root("labels_raw") / "keypoints" / key),
            files=(f"rev{revision}/annotations.coco.json",),
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
        "labels", "mice-17-openfield", [f"rev{revision}/annotations.coco.json"]
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


def test_two_skeletons_cannot_be_one_model(tmp_path: Path) -> None:
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    _ = _save_set(library, "two-points", {"m01": 2})
    image = (
        library.base_dir / "media/frames/kmeans/kmeans-d7968c97b0/m07/frame_000000.png"
    )
    image.parent.mkdir(parents=True)
    _ = image.write_bytes(b"png")
    three = AnnotationObject(
        keypoints=tuple(Keypoint(x=float(i), y=1.0, visibility=2) for i in range(3))
    )
    other = AnnotationSet(
        schema=KeypointSchema(names=("nose", "ear", "tail")),
        frames=(
            AnnotationFrame(
                image_path=image.relative_to(library.base_dir),
                width=64,
                height=48,
                objects=(three,),
                video="m07",
                frame_index=0,
            ),
        ),
        image_root=library.base_dir,
    )
    _ = write_keypoint_set_revision(
        library, set_key="three-points", annotations=other, origin={}
    )

    with pytest.raises(ValueError, match="One model has one skeleton"):
        _ = run_op(
            library,
            KIND,
            {"sets": [{"set_key": "two-points"}, {"set_key": "three-points"}]},
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

    (
        rats.get_root("labels_raw") / "keypoints/21-arena/rev1/annotations.coco.json"
    ).unlink()

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
