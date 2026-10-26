"""A trained model says what it was trained on, and prepared data is not a model.

Two gaps, one story. A training run root held weights and nothing that said how
they came to be: the data reference lived only in whatever submitted the run, so
nothing reading the dataset alone could say what a model had seen. And the
directory a model's training data is prepared into sits beside the models under
``models/``, where the inventory judged it by the trained-model rule and reported
every one as a finished row whose weights were missing.

A model also names the keypoints that it predicts. Its preparation row records
them, and for an older row they are read from the pose that the revision
declares.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from mosaic.core.annotations.model import KeypointSchema
from mosaic.core.annotations.pose_annotations import PoseDefinition, PoseFrame
from mosaic.core.annotations.projection import write_keypoint_set_revision
from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.dataset_indexes import iter_dataset_indexes
from mosaic.core.pipeline.inventory import inventory
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.core.pipeline.ops import run_op
from mosaic.tracking import register_ops
from mosaic.tracking.keypoint_sets import keypoint_schema_cell
from mosaic.tracking.ops.convert import (
    ConvertedDatasetIndexRow,
    converted_dataset_index,
)
from mosaic.tracking.ops.train import (
    TRAINING_PROVENANCE_FILENAME,
    LocalizerTrainParams,
    finalize_training,
)
from mosaic.tracking.ops.trained_models import trained_model_index
from mosaic.tracking.training_provenance import training_provenance
from tests.helpers import (
    MOUSE,
    make_dataset,
    pose_frame,
    pose_object,
    pose_set,
    register_trained_model,
    revision_file,
    rewrite_as_older_prepared_index,
)

register_ops()

KIND = "train-localizer"
RUN_ID = "train-localizer.0.1-abcdef0123"


def _finalize(ds: Dataset, data_dir: Path) -> Path:
    run_root = model_run_root(ds, KIND, RUN_ID)
    (run_root / "train").mkdir(parents=True)
    weights = run_root / "train" / "best.pt"
    _ = weights.write_bytes(b"weights")
    finalize_training(
        ds,
        KIND,
        RUN_ID,
        run_root,
        LocalizerTrainParams(dataset_dir=str(data_dir)),
        "",
        "",
        "",
        weights,
        run_root / "train" / "results.csv",
        1,
        data_path=data_dir,
        data_fingerprint="f1ngerpr1nt",
    )
    return run_root


def test_a_finished_run_records_what_it_was_trained_on(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds", name="prov")
    data_dir = ds.get_root("models") / "prepare-training-data" / "p-1"
    data_dir.mkdir(parents=True)

    run_root = _finalize(ds, data_dir)

    row = trained_model_index(model_index_path(ds, KIND)).read(run_id=RUN_ID).iloc[0]
    assert row["data_path"] == "models/prepare-training-data/p-1", "root-relative"
    assert row["data_fingerprint"] == "f1ngerpr1nt"

    document = json.loads((run_root / TRAINING_PROVENANCE_FILENAME).read_text())
    assert document["run_id"] == RUN_ID
    assert document["data"] == {
        "path": "models/prepare-training-data/p-1",
        "fingerprint": "f1ngerpr1nt",
    }
    assert document["params"]["dataset_dir"] == str(data_dir)


def test_data_outside_the_dataset_stays_absolute(tmp_path: Path) -> None:
    """External data is provenance of somewhere else, recorded as it was named."""
    ds = make_dataset(tmp_path / "ds", name="prov")
    outside = tmp_path / "elsewhere" / "yolo"
    outside.mkdir(parents=True)

    _ = _finalize(ds, outside)

    row = trained_model_index(model_index_path(ds, KIND)).read(run_id=RUN_ID).iloc[0]
    assert Path(str(row["data_path"])).is_absolute()


def test_the_data_path_is_visible_to_the_portability_passes(tmp_path: Path) -> None:
    """A path column the repair passes cannot see silently stops being portable."""
    ds = make_dataset(tmp_path / "ds", name="prov")
    _ = _finalize(ds, tmp_path / "elsewhere")

    models = [i for i in iter_dataset_indexes(ds) if i.root_key == "models"]
    from mosaic.core.dataset import _INDEX_PATH_COLUMNS  # pyright: ignore[reportPrivateUsage]

    assert models, "the models index is enumerated"
    assert "data_path" in _INDEX_PATH_COLUMNS["models"]


def test_an_index_written_before_the_columns_existed_is_adopted_on_write(
    tmp_path: Path,
) -> None:
    """Adopt on write, tolerate on read: a legacy index widens when it is next written.

    Reading leaves it as it is, so a read-only mount works. The next finished run
    publishes the wider schema and its own row in one write, and the row that was
    already there keeps its integer ``n_epochs`` rather than becoming ``3.0``.
    """
    ds = make_dataset(tmp_path / "ds", name="prov")
    index_path = model_index_path(ds, KIND)
    index_path.parent.mkdir(parents=True)
    legacy_run = "train-localizer.0.1-0000000000"
    legacy = pd.DataFrame(
        [
            {
                "run_id": legacy_run,
                "abs_path": f"models/{KIND}/{legacy_run}",
                "started_at": "",
                "finished_at": "",
                "kind": KIND,
                "base_model": "",
                "base_run_id": "",
                "best_model_path": f"models/{KIND}/{legacy_run}/best.pt",
                "metrics_path": "",
                "n_epochs": 3,
                "status": "finished",
            }
        ]
    )
    legacy.to_csv(index_path, index=False)
    assert "data_path" not in pd.read_csv(index_path).columns, (
        "reading rewrites nothing"
    )

    _ = _finalize(ds, tmp_path / "elsewhere")

    widened = pd.read_csv(index_path, dtype="string", keep_default_na=False)
    assert {"data_path", "data_fingerprint", "base_origin"} <= set(widened.columns)
    old_row = widened[widened["run_id"] == legacy_run].iloc[0]
    assert old_row["data_path"] == "", "unknown, not invented"
    assert old_row["n_epochs"] == "3", "an absent column must not widen the integers"


def _prepared(ds: Dataset, *, keep_artifact: bool) -> str:
    kind, run_id = "convert-points", "convert-points.0.2-aaaaaaaaaa"
    out = model_run_root(ds, kind, run_id)
    out.mkdir(parents=True)
    data_yaml = out / "data.yaml"
    if keep_artifact:
        _ = data_yaml.write_text("names: [bee]\n")
    index = converted_dataset_index(model_index_path(ds, kind))
    index.ensure()
    index.append(
        [
            ConvertedDatasetIndexRow(
                run_id=run_id,
                kind=kind,
                source_format="cvat_points_polo",
                data_yaml=ds.relative_to_root(data_yaml),
                class_names="bee",
                n_train=3,
                n_valid=1,
                n_test=0,
                status="finished",
                abs_path=Path(ds.relative_to_root(out)),
            )
        ]
    )
    index.mark_finished(run_id)
    return run_id


def test_prepared_data_is_not_reported_as_a_damaged_model(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds", name="prov")
    run_id = _prepared(ds, keep_artifact=True)

    assert inventory(ds, kinds=["trained-model"]).records == ()

    found = inventory(ds, kinds=["prepared-dataset"])
    assert [record.run_id for record in found.records] == [run_id]
    assert found.records[0].status == "complete"
    assert found.records[0].ref.kind == "prepared-dataset"


def test_prepared_data_whose_tree_is_gone_is_not_complete(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds", name="prov")
    _ = _prepared(ds, keep_artifact=False)

    found = inventory(ds, kinds=["prepared-dataset"])

    assert found.records[0].status != "complete"


# ------------------------------------------------- the keypoints that a model predicts

PREPARE: Final = "prepare-training-data"
SET_KEY: Final = "two-poses"
EARS: Final = PoseDefinition(
    id=5,
    name="eared",
    schema=KeypointSchema(
        names=("nose", "left_ear", "right_ear"),
        skeleton=((0, 1), (0, 2)),
        symmetries=((1, 2),),
    ),
)


@pytest.fixture
def two_poses(tmp_path: Path) -> Dataset:
    """A dataset that saved one set of two poses, ``mouse`` and ``eared``.

    Two recordings of three finished frames, each frame holding one object of
    each pose.
    """
    ds = make_dataset(tmp_path / "ds", name="prov")
    frames: list[PoseFrame] = []
    for sequence in ("m01", "m02"):
        for index in range(3):
            image = Path(f"media/frames/{sequence}/frame_{index:06d}.png")
            (ds.base_dir / image).parent.mkdir(parents=True, exist_ok=True)
            _ = (ds.base_dir / image).write_bytes(f"{sequence}/{index}".encode())
            frames.append(
                pose_frame(
                    image,
                    pose_object((10.0 + index, 12.0), (30.0, 20.0)),
                    pose_object(
                        (40.0, 20.0), (36.0, 16.0), (44.0, 16.0), pose_id=EARS.id
                    ),
                    sequence=sequence,
                    frame_index=index,
                )
            )
    state = pose_set(frames, poses=(MOUSE, EARS), image_root=ds.base_dir)
    _ = write_keypoint_set_revision(ds, set_key=SET_KEY, annotations=state, origin={})
    return ds


def _model(ds: Dataset, pose: PoseDefinition, **overrides: object) -> str:
    """A model registered over a preparation of the set narrowed to *pose*."""
    params: dict[str, object] = {
        "sets": [{"set_key": SET_KEY}],
        "pose": pose.name,
        # Two recordings are too few to draw whole ones into both splits.
        "split_by": "frame",
        **overrides,
    }
    prepared = run_op(ds, PREPARE, params)
    run_id = "train-pose.0.2-feedfacade"
    weights = model_run_root(ds, "train-pose", run_id) / "best.pt"
    weights.parent.mkdir(parents=True)
    _ = weights.write_bytes(b"weights")
    data_yaml = model_run_root(ds, PREPARE, prepared) / "data.yaml"
    register_trained_model(ds, "train-pose", run_id, weights, data_path=data_yaml)
    return run_id


def _lose_the_revision(ds: Dataset) -> None:
    (ds.get_root("labels_raw") / "keypoints" / SET_KEY / revision_file(1)).unlink()


def _reported(document: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
    """The keypoint names, skeleton and mirror pairs that ``as_json`` reports."""
    return document["keypoint_names"], document["skeleton"], document["symmetries"]


def test_a_model_names_the_keypoints_its_preparation_recorded(
    two_poses: Dataset,
) -> None:
    """The preparation row records the keypoints, and the revision is not read."""
    model = _model(two_poses, EARS)
    _lose_the_revision(two_poses)

    found = training_provenance(two_poses, "train-pose", model)

    assert found.keypoint_schema == EARS.schema
    assert found.keypoint_names == ("nose", "left_ear", "right_ear")
    assert _reported(found.as_json()) == (
        ["nose", "left_ear", "right_ear"],
        [[0, 1], [0, 2]],
        [[1, 2]],
    )


def test_a_model_reports_its_pairs_in_ascending_order(two_poses: Dataset) -> None:
    """The report lists pairs ascending, in the order of a revision file."""
    model = _model(two_poses, EARS)
    path = model_index_path(two_poses, PREPARE)
    written = pd.read_csv(path, dtype=str, keep_default_na=False)
    written["keypoint_schema"] = keypoint_schema_cell(
        replace(EARS.schema, skeleton=((0, 2), (0, 1)))
    )
    written.to_csv(path, index=False)

    found = training_provenance(two_poses, "train-pose", model)

    assert found.as_json()["skeleton"] == [[0, 1], [0, 2]]


def test_a_points_model_names_the_one_keypoint_it_was_trained_on(
    two_poses: Dataset,
) -> None:
    model = _model(two_poses, EARS, target="polo", point_index=2)

    found = training_provenance(two_poses, "train-pose", model)

    assert found.keypoint_schema == KeypointSchema(names=("right_ear",))
    assert _reported(found.as_json()) == (["right_ear"], [], [])


@pytest.mark.parametrize("pose", [MOUSE, EARS], ids=["mouse", "eared"])
def test_an_older_row_names_the_keypoints_of_the_pose_its_revision_declares(
    two_poses: Dataset, pose: PoseDefinition
) -> None:
    """The revision declares both poses, and the recorded pose id picks one."""
    model = _model(two_poses, pose)
    rewrite_as_older_prepared_index(two_poses)

    found = training_provenance(two_poses, "train-pose", model)

    assert found.pose_id == pose.id
    assert found.keypoint_schema == pose.schema


def test_an_older_points_row_names_no_keypoints(two_poses: Dataset) -> None:
    """The row does not record which keypoint of the pose the model learned."""
    model = _model(two_poses, EARS, target="polo", point_index=2)
    rewrite_as_older_prepared_index(two_poses)

    found = training_provenance(two_poses, "train-pose", model)

    assert found.pose_id == EARS.id, "the pose and its revision are still known"
    assert found.keypoint_schema is None
    assert found.keypoint_names == ()
    assert _reported(found.as_json()) == ([], [], [])


def test_an_older_row_whose_revision_is_gone_names_no_keypoints(
    two_poses: Dataset,
) -> None:
    model = _model(two_poses, EARS)
    rewrite_as_older_prepared_index(two_poses)
    _lose_the_revision(two_poses)

    found = training_provenance(two_poses, "train-pose", model)

    assert found.keypoint_schema is None
    assert found.keypoint_names == ()


def test_a_sleap_model_trained_on_an_external_slp_names_no_keypoints(
    tmp_path: Path,
) -> None:
    """The labels were not written by a preparation, which records the keypoints."""
    ds = make_dataset(tmp_path / "ds", name="prov")
    labels = tmp_path / "elsewhere" / "session.slp"
    labels.parent.mkdir(parents=True)
    _ = labels.write_bytes(b"slp")
    run_id = "train-sleap.0.3-0ddba11000"
    directory = model_run_root(ds, "train-sleap", run_id) / "model"
    directory.mkdir(parents=True)
    weights = directory / "best.ckpt"
    _ = weights.write_bytes(b"weights")
    register_trained_model(
        ds, "train-sleap", run_id, weights, directory=directory, data_path=labels
    )

    found = training_provenance(ds, "train-sleap", run_id)

    assert "not written by a preparation run" in found.stopped_at
    assert found.keypoint_schema is None
    assert _reported(found.as_json()) == ([], [], [])
