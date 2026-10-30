"""A preparation step is planned against the prepared dataset that it wrote.

A preparation writes one directory under ``models/<kind>/<run_id>/`` and a row in
that kind's index. The plan looks the step up by that artifact, so a finished
preparation reads as complete and one that never ran reads as absent. The cases
are every op that writes a prepared dataset, ``convert-points`` among them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest

from mosaic.core.annotations.projection import write_keypoint_set_revision
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import PlannedStep, Recipe, plan_pipeline
from mosaic.core.pipeline.inventory.model import PreparedDatasetRef
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.tracking import register_ops
from mosaic.tracking.ops.convert import (
    ConvertedDatasetIndexRow,
    converted_dataset_index,
)
from mosaic.tracking.ops.prepare import PreparedDatasetIndexRow, prepared_dataset_index
from mosaic.tracking.ops.prepare_embedding import (
    EmbeddingDataIndexRow,
    embedding_data_index,
)
from tests.helpers import (
    make_dataset,
    minimal_op_params,
    pose_set,
    write_media_index,
)

register_ops()

STEP: Final = "prepare"

CONVERT: Final = "convert-points"
EMBEDDING: Final = "prepare-embedding-data"
TRAINING: Final = "prepare-training-data"

ARTIFACT_NAMES: Final = {
    CONVERT: "data.yaml",
    EMBEDDING: "labels.slp",
    TRAINING: "labels.slp",
}
"""The file each kind records as what a trainer is handed."""

type Finish = Callable[[Dataset, str, Path], None]
"""Record a finished preparation of the given run whose artifact is the given file."""


def _finish_convert(ds: Dataset, run_id: str, artifact: Path) -> None:
    index = converted_dataset_index(model_index_path(ds, CONVERT))
    index.append(
        [
            ConvertedDatasetIndexRow(
                run_id=run_id,
                kind=CONVERT,
                source_format="cvat_points_polo",
                data_yaml=ds.relative_to_root(artifact),
                class_names="individual",
                n_train=1,
                n_valid=0,
                n_test=0,
                status="finished",
                abs_path=Path(ds.relative_to_root(artifact.parent)),
            )
        ]
    )
    index.mark_finished(run_id)


def _finish_embedding(ds: Dataset, run_id: str, artifact: Path) -> None:
    index = embedding_data_index(model_index_path(ds, EMBEDDING))
    index.append(
        [
            EmbeddingDataIndexRow(
                run_id=run_id,
                kind=EMBEDDING,
                artifact_path=ds.relative_to_root(artifact),
                sleap_run_id="sleap.1.6-abcdef0123",
                consumed_predictions="[]",
                n_windows=1,
                n_frames=2,
                n_train=1,
                n_valid=1,
                n_tracks=2,
                status="finished",
                abs_path=Path(ds.relative_to_root(artifact.parent)),
            )
        ]
    )
    index.mark_finished(run_id)


def _finish_training(ds: Dataset, run_id: str, artifact: Path) -> None:
    index = prepared_dataset_index(model_index_path(ds, TRAINING))
    index.append(
        [
            PreparedDatasetIndexRow(
                run_id=run_id,
                kind=TRAINING,
                target="sleap",
                artifact_path=ds.relative_to_root(artifact),
                consumed_sets="[]",
                n_frames=2,
                n_train=1,
                n_valid=1,
                n_test=0,
                status="finished",
                abs_path=Path(ds.relative_to_root(artifact.parent)),
            )
        ]
    )
    index.mark_finished(run_id)


def _convert_dataset(tmp_path: Path) -> Dataset:
    """A dataset whose annotations and images are on disk, as the identity reads them."""
    ds = make_dataset(tmp_path / "ds")
    params = minimal_op_params(CONVERT)
    xml = Path(ds.resolve_path(str(params["cvat_xml"])))
    images = Path(ds.resolve_path(str(params["images_dir"])))
    images.mkdir(parents=True)
    _ = xml.write_text("<annotations/>")
    _ = (images / "frame.png").write_bytes(b"")
    return ds


def _embedding_dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, ["s1", "s2"], uids={"s1": "u1", "s2": "u2"})
    return ds


def _training_dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    _ = write_keypoint_set_revision(
        ds,
        set_key="set-a",
        annotations=pose_set((), image_root=ds.base_dir),
        origin={},
    )
    return ds


def _params(kind: str) -> dict[str, object]:
    if kind == EMBEDDING:
        return {
            "sleap_run_id": "sleap.1.6-abcdef0123",
            "train_entries": [{"sequence": "s1"}],
            "val_entries": [{"sequence": "s2"}],
        }
    return minimal_op_params(kind)


def _planned(ds: Dataset, kind: str) -> PlannedStep:
    recipe = Recipe.model_validate(
        {"steps": [{"id": STEP, "type": "op", "kind": kind, "params": _params(kind)}]}
    )
    return plan_pipeline(ds, recipe).step(STEP)


type Case = tuple[str, Callable[[Path], Dataset], Finish]

CASES: Final[dict[str, Case]] = {
    CONVERT: (CONVERT, _convert_dataset, _finish_convert),
    EMBEDDING: (EMBEDDING, _embedding_dataset, _finish_embedding),
    TRAINING: (TRAINING, _training_dataset, _finish_training),
}

KINDS = pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())


def _prepared(ds: Dataset, kind: str, finish: Finish, *, write_file: bool) -> str:
    """Finish the planned preparation of *kind*, and return its run id."""
    run_id = _planned(ds, kind).run_id
    assert run_id is not None
    artifact = model_run_root(ds, kind, run_id) / ARTIFACT_NAMES[kind]
    if write_file:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        _ = artifact.write_bytes(b"labels")
    finish(ds, run_id, artifact)
    return run_id


@KINDS
def test_a_planned_preparation_names_its_prepared_dataset(
    tmp_path: Path, case: Case
) -> None:
    kind, build, _finish = case
    planned = _planned(build(tmp_path), kind)

    assert planned.run_id is not None
    assert planned.artifact == PreparedDatasetRef(op_kind=kind, run_id=planned.run_id)


@KINDS
def test_a_preparation_that_never_ran_reads_absent(tmp_path: Path, case: Case) -> None:
    kind, build, _finish = case

    planned = _planned(build(tmp_path), kind)

    assert planned.status == "absent"


@KINDS
def test_a_finished_preparation_reads_complete(tmp_path: Path, case: Case) -> None:
    kind, build, finish = case
    ds = build(tmp_path)
    _ = _prepared(ds, kind, finish, write_file=True)

    planned = _planned(ds, kind)

    assert planned.status == "complete"
    assert planned.coverage.missing == frozenset()


@KINDS
def test_a_preparation_whose_artifact_is_gone_does_not_read_complete(
    tmp_path: Path, case: Case
) -> None:
    kind, build, finish = case
    ds = build(tmp_path)
    _ = _prepared(ds, kind, finish, write_file=False)

    planned = _planned(ds, kind)

    assert planned.status == "absent"
