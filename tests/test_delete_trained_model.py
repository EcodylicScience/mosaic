"""Deleting a trained model: its index row first, then its run root.

A model resolves through its index row, so the row goes first and the model stops
resolving before any file is touched. Everything a request can get wrong is
refused before either happens, and so is a model whose run root another execution
is still writing.
"""

from __future__ import annotations

import datetime
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.manifest import LibraryLink
from mosaic.core.pipeline.index_csv import IndexCSV, index_records
from mosaic.core.pipeline.inventory import inventory
from mosaic.core.pipeline.markers import InflightMarker, read_inflight, write_inflight
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.runlog import JsonlRunLog, run_log_path
from mosaic.tracking import delete_trained_model
from mosaic.tracking.model_refs import ModelNotFoundError, resolve_op_model
from mosaic.tracking.ops.delete_model import (
    DeletionRefusalReason,
    ModelDeletionRefusedError,
    TrainedModelInUseError,
)
from mosaic.tracking.ops._common import RunRootHeld, claim_run_root
from mosaic.tracking.ops.prepare import PreparedDatasetIndexRow, prepared_dataset_index
from mosaic.tracking.ops.train import TrainedModelIndexRow, trained_model_index
from tests.helpers import make_dataset, register_trained_model

KIND = "train-pose"
RUN_ID = "train-pose.0.2-abcdef0123"


def _register(
    ds: Dataset, run_id: str = RUN_ID, *, data_path: Path | None = None
) -> None:
    """One finished ``train-pose`` run, registered the way the op registers it."""
    weights = model_run_root(ds, KIND, run_id) / "train" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    _ = weights.write_bytes(b"weights")
    register_trained_model(ds, KIND, run_id, weights, data_path=data_path)


def _registered_runs(ds: Dataset) -> list[str]:
    frame = trained_model_index(model_index_path(ds, KIND)).read()
    return [str(run_id) for run_id in frame["run_id"]]


def _assert_untouched(ds: Dataset) -> None:
    assert _registered_runs(ds) == [RUN_ID]
    assert model_run_root(ds, KIND, RUN_ID).is_dir()
    assert resolve_op_model(ds, "infer-pose", RUN_ID).run_id == RUN_ID


def _claim(ds: Dataset, execution_id: str, *, expires_in: float) -> None:
    """Claim the run root for *execution_id*, expiring *expires_in* seconds from now."""
    expires_at = datetime.datetime.now(datetime.UTC) + datetime.timedelta(
        seconds=expires_in
    )
    write_inflight(
        model_run_root(ds, KIND, RUN_ID),
        InflightMarker(
            execution_id=execution_id,
            host="trainer",
            pid=4242,
            expires_at=expires_at.isoformat(),
        ),
    )


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    """A dataset holding one registered, finished ``train-pose`` model."""
    dataset = make_dataset(tmp_path / "dataset")
    _register(dataset)
    return dataset


def test_a_finished_model_is_deleted_and_no_longer_resolves(ds: Dataset) -> None:
    deleted = delete_trained_model(ds, KIND, RUN_ID)

    assert deleted.row["run_id"] == RUN_ID
    assert deleted.directory_removed
    assert deleted.removal_error == ""
    assert _registered_runs(ds) == []
    assert not model_run_root(ds, KIND, RUN_ID).exists()
    with pytest.raises(ModelNotFoundError):
        _ = resolve_op_model(ds, "infer-pose", RUN_ID)


def test_deleting_one_model_leaves_another_of_its_kind_alone(ds: Dataset) -> None:
    other = "train-pose.0.2-1111111111"
    _register(ds, other)

    _ = delete_trained_model(ds, KIND, RUN_ID)

    assert _registered_runs(ds) == [other]
    assert (model_run_root(ds, KIND, other) / "train" / "weights" / "best.pt").is_file()
    assert resolve_op_model(ds, "infer-pose", other).run_id == other


def test_the_prepared_data_a_model_trained_on_is_left_alone(tmp_path: Path) -> None:
    """Other trainings may read it, and it is a run of its own."""
    ds = make_dataset(tmp_path / "dataset")
    prepared_kind, prepared_id = (
        "prepare-training-data",
        "prepare-training-data.0.2-2222222222",
    )
    prepared_root = model_run_root(ds, prepared_kind, prepared_id)
    prepared_root.mkdir(parents=True)
    data_yaml = prepared_root / "data.yaml"
    _ = data_yaml.write_text("train: train/images\nval: valid/images\n")
    prepared_index = model_index_path(ds, prepared_kind)
    prepared_dataset_index(prepared_index).append(
        [
            PreparedDatasetIndexRow(
                run_id=prepared_id,
                kind=prepared_kind,
                target="yolo-pose",
                artifact_path=ds.relative_to_root(data_yaml),
                consumed_sets="[]",
                n_frames=0,
                n_train=0,
                n_valid=0,
                n_test=0,
                status="finished",
                abs_path=Path(ds.relative_to_root(prepared_root)),
            )
        ]
    )
    _register(ds, data_path=data_yaml)
    (model_row,) = index_records(trained_model_index(model_index_path(ds, KIND)).read())
    assert prepared_id in model_row["data_path"], "the model names what it trained on"
    index_before = prepared_index.read_bytes()

    _ = delete_trained_model(ds, KIND, RUN_ID)

    assert prepared_index.read_bytes() == index_before
    assert [p.name for p in prepared_root.iterdir()] == ["data.yaml"]


def test_a_model_whose_run_root_a_live_execution_holds_is_refused(
    ds: Dataset,
) -> None:
    """A retraining holds the root, and removing it would pull files from under it."""
    _claim(ds, "TRAINING", expires_in=3600)

    with pytest.raises(TrainedModelInUseError, match="TRAINING") as refused:
        _ = delete_trained_model(ds, KIND, RUN_ID)

    assert refused.value.execution_id == "TRAINING"
    assert refused.value.expires_at, "says when the claim lapses"
    _assert_untouched(ds)


def _during_the_drop(
    monkeypatch: pytest.MonkeyPatch, arrive: Callable[[], None]
) -> None:
    """Run *arrive* once, as the delete drops the row after its checks have passed."""
    original = IndexCSV[TrainedModelIndexRow].drop_runs
    arrived: list[bool] = []

    def drop(
        self: IndexCSV[TrainedModelIndexRow],
        run_ids: Iterable[str],
        *,
        dry_run: bool = False,
    ) -> pd.DataFrame:
        if not arrived:
            arrived.append(True)
            arrive()
        return original(self, run_ids, dry_run=dry_run)

    monkeypatch.setattr(IndexCSV, "drop_runs", drop)


def test_a_training_arriving_mid_delete_cannot_claim_the_run_root(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the row is gone, a resubmitted training finds no model and claims the root.

    The delete holds the claim, so the training is refused and the root removed.
    """
    refusals: list[RunRootHeld] = []

    def retrain() -> None:
        with pytest.raises(RunRootHeld) as refused:
            _ = claim_run_root(
                ds, "RETRAIN", model_run_root(ds, KIND, RUN_ID), KIND, 60
            )
        refusals.append(refused.value)

    _during_the_drop(monkeypatch, retrain)

    deleted = delete_trained_model(ds, KIND, RUN_ID)

    assert len(refusals) == 1
    assert deleted.directory_removed
    assert not model_run_root(ds, KIND, RUN_ID).exists()


def test_a_second_delete_arriving_mid_delete_is_told_the_model_is_in_use(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    refusals: list[TrainedModelInUseError] = []

    def delete_again() -> None:
        with pytest.raises(TrainedModelInUseError) as refused:
            _ = delete_trained_model(ds, KIND, RUN_ID)
        refusals.append(refused.value)

    _during_the_drop(monkeypatch, delete_again)

    deleted = delete_trained_model(ds, KIND, RUN_ID)

    assert len(refusals) == 1
    assert deleted.directory_removed


@pytest.mark.parametrize("claim", ["expired", "orphaned"])
def test_a_claim_nobody_holds_any_more_does_not_block(ds: Dataset, claim: str) -> None:
    if claim == "expired":
        _claim(ds, "GONE", expires_in=-60)
    else:
        _claim(ds, "FINISHED", expires_in=3600)
        log = JsonlRunLog(run_log_path(ds.base_dir, "FINISHED"), "FINISHED")
        log.started(kind=KIND, target=RUN_ID)
        log.finished()
        log.close()

    deleted = delete_trained_model(ds, KIND, RUN_ID)

    assert deleted.directory_removed
    assert _registered_runs(ds) == []


@pytest.mark.parametrize(
    ("kind", "run_id", "reason"),
    [
        pytest.param(
            "prepare-training-data",
            "prepare-training-data.0.1-abcdef0123",
            "prepared_data",
            id="prepared-data",
        ),
        pytest.param("..", RUN_ID, "not_a_run_of_kind", id="a-kind-leaving-the-root"),
        pytest.param(
            "train-pose/..", RUN_ID, "not_a_run_of_kind", id="a-kind-of-two-parts"
        ),
        pytest.param(KIND, "..", "not_a_run_of_kind", id="a-run-id-leaving-the-root"),
        pytest.param(KIND, "best.pt", "not_a_run_of_kind", id="not-a-run-id"),
        pytest.param(
            KIND,
            "train-sleap.0.1-abcdef0123",
            "not_a_run_of_kind",
            id="a-run-of-another-kind",
        ),
    ],
)
def test_a_request_naming_no_trained_model_is_refused_before_anything_changes(
    ds: Dataset, kind: str, run_id: str, reason: DeletionRefusalReason
) -> None:
    with pytest.raises(ModelDeletionRefusedError) as refused:
        _ = delete_trained_model(ds, kind, run_id)

    assert refused.value.reason == reason
    _assert_untouched(ds)


def test_a_run_root_reached_through_a_link_out_of_the_models_root_is_refused(
    tmp_path: Path,
) -> None:
    """Removal follows the resolved path, so it must stay under ``models``."""
    ds = make_dataset(tmp_path / "dataset")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (ds.get_root("models") / KIND).symlink_to(elsewhere, target_is_directory=True)
    _register(ds)

    with pytest.raises(ModelDeletionRefusedError) as refused:
        _ = delete_trained_model(ds, KIND, RUN_ID)

    assert refused.value.reason == "outside_models_root"
    assert (elsewhere / RUN_ID).is_dir()
    assert _registered_runs(ds) == [RUN_ID]


@pytest.mark.parametrize("target", ["another-run", "outside"])
def test_a_run_root_that_is_itself_a_link_is_refused(
    tmp_path: Path, target: str
) -> None:
    """Emptying the root through the link would empty whatever it points at."""
    ds = make_dataset(tmp_path / "dataset")
    other = (
        model_run_root(ds, KIND, "train-pose.0.2-1111111111")
        if target == "another-run"
        else tmp_path / "elsewhere"
    )
    (other / "train" / "weights").mkdir(parents=True)
    _ = (other / "train" / "weights" / "best.pt").write_bytes(b"other weights")
    link = model_run_root(ds, KIND, RUN_ID)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(other, target_is_directory=True)
    register_trained_model(ds, KIND, RUN_ID, link / "train" / "weights" / "best.pt")

    with pytest.raises(ModelDeletionRefusedError) as refused:
        _ = delete_trained_model(ds, KIND, RUN_ID)

    assert refused.value.reason == "run_root_is_a_link"
    assert sorted(p.name for p in other.rglob("*")) == ["best.pt", "train", "weights"]
    assert _registered_runs(ds) == [RUN_ID]


@pytest.mark.parametrize(
    ("kind", "run_id"),
    [
        pytest.param(KIND, "train-pose.0.2-0123456789", id="unregistered-run"),
        pytest.param(
            "train-sleap", "train-sleap.0.1-abcdef0123", id="a-kind-never-trained-here"
        ),
        pytest.param(
            "infer-pose", "infer-pose.0.1-abcdef0123", id="an-op-that-trains-nothing"
        ),
        pytest.param("no-such-op", "no-such-op.0.1-abcdef0123", id="unknown"),
    ],
)
def test_a_run_no_index_registers_is_not_found(
    ds: Dataset, kind: str, run_id: str
) -> None:
    with pytest.raises(ModelNotFoundError) as missing:
        _ = delete_trained_model(ds, kind, run_id)

    assert missing.value.reference == run_id
    _assert_untouched(ds)


def test_a_model_the_inventory_reports_is_deletable_though_no_op_trains_it(
    tmp_path: Path,
) -> None:
    """TREx identity weights are registered under ``train-identity``.

    No op writes that kind.
    """
    ds = make_dataset(tmp_path / "dataset")
    kind, run_id = "train-identity", "train-identity.0.1-abcdef0123"
    weights = model_run_root(ds, kind, run_id) / "identity_model.pth"
    weights.parent.mkdir(parents=True)
    _ = weights.write_bytes(b"identity weights")
    register_trained_model(ds, kind, run_id, weights)
    reported = inventory(ds, kinds=["trained-model"]).records
    assert [(r.name, r.run_id) for r in reported] == [(kind, run_id)]

    deleted = delete_trained_model(ds, kind, run_id)

    assert deleted.directory_removed
    assert inventory(ds, kinds=["trained-model"]).records == ()


def test_a_row_whose_run_root_is_already_gone_is_still_dropped(ds: Dataset) -> None:
    shutil.rmtree(model_run_root(ds, KIND, RUN_ID))

    deleted = delete_trained_model(ds, KIND, RUN_ID)

    assert not deleted.directory_removed
    assert deleted.removal_error == ""
    assert _registered_runs(ds) == []


def test_a_run_root_that_cannot_be_removed_is_reported_once_the_row_is_gone(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model is already unresolvable, so the caller hears about the files."""

    def refuse(path: str | Path, *_args: object, **_kwargs: object) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(shutil, "rmtree", refuse)

    deleted = delete_trained_model(ds, KIND, RUN_ID)

    assert not deleted.directory_removed
    assert "Permission denied" in deleted.removal_error
    assert _registered_runs(ds) == []
    assert read_inflight(model_run_root(ds, KIND, RUN_ID)) is None, "claim released"
    with pytest.raises(ModelNotFoundError):
        _ = resolve_op_model(ds, "infer-pose", RUN_ID)


def test_a_model_in_a_linked_library_is_deleted_through_the_library(
    tmp_path: Path,
) -> None:
    """The library owns the model; a project linking it stops resolving it."""
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    project = make_dataset(tmp_path / "52", name="project")
    _register(library)
    _ = project.add_library(LibraryLink(id="group", path="../libraries/7"))
    assert resolve_op_model(project, "infer-pose", RUN_ID).library_id == "group"

    deleted = delete_trained_model(library, KIND, RUN_ID)

    assert deleted.directory_removed
    with pytest.raises(ModelNotFoundError):
        _ = resolve_op_model(project, "infer-pose", RUN_ID)
