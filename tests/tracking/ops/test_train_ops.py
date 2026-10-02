"""The training ops under the job contract, with the trainer faked.

The run-log lifecycle, lineage from a base model, cancel and a killed trainer,
and the trained-model index row each run registers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.pipeline.job import CancelToken, Cancelled
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.subprocess_util import ProcessCancelled
from mosaic.tracking.external.runner.ultralytics_protocol import TrainRequestBase
from mosaic.tracking.pose_training.ultralytics_train import TrainingOutcome
from mosaic.core.pipeline.run_log import read_run, read_runs, run_log_dir
from mosaic.tracking import resolve_model

from tests.helpers import FakeTrainer, stub_media_dataset


def _install_fake_pose_trainer(monkeypatch) -> FakeTrainer:
    """Replace the two seams that reach the training environment."""
    trainer = FakeTrainer()
    trainer.install(monkeypatch)
    return trainer


def test_train_pose_lifecycle_and_lineage(tmp_path, monkeypatch):
    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    _install_fake_pose_trainer(monkeypatch)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("kpt_shape: [4, 3]\n")

    r1 = run_op(
        ds, "train-pose", {"data": str(data_yaml), "epochs": 2, "device": "cpu"}
    )
    assert r1.startswith("train-pose.")
    row = read_run(
        run_log_dir(ds.base_dir),
        read_runs(run_log_dir(ds.base_dir), kind="train-pose")[0]["execution_id"],
    )
    assert row["status"] == "finished" and row["run_id"] == r1
    # per-epoch on_epoch_end advances the coarse runs-row counter (2 epochs -> 2/2),
    # so `status --json` progress_done tracks training epochs, not just the stream.
    assert row["progress_done"] == 2 and row["progress_total"] == 2

    # model index row written with the best.pt path
    from mosaic.tracking.ops.train import trained_model_index
    from mosaic.core.pipeline.models import model_index_path

    midx = trained_model_index(model_index_path(ds, "train-pose"))
    mdf = midx.read(run_id=r1)
    assert len(mdf) == 1
    assert mdf.iloc[0]["best_model_path"].endswith("best.pt")
    assert mdf.iloc[0]["base_run_id"] == ""

    # resolve_model turns the run_id into its best.pt (train->track handoff)
    resolved = resolve_model(ds, r1, "train-pose")
    assert resolved.path.name == "best.pt"
    assert resolved.run_id == r1
    # A registered model is named by its run, so the digest never reaches
    # identity -- but it is measured and recorded either way.
    assert resolved.model_id == r1
    assert len(resolved.digest) == 16

    # retrain from r1 -> lineage recorded
    r2 = run_op(
        ds, "train-pose", {"data": str(data_yaml), "epochs": 2, "base_model": r1}
    )
    mdf2 = trained_model_index(model_index_path(ds, "train-pose")).read(run_id=r2)
    assert mdf2.iloc[0]["base_run_id"] == r1
    assert mdf2.iloc[0]["base_digest"] == resolved.digest


def test_train_pose_cancel(tmp_path, monkeypatch):
    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    _install_fake_pose_trainer(monkeypatch)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("kpt_shape: [4, 3]\n")

    token = CancelToken()
    token.cancel()  # already cancelled -> ctx.check_cancel() after the tool raises
    with pytest.raises(Cancelled):
        run_op(
            ds, "train-pose", {"data": str(data_yaml), "epochs": 1}, cancel_token=token
        )
    assert (
        read_runs(run_log_dir(ds.base_dir), kind="train-pose")[0]["status"]
        == "cancelled"
    )


class _KilledTrainer(FakeTrainer):
    """A trainer that did not stop within its grace, and was killed."""

    def __call__(
        self,
        request: TrainRequestBase,
        /,
        *,
        work_dir: Path,
        idle_timeout: float,
        cancel_check: object = None,
        on_output: object = None,
        **_kwargs: object,
    ) -> TrainingOutcome:
        raise ProcessCancelled(["yolo", "train", request.run_name])


def test_a_killed_trainer_records_a_cancelled_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_op converts a killed tool, so the trainer does not have to.

    The token is not set, so no cancel check in the op can raise instead.
    """
    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    _KilledTrainer().install(monkeypatch)
    data_yaml = tmp_path / "data.yaml"
    _ = data_yaml.write_text("kpt_shape: [4, 3]\n")

    with pytest.raises(Cancelled):
        _ = run_op(ds, "train-pose", {"data": str(data_yaml), "epochs": 1})

    (run,) = read_runs(run_log_dir(ds.base_dir), kind="train-pose")
    assert run["status"] == "cancelled"
    assert run["error_json"] == ""


def test_a_tool_that_reports_a_cancel_is_never_registered(tmp_path, monkeypatch):
    """A truncated model must not land under a finished run's identifier.

    The token is normally already set and the check above has raised. This is the
    other order: the tool stopped short because it was asked to, while this
    process no longer thinks it was.
    """
    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    trainer = FakeTrainer(epochs_run=1, stop="cancelled")
    trainer.install(monkeypatch)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("kpt_shape: [4, 3]\n")

    with pytest.raises(Cancelled):
        run_op(ds, "train-pose", {"data": str(data_yaml), "epochs": 8})

    from mosaic.core.pipeline.models import model_index_path
    from mosaic.tracking.ops.train import trained_model_index

    index = model_index_path(ds, "train-pose")
    assert not index.exists() or trained_model_index(index).read().empty


def test_the_index_records_the_epochs_that_actually_ran(tmp_path, monkeypatch):
    """``patience`` stops a run short, and a forty-epoch model is not a three-hundred."""
    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    trainer = FakeTrainer(epochs_run=3, stop="early_stopped")
    trainer.install(monkeypatch)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("kpt_shape: [4, 3]\n")

    run_id = run_op(ds, "train-pose", {"data": str(data_yaml), "epochs": 40})

    from mosaic.core.pipeline.models import model_index_path
    from mosaic.tracking.ops.train import trained_model_index

    rows = trained_model_index(model_index_path(ds, "train-pose")).read(run_id=run_id)
    assert int(rows.iloc[0]["n_epochs"]) == 3


def test_train_localizer_mints_and_registers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The localizer op reaches its minter, and records what it produced.

    Nothing used to run this op. The registry tests assert the kind is
    registered and the golden corpus calls ``train_run_id`` directly with fixed
    arguments, so both stayed green while ``TrainLocalizerOp.run`` passed it one
    argument too many -- a ``TypeError`` on every real localizer training run.
    Only the type checker saw it. This is the test that would have.
    """
    import mosaic.tracking.pose_training.localizer_train as lt
    from mosaic.core.pipeline.models import model_index_path
    from mosaic.tracking.ops.train import trained_model_index

    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    dataset_dir = tmp_path / "patches"
    (dataset_dir / "train").mkdir(parents=True)
    _ = (dataset_dir / "train" / "patches.npy").write_bytes(b"patches")

    def fake_train_localizer(
        dataset_dir: str | Path,
        *,
        project: str | Path,
        name: str,
        epochs: int = 1,
        **kw: object,
    ) -> lt.TrainingResult:
        run_dir = Path(project) / name
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        weights = run_dir / "weights" / "best.pt"
        _ = weights.write_bytes(b"localizer weights")
        _ = (run_dir / "results.csv").write_text("epoch,loss\n0,0.1\n")
        return lt.TrainingResult(
            best_model_path=weights,
            last_model_path=weights,
            run_dir=run_dir,
            best_epoch=0,
            best_val_loss=0.1,
        )

    monkeypatch.setattr(lt, "train_localizer", fake_train_localizer)

    run_id = run_op(
        ds,
        "train-localizer",
        {"dataset_dir": str(dataset_dir), "epochs": 1, "device": "cpu"},
    )
    assert run_id.startswith("train-localizer.")

    index = trained_model_index(model_index_path(ds, "train-localizer"))
    row = index.read().iloc[0]
    assert row["run_id"] == run_id
    assert row["status"] == "finished"
    assert str(row["best_model_path"]).endswith("best.pt")


def test_point_train_default_model_is_polo26n():
    from mosaic.tracking.ops.train import PointTrainParams

    assert PointTrainParams(data="d.yaml").model == "polo26n.yaml"
