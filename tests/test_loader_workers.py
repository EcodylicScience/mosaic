"""A training run's data-loader worker count is throughput, never identity.

How many processes load data changes how long a run takes, not what it
trains. Until each op had a field for it, the only way to set it was the op's
overrides bag -- and everything in that bag is hashed, because a learning rate
set there is a different model. So retuning the loader minted a new
identifier and retrained a model nothing about had changed. sleap-nn asks for
exactly that retuning on every run: its own warning suggests ``num_workers=31``.

Each op now carries the count as a ``HASH_EXCLUDE`` field and writes the tool's
own key from it, and the bag refuses that key, so there is one way to set it and
it costs nothing. What these hold, for SLEAP, Lightning Pose and YOLO alike:

- the field does not reach the identity, and changing it reuses the model;
- the tool is told, under the key it reads;
- the bag refuses the key, naming the field to use instead.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mosaic.core.dataset import Dataset
from mosaic.core.params import Params
from mosaic.core.pipeline.ops import run_op
from mosaic.tracking import register_ops
from mosaic.tracking.ops.train import PointTrainParams, PoseTrainParams
from mosaic.tracking.ops.train_litpose import TrainLitposeParams
from mosaic.tracking.ops.train_sleap import TrainSleapParams

from tests.helpers import FakeTrainer, make_dataset

register_ops()


# --- the identity ----------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "with_workers"),
    [
        (TrainSleapParams(labels="s.slp"), {"num_workers": 8}),
        (TrainLitposeParams(project="p"), {"num_workers": 8}),
        (PoseTrainParams(data="d.yaml"), {"workers": 8}),
        (PointTrainParams(data="d.yaml"), {"workers": 8}),
    ],
    ids=["train-sleap", "train-litpose", "train-pose", "train-points"],
)
def test_the_worker_count_does_not_reach_the_identity(
    params: Params, with_workers: dict[str, int]
) -> None:
    """Recorded with the run, hashed into nothing.

    ``model_dump`` still carries it -- ``params.json`` records what a run was
    given -- and only the identity leaves it out.
    """
    tuned = params.model_copy(update=with_workers)

    assert tuned.identity_dump() == params.identity_dump()
    assert tuned.model_dump() != params.model_dump(), "the value is still recorded"


# --- the refusal -----------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "trainer_config.train_data_loader.num_workers",
        "trainer_config.val_data_loader.num_workers",
        "+trainer_config.train_data_loader.num_workers",
        "++trainer_config.val_data_loader.num_workers",
    ],
)
def test_sleap_overrides_refuse_the_worker_count(key: str) -> None:
    """However it is spelled: the bag passes a ``+`` or ``++`` through as written."""
    with pytest.raises(ValidationError, match="num_workers instead"):
        _ = TrainSleapParams(labels="s.slp", sleap_overrides={key: 8})


def test_sleap_overrides_still_take_everything_else() -> None:
    """The refusal is one key wide, not a bag that stopped working."""
    params = TrainSleapParams(
        labels="s.slp", sleap_overrides={"trainer_config.optimizer.lr": 5e-4}
    )
    assert params.sleap_overrides == {"trainer_config.optimizer.lr": 5e-4}


def test_litpose_overrides_refuse_the_worker_count() -> None:
    with pytest.raises(ValidationError, match="num_workers instead"):
        _ = TrainLitposeParams(
            project="p", litpose_overrides={"training.num_workers": 8}
        )


def test_train_overrides_refuse_the_worker_count() -> None:
    """No new code: the guard already refuses any key a field owns."""
    with pytest.raises(ValidationError, match="workers"):
        _ = PoseTrainParams(data="d.yaml", train_overrides={"workers": 8})


@pytest.mark.parametrize("value", [-1, -8])
def test_a_negative_worker_count_is_refused_at_submit(value: int) -> None:
    """Refused for its value. Matching the message is what tells that apart from
    the field being refused for not existing, which would also raise."""
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        _ = TrainSleapParams(labels="s.slp", num_workers=value)


# --- SLEAP, end to end -----------------------------------------------------


class _Sleap:
    """sleap-nn, faked at the two subprocess seams, counting what it trains."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from mosaic.tracking.sleap import probe as probe_module
        from mosaic.tracking.sleap import training as training_module

        self.argv: list[list[str]] = []
        (tmp_path / "bin").mkdir(exist_ok=True)
        for name in ("sleap-nn-train", "python"):
            _ = (tmp_path / "bin" / name).write_text("")
        monkeypatch.setenv("MOSAIC_SLEAP_BIN", str(tmp_path / "bin" / "sleap-nn-train"))
        monkeypatch.setattr(probe_module, "run_supervised", self._probe)
        monkeypatch.setattr(training_module, "run_supervised", self._train)
        self.labels = tmp_path / "session.slp"
        _ = self.labels.write_bytes(b"slp")

    def _probe(self, argv: Sequence[str], **kw: object) -> tuple[str, str, int]:
        answer: dict[str, object] = {"has_sleap_nn": True, "n_tracks": 2}
        _ = Path(argv[argv.index("-c") + 3]).write_text(json.dumps(answer))
        return ("probed", "", 0)

    def _train(self, argv: Sequence[str], **kw: object) -> tuple[str, str, int]:
        self.argv.append(list(argv))
        run_root = Path(argv[argv.index("--config-dir") + 1])
        config = yaml.safe_load((run_root / "config.yaml").read_text())
        produced = run_root / config["trainer_config"]["run_name"]
        produced.mkdir(parents=True, exist_ok=True)
        _ = (produced / "best.ckpt").write_bytes(b"weights")
        _ = (produced / "training_config.yaml").write_text(
            "head_configs:\n  centroid: {}\n"
        )
        return ("done", "", 0)

    def run(self, ds: Dataset, **params: object) -> str:
        base: dict[str, object] = {
            "labels": str(self.labels),
            "max_epochs": 1,
            "head": "centroid",
        }
        return run_op(ds, "train-sleap", {**base, **params})


def test_sleap_reuses_the_model_when_only_the_worker_count_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The promise, as the user meets it: retuning the loader is free."""
    ds = make_dataset(tmp_path, save=False)
    sleap = _Sleap(tmp_path, monkeypatch)

    first = sleap.run(ds)
    again = sleap.run(ds, num_workers=8)

    assert again == first
    assert len(sleap.argv) == 1, "the second run was a cache hit, not a retrain"


def test_sleap_hands_the_worker_count_to_both_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Appended with ``+``, because the generated config declares neither loader."""
    ds = make_dataset(tmp_path, save=False)
    sleap = _Sleap(tmp_path, monkeypatch)

    _ = sleap.run(ds, num_workers=8)

    argv = sleap.argv[-1]
    assert "+trainer_config.train_data_loader.num_workers=8" in argv
    assert "+trainer_config.val_data_loader.num_workers=8" in argv


def test_sleap_sends_nothing_when_the_worker_count_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset is sleap-nn's own default, not mosaic's restatement of it."""
    ds = make_dataset(tmp_path, save=False)
    sleap = _Sleap(tmp_path, monkeypatch)

    _ = sleap.run(ds)

    assert not [arg for arg in sleap.argv[-1] if "num_workers" in arg]


# --- Lightning Pose, end to end --------------------------------------------


class _Litpose:
    """Lightning Pose, faked at its subprocess seam, counting what it trains."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from mosaic.tracking.litpose import training as training_module

        self.argv: list[list[str]] = []
        (tmp_path / "bin").mkdir(exist_ok=True)
        for name in ("litpose", "python"):
            _ = (tmp_path / "bin" / name).write_text("")
        monkeypatch.setenv("MOSAIC_LITPOSE_BIN", str(tmp_path / "bin" / "litpose"))
        monkeypatch.setattr(training_module, "run_supervised", self._train)
        self.project = tmp_path / "project"
        self.project.mkdir()
        _ = (self.project / "config.yaml").write_text("data:\n  num_keypoints: 3\n")
        self.base_config = tmp_path / "litpose_default.yaml"
        _ = self.base_config.write_text("training:\n  num_gpus: 0\n")

    def _train(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        **kw: object,
    ) -> tuple[str, str, int]:
        self.argv.append(list(argv))
        run_root = Path(argv[argv.index("-c") + 4])
        checkpoints = run_root / "tb_logs" / "run" / "version_0" / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        _ = (run_root / "config.yaml").write_text("model:\n  model_type: heatmap\n")
        _ = (checkpoints / "best.ckpt").write_bytes(b"weights")
        return ("done", "", 0)

    def run(self, ds: Dataset, **params: object) -> str:
        base: dict[str, object] = {
            "project": str(self.project),
            "base_config": str(self.base_config),
            "max_epochs": 2,
        }
        return run_op(ds, "train-litpose", {**base, **params})


def test_litpose_reuses_the_model_when_only_the_worker_count_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path, save=False)
    litpose = _Litpose(tmp_path, monkeypatch)

    first = litpose.run(ds)
    again = litpose.run(ds, num_workers=8)

    assert again == first
    assert len(litpose.argv) == 1, "the second run was a cache hit, not a retrain"


def test_litpose_hands_the_worker_count_to_the_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path, save=False)
    litpose = _Litpose(tmp_path, monkeypatch)

    _ = litpose.run(ds, num_workers=8)
    assert "training.num_workers=8" in litpose.argv[-1]


def test_litpose_sends_nothing_when_the_worker_count_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path, save=False)
    litpose = _Litpose(tmp_path, monkeypatch)

    _ = litpose.run(ds)
    assert not [arg for arg in litpose.argv[-1] if "num_workers" in arg]


# --- YOLO, end to end ------------------------------------------------------


def _data_yaml(tmp_path: Path) -> Path:
    directory = tmp_path / "converted"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "data.yaml"
    _ = path.write_text("kpt_shape: [4, 3]\n")
    return path


def test_yolo_reuses_the_model_when_only_the_worker_count_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path, save=False)
    trainer = FakeTrainer()
    trainer.install(monkeypatch)
    params: dict[str, object] = {
        "data": str(_data_yaml(tmp_path)),
        "epochs": 2,
        "device": "cpu",
    }

    first = run_op(ds, "train-pose", dict(params))
    again = run_op(ds, "train-pose", {**params, "workers": 8})

    assert again == first
    assert trainer.calls == 1, "the second run was a cache hit, not a retrain"


@pytest.mark.parametrize("workers", [None, 8])
def test_yolo_sends_the_worker_count_in_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int | None
) -> None:
    """Across the wire as a field; the runner is what turns it into a kwarg."""
    ds = make_dataset(tmp_path, save=False)
    trainer = FakeTrainer()
    trainer.install(monkeypatch)
    params: dict[str, object] = {
        "data": str(_data_yaml(tmp_path)),
        "epochs": 2,
        "device": "cpu",
        "workers": workers,
    }

    _ = run_op(ds, "train-pose", params)
    assert trainer.last_request.workers == workers
