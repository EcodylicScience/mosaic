"""A cancelled YOLO or POLO training run stops at an epoch boundary.

Ultralytics cannot be interrupted inside an epoch, so the only clean stop is a
flag it reads between them: ``trainer.stop = True``, set from the callback
mosaic registers on ``on_train_epoch_end``. What that buys is the whole point --
the trainer writes ``last.pt`` and appends to ``results.csv`` every epoch, so a
run stopped this way leaves a complete checkpoint and a complete curve, where a
process kill loses whichever epoch was in flight.

Nothing exercised any of that. The two existing training suites replace
``train_pose_model`` itself, so the callback bridge below it -- the event name,
the metric casting, the stop flag -- had no coverage at all, and a change that
swapped the cooperative stop for a kill would have passed the suite.

The stand-in is Ultralytics' contract rather than Ultralytics: ``add_callback``
registers, ``train`` runs epochs and consults ``trainer.stop`` between them, and
``last.pt`` and ``results.csv`` grow as it goes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pytest

import mosaic.tracking.pose_training.train as tr

EPOCH_EVENT = "on_train_epoch_end"
"""The event mosaic registers on.

Spelled here as well as in the source because Ultralytics' callback registry is a
``defaultdict(list)``: a misspelled name registers a callback that is never
called, cancellation silently stops working, and nothing raises. The name is the
whole contract, so it is asserted rather than assumed.
"""


class EpochCallback(Protocol):
    """What ``_register_ultralytics_callback`` bridges to."""

    def on_epoch_end(
        self, epoch: int, total_epochs: int, metrics: dict[str, float]
    ) -> None: ...


class _Loss:
    """A stand-in for the loss tensor a trainer exposes, which needs ``.item()``."""

    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _Trainer:
    """The object Ultralytics hands its callbacks."""

    def __init__(self, run_dir: Path) -> None:
        self.epoch = 0
        self.loss = _Loss(0.5)
        self.metrics: dict[str, object] = {"mAP50": 0.25}
        self.stop = False
        self._run_dir = run_dir

    def write_epoch(self) -> None:
        """What the real trainer leaves on disk at the end of every epoch."""
        weights = self._run_dir / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        _ = (weights / "last.pt").write_bytes(b"checkpoint")
        with (self._run_dir / "results.csv").open("a") as handle:
            _ = handle.write(f"{self.epoch},{self.loss.item()}\n")


class _Recorder:
    """A progress callback that keeps every epoch it was told about."""

    def __init__(self) -> None:
        self.epochs: list[tuple[int, int, dict[str, float]]] = []

    def on_epoch_end(
        self, epoch: int, total_epochs: int, metrics: dict[str, float]
    ) -> None:
        self.epochs.append((epoch, total_epochs, dict(metrics)))


class _FakeYolo:
    """Ultralytics' surface, as far as the callback bridge can see it."""

    trainers: list[_Trainer] = []

    def __init__(self, _model: str) -> None:
        self.registered: dict[str, list[Callable[[_Trainer], None]]] = {}

    def add_callback(self, event: str, func: Callable[[_Trainer], None]) -> None:
        self.registered.setdefault(event, []).append(func)

    def train(self, **kwargs: object) -> None:
        """Run epochs, honoring ``trainer.stop`` between them, as Ultralytics does."""
        run_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
        run_dir.mkdir(parents=True, exist_ok=True)
        trainer = _Trainer(run_dir)
        _FakeYolo.trainers.append(trainer)
        total = int(str(kwargs["epochs"]))
        for epoch in range(total):
            trainer.epoch = epoch
            trainer.write_epoch()
            for func in self.registered.get(EPOCH_EVENT, []):
                func(trainer)
            if trainer.stop:
                return


@pytest.fixture(autouse=True)
def fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the stand-in behind both trainers, and forget the previous test's."""
    _FakeYolo.trainers = []
    monkeypatch.setattr(tr, "_require_ultralytics", lambda: _FakeYolo)
    monkeypatch.setattr(tr, "_require_polo", lambda: _FakeYolo)


def _train(
    tmp_path: Path,
    *,
    callback: EpochCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Path:
    """Run the pose trainer into *tmp_path*, and return the run directory."""
    data_yaml = tmp_path / "data.yaml"
    _ = data_yaml.write_text("kpt_shape: [4, 3]\n")
    _ = tr.train_pose_model(
        data_yaml,
        epochs=6,
        project=str(tmp_path / "run"),
        name="train",
        callback=callback,
        cancel_check=cancel_check,
    )
    return tmp_path / "run" / "train"


def test_a_cancel_stops_at_an_epoch_boundary(tmp_path: Path) -> None:
    """The behavior this bridge exists for, and the one a kill would lose."""
    fired = {"calls": 0}

    def cancel_after_two() -> bool:
        fired["calls"] += 1
        return fired["calls"] >= 2

    run_dir = _train(tmp_path, cancel_check=cancel_after_two)

    trainer = _FakeYolo.trainers[-1]
    assert trainer.stop is True, "the cancel must reach the trainer as its stop flag"
    assert trainer.epoch == 1, (
        "the run must end at the boundary of the epoch the cancel arrived in, "
        f"not run on to {trainer.epoch}"
    )
    assert (run_dir / "weights" / "last.pt").is_file(), (
        "a cooperative stop leaves the checkpoint the epoch wrote"
    )
    assert (run_dir / "results.csv").read_text().splitlines() == ["0,0.5", "1,0.5"], (
        "and leaves the curve complete up to the epoch it stopped on"
    )


def test_an_uncancelled_run_trains_every_epoch(tmp_path: Path) -> None:
    """The stop flag is not set when nothing asked for it."""
    run_dir = _train(tmp_path, cancel_check=lambda: False)

    assert _FakeYolo.trainers[-1].stop is False
    assert len((run_dir / "results.csv").read_text().splitlines()) == 6


def test_every_epoch_reaches_the_progress_callback(tmp_path: Path) -> None:
    """Position and denominator both, since a run-log reduces the pair."""
    recorder = _Recorder()
    _ = _train(tmp_path, callback=recorder)

    assert [epoch for epoch, _total, _metrics in recorder.epochs] == [0, 1, 2, 3, 4, 5]
    assert {total for _epoch, total, _metrics in recorder.epochs} == {6}


def test_the_metrics_are_cast_and_the_uncastable_are_dropped(tmp_path: Path) -> None:
    """The loss arrives through ``.item()``; a metric that will not cast is skipped.

    Pinned because the casting is what has to survive a move across a process
    boundary, where a metric can no longer be a live tensor and the dropping has
    to happen somewhere specific.
    """
    recorder = _Recorder()

    def unsupported_metric(trainer: _Trainer) -> None:
        trainer.metrics = {"mAP50": 0.25, "notes": "not a number"}

    original = _FakeYolo.train

    def train_with_odd_metrics(self: _FakeYolo, **kwargs: object) -> None:
        self.registered.setdefault(EPOCH_EVENT, []).insert(0, unsupported_metric)
        original(self, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_FakeYolo, "train", train_with_odd_metrics)
        _ = _train(tmp_path, callback=recorder)

    _epoch, _total, metrics = recorder.epochs[0]
    assert metrics == {"loss": 0.5, "mAP50": 0.25}, (
        "the loss is read through .item(), the numeric metric is cast, and the "
        "string is dropped rather than raising"
    )


def test_the_bridge_registers_on_the_event_ultralytics_calls(tmp_path: Path) -> None:
    """A misspelled event registers happily and never fires, so the name is pinned."""
    captured: list[_FakeYolo] = []
    original_init = _FakeYolo.__init__

    def remember(self: _FakeYolo, model: str) -> None:
        original_init(self, model)
        captured.append(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_FakeYolo, "__init__", remember)
        _ = _train(tmp_path, cancel_check=lambda: False)

    assert list(captured[-1].registered) == [EPOCH_EVENT]


def test_no_callback_is_registered_when_nothing_asked_for_one(tmp_path: Path) -> None:
    """A run with neither a progress callback nor a cancel check registers nothing."""
    captured: list[_FakeYolo] = []
    original_init = _FakeYolo.__init__

    def remember(self: _FakeYolo, model: str) -> None:
        original_init(self, model)
        captured.append(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_FakeYolo, "__init__", remember)
        _ = _train(tmp_path)

    assert captured[-1].registered == {}


def test_the_point_trainer_cancels_the_same_way(tmp_path: Path) -> None:
    """``train-points`` shares the bridge, so it shares the guarantee."""
    data_yaml = tmp_path / "data.yaml"
    _ = data_yaml.write_text("names: [bee]\nradii: {0: 5.0}\n")
    fired = {"calls": 0}

    def cancel_after_two() -> bool:
        fired["calls"] += 1
        return fired["calls"] >= 2

    tr.train_point_model(
        data_yaml,
        epochs=6,
        project=str(tmp_path / "run"),
        name="train",
        cancel_check=cancel_after_two,
    )

    assert _FakeYolo.trainers[-1].stop is True
    assert _FakeYolo.trainers[-1].epoch == 1
    assert (tmp_path / "run" / "train" / "weights" / "last.pt").is_file()
