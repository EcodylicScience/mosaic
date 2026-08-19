"""The runner's training subcommands, driven against a stand-in Ultralytics.

Training is the one path where *how a run ends* is not recoverable from disk:
a completed run, a run stopped by ``patience`` and a run stopped by a cancel all
leave ``best.pt``, ``last.pt`` and a ``results.csv``. So the runner has to say
which it was, and the epoch boundary it stops on has to be real -- that is the
whole reason a sentinel file exists rather than a process kill.

None of that can be tested against the real library here: mosaic's environment
holds no Ultralytics, and installing one would defeat the separation these tests
sit under. What is injected instead is Ultralytics' *contract* -- a model that
registers callbacks, a trainer that runs epochs, honors ``stop`` between them and
writes what a real one writes -- so everything under test is the runner's own
code, running for real.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import mosaic.tracking.external.runner as runner_package

pytestmark = pytest.mark.tracker

TOTAL_EPOCHS = 5


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """The runner program, imported the way it is when spawned, then unimported."""
    directory = str(Path(runner_package.__file__).parent)
    inserted = directory not in sys.path
    if inserted:
        sys.path.insert(0, directory)
    try:
        yield importlib.import_module("ultralytics_runner")
    finally:
        if inserted:
            sys.path.remove(directory)
        for name in ("ultralytics_runner", "ultralytics_protocol"):
            _ = sys.modules.pop(name, None)


class _Trainer:
    """What Ultralytics hands a callback, and what it leaves on disk."""

    def __init__(self, save_dir: Path, epochs: int) -> None:
        self.save_dir = save_dir
        self.epochs = epochs
        self.epoch = 0
        self.loss = SimpleNamespace(item=lambda: 0.5)
        self.metrics: dict[str, object] = {"mAP50": 0.25, "notes": "not a number"}
        self.stop = False

    def write_epoch(self) -> None:
        weights = self.save_dir / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        _ = (weights / "last.pt").write_bytes(b"checkpoint")
        _ = (weights / "best.pt").write_bytes(b"checkpoint")
        with (self.save_dir / "results.csv").open("a") as handle:
            _ = handle.write(f"{self.epoch},0.5\n")


class _FakeYolo:
    """The model surface the training path touches."""

    last: "_FakeYolo | None" = None

    def __init__(self, model: str) -> None:
        self.model = model
        self.callbacks: dict[str, list[Callable[[_Trainer], None]]] = {}
        self.kwargs: dict[str, object] = {}
        self.trainer: _Trainer | None = None
        self.stop_after: int | None = None
        _FakeYolo.last = self

    def add_callback(self, event: str, func: Callable[[_Trainer], None]) -> None:
        self.callbacks.setdefault(event, []).append(func)

    def train(self, **kwargs: object) -> None:
        self.kwargs = dict(kwargs)
        save_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
        save_dir.mkdir(parents=True, exist_ok=True)
        total = int(str(kwargs["epochs"]))
        trainer = _Trainer(save_dir, total)
        self.trainer = trainer
        for func in self.callbacks.get("on_train_start", []):
            func(trainer)
        for epoch in range(total):
            trainer.epoch = epoch
            trainer.write_epoch()
            for func in self.callbacks.get("on_train_epoch_end", []):
                func(trainer)
            if trainer.stop:
                return
            if self.stop_after is not None and epoch + 1 >= self.stop_after:
                return  # what `patience` looks like from out here


def _install_ultralytics(
    monkeypatch: pytest.MonkeyPatch, *, events: tuple[str, ...] | None = None
) -> None:
    """Put a stand-in Ultralytics where the runner's deferred imports find it."""
    known = events if events is not None else ("on_train_start", "on_train_epoch_end")
    ultralytics = ModuleType("ultralytics")
    setattr(ultralytics, "YOLO", _FakeYolo)
    base = ModuleType("ultralytics.utils.callbacks.base")
    setattr(base, "default_callbacks", {name: [] for name in known})
    for name, module in (
        ("ultralytics", ultralytics),
        ("ultralytics.utils", ModuleType("ultralytics.utils")),
        ("ultralytics.utils.callbacks", ModuleType("ultralytics.utils.callbacks")),
        ("ultralytics.utils.callbacks.base", base),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def _request(runner: ModuleType, tmp_path: Path, **overrides: object):
    """A pose training request with everything resolved, as mosaic sends it."""
    fields: dict[str, object] = {
        "model": "yolo11n-pose.pt",
        "data_yaml": str(tmp_path / "data.yaml"),
        "epochs": TOTAL_EPOCHS,
        "imgsz": 640,
        "batch": 16,
        "device": "cpu",
        "patience": 50,
        "project_dir": str(tmp_path / "run"),
        "run_name": "train",
        "resume": False,
        "augment": {"fliplr": 0.5, "mosaic": 0.5},
        "train_overrides": {},
        "cancel_sentinel": str(tmp_path / "cancel"),
    }
    fields.update(overrides)
    return runner.TrainPoseRequest(**fields)


def _lines(captured: str) -> list[dict[str, object]]:
    """Every event line, parsed, refusing the non-JSON a bare NaN would be."""

    def refuse(token: str) -> float:
        raise AssertionError(f"{token!r} is not JSON, and this stream must be")

    return [
        json.loads(line, parse_constant=refuse)
        for line in captured.splitlines()
        if line.strip()
    ]


def test_a_full_run_reports_every_epoch_and_says_it_completed(
    runner: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_ultralytics(monkeypatch)
    response = runner.run_train_pose(_request(runner, tmp_path))

    assert response.stop == "completed"
    assert response.epochs_completed == TOTAL_EPOCHS
    assert response.save_dir == str(tmp_path / "run" / "train")

    events = _lines(capsys.readouterr().out)
    epochs = [event for event in events if event["event"] == "epoch"]
    assert [event["epoch"] for event in epochs] == list(range(TOTAL_EPOCHS))
    assert {event["total_epochs"] for event in epochs} == {TOTAL_EPOCHS}


def test_the_startup_stretch_is_announced_at_both_ends(
    runner: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Twice: once when Ultralytics is imported, once when training begins.

    Between them lie the weights load, the dataset scan, the label cache and the
    AMP check -- minutes on a large set, and silent. One line at each end is what
    lets an inactivity window be chosen against an epoch rather than against the
    sum of an epoch and all of that.
    """
    _install_ultralytics(monkeypatch)
    _ = runner.run_train_pose(_request(runner, tmp_path))

    events = _lines(capsys.readouterr().out)
    started = [
        index for index, event in enumerate(events) if event["event"] == "started"
    ]
    first_epoch = next(
        index for index, event in enumerate(events) if event["event"] == "epoch"
    )
    assert len(started) == 2, "one line on import, one when the trainer starts"
    assert all(index < first_epoch for index in started)


def test_a_sentinel_stops_the_run_at_an_epoch_boundary(
    runner: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The behavior a process kill cannot give, and the reason for the file."""
    _install_ultralytics(monkeypatch)
    sentinel = tmp_path / "cancel"
    request = _request(runner, tmp_path)

    original = _FakeYolo.train

    def train_and_cancel_midway(self: _FakeYolo, **kwargs: object) -> None:
        def touch_after_second(trainer: _Trainer) -> None:
            if trainer.epoch == 1:
                _ = sentinel.write_text("cancelled")

        self.callbacks.setdefault("on_train_epoch_end", []).insert(
            0, touch_after_second
        )
        original(self, **kwargs)

    monkeypatch.setattr(_FakeYolo, "train", train_and_cancel_midway)
    response = runner.run_train_pose(request)

    assert response.stop == "cancelled"
    assert response.epochs_completed == 2, "the epoch it was in still finished"
    run_dir = tmp_path / "run" / "train"
    assert (run_dir / "weights" / "last.pt").is_file()
    assert (run_dir / "results.csv").read_text().splitlines() == ["0,0.5", "1,0.5"]

    epochs = [e for e in _lines(capsys.readouterr().out) if e["event"] == "epoch"]
    assert len(epochs) == 2, "the stopped epoch is reported like any other"


def test_a_run_that_stops_short_on_its_own_is_not_a_cancellation(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``patience`` and a cancel look identical on disk, so they are told apart here."""
    _install_ultralytics(monkeypatch)
    request = _request(runner, tmp_path)

    original = _FakeYolo.train

    def train_with_patience(self: _FakeYolo, **kwargs: object) -> None:
        self.stop_after = 3
        original(self, **kwargs)

    monkeypatch.setattr(_FakeYolo, "train", train_with_patience)
    response = runner.run_train_pose(request)

    assert response.stop == "early_stopped"
    assert response.epochs_completed == 3


def test_a_sentinel_on_the_last_epoch_is_still_a_finished_model(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancel that loses the race has cost nothing, and must not discard a model."""
    _install_ultralytics(monkeypatch)
    sentinel = tmp_path / "cancel"
    _ = sentinel.write_text("cancelled")

    original = _FakeYolo.train

    def train_ignoring_stop(self: _FakeYolo, **kwargs: object) -> None:
        kwargs["epochs"] = 1
        original(self, **kwargs)

    monkeypatch.setattr(_FakeYolo, "train", train_ignoring_stop)
    response = runner.run_train_pose(_request(runner, tmp_path, epochs=1))

    assert response.stop == "completed"


def test_the_metrics_that_cross_are_finite_floats_and_nothing_else(
    runner: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A NaN would serialize as null and cost the whole epoch line, not one key."""
    _install_ultralytics(monkeypatch)
    original = _FakeYolo.train

    def train_with_a_nan(self: _FakeYolo, **kwargs: object) -> None:
        def spoil(trainer: _Trainer) -> None:
            trainer.metrics = {
                "mAP50": 0.25,
                "notes": "not a number",
                "mAP75": float("nan"),
                "ceiling": float("inf"),
            }

        self.callbacks.setdefault("on_train_epoch_end", []).insert(0, spoil)
        original(self, **kwargs)

    monkeypatch.setattr(_FakeYolo, "train", train_with_a_nan)
    _ = runner.run_train_pose(_request(runner, tmp_path))

    epochs = [e for e in _lines(capsys.readouterr().out) if e["event"] == "epoch"]
    assert epochs, "the epoch survived rather than being dropped for one bad key"
    assert epochs[0]["metrics"] == {"loss": 0.5, "mAP50": 0.25}


def test_an_environment_without_the_callback_refuses_rather_than_going_quiet(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure no test in mosaic's own environment could otherwise catch.

    Ultralytics' registry is a ``defaultdict(list)`` and ``add_callback`` appends
    without asking whether the key is ever called, so a renamed event would leave
    progress unreported and a cancel unhonored, with nothing raised anywhere.
    """
    _install_ultralytics(monkeypatch, events=("on_train_start",))

    with pytest.raises(runner.RunnerError, match="on_train_epoch_end"):
        _ = runner.run_train_pose(_request(runner, tmp_path))


def test_the_caller_overrides_beat_the_resolved_augmentation(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two bags, applied in order, which is why they stay two on the wire."""
    _install_ultralytics(monkeypatch)
    request = _request(
        runner,
        tmp_path,
        augment={"fliplr": 0.5, "mosaic": 0.5},
        train_overrides={"fliplr": 0.9, "lr0": 0.0044},
    )
    _ = runner.run_train_pose(request)

    assert _FakeYolo.last is not None
    kwargs = _FakeYolo.last.kwargs
    assert kwargs["fliplr"] == 0.9, "the override wins"
    assert kwargs["mosaic"] == 0.5, "the preset key it did not name survives"
    assert kwargs["lr0"] == 0.0044


def test_the_run_directory_is_pinned_rather_than_incremented(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ultralytics renames a busy directory; mosaic reads back from the composed one."""
    _install_ultralytics(monkeypatch)
    _ = runner.run_train_pose(_request(runner, tmp_path))

    assert _FakeYolo.last is not None
    assert _FakeYolo.last.kwargs["exist_ok"] is True


def test_a_resumed_run_carries_the_keyword_and_the_checkpoint(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mosaic finds the checkpoint; this program is only told what to do with it."""
    _install_ultralytics(monkeypatch)
    checkpoint = str(tmp_path / "run" / "train" / "weights" / "last.pt")
    _ = runner.run_train_pose(_request(runner, tmp_path, model=checkpoint, resume=True))

    assert _FakeYolo.last is not None
    assert _FakeYolo.last.model == checkpoint
    assert _FakeYolo.last.kwargs["resume"] is True


def test_point_training_carries_the_fork_only_arguments(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``dor`` reaches a real keyword here, where on the inference path it reaches none."""
    _install_ultralytics(monkeypatch)
    request = runner.TrainPointsRequest(
        model="polo26n.yaml",
        data_yaml=str(tmp_path / "data.yaml"),
        epochs=2,
        imgsz=640,
        batch=16,
        device="cpu",
        patience=50,
        project_dir=str(tmp_path / "run"),
        run_name="train",
        resume=False,
        augment={},
        train_overrides={},
        cancel_sentinel=str(tmp_path / "cancel"),
        loc=7.5,
        loc_loss="hausdorff",
        dor=0.6,
    )
    response = runner.run_train_points(request)

    assert response.stop == "completed"
    assert _FakeYolo.last is not None
    kwargs = _FakeYolo.last.kwargs
    assert kwargs["task"] == "locate"
    assert kwargs["loc"] == 7.5
    assert kwargs["loc_loss"] == "hausdorff"
    assert kwargs["dor"] == 0.6


def test_the_heartbeat_speaks_while_an_epoch_runs(
    runner: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An epoch can outlast every window that supervises it, so silence is broken."""
    _install_ultralytics(monkeypatch)
    monkeypatch.setattr(runner, "_HEARTBEAT_SECONDS", 0.05)
    original = _FakeYolo.train

    def slow_train(self: _FakeYolo, **kwargs: object) -> None:
        import time

        time.sleep(0.3)
        original(self, **kwargs)

    monkeypatch.setattr(_FakeYolo, "train", slow_train)
    _ = runner.run_train_pose(_request(runner, tmp_path))

    events = _lines(capsys.readouterr().out)
    assert any(event["event"] == "heartbeat" for event in events)
    assert all(set(e) == {"event"} for e in events if e["event"] == "heartbeat"), (
        "a heartbeat carries no position: the only number available mid-epoch is "
        "a counter that would stall rather than wait"
    )


def test_a_probe_with_no_model_asks_about_the_environment_alone(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh training run has no local weights, and must not download to preflight."""
    loaded: list[str] = []
    _install_ultralytics(monkeypatch)

    def record_load(path: str) -> tuple[object | None, str]:
        loaded.append(path)
        return None, ""

    def importable(name: str) -> bool:
        return name == "ultralytics"

    def no_locate() -> bool:
        return False

    def no_tracker_table(root: Path, tracker: str) -> dict[str, bool]:
        return {}

    monkeypatch.setattr(runner, "_load_model", record_load)
    monkeypatch.setattr(runner, "_is_importable", importable)
    monkeypatch.setattr(runner, "_has_locate", no_locate)
    monkeypatch.setattr(runner, "_installed_tracker_table", no_tracker_table)
    ultralytics = sys.modules["ultralytics"]
    setattr(ultralytics, "__version__", "8.4.63")
    ultralytics.__file__ = "/nowhere/ultralytics/__init__.py"
    trackers = ModuleType("ultralytics.trackers.track")
    setattr(trackers, "TRACKER_MAP", {})
    monkeypatch.setitem(sys.modules, "ultralytics.trackers", ModuleType("x"))
    monkeypatch.setitem(sys.modules, "ultralytics.trackers.track", trackers)

    response = runner.run_probe(runner.ProbeRequest(model_path=""))

    assert loaded == [], "nothing was loaded, so nothing could be downloaded"
    assert response.model_task == ""
    assert response.n_keypoints == 0
    assert response.model_load_error == ""
    assert response.has_ultralytics is True


def test_point_training_stops_at_an_epoch_boundary_too(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both subcommands share the callback, so both share the guarantee."""
    _install_ultralytics(monkeypatch)
    sentinel = tmp_path / "cancel"
    original = _FakeYolo.train

    def train_and_cancel_midway(self: _FakeYolo, **kwargs: object) -> None:
        def touch_after_first(trainer: _Trainer) -> None:
            if trainer.epoch == 0:
                _ = sentinel.write_text("cancelled")

        self.callbacks.setdefault("on_train_epoch_end", []).insert(0, touch_after_first)
        original(self, **kwargs)

    monkeypatch.setattr(_FakeYolo, "train", train_and_cancel_midway)
    request = runner.TrainPointsRequest(
        model="polo26n.yaml",
        data_yaml=str(tmp_path / "data.yaml"),
        epochs=TOTAL_EPOCHS,
        imgsz=640,
        batch=16,
        device="cpu",
        patience=50,
        project_dir=str(tmp_path / "run"),
        run_name="train",
        resume=False,
        augment={},
        train_overrides={},
        cancel_sentinel=str(sentinel),
        loc=5.0,
        loc_loss="mse",
        dor=0.8,
    )
    response = runner.run_train_points(request)

    assert response.stop == "cancelled"
    assert response.epochs_completed == 1
    assert (tmp_path / "run" / "train" / "weights" / "last.pt").is_file()
