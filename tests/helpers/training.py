"""A stand-in for the training tool, for suites that drive a training op.

The trainer runs in an environment mosaic does not install, so every suite that
exercises ``train-pose`` or ``train-points`` replaces the two module-scope seams
that reach it: the probe that asks what the environment holds, and the launcher
that spawns it.

**The progress it reports is written as the runner writes it** -- one JSON line
per epoch, handed to the ``on_output`` callback the op composed -- rather than
called straight into ``ctx.progress``. That distinction is the whole value of the
fake: a stand-in that called the callback directly would satisfy every assertion
about the run-log while proving nothing about whether mosaic can read what the
tool actually says.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from mosaic.tracking.external.runner.ultralytics_protocol import (
    EpochEvent,
    ProbeResponse,
    TrainRequestBase,
    TrainStop,
)
from mosaic.tracking.pose_training.ultralytics_train import TrainingOutcome

__all__ = ["FakeTrainer", "healthy_probe"]


def healthy_probe(**overrides: object) -> ProbeResponse:
    """What a well-built environment reports, with the fields a case is about replaced."""
    fields: dict[str, object] = {
        "has_ultralytics": True,
        "has_lap": True,
        "has_locate": True,
        "ultralytics_version": "8.4.63",
        "tracker_names": ["botsort"],
        "model_task": "",
        "n_keypoints": 0,
        "model_load_error": "",
        "installed_tracker_table": {},
    }
    fields.update(overrides)
    return ProbeResponse.model_validate(fields)


@dataclass
class FakeTrainer:
    """Records what the op asked for, and produces what a real run leaves behind."""

    epochs_run: int | None = None
    """How many epochs to report, or ``None`` for however many were requested."""

    stop: TrainStop = "completed"
    save_dir_override: Path | None = None
    """Where to claim the model landed, for the case where that disagrees."""

    calls: int = 0
    requests: list[TrainRequestBase] = field(default_factory=list)
    probes: list[str] = field(default_factory=list)

    @property
    def last_request(self) -> TrainRequestBase:
        return self.requests[-1]

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace both seams: the environment probe, and the launcher."""
        import mosaic.tracking.common.ultralytics_env as tool_env
        import mosaic.tracking.pose_training.ultralytics_train as train_tool

        def fake_probe(model_path: str, **_kwargs: object) -> ProbeResponse:
            self.probes.append(model_path)
            return healthy_probe(
                model_task="" if not model_path else "pose",
                n_keypoints=0 if not model_path else 4,
            )

        monkeypatch.setattr(tool_env, "probe_environment", fake_probe)
        monkeypatch.setattr(train_tool, "run_pose_training_tool", self)
        monkeypatch.setattr(train_tool, "run_point_training_tool", self)

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
        del idle_timeout, cancel_check
        self.calls += 1
        self.requests.append(request)

        save_dir = Path(request.project_dir) / request.run_name
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        _ = (save_dir / "weights" / "best.pt").write_bytes(b"weights")
        _ = (save_dir / "weights" / "last.pt").write_bytes(b"weights")

        completed = request.epochs if self.epochs_run is None else self.epochs_run
        rows = ["epoch,loss"]
        for epoch in range(completed):
            rows.append(f"{epoch},0.1")
            if callable(on_output):
                on_output(
                    EpochEvent(
                        event="epoch",
                        epoch=epoch,
                        total_epochs=request.epochs,
                        metrics={"loss": 0.1},
                    ).model_dump_json()
                )
        _ = (save_dir / "results.csv").write_text("\n".join(rows) + "\n")

        return TrainingOutcome(
            save_dir=self.save_dir_override or save_dir,
            epochs_completed=completed,
            stop=self.stop,
        )
