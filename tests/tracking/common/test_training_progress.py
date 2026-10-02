"""Reading epochs off a PyTorch Lightning progress bar.

Every line quoted here is **verbatim from a real run** -- sleap-nn and Lightning
Pose, three epochs each, on a CUDA box -- rather than written to suit the parser.
That is the whole point of the fixtures: the format is another project's, mosaic
does not control it, and a parser tested against lines invented for it proves
only that it is self-consistent.

What the capture settled, and what these hold:

- both tools print the bar on **standard output**, with Lightning's and Hydra's
  own logging on standard error;
- the two bars are the same shape, because both drive a Lightning ``Trainer``,
  so one reader serves both;
- the bar is **not monotonic within an epoch** -- the last line carrying
  ``Epoch 0`` is the bar already reset for the epoch after it -- so what advances
  is the epoch number.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from mosaic.core.pipeline.job import CancelToken, JobContext
from mosaic.core.pipeline.progress import NullProgressCallback
from mosaic.tracking.common.training_progress import (
    epoch_reporter,
    lightning_epoch,
    report_epoch,
)

# --- captured output -------------------------------------------------------

SLEAP_FULL = [
    "Epoch 0: 100%|##########| 2761/2761 [04:19<00:00, 10.64it/s, loss=2.84e-04]",
    "Epoch 1: 100%|##########| 2761/2761 [04:06<00:00, 11.19it/s, loss=2.58e-04, "
    "val/loss=3.03e-04]",
    "Epoch 2: 100%|##########| 2761/2761 [04:02<00:00, 11.38it/s, loss=9.95e-05, "
    "val/loss=1.78e-04]",
]
"""sleap-nn, the redraw in which each epoch's bar reached its own total."""

SLEAP_PARTIAL = "Epoch 0:   0%|          | 1/2761 [00:00<10:03,  4.58it/s, loss=0.0012]"
"""sleap-nn, mid-epoch. The overwhelming majority of the lines look like this."""

SLEAP_TRANSITION = (
    "Epoch 1:   0%|          | 0/2761 [00:00<?, ?it/s, loss=2.51e-04, "
    "val/loss=3.03e-04]"
)
"""sleap-nn, the bar reset for epoch 1 -- which is proof epoch 0 ended."""

LITPOSE_FULL = [
    "Epoch 0: 100%|##########| 614/614 [02:19<00:00,  4.41it/s, v_num=0, "
    "train_supervised_loss=0.021, train_heatmap_mse_loss=0.042]",
    "Epoch 1: 100%|##########| 614/614 [02:17<00:00,  4.46it/s, v_num=0, "
    "train_supervised_loss=0.0205, train_heatmap_mse_loss=0.0411, "
    "val_supervised_loss=0.0199, val_heatmap_mse_loss=0.0398]",
    "Epoch 2: 100%|##########| 614/614 [02:06<00:00,  4.84it/s, v_num=0, "
    "train_supervised_loss=0.0118, train_heatmap_mse_loss=0.0235, "
    "val_supervised_loss=0.0199, val_heatmap_mse_loss=0.0398]",
]
"""Lightning Pose, the same redraw. Different metric names, identical shape."""

NOT_THE_BAR = [
    "INFO: GPU available: True (cuda), used: True",
    "INFO: `Trainer.fit` stopped: `max_epochs=3` reached.",
    "            (stack0_enc0_conv0): Conv2d(3, 32, kernel_size=(3, 3), "
    "stride=(1, 1), padding=same)",
    "Note: 'sleap-nn-train' is a legacy command. Consider using 'sleap train'.",
    "",
]
"""Also captured. The architecture dump alone is thousands of lines a run."""


def _context() -> tuple[JobContext, list[tuple[int, int, dict[str, float]]]]:
    """A job whose reported epochs are collected rather than written down."""
    reported: list[tuple[int, int, dict[str, float]]] = []

    class Recorder(NullProgressCallback):
        def on_epoch_end(
            self, epoch: int, total_epochs: int, metrics: dict[str, float]
        ) -> None:
            reported.append((epoch, total_epochs, dict(metrics)))

    ctx = JobContext(
        execution_id="01JQ0000000000000000000000",
        kind="train-sleap",
        target="train-sleap",
        run_log=None,
        progress=Recorder(),
        cancel_token=CancelToken(),
    )
    return ctx, reported


# --- the reader ------------------------------------------------------------


@pytest.mark.parametrize("line", NOT_THE_BAR)
def test_what_is_not_a_bar_is_answered_none(line: str) -> None:
    """It runs on the reader thread, where a raise reaches nobody."""
    assert lightning_epoch(line) is None


def test_a_full_bar_reports_where_the_trainer_is() -> None:
    event = lightning_epoch(SLEAP_FULL[2])
    assert event is not None
    assert (event.epoch, event.done, event.total) == (2, 2761, 2761)


def test_the_metrics_are_read_rather_than_named() -> None:
    """The keys are the tool's own, and the two tools do not share one.

    A parser naming them would report Lightning Pose's epochs with no metrics at
    all, and would go silent on any trainer that renamed a loss.
    """
    sleap = lightning_epoch(SLEAP_FULL[1])
    litpose = lightning_epoch(LITPOSE_FULL[1])
    assert sleap is not None and litpose is not None
    assert sleap.metrics == {"loss": 2.58e-04, "val/loss": 3.03e-04}
    assert litpose.metrics == {
        "v_num": 0.0,
        "train_supervised_loss": 0.0205,
        "train_heatmap_mse_loss": 0.0411,
        "val_supervised_loss": 0.0199,
        "val_heatmap_mse_loss": 0.0398,
    }


def test_the_rate_and_the_clock_are_not_mistaken_for_metrics() -> None:
    """``4.41it/s`` and ``[02:19<00:00`` sit in every line a metric does."""
    event = lightning_epoch(LITPOSE_FULL[0])
    assert event is not None
    assert set(event.metrics) == {
        "v_num",
        "train_supervised_loss",
        "train_heatmap_mse_loss",
    }


# --- what counts as an epoch ending ----------------------------------------


@pytest.mark.parametrize(
    ("tool", "lines"), [("sleap", SLEAP_FULL), ("litpose", LITPOSE_FULL)]
)
def test_every_epoch_is_reported_once_including_the_last(
    tool: str, lines: list[str]
) -> None:
    """The last epoch is the reason the full bar is read at all.

    Nothing follows it, so a reader that only noticed the epoch number advancing
    would finish a three-epoch run reporting two.
    """
    ctx, reported = _context()
    on_line = epoch_reporter(ctx, 3)
    for line in lines:
        on_line(line)

    assert [epoch for epoch, _, _ in reported] == [0, 1, 2]
    assert all(total == 3 for _, total, _ in reported), (
        "the denominator comes from the parameters; the bar counts batches"
    )


def test_lightning_redraws_the_full_bar_and_it_is_reported_once() -> None:
    """Measured at four redraws per epoch, five on the last."""
    ctx, reported = _context()
    on_line = epoch_reporter(ctx, 3)
    for _ in range(5):
        on_line(SLEAP_FULL[0])

    assert [epoch for epoch, _, _ in reported] == [0]


def test_a_bar_that_stopped_short_still_ended_when_the_next_began() -> None:
    """The other half, for an epoch whose final redraw was never seen.

    A run logged at an unlucky moment, or one whose bar a future Lightning draws
    differently, still advances its epoch number.
    """
    ctx, reported = _context()
    on_line = epoch_reporter(ctx, 3)
    on_line(SLEAP_PARTIAL)
    assert reported == [], "an epoch in progress has not ended"

    on_line(SLEAP_TRANSITION)
    assert [epoch for epoch, _, _ in reported] == [0]
    assert reported[0][2] == {"loss": 2.51e-04, "val/loss": 3.03e-04}, (
        "the reset bar carries the metrics of the epoch that just finished"
    )


def test_the_two_rules_cannot_double_count_one_epoch() -> None:
    """Both fire for epoch 0 here, and it is reported once."""
    ctx, reported = _context()
    on_line = epoch_reporter(ctx, 3)
    on_line(SLEAP_FULL[0])
    on_line(SLEAP_TRANSITION)

    assert [epoch for epoch, _, _ in reported] == [0]


def test_reporting_never_goes_backwards() -> None:
    """The bar is not monotonic within an epoch, so the reader has to be.

    The last line carrying ``Epoch 0`` is the bar already reset to ``0/2761``
    for epoch 1, which arrives *after* epoch 1 has been reported on a resumed
    or reordered read.
    """
    ctx, reported = _context()
    on_line = epoch_reporter(ctx, 3)
    for line in (*SLEAP_FULL, SLEAP_PARTIAL, SLEAP_TRANSITION, SLEAP_FULL[0]):
        on_line(line)

    assert [epoch for epoch, _, _ in reported] == [0, 1, 2]


# --- how an epoch is written down ------------------------------------------


def test_an_epoch_sets_the_liveness_count_as_well_as_the_position() -> None:
    """The trap :func:`report_epoch` exists to hold in one place.

    ``phase_activity`` heartbeats with no argument, and the run-log reduction
    takes ``progress_done`` from whichever spoke last -- so an epoch that set
    only the position would be undone by the next liveness heartbeat.
    """
    ctx, _ = _context()
    report_epoch(ctx, 4, 10, {"loss": 0.5})

    assert ctx.done == 5, "the count is the epochs finished, not the index"
    ctx.heartbeat()
    assert ctx.done == 5, "a bare liveness heartbeat must not reset it"


# --- the op actually wires it ----------------------------------------------


def test_train_sleap_reports_the_epochs_its_trainer_prints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The half a structural check cannot make: that the reporter is connected.

    ``tests/test_progress_reporting.py`` holds that the op passes *a* reporting
    argument. This holds that what it passes turns the trainer's real output
    into a moving numerator, read back out of the run-log -- which is the only
    channel a queued job has, since it is spawned with both streams on DEVNULL.
    """
    from mosaic.core.pipeline.models import model_run_root
    from mosaic.core.pipeline.ops import run_op
    from mosaic.runlog import reduce_run_log, run_log_dir
    from mosaic.tracking import register_ops
    from mosaic.tracking.sleap import probe as probe_module
    from mosaic.tracking.sleap import training as training_module

    from tests.helpers import make_dataset

    register_ops()
    ds = make_dataset(tmp_path, save=False)
    (tmp_path / "bin").mkdir(exist_ok=True)
    for name in ("sleap-nn-train", "python"):
        _ = (tmp_path / "bin" / name).write_text("")
    monkeypatch.setenv("MOSAIC_SLEAP_BIN", str(tmp_path / "bin" / "sleap-nn-train"))

    def probe(argv: Sequence[str], **kw: object) -> tuple[str, str, int]:
        answer: dict[str, object] = {"has_sleap_nn": True, "n_tracks": 2}
        _ = Path(argv[argv.index("-c") + 3]).write_text(json.dumps(answer))
        return ("probed", "", 0)

    monkeypatch.setattr(probe_module, "run_supervised", probe)

    def train(
        argv: Sequence[str],
        *,
        on_output: Callable[[str], None] | None = None,
        **kw: object,
    ) -> tuple[str, str, int]:
        run_root = Path(argv[argv.index("--config-dir") + 1])
        produced = run_root / "model"
        produced.mkdir(parents=True, exist_ok=True)
        _ = (produced / "best.ckpt").write_bytes(b"weights")
        _ = (produced / "training_config.yaml").write_text(
            "head_configs:\n  centroid: {}\n"
        )
        assert on_output is not None, "the op must read the trainer's output"
        for line in (SLEAP_PARTIAL, *SLEAP_FULL):
            on_output(line)
        return ("done", "", 0)

    monkeypatch.setattr(training_module, "run_supervised", train)

    labels = tmp_path / "session.slp"
    _ = labels.write_bytes(b"slp")
    run_id = run_op(
        ds, "train-sleap", {"labels": str(labels), "max_epochs": 3, "head": "centroid"}
    )
    assert model_run_root(ds, "train-sleap", run_id).is_dir()

    logs = sorted(run_log_dir(ds.base_dir).glob("*.jsonl"))
    assert len(logs) == 1
    snapshot = reduce_run_log(logs[0])
    assert snapshot is not None
    assert (snapshot["progress_done"], snapshot["progress_total"]) == (3, 3), (
        "every epoch the trainer printed reached the ledger"
    )
