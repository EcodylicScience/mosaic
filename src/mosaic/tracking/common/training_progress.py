"""Reading a training tool's epochs off its output, and reporting them.

Two ways a tool tells mosaic an epoch ended, and one way mosaic writes it down.

**The reporting shape is one thing.** :func:`report_epoch` is it, and both
readers call it, because the pairing it performs is a trap rather than a detail:
``ctx.progress.on_epoch_end`` and ``ctx.heartbeat`` both land in the run-log and
``reduce_run_log`` takes ``progress_done`` from whichever spoke last, so an
epoch reported without the count is undone by the next liveness heartbeat.

**The reading is per tool.** The Ultralytics runner is mosaic's own program and
writes a JSON line, read in
:mod:`~mosaic.tracking.common.ultralytics_env`. sleap-nn and Lightning Pose are
other people's programs and write a PyTorch Lightning progress bar, read here.
The bar is what those two have in common: both drive a Lightning ``Trainer``,
so both print the same ``TQDMProgressBar``, and one reader serves them.

Measured on a real run of each (sleap-nn 0.3 and Lightning Pose, 3 epochs,
NVIDIA RTX 4000 Ada). Three findings shape what follows:

- **The bar is on standard output**, with the surrounding Lightning and Hydra
  logging on standard error. So the reader belongs on ``run_supervised``'s
  *on_output*, while liveness belongs on *on_activity*, which takes both.
- **tqdm redraws with a carriage return**, which the subprocess reader's text
  mode turns into a line ending -- which is why there is a line to read at all,
  around 20 of them a second.
- **The bar is not monotonic within an epoch.** The last line carrying
  ``Epoch 0`` is the bar already reset to ``0/2761`` for the epoch after it, so
  what advances is the epoch *number*, never the fraction.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from mosaic.core.pipeline.job import JobContext

__all__ = [
    "LightningEpoch",
    "epoch_reporter",
    "lightning_epoch",
    "report_epoch",
]


_BAR: Final = re.compile(r"^Epoch (\d+):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)")
"""One redraw of a PyTorch Lightning progress bar.

``Epoch 2: 100%|##########| 2761/2761 [04:02<00:00, 11.38it/s, loss=9.95e-05]``

Anchored, so the model-architecture dump both tools print before training --
thousands of lines of it -- is rejected on the first character.
"""

_METRIC: Final = re.compile(r"([A-Za-z_][\w./]*)=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
"""One ``key=value`` in the bar's trailing postfix.

The keys are the tool's own and differ between them -- ``loss`` and ``val/loss``
from sleap-nn, ``train_supervised_loss`` and ``v_num`` from Lightning Pose -- so
they are read rather than named. A value Lightning renders as ``nan`` or ``inf``
does not match and is dropped, which is the honest reading: the callback takes
floats, and a metric that is not one says nothing about the epoch.
"""


@dataclass(frozen=True, slots=True)
class LightningEpoch:
    """One redraw of the bar: where a trainer says it is.

    Attributes:
        epoch: The epoch being *worked on*, zero-based and the trainer's own, so
            a resumed run's first is not zero.
        done: Batches finished within it.
        total: Batches it holds.
        metrics: What the bar's postfix carried, which is the trainer's running
            state at that redraw rather than a validated final figure. A metric
            validation has not refreshed yet is the previous epoch's.
    """

    epoch: int
    done: int
    total: int
    metrics: dict[str, float]


def lightning_epoch(line: str) -> LightningEpoch | None:
    """The bar redraw *line* carries, or ``None`` for anything else.

    Tolerant by construction: this runs on the subprocess reader thread, where a
    line is whatever the child wrote -- a torn redraw, a warning another library
    put on standard output, a bar a future Lightning draws differently -- and
    none of those is worth raising over where raising would be swallowed.
    """
    match = _BAR.match(line)
    if match is None:
        return None
    # Only past the bar itself, so the rate and the clock cannot be mistaken for
    # metrics. Neither carries an `=`, but the postfix is where metrics live and
    # reading elsewhere would only widen what can go wrong.
    postfix = line[match.end() :]
    metrics = {key: float(value) for key, value in _METRIC.findall(postfix)}
    return LightningEpoch(
        epoch=int(match[1]), done=int(match[2]), total=int(match[3]), metrics=metrics
    )


def report_epoch(
    ctx: JobContext, epoch: int, total_epochs: int, metrics: dict[str, float]
) -> None:
    """Record that *epoch* ended, as both a position and a liveness count.

    The two calls are one act. ``ctx.heartbeat(epoch + 1)`` is not decoration:
    ``phase_activity`` calls ``ctx.heartbeat()`` with no argument, and the
    run-log reduction takes ``progress_done`` from whichever of the two spoke
    last -- so without the count set here, the next liveness heartbeat after an
    epoch resets the reduced progress to zero.
    """
    ctx.progress.on_epoch_end(epoch, total_epochs, metrics)
    ctx.heartbeat(epoch + 1)


def epoch_reporter(ctx: JobContext, total_epochs: int) -> Callable[[str], None]:
    """Report each epoch a Lightning trainer finishes, read off its progress bar.

    Deliberately **not** throttled. It parses every line, of which there are
    around twenty a second, but reports only at an epoch boundary, of which
    there are as many as the run has epochs.

    An epoch has ended when either of two things is seen, and both are needed:

    - **its bar reached its own total.** This is what catches the *last* epoch,
      which nothing follows. Lightning draws that line four or five times, so
      the report is held to the first.
    - **a higher epoch number appeared.** A bar whose final redraw stopped short
      -- which is what a run killed or logged at an unlucky moment leaves --
      still ended when the next one began.

    Reporting is monotonic and once per epoch, so the two rules cannot
    double-count and an epoch skipped by one is caught by the other.

    Args:
        ctx: The job whose run-log the epochs land in.
        total_epochs: The denominator, which the bar does not carry -- it counts
            batches within an epoch, not epochs within a run -- so it comes from
            the parameters the run was submitted with.

    Returns:
        A callback for ``run_supervised``'s *on_output*. Liveness is a separate
        question with a separate answer: pass
        :func:`~mosaic.tracking.common.entry.phase_activity` as *on_activity*.
    """
    reported = [-1]

    def on_line(line: str) -> None:
        event = lightning_epoch(line)
        if event is None:
            return
        if event.total > 0 and event.done == event.total:
            ended = event.epoch
        elif event.epoch >= 1:
            ended = event.epoch - 1
        else:
            return
        if ended <= reported[0]:
            return
        reported[0] = ended
        report_epoch(ctx, ended, total_epochs, event.metrics)

    return on_line
