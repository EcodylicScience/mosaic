"""Training of a sequence network: the batches, the loop and the determinism around it.

A network trains on random crops of the runs of consecutive labeled frames, and
predicts a whole series in one pass. The batches come from a numpy generator, so
which frames a batch holds does not depend on the device. The network's own
randomness (its initial weights and its dropout) comes from torch's generator,
which :func:`deterministic_torch` seeds and the training function reseeds for each
model.

The order of the random draws is part of what a run id names. Changing it, the loop
or the standardizer changes the numbers a seed gives, and bumps the version of the
feature.

This module imports no torch. The functions that need it import it when called.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol

import numpy as np

from .classifier_common import balanced_class_weights

if TYPE_CHECKING:
    import numpy.typing as npt
    from torch import nn

__all__ = [
    "Sampler",
    "TrainingRun",
    "class_weights",
    "deterministic_torch",
    "network_state",
    "predict_probabilities",
    "seed_torch",
    "sub_seed",
    "train_network",
]

CUBLAS_WORKSPACE: Final = ":4096:8"
"""The setting that makes cuBLAS deterministic, which torch's deterministic mode demands."""

CHECK_EVERY: Final = 50
"""Steps between two looks at the cancel token."""


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """A stretch of consecutive labeled frames of one series.

    Attributes:
        entry: The ``(group, sequence)`` it comes from.
        x: The frames, one row each, as they are in the table.
        y: The class index of each frame, in the order of the model's classes.
        first_frame: The ``frame`` of its first row, which the rows then follow
            one by one. Set only where a fit cuts runs by frame, for
            cross-validation by block.
    """

    entry: tuple[str, str]
    x: npt.NDArray[np.float32]
    y: npt.NDArray[np.intp]
    first_frame: int | None = None


def sub_seed(seed: int, role: str, index: int) -> int:
    """Return a seed for one model of a run, from the run's seed.

    The full model of a run uses the run's seed itself. A fold model uses
    ``sub_seed(seed, "fold", k)``, so that no two models of two runs of
    neighboring seeds share one.
    """
    digest = hashlib.sha256(json.dumps([seed, role, index]).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def class_weights(
    counts: npt.NDArray[np.intp], how: Literal["sqrt", "balanced", "none"]
) -> npt.NDArray[np.float64]:
    """Return the weight of each class in the loss, from its frames in the training runs.

    ``balanced`` weights a class by the inverse of its frequency, ``sqrt`` by the
    square root of that, and both are scaled to a mean of 1 over the classes the
    training runs hold. A class with no frame has weight 0: it is never a target.

    Args:
        counts: The training frames of each class.
        how: The weighting.
    """
    present = counts > 0
    if how == "none":
        return present.astype(np.float64)
    weights = balanced_class_weights(counts)
    if how == "sqrt":
        weights = np.sqrt(weights)
    return weights / weights[present].mean()


@dataclass(frozen=True, slots=True)
class Sampler:
    """Draws batches of crops from runs laid end to end in one array.

    A batch is ``batch_size`` crops of the runs, each run drawn with a probability
    proportional to its length and with replacement, each crop starting uniformly
    inside its run. The crops are cut to the shortest crop of the batch.

    Attributes:
        offsets: Where each run starts in the array.
        lengths: The frames of each run.
        crop_length: The longest crop.
        batch_size: The crops in a batch.
    """

    offsets: npt.NDArray[np.intp]
    lengths: npt.NDArray[np.intp]
    crop_length: int
    batch_size: int

    @classmethod
    def of(
        cls, lengths: npt.NDArray[np.intp], crop_length: int, batch_size: int
    ) -> Sampler:
        """Return the sampler of runs of these *lengths*, laid end to end in this order."""
        ends = np.cumsum(lengths)
        return cls(
            offsets=(ends - lengths).astype(np.intp),
            lengths=lengths,
            crop_length=crop_length,
            batch_size=batch_size,
        )

    def draw(
        self, rng: np.random.Generator, probabilities: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.intp], int]:
        """Return the first frame of each crop of a batch and the length they share.

        The order of the draws is fixed: one call chooses every run of the batch,
        then one call per crop, in the order chosen, places its start.
        """
        chosen = rng.choice(len(self.lengths), self.batch_size, p=probabilities)
        starts = np.empty(self.batch_size, dtype=np.intp)
        shortest = self.crop_length
        for slot, run in enumerate(chosen):
            run_length = int(self.lengths[run])
            crop = min(self.crop_length, run_length)
            shortest = min(shortest, crop)
            starts[slot] = int(self.offsets[run]) + int(
                rng.integers(0, run_length - crop + 1)
            )
        return starts, shortest

    def probabilities(self) -> npt.NDArray[np.float64]:
        """Return the chance of each run, proportional to its length."""
        return self.lengths / self.lengths.sum()


@contextlib.contextmanager
def deterministic_torch(*, device: str, n_threads: int) -> Generator[None]:
    """Run a block under the settings that make a fit repeat, then restore them.

    Torch's deterministic algorithms are on, with a warning and not an error where
    an operation has no deterministic form, cuDNN is deterministic with its
    benchmark off, and the CPU runs ``n_threads`` threads. On CUDA the
    cuBLAS workspace variable is set before the first CUDA call of the process. A
    process that already used CUDA warns and goes on. The results are equal on the
    CPU at one thread count. They are not promised equal across thread counts.
    """
    import warnings

    import torch

    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cudnn = (
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    previous_threads = torch.get_num_threads()
    set_workspace = False
    if device == "cuda":
        if torch.cuda.is_initialized() and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            warnings.warn(
                "CUDA was initialized before the fit, so cuBLAS may not be "
                "deterministic. Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before the "
                "process starts",
                stacklevel=2,
            )
        elif "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE
            set_workspace = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(n_threads)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(
            previous_algorithms, warn_only=previous_warn_only
        )
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = (
            previous_cudnn
        )
        torch.set_num_threads(previous_threads)
        if set_workspace:
            _ = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


class _Loss(Protocol):
    """The part of a loss tensor that the loop calls."""

    def backward(self) -> None: ...


class _Optimizer(Protocol):
    """The part of an optimizer that the loop calls."""

    def step(self) -> None: ...


def backpropagate(loss: _Loss) -> None:
    """Differentiate *loss*, through a signature that the type checker can read."""
    loss.backward()


def advance(optimizer: _Optimizer) -> None:
    """Take an optimizer step, through a signature that the type checker can read."""
    optimizer.step()


def seed_torch(seed: int, *, device: str) -> None:
    """Seed torch's generators for the CPU and, on ``cuda``, for every GPU."""
    import torch

    _ = torch.default_generator.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def train_network(
    network: nn.Module,
    frames: npt.NDArray[np.float32],
    classes: npt.NDArray[np.intp],
    sampler: Sampler,
    weights: npt.NDArray[np.float64],
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    pct_start: float,
    grad_clip: float,
    seed: int,
    device: str,
    check: Callable[[], None],
) -> None:
    """Train *network* in place on random crops of *frames*.

    The loop is AdamW under a one-cycle schedule, cross-entropy weighted by
    *weights*, gradients clipped to *grad_clip*, and the weights of the last step
    kept. The generator of the batches is seeded with *seed*. Torch's generator is
    seeded by the caller before it builds the network, so that the initial weights
    and the dropout draw from one stream, in that order.

    Args:
        network: The module, still in its initial state, on the CPU.
        frames: The standardized frames of all runs, laid end to end.
        classes: The class index of each frame.
        sampler: Where the runs are, and how a batch is cut.
        weights: The weight of each class in the loss.
        steps: Optimizer steps.
        learning_rate: The peak of the schedule.
        weight_decay: The optimizer's weight decay.
        pct_start: The share of the steps that the schedule climbs for.
        grad_clip: The norm that gradients are clipped to.
        seed: Seeds the batches.
        device: Where to train.
        check: Called every :data:`CHECK_EVERY` steps, and raises to stop.
    """
    import torch
    import torch.nn.functional as functional

    rng = np.random.default_rng(seed)
    network.to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=steps, pct_start=pct_start
    )
    class_weight = torch.as_tensor(weights.astype(np.float32)).to(device)
    frame_tensor = torch.as_tensor(frames).to(device)
    class_tensor = torch.as_tensor(classes.astype(np.int64)).to(device)
    probabilities = sampler.probabilities()
    n_classes = int(class_weight.shape[0])
    network.train()
    for step in range(steps):
        if step % CHECK_EVERY == 0:
            check()
        starts, length = sampler.draw(rng, probabilities)
        batch = torch.stack([frame_tensor[s : s + length] for s in starts.tolist()])
        target = torch.stack([class_tensor[s : s + length] for s in starts.tolist()])
        loss = functional.cross_entropy(
            network(batch).reshape(-1, n_classes),
            target.reshape(-1),
            weight=class_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        backpropagate(loss)
        _ = torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
        advance(optimizer)
        schedule.step()


def network_state(network: nn.Module) -> dict[str, npt.NDArray[np.float32]]:
    """Return the parameters and buffers of *network* as numpy arrays on the CPU."""
    return {
        name: value.detach().cpu().numpy()
        for name, value in network.state_dict().items()
    }


def predict_probabilities(
    network: nn.Module,
    frames: npt.NDArray[np.float32],
    *,
    device: str,
) -> npt.NDArray[np.float64]:
    """Return the class probabilities of every frame of one stretch, from one pass.

    Args:
        network: A module in evaluation mode, on *device*.
        frames: One row per frame, standardized, of consecutive frames.
        device: Where *network* is.
    """
    import torch

    with torch.no_grad():
        logits = network(torch.as_tensor(frames).to(device)[None])
        probabilities = torch.softmax(logits[0], dim=-1)
    return probabilities.cpu().numpy().astype(np.float64)
