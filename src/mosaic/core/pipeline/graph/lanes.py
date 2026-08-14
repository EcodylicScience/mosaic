"""Which queue a step is offered to, decided from what the step declares.

A **lane is a queue name, not a reservation.** Each maps to a *resource class*
whose capacity is a semaphore ceiling on a chosen bottleneck, not a pinned
allocation, and two lanes may share one class deliberately -- training and
inference share the GPU so a single card is not double-booked. An idle lane
therefore costs nothing, which is what makes a lane per kind of work reasonable
rather than wasteful.

**This lives in mosaic, and it did not before.** The rule was in mosaic-api,
derived from mosaic's own op registry, while ``plan_pipeline`` has to put a lane
on every step it plans. mosaic cannot import ``mosaic_queue`` -- that package
already depends on this one, so the reverse is a cycle -- so for planning to
assign a lane the rule has to be here. The lane *vocabulary* stays owned by
mosaic-queue, which is why the return is a plain string rather than a closed
type: constraining it belongs upstream, beside the queues it names.

Decided from a :class:`~...compatibility.Declaration` rather than from the
registries, so this is one more read path that does not pay for importing the
feature library.
"""

from __future__ import annotations

from typing import Final

from .compatibility import Declaration

__all__ = [
    "DEFAULT_LANE",
    "GPU_INFER_LANE",
    "GPU_TRAIN_LANE",
    "lane_for",
    "resource_class_of",
]

DEFAULT_LANE: Final = "feature-compute"
"""Where anything that is not GPU work is offered."""

GPU_TRAIN_LANE: Final = "gpu-train"
"""Training, pooled separately from inference so fair-share can weigh them apart."""

GPU_INFER_LANE: Final = "gpu-infer"
"""Inference and anything else wanting a GPU. Shares the ``gpu`` class with training."""


def resource_class_of(declared: Declaration) -> str:
    """The bottleneck *declared* contends for.

    Read straight off the declaration, which read it off what the feature or op
    itself says. A new heavy step routes correctly by declaring
    ``resource_class``; nothing here needs editing, and there is no per-step name
    list to keep current.
    """
    return declared.resource_class or "cpu"


def lane_for(declared: Declaration) -> str:
    """Which lane *declared*'s work is offered to.

    GPU work splits by category and everything else does not, which is the whole
    rule. The split exists so a fair-share pool can weigh a training job against
    an inference job rather than treating a card as one undifferentiated queue;
    both still map to the one ``gpu`` resource class, so the card is not
    double-booked.
    """
    if resource_class_of(declared) != "gpu":
        return DEFAULT_LANE
    if declared.produces.kind == "op" and declared.category == "train":
        return GPU_TRAIN_LANE
    return GPU_INFER_LANE
