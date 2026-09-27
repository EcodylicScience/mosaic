"""Which queue a step is offered to, decided from what the step declares.

A **lane is a queue name, not a reservation.** The queue maps each lane to a
*pool* of workers, and two lanes may share one pool deliberately -- training and
inference share the GPU workers so a single card is not double-booked. An idle
lane therefore costs nothing, which is what makes a lane per kind of work
reasonable rather than wasteful.

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
    "TRANSCODE_LANE",
    "lane_for",
    "lane_for_step",
    "resource_class_of",
]

DEFAULT_LANE: Final = "feature-compute"
"""Where anything that is not GPU work is offered."""

GPU_TRAIN_LANE: Final = "gpu-train"
"""Training, pooled separately from inference so fair-share can weigh them apart."""

GPU_INFER_LANE: Final = "gpu-infer"
"""Inference and anything else wanting a GPU. Shares the ``gpu`` class with training."""

TRANSCODE_LANE: Final = "transcode"
"""Media work bound by ffmpeg: transcodes and the joined and store exports.

Apart from ``feature-compute`` because the two want opposite sizing: one well-tuned
ffmpeg process uses a machine's cores and its disk better than several side by side,
while CPU features scale with the number of workers running them.
"""


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

    GPU work splits into training and everything else: an op of category
    ``train``, or a GPU feature of category ``global`` (which fits a model), is
    training. The split exists so the queue can weigh a training job against an
    inference job -- and keep a card for inference when it is configured to --
    rather than treating the GPUs as one undifferentiated queue. Media work bound
    by ffmpeg -- the ``transcode`` category, or a declared ``heavy`` class -- has a
    lane of its own. Everything else is ``feature-compute``.
    """
    resource_class = resource_class_of(declared)
    if resource_class == "gpu":
        fits = (
            declared.category == "train"
            if declared.produces.kind == "op"
            else declared.category == "global"
        )
        return GPU_TRAIN_LANE if fits else GPU_INFER_LANE
    if declared.category == "transcode" or resource_class == "heavy":
        return TRANSCODE_LANE
    return DEFAULT_LANE


def lane_for_step(name: str) -> str:
    """The lane a step named *name* -- a feature slug or an op kind -- is offered to.

    The by-name entry point, for a caller that holds a submitted job rather than a
    declaration. It reads the declaration catalog, and so pays the feature-library
    import the planning read paths are kept free of; that import is deferred into
    the call so nothing that only imports this module pays it.

    Raises:
        KeyError: no feature or op of that name is registered.
    """
    from .resolve import declaration_catalog

    declared = declaration_catalog().get(name)
    if declared is None:
        message = f"no feature or op named {name!r} is registered"
        raise KeyError(message)
    return lane_for(declared)
