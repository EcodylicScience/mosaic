"""The label controls that the runs upstream of a reader were given.

A step that scores or tunes on ground-truth labels compares the truth classes
of its rows with the classes its input predicts. A ``remap`` or ``mask`` given
to the classifier and not to this step leaves truth classes the classifier never
trained on, and the step refuses them, but only where controls are in play:
without them, such a class is scored as a miss. Steps such as ``xgboost``,
``temporal-stack`` or ``smooth-probabilities`` sit between a classifier and its
reader with no labels of their own, so the reader looks at every run upstream
of its inputs.

A run carries controls when a top-level value of its recorded params is a
mapping holding a non-null ``remap`` or ``mask``. The read needs no feature
class, so ``core`` reads it without importing ``behavior``. A run whose
``params.json`` is absent or unreadable carries none.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from mosaic.core.pipeline.inventory.params import (
    RunKey,
    RunParams,
    features_roots,
    run_key_of,
    upstream_runs,
)
from mosaic.core.pipeline.label_controls import LabelControls

__all__ = ["lineage_label_controls"]


class _RecordedControls(BaseModel):
    """The ``remap`` and ``mask`` of one recorded params value, when it holds them.

    A value of another shape validates to neither, so any mapping is read
    without knowing which model wrote it.
    """

    model_config = ConfigDict(extra="ignore")

    remap: dict[int, int] | None = None
    mask: list[int] | None = None


def _run_controls(params: RunParams) -> tuple[set[tuple[int, int]], set[int]]:
    """The controls on one run's top-level params values."""
    remap: set[tuple[int, int]] = set()
    mask: set[int] = set()
    for value in params.params.values():
        if not isinstance(value, Mapping):
            continue
        try:
            recorded = _RecordedControls.model_validate(value)
        except ValidationError:
            continue
        remap |= set((recorded.remap or {}).items())
        mask |= set(recorded.mask or ())
    return remap, mask


def lineage_label_controls(input_run_roots: Iterable[Path]) -> LabelControls:
    """The union of ``remap`` and ``mask`` over the input runs and their upstreams.

    Reads each run's ``params.json`` once, walking with
    :func:`~mosaic.core.pipeline.inventory.params.upstream_runs` from the runs at
    *input_run_roots*, those runs included.

    Args:
        input_run_roots: The run directories of a reader's inputs, each
            ``<features>/<storage>/<run_id>``, all in one dataset.
    """
    roots = list(input_run_roots)
    if not roots:
        return LabelControls()
    starts: list[RunKey] = [run_key_of(root) for root in roots]
    remap: set[tuple[int, int]] = set()
    mask: set[int] = set()
    for _, params in upstream_runs(features_roots(roots[0]), starts):
        if params is None:
            continue
        run_remap, run_mask = _run_controls(params)
        remap |= run_remap
        mask |= run_mask
    return LabelControls(remap=frozenset(remap), mask=frozenset(mask))
