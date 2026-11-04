"""Which runs a request may pin its steps to, and when a pinned run is finished.

A pin (``Request.pins``) names a run of a step's planned storage that the step
then serves instead of executing. Only a step that fits over its scope
(``fits`` and ``scope_dependent``) is pinned: a template step, or a fitter such
as ``arhmm`` without a ``model``. A scope-free fitter reads the pinned run and
would label no new entry, and a fitter given a ``model`` fits nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..index import feature_index, feature_index_path
from ..inventory.model import FeatureRunRef
from .model import COMPLETE_STATUSES, FeatureStepSpec
from .step_preflight import StepRefused

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mosaic.core.dataset import Dataset

    from ..inventory.model import AnyRecord, DatasetInventory
    from .model import Recipe
    from .plan import Plan

__all__ = ["pinned_run_complete", "refuse_unusable_pins"]


def pinned_run_complete(record: AnyRecord | None) -> bool:
    """Whether a pinned run is finished, read off an unscoped inventory's record.

    The record's status is measured against the entries the run's invocations
    asked for (:func:`~mosaic.core.pipeline.inventory.scan.feature_run_target`),
    so a run fit on other sequences than a request's still reads complete.
    """
    return (
        record is not None
        and bool(record.finished_at)
        and record.status in COMPLETE_STATUSES
    )


def refuse_unusable_pins(
    ds: Dataset,
    recipe: Recipe,
    plan: Plan,
    pins: Mapping[str, str],
    *,
    inventory: DatasetInventory,
) -> None:
    """Refuse the first pin that names no fitted step or no finished run.

    Args:
        ds: The planned dataset.
        recipe: The recipe the pins name steps of.
        plan: The plan resolved with the pins.
        pins: The run id of each pinned step, keyed by step id.
        inventory: An unscoped inventory of the feature runs.

    Raises:
        StepRefused: Named for the pin's step: ``pin_unknown_step`` for a key
            naming no feature step of *recipe*, ``pin_not_fitted`` for a step
            that does not fit over its scope, ``pin_run_missing`` for a run the
            step storage's index lacks, and ``pin_run_unfinished`` for a run
            that has not finished.
    """
    for step_id, run_id in sorted(pins.items()):
        if not any(
            isinstance(step, FeatureStepSpec) and step.id == step_id
            for step in recipe.steps
        ):
            raise StepRefused(
                "pin_unknown_step",
                step_id,
                f"pins names step {step_id!r}, which is not a feature step of "
                f"this recipe. Pin a feature step by its id.",
                {"run_id": run_id},
            )
        planned = plan.step(step_id)
        if not (planned.fits and planned.scope_dependent):
            raise StepRefused(
                "pin_not_fitted",
                step_id,
                f"step {step_id!r} does not fit over its scope, so a pin has "
                f"nothing to keep: a pin names a template step, or a fitter such "
                f"as arhmm without a model. Leave {step_id!r} unpinned, and it "
                f"reads the runs above it.",
                {"run_id": run_id, "feature": planned.runs},
            )
        storage = planned.storage_name
        try:
            _ = feature_index(feature_index_path(ds, storage)).read(run_id=run_id)
        except FileNotFoundError:
            raise StepRefused(
                "pin_run_missing",
                step_id,
                f"step {step_id!r} is pinned to run {run_id!r}, which the index of "
                f"{storage!r} does not hold. Pin a run that this step's storage "
                f"lists.",
                {"run_id": run_id, "storage": storage},
            ) from None
        if not pinned_run_complete(
            inventory.record(FeatureRunRef(name=storage, run_id=run_id))
        ):
            raise StepRefused(
                "pin_run_unfinished",
                step_id,
                f"step {step_id!r} is pinned to run {run_id!r} of {storage!r}, "
                f"which has not finished every entry it was asked for. Finish "
                f"the run, or pin one that has finished.",
                {"run_id": run_id, "storage": storage},
            )
