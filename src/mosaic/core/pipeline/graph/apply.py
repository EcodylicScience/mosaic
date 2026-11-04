"""What an apply of a finished request keeps from it: its pins, its tracks, its locked steps.

An apply (``Request.applies_request_id``) names a finished request of the same
dataset and reuses its fit over other entries. Every step that fits, and every
step one reads, is **locked**: it must be the step the named request ran, or
the apply would read a fit learned from other inputs. Each locked step that
fits over its scope is pinned to the run its attempt recorded, and the apply
reads the tracks variant the named request read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mosaic.runlog import read_run, run_log_dir

from .model import FeatureStepSpec
from .request import load_recipe_for_request
from .resolve import declared_fit
from .rollup import request_rollup, request_verdict
from .step_preflight import StepRefused
from .store import RequestUnreadable, load_request
from .topo import ancestors_of, topological_order

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mosaic.core.dataset import Dataset
    from mosaic.core.json_value import JsonValue

    from .compatibility import DeclarationCatalog
    from .model import Recipe, Request, Step

__all__ = ["AppliedFit", "derive_applied_fit"]


@dataclass(frozen=True, slots=True)
class AppliedFit:
    """What an apply takes from the request it names.

    Attributes:
        request_id: The named request.
        pins: The run each locked step that fits over its scope serves, keyed
            by step id: the run its attempt in the named request recorded.
        locked: The steps that fit and the steps they read, in topological
            order.
        tracks_run_id: The tracks variant the named request read, ``None`` for
            each entry's one variant.
    """

    request_id: str
    pins: Mapping[str, str] = field(default_factory=lambda: dict[str, str]())
    locked: tuple[str, ...] = ()
    tracks_run_id: str | None = None


def derive_applied_fit(
    ds: Dataset,
    recipe: Recipe,
    applies_request_id: str,
    *,
    tracks_run_id: str | None,
    catalog: DeclarationCatalog,
) -> AppliedFit:
    """Derive an apply of *recipe* from the finished request it names.

    Args:
        ds: The dataset holding the named request.
        recipe: The recipe the apply runs.
        applies_request_id: The request whose fit the apply reuses.
        tracks_run_id: The tracks variant the apply names, or ``None`` to take
            the named request's.
        catalog: The declarations that say which steps fit.

    Returns:
        The pins, the locked steps and the tracks variant of the apply.

    Raises:
        StepRefused: ``apply_request_missing`` for a request the dataset does
            not hold or that does not read, ``apply_request_unfinished`` for
            one that has not finished, ``apply_step_changed``, named for the
            step, for a locked step that differs from the named request's
            recipe or that it lacks, and ``apply_tracks_changed`` for a tracks
            variant other than the named request's.
    """
    named = _named_request(ds, applies_request_id)
    verdict = request_verdict(request_rollup(ds.base_dir, named).steps)
    if verdict != "finished":
        raise StepRefused(
            "apply_request_unfinished",
            "",
            f"applies_request_id names request {applies_request_id!r}, which reads "
            f"{verdict!r}. An apply reuses a finished fit; finish that request, "
            f"or name one that has finished.",
            {"applies_request_id": applies_request_id, "status": verdict},
        )
    fitted = _fitting_steps(recipe, catalog)
    locked_ids = frozenset[str]().union(
        *(ancestors_of(recipe, step_id) for step_id in fitted)
    )
    locked = tuple(
        step.id for step in topological_order(recipe) if step.id in locked_ids
    )
    named_recipe = load_recipe_for_request(ds.base_dir, named)
    for step_id in locked:
        _refuse_changed(recipe.step(step_id), named_recipe, applies_request_id)
    if tracks_run_id is not None and tracks_run_id != named.tracks_run_id:
        raise StepRefused(
            "apply_tracks_changed",
            "",
            f"the apply reads tracks variant {tracks_run_id!r}, and request "
            f"{applies_request_id!r} fit on {named.tracks_run_id!r}. Its fit "
            f"learned from those tracks; leave the variant unset to read them.",
            {
                "applies_request_id": applies_request_id,
                "tracks_run_id": tracks_run_id,
                "fit_tracks_run_id": named.tracks_run_id,
            },
        )
    return AppliedFit(
        request_id=applies_request_id,
        pins=_recorded_pins(ds, named, locked, fitted),
        locked=locked,
        tracks_run_id=named.tracks_run_id,
    )


def _named_request(ds: Dataset, request_id: str) -> Request:
    """The request an apply names, or ``apply_request_missing``."""
    try:
        return load_request(ds.base_dir, request_id)
    except (FileNotFoundError, RequestUnreadable) as exc:
        raise StepRefused(
            "apply_request_missing",
            "",
            f"applies_request_id names request {request_id!r}, which this "
            f"dataset does not hold as a readable request: {exc}. Name a "
            f"finished request of this dataset.",
            {"applies_request_id": request_id},
        ) from exc


def _fitting_steps(recipe: Recipe, catalog: DeclarationCatalog) -> dict[str, bool]:
    """Whether each feature step that fits does so over its scope, by step id."""
    fitted: dict[str, bool] = {}
    for step in recipe.steps:
        if not isinstance(step, FeatureStepSpec):
            continue
        declared = catalog.get(step.feature)
        if declared is None:
            continue
        fits, scope_dependent = declared_fit(step.feature, step.params, declared)
        if fits:
            fitted[step.id] = scope_dependent
    return fitted


def _compared(step: Step) -> dict[str, JsonValue]:
    """What a locked step must keep: every field but its ``id`` and ``after``.

    What it runs, what it reads, its ``tracks`` reference, its params and its
    run arguments all reach what the fit learned. ``after`` orders and reaches
    nothing. The params are canonical at parse, so ``5`` and ``5.0`` compare
    equal, but they compare as written otherwise: a key restated at its
    default differs from the key left out.
    """
    return step.model_dump(mode="json", exclude={"id", "after"})


def _refuse_changed(step: Step, named_recipe: Recipe, applies_request_id: str) -> None:
    """Refuse a locked *step* that the named request's recipe ran otherwise."""
    try:
        before = named_recipe.step(step.id)
    except KeyError:
        before = None
    if before is not None and _compared(before) == _compared(step):
        return
    found = "lacks it" if before is None else "ran it otherwise"
    raise StepRefused(
        "apply_step_changed",
        step.id,
        f"step {step.id!r} fits or feeds a fit, so an apply keeps it as request "
        f"{applies_request_id!r} ran it, and that request's recipe {found}. "
        f"Restore the step, or submit without applies_request_id to fit again.",
        {"applies_request_id": applies_request_id},
    )


def _recorded_pins(
    ds: Dataset,
    named: Request,
    locked: tuple[str, ...],
    fitted: Mapping[str, bool],
) -> dict[str, str]:
    """The run each locked step that fits over its scope recorded in *named*."""
    logs = run_log_dir(ds.base_dir)
    pins: dict[str, str] = {}
    for step_id in locked:
        if not fitted.get(step_id, False):
            continue
        snapshot = read_run(logs, named.execution_of(step_id))
        if snapshot is None or not snapshot["run_id"]:
            raise StepRefused(
                "apply_request_unfinished",
                "",
                f"step {step_id!r} of request {named.request_id!r} recorded no "
                f"run, so an apply has no fit of it to reuse.",
                {"applies_request_id": named.request_id, "step": step_id},
            )
        pins[step_id] = snapshot["run_id"]
    return pins
