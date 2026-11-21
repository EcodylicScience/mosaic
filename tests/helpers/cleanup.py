"""What the tests of a cleanup and of a retirement share.

Both judge runs under :class:`~mosaic.core.pipeline.graph.CleanupPolicy`, read the
reasons the judging gave, end the attempts that hold runs, and watch the removals.
The retired recipe the tests of a retirement use is ``speed_heading_recipe``, which
runs ``speed-angvel`` and ``heading`` as two storages.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Literal

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline import feature_run_removal
from mosaic.core.pipeline.graph import (
    CleanupOutcome,
    CleanupPlan,
    CleanupPolicy,
    Recipe,
    RetirementKeeper,
    RetirementSpec,
    SupersededRun,
    plan_pipeline,
    plan_retirement,
    run_pipeline,
)
from mosaic.core.pipeline.graph import cleanup as cleanup_module
from mosaic.runlog import JsonlRunLog, run_log_path
from tests.helpers.recipes import run_id_of, speed_heading_recipe

LOOSE = CleanupPolicy(keep_newest=0, min_age_hours=0)
"""The policy mosaic-api plans a cleanup under."""

SPEED_STORAGE = "speed-angvel__from__tracks"
"""The storage of the retired recipe's ``speed`` step."""

HEADING_STORAGE = "heading__from__tracks"
"""The storage of the retired recipe's ``heading`` step."""


def managed_dataset(dataset: Dataset) -> Dataset:
    """Return *dataset* loaded again after its manifest says ``managed_by: mosaic-api``."""
    with dataset.manifest_path.open("a", encoding="utf-8") as manifest:
        _ = manifest.write("managed_by: mosaic-api\n")
    return Dataset(manifest_path=dataset.manifest_path).load(ensure_roots=False)


def end_attempt(ds: Dataset, execution_id: str) -> None:
    """Write the ``finished`` event that ends attempt *execution_id*."""
    with JsonlRunLog(run_log_path(ds.base_dir, execution_id), execution_id) as log:
        log.finished()


def kept_reasons(outcome: CleanupOutcome) -> dict[str, str]:
    """The reason *outcome* kept each approved run it kept, by run id."""
    return {run_id: reason for _, run_id, reason in outcome.kept}


def planned_run_ids(ds: Dataset, recipe: Recipe) -> dict[str, str]:
    """What each step of *recipe* that names a run is called on *ds*."""
    return {
        step_id: run_id
        for step_id, run_id in plan_pipeline(ds, recipe).run_ids.items()
        if run_id is not None
    }


def runs_by_id(cleanup: CleanupPlan) -> dict[str, SupersededRun]:
    """The runs *cleanup* judged, by run id."""
    return {run.run_id: run for run in cleanup.runs}


def reasons_by_run(cleanup: CleanupPlan) -> dict[str, str]:
    """The reason *cleanup* gave each run it judged, by run id."""
    return {run.run_id: run.reason for run in cleanup.runs}


def _around[**P, R](
    function: Callable[P, R],
    *,
    before: Callable[[], None],
    after: Callable[[], None],
) -> Callable[P, R]:
    """Return *function* with its signature, calling *before* and *after* it."""

    @functools.wraps(function)
    def observed(*args: P.args, **kwargs: P.kwargs) -> R:
        before()
        result = function(*args, **kwargs)
        after()
        return result

    return observed


def _nothing() -> None:
    """Stand for an observer that is not given."""


def act_after_judging(
    monkeypatch: pytest.MonkeyPatch,
    attribute: Literal["judge_runs", "plan_cleanup"],
    act: Callable[[], None],
) -> None:
    """Call *act* once ``cleanup.<attribute>`` has judged its runs, before any removal.

    ``judge_runs`` is what a retirement judges with, and ``plan_cleanup`` what a
    request's cleanup judges with.
    """
    if attribute == "judge_runs":
        wrapped = _around(cleanup_module.judge_runs, before=_nothing, after=act)
    else:
        wrapped = _around(cleanup_module.plan_cleanup, before=_nothing, after=act)
    monkeypatch.setattr(cleanup_module, attribute, wrapped)


def observe_removals(
    monkeypatch: pytest.MonkeyPatch,
    *,
    before: Callable[[], None] = _nothing,
    after: Callable[[], None] = _nothing,
) -> None:
    """Call *before* and *after* each ``remove_feature_run`` call, which still runs."""
    monkeypatch.setattr(
        feature_run_removal,
        "remove_feature_run",
        _around(feature_run_removal.remove_feature_run, before=before, after=after),
    )


class OldRuns:
    """The finished request of ``speed_heading_recipe``, its runs and its retirement.

    Attributes:
        digest: The digest of the recipe to retire.
        speed: The run id of its ``speed`` step.
        heading: The run id of its ``heading`` step.
        spec: The approved retirement of the recipe, planned under :data:`LOOSE`.
    """

    def __init__(
        self,
        ds: Dataset,
        keepers: Sequence[RetirementKeeper] | None = None,
        *,
        manager: str = "",
    ) -> None:
        ran = run_pipeline(ds, speed_heading_recipe())
        self.digest: str = ran.recipe_digest
        self.speed: str = run_id_of(ran, "speed")
        self.heading: str = run_id_of(ran, "heading")
        self.spec: RetirementSpec = plan_retirement(
            ds, [self.digest], keepers or [], LOOSE, manager
        ).spec

    def exist(self, ds: Dataset) -> tuple[bool, bool]:
        """Whether the ``speed`` run and the ``heading`` run are on disk."""
        from mosaic.core.pipeline.index import feature_run_root

        return (
            feature_run_root(ds, SPEED_STORAGE, self.speed).is_dir(),
            feature_run_root(ds, HEADING_STORAGE, self.heading).is_dir(),
        )
