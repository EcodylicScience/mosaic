"""Maintaining a dataset that holds feature definitions: cleanup, reconcile, verify.

Each is a read of the definitions: it holds a ``load=False`` scope, rebuilds a
run's feature from the sidecar of the digest the run recorded, and imports no
source. Verify alone loads a source, the version a step's plan pins, to refit
that step.
"""

from __future__ import annotations

import datetime
import functools
import hashlib
import shutil
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline import reconcile as reconcile_module
from mosaic.core.pipeline.definitions.scope import (
    DefinitionScope,
    current_scope,
    definition_scope,
)
from mosaic.core.pipeline.definitions.store import (
    DEFINITIONS_ROOT,
    FeatureDefinitionsNotAllowed,
    definition_module_name,
    list_versions,
)
from mosaic.core.pipeline.definitions.define import (
    define_feature,
    load_feature_definition,
)
from mosaic.core.pipeline.graph import (
    CleanupPlan,
    CleanupPolicy,
    Plan,
    Recipe,
    Request,
    RetirementKeeper,
    plan_fingerprint,
    plan_pipeline,
    plan_retirement,
    plan_with_cleanup,
    run_cleanup,
    run_pipeline,
    submit_request,
)
from mosaic.core.pipeline.graph import cleanup as cleanup_module
from mosaic.core.pipeline.graph import plan as plan_module
from mosaic.core.pipeline.graph import resolve as resolve_module
from mosaic.core.pipeline.graph.resolve import run_reference_fields
from mosaic.core.pipeline.graph.step import execute_step
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.job import CancelToken
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.verify import verify_plan
from mosaic.core.scope import Scope
from tests.helpers import (
    APPLYING_NAME,
    APPLYING_STORAGE,
    APPLYING_SOURCE,
    GLOBAL_NAME,
    GLOBAL_SOURCE,
    LOOSE,
    MINIMAL_NAME,
    MINIMAL_SOURCE,
    add_track_sequences,
    defined_dataset,
    fit_over,
    forget_imported_definitions,
    imported_definitions,
    make_dataset,
    named_recipe,
    noting_definition_loads,
    reasons_by_run,
    record_open_attempt,
    run_id_of,
    run_speed,
    speed_recipe_stepping,
    tree_snapshot,
)

SLUG: Final = f"def:{MINIMAL_NAME}"
STORAGE: Final = f"def.{MINIMAL_NAME}__from__tracks"
SECOND_SOURCE: Final = MINIMAL_SOURCE.replace(
    "scale: float = 1.0", "scale: float = 2.0"
).replace('version = "1"', 'version = "2"')
"""A second version of ``MINIMAL_SOURCE``, of another default and version."""

LABELED_NAME: Final = "labeled-distance"
LABELED_SOURCE: Final = MINIMAL_SOURCE.replace(
    f'name = "{MINIMAL_NAME}"', f'name = "{LABELED_NAME}"'
).replace("scale: float = 1.0", 'scale: float = 1.0\n        label: str = "a"')
"""A definition whose params hold a ``str``, which may name a model or a run."""

UNLABELED_SOURCE: Final = LABELED_SOURCE.replace(
    '\n        label: str = "a"', ""
).replace('version = "1"', 'version = "2"')
"""The next version of ``LABELED_SOURCE``, which holds no ``str`` param."""

LATER: Final = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=30)
"""An instant after every request's hold window."""


def _recipe(feature: str) -> Recipe:
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "maintained",
            "steps": [
                {
                    "id": "d",
                    "type": "feature",
                    "feature": feature,
                    "inputs": ["tracks"],
                }
            ],
        }
    )


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _run_of(name: str, source: str, dataset: Dataset) -> str:
    """Run ``def:<name>`` pinned to the version that *source* defined."""
    recipe = _recipe(f"def:{name}@{_digest(source)}")
    (outcome,) = run_pipeline(dataset, recipe).outcomes
    assert outcome.run_id is not None
    return outcome.run_id


def _cleanup(
    dataset: Dataset, recipe: Recipe, policy: CleanupPolicy = LOOSE
) -> CleanupPlan:
    plan = plan_with_cleanup(dataset, recipe, cleanup=policy, now=LATER)
    assert plan.cleanup is not None
    return plan.cleanup


def _then[**P, R](fn: Callable[P, R], act: Callable[[], object]) -> Callable[P, R]:
    """*fn*, which runs *act* after each call and returns what the call returned."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        found = fn(*args, **kwargs)
        _ = act()
        return found

    return wrapper


def _noting_scopes(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, name: str
) -> list[DefinitionScope | None]:
    """Replace ``module.<name>`` with itself, noting the scope each call finds."""
    seen: list[DefinitionScope | None] = []
    original: Callable[..., object] = vars(module)[name]

    @functools.wraps(original)
    def noting(*args: object, **kwargs: object) -> object:
        seen.append(current_scope())
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, noting)
    return seen


def _finding_of(dataset: Dataset, run_id: str) -> tuple[str, str]:
    """The verdict and the reason a dry-run reconcile reports for *run_id*."""
    report = dataset.reconcile(only=("features",))
    (finding,) = (each for each in report.findings if each.old_address == run_id)
    return finding.verdict, finding.reason


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return defined_dataset(tmp_path, {MINIMAL_NAME: MINIMAL_SOURCE})


# --- cleanup ----------------------------------------------------------------------


@pytest.fixture
def speed_dataset(tmp_path: Path) -> Dataset:
    """A dataset holding a superseded ``speed-angvel`` run beside two definitions."""
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, ("g", "s1"), ("g", "s2"))
    _ = define_feature(ds, MINIMAL_NAME, MINIMAL_SOURCE)
    _ = define_feature(ds, LABELED_NAME, LABELED_SOURCE)
    return ds


def test_a_definition_storage_whose_params_hold_a_str_reads_every_run(
    speed_dataset: Dataset,
) -> None:
    """A run of unknown params reads every run when its sidecar declares a ``str``."""
    old = run_speed(speed_dataset, 1)
    _ = record_open_attempt(
        speed_dataset, "", target=f"def.{LABELED_NAME}__from__tracks"
    )

    cleanup = _cleanup(speed_dataset, speed_recipe_stepping(2))

    assert reasons_by_run(cleanup) == {old: "read_by"}
    assert not imported_definitions()


def test_a_definition_storage_whose_params_hold_no_str_reads_no_run(
    speed_dataset: Dataset,
) -> None:
    old = run_speed(speed_dataset, 1)
    _ = record_open_attempt(speed_dataset, "", target=STORAGE)

    cleanup = _cleanup(speed_dataset, speed_recipe_stepping(2))

    assert reasons_by_run(cleanup) == {old: "superseded"}


def test_a_run_of_unknown_version_reads_every_run_when_any_version_declares_a_str(
    speed_dataset: Dataset,
) -> None:
    """The version that wrote a run with no ``params.json`` is unknown."""
    _ = define_feature(speed_dataset, LABELED_NAME, UNLABELED_SOURCE)
    old = run_speed(speed_dataset, 1)
    _ = record_open_attempt(
        speed_dataset, "", target=f"def.{LABELED_NAME}__from__tracks"
    )

    cleanup = _cleanup(speed_dataset, speed_recipe_stepping(2))

    assert reasons_by_run(cleanup) == {old: "read_by"}


# --- a run of a version that is not current -------------------------------------------


def test_a_run_of_a_version_other_than_current_is_superseded_by_the_rules_of_every_run(
    dataset: Dataset,
) -> None:
    _ = define_feature(dataset, MINIMAL_NAME, SECOND_SOURCE)
    first = _run_of(MINIMAL_NAME, MINIMAL_SOURCE, dataset)
    second = _run_of(MINIMAL_NAME, SECOND_SOURCE, dataset)
    assert first != second

    current = _cleanup(dataset, _recipe(SLUG))
    newest = _cleanup(
        dataset, _recipe(SLUG), CleanupPolicy(keep_newest=1, min_age_hours=0)
    )
    first_digest = _digest(MINIMAL_SOURCE)
    pinned = _cleanup(dataset, _recipe(f"{SLUG}@{first_digest}"))

    assert [(run.run_id, run.reason) for run in current.runs] == [(first, "superseded")]
    assert [(run.run_id, run.reason) for run in newest.runs] == [(first, "newest")]
    assert [(run.run_id, run.reason) for run in pinned.runs] == [(second, "superseded")]


def test_a_running_request_of_a_pinned_version_holds_every_run_of_the_definition(
    dataset: Dataset,
) -> None:
    _ = define_feature(dataset, MINIMAL_NAME, SECOND_SOURCE)
    first = _run_of(MINIMAL_NAME, MINIMAL_SOURCE, dataset)
    second = _run_of(MINIMAL_NAME, SECOND_SOURCE, dataset)
    pinned = _recipe(f"{SLUG}@{_digest(SECOND_SOURCE)}")
    named = Recipe.model_validate({**pinned.model_dump(mode="json"), "name": "other"})
    _ = submit_request(dataset, named)

    cleanup = _cleanup(dataset, _recipe(SLUG))

    assert {run.run_id: run.reason for run in cleanup.runs} == {
        first: "held_by_request"
    }
    assert second not in reasons_by_run(cleanup)


def test_a_request_of_a_pinned_version_submitted_during_a_cleanup_holds_its_runs(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = define_feature(dataset, MINIMAL_NAME, SECOND_SOURCE)
    first_class = load_feature_definition(dataset, f"{SLUG}@{_digest(MINIMAL_SOURCE)}")
    first = run_feature(dataset, first_class()).run_id
    recipe = _recipe(SLUG)
    plan = plan_with_cleanup(dataset, recipe, cleanup=LOOSE, now=LATER)
    request = submit_request(
        dataset, recipe, cleanup=LOOSE, expect_plan=plan_fingerprint(plan)
    ).request
    assert request.cleanup is not None
    assert [target.run_id for target in request.cleanup.approved] == [first]
    _ = execute_step(dataset, request, "d")
    pinned = _recipe(f"{SLUG}@{_digest(MINIMAL_SOURCE)[:12]}")
    named = Recipe.model_validate({**pinned.model_dump(mode="json"), "name": "other"})
    monkeypatch.setattr(
        cleanup_module,
        "plan_cleanup",
        _then(cleanup_module.plan_cleanup, lambda: submit_request(dataset, named)),
    )

    outcome = run_cleanup(dataset, request, token=CancelToken(), now=LATER)

    assert [(run_id, reason) for _, run_id, reason in outcome.kept] == [
        (first, "held_by_request")
    ]
    assert feature_run_root(dataset, STORAGE, str(first)).is_dir()


def test_cleanup_of_unknown_params_reads_the_sidecars_under_a_load_scope(
    speed_dataset: Dataset,
) -> None:
    old = run_speed(speed_dataset, 1)
    _ = record_open_attempt(
        speed_dataset, "", target=f"def.{LABELED_NAME}__from__tracks"
    )
    forget_imported_definitions()

    with definition_scope(speed_dataset, load=True):
        cleanup = _cleanup(speed_dataset, speed_recipe_stepping(2))

    assert reasons_by_run(cleanup) == {old: "read_by"}
    assert not imported_definitions()


def test_a_definition_whose_sidecar_cannot_be_read_may_declare_a_run_reference(
    speed_dataset: Dataset,
) -> None:
    old = run_speed(speed_dataset, 1)
    _ = record_open_attempt(speed_dataset, "", target=STORAGE)
    directory = speed_dataset.get_root(DEFINITIONS_ROOT) / MINIMAL_NAME
    for version in list_versions(directory):
        _ = (directory / f"{version.stem}.json").write_text("{")

    cleanup = _cleanup(speed_dataset, speed_recipe_stepping(2))

    assert reasons_by_run(cleanup) == {old: "read_by"}


def test_a_definition_whose_directory_was_removed_may_declare_a_run_reference(
    speed_dataset: Dataset,
) -> None:
    old = run_speed(speed_dataset, 1)
    _ = record_open_attempt(speed_dataset, "", target=STORAGE)
    shutil.rmtree(speed_dataset.get_root(DEFINITIONS_ROOT) / MINIMAL_NAME)

    cleanup = _cleanup(speed_dataset, speed_recipe_stepping(2))

    assert reasons_by_run(cleanup) == {old: "read_by"}


# --- reconcile and verify ----------------------------------------------------------


def test_reconcile_reads_the_first_versions_run_after_an_edit(
    dataset: Dataset,
) -> None:
    first = _run_of(MINIMAL_NAME, MINIMAL_SOURCE, dataset)
    _ = define_feature(dataset, MINIMAL_NAME, SECOND_SOURCE)
    forget_imported_definitions()
    before = tree_snapshot(dataset.base_dir)

    verdict, reason = _finding_of(dataset, first)

    assert (verdict, reason) == ("ok", "identifier and scheme current")
    assert not imported_definitions()
    assert tree_snapshot(dataset.base_dir) == before


def test_reconcile_reads_a_run_under_a_scope_pinned_to_another_version(
    dataset: Dataset,
) -> None:
    first = _run_of(MINIMAL_NAME, MINIMAL_SOURCE, dataset)
    _ = define_feature(dataset, MINIMAL_NAME, SECOND_SOURCE)
    pins = {MINIMAL_NAME: _digest(SECOND_SOURCE)}

    with definition_scope(dataset, load=False, pins=pins):
        verdict, reason = _finding_of(dataset, first)

    assert (verdict, reason) == ("ok", "identifier and scheme current")


def test_reconcile_reports_a_run_whose_version_was_removed_by_hand(
    dataset: Dataset,
) -> None:
    first = _run_of(MINIMAL_NAME, MINIMAL_SOURCE, dataset)
    _ = define_feature(dataset, MINIMAL_NAME, SECOND_SOURCE)
    first_digest = _digest(MINIMAL_SOURCE)
    directory = dataset.get_root(DEFINITIONS_ROOT) / MINIMAL_NAME
    for version in list_versions(directory):
        if version.digest == first_digest:
            for path in directory.glob(f"{version.stem}.*"):
                path.unlink()
    forget_imported_definitions()

    verdict, reason = _finding_of(dataset, first)

    assert verdict == "unresolvable_pre_provenance"
    assert first_digest[:12] in reason
    assert not imported_definitions()


def test_reconcile_reports_a_definition_run_that_records_no_definition(
    dataset: Dataset,
) -> None:
    first = _run_of(MINIMAL_NAME, MINIMAL_SOURCE, dataset)
    params = feature_run_root(dataset, STORAGE, first) / "params.json"
    _ = params.write_text(params.read_text().replace('"_definition"', '"_unrecorded"'))

    verdict, reason = _finding_of(dataset, first)

    assert verdict == "unresolvable_pre_provenance"
    assert "records no definition" in reason


def test_verify_refits_the_first_versions_run_after_an_edit(
    tmp_path: Path,
) -> None:
    """The step is refitted from the version its plan pins, and from no other."""
    ds = defined_dataset(tmp_path, {GLOBAL_NAME: GLOBAL_SOURCE})
    first = _run_of(GLOBAL_NAME, GLOBAL_SOURCE, ds)
    _ = define_feature(ds, GLOBAL_NAME, GLOBAL_SOURCE.replace('"1"', '"2"'))
    digest = _digest(GLOBAL_SOURCE)
    plan = plan_pipeline(ds, _recipe(f"def:{GLOBAL_NAME}@{digest}"))
    assert plan.step("d").run_id == first
    forget_imported_definitions()
    before = tree_snapshot(ds.base_dir)

    report = verify_plan(ds, plan)

    (verdict,) = report.steps
    assert (verdict.status, verdict.role) == ("ok", "fit"), verdict.reason
    assert report.ok
    assert imported_definitions() == [definition_module_name(GLOBAL_NAME, digest)]
    assert tree_snapshot(ds.base_dir) == before
    assert current_scope() is None


SPREAD_NAME: Final = "spread-distance"
SPREAD_SOURCE: Final = GLOBAL_SOURCE.replace(
    f'name = "{GLOBAL_NAME}"', f'name = "{SPREAD_NAME}"'
)
"""A second fitting definition, of another name."""


def test_verify_loads_each_definition_only_for_its_own_refit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = defined_dataset(
        tmp_path, {GLOBAL_NAME: GLOBAL_SOURCE, SPREAD_NAME: SPREAD_SOURCE}
    )
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [
                {
                    "id": step_id,
                    "type": "feature",
                    "feature": f"def:{name}",
                    "inputs": ["tracks"],
                }
                for step_id, name in (("a", GLOBAL_NAME), ("b", SPREAD_NAME))
            ],
        }
    )
    _ = run_pipeline(ds, recipe)
    plan = plan_pipeline(ds, recipe)
    forget_imported_definitions()
    events = noting_definition_loads(monkeypatch, resolve_module)

    report = verify_plan(ds, plan)

    assert [verdict.status for verdict in report.steps] == ["ok", "ok"]
    planned = [plan.step(step_id).spec.feature for step_id in ("a", "b")]
    assert [slug for kind, slug in events if kind == "load"] == planned
    for index, (kind, slug) in enumerate(events):
        if kind == "load":
            assert events[index - 1] == ("build", slug)


# --- a disabled scope ---------------------------------------------------------------


@pytest.fixture(params=["no-dataset", "same-dataset", "another-dataset"])
def disabled(
    request: pytest.FixtureRequest, dataset: Dataset, tmp_path: Path
) -> Dataset | None:
    """The dataset a disabled scope is entered for."""
    return {
        "no-dataset": None,
        "same-dataset": dataset,
        "another-dataset": make_dataset(tmp_path / "another"),
    }[request.param]


def test_reconcile_inside_a_disabled_scope_keeps_it_disabled(
    dataset: Dataset, disabled: Dataset | None
) -> None:
    first = _run_of(MINIMAL_NAME, MINIMAL_SOURCE, dataset)
    forget_imported_definitions()

    with definition_scope(disabled, load=False, allowed=False) as outer:
        verdict, reason = _finding_of(dataset, first)
        assert current_scope() is outer

    assert verdict == "unresolvable_pre_provenance"
    assert "not allowed" in reason
    assert outer.allowed is False
    assert not imported_definitions()


def test_verify_inside_a_disabled_scope_keeps_it_disabled(
    dataset: Dataset, disabled: Dataset | None
) -> None:
    _ = _run_of(MINIMAL_NAME, MINIMAL_SOURCE, dataset)
    plan: Plan = plan_pipeline(dataset, _recipe(SLUG))
    forget_imported_definitions()

    with definition_scope(disabled, load=False, allowed=False) as outer:
        with pytest.raises(FeatureDefinitionsNotAllowed):
            _ = verify_plan(dataset, plan)
        assert current_scope() is outer

    assert not imported_definitions()


@pytest.fixture
def approved_cleanup(dataset: Dataset) -> Request:
    """A request whose steps ran and whose cleanup is approved."""
    recipe = _recipe(SLUG)
    plan = plan_with_cleanup(dataset, recipe, cleanup=LOOSE)
    request = submit_request(
        dataset, recipe, cleanup=LOOSE, expect_plan=plan_fingerprint(plan)
    ).request
    _ = execute_step(dataset, request, "d")
    return request


def test_cleanup_inside_a_disabled_scope_keeps_it_disabled(
    dataset: Dataset,
    approved_cleanup: Request,
    disabled: Dataset | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _noting_scopes(monkeypatch, plan_module, "plan_pipeline")
    forget_imported_definitions()

    with definition_scope(disabled, load=False, allowed=False) as outer:
        outcome = run_cleanup(dataset, approved_cleanup, token=CancelToken())
        assert current_scope() is outer

    assert seen == [outer]
    assert outcome.status == "failed"
    assert "feature definitions are not allowed here" in outcome.error
    assert not imported_definitions()


def test_cleanup_enters_a_scope_of_its_dataset(
    dataset: Dataset, approved_cleanup: Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _noting_scopes(monkeypatch, plan_module, "plan_pipeline")

    outcome = run_cleanup(dataset, approved_cleanup, token=CancelToken())

    (scope,) = seen
    assert scope is not None
    assert (scope.base_dir, scope.load, scope.allowed) == (
        dataset.base_dir,
        False,
        True,
    )
    assert approved_cleanup.definitions
    assert scope.pins == approved_cleanup.definitions
    assert outcome.status == "finished"
    assert current_scope() is None


def test_dataset_reconcile_enters_a_scope_of_its_dataset(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _noting_scopes(monkeypatch, reconcile_module, "identity_reconcilers")

    _ = dataset.reconcile(only=("features",))

    (scope,) = seen
    assert scope is not None
    assert (scope.base_dir, scope.load, scope.allowed) == (
        dataset.base_dir,
        False,
        True,
    )
    assert current_scope() is None


# --- a fit that an apply names ---------------------------------------------


def _fit_recipe(name: str) -> Recipe:
    """The fit of ``def:centered-distance`` as a recipe named *name*."""
    return named_recipe(_recipe(f"def:{APPLYING_NAME}"), name)


def _apply_recipe(fit: str) -> Recipe:
    """``def:centered-distance`` applying the fit run *fit* by ``model``."""
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "applying",
            "steps": [
                {
                    "id": "d",
                    "type": "feature",
                    "feature": f"def:{APPLYING_NAME}",
                    "inputs": ["tracks"],
                    "params": {"model": {"feature": APPLYING_STORAGE, "run_id": fit}},
                }
            ],
        }
    )


class _FitAndApply:
    """A fit, its apply and a superseded sibling fit of one definition.

    The fit covers ``g/s1`` and the sibling ``g/s2``. Each is written by the
    request of its own recipe when *recorded* is true, and by a bare run
    otherwise, which no request records.

    Attributes:
        dataset: The dataset that holds the three runs.
        fit: The fit's run id.
        sibling: The sibling fit's run id.
        apply_recipe: The recipe that applies the fit.
        apply: The apply's run id.
        digests: The digests of the recipes that wrote the fit and the sibling,
            empty unless *recorded*.
    """

    def __init__(self, tmp_path: Path, *, recorded: bool) -> None:
        self.dataset: Dataset = defined_dataset(
            tmp_path, {APPLYING_NAME: APPLYING_SOURCE}
        )
        self.digests: list[str] = []
        self.fit: str = self._fit_over("fit", ("g", "s1"), recorded)
        self.sibling: str = self._fit_over("sibling", ("g", "s2"), recorded)
        self.apply_recipe: Recipe = _apply_recipe(self.fit)
        self.apply: str = run_id_of(run_pipeline(self.dataset, self.apply_recipe), "d")
        assert len({self.fit, self.sibling, self.apply}) == 3
        forget_imported_definitions()

    def _fit_over(self, name: str, entry: tuple[str, str], recorded: bool) -> str:
        """The run id of a fit over *entry* alone, recorded or not."""
        scope = Scope(entries=[entry])
        if recorded:
            ran = run_pipeline(self.dataset, _fit_recipe(name), scope=scope)
            self.digests.append(ran.recipe_digest)
            return run_id_of(ran, "d")
        return fit_over(self.dataset, f"def:{APPLYING_NAME}", entry)


def test_a_definitions_model_is_a_field_that_may_name_a_run(tmp_path: Path) -> None:
    ds = defined_dataset(tmp_path, {APPLYING_NAME: APPLYING_SOURCE})
    forget_imported_definitions()

    with definition_scope(ds, load=False):
        assert "model" in run_reference_fields(f"def:{APPLYING_NAME}")

    assert not imported_definitions()


def test_cleanup_keeps_the_fit_an_apply_names_and_removes_its_sibling(
    tmp_path: Path,
) -> None:
    runs = _FitAndApply(tmp_path, recorded=False)

    cleanup = _cleanup(runs.dataset, runs.apply_recipe)

    assert reasons_by_run(cleanup) == {runs.fit: "read_by", runs.sibling: "superseded"}
    assert not imported_definitions()


def test_retirement_keeps_the_fit_a_kept_apply_names_and_removes_its_sibling(
    tmp_path: Path,
) -> None:
    runs = _FitAndApply(tmp_path, recorded=True)

    plan = plan_retirement(
        runs.dataset,
        runs.digests,
        [RetirementKeeper(recipe=runs.apply_recipe, scope=Scope())],
        LOOSE,
        "",
        now=LATER,
    )

    assert reasons_by_run(plan.cleanup) == {
        runs.fit: "read_by",
        runs.sibling: "superseded",
    }
    assert not imported_definitions()
