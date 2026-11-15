"""Planning, submitting and running a recipe that names a feature definition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Iterator
from typing import ClassVar, Final

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.definitions.define import (
    define_feature,
    load_feature_definition,
)
from mosaic.core.pipeline.definitions.scope import (
    DefinitionScope,
    current_scope,
    definition_class_for_slug,
    definition_scope,
    definition_sidecar_for_slug,
)
from mosaic.core.pipeline.definitions.sidecar import DefinitionSidecar
from mosaic.core.pipeline.definitions.store import (
    DefinitionDigestAmbiguous,
    DefinitionMismatch,
    FeatureDefinitionsNotAllowed,
    definition_module_name,
    parse_definition_slug,
)
from mosaic.core.pipeline.graph import (
    Plan,
    Recipe,
    Request,
    ReviewedPlan,
    execute_step,
    load_request,
    plan_pipeline,
    run_pipeline,
    save_request,
    submit_request,
)
from mosaic.core.pipeline.graph import step as step_module
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.inventory.cache import InventoryCache
from mosaic.core.pipeline.pipeline import FeatureStep, Pipeline
from mosaic.core.pipeline.types import Inputs
from mosaic.core.scope import Scope
from tests.helpers import (
    CANARY_NAME,
    CANARY_SOURCE,
    CANARY_VARIABLE,
    MINIMAL_NAME,
    MINIMAL_SOURCE,
    NARROWING_NAME,
    NARROWING_SOURCE,
    PlainFeature,
    add_track_sequences,
    defined_dataset,
    forget_imported_definitions,
    imported_definitions,
    make_dataset,
    noting_definition_loads,
)

SLUG: Final = f"def:{MINIMAL_NAME}"
STORAGE: Final = f"def.{MINIMAL_NAME}__from__tracks"
SECOND_SOURCE: Final = MINIMAL_SOURCE.replace(
    "scale: float = 1.0", "scale: float = 2.0"
).replace('version = "1"', 'version = "2"')
"""A second version of ``MINIMAL_SOURCE``, of another default and version."""

BUILTIN_REQUEST_FILE: Final = """{
  "request_id": "01JC0000000000000000000000",
  "recipe_digest": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "owner": "someone",
  "created_at": "2026-11-14T00:00:00Z",
  "scope": {
    "entries": null,
    "groups": [
      "g"
    ],
    "sequences": null
  },
  "bind": {},
  "allow_partial": false,
  "max_concurrent_steps": null,
  "step_executions": {
    "speed": "01JC0000000000000000000001"
  },
  "step_versions": {
    "speed": "3"
  }
}"""
"""The file ``save_request`` wrote for ``_builtin_request()`` before requests
recorded definitions."""


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _recipe(feature: str) -> Recipe:
    return Recipe.model_validate(
        {
            "schema_version": 1,
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


def _planned_run_id(plan: Plan) -> str:
    (step,) = plan.steps
    assert step.run_id is not None
    return step.run_id


def _planned_definitions(plan: Plan) -> dict[str, str]:
    """Each definition a plan names, by name, with the digest its spec pins."""
    pinned: dict[str, str] = {}
    for step in plan.steps:
        if step.spec.feature.startswith("def:"):
            name, digest = parse_definition_slug(step.spec.feature)
            assert digest is not None
            pinned[name] = digest
    return pinned


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return defined_dataset(tmp_path, {MINIMAL_NAME: MINIMAL_SOURCE})


# --- planning imports no definition ------------------------------------------


def test_a_plan_pins_the_full_digest_and_imports_no_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = defined_dataset(tmp_path, {CANARY_NAME: CANARY_SOURCE})
    slug = f"def:{CANARY_NAME}"
    _ = load_feature_definition(ds, slug)
    forget_imported_definitions()
    canary = tmp_path / "canary.jsonl"
    monkeypatch.setenv(CANARY_VARIABLE, str(canary))

    plan = plan_pipeline(ds, _recipe(slug))

    assert not imported_definitions()
    assert not canary.exists()
    (step,) = plan.steps
    assert step.spec.feature == f"{slug}@{_digest(CANARY_SOURCE)}"
    assert step.storage_name == f"def.{CANARY_NAME}__from__tracks"
    assert step.run_id is not None
    assert current_scope() is None


def test_a_recipe_pinned_to_a_version_plans_it_after_current_moved(
    dataset: Dataset,
) -> None:
    first = _digest(MINIMAL_SOURCE)
    before = _planned_run_id(plan_pipeline(dataset, _recipe(SLUG)))
    _ = define_feature(dataset, MINIMAL_NAME, SECOND_SOURCE)

    moved = plan_pipeline(dataset, _recipe(SLUG))
    by_digest = plan_pipeline(dataset, _recipe(f"{SLUG}@{first}"))
    by_prefix = plan_pipeline(dataset, _recipe(f"{SLUG}@{first[:12]}"))

    assert _planned_run_id(moved) != before
    assert _planned_run_id(by_digest) == before
    assert _planned_run_id(by_prefix) == before
    assert by_prefix.steps[0].spec.feature == f"{SLUG}@{first}"
    assert by_prefix.steps[0].storage_name == STORAGE


def _forged(sidecar: DefinitionSidecar, digest: str) -> DefinitionSidecar:
    return sidecar.model_copy(update={"digest": digest})


def test_an_ambiguous_prefix_is_refused_by_its_own_error(dataset: Dataset) -> None:
    with definition_scope(dataset, load=False):
        sidecar = definition_sidecar_for_slug(SLUG)
    assert sidecar is not None
    first = "ab" * 6 + "0" * 52
    second = "ab" * 6 + "1" * 52
    sidecars = {first: _forged(sidecar, first), second: _forged(sidecar, second)}
    recipe = _recipe(f"{SLUG}@{'ab' * 6}")

    with definition_scope(None, load=False, sidecars=sidecars):
        with pytest.raises(DefinitionDigestAmbiguous) as planned:
            _ = plan_pipeline(dataset, recipe)
        with pytest.raises(DefinitionDigestAmbiguous) as submitted:
            _ = submit_request(dataset, recipe)

    for caught in (planned, submitted):
        assert caught.value.error_reason() == "definition_digest_ambiguous"
        detail = caught.value.error_detail()
        assert detail is not None
        assert detail["matches"] == [first, second]


# --- a request pins the digests --------------------------------------------------


@pytest.mark.parametrize("reviewed", [False, True])
def test_a_reviewed_plan_pins_the_digests_it_recorded(
    dataset: Dataset, reviewed: bool
) -> None:
    inventory = InventoryCache(dataset)
    plan = plan_pipeline(dataset, _recipe(SLUG), inventory=inventory)
    generation = inventory.generation
    _ = define_feature(dataset, MINIMAL_NAME, SECOND_SOURCE)

    submitted = submit_request(
        dataset,
        _recipe(SLUG),
        inventory=inventory,
        reviewed=ReviewedPlan(plan, generation) if reviewed else None,
    )

    expected = _digest(MINIMAL_SOURCE if reviewed else SECOND_SOURCE)
    assert _planned_definitions(submitted.plan) == {MINIMAL_NAME: expected}
    assert submitted.request.definitions == {MINIMAL_NAME: expected}
    assert submitted.request.step_versions == {"d": "2" if not reviewed else "1"}


def test_a_request_records_the_digest_and_runs_it_after_current_moved(
    dataset: Dataset,
) -> None:
    submitted = submit_request(dataset, _recipe(SLUG))
    first = _digest(MINIMAL_SOURCE)
    assert submitted.request.definitions == {MINIMAL_NAME: first}
    assert submitted.request.step_versions == {"d": "1"}
    document = json.loads(submitted.request_path.read_text())
    assert document["definitions"] == {MINIMAL_NAME: first}
    assert load_request(dataset.base_dir, submitted.request.request_id) == (
        submitted.request
    )
    _ = define_feature(dataset, MINIMAL_NAME, SECOND_SOURCE)

    outcome = execute_step(dataset, submitted.request, "d")

    assert outcome.state == "ran"
    assert outcome.run_id == _planned_run_id(submitted.plan)
    assert feature_run_root(dataset, STORAGE, outcome.run_id).is_dir()
    assert current_scope() is None


def _two_versions_recipe(first: str) -> Recipe:
    """Step ``a`` names the current version and step ``b`` the version *first*."""
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [
                {"id": "a", "type": "feature", "feature": SLUG, "inputs": ["tracks"]},
                {
                    "id": "b",
                    "type": "feature",
                    "feature": f"{SLUG}@{first}",
                    "inputs": ["tracks"],
                },
            ],
        }
    )


def test_two_versions_of_one_definition_in_one_recipe_run_as_planned(
    dataset: Dataset,
) -> None:
    first = _digest(MINIMAL_SOURCE)
    second = _digest(SECOND_SOURCE)
    _ = define_feature(dataset, MINIMAL_NAME, SECOND_SOURCE)
    recipe = _two_versions_recipe(first[:12])

    submitted = submit_request(dataset, recipe)

    planned = {step.step_id: step for step in submitted.plan.steps}
    assert planned["a"].spec.feature == f"{SLUG}@{second}"
    assert planned["b"].spec.feature == f"{SLUG}@{first}"
    assert submitted.request.definitions == {MINIMAL_NAME: second}
    assert submitted.request.step_versions == {"a": "2", "b": "1"}
    executed = {
        step_id: execute_step(dataset, submitted.request, step_id).run_id
        for step_id in ("a", "b")
    }
    assert executed == {step_id: planned[step_id].run_id for step_id in ("a", "b")}
    assert executed["a"] != executed["b"]


def test_run_pipeline_runs_two_versions_of_one_definition_as_planned(
    dataset: Dataset,
) -> None:
    first = _digest(MINIMAL_SOURCE)
    _ = define_feature(dataset, MINIMAL_NAME, SECOND_SOURCE)
    recipe = _two_versions_recipe(first)
    planned = {
        step.step_id: step.run_id for step in plan_pipeline(dataset, recipe).steps
    }

    run = run_pipeline(dataset, recipe)

    assert {outcome.step_id: outcome.run_id for outcome in run.outcomes} == planned
    assert all(outcome.state == "ran" for outcome in run.outcomes)


def _builtin_request() -> Request:
    return Request(
        request_id="01JC0000000000000000000000",
        recipe_digest="a" * 64,
        owner="someone",
        created_at="2026-11-14T00:00:00Z",
        scope=Scope(groups=["g"]),
        step_executions={"speed": "01JC0000000000000000000001"},
        step_versions={"speed": "3"},
    )


def test_a_request_of_builtins_only_writes_the_file_it_wrote_before(
    tmp_path: Path,
) -> None:
    path = save_request(tmp_path, _builtin_request())
    assert path.read_text() == BUILTIN_REQUEST_FILE


# --- running in process ----------------------------------------------------------


def test_run_pipeline_runs_a_definition_under_its_planned_run_id(
    dataset: Dataset,
) -> None:
    planned = _planned_run_id(plan_pipeline(dataset, _recipe(SLUG)))

    run = run_pipeline(dataset, _recipe(SLUG))

    (outcome,) = run.outcomes
    assert (outcome.state, outcome.run_id) == ("ran", planned)
    assert feature_run_root(dataset, STORAGE, planned).is_dir()
    assert current_scope() is None


# --- a step loads its own definition alone ----------------------------------------


def _mixed_recipe() -> Recipe:
    """A built-in step and three definition steps, of which ``n`` fails its load."""
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                },
                {
                    "id": "c",
                    "type": "feature",
                    "feature": f"def:{CANARY_NAME}",
                    "inputs": ["tracks"],
                },
                {"id": "m", "type": "feature", "feature": SLUG, "inputs": ["tracks"]},
                {
                    "id": "n",
                    "type": "feature",
                    "feature": f"def:{NARROWING_NAME}",
                    "inputs": ["tracks"],
                    "params": {"wide": True},
                },
            ],
        }
    )


@pytest.fixture
def mixed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Dataset, Path]:
    """A dataset defining every definition of ``_mixed_recipe``, and its canary file."""
    ds = defined_dataset(
        tmp_path,
        {
            CANARY_NAME: CANARY_SOURCE,
            MINIMAL_NAME: MINIMAL_SOURCE,
            NARROWING_NAME: NARROWING_SOURCE,
        },
    )
    forget_imported_definitions()
    canary = tmp_path / "canary.jsonl"
    monkeypatch.setenv(CANARY_VARIABLE, str(canary))
    return ds, canary


def test_a_step_imports_its_own_definition_and_no_other(
    mixed: tuple[Dataset, Path],
) -> None:
    ds, canary = mixed
    submitted = submit_request(ds, _mixed_recipe())
    planned = {step.step_id: step.run_id for step in submitted.plan.steps}

    builtin = execute_step(ds, submitted.request, "speed")
    assert builtin.state == "ran"
    assert not imported_definitions()

    minimal = execute_step(ds, submitted.request, "m")
    assert (minimal.state, minimal.run_id) == ("ran", planned["m"])
    assert imported_definitions() == [
        definition_module_name(MINIMAL_NAME, _digest(MINIMAL_SOURCE))
    ]
    assert not canary.exists()
    forget_imported_definitions()

    counted = execute_step(ds, submitted.request, "c")
    assert (counted.state, counted.run_id) == ("ran", planned["c"])
    assert imported_definitions() == [
        definition_module_name(CANARY_NAME, _digest(CANARY_SOURCE))
    ]
    assert len(canary.read_text().splitlines()) == 1
    assert current_scope() is None


def test_a_step_loads_the_version_its_request_pinned(
    mixed: tuple[Dataset, Path],
) -> None:
    ds, _ = mixed
    submitted = submit_request(ds, _mixed_recipe())
    _ = define_feature(ds, MINIMAL_NAME, SECOND_SOURCE)

    outcome = execute_step(ds, submitted.request, "m")

    assert imported_definitions() == [
        definition_module_name(MINIMAL_NAME, _digest(MINIMAL_SOURCE))
    ]
    assert outcome.run_id == submitted.plan.step("m").run_id


def test_a_step_runs_whatever_another_definition_of_its_recipe_fails(
    mixed: tuple[Dataset, Path],
) -> None:
    ds, _ = mixed
    submitted = submit_request(ds, _mixed_recipe())

    assert execute_step(ds, submitted.request, "speed").state == "ran"
    assert execute_step(ds, submitted.request, "m").state == "ran"
    with pytest.raises(DefinitionMismatch) as raised:
        _ = execute_step(ds, submitted.request, "n")
    assert raised.value.fields == ("scope_dependent",)


CONSUMER_NAME: Final = "consumer-distance"
CONSUMER_SOURCE: Final = (
    MINIMAL_SOURCE.replace(f'name = "{MINIMAL_NAME}"', f'name = "{CONSUMER_NAME}"')
    .replace(
        "import DependencyLookup, InputStream, TrackInputs",
        "import DependencyLookup, InputStream, Inputs, Result",
    )
    .replace("class Inputs(TrackInputs):", "class Inputs(Inputs[Result]):")
    .replace('self.Inputs(("tracks",))', "self.Inputs(())")
    .replace('np.hypot(df["X"], df["Y"])', 'df["distance"]')
)
"""A definition that reads another feature's ``distance`` column."""

SAMPLE_ONLY_CANARY_SOURCE: Final = (
    CANARY_SOURCE.replace('version = "1"', 'version = "2"')
    .replace(
        "from mosaic.core.params import Params as ParamsBase\n",
        "from mosaic.core.params import Params as ParamsBase\n"
        "from mosaic.core.pipeline.column_contract import Provides\n",
    )
    .replace(
        "    consumed_roots = ()\n",
        "    consumed_roots = ()\n    provides = Provides()\n",
    )
)
"""A later version of ``CANARY_SOURCE`` that declares it writes only a sample."""


def test_a_step_reading_a_definition_step_imports_only_its_own_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The producer is read from the sidecar of the version its run recorded."""
    ds = defined_dataset(
        tmp_path, {CANARY_NAME: CANARY_SOURCE, CONSUMER_NAME: CONSUMER_SOURCE}
    )
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [
                {
                    "id": "c",
                    "type": "feature",
                    "feature": f"def:{CANARY_NAME}@{_digest(CANARY_SOURCE)}",
                    "inputs": ["tracks"],
                },
                {
                    "id": "k",
                    "type": "feature",
                    "feature": f"def:{CONSUMER_NAME}",
                    "inputs": [{"step": "c"}],
                },
            ],
        }
    )
    submitted = submit_request(ds, recipe)
    assert execute_step(ds, submitted.request, "c").state == "ran"
    _ = define_feature(ds, CANARY_NAME, SAMPLE_ONLY_CANARY_SOURCE)
    forget_imported_definitions()
    canary = tmp_path / "canary.jsonl"
    monkeypatch.setenv(CANARY_VARIABLE, str(canary))

    outcome = execute_step(ds, submitted.request, "k")

    assert (outcome.state, outcome.run_id) == ("ran", submitted.plan.step("k").run_id)
    assert imported_definitions() == [
        definition_module_name(CONSUMER_NAME, _digest(CONSUMER_SOURCE))
    ]
    assert not canary.exists()


def test_run_pipeline_loads_each_definition_when_its_step_runs(
    mixed: tuple[Dataset, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, _ = mixed
    events = noting_definition_loads(monkeypatch, step_module)
    recipe = Recipe.model_validate(
        {
            **_mixed_recipe().model_dump(mode="json"),
            "steps": [
                step
                for step in _mixed_recipe().model_dump(mode="json")["steps"]
                if step["id"] != "n"
            ],
        }
    )

    run = run_pipeline(ds, recipe)

    assert {outcome.step_id: outcome.state for outcome in run.outcomes} == {
        "speed": "ran",
        "c": "ran",
        "m": "ran",
    }
    loads = [index for index, (kind, _) in enumerate(events) if kind == "load"]
    assert len(loads) == 2
    for index in loads:
        assert events[index - 1] == ("build", events[index][1])


# --- a caller's scope of sidecars --------------------------------------------------


@pytest.fixture
def held_sidecar(dataset: Dataset) -> DefinitionSidecar:
    with definition_scope(dataset, load=False):
        sidecar = definition_sidecar_for_slug(SLUG)
    assert sidecar is not None
    return sidecar


@pytest.fixture
def plain(tmp_path: Path) -> Dataset:
    """A dataset with the same tracks and no definition."""
    held = make_dataset(tmp_path / "plain")
    add_track_sequences(held, ("g", "s1"), ("g", "s2"))
    return held


def test_a_caller_scope_of_sidecars_plans_without_the_dataset_holding_them(
    plain: Dataset, held_sidecar: DefinitionSidecar, dataset: Dataset
) -> None:
    digest = held_sidecar.digest
    pinned = f"{SLUG}@{digest}"
    expected = _planned_run_id(plan_pipeline(dataset, _recipe(SLUG)))

    with definition_scope(None, load=False, sidecars={digest: held_sidecar}):
        plan = plan_pipeline(plain, _recipe(pinned))

    assert _planned_run_id(plan) == expected
    assert plan.steps[0].spec.feature == pinned
    assert plan.steps[0].storage_name == STORAGE


def test_the_live_pipeline_predicts_the_planned_run_id_under_a_caller_scope(
    held_sidecar: DefinitionSidecar, dataset: Dataset
) -> None:
    """A run the planner named reads as the live pipeline's cached step."""
    digest = held_sidecar.digest
    (outcome,) = run_pipeline(dataset, _recipe(SLUG)).outcomes

    with definition_scope(None, load=False, sidecars={digest: held_sidecar}):
        declared = definition_class_for_slug(f"{SLUG}@{digest}")
        assert declared is not None
        pipe = Pipeline().add(FeatureStep("d", declared))
        (status,) = pipe.status(dataset).to_dict("records")

    assert (status["cached"], status["run_id"]) == (True, outcome.run_id)


# --- a disabled scope ------------------------------------------------------------


@pytest.fixture(params=["no-dataset", "same-dataset", "another-dataset"])
def disabled(
    request: pytest.FixtureRequest, dataset: Dataset, plain: Dataset
) -> Dataset | None:
    """The dataset a disabled scope is entered for."""
    return {"no-dataset": None, "same-dataset": dataset, "another-dataset": plain}[
        request.param
    ]


def test_planning_inside_a_disabled_scope_refuses_a_definition(
    dataset: Dataset, disabled: Dataset | None
) -> None:
    with definition_scope(disabled, load=False, allowed=False) as outer:
        with pytest.raises(FeatureDefinitionsNotAllowed) as planned:
            _ = plan_pipeline(dataset, _recipe(SLUG))
        with pytest.raises(FeatureDefinitionsNotAllowed) as submitted:
            _ = submit_request(dataset, _recipe(SLUG))
        with pytest.raises(FeatureDefinitionsNotAllowed) as ran:
            _ = run_pipeline(dataset, _recipe(SLUG))
        assert current_scope() is outer
    for caught in (planned, submitted, ran):
        assert caught.value.error_reason() == "feature_definitions_not_allowed"
        assert caught.value.slug == SLUG
    assert not imported_definitions()


def test_a_step_inside_a_disabled_scope_refuses_its_definition(
    dataset: Dataset, disabled: Dataset | None
) -> None:
    submitted = submit_request(dataset, _recipe(SLUG))

    with definition_scope(disabled, load=False, allowed=False) as outer:
        with pytest.raises(FeatureDefinitionsNotAllowed) as executed:
            _ = execute_step(dataset, submitted.request, "d")
        with pytest.raises(FeatureDefinitionsNotAllowed) as ran:
            _ = run_pipeline(dataset, _recipe(SLUG), request=submitted.request)
        assert current_scope() is outer
    for caught in (executed, ran):
        assert caught.value.error_reason() == "feature_definitions_not_allowed"
    assert not imported_definitions()


class _ScopeNoting(PlainFeature):
    """A feature that notes the definition scope it is built under."""

    name = "scope-noting"
    consumed_roots: tuple[str, ...] = ()
    seen: ClassVar[list[DefinitionScope | None]] = []

    def __init__(
        self, inputs: Inputs | None = None, params: dict[str, object] | None = None
    ) -> None:
        super().__init__(inputs, params)
        _ScopeNoting.seen.append(current_scope())


@pytest.fixture
def noting() -> Iterator[type[_ScopeNoting]]:
    _ScopeNoting.seen.clear()
    yield _ScopeNoting
    _ScopeNoting.seen.clear()


def test_the_live_pipeline_keeps_a_disabled_scope(
    plain: Dataset, disabled: Dataset | None, noting: type[_ScopeNoting]
) -> None:
    with definition_scope(disabled, load=False, allowed=False) as outer:
        _ = Pipeline().add(FeatureStep("d", noting)).status(plain)

    assert noting.seen
    assert all(seen is outer for seen in noting.seen)


def test_the_live_pipeline_enters_a_scope_of_its_dataset(
    plain: Dataset, noting: type[_ScopeNoting]
) -> None:
    _ = Pipeline().add(FeatureStep("d", noting)).status(plain)

    assert noting.seen
    for seen in noting.seen:
        assert seen is not None
        assert (seen.base_dir, seen.load) == (plain.base_dir, False)
