"""Feature definitions over the MCP tools: the operator's flag and ``define_feature``.

With ``allow_feature_definitions`` off, every tool call runs under a disabled
definition scope: the catalogs list no ``def:`` step type, a recipe naming one is
refused with ``feature_definitions_not_allowed`` before it is validated, and the
``define_feature`` action of ``manage_dataset`` is refused the same way. With it on,
each call reads the definitions of the dataset it names, and only those. No call
imports a definition's source.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from pydantic import BaseModel

from mosaic.agent.mcp.local import LocalBackend
from mosaic.agent.mcp.resources import RESOURCES, ResourceText
from mosaic.agent.mcp.spec import DatasetChange, SubmittedPipeline, call_tool
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.definitions.define import define_feature
from mosaic.core.pipeline.definitions.scope import (
    DefinitionScope,
    clear_definition_caches,
    current_scope,
    definition_scope,
)
from mosaic.core.pipeline.definitions.store import FeatureDefinitionsNotAllowed
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.graph import (
    ValidationReport,
    read_recipe,
    recipe_digest,
    save_recipe,
)
from mosaic.core.pipeline.graph import plan as plan_module
from mosaic.core.pipeline.graph.recipe_store import recipe_path
from mosaic.core.pipeline.graph import step as step_module
from mosaic.core.pipeline.index import feature_run_root
from tests.helpers import (
    MINIMAL_NAME,
    MINIMAL_SOURCE,
    SPEED_RECIPE,
    RecordingStarter,
    defined_dataset,
    forget_imported_definitions,
    imported_definitions,
    noting_definition_loads,
    observed,
)

SLUG: Final = f"def:{MINIMAL_NAME}"
STORAGE: Final = f"def.{MINIMAL_NAME}__from__tracks"

DEFINITION_RECIPE: Final[dict[str, object]] = {
    "schema_version": 1,
    "name": "distance",
    "steps": [{"id": "d", "type": "feature", "feature": SLUG, "inputs": ["tracks"]}],
}

REPORT_SOURCE: Final = MINIMAL_SOURCE.replace(
    "    consumed_roots = ()\n",
    '    consumed_roots = ()\n    report_file = "report.json"\n',
)

EDITED_SOURCE: Final = MINIMAL_SOURCE.replace(
    "scale: float = 1.0", "scale: float = 2.0"
)


@pytest.fixture(autouse=True)
def no_imported_definitions() -> Iterator[None]:
    """Start and end each test with no definition imported and no sidecar cached."""
    forget_imported_definitions()
    clear_definition_caches()
    yield
    forget_imported_definitions()
    clear_definition_caches()


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return defined_dataset(tmp_path, {MINIMAL_NAME: MINIMAL_SOURCE})


@pytest.fixture
def starter() -> RecordingStarter:
    return RecordingStarter()


def _backend(
    dataset: Dataset, starter: RecordingStarter, *, allowed: bool
) -> LocalBackend:
    return LocalBackend(
        [dataset.manifest_path],
        warm=False,
        start_driver=starter,
        allow_feature_definitions=allowed,
    )


@pytest.fixture
def off(dataset: Dataset, starter: RecordingStarter) -> LocalBackend:
    return _backend(dataset, starter, allowed=False)


@pytest.fixture
def on(dataset: Dataset, starter: RecordingStarter) -> LocalBackend:
    return _backend(dataset, starter, allowed=True)


def _uuid(dataset: Dataset) -> str:
    uuid = dataset.manifest.uuid
    assert uuid is not None
    return uuid


def _not_allowed(result: BaseModel) -> ErrorEnvelope:
    assert isinstance(result, ErrorEnvelope), result
    assert (result.kind, result.reason) == (
        "unavailable",
        "feature_definitions_not_allowed",
    )
    return result


def _listed(backend: LocalBackend, dataset: Dataset | None) -> str:
    arguments: dict[str, object] = {"query": "distance", "limit": 100}
    if dataset is not None:
        arguments["dataset"] = _uuid(dataset)
    listed = call_tool(backend, "describe_step_types", arguments)
    assert not isinstance(listed, ErrorEnvelope), listed
    return listed.model_dump_json()


# ------------------------------------------------------------------ flag off


def test_define_feature_is_refused_without_the_flag(
    off: LocalBackend, dataset: Dataset
) -> None:
    refused = call_tool(
        off,
        "manage_dataset",
        {
            "action": "define_feature",
            "dataset": _uuid(dataset),
            "name": "another-distance",
            "source": MINIMAL_SOURCE,
        },
    )

    _ = _not_allowed(refused)
    assert not (dataset.base_dir / "feature_definitions" / "another-distance").exists()


def test_the_catalog_lists_no_definition_without_the_flag(
    off: LocalBackend, dataset: Dataset
) -> None:
    assert SLUG not in _listed(off, dataset)
    assert SLUG not in _listed(off, None)


@pytest.mark.parametrize(
    "tool",
    ["validate_recipe", "suggest_steps", "patch_recipe", "plan_recipe"],
)
def test_a_read_of_a_definition_recipe_is_refused_without_the_flag(
    tool: str, off: LocalBackend, dataset: Dataset
) -> None:
    cases: dict[str, dict[str, object]] = {
        "validate_recipe": {"recipe": DEFINITION_RECIPE},
        "suggest_steps": {
            "recipe": DEFINITION_RECIPE,
            "mode": "attach",
            "step": "d",
        },
        "patch_recipe": {
            "dataset": _uuid(dataset),
            "base": DEFINITION_RECIPE,
            "ops": [{"op": "set_param", "step": "d", "path": "scale", "value": 2.0}],
        },
        "plan_recipe": {"dataset": _uuid(dataset), "recipe": DEFINITION_RECIPE},
    }

    refused = _not_allowed(call_tool(off, tool, cases[tool]))

    assert refused.detail == {"tool": tool, "slug": SLUG}


def test_a_patch_adding_a_definition_is_refused_and_saves_nothing_without_the_flag(
    off: LocalBackend, dataset: Dataset
) -> None:
    saved = recipe_path(dataset.base_dir, "0" * 64).parent
    before = sorted(saved.rglob("*")) if saved.exists() else []
    added = {"id": "d", "type": "feature", "feature": SLUG, "inputs": ["tracks"]}

    refused = call_tool(
        off,
        "patch_recipe",
        {
            "dataset": _uuid(dataset),
            "base": SPEED_RECIPE,
            "ops": [{"op": "add_step", "document": added}],
        },
    )

    assert _not_allowed(refused).detail == {"tool": "patch_recipe", "slug": SLUG}
    assert (sorted(saved.rglob("*")) if saved.exists() else []) == before


def test_a_submission_naming_a_definition_is_refused_without_the_flag(
    off: LocalBackend, dataset: Dataset, starter: RecordingStarter
) -> None:
    refused = call_tool(
        off,
        "submit_pipeline",
        {"dataset": _uuid(dataset), "recipe": DEFINITION_RECIPE, "confirm": True},
    )

    _ = _not_allowed(refused)
    assert starter.started == []


def test_a_saved_definition_recipe_named_by_digest_is_refused_without_the_flag(
    off: LocalBackend, dataset: Dataset, starter: RecordingStarter
) -> None:
    recipe = read_recipe(DEFINITION_RECIPE)
    _ = save_recipe(dataset.base_dir, recipe)
    named = {"digest": recipe_digest(recipe)}

    validated = call_tool(
        off, "validate_recipe", {"dataset": _uuid(dataset), "recipe": named}
    )
    submitted = call_tool(
        off,
        "submit_pipeline",
        {"dataset": _uuid(dataset), "recipe": named, "confirm": True},
    )

    _ = _not_allowed(validated)
    _ = _not_allowed(submitted)
    assert starter.started == []


def test_a_built_in_recipe_shows_no_definition_without_the_flag(
    off: LocalBackend, dataset: Dataset
) -> None:
    suggested = call_tool(
        off,
        "suggest_steps",
        {
            "dataset": _uuid(dataset),
            "recipe": SPEED_RECIPE,
            "mode": "attach",
            "step": "speed",
        },
    )
    planned = call_tool(
        off, "plan_recipe", {"dataset": _uuid(dataset), "recipe": SPEED_RECIPE}
    )

    for result in (suggested, planned):
        assert not isinstance(result, ErrorEnvelope), result
        dumped = result.model_dump_json()
        assert SLUG not in dumped
        assert "def." not in dumped


def test_each_call_keeps_its_scope_disabled_without_the_flag(
    off: LocalBackend, dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _noting_scopes(monkeypatch)

    _ = call_tool(
        off, "plan_recipe", {"dataset": _uuid(dataset), "recipe": SPEED_RECIPE}
    )

    assert seen
    for scope in seen:
        assert scope is not None
        assert scope.base_dir is not None
        assert (scope.base_dir.resolve(), scope.load, scope.allowed) == (
            dataset.base_dir.resolve(),
            False,
            False,
        )
    assert current_scope() is None


def test_a_callers_disabled_scope_stays_disabled_with_the_flag(
    on: LocalBackend, dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _noting_scopes(monkeypatch)

    with definition_scope(None, load=False, allowed=False) as outer:
        refused = call_tool(
            on, "plan_recipe", {"dataset": _uuid(dataset), "recipe": SPEED_RECIPE}
        )
        definition = call_tool(
            on,
            "validate_recipe",
            {"dataset": _uuid(dataset), "recipe": DEFINITION_RECIPE},
        )
        assert current_scope() is outer

    assert not isinstance(refused, ErrorEnvelope), refused
    _ = _not_allowed(definition)
    assert seen
    assert all(scope is outer for scope in seen)


def _noting_scopes(monkeypatch: pytest.MonkeyPatch) -> list[DefinitionScope | None]:
    """Note the definition scope that each plan validates its recipe under."""
    seen: list[DefinitionScope | None] = []

    def note(*_args: object, **_kwargs: object) -> None:
        seen.append(current_scope())

    monkeypatch.setattr(
        plan_module,
        "reject_unless_valid",
        observed(plan_module.reject_unless_valid, note),
    )
    return seen


# ------------------------------------------------------------------- flag on


def test_define_feature_returns_its_record(on: LocalBackend, dataset: Dataset) -> None:
    defined = call_tool(
        on,
        "manage_dataset",
        {
            "action": "define_feature",
            "dataset": _uuid(dataset),
            "name": "edited-distance",
            "source": EDITED_SOURCE.replace(MINIMAL_NAME, "edited-distance"),
        },
    )
    again = call_tool(
        on,
        "manage_dataset",
        {
            "action": "define_feature",
            "dataset": _uuid(dataset),
            "name": "edited-distance",
            "source": EDITED_SOURCE.replace(MINIMAL_NAME, "edited-distance"),
        },
    )

    assert isinstance(defined, DatasetChange), defined
    assert isinstance(again, DatasetChange), again
    assert defined.action == "define_feature"
    assert defined.definition is not None
    assert (defined.definition.name, defined.definition.reused) == (
        "edited-distance",
        False,
    )
    assert again.definition is not None
    assert again.definition.reused
    assert again.definition.digest == defined.definition.digest
    assert "def:edited-distance" in _listed(on, dataset)
    assert not imported_definitions()


def test_a_refused_source_returns_its_findings(
    on: LocalBackend, dataset: Dataset
) -> None:
    refused = call_tool(
        on,
        "manage_dataset",
        {
            "action": "define_feature",
            "dataset": _uuid(dataset),
            "name": "broken-distance",
            "source": "this is not python\n",
        },
    )

    assert isinstance(refused, ErrorEnvelope), refused
    assert (refused.kind, refused.reason) == ("invalid", "definition_invalid")
    assert refused.detail is not None
    findings = refused.detail["findings"]
    assert isinstance(findings, list)
    assert findings


@pytest.mark.parametrize("missing", ["name", "source"])
def test_define_feature_names_a_missing_field(
    missing: str, on: LocalBackend, dataset: Dataset
) -> None:
    arguments: dict[str, object] = {
        "action": "define_feature",
        "dataset": _uuid(dataset),
        "name": "another-distance",
        "source": MINIMAL_SOURCE,
    }
    del arguments[missing]

    refused = call_tool(on, "manage_dataset", arguments)

    assert isinstance(refused, ErrorEnvelope), refused
    assert (refused.reason, refused.detail) == (
        "field_required",
        {"action": "define_feature", "field": missing},
    )


def test_define_feature_is_refused_on_a_managed_dataset(
    dataset: Dataset, starter: RecordingStarter
) -> None:
    with dataset.manifest_path.open("a", encoding="utf-8") as manifest:
        _ = manifest.write("managed_by: mosaic-api\n")
    backend = _backend(dataset, starter, allowed=True)

    refused = call_tool(
        backend,
        "manage_dataset",
        {
            "action": "define_feature",
            "dataset": _uuid(dataset),
            "name": "another-distance",
            "source": MINIMAL_SOURCE,
        },
    )

    assert isinstance(refused, ErrorEnvelope), refused
    assert refused.reason == "dataset_managed"


def test_a_submission_with_the_flag_imports_no_definition(
    on: LocalBackend,
    dataset: Dataset,
    starter: RecordingStarter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = noting_definition_loads(monkeypatch, step_module)

    submitted = call_tool(
        on,
        "submit_pipeline",
        {"dataset": _uuid(dataset), "recipe": DEFINITION_RECIPE, "confirm": True},
    )

    assert isinstance(submitted, SubmittedPipeline), submitted
    assert len(starter.started) == 1
    assert [event for event in events if event[0] == "load"] == []
    assert not imported_definitions()


# ---------------------------------------------------- two datasets, one server


def test_each_dataset_lists_its_own_definitions(
    tmp_path: Path, starter: RecordingStarter
) -> None:
    holding = defined_dataset(tmp_path / "holding", {MINIMAL_NAME: MINIMAL_SOURCE})
    other = defined_dataset(tmp_path / "other", {})
    backend = LocalBackend(
        [holding.manifest_path, other.manifest_path],
        warm=False,
        start_driver=starter,
        allow_feature_definitions=True,
    )

    validated = call_tool(
        backend,
        "validate_recipe",
        {"dataset": _uuid(other), "recipe": DEFINITION_RECIPE},
    )

    assert SLUG in _listed(backend, holding)
    assert SLUG not in _listed(backend, other)
    assert SLUG not in _listed(backend, None)
    assert isinstance(validated, ValidationReport), validated
    assert not validated.valid
    assert [(problem.step, problem.where) for problem in validated.problems] == [
        ("d", "feature")
    ]
    assert "feature_definitions_not_allowed" not in validated.problems[0].message
    assert not imported_definitions()


def test_a_whole_definition_recipe_without_a_dataset_names_the_dataset_it_needs(
    on: LocalBackend,
) -> None:
    validated = call_tool(on, "validate_recipe", {"recipe": DEFINITION_RECIPE})

    assert isinstance(validated, ValidationReport), validated
    (problem,) = validated.problems
    assert (problem.step, problem.where) == ("d", "feature")
    assert "which only a dataset holds" in problem.message


# ------------------------------------------------------------- running server


def test_a_definition_added_while_serving_is_listed_at_the_next_call(
    dataset: Dataset, starter: RecordingStarter
) -> None:
    backend = LocalBackend(
        [dataset.manifest_path], start_driver=starter, allow_feature_definitions=True
    )
    backend.wait_until_warm()
    assert "def:later-distance" not in _listed(backend, dataset)

    _ = define_feature(
        dataset,
        "later-distance",
        MINIMAL_SOURCE.replace(MINIMAL_NAME, "later-distance"),
    )

    assert "def:later-distance" in _listed(backend, dataset)
    assert not imported_definitions()


# ------------------------------------------------------------- report resource


def _read_report(backend: LocalBackend, dataset: Dataset, run_id: str) -> object:
    (report,) = (spec for spec in RESOURCES if spec.name == "report")
    return report.read(
        backend, {"id": _uuid(dataset), "name": STORAGE, "run_id": run_id}
    )


@pytest.fixture
def reported(tmp_path: Path) -> tuple[Dataset, str]:
    """A dataset whose definition names a report, and a run of it holding one."""
    dataset = defined_dataset(tmp_path, {MINIMAL_NAME: REPORT_SOURCE})
    run_id = "1-0123456789"
    run_root = feature_run_root(dataset, STORAGE, run_id)
    run_root.mkdir(parents=True)
    _ = (run_root / "report.json").write_text('{"rows": 3}', encoding="utf-8")
    return dataset, run_id


def test_a_definition_runs_report_reads_with_the_flag(
    reported: tuple[Dataset, str], starter: RecordingStarter
) -> None:
    dataset, run_id = reported
    backend = _backend(dataset, starter, allowed=True)

    read = _read_report(backend, dataset, run_id)

    assert isinstance(read, ResourceText)
    assert json.loads(read.text)["report"] == {"rows": 3}
    assert not imported_definitions()


def test_a_definition_runs_report_is_refused_without_the_flag(
    reported: tuple[Dataset, str], starter: RecordingStarter
) -> None:
    dataset, run_id = reported
    backend = _backend(dataset, starter, allowed=False)

    with pytest.raises(FeatureDefinitionsNotAllowed) as refused:
        _ = _read_report(backend, dataset, run_id)

    assert refused.value.error_reason() == "feature_definitions_not_allowed"
