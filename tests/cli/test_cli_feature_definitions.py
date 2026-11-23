"""The CLI verbs that list, describe, validate, plan and run a feature definition.

A definition is named by ``def:<name>`` and found through the dataset a verb is
pointed at. Every verb that resolves one holds a ``load=False`` definition scope, so
it reads sidecars and imports nothing, except ``run --feature``, which loads the one
definition it builds. A caller's disabled scope stays disabled through each.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline import run as run_module
from mosaic.core.pipeline.definitions import loader
from mosaic.core.pipeline.definitions.define import define_feature
from mosaic.core.pipeline.definitions.scope import (
    DefinitionScope,
    current_scope,
    definition_scope,
)
from mosaic.core.pipeline.definitions.store import definition_module_name
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.graph import (
    CatalogIndexEntry,
    StepTypeDetail,
    ValidationReport,
)
from mosaic.core.pipeline.graph import run as graph_run
from mosaic.core.pipeline.graph import status as status_module
from mosaic.core.pipeline.graph import step as step_module
from mosaic.core.pipeline.index import feature_run_root
from tests.helpers import (
    APPLYING_NAME,
    APPLYING_SECOND_SOURCE,
    APPLYING_SOURCE,
    MINIMAL_NAME,
    MINIMAL_SOURCE,
    NARROWING_NAME,
    NARROWING_SOURCE,
    defined_dataset,
    imported_definitions,
    invoke_json,
    invoke_ok,
    noting_definition_loads,
)

pytestmark = pytest.mark.usefixtures("fresh_definitions")

SLUG: Final = f"def:{MINIMAL_NAME}"
STORAGE: Final = f"def.{MINIMAL_NAME}__from__tracks"
MOVED_SOURCE: Final = MINIMAL_SOURCE.replace(
    "scale: float = 1.0", "scale: float = 2.0"
).replace('version = "1"', 'version = "2"')
"""A second version of ``MINIMAL_SOURCE``, which becomes the current one."""
SOURCES: Final = {MINIMAL_NAME: MINIMAL_SOURCE, NARROWING_NAME: NARROWING_SOURCE}
INDEX_ROWS: Final = TypeAdapter(list[CatalogIndexEntry])
PLAN_STEPS: Final = TypeAdapter(list[dict[str, object]])


def _recipe_file(tmp_path: Path, *names: str) -> Path:
    """A recipe of one independent step per definition name, each over the tracks."""
    document = {
        "schema_version": 1,
        "name": "definitions",
        "steps": [
            {
                "id": f"step{index}",
                "type": "feature",
                "feature": f"def:{name}",
                "inputs": ["tracks"],
            }
            for index, name in enumerate(names)
        ],
    }
    path = tmp_path / "recipe.json"
    _ = path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return defined_dataset(tmp_path, SOURCES)


@pytest.fixture
def manifest(dataset: Dataset) -> str:
    return str(dataset.manifest_path)


def _refused(args: list[str]) -> ErrorEnvelope:
    """Run *args*, assert exit 1, and return the error printed on standard output."""
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, result.output
    return ErrorEnvelope.model_validate_json(result.stdout)


# --- features list ---------------------------------------------------------------------


def test_features_list_with_a_manifest_shows_each_definition(manifest: str) -> None:
    result = invoke_ok(["features", "list", "--manifest", manifest, "--json"])

    rows = {row.name: row for row in INDEX_ROWS.validate_json(result.stdout)}

    row = rows[SLUG]
    assert row.definition is not None
    assert row.definition.name == MINIMAL_NAME
    assert row.version == "1"
    assert f"def:{NARROWING_NAME}" in rows
    assert "speed-angvel" in rows
    assert rows["speed-angvel"].definition is None


def test_features_list_without_a_manifest_lists_the_built_ins(manifest: str) -> None:
    result = invoke_ok(["features", "list", "--json"])

    names = [row.name for row in INDEX_ROWS.validate_json(result.stdout)]

    assert "speed-angvel" in names
    assert not any(name.startswith("def:") for name in names)


def test_the_table_gives_each_definition_its_digest_prefix_and_date(
    dataset: Dataset, manifest: str
) -> None:
    rows = INDEX_ROWS.validate_json(
        invoke_ok(["features", "list", "--manifest", manifest, "--json"]).stdout
    )
    (definition,) = [row.definition for row in rows if row.name == SLUG]
    assert definition is not None
    assert definition.digest is not None
    assert definition.defined_on is not None

    table = invoke_ok(["features", "list", "--manifest", manifest]).stdout

    header, *lines = table.splitlines()
    columns = header.split("\t")
    (line,) = [line for line in lines if line.startswith(f"{SLUG}\t")]
    cells = dict(zip(columns, line.split("\t"), strict=True))
    assert cells["digest"] == definition.digest[:12]
    assert cells["defined"] == definition.defined_on.isoformat()
    builtin = next(line for line in lines if line.startswith("speed-angvel\t"))
    assert dict(zip(columns, builtin.split("\t"), strict=True))["digest"] == ""


def test_the_table_without_a_manifest_has_no_definition_columns() -> None:
    header = invoke_ok(["features", "list"]).stdout.splitlines()[0]

    assert "digest" not in header.split("\t")


# --- features describe -----------------------------------------------------------------


def test_describing_a_definition_without_a_manifest_names_it_unknown() -> None:
    result = CliRunner().invoke(app, ["features", "describe", SLUG])

    assert result.exit_code == 1
    assert f"Unknown feature '{SLUG}'" in result.stderr


def test_describing_a_definition_with_a_manifest_gives_its_rebuilt_params(
    manifest: str,
) -> None:
    result = invoke_ok(["features", "describe", SLUG, "--manifest", manifest, "--json"])

    detail = StepTypeDetail.model_validate_json(result.stdout)

    assert detail.name == SLUG
    assert detail.definition is not None
    assert detail.definition.name == MINIMAL_NAME
    (scale,) = [row for row in detail.params if row.path == "scale"]
    assert scale.default == 1.0
    assert not imported_definitions()


def test_describing_an_unknown_definition_lists_the_datasets_definitions(
    manifest: str,
) -> None:
    result = CliRunner().invoke(
        app, ["features", "describe", "def:absent", "--manifest", manifest]
    )

    assert result.exit_code == 1
    assert "Unknown feature 'def:absent'" in result.stderr
    assert SLUG in result.stderr
    assert f"def:{NARROWING_NAME}" in result.stderr


# --- pipeline validate -----------------------------------------------------------------


def test_validating_a_definition_recipe_without_a_manifest_reports_the_step(
    tmp_path: Path,
) -> None:
    recipe = _recipe_file(tmp_path, MINIMAL_NAME)

    result = CliRunner().invoke(
        app, ["pipeline", "validate", "--recipe", f"@{recipe}", "--json"]
    )

    assert result.exit_code == 1
    report = ValidationReport.model_validate_json(result.stdout)
    (problem,) = report.problems
    assert (problem.step, problem.where) == ("step0", "feature")


def test_validating_a_definition_recipe_with_a_manifest_passes(
    tmp_path: Path, manifest: str
) -> None:
    recipe = _recipe_file(tmp_path, MINIMAL_NAME, NARROWING_NAME)

    result = invoke_ok(
        [
            "pipeline",
            "validate",
            "--recipe",
            f"@{recipe}",
            "--manifest",
            manifest,
            "--json",
        ]
    )

    assert ValidationReport.model_validate_json(result.stdout).valid
    assert not imported_definitions()


# --- run --feature ---------------------------------------------------------------------


def test_running_a_definition_writes_its_storage(
    dataset: Dataset, manifest: str
) -> None:
    payload = invoke_json(["run", "--feature", SLUG, "--manifest", manifest, "--json"])

    assert payload["feature"] == STORAGE
    run_id = payload["run_id"]
    assert isinstance(run_id, str)
    root = feature_run_root(dataset, STORAGE, run_id)
    assert sorted(path.name for path in root.glob("*.parquet")) == [
        "g__s1.parquet",
        "g__s2.parquet",
    ]
    assert root.parent == dataset.base_dir / "features" / STORAGE


def test_running_an_unknown_definition_lists_the_datasets_definitions(
    manifest: str,
) -> None:
    result = CliRunner().invoke(
        app, ["run", "--feature", "def:absent", "--manifest", manifest, "--json"]
    )

    assert result.exit_code == 1
    assert "Unknown feature 'def:absent'" in result.stderr
    assert SLUG in result.stderr
    assert f"def:{NARROWING_NAME}" in result.stderr


def test_running_a_definition_loads_it_only_to_build_it(
    manifest: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run itself holds no ``load=True`` scope and imports nothing more."""
    loading: list[bool | None] = []
    running: list[DefinitionScope | None] = []
    load = loader.load_definition

    def noting_load(definitions_dir: Path, name: str, digest: str) -> type:
        scope = current_scope()
        loading.append(None if scope is None else scope.load)
        return load(definitions_dir, name, digest)

    monkeypatch.setattr(loader, "load_definition", noting_load)
    _seen_by(monkeypatch, run_module, "run_feature", running)

    _ = invoke_json(["run", "--feature", SLUG, "--manifest", manifest, "--json"])

    assert loading == [True]
    (scope,) = running
    assert scope is None or not scope.load
    assert imported_definitions() == [
        definition_module_name(MINIMAL_NAME, _digest_of(MINIMAL_SOURCE))
    ]


def test_running_an_apply_of_another_versions_fit_is_refused(tmp_path: Path) -> None:
    applying = defined_dataset(tmp_path / "applying", {APPLYING_NAME: APPLYING_SOURCE})
    manifest = str(applying.manifest_path)
    fit_digest = _digest_of(APPLYING_SOURCE)
    fit = invoke_json(
        [
            "run",
            "--feature",
            f"def:{APPLYING_NAME}@{fit_digest}",
            "--manifest",
            manifest,
            "--json",
        ]
    )
    second = define_feature(applying, APPLYING_NAME, APPLYING_SECOND_SOURCE)
    model = {"feature": fit["feature"], "run_id": fit["run_id"]}

    error = _refused(
        [
            "run",
            "--feature",
            f"def:{APPLYING_NAME}@{second.digest}",
            "--params",
            json.dumps({"model": model}),
            "--manifest",
            manifest,
            "--json",
        ]
    )

    assert error.reason == "definition_model_refused"
    assert error.detail is not None
    assert error.detail["reason"] == "other_version"
    assert f"run this step as def:{APPLYING_NAME}@{fit_digest[:12]}" in error.message


def _seen_by(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    name: str,
    seen: list[DefinitionScope | None],
) -> None:
    """Replace ``module.name`` with a call to it that notes the scope in force."""
    original: Callable[..., object] = getattr(module, name)

    def noting(*args: object, **kwargs: object) -> object:
        seen.append(current_scope())
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, noting)


def _digest_of(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


# --- loading on demand -----------------------------------------------------------------


def test_pipeline_run_loads_each_definition_only_when_its_step_is_built(
    tmp_path: Path, manifest: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _recipe_file(tmp_path, MINIMAL_NAME, NARROWING_NAME)
    events = noting_definition_loads(monkeypatch, step_module)

    payload = invoke_json(
        ["pipeline", "run", "--recipe", f"@{recipe}", "--manifest", manifest, "--json"]
    )

    assert len(PLAN_STEPS.validate_python(payload["steps"])) == 2
    loads = [index for index, (kind, _) in enumerate(events) if kind == "load"]
    assert len(loads) == 2
    for index in loads:
        assert events[index - 1] == ("build", events[index][1])


def test_running_one_step_of_a_request_imports_only_its_definition(
    tmp_path: Path, manifest: str
) -> None:
    recipe = _recipe_file(tmp_path, MINIMAL_NAME, NARROWING_NAME)
    submitted = invoke_json(
        [
            "pipeline",
            "submit",
            "--recipe",
            f"@{recipe}",
            "--manifest",
            manifest,
            "--json",
        ]
    )
    request_id = submitted["request_id"]
    assert isinstance(request_id, str)
    assert not imported_definitions()

    payload = invoke_json(
        [
            "run",
            "--graph-request",
            request_id,
            "--step",
            "step1",
            "--manifest",
            manifest,
            "--json",
        ]
    )

    assert payload["state"] == "ran"
    assert imported_definitions() == [
        definition_module_name(NARROWING_NAME, _digest_of(NARROWING_SOURCE))
    ]


def test_sizing_a_definition_step_reads_the_version_its_request_pinned(
    tmp_path: Path, dataset: Dataset, manifest: str
) -> None:
    """The command sizes a step as the queue asks it to, with no scope entered."""
    recipe = _recipe_file(tmp_path, MINIMAL_NAME)
    submitted = invoke_json(
        [
            "pipeline",
            "submit",
            "--recipe",
            f"@{recipe}",
            "--manifest",
            manifest,
            "--json",
        ]
    )
    request_id = submitted["request_id"]
    assert isinstance(request_id, str)
    planned = PLAN_STEPS.validate_python(submitted["steps"])[0]["run_id"]
    _ = define_feature(dataset, MINIMAL_NAME, MOVED_SOURCE)
    assert current_scope() is None

    payload = invoke_json(
        [
            "pipeline",
            "input-size",
            "--request",
            request_id,
            "--step",
            "step0",
            "--manifest",
            manifest,
            "--json",
        ]
    )

    assert payload["inputs_ready"] is True
    assert payload["entries"] == 2
    assert payload["entries_to_compute"] == 2
    assert payload["run_id"] == planned
    assert not imported_definitions()


# --- a caller's disabled scope ---------------------------------------------------------


def _disabled_list(manifest: str, recipe: Path) -> None:
    result = invoke_ok(["features", "list", "--manifest", manifest, "--json"])
    names = [row.name for row in INDEX_ROWS.validate_json(result.stdout)]
    assert not any(name.startswith("def:") for name in names)


def _disabled_describe(manifest: str, recipe: Path) -> None:
    error = _refused(["features", "describe", SLUG, "--manifest", manifest, "--json"])
    assert error.reason == "feature_definitions_not_allowed"


def _disabled_validate(manifest: str, recipe: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "pipeline",
            "validate",
            "--recipe",
            f"@{recipe}",
            "--manifest",
            manifest,
            "--json",
        ],
    )
    assert result.exit_code == 1
    report = ValidationReport.model_validate_json(result.stdout)
    assert all("feature_definitions_not_allowed" in p.message for p in report.problems)
    assert report.problems


def _disabled_plan(manifest: str, recipe: Path) -> None:
    error = _refused(
        ["pipeline", "plan", "--recipe", f"@{recipe}", "--manifest", manifest, "--json"]
    )
    assert error.reason == "feature_definitions_not_allowed"


def _disabled_run_recipe(manifest: str, recipe: Path) -> None:
    error = _refused(
        ["pipeline", "run", "--recipe", f"@{recipe}", "--manifest", manifest, "--json"]
    )
    assert error.reason == "feature_definitions_not_allowed"


def _disabled_submit(manifest: str, recipe: Path) -> None:
    error = _refused(
        [
            "pipeline",
            "submit",
            "--recipe",
            f"@{recipe}",
            "--manifest",
            manifest,
            "--json",
        ]
    )
    assert error.reason == "feature_definitions_not_allowed"


def _disabled_run_feature(manifest: str, recipe: Path) -> None:
    error = _refused(["run", "--feature", SLUG, "--manifest", manifest, "--json"])
    assert error.reason == "feature_definitions_not_allowed"


DISABLED_VERBS: Final[dict[str, Callable[[str, Path], None]]] = {
    "features list": _disabled_list,
    "features describe": _disabled_describe,
    "pipeline validate": _disabled_validate,
    "pipeline plan": _disabled_plan,
    "pipeline run --recipe": _disabled_run_recipe,
    "pipeline submit": _disabled_submit,
    "run --feature": _disabled_run_feature,
}


@pytest.mark.parametrize("verb", sorted(DISABLED_VERBS))
@pytest.mark.parametrize("held", ["no-dataset", "same-dataset"])
def test_a_verb_inside_a_disabled_scope_keeps_it_disabled(
    verb: str,
    held: str,
    tmp_path: Path,
    dataset: Dataset,
    manifest: str,
) -> None:
    recipe = _recipe_file(tmp_path, MINIMAL_NAME)

    with definition_scope(
        None if held == "no-dataset" else dataset, load=False, allowed=False
    ) as outer:
        DISABLED_VERBS[verb](manifest, recipe)
        assert current_scope() is outer

    assert not imported_definitions()
    assert not (dataset.base_dir / "features" / STORAGE).exists()


@pytest.mark.parametrize("verb", ["status", "run --request", "reconcile"])
def test_a_reading_verb_holds_the_callers_disabled_scope(
    verb: str,
    tmp_path: Path,
    dataset: Dataset,
    manifest: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = _recipe_file(tmp_path, MINIMAL_NAME)
    submitted = invoke_json(
        [
            "pipeline",
            "submit",
            "--recipe",
            f"@{recipe}",
            "--manifest",
            manifest,
            "--json",
        ]
    )
    request_id = submitted["request_id"]
    assert isinstance(request_id, str)
    seen: list[DefinitionScope | None] = []
    arguments = {
        "status": ["pipeline", "status", "--request", request_id],
        "run --request": ["pipeline", "run", "--request", request_id],
        "reconcile": ["reconcile"],
    }[verb]
    watched = {
        "status": (status_module, "request_status_wire"),
        "run --request": (graph_run, "run_request"),
        "reconcile": (Dataset, "reconcile"),
    }[verb]
    _seen_by(monkeypatch, *watched, seen)

    with definition_scope(dataset, load=False, allowed=False) as outer:
        _ = CliRunner().invoke(app, [*arguments, "--manifest", manifest, "--json"])
        assert current_scope() is outer

    assert seen
    assert all(scope is outer for scope in seen)
