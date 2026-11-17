"""``--tag`` on the pipeline verbs: the entries a sequence tag selects are the scope.

The library's selection is in ``tests/core/annotations/test_tag_selection.py`` and the
JSON selector of a step scope in ``tests/cli/test_cli_pipeline_step_scope.py``. These
check what the flag adds: the lowering to entries, the revision the request records,
and the refusals of the flag beside another way of naming entries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.typed_attribute import TypedAttributeType
from mosaic.core.pipeline.graph import PlanWire, Request, load_request
from tests.helpers import (
    calms21_dataset,
    invoke_json,
    tag_table,
    tree_snapshot,
    write_tags,
)

runner = CliRunner()

RECIPE = {
    "schema_version": 1,
    "name": "tagged",
    "steps": [
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "inputs": ["tracks"],
        },
        {
            "id": "templates",
            "type": "feature",
            "feature": "extract-templates",
            "inputs": [{"step": "speed"}],
            "params": {"n_templates": 4},
        },
    ],
}

TYPES: Final[dict[str, TypedAttributeType]] = {"site": "categorical", "focal": "label"}

SITES = {
    ("train", "s1"): {"site": "a", "focal": None},
    ("train", "s2"): {"site": "b"},
    ("test", "s3"): {"site": "a"},
}


@pytest.fixture
def tagged(tmp_path: Path) -> Dataset:
    """Converted tracks of ``train/s1``, ``train/s2`` and ``test/s3``, tagged by site."""
    ds = calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": {"s1": 30, "s2": 30}}, "test": {"a": {"s3": 30}}},
        seed=2,
    )
    _ = ds.convert_all_tracks()
    _ = write_tags(ds, tag_table(SITES, TYPES))
    return ds


@pytest.fixture
def recipe(tmp_path: Path) -> str:
    path = tmp_path / "recipe.json"
    _ = path.write_text(json.dumps(RECIPE), encoding="utf-8")
    return f"@{path}"


def _verb(verb: str, ds: Dataset, recipe: str, *extra: str) -> list[str]:
    return [
        "pipeline",
        verb,
        "--recipe",
        recipe,
        "--manifest",
        str(ds.manifest_path),
        *extra,
        "--json",
    ]


def _run_ids(plan: dict[str, object]) -> dict[str, str | None]:
    return {step.step: step.run_id for step in PlanWire.model_validate(plan).steps}


def _request(ds: Dataset, submitted: dict[str, object]) -> Request:
    return load_request(ds.base_dir, str(submitted["request_id"]))


def _planned_ids(plan: dict[str, object]) -> dict[str, str]:
    """The run ids of a plan as ``pipeline submit`` prints them, ``-`` for none."""
    return {step: run_id or "-" for step, run_id in _run_ids(plan).items()}


def _step_run_ids(submitted: dict[str, object]) -> dict[str, str]:
    steps = TypeAdapter(list[dict[str, object]]).validate_python(submitted["steps"])
    return {str(step["step"]): str(step["run_id"]) for step in steps}


def test_plan_by_tag_plans_the_run_ids_of_the_same_entries_named_one_by_one(
    tagged: Dataset, recipe: str
) -> None:
    before = tree_snapshot(tagged.base_dir)

    by_tag = invoke_json(_verb("plan", tagged, recipe, "--tag", "site=a"))
    by_name = invoke_json(
        _verb("plan", tagged, recipe, "--entry", "train:s1", "--entry", "test:s3")
    )

    assert by_tag["scope_size"] == 2
    assert _run_ids(by_tag) == _run_ids(by_name)
    assert tree_snapshot(tagged.base_dir) == before


def test_submit_by_tag_records_the_entries_and_the_revision_it_read(
    tagged: Dataset, recipe: str
) -> None:
    submitted = invoke_json(_verb("submit", tagged, recipe, "--tag", "site=a"))

    assert submitted["scope"] == [["test", "s3"], ["train", "s1"]]
    request = _request(tagged, submitted)
    assert request.scope.entries == [("test", "s3"), ("train", "s1")]
    assert request.tag_revision is not None
    assert request.tag_revision.key == "dataset"
    assert request.tag_revision.revision == 1
    text = Path(str(submitted["request_path"])).read_text(encoding="utf-8")
    assert "site" not in text


def test_a_revision_written_after_the_submit_leaves_the_requests_run_ids(
    tagged: Dataset, recipe: str
) -> None:
    submitted = invoke_json(_verb("submit", tagged, recipe, "--tag", "site=a"))
    submitted_ids = _step_run_ids(submitted)
    moved = {**SITES, ("train", "s2"): {"site": "a"}, ("test", "s3"): {"site": "b"}}
    _ = write_tags(tagged, tag_table(moved, TYPES))

    replanned = invoke_json(_verb("plan", tagged, recipe, "--tag", "site=a"))
    by_name = invoke_json(
        _verb("plan", tagged, recipe, "--entry", "train:s1", "--entry", "test:s3")
    )

    assert _planned_ids(replanned) != submitted_ids
    assert _planned_ids(by_name) == submitted_ids
    recorded = _request(tagged, submitted).tag_revision
    assert recorded is not None
    assert recorded.revision == 1


def test_a_tag_beside_a_group_keeps_the_selected_entries_of_that_group(
    tagged: Dataset, recipe: str
) -> None:
    plan = invoke_json(
        _verb("plan", tagged, recipe, "--tag", "site=a", "--group", "train")
    )

    assert plan["scope_size"] == 1
    assert _run_ids(plan) == _run_ids(
        invoke_json(_verb("plan", tagged, recipe, "--entry", "train:s1"))
    )


def test_a_group_with_no_selected_entry_is_refused(
    tagged: Dataset, recipe: str
) -> None:
    result = runner.invoke(
        app,
        _verb("plan", tagged, recipe, "--tag", "site=b", "--group", "test"),
    )

    assert result.exit_code == 1
    assert json.loads(result.stdout)["kind"] == "invalid"


def test_a_tag_beside_an_entry_is_refused(tagged: Dataset, recipe: str) -> None:
    before = tree_snapshot(tagged.base_dir)

    result = runner.invoke(
        app,
        _verb("plan", tagged, recipe, "--tag", "site=a", "--entry", "test:s3"),
    )

    assert result.exit_code == 1
    assert "--entry" in result.stderr
    assert "--tag" in result.stderr
    assert tree_snapshot(tagged.base_dir) == before


def test_a_tag_named_without_a_value_selects_a_label(
    tagged: Dataset, recipe: str
) -> None:
    plan = invoke_json(_verb("plan", tagged, recipe, "--tag", "focal"))

    assert plan["scope_size"] == 1
    assert _run_ids(plan) == _run_ids(
        invoke_json(_verb("plan", tagged, recipe, "--entry", "train:s1"))
    )


def test_a_tag_with_no_name_is_refused(tagged: Dataset, recipe: str) -> None:
    result = runner.invoke(app, _verb("plan", tagged, recipe, "--tag", "=a"))

    assert result.exit_code == 1
    assert "--tag" in result.stderr


def test_repeated_values_of_one_tag_are_alternatives(
    tagged: Dataset, recipe: str
) -> None:
    plan = invoke_json(
        _verb("plan", tagged, recipe, "--tag", "site=a", "--tag", "site=b")
    )

    assert plan["scope_size"] == 3


def test_a_tag_selection_of_the_whole_scope_and_a_step_reads_one_revision(
    tagged: Dataset, recipe: str
) -> None:
    plan = invoke_json(
        _verb(
            "plan",
            tagged,
            recipe,
            "--tag",
            "site=a",
            "--step-scope",
            'templates={"sequence_tags": {"focal": []}}',
        )
    )

    targets = {
        step.step: step.coverage.target for step in PlanWire.model_validate(plan).steps
    }
    assert targets == {"speed": 2, "templates": 1}


def test_verify_by_tag_passes_after_a_run_by_tag(tagged: Dataset, recipe: str) -> None:
    ran = runner.invoke(app, _verb("run", tagged, recipe, "--tag", "site=a"))
    assert ran.exit_code == 0, ran.stderr
    before = tree_snapshot(tagged.base_dir)

    result = runner.invoke(app, _verb("verify", tagged, recipe, "--tag", "site=a"))

    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True
    assert tree_snapshot(tagged.base_dir) == before


def test_run_by_tag_runs_over_the_selection(tagged: Dataset, recipe: str) -> None:
    ran = runner.invoke(app, _verb("run", tagged, recipe, "--tag", "site=a"))
    assert ran.exit_code == 0, ran.stderr

    by_tag = invoke_json(_verb("plan", tagged, recipe, "--tag", "site=a"))
    other = invoke_json(_verb("plan", tagged, recipe, "--tag", "site=b"))

    assert {
        step.step: step.status for step in PlanWire.model_validate(by_tag).steps
    } == {
        "speed": "complete",
        "templates": "complete",
    }
    assert {
        step.step: step.status for step in PlanWire.model_validate(other).steps
    } != {
        "speed": "complete",
        "templates": "complete",
    }


def test_a_tag_is_not_used_with_a_request(tagged: Dataset, recipe: str) -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--request",
            "any",
            "--manifest",
            str(tagged.manifest_path),
            "--tag",
            "site=a",
        ],
    )

    assert result.exit_code == 1
    assert "--tag" in result.stderr
