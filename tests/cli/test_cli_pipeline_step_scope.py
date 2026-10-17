"""``--step-scope`` and ``--group`` on ``mosaic pipeline plan``, ``run`` and ``submit``.

A step scope narrows one step of a request, and ``--group`` narrows the
request to whole groups, as the MCP ``{"groups": [...]}`` selector does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from tests.helpers import calms21_dataset, invoke_json, tree_snapshot

runner = CliRunner()

RECIPE = {
    "schema_version": 1,
    "name": "split",
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


@pytest.fixture
def split(tmp_path: Path) -> Dataset:
    ds = calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": {"s1": 30, "s2": 30}}, "test": {"a": {"s3": 30}}},
        seed=2,
    )
    _ = ds.convert_all_tracks()
    return ds


def _recipe_file(tmp_path: Path) -> str:
    path = tmp_path / "recipe.json"
    _ = path.write_text(json.dumps(RECIPE), encoding="utf-8")
    return f"@{path}"


def _plan(ds: Dataset, recipe: str, *extra: str) -> dict[str, object]:
    return invoke_json(
        [
            "pipeline",
            "plan",
            "--recipe",
            recipe,
            "--manifest",
            str(ds.manifest_path),
            *extra,
            "--json",
        ]
    )


def _targets(plan: dict[str, object]) -> dict[str, int]:
    from mosaic.core.pipeline.graph import PlanWire

    wire = PlanWire.model_validate(plan)
    return {step.step: step.coverage.target for step in wire.steps}


def test_plan_with_step_scope_narrows_the_step(split: Dataset, tmp_path: Path) -> None:
    before = tree_snapshot(split.base_dir)

    plan = _plan(
        split,
        _recipe_file(tmp_path),
        "--step-scope",
        'templates={"groups": ["train"]}',
    )

    assert _targets(plan) == {"speed": 3, "templates": 2}
    assert tree_snapshot(split.base_dir) == before


@pytest.mark.parametrize(
    "token",
    ["templates", '={"groups": ["train"]}', "templates=[1, 2]", 'templates={"x": 1}'],
)
def test_a_malformed_step_scope_is_refused(
    split: Dataset, tmp_path: Path, token: str
) -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            "plan",
            "--recipe",
            _recipe_file(tmp_path),
            "--manifest",
            str(split.manifest_path),
            "--step-scope",
            token,
        ],
    )

    assert result.exit_code == 1
    assert "--step-scope" in result.stderr


def test_step_scope_with_request_is_refused(split: Dataset) -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--request",
            "01M00000000000000000000000",
            "--manifest",
            str(split.manifest_path),
            "--step-scope",
            'templates={"groups": ["train"]}',
        ],
    )

    assert result.exit_code == 1
    assert "not used with --request" in result.stderr


def test_group_option_selects_groups(split: Dataset, tmp_path: Path) -> None:
    plan = _plan(split, _recipe_file(tmp_path), "--group", "test")
    submitted = invoke_json(
        [
            "pipeline",
            "submit",
            "--recipe",
            _recipe_file(tmp_path),
            "--manifest",
            str(split.manifest_path),
            "--group",
            "train",
            "--step-scope",
            'templates={"entries": [["train", "s1"]]}',
            "--json",
        ]
    )

    assert plan["scope_size"] == 1
    assert submitted["scope"] == [["train", "s1"], ["train", "s2"]]
    assert submitted["step_scopes"] == {
        "templates": {"entries": [["train", "s1"]], "groups": None, "sequences": None}
    }
