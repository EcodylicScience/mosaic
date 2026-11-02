"""``--recipe`` as a bare file path, beside ``@file.json`` and inline JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from tests.helpers import invoke_json, make_dataset

runner = CliRunner()

RECIPE = {
    "schema_version": 1,
    "name": "speed",
    "steps": [
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "inputs": ["tracks"],
        }
    ],
}


@pytest.fixture
def recipe_file(tmp_path: Path) -> Path:
    path = tmp_path / "recipe.json"
    _ = path.write_text(json.dumps(RECIPE), encoding="utf-8")
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return make_dataset(tmp_path / "ds", name="ds")


def _plan(dataset: Dataset, recipe: str) -> dict[str, object]:
    return invoke_json(
        [
            "pipeline",
            "plan",
            "--recipe",
            recipe,
            "--manifest",
            str(dataset.manifest_path),
            "--json",
        ]
    )


def test_a_bare_path_plans_as_the_at_file_form_and_inline_json_do(
    dataset: Dataset, recipe_file: Path
) -> None:
    bare = _plan(dataset, str(recipe_file))
    at_file = _plan(dataset, f"@{recipe_file}")
    inline = _plan(dataset, json.dumps(RECIPE))

    assert bare["fingerprint"] == at_file["fingerprint"] == inline["fingerprint"]
    assert bare["recipe_digest"] == at_file["recipe_digest"]


def test_validate_reads_a_bare_path(recipe_file: Path) -> None:
    report = invoke_json(
        ["pipeline", "validate", "--recipe", str(recipe_file), "--json"]
    )

    assert report["valid"] is True


def test_a_value_that_names_no_file_and_is_not_json_says_both(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    result = runner.invoke(app, ["pipeline", "validate", "--recipe", str(missing)])

    assert result.exit_code == 1
    assert f"no file named {str(missing)!r}, and not inline JSON" in result.stderr


def test_long_inline_json_with_leading_whitespace_is_read_as_json() -> None:
    padded = {**RECIPE, "name": "x" * 2_000}

    report = invoke_json(
        ["pipeline", "validate", "--recipe", "  " + json.dumps(padded), "--json"]
    )

    assert report["valid"] is True


def test_a_value_too_long_to_be_a_path_is_read_as_inline_json() -> None:
    result = runner.invoke(app, ["pipeline", "validate", "--recipe", "x" * 5_000])

    assert result.exit_code == 1
    assert "not inline JSON" in result.stderr


def test_a_bare_path_that_is_not_json_names_the_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    _ = broken.write_text("{not json", encoding="utf-8")

    result = runner.invoke(app, ["pipeline", "validate", "--recipe", str(broken)])

    assert result.exit_code == 1
    assert f"Invalid JSON ({broken})" in result.stderr
