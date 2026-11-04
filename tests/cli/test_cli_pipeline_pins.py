"""``--pin`` and ``--applies-request`` on ``mosaic pipeline plan``, ``submit`` and ``run``.

``--pin STEP=RUN_ID`` serves a fitted step from a finished run, and
``--applies-request`` reuses the fit of a finished request over other
sequences. The two are alternatives, and a malformed pin is a usage error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    PlanWire,
    Recipe,
    Request,
    load_request,
    requests_root,
    run_pipeline,
)
from mosaic.core.scope import Scope
from tests.helpers import calms21_dataset, invoke_json

runner = CliRunner()

_ROWS = TypeAdapter(list[dict[str, object]])
"""The steps of a run, each an object."""

TRAIN = Scope(groups=["train"])

STEPS: list[dict[str, JsonValue]] = [
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
    {
        "id": "kmeans",
        "type": "feature",
        "feature": "global-kmeans",
        "inputs": [{"step": "speed"}],
        "params": {"templates": {"step": "templates"}, "k": 2},
    },
]


@pytest.fixture
def split(tmp_path: Path) -> Dataset:
    """Converted tracks of a train group of two sequences and a test group of one."""
    ds = calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": {"s1": 40, "s2": 30}}, "test": {"a": {"s3": 30}}},
        seed=5,
    )
    _ = ds.convert_all_tracks()
    return ds


@pytest.fixture
def fit(split: Dataset) -> tuple[Request, dict[str, str]]:
    """The finished request that fit the recipe over the train group, and its runs."""
    recipe = Recipe.model_validate(
        {"schema_version": 1, "name": "pins", "steps": STEPS}
    )
    run = run_pipeline(split, recipe, scope=TRAIN)
    runs = {outcome.step_id: outcome.run_id for outcome in run.outcomes}
    assert all(runs.values())
    return load_request(split.base_dir, run.request_id), runs


def _recipe_file(tmp_path: Path) -> str:
    path = tmp_path / "recipe.json"
    document = {"schema_version": 1, "name": "pins", "steps": STEPS}
    _ = path.write_text(json.dumps(document), encoding="utf-8")
    return f"@{path}"


def _args(verb: str, ds: Dataset, tmp_path: Path, *extra: str) -> list[str]:
    """The arguments of *verb* over the test group, without ``--json``."""
    return [
        "pipeline",
        verb,
        "--recipe",
        _recipe_file(tmp_path),
        "--manifest",
        str(ds.manifest_path),
        "--group",
        "test",
        *extra,
    ]


def _request_count(ds: Dataset) -> int:
    root = requests_root(ds.base_dir)
    return len(list(root.glob("*.json"))) if root.exists() else 0


def test_submit_pin_records_the_pin(
    split: Dataset, fit: tuple[Request, dict[str, str]], tmp_path: Path
) -> None:
    _, runs = fit

    submitted = invoke_json(
        [
            *_args(
                "submit", split, tmp_path, "--pin", f"templates={runs['templates']}"
            ),
            "--json",
        ]
    )

    request = load_request(split.base_dir, str(submitted["request_id"]))
    assert request.pins == {"templates": runs["templates"]}
    assert request.applies_request_id is None


def test_submit_applies_request_records_the_apply(
    split: Dataset, fit: tuple[Request, dict[str, str]], tmp_path: Path
) -> None:
    named, runs = fit

    submitted = invoke_json(
        [
            *_args("submit", split, tmp_path, "--applies-request", named.request_id),
            "--json",
        ]
    )

    request = load_request(split.base_dir, str(submitted["request_id"]))
    assert request.applies_request_id == named.request_id
    assert request.pins == {"templates": runs["templates"]}
    assert request.locked == ["speed", "templates", "kmeans"]


def test_plan_json_prints_the_locked_steps(
    split: Dataset, fit: tuple[Request, dict[str, str]], tmp_path: Path
) -> None:
    named, _ = fit

    plan = PlanWire.model_validate(
        invoke_json(
            [
                *_args("plan", split, tmp_path, "--applies-request", named.request_id),
                "--json",
            ]
        )
    )

    assert plan.locked_steps == ("speed", "templates", "kmeans")


def test_plan_text_names_the_locked_steps(
    split: Dataset, fit: tuple[Request, dict[str, str]], tmp_path: Path
) -> None:
    named, _ = fit

    result = runner.invoke(
        app, _args("plan", split, tmp_path, "--applies-request", named.request_id)
    )

    assert result.exit_code == 0, result.output
    assert "locked: speed, templates, kmeans" in result.output


def test_run_applies_the_fit_and_serves_the_pinned_step_from_the_cache(
    split: Dataset, fit: tuple[Request, dict[str, str]], tmp_path: Path
) -> None:
    named, runs = fit

    done = invoke_json(
        [
            *_args("run", split, tmp_path, "--applies-request", named.request_id),
            "--json",
        ]
    )

    served = {
        row["step"]: (row["state"], row["run_id"])
        for row in _ROWS.validate_python(done["steps"])
    }
    assert served["templates"] == ("cached", runs["templates"])


@pytest.mark.parametrize("token", ["speed", "=RUN", "speed=", "  =RUN"])
def test_a_malformed_pin_is_a_usage_error(
    split: Dataset, tmp_path: Path, token: str
) -> None:
    before = _request_count(split)

    result = runner.invoke(app, _args("submit", split, tmp_path, "--pin", token))

    assert result.exit_code == 2
    assert "--pin" in result.output
    assert "STEP=RUN_ID" in result.output
    assert _request_count(split) == before


def test_a_step_pinned_twice_is_a_usage_error(split: Dataset, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        _args(
            "submit", split, tmp_path, "--pin", "templates=A", "--pin", "templates=B"
        ),
    )

    assert result.exit_code == 2
    assert "templates" in result.output
    assert "twice" in result.output


@pytest.mark.parametrize("verb", ["plan", "submit", "run"])
def test_pins_beside_an_apply_are_a_usage_error(
    split: Dataset, tmp_path: Path, verb: str
) -> None:
    before = _request_count(split)

    result = runner.invoke(
        app,
        _args(
            verb, split, tmp_path, "--pin", "templates=RUN", "--applies-request", "R"
        ),
    )

    assert result.exit_code == 2
    assert "--pin" in result.output
    assert "--applies-request" in result.output
    assert _request_count(split) == before


@pytest.mark.parametrize(
    "extra", [["--pin", "templates=RUN"], ["--applies-request", "R"]]
)
def test_run_request_refuses_the_pin_options(
    split: Dataset, tmp_path: Path, extra: list[str]
) -> None:
    submitted = invoke_json([*_args("submit", split, tmp_path), "--json"])

    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--manifest",
            str(split.manifest_path),
            "--request",
            str(submitted["request_id"]),
            *extra,
        ],
    )

    assert result.exit_code == 1
    assert extra[0] in result.output
    assert "not used with --request" in result.output
