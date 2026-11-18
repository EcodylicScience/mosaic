"""``--step-scope`` and ``--group`` on ``mosaic pipeline plan``, ``run`` and ``submit``.

A step scope narrows one step of a request, and ``--group`` narrows the
request to whole groups, as the MCP ``{"groups": [...]}`` selector does.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.selections import save_selection
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


def test_the_tokens_a_refusal_gives_plan_without_problems(
    split: Dataset, tmp_path: Path
) -> None:
    from mosaic.core.pipeline.graph import PlanWire

    recipe = _recipe_file(tmp_path)
    refused = runner.invoke(
        app,
        [
            "pipeline",
            "plan",
            "--recipe",
            recipe,
            "--manifest",
            str(split.manifest_path),
            "--step-scope",
            'speed={"groups": ["train"]}',
            "--json",
        ],
    )

    assert refused.exit_code == 1
    problems = PlanWire.model_validate_json(refused.stdout).problems
    assert [(problem.step, problem.where) for problem in problems] == [
        ("templates", "inputs[0]")
    ]
    pasted = re.search(r"or pass (.*) to mosaic pipeline plan", problems[0].message)
    assert pasted is not None
    tokens = shlex.split(pasted.group(1))
    assert tokens == [
        "--step-scope",
        'speed={"groups":["train"]}',
        "--step-scope",
        'templates={"groups":["train"]}',
    ]
    plan = _plan(split, recipe, *tokens)
    assert plan["problems"] == []
    assert _targets(plan) == {"speed": 2, "templates": 2}


@pytest.mark.parametrize(
    "token",
    [
        "templates",
        '={"groups": ["train"]}',
        "templates=[1, 2]",
        'templates={"x": 1}',
        'templates={"entries": [["train", "s1", "cam0"]]}',
    ],
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


def _scopes_file(tmp_path: Path, scopes: object) -> str:
    path = tmp_path / "scopes.json"
    _ = path.write_text(json.dumps(scopes), encoding="utf-8")
    return f"@{path}"


def test_a_step_scope_file_narrows_each_step_it_maps(
    split: Dataset, tmp_path: Path
) -> None:
    scopes = {"speed": {"groups": ["train"]}, "templates": {"sequences": ["s1"]}}

    plan = _plan(
        split,
        _recipe_file(tmp_path),
        "--step-scope",
        _scopes_file(tmp_path, scopes),
    )

    assert _targets(plan) == {"speed": 2, "templates": 1}


def test_a_step_in_a_step_scope_file_and_a_token_is_refused(
    split: Dataset, tmp_path: Path
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
            _scopes_file(tmp_path, {"templates": {"groups": ["train"]}}),
            "--step-scope",
            'templates={"groups": ["test"]}',
        ],
    )

    assert result.exit_code == 1
    assert "names step 'templates' twice" in result.stderr


@pytest.mark.parametrize(
    "scopes",
    [
        [{"groups": ["train"]}],
        {"templates": [1, 2]},
        {"templates": {"x": 1}},
        {"": {"groups": ["train"]}},
    ],
)
def test_a_malformed_step_scope_file_is_refused(
    split: Dataset, tmp_path: Path, scopes: object
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
            _scopes_file(tmp_path, scopes),
        ],
    )

    assert result.exit_code == 1
    assert "--step-scope" in result.stderr


@pytest.mark.parametrize(
    "scopes", [{"templates": [1, 2]}, {"templates": {"x": 1}}], ids=["list", "keys"]
)
def test_an_error_in_a_step_scope_file_names_its_step(
    split: Dataset, tmp_path: Path, scopes: object
) -> None:
    named = _scopes_file(tmp_path, scopes)

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
            named,
        ],
    )

    assert result.exit_code == 1
    assert f"{named} [templates]" in result.stderr


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


# --- a selection by sequence tag ---------------------------------------------

BY_SITE = 'templates={"sequence_tags": {"site": ["alpha"]}}'
"""The step scope that selects the entries recorded at the site ``alpha``."""


@pytest.fixture
def sited(split: Dataset) -> Dataset:
    """``split`` with its entries tagged: ``alpha`` at ``train/s1`` and ``test/s3``."""
    _ = write_tags(
        split,
        tag_table(
            {
                ("train", "s1"): {"site": "alpha"},
                ("train", "s2"): {"site": "beta"},
                ("test", "s3"): {"site": "alpha"},
            },
            {"site": "categorical"},
        ),
    )
    return split


def _run_ids(plan: dict[str, object]) -> dict[str, str | None]:
    from mosaic.core.pipeline.graph import PlanWire

    return {step.step: step.run_id for step in PlanWire.model_validate(plan).steps}


def test_plan_with_a_tag_step_scope_plans_the_selected_entries(
    sited: Dataset, tmp_path: Path
) -> None:
    recipe = _recipe_file(tmp_path)
    before = tree_snapshot(sited.base_dir)

    by_tag = _plan(sited, recipe, "--step-scope", BY_SITE)
    by_name = _plan(
        sited,
        recipe,
        "--step-scope",
        'templates={"entries": [["train", "s1"], ["test", "s3"]]}',
    )

    assert _targets(by_tag) == {"speed": 3, "templates": 2}
    assert _run_ids(by_tag) == _run_ids(by_name)
    assert tree_snapshot(sited.base_dir) == before


def test_a_tag_step_scope_of_a_file_is_lowered_too(
    sited: Dataset, tmp_path: Path
) -> None:
    scopes = {"templates": {"sequence_tags": {"site": ["beta"]}}}

    plan = _plan(
        sited, _recipe_file(tmp_path), "--step-scope", _scopes_file(tmp_path, scopes)
    )

    assert _targets(plan) == {"speed": 3, "templates": 1}


def _submit(ds: Dataset, recipe: str, *extra: str) -> dict[str, object]:
    return invoke_json(
        [
            "pipeline",
            "submit",
            "--recipe",
            recipe,
            "--manifest",
            str(ds.manifest_path),
            *extra,
            "--json",
        ]
    )


def test_submit_with_a_tag_step_scope_records_entries_and_the_revision_read(
    sited: Dataset, tmp_path: Path
) -> None:
    submitted = _submit(sited, _recipe_file(tmp_path), "--step-scope", BY_SITE)

    assert submitted["step_scopes"] == {
        "templates": {
            "entries": [["test", "s3"], ["train", "s1"]],
            "groups": None,
            "sequences": None,
        }
    }
    text = Path(str(submitted["request_path"])).read_text(encoding="utf-8")
    assert "site" not in text
    assert "alpha" not in text
    request = json.loads(text)
    assert request["tag_revision"]["key"] == "dataset"
    assert request["tag_revision"]["revision"] == 1


def test_a_submission_that_selects_no_tag_records_no_revision(
    sited: Dataset, tmp_path: Path
) -> None:
    submitted = _submit(
        sited,
        _recipe_file(tmp_path),
        "--step-scope",
        'templates={"groups": ["train"]}',
    )

    request = json.loads(
        Path(str(submitted["request_path"])).read_text(encoding="utf-8")
    )
    assert "tag_revision" not in request


@pytest.mark.parametrize("verb", ["plan", "verify", "submit", "run"])
def test_a_tag_step_scope_that_selects_nothing_is_an_invalid_envelope(
    sited: Dataset, tmp_path: Path, verb: str
) -> None:
    before = tree_snapshot(sited.base_dir)

    result = runner.invoke(
        app,
        [
            "pipeline",
            verb,
            "--recipe",
            _recipe_file(tmp_path),
            "--manifest",
            str(sited.manifest_path),
            "--step-scope",
            'templates={"sequence_tags": {"site": ["gamma"]}}',
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["kind"] == "invalid"
    assert "'gamma'" in payload["message"]
    assert payload["detail"] == {"revision": 1, "tags": {"site": ["gamma"]}}
    assert tree_snapshot(sited.base_dir) == before


def test_a_tag_step_scope_beside_another_key_is_refused(
    sited: Dataset, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            "plan",
            "--recipe",
            _recipe_file(tmp_path),
            "--manifest",
            str(sited.manifest_path),
            "--step-scope",
            'templates={"sequence_tags": {"site": ["alpha"]}, "groups": ["train"]}',
        ],
    )

    assert result.exit_code == 1
    assert "--step-scope" in result.stderr
    assert "sequence_tags" in result.stderr


def test_a_tag_step_scope_of_a_dataset_without_tags_says_how_to_import_them(
    split: Dataset, tmp_path: Path
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
            BY_SITE,
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["kind"] == "invalid"
    assert "mosaic sequence-tags import" in payload["message"]
    assert payload["detail"] is None


@pytest.fixture
def selected(split: Dataset) -> Dataset:
    """The split dataset with the selections ``day1`` and ``one``."""
    _ = save_selection(split, "day1", [("train", "s1"), ("train", "s2")])
    _ = save_selection(split, "one", [("train", "s1")])
    return split


def test_plan_resolves_a_selection_for_the_request_and_a_step(
    selected: Dataset, tmp_path: Path
) -> None:
    before = tree_snapshot(selected.base_dir)

    plan = _plan(
        selected,
        _recipe_file(tmp_path),
        "--entry",
        "@day1",
        "--step-scope",
        'templates={"selection": "one"}',
    )

    assert _targets(plan) == {"speed": 2, "templates": 1}
    assert tree_snapshot(selected.base_dir) == before


def test_submit_records_the_selections_beside_the_entries(
    selected: Dataset, tmp_path: Path
) -> None:
    from mosaic.core.pipeline.graph import load_request

    submitted = _submit(
        selected,
        _recipe_file(tmp_path),
        "--entry",
        "@day1",
        "--step-scope",
        'templates={"selection": "one"}',
    )

    assert submitted["scope"] == [["train", "s1"], ["train", "s2"]]
    request = load_request(selected.base_dir, str(submitted["request_id"]))
    assert request.selections == {"": "day1", "templates": "one"}
    assert request.scope is not None
    assert request.scope.entries == [("train", "s1"), ("train", "s2")]
    assert request.step_scopes["templates"].entries == [("train", "s1")]


def test_a_request_without_a_selection_records_none(
    selected: Dataset, tmp_path: Path
) -> None:
    from mosaic.core.pipeline.graph import load_request

    submitted = _submit(selected, _recipe_file(tmp_path), "--group", "train")

    request = load_request(selected.base_dir, str(submitted["request_id"]))
    assert request.selections == {}


def test_run_records_the_selection_it_ran_over(
    selected: Dataset, tmp_path: Path
) -> None:
    from mosaic.core.pipeline.graph import load_request

    done = invoke_json(
        [
            "pipeline",
            "run",
            "--recipe",
            _recipe_file(tmp_path),
            "--manifest",
            str(selected.manifest_path),
            "--entry",
            "@day1",
            "--json",
        ]
    )

    assert done["scope"] == [["train", "s1"], ["train", "s2"]]
    request = load_request(selected.base_dir, str(done["request_id"]))
    assert request.selections == {"": "day1"}


def test_a_selection_key_stands_alone_in_a_step_scope(
    selected: Dataset, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            "plan",
            "--recipe",
            _recipe_file(tmp_path),
            "--manifest",
            str(selected.manifest_path),
            "--step-scope",
            'templates={"selection": "one", "groups": ["train"]}',
        ],
    )

    assert result.exit_code == 1
    assert "--step-scope" in result.stderr
    assert "selection" in result.stderr
    assert "Extra inputs" not in result.stderr


def test_a_selection_token_is_refused_beside_a_group(
    selected: Dataset, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            "plan",
            "--recipe",
            _recipe_file(tmp_path),
            "--manifest",
            str(selected.manifest_path),
            "--entry",
            "@day1",
            "--group",
            "train",
        ],
    )

    assert result.exit_code == 1
    assert "--entry" in result.stderr
    assert "--group" in result.stderr


@pytest.mark.parametrize(
    "flags",
    [
        ["--entry", "@nope"],
        ["--step-scope", 'templates={"selection": "nope"}'],
    ],
    ids=["request", "step"],
)
@pytest.mark.parametrize("verb", ["plan", "submit", "verify"])
def test_a_missing_selection_is_refused_naming_the_selections_present(
    selected: Dataset, tmp_path: Path, verb: str, flags: list[str]
) -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            verb,
            "--recipe",
            _recipe_file(tmp_path),
            "--manifest",
            str(selected.manifest_path),
            *flags,
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["reason"] == "selection_not_found"
    assert "day1" in payload["message"]
    assert "one" in payload["message"]
