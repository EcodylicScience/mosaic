"""Recording a submission, and the command each of its steps is run by.

Two documents land in the dataset and every step gets an attempt id before
anything runs, which is what makes the request complete at submit rather than
filled in as work lands. The argv is checked against the CLI that has to accept
it, because a command nothing can run is the one failure a shape assertion would
not catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    Recipe,
    declared_version,
    load_request,
    recipe_digest,
    recipe_path,
    request_path,
    step_argv,
    submit_request,
)
from tests.helpers import add_tracks_variant, make_dataset

runner = CliRunner()

type Document = dict[str, object]

VARIANT = "convert-trex.0.2-1111111111"

CHAIN: Document = {
    "schema_version": 1,
    "name": "speed then templates",
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
def tracked(tmp_path: Path) -> Dataset:
    """Two schema-valid track tables, and nothing computed."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return dataset


def _recipe_file(tmp_path: Path, document: Document) -> Path:
    path = tmp_path / "recipe.json"
    _ = path.write_text(json.dumps(document))
    return path


def test_submitting_writes_the_recipe_and_the_request(tracked: Dataset) -> None:
    """The dataset records which pipeline was applied to it, and by whom."""
    recipe = Recipe.model_validate(CHAIN)

    submitted = submit_request(tracked, recipe, owner="jacob")

    assert submitted.recipe_path == recipe_path(tracked.base_dir, recipe_digest(recipe))
    assert submitted.request_path == request_path(
        tracked.base_dir, submitted.request.request_id
    )
    assert submitted.recipe_path.exists()
    read_back = load_request(tracked.base_dir, submitted.request.request_id)
    assert read_back == submitted.request
    assert read_back.owner == "jacob"


def test_every_step_is_assigned_its_own_attempt(tracked: Dataset) -> None:
    """Assigned before anything runs, and no two steps share one.

    Sharing would make two steps' outputs indistinguishable in the run-log, and
    the run-log is what a child reads to pin its parent's identity.
    """
    submitted = submit_request(tracked, Recipe.model_validate(CHAIN))

    assigned = submitted.request.step_executions
    assert sorted(assigned) == ["speed", "templates"]
    assert len(set(assigned.values())) == 2


def test_the_versions_every_step_resolved_against_are_pinned(
    tracked: Dataset,
) -> None:
    """Only this map says what *ran*; the digest says only what the recipe is."""
    submitted = submit_request(tracked, Recipe.model_validate(CHAIN))

    assert submitted.request.step_versions == {
        "speed": declared_version("feature", "speed-angvel"),
        "templates": declared_version("feature", "extract-templates"),
    }
    assert all(submitted.request.step_versions.values())


def test_an_op_step_pins_its_version_too(tracked: Dataset) -> None:
    """The map covers every step, not only the feature ones.

    An op's version is a visible segment of its run identifier rather than a hash
    term, so a bump makes the step read as absent rather than as complete -- the
    same request spanning two identity regimes either way.
    """
    with_op: Document = {
        "schema_version": 1,
        "steps": [
            {
                "id": "points",
                "type": "op",
                "kind": "convert-points",
                "params": {
                    "cvat_xml": "annotations.xml",
                    "images_dir": "images",
                    "class_names": ["bee"],
                    "radii": {"bee": 8.0},
                },
            }
        ],
    }

    submitted = submit_request(tracked, Recipe.model_validate(with_op))

    assert submitted.request.step_versions["points"] == declared_version(
        "op", "convert-points"
    )
    # Its sources are not on disk, so its identity is honestly unresolvable --
    # which is a reported state, not a reason a submission cannot be recorded.
    assert submitted.plan.step("points").run_id is None


def test_the_argv_names_the_request_and_never_a_second_path(
    tracked: Dataset,
) -> None:
    """The request is found from the manifest's parent.

    A path flag of its own would not be covered by the translation a queue
    applies when the executing side mounts the dataset somewhere else, and it
    would break precisely on the substrate that does.
    """
    submitted = submit_request(tracked, Recipe.model_validate(CHAIN))

    argv = step_argv(tracked.manifest_path, submitted.request, "speed")

    assert argv[:3] == ["mosaic", "run", "--json"]
    assert "--graph-request" in argv
    assert argv[argv.index("--graph-request") + 1] == submitted.request.request_id
    assert argv[argv.index("--step") + 1] == "speed"
    assert argv[argv.index("--execution-id") + 1] == submitted.request.execution_of(
        "speed"
    )
    paths = [word for word in argv if str(tracked.base_dir) in word]
    assert paths == [str(tracked.manifest_path)]


def test_the_submitted_argv_is_what_mosaic_run_accepts(tracked: Dataset) -> None:
    """The one assertion a shape check cannot make: the command actually runs."""
    submitted = submit_request(tracked, Recipe.model_validate(CHAIN))
    argv = step_argv(tracked.manifest_path, submitted.request, "speed")

    result = runner.invoke(app, argv[1:])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["step"] == "speed"
    assert payload["request_id"] == submitted.request.request_id
    assert payload["run_id"]


# --- the CLI ---------------------------------------------------------------------


def test_submit_prints_a_command_per_step_and_its_dependencies(
    tracked: Dataset, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            "submit",
            "--recipe",
            f"@{_recipe_file(tmp_path, CHAIN)}",
            "--manifest",
            str(tracked.manifest_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["request_id"]
    steps = {step["step"]: step for step in payload["steps"]}
    assert steps["speed"]["depends_on"] == []
    assert steps["templates"]["depends_on"] == [steps["speed"]["execution_id"]]
    assert steps["templates"]["parents"] == ["speed"]
    assert steps["speed"]["lane"]


def test_submit_runs_nothing(tracked: Dataset, tmp_path: Path) -> None:
    """It records a submission. Something else decides when the work happens."""
    result = runner.invoke(
        app,
        [
            "pipeline",
            "submit",
            "--recipe",
            f"@{_recipe_file(tmp_path, CHAIN)}",
            "--manifest",
            str(tracked.manifest_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert not list(Path(tracked.get_root("features")).glob("*/*/*.parquet"))


def test_a_shell_loop_over_the_printed_commands_runs_the_graph(
    tracked: Dataset, tmp_path: Path
) -> None:
    """The claim this phase makes: a driver needs only to start processes in order."""
    submitted = runner.invoke(
        app,
        [
            "pipeline",
            "submit",
            "--recipe",
            f"@{_recipe_file(tmp_path, CHAIN)}",
            "--manifest",
            str(tracked.manifest_path),
            "--json",
        ],
    )
    payload = json.loads(submitted.stdout)

    for step in payload["steps"]:
        ran = runner.invoke(app, list(step["argv"])[1:])
        assert ran.exit_code == 0, ran.output

    status = runner.invoke(
        app,
        [
            "pipeline",
            "status",
            "--manifest",
            str(tracked.manifest_path),
            "--request",
            payload["request_id"],
            "--json",
        ],
    )
    assert json.loads(status.stdout)["status"] == "finished"
