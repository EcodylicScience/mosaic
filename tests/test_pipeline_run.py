"""Running a plan in this process, and the four ``mosaic pipeline`` verbs.

The no-queue path, which is what a notebook and a bare compute node have. Two
properties carry the suite: a run does what the plan said it would and records
the identifiers it resolved, and a second run of the same recipe does nothing at
all. The third is the refusal -- a step whose identity covers its scope must not
quietly run over less of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    CoverageShortfall,
    Recipe,
    build_step_feature,
    plan_pipeline,
    run_pipeline,
)
from mosaic.core.pipeline.run import run_feature
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
        {
            "id": "scaler",
            "type": "feature",
            "feature": "global-scaler",
            "inputs": [{"step": "speed"}],
            "params": {
                "templates": {"step": "templates", "pattern": "templates.parquet"}
            },
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


def _recorded_params(dataset: Dataset, run_id: str) -> str:
    """The provenance one run wrote, as text, whichever storage it landed under.

    Read off disk rather than from the plan, because what is being asked is what
    the *run* pinned -- the plan's answer is the prediction under test.
    """
    found = list(Path(dataset.get_root("features")).glob(f"*/{run_id}/params.json"))
    assert found, f"no params.json for run {run_id}"
    return found[0].read_text()


UNPINNED_CHAIN: Document = json.loads(
    json.dumps(CHAIN).replace(', "pattern": "templates.parquet"', "")
)
"""``CHAIN`` with the scaler's artifact reference naming no file.

This shape once validated, planned and ran while the scaler fitted on
``seq_a.parquet`` -- one sequence's pass-through table -- because the derived
``*.parquet`` glob matched the per-entry outputs beside the templates matrix and
they sort first.
"""


# --- the runner ------------------------------------------------------------------


def test_an_unpinned_artifact_reference_reads_the_declared_artifact(
    tracked: Dataset,
) -> None:
    """The consumer's declared type names the file, so the recipe need not.

    Two sequences, so the producer's run root holds ``seq_a.parquet`` and
    ``seq_b.parquet`` beside ``templates.parquet`` and
    ``template_provenance.parquet`` -- every arrangement in which taking the
    first sorted match is wrong. What closes it is that the scaler fits on the
    template matrix, which its own column list is read from.
    """
    recipe = Recipe.model_validate(UNPINNED_CHAIN)

    done = run_pipeline(tracked, recipe)

    assert [outcome.state for outcome in done.outcomes] == ["ran", "ran", "ran"]
    scaler = done.outcomes[-1]
    assert scaler.run_id is not None
    assert '"pattern": "templates.parquet"' in _recorded_params(tracked, scaler.run_id)

    templates_run = done.outcomes[1].run_id
    assert templates_run is not None
    produced = _table(tracked, templates_run, "templates.parquet")
    fitted = _table(tracked, scaler.run_id, "scaled_templates.parquet")
    assert list(fitted.columns) == list(produced.columns)
    assert len(fitted) == len(produced)


def _table(dataset: Dataset, run_id: str, name: str) -> pd.DataFrame:
    """One named artifact out of whichever storage *run_id* landed under."""
    found = list(Path(dataset.get_root("features")).glob(f"*/{run_id}/{name}"))
    assert found, f"no {name} for run {run_id}"
    return pd.read_parquet(found[0])


def test_a_run_records_the_identifiers_the_plan_resolved(tracked: Dataset) -> None:
    """The plan is a preview, and here the preview turns out to have been right."""
    recipe = Recipe.model_validate(CHAIN)
    planned = plan_pipeline(tracked, recipe).run_ids

    done = run_pipeline(tracked, recipe)

    assert [outcome.state for outcome in done.outcomes] == ["ran", "ran", "ran"]
    assert {outcome.step_id: outcome.run_id for outcome in done.outcomes} == planned
    assert not any(outcome.diverged for outcome in done.outcomes)


def test_a_second_run_does_nothing(tracked: Dataset) -> None:
    """Every step is served from its own directory, which is what caching means."""
    recipe = Recipe.model_validate(CHAIN)
    first = run_pipeline(tracked, recipe)

    again = run_pipeline(tracked, recipe)

    assert [outcome.state for outcome in again.outcomes] == ["cached"] * 3
    assert again.ran == ()
    assert {outcome.run_id for outcome in again.outcomes} == {
        outcome.run_id for outcome in first.outcomes
    }


def test_a_step_already_computed_is_not_recomputed(tracked: Dataset) -> None:
    """A partly built graph resumes rather than starting again."""
    recipe = Recipe.model_validate(CHAIN)
    plan = plan_pipeline(tracked, recipe)
    _ = run_feature(tracked, build_step_feature(plan.step("speed").spec), track=False)

    done = run_pipeline(tracked, recipe)

    assert done.outcomes[0].state == "cached"
    assert [outcome.state for outcome in done.outcomes[1:]] == ["ran", "ran"]


UNCOMPUTABLE: list[tuple[str, str]] = [("", "seq_a"), ("", "seq_b"), ("", "seq_gone")]
"""A submission naming one entry more than the dataset can process.

An explicit narrowing is not widened or narrowed by what is on disk -- it is the
submission saying what it wants -- so this is a scope the graph cannot fill, which
is the state a shortfall describes. A partly-computed upstream is *not*: the
runner completes it first, which is what it is for.
"""


def test_a_scope_dependent_step_refuses_a_short_upstream(tracked: Dataset) -> None:
    """The refusal that makes a shortfall a decision rather than a default.

    ``speed`` computes what it can and is correct in doing so: its outputs are
    per entry, and the missing one arrives later under the same identifier. The
    fit below it is not like that -- over two of three entries it is a different
    artifact from the one that was asked for, under a name saying it is the same.
    """
    with pytest.raises(CoverageShortfall) as raised:
        _ = run_pipeline(
            tracked, Recipe.model_validate(CHAIN), intended_entries=UNCOMPUTABLE
        )

    assert raised.value.step_id == "templates"
    assert raised.value.covered == 2
    assert raised.value.target == 3
    assert "seq_gone" in str(raised.value)


def test_allow_partial_is_the_gesture_that_answers_the_refusal(
    tracked: Dataset,
) -> None:
    """Recorded as a choice, and the run then proceeds over what there is.

    The fit lands somewhere other than where it was planned to -- it covers the
    entries it was actually given, and those are its name -- and the step below
    it still resolves, because what governs is what the run recorded rather than
    what the plan predicted.
    """
    done = run_pipeline(
        tracked,
        Recipe.model_validate(CHAIN),
        intended_entries=UNCOMPUTABLE,
        allow_partial=True,
    )

    assert [outcome.state for outcome in done.outcomes] == ["ran", "ran", "ran"]
    templates = next(o for o in done.outcomes if o.step_id == "templates")
    assert templates.diverged, "its identity covers the set it was fitted on"
    scaler = next(o for o in done.outcomes if o.step_id == "scaler")
    assert templates.run_id in _recorded_params(tracked, scaler.run_id)


def test_a_partly_computed_upstream_is_completed_rather_than_refused(
    tracked: Dataset,
) -> None:
    """The case a shortfall is *not*, kept apart because the two look alike.

    Half a scope already computed is a resumable run, and the runner asks the
    step for the remainder. Only a scope nothing can fill is a shortfall.
    """
    recipe = Recipe.model_validate(CHAIN)
    plan = plan_pipeline(tracked, recipe)
    _ = run_feature(
        tracked,
        build_step_feature(plan.step("speed").spec),
        entries=[("", "seq_a")],
        track=False,
    )

    done = run_pipeline(tracked, recipe)

    assert [outcome.state for outcome in done.outcomes] == ["ran", "ran", "ran"]


def test_a_narrowed_run_covers_only_what_was_asked_for(tracked: Dataset) -> None:
    """The narrowing is the submission speaking, and the run honours it."""
    recipe = Recipe.model_validate(CHAIN)

    done = run_pipeline(tracked, recipe, intended_entries=[("", "seq_a")])

    assert done.scope == frozenset({("", "seq_a")})
    assert all(outcome.state == "ran" for outcome in done.outcomes)


# --- the CLI ---------------------------------------------------------------------


def test_validate_says_a_good_recipe_is_good(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["pipeline", "validate", "--recipe", f"@{_recipe_file(tmp_path, CHAIN)}"]
    )

    assert result.exit_code == 0, result.output
    assert "valid" in result.stdout


def test_validate_exits_non_zero_and_names_the_fault(tmp_path: Path) -> None:
    """A recipe checked in CI has to fail the shell, not only print."""
    broken: Document = {
        "steps": [{"id": "a", "type": "feature", "feature": "no-such-feature"}]
    }
    result = runner.invoke(
        app, ["pipeline", "validate", "--recipe", f"@{_recipe_file(tmp_path, broken)}"]
    )

    assert result.exit_code == 1
    assert "no-such-feature" in result.output


def test_validate_opens_no_dataset(tmp_path: Path) -> None:
    """It takes no manifest at all, which is the property rather than a default."""
    result = runner.invoke(
        app,
        [
            "pipeline",
            "validate",
            "--recipe",
            f"@{_recipe_file(tmp_path, CHAIN)}",
            "--json",
        ],
    )

    assert json.loads(result.stdout) == {"valid": True, "problems": []}


def test_show_reads_the_graph_without_a_dataset(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["pipeline", "show", "--recipe", f"@{_recipe_file(tmp_path, CHAIN)}", "--json"],
    )

    payload = json.loads(result.stdout)
    assert [step["id"] for step in payload["steps"]] == ["speed", "templates", "scaler"]
    assert {
        (edge["producer"], edge["consumer"], edge["site"]) for edge in payload["edges"]
    } == {
        ("speed", "templates", "inputs"),
        ("speed", "scaler", "inputs"),
        ("templates", "scaler", "params"),
    }


def test_plan_reports_identities_and_what_is_left(
    tracked: Dataset, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            "plan",
            "--recipe",
            f"@{_recipe_file(tmp_path, CHAIN)}",
            "--manifest",
            str(tracked.manifest_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["complete"] is False
    assert [step["step"] for step in payload["steps"]] == [
        "speed",
        "templates",
        "scaler",
    ]
    assert all(step["run_id"] != "-" for step in payload["steps"])
    assert payload["steps"][1]["reason"] == "waiting on speed"


def test_run_executes_the_graph_and_then_finds_it_complete(
    tracked: Dataset, tmp_path: Path
) -> None:
    """The end-to-end path a person actually takes: run, then plan again."""
    recipe_path = _recipe_file(tmp_path, CHAIN)
    manifest = str(tracked.manifest_path)

    ran = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--recipe",
            f"@{recipe_path}",
            "--manifest",
            manifest,
            "--json",
        ],
    )
    assert ran.exit_code == 0, ran.output
    assert [step["state"] for step in json.loads(ran.stdout)["steps"]] == ["ran"] * 3

    again = runner.invoke(
        app,
        [
            "pipeline",
            "plan",
            "--recipe",
            f"@{recipe_path}",
            "--manifest",
            manifest,
            "--json",
        ],
    )
    assert json.loads(again.stdout)["complete"] is True


def test_run_refuses_a_shortfall_with_a_non_zero_exit(
    tracked: Dataset, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--recipe",
            f"@{_recipe_file(tmp_path, CHAIN)}",
            "--manifest",
            str(tracked.manifest_path),
            *[f"--entry={group}:{sequence}" for group, sequence in UNCOMPUTABLE],
        ],
    )

    assert result.exit_code == 1
    assert "allow_partial" in result.output
