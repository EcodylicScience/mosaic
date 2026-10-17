"""The fingerprint of a plan, and a submission refused when its plan has changed.

A caller reviews a plan and submits it later. The fingerprint names what the plan
found on the dataset. A submission given it is refused when a feature run or a
tracks entry has appeared in between.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.error_envelope import ErrorEnvelope, error_envelope
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import serialize as serialize_module
from mosaic.core.pipeline.graph import (
    CleanupPolicy,
    PlanChanged,
    Recipe,
    plan_fingerprint,
    plan_pipeline,
    plan_wire,
    plan_with_cleanup,
    pipelines_root,
    requests_matching,
    run_pipeline,
    submit_request,
)
from mosaic.core.scope import Scope
from tests.helpers import (
    SPEED_RECIPE,
    add_tracks_variant,
    invoke_json,
    make_dataset,
    run_fresh,
    run_speed,
    speed_recipe,
    speed_recipe_stepping,
)

VARIANT = "convert-trex.0.2-1111111111"

runner = CliRunner()


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two schema-valid track tables before any feature run."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return dataset


def _fingerprint(dataset: Dataset) -> str:
    return plan_fingerprint(plan_pipeline(dataset, speed_recipe()))


def _recipe_argument(tmp_path: Path) -> str:
    path = tmp_path / "speed.recipe.json"
    _ = path.write_text(json.dumps(SPEED_RECIPE))
    return f"@{path}"


def test_one_plan_has_one_fingerprint_in_every_interpreter(tracked: Dataset) -> None:
    """The fingerprint sorts every set, whose iteration order differs per process."""
    code = (
        "import json\n"
        "from mosaic.core.dataset import open_dataset\n"
        "from mosaic.core.pipeline.graph import Recipe, plan_fingerprint, plan_pipeline\n"
        f"recipe = Recipe.model_validate(json.loads({json.dumps(SPEED_RECIPE)!r}))\n"
        f"dataset = open_dataset({str(tracked.manifest_path)!r})\n"
        "print(plan_fingerprint(plan_pipeline(dataset, recipe)))\n"
    )

    assert run_fresh(code) == _fingerprint(tracked)


def test_a_new_tracks_entry_changes_the_fingerprint(tracked: Dataset) -> None:
    before = _fingerprint(tracked)

    add_tracks_variant(tracked, VARIANT, "seq_c", std_format="mosaic_v1")

    assert _fingerprint(tracked) != before


def test_a_new_feature_run_changes_the_fingerprint(tracked: Dataset) -> None:
    before = _fingerprint(tracked)

    _ = run_pipeline(tracked, speed_recipe())

    assert _fingerprint(tracked) != before


def test_the_same_tables_in_another_dataset_have_another_fingerprint(
    tracked: Dataset, tmp_path: Path
) -> None:
    """The manifest's uuid enters the fingerprint. A plan names its dataset."""
    other = make_dataset(tmp_path / "other")
    add_tracks_variant(other, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")

    assert _fingerprint(other) != _fingerprint(tracked)


def test_the_plan_wire_reports_the_fingerprint(tracked: Dataset) -> None:
    plan = plan_pipeline(tracked, speed_recipe())

    assert plan_wire(plan).fingerprint == plan_fingerprint(plan)


def test_the_fingerprint_ignores_the_sample_size(tracked: Dataset) -> None:
    plan = plan_pipeline(tracked, speed_recipe())

    assert plan_wire(plan, missing_sample=0).fingerprint == plan_fingerprint(plan)
    assert plan_wire(plan, missing_sample=1).fingerprint == plan_fingerprint(plan)


def test_the_fingerprint_ignores_open_requests(tracked: Dataset) -> None:
    before = _fingerprint(tracked)
    _ = submit_request(tracked, speed_recipe())
    plan = plan_pipeline(tracked, speed_recipe())
    unended = requests_matching(tracked.base_dir, plan.recipe_digest, Scope())

    wire = plan_wire(plan, open_requests=unended)

    assert len(wire.open_requests) == 1
    assert wire.fingerprint == plan_fingerprint(plan) == before


def test_cautions_and_whole_dataset_leave_the_fingerprint_unchanged(
    tracked: Dataset,
) -> None:
    """A plan over the whole dataset and one naming its every entry share one."""
    whole = plan_pipeline(tracked, speed_recipe(), scope=Scope())
    named = plan_pipeline(
        tracked, speed_recipe(), scope=Scope(entries=[("", "seq_a"), ("", "seq_b")])
    )

    assert plan_wire(whole).cautions != plan_wire(named).cautions == ()
    assert plan_fingerprint(whole) == plan_fingerprint(named)


def test_a_current_fingerprint_submits(tracked: Dataset) -> None:
    fingerprint = _fingerprint(tracked)

    submitted = submit_request(tracked, speed_recipe(), expect_plan=fingerprint)

    assert submitted.request_path.exists()
    assert plan_fingerprint(submitted.plan) == fingerprint


def test_a_stale_fingerprint_is_a_conflict_and_writes_nothing(
    tracked: Dataset,
) -> None:
    stale = _fingerprint(tracked)
    add_tracks_variant(tracked, VARIANT, "seq_c", std_format="mosaic_v1")

    with pytest.raises(PlanChanged) as raised:
        _ = submit_request(tracked, speed_recipe(), expect_plan=stale)

    envelope = error_envelope(raised.value)
    assert envelope.kind == "conflict"
    assert envelope.detail == {"expected": stale, "found": _fingerprint(tracked)}
    assert not pipelines_root(tracked.base_dir).exists()


# --- the CLI ---------------------------------------------------------------------


def test_plan_json_reports_the_fingerprint(tracked: Dataset, tmp_path: Path) -> None:
    payload = invoke_json(
        [
            "pipeline",
            "plan",
            "--recipe",
            _recipe_argument(tmp_path),
            "--manifest",
            str(tracked.manifest_path),
            "--json",
        ]
    )

    assert payload["fingerprint"] == _fingerprint(tracked)


def test_submit_with_a_stale_expected_plan_exits_with_a_conflict(
    tracked: Dataset, tmp_path: Path
) -> None:
    stale = _fingerprint(tracked)
    add_tracks_variant(tracked, VARIANT, "seq_c", std_format="mosaic_v1")

    result = runner.invoke(
        app,
        [
            "pipeline",
            "submit",
            "--recipe",
            _recipe_argument(tmp_path),
            "--manifest",
            str(tracked.manifest_path),
            "--expect-plan",
            stale,
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert ErrorEnvelope.model_validate_json(result.stdout).kind == "conflict"
    assert not pipelines_root(tracked.base_dir).exists()


def test_submit_with_the_current_expected_plan_reports_it(
    tracked: Dataset, tmp_path: Path
) -> None:
    fingerprint = _fingerprint(tracked)

    payload = invoke_json(
        [
            "pipeline",
            "submit",
            "--recipe",
            _recipe_argument(tmp_path),
            "--manifest",
            str(tracked.manifest_path),
            "--expect-plan",
            fingerprint,
            "--json",
        ]
    )

    assert payload["plan_fingerprint"] == fingerprint


def test_an_integral_float_and_its_integer_plan_one_fingerprint(
    tracked: Dataset,
) -> None:
    def fingerprint(step_size: object) -> str:
        document = json.loads(json.dumps(SPEED_RECIPE))
        document["steps"][0]["params"] = {"step_size": step_size}
        return plan_fingerprint(plan_pipeline(tracked, Recipe.model_validate(document)))

    assert fingerprint(1.0) == fingerprint(1)


LOOSE = CleanupPolicy(keep_newest=0, min_age_hours=0)


def test_a_plan_without_cleanup_fingerprints_as_before(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = run_speed(tracked, 1)
    digested: list[Mapping[str, object]] = []
    digest = serialize_module.document_digest

    def capture(document: dict[str, JsonValue]) -> str:
        digested.append(document)
        return digest(document)

    monkeypatch.setattr(serialize_module, "document_digest", capture)
    plain = plan_pipeline(tracked, speed_recipe_stepping(2))
    unasked = plan_with_cleanup(tracked, speed_recipe_stepping(2), cleanup=None)

    assert plan_fingerprint(unasked) == plan_fingerprint(plain)
    assert set(digested[-1]) == {"dataset", "recipe", "steps"}
    assert plan_wire(unasked).cleanup is None


def test_a_new_superseded_run_changes_a_cleanup_fingerprint(tracked: Dataset) -> None:
    _ = run_speed(tracked, 1)
    recipe = speed_recipe_stepping(2)
    before = plan_fingerprint(plan_with_cleanup(tracked, recipe, cleanup=LOOSE))
    plain = plan_fingerprint(plan_pipeline(tracked, recipe))

    _ = run_speed(tracked, 3)

    after = plan_fingerprint(plan_with_cleanup(tracked, recipe, cleanup=LOOSE))
    assert after != before
    assert plan_fingerprint(plan_pipeline(tracked, recipe)) == plain
    assert before != plain


def test_the_cleanup_policy_enters_the_fingerprint(tracked: Dataset) -> None:
    _ = run_speed(tracked, 1)
    recipe = speed_recipe_stepping(2)

    loose = plan_with_cleanup(tracked, recipe, cleanup=LOOSE)
    strict = plan_with_cleanup(
        tracked, recipe, cleanup=CleanupPolicy(keep_newest=5, min_age_hours=0)
    )

    assert plan_fingerprint(loose) != plan_fingerprint(strict)
    assert plan_wire(loose).fingerprint == plan_fingerprint(loose)
