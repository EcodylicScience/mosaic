"""``--tracks-run-id`` on ``mosaic pipeline plan``, ``submit`` and ``run``.

The flag names the tracks variant that each step reading the tracks without a
``tracks`` reference reads. Over a dataset where one entry holds two variants,
a recipe cannot choose, and the request answers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import PlanWire, load_request
from mosaic.core.pipeline.refusal import REFUSED_EXIT_CODE
from tests.helpers import add_tracks_variant, invoke_json, make_dataset, tree_snapshot

runner = CliRunner()

_ROWS = TypeAdapter(list[dict[str, object]])
"""The steps of a run, each an object."""

VARIANT = "convert-trex.0.2-1111111111"
OTHER_VARIANT = "convert-other.0.1-2222222222"

RECIPE = {
    "schema_version": 1,
    "steps": [
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "inputs": ["tracks"],
            "params": {"step_size": 1},
        }
    ],
}


@pytest.fixture
def two_variants(tmp_path: Path) -> Dataset:
    """Entry ``seq_a`` holds two variants of one schema family."""
    dataset = make_dataset(tmp_path / "ds")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    add_tracks_variant(dataset, OTHER_VARIANT, "seq_a", std_format="mosaic_v1")
    return dataset


def _recipe_file(tmp_path: Path) -> str:
    path = tmp_path / "recipe.json"
    _ = path.write_text(json.dumps(RECIPE), encoding="utf-8")
    return f"@{path}"


def _args(verb: str, ds: Dataset, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "pipeline",
        verb,
        "--recipe",
        _recipe_file(tmp_path),
        "--manifest",
        str(ds.manifest_path),
        *extra,
        "--json",
    ]


def test_plan_with_a_variant_resolves_and_writes_nothing(
    two_variants: Dataset, tmp_path: Path
) -> None:
    before = tree_snapshot(two_variants.base_dir)

    named = invoke_json(
        _args("plan", two_variants, tmp_path, "--tracks-run-id", OTHER_VARIANT)
    )
    other = invoke_json(
        _args("plan", two_variants, tmp_path, "--tracks-run-id", VARIANT)
    )

    assert tree_snapshot(two_variants.base_dir) == before
    assert named["fingerprint"] != other["fingerprint"]
    assert [step.step for step in PlanWire.model_validate(named).steps] == ["speed"]


def test_plan_without_a_variant_is_refused_by_name(
    two_variants: Dataset, tmp_path: Path
) -> None:
    result = runner.invoke(app, _args("plan", two_variants, tmp_path))

    assert result.exit_code == 1
    assert json.loads(result.stdout)["reason"] == "variant_ambiguous"


def test_submit_records_the_variant_the_plan_named(
    two_variants: Dataset, tmp_path: Path
) -> None:
    planned = invoke_json(
        _args("plan", two_variants, tmp_path, "--tracks-run-id", OTHER_VARIANT)
    )

    submitted = invoke_json(
        _args(
            "submit",
            two_variants,
            tmp_path,
            "--tracks-run-id",
            OTHER_VARIANT,
            "--expect-plan",
            str(planned["fingerprint"]),
        )
    )

    request = load_request(two_variants.base_dir, str(submitted["request_id"]))
    assert request.tracks_run_id == OTHER_VARIANT
    assert submitted["plan_fingerprint"] == planned["fingerprint"]


def test_submit_without_a_variant_is_refused_by_name(
    two_variants: Dataset, tmp_path: Path
) -> None:
    result = runner.invoke(app, _args("submit", two_variants, tmp_path))

    assert result.exit_code == 1
    assert json.loads(result.stdout)["reason"] == "variant_ambiguous"
    assert not (two_variants.base_dir / ".mosaic" / "pipelines").exists()


def test_submit_with_cleanup_passes_the_variant_to_both_plans(
    two_variants: Dataset, tmp_path: Path
) -> None:
    """The cleanup path plans twice, and the second plan must name the variant too."""
    planned = invoke_json(
        _args(
            "plan",
            two_variants,
            tmp_path,
            "--tracks-run-id",
            OTHER_VARIANT,
            "--cleanup",
            "superseded",
        )
    )

    submitted = invoke_json(
        _args(
            "submit",
            two_variants,
            tmp_path,
            "--tracks-run-id",
            OTHER_VARIANT,
            "--cleanup",
            "superseded",
            "--expect-plan",
            str(planned["fingerprint"]),
        )
    )

    request = load_request(two_variants.base_dir, str(submitted["request_id"]))
    assert request.tracks_run_id == OTHER_VARIANT
    assert request.cleanup is not None


def test_run_with_a_variant_runs_over_the_entry_that_variant_holds(
    two_variants: Dataset, tmp_path: Path
) -> None:
    done = invoke_json(
        _args("run", two_variants, tmp_path, "--tracks-run-id", OTHER_VARIANT)
    )

    request = load_request(two_variants.base_dir, str(done["request_id"]))
    assert request.tracks_run_id == OTHER_VARIANT
    assert done["scope"] == [["", "seq_a"]]
    [step] = _ROWS.validate_python(done["steps"])
    assert step["state"] == "ran"


def test_run_without_a_variant_exits_with_the_refusal_code(
    two_variants: Dataset, tmp_path: Path
) -> None:
    result = runner.invoke(app, _args("run", two_variants, tmp_path))

    assert result.exit_code == REFUSED_EXIT_CODE
    assert json.loads(result.stdout)["reason"] == "variant_ambiguous"


def test_run_of_a_request_refuses_a_variant(
    two_variants: Dataset, tmp_path: Path
) -> None:
    """A request records its variant at submit."""
    submitted = invoke_json(
        _args("submit", two_variants, tmp_path, "--tracks-run-id", OTHER_VARIANT)
    )

    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--manifest",
            str(two_variants.manifest_path),
            "--request",
            str(submitted["request_id"]),
            "--tracks-run-id",
            OTHER_VARIANT,
        ],
    )

    assert result.exit_code == 1
    assert "--tracks-run-id" in result.output
    assert "not used with --request" in result.output
