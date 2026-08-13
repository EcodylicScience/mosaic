"""``mosaic inventory``: the verb that answers what a dataset holds."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.cli._features import build_feature
from mosaic.core.dataset import Dataset

runner = CliRunner()


def _json(ds: Dataset, *extra: str) -> dict[str, object]:
    result = runner.invoke(
        app,
        ["inventory", "--manifest", str(ds.manifest_path), "--json", *extra],
    )
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_it_reports_a_computed_run(scenario_dataset: Dataset) -> None:
    run_id = str(
        scenario_dataset.run_feature(build_feature("speed-angvel", None, None)).run_id
    )

    payload = _json(scenario_dataset, "--kind", "feature")

    artifacts = payload["artifacts"]
    assert any(
        entry["run_id"] == run_id and entry["status"] == "complete"
        for entry in artifacts
    )


def test_coverage_crosses_the_wire_as_a_summary(scenario_dataset: Dataset) -> None:
    """A wide dataset under several kinds would otherwise emit megabytes, and a
    count beside a sample is what a reader acts on."""
    _ = scenario_dataset.run_feature(build_feature("speed-angvel", None, None))

    payload = _json(scenario_dataset, "--kind", "feature")

    entry = payload["artifacts"][0]
    assert entry["covered"] == 2
    assert entry["target"] == 2
    assert entry["missing_sample"] == []
    assert entry["has_more_missing"] is False


def test_the_ops_kinds_are_reported_rather_than_unavailable(
    scenario_dataset: Dataset,
) -> None:
    """The command imports the producers, so a user never meets the layering."""
    payload = _json(scenario_dataset)

    assert payload["unavailable_kinds"] == []


def test_a_dataset_with_nothing_computed_says_so(
    make_media_dataset, tmp_path: Path
) -> None:
    ds = make_media_dataset(tmp_path / "bare")

    result = runner.invoke(app, ["inventory", "--manifest", str(ds.manifest_path)])

    assert result.exit_code == 0
    assert "No artifacts" in result.output or result.output.strip()


def test_json_is_exactly_one_value_on_stdout(scenario_dataset: Dataset) -> None:
    """The stream contract every read-only verb keeps: breadcrumbs go to stderr."""
    _ = scenario_dataset.run_feature(build_feature("speed-angvel", None, None))

    result = runner.invoke(
        app, ["inventory", "--manifest", str(scenario_dataset.manifest_path), "--json"]
    )

    assert result.exit_code == 0
    _ = json.loads(result.stdout)


def test_an_unknown_kind_is_refused_rather_than_reporting_nothing(
    scenario_dataset: Dataset,
) -> None:
    """A misspelled kind that silently reports nothing reads as "this dataset
    holds none of those" -- the same output as the true answer."""
    result = runner.invoke(
        app,
        [
            "inventory",
            "--manifest",
            str(scenario_dataset.manifest_path),
            "--kind",
            "featrue",
        ],
    )

    assert result.exit_code != 0
    assert "unknown artifact kind" in result.output
