"""``mosaic inventory``: the verb that answers what a dataset holds."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.cli._features import build_feature
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.inventory import ARTIFACT_KINDS, InventoryWire
from tests.helpers import invoke_ok

runner = CliRunner()


def _wire(ds: Dataset, *arguments: str) -> InventoryWire:
    """Run ``mosaic inventory`` with *arguments* and read its stdout as the wire."""
    result = invoke_ok(
        ["inventory", *arguments, "--manifest", str(ds.manifest_path), "--json"]
    )
    return InventoryWire.model_validate_json(result.stdout)


def test_it_reports_a_computed_run(scenario_dataset: Dataset) -> None:
    run_id = str(
        scenario_dataset.run_feature(build_feature("speed-angvel", None, None)).run_id
    )

    wire = _wire(scenario_dataset, "--kind", "feature")

    assert any(
        artifact.run_id == run_id and artifact.status == "complete"
        for artifact in wire.artifacts
    )


def test_coverage_crosses_the_wire_as_a_summary(scenario_dataset: Dataset) -> None:
    """A wide dataset under several kinds would otherwise emit megabytes, and a
    count beside a sample is what a reader acts on."""
    _ = scenario_dataset.run_feature(build_feature("speed-angvel", None, None))

    coverage = _wire(scenario_dataset, "--kind", "feature").artifacts[0].coverage
    assert (coverage.covered, coverage.target) == (2, 2)
    assert coverage.missing_sample == ()
    assert coverage.has_more is False


def test_the_ops_kinds_are_reported_rather_than_unavailable(
    scenario_dataset: Dataset,
) -> None:
    """The command imports the producers, so a user never meets the layering."""
    every_kind = [argument for kind in ARTIFACT_KINDS for argument in ("--kind", kind)]

    assert _wire(scenario_dataset, *every_kind).unavailable_kinds == ()


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


def test_show_reports_one_artifact_in_depth(scenario_dataset: Dataset) -> None:
    run_id = str(
        scenario_dataset.run_feature(build_feature("speed-angvel", None, None)).run_id
    )
    ref = f"feature/speed-angvel__from__tracks/{run_id}"

    [artifact] = _wire(scenario_dataset, "show", ref).artifacts

    assert artifact.ref == ref
    assert artifact.detail is not None
    assert artifact.detail.rows == 2


def test_show_takes_a_ref_from_the_plain_listing(scenario_dataset: Dataset) -> None:
    """A tracks variant's ref is not its printed name and run id joined, so the
    listing prints the ref itself."""
    listing = invoke_ok(
        ["inventory", "--manifest", str(scenario_dataset.manifest_path)]
    ).stdout.splitlines()
    header = listing[0].split("\t")
    refs = [line.split("\t")[header.index("ref")] for line in listing[1:]]
    [ref] = [ref for ref in refs if ref.startswith("tracks-variant/")]

    [artifact] = _wire(scenario_dataset, "show", ref).artifacts

    assert artifact.kind == "tracks-variant"
    assert artifact.ref == ref


def test_show_reads_the_manifest_given_before_it(scenario_dataset: Dataset) -> None:
    run_id = str(
        scenario_dataset.run_feature(build_feature("speed-angvel", None, None)).run_id
    )
    ref = f"feature/speed-angvel__from__tracks/{run_id}"

    result = invoke_ok(
        [
            "inventory",
            "--manifest",
            str(scenario_dataset.manifest_path),
            "show",
            ref,
            "--json",
        ]
    )

    [artifact] = InventoryWire.model_validate_json(result.stdout).artifacts
    assert artifact.ref == ref


def test_a_listing_option_before_show_is_refused(scenario_dataset: Dataset) -> None:
    """The listing's options would otherwise be dropped without a word."""
    result = runner.invoke(
        app,
        [
            "inventory",
            "--kind",
            "feature",
            "show",
            "feature/nothing/0.1-0000000000",
            "--manifest",
            str(scenario_dataset.manifest_path),
        ],
    )

    assert result.exit_code == 2
    assert "--kind given before 'show'" in result.output


def test_show_of_an_unknown_ref_reports_not_found(scenario_dataset: Dataset) -> None:
    result = runner.invoke(
        app,
        [
            "inventory",
            "show",
            "feature/nothing/0.1-0000000000",
            "--manifest",
            str(scenario_dataset.manifest_path),
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert ErrorEnvelope.model_validate_json(result.stdout).kind == "not_found"


def test_a_limited_listing_pages_through_every_artifact(
    scenario_dataset: Dataset,
) -> None:
    _ = scenario_dataset.run_feature(build_feature("speed-angvel", None, None))
    whole = _wire(scenario_dataset)

    refs: list[str] = []
    cursor: list[str] = []
    while True:
        page = _wire(scenario_dataset, "--limit", "1", *cursor)
        refs.extend(artifact.ref for artifact in page.artifacts)
        if page.next_cursor is None:
            break
        cursor = ["--cursor", page.next_cursor]

    assert refs == [artifact.ref for artifact in whole.artifacts]


def test_status_not_and_entry_narrow_the_listing(scenario_dataset: Dataset) -> None:
    _ = scenario_dataset.run_feature(build_feature("speed-angvel", None, None))

    narrowed = _wire(scenario_dataset, "--kind", "feature", "--entry", ":seq_a")
    without = _wire(scenario_dataset, "--kind", "feature", "--status-not", "complete")

    assert narrowed.artifacts[0].coverage.target == 1
    assert without.artifacts == ()


def test_an_unknown_status_is_refused(scenario_dataset: Dataset) -> None:
    result = runner.invoke(
        app,
        [
            "inventory",
            "--manifest",
            str(scenario_dataset.manifest_path),
            "--status-not",
            "finished",
        ],
    )

    assert result.exit_code != 0
    assert "unknown status" in result.output
