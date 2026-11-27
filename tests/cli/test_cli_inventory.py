"""``mosaic inventory``: the verb that answers what a dataset holds."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.cli._features import build_feature
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline import identity_scheme as identity_scheme_module
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.inventory import ARTIFACT_KINDS, InventoryWire
from mosaic.core.selections import save_selection
from tests.helpers import (
    add_tracks_variant,
    invoke_ok,
    kmeans_fit,
    make_dataset,
    old_merged_run,
    reconvert_entry,
    run_step,
)

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


def test_the_kind_help_lists_every_artifact_kind() -> None:
    result = invoke_ok(["inventory", "--help"])
    after_kind = result.output.split("--kind", 1)[1]
    kind_help = re.split(r"--[a-z]", after_kind, maxsplit=1)[0]

    assert set(ARTIFACT_KINDS) <= set(re.findall(r"[a-z-]+", kind_help))


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


def _fitted_dataset(base: Path, *, converted_again: bool) -> tuple[Dataset, str]:
    """A k-means fit over two tables, and the ref of its templates run.

    With *converted_again*, one entry's table is converted again and the run
    that feeds the fit computes it again. The fit then read inputs that moved.
    """
    ds = make_dataset(base)
    variant = "convert-trex.0.2-1111111111"
    add_tracks_variant(ds, variant, "seq_a", "seq_b", std_format="mosaic_v1")
    runs = kmeans_fit(ds)
    if converted_again:
        reconvert_entry(ds, ("", "seq_a"), composition="converted-again")
        _ = run_step(ds, "speed-angvel", None, {})
    return ds, f"feature/{runs.templates.feature}/{runs.templates.run_id}"


def test_show_prints_the_refit_of_a_fit_whose_inputs_changed(tmp_path: Path) -> None:
    ds, ref = _fitted_dataset(tmp_path / "moved", converted_again=True)

    result = invoke_ok(["inventory", "show", ref, "--manifest", str(ds.manifest_path)])

    assert "Refit: 1 moved (seq_a)" in result.stdout


def test_show_prints_no_refit_for_a_fit_whose_inputs_are_unchanged(
    tmp_path: Path,
) -> None:
    ds, ref = _fitted_dataset(tmp_path / "steady", converted_again=False)

    result = invoke_ok(["inventory", "show", ref, "--manifest", str(ds.manifest_path)])

    assert "Refit" not in result.stdout


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


def test_a_selection_token_narrows_the_listing_to_its_entries(
    scenario_dataset: Dataset,
) -> None:
    _ = scenario_dataset.run_feature(build_feature("speed-angvel", None, None))
    _ = save_selection(scenario_dataset, "a-only", [("", "seq_a")])

    narrowed = _wire(scenario_dataset, "--kind", "feature", "--entry", "@a-only")

    assert narrowed.artifacts[0].coverage.target == 1


def test_a_missing_selection_is_refused_by_inventory(
    scenario_dataset: Dataset,
) -> None:
    result = runner.invoke(
        app,
        [
            "inventory",
            "--manifest",
            str(scenario_dataset.manifest_path),
            "--entry",
            "@nope",
            "--json",
        ],
    )

    assert result.exit_code == 1
    envelope = ErrorEnvelope.model_validate_json(result.stdout)
    assert envelope.reason == "selection_not_found"


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


def test_a_run_merged_before_summary_keys_joined_is_noted_as_reconcile_pending(
    scenario_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(identity_scheme_module, "FEATURE_IDENTITY_SCHEME", "7")
    result, _ = old_merged_run(scenario_dataset, unjoined="seq_b")

    listed = runner.invoke(
        app,
        [
            "inventory",
            "--kind",
            "feature",
            "--manifest",
            str(scenario_dataset.manifest_path),
        ],
    )

    assert listed.exit_code == 0
    assert (
        f"note: 2 entry(ies) of {result.run_id} were merged before summary keys "
        "joined; run mosaic reconcile --apply"
    ) in listed.output
