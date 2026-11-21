"""``mosaic pipeline retire``: preview a retirement of saved recipes, then apply it.

The retired recipe ``old`` runs ``speed-angvel`` at step size 1 and ``heading``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline import feature_run_removal, graph
from mosaic.core.pipeline.error_kind import ClassifiedError
from mosaic.core.pipeline.graph import (
    CleanupOutcome,
    CleanupStatusWire,
    RetirementPlan,
    RetirementPlanWire,
    RetirementSpec,
    run_pipeline,
)
from mosaic.core.pipeline.graph import retire
from mosaic.core.pipeline.job import Cancelled, JobContext
from mosaic.core.pipeline.index import feature_run_root
from tests.helpers import (
    add_tracks_variant,
    invoke_json,
    make_dataset,
    named_recipe,
    run_id_of,
    speed_heading_recipe,
    speed_recipe_stepping,
)

VARIANT = "convert-trex.0.2-1111111111"

STORAGE = "speed-angvel__from__tracks"

HEADING = "heading__from__tracks"

LOOSE = ["--keep-newest", "0", "--min-age-hours", "0"]

UNKNOWN = "f" * 16
"""A well-formed digest that names no saved recipe."""

runner = CliRunner()


@pytest.fixture
def dataset(tmp_path: Path, lapsed_request_holds: None) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


class _Old:
    """The finished request of the recipe to retire, and the runs it wrote."""

    def __init__(self, ds: Dataset) -> None:
        ran = run_pipeline(ds, speed_heading_recipe())
        self.digest: str = ran.recipe_digest
        self.speed: str = run_id_of(ran, "speed")
        self.heading: str = run_id_of(ran, "heading")

    def exist(self, ds: Dataset) -> tuple[bool, bool]:
        return (
            feature_run_root(ds, STORAGE, self.speed).is_dir(),
            feature_run_root(ds, HEADING, self.heading).is_dir(),
        )


def _retire(ds: Dataset, *arguments: str) -> list[str]:
    return ["pipeline", "retire", "--manifest", str(ds.manifest_path), *arguments]


def _status(report: dict[str, object]) -> CleanupStatusWire:
    """The status wire of an applied retirement's JSON, without the keys it adds."""
    fields = CleanupStatusWire.model_fields
    return CleanupStatusWire.model_validate(
        {key: value for key, value in report.items() if key in fields}
    )


def _managed(dataset: Dataset) -> None:
    with dataset.manifest_path.open("a", encoding="utf-8") as manifest:
        _ = manifest.write("managed_by: mosaic-api\n")


def test_the_dry_run_lists_the_removable_runs_and_removes_none(
    dataset: Dataset,
) -> None:
    old = _Old(dataset)

    plan = RetirementPlanWire.model_validate(
        invoke_json(_retire(dataset, "--retire", old.digest, *LOOSE, "--json"))
    )
    printed = runner.invoke(app, _retire(dataset, "--retire", old.digest, *LOOSE))

    assert plan.retired == (old.digest,)
    assert {run.run_id for run in plan.runs if run.removable} == {
        old.speed,
        old.heading,
    }
    assert plan.removable_runs == 2
    assert old.exist(dataset) == (True, True)
    assert printed.exit_code == 0, printed.output
    assert plan.fingerprint in printed.stdout
    assert old.speed in printed.stdout
    assert old.heading in printed.stdout


def test_apply_removes_the_runs_and_reports_them(dataset: Dataset) -> None:
    old = _Old(dataset)

    report = invoke_json(
        _retire(dataset, "--retire", old.digest, *LOOSE, "--apply", "--json")
    )

    wire = _status(report)
    assert (wire.status, wire.approved_runs, wire.removed_runs) == ("finished", 2, 2)
    assert {run.run_id for run in wire.removed} == {old.speed, old.heading}
    assert report["execution_id"] == wire.execution_id
    assert str(report["run_id"]).startswith("retire-recipes.0.1-")
    assert report["entries_failed"] == 0
    assert old.exist(dataset) == (False, False)


def test_apply_prints_the_removals(dataset: Dataset) -> None:
    old = _Old(dataset)

    result = runner.invoke(
        app, _retire(dataset, "--retire", old.digest, *LOOSE, "--apply")
    )

    assert result.exit_code == 0, result.output
    assert "removed 2 of 2 approved run(s)" in result.stdout
    assert old.exist(dataset) == (False, False)


@pytest.mark.parametrize("as_json", [False, True])
def test_an_unreadable_digest_is_listed_and_the_rest_applied(
    dataset: Dataset, as_json: bool
) -> None:
    old = _Old(dataset)

    result = runner.invoke(
        app,
        _retire(
            dataset,
            "--retire",
            UNKNOWN,
            "--retire",
            old.digest,
            *LOOSE,
            "--apply",
            *(["--json"] if as_json else []),
        ),
    )

    assert result.exit_code == 0, result.output
    assert f"not retired: {UNKNOWN} (recipe_unreadable)" in result.stderr
    assert old.exist(dataset) == (False, False)
    if as_json:
        report = json.loads(result.stdout)
        assert [(each["digest"], each["reason"]) for each in report["not_retired"]] == [
            (UNKNOWN, "recipe_unreadable")
        ]


def test_the_dry_run_lists_an_unreadable_digest(dataset: Dataset) -> None:
    old = _Old(dataset)

    plan = RetirementPlanWire.model_validate(
        invoke_json(
            _retire(dataset, "--retire", UNKNOWN, "--retire", old.digest, "--json")
        )
    )
    printed = runner.invoke(
        app, _retire(dataset, "--retire", UNKNOWN, "--retire", old.digest)
    )

    [refused] = plan.not_retired
    assert (refused.digest, refused.reason) == (UNKNOWN, "recipe_unreadable")
    assert printed.exit_code == 0, printed.output
    assert UNKNOWN in printed.stdout
    assert "recipe_unreadable" in printed.stdout


def test_a_managed_dataset_keeps_every_run(dataset: Dataset) -> None:
    old = _Old(dataset)
    _managed(dataset)

    plan = RetirementPlanWire.model_validate(
        invoke_json(_retire(dataset, "--retire", old.digest, *LOOSE, "--json"))
    )
    report = invoke_json(
        _retire(dataset, "--retire", old.digest, *LOOSE, "--apply", "--json")
    )

    assert {run.run_id: run.reason for run in plan.runs} == {
        old.speed: "managed_dataset",
        old.heading: "managed_dataset",
    }
    assert (report["approved_runs"], report["removed_runs"]) == (0, 0)
    assert old.exist(dataset) == (True, True)


def test_a_keeper_keeps_the_runs_its_plan_expects(
    dataset: Dataset, tmp_path: Path
) -> None:
    old = _Old(dataset)
    keeper = tmp_path / "keeper.recipe.json"
    _ = keeper.write_text(
        named_recipe(speed_recipe_stepping(1), "keeper").model_dump_json()
    )
    keeping = ["--retire", old.digest, "--keeper", f"@{keeper}", *LOOSE]

    plan = RetirementPlanWire.model_validate(
        invoke_json(_retire(dataset, *keeping, "--json"))
    )
    report = invoke_json(_retire(dataset, *keeping, "--apply", "--json"))

    assert [run.run_id for run in plan.runs] == [old.heading]
    assert len(plan.keepers) == 1
    assert report["removed_runs"] == 1
    assert old.exist(dataset) == (True, False)


def test_a_keeper_is_planned_over_the_entries_named(
    dataset: Dataset, tmp_path: Path
) -> None:
    old = _Old(dataset)
    keeper = tmp_path / "keeper.recipe.json"
    _ = keeper.write_text(
        named_recipe(speed_recipe_stepping(2), "keeper").model_dump_json()
    )
    keeping = ["--retire", old.digest, "--keeper", f"@{keeper}", *LOOSE, "--json"]

    narrowed = RetirementPlanWire.model_validate(
        invoke_json(_retire(dataset, *keeping, "--entry", ":seq_a"))
    )
    whole = RetirementPlanWire.model_validate(invoke_json(_retire(dataset, *keeping)))

    assert {run.run_id: run.reason for run in narrowed.runs}[old.speed] == (
        "wider_coverage"
    )
    assert {run.run_id: run.removable for run in whole.runs}[old.speed]


def test_a_retirement_whose_every_removal_raised_exits_one(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = _Old(dataset)

    def refuse(path: Path, *_: object, **__: object) -> None:
        raise OSError(f"busy: {path}")

    monkeypatch.setattr(feature_run_removal.shutil, "rmtree", refuse)

    result = runner.invoke(
        app, _retire(dataset, "--retire", old.digest, *LOOSE, "--apply", "--json")
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    report = _status(payload)
    assert (report.status, report.failed_runs, report.removed_runs) == ("failed", 2, 0)
    assert report.error is not None
    assert (payload["entries_failed"], payload["not_retired"]) == (2, [])
    assert "run_id" not in payload
    assert old.exist(dataset) == (True, True)


def test_a_cancelled_retirement_exits_130(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = _Old(dataset)

    def cancel(*_: object, **__: object) -> CleanupOutcome:
        raise Cancelled()

    monkeypatch.setattr(retire, "run_retirement", cancel)

    result = runner.invoke(
        app, _retire(dataset, "--retire", old.digest, *LOOSE, "--apply", "--json")
    )

    assert result.exit_code == 130, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "cancelled"
    assert set(payload) == {"execution_id", "status"}
    assert old.exist(dataset) == (True, True)


class _Busy(ClassifiedError):
    error_kind = "conflict"


def test_a_classified_error_of_the_removal_is_reported(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = _Old(dataset)

    def busy(*_: object, **__: object) -> CleanupOutcome:
        raise _Busy("the dataset is busy")

    monkeypatch.setattr(retire, "run_retirement", busy)

    result = runner.invoke(
        app, _retire(dataset, "--retire", old.digest, *LOOSE, "--apply", "--json")
    )

    assert result.exit_code == 1, result.output
    envelope = json.loads(result.stdout)
    assert (envelope["kind"], envelope["message"]) == (
        "conflict",
        "the dataset is busy",
    )
    assert old.exist(dataset) == (True, True)


def test_a_classified_error_of_the_plan_is_reported(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = _Old(dataset)

    def busy(*_: object, **__: object) -> RetirementPlan:
        raise _Busy("the keeper does not plan")

    monkeypatch.setattr(graph, "plan_retirement", busy)

    result = runner.invoke(app, _retire(dataset, "--retire", old.digest, "--json"))

    assert result.exit_code == 1, result.output
    envelope = json.loads(result.stdout)
    assert (envelope["kind"], envelope["message"]) == (
        "conflict",
        "the keeper does not plan",
    )


def test_what_the_removal_prints_goes_to_stderr(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = _Old(dataset)
    real = retire.run_retirement

    def noisy(
        ds: Dataset, plan: RetirementSpec, manager: str, *, ctx: JobContext
    ) -> CleanupOutcome:
        print("removing")
        return real(ds, plan, manager, ctx=ctx)

    monkeypatch.setattr(retire, "run_retirement", noisy)

    result = runner.invoke(
        app, _retire(dataset, "--retire", old.digest, *LOOSE, "--apply", "--json")
    )

    assert result.exit_code == 0, result.output
    assert _status(json.loads(result.stdout)).removed_runs == 2
    assert "removing" in result.stderr


def test_retire_needs_a_digest(dataset: Dataset) -> None:
    result = runner.invoke(app, _retire(dataset))

    assert result.exit_code == 2
    assert "--retire" in result.stderr
