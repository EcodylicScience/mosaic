"""The op ``retire-recipes``: an approved retirement run as a queue job.

The retired recipe ``old`` runs ``speed-angvel`` and ``heading``. Its plan is
approved, and the op receives the approval as params.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline import feature_run_removal
from mosaic.core.pipeline.graph import (
    CleanupPolicy,
    CleanupTarget,
    Recipe,
    RetirementSpec,
    check_recipe,
    plan_retirement,
    run_pipeline,
)
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.job import Cancelled, CancelToken
from mosaic.core.pipeline.ops import registered_op, run_op
from mosaic.runlog import (
    new_execution_id,
    read_removal_records,
    reduce_run_log,
    run_log_dir,
    run_log_path,
)
from tests.helpers import (
    add_tracks_variant,
    make_dataset,
    run_id_of,
    speed_heading_recipe,
)

runner = CliRunner()

VARIANT = "convert-trex.0.2-1111111111"

STORAGE = "speed-angvel__from__tracks"

HEADING = "heading__from__tracks"

LOOSE = CleanupPolicy(keep_newest=0, min_age_hours=0)

KIND = "retire-recipes"

MAX_ARGUMENT_BYTES = 128 * 1024
"""Linux's cap on one command-line argument (``MAX_ARG_STRLEN``)."""


@pytest.fixture
def dataset(tmp_path: Path, lapsed_request_holds: None) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


class _Old:
    """The retired recipe's runs and the approved retirement."""

    def __init__(self, ds: Dataset) -> None:
        ran = run_pipeline(ds, speed_heading_recipe())
        self.speed: str = run_id_of(ran, "speed")
        self.heading: str = run_id_of(ran, "heading")
        self.spec: RetirementSpec = plan_retirement(
            ds, [ran.recipe_digest], [], LOOSE, ""
        ).spec

    def params(self, spec: RetirementSpec | None = None) -> dict[str, object]:
        return {"retirement": (spec or self.spec).model_dump(mode="json")}


def _exists(ds: Dataset, storage: str, run_id: str) -> bool:
    return feature_run_root(ds, storage, run_id).is_dir()


def _run_cli(dataset: Dataset, *params: str) -> tuple[int, dict[str, object]]:
    result = runner.invoke(
        app,
        [
            "run",
            "--manifest",
            str(dataset.manifest_path),
            "--json",
            "--kind",
            KIND,
            "--params",
            *params,
        ],
    )
    return result.exit_code, json.loads(result.stdout)


def _fail_first_removal(monkeypatch: pytest.MonkeyPatch) -> None:
    rmtree = feature_run_removal.shutil.rmtree
    calls: list[Path] = []

    def refuse_first(path: Path) -> None:
        calls.append(path)
        if len(calls) == 1:
            raise OSError(f"busy: {path}")
        rmtree(path)

    monkeypatch.setattr(feature_run_removal.shutil, "rmtree", refuse_first)


# --- the registration ----------------------------------------------------------------


def test_the_op_declares_a_scope_free_cleanup_in_the_pipeline_domain() -> None:
    op = registered_op(KIND)

    assert (op.domain, op.category, op.version) == ("pipeline", "cleanup", "0.1")
    assert (op.scope_takes, op.scope_dependent) == ("none", False)


def test_the_run_id_is_the_identity_of_the_params(dataset: Dataset) -> None:
    old = _Old(dataset)
    op = registered_op(KIND)()
    params = op.Params.model_validate(old.params())

    identity = op.plan_identity(dataset, params, dataset.resolve_scope(None))
    other = op.Params.model_validate(
        old.params(RetirementSpec(planned_at="2026-01-01T00:00:00+00:00"))
    )

    assert identity.run_id.startswith(f"{KIND}.0.1-")
    assert (
        identity.run_id
        == op.plan_identity(dataset, params, dataset.resolve_scope(None)).run_id
    )
    assert (
        op.plan_identity(dataset, other, dataset.resolve_scope(None)).run_id
        != identity.run_id
    )


# --- running it -----------------------------------------------------------------------


def test_run_op_removes_the_approved_runs_and_logs_the_attempt(
    dataset: Dataset,
) -> None:
    old = _Old(dataset)
    execution_id = new_execution_id()

    run_id = run_op(dataset, KIND, old.params(), execution_id=execution_id)

    assert run_id.startswith(f"{KIND}.0.1-")
    assert not _exists(dataset, STORAGE, old.speed)
    assert not _exists(dataset, HEADING, old.heading)
    snapshot = reduce_run_log(run_log_path(dataset.base_dir, execution_id))
    assert snapshot is not None
    assert (snapshot["kind"], snapshot["status"], snapshot["run_id"]) == (
        KIND,
        "finished",
        run_id,
    )
    records = read_removal_records(run_log_dir(dataset.base_dir), execution_id)
    assert {(record["run_id"], record["state"]) for record in records} == {
        (old.speed, "removed"),
        (old.heading, "removed"),
    }


@pytest.mark.parametrize(
    ("manager", "removed"),
    [("mosaic-api", True), ("someone-else", False), ("", False), (None, False)],
)
def test_the_manager_param_lifts_the_hold_of_a_managed_dataset(
    dataset: Dataset, manager: str | None, removed: bool
) -> None:
    """``None`` leaves the param out, which acts for no manager."""
    ran = run_pipeline(dataset, speed_heading_recipe())
    with dataset.manifest_path.open("a", encoding="utf-8") as manifest:
        _ = manifest.write("managed_by: mosaic-api\n")
    managed = Dataset(manifest_path=dataset.manifest_path).load(ensure_roots=False)
    spec = plan_retirement(managed, [ran.recipe_digest], [], LOOSE, "mosaic-api").spec

    params: dict[str, object] = {"retirement": spec.model_dump(mode="json")}
    if manager is not None:
        params["manager"] = manager

    _ = run_op(managed, KIND, params)

    assert _exists(managed, STORAGE, run_id_of(ran, "speed")) is not removed


def test_a_rerun_judges_again_and_removes_nothing_twice(dataset: Dataset) -> None:
    old = _Old(dataset)
    _ = run_op(dataset, KIND, old.params(), overwrite=True)
    execution_id = new_execution_id()

    _ = run_op(dataset, KIND, old.params(), overwrite=False, execution_id=execution_id)

    records = read_removal_records(run_log_dir(dataset.base_dir), execution_id)
    assert {record["state"] for record in records} == {"kept"}


def test_a_cancel_stops_the_op_as_a_cancelled_attempt(dataset: Dataset) -> None:
    old = _Old(dataset)
    token = CancelToken()
    token.cancel()
    execution_id = new_execution_id()

    with pytest.raises(Cancelled):
        _ = run_op(
            dataset,
            KIND,
            old.params(),
            execution_id=execution_id,
            cancel_token=token,
        )

    snapshot = reduce_run_log(run_log_path(dataset.base_dir, execution_id))
    assert snapshot is not None
    assert snapshot["status"] == "cancelled"
    assert _exists(dataset, STORAGE, old.speed)


# --- mosaic run ---------------------------------------------------------------------------


def test_mosaic_run_reports_a_clean_retirement_as_finished(dataset: Dataset) -> None:
    old = _Old(dataset)

    code, payload = _run_cli(dataset, json.dumps(old.params()))

    assert code == 0
    assert (payload["kind"], payload["status"], payload["entries_failed"]) == (
        KIND,
        "finished",
        0,
    )
    assert not _exists(dataset, STORAGE, old.speed)


def test_mosaic_run_reports_partial_after_one_failed_removal(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = _Old(dataset)
    _fail_first_removal(monkeypatch)

    code, payload = _run_cli(dataset, json.dumps(old.params()))

    assert code == 0
    assert (payload["status"], payload["entries_failed"]) == ("partial", 1)
    assert [
        _exists(dataset, STORAGE, old.speed),
        _exists(dataset, HEADING, old.heading),
    ].count(False) == 1


def test_mosaic_run_reads_params_past_the_argument_limit_from_a_file(
    dataset: Dataset, tmp_path: Path
) -> None:
    old = _Old(dataset)
    extra = [
        CleanupTarget(storage=STORAGE, run_id=f"0.1-{index:010d}")
        for index in range(2500)
    ]
    many = old.spec.model_copy(update={"approved": [*old.spec.approved, *extra]})
    document = json.dumps(old.params(many))
    assert len(document.encode()) > MAX_ARGUMENT_BYTES
    params_file = tmp_path / "params" / "retire.json"
    params_file.parent.mkdir()
    _ = params_file.write_text(document)

    code, payload = _run_cli(dataset, f"@{params_file}")

    assert code == 0
    assert (payload["status"], payload["entries_failed"]) == ("finished", 0)
    assert not _exists(dataset, STORAGE, old.speed)
    assert not _exists(dataset, HEADING, old.heading)


# --- no recipe runs it -----------------------------------------------------------------------


def test_a_recipe_step_of_the_op_is_refused() -> None:
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "retiring",
            "steps": [{"id": "retire", "type": "op", "kind": KIND, "params": {}}],
        }
    )

    [problem] = check_recipe(recipe)

    assert problem.where == "kind"
    assert KIND in problem.message
    assert "may not run as a graph step" in problem.message
