"""A submission beside an open request that computes one of its runs.

A request records the runs its plan leaves to compute. A feature run takes no
claim, so two requests computing one run at once both compute it, whatever their
recipes. A submission reports the open requests, and refuses as a conflict when
asked to. A request that recorded no runs is matched by its recipe and scope.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.error_envelope import ErrorEnvelope, error_envelope
from mosaic.core.pipeline.graph import (
    BoundRef,
    DeclarationCatalog,
    DuplicateRequest,
    Plan,
    Recipe,
    OpenDuplicate,
    Request,
    execute_step,
    computed_runs,
    computes_path,
    describe_duplicates,
    load_computes,
    open_duplicates,
    plan_pipeline,
    recipe_path,
    request_path,
    requests_root,
    run_pipeline,
    save_request,
    submit_request,
)
from mosaic.core.pipeline.graph import request as request_module
from mosaic.core.pipeline.graph.digest import document_digest
from mosaic.core.pipeline.inventory.cache import InventoryCache
from mosaic.core.scope import Scope
from mosaic.runlog import JsonlRunLog, run_log_path
from tests.helpers import (
    CHAIN_RECIPE,
    CONVERT_TRACKS_STEP,
    SPEED_RECIPE,
    add_tracks_variant,
    calms21_dataset,
    exited_pid,
    invoke_json,
    make_dataset,
    named_recipe,
    run_speed,
    speed_recipe,
    speed_recipe_stepping,
    write_driver_lease,
)

SPEED_STORAGE = "speed-angvel__from__tracks"

VARIANT = "convert-trex.0.2-1111111111"
SEQ_A = Scope(entries=[("", "seq_a")])
SEQ_B = Scope(entries=[("", "seq_b")])
LEGACY = "0000000000000000000000000A"
"""An older request id, which sorts before every one minted now."""
MISSING = "0000000000000000000000000B"

runner = CliRunner()


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two schema-valid track tables before any feature run."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return dataset


def _request_count(dataset: Dataset) -> int:
    return len(list(requests_root(dataset.base_dir).glob("*.json")))


def _open(dataset: Dataset, recipe: Recipe | None = None) -> tuple[str, ...]:
    """The open requests that *recipe*, or the speed recipe, meets over every entry."""
    recipe = speed_recipe() if recipe is None else recipe
    plan = plan_pipeline(dataset, recipe)
    found = open_duplicates(
        dataset.base_dir, plan.recipe_digest, Scope(), computed_runs(plan)
    )
    return tuple(duplicate.request_id for duplicate in found)


def _ids(duplicates: tuple[OpenDuplicate, ...]) -> tuple[str, ...]:
    return tuple(duplicate.request_id for duplicate in duplicates)


def test_an_open_request_over_the_same_scope_is_reported(tracked: Dataset) -> None:
    first = submit_request(tracked, speed_recipe())

    second = submit_request(tracked, speed_recipe())

    assert _ids(second.duplicates) == (first.request.request_id,)


def test_open_requests_are_listed_newest_first(tracked: Dataset) -> None:
    first = submit_request(tracked, speed_recipe())
    second = submit_request(tracked, speed_recipe())

    assert _open(tracked) == (second.request.request_id, first.request.request_id)


def test_a_request_over_an_overlapping_scope_is_reported(tracked: Dataset) -> None:
    first = submit_request(tracked, speed_recipe(), scope=SEQ_A)

    second = submit_request(tracked, speed_recipe())

    assert _ids(second.duplicates) == (first.request.request_id,)


def test_a_finished_request_is_not_open(tracked: Dataset) -> None:
    first = submit_request(tracked, speed_recipe())
    outcome = execute_step(tracked, first.request, "speed")
    assert outcome.state == "ran"

    second = submit_request(tracked, speed_recipe())

    assert second.duplicates == ()


def test_a_request_over_a_disjoint_scope_is_not_reported(tracked: Dataset) -> None:
    _ = submit_request(tracked, speed_recipe(), scope=SEQ_A)

    second = submit_request(tracked, speed_recipe(), scope=SEQ_B)

    assert second.duplicates == ()


def test_a_request_computing_other_runs_is_not_reported(tracked: Dataset) -> None:
    _ = submit_request(tracked, speed_recipe())

    second = submit_request(tracked, speed_recipe_stepping(2))

    assert second.duplicates == ()


def test_a_request_whose_driver_lease_is_stale_is_not_open(tracked: Dataset) -> None:
    orphaned = submit_request(tracked, speed_recipe()).request
    _ = write_driver_lease(tracked.base_dir, orphaned.request_id, stale=True)

    assert _open(tracked) == ()


def test_a_request_with_a_stale_lease_and_a_running_step_is_open(
    tracked: Dataset,
) -> None:
    """The step that a killed driver left running still computes the work."""
    driven = submit_request(tracked, speed_recipe()).request
    execution_id = next(iter(driven.step_executions.values()))
    path = run_log_path(tracked.base_dir, execution_id)
    with JsonlRunLog(path, execution_id) as log:
        log.started(
            kind="feature", target="speed", host=socket.gethostname(), pid=os.getpid()
        )
    _ = write_driver_lease(tracked.base_dir, driven.request_id, stale=True)

    assert _open(tracked) == (driven.request_id,)


def test_a_request_with_a_live_driver_lease_is_open(tracked: Dataset) -> None:
    driven = submit_request(tracked, speed_recipe()).request
    _ = write_driver_lease(tracked.base_dir, driven.request_id, stale=False)

    assert _open(tracked) == (driven.request_id,)


def test_a_request_whose_step_is_dead_is_not_an_open_duplicate(
    tracked: Dataset,
) -> None:
    """No lease, as for a request a queue drives, and the step's process exited."""
    dead = submit_request(tracked, speed_recipe()).request
    execution_id = next(iter(dead.step_executions.values()))
    with JsonlRunLog(run_log_path(tracked.base_dir, execution_id), execution_id) as log:
        log.started(
            kind="feature", target="speed", host=socket.gethostname(), pid=exited_pid()
        )

    assert _open(tracked) == ()
    assert submit_request(tracked, speed_recipe()).duplicates == ()


def test_a_dataset_without_requests_has_none_open(tracked: Dataset) -> None:
    assert _open(tracked) == ()


def test_a_refused_duplicate_is_a_conflict_and_writes_nothing(
    tracked: Dataset,
) -> None:
    first = submit_request(tracked, speed_recipe())

    with pytest.raises(DuplicateRequest) as raised:
        _ = submit_request(tracked, speed_recipe(), refuse_duplicates=True)

    envelope = error_envelope(raised.value)
    assert envelope.kind == "conflict"
    assert envelope.detail == {
        "recipe_digest": first.request.recipe_digest,
        "duplicates": [
            {
                "request_id": first.request.request_id,
                "steps": ["speed"],
                "steps_omitted": 0,
            }
        ],
    }
    assert _request_count(tracked) == 1


def test_refusing_duplicates_submits_when_none_is_open(tracked: Dataset) -> None:
    _ = submit_request(tracked, speed_recipe(), scope=SEQ_A)

    second = submit_request(
        tracked, speed_recipe(), scope=SEQ_B, refuse_duplicates=True
    )

    assert second.request_path.exists()


def test_a_request_written_while_planning_refuses_the_submission(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check runs again after planning, beside the write of the request."""
    plan_pipeline = request_module.plan_pipeline
    written: list[str] = []

    def plan_while_another_submits(
        dataset: Dataset,
        recipe: Recipe,
        *,
        request: Request | None = None,
        inventory: InventoryCache | None = None,
        catalog: DeclarationCatalog | None = None,
        revalidate: bool = True,
    ) -> Plan:
        monkeypatch.setattr(request_module, "plan_pipeline", plan_pipeline)
        written.append(submit_request(dataset, recipe).request.request_id)
        return plan_pipeline(
            dataset,
            recipe,
            request=request,
            inventory=inventory,
            catalog=catalog,
            revalidate=revalidate,
        )

    monkeypatch.setattr(request_module, "plan_pipeline", plan_while_another_submits)

    with pytest.raises(DuplicateRequest) as raised:
        _ = submit_request(tracked, speed_recipe(), refuse_duplicates=True)

    assert [each.request_id for each in raised.value.duplicates] == written
    assert _request_count(tracked) == 1


def test_an_unreadable_request_file_is_skipped(tracked: Dataset) -> None:
    first = submit_request(tracked, speed_recipe())
    broken = request_path(tracked.base_dir, "0" * 26)
    _ = broken.write_text("{not json")

    with pytest.warns(UserWarning, match="skipped request file"):
        second = submit_request(tracked, speed_recipe())

    assert _ids(second.duplicates) == (first.request.request_id,)


def test_an_invalid_request_of_the_recipe_is_skipped(tracked: Dataset) -> None:
    first = submit_request(tracked, speed_recipe())
    invalid = {"recipe_digest": first.request.recipe_digest, "unknown": True}
    _ = request_path(tracked.base_dir, "0" * 26).write_text(json.dumps(invalid))

    with pytest.warns(UserWarning, match="skipped request file"):
        second = submit_request(tracked, speed_recipe())

    assert _ids(second.duplicates) == (first.request.request_id,)


# --- the CLI ---------------------------------------------------------------------


def _speed_with(step_size: object) -> dict[str, object]:
    """The speed recipe with *step_size* written as given."""
    document = json.loads(json.dumps(SPEED_RECIPE))
    document["steps"][0]["params"] = {"step_size": step_size}
    return document


def test_a_float_and_an_integer_submission_are_duplicates(tracked: Dataset) -> None:
    """A hand-written ``1.0`` and a client's ``1`` are one recipe, so one request."""
    first = submit_request(tracked, Recipe.model_validate(_speed_with(1.0)))

    second = submit_request(tracked, Recipe.model_validate(_speed_with(1)))

    assert second.request.recipe_digest == first.request.recipe_digest
    assert _ids(second.duplicates) == (first.request.request_id,)


def _submit_args(dataset: Dataset, tmp_path: Path, *extra: str) -> list[str]:
    path = tmp_path / "speed.recipe.json"
    _ = path.write_text(json.dumps(SPEED_RECIPE))
    return [
        "pipeline",
        "submit",
        "--recipe",
        f"@{path}",
        "--manifest",
        str(dataset.manifest_path),
        "--json",
        *extra,
    ]


def test_submit_json_reports_the_open_requests(
    tracked: Dataset, tmp_path: Path
) -> None:
    first = invoke_json(_submit_args(tracked, tmp_path))

    result = runner.invoke(app, _submit_args(tracked, tmp_path))

    assert result.exit_code == 0
    assert first["duplicates"] == []
    assert json.loads(result.stdout)["duplicates"] == [
        {"request_id": first["request_id"], "steps": ["speed"], "steps_omitted": 0}
    ]
    assert "warning:" in result.stderr
    assert str(first["request_id"]) in result.stderr


def test_submit_refusing_duplicates_exits_with_a_conflict(
    tracked: Dataset, tmp_path: Path
) -> None:
    _ = invoke_json(_submit_args(tracked, tmp_path))

    result = runner.invoke(app, _submit_args(tracked, tmp_path, "--refuse-duplicates"))

    assert result.exit_code == 1
    assert ErrorEnvelope.model_validate_json(result.stdout).kind == "conflict"
    assert _request_count(tracked) == 1


def test_pipeline_run_warns_of_an_open_request(
    tracked: Dataset, tmp_path: Path
) -> None:
    first = invoke_json(_submit_args(tracked, tmp_path))
    args = _submit_args(tracked, tmp_path)
    args[1] = "run"

    result = runner.invoke(app, args)

    assert result.exit_code == 0
    assert "warning:" in result.stderr
    assert str(first["request_id"]) in result.stderr


def test_pipeline_run_with_a_float_file_warns_of_an_integer_request(
    tracked: Dataset, tmp_path: Path
) -> None:
    first = submit_request(tracked, Recipe.model_validate(_speed_with(1)))
    path = tmp_path / "speed.float.recipe.json"
    _ = path.write_text(json.dumps(_speed_with(1.0)))

    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--recipe",
            f"@{path}",
            "--manifest",
            str(tracked.manifest_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert first.request.request_id in result.stderr


def test_a_request_saved_under_the_legacy_digest_is_open_for_its_recipe(
    tracked: Dataset,
) -> None:
    """A request an earlier canonical form named, which recorded no runs."""
    recipe = Recipe.model_validate(_speed_with(1))
    submitted = submit_request(tracked, recipe)
    legacy = _speed_with(1.0)
    name = document_digest(cast("JsonValue", legacy))
    _ = recipe_path(tracked.base_dir, name).write_text(json.dumps(legacy))
    older = submitted.request.model_copy(
        update={"request_id": LEGACY, "recipe_digest": name}
    )
    _ = save_request(tracked.base_dir, older)

    assert name != submitted.request.recipe_digest
    assert _open(tracked, recipe) == (submitted.request.request_id, LEGACY)


def test_a_request_whose_recipe_file_is_missing_is_not_a_duplicate(
    tracked: Dataset,
) -> None:
    """A request that recorded no runs is matched by its recipe, which is gone."""
    submitted = submit_request(tracked, speed_recipe())
    orphan = submitted.request.model_copy(
        update={"request_id": MISSING, "recipe_digest": "0" * 16}
    )
    _ = save_request(tracked.base_dir, orphan)

    assert _open(tracked) == (submitted.request.request_id,)


# --- the runs a request computes ------------------------------------------------


def _chain() -> Recipe:
    return Recipe.model_validate(CHAIN_RECIPE)


def test_a_request_of_another_recipe_computing_the_same_fit_is_reported(
    tracked: Dataset,
) -> None:
    """The fit is one run whichever recipe asks for it."""
    _ = run_pipeline(tracked, speed_recipe())
    first = submit_request(tracked, _chain())

    second = submit_request(tracked, named_recipe(_chain(), "another name"))

    assert second.request.recipe_digest != first.request.recipe_digest
    assert second.duplicates == (
        OpenDuplicate(first.request.request_id, steps=("templates",)),
    )


def test_fits_over_disjoint_step_scopes_of_one_recipe_are_not_reported(
    tracked: Dataset,
) -> None:
    _ = run_pipeline(tracked, speed_recipe())
    _ = submit_request(tracked, _chain(), step_scopes={"templates": SEQ_A})

    second = submit_request(
        tracked, _chain(), step_scopes={"templates": SEQ_B}, refuse_duplicates=True
    )

    assert second.duplicates == ()


def test_one_fit_named_by_a_step_scope_and_by_a_request_scope_is_reported(
    tracked: Dataset,
) -> None:
    """A fit over ``seq_a`` is one run, whichever scope narrows it to ``seq_a``."""
    _ = run_pipeline(tracked, speed_recipe())
    first = submit_request(tracked, _chain(), scope=SEQ_A)

    second = submit_request(
        tracked,
        named_recipe(_chain(), "another name"),
        step_scopes={"templates": SEQ_A},
    )

    assert second.duplicates == (
        OpenDuplicate(first.request.request_id, steps=("templates",)),
    )


def test_a_request_saved_without_its_runs_is_matched_by_recipe_and_scope(
    tracked: Dataset,
) -> None:
    first = submit_request(tracked, speed_recipe())
    computes_path(tracked.base_dir, first.request.request_id).unlink()

    second = submit_request(tracked, speed_recipe())
    other = submit_request(tracked, named_recipe(speed_recipe(), "another name"))

    assert load_computes(tracked.base_dir, first.request.request_id) is None
    assert second.duplicates == (OpenDuplicate(first.request.request_id, steps=()),)
    assert first.request.request_id not in {d.request_id for d in other.duplicates}


def test_a_bound_step_computes_nothing_and_the_step_below_reads_its_run(
    tracked: Dataset,
) -> None:
    """A bound step is the pinned run, and the step below computes from it."""
    older = run_speed(tracked, 3)
    bind = {"speed": BoundRef(feature=SPEED_STORAGE, run_id=older)}

    bound = submit_request(tracked, _chain(), bind=bind)

    computes = load_computes(tracked.base_dir, bound.request.request_id) or []
    assert [run.step for run in computes] == ["templates"]
    assert computes[0].run_id == bound.plan.step("templates").run_id
    assert bound.plan.step("templates").spec.inputs == (
        {"feature": SPEED_STORAGE, "run_id": older},
    )


def test_a_request_file_keeps_the_keys_an_older_mosaic_reads(tracked: Dataset) -> None:
    """The computed runs are a file of their own, so the request file is unchanged."""
    submitted = submit_request(tracked, speed_recipe())

    document = json.loads(submitted.request_path.read_text())

    assert set(document) == {
        "request_id",
        "recipe_digest",
        "owner",
        "created_at",
        "scope",
        "bind",
        "allow_partial",
        "max_concurrent_steps",
        "step_executions",
        "step_versions",
    }
    assert load_computes(tracked.base_dir, submitted.request.request_id) is not None


def test_per_entry_runs_over_disjoint_step_scopes_are_not_reported(
    tracked: Dataset,
) -> None:
    """One speed run, asked for ``seq_a`` by one request and ``seq_b`` by the other."""
    _ = submit_request(tracked, speed_recipe(), step_scopes={"speed": SEQ_A})

    second = submit_request(tracked, speed_recipe(), step_scopes={"speed": SEQ_B})

    assert second.duplicates == ()


def test_an_op_taking_no_scope_is_reported_under_disjoint_request_scopes(
    tmp_path: Path,
) -> None:
    """A conversion writes every entry its files hold, whatever the scope."""
    raw = calms21_dataset(
        tmp_path / "raw", {"train": {"a": {"s1": 30}}, "test": {"a": {"s2": 30}}}
    )
    converting = Recipe.model_validate(
        {"schema_version": 1, "name": "convert", "steps": [CONVERT_TRACKS_STEP]}
    )
    first = submit_request(raw, converting, scope=Scope(groups=["train"]))

    second = submit_request(raw, converting, scope=Scope(groups=["test"]))

    assert second.duplicates == (
        OpenDuplicate(first.request.request_id, steps=("tracks",)),
    )


def test_a_duplicate_names_three_steps_and_counts_the_rest() -> None:
    duplicate = OpenDuplicate("r", steps=("a", "b", "c", "d", "e"))

    assert describe_duplicates([duplicate]) == "r (steps 'a', 'b', 'c' and 2 more)"
    assert duplicate.as_json() == {
        "request_id": "r",
        "steps": ["a", "b", "c"],
        "steps_omitted": 2,
    }
