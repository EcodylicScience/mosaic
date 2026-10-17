"""A submission beside an open request of its recipe over an overlapping scope.

Two such requests run the same steps over the same entries. A feature run takes no
claim, so both compute it. A submission reports the open requests, and refuses as a
conflict when asked to.
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
    DeclarationCatalog,
    DuplicateRequest,
    Plan,
    Recipe,
    execute_step,
    open_requests_matching,
    recipe_digest,
    recipe_path,
    request_path,
    requests_root,
    save_request,
    submit_request,
)
from mosaic.core.pipeline.graph import request as request_module
from mosaic.core.pipeline.graph.digest import document_digest
from mosaic.core.pipeline.inventory.cache import InventoryCache
from mosaic.core.scope import Scope
from mosaic.runlog import JsonlRunLog, run_log_path
from tests.helpers import (
    SPEED_RECIPE,
    add_tracks_variant,
    invoke_json,
    make_dataset,
    speed_recipe,
    write_driver_lease,
)

VARIANT = "convert-trex.0.2-1111111111"
SEQ_A = Scope(entries=[("", "seq_a")])
SEQ_B = Scope(entries=[("", "seq_b")])

runner = CliRunner()


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two schema-valid track tables before any feature run."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return dataset


def _request_count(dataset: Dataset) -> int:
    return len(list(requests_root(dataset.base_dir).glob("*.json")))


def test_an_open_request_over_the_same_scope_is_reported(tracked: Dataset) -> None:
    first = submit_request(tracked, speed_recipe())

    second = submit_request(tracked, speed_recipe())

    assert second.duplicates == (first.request.request_id,)


def test_open_requests_are_listed_newest_first(tracked: Dataset) -> None:
    first = submit_request(tracked, speed_recipe())
    second = submit_request(tracked, speed_recipe())

    found = open_requests_matching(
        tracked.base_dir, recipe_digest(speed_recipe()), Scope()
    )

    assert found == (second.request.request_id, first.request.request_id)


def test_a_request_over_an_overlapping_scope_is_reported(tracked: Dataset) -> None:
    first = submit_request(tracked, speed_recipe(), scope=SEQ_A)

    second = submit_request(tracked, speed_recipe())

    assert second.duplicates == (first.request.request_id,)


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


def test_a_request_of_another_recipe_is_not_reported(tracked: Dataset) -> None:
    _ = submit_request(tracked, speed_recipe())

    found = open_requests_matching(tracked.base_dir, "0" * 64, Scope())

    assert found == ()


def test_a_request_whose_driver_lease_is_stale_is_not_open(tracked: Dataset) -> None:
    orphaned = submit_request(tracked, speed_recipe()).request
    _ = write_driver_lease(tracked.base_dir, orphaned.request_id, stale=True)

    found = open_requests_matching(
        tracked.base_dir, recipe_digest(speed_recipe()), Scope()
    )

    assert found == ()


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

    found = open_requests_matching(
        tracked.base_dir, recipe_digest(speed_recipe()), Scope()
    )

    assert found == (driven.request_id,)


def test_a_request_with_a_live_driver_lease_is_open(tracked: Dataset) -> None:
    driven = submit_request(tracked, speed_recipe()).request
    _ = write_driver_lease(tracked.base_dir, driven.request_id, stale=False)

    found = open_requests_matching(
        tracked.base_dir, recipe_digest(speed_recipe()), Scope()
    )

    assert found == (driven.request_id,)


def test_a_dataset_without_requests_has_none_open(tracked: Dataset) -> None:
    found = open_requests_matching(
        tracked.base_dir, recipe_digest(speed_recipe()), Scope()
    )

    assert found == ()


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
        "request_ids": [first.request.request_id],
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
        scope: Scope | None = None,
        inventory: InventoryCache | None = None,
        catalog: DeclarationCatalog | None = None,
    ) -> Plan:
        monkeypatch.setattr(request_module, "plan_pipeline", plan_pipeline)
        written.append(submit_request(dataset, recipe).request.request_id)
        return plan_pipeline(
            dataset, recipe, scope=scope, inventory=inventory, catalog=catalog
        )

    monkeypatch.setattr(request_module, "plan_pipeline", plan_while_another_submits)

    with pytest.raises(DuplicateRequest) as raised:
        _ = submit_request(tracked, speed_recipe(), refuse_duplicates=True)

    assert raised.value.request_ids == tuple(written)
    assert _request_count(tracked) == 1


def test_an_unreadable_request_file_is_skipped(tracked: Dataset) -> None:
    first = submit_request(tracked, speed_recipe())
    broken = request_path(tracked.base_dir, "0" * 26)
    _ = broken.write_text("{not json")

    with pytest.warns(UserWarning, match="skipped request file"):
        second = submit_request(tracked, speed_recipe())

    assert second.duplicates == (first.request.request_id,)


def test_an_invalid_request_of_the_recipe_is_skipped(tracked: Dataset) -> None:
    first = submit_request(tracked, speed_recipe())
    invalid = {"recipe_digest": first.request.recipe_digest, "unknown": True}
    _ = request_path(tracked.base_dir, "0" * 26).write_text(json.dumps(invalid))

    with pytest.warns(UserWarning, match="skipped request file"):
        second = submit_request(tracked, speed_recipe())

    assert second.duplicates == (first.request.request_id,)


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
    assert second.duplicates == (first.request.request_id,)


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
    assert json.loads(result.stdout)["duplicates"] == [first["request_id"]]
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
    """A request an earlier canonical form named is still a duplicate."""
    submitted = submit_request(tracked, Recipe.model_validate(_speed_with(1)))
    legacy = _speed_with(1.0)
    name = document_digest(cast("JsonValue", legacy))
    _ = recipe_path(tracked.base_dir, name).write_text(json.dumps(legacy))
    older = submitted.request.model_copy(
        update={"request_id": "00-legacy", "recipe_digest": name}
    )
    _ = save_request(tracked.base_dir, older)

    found = open_requests_matching(
        tracked.base_dir, submitted.request.recipe_digest, Scope()
    )

    assert name != submitted.request.recipe_digest
    assert found == (submitted.request.request_id, "00-legacy")


def test_a_request_whose_recipe_file_is_missing_is_not_a_duplicate(
    tracked: Dataset,
) -> None:
    submitted = submit_request(tracked, speed_recipe())
    orphan = submitted.request.model_copy(
        update={"request_id": "00-missing", "recipe_digest": "0" * 16}
    )
    _ = save_request(tracked.base_dir, orphan)

    found = open_requests_matching(
        tracked.base_dir, submitted.request.recipe_digest, Scope()
    )

    assert found == (submitted.request.request_id,)
