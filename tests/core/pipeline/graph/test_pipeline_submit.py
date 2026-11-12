"""Recording a submission, and the command each of its steps is run by.

Two documents land in the dataset and every step gets an attempt id before
anything runs, which is what makes the request complete at submit rather than
filled in as work lands. The argv is checked against the CLI that has to accept
it, because a command nothing can run is the one failure a shape assertion would
not catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    BoundRef,
    CleanupNeedsPlan,
    CleanupOnManagedDataset,
    CleanupPolicy,
    CleanupSpec,
    CleanupTarget,
    Plan,
    PlanChanged,
    Recipe,
    Request,
    RequestIdMalformed,
    RetireNeedsCleanup,
    RetireUnknownRecipe,
    ReviewedPlan,
    declared_version,
    load_request,
    plan_fingerprint,
    plan_pipeline,
    plan_with_cleanup,
    recipe_digest,
    recipe_path,
    request_path,
    requests_root,
    run_pipeline,
    save_request,
    step_argv,
    submit_request,
)
from mosaic.core.pipeline.inventory.cache import InventoryCache
from mosaic.core.pipeline.parallelism import ParallelSettings
from mosaic.core.scope import Scope
from tests.helpers import (
    add_tracks_variant,
    make_dataset,
    run_speed,
    speed_heading_recipe,
    speed_recipe_stepping,
)

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


def test_submitting_writes_the_recipe_and_the_request(tracked: Dataset) -> None:
    """The dataset records which pipeline was applied to it, and by whom."""
    recipe = Recipe.model_validate(CHAIN)

    submitted = submit_request(tracked, recipe, owner="jacob")

    assert submitted.recipe_path == recipe_path(tracked.base_dir, recipe_digest(recipe))
    assert submitted.request_path == request_path(
        tracked.base_dir, submitted.request.request_id
    )
    assert submitted.recipe_path.exists()
    read_back = load_request(tracked.base_dir, submitted.request.request_id)
    assert read_back == submitted.request
    assert read_back.owner == "jacob"


def test_an_external_id_that_is_not_a_ulid_writes_nothing(tracked: Dataset) -> None:
    with pytest.raises(RequestIdMalformed):
        _ = submit_request(tracked, Recipe.model_validate(CHAIN), request_id="../x")

    assert not (tracked.base_dir / ".mosaic" / "pipelines").exists()


def test_a_submission_revalidates_a_held_inventory(tracked: Dataset) -> None:
    held = InventoryCache(tracked)
    _ = held.get()
    _ = run_speed(tracked, 1)

    submitted = submit_request(tracked, speed_recipe_stepping(1), inventory=held)

    assert submitted.plan.step("speed").status == "complete"


def test_a_submission_reads_a_held_inventory_as_held_without_revalidation(
    tracked: Dataset,
) -> None:
    held = InventoryCache(tracked)
    _ = held.get()
    _ = run_speed(tracked, 1)

    submitted = submit_request(
        tracked, speed_recipe_stepping(1), inventory=held, revalidate=False
    )

    assert submitted.plan.step("speed").status == "absent"


def test_every_step_is_assigned_its_own_attempt(tracked: Dataset) -> None:
    """Assigned before anything runs, and no two steps share one.

    Sharing would make two steps' outputs indistinguishable in the run-log, and
    the run-log is what a child reads to pin its parent's identity.
    """
    submitted = submit_request(tracked, Recipe.model_validate(CHAIN))

    assigned = submitted.request.step_executions
    assert sorted(assigned) == ["speed", "templates"]
    assert len(set(assigned.values())) == 2


def test_the_versions_every_step_resolved_against_are_pinned(
    tracked: Dataset,
) -> None:
    """Only this map says what *ran*; the digest says only what the recipe is."""
    submitted = submit_request(tracked, Recipe.model_validate(CHAIN))

    assert submitted.request.step_versions == {
        "speed": declared_version("feature", "speed-angvel"),
        "templates": declared_version("feature", "extract-templates"),
    }
    assert all(submitted.request.step_versions.values())


def test_an_op_step_pins_its_version_too(tracked: Dataset) -> None:
    """The map covers every step, not only the feature ones.

    An op's version is a visible segment of its run identifier rather than a hash
    term, so a bump makes the step read as absent rather than as complete -- the
    same request spanning two identity regimes either way.
    """
    with_op: Document = {
        "schema_version": 1,
        "steps": [
            {
                "id": "points",
                "type": "op",
                "kind": "convert-points",
                "params": {
                    "cvat_xml": "annotations.xml",
                    "images_dir": "images",
                    "class_names": ["bee"],
                    "radii": {"bee": 8.0},
                },
            }
        ],
    }

    submitted = submit_request(tracked, Recipe.model_validate(with_op))

    assert submitted.request.step_versions["points"] == declared_version(
        "op", "convert-points"
    )
    # Its sources are not on disk, so its identity is honestly unresolvable --
    # which is a reported state, not a reason a submission cannot be recorded.
    assert submitted.plan.step("points").run_id is None


def test_the_argv_names_the_request_and_never_a_second_path(
    tracked: Dataset,
) -> None:
    """The request is found from the manifest's parent.

    A path flag of its own would not be covered by the translation a queue
    applies when the executing side mounts the dataset somewhere else, and it
    would break precisely on the substrate that does.
    """
    submitted = submit_request(tracked, Recipe.model_validate(CHAIN))

    argv = step_argv(tracked.manifest_path, submitted.request, "speed")

    assert argv[:3] == ["mosaic", "run", "--json"]
    assert "--graph-request" in argv
    assert argv[argv.index("--graph-request") + 1] == submitted.request.request_id
    assert argv[argv.index("--step") + 1] == "speed"
    assert argv[argv.index("--execution-id") + 1] == submitted.request.execution_of(
        "speed"
    )
    paths = [word for word in argv if str(tracked.base_dir) in word]
    assert paths == [str(tracked.manifest_path)]


def test_the_submitted_argv_is_what_mosaic_run_accepts(tracked: Dataset) -> None:
    """The one assertion a shape check cannot make: the command actually runs."""
    submitted = submit_request(tracked, Recipe.model_validate(CHAIN))
    argv = step_argv(tracked.manifest_path, submitted.request, "speed")

    result = runner.invoke(app, argv[1:])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["step"] == "speed"
    assert payload["request_id"] == submitted.request.request_id
    assert payload["run_id"]


def test_the_argv_of_a_step_carries_workers_only_above_one(tracked: Dataset) -> None:
    request = submit_request(tracked, Recipe.model_validate(CHAIN)).request

    assert "--workers" not in step_argv(tracked.manifest_path, request, "speed")
    assert "--workers" not in step_argv(
        tracked.manifest_path, request, "speed", parallel=ParallelSettings(workers=1)
    )
    argv = step_argv(
        tracked.manifest_path,
        request,
        "speed",
        parallel=ParallelSettings(
            workers=3, mode="process", min_free_memory_fraction=0.1
        ),
    )
    assert argv[argv.index("--workers") + 1] == "3"
    assert argv[argv.index("--parallel-mode") + 1] == "process"
    assert argv[argv.index("--min-free-memory-fraction") + 1] == "0.1"


def test_the_argv_with_workers_is_what_mosaic_run_accepts(tracked: Dataset) -> None:
    submitted = submit_request(tracked, Recipe.model_validate(CHAIN))
    argv = step_argv(
        tracked.manifest_path,
        submitted.request,
        "speed",
        parallel=ParallelSettings(workers=2, min_free_memory_fraction=0.05),
    )

    result = runner.invoke(app, argv[1:])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["step"] == "speed"


# --- the CLI ---------------------------------------------------------------------


def test_submit_prints_a_command_per_step_and_its_dependencies(
    tracked: Dataset, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            "submit",
            "--recipe",
            f"@{_recipe_file(tmp_path, CHAIN)}",
            "--manifest",
            str(tracked.manifest_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["request_id"]
    steps = {step["step"]: step for step in payload["steps"]}
    assert steps["speed"]["depends_on"] == []
    assert steps["templates"]["depends_on"] == [steps["speed"]["execution_id"]]
    assert steps["templates"]["parents"] == ["speed"]
    assert steps["speed"]["lane"]


def test_submit_runs_nothing(tracked: Dataset, tmp_path: Path) -> None:
    """It records a submission. Something else decides when the work happens."""
    result = runner.invoke(
        app,
        [
            "pipeline",
            "submit",
            "--recipe",
            f"@{_recipe_file(tmp_path, CHAIN)}",
            "--manifest",
            str(tracked.manifest_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert not list(Path(tracked.get_root("features")).glob("*/*/*.parquet"))


def test_a_shell_loop_over_the_printed_commands_runs_the_graph(
    tracked: Dataset, tmp_path: Path
) -> None:
    """The claim this phase makes: a driver needs only to start processes in order."""
    submitted = runner.invoke(
        app,
        [
            "pipeline",
            "submit",
            "--recipe",
            f"@{_recipe_file(tmp_path, CHAIN)}",
            "--manifest",
            str(tracked.manifest_path),
            "--json",
        ],
    )
    payload = json.loads(submitted.stdout)

    for step in payload["steps"]:
        ran = runner.invoke(app, list(step["argv"])[1:])
        assert ran.exit_code == 0, ran.output

    status = runner.invoke(
        app,
        [
            "pipeline",
            "status",
            "--manifest",
            str(tracked.manifest_path),
            "--request",
            payload["request_id"],
            "--json",
        ],
    )
    assert json.loads(status.stdout)["status"] == "finished"


# --- a cleanup ----------------------------------------------------------------------

LOOSE = CleanupPolicy(keep_newest=0, min_age_hours=0)

SPEED_STORAGE = "speed-angvel__from__tracks"


def _cleanup_fingerprint(
    dataset: Dataset, recipe: Recipe, bind: dict[str, BoundRef] | None = None
) -> str:
    return plan_fingerprint(
        plan_with_cleanup(dataset, recipe, cleanup=LOOSE, bind=bind)
    )


def test_cleanup_requires_an_expected_plan(tracked: Dataset) -> None:
    with pytest.raises(CleanupNeedsPlan) as raised:
        _ = submit_request(tracked, speed_recipe_stepping(2), cleanup=LOOSE)

    assert raised.value.error_kind == "invalid"
    assert not requests_root(tracked.base_dir).exists()


def test_cleanup_is_refused_on_a_managed_dataset(tracked: Dataset) -> None:
    with tracked.manifest_path.open("a", encoding="utf-8") as manifest:
        _ = manifest.write("managed_by: mosaic-api\n")
    managed = Dataset(manifest_path=tracked.manifest_path).load(ensure_roots=False)

    with pytest.raises(CleanupOnManagedDataset) as raised:
        _ = submit_request(
            managed, speed_recipe_stepping(2), cleanup=LOOSE, expect_plan="x"
        )

    assert raised.value.error_detail() == {"managed_by": "mosaic-api"}


def test_the_request_records_the_approved_set_and_policy(tracked: Dataset) -> None:
    old = run_speed(tracked, 1)
    recipe = speed_recipe_stepping(2)
    fingerprint = _cleanup_fingerprint(tracked, recipe)

    submitted = submit_request(tracked, recipe, cleanup=LOOSE, expect_plan=fingerprint)

    spec = load_request(tracked.base_dir, submitted.request.request_id).cleanup
    assert spec is not None
    assert (spec.mode, spec.policy, spec.fingerprint) == (
        "superseded",
        LOOSE,
        fingerprint,
    )
    assert spec.approved == [CleanupTarget(storage=SPEED_STORAGE, run_id=old)]
    assert spec.retired == []
    saved = json.loads(
        request_path(tracked.base_dir, submitted.request.request_id).read_text()
    )
    assert "retired" not in saved["cleanup"]
    assert spec.execution_id not in submitted.request.step_executions.values()
    assert submitted.plan.cleanup is not None


def test_a_bound_run_is_never_approved(tracked: Dataset) -> None:
    old = run_speed(tracked, 1)
    other = run_speed(tracked, 3)
    recipe = speed_recipe_stepping(2)
    bind = {"speed": BoundRef(feature=SPEED_STORAGE, run_id=old)}

    submitted = submit_request(
        tracked,
        recipe,
        bind=bind,
        cleanup=LOOSE,
        expect_plan=_cleanup_fingerprint(tracked, recipe, bind),
    )

    spec = submitted.request.cleanup
    assert spec is not None
    assert spec.approved == [CleanupTarget(storage=SPEED_STORAGE, run_id=other)]


def test_a_fingerprint_without_cleanup_does_not_submit_a_cleanup(
    tracked: Dataset,
) -> None:
    _ = run_speed(tracked, 1)
    recipe = speed_recipe_stepping(2)
    plain = plan_fingerprint(plan_pipeline(tracked, recipe))

    with pytest.raises(PlanChanged):
        _ = submit_request(tracked, recipe, cleanup=LOOSE, expect_plan=plain)
    submitted = submit_request(tracked, recipe, expect_plan=plain)

    assert submitted.request.cleanup is None
    saved = json.loads(
        request_path(tracked.base_dir, submitted.request.request_id).read_text()
    )
    assert "cleanup" not in saved


# --- the request file an older mosaic reads --------------------------------------------

_FILE_HEAD = """{
  "request_id": "%s",
  "recipe_digest": "0123456789abcdef",
  "owner": "",
  "created_at": "2026-10-18T00:00:00Z",
  "scope": {
    "entries": null,
    "groups": null,
    "sequences": null
  },
  "bind": {},
  "allow_partial": false,
  "max_concurrent_steps": null,
  "step_executions": {
    "speed": "E1"
  },
  "step_versions": {
    "speed": "0.2"
  }"""

_CLEANUP_BODY = """,
  "cleanup": {
    "mode": "superseded",
    "policy": {
      "keep_newest": 1,
      "min_age_hours": 24.0
    },
    "execution_id": "E2",
    "approved": [
      {
        "storage": "s",
        "run_id": "0.2-aaaaaaaaaa"
      }
    ],
    "fingerprint": "f"%s
  }
}"""


CLEANING = "01JC0000000000000000000001"
PLAIN = "01JC0000000000000000000002"


def _request(*, retired: list[str] | None = None, cleanup: bool = True) -> Request:
    spec = (
        CleanupSpec(
            execution_id="E2",
            approved=[CleanupTarget(storage="s", run_id="0.2-aaaaaaaaaa")],
            fingerprint="f",
            retired=retired or [],
        )
        if cleanup
        else None
    )
    return Request(
        request_id=CLEANING if cleanup else PLAIN,
        recipe_digest="0123456789abcdef",
        created_at="2026-10-18T00:00:00Z",
        step_executions={"speed": "E1"},
        step_versions={"speed": "0.2"},
        cleanup=spec,
    )


def test_a_request_without_a_retirement_is_written_as_an_older_mosaic_wrote_it(
    tmp_path: Path,
) -> None:
    """Pinned to the bytes written before ``retired`` existed."""
    plain = save_request(tmp_path, _request(cleanup=False))
    cleaning = save_request(tmp_path, _request())

    assert plain.read_text() == (_FILE_HEAD % PLAIN) + "\n}"
    assert cleaning.read_text() == (_FILE_HEAD % CLEANING) + (_CLEANUP_BODY % "")


def test_a_retirement_is_written_and_read_back(tmp_path: Path) -> None:
    request = _request(retired=["51b2573baa1b4a84"])

    path = save_request(tmp_path, request)

    assert path.read_text() == (_FILE_HEAD % CLEANING) + (
        _CLEANUP_BODY % ',\n    "retired": [\n      "51b2573baa1b4a84"\n    ]'
    )
    loaded = load_request(tmp_path, CLEANING)
    assert loaded.cleanup is not None
    assert loaded.cleanup.retired == ["51b2573baa1b4a84"]
    assert loaded == request


def test_the_request_records_the_recipes_it_retires(
    tracked: Dataset, lapsed_request_holds: None
) -> None:
    ran = run_pipeline(tracked, speed_heading_recipe())
    recipe = speed_recipe_stepping(2)
    fingerprint = plan_fingerprint(
        plan_with_cleanup(tracked, recipe, cleanup=LOOSE, retire=[ran.recipe_digest])
    )

    submitted = submit_request(
        tracked,
        recipe,
        cleanup=LOOSE,
        expect_plan=fingerprint,
        retire=[ran.recipe_digest, ran.recipe_digest],
    )

    spec = load_request(tracked.base_dir, submitted.request.request_id).cleanup
    assert spec is not None
    assert spec.retired == [ran.recipe_digest]
    assert {target.storage for target in spec.approved} == {
        SPEED_STORAGE,
        "heading__from__tracks",
    }
    assert submitted.plan.cleanup is not None
    assert submitted.plan.cleanup.retired == (ran.recipe_digest,)


def test_a_submission_that_retires_without_a_cleanup_is_refused(
    tracked: Dataset,
) -> None:
    ran = run_pipeline(tracked, speed_heading_recipe())

    with pytest.raises(RetireNeedsCleanup):
        _ = submit_request(
            tracked, speed_recipe_stepping(2), retire=[ran.recipe_digest]
        )

    assert list(requests_root(tracked.base_dir).glob("*.json")) == [
        request_path(tracked.base_dir, ran.request_id)
    ]


def test_a_submission_that_retires_an_unknown_recipe_writes_nothing(
    tracked: Dataset,
) -> None:
    recipe = speed_recipe_stepping(2)
    fingerprint = _cleanup_fingerprint(tracked, recipe)

    with pytest.raises(RetireUnknownRecipe):
        _ = submit_request(
            tracked,
            recipe,
            cleanup=LOOSE,
            expect_plan=fingerprint,
            retire=["0123456789abcdef"],
        )

    assert not requests_root(tracked.base_dir).exists()
    assert not recipe_path(tracked.base_dir, recipe_digest(recipe)).exists()


# --- a bound submission -------------------------------------------------------

SEQ_A = Scope(entries=[("", "seq_a")])


def _clustering() -> Recipe:
    """The chain with a ``global-kmeans`` step that reads the templates."""
    document = json.loads(json.dumps(CHAIN))
    document["name"] = "speed, templates and clusters"
    document["steps"].append(
        {
            "id": "kmeans",
            "type": "feature",
            "feature": "global-kmeans",
            "inputs": [{"step": "speed"}],
            "params": {"templates": {"step": "templates"}, "k": 2},
        }
    )
    return Recipe.model_validate(document)


def _bound_to_older_speed(dataset: Dataset) -> dict[str, BoundRef]:
    """Bind the speed step to a speed run that the graph does not make."""
    older = run_speed(dataset, 3)
    return {"speed": BoundRef(feature=SPEED_STORAGE, run_id=older)}


def _bound_plan(dataset: Dataset, recipe: Recipe, bind: dict[str, BoundRef]) -> Plan:
    pinned = Request(request_id="pins", recipe_digest=recipe_digest(recipe), bind=bind)
    return plan_pipeline(dataset, recipe, scope=SEQ_A, request=pinned)


def test_a_bound_submission_returns_the_plan_of_the_bound_graph(
    tracked: Dataset,
) -> None:
    recipe = _clustering()
    bind = _bound_to_older_speed(tracked)
    bound = _bound_plan(tracked, recipe, bind)

    submitted = submit_request(tracked, recipe, scope=SEQ_A, bind=bind)

    assert submitted.plan.step("speed").status == "bound"
    assert submitted.plan.step("templates").spec.inputs == (
        {"feature": SPEED_STORAGE, "run_id": bind["speed"].run_id},
    )
    assert [step.run_id for step in submitted.plan.steps] == [
        step.run_id for step in bound.steps
    ]
    assert plan_fingerprint(submitted.plan) == plan_fingerprint(bound)


def test_a_bound_submission_accepts_the_fingerprint_of_the_bound_plan(
    tracked: Dataset,
) -> None:
    recipe = _clustering()
    bind = _bound_to_older_speed(tracked)
    reviewed = plan_fingerprint(_bound_plan(tracked, recipe, bind))

    submitted = submit_request(
        tracked, recipe, scope=SEQ_A, bind=bind, expect_plan=reviewed
    )

    assert plan_fingerprint(submitted.plan) == reviewed


# --- the tracks variant -----------------------------------------------------------


def test_a_request_without_a_variant_writes_no_tracks_key(tracked: Dataset) -> None:
    """Left out when unset, so an older mosaic reads the file."""
    submitted = submit_request(tracked, Recipe.model_validate(CHAIN))

    written = json.loads(submitted.request_path.read_text())

    assert "tracks_run_id" not in written


def test_a_request_with_a_variant_records_it(tracked: Dataset) -> None:
    submitted = submit_request(
        tracked, Recipe.model_validate(CHAIN), tracks_run_id=VARIANT
    )

    written = json.loads(submitted.request_path.read_text())

    assert written["tracks_run_id"] == VARIANT
    loaded = load_request(tracked.base_dir, submitted.request.request_id)
    assert loaded.tracks_run_id == VARIANT


def test_a_request_document_without_the_field_loads(tmp_path: Path) -> None:
    path = save_request(tmp_path, _request(cleanup=False))
    assert "tracks_run_id" not in path.read_text()

    loaded = load_request(tmp_path, PLAIN)

    assert loaded.tracks_run_id is None


@pytest.mark.parametrize("blank", ["", "  "])
def test_a_request_refuses_a_blank_variant(blank: str) -> None:
    with pytest.raises(ValueError, match="names no variant"):
        _ = Request(request_id="R", recipe_digest="0" * 16, tracks_run_id=blank)


# --- a reviewed plan ---------------------------------------------------------


def _count_plans(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the plans that a submission resolves."""
    import mosaic.core.pipeline.graph.request as request_module

    resolved: list[int] = []
    real = request_module.plan_pipeline

    def counted(*args: object, **kwargs: object) -> Plan:
        resolved.append(1)
        return real(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(request_module, "plan_pipeline", counted)
    return resolved


def test_a_reviewed_plan_is_recorded_without_planning_again(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = Recipe.model_validate(CHAIN)
    held = InventoryCache(tracked)
    plan = plan_pipeline(tracked, recipe, inventory=held)
    resolved = _count_plans(monkeypatch)

    submitted = submit_request(
        tracked,
        recipe,
        inventory=held,
        expect_plan=plan_fingerprint(plan),
        reviewed=ReviewedPlan(plan, held.generation),
    )

    assert resolved == []
    assert submitted.plan is plan


def test_a_reviewed_plan_is_resolved_again_once_an_index_moved(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Computing a run writes its index, and the plan then names the run complete."""
    recipe = speed_recipe_stepping(1)
    held = InventoryCache(tracked)
    plan = plan_pipeline(tracked, recipe, inventory=held)
    reviewed = ReviewedPlan(plan, held.generation)
    _ = run_speed(tracked, 1)
    resolved = _count_plans(monkeypatch)

    submitted = submit_request(tracked, recipe, inventory=held, reviewed=reviewed)

    assert resolved == [1]
    assert submitted.plan is not plan
    assert plan_fingerprint(submitted.plan) != plan_fingerprint(plan)


def test_a_plan_reviewed_before_an_index_moved_is_refused_by_its_fingerprint(
    tracked: Dataset,
) -> None:
    recipe = speed_recipe_stepping(1)
    held = InventoryCache(tracked)
    plan = plan_pipeline(tracked, recipe, inventory=held)
    _ = run_speed(tracked, 1)

    with pytest.raises(PlanChanged):
        _ = submit_request(
            tracked,
            recipe,
            inventory=held,
            expect_plan=plan_fingerprint(plan),
            reviewed=ReviewedPlan(plan, held.generation),
        )
