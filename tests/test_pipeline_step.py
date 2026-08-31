"""Running one step of one request, and what it refuses to run over.

The suite that de-risks deciding a whole graph at submit. Every property here is
about a step re-planning *itself*: what it pins, what it checks, and how it says
no -- and each is cheaper to get wrong here, in one process, than inside a queue.

The sharpest is the cross-binding case. Two requests running one feature with
different params on one dataset must each bind to their own upstream, and a step
that re-resolved its input by feature name would not: the fallback rule there is
wall clock, so the second step of one request would pick up the other's output
because its index row landed a second later.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import mosaic.core.pipeline.graph.step as step_module
from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    REFUSED_EXIT_CODE,
    CoverageShortfall,
    Plan,
    PlannedStep,
    Recipe,
    StepRefused,
    StepSpec,
    asked_of,
    execute_step,
    load_recipe_for_request,
    recipe_path,
    request_rollup,
    save_recipe,
    submit_request,
)
from mosaic.core.scope import Scope
from mosaic.runlog import read_run, run_log_dir
from tests.helpers import add_tracks_variant, make_dataset

runner = CliRunner()

type Document = dict[str, object]

VARIANT = "convert-trex.0.2-1111111111"


def _chain(step_size: int = 1) -> Document:
    """A two-step chain whose first step's params can be varied.

    ``step_size`` moves ``speed``'s identity without changing anything else, so
    two requests over one dataset produce two genuinely different upstreams --
    which is the only way to tell binding from coincidence.
    """
    return {
        "schema_version": 1,
        "name": "speed then templates",
        "steps": [
            {
                "id": "speed",
                "type": "feature",
                "feature": "speed-angvel",
                "inputs": ["tracks"],
                "params": {"step_size": step_size},
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


def _params_text(dataset: Dataset, run_id: str) -> str:
    """The provenance one run wrote, whichever storage it landed under."""
    found = list(Path(dataset.get_root("features")).glob(f"*/{run_id}/params.json"))
    assert found, f"no params.json for run {run_id}"
    return found[0].read_text()


# --- binding -----------------------------------------------------------------


def test_a_step_pins_its_parent_from_that_parent_s_run_log(
    tracked: Dataset,
) -> None:
    """What the run-log recorded governs, not what the plan predicted."""
    submitted = submit_request(tracked, Recipe.model_validate(_chain()))
    request = submitted.request

    speed = execute_step(tracked, request, "speed")
    templates = execute_step(tracked, request, "templates")

    logged = read_run(run_log_dir(tracked.base_dir), request.execution_of("speed"))
    assert logged is not None
    assert logged["run_id"] == speed.run_id
    assert speed.run_id in _params_text(tracked, templates.run_id)


def test_a_feature_step_records_the_tracks_variant_it_read(tracked: Dataset) -> None:
    """Which tables a step read, on the step's own record.

    The queue cannot derive this from the spec: a step-addressed argv carries no
    ``--tracks-run-id`` at all, because the step resolves its own variant out of
    the recipe. So mosaic is the only party that can put it on the wire, and it
    goes on early -- beside the identity -- so a run killed halfway still says
    what it was reading.
    """
    request = submit_request(tracked, Recipe.model_validate(_chain())).request
    _ = execute_step(tracked, request, "speed")
    _ = execute_step(tracked, request, "templates")

    speed_log = read_run(run_log_dir(tracked.base_dir), request.execution_of("speed"))
    assert speed_log is not None
    assert speed_log["tracks_variant"] == VARIANT

    # A feature reading another feature's output read no tracks, and says so
    # rather than inheriting its parent's answer.
    templates_log = read_run(
        run_log_dir(tracked.base_dir), request.execution_of("templates")
    )
    assert templates_log is not None
    assert templates_log["tracks_variant"] == ""


def test_a_cached_step_records_a_cache_hit_and_what_its_artifact_holds(
    tracked: Dataset,
) -> None:
    """A step that ran nothing is as legible as one that ran everything.

    ``entries_written`` is deliberately the artifact's coverage rather than zero:
    the field counts what the scope holds at the end of the attempt, and a cached
    step holds all of it. Zero would read as a step that lost everything, which is
    the misreading the field exists to prevent.
    """
    first = submit_request(tracked, Recipe.model_validate(_chain())).request
    ran = execute_step(tracked, first, "speed")
    assert ran.state == "ran"

    second = submit_request(tracked, Recipe.model_validate(_chain())).request
    cached = execute_step(tracked, second, "speed")
    assert cached.state == "cached"

    logged = read_run(run_log_dir(tracked.base_dir), second.execution_of("speed"))
    assert logged is not None
    assert logged["status"] == "finished"
    assert logged["cache_hit"] is True
    # The log and the outcome cannot disagree about one step: both come from the
    # same coverage expression.
    assert logged["entries_written"] == cached.covered


def test_two_requests_on_one_dataset_each_bind_to_their_own_upstream(
    tracked: Dataset,
) -> None:
    """The cross-binding regression.

    Both requests run ``speed`` with different params, so both upstreams exist
    on disk at once and only one of them is each consumer's. Resolving by feature
    name would fall through to the latest-run rule -- wall clock -- and the later
    of the two would win for both.
    """
    first = submit_request(tracked, Recipe.model_validate(_chain(step_size=1))).request
    second = submit_request(tracked, Recipe.model_validate(_chain(step_size=2))).request

    speed_a = execute_step(tracked, first, "speed")
    speed_b = execute_step(tracked, second, "speed")
    assert speed_a.run_id != speed_b.run_id, "the two upstreams must differ at all"

    # Deliberately out of submission order: the consumer of the *earlier* request
    # runs after the *later* request's producer, which is exactly the interleave
    # a wall-clock rule gets wrong.
    templates_a = execute_step(tracked, first, "templates")
    templates_b = execute_step(tracked, second, "templates")

    assert speed_a.run_id in _params_text(tracked, templates_a.run_id)
    assert speed_b.run_id not in _params_text(tracked, templates_a.run_id)
    assert speed_b.run_id in _params_text(tracked, templates_b.run_id)
    assert templates_a.run_id != templates_b.run_id


def test_a_parent_served_from_an_earlier_request_is_not_a_fault(
    tracked: Dataset,
) -> None:
    """An ancestor with no run-log is a cache hit, not a missing identity."""
    done = submit_request(tracked, Recipe.model_validate(_chain())).request
    _ = execute_step(tracked, done, "speed")

    fresh = submit_request(tracked, Recipe.model_validate(_chain())).request
    templates = execute_step(tracked, fresh, "templates")

    assert templates.state == "ran"
    assert templates.run_id


def test_a_parent_that_recorded_no_identity_stops_its_children(
    tracked: Dataset,
) -> None:
    """Nothing below a parent that never named its output can be addressed."""
    request = submit_request(tracked, Recipe.model_validate(_chain())).request
    log_path = run_log_dir(tracked.base_dir) / f"{request.execution_of('speed')}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _ = log_path.write_text(
        json.dumps({"t": "2026-01-01T00:00:00+00:00", "ev": "started"})
        + "\n"
        + json.dumps({"t": "2026-01-01T00:00:01+00:00", "ev": "finished"})
        + "\n"
    )

    with pytest.raises(StepRefused) as raised:
        _ = execute_step(tracked, request, "templates")

    assert raised.value.reason == "parent_unrecorded"


# --- the recipe a request names ----------------------------------------------


def test_a_missing_recipe_refuses_by_name(tracked: Dataset) -> None:
    request = submit_request(tracked, Recipe.model_validate(_chain())).request
    recipe_path(tracked.base_dir, request.recipe_digest).unlink()

    with pytest.raises(StepRefused) as raised:
        _ = execute_step(tracked, request, "speed")

    assert raised.value.reason == "recipe_missing"


def test_a_recipe_that_no_longer_digests_to_its_name_refuses(
    tracked: Dataset,
) -> None:
    """Edited underneath an open request, which is worse than absent.

    Every step below the edit would resolve under a graph its predecessors never
    ran, and nothing on disk would say so.
    """
    request = submit_request(tracked, Recipe.model_validate(_chain())).request
    path = recipe_path(tracked.base_dir, request.recipe_digest)
    document = json.loads(path.read_text())
    document["name"] = "something else entirely"
    _ = path.write_text(json.dumps(document))

    with pytest.raises(StepRefused) as raised:
        _ = load_recipe_for_request(tracked.base_dir, request)

    assert raised.value.reason == "digest_mismatch"


def test_a_producer_version_that_moved_refuses_rather_than_re_planning(
    tracked: Dataset,
) -> None:
    """A request completes against the versions it started with.

    Resolving early steps under the old versions and later ones under the new
    would put one submission in two identity regimes, with its later steps
    reading as absent and no fault anywhere to point at.
    """
    submitted = submit_request(tracked, Recipe.model_validate(_chain()))
    moved = submitted.request.model_copy(
        update={"step_versions": {"speed": "0.0-not-the-installed-one"}}
    )

    with pytest.raises(StepRefused) as raised:
        _ = execute_step(tracked, moved, "speed")

    assert raised.value.reason == "version_moved"
    assert raised.value.detail["moved_step"] == "speed"


# --- refusing before doing work ----------------------------------------------


def test_a_scope_dependent_step_refuses_a_short_upstream(tracked: Dataset) -> None:
    """The refusal that makes a shortfall a decision rather than a default."""
    uncomputable = Scope(entries=[("", "seq_a"), ("", "seq_b"), ("", "seq_gone")])
    request = submit_request(
        tracked, Recipe.model_validate(_chain()), scope=uncomputable
    ).request
    _ = execute_step(tracked, request, "speed")

    with pytest.raises(CoverageShortfall) as raised:
        _ = execute_step(tracked, request, "templates")

    assert raised.value.reason == "coverage_shortfall"
    assert raised.value.covered == 2
    assert raised.value.target == 3


def test_a_refusal_is_recorded_as_a_failed_attempt_carrying_its_reason(
    tracked: Dataset,
) -> None:
    """No new terminal status: the ledger reads ``failed`` with a reason beside it.

    Adding one would mean adding a member to the terminal set three repositories
    read and mosaic-api's sweeper reaps, which is why ``partial`` was kept out of
    it too.
    """
    uncomputable = Scope(entries=[("", "seq_a"), ("", "seq_b"), ("", "seq_gone")])
    request = submit_request(
        tracked, Recipe.model_validate(_chain()), scope=uncomputable
    ).request
    _ = execute_step(tracked, request, "speed")

    with pytest.raises(CoverageShortfall):
        _ = execute_step(tracked, request, "templates")

    logged = read_run(run_log_dir(tracked.base_dir), request.execution_of("templates"))
    assert logged is not None
    assert logged["status"] == "failed"
    assert json.loads(logged["error_json"])["reason"] == "coverage_shortfall"
    # A step that refused before planning claims nothing about coverage. Zero here
    # is the honest answer, not a report that its artifact is empty.
    assert logged["entries_written"] == 0
    assert logged["cache_hit"] is False


def test_a_finished_upstream_that_wrote_nothing_stops_its_consumer(
    tracked: Dataset,
) -> None:
    """Coverage counts cannot see wrongness; this is the one shape they can.

    Never-ran and ran-and-produced-nothing both read as absent, so the recorded
    finish is what tells them apart -- and chaining onto an empty directory is
    what happens when nothing does.
    """
    request = submit_request(tracked, Recipe.model_validate(_chain())).request
    speed = execute_step(tracked, request, "speed")
    run_root = next(Path(tracked.get_root("features")).glob(f"*/{speed.run_id}"))
    for parquet in run_root.glob("*.parquet"):
        parquet.unlink()

    with pytest.raises(StepRefused) as raised:
        _ = execute_step(tracked, request, "templates")

    assert raised.value.reason == "upstream_empty"


def test_a_dataset_that_became_mixed_refuses_by_name(tracked: Dataset) -> None:
    """Mixing centimetres with pixels is mixing units and landmarks.

    Submitted clean and mixed afterwards, which is the case a step can meet and a
    submission cannot: the term a feature identifier carries for its tables is
    scope-free, so a mixed dataset resolves no feature identity at all. Checking
    first is what turns that into a refusal naming the schemas rather than an
    exception out of the middle of a hash.
    """
    request = submit_request(tracked, Recipe.model_validate(_chain())).request
    add_tracks_variant(
        tracked, "convert-old.0.1-2222222222", "seq_c", std_format="trex_v1"
    )

    with pytest.raises(StepRefused) as raised:
        _ = execute_step(tracked, request, "speed")

    assert raised.value.reason == "schema_family_mismatch"


def test_a_mixed_dataset_cannot_be_submitted_either(tmp_path: Path) -> None:
    """The same refusal at the other entry point, so neither leaks a bare error."""
    dataset = make_dataset(tmp_path / "mixed")
    add_tracks_variant(dataset, VARIANT, "seq_a", std_format="mosaic_v1")
    add_tracks_variant(
        dataset, "convert-old.0.1-2222222222", "seq_b", std_format="trex_v1"
    )

    with pytest.raises(StepRefused) as raised:
        _ = submit_request(dataset, Recipe.model_validate(_chain()))

    assert raised.value.reason == "schema_family_mismatch"


# --- the CLI -----------------------------------------------------------------


def test_the_cli_exits_with_the_reserved_code_on_a_refusal(
    tracked: Dataset,
) -> None:
    """A driver tells a refusal from a crash without parsing anything."""
    uncomputable = Scope(entries=[("", "seq_a"), ("", "seq_b"), ("", "seq_gone")])
    request = submit_request(
        tracked, Recipe.model_validate(_chain()), scope=uncomputable
    ).request
    _ = execute_step(tracked, request, "speed")

    result = runner.invoke(
        app,
        [
            "run",
            "--json",
            "--manifest",
            str(tracked.manifest_path),
            "--graph-request",
            request.request_id,
            "--step",
            "templates",
            "--execution-id",
            request.execution_of("templates"),
        ],
    )

    assert result.exit_code == REFUSED_EXIT_CODE
    payload = json.loads(result.stdout)
    assert payload["status"] == "refused"
    assert payload["reason"] == "coverage_shortfall"


def test_the_cli_refuses_a_request_without_a_step(tracked: Dataset) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--manifest",
            str(tracked.manifest_path),
            "--graph-request",
            "whatever",
        ],
    )

    assert result.exit_code == 1
    assert "--graph-request and --step" in result.output


@pytest.mark.parametrize("flag", ["--entries", "--groups", "--sequences"])
def test_the_cli_refuses_a_scope_flag_on_a_step(tracked: Dataset, flag: str) -> None:
    """A step covers the entries its plan resolved, and nothing reads a flag.

    Accepted and dropped before these flags existed, for ``--entries`` alone.
    Asserted per flag, because one refusal reading only ``entries`` would let
    the other two through and still exit zero.
    """
    request = submit_request(tracked, Recipe.model_validate(_chain())).request
    result = runner.invoke(
        app,
        [
            "run",
            "--manifest",
            str(tracked.manifest_path),
            "--graph-request",
            request.request_id,
            "--step",
            "speed",
            flag,
            "seq_a",
        ],
    )

    assert result.exit_code == 1
    assert flag in result.output
    assert "A step covers the entries its plan resolved" in result.output


def test_a_graph_op_step_passes_overwrite_to_the_op(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A step run with ``--overwrite`` recomputes rather than refusing.

    The refusal this replaces stood while six ops read a params field instead
    of the argument. Asserted on what ``run_op`` received. A step that
    finishes proves nothing, because an op that discarded the flag finishes
    too.
    """
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [
                {
                    "id": "regrid",
                    "type": "op",
                    "kind": "resample-tracks",
                    "params": {"target_fps": 30.0},
                }
            ],
        }
    )
    submitted = submit_request(tracked, recipe)
    planned_run_id = submitted.plan.step("regrid").run_id or ""
    seen: list[bool] = []

    def _capture(*args: object, overwrite: bool = False, **kwargs: object) -> str:
        seen.append(overwrite)
        return planned_run_id

    monkeypatch.setattr(step_module, "run_op", _capture)
    _ = execute_step(tracked, submitted.request, "regrid", overwrite=True)

    assert seen == [True]


def test_an_op_step_is_given_the_entries_its_plan_resolved(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dispatch names the entries the plan named, rather than leaving it unset.

    The planner resolves a step's target scope to mint its identity, and an
    unset selector covers every indexed entry. A dispatch that left it unset
    would extract, track or transcode the whole dataset under an identifier
    minted for the entries the plan asked for.

    Asserted on what ``run_op`` received. A step that completes proves nothing
    here, because an op reaching more entries than it was asked for still
    finishes.
    """
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [
                {
                    "id": "regrid",
                    "type": "op",
                    "kind": "resample-tracks",
                    "params": {"target_fps": 30.0},
                }
            ],
        }
    )
    submitted = submit_request(tracked, recipe, scope=Scope(entries=[("", "seq_a")]))
    planned_run_id = submitted.plan.step("regrid").run_id or ""
    seen: list[Scope | None] = []

    def _capture(*args: object, scope: Scope | None = None, **kwargs: object) -> str:
        seen.append(scope)
        return planned_run_id

    monkeypatch.setattr(step_module, "run_op", _capture)
    _ = execute_step(tracked, submitted.request, "regrid")

    assert seen == [Scope(entries=[("", "seq_a")])]


def test_a_scope_free_op_step_is_given_no_scope_inside_a_narrowed_graph(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A training step reads a directory, and a narrowed graph must not scope it.

    Every op step used to be handed the plan's entries. An op declaring
    ``scope_takes = "none"`` refuses those. A graph narrowed to any entry
    therefore could not run a training step at all, failing with a scope
    refusal before the trainer was reached.

    Asserted on what ``run_op`` received, because the refusal it caused is
    raised inside ``run_op`` and a test that only checked for an exception
    would pass on any other failure.
    """
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [
                {
                    "id": "train",
                    "type": "op",
                    "kind": "train-pose",
                    "params": {"data": "datasets/pose/data.yaml", "epochs": 1},
                }
            ],
        }
    )
    submitted = submit_request(tracked, recipe, scope=Scope(entries=[("", "seq_a")]))
    seen: list[Scope | None] = []

    def _capture(*args: object, scope: Scope | None = None, **kwargs: object) -> str:
        seen.append(scope)
        return "train-pose.0.2-0000000000"

    monkeypatch.setattr(step_module, "run_op", _capture)
    _ = execute_step(tracked, submitted.request, "train")

    assert seen == [Scope()], "a scope-free op is asked to cover nothing"


# --- the request as a whole ---------------------------------------------------


def test_a_request_is_running_until_every_step_is_terminal(
    tracked: Dataset,
) -> None:
    request = submit_request(tracked, Recipe.model_validate(_chain())).request
    assert request_rollup(tracked.base_dir, request).status == "running"

    _ = execute_step(tracked, request, "speed")
    assert request_rollup(tracked.base_dir, request).status == "running"

    _ = execute_step(tracked, request, "templates")
    rollup = request_rollup(tracked.base_dir, request)
    assert rollup.status == "finished"
    assert rollup.is_terminal


def test_a_refused_step_closes_its_request_as_failed(tracked: Dataset) -> None:
    """The steps below a refusal are never dispatched, so waiting for them to
    start is waiting forever."""
    uncomputable = Scope(entries=[("", "seq_a"), ("", "seq_b"), ("", "seq_gone")])
    request = submit_request(
        tracked, Recipe.model_validate(_chain()), scope=uncomputable
    ).request
    _ = execute_step(tracked, request, "speed")
    with pytest.raises(CoverageShortfall):
        _ = execute_step(tracked, request, "templates")

    assert request_rollup(tracked.base_dir, request).status == "failed"


def test_two_dispatches_of_one_training_run_do_not_both_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one place duplicate execution is corruption rather than waste.

    Two trainers in one run root interleave nondeterministic checkpoints, and the
    reuse gate cannot stop it: it asks whether the run is complete, and neither
    dispatch is. The claim that does stop it already exists, so what this pins is
    that running a training step through a graph reaches it rather than
    introducing a second one beside it.
    """
    from mosaic.core.pipeline.markers import new_inflight, write_inflight
    from mosaic.core.pipeline.models import model_run_root
    from mosaic.tracking.ops._common import RunRootHeld
    from tests.test_training_reuse import _Counter, _data_yaml
    from tests.test_tracking_ops import _make_dataset

    dataset = _make_dataset(tmp_path)
    trainer = _Counter()
    trainer.install(monkeypatch)
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [
                {
                    "id": "train",
                    "type": "op",
                    "kind": "train-pose",
                    "params": {
                        "data": str(_data_yaml(tmp_path)),
                        "epochs": 2,
                        "device": "cpu",
                    },
                }
            ],
        }
    )
    submitted = submit_request(dataset, recipe)
    planned = submitted.plan.step("train")
    assert planned.model_run_id, "a training step names its model before it runs"

    # A peer holds the run root: a live claim whose execution has no terminal
    # run-log, which is exactly what a second concurrent dispatch looks like.
    write_inflight(
        model_run_root(dataset, "train-pose", planned.model_run_id),
        new_inflight(
            execution_id="SOMEONE-ELSE",
            host="otherhost",
            pid=4242,
            phase=None,
            idle_seconds=3600.0,
        ),
    )

    with pytest.raises(RunRootHeld, match="SOMEONE-ELSE"):
        _ = execute_step(dataset, submitted.request, "train")

    assert trainer.calls == 0


def test_a_recipe_is_copied_in_so_the_dataset_records_what_ran(
    tracked: Dataset, tmp_path: Path
) -> None:
    """A dataset can be handed to someone else with its pipelines intact."""
    recipe = Recipe.model_validate(_chain())
    where = save_recipe(tracked.base_dir, recipe)

    assert where.exists()
    assert where.parent == tracked.base_dir / ".mosaic" / "pipelines"
    _ = tmp_path


# --- what a step is asked for ---------------------------------------------------


def _planned(entries: Scope) -> PlannedStep:
    """A feature step whose spec narrows to *entries* and nothing else."""
    return PlannedStep(
        step_id="speed",
        kind="feature",
        runs="speed-angvel",
        spec=StepSpec(step_id="speed", kind="feature", entries=entries),
    )


def test_an_unset_selector_asks_for_the_whole_plan_scope() -> None:
    """Decided by is_unset, because every Scope instance is truthy.

    A truthiness test on the selector never reaches the fallback, and the step
    is asked for nothing where it should be asked for everything.
    """
    plan = Plan(recipe_digest="d", scope=frozenset({("", "seq_a"), ("", "seq_b")}))

    assert asked_of(_planned(Scope()), plan) == (("", "seq_a"), ("", "seq_b"))


def test_a_named_selector_asks_for_what_it_names() -> None:
    plan = Plan(recipe_digest="d", scope=frozenset({("", "seq_a"), ("", "seq_b")}))
    narrowed = _planned(Scope(entries=[("", "seq_b")]))

    assert asked_of(narrowed, plan) == (("", "seq_b"),)


def test_a_selector_only_an_index_can_enumerate_is_refused() -> None:
    """Neither answer fits a selector naming groups or sequences.

    The unset answer asks for every entry in the plan and the named one asks for
    none. ``plan_pipeline`` enumerates such a selector against the tracks
    universe before a step is planned, and a step spec that skipped it is a
    fault the caller has to hear about.
    """
    plan = Plan(recipe_digest="d", scope=frozenset({("", "seq_a"), ("", "seq_b")}))

    with pytest.raises(ValueError, match="asked_of returns the entries") as raised:
        _ = asked_of(_planned(Scope(groups=["A"])), plan)

    message = str(raised.value)
    assert "'speed'" in message
    assert "names groups or sequences" in message
    assert "Scope(entries=[...])" in message


def test_a_sequence_selector_is_refused_the_same_way() -> None:
    plan = Plan(recipe_digest="d", scope=frozenset({("", "seq_a")}))

    with pytest.raises(ValueError, match="names groups or sequences"):
        _ = asked_of(_planned(Scope(sequences=["seq_a"])), plan)


def test_a_scope_free_op_step_is_asked_for_no_entry() -> None:
    """A training step computes no entry, and its unset selector says so.

    The unset-means-everything rule is a feature rule. An op declaring
    ``scope_takes = "none"`` reads a prepared directory, and ``asked_of``'s
    callers query the failure store by entry -- so answering with the plan
    scope would quarantine the step over entries it never touched.
    """
    plan = Plan(recipe_digest="d", scope=frozenset({("", "seq_a"), ("", "seq_b")}))
    training = PlannedStep(
        step_id="train",
        kind="op",
        runs="train-pose",
        spec=StepSpec(
            step_id="train", kind="op", op_kind="train-pose", entries=Scope()
        ),
    )

    assert asked_of(training, plan) == ()


def test_a_scoped_op_step_is_still_asked_for_the_whole_plan_scope() -> None:
    """The narrowing above is by declaration, not by a step being an op."""
    plan = Plan(recipe_digest="d", scope=frozenset({("", "seq_a"), ("", "seq_b")}))
    resample = PlannedStep(
        step_id="resample",
        kind="op",
        runs="resample-tracks",
        spec=StepSpec(
            step_id="resample",
            kind="op",
            op_kind="resample-tracks",
            entries=Scope(),
        ),
    )

    assert asked_of(resample, plan) == (("", "seq_a"), ("", "seq_b"))
