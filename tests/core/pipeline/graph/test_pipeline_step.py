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
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

import mosaic.core.pipeline.graph.step as step_module
import mosaic.core.pipeline.parallelism as parallelism
from mosaic.behavior.feature_library.kpms import KPMS_LICENSE_ENV, KPMS_PYTHON_ENV
from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    BoundRef,
    CoverageShortfall,
    Plan,
    PlannedStep,
    Recipe,
    RecipeInvalid,
    Request,
    StepAttempt,
    StepRefused,
    StepSpec,
    asked_of,
    declaration_catalog,
    execute_step,
    plan_pipeline,
    load_recipe_for_request,
    recipe_path,
    record_step_ended,
    request_rollup,
    request_verdict,
    request_verdict_of,
    save_recipe,
    storage_name_of,
    submit_request,
)
from mosaic.core.entry import to_safe_name
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.storage_names import NAME_MAX_BYTES, SHORTENED_MARKER
from mosaic.core.pipeline.graph.claims import FileFailureStore
from mosaic.core.pipeline.graph.digest import document_digest, recipe_digest
from mosaic.core.pipeline.job import Cancelled
from mosaic.core.pipeline.parallelism import ParallelSettings
from mosaic.core.pipeline.ops import registered_op
from mosaic.core.pipeline.refusal import REFUSED_EXIT_CODE, ToolUnavailable
from mosaic.core.pipeline.tracking_roots import ToolCodecError
from mosaic.core.scope import Scope
from mosaic.core.pipeline.gpu_lease import GPU_CARD_VAR
from mosaic.runlog import (
    TERMINAL_STATUSES,
    JsonlRunLog,
    read_run,
    run_log_dir,
    run_log_path,
)
from tests.helpers import (
    FakeTrainer,
    add_tracks_variant,
    chain_with_sibling_recipe,
    hide_package,
    install_lightning_action_stub,
    kpms_params,
    lightning_action_params,
    make_dataset,
    run_speed,
    write_data_yaml,
    write_driver_lease,
    recipe_saved_by_0_17,
)

runner = CliRunner()

SPEED_STORAGE = "speed-angvel__from__tracks"

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


@pytest.mark.parametrize(
    ("raised", "attempts"),
    [(Cancelled(), 0), (RuntimeError("the feature failed"), 1)],
    ids=["cancelled", "failed"],
)
def test_a_cancelled_feature_step_is_not_counted_as_a_step_failure(
    tracked: Dataset,
    monkeypatch: pytest.MonkeyPatch,
    raised: Exception,
    attempts: int,
) -> None:
    """Backoff and quarantine read the count, and a cancel is not a failure."""
    recipe = Recipe.model_validate(_chain())
    request = submit_request(tracked, recipe).request
    planned = plan_pipeline(tracked, recipe).step("speed")

    def run_feature(*_args: object, **_kwargs: object) -> None:
        raise raised

    monkeypatch.setattr(step_module, "run_feature", run_feature)

    with pytest.raises(type(raised)):
        _ = execute_step(tracked, request, "speed")

    key = (planned.storage_name, planned.run_id or "")
    assert FileFailureStore(tracked.base_dir).step_record(key).attempts == attempts


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
    assert "saved" in raised.value.detail


def test_a_recipe_file_is_accepted_when_its_bytes_are_what_its_name_digests(
    tracked: Dataset,
) -> None:
    """A file an earlier canonical form wrote still loads for its request.

    The document is saved without the defaults pydantic fills, so the recipe it
    holds digests differently from its name, while the bytes match the name.
    """
    request = submit_request(tracked, Recipe.model_validate(_chain())).request
    raw = cast("dict[str, JsonValue]", _chain())
    name = document_digest(raw)
    _ = recipe_path(tracked.base_dir, name).write_text(json.dumps(raw))
    legacy = request.model_copy(update={"recipe_digest": name})

    recipe = load_recipe_for_request(tracked.base_dir, legacy)

    assert recipe == Recipe.model_validate(raw)
    assert recipe_digest(recipe) != name


def _legacy_request(dataset: Dataset, document: dict[str, JsonValue]) -> Request:
    """A request naming *document*, saved as an earlier mosaic wrote it.

    The file holds the document as written and is named by its own digest, which
    is what 0.17's ``save_recipe`` did for whole-number floats in params.
    """
    request = submit_request(dataset, Recipe.model_validate(_chain())).request
    name = document_digest(document)
    _ = recipe_path(dataset.base_dir, name).write_text(json.dumps(document))
    return request.model_copy(update={"recipe_digest": name})


def test_a_request_saved_before_numbers_were_normalized_still_loads(
    tracked: Dataset,
) -> None:
    legacy = recipe_saved_by_0_17()
    request = _legacy_request(tracked, legacy)
    assert request.recipe_digest == "a382f3f695561521"

    recipe = load_recipe_for_request(tracked.base_dir, request)

    assert recipe_digest(recipe) == "51f069d7b95bc6a4"


def test_a_legacy_recipe_edited_underneath_its_request_still_refuses(
    tracked: Dataset,
) -> None:
    request = _legacy_request(tracked, recipe_saved_by_0_17())
    path = recipe_path(tracked.base_dir, request.recipe_digest)
    document = json.loads(path.read_text())
    document["name"] = "something else entirely"
    _ = path.write_text(json.dumps(document))

    with pytest.raises(StepRefused) as raised:
        _ = load_recipe_for_request(tracked.base_dir, request)

    assert raised.value.reason == "digest_mismatch"
    assert "saved" in raised.value.detail


def test_a_legacy_request_runs_its_step(tracked: Dataset) -> None:
    """A whole-number float written by an earlier mosaic runs as its integer."""
    document = cast("dict[str, JsonValue]", _chain())
    steps = cast("list[dict[str, JsonValue]]", document["steps"])
    steps[0]["params"] = {"step_size": 1.0}
    request = _legacy_request(tracked, document)

    outcome = execute_step(tracked, request, "speed")

    assert outcome.run_id
    logged = read_run(run_log_dir(tracked.base_dir), request.execution_of("speed"))
    assert logged is not None and logged["status"] == "finished"


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


def test_a_step_attempt_reads_the_card_its_run_log_records(tracked: Dataset) -> None:
    request = submit_request(tracked, Recipe.model_validate(_chain())).request
    with JsonlRunLog(
        run_log_path(tracked.base_dir, request.execution_of("speed")),
        request.execution_of("speed"),
    ) as log:
        log.started(kind="feature", target="speed", gpu_card="1:GPU-abc")
    with JsonlRunLog(
        run_log_path(tracked.base_dir, request.execution_of("templates")),
        request.execution_of("templates"),
    ) as log:
        log.started(kind="feature", target="templates")

    rollup = request_rollup(tracked.base_dir, request)

    assert [(step.step_id, step.gpu_card) for step in rollup.steps] == [
        ("speed", "1:GPU-abc"),
        ("templates", ""),
    ]


def test_an_unstarted_step_attempt_reads_no_card(tracked: Dataset) -> None:
    request = submit_request(tracked, Recipe.model_validate(_chain())).request

    rollup = request_rollup(tracked.base_dir, request)

    assert [step.gpu_card for step in rollup.steps] == ["", ""]


def test_a_step_that_runs_nothing_records_the_card_of_its_process(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = submit_request(tracked, Recipe.model_validate(_chain())).request
    _ = execute_step(tracked, first, "speed")
    second = submit_request(tracked, Recipe.model_validate(_chain())).request
    monkeypatch.setenv(GPU_CARD_VAR, "0:GPU-abc")

    cached = execute_step(tracked, second, "speed")

    assert cached.state == "cached"
    (speed, _templates) = request_rollup(tracked.base_dir, second).steps
    assert speed.gpu_card == "0:GPU-abc"


def test_a_step_a_driver_records_ended_names_no_card(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = submit_request(tracked, Recipe.model_validate(_chain())).request
    monkeypatch.setenv(GPU_CARD_VAR, "0:GPU-abc")

    record_step_ended(
        tracked,
        request.execution_of("speed"),
        step_id="speed",
        owner="",
        status="cancelled",
    )

    (speed, _templates) = request_rollup(tracked.base_dir, request).steps
    assert (speed.status, speed.gpu_card) == ("cancelled", "")


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


# --- two tracks variants of one entry -----------------------------------------


OTHER_VARIANT = "convert-other.0.1-2222222222"
VARIANT_B = "convert-old.0.1-3333333333"


def _sweep(*speeds: float) -> Recipe:
    """A tracker sweep: one ``trex`` op per setting, reading no tracks table."""
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [
                {
                    "id": f"trex_{speed}",
                    "type": "op",
                    "kind": "trex",
                    "params": {
                        "track_max_individuals": 2,
                        "track_max_speed": speed,
                    },
                }
                for speed in speeds
            ],
        }
    )


def _tracked_then_read() -> Document:
    """A tracker, a feature pinned to what it writes, and a reader of that feature."""
    return {
        "schema_version": 1,
        "steps": [
            {
                "id": "track",
                "type": "op",
                "kind": "trex",
                "params": {"track_max_individuals": 2},
            },
            {
                "id": "speed",
                "type": "feature",
                "feature": "speed-angvel",
                "inputs": ["tracks"],
                "tracks": {"step": "track"},
                "params": {"step_size": 1},
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


def test_two_variants_of_one_family_do_not_refuse_a_sweep(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tracker sweep is exactly the dataset that holds two variants of an entry.

    The schema check asks about families, and a choice between variants is no
    business of a step that reads no tracks. Asserted on ``run_op`` being reached,
    because the refusal this replaces was raised before it.
    """
    add_tracks_variant(tracked, OTHER_VARIANT, "seq_a", std_format="mosaic_v1")
    submitted = submit_request(tracked, _sweep(10, 20))
    ran: list[str] = []

    def _capture(*args: object, **kwargs: object) -> str:
        ran.append(submitted.plan.step("trex_10").run_id or "")
        return ran[-1]

    monkeypatch.setattr(step_module, "run_op", _capture)
    _ = execute_step(tracked, submitted.request, "trex_10")

    assert len(ran) == 1


def test_two_families_on_one_entry_refuse_by_name(tracked: Dataset) -> None:
    """A choice between variants does not hide a mixture: each family is named."""
    add_tracks_variant(tracked, OTHER_VARIANT, "seq_a", std_format="trex_v1")

    with pytest.raises(StepRefused) as raised:
        _ = submit_request(tracked, _sweep(10))

    assert raised.value.reason == "schema_family_mismatch"
    assert "mosaic_v1" in str(raised.value)
    assert "trex_v1" in str(raised.value)


def test_an_unlabelled_row_is_superseded_before_families_are_compared(
    tmp_path: Path,
) -> None:
    """The rule that picks an entry's rows is the resolver's, so the two agree.

    Beside a labelled row of its entry an unlabelled one is a record and not a
    table anything reads, so its family does not count against the dataset.
    """
    dataset = make_dataset(tmp_path / "superseded")
    add_tracks_variant(dataset, "", "seq_a", std_format="trex_v1")
    add_tracks_variant(dataset, VARIANT, "seq_a", std_format="mosaic_v1")
    add_tracks_variant(dataset, OTHER_VARIANT, "seq_a", std_format="mosaic_v1")

    assert submit_request(dataset, _sweep(10)).request.request_id


def test_a_row_without_its_table_does_not_count_as_a_family(tracked: Dataset) -> None:
    """A table that is gone cannot be read, so it cannot be read beside another."""
    add_tracks_variant(tracked, OTHER_VARIANT, "seq_a", std_format="trex_v1")
    gone = Path(tracked.get_root("tracks")) / OTHER_VARIANT
    for table in gone.glob("*.parquet"):
        table.unlink()

    assert submit_request(tracked, _sweep(10)).request.request_id


def test_an_unpinned_reader_of_an_ambiguous_entry_is_refused_at_submit(
    tracked: Dataset,
) -> None:
    """The refusal names the entry and its variants, and how a request answers."""
    add_tracks_variant(tracked, OTHER_VARIANT, "seq_a", std_format="mosaic_v1")

    with pytest.raises(StepRefused) as raised:
        _ = submit_request(tracked, Recipe.model_validate(_chain()))

    assert raised.value.reason == "variant_ambiguous"
    assert raised.value.step_id == "speed"
    assert raised.value.detail == {
        "entry": ["", "seq_a"],
        "variants": sorted([VARIANT, OTHER_VARIANT]),
        "steps": ["speed"],
    }
    assert "tracks_run_id" in str(raised.value)
    assert "--tracks-run-id" in str(raised.value)
    assert '"tracks": {"step":' in str(raised.value)


def test_plan_pipeline_refuses_an_ambiguous_entry_before_any_step_resolves(
    tracked: Dataset,
) -> None:
    """Refused before the walk; a plan that records its problems keeps the step's."""
    add_tracks_variant(tracked, OTHER_VARIANT, "seq_a", std_format="mosaic_v1")
    recipe = Recipe.model_validate(_chain())

    with pytest.raises(StepRefused) as raised:
        _ = plan_pipeline(tracked, recipe, refuse=True)

    assert raised.value.reason == "variant_ambiguous"
    [problem] = plan_pipeline(tracked, recipe, refuse=False).problems
    assert (problem.step, problem.where) == ("speed", "tracks")


def test_an_ambiguous_entry_outside_the_scope_still_refuses_an_unpinned_reader(
    tracked: Dataset,
) -> None:
    """A reader with no variant hashes every entry's, whatever the scope."""
    add_tracks_variant(tracked, OTHER_VARIANT, "seq_a", std_format="mosaic_v1")
    recipe = Recipe.model_validate(_chain())
    elsewhere = Scope(entries=[("", "seq_b")])

    with pytest.raises(StepRefused) as raised:
        _ = submit_request(tracked, recipe, scope=elsewhere)

    assert raised.value.reason == "variant_ambiguous"
    assert raised.value.detail["entry"] == ["", "seq_a"]
    submitted = submit_request(tracked, recipe, scope=elsewhere, tracks_run_id=VARIANT)
    assert submitted.request.tracks_run_id == VARIANT


def test_a_scoped_reader_over_a_mixed_family_dataset_is_refused(
    tracked: Dataset,
) -> None:
    """Without a variant the family check reads every entry, as the identity does."""
    add_tracks_variant(tracked, VARIANT_B, "seq_b", std_format="trex_v1")

    with pytest.raises(StepRefused) as raised:
        _ = submit_request(
            tracked,
            Recipe.model_validate(_chain()),
            scope=Scope(entries=[("", "seq_a")]),
        )

    assert raised.value.reason == "schema_family_mismatch"


def test_an_unpinned_reader_refuses_itself_when_the_ambiguity_arrives_later(
    tracked: Dataset,
) -> None:
    """As when an op of the same graph writes a second variant of an entry."""
    request = submit_request(tracked, Recipe.model_validate(_chain())).request
    add_tracks_variant(tracked, OTHER_VARIANT, "seq_a", std_format="mosaic_v1")

    with pytest.raises(StepRefused) as raised:
        _ = execute_step(tracked, request, "speed")

    assert raised.value.reason == "variant_ambiguous"
    logged = read_run(run_log_dir(tracked.base_dir), request.execution_of("speed"))
    assert logged is not None
    assert logged["status"] == "failed"
    assert json.loads(logged["error_json"])["reason"] == "variant_ambiguous"


def test_an_unresolved_reader_keeps_what_its_guards_read(tracked: Dataset) -> None:
    """The held-out and reach checks still ask a step whose identity is open."""
    add_tracks_variant(tracked, OTHER_VARIANT, "seq_a", std_format="mosaic_v1")
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [
                {
                    "id": "pca",
                    "type": "feature",
                    "feature": "pair-posedistance-pca",
                    "inputs": ["tracks"],
                }
            ],
        }
    )

    step = plan_pipeline(tracked, recipe, refuse=False).step("pca")

    assert step.run_id is None
    assert step.fits
    assert step.scope_dependent


def test_only_the_unpinned_reader_is_refused(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A problem belongs to its step, so a sibling op of the request still runs."""
    document = _chain()
    steps = cast("list[Document]", document["steps"])
    steps.append(
        {
            "id": "trex",
            "type": "op",
            "kind": "trex",
            "params": {"track_max_individuals": 2},
        }
    )
    request = submit_request(tracked, Recipe.model_validate(document)).request
    add_tracks_variant(tracked, OTHER_VARIANT, "seq_a", std_format="mosaic_v1")
    ran: list[str] = []

    def _capture(*args: object, **kwargs: object) -> str:
        ran.append("trex")
        return "trex.0.1-0000000000"

    monkeypatch.setattr(step_module, "run_op", _capture)
    _ = execute_step(tracked, request, "trex")

    assert ran == ["trex"]


# --- the tracks variant a request names --------------------------------------


def _speed_only() -> Recipe:
    """One ``speed-angvel`` step reading the tracks, with no ``tracks`` reference."""
    return Recipe.model_validate(
        {
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
    )


def test_a_request_variant_plans_submits_and_runs_one_entry_of_two_variants(
    tracked: Dataset,
) -> None:
    """The request answers the choice that the recipe leaves open."""
    from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
    from mosaic.core.pipeline.run import resolve_feature_identity

    add_tracks_variant(tracked, OTHER_VARIANT, "seq_a", std_format="mosaic_v1")

    submitted = submit_request(tracked, _speed_only(), tracks_run_id=OTHER_VARIANT)
    outcome = execute_step(tracked, submitted.request, "speed")

    expected, _ = resolve_feature_identity(
        tracked,
        SpeedAngvel(params={"step_size": 1}),
        scope=Scope(entries=[("", "seq_a")]),
        tracks_run_id=OTHER_VARIANT,
    )
    assert submitted.request.tracks_run_id == OTHER_VARIANT
    assert submitted.plan.scope == frozenset({("", "seq_a")})
    assert outcome.run_id == expected
    assert outcome.covered == 1
    logged = read_run(
        run_log_dir(tracked.base_dir), submitted.request.execution_of("speed")
    )
    assert logged is not None
    assert logged["tracks_variant"] == OTHER_VARIANT


def test_a_request_variant_names_the_run_a_single_feature_run_names(
    tracked: Dataset,
) -> None:
    """A step and a single run over one variant serve each other's cache."""
    from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
    from mosaic.core.pipeline.run import run_feature

    add_tracks_variant(tracked, OTHER_VARIANT, "seq_a", std_format="mosaic_v1")
    single = run_feature(
        tracked,
        SpeedAngvel(params={"step_size": 1}),
        scope=Scope(entries=[("", "seq_a")]),
        tracks_run_id=OTHER_VARIANT,
    )

    submitted = submit_request(tracked, _speed_only(), tracks_run_id=OTHER_VARIANT)

    assert submitted.plan.step("speed").run_id == single.run_id
    assert submitted.plan.step("speed").status == "complete"


def test_a_deferred_feature_step_carries_the_request_variant(tmp_path: Path) -> None:
    """A step whose identity waits on a conversion still knows which tables it reads."""
    from tests.helpers import calms21_dataset, converting_recipe

    dataset = calms21_dataset(
        tmp_path / "ds", {"train": {"a": {"s1": 40}}}, seed=3, n_classes=2
    )
    variant = dataset.track_conversion_variant("calms21_npy")
    recipe = converting_recipe(
        {
            "id": "pca",
            "type": "feature",
            "feature": "pair-posedistance-pca",
            "inputs": ["tracks"],
            "after": ["tracks"],
        },
    )

    plan = plan_pipeline(
        dataset, recipe, request=_request_naming(variant), refuse=False
    )

    pca = plan.step("pca")
    assert pca.run_id is None
    assert pca.spec.tracks_run_id == variant


def _request_naming(variant: str) -> Request:
    return Request(request_id="R", recipe_digest="0" * 16, tracks_run_id=variant)


def test_a_resample_then_read_recipe_runs_and_resubmits(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A feature pinned to an op's tracks needs no request variant, then or after.

    ``run_op`` is stubbed to write the variant the plan minted for every entry,
    as the op does, because the fixture's tables carry no frame times to
    re-grid.
    """
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [
                {
                    "id": "regrid",
                    "type": "op",
                    "kind": "resample-tracks",
                    "params": {"target_fps": 15.0},
                },
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                    "tracks": {"step": "regrid"},
                    "params": {"step_size": 1},
                },
            ],
        }
    )

    first = submit_request(tracked, recipe)
    regrid = first.plan.step("regrid")
    resampled = regrid.tracks_variant
    assert resampled

    def _resample(*args: object, **kwargs: object) -> str:
        add_tracks_variant(
            tracked,
            resampled,
            "seq_a",
            "seq_b",
            std_format="mosaic_v1",
            producer_run_id=regrid.run_id or "",
        )
        return regrid.run_id or ""

    monkeypatch.setattr(step_module, "run_op", _resample)
    _ = execute_step(tracked, first.request, "regrid")
    speed = execute_step(tracked, first.request, "speed")

    logged = read_run(
        run_log_dir(tracked.base_dir), first.request.execution_of("speed")
    )
    assert logged is not None
    assert logged["tracks_variant"] == resampled
    assert speed.covered == 2

    again = submit_request(tracked, recipe)
    assert execute_step(tracked, again.request, "regrid").run_id
    assert execute_step(tracked, again.request, "speed").run_id == speed.run_id


def test_a_chained_op_with_no_source_is_a_problem_on_an_ambiguous_entry(
    tracked: Dataset,
) -> None:
    """An op that reads tracks needs a variant named as much as a feature does."""
    add_tracks_variant(tracked, OTHER_VARIANT, "seq_a", std_format="mosaic_v1")
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

    plan = plan_pipeline(tracked, recipe, refuse=False)

    [problem] = plan.problems
    assert (problem.step, problem.where) == (
        "regrid",
        "params.source_tracks_run_id",
    )
    assert plan.step("regrid").run_id is None


def test_a_pinned_feature_and_its_reader_run_on_a_multi_variant_dataset(
    tracked: Dataset,
) -> None:
    """The pin answers the choice, for the feature and for what reads the feature."""
    recipe = Recipe.model_validate(_tracked_then_read())
    planned = plan_pipeline(tracked, recipe).step("track")
    add_tracks_variant(
        tracked,
        planned.tracks_variant or "",
        "seq_a",
        "seq_b",
        std_format="trex_v2",
        producer_run_id=planned.run_id or "",
    )

    request = submit_request(tracked, recipe).request

    assert execute_step(tracked, request, "speed").covered == 2
    assert execute_step(tracked, request, "templates").covered >= 1


def test_a_bound_tracks_reader_is_not_refused_on_an_ambiguous_entry(
    tracked: Dataset,
) -> None:
    """A bound step is the run it names, so it reads no tracks to choose."""
    older = run_speed(tracked, 3)
    bind = {"speed": BoundRef(feature=SPEED_STORAGE, run_id=older)}
    recipe = Recipe.model_validate(_chain())
    request = submit_request(tracked, recipe, bind=bind).request
    add_tracks_variant(tracked, OTHER_VARIANT, "seq_a", std_format="mosaic_v1")

    plan = plan_pipeline(tracked, recipe, request=request, refuse=True)

    assert plan.step("speed").status == "bound"
    assert plan.step("speed").spec.tracks_run_id is None
    assert execute_step(tracked, request, "speed").run_id == older


def test_a_bound_tracks_reader_over_an_ambiguous_entry_is_submitted_as_its_step_plans(
    tracked: Dataset,
) -> None:
    """The submission plans with its bind, so it fingerprints the plan a step makes."""
    from mosaic.core.pipeline.graph import plan_fingerprint

    older = run_speed(tracked, 3)
    bind = {"speed": BoundRef(feature=SPEED_STORAGE, run_id=older)}
    add_tracks_variant(tracked, OTHER_VARIANT, "seq_a", std_format="mosaic_v1")
    recipe = Recipe.model_validate(_chain())

    submitted = submit_request(tracked, recipe, bind=bind)
    # What execute_step plans for a step with no ancestors and no exclusions.
    stepped = plan_pipeline(tracked, recipe, request=submitted.request, refuse=False)

    assert plan_fingerprint(stepped) == plan_fingerprint(submitted.plan)
    assert execute_step(tracked, submitted.request, "speed").run_id == older


def test_a_pinned_tracks_reader_is_not_refused_on_an_ambiguous_entry(
    tracked: Dataset,
) -> None:
    """A pinned step is not executed, so it reads no tracks to choose."""
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "a fit over the tracks",
            "steps": [
                {
                    "id": "fit",
                    "type": "feature",
                    "feature": "pair-posedistance-pca",
                    "inputs": ["tracks"],
                    "params": {"n_components": 2, "pose": {"pose_n": 3}},
                }
            ],
        }
    )
    fitted = execute_step(tracked, submit_request(tracked, recipe).request, "fit")
    add_tracks_variant(tracked, OTHER_VARIANT, "seq_a", std_format="mosaic_v1")

    plan = plan_pipeline(tracked, recipe, pins={"fit": fitted.run_id}, refuse=True)
    request = submit_request(tracked, recipe, pins={"fit": fitted.run_id}).request

    assert plan.step("fit").status == "complete"
    outcome = execute_step(tracked, request, "fit")
    assert (outcome.state, outcome.run_id) == ("cached", fitted.run_id)


def test_a_finished_unpinned_parent_keeps_its_run_under_a_later_variant(
    tracked: Dataset,
) -> None:
    """What a step was called is a fact, and an ambiguity since does not undo it."""
    request = submit_request(tracked, Recipe.model_validate(_chain())).request
    speed = execute_step(tracked, request, "speed")
    add_tracks_variant(tracked, OTHER_VARIANT, "seq_a", std_format="mosaic_v1")

    templates = execute_step(tracked, request, "templates")

    assert speed.run_id
    assert templates.run_id


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


def test_run_graph_request_refuses_tracks_run_id(tracked: Dataset) -> None:
    """A request records its variant at submit, and a step reads it from there."""
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
            "--tracks-run-id",
            VARIANT,
        ],
    )

    assert result.exit_code == 1
    assert "--tracks-run-id" in result.output
    assert "mosaic pipeline submit --tracks-run-id" in result.output


def test_run_graph_request_refuses_labels_run_id(tracked: Dataset) -> None:
    """A step's labels are chosen in the recipe, so the flag is refused."""
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
            "--labels-run-id",
            "labels.0.1-abc123def0",
        ],
    )

    assert result.exit_code == 1
    assert "--labels-run-id" in result.output
    assert "--graph-request" in result.output
    assert "recipe" in result.output


# --- what an op step hands its op --------------------------------------------


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


def _fail_speed(dataset: Dataset, request: Request) -> None:
    record_step_ended(
        dataset,
        request.execution_of("speed"),
        step_id="speed",
        owner="",
        status="failed",
        error_json=json.dumps({"type": "RuntimeError", "message": "speed broke"}),
    )


def test_a_failed_request_reads_running_while_its_driver_can_start_a_step(
    tracked: Dataset,
) -> None:
    """``sibling`` does not descend from ``speed``, and its driver still starts it."""
    request = submit_request(tracked, chain_with_sibling_recipe()).request
    _fail_speed(tracked, request)
    assert request_rollup(tracked.base_dir, request).status == "failed"

    _ = write_driver_lease(tracked.base_dir, request.request_id, stale=True)

    rollup = request_rollup(tracked.base_dir, request)
    assert rollup.status == "running"
    assert request_verdict(rollup.steps) == "failed"
    assert request_verdict_of(tracked.base_dir, rollup) == "orphaned"
    unread = request_rollup(tracked.base_dir, request, read_lease=False)
    assert unread.status == "failed"


def test_a_failed_request_whose_unstarted_steps_descend_from_it_reads_failed(
    tracked: Dataset,
) -> None:
    """``templates`` reads the failed ``speed``. A live driver does not start it."""
    request = submit_request(tracked, Recipe.model_validate(_chain())).request
    _fail_speed(tracked, request)
    _ = write_driver_lease(tracked.base_dir, request.request_id, stale=False)

    rollup = request_rollup(tracked.base_dir, request)

    assert rollup.status == "failed"
    assert request_verdict_of(tracked.base_dir, rollup) == "failed"


def test_a_failed_request_whose_recipe_does_not_read_stays_running_under_its_lease(
    tracked: Dataset,
) -> None:
    """The steps below the failure are read from the recipe, which is gone."""
    request = submit_request(tracked, Recipe.model_validate(_chain())).request
    _fail_speed(tracked, request)
    _ = write_driver_lease(tracked.base_dir, request.request_id, stale=False)
    recipe_path(tracked.base_dir, request.recipe_digest).unlink()

    assert request_rollup(tracked.base_dir, request).status == "running"


def test_a_status_recorded_outside_the_run_logs_closes_the_request() -> None:
    """A step whose dispatch failed does not write a run-log, and the step below
    it is not dispatched. The control plane that recorded both ends passes them in."""
    finished = StepAttempt("smooth", "e1", status="finished")
    undispatched = StepAttempt("speed", "e2")
    below = StepAttempt("delta", "e3")
    assert request_verdict((finished, undispatched, below)) == "running"

    failed = (
        finished,
        replace(undispatched, status="failed"),
        replace(below, status="failed"),
    )
    assert request_verdict(failed) == "failed"
    stopped = (finished, replace(undispatched, status="cancelled"), below)
    assert request_verdict(stopped) == "cancelled"


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
    from mosaic.core.pipeline.entry_claim import RunRootHeld
    from mosaic.core.pipeline.markers import new_inflight, write_inflight
    from mosaic.core.pipeline.models import model_run_root

    dataset = make_dataset(tmp_path)
    trainer = FakeTrainer()
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
                        "data": str(write_data_yaml(tmp_path)),
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

    with pytest.raises(RunRootHeld, match="SOMEONE-ELSE") as refused:
        _ = execute_step(dataset, submitted.request, "train")

    assert refused.value.reason == "run_root_held"
    assert refused.value.detail["holder_execution_id"] == "SOMEONE-ELSE"
    assert refused.value.detail["holder_host"] == "otherhost"
    assert refused.value.detail["holder_pid"] == 4242

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


# --- storage names too long for a directory ----------------------------------


def _long_lineage() -> Document:
    """A step over two multi-input parents, whose full storage name is 281 bytes.

    Each parent stacks four ``speed-angvel`` runs, so ``combined`` would store
    under ``temporal-stack__from__`` and both parents' four-input lineages.
    """
    speeds: list[JsonValue] = [{"step": f"speed{size}"} for size in range(1, 5)]
    return {
        "schema_version": 1,
        "name": "a lineage too long for a directory",
        "steps": [
            *(
                {
                    "id": f"speed{size}",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                    "params": {"step_size": size},
                }
                for size in range(1, 5)
            ),
            {
                "id": "narrow",
                "type": "feature",
                "feature": "temporal-stack",
                "inputs": speeds,
                "params": {"half": 2},
            },
            {
                "id": "wide",
                "type": "feature",
                "feature": "temporal-stack",
                "inputs": speeds,
                "params": {"half": 3},
            },
            {
                "id": "combined",
                "type": "feature",
                "feature": "temporal-stack",
                "inputs": [{"step": "narrow"}, {"step": "wide"}],
                "params": {"half": 1},
            },
        ],
    }


def _run_parents_of_combined(dataset: Dataset, request: Request) -> None:
    for step_id in ("speed1", "speed2", "speed3", "speed4", "narrow", "wide"):
        assert execute_step(dataset, request, step_id).state == "ran"


def test_a_step_whose_full_storage_name_exceeds_a_filename_runs(
    tracked: Dataset,
) -> None:
    recipe = Recipe.model_validate(_long_lineage())
    planned = plan_pipeline(tracked, recipe).step("combined")
    request = submit_request(tracked, recipe).request
    _run_parents_of_combined(tracked, request)

    combined = execute_step(tracked, request, "combined")

    assert combined.state == "ran"
    assert SHORTENED_MARKER in planned.storage_name
    run_root = Path(tracked.get_root("features")) / planned.storage_name
    assert (run_root / combined.run_id / "params.json").is_file()


def test_a_failed_step_with_an_abbreviated_name_records_its_failure(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure record is written, and the attempt counts toward backoff."""
    recipe = Recipe.model_validate(_long_lineage())
    request = submit_request(tracked, recipe).request
    _run_parents_of_combined(tracked, request)
    planned = plan_pipeline(tracked, recipe).step("combined")

    def run_feature(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("the feature failed")

    monkeypatch.setattr(step_module, "run_feature", run_feature)

    with pytest.raises(RuntimeError, match="the feature failed"):
        _ = execute_step(tracked, request, "combined")

    key = (planned.storage_name, planned.run_id or "")
    assert FileFailureStore(tracked.base_dir).step_record(key).attempts == 1


def test_a_name_that_fits_until_encoded_records_and_clears_claims(
    tracked: Dataset,
) -> None:
    """Percent-encoding lengthens each ``+`` by 2 bytes, past one path component."""
    storage = storage_name_of("temporal-stack", ["tracks"] * 33)
    assert len(storage.encode("utf-8")) <= NAME_MAX_BYTES
    assert len(to_safe_name(storage)) > NAME_MAX_BYTES
    store = FileFailureStore(tracked.base_dir)
    step_key = (storage, "0.5-0000000000")
    entry_key = (storage, "0.5-0000000000", "", "seq_a")

    _ = store.note_step_failure(step_key, error="the step failed")
    _ = store.note_entry_failure(entry_key, error="the entry failed")

    assert store.step_record(step_key).attempts == 1
    assert store.entry_record(entry_key).attempts == 1
    store.clear_step(step_key)
    store.clear_entry(entry_key)
    assert store.step_record(step_key).attempts == 0
    assert store.entry_record(entry_key).attempts == 0


# --- parallelism -------------------------------------------------------------


class _Stopped(Exception):
    """Ends a step at the call of ``run_feature``, after its arguments are seen."""


def _run_feature_arguments(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Replace ``run_feature`` with a stub that records its keyword arguments."""
    seen: dict[str, object] = {}

    def run_feature(*_args: object, **kwargs: object) -> None:
        seen.update(kwargs)
        raise _Stopped

    monkeypatch.setattr(step_module, "run_feature", run_feature)
    monkeypatch.setattr(parallelism, "host_cpus", lambda: 8)
    return seen


def test_a_parallelizable_step_is_run_with_the_workers_it_was_given(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = submit_request(tracked, Recipe.model_validate(_chain())).request
    seen = _run_feature_arguments(monkeypatch)
    settings = ParallelSettings(workers=3, mode="process", min_free_memory_fraction=0.2)

    with pytest.raises(_Stopped):
        _ = execute_step(tracked, request, "speed", parallel=settings)

    assert seen["parallel_workers"] == 3
    assert seen["parallel_mode"] == "process"
    assert seen["min_free_memory_fraction"] == 0.2


def test_a_step_given_no_workers_runs_in_turn(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = submit_request(tracked, Recipe.model_validate(_chain())).request
    seen = _run_feature_arguments(monkeypatch)

    with pytest.raises(_Stopped):
        _ = execute_step(tracked, request, "speed")

    assert seen["parallel_workers"] is None
    assert seen["min_free_memory_fraction"] == 0.0


def test_a_step_that_fits_across_every_entry_is_not_given_workers(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = submit_request(tracked, Recipe.model_validate(_chain())).request
    _ = execute_step(tracked, request, "speed")
    seen = _run_feature_arguments(monkeypatch)

    with pytest.raises(_Stopped):
        _ = execute_step(
            tracked, request, "templates", parallel=ParallelSettings(workers=4)
        )

    assert seen["parallel_workers"] is None


def test_the_worker_count_is_not_part_of_a_steps_identity(tracked: Dataset) -> None:
    recipe = Recipe.model_validate(_chain())
    request = submit_request(tracked, recipe).request
    planned = plan_pipeline(tracked, recipe).step("speed")

    outcome = execute_step(
        tracked, request, "speed", parallel=ParallelSettings(workers=2)
    )

    assert outcome.run_id == planned.run_id


# --- a worker without the tool ------------------------------------------------


def _tool_recipe(feature: str) -> Recipe:
    """``speed-angvel`` beside a step that needs a tool, both reading the tracks."""
    tool_step: Document
    if feature == "kpms":
        tool_step = {
            "id": "tool",
            "type": "feature",
            "feature": "kpms",
            "inputs": ["tracks"],
            "params": kpms_params(),
        }
    else:
        tool_step = {
            "id": "tool",
            "type": "feature",
            "feature": "lightning-action",
            "inputs": [{"step": "speed"}],
            "params": lightning_action_params(),
        }
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "a tool beside a sibling",
            "steps": [
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                },
                tool_step,
            ],
        }
    )


def _submit_with_kpms(tracked: Dataset, monkeypatch: pytest.MonkeyPatch) -> Request:
    """Submit the kpms recipe where the license is accepted and an interpreter found."""
    monkeypatch.setenv(KPMS_LICENSE_ENV, "1")
    monkeypatch.setenv(KPMS_PYTHON_ENV, sys.executable)
    declaration_catalog.cache_clear()
    return submit_request(tracked, _tool_recipe("kpms")).request


def _run_step_json(
    dataset: Dataset, request: Request, step_id: str
) -> tuple[int, dict[str, object]]:
    result = runner.invoke(
        app,
        [
            "run",
            "--json",
            "--manifest",
            str(dataset.manifest_path),
            "--graph-request",
            request.request_id,
            "--step",
            step_id,
            "--execution-id",
            request.execution_of(step_id),
        ],
    )
    return result.exit_code, json.loads(result.stdout)


def _recorded_error(
    dataset: Dataset, request: Request, step_id: str
) -> dict[str, object]:
    logged = read_run(run_log_dir(dataset.base_dir), request.execution_of(step_id))
    assert logged is not None
    assert logged["status"] == "failed"
    return json.loads(logged["error_json"])


def _terminal_events(dataset: Dataset, execution_id: str) -> list[str]:
    lines = run_log_path(dataset.base_dir, execution_id).read_text().splitlines()
    events = [json.loads(line)["ev"] for line in lines]
    return [event for event in events if event in TERMINAL_STATUSES]


@pytest.mark.usefixtures("fresh_catalogs")
def test_kpms_without_its_license_exits_65(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _submit_with_kpms(tracked, monkeypatch)
    monkeypatch.delenv(KPMS_LICENSE_ENV)
    declaration_catalog.cache_clear()

    code, payload = _run_step_json(tracked, request, "tool")

    assert code == REFUSED_EXIT_CODE
    assert payload["status"] == "refused"
    assert payload["reason"] == "tool_unavailable"
    recorded = _recorded_error(tracked, request, "tool")
    assert recorded["reason"] == "tool_unavailable"
    assert (recorded["tool"], recorded["cause"]) == ("keypoint-moseq", "license")
    assert recorded["step"] == "tool"


@pytest.mark.usefixtures("fresh_catalogs")
def test_lightning_action_not_installed_exits_65(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = install_lightning_action_stub(monkeypatch)
    declaration_catalog.cache_clear()
    request = submit_request(tracked, _tool_recipe("lightning-action")).request
    hide_package(monkeypatch, "lightning_action")
    declaration_catalog.cache_clear()

    code, payload = _run_step_json(tracked, request, "tool")

    assert code == REFUSED_EXIT_CODE
    assert payload["reason"] == "tool_unavailable"
    recorded = _recorded_error(tracked, request, "tool")
    assert recorded["reason"] == "tool_unavailable"
    assert (recorded["tool"], recorded["cause"]) == ("lightning-action", "environment")


@pytest.mark.usefixtures("fresh_catalogs")
def test_a_sibling_step_runs_on_a_worker_without_the_tool(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _submit_with_kpms(tracked, monkeypatch)
    monkeypatch.delenv(KPMS_LICENSE_ENV)
    declaration_catalog.cache_clear()

    outcome = execute_step(tracked, request, "speed")

    assert outcome.state == "ran"
    with pytest.raises(ToolUnavailable):
        _ = execute_step(tracked, request, "tool")
    # Neither a submission nor a plan accepts the step on this installation.
    with pytest.raises(RecipeInvalid):
        _ = submit_request(tracked, _tool_recipe("kpms"))
    with pytest.raises(RecipeInvalid):
        _ = plan_pipeline(tracked, _tool_recipe("kpms"))


@pytest.mark.usefixtures("fresh_catalogs")
def test_a_refusal_before_the_run_log_records_the_step_ended(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _submit_with_kpms(tracked, monkeypatch)
    monkeypatch.delenv(KPMS_LICENSE_ENV)
    declaration_catalog.cache_clear()
    attempt = request.execution_of("tool")
    _ = execute_step(tracked, request, "speed")

    with pytest.raises(ToolUnavailable):
        _ = execute_step(tracked, request, "tool")

    assert _terminal_events(tracked, attempt) == ["failed"]
    assert _recorded_error(tracked, request, "tool")["reason"] == "tool_unavailable"
    assert request_rollup(tracked.base_dir, request).status == "failed"


def test_a_tracker_codec_refusal_leaves_one_terminal_event(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    request = submit_request(tracked, recipe).request

    def _refuse(*args: object, **kwargs: object) -> str:
        raise ToolCodecError("clip0.mp4 is av1")

    monkeypatch.setattr(registered_op("resample-tracks"), "run", _refuse)

    with pytest.raises(ToolCodecError):
        _ = execute_step(tracked, request, "regrid")

    assert _terminal_events(tracked, request.execution_of("regrid")) == ["failed"]
    assert _recorded_error(tracked, request, "regrid")["reason"] == "undecodable_codec"


@pytest.mark.usefixtures("fresh_catalogs")
def test_a_tool_unavailable_from_fit_leaves_one_terminal_event(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The declaration still reads the license as accepted when ``fit`` finds it gone."""
    request = _submit_with_kpms(tracked, monkeypatch)
    accepted = declaration_catalog()
    monkeypatch.delenv(KPMS_LICENSE_ENV)

    with pytest.raises(ToolUnavailable):
        _ = execute_step(tracked, request, "tool", catalog=accepted)

    attempt = request.execution_of("tool")
    assert _terminal_events(tracked, attempt) == ["failed"]
    recorded = _recorded_error(tracked, request, "tool")
    assert (recorded["reason"], recorded["cause"]) == ("tool_unavailable", "license")
