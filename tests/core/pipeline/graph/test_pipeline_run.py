"""Running a plan in this process, and the four ``mosaic pipeline`` verbs.

The no-queue path, which is what a notebook and a bare compute node have. Two
properties carry the suite: a run does what the plan said it would and records
the identifiers it resolved, and a second run of the same recipe does nothing at
all. The third is the refusal -- a step whose identity covers its scope must not
quietly run over less of it.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner, Result

from mosaic.behavior.feature_library.global_kmeans import GlobalKMeansClustering
from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.cancel import RequestCancel, cancel_request
from mosaic.core.pipeline.graph import run as run_module
from mosaic.core.pipeline.graph import validate as validate_module
from mosaic.core.pipeline.graph import (
    CoverageShortfall,
    DriverLease,
    DriverRunning,
    Recipe,
    Request,
    StepOutcome,
    build_step_feature,
    driver_lease_path,
    execute_step,
    plan_pipeline,
    plan_wire,
    read_driver_lease,
    recipe_digest,
    request_rollup,
    request_verdict_of,
    run_pipeline,
    submit_request,
)
from mosaic.core.pipeline.job import Cancelled, CancelToken
from mosaic.core.pipeline.refusal import REFUSED_EXIT_CODE
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.run_log import (
    JsonlRunLog,
    read_run,
    run_log_dir,
    run_log_path,
)
from mosaic.core.scope import Scope
from tests.helpers import (
    add_tracks_variant,
    catalog_without_contract,
    chain_recipe,
    make_dataset,
    write_driver_lease,
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
        {
            "id": "scaler",
            "type": "feature",
            "feature": "global-scaler",
            "inputs": [{"step": "speed"}],
            "params": {
                "templates": {"step": "templates", "pattern": "templates.parquet"}
            },
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


def _recorded_params(dataset: Dataset, run_id: str) -> str:
    """The provenance one run wrote, as text, whichever storage it landed under.

    Read off disk rather than from the plan, because what is being asked is what
    the *run* pinned -- the plan's answer is the prediction under test.
    """
    found = list(Path(dataset.get_root("features")).glob(f"*/{run_id}/params.json"))
    assert found, f"no params.json for run {run_id}"
    return found[0].read_text()


UNPINNED_CHAIN: Document = json.loads(
    json.dumps(CHAIN).replace(', "pattern": "templates.parquet"', "")
)
"""``CHAIN`` with the scaler's artifact reference naming no file.

This shape once validated, planned and ran while the scaler fitted on
``seq_a.parquet`` -- one sequence's pass-through table -- because the derived
``*.parquet`` glob matched the per-entry outputs beside the templates matrix and
they sort first.
"""


# --- the runner ------------------------------------------------------------------


def test_an_unpinned_artifact_reference_reads_the_declared_artifact(
    tracked: Dataset,
) -> None:
    """The consumer's declared type names the file, so the recipe need not.

    Two sequences, so the producer's run root holds ``seq_a.parquet`` and
    ``seq_b.parquet`` beside ``templates.parquet`` and
    ``template_provenance.parquet`` -- every arrangement in which taking the
    first sorted match is wrong. What closes it is that the scaler fits on the
    template matrix, which its own column list is read from.
    """
    recipe = Recipe.model_validate(UNPINNED_CHAIN)

    done = run_pipeline(tracked, recipe)

    assert [outcome.state for outcome in done.outcomes] == ["ran", "ran", "ran"]
    scaler = done.outcomes[-1]
    assert scaler.run_id is not None
    assert '"pattern": "templates.parquet"' in _recorded_params(tracked, scaler.run_id)

    templates_run = done.outcomes[1].run_id
    assert templates_run is not None
    produced = _table(tracked, templates_run, "templates.parquet")
    fitted = _table(tracked, scaler.run_id, "scaled_templates.parquet")
    assert list(fitted.columns) == list(produced.columns)
    assert len(fitted) == len(produced)


def _table(dataset: Dataset, run_id: str, name: str) -> pd.DataFrame:
    """One named artifact out of whichever storage *run_id* landed under."""
    found = list(Path(dataset.get_root("features")).glob(f"*/{run_id}/{name}"))
    assert found, f"no {name} for run {run_id}"
    return pd.read_parquet(found[0])


CLUSTERING: Document = {
    "schema_version": 1,
    "name": "simple clustering",
    "steps": [
        {
            "id": "smooth",
            "type": "feature",
            "feature": "trajectory-smooth",
            "inputs": ["tracks"],
        },
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "inputs": [{"step": "smooth"}],
            "params": {"step_size": 4},
        },
        {
            "id": "groups",
            "type": "feature",
            "feature": "ffgroups-metrics",
            "inputs": [{"step": "speed"}, {"step": "smooth"}],
        },
        {
            "id": "templates",
            "type": "feature",
            "feature": "extract-templates",
            "inputs": [{"step": "groups"}, {"step": "speed"}],
            "params": {"n_templates": 500},
        },
        {
            "id": "kmeans",
            "type": "feature",
            "feature": "global-kmeans",
            "inputs": [{"step": "templates"}],
            "params": {"templates": {"step": "templates"}, "k": 1, "n_init": 1},
        },
    ],
}
"""A k-means fit over a windowed feature, which leaves NaN at each sequence's edges.

``speed-angvel`` with ``step_size: 4`` writes NaN ``speed_step`` on the first
and last rows of every sequence. The pool holds more rows than the fixture, and
every row is a candidate template. The fixture's individuals move at a constant
speed: the finite templates are one point and ``k`` is 1.
"""


def test_a_global_fit_runs_over_a_feature_with_nan_edges(tracked: Dataset) -> None:
    """Templates are sampled from the finite rows, and k-means fits on them.

    The fit once raised scikit-learn's "Input X contains NaN", because the
    templates included the edge rows.
    """
    done = run_pipeline(tracked, Recipe.model_validate(CLUSTERING))

    assert [outcome.state for outcome in done.outcomes] == ["ran"] * 5
    templates_run = done.outcomes[3].run_id
    assert templates_run is not None
    provenance = _table(tracked, templates_run, "template_provenance.parquet")
    assert (provenance["dropped"] > 0).all()
    templates = _table(tracked, templates_run, "templates.parquet")
    assert templates.notna().all().all()


NAMED_MEASUREMENTS: Document = {
    "schema_version": 1,
    "name": "clustering on named measurements",
    "steps": [
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "inputs": ["tracks"],
            "params": {"step_size": 2},
        },
        {
            "id": "templates",
            "type": "feature",
            "feature": "extract-templates",
            "inputs": [{"step": "speed"}],
            "params": {"n_templates": 100, "columns": ["speed_step", "speed"]},
        },
        {
            "id": "scaler",
            "type": "feature",
            "feature": "global-scaler",
            "inputs": [{"step": "speed"}],
            "params": {"templates": {"step": "templates"}},
        },
        {
            "id": "scaled_templates",
            "type": "feature",
            "feature": "extract-templates",
            "inputs": [{"step": "scaler"}],
            "params": {
                "n_templates": 10,
                "strategy": "farthest_first",
                "pool": {"size": 40},
            },
        },
        {
            "id": "kmeans",
            "type": "feature",
            "feature": "global-kmeans",
            "inputs": [{"step": "scaler"}],
            "params": {"templates": {"step": "scaled_templates"}, "k": 1, "n_init": 1},
        },
    ],
}
"""A clustering on two of the four measurements of ``speed-angvel``.

The first templates name ``speed`` and ``speed_step``, and leave out ``vx`` and
``vy``. The resampled templates read the scaler's output and leave ``columns``
unset.
"""


def test_a_clustering_fits_only_the_named_measurements(tracked: Dataset) -> None:
    """The scaler writes the fitted columns alone, and every later fit reads them.

    The scaler once kept every column of its input and scaled only the fitted
    ones, and the resampled templates sampled ``vx`` and ``vy`` unscaled beside
    the two named columns.
    """
    done = run_pipeline(tracked, Recipe.model_validate(NAMED_MEASUREMENTS))

    assert [outcome.state for outcome in done.outcomes] == ["ran"] * 5
    run_ids = {outcome.step_id: outcome.run_id for outcome in done.outcomes}
    named = ["speed", "speed_step"]
    for step in ("templates", "scaled_templates"):
        run_id = run_ids[step]
        assert run_id is not None
        assert list(_table(tracked, run_id, "templates.parquet").columns) == named
    kmeans_run = run_ids["kmeans"]
    assert kmeans_run is not None
    found = list(
        Path(tracked.get_root("features")).glob(f"*/{kmeans_run}/model.joblib")
    )
    assert len(found) == 1
    model = GlobalKMeansClustering.ModelArtifact().from_path(found[0])
    assert model["feature_columns"] == named


def test_a_run_records_the_identifiers_the_plan_resolved(tracked: Dataset) -> None:
    """The plan is a preview, and here the preview turns out to have been right."""
    recipe = Recipe.model_validate(CHAIN)
    planned = plan_pipeline(tracked, recipe).run_ids

    done = run_pipeline(tracked, recipe)

    assert [outcome.state for outcome in done.outcomes] == ["ran", "ran", "ran"]
    assert {outcome.step_id: outcome.run_id for outcome in done.outcomes} == planned
    assert not any(outcome.diverged for outcome in done.outcomes)


def test_a_second_run_does_nothing(tracked: Dataset) -> None:
    """Every step is served from its own directory, which is what caching means."""
    recipe = Recipe.model_validate(CHAIN)
    first = run_pipeline(tracked, recipe)

    again = run_pipeline(tracked, recipe)

    assert [outcome.state for outcome in again.outcomes] == ["cached"] * 3
    assert again.ran == ()
    assert {outcome.run_id for outcome in again.outcomes} == {
        outcome.run_id for outcome in first.outcomes
    }


def test_a_step_already_computed_is_not_recomputed(tracked: Dataset) -> None:
    """A partly built graph resumes rather than starting again."""
    recipe = Recipe.model_validate(CHAIN)
    plan = plan_pipeline(tracked, recipe)
    _ = run_feature(tracked, build_step_feature(plan.step("speed").spec), track=False)

    done = run_pipeline(tracked, recipe)

    assert done.outcomes[0].state == "cached"
    assert [outcome.state for outcome in done.outcomes[1:]] == ["ran", "ran"]


UNCOMPUTABLE = Scope(entries=[("", "seq_a"), ("", "seq_b"), ("", "seq_gone")])
"""A submission naming one entry more than the dataset can process.

An explicit narrowing is not widened or narrowed by what is on disk -- it is the
submission saying what it wants -- so this is a scope the graph cannot fill, which
is the state a shortfall describes. A partly-computed upstream is *not*: the
runner completes it first, which is what it is for.
"""


def test_a_scope_dependent_step_refuses_a_short_upstream(tracked: Dataset) -> None:
    """The refusal that makes a shortfall a decision rather than a default.

    ``speed`` computes what it can and is correct in doing so: its outputs are
    per entry, and the missing one arrives later under the same identifier. The
    fit below it is not like that -- over two of three entries it is a different
    artifact from the one that was asked for, under a name saying it is the same.
    """
    with pytest.raises(CoverageShortfall) as raised:
        _ = run_pipeline(tracked, Recipe.model_validate(CHAIN), scope=UNCOMPUTABLE)

    assert raised.value.step_id == "templates"
    assert raised.value.covered == 2
    assert raised.value.target == 3
    assert "seq_gone" in str(raised.value)


def test_allow_partial_is_the_gesture_that_answers_the_refusal(
    tracked: Dataset,
) -> None:
    """Recorded as a choice, and the run then proceeds over what there is.

    The fit lands somewhere other than where it was planned to -- it covers the
    entries it was actually given, and those are its name -- and the step below
    it still resolves, because what governs is what the run recorded rather than
    what the plan predicted.
    """
    done = run_pipeline(
        tracked,
        Recipe.model_validate(CHAIN),
        scope=UNCOMPUTABLE,
        allow_partial=True,
    )

    assert [outcome.state for outcome in done.outcomes] == ["ran", "ran", "ran"]
    templates = next(o for o in done.outcomes if o.step_id == "templates")
    assert templates.diverged, "its identity covers the set it was fitted on"
    scaler = next(o for o in done.outcomes if o.step_id == "scaler")
    assert templates.run_id in _recorded_params(tracked, scaler.run_id)


def test_a_partly_computed_upstream_is_completed_rather_than_refused(
    tracked: Dataset,
) -> None:
    """The case a shortfall is *not*, kept apart because the two look alike.

    Half a scope already computed is a resumable run, and the runner asks the
    step for the remainder. Only a scope nothing can fill is a shortfall.
    """
    recipe = Recipe.model_validate(CHAIN)
    plan = plan_pipeline(tracked, recipe)
    _ = run_feature(
        tracked,
        build_step_feature(plan.step("speed").spec),
        scope=Scope(entries=[("", "seq_a")]),
        track=False,
    )

    done = run_pipeline(tracked, recipe)

    assert [outcome.state for outcome in done.outcomes] == ["ran", "ran", "ran"]


def test_a_narrowed_run_covers_only_what_was_asked_for(tracked: Dataset) -> None:
    """The narrowing is the submission speaking, and the run honours it."""
    recipe = Recipe.model_validate(CHAIN)

    done = run_pipeline(tracked, recipe, scope=Scope(entries=[("", "seq_a")]))

    assert done.scope == frozenset({("", "seq_a")})
    assert all(outcome.state == "ran" for outcome in done.outcomes)


# --- the CLI ---------------------------------------------------------------------


def test_validate_says_a_good_recipe_is_good(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["pipeline", "validate", "--recipe", f"@{_recipe_file(tmp_path, CHAIN)}"]
    )

    assert result.exit_code == 0, result.output
    assert "valid" in result.stdout


def test_validate_exits_non_zero_and_names_the_fault(tmp_path: Path) -> None:
    """A recipe checked in CI has to fail the shell, not only print."""
    broken: Document = {
        "steps": [{"id": "a", "type": "feature", "feature": "no-such-feature"}]
    }
    result = runner.invoke(
        app, ["pipeline", "validate", "--recipe", f"@{_recipe_file(tmp_path, broken)}"]
    )

    assert result.exit_code == 1
    assert "no-such-feature" in result.output


@pytest.fixture
def undeclared_below_speed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation reads a catalog in which the features below ``speed`` declare no columns."""
    catalog = catalog_without_contract("extract-templates", "global-scaler")
    monkeypatch.setattr(validate_module, "declaration_catalog", lambda: catalog)


@pytest.mark.usefixtures("undeclared_below_speed")
def test_validate_opens_no_dataset(tmp_path: Path) -> None:
    """It takes no manifest at all, which is the property rather than a default."""
    result = runner.invoke(
        app,
        [
            "pipeline",
            "validate",
            "--recipe",
            f"@{_recipe_file(tmp_path, CHAIN)}",
            "--json",
        ],
    )

    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["digest"] == recipe_digest(Recipe.model_validate(CHAIN))
    assert payload["problems"] == []
    # Neither feature below speed-angvel declares the columns it reads. Both
    # are listed as unchecked, which is not a caution and leaves the recipe valid.
    assert payload["unchecked"] == ["templates", "scaler"]
    assert payload["cautions"] == []


@pytest.mark.usefixtures("undeclared_below_speed")
def test_validate_names_the_unchecked_steps_in_one_line(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["pipeline", "validate", "--recipe", f"@{_recipe_file(tmp_path, CHAIN)}"]
    )

    assert result.exit_code == 0, result.output
    assert result.stderr.splitlines() == ["Columns not checked for templates, scaler"]
    assert "caution" not in result.output


def test_show_reads_the_graph_without_a_dataset(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["pipeline", "show", "--recipe", f"@{_recipe_file(tmp_path, CHAIN)}", "--json"],
    )

    payload = json.loads(result.stdout)
    assert [step["id"] for step in payload["steps"]] == ["speed", "templates", "scaler"]
    assert {
        (edge["producer"], edge["consumer"], edge["site"]) for edge in payload["edges"]
    } == {
        ("speed", "templates", "inputs"),
        ("speed", "scaler", "inputs"),
        ("templates", "scaler", "params"),
    }


def test_show_reports_a_malformed_recipe_as_an_invalid_envelope() -> None:
    recipe = {"schema_version": 1, "steps": [{"id": "a", "type": "bogus"}]}

    result = runner.invoke(
        app, ["pipeline", "show", "--recipe", json.dumps(recipe), "--json"]
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["kind"] == "invalid"
    assert payload["message"] in result.stderr
    assert [(p["step"], p["where"]) for p in payload["detail"]["problems"]] == [
        ("a", "type")
    ]


def test_plan_reports_a_malformed_recipe_as_an_invalid_envelope_on_stdout(
    tracked: Dataset,
) -> None:
    recipe = {"schema_version": 1, "steps": [{"id": "a", "type": "bogus"}]}

    result = runner.invoke(
        app,
        [
            "pipeline",
            "plan",
            "--manifest",
            str(tracked.manifest_path),
            "--recipe",
            json.dumps(recipe),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["kind"] == "invalid"
    assert payload["message"] in result.stderr


def test_plan_reports_identities_and_what_is_left(
    tracked: Dataset, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            "plan",
            "--recipe",
            f"@{_recipe_file(tmp_path, CHAIN)}",
            "--manifest",
            str(tracked.manifest_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["complete"] is False
    assert [step["step"] for step in payload["steps"]] == [
        "speed",
        "templates",
        "scaler",
    ]
    assert all(step["run_id"] is not None for step in payload["steps"])
    assert payload["steps"][1]["reason"] == {
        "kind": "deps_incomplete",
        "blocking": ["speed"],
    }
    in_process = plan_pipeline(tracked, Recipe.model_validate(CHAIN))
    assert payload == plan_wire(in_process).model_dump(mode="json")


def test_run_executes_the_graph_and_then_finds_it_complete(
    tracked: Dataset, tmp_path: Path
) -> None:
    """The end-to-end path a person actually takes: run, then plan again."""
    recipe_path = _recipe_file(tmp_path, CHAIN)
    manifest = str(tracked.manifest_path)

    ran = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--recipe",
            f"@{recipe_path}",
            "--manifest",
            manifest,
            "--json",
        ],
    )
    assert ran.exit_code == 0, ran.output
    assert [step["state"] for step in json.loads(ran.stdout)["steps"]] == ["ran"] * 3

    again = runner.invoke(
        app,
        [
            "pipeline",
            "plan",
            "--recipe",
            f"@{recipe_path}",
            "--manifest",
            manifest,
            "--json",
        ],
    )
    assert json.loads(again.stdout)["complete"] is True


def _run_short(tracked: Dataset, tmp_path: Path, *flags: str) -> Result:
    """``mosaic pipeline run`` over a scope a step of CHAIN cannot fully cover."""
    return runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--recipe",
            f"@{_recipe_file(tmp_path, CHAIN)}",
            "--manifest",
            str(tracked.manifest_path),
            *[f"--entry={group}:{sequence}" for group, sequence in UNCOMPUTABLE],
            *flags,
        ],
    )


def test_run_refuses_a_shortfall_with_the_refusal_exit_code(
    tracked: Dataset, tmp_path: Path
) -> None:
    result = _run_short(tracked, tmp_path)

    assert result.exit_code == REFUSED_EXIT_CODE
    assert isinstance(result.exception, SystemExit), result.exception
    assert "[mosaic] refused (coverage_shortfall)" in result.stderr
    assert "allow_partial" in result.stderr


def test_a_shortfall_is_a_json_refusal_under_json(
    tracked: Dataset, tmp_path: Path
) -> None:
    """The shape a codec refusal has, naming the step and its attempt.

    The step here stalls: its attempt finishes, and the run refuses to go on
    because coverage did not move.
    """
    result = _run_short(tracked, tmp_path, "--json")

    assert result.exit_code == REFUSED_EXIT_CODE
    payload = json.loads(result.stdout)
    assert payload["status"] == "refused"
    assert payload["reason"] == "coverage_shortfall"
    assert payload["step"] == json.loads(payload["error_json"])["step"] == "speed"
    attempt = read_run(run_log_dir(tracked.base_dir), payload["execution_id"])
    assert attempt is not None, "the attempt of the step that was refused"


POINT_MODEL_TRACKED: Document = {
    "schema_version": 1,
    "name": "track with a point model",
    "steps": [
        {
            "id": "track",
            "type": "op",
            "kind": "ultralytics",
            "params": {"model_path": "train-points.0.2-abcdef0123"},
        },
    ],
}
"""A tracker step handed a model its op does not run, which planning refuses."""


@pytest.mark.parametrize("verb", ["plan", "run", "submit"])
def test_a_model_the_op_does_not_run_is_refused_as_a_message(
    verb: str, tracked: Dataset, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            verb,
            "--recipe",
            f"@{_recipe_file(tmp_path, POINT_MODEL_TRACKED)}",
            "--manifest",
            str(tracked.manifest_path),
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), result.exception
    assert "runs models trained by train-pose" in result.stderr


# --- the in-process driver lease ---------------------------------------------


def _finish_step(
    ds: Dataset, request: Request, step_id: str, *, state: str = "ran"
) -> StepOutcome:
    """Record *step_id*'s attempt finished, as a step that ran does."""
    execution_id = request.execution_of(step_id)
    with JsonlRunLog(run_log_path(ds.base_dir, execution_id), execution_id) as log:
        log.started(kind="feature", target=step_id, pid=os.getpid())
        log.finished()
    return StepOutcome(
        step_id=step_id,
        kind="feature",
        run_id="0.1-0000000000",
        state="stalled" if state == "stalled" else "ran",
        execution_id=execution_id,
        covered=1,
        target=2,
    )


def test_an_in_process_run_holds_the_lease_while_a_step_runs(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[DriverLease | None] = []

    def observed(ds: Dataset, request: Request, step_id: str, **kwargs: object):
        seen.append(read_driver_lease(ds.base_dir, request.request_id))
        return execute_step(ds, request, step_id, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(run_module, "execute_step", observed)

    _ = run_pipeline(tracked, chain_recipe())

    assert [
        (lease.in_process, lease.step_id, lease.step_pid)
        for lease in seen
        if lease is not None
    ] == [(True, "speed", os.getpid()), (True, "templates", os.getpid())]


def test_an_in_process_run_releases_its_lease_when_the_request_ends(
    tracked: Dataset,
) -> None:
    done = run_pipeline(tracked, chain_recipe())

    assert not driver_lease_path(tracked.base_dir, done.request_id).exists()


def test_an_in_process_run_that_stalls_keeps_its_lease_and_reads_orphaned(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The process outlives the run, as a notebook kernel does, and stops refreshing."""

    def stalls(ds: Dataset, request: Request, step_id: str, **_: object):
        return _finish_step(ds, request, step_id, state="stalled")

    monkeypatch.setattr(run_module, "execute_step", stalls)
    submitted = submit_request(tracked, chain_recipe()).request

    with pytest.raises(CoverageShortfall):
        _ = run_pipeline(tracked, chain_recipe(), request=submitted)

    lease = read_driver_lease(tracked.base_dir, submitted.request_id)
    assert lease is not None and lease.in_process
    rollup = request_rollup(tracked.base_dir, submitted)
    later = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1)
    assert request_verdict_of(tracked.base_dir, rollup) == "running"
    assert request_verdict_of(tracked.base_dir, rollup, now=later) == "orphaned"


def test_an_in_process_run_refuses_a_request_another_driver_holds(
    tracked: Dataset,
) -> None:
    submitted = submit_request(tracked, chain_recipe()).request
    held = write_driver_lease(tracked.base_dir, submitted.request_id, stale=False)

    with pytest.raises(DriverRunning):
        _ = run_pipeline(tracked, chain_recipe(), request=submitted)

    assert read_driver_lease(tracked.base_dir, submitted.request_id) == held
    assert request_rollup(tracked.base_dir, submitted).steps[0].status == ""


def test_a_cancelled_in_process_run_records_its_unended_steps_cancelled(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    def cancelled(
        ds: Dataset,
        request: Request,
        step_id: str,
        *,
        cancel_token: CancelToken,
        **_: object,
    ):
        cancel_token.cancel()
        raise Cancelled()

    monkeypatch.setattr(run_module, "execute_step", cancelled)
    submitted = submit_request(tracked, chain_recipe()).request

    with pytest.raises(Cancelled):
        _ = run_pipeline(tracked, chain_recipe(), request=submitted)

    rollup = request_rollup(tracked.base_dir, submitted)
    assert [attempt.status for attempt in rollup.steps] == ["cancelled", "cancelled"]
    assert not driver_lease_path(tracked.base_dir, submitted.request_id).exists()


def test_an_in_process_run_stops_before_a_step_a_cancel_recorded(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``mosaic cancel --request`` records the steps of an in-process run."""
    ran: list[str] = []
    cancels: list[RequestCancel] = []

    def then_cancelled(ds: Dataset, request: Request, step_id: str, **_: object):
        ran.append(step_id)
        outcome = _finish_step(ds, request, step_id)
        cancels.append(cancel_request(ds, request.request_id))
        return outcome

    monkeypatch.setattr(run_module, "execute_step", then_cancelled)
    submitted = submit_request(tracked, chain_recipe()).request

    with pytest.raises(Cancelled):
        _ = run_pipeline(tracked, chain_recipe(), request=submitted)

    assert ran == ["speed"]
    assert [(cancel.outcome, cancel.driver_pid) for cancel in cancels] == [
        ("recorded", 0)
    ]
    rollup = request_rollup(tracked.base_dir, submitted)
    assert [attempt.status for attempt in rollup.steps] == ["finished", "cancelled"]


def test_an_in_process_step_that_raises_unrecorded_leaves_no_step_named(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A step that raises before its run-log exists is not this process's to wait on.

    The process outlives the run, as a notebook kernel does. A lease left naming
    the step with this pid would read it as running for as long as the kernel
    lives, and the request would never read ``orphaned``.
    """

    def raises(ds: Dataset, request: Request, step_id: str, **_: object):
        raise RuntimeError("raised before the step's attempt began")

    monkeypatch.setattr(run_module, "execute_step", raises)
    submitted = submit_request(tracked, chain_recipe()).request

    with pytest.raises(RuntimeError, match="before the step's attempt began"):
        _ = run_pipeline(tracked, chain_recipe(), request=submitted)

    lease = read_driver_lease(tracked.base_dir, submitted.request_id)
    assert lease is not None
    assert (lease.step_id, lease.step_pid) == ("", 0)
    rollup = request_rollup(tracked.base_dir, submitted)
    later = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1)
    assert request_verdict_of(tracked.base_dir, rollup, now=later) == "orphaned"


def test_a_request_stopped_in_this_process_runs_again_at_once(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kept lease of an earlier call here does not refuse the next one.

    A notebook that meets a shortfall runs the same request again with
    ``allow_partial`` within the lease's grace.
    """
    states = iter(["stalled", "ran", "ran"])

    def steps(ds: Dataset, request: Request, step_id: str, **_: object):
        return _finish_step(ds, request, step_id, state=next(states))

    monkeypatch.setattr(run_module, "execute_step", steps)
    submitted = submit_request(tracked, chain_recipe()).request
    with pytest.raises(CoverageShortfall):
        _ = run_pipeline(tracked, chain_recipe(), request=submitted)
    assert read_driver_lease(tracked.base_dir, submitted.request_id) is not None

    done = run_pipeline(tracked, chain_recipe(), request=submitted, allow_partial=True)

    assert [outcome.step_id for outcome in done.outcomes] == ["speed", "templates"]
    assert not driver_lease_path(tracked.base_dir, submitted.request_id).exists()


# --- the supervised chain, as one request -----------------------------------------


def test_the_supervised_chain_runs_as_one_request(tmp_path: Path) -> None:
    """Convert, fit on the training groups and score on the test group, at once.

    One recipe and one request: the conversions, the features, templates and a
    scaler fitted on the training groups, labeled templates drawn from them, a
    classifier applied to the test group and its evaluation there. Step scopes
    make the split, and the labels reference pins the converted labels.
    """
    from mosaic.core.evaluation import EvaluationReport
    from mosaic.core.pipeline.fit_scope import read_fit_scope
    from mosaic.core.scope import Scope
    from tests.helpers import CONVERT_LABELS_STEP, CONVERT_TRACKS_STEP, calms21_dataset

    ds = calms21_dataset(
        tmp_path / "ds",
        {
            "train": {"a": {"s1": 300, "s2": 300, "s4": 300}},
            "test": {"a": {"s3": 300}},
        },
        seed=11,
        n_classes=3,
    )
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "supervised",
            "steps": [
                CONVERT_TRACKS_STEP,
                CONVERT_LABELS_STEP,
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                    "tracks": {"step": "tracks"},
                },
                {
                    "id": "templates",
                    "type": "feature",
                    "feature": "extract-templates",
                    "inputs": [{"step": "speed"}],
                    "params": {"n_templates": 50},
                },
                {
                    "id": "scaler",
                    "type": "feature",
                    "feature": "global-scaler",
                    "inputs": [{"step": "speed"}],
                    "params": {"templates": {"step": "templates"}},
                },
                {
                    "id": "labeled",
                    "type": "feature",
                    "feature": "extract-labeled-templates",
                    "inputs": [{"step": "scaler"}],
                    "params": {"labels": {"step": "labels"}, "n_total": 120},
                },
                {
                    "id": "classifier",
                    "type": "feature",
                    "feature": "xgboost",
                    "inputs": [{"step": "scaler"}],
                    "params": {
                        "templates": {"step": "labeled"},
                        "default_class": 0,
                        "n_estimators": 5,
                    },
                },
                {
                    "id": "evaluate",
                    "type": "feature",
                    "feature": "evaluate-labels",
                    "inputs": [{"step": "classifier"}],
                    "params": {"labels": {"step": "labels"}},
                },
            ],
        }
    )
    train, test = Scope(groups=["train"]), Scope(groups=["test"])
    scopes = {
        "templates": train,
        "labeled": train,
        "classifier": test,
        "evaluate": test,
    }

    planned = plan_pipeline(ds, recipe, step_scopes=scopes)
    run = run_pipeline(ds, recipe, step_scopes=scopes)

    assert planned.step("evaluate").run_id is None
    outcomes = {outcome.step_id: outcome for outcome in run.outcomes}
    assert [outcome.state for outcome in run.outcomes] == ["ran"] * 8
    assert (outcomes["evaluate"].covered, outcomes["evaluate"].target) == (1, 1)
    after = plan_pipeline(ds, recipe, step_scopes=scopes)
    assert after.is_complete
    for step_id in ("tracks", "labels", "speed"):
        assert outcomes[step_id].run_id == planned.step(step_id).run_id
    features = ds.get_root("features")
    classifier = after.step("classifier")
    fitted = read_fit_scope(features / classifier.storage_name / str(classifier.run_id))
    assert fitted is not None
    assert fitted.entries == (("train", "s1"), ("train", "s2"), ("train", "s4"))
    evaluate = after.step("evaluate")
    report = EvaluationReport.model_validate_json(
        (
            features / evaluate.storage_name / str(evaluate.run_id) / "metrics.json"
        ).read_text()
    )
    assert report.n_entries == 1
    assert report.n_scored > 0
