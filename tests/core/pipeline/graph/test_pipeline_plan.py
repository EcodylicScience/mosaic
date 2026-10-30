"""The load-bearing property: what a plan predicts is what execution records.

Everything after this phase is plumbing around a function already known to be
right, so this is where being right is established. The claim is narrow and
checkable: for every step of a graph, the ``run_id`` ``plan_pipeline`` resolves
before anything runs is the identifier the run then writes to disk -- across a
feature chain, across a params reference, across an op-to-feature edge, and on a
**cold** dataset where none of the upstream outputs exist yet.

The cold case is the one that used to be wrong, and it is wrong in a way that
reads as right: prediction routed through a manifest built over an index that
does not exist, which reports an *empty* scope, so a ``scope_dependent`` step
hashed ``[]`` where execution would hash the real entries. The identifier looked
ordinary and named a directory nothing would ever write.

The other half of the claim is the refusal. When fewer entries complete than were
intended, a ``scope_dependent`` step's identities genuinely **differ** -- a model
fitted on one sequence is not the model fitted on two -- and that is asserted here
rather than left as folklore.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from mosaic_media import MediaFacts, MediaProbeError
from typer.testing import CliRunner

import mosaic.tracking.trex.dataset_runs as trex_runs
from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    CoverageShort,
    DepsIncomplete,
    DepsIncompleteWire,
    LARGE_SCOPE_ENTRIES,
    IdentityUnresolved,
    Plan,
    PlanWire,
    Recipe,
    RecipeInvalid,
    RecipeRefused,
    build_step_feature,
    plan_pipeline,
    plan_wire,
    recipe_digest,
    requests_matching,
    run_pipeline,
    submit_request,
)
from mosaic.core.pipeline.error_kind import ClassifiedError
from mosaic.core.pipeline.preprocess import PreprocessParams, preprocess_identity
from mosaic.core.pipeline.preprocess_layout import media_variants_root
from mosaic.core.pipeline.run import run_feature
from mosaic.core.scope import Scope
from mosaic.core.pipeline.tracks_index import (
    read_tracks_index,
    variant_for_producer_run,
)
from mosaic.tracking.trex.params import TrexParams
from tests.helpers import (
    SPEED_RECIPE,
    FakeTrex,
    MediaClip,
    add_media_variant,
    add_tracks_variant,
    add_transcode_derivative,
    assert_tree_unchanged,
    clip_facts,
    finish_media_variant,
    install_fake_trex,
    make_dataset,
    scope_over,
    speed_recipe,
    tree_snapshot,
    write_driver_lease,
    write_media_index,
    write_sleap_model,
)

type Document = dict[str, object]

VARIANT = "convert-trex.0.2-1111111111"
"""The tracks recipe the fixture's tables answer to."""


# --- fixtures ------------------------------------------------------------------


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """A dataset with two schema-valid track tables and nothing computed.

    Cold in the sense that matters here: the tracks exist, so there is a scope,
    and no feature has run, so every upstream a downstream step names is absent.
    """
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return dataset


def execute(dataset: Dataset, plan: Plan) -> dict[str, str]:
    """Run every feature step of *plan* in order, returning what each was called.

    Each step is run from its own ``spec`` -- the entries, the tracks pin -- so
    what is compared is the identifier of the run the plan asked for, not of some
    other run with the same feature.
    """
    executed: dict[str, str] = {}
    for planned in plan.steps:
        feature = build_step_feature(planned.spec)
        result = run_feature(
            dataset,
            feature,
            scope=None if planned.spec.entries.is_unset else planned.spec.entries,
            tracks_run_id=planned.spec.tracks_run_id,
            track=False,
        )
        executed[planned.step_id] = result.run_id
    return executed


# --- the chain -----------------------------------------------------------------

FIVE_STEPS: Document = {
    "schema_version": 1,
    "name": "five steps",
    "steps": [
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "inputs": ["tracks"],
        },
        {
            "id": "nn",
            "type": "feature",
            "feature": "nearest-neighbor",
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
        {
            "id": "stack",
            "type": "feature",
            "feature": "temporal-stack",
            "inputs": [{"step": "scaler"}],
        },
    ],
}


def test_every_step_resolves_to_the_identifier_execution_records(
    tracked: Dataset,
) -> None:
    """The phase gate, over a five-step graph covering all three edge kinds.

    ``speed`` and ``nn`` read tracks; ``templates`` and ``scaler`` read a
    feature's output; ``scaler`` also carries a params reference, which is a
    different substitution with a different field type; and ``stack`` sits three
    deep, so its storage name nests two suffixes.
    """
    recipe = Recipe.model_validate(FIVE_STEPS)
    plan = plan_pipeline(tracked, recipe)

    assert not any(planned.run_id is None for planned in plan.steps)
    assert execute(tracked, plan) == plan.run_ids


def test_the_plan_lists_the_open_and_orphaned_requests_of_its_recipe(
    tracked: Dataset, tmp_path: Path
) -> None:
    running = submit_request(tracked, speed_recipe()).request
    orphaned = submit_request(tracked, speed_recipe()).request
    _ = write_driver_lease(tracked.base_dir, orphaned.request_id, stale=True)
    finished = submit_request(tracked, speed_recipe()).request
    _ = run_pipeline(tracked, speed_recipe(), request=finished)
    _ = submit_request(tracked, speed_recipe(), scope=Scope(entries=[("", "seq_b")]))
    recipe = tmp_path / "speed.json"
    _ = recipe.write_text(json.dumps(SPEED_RECIPE))

    listed = requests_matching(
        tracked.base_dir, recipe_digest(speed_recipe()), Scope(entries=[("", "seq_a")])
    )
    result = CliRunner().invoke(
        app,
        [
            "pipeline",
            "plan",
            "-m",
            str(tracked.manifest_path),
            "-r",
            f"@{recipe}",
            "--entry",
            "seq_a",
            "--json",
        ],
    )

    assert [(summary.request_id, summary.verdict) for summary in listed] == [
        (orphaned.request_id, "orphaned"),
        (running.request_id, "running"),
    ]
    assert result.exit_code == 0, result.output
    wire = PlanWire.model_validate_json(result.stdout)
    assert wire.open_requests == listed


def test_a_cold_scope_dependent_step_predicts_its_scope(tracked: Dataset) -> None:
    """The case the previous resolver got wrong, asserted on its own.

    ``extract-templates`` is ``scope_dependent``: the set of entries it was
    fitted over is part of what it is called. Its input has not been computed
    when the plan is made, so a resolver reading the upstream index would find
    nothing and hash an empty scope.
    """
    recipe = Recipe.model_validate(FIVE_STEPS)
    plan = plan_pipeline(tracked, recipe)
    templates = plan.step("templates")

    assert build_step_feature(templates.spec).scope_dependent
    assert templates.spec.entries == Scope(entries=[("", "seq_a"), ("", "seq_b")])
    assert execute(tracked, plan)["templates"] == templates.run_id


def test_a_feature_chain_resolves_with_nothing_on_disk(tmp_path: Path) -> None:
    """A four-step chain's identities come from params and the step above alone.

    Two datasets holding nothing at all -- no tracks, no media, no runs -- give
    every step of the chain the same identifier, which is what "reads nothing
    from disk on those edges" means as a measurement rather than as a claim.
    """
    chain: Document = {
        "steps": [
            {
                "id": "a",
                "type": "feature",
                "feature": "speed-angvel",
                "inputs": ["tracks"],
            },
            {
                "id": "b",
                "type": "feature",
                "feature": "temporal-stack",
                "inputs": [{"step": "a"}],
            },
            {
                "id": "c",
                "type": "feature",
                "feature": "global-scaler",
                "inputs": [{"step": "b"}],
                "params": {
                    "templates": {"feature": "extract-templates", "run_id": None}
                },
            },
            {
                "id": "d",
                "type": "feature",
                "feature": "temporal-stack",
                "inputs": [{"step": "c"}],
            },
        ]
    }
    recipe = Recipe.model_validate(chain)
    first = plan_pipeline(make_dataset(tmp_path / "one"), recipe)
    second = plan_pipeline(make_dataset(tmp_path / "two"), recipe)

    assert first.run_ids == second.run_ids
    assert len(set(first.run_ids.values())) == 4, "four steps, four identities"
    # Four nestings: the three steps above it, and the tracks the first reads.
    assert first.step("d").storage_name.count("__from__") == 4


def test_two_datasets_give_a_scope_free_step_one_identity(tmp_path: Path) -> None:
    """The cross-dataset property, which is what makes a recipe portable.

    A scope-free step reads the same identifier on two datasets holding the same
    tracks recipe, so the same analysis lands under the same directory name on
    both and they are comparable on disk with no shared database.
    """
    recipe = Recipe.model_validate(
        {
            "steps": [
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                }
            ]
        }
    )
    here = make_dataset(tmp_path / "here")
    add_tracks_variant(here, VARIANT, "seq_a", std_format="mosaic_v1")
    there = make_dataset(tmp_path / "there")
    add_tracks_variant(there, VARIANT, "other_seq", "third_seq", std_format="mosaic_v1")

    assert (
        plan_pipeline(here, recipe).step("speed").run_id
        == plan_pipeline(there, recipe).step("speed").run_id
    )


def test_fewer_entries_than_intended_gives_a_different_identity(
    tracked: Dataset,
) -> None:
    """Pinned, so the refusal downstream of it is grounded rather than folklore.

    A ``scope_dependent`` step fitted over one sequence *is* a different artifact
    from the same step fitted over two, and mosaic says so by giving it a
    different name. That is why a shortfall is a decision for a person rather
    than something a runner quietly proceeds through.
    """
    recipe = Recipe.model_validate(FIVE_STEPS)
    plan = plan_pipeline(tracked, recipe)
    speed = plan.step("speed")
    _ = run_feature(tracked, build_step_feature(speed.spec), track=False)

    templates = build_step_feature(plan.step("templates").spec)
    partial = run_feature(
        tracked, templates, scope=Scope(entries=[("", "seq_a")]), track=False
    )

    assert partial.run_id != plan.step("templates").run_id


# --- an op edge ----------------------------------------------------------------


@pytest.fixture
def fake_trex(monkeypatch: pytest.MonkeyPatch) -> FakeTrex:
    """TREx's two phases, replaced by fakes that write what TREx writes.

    The tool never runs, and its output is convertible. The bridge that publishes
    into ``tracks/`` is exercised rather than stubbed.
    """
    return install_fake_trex(monkeypatch)


TREX: Document = {
    "id": "trex",
    "type": "op",
    "kind": "trex",
    "params": {"track_max_individuals": 2},
}

SPEED_BELOW_TREX: Document = {
    "id": "speed",
    "type": "feature",
    "feature": "speed-angvel",
    "inputs": ["tracks"],
    "tracks": {"step": "trex"},
}

TREX_TO_SPEED: Document = {"steps": [TREX, SPEED_BELOW_TREX]}


@pytest.mark.tracker
def test_a_tracker_step_resolves_the_variant_its_run_writes(
    tmp_path: Path, fake_trex: FakeTrex
) -> None:
    """The op-to-feature edge, which resolves before the op has run.

    What a feature reads from a tracker is the ``tracks/`` **variant**, a
    different identifier from the op's run id. The plan mints it by calling the
    real payload builder, so it stays right whatever that payload later becomes
    -- and the run then writes tables under exactly that name.
    """
    dataset = make_dataset(tmp_path / "media")
    write_media_index(dataset, ["vid1"], uids={"vid1": "uuid-1"})
    plan = plan_pipeline(dataset, Recipe.model_validate(TREX_TO_SPEED))
    predicted = plan.step("trex")

    produced = trex_runs.run_trex(
        dataset,
        TrexParams(track_max_individuals=2),
        scope_over(("", "vid1")),
    )

    assert produced == predicted.run_id
    index = read_tracks_index(dataset)
    assert variant_for_producer_run(index, produced) == predicted.tracks_variant
    assert plan.step("speed").spec.tracks_run_id == predicted.tracks_variant


@pytest.mark.tracker
def test_a_feature_below_a_tracker_resolves_to_what_it_then_records(
    tmp_path: Path, fake_trex: FakeTrex
) -> None:
    """The whole edge, end to end: predicted before, recorded after."""
    dataset = make_dataset(tmp_path / "media")
    write_media_index(dataset, ["vid1"], uids={"vid1": "uuid-1"})
    plan = plan_pipeline(dataset, Recipe.model_validate(TREX_TO_SPEED))
    predicted = plan.step("speed").run_id

    _ = trex_runs.run_trex(
        dataset,
        TrexParams(track_max_individuals=2),
        scope_over(("", "vid1")),
    )
    speed = plan.step("speed")
    result = run_feature(
        dataset,
        build_step_feature(speed.spec),
        tracks_run_id=speed.spec.tracks_run_id,
        track=False,
    )

    assert result.run_id == predicted


# --- a media variant -------------------------------------------------------------


@pytest.fixture
def with_media(tmp_path: Path) -> Dataset:
    """Return a dataset whose only content is two indexed 640x480 videos."""
    dataset = make_dataset(tmp_path / "media")
    write_media_index(dataset, ["vid1", "vid2"], uids={"vid1": "u1", "vid2": "u2"})
    return dataset


@pytest.fixture
def sleap_model(tmp_path: Path) -> Path:
    """Write a model directory to disk, so that a SLEAP step's identity can be read."""
    return write_sleap_model(tmp_path / "sleap-model")


def _crop(width: int = 320) -> Document:
    return {"step": "crop", "x": 0, "y": 0, "width": width, "height": 240}


def _variant_then_sleap(model: Path, width: int = 320) -> Recipe:
    """Return a recipe of a variant and a tracker that reads it via ``media``."""
    return Recipe.model_validate(
        {
            "steps": [
                {
                    "id": "pre",
                    "type": "op",
                    "kind": "preprocess",
                    "params": {"steps": [_crop(width)]},
                },
                {
                    "id": "sleap",
                    "type": "op",
                    "kind": "sleap",
                    "params": {"model_paths": [str(model)], "media": {"step": "pre"}},
                },
            ]
        }
    )


def _variant_id(steps: list[Document], media: str = "") -> str:
    """Return the run id of a variant of *steps* over *media*, minted by the op."""
    params = PreprocessParams.model_validate({"steps": steps, "media": media})
    return preprocess_identity(params).run_id


def test_a_tracker_reading_a_variant_resolves_both_identities_before_anything_runs(
    with_media: Dataset, sleap_model: Path
) -> None:
    """The variant's identity is its params, and the tracker's names the variant."""
    plan = plan_pipeline(with_media, _variant_then_sleap(sleap_model))
    pre = plan.step("pre")
    sleap = plan.step("sleap")

    assert pre.run_id == _variant_id([_crop()])
    assert sleap.run_id is not None
    assert sleap.spec.params["media"] == pre.run_id
    assert sleap.parents == ("pre",)
    assert not media_variants_root(with_media).exists(), "planning wrote nothing"


def test_a_changed_crop_moves_the_variant_and_the_tracker_below_it(
    with_media: Dataset, sleap_model: Path
) -> None:
    before = plan_pipeline(with_media, _variant_then_sleap(sleap_model, width=320))
    after = plan_pipeline(with_media, _variant_then_sleap(sleap_model, width=322))

    assert after.step("pre").run_id != before.step("pre").run_id
    assert after.step("sleap").run_id != before.step("sleap").run_id


_BOTH_VIDEOS = Scope(entries=[("", "vid1"), ("", "vid2")])
"""The scope is named instead of left to the planner. The planner plans a graph
without a tracks writer over the tracks universe, and these datasets do not have
tracks yet."""


def test_a_variant_after_a_transcode_plans(with_media: Dataset) -> None:
    """The transcode is ordering only, and the variant's identity is its params."""
    plan = plan_pipeline(
        with_media,
        scope=_BOTH_VIDEOS,
        recipe=Recipe.model_validate(
            {
                "steps": [
                    {
                        "id": "transcode",
                        "type": "op",
                        "kind": "transcode",
                        "params": {"target": "analysis"},
                    },
                    {
                        "id": "pre",
                        "type": "op",
                        "kind": "preprocess",
                        "params": {"steps": [_crop()]},
                        "after": ["transcode"],
                    },
                ]
            }
        ),
    )

    assert plan.step("transcode").run_id is not None
    assert plan.step("pre").run_id == _variant_id([_crop()])
    assert plan.step("pre").parents == ("transcode",)


def test_a_variant_after_another_variant_names_it(with_media: Dataset) -> None:
    """A chained variant's identity includes its upstream's, before either exists."""
    plan = plan_pipeline(
        with_media,
        scope=_BOTH_VIDEOS,
        recipe=Recipe.model_validate(
            {
                "steps": [
                    {
                        "id": "pre",
                        "type": "op",
                        "kind": "preprocess",
                        "params": {"steps": [_crop()]},
                    },
                    {
                        "id": "gray",
                        "type": "op",
                        "kind": "preprocess",
                        "params": {
                            "steps": [{"step": "grayscale"}],
                            "media": {"step": "pre"},
                        },
                    },
                ]
            }
        ),
    )
    upstream = plan.step("pre").run_id
    assert upstream is not None

    assert plan.step("gray").spec.params["media"] == upstream
    assert plan.step("gray").run_id == _variant_id(
        [{"step": "grayscale"}], media=upstream
    )


def test_a_variant_over_some_entries_reads_complete_for_them(
    with_media: Dataset, sleap_model: Path
) -> None:
    """An entry is covered when its consumed camera has a row and a file."""
    recipe = _variant_then_sleap(sleap_model)
    variant = plan_pipeline(with_media, recipe).step("pre").run_id
    assert variant is not None
    _ = add_media_variant(with_media, variant, "vid1")
    finish_media_variant(with_media, variant)

    whole = plan_pipeline(with_media, recipe)
    narrowed = plan_pipeline(with_media, recipe, scope=Scope(entries=[("", "vid1")]))

    assert whole.step("pre").status == "partial"
    assert whole.step("pre").reason == CoverageShort(
        covered=1, target=2, missing=(("", "vid2"),)
    )
    assert whole.step("sleap").reason == DepsIncomplete(blocking=("pre",))
    assert narrowed.step("pre").status == "complete"
    assert narrowed.step("pre").coverage.covered == frozenset({("", "vid1")})
    assert narrowed.step("sleap").reason is None


# --- what a plan says about work that is not simply running ---------------------


def test_a_step_below_an_incomplete_one_says_what_blocks_it(tracked: Dataset) -> None:
    recipe = Recipe.model_validate(FIVE_STEPS)
    plan = plan_pipeline(tracked, recipe)

    assert plan.step("speed").reason is None, "nothing above it, nothing computed"
    assert plan.step("templates").reason == DepsIncomplete(blocking=("speed",))


def test_a_completed_graph_reads_complete_and_pends_nothing(tracked: Dataset) -> None:
    """Re-planning after a run is what a cache hit looks like from outside."""
    recipe = Recipe.model_validate(FIVE_STEPS)
    _ = execute(tracked, plan_pipeline(tracked, recipe))

    again = plan_pipeline(tracked, recipe)

    assert again.is_complete
    assert again.pending == ()
    assert all(planned.reason is None for planned in again.steps)


def test_a_partly_covered_step_reports_what_is_covered(tracked: Dataset) -> None:
    """``partial`` is its own answer: 1 of 2 is not "nothing has run"."""
    recipe = Recipe.model_validate(
        {
            "steps": [
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                }
            ]
        }
    )
    speed = plan_pipeline(tracked, recipe).step("speed")
    _ = run_feature(
        tracked,
        build_step_feature(speed.spec),
        scope=Scope(entries=[("", "seq_a")]),
        track=False,
    )

    again = plan_pipeline(tracked, recipe).step("speed")

    assert again.status == "partial"
    assert again.reason == CoverageShort(covered=1, target=2, missing=(("", "seq_b"),))
    assert again.spec.entries == Scope(entries=[("", "seq_b")]), (
        "ask only for what is missing"
    )


def test_a_scope_dependent_step_is_asked_for_all_of_its_scope(
    tracked: Dataset,
) -> None:
    """Its identity *is* its scope, so a remainder would be a different run.

    The narrowing that is right for a scope-free step is exactly wrong here: a
    fit over what remains, written under the name of a fit over everything, is
    the one outcome the identity scheme exists to prevent.
    """
    recipe = Recipe.model_validate(FIVE_STEPS)
    plan = plan_pipeline(tracked, recipe)
    _ = run_feature(
        tracked,
        build_step_feature(plan.step("speed").spec),
        track=False,
    )
    templates = plan.step("templates")
    _ = run_feature(
        tracked,
        build_step_feature(templates.spec),
        scope=Scope(entries=[("", "seq_a")]),
        track=False,
    )

    again = plan_pipeline(tracked, recipe).step("templates")

    assert again.spec.entries == Scope(entries=[("", "seq_a"), ("", "seq_b")])


# --- the plan as JSON -----------------------------------------------------------


def test_the_plan_round_trips_through_json(tracked: Dataset) -> None:
    """An editor previews a plan as its wire model, which parses back unchanged."""
    wire = plan_wire(plan_pipeline(tracked, Recipe.model_validate(FIVE_STEPS)))

    assert PlanWire.model_validate_json(wire.model_dump_json()) == wire
    assert wire.scope_size == 2
    steps = {planned.step: planned for planned in wire.steps}
    speed = steps["speed"]
    assert speed.coverage.target == 2
    assert speed.coverage.covered == 0
    assert speed.coverage.missing_sample == (("", "seq_a"), ("", "seq_b"))
    assert not speed.coverage.has_more
    assert speed.asked == 2
    assert steps["templates"].reason == DepsIncompleteWire(blocking=("speed",))


def test_plan_wire_bounds_the_missing_sample(tracked: Dataset) -> None:
    plan = plan_pipeline(tracked, Recipe.model_validate(FIVE_STEPS))

    bounded = plan_wire(plan, missing_sample=1)

    speed = {planned.step: planned for planned in bounded.steps}["speed"]
    assert speed.coverage.missing_sample == (("", "seq_a"),)
    assert speed.coverage.has_more
    assert speed.coverage.target == 2
    assert bounded.fingerprint == plan_wire(plan).fingerprint


# --- cautions ------------------------------------------------------------------


def test_a_plan_over_the_whole_dataset_is_cautioned(tracked: Dataset) -> None:
    plan = plan_pipeline(tracked, speed_recipe())

    wire = plan_wire(plan)

    assert plan.covers_dataset
    assert [(caution.step, caution.where) for caution in wire.cautions] == [
        ("", "scope")
    ]
    assert "all 2 entries" in wire.cautions[0].message
    assert wire.problems == ()


def test_a_plan_over_named_entries_is_not_cautioned(tracked: Dataset) -> None:
    plan = plan_pipeline(
        tracked, speed_recipe(), scope=Scope(entries=[("", "seq_a"), ("", "seq_b")])
    )

    assert not plan.covers_dataset
    assert plan_wire(plan).cautions == ()


def test_a_step_asked_for_more_than_large_scope_entries_is_cautioned(
    tmp_path: Path,
) -> None:
    sequences = [f"seq_{index:03d}" for index in range(LARGE_SCOPE_ENTRIES + 1)]
    large = make_dataset(tmp_path / "large")
    add_tracks_variant(large, VARIANT, *sequences, n_rows=2, std_format="mosaic_v1")

    plan = plan_pipeline(
        large,
        speed_recipe(),
        scope=Scope(entries=[("", sequence) for sequence in sequences]),
    )

    assert [str(caution) for caution in plan_wire(plan).cautions] == [
        f"step 'speed' scope: asked for {LARGE_SCOPE_ENTRIES + 1} entries, more "
        f"than {LARGE_SCOPE_ENTRIES}"
    ]


def test_the_cli_plan_notes_each_caution_below_its_table(
    tracked: Dataset, tmp_path: Path
) -> None:
    recipe = tmp_path / "speed.json"
    _ = recipe.write_text(json.dumps(SPEED_RECIPE))

    result = CliRunner().invoke(
        app, ["pipeline", "plan", "-m", str(tracked.manifest_path), "-r", f"@{recipe}"]
    )

    assert result.exit_code == 0, result.output
    assert "caution: scope: the plan covers all 2 entries" in result.stderr


# --- an identity nothing can resolve yet ----------------------------------------

DEFERRED_TRAINING: Document = {
    "steps": [
        {
            "id": "train",
            "type": "op",
            "kind": "train-pose",
            "params": {"data": "annotations/pose.yaml"},
        },
        {
            "id": "infer",
            "type": "op",
            "kind": "infer-pose",
            "params": {"model": {"step": "train"}},
        },
    ]
}


def test_an_op_whose_data_is_not_written_yet_says_so(tmp_path: Path) -> None:
    """A model's identity covers what it was trained on, which has to be read.

    Saying so is honest where guessing is not: an invented identifier is a wrong
    answer in a preview, and it costs nothing at execution, where the directory
    does exist and the step resolves its own identity.
    """
    dataset = make_dataset(tmp_path / "ds")
    write_media_index(dataset, ["vid1"], uids={"vid1": "uuid-1"})

    plan = plan_pipeline(dataset, Recipe.model_validate(DEFERRED_TRAINING))

    train = plan.step("train")
    assert train.run_id is None
    assert isinstance(train.reason, IdentityUnresolved)
    assert "not on disk yet" in train.reason.because


def test_an_unresolved_identity_cascades_to_everything_below_it(
    tmp_path: Path,
) -> None:
    """And it names the step that could not resolve, not the one reporting."""
    dataset = make_dataset(tmp_path / "ds")
    write_media_index(dataset, ["vid1"], uids={"vid1": "uuid-1"})

    infer = plan_pipeline(dataset, Recipe.model_validate(DEFERRED_TRAINING)).step(
        "infer"
    )

    assert infer.run_id is None
    assert isinstance(infer.reason, IdentityUnresolved)
    assert infer.reason.step == "train"
    assert infer.spec.op_kind == "infer-pose", "what to run is still known"


def _prepare_then_train_embedding(device: str = "auto") -> Recipe:
    """A recipe that cuts embedding labels and trains on them."""
    return Recipe.model_validate(
        {
            "steps": [
                {
                    "id": "prepare",
                    "type": "op",
                    "kind": "prepare-embedding-data",
                    "params": {
                        "sleap_run_id": "sleap.1.6-abcdef0123",
                        "train_entries": [{"sequence": "vid1"}],
                        "val_entries": [{"sequence": "vid2"}],
                    },
                },
                {
                    "id": "train",
                    "type": "op",
                    "kind": "train-sleap",
                    "params": {
                        "labels": {"step": "prepare"},
                        "head": "embedding",
                        "device": device,
                    },
                },
            ]
        }
    )


def test_a_training_step_on_labels_not_written_yet_defers_its_label_checks(
    tmp_path: Path,
) -> None:
    """The preparation's row and labels do not exist at plan time.

    The checks that read them wait, and the step's identity is deferred, as it
    is for any training step whose data is not on disk.
    """
    dataset = make_dataset(tmp_path / "ds")
    write_media_index(dataset, ["vid1", "vid2"], uids={"vid1": "u1", "vid2": "u2"})

    plan = plan_pipeline(dataset, _prepare_then_train_embedding())

    assert plan.step("prepare").run_id is not None
    train = plan.step("train")
    assert train.run_id is None
    assert isinstance(train.reason, IdentityUnresolved)
    assert "not on disk yet" in train.reason.because


def test_a_training_step_whose_settings_cannot_train_is_invalid_at_plan(
    tmp_path: Path,
) -> None:
    """A refusal that reads no data is raised before anything is enqueued."""
    dataset = make_dataset(tmp_path / "ds")
    write_media_index(dataset, ["vid1", "vid2"], uids={"vid1": "u1", "vid2": "u2"})

    with pytest.raises(ClassifiedError, match="one device") as refused:
        _ = plan_pipeline(dataset, _prepare_then_train_embedding(device="0,1"))
    assert refused.value.error_kind == "invalid"


# --- the scope a graph is planned over ------------------------------------------


def test_a_narrowing_governs_the_scope_and_moves_a_scope_dependent_identity(
    tracked: Dataset,
) -> None:
    """An explicit narrowing is the submission speaking, and it is not widened."""
    recipe = Recipe.model_validate(FIVE_STEPS)
    whole = plan_pipeline(tracked, recipe)
    narrowed = plan_pipeline(tracked, recipe, scope=Scope(entries=[("", "seq_a")]))

    assert narrowed.scope == frozenset({("", "seq_a")})
    assert narrowed.step("speed").run_id == whole.step("speed").run_id
    assert narrowed.step("templates").run_id != whole.step("templates").run_id


def test_a_graph_that_makes_its_own_tracks_is_planned_over_its_media(
    tmp_path: Path,
) -> None:
    """There are no tracks yet, and the entries are still knowable.

    A tracker turns videos into tracks one entry at a time, and the videos are on
    disk before anything is planned -- so the scope is exact rather than guessed.
    """
    dataset = make_dataset(tmp_path / "media")
    write_media_index(dataset, ["vid1", "vid2"], uids={"vid1": "u1", "vid2": "u2"})

    plan = plan_pipeline(dataset, Recipe.model_validate(TREX_TO_SPEED))

    assert plan.scope == frozenset({("", "vid1"), ("", "vid2")})


TRANSCODE_FOR_ANALYSIS: Document = {
    "id": "transcode",
    "type": "op",
    "kind": "transcode",
    "params": {"target": "analysis"},
}

TRANSCODE_THEN_TREX: Document = {
    "steps": [
        TRANSCODE_FOR_ANALYSIS,
        {**TREX, "after": ["transcode"]},
        SPEED_BELOW_TREX,
    ]
}


def _one_clip(tmp_path: Path, rotation: int) -> Dataset:
    """Return a dataset whose one clip, ``vid1``, is turned by *rotation* degrees."""
    dataset = make_dataset(tmp_path / "media")
    write_media_index(
        dataset, [MediaClip(sequence="vid1", video_uuid="u1", rotation=rotation)]
    )
    return dataset


def _awaiting_transcode(tmp_path: Path) -> Dataset:
    """Return a dataset whose one clip, ``vid1``, awaits an analysis transcode.

    A rotated clip's verdict requires one, and the dataset lacks its derivative.
    """
    return _one_clip(tmp_path, 90)


def _upright(tmp_path: Path) -> Dataset:
    """Return a dataset whose one clip, ``vid1``, reads without a transcode."""
    return _one_clip(tmp_path, 0)


def _transcoded(tmp_path: Path) -> Dataset:
    """Return a dataset whose rotated clip, ``vid1``, has its analysis derivative."""

    def encode(_source: Path, derivative: Path) -> MediaFacts:
        _ = derivative.write_bytes(b"derivative")
        return clip_facts(video_uuid="u1-analysis")

    dataset = _awaiting_transcode(tmp_path)
    _ = add_transcode_derivative(dataset, "vid1", target="analysis", encode=encode)
    return dataset


@pytest.mark.parametrize(
    "scope",
    [None, Scope(sequences=["vid1"])],
    ids=["whole-dataset", "named-sequence"],
)
def test_a_graph_that_transcodes_first_is_planned_before_the_derivative_exists(
    tmp_path: Path, scope: Scope | None
) -> None:
    """An entry whose media cannot be read yet is still in the scope.

    A rotated clip's verdict requires an analysis transcode, and the derivative
    that the tracker reads is what the graph's first step makes. The plan lists
    the entry without routing its media, and the tracker routes it when it runs,
    after the transcode.
    """
    dataset = _awaiting_transcode(tmp_path)
    with pytest.raises(MediaProbeError, match="transcode it first"):
        _ = dataset.resolve_media("", "vid1")

    plan = plan_pipeline(
        dataset, Recipe.model_validate(TRANSCODE_THEN_TREX), scope=scope
    )

    assert plan.scope == frozenset({("", "vid1")})
    assert plan.step("transcode").spec.entries == Scope(entries=[("", "vid1")])
    assert plan.step("trex").run_id is not None


def test_a_tracker_over_media_awaiting_a_transcode_is_refused_at_its_after(
    tmp_path: Path,
) -> None:
    """The tracker reads the analysis derivative, which no step above it makes.

    The problem names the entry and the step to add, and planning writes nothing.
    """
    dataset = _awaiting_transcode(tmp_path)
    before = tree_snapshot(Path(dataset.base_dir))

    with pytest.raises(RecipeRefused) as raised:
        _ = plan_pipeline(dataset, Recipe.model_validate(TREX_TO_SPEED))

    (problem,) = raised.value.problems
    assert (problem.step, problem.where) == ("trex", "after")
    assert "1 entry requires an analysis transcode" in problem.message
    assert "('', 'vid1')" in problem.message
    assert json.dumps(TRANSCODE_FOR_ANALYSIS) in problem.message
    assert '"after": ["transcode"] on this step' in problem.message
    assert_tree_unchanged(Path(dataset.base_dir), before)


def test_the_problem_names_a_few_entries_and_counts_the_rest(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "media")
    write_media_index(
        dataset,
        [
            MediaClip(sequence=f"vid{n}", video_uuid=f"u{n}", rotation=90)
            for n in range(7)
        ],
    )

    plan = plan_pipeline(dataset, Recipe.model_validate(TREX_TO_SPEED), refuse=False)

    (problem,) = plan.problems
    assert "7 entries require an analysis transcode" in problem.message
    assert "('', 'vid4') and 2 more" in problem.message
    assert "vid5" not in problem.message


@pytest.mark.parametrize(
    "steps",
    [
        [TRANSCODE_FOR_ANALYSIS, TREX],
        [
            {**TRANSCODE_FOR_ANALYSIS, "params": {"target": "playback"}},
            {**TREX, "after": ["transcode"]},
        ],
    ],
    ids=["analysis-transcode-unordered", "playback-transcode-upstream"],
)
def test_a_transcode_that_does_not_make_the_derivative_first_leaves_the_problem(
    tmp_path: Path, steps: list[Document]
) -> None:
    """Only an analysis transcode upstream of the tracker makes its derivative first."""
    recipe = Recipe.model_validate({"steps": steps})

    plan = plan_pipeline(_awaiting_transcode(tmp_path), recipe, refuse=False)

    assert [(problem.step, problem.where) for problem in plan.problems] == [
        ("trex", "after")
    ]


def test_an_unordered_analysis_transcode_is_named_for_after(tmp_path: Path) -> None:
    """The fix orders the transcode the recipe has, rather than adding a second."""
    recipe = Recipe.model_validate(
        {"steps": [{**TRANSCODE_FOR_ANALYSIS, "id": "make_analysis"}, TREX]}
    )

    plan = plan_pipeline(_awaiting_transcode(tmp_path), recipe, refuse=False)

    (problem,) = plan.problems
    assert 'Add "after": ["make_analysis"] on this step' in problem.message
    assert "Add the step" not in problem.message


@pytest.mark.parametrize(
    "readable", [_upright, _transcoded], ids=["upright", "transcoded"]
)
def test_media_the_tracker_can_read_raises_no_problem(
    tmp_path: Path, readable: Callable[[Path], Dataset]
) -> None:
    plan = plan_pipeline(
        readable(tmp_path), Recipe.model_validate(TREX_TO_SPEED), refuse=False
    )

    assert plan.problems == ()


def test_a_tracker_reading_a_media_variant_is_not_checked(tmp_path: Path) -> None:
    """The tracker reads the variant's file in place of the analysis derivative."""
    recipe = Recipe.model_validate(
        {
            "steps": [
                {
                    "id": "pre",
                    "type": "op",
                    "kind": "preprocess",
                    "params": {"steps": [_crop()]},
                },
                {
                    **TREX,
                    "params": {"track_max_individuals": 2, "media": {"step": "pre"}},
                },
            ]
        }
    )

    plan = plan_pipeline(_awaiting_transcode(tmp_path), recipe, refuse=False)

    assert plan.problems == ()


def test_the_cli_plan_lists_the_media_awaiting_a_transcode(tmp_path: Path) -> None:
    dataset = _awaiting_transcode(tmp_path)
    recipe_path = tmp_path / "recipe.json"
    _ = recipe_path.write_text(json.dumps(TREX_TO_SPEED))

    result = CliRunner().invoke(
        app,
        [
            "pipeline",
            "plan",
            "--recipe",
            f"@{recipe_path}",
            "--manifest",
            str(dataset.manifest_path),
            "--json",
        ],
    )

    assert result.exit_code == 1, result.output
    wire = PlanWire.model_validate_json(result.stdout)
    assert [(problem.step, problem.where) for problem in wire.problems] == [
        ("trex", "after")
    ]


def test_a_dataset_with_neither_tracks_nor_media_plans_over_nothing(
    tmp_path: Path,
) -> None:
    """Empty is an answer. Refusing would take down the plan that explains why."""
    plan = plan_pipeline(
        make_dataset(tmp_path / "bare"), Recipe.model_validate(TREX_TO_SPEED)
    )

    assert plan.scope == frozenset()
    assert plan.step("trex").run_id is not None, "a tracker's identity is its settings"


def test_the_live_pipeline_and_the_planner_agree(tracked: Dataset) -> None:
    """One answer to what a step will be called, held by a test rather than a rule.

    The notebooks drive the live ``Pipeline``, so a second resolver that drifted
    would be the answer users meet first. Both go through the same identity site
    now, and this is what notices if one of them stops.
    """
    from mosaic.behavior.feature_library import FEATURES
    from mosaic.core.pipeline.pipeline import FeatureStep, Pipeline

    live = Pipeline()
    _ = live.add(FeatureStep("speed", FEATURES["SpeedAngvel"], None))
    _ = live.add(
        FeatureStep(
            "templates",
            FEATURES["ExtractTemplates"],
            {"n_templates": 4},
            input_names=["speed"],
        )
    )
    graph = plan_pipeline(
        tracked,
        Recipe.model_validate(
            {
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
                ]
            }
        ),
    )

    resolved = live._resolve_step_cache(tracked)  # pyright: ignore[reportPrivateUsage]

    assert [info["expected_run_id"] for info in resolved] == [
        graph.step("speed").run_id,
        graph.step("templates").run_id,
    ]


# --- planning writes nothing ----------------------------------------------------


def test_planning_leaves_the_dataset_untouched(tracked: Dataset) -> None:
    """It is a read, and a caller previewing a graph must be able to trust that."""

    root = Path(tracked.base_dir)
    before = tree_snapshot(root)
    _ = plan_pipeline(tracked, Recipe.model_validate(FIVE_STEPS))

    assert_tree_unchanged(root, before)


def test_an_invalid_recipe_is_refused_before_the_dataset_is_read(
    tmp_path: Path,
) -> None:
    """A malformed graph is refused as a document rather than half-planned.

    The dataset here holds nothing at all, so the only thing that can refuse is
    validation -- and the message names the slug rather than an empty scope.
    """
    with pytest.raises(RecipeInvalid) as raised:
        _ = plan_pipeline(
            make_dataset(tmp_path / "bare"),
            Recipe.model_validate(
                {"steps": [{"id": "a", "type": "feature", "feature": "no-such-thing"}]}
            ),
        )

    assert "no-such-thing" in str(raised.value)


def test_two_inputs_resolving_to_one_run_are_refused_before_planning(
    tracked: Dataset,
) -> None:
    """Planning gives the two speed steps one run id, which one step reads once."""

    def speed(step_id: str) -> Document:
        return {"id": step_id, "type": "feature", "feature": "speed-angvel"}

    recipe = Recipe.model_validate(
        {
            "steps": [
                speed("speed_a"),
                speed("speed_b"),
                {
                    "id": "aggregate",
                    "type": "feature",
                    "feature": "frame-aggregate",
                    "inputs": [{"step": "speed_a"}, {"step": "speed_b"}],
                    "params": {"column": "speed"},
                },
            ]
        }
    )

    with pytest.raises(RecipeInvalid) as raised:
        _ = plan_pipeline(tracked, recipe)

    (problem,) = raised.value.problems
    assert (problem.step, problem.where) == ("aggregate", "inputs")
    assert "They resolve to one run" in problem.message


def test_inputs_that_resolve_to_one_run_only_here_are_a_dataset_problem(
    tracked: Dataset,
) -> None:
    """A frame filter and the time filter equal to it validate as two runs.

    At 30 frames per second, frame 30 and one second name the same frames, so
    the plan resolves both steps to one run id and refuses the step reading both.
    """
    tracked.meta["fps_default"] = 30.0
    recipe = Recipe.model_validate(
        {
            "steps": [
                {
                    "id": "by_frame",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "run": {"filter_start_frame": 30},
                },
                {
                    "id": "by_time",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "run": {"filter_start_time": 1.0},
                },
                {
                    "id": "aggregate",
                    "type": "feature",
                    "feature": "frame-aggregate",
                    "inputs": [{"step": "by_frame"}, {"step": "by_time"}],
                    "params": {"column": "speed"},
                },
            ]
        }
    )

    with pytest.raises(RecipeRefused) as raised:
        _ = plan_pipeline(tracked, recipe)

    (problem,) = raised.value.problems
    assert (problem.step, problem.where) == ("aggregate", "inputs")
    assert "'by_frame' and 'by_time' run speed-angvel" in problem.message
    assert "name one run on this dataset" in problem.message
    assert raised.value.plan.run_ids["aggregate"] is None
