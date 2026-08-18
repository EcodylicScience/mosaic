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

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import mosaic.tracking.trex.dataset_runs as trex_runs
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    CoverageShort,
    DepsIncomplete,
    IdentityUnresolved,
    Plan,
    Recipe,
    RecipeInvalid,
    build_step_feature,
    plan_pipeline,
)
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.tracks_index import (
    read_tracks_index,
    variant_for_producer_run,
)
from mosaic.tracking.trex.run import TRexConvertResult, TRexTrackResult
from tests.helpers import add_tracks_variant, make_dataset, write_media_index

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
            entries=list(planned.spec.entries) or None,
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
    assert templates.spec.entries == (("", "seq_a"), ("", "seq_b"))
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
    partial = run_feature(tracked, templates, entries=[("", "seq_a")], track=False)

    assert partial.run_id != plan.step("templates").run_id


# --- an op edge ----------------------------------------------------------------


@pytest.fixture
def fake_trex(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """TREx's two phases, replaced by fakes that write what TREx writes.

    The established shape for this suite: the tool never runs, and what it leaves
    behind is real enough to convert, so the bridge that publishes into
    ``tracks/`` is exercised rather than stubbed.
    """

    def convert(
        video_path: Path | list[Path],
        seq_dir: Path,
        *,
        output_name: str | None = None,
        **_kwargs: object,
    ) -> TRexConvertResult:
        given = (
            [video_path] if isinstance(video_path, (str, Path)) else list(video_path)
        )
        home = Path(seq_dir)
        home.mkdir(parents=True, exist_ok=True)
        stem = output_name if output_name is not None else Path(given[0]).stem
        pv_path = home / f"{stem}.pv"
        _ = pv_path.write_bytes(b"pv")
        # TREx writes one beside every conversion, and it carries the detection
        # parameters into tracking; a fake that omits it exercises a degraded
        # path rather than the ordinary one.
        settings_path = home / f"{stem}.settings"
        _ = settings_path.write_text("detect_type = yolo\n")
        return TRexConvertResult(
            pv_path=pv_path,
            settings_path=settings_path,
            background_path=None,
            stdout="",
            stderr="",
        )

    def track(pv_path: Path, seq_dir: Path, **_kwargs: object) -> TRexTrackResult:
        data_dir = Path(seq_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            data_dir / "fish0.npz",
            frame=np.arange(4),
            time=np.arange(4) / 30.0,
            cm_per_pixel=np.array([1.0]),
            **{
                "X#wcentroid": np.arange(4, dtype=float),
                "Y#wcentroid": np.arange(4, dtype=float),
            },
        )
        _ = (Path(seq_dir) / f"{Path(pv_path).stem}.results").write_bytes(b"results")
        return TRexTrackResult()

    monkeypatch.setattr(trex_runs, "run_trex_convert", convert)
    monkeypatch.setattr(trex_runs, "run_trex_track", track)
    yield


TREX_TO_SPEED: Document = {
    "steps": [
        {
            "id": "trex",
            "type": "op",
            "kind": "trex",
            "params": {"track_max_individuals": 2},
        },
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "inputs": ["tracks"],
            "tracks": {"step": "trex"},
        },
    ]
}


@pytest.mark.tracker
def test_a_tracker_step_resolves_the_variant_its_run_writes(
    tmp_path: Path, fake_trex: None
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
        dataset, entries=[("", "vid1")], track_max_individuals=2
    )

    assert produced == predicted.run_id
    index = read_tracks_index(dataset)
    assert variant_for_producer_run(index, produced) == predicted.tracks_variant
    assert plan.step("speed").spec.tracks_run_id == predicted.tracks_variant


@pytest.mark.tracker
def test_a_feature_below_a_tracker_resolves_to_what_it_then_records(
    tmp_path: Path, fake_trex: None
) -> None:
    """The whole edge, end to end: predicted before, recorded after."""
    dataset = make_dataset(tmp_path / "media")
    write_media_index(dataset, ["vid1"], uids={"vid1": "uuid-1"})
    plan = plan_pipeline(dataset, Recipe.model_validate(TREX_TO_SPEED))
    predicted = plan.step("speed").run_id

    _ = trex_runs.run_trex(dataset, entries=[("", "vid1")], track_max_individuals=2)
    speed = plan.step("speed")
    result = run_feature(
        dataset,
        build_step_feature(speed.spec),
        tracks_run_id=speed.spec.tracks_run_id,
        track=False,
    )

    assert result.run_id == predicted


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
        tracked, build_step_feature(speed.spec), entries=[("", "seq_a")], track=False
    )

    again = plan_pipeline(tracked, recipe).step("speed")

    assert again.status == "partial"
    assert again.reason == CoverageShort(covered=1, target=2, missing=(("", "seq_b"),))
    assert again.spec.entries == (("", "seq_b"),), "ask only for what is missing"


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
        entries=[("", "seq_a")],
        track=False,
    )

    again = plan_pipeline(tracked, recipe).step("templates")

    assert again.spec.entries == (("", "seq_a"), ("", "seq_b"))


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


# --- the scope a graph is planned over ------------------------------------------


def test_a_narrowing_governs_the_scope_and_moves_a_scope_dependent_identity(
    tracked: Dataset,
) -> None:
    """An explicit narrowing is the submission speaking, and it is not widened."""
    recipe = Recipe.model_validate(FIVE_STEPS)
    whole = plan_pipeline(tracked, recipe)
    narrowed = plan_pipeline(tracked, recipe, intended_entries=[("", "seq_a")])

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

    def snapshot() -> dict[str, float]:
        return {
            str(path): path.stat().st_mtime
            for path in sorted(Path(tracked.base_dir).rglob("*"))
            if path.is_file()
        }

    before = snapshot()
    _ = plan_pipeline(tracked, Recipe.model_validate(FIVE_STEPS))

    assert snapshot() == before


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
