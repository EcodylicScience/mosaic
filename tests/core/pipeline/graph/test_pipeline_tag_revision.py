"""A plan reads the revision of the tag table that its run will read."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.annotations.tag_table import (
    TagRead,
    TagTable,
    read_tag_table,
    tag_table_payload,
)
from mosaic.core.entry import Entry
from mosaic.core.params import Params
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.pipeline.types import Inputs, InputStream, TrackInput
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    IdentityUnresolved,
    CleanupPolicy,
    Plan,
    PlanChanged,
    Problem,
    Recipe,
    RecipeRefused,
    Request,
    build_step_feature,
    plan_fingerprint,
    plan_pipeline,
    plan_with_cleanup,
    request_path,
    run_pipeline,
    submit_request,
)
from mosaic.core.pipeline.graph import request as request_module
from mosaic.core.pipeline.graph import step as step_module
from mosaic.core.pipeline.label_series_index import (
    LabelSeriesTamperedError,
)
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.label_series import LabelRevisionRef
from mosaic.core.typed_attribute import TypedAttributeType
from mosaic.core.pipeline.verify import verify_plan
from tests.helpers import (
    index_a_second_origin,
    add_tracks_variant,
    make_dataset,
    registered_features,
    speed_recipe,
    tag_table,
    write_tags,
)

VARIANT = "convert-trex.0.2-1111111111"
ENTRIES = (("", "seq_a"), ("", "seq_b"))
TYPES: dict[str, TypedAttributeType] = {"treatment": "categorical"}
SERIES_KEY = ("tags", "dataset")


def _recipe(*fields: str, consumer: bool = True) -> Recipe:
    steps: list[dict[str, object]] = [
        {
            "id": "tags",
            "type": "feature",
            "feature": "sequence-tag-columns",
            "inputs": ["tracks"],
            "params": {"fields": list(fields)},
        }
    ]
    if consumer:
        steps.append(
            {
                "id": "motion",
                "type": "feature",
                "feature": "collective-motion-metrics",
                "inputs": ["tracks", {"step": "tags"}],
                "params": {"subgroup_col": "treatment"},
            }
        )
    return Recipe.model_validate(
        {"schema_version": 1, "name": "tagged", "steps": steps}
    )


def _table(treatment: str = "a") -> TagTable:
    return tag_table({entry: {"treatment": treatment} for entry in ENTRIES}, TYPES)


def _run_ids(plan: Plan) -> dict[str, str | None]:
    return {step.step_id: step.run_id for step in plan.steps}


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    dataset = make_dataset(tmp_path / "tagged")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return dataset


def test_the_planned_run_ids_are_the_ones_run_feature_writes(ds: Dataset) -> None:
    _ = write_tags(ds, _table())
    recipe = _recipe("treatment")
    plan = plan_pipeline(ds, recipe)

    executed: dict[str, str | None] = {}
    for planned in plan.steps:
        result = run_feature(
            ds,
            build_step_feature(planned.spec),
            tracks_run_id=planned.spec.tracks_run_id,
            track=False,
        )
        executed[planned.step_id] = result.run_id

    assert None not in executed.values()
    assert _run_ids(plan) == executed


def test_a_table_passed_unsaved_plans_the_run_ids_of_the_table_once_written(
    ds: Dataset,
) -> None:
    recipe = _recipe("treatment")
    table = _table()

    unsaved = plan_pipeline(
        ds, recipe, series_payloads={SERIES_KEY: tag_table_payload(table)}
    )
    _ = write_tags(ds, table)
    written = plan_pipeline(ds, recipe)

    assert _run_ids(unsaved) == _run_ids(written)
    assert None not in _run_ids(written).values()


def test_without_a_revision_or_a_payload_the_identity_is_unresolved(
    ds: Dataset,
) -> None:
    plan = plan_pipeline(ds, _recipe("treatment"))

    assert plan.problems == ()
    assert _run_ids(plan) == {"tags": None, "motion": None}
    reason = plan.step("tags").reason
    assert isinstance(reason, IdentityUnresolved)
    assert reason.step == "tags"
    assert "tags" in reason.because
    assert isinstance(plan.step("motion").reason, IdentityUnresolved)
    assert plan.tags_read is None


def test_the_plan_records_the_revision_it_read(ds: Dataset) -> None:
    _ = write_tags(ds, _table("a"))
    _ = write_tags(ds, _table("b"))

    plan = plan_pipeline(ds, _recipe("treatment"))

    assert plan.tags_read is not None
    assert plan.tags_read.key == "dataset"
    assert plan.tags_read.revision == 2
    assert plan.tags_read.origin_uuid == ds.uuid


def test_a_plan_of_no_reading_step_records_no_revision(ds: Dataset) -> None:
    _ = write_tags(ds, _table())
    plan = plan_pipeline(
        ds,
        Recipe.model_validate(
            {
                "schema_version": 1,
                "name": "speed",
                "steps": [
                    {
                        "id": "speed",
                        "type": "feature",
                        "feature": "speed-angvel",
                        "inputs": ["tracks"],
                    }
                ],
            }
        ),
    )

    assert plan.tags_read is None


def test_a_request_naming_a_revision_plans_that_revision(ds: Dataset) -> None:
    recipe = _recipe("treatment")
    _ = write_tags(ds, _table("a"))
    first = plan_pipeline(ds, recipe)
    _ = write_tags(ds, _table("b"))
    latest = plan_pipeline(ds, recipe)
    request = Request(
        request_id="R",
        recipe_digest="0" * 16,
        tag_revision=LabelRevisionRef(key="dataset", revision=1),
    )

    pinned = plan_pipeline(ds, recipe, request=request)

    assert _run_ids(pinned) == _run_ids(first)
    assert _run_ids(latest) != _run_ids(first)
    assert pinned.tags_read is not None
    assert pinned.tags_read.revision == 1


def test_a_request_may_name_only_the_dataset_key() -> None:
    with pytest.raises(ValueError, match="dataset"):
        _ = Request(
            request_id="R",
            recipe_digest="0" * 16,
            tag_revision=LabelRevisionRef(key="other", revision=1),
        )


def test_the_request_file_omits_an_unset_revision(tmp_path: Path) -> None:
    from mosaic.core.pipeline.graph.store import save_request

    unset = Request(request_id="01HZZZZZZZZZZZZZZZZZZZZZZZ", recipe_digest="0" * 16)
    named = Request(
        request_id="01HZZZZZZZZZZZZZZZZZZZZZZY",
        recipe_digest="0" * 16,
        tag_revision=LabelRevisionRef(key="dataset", revision=3),
    )

    unset_document = json.loads(save_request(tmp_path, unset).read_text())
    named_document = json.loads(save_request(tmp_path, named).read_text())

    assert "tag_revision" not in unset_document
    assert named_document["tag_revision"]["revision"] == 3


def test_a_series_payload_under_another_key_is_refused(ds: Dataset) -> None:
    with pytest.raises(ValueError, match=r"'keypoints', 'sets'"):
        _ = plan_pipeline(
            ds,
            _recipe("treatment"),
            series_payloads={("keypoints", "sets"): b"{}"},
        )


def test_a_sequence_tag_named_as_a_written_column_is_a_problem(ds: Dataset) -> None:
    named_frame = tag_table(
        {entry: {"frame": "x", "treatment": "a"} for entry in ENTRIES},
        {"frame": "text", "treatment": "categorical"},
    )
    _ = write_tags(ds, named_frame)

    plan = plan_pipeline(ds, _recipe("treatment"), refuse=False)

    assert (
        Problem(
            "tags",
            "params.fields",
            "tag 'frame' has the name of a column that this step writes; "
            "rename the tag to read it",
        )
        in plan.problems
    )
    assert plan.step("tags").run_id is None


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ((), "fields names no tag"),
        (("treatment", "nothing"), "'nothing' is not a sequence tag"),
    ],
)
def test_fields_that_name_no_tag_are_a_problem(
    ds: Dataset, fields: tuple[str, ...], message: str
) -> None:
    _ = write_tags(ds, _table())

    plan = plan_pipeline(ds, _recipe(*fields, consumer=False), refuse=False)

    found = [problem for problem in plan.problems if problem.step == "tags"]
    assert [problem.where for problem in found] == ["params.fields"]
    assert message in found[0].message
    assert isinstance(plan.step("tags").reason, IdentityUnresolved)


def test_a_revision_edited_in_place_is_refused_by_the_plan(ds: Dataset) -> None:
    saved = write_tags(ds, _table())
    _ = saved.path.write_bytes(saved.path.read_bytes() + b" ")

    with pytest.raises(LabelSeriesTamperedError, match="immutable"):
        _ = plan_pipeline(ds, _recipe("treatment"))


def test_two_origins_without_a_named_one_are_a_problem_of_each_reading_step(
    ds: Dataset,
) -> None:
    _ = write_tags(ds, _table())
    _ = index_a_second_origin(ds)

    plan = plan_pipeline(ds, _recipe("treatment"), refuse=False)

    assert [(problem.step, problem.where) for problem in plan.problems] == [
        ("tags", "tag_revision")
    ]
    assert "tag_revision" in plan.problems[0].message
    assert "origin_uuid" in plan.problems[0].message
    assert _run_ids(plan) == {"tags": None, "motion": None}
    assert isinstance(plan.step("tags").reason, IdentityUnresolved)


def test_a_keyword_revision_naming_no_origin_among_several_is_a_problem(
    ds: Dataset,
) -> None:
    _ = write_tags(ds, _table())
    _ = index_a_second_origin(ds)

    plan = plan_pipeline(
        ds,
        _recipe("treatment"),
        tag_revision=LabelRevisionRef(key="dataset"),
        refuse=False,
    )

    assert [(problem.step, problem.where) for problem in plan.problems] == [
        ("tags", "tag_revision")
    ]
    message = plan.problems[0].message
    assert message.endswith(" in tag_revision")
    assert "request" not in message
    assert "origin_uuid" in message
    assert _run_ids(plan) == {"tags": None, "motion": None}


def test_a_request_naming_an_origin_plans_among_several(ds: Dataset) -> None:
    _ = write_tags(ds, _table())
    other = index_a_second_origin(ds)
    request = Request(
        request_id="R",
        recipe_digest="0" * 16,
        tag_revision=LabelRevisionRef(key="dataset", origin_uuid=other),
    )

    plan = plan_pipeline(ds, _recipe("treatment"), request=request)

    assert plan.problems == ()
    assert None not in _run_ids(plan).values()


def _request_document(ds: Dataset, request: Request) -> dict[str, object]:
    return json.loads(request_path(ds.base_dir, request.request_id).read_text())


def _planned_at(ds: Dataset, recipe: Recipe, revision: int) -> dict[str, str | None]:
    request = Request(
        request_id="R",
        recipe_digest="0" * 16,
        tag_revision=LabelRevisionRef(key="dataset", revision=revision),
    )
    return _run_ids(plan_pipeline(ds, recipe, request=request))


def test_a_submission_records_the_latest_revision_of_a_reading_recipe(
    ds: Dataset,
) -> None:
    _ = write_tags(ds, _table("a"))
    _ = write_tags(ds, _table("b"))

    submitted = submit_request(ds, _recipe("treatment"))

    recorded = submitted.request.tag_revision
    assert recorded is not None
    assert (recorded.key, recorded.revision) == ("dataset", 2)
    assert recorded.origin_uuid == ds.uuid
    document = _request_document(ds, submitted.request)
    assert document["tag_revision"] == recorded.model_dump(mode="json")


def test_a_submission_of_a_recipe_that_reads_no_tags_leaves_the_key_out(
    ds: Dataset,
) -> None:
    _ = write_tags(ds, _table())

    submitted = submit_request(ds, speed_recipe())

    assert submitted.request.tag_revision is None
    assert "tag_revision" not in _request_document(ds, submitted.request)


def test_a_reference_without_a_revision_records_the_latest_without_a_reading_step(
    ds: Dataset,
) -> None:
    _ = write_tags(ds, _table("a"))
    _ = write_tags(ds, _table("b"))

    submitted = submit_request(
        ds, speed_recipe(), tag_revision=LabelRevisionRef(key="dataset")
    )

    recorded = submitted.request.tag_revision
    assert recorded is not None
    assert recorded.revision == 2
    document = _request_document(ds, submitted.request)
    tag_revision = document["tag_revision"]
    assert isinstance(tag_revision, dict)
    assert tag_revision["revision"] == 2


def test_a_reference_without_a_revision_is_resolved_before_the_plan_is_made(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _recipe("treatment")
    _ = write_tags(ds, _table("a"))

    def write_first[**P, R](function: Callable[P, R]) -> Callable[P, R]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _ = write_tags(ds, _table("b"))
            return function(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        request_module, "plan_pipeline", write_first(request_module.plan_pipeline)
    )

    submitted = submit_request(ds, recipe, tag_revision=LabelRevisionRef(key="dataset"))

    recorded = submitted.request.tag_revision
    assert recorded is not None
    assert recorded.revision == 1
    assert None not in _planned_at(ds, recipe, 2).values()
    assert _run_ids(submitted.plan) == _planned_at(ds, recipe, 1)
    assert _run_ids(submitted.plan) != _planned_at(ds, recipe, 2)


def test_a_revision_written_after_the_submit_leaves_the_step_at_the_submitted_one(
    ds: Dataset,
) -> None:
    recipe = _recipe("treatment")
    _ = write_tags(ds, _table("a"))
    submitted = submit_request(ds, recipe)
    _ = write_tags(ds, _table("b"))

    run = run_pipeline(ds, recipe, request=submitted.request)

    assert {outcome.step_id: outcome.run_id for outcome in run.outcomes} == _run_ids(
        submitted.plan
    )
    assert _run_ids(submitted.plan) == _planned_at(ds, recipe, 1)
    assert _run_ids(submitted.plan) != _planned_at(ds, recipe, 2)


def test_run_pipeline_passes_the_revision_it_is_given_to_its_request(
    ds: Dataset,
) -> None:
    recipe = _recipe("treatment")
    _ = write_tags(ds, _table("a"))
    _ = write_tags(ds, _table("b"))

    run = run_pipeline(
        ds, recipe, tag_revision=LabelRevisionRef(key="dataset", revision=1)
    )

    assert {outcome.step_id: outcome.run_id for outcome in run.outcomes} == (
        _planned_at(ds, recipe, 1)
    )


def test_an_edit_of_the_named_tag_changes_the_plan_that_was_reviewed(
    ds: Dataset,
) -> None:
    recipe = _recipe("treatment")
    _ = write_tags(ds, _table("a"))
    reviewed = plan_fingerprint(plan_pipeline(ds, recipe))
    _ = write_tags(ds, _table("b"))

    with pytest.raises(PlanChanged):
        _ = submit_request(ds, recipe, expect_plan=reviewed)


def test_an_edit_of_another_tag_leaves_the_plan_that_was_reviewed(
    ds: Dataset,
) -> None:
    types: dict[str, TypedAttributeType] = {
        "treatment": "categorical",
        "batch": "categorical",
    }

    def table(batch: str) -> TagTable:
        return tag_table(
            {entry: {"treatment": "a", "batch": batch} for entry in ENTRIES}, types
        )

    recipe = _recipe("treatment")
    _ = write_tags(ds, table("x"))
    reviewed = plan_fingerprint(plan_pipeline(ds, recipe))
    _ = write_tags(ds, table("y"))

    submitted = submit_request(ds, recipe, expect_plan=reviewed)

    assert submitted.request.tag_revision is not None
    assert submitted.request.tag_revision.revision == 2


def test_a_step_of_a_request_that_recorded_no_revision_runs_the_one_it_planned(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _recipe("treatment", consumer=False)
    submitted = submit_request(ds, recipe)
    assert submitted.request.tag_revision is None
    _ = write_tags(ds, _table("a"))

    def write_first[**P, R](function: Callable[P, R]) -> Callable[P, R]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _ = write_tags(ds, _table("b"))
            return function(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        step_module, "run_feature", write_first(step_module.run_feature)
    )

    run = run_pipeline(ds, recipe, request=submitted.request)

    assert None not in _planned_at(ds, recipe, 2).values()
    assert {outcome.step_id: outcome.run_id for outcome in run.outcomes} == (
        _planned_at(ds, recipe, 1)
    )


def test_a_reference_naming_no_origin_among_several_is_the_plans_problem(
    ds: Dataset,
) -> None:
    _ = write_tags(ds, _table())
    _ = index_a_second_origin(ds)

    with pytest.raises(RecipeRefused) as refused:
        _ = submit_request(
            ds, _recipe("treatment"), tag_revision=LabelRevisionRef(key="dataset")
        )

    assert [
        (problem.step, problem.where) for problem in refused.value.plan.problems
    ] == [("tags", "tag_revision")]
    assert not list((ds.base_dir / ".mosaic" / "pipelines" / "requests").glob("*.json"))


def test_a_keyword_revision_plans_that_revision_after_a_later_one_exists(
    ds: Dataset,
) -> None:
    recipe = _recipe("treatment")
    _ = write_tags(ds, _table("a"))
    first = plan_pipeline(ds, recipe)
    _ = write_tags(ds, _table("b"))
    latest = plan_pipeline(ds, recipe)

    pinned = plan_pipeline(
        ds, recipe, tag_revision=LabelRevisionRef(key="dataset", revision=1)
    )

    assert _run_ids(pinned) == _run_ids(first)
    assert _run_ids(pinned) != _run_ids(latest)
    assert pinned.tags_read is not None
    assert pinned.tags_read.revision == 1


def test_a_keyword_revision_wins_over_the_requests(ds: Dataset) -> None:
    recipe = _recipe("treatment")
    _ = write_tags(ds, _table("a"))
    _ = write_tags(ds, _table("b"))
    request = Request(
        request_id="R",
        recipe_digest="0" * 16,
        tag_revision=LabelRevisionRef(key="dataset", revision=1),
    )

    plan = plan_pipeline(
        ds,
        recipe,
        request=request,
        tag_revision=LabelRevisionRef(key="dataset", revision=2),
    )

    assert _run_ids(plan) == _planned_at(ds, recipe, 2)
    assert _run_ids(plan) != _planned_at(ds, recipe, 1)
    assert plan.tags_read is not None
    assert plan.tags_read.revision == 2


def test_the_request_revision_is_read_when_the_keyword_is_unset(ds: Dataset) -> None:
    recipe = _recipe("treatment")
    _ = write_tags(ds, _table("a"))
    _ = write_tags(ds, _table("b"))
    request = Request(
        request_id="R",
        recipe_digest="0" * 16,
        tag_revision=LabelRevisionRef(key="dataset", revision=1),
    )

    plan = plan_pipeline(ds, recipe, request=request, tag_revision=None)

    assert _run_ids(plan) == _planned_at(ds, recipe, 1)


def test_a_plan_without_the_keyword_reads_the_latest_revision(ds: Dataset) -> None:
    recipe = _recipe("treatment")
    _ = write_tags(ds, _table("a"))
    _ = write_tags(ds, _table("b"))

    plan = plan_pipeline(ds, recipe)

    assert _run_ids(plan) == _planned_at(ds, recipe, 2)
    assert plan_fingerprint(plan) == plan_fingerprint(
        plan_pipeline(ds, recipe, tag_revision=None)
    )


def test_plan_with_cleanup_passes_the_keyword_revision_on(ds: Dataset) -> None:
    recipe = _recipe("treatment")
    _ = write_tags(ds, _table("a"))
    _ = write_tags(ds, _table("b"))
    policy = CleanupPolicy(keep_newest=0, min_age_hours=0)

    for cleanup in (None, policy):
        plan = plan_with_cleanup(
            ds,
            recipe,
            cleanup=cleanup,
            tag_revision=LabelRevisionRef(key="dataset", revision=1),
        )

        assert _run_ids(plan) == _planned_at(ds, recipe, 1)
        assert plan.tags_read is not None
        assert plan.tags_read.revision == 1


class TaggedFit:
    """A fitting, scope-dependent feature whose output is the code of a sequence tag.

    It reads the tag table the way a tag-reading feature does, so its run id
    covers the tag and its output changes with the tag's value.
    """

    name = "test-tagged-fit"
    version = "0.1"
    category = "per-frame"
    emits: EmitsLevel = "as-input"
    accepts_overlap = False
    fits = True
    parallelizable = True
    scope_dependent = True
    consumed_roots: tuple[str, ...] = ()

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        tag: str = "treatment"

    def __init__(
        self,
        inputs: TaggedFit.Inputs | None = None,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs: TaggedFit.Inputs = inputs or self.Inputs(("tracks",))
        self.params: TaggedFit.Params = self.Params.from_overrides(params)
        self._codes: dict[Entry, float] = {}

    def tag_read(self) -> TagRead:
        return TagRead(
            sequence_tags=(self.params.tag,),
            writes=frozenset({"group", "sequence", "id", "frame", "time"}),
        )

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        (path,) = {*dependency_lookups["_tags"].values()}
        table = read_tag_table(path)
        self._codes = {
            (sequence.group, sequence.sequence): float(
                ord(str(sequence.tags[self.params.tag]))
            )
            for sequence in table.sequences
        }
        return False

    def fit(self, inputs: InputStream) -> None:
        return None

    def save_state(self, run_root: Path) -> None:
        return None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        entry = (str(df["group"].iloc[0]), str(df["sequence"].iloc[0]))
        return pd.DataFrame(
            {
                "frame": df["frame"],
                "id": df["id"],
                "group": df["group"],
                "sequence": df["sequence"],
                "code": self._codes[entry],
            }
        )


@pytest.fixture
def tagged_fit_registered() -> Iterator[None]:
    """Put the fixture feature in the registry for the length of one test."""
    with registered_features(TaggedFit):
        yield


@pytest.mark.usefixtures("tagged_fit_registered")
def test_verify_of_a_finished_request_fits_again_from_the_revision_the_run_read(
    ds: Dataset,
) -> None:
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "tagged fit",
            "steps": [
                {
                    "id": "coded",
                    "type": "feature",
                    "feature": "test-tagged-fit",
                    "inputs": ["tracks"],
                }
            ],
        }
    )
    _ = write_tags(ds, _table("a"))
    submitted = submit_request(ds, recipe)
    _ = run_pipeline(ds, recipe, request=submitted.request)
    _ = write_tags(ds, _table("b"))
    plan = plan_pipeline(ds, recipe, request=submitted.request)

    report = verify_plan(ds, plan)

    assert plan.tags_read is not None
    assert plan.tags_read.revision == 1
    assert [(verdict.step, verdict.status) for verdict in report.steps] == [
        ("coded", "ok")
    ]
    assert report.ok
    assert report.verified == 1
