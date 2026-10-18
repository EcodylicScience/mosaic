"""Retiring a saved recipe: what is refused, which requests and runs are the recipe's.

The judging of the runs is in ``test_pipeline_cleanup.py``. These cases cover the
reads that feed it: the digest checks, the match of a request to a retired recipe,
the runs a request wrote as opposed to found, and the list of recipes a person may
retire.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    RetireNeedsCleanup,
    RetireOpenRecipe,
    RetireOwnRecipe,
    RetireUnknownRecipe,
    load_request,
    recipe_digest,
    recipe_path,
    request_rollup,
    retirable_recipes,
    retired_recipe_names,
    run_pipeline,
    save_request,
    submit_request,
)
from mosaic.core.pipeline.graph.digest import document_digest
from mosaic.core.pipeline.graph.retirement import (
    NAME_LIMIT,
    exclusively_written,
    retired_digest_of,
    validate_retirement,
    written_runs,
)
from tests.helpers import (
    add_tracks_variant,
    make_dataset,
    named_recipe,
    run_speed,
    speed_recipe_stepping,
)

VARIANT = "convert-trex.0.2-1111111111"

STORAGE = "speed-angvel__from__tracks"


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


def _ran(dataset: Dataset, step_size: int, name: str) -> tuple[str, str]:
    """Run a recipe named *name* through a request, as ``(request id, digest)``."""
    recipe = named_recipe(speed_recipe_stepping(step_size), name)
    done = run_pipeline(dataset, recipe)
    return done.request_id, recipe_digest(recipe)


# --- what is refused ------------------------------------------------------------------


def test_nothing_to_retire_needs_no_cleanup(dataset: Dataset) -> None:
    assert validate_retirement(dataset.base_dir, "x", (), cleanup=False) == ()


def test_retiring_without_a_cleanup_is_refused(dataset: Dataset) -> None:
    _, old = _ran(dataset, 1, "old")

    with pytest.raises(RetireNeedsCleanup) as raised:
        _ = validate_retirement(dataset.base_dir, "x", [old], cleanup=False)

    assert raised.value.error_kind == "invalid"
    assert raised.value.error_reason() == "retire_needs_cleanup"


@pytest.mark.parametrize(
    "digest", ["0123456789abcdef", "../escape", "ABCDEF0123456789", "short", ""]
)
def test_a_digest_no_saved_recipe_has_is_refused(dataset: Dataset, digest: str) -> None:
    with pytest.raises(RetireUnknownRecipe) as raised:
        _ = validate_retirement(dataset.base_dir, "x", [digest], cleanup=True)

    assert raised.value.error_kind == "not_found"
    assert raised.value.error_detail() == {"digest": digest}


def test_a_saved_recipe_that_does_not_read_is_refused(dataset: Dataset) -> None:
    _, old = _ran(dataset, 1, "old")
    _ = recipe_path(dataset.base_dir, old).write_text("{not json")

    with pytest.raises(RetireUnknownRecipe, match="does not read"):
        _ = validate_retirement(dataset.base_dir, "x", [old], cleanup=True)


def test_the_recipe_being_planned_cannot_retire_itself(dataset: Dataset) -> None:
    _, old = _ran(dataset, 1, "old")

    with pytest.raises(RetireOwnRecipe) as raised:
        _ = validate_retirement(dataset.base_dir, old, [old], cleanup=True)

    assert raised.value.error_reason() == "retire_own_recipe"


def test_a_recipe_with_a_request_that_has_not_ended_is_refused(
    dataset: Dataset,
) -> None:
    done, old = _ran(dataset, 1, "old")
    unended = submit_request(
        dataset, named_recipe(speed_recipe_stepping(1), "old")
    ).request

    with pytest.raises(RetireOpenRecipe) as raised:
        _ = validate_retirement(dataset.base_dir, "x", [old], cleanup=True)

    assert raised.value.error_kind == "conflict"
    assert raised.value.error_detail() == {
        "digest": old,
        "request_ids": [unended.request_id],
    }
    assert done != unended.request_id


def test_the_digests_are_returned_sorted_without_repeats(dataset: Dataset) -> None:
    _, first = _ran(dataset, 1, "first")
    _, second = _ran(dataset, 2, "second")

    checked = validate_retirement(
        dataset.base_dir, "x", [second, first, second], cleanup=True
    )

    assert checked == tuple(sorted({first, second}))


def test_a_digest_in_an_earlier_canonical_form_names_the_current_one(
    dataset: Dataset,
) -> None:
    _, old = _ran(dataset, 1, "old")
    legacy = named_recipe(speed_recipe_stepping(1), "old").model_dump(mode="json")
    legacy["steps"][0]["params"]["step_size"] = 1.0
    legacy_digest = document_digest(legacy)
    _ = recipe_path(dataset.base_dir, legacy_digest).write_text(json.dumps(legacy))

    checked = validate_retirement(dataset.base_dir, "x", [legacy_digest], cleanup=True)

    assert legacy_digest != old
    assert checked == (old,)
    with pytest.raises(RetireOwnRecipe):
        _ = validate_retirement(dataset.base_dir, old, [legacy_digest], cleanup=True)


# --- which requests and runs are the recipe's -----------------------------------------


def test_a_request_belongs_to_a_retired_recipe_by_its_digest(dataset: Dataset) -> None:
    done, old = _ran(dataset, 1, "old")
    other, _ = _ran(dataset, 2, "other")
    cache: dict[str, str | None] = {}

    mine = load_request(dataset.base_dir, done)
    theirs = load_request(dataset.base_dir, other)

    assert retired_digest_of(dataset.base_dir, mine, {old}, cache) == old
    assert retired_digest_of(dataset.base_dir, theirs, {old}, cache) == ""
    assert retired_digest_of(dataset.base_dir, mine, set(), cache) == ""


def test_a_request_saved_under_an_earlier_form_belongs_to_the_current_digest(
    dataset: Dataset,
) -> None:
    done, old = _ran(dataset, 1, "old")
    legacy = named_recipe(speed_recipe_stepping(1), "old").model_dump(mode="json")
    legacy["steps"][0]["params"]["step_size"] = 1.0
    legacy_digest = document_digest(legacy)
    _ = recipe_path(dataset.base_dir, legacy_digest).write_text(json.dumps(legacy))
    request = load_request(dataset.base_dir, done).model_copy(
        update={"recipe_digest": legacy_digest}
    )

    found = retired_digest_of(dataset.base_dir, request, {old}, {})

    assert found == old


def test_a_request_created_after_the_cutoff_is_not_retired(dataset: Dataset) -> None:
    done, old = _ran(dataset, 1, "old")
    request = load_request(dataset.base_dir, done)
    assert request.created_at

    def retired(before: str) -> str:
        return retired_digest_of(dataset.base_dir, request, {old}, {}, before=before)

    assert retired("") == old
    assert retired("2999-01-01T00:00:00Z") == old
    assert retired(request.created_at) == ""
    assert retired("2000-01-01T00:00:00Z") == ""
    assert retired("not a time") == old


def test_a_request_wrote_the_run_it_computed_and_not_the_one_it_found(
    dataset: Dataset,
) -> None:
    first, _ = _ran(dataset, 1, "first")
    again, _ = _ran(dataset, 1, "again")

    wrote = written_runs(
        request_rollup(dataset.base_dir, load_request(dataset.base_dir, first))
    )
    found = written_runs(
        request_rollup(dataset.base_dir, load_request(dataset.base_dir, again))
    )

    [(storage, run_id)] = wrote
    assert storage == STORAGE
    assert run_id
    assert found == frozenset()


def test_a_run_a_notebook_also_used_is_not_exclusively_the_requests(
    dataset: Dataset,
) -> None:
    done, _ = _ran(dataset, 1, "old")
    request = load_request(dataset.base_dir, done)
    wrote = written_runs(request_rollup(dataset.base_dir, request))
    executions = set(request.step_executions.values())

    assert exclusively_written(dataset.base_dir, wrote, executions) == wrote
    _ = run_speed(dataset, 1)

    assert exclusively_written(dataset.base_dir, wrote, executions) == frozenset()
    assert exclusively_written(dataset.base_dir, [], executions) == frozenset()


# --- the recipes a person may retire --------------------------------------------------


def test_the_recipes_that_wrote_runs_are_listed_newest_first(dataset: Dataset) -> None:
    first, old = _ran(dataset, 1, "old")
    second, new = _ran(dataset, 2, "new")
    _ = _ran(dataset, 2, "new")

    listed, omitted = retirable_recipes(dataset.base_dir)

    assert omitted == 0
    assert [(row.name, row.digest) for row in listed] == [("new", new), ("old", old)]
    assert [(row.requests, row.runs, row.open) for row in listed] == [
        (2, 1, False),
        (1, 1, False),
    ]
    assert first in listed[1].request_ids
    assert second in listed[0].request_ids


def test_the_planned_recipe_is_left_out_and_the_list_is_bounded(
    dataset: Dataset,
) -> None:
    _, old = _ran(dataset, 1, "old")
    _, new = _ran(dataset, 2, "new")

    without, _ = retirable_recipes(dataset.base_dir, exclude_digest=new)
    bounded, omitted = retirable_recipes(dataset.base_dir, limit=1)

    assert [row.digest for row in without] == [old]
    assert [row.digest for row in bounded] == [new]
    assert omitted == 1


def test_a_recipe_that_wrote_nothing_is_not_listed(dataset: Dataset) -> None:
    _, old = _ran(dataset, 1, "old")
    _ = run_pipeline(dataset, named_recipe(speed_recipe_stepping(1), "only found"))

    listed, _ = retirable_recipes(dataset.base_dir)

    assert [row.digest for row in listed] == [old]


def test_an_unended_request_marks_its_recipe_open(dataset: Dataset) -> None:
    _, old = _ran(dataset, 1, "old")
    _ = submit_request(dataset, named_recipe(speed_recipe_stepping(1), "old"))

    [row], _ = retirable_recipes(dataset.base_dir)

    assert (row.digest, row.open, row.requests) == (old, True, 2)


def test_a_request_file_that_does_not_read_is_skipped(dataset: Dataset) -> None:
    _, old = _ran(dataset, 1, "old")
    _ = (
        dataset.base_dir / ".mosaic" / "pipelines" / "requests" / "ZZZ.json"
    ).write_text("{broken")

    listed, _ = retirable_recipes(dataset.base_dir)

    assert old in [row.digest for row in listed]


def _saved_only_in_an_earlier_form(dataset: Dataset, done: str, current: str) -> str:
    """Leave the recipe of request *done* only in the file of its earlier form."""
    legacy = named_recipe(speed_recipe_stepping(1), "old").model_dump(mode="json")
    legacy["steps"][0]["params"]["step_size"] = 1.0
    legacy_digest = document_digest(legacy)
    _ = recipe_path(dataset.base_dir, legacy_digest).write_text(json.dumps(legacy))
    recipe_path(dataset.base_dir, current).unlink()
    request = load_request(dataset.base_dir, done).model_copy(
        update={"recipe_digest": legacy_digest}
    )
    save_request(dataset.base_dir, request)
    return legacy_digest


def test_the_listed_digest_is_accepted_when_only_an_earlier_form_is_saved(
    dataset: Dataset,
) -> None:
    """``pipeline show`` and the list print the current form; the file may not exist."""
    done, current = _ran(dataset, 1, "old")
    legacy = _saved_only_in_an_earlier_form(dataset, done, current)
    assert not recipe_path(dataset.base_dir, current).exists()

    [row], _ = retirable_recipes(dataset.base_dir)
    by_current = validate_retirement(dataset.base_dir, "x", [current], cleanup=True)
    by_legacy = validate_retirement(dataset.base_dir, "x", [legacy], cleanup=True)
    names = retired_recipe_names(dataset.base_dir, [current])

    assert (row.digest, row.name) == (current, "old")
    assert by_current == by_legacy == (current,)
    assert names == {current: "old"}


def test_a_digest_that_no_saved_recipe_reaches_is_still_unknown(
    dataset: Dataset,
) -> None:
    done, current = _ran(dataset, 1, "old")
    _ = _saved_only_in_an_earlier_form(dataset, done, current)

    with pytest.raises(RetireUnknownRecipe):
        _ = validate_retirement(dataset.base_dir, "x", ["f" * 16], cleanup=True)


def test_only_the_runs_still_on_disk_make_a_recipe_retirable(
    dataset: Dataset,
) -> None:
    done, old = _ran(dataset, 1, "old")
    rollup = request_rollup(dataset.base_dir, load_request(dataset.base_dir, done))
    [(storage, run_id)] = written_runs(rollup)
    gone = {(storage, run_id)}

    everything, _ = retirable_recipes(dataset.base_dir)
    nothing, _ = retirable_recipes(
        dataset.base_dir, present=lambda s, r: (s, r) not in gone
    )
    present, _ = retirable_recipes(dataset.base_dir, present=lambda s, r: True)

    assert [row.digest for row in everything] == [old]
    assert nothing == ()
    assert [row.digest for row in present] == [old]


def test_a_recipe_name_is_cut_where_it_is_listed_and_named(dataset: Dataset) -> None:
    _, digest = _ran(dataset, 1, "n" * 300)

    [row], _ = retirable_recipes(dataset.base_dir)
    names = retired_recipe_names(dataset.base_dir, [digest])

    assert row.name == "n" * NAME_LIMIT
    assert names == {digest: "n" * NAME_LIMIT}
