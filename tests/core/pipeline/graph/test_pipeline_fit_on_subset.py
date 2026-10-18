"""A basis fitted on the train groups and applied to every group, in one recipe.

``pair-posedistance-pca`` fitted from the entries in its scope, so a test group
could not be projected onto a train-only basis: a different scope refitted. With
``model`` naming a fit step's ``model.joblib``, the apply step loads that basis,
and its run does not depend on the scope it is applied over.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.pairposedistancepca import PairPoseDistancePCA
from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    IdentityUnresolved,
    Recipe,
    check_recipe,
    plan_pipeline,
    plan_wire,
    run_pipeline,
)
from mosaic.core.pipeline.index import feature_index, feature_index_path
from mosaic.core.pipeline.types import Result
from mosaic.core.scope import Scope
from tests.helpers import calms21_dataset, converting_recipe

TRAIN = Scope(groups=["train"])
TEST = Scope(groups=["test"])
EVERY_GROUP = Scope(groups=["train", "test"])

STORAGE = "pair-posedistance-pca__from__tracks"

FIT: dict[str, JsonValue] = {
    "id": "fit",
    "type": "feature",
    "feature": "pair-posedistance-pca",
    "inputs": ["tracks"],
    "params": {"n_components": 3},
}
APPLY: dict[str, JsonValue] = {
    "id": "apply",
    "type": "feature",
    "feature": "pair-posedistance-pca",
    "inputs": ["tracks"],
    "params": {"n_components": 3, "model": {"step": "fit", "pattern": "model.joblib"}},
}


def _recipe(*steps: dict[str, JsonValue]) -> Recipe:
    return Recipe.model_validate(
        {"schema_version": 1, "name": "fit on train", "steps": list(steps)}
    )


@pytest.fixture
def split(tmp_path: Path) -> Dataset:
    """Converted tracks of a train group of two sequences and a test group of one."""
    ds = calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": {"s1": 40, "s2": 30}}, "test": {"a": {"s3": 30}}},
        seed=5,
    )
    _ = ds.convert_all_tracks()
    return ds


def _outputs(ds: Dataset, run_id: str) -> dict[tuple[str, str], pd.DataFrame]:
    """Each entry's table of one run of the pair feature, by (group, sequence)."""
    rows = feature_index(feature_index_path(ds, STORAGE)).read(
        run_id=run_id, filter_ext=".parquet"
    )
    return {
        (str(row["group"]), str(row["sequence"])): pd.read_parquet(
            ds.resolve_path(str(row["abs_path"]))
        )
        for _, row in rows.iterrows()
    }


def _bases(ds: Dataset) -> list[Path]:
    return sorted(Path(ds.get_root("features")).glob(f"{STORAGE}/*/model.joblib"))


# --- one basis -----------------------------------------------------------------


def test_a_fit_on_the_train_groups_is_applied_to_every_group_with_one_basis(
    split: Dataset,
) -> None:
    planned = plan_pipeline(split, _recipe(FIT, APPLY), step_scopes={"fit": TRAIN})
    run = run_pipeline(split, _recipe(FIT, APPLY), step_scopes={"fit": TRAIN})

    fit = next(outcome for outcome in run.outcomes if outcome.step_id == "fit")
    apply = next(outcome for outcome in run.outcomes if outcome.step_id == "apply")
    assert fit.run_id is not None and apply.run_id is not None
    assert (fit.covered, fit.target) == (2, 2)
    assert (apply.covered, apply.target) == (3, 3)
    # Each run is the one that the plan named before anything ran.
    assert (planned.step("fit").run_id, planned.step("apply").run_id) == (
        fit.run_id,
        apply.run_id,
    )

    # Only the fit step wrote a basis, so the apply step did not refit on the
    # test group it was also asked about.
    assert [path.parent.name for path in _bases(split)] == [fit.run_id]

    fitted = _outputs(split, fit.run_id)
    applied = _outputs(split, apply.run_id)
    assert set(fitted) == {("train", "s1"), ("train", "s2")}
    assert set(applied) == {("train", "s1"), ("train", "s2"), ("test", "s3")}

    # The same basis gives a train entry the same scores from either step.
    components = [f"PC{index}" for index in range(3)]
    for entry, table in fitted.items():
        assert np.allclose(table[components], applied[entry][components])


def test_the_test_group_is_projected_onto_the_basis_of_the_train_groups(
    split: Dataset,
) -> None:
    run = run_pipeline(split, _recipe(FIT, APPLY), step_scopes={"fit": TRAIN})
    fit = next(outcome for outcome in run.outcomes if outcome.step_id == "fit")
    apply = next(outcome for outcome in run.outcomes if outcome.step_id == "apply")
    assert fit.run_id is not None and apply.run_id is not None

    # Projected by hand onto the fit's own basis, the test group's tracks give
    # the scores the apply step wrote.
    projector = PairPoseDistancePCA(params={"n_components": 3})
    assert projector.load_state(_bases(split)[0].parent, {}, {})
    expected = projector.apply(split.load_tracks("test", "s3"))
    written = _outputs(split, apply.run_id)[("test", "s3")]
    components = ["PC0", "PC1", "PC2"]
    assert np.allclose(written[components], expected[components])
    # And the basis saw the train groups alone: both perspectives of s1 and s2.
    basis = joblib.load(_bases(split)[0])["ipca"]
    assert basis.n_samples_seen_ == 2 * (40 + 30)


def test_an_apply_whose_basis_has_gone_is_refused_rather_than_refitted(
    split: Dataset,
) -> None:
    """A missing model must not become a fit over the scope the apply was asked for."""
    _ = run_pipeline(split, _recipe(FIT), step_scopes={"fit": TRAIN})
    for basis in _bases(split):
        basis.unlink()

    with pytest.raises(FileNotFoundError, match="holds no"):
        _ = run_pipeline(split, _recipe(FIT, APPLY), step_scopes={"fit": TRAIN})

    assert _bases(split) == []


def test_process_workers_apply_the_basis_of_the_fit_step(split: Dataset) -> None:
    """A worker rebuilds the feature from its dumped params, ``model`` among them."""
    run = run_pipeline(split, _recipe(FIT), step_scopes={"fit": TRAIN})
    fit = run.outcomes[0]
    assert fit.run_id is not None
    # One perspective only, which a step may choose differently from the fit's.
    applying = PairPoseDistancePCA(
        params={
            "n_components": 3,
            "duplicate_perspective": False,
            "model": PairPoseDistancePCA.ModelArtifact().from_result(
                Result(feature=STORAGE, run_id=fit.run_id)
            ),
        }
    )

    result = split.run_feature(applying, parallel_workers=2, parallel_mode="process")

    projector = PairPoseDistancePCA(
        params={"n_components": 3, "duplicate_perspective": False}
    )
    assert projector.load_state(_bases(split)[0].parent, {}, {})
    expected = projector.apply(split.load_tracks("test", "s3"))
    written = _outputs(split, result.run_id)
    assert set(written) == {("train", "s1"), ("train", "s2"), ("test", "s3")}
    components = ["PC0", "PC1", "PC2"]
    assert set(written[("test", "s3")]["perspective"]) == {0}
    assert np.allclose(written[("test", "s3")][components], expected[components])
    assert len(_bases(split)) == 1


# --- the apply step's identity --------------------------------------------------


@pytest.mark.parametrize(
    "scope", [TEST, TRAIN, EVERY_GROUP], ids=["test", "train", "every-group"]
)
def test_the_apply_run_id_does_not_depend_on_the_scope_it_is_applied_over(
    split: Dataset, scope: Scope
) -> None:
    recipe = _recipe(FIT, APPLY)
    reference = plan_pipeline(split, recipe, step_scopes={"fit": TRAIN})
    narrowed = plan_pipeline(split, recipe, step_scopes={"fit": TRAIN, "apply": scope})

    assert reference.problems == ()
    assert narrowed.problems == ()
    assert narrowed.step("apply").run_id is not None
    assert narrowed.step("apply").run_id == reference.step("apply").run_id


def test_the_fit_run_id_still_follows_its_scope_and_the_apply_follows_the_fit(
    split: Dataset,
) -> None:
    recipe = _recipe(FIT, APPLY)

    on_train = plan_pipeline(split, recipe, step_scopes={"fit": TRAIN})
    on_test = plan_pipeline(split, recipe, step_scopes={"fit": TEST})

    assert on_train.step("fit").run_id != on_test.step("fit").run_id
    assert on_train.step("apply").run_id != on_test.step("apply").run_id


def test_an_apply_run_over_a_narrower_scope_is_the_same_run(split: Dataset) -> None:
    """Applying to the test group after the whole set adds no second run."""
    first = run_pipeline(split, _recipe(FIT, APPLY), step_scopes={"fit": TRAIN})
    before = sorted(path.name for path in Path(split.get_root("features")).glob("*/*"))

    again = run_pipeline(
        split, _recipe(FIT, APPLY), step_scopes={"fit": TRAIN, "apply": TEST}
    )

    after = sorted(path.name for path in Path(split.get_root("features")).glob("*/*"))
    assert after == before
    assert [(o.step_id, o.run_id) for o in again.outcomes] == [
        (o.step_id, o.run_id) for o in first.outcomes
    ]


def test_an_apply_is_deferred_with_a_fit_that_waits_for_the_conversion(
    tmp_path: Path,
) -> None:
    """With no tracks yet the fit has no entries to be named for, and nor has its apply."""
    ds = calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": {"s1": 40, "s2": 30}}, "test": {"a": {"s3": 30}}},
        seed=5,
    )
    converted: dict[str, JsonValue] = {"tracks": {"step": "tracks"}}
    recipe = converting_recipe({**FIT, **converted}, {**APPLY, **converted})

    plan = plan_pipeline(ds, recipe, step_scopes={"fit": TRAIN})

    for step_id in ("fit", "apply"):
        assert plan.step(step_id).run_id is None
        assert isinstance(plan.step(step_id).reason, IdentityUnresolved)
    run = run_pipeline(ds, recipe, step_scopes={"fit": TRAIN})
    after = plan_pipeline(ds, recipe, step_scopes={"fit": TRAIN})
    executed = {outcome.step_id: outcome.run_id for outcome in run.outcomes}
    assert after.is_complete
    for step_id in ("fit", "apply"):
        assert executed[step_id] == after.step(step_id).run_id is not None


# --- the plan and validation -----------------------------------------------------


def test_the_plan_shows_the_apply_step_depending_on_the_fit_step(
    split: Dataset,
) -> None:
    plan = plan_pipeline(
        split, _recipe(FIT, APPLY), step_scopes={"fit": TRAIN}, refuse=False
    )

    assert plan.problems == ()
    assert plan.step("apply").parents == ("fit",)
    assert plan.step("apply").status_parents == ("fit",)
    wire = {step.step: step for step in plan_wire(plan).steps}
    assert wire["apply"].parents == ("fit",)
    assert wire["apply"].run_id is not None
    assert wire["apply"].run_id != wire["fit"].run_id


def test_the_apply_step_is_asked_for_every_entry_the_fit_did_not_cover(
    split: Dataset,
) -> None:
    plan = plan_pipeline(split, _recipe(FIT, APPLY), step_scopes={"fit": TRAIN})

    assert plan.step("fit").coverage.target == {("train", "s1"), ("train", "s2")}
    assert plan.step("apply").coverage.target == {
        ("train", "s1"),
        ("train", "s2"),
        ("test", "s3"),
    }


def test_a_model_naming_a_step_of_another_feature_is_refused() -> None:
    recipe = _recipe(
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "inputs": ["tracks"],
        },
        {
            **APPLY,
            "params": {"model": {"step": "speed", "pattern": "model.joblib"}},
        },
    )

    problems = [p for p in check_recipe(recipe) if p.where == "params.model"]

    assert len(problems) == 1
    assert problems[0].step == "apply"
    assert "speed-angvel" in problems[0].message
    assert "pair-posedistance-pca" in problems[0].message


def test_a_model_naming_a_step_of_the_same_feature_validates() -> None:
    assert check_recipe(_recipe(FIT, APPLY)) == ()
