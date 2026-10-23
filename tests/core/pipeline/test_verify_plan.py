"""``verify_plan``: a finished recipe fitted again and compared with its stored runs.

The recipe is the classifier recipe of the held-out protocol, with an
``evaluate-labels`` step after its ``xgboost`` step. Each case runs it once and then
changes one stored thing, because a verification is only worth its name if it
fails when the stored answer is not the one the recipe produces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset, open_dataset
from mosaic.core.pipeline.graph import Plan, Recipe, plan_pipeline, run_pipeline
from mosaic.core.pipeline.verify import (
    ScratchRefused,
    StepVerdict,
    VerifyReport,
    verify_plan,
)
from mosaic.core.pipeline.verify import _Found, _walk  # pyright: ignore[reportPrivateUsage]
from tests.helpers import (
    assert_tree_unchanged,
    calms21_dataset,
    classifier_recipe,
    tree_snapshot,
)

ENTRIES = {"train": {"a": {"s1": 90, "s2": 90, "s3": 90}}}


def _recipe() -> Recipe:
    base = classifier_recipe("clf").model_dump(mode="json")
    evaluation = {
        "id": "score",
        "type": "feature",
        "feature": "evaluate-labels",
        "inputs": [{"step": "clf"}],
        "params": {"labels": {"step": "labels"}, "perspective": 0},
    }
    return Recipe.model_validate({**base, "steps": [*base["steps"], evaluation]})


@pytest.fixture
def study(tmp_path: Path) -> Dataset:
    """A dataset the recipe has run over, as a reader opens it."""
    ds = calms21_dataset(tmp_path / "study", ENTRIES, seed=11, n_classes=2)
    _ = run_pipeline(ds, _recipe())
    return open_dataset(ds.manifest_path, ensure_roots=False)


def _plan(ds: Dataset) -> Plan:
    return plan_pipeline(ds, _recipe())


def _verdict(report: VerifyReport, step: str) -> StepVerdict:
    return next(verdict for verdict in report.steps if verdict.step == step)


def _root(ds: Dataset, step: str) -> Path:
    planned = _plan(ds).step(step)
    return ds.get_root("features") / planned.storage_name / str(planned.run_id)


def test_a_recipe_fitted_again_matches_what_it_stored(study: Dataset) -> None:
    report = verify_plan(study, _plan(study))

    assert report.ok
    clf = _verdict(report, "clf")
    assert (clf.role, clf.status, clf.tolerance) == ("fit", "ok", 0.0)
    assert clf.max_abs_diff == 0.0
    assert clf.argmax_agreement == 1.0
    assert clf.entries_checked == 3
    assert {state.file for state in clf.state} >= {"xgboost_model.joblib"}
    assert _verdict(report, "labeled").status == "ok"
    score = _verdict(report, "score")
    assert (score.role, score.status) == ("evaluation", "ok")
    assert score.reason == "scored the predictions of the new fit"
    assert report.verified == 3


def test_a_step_that_fits_nothing_is_skipped_with_the_reason(study: Dataset) -> None:
    report = verify_plan(study, _plan(study))

    for step in ("tracks", "labels", "heading", "pairs"):
        assert _verdict(report, step).status == "skipped"
    assert _verdict(report, "heading").reason == "it fits nothing"
    assert _verdict(report, "tracks").reason == "an op step is not fitted here"


def test_a_stored_output_that_was_changed_fails_and_names_the_entry(
    study: Dataset,
) -> None:
    [path] = sorted(_root(study, "clf").glob("train__s2*.parquet"))
    stored = pd.read_parquet(path)
    stored["prob_1"] = stored["prob_1"] + 0.01
    stored.to_parquet(path)

    report = verify_plan(study, _plan(study))

    assert not report.ok
    clf = _verdict(report, "clf")
    assert clf.status == "differs"
    assert clf.max_abs_diff == pytest.approx(0.01, abs=1e-6)
    assert [e.sequence for e in clf.entries if e.status == "differs"] == ["s2"]
    assert [e.sequence for e in clf.entries if e.status == "ok"] == ["s1", "s3"]


def test_a_value_inside_the_tolerance_still_fails_for_an_exact_feature(
    study: Dataset,
) -> None:
    [path] = sorted(_root(study, "clf").glob("train__s1*.parquet"))
    stored = pd.read_parquet(path)
    stored["prob_1"] = stored["prob_1"] + 1e-6
    stored.to_parquet(path)

    report = verify_plan(study, _plan(study), atol=1.0)

    assert _verdict(report, "clf").status == "differs"


def test_a_stored_score_that_was_changed_fails(study: Dataset) -> None:
    path = _root(study, "score") / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["macro_f1"] = metrics["macro_f1"] - 0.05
    _ = path.write_text(json.dumps(metrics))

    report = verify_plan(study, _plan(study))

    score = _verdict(report, "score")
    assert score.status == "differs"
    assert score.scores and score.scores[0].startswith(".macro_f1")
    assert _verdict(report, "clf").status == "ok"


def test_a_stored_state_file_that_was_changed_fails(study: Dataset) -> None:
    path = _root(study, "clf") / "reports.json"
    reports = json.loads(path.read_text())
    reports["0"]["precision"] = 0.123
    _ = path.write_text(json.dumps(reports))

    report = verify_plan(study, _plan(study))

    clf = _verdict(report, "clf")
    assert clf.status == "differs"
    assert [s.file for s in clf.state if s.status == "differs"] == ["reports.json"]


def test_the_dataset_is_left_as_it_was_and_the_fit_goes_to_the_scratch(
    study: Dataset, tmp_path: Path
) -> None:
    before = tree_snapshot(study.base_dir)
    scratch = tmp_path / "scratch"

    report = verify_plan(study, _plan(study), scratch=scratch)

    assert report.ok
    assert_tree_unchanged(study.base_dir, before)
    # A bundle in the scratch is proof that the step was fitted, not loaded.
    assert (scratch / "clf" / "state" / "xgboost_model.joblib").exists()


def test_a_scratch_directory_inside_the_dataset_is_refused(study: Dataset) -> None:
    with pytest.raises(ScratchRefused, match="inside the dataset"):
        _ = verify_plan(study, _plan(study), scratch=study.base_dir / "scratch")


def test_a_scratch_directory_holding_other_files_is_refused(
    study: Dataset, tmp_path: Path
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    _ = (scratch / "notes.txt").write_text("keep")

    with pytest.raises(ScratchRefused, match="holds other files"):
        _ = verify_plan(study, _plan(study), scratch=scratch)

    assert (scratch / "notes.txt").read_text() == "keep"


def test_a_scratch_directory_of_an_earlier_verification_is_cleared_and_reused(
    study: Dataset, tmp_path: Path
) -> None:
    scratch = tmp_path / "scratch"
    _ = verify_plan(study, _plan(study), scratch=scratch)

    again = verify_plan(study, _plan(study), scratch=scratch)

    assert again.ok


def test_a_report_that_compared_nothing_is_not_ok_and_a_step_with_no_run_is_skipped(
    tmp_path: Path,
) -> None:
    ds = calms21_dataset(tmp_path / "fresh", ENTRIES, seed=11, n_classes=2)
    _ = ds.convert_all_tracks()

    report = verify_plan(
        open_dataset(ds.manifest_path, ensure_roots=False),
        plan_pipeline(ds, _recipe(), refuse=False),
    )

    assert not report.ok
    assert report.verified == 0
    assert _verdict(report, "clf").status == "skipped"


def test_a_step_whose_stored_output_is_gone_is_not_vouched_for(study: Dataset) -> None:
    [path] = sorted(_root(study, "clf").glob("train__s1*.parquet"))
    path.unlink()

    report = verify_plan(study, _plan(study))

    assert not report.ok
    clf = _verdict(report, "clf")
    assert clf.status == "skipped"
    assert "no complete run" in clf.reason


def test_a_run_whose_outputs_are_all_gone_does_not_pass(study: Dataset) -> None:
    for path in _root(study, "clf").glob("train__*.parquet"):
        path.unlink()

    report = verify_plan(study, _plan(study))

    assert not report.ok
    assert _verdict(report, "clf").status == "skipped"


@pytest.mark.parametrize(
    ("stored", "fresh"),
    [
        (0.5, "abc"),
        (0.5, None),
        ({"a": 1}, [1]),
        ([1], {"a": 1}),
        ({"f1": 0.9}, {"f1": None}),
        ({"f1": None}, {"f1": 0.9}),
        ({"f1": "x"}, {"f1": 0.9}),
        (1.0, True),
    ],
)
def test_two_values_of_different_kinds_are_a_mismatch(
    stored: object, fresh: object
) -> None:
    found = _Found()

    _walk(stored, fresh, "", found, exact_models=True)

    assert not found.within(1.0)
    assert found.mismatches


def test_an_object_it_cannot_read_and_a_value_too_deep_are_named_not_passed() -> None:
    class Opaque:
        __slots__ = ("x",)

    deep: object = 1.0
    for _ in range(10):
        deep = {"k": deep}
    found = _Found()

    _walk(Opaque(), Opaque(), "", found, exact_models=True)
    _walk(deep, deep, "", found, exact_models=True)

    assert found.compared == 0
    assert len(found.unchecked) == 2
    assert "Opaque" in found.unchecked[0]
    assert "deeper" in found.unchecked[1]


def test_equal_values_count_as_compared() -> None:
    found = _Found()

    _walk(
        {"f1": 0.9, "name": "a", "w": [1, 2]},
        {"f1": 0.9, "name": "a", "w": [1, 2]},
        "",
        found,
        exact_models=True,
    )

    assert found.within(0.0)
    assert found.compared == 4
