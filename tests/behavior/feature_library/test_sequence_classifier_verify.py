"""``pipeline verify`` of a network fit: equal at one thread count, tolerant by declaration.

A ``sequence-classifier`` has an ``n_threads`` param, so a verification holds it to
``atol`` and not to equality, and the ``evaluate-labels`` step that scores it
inherits that. The cases pin both halves: a fit at the same thread count reproduces
the stored run, and a stored value moved by more than ``atol`` fails while one moved
by less passes, which an exact feature would not.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset, open_dataset
from mosaic.core.pipeline.graph import Recipe, plan_pipeline, run_pipeline
from mosaic.core.pipeline.verify import DEFAULT_ATOL, VerifyReport, verify_plan
from mosaic.core.scope import Scope
from tests.helpers import (
    CONVERT_LABELS_STEP,
    CONVERT_TRACKS_STEP,
    assert_tree_unchanged,
    calms21_dataset,
    tree_snapshot,
)

pytestmark = pytest.mark.identity

SCOPES = {
    "labeled": Scope(groups=["train"]),
    "classifier": Scope(groups=["train"]),
    "evaluate": Scope(groups=["train"]),
}


def _recipe() -> Recipe:
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "sequence-network",
            "steps": [
                CONVERT_TRACKS_STEP,
                CONVERT_LABELS_STEP,
                {
                    "id": "heading",
                    "type": "feature",
                    "feature": "heading",
                    "inputs": ["tracks"],
                    "tracks": {"step": "tracks"},
                    "params": {"method": "two_point", "front_idx": 0, "rear_idx": 3},
                },
                {
                    "id": "pairs",
                    "type": "feature",
                    "feature": "pair-position",
                    "inputs": ["tracks", {"step": "heading"}],
                    "tracks": {"step": "tracks"},
                },
                {
                    "id": "labeled",
                    "type": "feature",
                    "feature": "extract-labeled-templates",
                    "inputs": [{"step": "pairs"}],
                    "params": {
                        "labels": {"step": "labels"},
                        "n_total": 200,
                        "test_fraction": 0.0,
                    },
                },
                {
                    "id": "classifier",
                    "type": "feature",
                    "feature": "sequence-classifier",
                    "inputs": [{"step": "pairs"}],
                    "params": {
                        "templates": {"step": "labeled"},
                        "labels": {"step": "labels"},
                        "default_class": 0,
                        "device": "cpu",
                        "head": "tcn",
                        "head_config": {"hidden": 8, "blocks": 2},
                        "steps": 12,
                        "batch_size": 4,
                        "crop_length": 60,
                        "learning_rate": 0.01,
                        "cv_folds": 2,
                    },
                },
                {
                    "id": "evaluate",
                    "type": "feature",
                    "feature": "evaluate-labels",
                    "inputs": [{"step": "classifier"}],
                    "params": {
                        "labels": {"step": "labels"},
                        "perspective": 0,
                        "average_over": [0, 1],
                    },
                },
            ],
        }
    )


@pytest.fixture
def study(tmp_path: Path) -> Dataset:
    ds = calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": {f"s{i}": 120 for i in range(1, 5)}}},
        seed=11,
        n_classes=2,
    )
    _ = run_pipeline(ds, _recipe(), step_scopes=SCOPES)
    return open_dataset(ds.manifest_path, ensure_roots=False)


def _verify(ds: Dataset, atol: float = DEFAULT_ATOL) -> VerifyReport:
    plan = plan_pipeline(ds, _recipe(), step_scopes=SCOPES)
    return verify_plan(ds, plan, atol=atol)


def _first_output(ds: Dataset) -> Path:
    planned = plan_pipeline(ds, _recipe(), step_scopes=SCOPES).step("classifier")
    root = ds.get_root("features") / planned.storage_name / str(planned.run_id)
    return sorted(root.glob("train__s1*.parquet"))[0]


def test_a_network_fitted_again_at_one_thread_reproduces_the_stored_run(
    study: Dataset,
) -> None:
    report = _verify(study)

    assert report.ok, report
    classifier = next(s for s in report.steps if s.step == "classifier")
    assert classifier.status == "ok"
    assert classifier.tolerance == report.atol
    assert classifier.max_abs_diff == 0.0
    assert classifier.argmax_agreement == 1.0
    assert {s.file for s in classifier.state} >= {"cv_folds.parquet"}
    evaluate = next(s for s in report.steps if s.step == "evaluate")
    assert (evaluate.status, evaluate.tolerance) == ("ok", report.atol)
    assert evaluate.reason == "scored the predictions of the new fit"


def test_a_value_inside_the_tolerance_passes_and_one_outside_it_fails(
    study: Dataset,
) -> None:
    path = _first_output(study)
    stored = pd.read_parquet(path)
    stored["prob_1"] = stored["prob_1"] + 1e-7
    stored.to_parquet(path)

    assert _verify(study).ok
    stored["prob_1"] = stored["prob_1"] + 1e-3
    stored.to_parquet(path)

    report = _verify(study)

    assert not report.ok
    classifier = next(s for s in report.steps if s.step == "classifier")
    assert classifier.status == "differs"
    assert classifier.max_abs_diff == pytest.approx(1e-3, rel=0.05)
    assert _verify(study, atol=1e-2).ok


def test_verifying_a_network_writes_nothing_under_the_dataset_or_beside_it(
    study: Dataset,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A private temp root: a parallel worker's verify run shares the system one.
    scratch = tmp_path_factory.mktemp("scratch")
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    before = tree_snapshot(study.base_dir)

    report = _verify(study)

    assert report.ok
    assert_tree_unchanged(study.base_dir, before)
    assert not list(scratch.glob("mosaic-verify-*"))
