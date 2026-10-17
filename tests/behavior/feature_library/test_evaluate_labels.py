"""``evaluate-labels``: a run's predictions scored against ground-truth labels.

Each row is labeled by its own frame and direction, so the intruder-focal row of
a resident-labeled pair is unlabeled and dropped. The report covers every entry
scored, the output holds one row per entry, and in clusters mode a label map
fitted on other entries maps clusters to classes before they are scored.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library import evaluate_labels as module
from mosaic.behavior.feature_library.evaluate_labels import (
    CONFUSION_FILE,
    METRICS_FILE,
    EvaluateLabels,
)
from mosaic.core.evaluation import EvaluationReport
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.types import InputStream, Result
from mosaic.core.scope import Scope
from tests.helpers import write_sparse_labels

FRAMES = 10


def _rows(group: str, sequence: str, predicted: list[int]) -> pd.DataFrame:
    """Pair rows of individuals 0 and 1 at every frame, focal on each in turn."""
    frames = np.repeat(np.arange(FRAMES), 2)
    return pd.DataFrame(
        {
            "group": group,
            "sequence": sequence,
            "frame": frames,
            "id1": np.tile([0, 1], FRAMES),
            "id2": np.tile([1, 0], FRAMES),
            "perspective": np.tile([0, 1], FRAMES),
            "predicted_label": np.repeat(predicted, 2),
        }
    )


def _labels(path: Path, first: int = 0, second: int = 1) -> Path:
    """Resident-to-intruder events: class *first* on frames 0-4, *second* on 5-9."""
    events = [(frame, 0, 1, first if frame < 5 else second) for frame in range(FRAMES)]
    return write_sparse_labels(path, events)


def _feature(params: dict[str, object] | None = None) -> EvaluateLabels:
    return EvaluateLabels(
        EvaluateLabels.Inputs((Result(feature="xgboost"),)), params or {}
    )


def _run(
    feature: EvaluateLabels,
    entries: dict[tuple[str, str], pd.DataFrame],
    lookup: dict[tuple[str, str], Path],
    run_root: Path,
    artifacts: dict[str, Path] | None = None,
) -> list[pd.DataFrame]:
    """Fit over *entries*, save, and apply each, as ``run_feature`` does."""
    loaded = feature.load_state(run_root, artifacts or {}, {"labels": lookup})
    assert not loaded
    items = [(make_entry_key(*key), frame) for key, frame in entries.items()]
    feature.fit(InputStream(lambda: iter(items), n_entries=len(items)))
    feature.save_state(run_root)
    return [feature.apply(frame) for frame in entries.values()]


GOOD = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
"""Right on frames 0-7, wrong on 8 and 9."""


def test_writes_metrics_and_confusion_through_the_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[str] = []
    real_parquet = module.write_parquet_atomic
    real_atomic = module.atomic_write

    def parquet(frame: pd.DataFrame, path: Path) -> object:
        written.append(path.name)
        return real_parquet(frame, path)

    def atomic(path: Path, fill: Callable[[Path], object]) -> None:
        written.append(path.name)
        real_atomic(path, fill)

    monkeypatch.setattr(module, "write_parquet_atomic", parquet)
    monkeypatch.setattr(module, "atomic_write", atomic)
    lookup = {("g", "s"): _labels(tmp_path / "labels.npz")}

    _ = _run(
        _feature({"perspective": 0}),
        {("g", "s"): _rows("g", "s", GOOD)},
        lookup,
        tmp_path / "run",
    )

    assert sorted(written) == [CONFUSION_FILE, METRICS_FILE]
    report = EvaluationReport.model_validate(
        json.loads((tmp_path / "run" / METRICS_FILE).read_text())
    )
    assert (report.n_scored, report.n_unlabeled) == (10, 0)
    assert report.accuracy == pytest.approx(0.8)
    assert [score.name for score in report.per_class] == ["attack", "investigation"]
    confusion = pd.read_parquet(tmp_path / "run" / CONFUSION_FILE)
    assert confusion.to_dict("list") == {
        "label": [0, 1],
        "predicted_0": [5, 2],
        "predicted_1": [0, 3],
    }


def test_the_other_direction_is_unlabeled_and_dropped(tmp_path: Path) -> None:
    lookup = {("g", "s"): _labels(tmp_path / "labels.npz")}

    _ = _run(
        _feature({"perspective": "pooled"}),
        {("g", "s"): _rows("g", "s", GOOD)},
        lookup,
        tmp_path / "run",
    )

    report = EvaluationReport.model_validate_json(
        (tmp_path / "run" / METRICS_FILE).read_text()
    )
    assert (report.n_scored, report.n_unlabeled) == (10, 10)


def test_apply_returns_one_row_per_entry(tmp_path: Path) -> None:
    entries = {
        ("g", "a"): _rows("g", "a", GOOD),
        ("g", "b"): _rows("g", "b", [0] * 5 + [1] * 5),
        ("g", "c"): _rows("g", "c", GOOD),
    }
    lookup = {
        ("g", "a"): _labels(tmp_path / "a.npz"),
        ("g", "b"): _labels(tmp_path / "b.npz"),
    }

    outputs = _run(_feature({"perspective": 0}), entries, lookup, tmp_path / "run")

    rows = pd.concat(outputs, ignore_index=True)
    assert rows["sequence"].tolist() == ["a", "b", "c"]
    assert rows["n_scored"].tolist() == [10, 10, 0]
    assert rows["accuracy"].tolist()[:2] == pytest.approx([0.8, 1.0])
    assert np.isnan(rows["accuracy"].tolist()[2])
    report = EvaluationReport.model_validate_json(
        (tmp_path / "run" / METRICS_FILE).read_text()
    )
    assert (report.n_entries, report.n_scored) == (2, 20)


def test_an_entry_without_labels_fails_under_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="has no label file"):
        _ = _run(
            _feature({"perspective": 0, "missing_label": "error"}),
            {("g", "s"): _rows("g", "s", GOOD)},
            {},
            tmp_path / "run",
        )


def test_labels_come_from_the_dependency_lookup(tmp_path: Path) -> None:
    """The run scores against whichever file its lookup names, a labels variant."""
    entries = {("g", "s"): _rows("g", "s", [0] * 5 + [1] * 5)}
    agreeing = {("g", "s"): _labels(tmp_path / "agree.npz")}
    swapped = {("g", "s"): _labels(tmp_path / "swap.npz", first=1, second=0)}

    agree = _run(_feature({"perspective": 0}), entries, agreeing, tmp_path / "a")
    swap = _run(_feature({"perspective": 0}), entries, swapped, tmp_path / "b")

    assert agree[0]["accuracy"].tolist() == [1.0]
    assert swap[0]["accuracy"].tolist() == [0.0]


def test_label_map_maps_clusters_before_scoring(tmp_path: Path) -> None:
    """Clusters 7 and 9 stand for classes 0 and 1 by a map fitted elsewhere."""
    rows = _rows("g", "s", [7] * 5 + [9] * 5).rename(
        columns={"predicted_label": "cluster"}
    )
    label_map = tmp_path / "label_map.parquet"
    pd.DataFrame({"cluster": [7, 9], "label": [0, 1], "n_rows": [5, 5]}).to_parquet(
        label_map
    )
    params: dict[str, object] = {
        "mode": "clusters",
        "prediction_column": "cluster",
        "perspective": 0,
        "label_map": {"feature": "fit-label-map"},
    }
    lookup = {("g", "s"): _labels(tmp_path / "labels.npz")}

    _ = _run(
        _feature(params),
        {("g", "s"): rows},
        lookup,
        tmp_path / "run",
        artifacts={"label_map": label_map},
    )

    report = EvaluationReport.model_validate_json(
        (tmp_path / "run" / METRICS_FILE).read_text()
    )
    assert report.accuracy == 1.0
    assert report.label_map == {7: 0, 9: 1}
    assert report.clusters["purity"] == 1.0


def test_without_a_label_map_clusters_score_their_purity(tmp_path: Path) -> None:
    rows = _rows("g", "s", [3] * 4 + [4] * 6).rename(
        columns={"predicted_label": "cluster"}
    )
    lookup = {("g", "s"): _labels(tmp_path / "labels.npz")}

    _ = _run(
        _feature(
            {"mode": "clusters", "prediction_column": "cluster", "perspective": 0}
        ),
        {("g", "s"): rows},
        lookup,
        tmp_path / "run",
    )

    report = EvaluationReport.model_validate_json(
        (tmp_path / "run" / METRICS_FILE).read_text()
    )
    assert report.accuracy == pytest.approx(report.clusters["purity"])
    assert report.accuracy == pytest.approx(0.9)
    assert report.label_map == {3: 0, 4: 1}


def test_scored_entries_enter_the_run_id(tmp_path: Path) -> None:
    """Scoring the test entries and scoring the training entries are two runs."""
    from mosaic.core.pipeline.graph import Recipe, plan_pipeline
    from tests.helpers import calms21_dataset

    ds = calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": {"s1": 20, "s2": 20}}, "test": {"a": {"s3": 20}}},
        seed=1,
    )
    _ = ds.convert_all_tracks()
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "score",
            "steps": [
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                },
                {
                    "id": "score",
                    "type": "feature",
                    "feature": "evaluate-labels",
                    "inputs": [{"step": "speed"}],
                    "params": {"prediction_column": "speed"},
                },
            ],
        }
    )

    def scored(group: str) -> str | None:
        plan = plan_pipeline(ds, recipe, step_scopes={"score": Scope(groups=[group])})
        return plan.step("score").run_id

    assert scored("test") != scored("train")
