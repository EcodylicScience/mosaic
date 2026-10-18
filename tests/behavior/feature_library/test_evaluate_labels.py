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
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library import evaluate_labels as module
from mosaic.behavior.feature_library.evaluate_labels import (
    CLUSTER_CONFUSION_FILE,
    CONFUSION_FILE,
    METRICS_FILE,
    EvaluateLabels,
    read_report,
)
from mosaic.core.evaluation import (
    EvaluationReport,
    cluster_label_scores,
    pool_entry_counts,
    pool_reports,
)
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.types import InputStream, Result
from mosaic.core.scope import Scope
from tests.helpers import write_sparse_labels

if TYPE_CHECKING:
    from mosaic.core.pipeline.graph import Recipe

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


# --- where the label map came from ----------------------------------------------


def _cluster_params(**extra: object) -> dict[str, object]:
    return {
        "mode": "clusters",
        "prediction_column": "cluster",
        "perspective": 0,
        **extra,
    }


def _report_of(run_root: Path) -> EvaluationReport:
    return EvaluationReport.model_validate_json((run_root / METRICS_FILE).read_text())


def test_the_version_names_the_report_that_says_where_its_map_came_from() -> None:
    assert EvaluateLabels.version == "0.2"


def test_a_report_says_where_its_label_map_came_from(tmp_path: Path) -> None:
    """Purity is the map fitted on the rows scored; a held-out map is a referenced one."""
    rows = _rows("g", "s", [7] * 5 + [9] * 5).rename(
        columns={"predicted_label": "cluster"}
    )
    lookup = {("g", "s"): _labels(tmp_path / "labels.npz")}
    label_map = tmp_path / "label_map.parquet"
    pd.DataFrame({"cluster": [7, 9], "label": [0, 1], "n_rows": [5, 5]}).to_parquet(
        label_map
    )

    _ = _run(
        _feature(_cluster_params(label_map={"feature": "fit-label-map"})),
        {("g", "s"): rows},
        lookup,
        tmp_path / "held-out",
        artifacts={"label_map": label_map},
    )
    _ = _run(_feature(_cluster_params()), {("g", "s"): rows}, lookup, tmp_path / "own")
    _ = _run(
        _feature({"perspective": 0}),
        {("g", "s"): _rows("g", "s", GOOD)},
        lookup,
        tmp_path / "classifier",
    )

    assert _report_of(tmp_path / "held-out").label_map_source == "fit-label-map"
    assert _report_of(tmp_path / "own").label_map_source == "scored_rows"
    classifier = _report_of(tmp_path / "classifier")
    assert classifier.label_map_source == ""
    assert classifier.label_map == {}


def test_a_report_saved_before_label_map_source_loads_with_it_unset(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    lookup = {("g", "s"): _labels(tmp_path / "labels.npz")}
    _ = _run(
        _feature({"perspective": 0}),
        {("g", "s"): _rows("g", "s", GOOD)},
        lookup,
        run_root,
    )
    saved = json.loads((run_root / METRICS_FILE).read_text())
    del saved["label_map_source"]
    (run_root / METRICS_FILE).write_text(json.dumps(saved))

    loaded = read_report(run_root)
    reloaded = _feature({"perspective": 0}).load_state(run_root, {}, {"labels": lookup})

    assert loaded is not None
    assert loaded.label_map_source == ""
    assert reloaded


# --- the clustering, beside its purity ----------------------------------------------


def test_a_cluster_report_carries_the_agreement_scores_and_the_contingency(
    tmp_path: Path,
) -> None:
    """Cluster 3 is class 0 alone; cluster 4 holds one row of class 0 and five of class 1."""
    rows = _rows("g", "s", [3] * 4 + [4] * 6).rename(
        columns={"predicted_label": "cluster"}
    )
    lookup = {("g", "s"): _labels(tmp_path / "labels.npz")}

    _ = _run(_feature(_cluster_params()), {("g", "s"): rows}, lookup, tmp_path / "run")

    report = _report_of(tmp_path / "run")
    expected = cluster_label_scores(
        np.array([0] * 5 + [1] * 5), np.array([3] * 4 + [4] * 6)
    )
    for key in ("homogeneity", "completeness", "v_measure"):
        assert report.clusters[key] == pytest.approx(float(str(expected[key])))
    assert 0.0 < report.clusters["v_measure"] < 1.0
    contingency = pd.read_parquet(tmp_path / "run" / CLUSTER_CONFUSION_FILE)
    assert contingency.to_dict("list") == {
        "cluster": [3, 4],
        "class_0": [4, 1],
        "class_1": [0, 5],
    }


def test_the_contingency_is_written_through_the_parquet_writer_in_clusters_mode_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[str] = []
    real_parquet = module.write_parquet_atomic

    def parquet(frame: pd.DataFrame, path: Path) -> object:
        written.append(path.name)
        return real_parquet(frame, path)

    monkeypatch.setattr(module, "write_parquet_atomic", parquet)
    lookup = {("g", "s"): _labels(tmp_path / "labels.npz")}
    clusters = _rows("g", "s", [3] * 4 + [4] * 6).rename(
        columns={"predicted_label": "cluster"}
    )

    _ = _run(
        _feature(_cluster_params()), {("g", "s"): clusters}, lookup, tmp_path / "c"
    )
    assert sorted(written) == [CLUSTER_CONFUSION_FILE, CONFUSION_FILE]

    written.clear()
    _ = _run(
        _feature({"perspective": 0}),
        {("g", "s"): _rows("g", "s", GOOD)},
        lookup,
        tmp_path / "k",
    )
    assert written == [CONFUSION_FILE]
    assert not (tmp_path / "k" / CLUSTER_CONFUSION_FILE).exists()
    assert _report_of(tmp_path / "k").clusters == {}


# --- each entry's counts ----------------------------------------------------------


def test_each_entry_row_carries_its_true_and_false_counts_per_class(
    tmp_path: Path,
) -> None:
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
    counts = ["tp_0", "fp_0", "fn_0", "tp_1", "fp_1", "fn_1"]
    assert rows[counts].values.tolist() == [
        [5, 2, 0, 3, 0, 2],
        [5, 0, 0, 5, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    assert all(rows[column].dtype == np.int64 for column in counts)


def test_the_counts_of_every_entry_pool_to_the_score_of_the_report_exactly(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(2)
    entries = {
        ("g", name): _rows("g", name, rng.integers(0, 3, size=FRAMES).tolist())
        for name in "abcd"
    }
    lookup = {
        ("g", name): _labels(tmp_path / f"{name}.npz", first=int(i % 2), second=2)
        for i, name in enumerate("abcd")
    }

    outputs = _run(_feature({"perspective": 0}), entries, lookup, tmp_path / "run")

    report = _report_of(tmp_path / "run")
    pooled = pool_entry_counts(pd.concat(outputs, ignore_index=True))
    assert pooled["macro_f1"] == report.macro_f1
    assert pooled["accuracy"] == report.accuracy
    assert pooled["per_class"] == [
        row.model_dump(exclude={"name"}) for row in report.per_class
    ]


def test_a_cluster_the_map_does_not_name_counts_as_missed_rows(tmp_path: Path) -> None:
    """Cluster 9 stands for class 1 but is outside the map, so those rows are false negatives."""
    rows = _rows("g", "s", [7] * 5 + [9] * 5).rename(
        columns={"predicted_label": "cluster"}
    )
    label_map = tmp_path / "label_map.parquet"
    pd.DataFrame({"cluster": [7], "label": [0], "n_rows": [5]}).to_parquet(label_map)
    lookup = {("g", "s"): _labels(tmp_path / "labels.npz")}

    outputs = _run(
        _feature(_cluster_params(label_map={"feature": "fit-label-map"})),
        {("g", "s"): rows},
        lookup,
        tmp_path / "run",
        artifacts={"label_map": label_map},
    )

    row = outputs[0].iloc[0]
    assert [int(row[c]) for c in ("tp_0", "fp_0", "fn_0", "tp_1", "fp_1", "fn_1")] == [
        5,
        0,
        0,
        0,
        0,
        5,
    ]
    report = _report_of(tmp_path / "run")
    pooled = pool_entry_counts(outputs[0])
    assert pooled["macro_f1"] == report.macro_f1 == 0.5


def test_runs_over_disjoint_entries_pool_to_the_run_over_all_of_them(
    tmp_path: Path,
) -> None:
    """Per-fold runs, which the entry-coverage rule keeps apart, pool by their reports."""
    rng = np.random.default_rng(5)
    entries = {
        ("g", name): _rows("g", name, rng.integers(0, 2, size=FRAMES).tolist())
        for name in "abcde"
    }
    lookup = {
        ("g", name): _labels(tmp_path / f"{name}.npz", first=0, second=1)
        for name in "abcde"
    }
    names = [("g", "a"), ("g", "b")], [("g", "c"), ("g", "d"), ("g", "e")]

    for index, fold in enumerate(names):
        _ = _run(
            _feature({"perspective": 0}),
            {key: entries[key] for key in fold},
            lookup,
            tmp_path / f"fold{index}",
        )
    _ = _run(_feature({"perspective": 0}), entries, lookup, tmp_path / "all")

    pooled = pool_reports([_report_of(tmp_path / f"fold{i}") for i in (0, 1)])
    whole = _report_of(tmp_path / "all")
    assert pooled.macro_f1 == whole.macro_f1
    assert pooled.confusion == whole.confusion
    assert pooled.per_class == whole.per_class
    assert (pooled.n_entries, pooled.n_scored) == (whole.n_entries, whole.n_scored)


def test_runs_that_miss_a_cluster_of_the_map_pool_with_those_rows_as_wrong(
    tmp_path: Path,
) -> None:
    """Cluster 9 is outside the map in both folds: pooled, 10 of 20 rows are missed."""
    label_map = tmp_path / "label_map.parquet"
    pd.DataFrame({"cluster": [7], "label": [0], "n_rows": [5]}).to_parquet(label_map)
    entries = {
        ("g", name): _rows("g", name, [7] * 5 + [9] * 5).rename(
            columns={"predicted_label": "cluster"}
        )
        for name in "ab"
    }
    lookup = {key: _labels(tmp_path / f"{key[1]}.npz") for key in entries}
    params = _cluster_params(label_map={"feature": "fit-label-map"})

    for name in "ab":
        _ = _run(
            _feature(params),
            {("g", name): entries[("g", name)]},
            lookup,
            tmp_path / f"fold-{name}",
            artifacts={"label_map": label_map},
        )
    _ = _run(
        _feature(params),
        entries,
        lookup,
        tmp_path / "all",
        artifacts={"label_map": label_map},
    )

    pooled = pool_reports([_report_of(tmp_path / f"fold-{name}") for name in "ab"])
    whole = _report_of(tmp_path / "all")
    assert pooled.accuracy == whole.accuracy == 0.5
    assert pooled.macro_f1 == whole.macro_f1
    assert pooled.confusion == whole.confusion == [[10, 0], [0, 0]]
    assert pooled.per_class == whole.per_class
    assert (pooled.n_scored, pooled.label_map_source) == (20, "fit-label-map")
    assert pooled.label_map == whole.label_map == {7: 0}


# --- what the map was fitted over ------------------------------------------------


def _cluster_recipe() -> Recipe:
    from mosaic.core.pipeline.graph import Recipe
    from tests.helpers import CONVERT_LABELS_STEP, CONVERT_TRACKS_STEP

    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "held-out-map",
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
                    "params": {"n_templates": 40},
                },
                {
                    "id": "kmeans",
                    "type": "feature",
                    "feature": "global-kmeans",
                    "inputs": [{"step": "speed"}],
                    "params": {"templates": {"step": "templates"}, "k": 3},
                },
                {
                    "id": "map",
                    "type": "feature",
                    "feature": "fit-label-map",
                    "inputs": [{"step": "kmeans"}],
                    "params": {"labels": {"step": "labels"}},
                },
                {
                    "id": "score",
                    "type": "feature",
                    "feature": "evaluate-labels",
                    "inputs": [{"step": "kmeans"}],
                    "params": {
                        "labels": {"step": "labels"},
                        "mode": "clusters",
                        "prediction_column": "cluster",
                        "label_map": {"step": "map"},
                    },
                },
                {
                    "id": "purity",
                    "type": "feature",
                    "feature": "evaluate-labels",
                    "inputs": [{"step": "kmeans"}],
                    "params": {
                        "labels": {"step": "labels"},
                        "mode": "clusters",
                        "prediction_column": "cluster",
                    },
                },
            ],
        }
    )


def test_the_fit_scope_of_an_evaluation_names_the_map_it_applied(
    tmp_path: Path,
) -> None:
    """Scored on the test group, mapped on the training group: both are on record."""
    from mosaic.core.pipeline.fit_scope import FitScope, read_fit_scope
    from mosaic.core.pipeline.graph import plan_pipeline, run_pipeline
    from tests.helpers import calms21_dataset

    ds = calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": {"s1": 200, "s2": 200}}, "test": {"a": {"s3": 200}}},
        seed=3,
        n_classes=3,
    )
    recipe = _cluster_recipe()
    scopes = {
        "templates": Scope(groups=["train"]),
        "map": Scope(groups=["train"]),
        "score": Scope(groups=["test"]),
        "purity": Scope(groups=["test"]),
    }

    _ = run_pipeline(ds, recipe, step_scopes=scopes)

    plan = plan_pipeline(ds, recipe, step_scopes=scopes)
    features = ds.get_root("features")

    def run_root(step: str) -> Path:
        planned = plan.step(step)
        return features / planned.storage_name / str(planned.run_id)

    def fitted(step: str) -> FitScope:
        record = read_fit_scope(run_root(step))
        assert record is not None
        return record

    mapped, scored, own = fitted("map"), fitted("score"), fitted("purity")
    assert mapped.entries == (("train", "s1"), ("train", "s2"))
    assert scored.entries == (("test", "s3"),)
    assert scored.source == "stream"
    assert [(ref.field, ref.feature, ref.run_id, ref.known) for ref in scored.via] == [
        ("label_map", plan.step("map").storage_name, plan.step("map").run_id, True)
    ]
    assert set(mapped.entries or ()).isdisjoint(scored.entries or ())
    assert (own.entries, own.via) == ((("test", "s3"),), ())
    assert _report_of(run_root("score")).label_map_source == "fit-label-map"
    assert _report_of(run_root("purity")).label_map_source == "scored_rows"
