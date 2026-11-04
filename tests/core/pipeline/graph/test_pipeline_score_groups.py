"""A group's score is pooled from the counts of its entries."""

from __future__ import annotations

import pandas as pd
import pytest

from mosaic.core.evaluation import EvaluationReport, pool_entry_counts
from mosaic.core.pipeline.graph.score import pooled_group_scores


def _entry(counts: dict[int, tuple[int, int, int]]) -> pd.DataFrame:
    """One entry's row of ``tp_<class>``, ``fp_<class>`` and ``fn_<class>``."""
    row: dict[str, int] = {}
    for label, (hit, extra, missed) in counts.items():
        row[f"tp_{label}"] = hit
        row[f"fp_{label}"] = extra
        row[f"fn_{label}"] = missed
    return pd.DataFrame([row])


def _report(average_over: list[int]) -> EvaluationReport:
    return EvaluationReport.model_validate(
        {
            "mode": "classifier",
            "n_entries": 3,
            "n_scored": 0,
            "n_unlabeled": 0,
            "n_missing_prediction": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "average_over": average_over,
            "classes": [0, 1, 2],
            "per_class": [
                {
                    "label": label,
                    "name": name,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "support": 0,
                    "predicted": 0,
                }
                for label, name in ((0, "attack"), (1, "mount"), (2, "other"))
            ],
            "confusion": [[0] * 3] * 3,
        }
    )


# Entry a: class 0 has 3 hits, 1 false positive, 1 miss; class 1 has 2, 0 and 2.
# Entry b: class 0 has 1, 1 and 1; class 1 has 4, 1 and 0. Class 2 is in neither.
# Pooled: class 0 is 4, 2, 2 and so F1 2/3; class 1 is 6, 1, 2 and so F1 4/5.
ENTRIES = {
    "g": [
        _entry({0: (3, 1, 1), 1: (2, 0, 2), 2: (0, 0, 0)}),
        _entry({0: (1, 1, 1), 1: (4, 1, 0), 2: (0, 0, 0)}),
    ]
}


def test_a_group_macro_f1_pools_the_counts_of_its_two_entries() -> None:
    [score] = pooled_group_scores(ENTRIES, _report([0, 1])).values()

    assert score.n_entries == 2
    assert score.n_scored == 14
    assert score.accuracy == pytest.approx(10 / 14)
    assert score.per_class_f1 == {
        "attack": pytest.approx(2 / 3),
        "mount": pytest.approx(4 / 5),
        "other": 0.0,
    }
    assert score.macro_f1 == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_it_is_not_the_mean_of_the_entries_own_macro_f1() -> None:
    [score] = pooled_group_scores(ENTRIES, _report([0, 1])).values()

    # a: (3/4 + 2/3) / 2, b: (1/2 + 8/9) / 2.
    mean_of_entries = ((3 / 4 + 2 / 3) / 2 + (1 / 2 + 8 / 9) / 2) / 2
    assert score.macro_f1 != pytest.approx(mean_of_entries)


def test_a_class_the_whole_group_lacks_scores_zero_in_the_macro_f1() -> None:
    [score] = pooled_group_scores(ENTRIES, _report([0, 1, 2])).values()

    assert score.macro_f1 == pytest.approx((2 / 3 + 4 / 5 + 0.0) / 3)


def test_a_group_kappa_is_that_of_its_pooled_counts() -> None:
    [score] = pooled_group_scores(ENTRIES, _report([0, 1])).values()

    # 14 rows: 10 hits; labeled 6 and 8 of the two classes; predicted 6 and 7.
    observed, expected = 10 / 14, (6 * 6 + 8 * 7) / 14**2
    assert score.kappa == pytest.approx((observed - expected) / (1 - expected))
    pooled = pool_entry_counts(
        pd.concat(ENTRIES["g"], ignore_index=True), average_over=[0, 1]
    )
    assert score.kappa == pooled["kappa"]


def test_a_group_kappa_is_none_when_the_pooled_counts_leave_it_undefined() -> None:
    # Every row is labeled and predicted as class 0, so chance agreement is 1.
    entries = {"g": [_entry({0: (5, 0, 0), 1: (0, 0, 0), 2: (0, 0, 0)})]}

    [score] = pooled_group_scores(entries, _report([0, 1])).values()

    assert score.kappa is None
