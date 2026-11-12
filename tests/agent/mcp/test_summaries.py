"""The paragraph that a server returns beside a run summary."""

from __future__ import annotations

from mosaic.agent.mcp.summaries import summarize
from mosaic.core.pipeline.run_summary import (
    GroupWire,
    KruskalWire,
    MannWhitneyWire,
    RunSummaryWire,
    StatisticsWire,
)

RUN_ID = "0.2-1a2b3c4d5e"

KEYS = (
    {"strain": "DGRP_105", "sex": "female_x"},
    {"strain": "DGRP_208", "sex": "male_y"},
)

COLUMNS = ("speed_mean", "angvel_mean")


def _statistics(column: str, key: dict[str, str]) -> StatisticsWire:
    return StatisticsWire(
        column=column,
        key=dict(key),
        n=3,
        n_missing=0,
        mean=1.0,
        sd=0.5,
        sem=0.3,
        median=1.0,
    )


def _pair(column: str, p: float | None) -> MannWhitneyWire:
    return MannWhitneyWire(
        column=column,
        a=dict(KEYS[0]),
        b=dict(KEYS[1]),
        n_a=3,
        n_b=3,
        statistic=None if p is None else 2.0,
        p=p,
        p_holm=p,
        comparisons=1,
        reason="no_variation" if p is None else None,
    )


SUMMARY = RunSummaryWire(
    ref=f"feature/sequence-aggregate__from__speed-angvel__from__tracks/{RUN_ID}",
    by=("strain", "sex"),
    columns=COLUMNS,
    n_entries=7,
    missing=(("cohort_one", "trial_9"),),
    finished=True,
    entries_without_rows=(("cohort_two", "trial_4"),),
    rows_without_key=4,
    max_rows_per_entry=1,
    groups=tuple(GroupWire(key=dict(key), n_rows=3, n_entries=3) for key in KEYS),
    statistics=tuple(_statistics(column, key) for column in COLUMNS for key in KEYS),
    kruskal=(
        KruskalWire(
            column=COLUMNS[0], k=2, n=6, statistic=1.5, df=1, p=0.2, reason=None
        ),
        KruskalWire(
            column=COLUMNS[1],
            k=2,
            n=6,
            statistic=None,
            df=1,
            p=None,
            reason="no_variation",
        ),
    ),
    mann_whitney=(_pair(COLUMNS[0], 0.2), _pair(COLUMNS[1], None)),
)


def test_the_paragraph_names_the_run_id_and_the_counts() -> None:
    paragraph = summarize(SUMMARY)

    assert RUN_ID in paragraph
    for count in (
        "2 group(s)",
        "2 column(s)",
        "7 entries read",
        "1 missing",
        "4 row(s) without a key",
        "Kruskal-Wallis tested 1 column(s)",
        "Mann-Whitney tested 1 pair(s)",
    ):
        assert count in paragraph, (count, paragraph)


def test_the_paragraph_names_no_key_value_and_no_column() -> None:
    paragraph = summarize(SUMMARY)

    dataset_text = [
        *COLUMNS,
        *SUMMARY.by,
        *(value for key in KEYS for value in key.values()),
        "cohort_one",
        "trial_9",
        "cohort_two",
        "trial_4",
        "sequence-aggregate",
    ]
    assert [text for text in dataset_text if text in paragraph] == []


def test_a_summary_without_tests_names_no_test() -> None:
    untested = SUMMARY.model_copy(update={"kruskal": (), "mann_whitney": ()})

    paragraph = summarize(untested)

    assert "Kruskal" not in paragraph
    assert "Mann-Whitney" not in paragraph


def test_a_summary_with_kruskal_alone_names_no_pair() -> None:
    kruskal_only = SUMMARY.model_copy(update={"mann_whitney": ()})

    paragraph = summarize(kruskal_only)

    assert "Kruskal-Wallis tested 1 column(s)" in paragraph
    assert "Mann-Whitney" not in paragraph
    assert "pair(s)" not in paragraph


def test_a_summary_with_mann_whitney_alone_names_no_kruskal_column() -> None:
    pairs_only = SUMMARY.model_copy(update={"kruskal": ()})

    paragraph = summarize(pairs_only)

    assert "Mann-Whitney tested 1 pair(s)" in paragraph
    assert "Kruskal" not in paragraph
