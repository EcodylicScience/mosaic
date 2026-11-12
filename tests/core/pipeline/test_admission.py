"""The per-row memory estimate a run learns from its own entries."""

from __future__ import annotations

from mosaic.core.pipeline.admission import (
    KEPT_OBSERVATIONS,
    RowCostEstimate,
    WorkerMeasure,
    admission_budget,
)


def _worker(*, base: int, before: int, peak: int) -> WorkerMeasure:
    return WorkerMeasure(
        base_rss_bytes=base, peak_before_rss_bytes=before, peak_rss_bytes=peak
    )


def _estimate(base: int) -> RowCostEstimate:
    estimate = RowCostEstimate()
    estimate.parent_base(base)
    return estimate


def test_an_entry_alone_costs_its_growth_over_the_base_per_row() -> None:
    estimate = _estimate(100)

    estimate.observe(tree_rss=1_100, in_flight_rows=10, live_workers=0)
    estimate.observe(tree_rss=600, in_flight_rows=10, live_workers=0)

    assert estimate.bytes_per_row == 100


def test_two_entries_in_flight_share_the_growth_over_their_rows() -> None:
    estimate = _estimate(100)

    estimate.observe(tree_rss=2_100, in_flight_rows=10 + 10, live_workers=0)

    assert estimate.bytes_per_row == 100


def test_the_largest_observation_is_the_estimate() -> None:
    estimate = _estimate(0)

    estimate.observe(tree_rss=1_000, in_flight_rows=10, live_workers=0)
    estimate.observe(tree_rss=3_000, in_flight_rows=10, live_workers=0)
    estimate.observe(tree_rss=2_000, in_flight_rows=10, live_workers=0)

    assert estimate.bytes_per_row == 300


def test_a_worker_base_learned_late_is_taken_off_every_kept_observation() -> None:
    estimate = _estimate(100)
    # Two workers of 500 bytes each, and 20 rows at 10 bytes a row.
    estimate.observe(
        tree_rss=100 + 2 * 500 + 20 * 10, in_flight_rows=20, live_workers=2
    )

    assert estimate.bytes_per_row == (2 * 500 + 20 * 10) // 20

    estimate.worker_base(500)

    assert estimate.bytes_per_row == 10


def test_the_prediction_adds_the_bases_to_the_margined_rows() -> None:
    estimate = _estimate(100)
    estimate.worker_base(500)
    estimate.observe(tree_rss=100 + 500 + 10 * 10, in_flight_rows=10, live_workers=1)

    predicted = estimate.predicted_peak(
        rows_in_flight=10, rows_next=5, workers_after=2, margin=1.5
    )

    assert predicted == 100 + 2 * 500 + int(1.5 * 10 * (10 + 5))


def test_observations_without_rows_make_no_estimate() -> None:
    estimate = _estimate(100)

    estimate.observe(tree_rss=50_000, in_flight_rows=0, live_workers=0)

    assert estimate.bytes_per_row is None
    assert (
        estimate.predicted_peak(
            rows_in_flight=0, rows_next=10, workers_after=0, margin=1.0
        )
        == 100
    )


def test_memory_at_or_under_the_fixed_part_makes_no_estimate() -> None:
    estimate = _estimate(1_000)
    estimate.worker_base(500)

    estimate.observe(tree_rss=900, in_flight_rows=10, live_workers=0)
    estimate.observe(tree_rss=1_500, in_flight_rows=10, live_workers=1)

    assert estimate.bytes_per_row is None


def test_a_worker_peak_raises_the_estimate_past_what_the_samples_saw() -> None:
    estimate = _estimate(100)
    estimate.worker_base(500)
    estimate.observe(tree_rss=100 + 500 + 10 * 10, in_flight_rows=10, live_workers=1)

    # The worker reached 50 bytes a row between two samples.
    estimate.observe_worker(_worker(base=500, before=500, peak=500 + 10 * 50), rows=10)

    assert estimate.bytes_per_row == 50


def test_a_small_entry_after_a_large_one_on_a_worker_does_not_carry_its_peak() -> None:
    estimate = _estimate(100)
    estimate.worker_base(500)
    # The large entry reaches 50 bytes a row over its 100 rows.
    large_peak = 500 + 100 * 50
    estimate.observe_worker(_worker(base=500, before=600, peak=large_peak), rows=100)

    # The small entry on the same worker never rises past that high-water mark,
    # which would read as 500 bytes over its 10 rows.
    estimate.observe_worker(
        _worker(base=500, before=large_peak, peak=large_peak), rows=10
    )

    assert estimate.bytes_per_row == 50


def test_a_worker_peak_without_rows_or_growth_makes_no_estimate() -> None:
    estimate = _estimate(100)

    estimate.observe_worker(_worker(base=500, before=500, peak=900), rows=0)
    estimate.observe_worker(_worker(base=500, before=300, peak=400), rows=10)

    assert estimate.bytes_per_row is None


def test_only_the_last_observations_are_kept() -> None:
    estimate = _estimate(0)
    estimate.observe(tree_rss=1_000_000, in_flight_rows=1, live_workers=0)
    for _ in range(KEPT_OBSERVATIONS):
        estimate.observe(tree_rss=10, in_flight_rows=1, live_workers=0)

    # A worker base recomputes the estimate from the kept observations alone.
    estimate.worker_base(0)

    assert estimate.bytes_per_row == 10


# --- the budget ----------------------------------------------------------------


def test_the_budget_is_the_limit_less_the_floor_when_the_run_holds_what_is_used() -> (
    None
):
    budget = admission_budget(
        available_bytes=6_000, tree_rss_bytes=4_000, limit_bytes=10_000, floor=0.1
    )

    assert budget == 10_000 - 1_000


def test_memory_used_outside_the_run_narrows_the_budget() -> None:
    # 4,000 in use, of which the run holds 1,000: another process holds 3,000.
    budget = admission_budget(
        available_bytes=6_000, tree_rss_bytes=1_000, limit_bytes=10_000, floor=0.0
    )

    assert budget == 7_000


def test_the_budget_never_exceeds_the_limit() -> None:
    # A host's available memory counts reclaimable cache, so it may exceed what the
    # tree could add to it.
    budget = admission_budget(
        available_bytes=9_000, tree_rss_bytes=4_000, limit_bytes=10_000, floor=0.0
    )

    assert budget == 10_000
