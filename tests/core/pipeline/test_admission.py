"""The per-row memory estimate a run learns from its own entries."""

from __future__ import annotations

from mosaic.core.pipeline.admission import (
    KEPT_OBSERVATIONS,
    EntrySpan,
    PeakSample,
    RowCostEstimate,
    WorkerBases,
    WorkerMeasure,
    admission_budget,
    grown_over_bases,
    larger_peak,
)


def _worker(
    *,
    base: int,
    before: int,
    peak: int,
    base_nonreclaimable: int | None = None,
    peak_nonreclaimable: int | None = None,
) -> WorkerMeasure:
    return WorkerMeasure(
        base_rss_bytes=base,
        peak_before_rss_bytes=before,
        peak_rss_bytes=peak,
        base_nonreclaimable_bytes=base_nonreclaimable,
        peak_nonreclaimable_bytes=peak_nonreclaimable,
    )


def _estimate(base: int) -> RowCostEstimate:
    estimate = RowCostEstimate()
    estimate.parent_base(base)
    return estimate


def test_an_entry_alone_costs_its_growth_over_the_base_per_row() -> None:
    estimate = _estimate(100)

    estimate.observe(measured_bytes=1_100, in_flight_rows=10, live_workers=0)
    estimate.observe(measured_bytes=600, in_flight_rows=10, live_workers=0)

    assert estimate.bytes_per_row == 100


def test_two_entries_in_flight_share_the_growth_over_their_rows() -> None:
    estimate = _estimate(100)

    estimate.observe(measured_bytes=2_100, in_flight_rows=10 + 10, live_workers=0)

    assert estimate.bytes_per_row == 100


def test_the_largest_observation_is_the_estimate() -> None:
    estimate = _estimate(0)

    estimate.observe(measured_bytes=1_000, in_flight_rows=10, live_workers=0)
    estimate.observe(measured_bytes=3_000, in_flight_rows=10, live_workers=0)
    estimate.observe(measured_bytes=2_000, in_flight_rows=10, live_workers=0)

    assert estimate.bytes_per_row == 300


def test_a_worker_base_learned_late_is_taken_off_every_kept_observation() -> None:
    estimate = _estimate(100)
    # Two workers of 500 bytes each, and 20 rows at 10 bytes a row.
    estimate.observe(
        measured_bytes=100 + 2 * 500 + 20 * 10, in_flight_rows=20, live_workers=2
    )

    assert estimate.bytes_per_row == (2 * 500 + 20 * 10) // 20

    estimate.worker_base(500)

    assert estimate.bytes_per_row == 10


def test_the_prediction_adds_the_bases_to_the_margined_rows() -> None:
    estimate = _estimate(100)
    estimate.worker_base(500)
    estimate.observe(
        measured_bytes=100 + 500 + 10 * 10, in_flight_rows=10, live_workers=1
    )

    predicted = estimate.predicted_peak(
        rows_in_flight=10, rows_next=5, workers_after=2, margin=1.5
    )

    assert predicted == 100 + 2 * 500 + int(1.5 * 10 * (10 + 5))


def test_observations_without_rows_make_no_estimate() -> None:
    estimate = _estimate(100)

    estimate.observe(measured_bytes=50_000, in_flight_rows=0, live_workers=0)

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

    estimate.observe(measured_bytes=900, in_flight_rows=10, live_workers=0)
    estimate.observe(measured_bytes=1_500, in_flight_rows=10, live_workers=1)

    assert estimate.bytes_per_row is None


def test_a_worker_peak_raises_the_estimate_past_what_the_samples_saw() -> None:
    estimate = _estimate(100)
    estimate.worker_base(500)
    estimate.observe(
        measured_bytes=100 + 500 + 10 * 10, in_flight_rows=10, live_workers=1
    )

    # The worker reached 50 bytes a row between two samples.
    estimate.observe_worker(
        _worker(base=500, before=500, peak=500 + 10 * 50), rows=10, nonreclaimable=False
    )

    assert estimate.bytes_per_row == 50


def test_a_small_entry_after_a_large_one_on_a_worker_does_not_carry_its_peak() -> None:
    estimate = _estimate(100)
    estimate.worker_base(500)
    # The large entry reaches 50 bytes a row over its 100 rows.
    large_peak = 500 + 100 * 50
    estimate.observe_worker(
        _worker(base=500, before=600, peak=large_peak), rows=100, nonreclaimable=False
    )

    # The small entry on the same worker never rises past that high-water mark,
    # which would read as 500 bytes over its 10 rows.
    estimate.observe_worker(
        _worker(base=500, before=large_peak, peak=large_peak),
        rows=10,
        nonreclaimable=False,
    )

    assert estimate.bytes_per_row == 50


def test_a_worker_peak_without_rows_or_growth_makes_no_estimate() -> None:
    estimate = _estimate(100)

    estimate.observe_worker(
        _worker(base=500, before=500, peak=900), rows=0, nonreclaimable=False
    )
    estimate.observe_worker(
        _worker(base=500, before=300, peak=400), rows=10, nonreclaimable=False
    )

    assert estimate.bytes_per_row is None


def test_a_worker_nonreclaimable_peak_is_its_growth_per_row_with_no_rise_test() -> None:
    estimate = _estimate(100)
    # The peak equals the mark before the entry, which the RSS pair would read
    # as an earlier entry's peak. The sampler's peak is the entry's own.
    measure = _worker(
        base=500,
        before=900,
        peak=900,
        base_nonreclaimable=400,
        peak_nonreclaimable=400 + 10 * 50 + 1,
    )

    estimate.observe_worker(measure, rows=10, nonreclaimable=True)

    assert estimate.bytes_per_row == 51


def test_a_measure_without_both_nonreclaimable_figures_is_left_out_of_such_a_run() -> (
    None
):
    estimate = _estimate(100)

    estimate.observe_worker(
        _worker(base=0, before=0, peak=10_000, peak_nonreclaimable=9_000),
        rows=10,
        nonreclaimable=True,
    )
    estimate.observe_worker(
        _worker(base=0, before=0, peak=10_000, base_nonreclaimable=100),
        rows=10,
        nonreclaimable=True,
    )

    assert estimate.bytes_per_row is None


def test_a_nonreclaimable_estimate_ignores_the_rss_pair_of_a_worker() -> None:
    estimate = _estimate(100)

    estimate.observe_worker(
        _worker(
            base=0,
            before=0,
            peak=10_000,
            base_nonreclaimable=500,
            peak_nonreclaimable=500,
        ),
        rows=10,
        nonreclaimable=True,
    )

    assert estimate.bytes_per_row is None


def test_an_rss_estimate_ignores_the_nonreclaimable_pair_of_a_worker() -> None:
    estimate = _estimate(100)

    estimate.observe_worker(
        _worker(
            base=500,
            before=500,
            peak=500,
            base_nonreclaimable=0,
            peak_nonreclaimable=10_000,
        ),
        rows=10,
        nonreclaimable=False,
    )

    assert estimate.bytes_per_row is None


def test_only_the_last_observations_are_kept() -> None:
    estimate = _estimate(0)
    estimate.observe(measured_bytes=1_000_000, in_flight_rows=1, live_workers=0)
    for _ in range(KEPT_OBSERVATIONS):
        estimate.observe(measured_bytes=10, in_flight_rows=1, live_workers=0)

    # A worker base recomputes the estimate from the kept observations alone.
    estimate.worker_base(0)

    assert estimate.bytes_per_row == 10


# --- the budget ----------------------------------------------------------------


def test_the_budget_is_the_limit_less_the_floor_when_the_run_holds_what_is_used() -> (
    None
):
    budget = admission_budget(
        available_bytes=6_000, held_bytes=4_000, limit_bytes=10_000, floor=0.1
    )

    assert budget == 10_000 - 1_000


def test_memory_used_outside_the_run_narrows_the_budget() -> None:
    # 4,000 in use, of which the run holds 1,000: another process holds 3,000.
    budget = admission_budget(
        available_bytes=6_000, held_bytes=1_000, limit_bytes=10_000, floor=0.0
    )

    assert budget == 7_000


def test_the_budget_never_exceeds_the_limit() -> None:
    # A host's available memory counts reclaimable cache, so it may exceed what the
    # tree could add to it.
    budget = admission_budget(
        available_bytes=9_000, held_bytes=4_000, limit_bytes=10_000, floor=0.0
    )

    assert budget == 10_000


def test_the_budget_takes_the_figure_the_run_holds_in_its_own_measure() -> None:
    # The run holds 2,000 of the 4,000 in use, in whichever measure it fits.
    budget = admission_budget(
        available_bytes=6_000, held_bytes=2_000, limit_bytes=10_000, floor=0.0
    )

    assert budget == 8_000


# --- the peak sample -------------------------------------------------------------


def test_the_first_sample_with_an_entry_in_flight_is_the_peak() -> None:
    peak = larger_peak(None, bytes_=500, entries=1, rows=10, live_workers=0)

    assert peak == PeakSample(bytes_=500, entries=1, rows=10, live_workers=0)


def test_a_sample_with_no_entry_in_flight_is_never_the_peak() -> None:
    held = PeakSample(bytes_=500, entries=1, rows=10, live_workers=0)

    assert larger_peak(None, bytes_=9_000, entries=0, rows=0, live_workers=0) is None
    assert larger_peak(held, bytes_=9_000, entries=0, rows=0, live_workers=0) is held


def test_only_a_larger_figure_replaces_the_peak_sample() -> None:
    held = PeakSample(bytes_=500, entries=1, rows=10, live_workers=0)

    assert larger_peak(held, bytes_=500, entries=2, rows=20, live_workers=1) is held
    assert larger_peak(held, bytes_=400, entries=2, rows=20, live_workers=1) is held
    assert larger_peak(held, bytes_=501, entries=2, rows=20, live_workers=1) == (
        PeakSample(bytes_=501, entries=2, rows=20, live_workers=1)
    )


# --- the worker bases --------------------------------------------------------------


def test_an_rss_run_takes_the_largest_resident_base_of_its_workers() -> None:
    bases = WorkerBases()

    first = bases.learn(
        _worker(base=500, before=0, peak=0, base_nonreclaimable=900),
        nonreclaimable=False,
    )
    second = bases.learn(_worker(base=300, before=0, peak=0), nonreclaimable=False)

    assert (first, second) == (500, 500)


def test_a_nonreclaimable_run_takes_the_largest_such_base_of_its_workers() -> None:
    bases = WorkerBases()

    first = bases.learn(
        _worker(base=500, before=0, peak=0, base_nonreclaimable=200),
        nonreclaimable=True,
    )
    second = bases.learn(
        _worker(base=900, before=0, peak=0, base_nonreclaimable=100),
        nonreclaimable=True,
    )

    assert (first, second) == (200, 200)


def test_a_worker_without_the_figure_gives_a_nonreclaimable_run_no_base() -> None:
    bases = WorkerBases()

    learned = bases.learn(_worker(base=500, before=0, peak=0), nonreclaimable=True)

    assert learned is None
    assert (bases.rss_bytes, bases.nonreclaimable_bytes) == (500, 0)


def test_a_worker_without_the_figure_leaves_the_base_an_earlier_one_gave() -> None:
    bases = WorkerBases()
    _ = bases.learn(
        _worker(base=500, before=0, peak=0, base_nonreclaimable=200),
        nonreclaimable=True,
    )

    learned = bases.learn(_worker(base=700, before=0, peak=0), nonreclaimable=True)

    assert learned is None
    assert (bases.rss_bytes, bases.nonreclaimable_bytes) == (700, 200)


def _peak(bytes_: int, *, live_workers: int = 0) -> PeakSample:
    return PeakSample(bytes_=bytes_, entries=2, rows=50, live_workers=live_workers)


def test_the_peak_grew_over_the_parent_base_in_a_thread_run() -> None:
    assert grown_over_bases(_peak(900), base_bytes=300, worker_base_bytes=0) == 600


def test_the_peak_grew_over_the_parent_and_each_live_worker_base() -> None:
    peak = _peak(900, live_workers=3)

    assert grown_over_bases(peak, base_bytes=300, worker_base_bytes=100) == 300


def test_a_peak_under_the_bases_grew_nothing() -> None:
    peak = _peak(500, live_workers=3)

    assert grown_over_bases(peak, base_bytes=300, worker_base_bytes=100) == 0


def test_no_peak_sample_means_no_growth_figure() -> None:
    assert grown_over_bases(None, base_bytes=300, worker_base_bytes=100) == 0


def test_a_base_that_was_not_measured_is_not_subtracted_as_zero() -> None:
    peak = _peak(900, live_workers=2)

    assert grown_over_bases(peak, base_bytes=0, worker_base_bytes=100) == 0
    assert grown_over_bases(peak, base_bytes=300, worker_base_bytes=0) == 0
    assert grown_over_bases(_peak(900), base_bytes=0, worker_base_bytes=0) == 0


def test_a_process_workers_growth_is_its_peak_over_its_base() -> None:
    worker = _worker(
        base=1, before=1, peak=1, base_nonreclaimable=200, peak_nonreclaimable=750
    )

    assert worker.grown_bytes == 550


def test_a_process_worker_that_read_less_than_its_base_grew_nothing() -> None:
    worker = _worker(
        base=1, before=1, peak=1, base_nonreclaimable=200, peak_nonreclaimable=150
    )

    assert worker.grown_bytes == 0


def test_a_process_worker_without_both_figures_has_no_growth() -> None:
    assert (
        _worker(base=1, before=1, peak=1, base_nonreclaimable=200).grown_bytes is None
    )
    assert (
        _worker(base=1, before=1, peak=1, peak_nonreclaimable=200).grown_bytes is None
    )


def _span(*, most_in_flight: int, peak: int | None, rows: int = 10) -> EntrySpan:
    return EntrySpan(
        rows=rows,
        most_in_flight=most_in_flight,
        load_started=0.0,
        cpu_at_load=0.0,
        apply_seconds=1.0,
        peak_nonreclaimable_bytes=peak,
    )


def test_an_entry_that_ran_alone_grew_by_its_peak_over_the_base() -> None:
    assert _span(most_in_flight=1, peak=900).grown_bytes(base_bytes=300) == 600
    assert _span(most_in_flight=1, peak=200).grown_bytes(base_bytes=300) == 0


def test_an_entry_that_ran_beside_another_has_no_growth() -> None:
    assert _span(most_in_flight=2, peak=900).grown_bytes(base_bytes=300) is None


def test_an_entry_that_saw_no_sample_has_no_growth() -> None:
    assert _span(most_in_flight=1, peak=None).grown_bytes(base_bytes=300) is None


def test_an_unmeasured_base_gives_an_entry_no_growth() -> None:
    assert _span(most_in_flight=1, peak=900).grown_bytes(base_bytes=0) is None
