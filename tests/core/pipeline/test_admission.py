"""The per-row memory estimate a run learns from its own entries."""

from __future__ import annotations

import pytest

from mosaic.core.pipeline.admission import (
    KEPT_OBSERVATIONS,
    EntriesInFlight,
    EntrySpan,
    PeakSample,
    RowCostEstimate,
    WorkerBases,
    WorkerMeasure,
    admission_budget,
    grown_over_bases,
    larger_peak,
)
from mosaic.runlog import InFlightLevel


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


def _sample(
    bytes_: int,
    *,
    entries: int,
    rows: int,
    live_workers: int = 0,
    pair_frames: int = 0,
    pair_frames_bound: bool = False,
) -> PeakSample:
    return PeakSample(
        bytes_=bytes_,
        entries=entries,
        rows=rows,
        pair_frames=pair_frames,
        pair_frames_bound=pair_frames_bound,
        live_workers=live_workers,
    )


def _consider(
    held: PeakSample | None,
    bytes_: int,
    *,
    entries: int,
    rows: int,
    live_workers: int = 0,
    pair_frames: int = 0,
    pair_frames_bound: bool = False,
) -> PeakSample | None:
    return larger_peak(
        held,
        bytes_=bytes_,
        entries=entries,
        rows=rows,
        pair_frames=pair_frames,
        pair_frames_bound=pair_frames_bound,
        live_workers=live_workers,
    )


def test_the_first_sample_with_an_entry_in_flight_is_the_peak() -> None:
    peak = _consider(
        None, 500, entries=1, rows=10, pair_frames=6, pair_frames_bound=True
    )

    assert peak == _sample(
        500, entries=1, rows=10, pair_frames=6, pair_frames_bound=True
    )


def test_a_sample_with_no_entry_in_flight_is_never_the_peak() -> None:
    held = _sample(500, entries=1, rows=10)

    assert _consider(None, 9_000, entries=0, rows=0) is None
    assert _consider(held, 9_000, entries=0, rows=0) is held


def test_only_a_larger_figure_replaces_the_peak_sample() -> None:
    held = _sample(500, entries=1, rows=10)

    assert _consider(held, 500, entries=2, rows=20, live_workers=1) is held
    assert _consider(held, 400, entries=2, rows=20, live_workers=1) is held
    assert _consider(held, 501, entries=2, rows=20, live_workers=1) == (
        _sample(501, entries=2, rows=20, live_workers=1)
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
    return _sample(bytes_, entries=2, rows=50, live_workers=live_workers)


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


# --- the levels of the entries in flight ------------------------------------------


class _Time:
    """A clock and a CPU counter that a test moves by hand."""

    def __init__(self) -> None:
        self.now = 0.0
        self.cpu = 0.0

    def move(self, *, to: float, cpu: float) -> None:
        self.now = to
        self.cpu = cpu

    def in_flight(self) -> EntriesInFlight:
        return EntriesInFlight(cpu_seconds=lambda: self.cpu, clock=lambda: self.now)

    def enter(
        self,
        in_flight: EntriesInFlight,
        key: str,
        rows: int,
        *,
        pair_frames: int | None,
    ) -> None:
        in_flight.enter(
            key, rows, pair_frames=pair_frames, started=self.now, cpu=self.cpu
        )


def _two_overlapping(clock: _Time) -> EntriesInFlight:
    """Entry ``a`` (100 rows) in flight from 0 to 6 s, ``b`` (60 rows) from 2 to 8 s."""
    in_flight = clock.in_flight()
    clock.enter(in_flight, "a", 100, pair_frames=None)
    clock.move(to=2.0, cpu=2.0)
    clock.enter(in_flight, "b", 60, pair_frames=None)
    clock.move(to=6.0, cpu=8.0)
    _ = in_flight.leave("a", written=True)
    clock.move(to=8.0, cpu=9.0)
    _ = in_flight.leave("b", written=True)
    return in_flight


def test_two_overlapping_entries_give_levels_one_and_two() -> None:
    in_flight = _two_overlapping(_Time())

    one, two = in_flight.levels()

    # Level 1 is 0-2 s and 6-8 s, level 2 is 2-6 s. Each entry's rows are spread
    # evenly over its own 6 s: a has 100 / 6 rows a second, b has 60 / 6.
    assert one.in_flight == 1
    assert one.seconds == pytest.approx(4.0)
    assert one.rows == pytest.approx(100 * 2 / 6 + 60 * 2 / 6)
    assert one.cpu_seconds == pytest.approx((2.0 - 0.0) + (9.0 - 8.0))
    assert two.in_flight == 2
    assert two.seconds == pytest.approx(4.0)
    assert two.rows == pytest.approx(100 * 4 / 6 + 60 * 4 / 6)
    assert two.cpu_seconds == pytest.approx(8.0 - 2.0)


def test_the_levels_hold_the_time_and_rows_of_the_written_entries() -> None:
    levels = _two_overlapping(_Time()).levels()

    assert sum(level.seconds for level in levels) == pytest.approx(8.0)
    assert sum(level.rows for level in levels) == pytest.approx(160.0)


def test_an_entry_that_was_not_written_adds_nothing() -> None:
    clock = _Time()
    in_flight = clock.in_flight()
    clock.enter(in_flight, "a", 100, pair_frames=None)
    clock.move(to=3.0, cpu=3.0)
    span = in_flight.leave("a")

    assert span is not None
    assert in_flight.levels() == ()
    assert in_flight.count() == 0


def test_leaving_twice_adds_nothing_twice() -> None:
    clock = _Time()
    in_flight = clock.in_flight()
    clock.enter(in_flight, "a", 100, pair_frames=None)
    clock.move(to=3.0, cpu=3.0)
    _ = in_flight.leave("a", written=True)
    clock.move(to=9.0, cpu=9.0)

    assert in_flight.leave("a", written=True) is None

    assert in_flight.levels() == (
        InFlightLevel(in_flight=1, seconds=3.0, rows=100.0, cpu_seconds=3.0),
    )


def test_a_written_exit_after_an_unwritten_one_adds_nothing() -> None:
    clock = _Time()
    in_flight = clock.in_flight()
    clock.enter(in_flight, "a", 100, pair_frames=None)
    clock.move(to=3.0, cpu=3.0)
    _ = in_flight.leave("a")

    assert in_flight.leave("a", written=True) is None
    assert in_flight.levels() == ()


def test_a_level_never_reached_is_absent() -> None:
    clock = _Time()
    in_flight = clock.in_flight()
    clock.enter(in_flight, "a", 10, pair_frames=None)
    clock.move(to=1.0, cpu=1.0)
    _ = in_flight.leave("a", written=True)
    clock.move(to=2.0, cpu=1.5)
    clock.enter(in_flight, "b", 20, pair_frames=None)
    clock.move(to=4.0, cpu=3.5)
    _ = in_flight.leave("b", written=True)

    levels = in_flight.levels()

    assert [level.in_flight for level in levels] == [1]
    assert levels[0].seconds == pytest.approx(3.0)
    assert levels[0].rows == pytest.approx(30.0)


def test_an_open_flight_is_not_counted() -> None:
    clock = _Time()
    in_flight = clock.in_flight()
    clock.enter(in_flight, "a", 10, pair_frames=None)
    clock.enter(in_flight, "open", 99, pair_frames=None)
    clock.move(to=4.0, cpu=4.0)
    _ = in_flight.leave("a", written=True)

    assert in_flight.levels() == (
        InFlightLevel(in_flight=1, seconds=4.0, rows=10.0, cpu_seconds=4.0),
    )
    assert in_flight.count() == 1


def test_an_exit_that_is_not_written_still_lowers_the_count() -> None:
    clock = _Time()
    in_flight = clock.in_flight()
    clock.enter(in_flight, "a", 10, pair_frames=None)

    _ = in_flight.leave("a", written=False)

    assert in_flight.count() == 0
    assert in_flight.rows() == 0


def test_the_totals_sum_the_pair_frames_and_flag_any_bound() -> None:
    clock = _Time()
    in_flight = clock.in_flight()
    clock.enter(in_flight, "a", 10, pair_frames=6)
    clock.enter(in_flight, "none", 20, pair_frames=None)
    assert in_flight.totals() == (2, 30, 6, False)

    in_flight.enter("c", 5, pair_frames=4, pair_frames_bound=True, started=0.0, cpu=0.0)

    assert in_flight.totals() == (3, 35, 10, True)
    _ = in_flight.leave("c")
    assert in_flight.totals() == (2, 30, 6, False)
