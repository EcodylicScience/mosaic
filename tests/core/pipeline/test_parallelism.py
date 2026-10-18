"""The settings that decide how many entries a run computes at once, and their limits."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mosaic.core.pipeline.parallelism import (
    MAX_WORKERS_CEILING,
    ParallelLimits,
    ParallelRefused,
    ParallelSettings,
    effective_parallel,
    memory_use_text,
    parallel_flags,
)

GIB = 1024**3


def test_a_setting_beyond_the_ceiling_is_invalid() -> None:
    with pytest.raises(ValidationError):
        _ = ParallelSettings(workers=MAX_WORKERS_CEILING + 1)
    with pytest.raises(ValidationError):
        _ = ParallelSettings(workers=0)


def test_a_setting_is_one_worker_in_threads_with_no_floor() -> None:
    assert ParallelSettings() == ParallelSettings(
        workers=1, mode="thread", min_free_memory_fraction=0.0
    )


class TestEffectiveParallel:
    def test_no_request_is_one_worker(self) -> None:
        assert effective_parallel(None, parallelizable=True).workers == 1

    def test_a_feature_that_is_not_parallelizable_runs_in_turn(self) -> None:
        asked = ParallelSettings(workers=8, mode="process")

        assert effective_parallel(asked, parallelizable=False) == ParallelSettings()

    def test_there_is_at_most_one_worker_per_processor(self) -> None:
        asked = ParallelSettings(workers=8)

        assert effective_parallel(asked, parallelizable=True, cpus=3).workers == 3

    def test_the_count_is_kept_when_there_are_processors_to_spare(self) -> None:
        asked = ParallelSettings(workers=4, min_free_memory_fraction=0.2)

        found = effective_parallel(asked, parallelizable=True, cpus=16)

        assert found == asked

    def test_a_feature_that_needs_its_scope_runs_in_threads(self) -> None:
        asked = ParallelSettings(workers=4, mode="process")

        found = effective_parallel(
            asked, parallelizable=True, needs_scope=True, cpus=16
        )

        assert (found.workers, found.mode) == (4, "thread")


class TestParallelFlags:
    def test_one_worker_adds_no_flag(self) -> None:
        assert parallel_flags(None) == []
        assert parallel_flags(ParallelSettings(workers=1)) == []

    def test_several_workers_carry_the_mode_and_the_floor(self) -> None:
        flags = parallel_flags(
            ParallelSettings(workers=4, mode="process", min_free_memory_fraction=0.1)
        )

        assert flags == [
            "--workers",
            "4",
            "--parallel-mode",
            "process",
            "--min-free-memory-fraction",
            "0.1",
        ]


class TestParallelLimits:
    def test_a_request_naming_nothing_receives_the_ceiling_in_threads(self) -> None:
        limits = ParallelLimits(max_workers=4, allow_process_mode=True)

        found = limits.settings_for()

        assert (found.workers, found.mode) == (4, "thread")

    def test_a_request_may_ask_for_fewer_workers(self) -> None:
        limits = ParallelLimits(max_workers=4)

        assert limits.settings_for(workers=2).workers == 2

    def test_a_request_for_more_workers_than_the_limit_is_refused(self) -> None:
        limits = ParallelLimits(max_workers=4)

        with pytest.raises(ParallelRefused, match="limit of 4") as refused:
            _ = limits.settings_for(workers=5)

        assert refused.value.limit == "max_workers"
        assert refused.value.error_detail() == {"limit": "max_workers"}

    def test_process_mode_is_refused_unless_the_operator_allows_it(self) -> None:
        with pytest.raises(ParallelRefused, match="allow_process_mode") as refused:
            _ = ParallelLimits(max_workers=4).settings_for(mode="process")

        assert refused.value.limit == "allow_process_mode"

    def test_process_mode_is_given_when_the_operator_allows_it(self) -> None:
        limits = ParallelLimits(max_workers=4, allow_process_mode=True)

        assert limits.settings_for(mode="process").mode == "process"

    def test_the_floor_applies_to_every_request(self) -> None:
        limits = ParallelLimits(max_workers=2, min_free_memory_fraction=0.15)

        assert limits.settings_for(workers=1).min_free_memory_fraction == 0.15


def test_the_memory_use_names_the_workers_the_peak_and_the_host() -> None:
    text = memory_use_text(
        workers=4, mode="process", peak_rss_bytes=38 * GIB, total_bytes=64 * GIB
    )

    assert text == "workers=4, mode=process, peak memory 38.0 GB of 64.0 GB"


def test_the_memory_use_omits_a_peak_that_was_not_measured() -> None:
    text = memory_use_text(workers=2, mode="thread", peak_rss_bytes=0, total_bytes=0)

    assert text == "workers=2, mode=thread"
