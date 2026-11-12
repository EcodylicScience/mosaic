"""The settings that decide how many entries a run computes at once, and their limits."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mosaic.core.pipeline import parallelism
from mosaic.core.pipeline.parallelism import (
    MAX_WORKERS_CEILING,
    EntryOverMemoryLimit,
    MemoryFloorReached,
    ParallelLimits,
    ParallelRefused,
    ParallelSettings,
    effective_parallel,
    memory_use_text,
    parallel_flags,
    physical_cpus,
    process_capable,
)

GIB = 1024**3


def test_a_setting_beyond_the_ceiling_is_invalid() -> None:
    with pytest.raises(ValidationError):
        _ = ParallelSettings(workers=MAX_WORKERS_CEILING + 1)
    with pytest.raises(ValidationError):
        _ = ParallelSettings(workers=0)


def test_a_setting_is_one_worker_in_threads_with_no_floor() -> None:
    assert ParallelSettings() == ParallelSettings(
        workers=1, mode="thread", min_free_memory_fraction=0.0, admission_margin=0.0
    )


@pytest.mark.parametrize("margin", [0.0, 1.0, 1.5, 4.0])
def test_an_admission_margin_is_off_or_from_one_to_four(margin: float) -> None:
    assert ParallelSettings(workers=2, admission_margin=margin).admission_margin == (
        margin
    )


@pytest.mark.parametrize("margin", [-0.5, 0.5, 0.99, 4.01])
def test_an_admission_margin_outside_that_is_invalid(margin: float) -> None:
    with pytest.raises(ValidationError, match="admission_margin"):
        _ = ParallelSettings(workers=2, admission_margin=margin)


def _host(
    monkeypatch: pytest.MonkeyPatch,
    *,
    usable: int | None,
    logical: int | None,
    physical: int | None,
) -> None:
    """Fake a host: *usable* processors in the affinity mask, then psutil's counts."""
    if usable is None:
        monkeypatch.delattr(parallelism.os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(parallelism.os, "cpu_count", lambda: logical)
    else:

        def affinity(_pid: int) -> set[int]:
            return set(range(usable))

        monkeypatch.setattr(
            parallelism.os, "sched_getaffinity", affinity, raising=False
        )

    def count(*, logical: bool = True) -> int | None:
        return logical_processors if logical else physical

    logical_processors = logical
    monkeypatch.setattr(parallelism.psutil, "cpu_count", count)


class TestPhysicalCpus:
    def test_hyperthreads_are_not_counted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _host(monkeypatch, usable=40, logical=40, physical=20)

        assert physical_cpus() == 20

    def test_the_affinity_mask_narrows_the_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _host(monkeypatch, usable=8, logical=40, physical=20)

        assert physical_cpus() == 4

    def test_a_host_without_hyperthreads_counts_every_processor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _host(monkeypatch, usable=None, logical=10, physical=10)

        assert physical_cpus() == 10

    def test_a_count_psutil_cannot_give_counts_every_processor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _host(monkeypatch, usable=6, logical=6, physical=None)

        assert physical_cpus() == 6

    def test_there_is_at_least_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _host(monkeypatch, usable=1, logical=8, physical=4)

        assert physical_cpus() == 1


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
        asked = ParallelSettings(
            workers=4, min_free_memory_fraction=0.2, admission_margin=1.5
        )

        found = effective_parallel(asked, parallelizable=True, cpus=16)

        assert found == asked

    def test_a_feature_that_cannot_run_in_processes_runs_in_threads(self) -> None:
        asked = ParallelSettings(workers=4, mode="process")

        found = effective_parallel(
            asked, parallelizable=True, process_capable=False, cpus=16
        )

        assert (found.workers, found.mode) == (4, "thread")

    def test_a_process_request_gives_a_feature_that_needs_its_scope_threads(
        self,
    ) -> None:
        asked = ParallelLimits(max_workers=4).settings_for(mode="process")

        found = effective_parallel(
            asked, parallelizable=True, process_capable=False, cpus=16
        )

        assert (asked.mode, found.mode, found.workers) == ("process", "thread", 4)

    def test_a_default_request_gives_a_feature_that_is_not_parallelizable_one_worker(
        self,
    ) -> None:
        asked = ParallelLimits(max_workers=4).settings_for()

        found = effective_parallel(asked, parallelizable=False, cpus=16)

        assert found == ParallelSettings()


class _Parallel:
    parallelizable = True


class _Scoped(_Parallel):
    def set_scope(self, scope: object) -> None:
        _ = scope


class _Sequential:
    parallelizable = False


class TestProcessCapable:
    def test_a_parallelizable_feature_without_a_scope_may_run_in_processes(
        self,
    ) -> None:
        assert process_capable(_Parallel()) is True

    def test_a_feature_that_needs_its_scope_may_not(self) -> None:
        assert process_capable(_Scoped()) is False

    def test_a_feature_that_is_not_parallelizable_may_not(self) -> None:
        assert process_capable(_Sequential()) is False


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

    def test_an_admission_margin_travels_with_the_workers(self) -> None:
        flags = parallel_flags(ParallelSettings(workers=2, admission_margin=1.5))

        assert flags[-2:] == ["--admission-margin", "1.5"]

    def test_no_margin_adds_no_flag(self) -> None:
        assert "--admission-margin" not in parallel_flags(ParallelSettings(workers=2))


class TestParallelLimits:
    @pytest.mark.parametrize("allow_process_mode", [True, False])
    def test_a_request_naming_nothing_receives_the_ceiling_in_threads(
        self, allow_process_mode: bool
    ) -> None:
        limits = ParallelLimits(max_workers=4, allow_process_mode=allow_process_mode)

        found = limits.settings_for()

        assert (found.workers, found.mode) == (4, "thread")

    def test_the_defaults_are_the_physical_cores_with_processes_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(parallelism, "physical_cpus", lambda: 12)

        limits = ParallelLimits()

        assert (limits.max_workers, limits.allow_process_mode) == (12, True)

    def test_the_default_floor_keeps_a_tenth_of_the_memory_available(self) -> None:
        assert ParallelLimits().min_free_memory_fraction == 0.10

    def test_the_default_count_stays_within_the_ceiling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(parallelism, "physical_cpus", lambda: 128)

        assert ParallelLimits().max_workers == MAX_WORKERS_CEILING

    def test_a_request_may_name_threads_when_processes_are_allowed(self) -> None:
        limits = ParallelLimits(max_workers=4)

        assert limits.settings_for(mode="thread").mode == "thread"

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
            _ = ParallelLimits(max_workers=4, allow_process_mode=False).settings_for(
                mode="process"
            )

        assert refused.value.limit == "allow_process_mode"

    def test_process_mode_is_given_when_the_operator_allows_it(self) -> None:
        limits = ParallelLimits(max_workers=4, allow_process_mode=True)

        assert limits.settings_for(mode="process").mode == "process"

    def test_the_floor_applies_to_every_request(self) -> None:
        limits = ParallelLimits(max_workers=2, min_free_memory_fraction=0.15)

        assert limits.settings_for(workers=1).min_free_memory_fraction == 0.15


def test_the_memory_use_names_the_workers_the_peak_and_the_limit() -> None:
    text = memory_use_text(
        workers=4, mode="process", peak_rss_bytes=38 * GIB, limit_bytes=64 * GIB
    )

    assert text == "workers=4, mode=process, peak memory 38.0 GB of 64.0 GB"


def test_the_memory_use_omits_a_peak_that_was_not_measured() -> None:
    text = memory_use_text(workers=2, mode="thread", peak_rss_bytes=0, limit_bytes=0)

    assert text == "workers=2, mode=thread"


def test_the_memory_use_of_a_small_limit_is_stated_against_that_limit() -> None:
    text = memory_use_text(
        workers=2, mode="thread", peak_rss_bytes=6 * GIB, limit_bytes=8 * GIB
    )

    assert text == "workers=2, mode=thread, peak memory 6.0 GB of 8.0 GB"


def test_the_floor_names_the_container_when_a_cgroup_limit_applies() -> None:
    error = MemoryFloorReached(
        "workers=2, mode=thread",
        free_fraction=0.04,
        floor=0.1,
        limit_bytes=8 * GIB,
        source="cgroup_v2",
    )

    assert "the container has 4% of its 8.0 GB limit available" in str(error)
    assert "fewer workers" in str(error)


def test_the_floor_names_the_host_when_no_limit_applies() -> None:
    error = MemoryFloorReached(
        "workers=2, mode=thread",
        free_fraction=0.04,
        floor=0.1,
        limit_bytes=64 * GIB,
        source="host",
    )

    assert "the host has 4% of its 64.0 GB memory available" in str(error)


def test_an_entry_over_the_limit_names_itself_its_prediction_and_the_limit() -> None:
    error = EntryOverMemoryLimit(
        "g__s", predicted_bytes=12 * GIB, limit_bytes=8 * GIB, source="cgroup_v2"
    )

    text = str(error)
    assert "g__s" in text
    assert "12.0 GB" in text
    assert "container's 8.0 GB limit" in text
    assert "Run again with a larger memory limit." in text
    assert "thread mode" not in text
    assert "already written are kept" in text


def test_an_entry_over_the_limit_in_process_mode_also_offers_thread_mode() -> None:
    error = EntryOverMemoryLimit(
        "g__s",
        predicted_bytes=12 * GIB,
        limit_bytes=8 * GIB,
        source="cgroup_v2",
        parallel_mode="process",
    )

    assert "Run again with a larger memory limit, or in thread mode." in str(error)
