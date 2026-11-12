"""The configuration file of ``mosaic agent serve`` and the manifests it serves."""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.agent.mcp.config import (
    ServerConfig,
    ServerConfigInvalid,
    read_server_config,
    served_manifests,
)
from mosaic.core.pipeline.graph import CleanupPolicy
from mosaic.core.pipeline import parallelism
from mosaic.core.pipeline.parallelism import (
    MAX_WORKERS_CEILING,
    ParallelLimits,
    ParallelRefused,
)
from tests.helpers import make_dataset


def test_relative_paths_are_read_from_the_file_directory(tmp_path: Path) -> None:
    config = tmp_path / "conf" / "server.yaml"
    config.parent.mkdir()
    _ = config.write_text(
        "datasets: [../study]\ndiscover: [/data]\nallow_managed_writes: true\n",
        encoding="utf-8",
    )

    read = read_server_config(config)

    assert read.datasets == [tmp_path / "conf" / ".." / "study"]
    assert read.discover == [Path("/data")]
    assert read.allow_managed_writes


def test_an_empty_file_serves_nothing_and_allows_no_managed_writes(
    tmp_path: Path,
) -> None:
    config = tmp_path / "server.yaml"
    _ = config.write_text("", encoding="utf-8")

    assert read_server_config(config) == ServerConfig()


@pytest.mark.parametrize("text", ["datasets: 3\n", "unknown: true\n", "datasets: [\n"])
def test_a_file_that_does_not_fit_is_invalid(tmp_path: Path, text: str) -> None:
    config = tmp_path / "server.yaml"
    _ = config.write_text(text, encoding="utf-8")

    with pytest.raises(ServerConfigInvalid, match="server.yaml"):
        _ = read_server_config(config)


def test_the_served_manifests_join_the_given_the_listed_and_the_discovered(
    tmp_path: Path,
) -> None:
    found = make_dataset(tmp_path / "projects" / "found", name="found")
    config = ServerConfig(
        datasets=[tmp_path / "listed"], discover=[tmp_path / "projects"]
    )

    served = served_manifests([tmp_path / "given"], config)

    assert served == [tmp_path / "given", tmp_path / "listed", found.manifest_path]


def test_allow_cleanup_and_policy_read_from_the_file(tmp_path: Path) -> None:
    config = tmp_path / "server.yaml"
    _ = config.write_text(
        "allow_cleanup: [feature-runs]\ncleanup:\n  keep_newest: 0\n  min_age_hours: 1\n",
        encoding="utf-8",
    )

    read = read_server_config(config)

    assert read.allow_cleanup == ["feature-runs"]
    assert read.cleanup == CleanupPolicy(keep_newest=0, min_age_hours=1)
    assert ServerConfig().cleanup == CleanupPolicy()


@pytest.mark.parametrize(
    "text",
    [
        "allow_cleanup: [tracks]\n",
        "cleanup:\n  keep_newest: -1\n",
        "cleanup:\n  x: 1\n",
    ],
    ids=["unknown-class", "negative", "unknown-key"],
)
def test_a_cleanup_setting_that_does_not_fit_is_invalid(
    tmp_path: Path, text: str
) -> None:
    config = tmp_path / "server.yaml"
    _ = config.write_text(text, encoding="utf-8")

    with pytest.raises(ServerConfigInvalid):
        _ = read_server_config(config)


def test_the_parallelism_limits_read_from_the_file(tmp_path: Path) -> None:
    config = tmp_path / "server.yaml"
    _ = config.write_text(
        "max_workers: 6\nallow_process_mode: true\nmin_free_memory_fraction: 0.2\n",
        encoding="utf-8",
    )

    read = read_server_config(config)

    assert read.parallel_limits == ParallelLimits(
        max_workers=6, allow_process_mode=True, min_free_memory_fraction=0.2
    )


def test_the_default_limits_are_the_physical_cores_in_threads_with_a_memory_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parallelism, "physical_cpus", lambda: 6)

    limits = ServerConfig().parallel_limits

    assert limits.max_workers == 6
    assert limits.allow_process_mode
    assert limits.settings_for().mode == "thread"
    assert limits.settings_for(mode="process").mode == "process"
    assert limits.min_free_memory_fraction == 0.10


def test_an_operator_may_turn_process_mode_off(tmp_path: Path) -> None:
    config = tmp_path / "server.yaml"
    _ = config.write_text("allow_process_mode: false\n", encoding="utf-8")

    limits = read_server_config(config).parallel_limits

    with pytest.raises(ParallelRefused):
        _ = limits.settings_for(mode="process")


@pytest.mark.parametrize(
    "text",
    [
        "max_workers: 0\n",
        f"max_workers: {MAX_WORKERS_CEILING + 1}\n",
        "min_free_memory_fraction: 1.0\n",
        "min_free_memory_fraction: -0.1\n",
        "allow_process_mode: maybe\n",
    ],
    ids=["zero", "above-ceiling", "floor-one", "floor-negative", "not-a-bool"],
)
def test_a_parallelism_limit_that_does_not_fit_is_invalid(
    tmp_path: Path, text: str
) -> None:
    config = tmp_path / "server.yaml"
    _ = config.write_text(text, encoding="utf-8")

    with pytest.raises(ServerConfigInvalid):
        _ = read_server_config(config)


def test_the_dataset_source_and_recipe_roots_are_resolved_and_the_datasets_served(
    tmp_path: Path,
) -> None:
    config = tmp_path / "conf" / "server.yaml"
    config.parent.mkdir()
    found = make_dataset(tmp_path / "made" / "one", name="one")
    _ = config.write_text(
        "dataset_roots: [../made]\nsource_roots: [../raw]\nrecipe_roots: [../recipes]\n",
        encoding="utf-8",
    )

    read = read_server_config(config)

    assert read.dataset_roots == [(tmp_path / "made").resolve()]
    assert read.source_roots == [(tmp_path / "raw").resolve()]
    assert read.recipe_roots == [(tmp_path / "recipes").resolve()]
    assert [path.resolve() for path in served_manifests([], read)] == [
        found.manifest_path.resolve()
    ]


def test_the_card_limit_reads_from_the_file(tmp_path: Path) -> None:
    config = tmp_path / "server.yaml"
    _ = config.write_text("gpu_cards: [0, GPU-1111]\n", encoding="utf-8")

    assert read_server_config(config).gpu_cards == ["0", "GPU-1111"]


def test_a_quoted_card_index_reads_as_an_unquoted_one_does(tmp_path: Path) -> None:
    config = tmp_path / "server.yaml"
    _ = config.write_text('gpu_cards: ["1", 0]\n', encoding="utf-8")

    assert read_server_config(config).gpu_cards == ["1", "0"]


@pytest.mark.parametrize("card", ["true", "-1", "1.5", "[0]"])
def test_a_card_that_is_not_an_index_or_a_name_is_invalid(
    tmp_path: Path, card: str
) -> None:
    config = tmp_path / "server.yaml"
    _ = config.write_text(f"gpu_cards: [{card}]\n", encoding="utf-8")

    with pytest.raises(ServerConfigInvalid, match="gpu_cards"):
        _ = read_server_config(config)


def test_no_card_limit_leases_every_visible_card() -> None:
    assert ServerConfig().gpu_cards is None


def test_an_empty_card_limit_is_invalid(tmp_path: Path) -> None:
    config = tmp_path / "server.yaml"
    _ = config.write_text("gpu_cards: []\n", encoding="utf-8")

    with pytest.raises(ServerConfigInvalid, match="gpu_cards"):
        _ = read_server_config(config)
