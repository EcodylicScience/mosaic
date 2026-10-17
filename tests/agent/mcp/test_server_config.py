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
