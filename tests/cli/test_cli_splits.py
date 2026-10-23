"""``mosaic splits``: list, set and remove the role of a group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.manifest import read_manifest
from tests.helpers import make_dataset

runner = CliRunner()


def _listed(manifest: Path) -> dict[str, str]:
    result = runner.invoke(app, ["splits", "list", "-m", str(manifest), "--json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["splits"]


def test_set_list_and_remove_round_trip(tmp_path: Path) -> None:
    manifest = make_dataset(tmp_path / "ds").manifest_path

    assert _listed(manifest) == {}
    first = runner.invoke(
        app, ["splits", "set", "-m", str(manifest), "train", "s1", "s2"]
    )
    assert first.exit_code == 0, first.output
    _ = runner.invoke(app, ["splits", "set", "-m", str(manifest), "sealed", "s3"])

    assert _listed(manifest) == {"s1": "train", "s2": "train", "s3": "sealed"}
    assert read_manifest(manifest).splits == _listed(manifest)

    removed = runner.invoke(app, ["splits", "remove", "-m", str(manifest), "s3"])
    assert removed.exit_code == 0, removed.output
    assert _listed(manifest) == {"s1": "train", "s2": "train"}


def test_the_plain_listing_names_each_group_and_role(tmp_path: Path) -> None:
    manifest = make_dataset(tmp_path / "ds").manifest_path
    empty = runner.invoke(app, ["splits", "list", "-m", str(manifest)])
    assert "No group has a role" in empty.output

    _ = runner.invoke(app, ["splits", "set", "-m", str(manifest), "test", "s3"])
    shown = runner.invoke(app, ["splits", "list", "-m", str(manifest)])
    assert "s3" in shown.output and "test" in shown.output


def test_an_unknown_role_is_refused_and_writes_nothing(tmp_path: Path) -> None:
    manifest = make_dataset(tmp_path / "ds").manifest_path
    before = manifest.read_text(encoding="utf-8")

    result = runner.invoke(
        app, ["splits", "set", "-m", str(manifest), "validation", "s1"]
    )

    assert result.exit_code == 1
    assert "train, test, sealed" in result.output
    assert manifest.read_text(encoding="utf-8") == before


def test_an_empty_group_is_refused(tmp_path: Path) -> None:
    manifest = make_dataset(tmp_path / "ds").manifest_path

    result = runner.invoke(app, ["splits", "set", "-m", str(manifest), "test", ""])

    assert result.exit_code == 1
    assert "empty group" in result.output


def test_removing_a_group_with_no_role_says_so(tmp_path: Path) -> None:
    manifest = make_dataset(tmp_path / "ds").manifest_path

    result = runner.invoke(app, ["splits", "remove", "-m", str(manifest), "s1"])

    assert result.exit_code == 1
    assert "has a role" in result.output
