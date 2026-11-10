"""How the verbs report a planar feature handed a world-frame table.

A planar feature reads ``X`` and ``Y`` of a world-frame table and returns numbers
that look like measurements and are not. ``mosaic run`` refuses before the run
starts: it prints the refusal, exits with the code reserved for one, and writes no
run directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.refusal import REFUSED_EXIT_CODE

from tests.helpers import add_tracks_table, make_dataset, world_track_table

runner = CliRunner()


def _world_dataset(tmp_path: Path) -> Dataset:
    dataset = make_dataset(tmp_path / "ds")
    table = world_track_table("", "seq", n_rows=10, n_ids=2)
    add_tracks_table(
        dataset, "convert-x.0.1-aaaaaaaaaa", table, std_format="mosaic_world_v1"
    )
    return dataset


def test_a_planar_feature_is_a_refusal_not_a_traceback(tmp_path: Path) -> None:
    dataset = _world_dataset(tmp_path)

    result = runner.invoke(
        app, ["run", "--feature", "pair-egocentric", "-m", str(dataset.manifest_path)]
    )

    assert isinstance(result.exception, SystemExit), result.exception
    assert result.exit_code == REFUSED_EXIT_CODE
    assert "[mosaic] refused (schema_family_mismatch)" in result.stderr
    assert "world_m" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_planar_feature_is_a_json_refusal_naming_what_it_accepts(
    tmp_path: Path,
) -> None:
    dataset = _world_dataset(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "--feature",
            "pair-egocentric",
            "-m",
            str(dataset.manifest_path),
            "--json",
        ],
    )

    assert result.exit_code == REFUSED_EXIT_CODE
    payload = json.loads(result.stdout)
    assert payload["status"] == "refused"
    assert payload["reason"] == "schema_family_mismatch"
    assert payload["detail"] == {
        "feature": "pair-egocentric",
        "accepts": ["image_px", "plane_cm"],
        "found": ["world_m"],
    }


def test_a_refused_run_writes_no_feature_output(tmp_path: Path) -> None:
    dataset = _world_dataset(tmp_path)

    result = runner.invoke(
        app, ["run", "--feature", "pair-egocentric", "-m", str(dataset.manifest_path)]
    )

    assert result.exit_code == REFUSED_EXIT_CODE
    assert [p for p in dataset.get_root("features").rglob("*") if p.is_file()] == []
