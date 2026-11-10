"""``mosaic pipeline run`` of a recipe whose planar step reads world tracks it converts.

The conversion writes the tracks after submission, so the plan cannot see their
space. The planar step refuses at its start with ``schema_family_mismatch``, and
the command exits with the code reserved for a refusal.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.pipeline.refusal import REFUSED_EXIT_CODE
from tests.helpers import (
    WORLD_POINTS_FORMAT,
    raw_world_dataset,
    registered_world_points,
)

runner = CliRunner()

RECIPE = {
    "schema_version": 1,
    "steps": [
        {
            "id": "tracks",
            "type": "op",
            "kind": "convert-tracks",
            "params": {"source_format": WORLD_POINTS_FORMAT},
        },
        {
            "id": "pair",
            "type": "feature",
            "feature": "pair-egocentric",
            "inputs": ["tracks"],
            "params": {"pose": {"pose_n": 3}, "neck_idx": 1, "tail_base_idx": 2},
            "tracks": {"step": "tracks"},
        },
    ],
}


def test_run_exits_with_the_refusal_code_when_the_planar_step_starts(
    tmp_path: Path,
) -> None:
    recipe = tmp_path / "recipe.json"
    _ = recipe.write_text(json.dumps(RECIPE), encoding="utf-8")

    with registered_world_points():
        dataset = raw_world_dataset(tmp_path / "ds")
        result = runner.invoke(
            app,
            [
                "pipeline",
                "run",
                "--recipe",
                f"@{recipe}",
                "--manifest",
                str(dataset.manifest_path),
                "--json",
            ],
        )

    assert result.exit_code == REFUSED_EXIT_CODE, result.output
    payload = json.loads(result.stdout)
    assert payload["reason"] == "schema_family_mismatch"
