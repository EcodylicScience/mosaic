"""``plan_recipe`` previews the refusal of a planar step over world-frame tracks.

The preview plans without refusing, so the refusal a submission would raise
travels as a problem at the step's ``inputs``.
"""

from __future__ import annotations

from pathlib import Path

from mosaic.agent.mcp.local import LocalBackend
from mosaic.agent.mcp.spec import call_tool
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import PlanWire
from tests.helpers import (
    RecordingStarter,
    add_tracks_table,
    make_dataset,
    world_track_table,
)

RECIPE: dict[str, JsonValue] = {
    "schema_version": 1,
    "name": "planar-over-world",
    "steps": [
        {
            "id": "pair",
            "type": "feature",
            "feature": "pair-egocentric",
            "inputs": ["tracks"],
            "params": {"pose": {"pose_n": 3}, "neck_idx": 1, "tail_base_idx": 2},
        }
    ],
}


def test_the_preview_shows_the_space_refusal_as_a_problem(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds")
    add_tracks_table(
        dataset,
        "convert-world.0.1-aaaaaaaaaa",
        world_track_table("", "seq", n_rows=10, n_ids=2),
        std_format="mosaic_world_v1",
    )
    backend = LocalBackend(
        [dataset.manifest_path], warm=False, start_driver=RecordingStarter()
    )

    answered = call_tool(
        backend,
        "plan_recipe",
        {"dataset": dataset.manifest.uuid, "recipe": RECIPE},
    )

    assert isinstance(answered, PlanWire), answered
    assert [(problem.step, problem.where) for problem in answered.problems] == [
        ("pair", "inputs")
    ]
    assert "world_m" in answered.problems[0].message
