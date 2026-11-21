"""A definition computes in the planar spaces unless its class declares ``track_spaces``.

``define`` reads no tracks, so it cannot tell a definition that its dataset's
world-frame tracks refuse. The plan reads them and refuses the step, and the
refusal names the declaration that changes the answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.definitions.define import define_feature
from mosaic.core.pipeline.graph import Recipe, StepRefused, plan_pipeline, read_recipe
from mosaic.core.pipeline.track_space import (
    TrackSpaceRefused,
    refuse_unsupported_track_space,
)
from tests.helpers import (
    MINIMAL_NAME,
    MINIMAL_SOURCE,
    add_tracks_table,
    make_dataset,
    world_track_table,
)

WORLD_VARIANT: Final = "convert-world.0.1-aaaaaaaaaa"
SPACES_LINE: Final = "    consumed_roots = ()\n"
WORLD_SOURCE: Final = MINIMAL_SOURCE.replace(
    SPACES_LINE, SPACES_LINE + '    track_spaces = frozenset({"world_m"})\n'
)


def _world_dataset(root: Path, source: str) -> Dataset:
    dataset = make_dataset(root)
    _ = define_feature(dataset, MINIMAL_NAME, source)
    add_tracks_table(
        dataset,
        WORLD_VARIANT,
        world_track_table("", "seq", n_rows=10, n_ids=2),
        std_format="mosaic_world_v1",
    )
    return dataset


def _recipe() -> Recipe:
    return read_recipe(
        {
            "steps": [
                {
                    "id": "step",
                    "type": "feature",
                    "feature": f"def:{MINIMAL_NAME}",
                    "inputs": ["tracks"],
                    "tracks": None,
                }
            ]
        }
    )


def test_a_definition_that_declares_no_spaces_is_refused_over_world_tracks(
    tmp_path: Path,
) -> None:
    dataset = _world_dataset(tmp_path / "default", MINIMAL_SOURCE)

    with pytest.raises(StepRefused) as raised:
        _ = plan_pipeline(dataset, _recipe())

    assert raised.value.reason == "schema_family_mismatch"
    assert "track_spaces" in str(raised.value)


def test_a_definition_declaring_the_world_space_plans_over_world_tracks(
    tmp_path: Path,
) -> None:
    dataset = _world_dataset(tmp_path / "world", WORLD_SOURCE)

    plan = plan_pipeline(dataset, _recipe(), refuse=False)

    assert plan.problems == ()


def test_the_refusal_of_a_definition_names_the_declaration_that_fixes_it() -> None:
    with pytest.raises(TrackSpaceRefused) as raised:
        refuse_unsupported_track_space(
            frozenset({"image_px", "plane_cm"}), "def:shift-partner", {"world_m"}
        )

    message = str(raised.value)
    assert "Declare `track_spaces` on the definition's class" in message
    assert "define it again" in message


def test_the_refusal_of_a_built_in_feature_does_not_mention_a_definition() -> None:
    with pytest.raises(TrackSpaceRefused) as raised:
        refuse_unsupported_track_space(
            frozenset({"image_px", "plane_cm"}), "pair-egocentric", {"world_m"}
        )

    assert "definition" not in str(raised.value)
