"""The synthetic track file is a validated description of a generator run.

One TOML file holds every generator parameter, so the file's checksum identifies
the sequence it produces. The reader refuses a file the generator could not honor
before any frame is drawn, and each refusal names the field it concerns.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

import pytest

import mosaic.core.track_library as track_library
from mosaic.core.track_library.synthetic_spec import (
    MAX_TABLE_CELLS,
    SYNTHETIC_FORMAT,
    SyntheticSpec,
    read_synthetic_spec,
)
from tests.helpers import write_synthetic_spec


def refusal(tmp_path: Path, **overrides: object) -> str:
    """The message of the ValueError that reading a file with *overrides* raises."""
    path = write_synthetic_spec(tmp_path / "walk.toml", **overrides)
    with pytest.raises(ValueError) as caught:
        read_synthetic_spec(path)
    return str(caught.value)


def test_a_minimal_file_reads_with_every_default(tmp_path: Path) -> None:
    spec = read_synthetic_spec(write_synthetic_spec(tmp_path / "walk.toml"))

    assert spec.format == SYNTHETIC_FORMAT == "mosaic-synthetic/1"
    assert (spec.seed, spec.dimensions, spec.frames, spec.fps) == (7, 2, 200, 30.0)
    assert spec.individuals == 3
    assert spec.burn_in_frames == 0
    assert spec.arena.shape == "box"
    assert (spec.arena.width, spec.arena.height) == (1000.0, 1000.0)
    assert spec.arena.radius is None
    assert spec.arena.depth is None
    assert spec.arena.start_radius is None
    assert spec.motion.mean_speed == 100.0
    assert spec.motion.speed_sd == 0.0
    assert spec.motion.speed_tau_s == 1.0
    assert spec.motion.turning_sd == 0.5
    assert spec.zones is None
    assert spec.body is None
    assert spec.dropouts is None


def test_every_optional_table_reads_with_its_defaults(tmp_path: Path) -> None:
    spec = read_synthetic_spec(
        write_synthetic_spec(
            tmp_path / "walk.toml",
            zones={"repulsion": 10.0, "orientation": 60.0, "attraction": 150.0},
            body={},
            dropouts={},
        )
    )

    assert spec.zones is not None
    assert (spec.zones.field_of_view_deg, spec.zones.max_turn_rate) == (270.0, 6.0)
    assert spec.body is not None
    assert spec.body.keypoints == 0
    assert spec.body.length is None
    assert spec.body.keypoint_sd == 0.0
    assert spec.body.confidence_sd == 0.05
    assert spec.dropouts is not None
    assert (spec.dropouts.id_rate, spec.dropouts.id_mean_frames) == (0.0, 10.0)
    assert (spec.dropouts.gap_rate, spec.dropouts.gap_mean_frames) == (0.0, 5.0)


def test_a_full_file_reads_every_table(tmp_path: Path) -> None:
    spec = read_synthetic_spec(
        write_synthetic_spec(
            tmp_path / "walk.toml",
            dimensions=3,
            burn_in_frames=50,
            arena={
                "shape": "disk",
                "width": None,
                "height": None,
                "radius": 500.0,
                "depth": 50.0,
                "start_radius": 100.0,
            },
            motion={"speed_sd": 30.0, "speed_tau_s": 2.0, "turning_sd": 0.25},
            zones={"repulsion": 5.0, "orientation": 5.0, "attraction": 5.0},
            body={"keypoints": 5, "length": 30.0, "keypoint_sd": 1.0},
            dropouts={"id_rate": 0.01, "id_mean_frames": 4.0, "gap_rate": 0.02},
        )
    )

    assert spec.arena.shape == "disk"
    assert (spec.arena.radius, spec.arena.depth, spec.arena.start_radius) == (
        500.0,
        50.0,
        100.0,
    )
    assert spec.motion.max_speed == 100.0 + 4 * 30.0
    assert spec.body is not None and spec.body.keypoints == 5
    assert spec.dropouts is not None and spec.dropouts.gap_rate == 0.02


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"format": "mosaic-synthetic/2"}, "format"),
        ({"arena": {"shape": "disk", "radius": 400.0}}, "width"),
        ({"arena": {"shape": "disk", "width": None, "height": None}}, "radius"),
        ({"arena": {"radius": 400.0}}, "radius"),
        ({"arena": {"width": None}}, "width"),
        ({"arena": {"depth": 0.5}}, "depth"),
        ({"dimensions": 3}, "depth"),
        (
            {"zones": {"repulsion": 0.0, "orientation": 60.0, "attraction": 150.0}},
            "repulsion",
        ),
        (
            {"zones": {"repulsion": 70.0, "orientation": 60.0, "attraction": 150.0}},
            "orientation",
        ),
        (
            {"zones": {"repulsion": 10.0, "orientation": 60.0, "attraction": 50.0}},
            "attraction",
        ),
        ({"arena": {"start_radius": 501.0}}, "start_radius"),
        (
            {
                "arena": {
                    "shape": "disk",
                    "width": None,
                    "height": None,
                    "radius": 400.0,
                    "start_radius": 401.0,
                }
            },
            "start_radius",
        ),
        ({"motion": {"mean_speed": 40_000.0}}, "mean_speed"),
        ({"body": {"keypoints": 3}}, "length"),
        ({"body": {"keypoints": 33, "length": 30.0}}, "keypoints"),
        ({"individuals": 0}, "individuals"),
        ({"individuals": 301}, "individuals"),
        ({"frames": 0}, "frames"),
        ({"fps": 0.0}, "fps"),
        ({"burn_in_frames": -1}, "burn_in_frames"),
        ({"dropouts": {"id_rate": 1.0}}, "id_rate"),
        ({"dropouts": {"gap_rate": -0.1}}, "gap_rate"),
        ({"dropouts": {"id_mean_frames": 0.5}}, "id_mean_frames"),
        ({"dropouts": {"gap_mean_frames": 0.0}}, "gap_mean_frames"),
        ({"motion": {"speed_tau_s": 0.0}}, "speed_tau_s"),
    ],
)
def test_a_refused_file_names_its_field(
    tmp_path: Path, overrides: dict[str, object], field: str
) -> None:
    assert field in refusal(tmp_path, **overrides)


def test_a_file_over_the_cell_cap_is_refused(tmp_path: Path) -> None:
    frames = MAX_TABLE_CELLS // (100 * 5) + 1
    message = refusal(tmp_path, frames=frames, individuals=100)

    assert "frames" in message
    assert "individuals" in message


def test_the_cell_cap_counts_every_numeric_column(tmp_path: Path) -> None:
    centroid = read_synthetic_spec(write_synthetic_spec(tmp_path / "a.toml"))
    five = read_synthetic_spec(
        write_synthetic_spec(tmp_path / "b.toml", body={"keypoints": 5, "length": 30.0})
    )
    world = read_synthetic_spec(
        write_synthetic_spec(
            tmp_path / "c.toml",
            dimensions=3,
            arena={"depth": 50.0},
            body={"keypoints": 2, "length": 30.0},
        )
    )

    assert centroid.numeric_columns == 5
    assert five.numeric_columns == 5 + 5 * 3
    assert world.numeric_columns == 6 + 2 * 4


def test_a_file_at_the_cell_cap_is_accepted(tmp_path: Path) -> None:
    # 5 columns * 100 ids * frames == MAX_TABLE_CELLS exactly.
    frames = MAX_TABLE_CELLS // (100 * 5)
    assert frames * 100 * 5 == MAX_TABLE_CELLS

    spec = read_synthetic_spec(
        write_synthetic_spec(tmp_path / "walk.toml", frames=frames, individuals=100)
    )

    assert spec.frames * spec.individuals * spec.numeric_columns == MAX_TABLE_CELLS


def test_the_step_limit_is_inclusive_of_the_smallest_extent(tmp_path: Path) -> None:
    # max_speed / fps == 100 / 1 == the smallest extent: accepted.
    read_synthetic_spec(
        write_synthetic_spec(
            tmp_path / "walk.toml",
            fps=1.0,
            arena={"width": 100.0, "height": 400.0},
            motion={"mean_speed": 100.0},
        )
    )
    assert "mean_speed" in refusal(
        tmp_path,
        fps=1.0,
        arena={"width": 99.0, "height": 400.0},
        motion={"mean_speed": 100.0},
    )


def test_the_step_limit_counts_the_speed_spread_and_the_depth(tmp_path: Path) -> None:
    assert "mean_speed" in refusal(
        tmp_path,
        fps=1.0,
        arena={"width": 200.0, "height": 200.0},
        motion={"mean_speed": 100.0, "speed_sd": 30.0},
    )
    assert "mean_speed" in refusal(
        tmp_path,
        dimensions=3,
        fps=1.0,
        arena={"width": 200.0, "height": 200.0, "depth": 50.0},
        motion={"mean_speed": 100.0},
    )


@pytest.mark.parametrize("table", ["arena", "motion", "zones", "body", "dropouts"])
def test_an_unknown_key_is_refused_in_each_table(tmp_path: Path, table: str) -> None:
    tables: dict[str, dict[str, object]] = {
        "zones": {"repulsion": 1.0, "orientation": 2.0, "attraction": 3.0},
        "body": {},
        "dropouts": {},
    }
    tables.setdefault(table, {})["colour"] = 1

    assert "colour" in refusal(tmp_path, **tables)


def test_an_unknown_top_level_key_is_refused(tmp_path: Path) -> None:
    assert "colour" in refusal(tmp_path, colour=1)


def test_a_syntax_error_names_the_path(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("frames = = 3\n")

    with pytest.raises(ValueError, match=str(path).replace("\\", "\\\\")):
        read_synthetic_spec(path)


def test_a_model_refusal_names_the_path(tmp_path: Path) -> None:
    path = write_synthetic_spec(tmp_path / "walk.toml", individuals=0)

    with pytest.raises(ValueError, match="walk.toml"):
        read_synthetic_spec(path)


def test_the_spec_model_is_the_file_schema() -> None:
    assert set(SyntheticSpec.model_fields) == {
        "format",
        "seed",
        "dimensions",
        "frames",
        "fps",
        "individuals",
        "burn_in_frames",
        "arena",
        "motion",
        "zones",
        "body",
        "dropouts",
    }


def test_the_written_helper_file_reads_back_as_given(tmp_path: Path) -> None:
    path = write_synthetic_spec(
        tmp_path / "walk.toml",
        seed=11,
        arena={"width": 250.5},
        zones={"repulsion": 1.0, "orientation": 2.0, "attraction": 3.0},
        body={"keypoints": 4, "length": 12.0},
    )

    assert tomllib.loads(path.read_text()) == {
        "format": "mosaic-synthetic/1",
        "seed": 11,
        "dimensions": 2,
        "frames": 200,
        "fps": 30.0,
        "individuals": 3,
        "arena": {"shape": "box", "width": 250.5, "height": 1000.0},
        "motion": {"mean_speed": 100.0},
        "zones": {"repulsion": 1.0, "orientation": 2.0, "attraction": 3.0},
        "body": {"keypoints": 4, "length": 12.0},
    }


SPEC_MODULE = "mosaic.core.track_library.synthetic_spec"
LIBRARY = Path(track_library.__file__).parent
ALLOWED_THIRD_PARTY = {
    "synthetic_spec.py": {"pydantic", "mosaic.core.strict_model"},
    "synthetic_motion.py": {"numpy", SPEC_MODULE},
}


@pytest.mark.parametrize("name", list(ALLOWED_THIRD_PARTY))
def test_the_module_level_imports_stay_inside_the_allowed_modules(name: str) -> None:
    tree = ast.parse((LIBRARY / name).read_text())
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{name} imports relatively"
            imported.add(node.module or "")

    allowed = ALLOWED_THIRD_PARTY[name] | set(sys.stdlib_module_names) | {"__future__"}
    outside = {
        module
        for module in imported
        if module not in allowed and module.split(".")[0] not in allowed
    }
    assert not outside, f"{name} imports {sorted(outside)} at module level"


def test_a_none_override_of_a_table_removes_the_table(tmp_path: Path) -> None:
    path = write_synthetic_spec(tmp_path / "walk.toml", motion=None)

    assert "motion" not in tomllib.loads(path.read_text())
    with pytest.raises(ValueError, match="motion"):
        read_synthetic_spec(path)
