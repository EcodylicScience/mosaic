"""``mosaic arenas set`` and ``show``: a recording's layout from a file or a grid."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.arenas import ArenaKey, ArenaLayout, Rect
from mosaic.core.dataset import Dataset
from tests.helpers import add_track_sequences, invoke_json, invoke_ok, make_dataset


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, ("rec", "s1"), ("rec", "s2"), ("", "solo"))
    ds.set_continuous_groups(["rec"])
    return ds


def _manifest(ds: Dataset) -> list[str]:
    return ["--manifest", str(ds.manifest_path)]


def test_a_grid_is_set_for_a_divided_recording(dataset: Dataset) -> None:
    result = invoke_ok(
        ["arenas", "set", *_manifest(dataset), "-r", "rec", "--xs", "0,50,100"]
        + ["--ys", "0,40,80"]
    )

    found = dataset.arena_layouts()[ArenaKey("rec", "", "")]
    assert found.layout == ArenaLayout.grid(xs=(0, 50, 100), ys=(0, 40, 80))
    assert "Set 4 arenas of 'rec' (A1, A2, B1, B2)" in result.stdout


def test_a_strip_is_excluded_from_each_grid_arena_it_crosses(
    dataset: Dataset,
) -> None:
    _ = invoke_ok(
        ["arenas", "set", *_manifest(dataset), "-r", "rec", "--xs", "0,50,100"]
        + ["--ys", "0,40,80", "--exclude-x", "20,30", "--exclude-y", "75,80"]
    )

    found = dataset.arena_layouts()[ArenaKey("rec", "", "")]
    assert found.layout == ArenaLayout.grid(
        xs=(0, 50, 100),
        ys=(0, 40, 80),
        exclude={
            "A1": [Rect(x0=20, y0=0, x1=30, y1=40)],
            "A2": [
                Rect(x0=20, y0=40, x1=30, y1=80),
                Rect(x0=0, y0=75, x1=50, y1=80),
            ],
            "B2": [Rect(x0=50, y0=75, x1=100, y1=80)],
        },
    )


def test_a_layout_file_is_set_for_one_camera(dataset: Dataset, tmp_path: Path) -> None:
    layout = ArenaLayout.grid(xs=(0, 50), ys=(0, 40), capacity=None)
    path = tmp_path / "layout.json"
    _ = path.write_text(layout.model_dump_json())

    _ = invoke_ok(
        ["arenas", "set", *_manifest(dataset), "-r", "solo", "--layout", str(path)]
        + ["--camera", "top"]
    )

    listed = invoke_json(["arenas", "show", *_manifest(dataset), "--json"])
    assert listed == {
        "arena_layouts": [
            {
                "recording": "solo",
                "camera": "top",
                "arenas": ["A1"],
                "digest": layout.digest(),
            }
        ]
    }
    shown = invoke_json(
        ["arenas", "show", *_manifest(dataset), "-r", "solo", "--camera", "top"]
        + ["--json"]
    )
    assert shown["layout"] == layout.model_dump(mode="json")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["-r", "nowhere", "--xs", "0,10", "--ys", "0,10"], "no recording"),
        (["-r", "rec"], "give --layout"),
        (["-r", "rec", "--xs", "0,a", "--ys", "0,10"], "whole pixels"),
        (["-r", "rec", "--xs", "0,10", "--ys", "0,10", "--camera", "a/b"], "camera"),
        (["-r", "rec", "--xs", "0,10", "--ys", "0,10", "--exclude-x", "5"], "two"),
        (["-r", "rec", "--xs", "0,10", "--ys", "0,10", "--exclude-x", "8,2"], "lower"),
        (["-r", "rec", "--xs", "0,10", "--ys", "0,10", "--exclude-y", "5,5"], "lower"),
        (
            ["-r", "rec", "--xs", "0,10", "--ys", "0,10", "--exclude-x", "20,30"],
            "crosses no arena",
        ),
        (["-r", "rec", "--layout", "x.json", "--exclude-x", "1,2"], "grid"),
    ],
)
def test_a_refused_set_says_why_and_writes_nothing(
    dataset: Dataset, arguments: list[str], message: str
) -> None:
    result = CliRunner().invoke(app, ["arenas", "set", *_manifest(dataset), *arguments])

    assert result.exit_code == 1
    assert message in result.stderr
    assert dataset.arena_layouts() == {}


def test_show_names_a_recording_without_a_layout(dataset: Dataset) -> None:
    result = CliRunner().invoke(
        app, ["arenas", "show", *_manifest(dataset), "-r", "rec"]
    )

    assert result.exit_code == 1
    assert "no arena layout applies to 'rec'" in result.stderr
