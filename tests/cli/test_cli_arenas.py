"""``mosaic arenas set`` and ``show``: a recording's layout from a file, a grid or a circle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.arenas import ArenaKey, ArenaLayout, Rect
from mosaic.core.dataset import Dataset
from mosaic.core.regions import CircleShape, RectShape, Roi
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
                "assignment_digest": layout.assignment_digest(),
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


def _rois_file(tmp_path: Path, rois: list[Roi]) -> Path:
    path = tmp_path / "rois.json"
    _ = path.write_text(
        json.dumps([roi.model_dump(mode="json") for roi in rois]), encoding="utf-8"
    )
    return path


CENTER_DISC = Roi(name="center", shape=CircleShape(cx=0.0, cy=0.0, r=5.0))
NEST = Roi(
    name="nest",
    shape=RectShape(x0=2.0, y0=2.0, x1=8.0, y1=8.0),
)


def test_a_circle_is_set_as_a_one_arena_layout(dataset: Dataset) -> None:
    result = invoke_ok(
        ["arenas", "set", *_manifest(dataset), "-r", "rec", "--circle", "50,40,30"]
    )

    found = dataset.arena_layouts()[ArenaKey("rec", "", "")]
    assert found.layout == ArenaLayout.single(CircleShape(cx=50, cy=40, r=30))
    assert "Set 1 arenas of 'rec' (A1)" in result.stdout


def test_a_circle_past_the_frame_edge_is_clipped_to_the_frame(
    dataset: Dataset,
) -> None:
    _ = invoke_ok(
        ["arenas", "set", *_manifest(dataset), "-r", "rec", "--circle", "10.5,40,30"]
        + ["--capacity", "0"]
    )

    layout = dataset.arena_layouts()[ArenaKey("rec", "", "")].layout
    arena = layout.arenas[0]
    assert arena.boundary == CircleShape(cx=10.5, cy=40, r=30)
    assert arena.region == Rect(x0=0, y0=10, x1=41, y1=70)
    assert arena.capacity is None
    assert (layout.width, layout.height) == (41, 70)


def test_rois_are_given_to_every_arena_of_a_grid(
    dataset: Dataset, tmp_path: Path
) -> None:
    rois = _rois_file(tmp_path, [CENTER_DISC, NEST])

    _ = invoke_ok(
        ["arenas", "set", *_manifest(dataset), "-r", "rec", "--xs", "0,50,100"]
        + ["--ys", "0,40,80", "--rois", str(rois)]
    )

    layout = dataset.arena_layouts()[ArenaKey("rec", "", "")].layout
    assert [arena.rois for arena in layout.arenas] == [(CENTER_DISC, NEST)] * 4
    bare = ArenaLayout.grid(xs=(0, 50, 100), ys=(0, 40, 80))
    assert [arena.region for arena in layout.arenas] == [
        arena.region for arena in bare.arenas
    ]


def test_rois_are_given_to_a_circle(dataset: Dataset, tmp_path: Path) -> None:
    rois = _rois_file(tmp_path, [CENTER_DISC])

    _ = invoke_ok(
        ["arenas", "set", *_manifest(dataset), "-r", "rec", "--circle", "50,40,30"]
        + ["--rois", str(rois)]
    )

    layout = dataset.arena_layouts()[ArenaKey("rec", "", "")].layout
    assert layout == ArenaLayout.single(
        CircleShape(cx=50, cy=40, r=30), rois=[CENTER_DISC]
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--layout", "x.json", "--rois", "{rois}"], "--rois applies to a grid"),
        (["--circle", "50,40,30", "--xs", "0,10", "--ys", "0,10"], "--circle"),
        (["--circle", "50,40,30", "--layout", "x.json"], "--circle"),
        (["--circle", "50,40,30", "--exclude-x", "1,2"], "apply to a grid"),
        (["--circle", "50,40"], "--circle takes"),
        (["--circle", "50,40,a"], "--circle takes"),
        (["--circle", "50,40,0"], "greater than 0"),
        (["--circle", "-50,40,10"], "no part inside the frame"),
        (["--xs", "0,10", "--ys", "0,10", "--rois", "missing.json"], "missing.json"),
        (["--xs", "0,10", "--ys", "0,10", "--rois", "{bad}"], "Invalid JSON"),
        (["--xs", "0,10", "--ys", "0,10", "--rois", "{repeated}"], "repeats ROI names"),
    ],
)
def test_a_refused_circle_or_rois_says_why_and_writes_nothing(
    dataset: Dataset, tmp_path: Path, arguments: list[str], message: str
) -> None:
    files = {
        "{rois}": _rois_file(tmp_path, [CENTER_DISC]),
        "{repeated}": _rois_file(tmp_path, [CENTER_DISC, CENTER_DISC]),
        "{bad}": tmp_path / "bad.json",
    }
    _ = files["{bad}"].write_text("[", encoding="utf-8")
    resolved = [str(files[item]) if item in files else item for item in arguments]

    result = CliRunner().invoke(
        app, ["arenas", "set", *_manifest(dataset), "-r", "rec", *resolved]
    )

    assert result.exit_code == 1
    assert message in result.stderr
    assert dataset.arena_layouts() == {}


def test_show_prints_each_arenas_boundary_and_rois(
    dataset: Dataset, tmp_path: Path
) -> None:
    rois = _rois_file(tmp_path, [CENTER_DISC, NEST])
    _ = invoke_ok(
        ["arenas", "set", *_manifest(dataset), "-r", "rec", "--circle", "50,40,30"]
        + ["--rois", str(rois)]
    )
    _ = invoke_ok(
        ["arenas", "set", *_manifest(dataset), "-r", "solo", "--xs", "0,50"]
        + ["--ys", "0,40"]
    )

    circle = invoke_ok(["arenas", "show", *_manifest(dataset), "-r", "rec"])
    plain = invoke_ok(["arenas", "show", *_manifest(dataset), "-r", "solo"])

    assert "boundary circle" in circle.stdout
    assert "rois center, nest" in circle.stdout
    assert "boundary rect" in plain.stdout
    assert "rois" not in plain.stdout


def test_show_json_prints_the_boundary_and_rois(
    dataset: Dataset, tmp_path: Path
) -> None:
    rois = _rois_file(tmp_path, [CENTER_DISC])
    _ = invoke_ok(
        ["arenas", "set", *_manifest(dataset), "-r", "rec", "--circle", "50,40,30"]
        + ["--rois", str(rois)]
    )

    shown = invoke_json(["arenas", "show", *_manifest(dataset), "-r", "rec", "--json"])

    layout = ArenaLayout.single(CircleShape(cx=50, cy=40, r=30), rois=[CENTER_DISC])
    assert shown["layout"] == layout.model_dump(mode="json")
    arena = ArenaLayout.model_validate(shown["layout"]).arenas[0]
    assert arena.boundary == CircleShape(cx=50, cy=40, r=30)
    assert [roi.name for roi in arena.rois] == ["center"]
