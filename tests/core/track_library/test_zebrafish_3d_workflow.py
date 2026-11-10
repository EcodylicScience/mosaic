"""The ``zebrafish_3d_workflow_csv`` converter: a 3D workflow export into meters."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.schema import ensure_track_schema
from mosaic.core.track_converter import (
    TRACK_CONVERTERS,
    EntryHints,
    FrameRateUnknownError,
)
from mosaic.core.track_library.zebrafish_3d_workflow import (
    Zebrafish3dWorkflowConverter,
    Zebrafish3dWorkflowParams,
)
from tests.helpers import make_dataset, write_zebrafish_3d_csv

FRAMES = 12
FPS = 140.0


def _convert(
    path: Path,
    *,
    hints: EntryHints | None = None,
    **params: object,
) -> pd.DataFrame:
    overrides: dict[str, object] = {"fps": FPS, **params}
    return Zebrafish3dWorkflowConverter().convert(
        path,
        Zebrafish3dWorkflowParams.from_overrides(overrides),
        hints or EntryHints(group="wt", sequence="day1"),
    )


def _write(
    tmp_path: Path,
    *,
    fish: Sequence[int] = (1, 2),
    parts: Sequence[str] = ("head", "pec", "tail"),
    blank: Sequence[tuple[int, int, str | None]] = (),
) -> tuple[Path, npt.NDArray[np.float64]]:
    path = tmp_path / "zebrafish20240801_1428-pc2.csv"
    values = write_zebrafish_3d_csv(path, FRAMES, fish=fish, parts=parts, blank=blank)
    return path, values


def _pose(table: pd.DataFrame, fish_id: int, axis: str) -> npt.NDArray[np.float64]:
    rows = table[table["id"] == fish_id]
    return np.column_stack(
        [rows[f"pose{axis}{k}"].to_numpy(dtype=np.float64) for k in range(3)]
    )


# --- registration and the table -----------------------------------------------


def test_the_converter_declares_the_world_schema() -> None:
    converter = TRACK_CONVERTERS["zebrafish_3d_workflow_csv"]

    assert converter is Zebrafish3dWorkflowConverter
    assert converter.output_schema == "mosaic_world_v1"
    assert converter.carries_tracklets is None
    assert not converter.enumerable
    assert not converter.merges_per_sequence


def test_a_converted_table_validates_against_the_world_schema(tmp_path: Path) -> None:
    path, _ = _write(tmp_path)

    table = _convert(path)
    _, report = ensure_track_schema(
        table, "mosaic_world_v1", strict=True, source=str(path)
    )

    assert report["missing_required"] == []
    assert len(table) == FRAMES * 2
    assert not any(column.startswith("poseP") for column in table.columns)


def test_ids_are_the_fish_numbers_and_keypoints_follow_header_order(
    tmp_path: Path,
) -> None:
    path, values = _write(tmp_path, fish=(3, 7), parts=("tail", "head", "pec"))

    table = _convert(path)

    assert sorted(set(table["id"])) == [3, 7]
    for index, fish in enumerate((3, 7)):
        for axis_index, axis in enumerate("XYZ"):
            np.testing.assert_allclose(
                _pose(table, fish, axis),
                values[:, index, :, axis_index] * 0.01,
            )


def test_rows_are_sorted_by_frame_then_id_and_carry_the_hints(tmp_path: Path) -> None:
    path, _ = _write(tmp_path, fish=(2, 1))

    table = _convert(path)

    assert list(table["frame"]) == [frame for frame in range(FRAMES) for _ in (1, 2)]
    assert list(table["id"]) == [1, 2] * FRAMES
    assert set(table["group"]) == {"wt"}
    assert set(table["sequence"]) == {"day1"}


def test_the_sequence_falls_back_to_the_file_stem(tmp_path: Path) -> None:
    path, _ = _write(tmp_path)

    table = _convert(path, hints=EntryHints())

    assert set(table["sequence"]) == {"zebrafish20240801_1428-pc2"}
    assert set(table["group"]) == {""}


# --- units and axes -----------------------------------------------------------


def test_values_are_meters_under_the_default_unit(tmp_path: Path) -> None:
    path, values = _write(tmp_path)

    table = _convert(path)

    first = table[table["id"] == 1]
    np.testing.assert_allclose(first["poseX0"], values[:, 0, 0, 0] / 100)
    np.testing.assert_allclose(first["poseZ2"], values[:, 0, 2, 2] / 100)


def test_a_unit_factor_rescales_every_value(tmp_path: Path) -> None:
    path, values = _write(tmp_path)

    table = _convert(path, meters_per_unit=1.0)

    np.testing.assert_allclose(table[table["id"] == 2]["poseY1"], values[:, 1, 1, 1])


def test_axes_map_source_axes_to_world_axes(tmp_path: Path) -> None:
    path, values = _write(tmp_path)

    table = _convert(path, axes=("x", "-z", "y"))

    first = table[table["id"] == 1]
    source_x, source_y, source_z = (values[:, 0, :, axis] for axis in range(3))
    np.testing.assert_allclose(_pose(first, 1, "X"), source_x * 0.01)
    np.testing.assert_allclose(_pose(first, 1, "Y"), -source_z * 0.01)
    np.testing.assert_allclose(_pose(first, 1, "Z"), source_y * 0.01)
    np.testing.assert_allclose(first["Z"], source_y.mean(axis=1) * 0.01)


@pytest.mark.parametrize(
    "axes",
    [
        ("x", "z", "y"),
        ("-x", "y", "z"),
        ("-x", "-y", "-z"),
    ],
)
def test_a_reflection_is_refused_at_construction(axes: tuple[str, str, str]) -> None:
    with pytest.raises(ValidationError, match="reflection"):
        _ = Zebrafish3dWorkflowParams(axes=axes)


@pytest.mark.parametrize("axes", [("x", "-x", "y"), ("x", "y", "y"), ("-z", "x", "z")])
def test_a_repeated_axis_is_refused_at_construction(axes: tuple[str, str, str]) -> None:
    with pytest.raises(ValidationError, match="more than once"):
        _ = Zebrafish3dWorkflowParams(axes=axes)


@pytest.mark.parametrize("axes", [("x", "y", "w"), ("x", "y", "--z"), ("x", "y", "")])
def test_an_unknown_axis_is_refused_at_construction(axes: tuple[str, str, str]) -> None:
    with pytest.raises(ValidationError, match="axis"):
        _ = Zebrafish3dWorkflowParams(axes=axes)


@pytest.mark.parametrize(
    "axes",
    [
        ("x", "y", "z"),
        ("x", "-z", "y"),
        ("-x", "-y", "z"),
        ("y", "-x", "z"),
        ("y", "x", "-z"),
    ],
)
def test_a_proper_rotation_is_accepted(axes: tuple[str, str, str]) -> None:
    assert Zebrafish3dWorkflowParams(axes=axes).axes == axes


def test_a_unit_factor_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="meters_per_unit"):
        _ = Zebrafish3dWorkflowParams(meters_per_unit=0.0)


# --- gaps and the body center -------------------------------------------------


def test_an_all_empty_fish_frame_has_no_row_and_a_partly_empty_one_keeps_it(
    tmp_path: Path,
) -> None:
    path, _ = _write(
        tmp_path, blank=[(3, 1, None), (5, 2, "pec"), (6, 1, "head"), (6, 1, "tail")]
    )

    table = _convert(path)

    assert len(table) == FRAMES * 2 - 1
    assert table[(table["frame"] == 3) & (table["id"] == 1)].empty
    assert not table[(table["frame"] == 3) & (table["id"] == 2)].empty
    partly = table[(table["frame"] == 5) & (table["id"] == 2)].iloc[0]
    assert np.isnan(partly["poseX1"])
    assert not np.isnan(partly["poseX0"])
    mostly = table[(table["frame"] == 6) & (table["id"] == 1)].iloc[0]
    assert np.isnan(mostly["poseZ0"])
    assert np.isnan(mostly["poseZ2"])
    assert not np.isnan(mostly["poseZ1"])


def test_the_body_center_is_the_mean_of_the_complete_keypoints(tmp_path: Path) -> None:
    path, values = _write(tmp_path, blank=[(4, 1, "pec")])

    table = _convert(path)

    first = table[table["id"] == 1].set_index("frame")
    np.testing.assert_allclose(
        first.loc[0, ["X", "Y", "Z"]].to_numpy(dtype=np.float64),
        values[0, 0].mean(axis=0) * 0.01,
    )
    np.testing.assert_allclose(
        first.loc[4, ["X", "Y", "Z"]].to_numpy(dtype=np.float64),
        values[4, 0][[0, 2]].mean(axis=0) * 0.01,
    )


def test_a_keypoint_with_an_empty_axis_is_left_out_of_the_center(
    tmp_path: Path,
) -> None:
    path, values = _write(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    cells = lines[3].split(",")
    cells[header.index("fish1_pec_y")] = ""
    lines[3] = ",".join(cells)
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    row = _convert(path).query("frame == 2 and id == 1").iloc[0]

    assert np.isnan(row["poseY1"])
    assert row["poseX1"] == pytest.approx(values[2, 0, 1, 0] * 0.01)
    np.testing.assert_allclose(
        [row["X"], row["Y"], row["Z"]],
        values[2, 0][[0, 2]].mean(axis=0) * 0.01,
    )


def test_a_frame_with_no_complete_keypoint_has_no_center(tmp_path: Path) -> None:
    path, _ = _write(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    cells = lines[2].split(",")
    for column in ("fish1_head_x", "fish1_pec_y", "fish1_tail_z"):
        cells[header.index(column)] = ""
    lines[2] = ",".join(cells)
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    row = _convert(path).query("frame == 1 and id == 1").iloc[0]

    assert np.isnan([row["X"], row["Y"], row["Z"]]).all()


def test_a_file_with_nothing_but_gaps_gives_an_empty_valid_table(
    tmp_path: Path,
) -> None:
    path, _ = _write(
        tmp_path,
        blank=[(frame, number, None) for frame in range(FRAMES) for number in (1, 2)],
    )

    table = _convert(path)

    assert table.empty
    assert {"frame", "time", "id", "X", "Y", "Z", "group", "sequence"} <= set(
        table.columns
    )
    assert "poseZ2" in table.columns


# --- time ---------------------------------------------------------------------


def test_time_is_frame_over_the_explicit_rate(tmp_path: Path) -> None:
    path, _ = _write(tmp_path)

    table = _convert(path)

    np.testing.assert_allclose(
        table["time"].to_numpy(dtype=np.float64),
        table["frame"].to_numpy(dtype=np.float64) / 140.0,
    )


def test_time_follows_the_measured_rate_when_fps_is_unset(tmp_path: Path) -> None:
    path, _ = _write(tmp_path)

    table = _convert(path, hints=EntryHints(fps=70.0), fps=None)

    np.testing.assert_allclose(
        table["time"].to_numpy(dtype=np.float64),
        table["frame"].to_numpy(dtype=np.float64) / 70.0,
    )


def test_a_conversion_with_no_rate_is_refused(tmp_path: Path) -> None:
    path, _ = _write(tmp_path)

    with pytest.raises(FrameRateUnknownError, match="fps_default"):
        _ = _convert(path, fps=None)


# --- the header ---------------------------------------------------------------


def _with_header(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    _ = path.write_text(text.replace(old, new, 1), encoding="utf-8")


@pytest.mark.parametrize(
    ("old", "new", "named"),
    [
        ("frame_index", "frame", "frame_index"),
        ("fish1_head_x", "fish1_head_u", "fish1_head_u"),
        ("fish2_tail_z", "notes", "notes"),
        ("fish1_head_x", "Fish1_head_x", "Fish1_head_x"),
    ],
    ids=["no-frame-index", "stray-axis", "stray-column", "wrong-case"],
)
def test_a_header_outside_the_format_is_refused(
    tmp_path: Path, old: str, new: str, named: str
) -> None:
    path, _ = _write(tmp_path)
    _with_header(path, old, new)

    with pytest.raises(ValueError, match=named) as raised:
        _ = _convert(path)

    assert path.name in str(raised.value)


def test_a_keypoint_missing_an_axis_is_refused(tmp_path: Path) -> None:
    path, _ = _write(tmp_path)
    _with_header(path, "fish1_pec_z", "fish1_pec_x")

    with pytest.raises(ValueError, match="fish1_pec") as raised:
        _ = _convert(path)

    assert path.name in str(raised.value)


def test_fish_with_different_parts_are_refused(tmp_path: Path) -> None:
    path, _ = _write(tmp_path)
    _with_header(path, "fish2_tail_x", "fish2_fin_x")
    _with_header(path, "fish2_tail_y", "fish2_fin_y")
    _with_header(path, "fish2_tail_z", "fish2_fin_z")

    with pytest.raises(ValueError, match="fish 2") as raised:
        _ = _convert(path)

    assert path.name in str(raised.value)


def test_fish_with_parts_in_another_order_are_refused(tmp_path: Path) -> None:
    path, _ = _write(tmp_path)
    text = path.read_text(encoding="utf-8").splitlines()
    header = text[0].split(",")
    swapped = [
        column.replace("fish2_head", "fish2_X").replace("fish2_pec", "fish2_head")
        for column in header
    ]
    text[0] = ",".join(swapped).replace("fish2_X", "fish2_pec")
    _ = path.write_text("\n".join(text) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fish 2"):
        _ = _convert(path)


def test_a_repeated_column_is_refused(tmp_path: Path) -> None:
    path, _ = _write(tmp_path)
    _with_header(path, "fish2_head_x", "fish1_head_x")

    with pytest.raises(ValueError, match="fish1_head_x"):
        _ = _convert(path)


def test_an_empty_frame_index_is_refused(tmp_path: Path) -> None:
    path, _ = _write(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[3] = "," + lines[3].split(",", 1)[1]
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frame_index"):
        _ = _convert(path)


def test_a_file_with_no_header_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    _ = path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty.csv"):
        _ = _convert(path)


# --- determinism and the dataset ----------------------------------------------


def test_one_file_converts_to_equal_frames_twice(tmp_path: Path) -> None:
    path, _ = _write(tmp_path, blank=[(2, 1, None), (7, 2, "head")])

    assert_frame_equal(_convert(path), _convert(path))


def test_the_axes_enter_the_variant_identity() -> None:
    plain = Zebrafish3dWorkflowParams().identity_dump()
    turned = Zebrafish3dWorkflowParams(axes=("x", "-z", "y")).identity_dump()

    assert plain["axes"] != turned["axes"]
    assert plain["meters_per_unit"] == 0.01
    assert "fps" not in plain


def test_a_dataset_converts_a_file_with_its_default_rate(tmp_path: Path) -> None:
    dataset = make_dataset((tmp_path / "ds").resolve(), roots=("tracks_raw", "tracks"))
    dataset.meta["fps_default"] = FPS
    _ = write_zebrafish_3d_csv(
        dataset.base_dir / "raw" / "zebrafish20240801_1428-pc2.csv", FRAMES
    )
    _ = dataset.index_tracks_raw(
        [dataset.base_dir / "raw"],
        patterns=["*.csv"],
        src_format="zebrafish_3d_workflow_csv",
    )

    _ = dataset.convert_all_tracks()

    index = read_tracks_index(dataset)
    assert len(index) == 1
    row = index.iloc[0]
    assert row["std_format"] == "mosaic_world_v1"
    assert row["sequence"] == "zebrafish20240801_1428-pc2"
    table = dataset.load_tracks("", "zebrafish20240801_1428-pc2")
    assert len(table) == FRAMES * 2
    np.testing.assert_allclose(
        table["time"].to_numpy(dtype=np.float64),
        table["frame"].to_numpy(dtype=np.float64) / FPS,
    )
