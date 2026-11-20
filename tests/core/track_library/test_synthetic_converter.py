"""The ``synthetic`` and ``synthetic_3d`` converters write simulated tracks.

A synthetic track file declares every generator parameter. A conversion is a pure
function of the file and the id policy. The converters read the file, run the
simulation and hand the table to ``convert_all_tracks``, which writes it and records
its measures in the tracks index.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.pipeline.tracks_index import (
    frame_measures,
    read_carries_tracklets,
    read_ids_per_frame,
    read_pair_frames,
    read_tracks_index,
)
from mosaic.core.schema import ensure_track_schema
from mosaic.core.track_converter import TRACK_CONVERTERS, EntryHints
from mosaic.core.track_library import synthetic
from mosaic.core.track_library.synthetic import (
    Synthetic3dConverter,
    SyntheticConverter,
    SyntheticParams,
)
from mosaic.core.track_library.synthetic_motion import SyntheticColumns, simulate
from mosaic.core.track_library.synthetic_spec import SyntheticSpec
from tests.helpers import (
    SYNTHETIC_ARENA_3D,
    SYNTHETIC_DROPOUTS,
    invoke_json,
    make_dataset,
    synthetic_dataset,
    write_synthetic_spec,
)

BODY = {"keypoints": 3, "length": 40.0, "keypoint_sd": 1.0}


def _convert(
    converter: SyntheticConverter | Synthetic3dConverter,
    path: Path,
    *,
    relabel_after_gap: bool = False,
    hints: EntryHints | None = None,
) -> pd.DataFrame:
    params = SyntheticParams(relabel_after_gap=relabel_after_gap)
    return converter.convert(path, params, hints or EntryHints())


def _pairs(table: pd.DataFrame) -> tuple[int, float]:
    """``sum n_f (n_f - 1)`` and the mean ids per frame, counted from the rows."""
    per_frame = table.groupby("frame")["id"].nunique().to_numpy(dtype=np.int64)
    extent = int(table["frame"].max()) - int(table["frame"].min()) + 1
    return int((per_frame * (per_frame - 1)).sum()), float(per_frame.sum() / extent)


# --- a direct conversion ---------------------------------------------------------


def test_a_centroid_file_converts_to_a_table_mosaic_v1_accepts(tmp_path: Path) -> None:
    path = write_synthetic_spec(tmp_path / "walk.toml")

    table = _convert(SyntheticConverter(), path)

    _, report = ensure_track_schema(table, "mosaic_v1", strict=True, source=str(path))
    assert report["missing_required"] == []
    assert report["forbidden_present"] == []
    assert list(table.columns) == ["frame", "time", "id", "X", "Y", "group", "sequence"]
    assert not [name for name in table.columns if str(name).startswith("pose")]
    assert len(table) == 200 * 3


def test_time_is_the_frame_over_the_files_rate(tmp_path: Path) -> None:
    path = write_synthetic_spec(tmp_path / "walk.toml", fps=25.0)

    table = _convert(SyntheticConverter(), path)

    np.testing.assert_array_equal(
        table["time"].to_numpy(), table["frame"].to_numpy() / 25.0
    )


def test_a_file_with_a_body_carries_its_keypoints_with_confidence(
    tmp_path: Path,
) -> None:
    path = write_synthetic_spec(tmp_path / "walk.toml", body=BODY)

    table = _convert(SyntheticConverter(), path)

    _, report = ensure_track_schema(table, "mosaic_v1", strict=True, source=str(path))
    assert report["missing_required"] == []
    keypoint_columns = [str(name) for name in table.columns if str(name)[:4] == "pose"]
    assert keypoint_columns == [
        f"pose{axis}{k}" for k in range(3) for axis in ("X", "Y", "P")
    ]
    assert "Z" not in table.columns


def test_a_3d_file_has_z_and_a_z_per_keypoint_in_world_schema(tmp_path: Path) -> None:
    path = write_synthetic_spec(
        tmp_path / "walk.toml", dimensions=3, arena=SYNTHETIC_ARENA_3D, body=BODY
    )

    table = _convert(Synthetic3dConverter(), path)

    _, report = ensure_track_schema(
        table, "mosaic_world_v1", strict=True, source=str(path)
    )
    assert report["missing_required"] == []
    assert "Z" in table.columns
    assert {f"poseZ{k}" for k in range(3)} <= set(table.columns)
    assert float(table["Z"].max()) <= 500.0


def test_the_table_shares_the_simulated_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_synthetic_spec(tmp_path / "walk.toml", body=BODY)
    produced: list[SyntheticColumns] = []

    def recording_simulate(
        spec: SyntheticSpec, *, relabel_after_gap: bool = False
    ) -> SyntheticColumns:
        columns = simulate(spec, relabel_after_gap=relabel_after_gap)
        produced.append(columns)
        return columns

    monkeypatch.setattr(synthetic, "simulate", recording_simulate)

    table = _convert(SyntheticConverter(), path)

    (columns,) = produced
    named = list(columns.named_columns())
    assert len(named) == 4 + 3 * 3
    for name, array in named:
        assert np.shares_memory(table[name].to_numpy(), array), name


def test_group_and_sequence_come_from_the_hints_or_the_file_stem(
    tmp_path: Path,
) -> None:
    path = write_synthetic_spec(tmp_path / "walk_a.toml")
    converter = SyntheticConverter()

    stem = _convert(converter, path)
    hinted = _convert(converter, path, hints=EntryHints(group="g1", sequence="s1"))

    assert set(stem["group"]) == {""}
    assert set(stem["sequence"]) == {"walk_a"}
    assert set(hinted["group"]) == {"g1"}
    assert set(hinted["sequence"]) == {"s1"}


# --- the dimension of a file -----------------------------------------------------


def test_synthetic_refuses_a_3d_file_and_names_synthetic_3d(tmp_path: Path) -> None:
    path = write_synthetic_spec(
        tmp_path / "walk.toml", dimensions=3, arena=SYNTHETIC_ARENA_3D
    )

    with pytest.raises(ValueError, match="synthetic_3d") as caught:
        _ = _convert(SyntheticConverter(), path)

    assert str(path) in str(caught.value)


def test_synthetic_3d_refuses_a_2d_file_and_names_synthetic(tmp_path: Path) -> None:
    path = write_synthetic_spec(tmp_path / "walk.toml")

    with pytest.raises(ValueError, match=r"synthetic\b") as caught:
        _ = _convert(Synthetic3dConverter(), path)

    assert "synthetic_3d" in str(caught.value)
    assert str(path) in str(caught.value)


def test_the_two_formats_declare_their_schema_and_the_id_policy() -> None:
    flat, deep = SyntheticConverter(), Synthetic3dConverter()

    assert TRACK_CONVERTERS["synthetic"] is SyntheticConverter
    assert TRACK_CONVERTERS["synthetic_3d"] is Synthetic3dConverter
    assert (flat.output_schema, deep.output_schema) == ("mosaic_v1", "mosaic_world_v1")
    assert flat.tracklets(SyntheticParams()) is True
    assert flat.tracklets(SyntheticParams(relabel_after_gap=True)) is False
    assert deep.tracklets(SyntheticParams(relabel_after_gap=True)) is False


# --- a dataset -------------------------------------------------------------------


def test_a_dataset_converts_two_files_and_indexes_their_measures(
    tmp_path: Path,
) -> None:
    ds = synthetic_dataset(tmp_path / "ds", ["a", "b"])

    outcome = ds.convert_all_tracks()

    assert (outcome.converted, outcome.failed) == (2, 0)
    index = read_tracks_index(ds)
    assert sorted(index["sequence"]) == ["a", "b"]
    for _, row in index.iterrows():
        written = ds.resolve_path(str(row["abs_path"]))
        table = pd.read_parquet(written)
        measures = frame_measures(written)
        assert measures is not None
        pair_frames, ids_per_frame = _pairs(table)
        assert row["std_format"] == "mosaic_v1"
        assert int(row["n_rows"]) == len(table)
        assert read_carries_tracklets(row) is True
        assert read_pair_frames(row) == measures.pair_frames == pair_frames
        assert measures.ids_per_frame is not None
        assert measures.ids_per_frame == pytest.approx(ids_per_frame)
        assert read_ids_per_frame(row) == pytest.approx(ids_per_frame, abs=5e-5)


def test_relabeling_records_that_ids_do_not_follow_animals(tmp_path: Path) -> None:
    ds = synthetic_dataset(tmp_path / "ds", ["a"])
    default = ds.track_conversion_variant("synthetic")
    relabeling = ds.track_conversion_variant("synthetic", {"relabel_after_gap": True})

    outcome = ds.convert_all_tracks({"relabel_after_gap": True})

    assert outcome.variants == (relabeling,)
    assert relabeling != default
    (row,) = (row for _, row in read_tracks_index(ds).iterrows())
    assert read_carries_tracklets(row) is False
    assert row["run_id"] == relabeling


def test_a_3d_source_converts_to_the_world_schema(tmp_path: Path) -> None:
    ds = synthetic_dataset(tmp_path / "ds", ["a"], src_format="synthetic_3d")

    outcome = ds.convert_all_tracks()

    assert (outcome.converted, outcome.failed) == (1, 0)
    (row,) = (row for _, row in read_tracks_index(ds).iterrows())
    assert row["std_format"] == "mosaic_world_v1"
    assert "Z" in pd.read_parquet(ds.resolve_path(str(row["abs_path"]))).columns


def test_two_datasets_converting_one_file_agree_to_the_byte(tmp_path: Path) -> None:
    outputs: list[tuple[str, bytes]] = []
    for name in ("first", "second"):
        ds = make_dataset(tmp_path / name / "ds")
        _ = write_synthetic_spec(
            ds.base_dir / "raw" / "walk.toml", dropouts=SYNTHETIC_DROPOUTS, body=BODY
        )
        _ = ds.index_tracks_raw(
            [ds.base_dir / "raw"], patterns=["*.toml"], src_format="synthetic"
        )
        _ = ds.convert_all_tracks()
        (row,) = (row for _, row in read_tracks_index(ds).iterrows())
        outputs.append(
            (str(row["run_id"]), ds.resolve_path(str(row["abs_path"])).read_bytes())
        )

    assert outputs[0][0] == outputs[1][0]
    assert outputs[0][1] == outputs[1][1]


def test_a_file_that_fails_validation_is_a_failed_entry_and_the_rest_convert(
    tmp_path: Path,
) -> None:
    ds = synthetic_dataset(tmp_path / "ds", ["good"])
    _ = write_synthetic_spec(ds.base_dir / "raw" / "bad.toml", format="other/9")
    _ = ds.index_tracks_raw(
        [ds.base_dir / "raw"], patterns=["*.toml"], src_format="synthetic"
    )
    failures: list[tuple[str, BaseException]] = []

    outcome = ds.convert_all_tracks(
        on_failure=lambda path, error: failures.append((path, error))
    )

    assert (outcome.converted, outcome.failed) == (1, 1)
    assert len(failures) == 1
    assert "format must be" in str(failures[0][1])
    assert list(read_tracks_index(ds)["sequence"]) == ["good"]


def test_convert_tracks_reports_two_written(tmp_path: Path) -> None:
    ds = synthetic_dataset(tmp_path / "ds", ["a", "b"])

    payload = invoke_json(
        [
            "convert-tracks",
            "-m",
            str(ds.manifest_path),
            "--source-format",
            "synthetic",
            "--json",
        ]
    )

    assert payload["status"] == "ok"
    assert (payload["converted"], payload["written"], payload["failed"]) == (2, 2, 0)
