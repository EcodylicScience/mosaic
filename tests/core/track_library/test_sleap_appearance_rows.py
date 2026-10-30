"""The join of a SLEAP readback to the tracks table's ``(frame, id)`` slots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.track_converter import EntryHints
from mosaic.core.track_library.sleap import (
    AppearanceReadback,
    SleapAnalysisH5Converter,
    SleapConvertParams,
    appearance_rows,
    read_analysis_track_names,
    read_appearance_readback,
)
from tests.helpers import (
    ReadbackRecord,
    write_appearance_readback,
    write_sleap_analysis_h5,
)


def _readback(
    tmp_path: Path, records: Sequence[ReadbackRecord], *, dim: int
) -> AppearanceReadback:
    path = write_appearance_readback(tmp_path / "e.embeddings.npz", records, dim=dim)
    return read_appearance_readback(path)


def _rows(readback: AppearanceReadback, track_names: Sequence[str]) -> pd.DataFrame:
    return appearance_rows(readback, track_names, group="g", sequence="s")


def test_a_user_instance_takes_the_slot_over_predictions(tmp_path: Path) -> None:
    readback = _readback(
        tmp_path,
        [
            ReadbackRecord(4, "a", vector=(1.0, 0.0)),
            ReadbackRecord(4, "a", is_user=True, vector=(0.0, 1.0)),
            ReadbackRecord(4, "a", vector=(0.6, 0.8)),
        ],
        dim=2,
    )

    rows = _rows(readback, ["a"])

    assert rows[["frame", "id"]].to_numpy().tolist() == [[4, 0]]
    assert rows[["emb_0", "emb_1"]].to_numpy().tolist() == [[0.0, 1.0]]
    assert rows["has_vector"].tolist() == [True]


def test_the_last_prediction_in_file_order_takes_the_slot(tmp_path: Path) -> None:
    readback = _readback(
        tmp_path,
        [
            ReadbackRecord(2, "b", vector=(1.0, 0.0)),
            ReadbackRecord(2, "a", vector=(0.0, 1.0)),
            ReadbackRecord(2, "b", vector=(0.6, 0.8)),
        ],
        dim=2,
    )

    rows = _rows(readback, ["a", "b"])

    assert rows[["frame", "id"]].to_numpy().tolist() == [[2, 0], [2, 1]]
    np.testing.assert_allclose(
        rows[["emb_0", "emb_1"]].to_numpy(), [[0.0, 1.0], [0.6, 0.8]], rtol=1e-6
    )


def test_a_winner_without_a_vector_is_a_slot_without_a_vector(
    tmp_path: Path,
) -> None:
    readback = _readback(
        tmp_path,
        [
            ReadbackRecord(0, "a", vector=(1.0, 0.0)),
            ReadbackRecord(0, "a"),
            ReadbackRecord(1, "a", vector=(0.0, 1.0)),
        ],
        dim=2,
    )

    rows = _rows(readback, ["a"])

    assert rows["frame"].tolist() == [0, 1]
    assert rows["has_vector"].tolist() == [False, True]
    assert np.isnan(rows.loc[0, ["emb_0", "emb_1"]].to_numpy(dtype=float)).all()


def test_a_readback_without_vector_columns_gives_slots_without_vectors(
    tmp_path: Path,
) -> None:
    readback = _readback(
        tmp_path,
        [ReadbackRecord(0, "a"), ReadbackRecord(0, "b", is_user=True)],
        dim=0,
    )

    rows = _rows(readback, ["a", "b"])

    assert rows[["frame", "id"]].to_numpy().tolist() == [[0, 0], [0, 1]]
    assert rows["has_vector"].tolist() == [False, False]
    assert not [c for c in rows.columns if str(c).startswith("emb_")]


def test_a_track_without_instances_keeps_its_index(tmp_path: Path) -> None:
    readback = _readback(tmp_path, [ReadbackRecord(3, "c", vector=(1.0, 0.0))], dim=2)

    rows = _rows(readback, ["a", "b", "c"])

    assert rows["id"].tolist() == [2]


def test_the_columns_have_the_contract_dtypes(tmp_path: Path) -> None:
    readback = _readback(tmp_path, [ReadbackRecord(3, "a", vector=(1.0, 0.0))], dim=2)

    rows = _rows(readback, ["a"])

    assert list(rows.columns) == [
        "group",
        "sequence",
        "frame",
        "id",
        "has_vector",
        "emb_0",
        "emb_1",
    ]
    assert rows["frame"].dtype == np.int64
    assert rows["id"].dtype == np.int64
    assert rows["has_vector"].dtype == np.bool_
    assert rows["emb_0"].dtype == np.float32
    assert rows["group"].tolist() == ["g"]
    assert rows["sequence"].tolist() == ["s"]


def test_repeated_track_names_are_refused(tmp_path: Path) -> None:
    readback = _readback(tmp_path, [ReadbackRecord(0, "a")], dim=0)

    with pytest.raises(ValueError, match="'a'"):
        _ = _rows(readback, ["a", "b", "a"])


def test_a_record_whose_track_is_not_named_is_refused(tmp_path: Path) -> None:
    readback = _readback(tmp_path, [ReadbackRecord(0, "z")], dim=0)

    with pytest.raises(ValueError, match="'z'"):
        _ = _rows(readback, ["a"])


def test_ids_match_the_converter_on_the_analysis_track_axis(tmp_path: Path) -> None:
    n_frames, n_tracks, n_nodes = 3, 3, 2
    tracks = np.full((n_frames, n_tracks, n_nodes, 2), np.nan)
    tracks[0:2, 0] = 1.0
    tracks[1:3, 2] = 2.0
    names = ("left", "empty", "right")
    h5 = tmp_path / "e.analysis.h5"
    write_sleap_analysis_h5(h5, tracks, track_names=names)
    readback = _readback(
        tmp_path,
        [
            ReadbackRecord(0, "left", vector=(1.0, 0.0)),
            ReadbackRecord(1, "right", vector=(0.0, 1.0)),
            ReadbackRecord(1, "left", vector=(1.0, 0.0)),
            ReadbackRecord(2, "right", vector=(0.0, 1.0)),
        ],
        dim=2,
    )
    converted = SleapAnalysisH5Converter().convert(
        h5, SleapConvertParams(), EntryHints(group="g", sequence="s")
    )

    track_names = read_analysis_track_names(h5)
    rows = _rows(readback, track_names)

    assert track_names == names
    expected = sorted(converted[["frame", "id"]].itertuples(index=False, name=None))
    assert list(rows[["frame", "id"]].itertuples(index=False, name=None)) == [
        (int(f), int(i)) for f, i in expected
    ]


def test_an_analysis_file_without_track_names_is_refused(tmp_path: Path) -> None:
    h5 = tmp_path / "e.analysis.h5"
    write_sleap_analysis_h5(h5, np.zeros((1, 1, 1, 2)))

    with pytest.raises(ValueError, match="track_names"):
        _ = read_analysis_track_names(h5)


def test_an_empty_readback_gives_no_rows_with_the_contract_dtypes() -> None:
    rows = _rows(AppearanceReadback.empty(3), [])

    assert rows.empty
    assert pd.api.types.is_string_dtype(rows["group"])
    assert pd.api.types.is_string_dtype(rows["sequence"])
    assert (rows["frame"].dtype, rows["id"].dtype) == (np.int64, np.int64)
    assert [str(rows[f"emb_{k}"].dtype) for k in range(3)] == ["float32"] * 3
