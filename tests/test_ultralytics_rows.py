"""Row extraction and the tracks conversion, over hand-built results.

No Ultralytics, no dataset, no model. This is where the three defects of the
older per-frame inference path are pinned as *not* inherited: its ``frame`` is an
enumerate position rather than the reader's index, its ``id`` is a per-frame
detection ordinal rather than an identity, and its ``time`` is a frame ordinal
rather than seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.schema import ensure_track_schema
from mosaic.core.track_converter import EntryHints
from mosaic.core.track_library.ultralytics_tracks import (
    UltralyticsTracksConverter,
    UltralyticsTracksParams,
    raw_columns,
)
from mosaic.tracking.external.runner.ultralytics_protocol import (
    UltralyticsInteropError,
    rows_from_result,
)


# --- stand-ins for the Ultralytics result surface --------------------------


@dataclass
class FakeDetections:
    array: np.ndarray

    @property
    def data(self) -> np.ndarray:
        return self.array

    @property
    def id(self) -> np.ndarray | None:
        # A tracked Boxes carries the track id in column 4.
        return self.array[:, 4] if self.array.shape[1] >= 7 else None

    def cpu(self) -> FakeDetections:
        return self

    def numpy(self) -> FakeDetections:
        return self


@dataclass
class FakeResult:
    boxes: FakeDetections | None
    keypoints: FakeDetections | None = None


def _boxes(*rows: tuple[float, ...]) -> FakeDetections:
    """Rows of (x1, y1, x2, y2, track_id, conf, cls)."""
    return FakeDetections(np.array(rows, dtype=float))


def _keypoints(*per_detection: list[tuple[float, float, float]]) -> FakeDetections:
    return FakeDetections(np.array(per_detection, dtype=float))


# --- extraction -------------------------------------------------------------


def test_a_frame_with_no_tracks_contributes_no_rows() -> None:
    """``boxes.id is None`` is untracked raw detections, not an empty frame.

    Ultralytics' tracking callback returns early when the tracker produced no
    tracks, leaving the raw detections in place. Emitting them would put
    identity-less rows into a table whose whole subject is identity.
    """
    untracked = FakeDetections(np.array([[0.0, 0.0, 4.0, 4.0, 0.9, 0.0]]))
    assert rows_from_result(FakeResult(untracked), 3, n_keypoints=1) is None
    assert rows_from_result(FakeResult(None), 3, n_keypoints=1) is None


def test_the_frame_column_is_the_readers_true_index() -> None:
    """Not an enumerate position: with a window and a step, 100 means frame 100."""
    block = rows_from_result(
        FakeResult(_boxes((0.0, 0.0, 4.0, 4.0, 7.0, 0.9, 0.0))), 100, n_keypoints=1
    )
    assert block is not None
    assert block[0, 0] == 100.0


def test_the_id_column_is_the_tracker_identity() -> None:
    """Not a per-frame detection ordinal, which is the whole point of tracking."""
    block = rows_from_result(
        FakeResult(
            _boxes(
                (0.0, 0.0, 4.0, 4.0, 11.0, 0.9, 0.0),
                (9.0, 9.0, 12.0, 12.0, 4.0, 0.8, 0.0),
            )
        ),
        0,
        n_keypoints=1,
    )
    assert block is not None
    # Sorted by the tracker's own numbering, because list order is not a contract.
    assert list(block[:, 1]) == [4.0, 11.0]


def test_a_box_only_model_synthesizes_one_keypoint_at_the_box_centre() -> None:
    block = rows_from_result(
        FakeResult(_boxes((10.0, 20.0, 30.0, 40.0, 1.0, 0.75, 0.0))), 0, n_keypoints=1
    )
    assert block is not None
    assert block[0, 8] == 20.0  # kpx0: centre x
    assert block[0, 9] == 30.0  # kpy0: centre y
    assert block[0, 10] == 0.75  # kpp0: the detection confidence


def test_keypoints_stay_aligned_with_the_tracked_boxes() -> None:
    block = rows_from_result(
        FakeResult(
            _boxes(
                (0.0, 0.0, 4.0, 4.0, 2.0, 0.9, 0.0),
                (9.0, 9.0, 12.0, 12.0, 1.0, 0.8, 0.0),
            ),
            _keypoints(
                [(1.0, 1.0, 0.5), (2.0, 2.0, 0.6)],
                [(10.0, 10.0, 0.7), (11.0, 11.0, 0.8)],
            ),
        ),
        0,
        n_keypoints=2,
    )
    assert block is not None
    # Track 1 sorts first and must keep *its* keypoints, not track 2's.
    assert block[0, 1] == 1.0
    assert block[0, 8] == 10.0
    assert block[1, 1] == 2.0
    assert block[1, 8] == 1.0


def test_a_keypoint_count_mismatch_is_refused() -> None:
    with pytest.raises(UltralyticsInteropError, match="keypoint sets"):
        _ = rows_from_result(
            FakeResult(
                _boxes(
                    (0.0, 0.0, 4.0, 4.0, 1.0, 0.9, 0.0),
                    (5.0, 5.0, 9.0, 9.0, 2.0, 0.9, 0.0),
                ),
                _keypoints([(1.0, 1.0, 0.5)]),
            ),
            0,
            n_keypoints=1,
        )


# --- conversion -------------------------------------------------------------


def _write_raw(path: Path, rows: list[list[float]], n_keypoints: int) -> Path:
    columns = list(raw_columns(n_keypoints))
    values = (
        np.array(rows, dtype=float)
        if rows
        else np.empty((0, len(columns)), dtype=float)
    )
    table = pd.DataFrame(values, columns=columns)
    table = table.astype({"frame": "int64", "track_id": "int64", "cls": "int64"})
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return path


def _convert(path: Path, *, fps: float = 25.0) -> pd.DataFrame:
    return UltralyticsTracksConverter().convert(
        path, UltralyticsTracksParams(fps=fps), EntryHints(group="g", sequence="vid")
    )


def test_time_is_seconds_and_ids_are_densified(tmp_path: Path) -> None:
    rows = [
        [float(f), float(t), 0.0, 0.0, 4.0, 4.0, 0.9, 0.0, float(t + f), 1.0, 0.8]
        for f in (0, 1)
        for t in (3, 7, 11)
    ]
    df = _convert(_write_raw(tmp_path / "p.parquet", rows, 1))

    assert np.allclose(df["time"], df["frame"] / 25.0)
    assert sorted(df["id"].unique().tolist()) == [0, 1, 2]
    assert sorted(df["source_track_id"].unique().tolist()) == [3, 7, 11]


def test_a_gap_in_a_track_keeps_its_true_frame_numbers(tmp_path: Path) -> None:
    """A track that drops out and returns must not have its gap closed up.

    The converter used to differentiate position itself, and this test asserted
    the resulting speed was uniform across a gap. That arithmetic now belongs to
    ``speed-angvel``, which differentiates against ``frame`` for exactly the same
    reason -- so what the converter still owes is the honest frame index. Were it
    to renumber rows contiguously, every downstream velocity would silently treat
    the gap as one step.
    """
    rows = [
        [float(f), 1.0, 0.0, 0.0, 4.0, 4.0, 0.9, 0.0, float(f), 0.0, 0.8]
        for f in (0, 1, 3, 4)  # frame 2 missing
    ]
    df = _convert(_write_raw(tmp_path / "p.parquet", rows, 1), fps=10.0)
    assert df["frame"].to_list() == [0, 1, 3, 4]


def test_a_box_only_table_reports_no_heading_at_all(tmp_path: Path) -> None:
    """It used to infer one from direction of travel. That was not an orientation.

    A box tracker localizes no shape, so it has nothing to say about which way an
    animal faces -- a bee walking backwards would have been recorded as facing
    the way it went. Inferring it here also made a *derived* quantity look like a
    measurement, under a column name every other tracker filled differently.
    Anything wanting a heading now runs the ``heading`` feature and chooses.
    """
    moving = [
        [float(f), 1.0, 0.0, 0.0, 4.0, 4.0, 0.9, 0.0, float(f), 0.0, 0.8]
        for f in range(3)
    ]
    df = _convert(_write_raw(tmp_path / "p.parquet", moving, 1))
    assert "ANGLE" not in df.columns


def test_the_converted_table_satisfies_the_schema(tmp_path: Path) -> None:
    rows = [
        [
            float(f),
            1.0,
            0.0,
            0.0,
            4.0,
            4.0,
            0.9,
            0.0,
            float(f),
            0.0,
            0.8,
            float(f) + 2,
            1.0,
            0.7,
        ]
        for f in range(3)
    ]
    df = _convert(_write_raw(tmp_path / "p.parquet", rows, 2))
    _, report = ensure_track_schema(df, "mosaic_v1", strict=True)
    assert report["missing_required"] == []
    assert report["missing_prefixes"] == []
    assert report["missing_recommended"] == []


def test_an_empty_table_converts_to_the_empty_frame(tmp_path: Path) -> None:
    """Written even with no detections, so the reuse gate can find it."""
    df = _convert(_write_raw(tmp_path / "p.parquet", [], 1))
    assert df.empty
    assert list(df.columns) == ["frame", "time", "id", "group", "sequence"]
