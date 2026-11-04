"""Which event of a label file labels each row, and how many events match it.

``events_for_rows`` follows the rule ``labels_for_rows`` aligns by: a row with
integer ids takes the last event matching its individuals, and a row without
them takes the last event of its frame once reverse pair events are dropped. It
returns the event's position in the file and the count of events matching the
row, so a reader can gather other per-event columns from the same event.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.helpers import UNLABELED, RowEvents, events_for_frames
from mosaic.core.pipeline.loading import events_for_rows, labels_for_rows
from mosaic.core.pipeline.types import GroundTruthLabelsSource
from tests.helpers import write_sparse_labels

Event = tuple[int, int, int, int]

_EVENTS: dict[str, list[Event]] = {
    "individual": [(0, 0, -1, 1), (1, 1, -1, 2), (1, 0, -1, 0), (3, 0, -1, 2)],
    "pair": [(0, 0, 1, 1), (0, 1, 0, 2), (1, 0, 1, 0), (1, 0, 2, 3), (2, 1, 0, 1)],
    "scene": [(0, -1, -1, 1), (2, -1, -1, 2), (2, -1, -1, 0)],
}

_ROWS: dict[str, pd.DataFrame] = {
    "track rows": pd.DataFrame(
        {"frame": [0, 0, 1, 1, 2, 3, 4], "id": [0, 1] * 3 + [0]}
    ),
    "pair rows": pd.DataFrame(
        {
            "frame": [0, 0, 1, 1, 2, 2, 4],
            "id1": [0, 1, 0, 2, 1, 0, 0],
            "id2": [1, 0, 2, 0, 0, 1, 1],
        }
    ),
    "rows without ids": pd.DataFrame({"frame": [0, 1, 2, 3, 4]}),
    "rows with named ids": pd.DataFrame(
        {"frame": [0, 1, 2, 3], "id": ["a", "b", "a", "b"]}
    ),
}


@pytest.mark.parametrize("background_id", [UNLABELED, 3])
@pytest.mark.parametrize("rows", sorted(_ROWS))
@pytest.mark.parametrize("events", sorted(_EVENTS))
def test_the_labels_are_those_labels_for_rows_gives(
    tmp_path: Path, events: str, rows: str, background_id: int
) -> None:
    path = write_sparse_labels(
        tmp_path / "a.npz", _EVENTS[events], background_id=background_id
    )
    df = _ROWS[rows]

    found = events_for_rows(path, df)

    expected = labels_for_rows(path, df, GroundTruthLabelsSource(kind="behavior"))
    assert found.labels.tolist() == expected.tolist()


@pytest.mark.parametrize("rows", sorted(_ROWS))
@pytest.mark.parametrize("events", sorted(_EVENTS))
def test_the_label_at_each_matched_position_is_the_rows_label(
    tmp_path: Path, events: str, rows: str
) -> None:
    path = write_sparse_labels(tmp_path / "a.npz", _EVENTS[events], background_id=3)
    file_labels = np.array([event[3] for event in _EVENTS[events]], dtype=np.int64)

    found = events_for_rows(path, _ROWS[rows])

    matched = found.positions >= 0
    assert matched.any()
    assert file_labels[found.positions[matched]].tolist() == (
        found.labels[matched].tolist()
    )
    assert (found.matches[matched] >= 1).all()
    assert found.labels[~matched].tolist() == [3] * int((~matched).sum())
    assert found.matches[~matched].tolist() == [0] * int((~matched).sum())


def test_a_dense_file_gives_each_row_its_frames_position(tmp_path: Path) -> None:
    path = tmp_path / "dense.npz"
    np.savez(path, labels=np.array([2, 0, 1]))
    df = pd.DataFrame({"frame": [2, 0, 5, -1], "id": [0, 1, 0, 1]})

    found = events_for_rows(path, df)

    assert found.positions.tolist() == [2, 0, -1, -1]
    assert found.matches.tolist() == [1, 1, 0, 0]
    assert found.labels.tolist() == [1, 2, UNLABELED, UNLABELED]
    expected = labels_for_rows(path, df, GroundTruthLabelsSource(kind="behavior"))
    assert found.labels.tolist() == expected.tolist()


def test_a_calms21_event_matches_the_resident_row_alone(tmp_path: Path) -> None:
    """A CalMS21 event is resident ``0`` -> intruder ``1``."""
    path = write_sparse_labels(tmp_path / "a.npz", [(0, 0, 1, 2), (1, 0, 1, 1)])
    df = pd.DataFrame({"frame": [0, 0, 1, 1], "id": [0, 1, 0, 1]})

    found = events_for_rows(path, df)

    assert found.matches.tolist() == [1, 0, 1, 0]
    assert found.positions.tolist() == [0, -1, 1, -1]


def test_two_pairs_focal_on_one_individual_both_match_its_row(tmp_path: Path) -> None:
    path = write_sparse_labels(tmp_path / "a.npz", [(0, 0, 1, 2), (0, 0, 2, 1)])
    df = pd.DataFrame({"frame": [0, 0], "id": [0, 1]})

    found = events_for_rows(path, df)

    assert found.matches.tolist() == [2, 0]
    assert found.positions.tolist() == [1, -1]


def test_two_individual_events_both_count_on_a_row_without_ids(tmp_path: Path) -> None:
    path = write_sparse_labels(tmp_path / "a.npz", [(0, 0, -1, 2), (0, 1, -1, 1)])

    found = events_for_rows(path, pd.DataFrame({"frame": [0, 1]}))

    assert found.matches.tolist() == [2, 0]
    assert found.positions.tolist() == [1, -1]
    assert found.labels.tolist() == [1, UNLABELED]


def test_a_reverse_pair_event_is_not_counted_on_a_row_without_ids(
    tmp_path: Path,
) -> None:
    """Both directions of a pair are one interaction to a row that names no one."""
    path = write_sparse_labels(tmp_path / "a.npz", [(0, 0, 1, 2), (0, 1, 0, 1)])

    found = events_for_rows(path, pd.DataFrame({"frame": [0]}))

    assert found.matches.tolist() == [1]
    assert found.positions.tolist() == [0]
    assert found.labels.tolist() == [2]


def test_events_for_frames_takes_one_filter_for_every_row(tmp_path: Path) -> None:
    path = write_sparse_labels(tmp_path / "a.npz", [(0, 0, 1, 2), (0, 1, 0, 1)])

    found = events_for_frames(path, np.array([0, 1]), individual_filter=(1, 0))

    assert isinstance(found, RowEvents)
    assert found.positions.tolist() == [1, -1]
    assert found.matches.tolist() == [1, 0]
    assert found.positions.dtype == found.matches.dtype == found.labels.dtype
    assert found.labels.dtype == np.int64


def test_an_empty_file_matches_no_row(tmp_path: Path) -> None:
    path = write_sparse_labels(tmp_path / "a.npz", [], background_id=3)

    found = events_for_rows(path, pd.DataFrame({"frame": [0, 1], "id": [0, 1]}))

    assert found.positions.tolist() == [-1, -1]
    assert found.matches.tolist() == [0, 0]
    assert found.labels.tolist() == [3, 3]


def test_rows_without_a_frame_are_refused(tmp_path: Path) -> None:
    path = write_sparse_labels(tmp_path / "a.npz", [(0, 0, 1, 2)])

    with pytest.raises(ValueError, match="'frame'"):
        _ = events_for_rows(path, pd.DataFrame({"id": [0]}))
