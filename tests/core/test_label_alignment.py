"""A label file aligned to rows: what a frame with no event means, and whose row an event labels.

A frame no event covered used to read as class 0, which in CalMS21 is ``attack``,
and a pair event labelled both directions of the pair. These pin the rules that
replaced both: a file states its background class, and an event labels the rows
whose individuals it names.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mosaic.core.helpers import (
    UNLABELED,
    LegacyLabelFileWarning,
    load_labels_auto,
    load_labels_for_feature_frames,
)

CALMS21_NAMES = ["attack", "investigation", "mount", "other"]


def _sparse(
    path: Path,
    events: list[tuple[int, int, int, int]],
    *,
    names: list[str] = CALMS21_NAMES,
    background_id: int | None = None,
) -> Path:
    """Write an ``individual_pair_v1`` file of ``(frame, id1, id2, label)`` events."""
    payload: dict[str, object] = {
        "label_format": "individual_pair_v1",
        "frames": np.array([e[0] for e in events], dtype=np.int64),
        "individual_ids": np.array(
            [[e[1], e[2]] for e in events], dtype=np.int64
        ).reshape(-1, 2),
        "labels": np.array([e[3] for e in events], dtype=np.int64),
        "label_ids": np.arange(len(names), dtype=np.int32),
        "label_names": np.array(names, dtype=object),
    }
    if background_id is not None:
        payload["background_id"] = np.int32(background_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)  # pyright: ignore[reportArgumentType]
    return path


def _rows(*pairs: tuple[int, int]) -> np.ndarray:
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def test_a_frame_without_an_event_is_unlabeled_not_class_zero(tmp_path: Path) -> None:
    path = _sparse(
        tmp_path / "a.npz", [(0, 0, 1, 1), (1, 0, 1, 1)], background_id=UNLABELED
    )

    labels = load_labels_for_feature_frames(path, np.array([0, 1, 5]))

    assert labels.tolist() == [1, 1, UNLABELED]


def test_background_id_in_the_file_is_honoured(tmp_path: Path) -> None:
    path = _sparse(tmp_path / "a.npz", [(0, 0, 1, 1)], background_id=3)

    assert load_labels_for_feature_frames(path, np.array([0, 5])).tolist() == [1, 3]
    assert load_labels_auto(path, n_frames=3).tolist() == [1, 3, 3]


def test_a_legacy_boris_file_takes_its_background_from_the_variant_record(
    tmp_path: Path,
) -> None:
    """BORIS sorted its background alphabetically, so it was not always id 0.

    The variant's ``params.json`` records which name was the background, which
    is exact for a file written before ``background_id`` existed.
    """
    variant = tmp_path / "labels" / "behavior" / "boris.0.1-abc"
    path = _sparse(
        variant / "obs.npz", [(0, 0, -1, 0)], names=["approach", "none", "zigzag"]
    )
    _ = (variant / "params.json").write_text(
        json.dumps({"params": {"background_label": "none"}})
    )

    assert load_labels_for_feature_frames(path, np.array([0, 4])).tolist() == [0, 1]


def test_a_legacy_file_without_a_background_reads_unlabeled_and_warns(
    tmp_path: Path,
) -> None:
    path = _sparse(tmp_path / "a.npz", [(0, 0, 1, 2)])

    with pytest.warns(LegacyLabelFileWarning, match="convert-labels"):
        labels = load_labels_for_feature_frames(path, np.array([0, 1]))

    assert labels.tolist() == [2, UNLABELED]


def test_a_fully_covered_legacy_file_does_not_warn(tmp_path: Path) -> None:
    """The background is read only when a row needs it. Warnings are errors here."""
    path = _sparse(tmp_path / "a.npz", [(0, 0, 1, 2), (1, 0, 1, 3)])

    assert load_labels_for_feature_frames(path, np.array([0, 1])).tolist() == [2, 3]


def test_a_dense_frame_out_of_range_is_unlabeled_without_a_warning(
    tmp_path: Path,
) -> None:
    """A dense file labels every frame it holds, so a frame past it is unannotated."""
    path = tmp_path / "dense.npz"
    np.savez(path, labels=np.array([2, 2, 2]))

    labels = load_labels_for_feature_frames(path, np.array([0, 2, 7]))

    assert labels.tolist() == [2, 2, UNLABELED]


def test_pair_rows_take_their_own_directions_label(tmp_path: Path) -> None:
    """A CalMS21 event is resident -> intruder, so the intruder-focal row is unlabeled."""
    path = _sparse(tmp_path / "a.npz", [(0, 0, 1, 0)], background_id=UNLABELED)

    labels = load_labels_for_feature_frames(
        path, np.array([0, 0]), individual_filter=_rows((0, 1), (1, 0))
    )

    assert labels.tolist() == [0, UNLABELED]


def test_symmetric_labels_stored_both_ways_label_both_rows(tmp_path: Path) -> None:
    path = _sparse(
        tmp_path / "a.npz", [(0, 0, 1, 2), (0, 1, 0, 2)], background_id=UNLABELED
    )

    labels = load_labels_for_feature_frames(
        path, np.array([0, 0]), individual_filter=_rows((0, 1), (1, 0))
    )

    assert labels.tolist() == [2, 2]


def test_a_scene_label_applies_to_every_row(tmp_path: Path) -> None:
    path = _sparse(tmp_path / "a.npz", [(0, -1, -1, 3)], background_id=UNLABELED)

    labels = load_labels_for_feature_frames(
        path, np.array([0, 0, 0]), individual_filter=_rows((0, 1), (1, 0), (2, -1))
    )

    assert labels.tolist() == [3, 3, 3]


def test_an_individual_label_applies_to_its_pairs(tmp_path: Path) -> None:
    path = _sparse(tmp_path / "a.npz", [(0, 0, -1, 1)], background_id=UNLABELED)

    labels = load_labels_for_feature_frames(
        path, np.array([0, 0, 0]), individual_filter=_rows((0, 1), (1, 0), (0, -1))
    )

    assert labels.tolist() == [1, UNLABELED, 1]


def test_a_single_pair_filter_is_directed(tmp_path: Path) -> None:
    """The tuple form used to match both orderings of the pair."""
    path = _sparse(tmp_path / "a.npz", [(0, 1, 0, 2)], background_id=UNLABELED)

    labels = load_labels_for_feature_frames(
        path, np.array([0]), individual_filter=(0, 1)
    )

    assert labels.tolist() == [UNLABELED]


def test_a_filter_of_the_wrong_shape_is_refused(tmp_path: Path) -> None:
    path = _sparse(tmp_path / "a.npz", [(0, 0, 1, 2)], background_id=UNLABELED)

    with pytest.raises(ValueError, match="one \\(id1, id2\\) per feature frame"):
        _ = load_labels_for_feature_frames(
            path, np.array([0, 1]), individual_filter=_rows((0, 1))
        )
