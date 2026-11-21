"""Playback reads the ground-truth labels of an entry as every label reader does."""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.behavior.visualization_library.data_loading import load_ground_truth_labels
from tests.helpers import (
    SOURCE_ENTRIES,
    add_labels,
    copy_of,
    labelled_source,
    make_dataset,
)

OWN_VARIANT = "convert-labels-calms21_npy.0.3-bbbbbbbbbb"
"""A variant the copy converts one entry into."""


def test_a_copy_plays_back_its_sources_labels(tmp_path: Path) -> None:
    source = labelled_source(tmp_path / "source")
    copy = copy_of(source, tmp_path / "copy")
    group, sequence = SOURCE_ENTRIES[0]

    labels = load_ground_truth_labels(copy, "behavior", group, sequence)

    assert labels["label_id"].tolist() == [0]
    assert labels["label_name"].tolist() == ["attack"]


def test_an_own_variant_of_an_entry_is_played_back_over_the_linked_one(
    tmp_path: Path,
) -> None:
    source = labelled_source(tmp_path / "source")
    copy = copy_of(source, tmp_path / "copy")
    group, sequence = SOURCE_ENTRIES[0]
    _ = add_labels(copy, group, sequence, [(0, 0, 1, 2)], run_id=OWN_VARIANT)

    labels = load_ground_truth_labels(copy, "behavior", group, sequence)

    assert labels["label_name"].tolist() == ["mount"]


def test_two_own_variants_of_an_entry_raise(tmp_path: Path) -> None:
    source = labelled_source(tmp_path / "source")
    group, sequence = SOURCE_ENTRIES[0]
    _ = add_labels(source, group, sequence, [(0, 0, 1, 2)], run_id=OWN_VARIANT)

    with pytest.raises(ValueError, match="labels_run_id"):
        _ = load_ground_truth_labels(source, "behavior", group, sequence)


def test_absent_labels_raise_file_not_found(tmp_path: Path) -> None:
    empty = make_dataset(tmp_path / "empty")
    source = labelled_source(tmp_path / "source")

    with pytest.raises(FileNotFoundError, match="kind='behavior'"):
        _ = load_ground_truth_labels(empty, "behavior", "g", "s1")
    with pytest.raises(FileNotFoundError, match="sequence='other'"):
        _ = load_ground_truth_labels(source, "behavior", "g", "other")
