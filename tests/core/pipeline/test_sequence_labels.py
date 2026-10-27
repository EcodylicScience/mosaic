"""Tests for the token / display-label split (item 4.1's mosaic half).

The token is what every filename, directory and index join key is built from.
The label is what a human reads. The whole point is that changing the second
touches nothing the first names.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.pipeline.sequence_index import (
    SEQUENCE_LABEL_COLUMNS,
    SequenceLabelRow,
    amend_sequence_labels,
    read_sequence_labels,
    sequence_label_path,
    sequence_labels,
)

from tests.helpers import (
    DIVISION_CREATED_UTC,
    add_track_sequences,
    index_locked_out,
    make_dataset,
    store_dataset,
    video_store_maker,
    write_division_rows,
)

# Only ``tracks`` is declared: a label is a property of the sequence, and these
# tests read it beside the one index that names sequences.
TRACKS_ONLY = ("tracks",)


def test_an_unlabelled_sequence_is_called_by_its_token(tmp_path: Path) -> None:
    """The fallback is the ordinary path, not an error case.

    Every dataset predating this file is in exactly this state, so it has to read
    the way it always did with no caller branching.
    """
    ds = make_dataset(tmp_path, roots=TRACKS_ONLY)
    assert ds.display_name("", "seqA") == "seqA"
    assert ds.display_name("g", "seqA") == "g__seqA"
    assert ds.display_names() == {}
    assert not sequence_label_path(ds).exists()


def test_a_label_displaces_the_token_for_a_human_and_nothing_else(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path, roots=TRACKS_ONLY)
    add_track_sequences(ds, "seqA")
    before = sorted(p.name for p in ds.get_root("tracks").iterdir())

    ds.set_display_name("", "seqA", "Trial 1, morning")

    assert ds.display_name("", "seqA") == "Trial 1, morning"
    # Nothing the token names has moved.
    assert sorted(p.name for p in ds.get_root("tracks").iterdir()) == before
    assert ds.list_sequences() == ["seqA"]
    assert list(read_tracks_index(ds)["sequence"]) == ["seqA"]


def test_relabelling_replaces_rather_than_accumulates(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path, roots=TRACKS_ONLY)
    ds.set_display_name("", "seqA", "first")
    ds.set_display_name("", "seqA", "second")

    frame = read_sequence_labels(ds)
    assert list(frame["display_name"]) == ["second"]
    assert len(frame) == 1


def test_clearing_a_label_returns_the_sequence_to_its_token(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path, roots=TRACKS_ONLY)
    ds.set_display_name("", "seqA", "named")
    ds.set_display_name("", "seqA", "")
    assert ds.display_name("", "seqA") == "seqA"


def test_relabelling_keeps_the_sequence_s_lineage(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path, roots=TRACKS_ONLY)
    sequence_labels(sequence_label_path(ds)).append(
        [SequenceLabelRow(sequence="seqA", derived_from="trex.1.0-abcdef0123")]
    )

    ds.set_display_name("", "seqA", "named")

    row = read_sequence_labels(ds).iloc[0]
    assert (row["display_name"], row["derived_from"]) == (
        "named",
        "trex.1.0-abcdef0123",
    )


def test_an_amendment_reads_its_row_under_the_label_lock(tmp_path: Path) -> None:
    """A second writer cannot land between the row an amendment reads and its append.

    Read outside the lock, two writers of one entry both amend the same prior
    row, and the later append drops the earlier writer's cell.
    """
    ds = make_dataset(tmp_path, roots=TRACKS_ONLY)
    path = sequence_label_path(ds)
    locked_out: list[bool] = []

    def amend(row: SequenceLabelRow) -> SequenceLabelRow:
        locked_out.append(index_locked_out(path, timeout=0.2))
        return dataclasses.replace(row, display_name="named")

    amend_sequence_labels(ds, [("", "seqA")], amend)

    assert locked_out == [True]
    assert ds.display_name("", "seqA") == "named"


def test_a_label_file_without_a_header_reads_as_no_labels(tmp_path: Path) -> None:
    """The lock creates the file empty before a first writer's append lands."""
    ds = make_dataset(tmp_path, roots=TRACKS_ONLY)
    sequence_label_path(ds).touch()

    assert read_sequence_labels(ds).empty
    assert ds.display_names() == {}


def test_two_groups_may_label_the_same_sequence_name_differently(
    tmp_path: Path,
) -> None:
    """The token is ``(group, sequence)``, so the label is keyed on both."""
    ds = make_dataset(tmp_path, roots=TRACKS_ONLY)
    ds.set_display_name("control", "trial1", "Control, trial 1")
    ds.set_display_name("exp", "trial1", "Experimental, trial 1")

    assert ds.display_name("control", "trial1") == "Control, trial 1"
    assert ds.display_name("exp", "trial1") == "Experimental, trial 1"


def test_a_label_may_contain_what_a_token_may_not(tmp_path: Path) -> None:
    """Spaces, slashes, punctuation -- it never becomes a path component."""
    ds = make_dataset(tmp_path, roots=TRACKS_ONLY)
    ds.set_display_name("", "seqA", "Fish 3 / 2026-07-14 (re-run)")
    assert ds.display_name("", "seqA") == "Fish 3 / 2026-07-14 (re-run)"


def test_the_token_is_still_validated_as_one_path_component(tmp_path: Path) -> None:
    """Labelling is not a way to smuggle a slash into an entry name."""
    ds = make_dataset(tmp_path, roots=TRACKS_ONLY)
    with pytest.raises(ValueError, match="may not contain"):
        ds.set_display_name("", "a/b", "anything")


def test_the_label_file_lives_at_the_dataset_root(tmp_path: Path) -> None:
    """One file per dataset, not one per source root.

    A composition is a property of (sequence, root); a label is a property of the
    sequence. On a per-root row, a sequence with media and tracks would carry two
    labels that could disagree.
    """
    ds = make_dataset(tmp_path, roots=TRACKS_ONLY)
    ds.set_display_name("", "seqA", "named")

    path = sequence_label_path(ds)
    assert path == ds.base_dir / "sequences.csv"
    assert list(pd.read_csv(path, nrows=0).columns) == SEQUENCE_LABEL_COLUMNS


def test_get_sequence_metadata_carries_the_label_beside_the_token(
    tmp_path: Path,
) -> None:
    """Additive: every existing column is a join key and none of them moves."""
    ds = make_dataset(tmp_path, roots=TRACKS_ONLY)
    add_track_sequences(ds, "seqA", "seqB")
    ds.set_display_name("", "seqA", "Trial 1")

    frame = ds.get_sequence_metadata()
    labels = dict(zip(frame["sequence"], frame["display_name"]))
    assert labels == {"seqA": "Trial 1", "seqB": "seqB"}


def test_parse_hierarchy_still_reads_the_token(tmp_path: Path) -> None:
    """A factor parsed out of a freely-relabelled string would change on rename.

    ``validate_entry_name``'s error text steers users to encode hierarchy in the
    token with ``__``; that is where ``level_names`` keeps reading it.
    """
    ds = make_dataset(tmp_path, roots=TRACKS_ONLY)
    add_track_sequences(ds, "fish3__fast")
    ds.set_display_name("", "fish3__fast", "something entirely different")

    frame = ds.get_sequence_metadata(level_names=["individual", "speed"])
    assert list(frame["individual"]) == ["fish3"]
    assert list(frame["speed"]) == ["fast"]


# --- the wall-clock name of a division --------------------------------------------

DIVISION_TOKENS = ("20210219T192253Z", "20210219T192255Z", "20210219T192258Z")
"""The sequences of a store of chunks of 5 frames at 4 fps, divided every 10."""


def divided_store_dataset(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    store_metadata: Mapping[str, object] | None = None,
) -> Dataset:
    """A dataset whose store ``rec`` is indexed whole, ready to divide."""
    return store_dataset(
        tmp_path,
        make_media_dataset,
        video_store_maker(
            (5, 5, 5, 5, 2),
            fps=4.0,
            created_utc=DIVISION_CREATED_UTC,
            store_metadata=store_metadata,
        ),
    )


def divide(dataset: Dataset) -> None:
    """Divide the store ``rec`` of *dataset* every 10 frames."""
    _ = write_division_rows(
        dataset, dataset.get_root("media_raw") / "recordings" / "rec", 10
    )


@pytest.mark.media
def test_a_division_is_named_by_its_start_on_the_store_s_local_clock(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    dataset = divided_store_dataset(
        tmp_path, make_media_dataset, {"timezone_local": "America/Los_Angeles"}
    )

    divide(dataset)

    assert dataset.display_names() == {
        ("rec", DIVISION_TOKENS[0]): "2021-02-19 11:22:53 PST",
        ("rec", DIVISION_TOKENS[1]): "2021-02-19 11:22:55 PST",
        ("rec", DIVISION_TOKENS[2]): "2021-02-19 11:22:58 PST",
    }


@pytest.mark.media
def test_a_store_without_a_time_zone_names_its_divisions_in_utc(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    dataset = divided_store_dataset(tmp_path, make_media_dataset)

    divide(dataset)

    assert dataset.display_name("rec", DIVISION_TOKENS[0]) == "2021-02-19 19:22:53Z"


@pytest.mark.media
def test_a_division_keeps_its_label_and_a_later_write_names_the_unlabeled(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    dataset = divided_store_dataset(tmp_path, make_media_dataset)
    divide(dataset)
    dataset.set_display_name("rec", DIVISION_TOKENS[1], "the middle")
    dataset.set_display_name("rec", DIVISION_TOKENS[2], "")

    divide(dataset)

    assert dataset.display_names() == {
        ("rec", DIVISION_TOKENS[0]): "2021-02-19 19:22:53Z",
        ("rec", DIVISION_TOKENS[1]): "the middle",
        ("rec", DIVISION_TOKENS[2]): "2021-02-19 19:22:58Z",
    }


@pytest.mark.media
def test_a_division_written_before_its_label_is_named_by_a_later_write(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    dataset = divided_store_dataset(tmp_path, make_media_dataset)
    divide(dataset)
    sequence_label_path(dataset).unlink()

    divide(dataset)

    assert len(dataset.display_names()) == 3


@pytest.mark.media
def test_a_division_write_labels_a_row_without_a_name_and_keeps_its_other_cells(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    dataset = divided_store_dataset(tmp_path, make_media_dataset)
    divide(dataset)
    path = sequence_label_path(dataset)
    path.unlink()
    sequence_labels(path).append(
        [
            SequenceLabelRow(
                group="rec",
                sequence=DIVISION_TOKENS[1],
                display_group="g2",
                derived_from="run123",
            )
        ]
    )

    divide(dataset)

    frame = read_sequence_labels(dataset)
    row = frame[frame["sequence"] == DIVISION_TOKENS[1]].iloc[0]
    assert (row["display_name"], row["display_group"], row["derived_from"]) == (
        "2021-02-19 19:22:55Z",
        "g2",
        "run123",
    )
