"""Tests for the token / display-label split (item 4.1's mosaic half).

The token is what every filename, directory and index join key is built from.
The label is what a human reads. The whole point is that changing the second
touches nothing the first names.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.pipeline.sequence_index import (
    SEQUENCE_LABEL_COLUMNS,
    read_sequence_labels,
    sequence_label_path,
)


def _make_dataset(tmp_path: Path) -> Dataset:
    ds = Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={"tracks": str(tmp_path / "tracks")},
    )
    ds.ensure_roots()
    ds.save()
    return ds


def _add_tracks(ds: Dataset, *entries: tuple[str, str]) -> None:
    tracks = ds.get_root("tracks")
    rows: list[dict[str, str]] = []
    for group, sequence in entries:
        key = f"{group}__{sequence}" if group else sequence
        path = tracks / f"{key}.parquet"
        pd.DataFrame({"frame": np.arange(3), "id": np.zeros(3, dtype=int)}).to_parquet(
            path
        )
        rows.append(
            {
                "run_id": "",
                "group": group,
                "sequence": sequence,
                "abs_path": str(path),
            }
        )
    pd.DataFrame(rows).to_csv(tracks / "index.csv", index=False)


def test_an_unlabelled_sequence_is_called_by_its_token(tmp_path: Path) -> None:
    """The fallback is the ordinary path, not an error case.

    Every dataset predating this file is in exactly this state, so it has to read
    the way it always did with no caller branching.
    """
    ds = _make_dataset(tmp_path)
    assert ds.display_name("", "seqA") == "seqA"
    assert ds.display_name("g", "seqA") == "g__seqA"
    assert ds.display_names() == {}
    assert not sequence_label_path(ds).exists()


def test_a_label_displaces_the_token_for_a_human_and_nothing_else(
    tmp_path: Path,
) -> None:
    ds = _make_dataset(tmp_path)
    _add_tracks(ds, ("", "seqA"))
    before = sorted(p.name for p in ds.get_root("tracks").iterdir())

    ds.set_display_name("", "seqA", "Trial 1, morning")

    assert ds.display_name("", "seqA") == "Trial 1, morning"
    # Nothing the token names has moved.
    assert sorted(p.name for p in ds.get_root("tracks").iterdir()) == before
    assert ds.list_sequences() == ["seqA"]
    assert list(read_tracks_index(ds)["sequence"]) == ["seqA"]


def test_relabelling_replaces_rather_than_accumulates(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    ds.set_display_name("", "seqA", "first")
    ds.set_display_name("", "seqA", "second")

    frame = read_sequence_labels(ds)
    assert list(frame["display_name"]) == ["second"]
    assert len(frame) == 1


def test_clearing_a_label_returns_the_sequence_to_its_token(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    ds.set_display_name("", "seqA", "named")
    ds.set_display_name("", "seqA", "")
    assert ds.display_name("", "seqA") == "seqA"


def test_two_groups_may_label_the_same_sequence_name_differently(
    tmp_path: Path,
) -> None:
    """The token is ``(group, sequence)``, so the label is keyed on both."""
    ds = _make_dataset(tmp_path)
    ds.set_display_name("control", "trial1", "Control, trial 1")
    ds.set_display_name("exp", "trial1", "Experimental, trial 1")

    assert ds.display_name("control", "trial1") == "Control, trial 1"
    assert ds.display_name("exp", "trial1") == "Experimental, trial 1"


def test_a_label_may_contain_what_a_token_may_not(tmp_path: Path) -> None:
    """Spaces, slashes, punctuation -- it never becomes a path component."""
    ds = _make_dataset(tmp_path)
    ds.set_display_name("", "seqA", "Fish 3 / 2026-07-14 (re-run)")
    assert ds.display_name("", "seqA") == "Fish 3 / 2026-07-14 (re-run)"


def test_the_token_is_still_validated_as_one_path_component(tmp_path: Path) -> None:
    """Labelling is not a way to smuggle a slash into an entry name."""
    ds = _make_dataset(tmp_path)
    with pytest.raises(ValueError, match="may not contain"):
        ds.set_display_name("", "a/b", "anything")


def test_the_label_file_lives_at_the_dataset_root(tmp_path: Path) -> None:
    """One file per dataset, not one per source root.

    A composition is a property of (sequence, root); a label is a property of the
    sequence. On a per-root row, a sequence with media and tracks would carry two
    labels that could disagree.
    """
    ds = _make_dataset(tmp_path)
    ds.set_display_name("", "seqA", "named")

    path = sequence_label_path(ds)
    assert path == ds.base_dir / "sequences.csv"
    assert list(pd.read_csv(path, nrows=0).columns) == SEQUENCE_LABEL_COLUMNS


def test_get_sequence_metadata_carries_the_label_beside_the_token(
    tmp_path: Path,
) -> None:
    """Additive: every existing column is a join key and none of them moves."""
    ds = _make_dataset(tmp_path)
    _add_tracks(ds, ("", "seqA"), ("", "seqB"))
    ds.set_display_name("", "seqA", "Trial 1")

    frame = ds.get_sequence_metadata()
    labels = dict(zip(frame["sequence"], frame["display_name"]))
    assert labels == {"seqA": "Trial 1", "seqB": "seqB"}


def test_parse_hierarchy_still_reads_the_token(tmp_path: Path) -> None:
    """A factor parsed out of a freely-relabelled string would change on rename.

    ``validate_entry_name``'s error text steers users to encode hierarchy in the
    token with ``__``; that is where ``level_names`` keeps reading it.
    """
    ds = _make_dataset(tmp_path)
    _add_tracks(ds, ("", "fish3__fast"))
    ds.set_display_name("", "fish3__fast", "something entirely different")

    frame = ds.get_sequence_metadata(level_names=["individual", "speed"])
    assert list(frame["individual"]) == ["fish3"]
    assert list(frame["speed"]) == ["fast"]
