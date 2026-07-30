"""The declared per-entry layout -- item 9.2, and the empty group it must survive.

``group`` is an optional namespace and is empty on every dataset the control
plane creates, so the empty-group case is the *common* one rather than an edge.
These pin that the layout expresses it, that a two-level reading does not, and
that ``index_media`` can now read the layout back -- which nothing in mosaic
could do while the control plane was already writing it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.helpers import entry_directory, make_entry_key, parse_entry_key


def _write_mp4(path: Path, nframes: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for _ in range(nframes):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()


def _dataset(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest(name="layout", base_dir=tmp_path / "ds")
    return Dataset(manifest_path=manifest).load(ensure_roots=True)


# --- One level, and why -------------------------------------------------------


def test_an_empty_group_yields_one_level(tmp_path: Path) -> None:
    """The common case: every dataset the control plane creates has group=""."""
    assert entry_directory(tmp_path, "", "seq_a") == tmp_path / "seq_a"


def test_a_group_is_joined_not_nested(tmp_path: Path) -> None:
    """``a__b``, not ``a/b`` -- the convention every other root already uses."""
    assert entry_directory(tmp_path, "a", "b") == tmp_path / "a__b"


def test_the_layout_has_no_collision_a_two_level_one_would(tmp_path: Path) -> None:
    """Why one level rather than ``<group>/<sequence>``, stated as a test.

    Under two levels ``("", "a")`` and ``("a", "b")`` both want ``root/a`` -- one
    as a sequence directory, one as a group directory -- because ``Path(root) /
    "" / "seq"`` silently collapses. The entry key cannot collide that way.
    """
    flat = entry_directory(tmp_path, "", "a")
    grouped = entry_directory(tmp_path, "a", "b")

    assert flat != grouped.parent, "a two-level layout would nest under the other"
    assert flat.parent == grouped.parent == tmp_path
    # And the collapse that motivates it is real.
    assert tmp_path / "" / "a" == tmp_path / "a"


def test_a_directory_name_round_trips_to_its_entry(tmp_path: Path) -> None:
    """The layout is only readable if the key parses back."""
    for group, sequence in (("", "seq_a"), ("cohort", "seq_a"), ("", "odd name")):
        key = make_entry_key(group, sequence)
        assert parse_entry_key(key) == (group, sequence)


def test_a_sequence_containing_the_separator_keeps_it(tmp_path: Path) -> None:
    """Split on the *first* ``__``: the group is one level, the rest is sequence.

    ``parse_hierarchy`` reads further ``__`` as deeper levels, so a sequence may
    legitimately contain one and it must not be shorn off here.
    """
    assert parse_entry_key("cohort__day1__trial2") == ("cohort", "day1__trial2")


# --- Reading the layout back --------------------------------------------------


def test_index_media_reads_the_sequence_from_the_directory(
    tmp_path: Path, requires_ffprobe: None
) -> None:
    """Two clips in one entry directory are one sequence, not two.

    Under the stem heuristic they are two, which is the gap: the control plane
    has been writing this layout all along and nothing in mosaic could read it.
    """
    ds = _dataset(tmp_path)
    home = entry_directory(ds.get_root("media_raw"), "", "trial7")
    _write_mp4(home / "part1.mp4")
    _write_mp4(home / "part2.mp4")

    indexed = pd.read_csv(
        ds.index_media(
            [ds.get_root("media_raw")],
            extensions=(".mp4",),
            media_layout="per_sequence",
        )
    )

    assert set(indexed["sequence"]) == {"trial7"}
    assert len(indexed) == 2


def test_a_grouped_entry_directory_yields_both_halves(
    tmp_path: Path, requires_ffprobe: None
) -> None:
    """The group is recovered too, or the layout is only half readable."""
    ds = _dataset(tmp_path)
    home = entry_directory(ds.get_root("media_raw"), "cohortA", "trial7")
    _write_mp4(home / "part1.mp4")

    indexed = pd.read_csv(
        ds.index_media(
            [ds.get_root("media_raw")],
            extensions=(".mp4",),
            media_layout="per_sequence",
        )
    )

    assert list(indexed["group"]) == ["cohortA"]
    assert list(indexed["sequence"]) == ["trial7"]


def test_the_flat_layout_is_grandfathered(
    tmp_path: Path, requires_ffprobe: None
) -> None:
    """The default is unchanged, so no existing dataset is re-identified.

    ``sequence_match_mode="prefix"`` exists to serve split recordings under the
    flat layout; flipping the default would silently re-identify every dataset
    that relies on it.
    """
    ds = _dataset(tmp_path)
    _write_mp4(ds.get_root("media_raw") / "trial7.mp4")

    indexed = pd.read_csv(
        ds.index_media([ds.get_root("media_raw")], extensions=(".mp4",))
    )

    assert list(indexed["sequence"]) == ["trial7"]


def test_an_unknown_layout_is_refused(tmp_path: Path) -> None:
    """A typo must not silently fall back to the other mode.

    The parameter is a ``Literal``, so a bad *literal* is already a type error --
    which is the point, and why this passes a ``str`` instead. The runtime guard
    is for the callers whose value did not come from source: a CLI flag, a JSON
    payload, a queue spec.
    """
    ds = _dataset(tmp_path)
    from_the_wire: str = "per-sequence"

    with pytest.raises(ValueError, match="media_layout must be"):
        _ = ds.index_media([ds.get_root("media_raw")], media_layout=from_the_wire)
