"""Entry names survive the media-index read as the text they were written as.

A ``sequence`` such as ``0066`` is a name, not a number. ``pandas.read_csv``
infers such a column as int64 unless told otherwise, and the ``.astype(str)``
normalization that follows then yields ``"66"`` -- the padding is already gone
by the time it runs, so nothing downstream can tell that a rename happened.

That read is shared by ``match_media_rows`` and ``resolve_media_scope``, which
is how the damage spreads: frame extraction, transcode and every tracker take
their scope from it, so the run lands under a key matching neither the media
index it came from nor the tracks index beside it. A frame run for
``10-fish/0066`` filed itself under ``10-fish__66``, and looking it up by the
name it was asked for found nothing.
"""

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest, open_dataset
from mosaic.core.entry import EntryKeyCollisionError
from mosaic.core.pipeline.media_index import MediaIndexScope
from mosaic.core.pipeline.tracks_raw_index import TracksRawIndexScope
from tests.helpers import (
    MediaClip,
    add_media_sequence,
    add_tracks_table,
    make_dataset,
    track_table,
    write_media_index,
    write_mpeg4_mp4,
)

_HEADER = "name,group,sequence,group_safe,sequence_safe,abs_path,video_order\n"


def _dataset_with_media_rows(base: Path, rows: str) -> Dataset:
    """A dataset whose originals index holds exactly *rows*.

    Hand-seeded rather than scanned: the bug is in the read, so the write must
    stay out of it -- a scan that also lost the padding would make the test pass
    for the wrong reason.
    """
    new_dataset_manifest(name="entry-names", base_dir=base)
    dataset = open_dataset(base)
    index_path = dataset.get_root("media_raw") / "index.csv"
    index_path.write_text(_HEADER + rows)
    return dataset


def test_zero_padded_sequence_survives_the_index_read(tmp_path: Path) -> None:
    dataset = _dataset_with_media_rows(
        tmp_path / "ds",
        "0066.mp4,10-fish,0066,10-fish,0066,media_raw/0066.mp4,0\n",
    )

    matched = dataset.match_media_rows("10-fish", "0066")

    assert list(matched["sequence"]) == ["0066"]


def test_a_padded_sequence_is_not_matched_by_its_unpadded_spelling(
    tmp_path: Path,
) -> None:
    """``66`` and ``0066`` are different entries, and one must not answer for the
    other. Without this the coercion reads as harmless -- the lookup still
    "works", against the wrong row."""
    dataset = _dataset_with_media_rows(
        tmp_path / "ds",
        "0066.mp4,10-fish,0066,10-fish,0066,media_raw/0066.mp4,0\n"
        "66.mp4,10-fish,66,10-fish,66,media_raw/66.mp4,0\n",
    )

    assert list(dataset.match_media_rows("10-fish", "0066")["name"]) == ["0066.mp4"]
    assert list(dataset.match_media_rows("10-fish", "66")["name"]) == ["66.mp4"]


def test_an_all_digit_group_keeps_its_text(tmp_path: Path) -> None:
    """``group`` is coerced by the same read, and a numeric-looking one is
    ordinary -- a plate number, a cohort, a date."""
    dataset = _dataset_with_media_rows(
        tmp_path / "ds",
        "a.mp4,007,0066,007,0066,media_raw/a.mp4,0\n",
    )

    matched = dataset.match_media_rows("007", "0066")

    assert list(matched["group"]) == ["007"]


def test_numeric_columns_are_still_numeric(tmp_path: Path) -> None:
    """The fix reads the *schema's* non-numeric columns as text and leaves the
    rest alone; ``video_order`` sorts the clips of one entry, so widening it to
    text would order clip 10 before clip 2."""
    dataset = _dataset_with_media_rows(
        tmp_path / "ds",
        "a.mp4,g,0066,g,0066,media_raw/a.mp4,2\n"
        "b.mp4,g,0066,g,0066,media_raw/b.mp4,10\n",
    )

    matched = dataset.match_media_rows("g", "0066")

    assert list(matched["name"]) == ["a.mp4", "b.mp4"]
    assert list(matched["video_order"]) == [2, 10]


def test_a_media_write_refuses_a_key_another_entry_already_names(
    tmp_path: Path,
) -> None:
    """``("", "R__0900")`` would be stored under the key of ``("R", "0900")``."""
    dataset = make_dataset(tmp_path / "ds")
    write_media_index(
        dataset, [MediaClip(group="R", sequence="0900", filename="R-0900.mp4")]
    )

    with pytest.raises(EntryKeyCollisionError, match="R__0900"):
        add_media_sequence(dataset, "R__0900", videos=("a.mp4",))

    assert [str(row["sequence"]) for row in dataset.read_media_index()] == ["0900"]


def test_a_rescan_keeps_an_assigned_entry_whose_filename_spells_its_key(
    tmp_path: Path,
) -> None:
    """The check reads the entry a row is written under, not the filename's guess.

    A scan guesses ``("", "R__0900")`` from ``R__0900.mp4`` and then restores
    the assigned ``("R", "0900")``. The guess shares the assigned entry's key,
    and no row is written under it.
    """
    dataset = make_dataset(tmp_path / "ds")
    directory = dataset.get_root("media_raw") / "R"
    write_mpeg4_mp4(directory / "R__0900.mp4")
    _ = dataset.write_media_index(
        [MediaIndexScope(directory=directory, group="R", sequence="0900")],
        extensions=(".mp4",),
    )

    _ = dataset.index_media([directory], extensions=(".mp4",))

    rows = dataset.read_media_index()
    assert [
        (row["group"], row["sequence"], row["assignment_source"]) for row in rows
    ] == [("R", "0900", "assigned")]


def test_a_media_write_may_rename_an_entry_onto_the_key_it_replaces(
    tmp_path: Path,
) -> None:
    """The old entry leaves the index with its rows and frees its key."""
    dataset = make_dataset(tmp_path / "ds")
    directory = dataset.get_root("media_raw") / "R"
    write_mpeg4_mp4(directory / "a.mp4")
    for group, sequence in (("R", "0900"), ("", "R__0900")):
        _ = dataset.write_media_index(
            [MediaIndexScope(directory=directory, group=group, sequence=sequence)],
            extensions=(".mp4",),
        )

    rows = dataset.read_media_index()
    assert [(row["group"], row["sequence"]) for row in rows] == [("", "R__0900")]


def test_a_tracks_raw_write_may_rename_an_entry_onto_the_key_it_replaces(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    directory = dataset.get_root("tracks_raw") / "R"
    directory.mkdir(parents=True)
    _ = (directory / "a.csv").write_text("frame,x\n0,1\n")
    for group, sequence in (("R", "0900"), ("", "R__0900")):
        _ = dataset.write_tracks_raw_index(
            [
                TracksRawIndexScope(
                    directory=directory,
                    group=group,
                    sequence=sequence,
                    src_format="trex",
                )
            ],
            patterns="*.csv",
        )

    rows = dataset.read_tracks_raw_index()
    assert [(row["group"], row["sequence"]) for row in rows] == [("", "R__0900")]


def test_a_media_write_refuses_a_rename_while_the_tracks_index_names_the_entry(
    tmp_path: Path,
) -> None:
    """The tracks index still names the old entry, whose key the new one shares."""
    dataset = make_dataset(tmp_path / "ds")
    directory = dataset.get_root("media_raw") / "R"
    write_mpeg4_mp4(directory / "a.mp4")
    _ = dataset.write_media_index(
        [MediaIndexScope(directory=directory, group="R", sequence="0900")],
        extensions=(".mp4",),
    )
    add_tracks_table(dataset, "run", track_table("R", "0900"))

    with pytest.raises(EntryKeyCollisionError, match="R__0900"):
        _ = dataset.write_media_index(
            [MediaIndexScope(directory=directory, group="", sequence="R__0900")],
            extensions=(".mp4",),
        )

    rows = dataset.read_media_index()
    assert [(row["group"], row["sequence"]) for row in rows] == [("R", "0900")]


def test_a_tracks_raw_write_refuses_a_rename_while_the_tracks_index_names_the_entry(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    directory = dataset.get_root("tracks_raw") / "R"
    directory.mkdir(parents=True)
    _ = (directory / "a.csv").write_text("frame,x\n0,1\n")

    def write_raw(group: str, sequence: str) -> None:
        _ = dataset.write_tracks_raw_index(
            [
                TracksRawIndexScope(
                    directory=directory,
                    group=group,
                    sequence=sequence,
                    src_format="trex",
                )
            ],
            patterns="*.csv",
        )

    write_raw("R", "0900")
    add_tracks_table(dataset, "run", track_table("R", "0900"))

    with pytest.raises(EntryKeyCollisionError, match="R__0900"):
        write_raw("", "R__0900")

    rows = dataset.read_tracks_raw_index()
    assert [(row["group"], row["sequence"]) for row in rows] == [("R", "0900")]


def test_a_media_write_refuses_a_rename_onto_a_key_another_row_keeps(
    tmp_path: Path,
) -> None:
    """A row outside the write's directories keeps the old entry in the index."""
    dataset = make_dataset(tmp_path / "ds")
    media_root = dataset.get_root("media_raw")
    for name in ("first", "second"):
        write_mpeg4_mp4(media_root / name / "a.mp4")
    _ = dataset.write_media_index(
        [
            MediaIndexScope(directory=media_root / name, group="R", sequence="0900")
            for name in ("first", "second")
        ],
        extensions=(".mp4",),
    )

    with pytest.raises(EntryKeyCollisionError, match="R__0900"):
        _ = dataset.write_media_index(
            [
                MediaIndexScope(
                    directory=media_root / "first", group="", sequence="R__0900"
                )
            ],
            extensions=(".mp4",),
        )


def test_an_empty_group_request_does_not_match_a_continuous_group(
    tmp_path: Path,
) -> None:
    """A continuous group's sequence is named with its recording.

    Before the declaration ``rec`` is a namespace, and a request without a group
    matches its row through the sequence-cell tier. After it, the request
    matches no row.
    """
    dataset = _dataset_with_media_rows(
        tmp_path / "ds",
        "rec-0900.mp4,rec,0900,rec,0900,media_raw/rec-0900.mp4,0\n",
    )
    assert list(dataset.match_media_rows("", "0900")["group"]) == ["rec"]

    dataset.set_continuous_groups(["rec"])

    with pytest.raises(FileNotFoundError, match="0900"):
        _ = dataset.match_media_rows("", "0900")
    assert list(dataset.match_media_rows("rec", "0900")["group"]) == ["rec"]


def test_a_continuous_group_places_its_media_beside_two_tracks_variants(
    tmp_path: Path,
) -> None:
    """Two labeled tracks variants leave each sequence's place readable.

    Reading the frame extents without a run id refuses an entry with two
    labeled variants. A sequence's first frame counts the media frames of the
    sequences before it in a natural sort on their names, without reading a
    tracks run. Each sequence resolves to the rows of its entry.
    """
    dataset = make_dataset(tmp_path / "ds")
    write_media_index(
        dataset,
        [
            MediaClip(sequence="s10", group="rec", filename="s10.mp4", frame_count=30),
            MediaClip(sequence="s2", group="rec", filename="s2.mp4", frame_count=40),
        ],
    )
    for run_id in ("convert.a", "convert.b"):
        add_tracks_table(dataset, run_id, track_table("rec", "s2"))
        add_tracks_table(dataset, run_id, track_table("rec", "s10", start_frame=40))
    dataset.set_continuous_groups(["rec"])

    assert list(dataset.match_media_rows("rec", "s10")["sequence"]) == ["s10"]
    assert dataset.resolve_media("rec", "s10").first_frame == 40


@pytest.fixture
def declared_recording(tmp_path: Path) -> Dataset:
    """A dataset whose continuous group ``R20210219`` has one sequence."""
    dataset = make_dataset(tmp_path / "ds")
    add_tracks_table(dataset, "convert.a", track_table("R20210219", "partA"))
    dataset.set_continuous_groups(["R20210219"])
    return dataset


def test_a_media_write_refuses_a_sequence_named_like_a_continuous_group(
    declared_recording: Dataset,
) -> None:
    """The reverse of the declaration's refusal: two recordings would share a name."""
    with pytest.raises(EntryKeyCollisionError, match="continuous group 'R20210219'"):
        add_media_sequence(declared_recording, "R20210219", videos=("a.mp4",))

    assert declared_recording.read_media_index() == []


def test_a_tracks_write_refuses_a_sequence_named_like_a_continuous_group(
    declared_recording: Dataset,
) -> None:
    with pytest.raises(EntryKeyCollisionError, match="continuous group 'R20210219'"):
        add_tracks_table(declared_recording, "convert.a", track_table("", "R20210219"))

    assert declared_recording.list_entries() == [("R20210219", "partA")]


def test_a_sequence_indexed_before_the_group_was_declared_keeps_resolving(
    tmp_path: Path,
) -> None:
    """Only a new entry is refused, as for a key that two entries share."""
    dataset = make_dataset(tmp_path / "ds")
    add_tracks_table(dataset, "convert.a", track_table("R20210219", "partA"))
    add_tracks_table(dataset, "convert.a", track_table("", "R20210219"))
    dataset.manifest = dataset.manifest.model_copy(
        update={"continuous_groups": ("R20210219",)}
    )

    add_tracks_table(dataset, "convert.b", track_table("", "R20210219"))

    assert ("", "R20210219") in dataset.list_entries()
