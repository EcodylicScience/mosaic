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

from mosaic.core.dataset import Dataset, new_dataset_manifest, open_dataset

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
