"""Row selection and verdict-based routing in :meth:`Dataset.resolve_media`."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest
from mosaic_media import MediaProbeError, probe_media

from mosaic.core.dataset import AmbiguousMediaMatchError, Dataset
from mosaic.core.helpers import to_safe_name
from mosaic.core.media.facts_columns import MEDIA_INDEX_COLUMNS

from tests.helpers import write_mpeg4_mp4


def _make_dataset(tmp_path: Path) -> Dataset:
    for sub in ("media_raw", "media", "tracks"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={
            "media_raw": str(tmp_path / "media_raw"),
            "media": str(tmp_path / "media"),
            "tracks": str(tmp_path / "tracks"),
        },
    )


def _facts_json(path: Path) -> str:
    return json.dumps(dataclasses.asdict(probe_media(path)))


def _row(
    *,
    group: str,
    sequence: str,
    abs_path: Path,
    analysis_transcode: str = "",
    analysis_derivative_path: str = "",
    playback_derivative_path: str = "",
    source_path: str = "",
    camera: str = "",
    video_order: int = 0,
) -> dict[str, object]:
    """Build a full media-index row for *abs_path* with a real facts JSON cell."""
    return {
        "name": abs_path.name,
        "group": group,
        "sequence": sequence,
        "group_safe": to_safe_name(group) if group else "",
        "sequence_safe": to_safe_name(sequence),
        "abs_path": str(abs_path),
        "size_bytes": abs_path.stat().st_size,
        "mtime_iso": "",
        "width": 64,
        "height": 48,
        "fps": 30.0,
        "codec": "h264",
        "media_type": "video",
        "frame_count": probe_media(abs_path).frame_count,
        "analysis_transcode": analysis_transcode,
        "stream_transcode": "",
        "analysis_derivative_path": analysis_derivative_path,
        "playback_derivative_path": playback_derivative_path,
        "source_path": source_path,
        "media_facts": _facts_json(abs_path),
        "camera": camera,
        "video_order": video_order,
    }


def _write_index(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows, columns=MEDIA_INDEX_COLUMNS).to_csv(path, index=False)


def test_clean_row_returns_original_with_stored_facts(tmp_path: Path):
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "clean.mp4"
    write_mpeg4_mp4(original, frames=6)
    # The stored cell deliberately disagrees with the file (a fresh probe of
    # this clip returns 6), so a returned count of 999 can only have come from
    # the stored cell, never from a re-probe.
    stored_facts = dataclasses.replace(probe_media(original), frame_count=999)
    row = _row(group="g1", sequence="clean", abs_path=original)
    row["media_facts"] = json.dumps(dataclasses.asdict(stored_facts))
    _write_index(tmp_path / "media_raw" / "index.csv", [row])

    resolved = ds.resolve_media("g1", "clean")
    assert [p.resolve() for p in resolved.paths] == [original.resolve()]
    assert resolved.facts[0].frame_count == 999


def test_row_with_an_unreconstructable_facts_cell_raises_the_reprobe_remedy(
    tmp_path: Path,
):
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "clean.mp4"
    write_mpeg4_mp4(original, frames=6)
    row = _row(group="g1", sequence="clean", abs_path=original)
    # A media_facts cell written before the identity fields existed: it lacks
    # the keys MediaFacts now requires, so it cannot be reconstructed.
    row["media_facts"] = json.dumps({"width": 64, "height": 48})
    _write_index(tmp_path / "media_raw" / "index.csv", [row])

    with pytest.raises(
        MediaProbeError, match="mosaic reprobe-media --apply"
    ) as excinfo:
        ds.resolve_media("g1", "clean")
    assert str(original) in str(excinfo.value)


def test_row_with_no_facts_cell_raises_the_reprobe_remedy(tmp_path: Path):
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "clean.mp4"
    write_mpeg4_mp4(original, frames=6)
    row = _row(group="g1", sequence="clean", abs_path=original)
    row["media_facts"] = ""
    _write_index(tmp_path / "media_raw" / "index.csv", [row])

    with pytest.raises(MediaProbeError, match="mosaic reprobe-media --apply"):
        ds.resolve_media("g1", "clean")


def test_a_required_row_with_no_facts_still_reports_the_transcode_remedy(
    tmp_path: Path,
):
    """A required-and-unlinked row that also carries no facts reports the
    transcode remedy, not the reprobe remedy: the file the caller would open
    is the derivative, so the derivative's absence is the fault to report."""
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "orphan.mp4"
    write_mpeg4_mp4(original, frames=6)
    row = _row(
        group="g1",
        sequence="orphan",
        abs_path=original,
        analysis_transcode="required",
        analysis_derivative_path="",
    )
    row["media_facts"] = ""
    _write_index(tmp_path / "media_raw" / "index.csv", [row])

    with pytest.raises(
        MediaProbeError, match="requires an analysis transcode"
    ) as excinfo:
        ds.resolve_media("g1", "orphan")
    assert "reprobe-media" not in str(excinfo.value)


def test_required_row_routes_to_derivative_with_derivative_facts(tmp_path):
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "needs.mp4"
    write_mpeg4_mp4(original, frames=6)
    derivative = tmp_path / "media" / "g1__needs.analysis.mp4"
    write_mpeg4_mp4(derivative, frames=10)

    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [
            _row(
                group="g1",
                sequence="needs",
                abs_path=original,
                analysis_transcode="required",
                analysis_derivative_path="g1__needs.analysis.mp4",
            )
        ],
    )
    _write_index(
        tmp_path / "media" / "index.csv",
        [
            _row(
                group="g1",
                sequence="needs",
                abs_path=derivative,
                source_path="needs.mp4",
            )
        ],
    )

    resolved = ds.resolve_media("g1", "needs")
    assert [p.resolve() for p in resolved.paths] == [derivative.resolve()]
    # Derivative facts, not the original's (distinct frame counts).
    assert resolved.facts[0].frame_count == probe_media(derivative).frame_count
    assert resolved.facts[0].frame_count != probe_media(original).frame_count


def test_required_row_without_derivative_raises(tmp_path):
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "orphan.mp4"
    write_mpeg4_mp4(original, frames=6)
    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [
            _row(
                group="g1",
                sequence="orphan",
                abs_path=original,
                analysis_transcode="required",
                analysis_derivative_path="",
            )
        ],
    )

    with pytest.raises(MediaProbeError, match="requires an analysis transcode"):
        ds.resolve_media("g1", "orphan")


def test_legacy_media_only_required_row_raises(tmp_path):
    """A legacy dataset with only a ``media`` root (no ``media_raw`` split).

    A required row has no silent-degrade arm: routing must fail loud, telling
    the user to adopt the media_raw/media split and transcode, rather than
    opening the defective original.
    """
    (tmp_path / "media").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tracks").mkdir(parents=True, exist_ok=True)
    ds = Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={
            "media": str(tmp_path / "media"),
            "tracks": str(tmp_path / "tracks"),
        },
    )
    assert ds.resolve_media_root() == "media"
    original = tmp_path / "media" / "legacy.mp4"
    write_mpeg4_mp4(original, frames=6)
    _write_index(
        tmp_path / "media" / "index.csv",
        [
            _row(
                group="",
                sequence="legacy",
                abs_path=original,
                analysis_transcode="required",
            )
        ],
    )

    with pytest.raises(MediaProbeError, match="requires an analysis transcode"):
        ds.resolve_media("", "legacy")


def test_a_sequence_never_resolves_another_sequences_media(tmp_path: Path):
    """A sequence with no row of its own must not borrow one whose filename
    merely contains its name. Resolving ``clip`` to ``clip_a.mp4`` would
    register one sequence's frames under another's name."""
    ds = _make_dataset(tmp_path)
    other = tmp_path / "media_raw" / "clip_a.mp4"
    write_mpeg4_mp4(other, frames=6)
    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [_row(group="", sequence="clip_a", abs_path=other)],
    )

    with pytest.raises(FileNotFoundError, match=r"sequence 'clip'"):
        ds.resolve_media("", "clip")


def test_a_sequence_name_is_matched_literally_not_as_a_regex(tmp_path: Path):
    """``clip.a`` is not a literal substring of ``clipXa.mp4``; only regex
    interpretation of the requested name makes the two meet."""
    ds = _make_dataset(tmp_path)
    other = tmp_path / "media_raw" / "clipXa.mp4"
    write_mpeg4_mp4(other, frames=6)
    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [_row(group="", sequence="clipXa", abs_path=other)],
    )

    with pytest.raises(FileNotFoundError):
        ds.match_media_rows("", "clip.a")


def test_a_sequence_name_holding_a_regex_metacharacter_reports_no_match(
    tmp_path: Path,
):
    """An unmatched sequence reports no match whatever characters it holds. A
    name compiled as a pattern fails with a regex error instead, which no
    caller expects from a lookup."""
    ds = _make_dataset(tmp_path)
    other = tmp_path / "media_raw" / "clip_a.mp4"
    write_mpeg4_mp4(other, frames=6)
    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [_row(group="", sequence="clip_a", abs_path=other)],
    )

    with pytest.raises(FileNotFoundError):
        ds.match_media_rows("", "clip(1")


@pytest.mark.parametrize(
    ("requested", "resolves"),
    [
        ("trial", True),
        ("TRIAL", True),
        ("trial.mp4", True),
        ("trial.avi", True),
        ("trial.1", False),
        ("trial.v2", False),
        ("trial_a", False),
        ("trial.mp4.mp4", False),
        ("trial.", False),
        ("sub/trial.mp4", False),
    ],
)
def test_only_a_media_extension_is_stripped_from_a_request(
    tmp_path: Path, requested: str, resolves: bool
):
    """The tier bridges a request that carries a media extension the entry's own
    name lacks. A dotted suffix that is not one -- ``trial.1`` beside an entry
    ``trial`` -- names a different recording, and entry names carrying dots are
    ordinary: ``cam1.left`` and ``session.v2`` are real.

    Exactly one extension goes, and only from the end of the whole request: a
    second one, a bare trailing dot, and a leading directory all leave a name
    that is not the entry's. Which extension it is is deliberately not checked
    against the row's own file, so a caller naming a derivative's ``.mp4`` still
    reaches an entry whose original is raw ``.h264``.
    """
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "trial.mp4"
    write_mpeg4_mp4(original, frames=6)
    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [_row(group="", sequence="trial", abs_path=original)],
    )

    if resolves:
        assert list(ds.match_media_rows("", requested)["sequence"]) == ["trial"]
    else:
        with pytest.raises(FileNotFoundError):
            _ = ds.match_media_rows("", requested)


def test_an_entry_whose_own_name_ends_in_an_extension_matches_it_whole(
    tmp_path: Path,
):
    """An entry may be named ``trial.MP4``. Nothing above this tier reaches it
    from ``trial.mp4`` -- the exact tier is case-sensitive and so are safe
    names -- and stripping the extension leaves ``trial``, which is a different
    entry. Only comparing the request whole, alongside the stripped form,
    resolves it."""
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "recording.mp4"
    write_mpeg4_mp4(original, frames=6)
    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [_row(group="", sequence="trial.MP4", abs_path=original)],
    )

    assert list(ds.match_media_rows("", "trial.mp4")["sequence"]) == ["trial.MP4"]


def test_a_sequence_named_for_its_file_still_matches_that_row(tmp_path: Path):
    """A request carrying a file name where the row records the bare entry name.
    Narrowing this tier must keep it."""
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "clip_a.mp4"
    write_mpeg4_mp4(original, frames=6)
    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [_row(group="", sequence="clip_a", abs_path=original)],
    )

    resolved = ds.resolve_media("", "clip_a.mp4")
    assert [p.resolve() for p in resolved.paths] == [original.resolve()]


def test_a_sequence_differing_only_in_case_still_matches_its_row(tmp_path: Path):
    """The fallback's other legitimate shape: safe names are case-preserving,
    so only a case-insensitive comparison resolves ``CLIP_A`` to ``clip_a``."""
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "clip_a.mp4"
    write_mpeg4_mp4(original, frames=6)
    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [_row(group="", sequence="clip_a", abs_path=original)],
    )

    resolved = ds.resolve_media("", "CLIP_A")
    assert [p.resolve() for p in resolved.paths] == [original.resolve()]


def test_a_sequence_name_two_groups_share_raises_rather_than_guessing(
    tmp_path: Path,
):
    """Two groups holding a sequence of the same name is ordinary. Asked for
    that name with no group, the fallback has no ground to choose between them,
    so it must refuse instead of returning one group's media, or both groups'
    media concatenated into a fabricated timeline.

    The refusal is a :class:`MediaProbeError`, so a sweep that reports faults
    per entry vents it and keeps going rather than aborting.
    """
    ds = _make_dataset(tmp_path)
    first_dir = tmp_path / "media_raw" / "one"
    second_dir = tmp_path / "media_raw" / "two"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    first = first_dir / "clip.mp4"
    second = second_dir / "clip.mp4"
    write_mpeg4_mp4(first, frames=6)
    write_mpeg4_mp4(second, frames=6)
    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [
            _row(group="g1", sequence="clip", abs_path=first),
            _row(group="g2", sequence="clip", abs_path=second),
        ],
    )

    with pytest.raises(MediaProbeError) as excinfo:
        ds.match_media_rows("", "clip")
    assert isinstance(excinfo.value, AmbiguousMediaMatchError)
    assert "g1" in str(excinfo.value) and "g2" in str(excinfo.value)


def test_a_group_named_alongside_a_shared_sequence_name_resolves_that_group(
    tmp_path: Path,
):
    """Naming the group answers the refusal above: the fallback narrows to that
    group first, so only one entry survives and there is nothing to refuse."""
    ds = _make_dataset(tmp_path)
    first_dir = tmp_path / "media_raw" / "one"
    second_dir = tmp_path / "media_raw" / "two"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    first = first_dir / "clip.mp4"
    second = second_dir / "clip.mp4"
    write_mpeg4_mp4(first, frames=6)
    write_mpeg4_mp4(second, frames=6)
    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [
            _row(group="g1", sequence="clip", abs_path=first),
            _row(group="g2", sequence="clip", abs_path=second),
        ],
    )

    resolved = ds.resolve_media("g2", "CLIP")
    assert [p.resolve() for p in resolved.paths] == [second.resolve()]


def test_a_named_group_does_not_resolve_another_groups_file(tmp_path: Path):
    """A request naming a group asks for that group's media. The fallback
    answers on a filename, which carries no group, so it must not hand back
    another group's row just because the file is named for the sequence."""
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "clip_a.mp4"
    write_mpeg4_mp4(original, frames=6)
    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [_row(group="g1", sequence="clip_a", abs_path=original)],
    )

    with pytest.raises(FileNotFoundError):
        ds.resolve_media("g2", "clip_a.mp4")


def test_a_named_group_still_resolves_its_own_file(tmp_path: Path):
    """The group check narrows across groups only: within the requested group
    the fallback still bridges the extension the ``name`` cell carries."""
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "clip_a.mp4"
    write_mpeg4_mp4(original, frames=6)
    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [_row(group="g1", sequence="clip_a", abs_path=original)],
    )

    resolved = ds.resolve_media("g1", "clip_a.mp4")
    assert [p.resolve() for p in resolved.paths] == [original.resolve()]


def test_a_sequence_does_not_resolve_a_row_whose_file_merely_carries_its_name(
    tmp_path: Path,
):
    """A row's ``name`` is a filename, not an identity: entry ``session1`` may
    well hold a file called ``trial.mp4``. Matching on the file would hand the
    unrelated sequence ``trial`` that entry's media."""
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "trial.mp4"
    write_mpeg4_mp4(original, frames=6)
    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [_row(group="", sequence="session1", abs_path=original)],
    )

    with pytest.raises(FileNotFoundError, match=r"sequence 'trial'"):
        ds.resolve_media("", "trial")


def test_a_chunked_sequence_resolves_every_one_of_its_files_in_order(
    tmp_path: Path,
):
    """A sequence spanning several files is one entry, so the fallback must
    return all of its rows in ``video_order`` -- not the one whose filename
    happens to answer -- and must not read one entry's rows as two."""
    ds = _make_dataset(tmp_path)
    first = tmp_path / "media_raw" / "part1.mp4"
    second = tmp_path / "media_raw" / "part2.mp4"
    write_mpeg4_mp4(first, frames=6)
    write_mpeg4_mp4(second, frames=6)
    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [
            _row(group="g", sequence="rec", abs_path=first, video_order=1),
            _row(group="g", sequence="rec", abs_path=second, video_order=0),
        ],
    )

    resolved = ds.resolve_media("g", "REC")
    assert [p.resolve() for p in resolved.paths] == [
        second.resolve(),
        first.resolve(),
    ]


def test_a_chunk_filename_does_not_resolve_the_sequence_holding_it(
    tmp_path: Path,
):
    """Naming one file of a chunked sequence asks for something that is not an
    entry. Answering with that chunk alone would present a fragment of the
    recording as the whole of it."""
    ds = _make_dataset(tmp_path)
    first = tmp_path / "media_raw" / "part1.mp4"
    second = tmp_path / "media_raw" / "part2.mp4"
    write_mpeg4_mp4(first, frames=6)
    write_mpeg4_mp4(second, frames=6)
    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [
            _row(group="g", sequence="rec", abs_path=first, video_order=1),
            _row(group="g", sequence="rec", abs_path=second, video_order=0),
        ],
    )

    with pytest.raises(FileNotFoundError, match=r"sequence 'part1.mp4'"):
        ds.resolve_media("g", "part1.mp4")


def test_an_empty_sequence_resolves_nothing_through_the_fallback(tmp_path: Path):
    """An empty sequence names no entry, so the fallback has nothing to compare.
    Left to match, it answers for every row that shares its emptiness."""
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "clip_a.mp4"
    write_mpeg4_mp4(original, frames=6)
    row = _row(group="g1", sequence="clip_a", abs_path=original)
    row["sequence"] = ""
    row["sequence_safe"] = ""
    _write_index(tmp_path / "media_raw" / "index.csv", [row])

    with pytest.raises(FileNotFoundError):
        ds.resolve_media("g2", "")


def test_a_multi_camera_sequence_reached_by_the_fallback_still_refuses(
    tmp_path: Path,
):
    """The camera rule survives the fallback: concatenating two cameras would
    fabricate a timeline whichever tier selected the rows."""
    ds = _make_dataset(tmp_path)
    left = tmp_path / "media_raw" / "left.mp4"
    right = tmp_path / "media_raw" / "right.mp4"
    write_mpeg4_mp4(left, frames=6)
    write_mpeg4_mp4(right, frames=6)
    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [
            _row(group="g", sequence="rec", abs_path=left, camera="cam0"),
            _row(group="g", sequence="rec", abs_path=right, camera="cam1"),
        ],
    )

    with pytest.raises(MediaProbeError, match="spans 2 cameras"):
        ds.resolve_media("g", "REC")

    resolved = ds.resolve_media("g", "REC", camera="cam1")
    assert [p.resolve() for p in resolved.paths] == [right.resolve()]
