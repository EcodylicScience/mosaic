"""Tests for the per-sequence composition hashes (item 4.4's pure half).

No fixture and no filesystem: ``composition`` turns already-read rows into a
digest and nothing else, which is what lets these values be pinned as literals.
"""

from __future__ import annotations

from mosaic.core.pipeline.composition import (
    MediaMember,
    SourceMember,
    media_composition,
    media_composition_payload,
    source_composition_payload,
    tracks_raw_composition,
)


def _media(*specs: tuple[str, int, str]) -> list[MediaMember]:
    return [
        MediaMember(camera=camera, video_order=order, uid=uid)
        for camera, order, uid in specs
    ]


def _sources(*specs: tuple[str, str]) -> list[SourceMember]:
    return [
        SourceMember(name=name, digest=digest, algo="md5") for name, digest in specs
    ]


# --- the mixed ordering rule ------------------------------------------------


def test_the_camera_list_is_sorted_by_the_builder() -> None:
    """Two callers may hand the cameras over in either order and mean the same."""
    forward = media_composition(_media(("a", 0, "u1"), ("b", 0, "u2")))
    reversed_ = media_composition(_media(("b", 0, "u2"), ("a", 0, "u1")))
    assert forward == reversed_


def test_the_uid_list_within_a_camera_is_never_sorted() -> None:
    """Order inside a camera is ``video_order``, and it is the whole point.

    Sorting it would make two arrangements of one sequence hash alike -- exactly
    the change a media composition exists to detect.
    """
    first = media_composition(_media(("", 0, "u1"), ("", 1, "u2")))
    swapped = media_composition(_media(("", 0, "u2"), ("", 1, "u1")))
    assert first.digest != swapped.digest


def test_the_uid_order_is_video_order_not_argument_order() -> None:
    """The payload is built from the column, so the caller's iteration cannot leak."""
    as_given = media_composition(_media(("", 1, "u2"), ("", 0, "u1")))
    in_order = media_composition(_media(("", 0, "u1"), ("", 1, "u2")))
    assert as_given == in_order


# --- cardinality ------------------------------------------------------------


def test_a_repeated_uid_is_not_collapsed() -> None:
    """Two byte-identical videos in one sequence share one uuid, by design.

    A ``set`` would deduplicate at construction, before ``identity_ready`` ever
    saw it, and a two-video sequence would hash like a one-video one.
    """
    twice = media_composition(_media(("", 0, "u1"), ("", 1, "u1")))
    once = media_composition(_media(("", 0, "u1")))
    assert twice.digest != once.digest
    assert twice.member_count == 2


def test_a_repeated_source_file_is_not_collapsed() -> None:
    duplicated = tracks_raw_composition(_sources(("a.npy", "d1"), ("b.npy", "d1")))
    single = tracks_raw_composition(_sources(("a.npy", "d1")))
    assert duplicated.digest != single.digest
    assert duplicated.member_count == 2


# --- the three states -------------------------------------------------------


def test_a_member_with_no_identity_makes_the_whole_composition_unestablishable() -> (
    None
):
    """Partial is not an option: it would compare equal to a different sequence."""
    partial = media_composition(_media(("", 0, "u1"), ("", 1, "")))
    assert partial.digest == ""
    assert partial.member_count == 2, "the count still says how much was there"


def test_a_source_with_no_checksum_makes_the_composition_unestablishable() -> None:
    partial = tracks_raw_composition(_sources(("a.npy", "d1"), ("b.npy", "")))
    assert partial.digest == ""
    assert partial.member_count == 2


def test_zero_members_is_a_real_digest_not_an_empty_one() -> None:
    """ "Every video was deleted" is a detected change, not a disappearance.

    Computed-empty and not-establishable must be distinguishable, which is why
    an empty member list mints a comparable value rather than ``""``.
    """
    empty = media_composition([])
    assert empty.digest != ""
    assert empty.member_count == 0
    assert empty.digest != media_composition(_media(("", 0, "u1"))).digest


def test_an_unestablishable_composition_is_not_equal_to_another_one() -> None:
    """``"" == ""`` is never "unchanged", and the count is what says so."""
    two = media_composition(_media(("", 0, ""), ("", 1, "")))
    three = media_composition(_media(("", 0, ""), ("", 1, ""), ("", 2, "")))
    assert two.digest == three.digest == ""
    assert two != three


# --- domain separation ------------------------------------------------------


def test_the_two_kinds_do_not_share_a_digest() -> None:
    """A media composition and a source one over comparable data are distinct."""
    media = media_composition_payload(_media(("", 0, "u1")))
    source = source_composition_payload("tracks_raw", _sources(("u1", "d1")))
    assert media["kind"] != source["kind"]


def test_the_root_name_separates_two_source_compositions() -> None:
    """The same files under two roots must not mint one value."""
    members = _sources(("a.npy", "d1"))
    assert source_composition_payload("tracks_raw", members) != (
        source_composition_payload("labels_raw", members)
    )


def test_the_algorithm_is_inside_the_payload() -> None:
    """Switching it changes every digest rather than producing a look-alike."""
    md5 = tracks_raw_composition([SourceMember(name="a.npy", digest="d1", algo="md5")])
    other = tracks_raw_composition(
        [SourceMember(name="a.npy", digest="d1", algo="blake2b")]
    )
    assert md5.digest != other.digest


# --- the wrapper the golden corpus pins -------------------------------------


def test_the_media_payload_shape_is_the_documented_one() -> None:
    payload = media_composition_payload(
        _media(("camB", 0, "u3"), ("camA", 1, "u2"), ("camA", 0, "u1"))
    )
    assert payload == {
        "kind": "media_raw",
        "cameras": [["camA", ["u1", "u2"]], ["camB", ["u3"]]],
    }


def test_the_source_payload_shape_is_the_documented_one() -> None:
    payload = source_composition_payload(
        "tracks_raw", _sources(("b.npy", "d2"), ("a.npy", "d1"))
    )
    assert payload == {
        "kind": "tracks_raw",
        "files": [["a.npy", "md5", "d1"], ["b.npy", "md5", "d2"]],
    }
