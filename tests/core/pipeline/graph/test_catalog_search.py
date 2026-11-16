"""Step types of the catalog index ranked against a query by ``search_catalog``.

The hand-built rows isolate one ranking rule each. The last tests search the
registered catalog.
"""

from __future__ import annotations

import json

import pytest

from mosaic.core.pipeline.graph import (
    CatalogIndexEntry,
    catalog_index,
    catalog_listing,
    search_catalog,
)


def _entry(
    name: str,
    summary: str = "",
    category: str = "per-frame",
    keywords: tuple[str, ...] = (),
) -> CatalogIndexEntry:
    """An index row that differs from the others by its name, summary and words."""
    return CatalogIndexEntry(
        name=name,
        type="feature",
        version="0.1",
        category=category,
        summary=summary,
        lane="feature-compute",
        emits="individual",
        requires=None,
        provides=None,
        passthrough=False,
        payload_families=False,
        unavailable=None,
        license_note=None,
        keywords=keywords,
    )


def _names(entries: list[CatalogIndexEntry], query: str) -> list[str]:
    return [match.entry.name for match in search_catalog(entries, query)]


def test_each_tier_ranks_above_the_next() -> None:
    """The name, its words begun, the name's text, part of it, then summary only."""
    ranked = [
        _entry("speed-angvel"),
        _entry("angvel-speedup"),
        _entry("airspeed-angvel"),
        _entry("speed-filter", summary="Smooths angvel."),
        _entry("smoother", summary="Smooths speed and angvel."),
    ]

    assert _names(ranked[::-1], "speed angvel") == [entry.name for entry in ranked]


def test_a_word_that_occurs_nowhere_leaves_the_entry_out() -> None:
    entries = [_entry("speed-angvel", summary="Speed and angular velocity.")]

    assert _names(entries, "speed heading") == []
    assert _names(entries, "angular") == ["speed-angvel"]


def test_more_words_equal_to_name_words_rank_first_within_a_tier() -> None:
    entries = [_entry("speedy-angvel"), _entry("speed-angvel-extra")]

    assert _names(entries, "speed angvel") == ["speed-angvel-extra", "speedy-angvel"]


def test_a_word_in_the_summary_ranks_above_one_in_the_category_only() -> None:
    entries = [
        _entry("aa", category="per-frame"),
        _entry("zz-longer", summary="One value per frame.", category="summary"),
    ]

    assert _names(entries, "frame") == ["zz-longer", "aa"]


def test_a_word_equal_to_the_singular_of_a_name_word_counts_as_equal() -> None:
    entries = [_entry("track-subsample"), _entry("tracks"), _entry("trackers")]

    assert _names(entries, "track") == ["tracks", "track-subsample", "trackers"]


def test_ties_go_to_the_shorter_name_then_the_name() -> None:
    entries = [_entry("speed-long"), _entry("speed-b"), _entry("speed-a")]

    assert _names(entries, "speed") == ["speed-a", "speed-b", "speed-long"]


def test_a_word_found_only_in_the_keywords_matches_as_a_summary_word() -> None:
    entries = [_entry("pairwise", keywords=("social force",)), _entry("heading")]

    ranked = search_catalog(entries, "force")

    assert [match.entry.name for match in ranked] == ["pairwise"]
    assert ranked[0].key[0] == "000004"


def test_a_name_word_and_a_keyword_word_match_at_tier_three() -> None:
    entries = [_entry("nn-delta-bins", keywords=("social force",))]

    ranked = search_catalog(entries, "delta force")

    assert [match.key[0] for match in ranked] == ["000003"]


def test_a_keyword_never_outranks_a_name_match() -> None:
    entries = [
        _entry("annotator", keywords=("force map",)),
        _entry("force-map"),
        _entry("force-field", summary="A map of forces."),
    ]

    assert _names(entries, "force map") == [
        "force-map",
        "force-field",
        "annotator",
    ]


def test_a_keyword_listed_in_a_name_only_query_changes_nothing() -> None:
    plain = [_entry("speed-angvel"), _entry("speed-b")]
    keyed = [_entry("speed-angvel", keywords=("velocity",)), _entry("speed-b")]

    assert _names(plain, "speed") == _names(keyed, "speed")


@pytest.mark.parametrize(
    "query", ["speed_angvel", "speed angvel", "Speed-Angvel", "  SPEED -- angvel "]
)
def test_case_spaces_hyphens_and_underscores_are_alike(query: str) -> None:
    entries = [_entry("angvel-speed"), _entry("speed-angvel"), _entry("speed")]

    ranked = search_catalog(entries, query)

    assert [match.entry.name for match in ranked] == ["speed-angvel", "angvel-speed"]
    assert ranked[0].key[0] == "000000"
    assert ranked[1].key[0] == "000001"


@pytest.mark.parametrize("query", ["", "   ", "-_"])
def test_a_query_without_words_lists_every_entry_by_name(query: str) -> None:
    entries = [_entry("speed-angvel"), _entry("heading"), _entry("body-scale")]

    ranked = search_catalog(entries, query)

    assert [match.entry.name for match in ranked] == [
        "body-scale",
        "heading",
        "speed-angvel",
    ]
    assert [match.key[-1] for match in ranked] == [
        "body-scale",
        "heading",
        "speed-angvel",
    ]
    assert {len(match.key) for match in ranked} == {
        len(search_catalog(entries, "speed")[0].key)
    }


def test_the_ranking_does_not_depend_on_the_input_order() -> None:
    entries = [
        _entry("speed-angvel"),
        _entry("nn-delta-bins", summary="Bin dangle and dspeed."),
        _entry("speed-b"),
        _entry("speed-a"),
        _entry("airspeed"),
    ]

    forward = search_catalog(entries, "speed")
    backward = search_catalog(entries[::-1], "speed")

    assert forward == backward
    assert len({match.key for match in forward}) == len(forward)
    assert [match.key for match in forward] == sorted(match.key for match in forward)


@pytest.mark.parametrize(
    ("query", "first"),
    [
        ("speed", "speed-angvel"),
        ("angvel", "speed-angvel"),
        ("kmeans", "global-kmeans"),
        ("track", "tracks"),
        ("egocentric crop", "egocentric-crop"),
        ("egocentric_crop", "egocentric-crop"),
    ],
)
def test_the_registered_catalog_ranks_the_named_step_first(
    query: str, first: str
) -> None:
    ranked = _names(list(catalog_index().entries), query)

    assert ranked[0] == first


@pytest.mark.parametrize(
    "query", ["social force", "social force map", "force map", "Force_Map"]
)
def test_the_registered_catalog_finds_the_social_force_features(query: str) -> None:
    found = set(_names(list(catalog_index().entries), query))

    assert {"nn-delta-response", "nn-delta-bins", "neighbor-delta-response"} <= found


def test_a_listing_groups_names_by_category_and_keeps_summaries_that_fit() -> None:
    entries = [
        _entry("speed", summary="Speed per frame.", category="per-frame"),
        _entry("kmeans", summary="Clusters templates.", category="global"),
        _entry("heading", summary="Heading per frame.", category="per-frame"),
        _entry("tracks", summary="The tracks.", category=""),
    ]

    listed = catalog_listing(entries, budget=1000)

    assert listed == {
        "per-frame": {"speed": "Speed per frame.", "heading": "Heading per frame."},
        "global": {"kmeans": "Clusters templates."},
        "": {"tracks": "The tracks."},
    }


def test_a_listing_over_its_budget_shortens_every_summary_at_a_word() -> None:
    long = "Per-frame body heading derived from pose keypoints and smoothed."
    entries = [_entry(f"step-{index}", summary=long) for index in range(10)]

    listed = catalog_listing(entries, budget=600)
    summaries = set(listed["per-frame"].values())

    assert len(json.dumps(listed, separators=(",", ":"))) <= 600
    assert len(summaries) == 1
    (summary,) = summaries
    assert summary.endswith("...")
    assert long.startswith(summary.removesuffix("..."))
    assert long[len(summary) - 3] == " "


def test_a_listing_too_long_for_any_summary_lists_names_alone() -> None:
    entries = [_entry(f"step-{index}", summary="Some words.") for index in range(10)]

    listed = catalog_listing(entries, budget=50)

    assert listed == {"per-frame": {entry.name: "" for entry in entries}}
