"""The storage name of a feature run, and the slug read back from it."""

from __future__ import annotations

import pytest

from mosaic.core.pipeline.storage_names import derive_storage_name, feature_slug_of


def test_a_definition_stores_under_a_dotted_name() -> None:
    assert derive_storage_name("def:x", "tracks") == "def.x__from__tracks"
    assert derive_storage_name("def:x", None) == "def.x"


def test_every_version_of_a_definition_stores_under_its_name() -> None:
    digest = "ab" * 32
    assert derive_storage_name(f"def:x@{digest}", "tracks") == "def.x__from__tracks"
    assert derive_storage_name(f"def:x@{digest[:12]}", None) == "def.x"


def test_the_slug_of_a_definition_storage_name_is_the_def_slug() -> None:
    assert feature_slug_of("def.x__from__tracks") == "def:x"
    assert feature_slug_of("def.speed-x") == "def:speed-x"


def test_a_shortened_definition_storage_name_keeps_its_slug() -> None:
    name = derive_storage_name("def:x", "+".join(f"input-{i}" for i in range(80)))
    assert name.startswith("def.x__from__")
    assert feature_slug_of(name) == "def:x"


@pytest.mark.parametrize(
    "slug", ["speed", "speed-x", "nearest-neighbor", "pair-series-correlation"]
)
def test_a_built_in_slug_round_trips_unchanged(slug: str) -> None:
    assert feature_slug_of(derive_storage_name(slug, "tracks")) == slug
    assert derive_storage_name(slug, "tracks") == f"{slug}__from__tracks"
    assert feature_slug_of(slug) == slug
