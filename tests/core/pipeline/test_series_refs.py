"""Resolving references to revisions of a label series, over the ``tags`` series."""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.annotations.tag_table import tag_table_payload
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.label_series_index import (
    LabelSeriesTamperedError,
    SeveralOrigins,
    resolve_series_refs,
)
from mosaic.core.pipeline.ops import IdentityDeferred
from mosaic.core.pipeline.label_series import (
    TAG_TABLE_KEY,
    TAGS_SERIES,
    LabelRevisionRef,
)
from mosaic.core.typed_attribute import TypedAttributeType
from tests.helpers import index_a_second_origin, make_dataset, tag_table, write_tags

TYPES: dict[str, TypedAttributeType] = {"treatment": "categorical"}


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    return make_dataset(tmp_path / "ds")


def _write_two_revisions(ds: Dataset) -> None:
    _ = write_tags(ds, tag_table({("g", "s1"): {"treatment": "a"}}, TYPES))
    _ = write_tags(ds, tag_table({("g", "s1"): {"treatment": "b"}}, TYPES))


def test_a_reference_without_a_revision_resolves_to_the_latest(ds: Dataset) -> None:
    _write_two_revisions(ds)

    (resolved,) = resolve_series_refs(
        ds, TAGS_SERIES, (LabelRevisionRef(key=TAG_TABLE_KEY),), kind="any-op"
    )

    expected = tag_table_payload(tag_table({("g", "s1"): {"treatment": "b"}}, TYPES))
    assert resolved.series == TAGS_SERIES
    assert resolved.key == TAG_TABLE_KEY
    assert resolved.origin_uuid == (ds.uuid or "")
    assert resolved.revision == 2
    assert resolved.payload.read_bytes() == expected
    assert resolved.digest == file_digest(resolved.payload)


def test_a_named_revision_resolves_to_that_revision(ds: Dataset) -> None:
    _write_two_revisions(ds)

    (resolved,) = resolve_series_refs(
        ds,
        TAGS_SERIES,
        (LabelRevisionRef(key=TAG_TABLE_KEY, revision=1),),
        kind="any-op",
    )

    expected = tag_table_payload(tag_table({("g", "s1"): {"treatment": "a"}}, TYPES))
    assert resolved.revision == 1
    assert resolved.payload.read_bytes() == expected


def test_resolution_keeps_the_order_of_the_references(ds: Dataset) -> None:
    _write_two_revisions(ds)
    refs = (
        LabelRevisionRef(key=TAG_TABLE_KEY, revision=2),
        LabelRevisionRef(key=TAG_TABLE_KEY, revision=1),
    )

    resolved = resolve_series_refs(ds, TAGS_SERIES, refs, kind="any-op")

    assert [item.revision for item in resolved] == [2, 1]


def test_a_missing_key_defers_naming_the_key(ds: Dataset) -> None:
    _write_two_revisions(ds)

    with pytest.raises(IdentityDeferred, match="other-key") as raised:
        _ = resolve_series_refs(
            ds, TAGS_SERIES, (LabelRevisionRef(key="other-key"),), kind="any-op"
        )

    assert raised.value.kind == "any-op"


def test_a_series_with_no_revision_defers_naming_the_key(ds: Dataset) -> None:
    with pytest.raises(IdentityDeferred, match=TAG_TABLE_KEY):
        _ = resolve_series_refs(
            ds, TAGS_SERIES, (LabelRevisionRef(key=TAG_TABLE_KEY),), kind="any-op"
        )


def test_a_missing_revision_defers_naming_the_revision(ds: Dataset) -> None:
    _write_two_revisions(ds)

    with pytest.raises(IdentityDeferred, match="revision 7"):
        _ = resolve_series_refs(
            ds,
            TAGS_SERIES,
            (LabelRevisionRef(key=TAG_TABLE_KEY, revision=7),),
            kind="any-op",
        )


def test_a_payload_that_is_not_on_disk_defers(ds: Dataset) -> None:
    saved = write_tags(ds, tag_table({("g", "s1"): {"treatment": "a"}}, TYPES))
    saved.path.unlink()

    with pytest.raises(IdentityDeferred, match="not on disk"):
        _ = resolve_series_refs(
            ds, TAGS_SERIES, (LabelRevisionRef(key=TAG_TABLE_KEY),), kind="any-op"
        )


def test_a_payload_edited_in_place_is_refused(ds: Dataset) -> None:
    saved = write_tags(ds, tag_table({("g", "s1"): {"treatment": "a"}}, TYPES))
    _ = saved.path.write_bytes(saved.path.read_bytes() + b" ")

    with pytest.raises(LabelSeriesTamperedError, match="immutable"):
        _ = resolve_series_refs(
            ds, TAGS_SERIES, (LabelRevisionRef(key=TAG_TABLE_KEY),), kind="any-op"
        )


def test_two_origins_without_an_origin_uuid_raise_key_error(ds: Dataset) -> None:
    _ = write_tags(ds, tag_table({("g", "s1"): {"treatment": "a"}}, TYPES))
    _ = index_a_second_origin(ds)

    with pytest.raises(KeyError, match="origin_uuid"):
        _ = resolve_series_refs(
            ds, TAGS_SERIES, (LabelRevisionRef(key=TAG_TABLE_KEY),), kind="any-op"
        )


def test_the_several_origins_error_names_the_series_key_and_origins(
    ds: Dataset,
) -> None:
    _ = write_tags(ds, tag_table({("g", "s1"): {"treatment": "a"}}, TYPES))
    other = index_a_second_origin(ds)

    with pytest.raises(SeveralOrigins) as raised:
        _ = resolve_series_refs(
            ds, TAGS_SERIES, (LabelRevisionRef(key=TAG_TABLE_KEY),), kind="any-op"
        )

    assert raised.value.series == TAGS_SERIES
    assert raised.value.key == TAG_TABLE_KEY
    assert raised.value.origins == tuple(sorted((ds.uuid or "", other)))
    assert isinstance(raised.value, KeyError)


def test_an_origin_uuid_selects_among_origins(ds: Dataset) -> None:
    _ = write_tags(ds, tag_table({("g", "s1"): {"treatment": "a"}}, TYPES))
    other = index_a_second_origin(ds)

    (resolved,) = resolve_series_refs(
        ds,
        TAGS_SERIES,
        (LabelRevisionRef(key=TAG_TABLE_KEY, origin_uuid=other),),
        kind="any-op",
    )

    assert resolved.origin_uuid == other
