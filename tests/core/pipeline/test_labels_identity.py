"""Unit tests for the label variant identity (item 9.3).

The label sibling of ``test_tracks_identity``: pin the shape of the minter and
the payload wrapper, and the domain separation that keeps a label variant from
ever reading like a tracks variant of the same registered format.
"""

from __future__ import annotations

from mosaic.core.pipeline.composition import (
    SourceMember,
    labels_raw_composition,
    tracks_raw_composition,
)
from mosaic.core.pipeline.labels_identity import (
    LABELS_IDENTITY_SCHEME,
    label_convert_variant_payload,
    label_converter_op,
    labels_run_id,
)
from mosaic.core.pipeline.op_identity import parse_op_run_id
from mosaic.core.pipeline.tracks_identity import converter_op


def test_scheme_is_born_marked() -> None:
    assert LABELS_IDENTITY_SCHEME == "1"


def test_op_prefix_distinguishes_labels_from_tracks() -> None:
    # calms21_npy is registered as both a track and a label converter, so the op
    # segment is what keeps their variants apart on disk and in the digest.
    assert label_converter_op("calms21_npy") == "convert-labels-calms21_npy"
    assert label_converter_op("calms21_npy") != converter_op("calms21_npy")


def test_run_id_shape_is_op_version_digest() -> None:
    run_id = labels_run_id(
        label_converter_op("boris_aggregated_csv"),
        "0.1",
        label_convert_variant_payload("behavior", {"fps": 30.0}),
    )
    parsed = parse_op_run_id(run_id)
    assert parsed is not None
    assert parsed.kind == "convert-labels-boris_aggregated_csv"
    assert parsed.version == "0.1"
    assert len(parsed.digest) == 10


def test_kind_term_separates_two_kinds_sharing_a_format() -> None:
    # One src_format feeding two kinds must not mint one identifier.
    behavior = labels_run_id(
        label_converter_op("f"), "0.1", label_convert_variant_payload("behavior", {})
    )
    id_tags = labels_run_id(
        label_converter_op("f"), "0.1", label_convert_variant_payload("id_tags", {})
    )
    assert behavior != id_tags


def test_params_participate_in_identity() -> None:
    a = labels_run_id(
        label_converter_op("f"),
        "0.1",
        label_convert_variant_payload("behavior", {"x": 1}),
    )
    b = labels_run_id(
        label_converter_op("f"),
        "0.1",
        label_convert_variant_payload("behavior", {"x": 2}),
    )
    assert a != b


def test_version_is_visible_and_out_of_the_digest() -> None:
    payload = label_convert_variant_payload("behavior", {"x": 1})
    v1 = labels_run_id(label_converter_op("f"), "0.1", payload)
    v2 = labels_run_id(label_converter_op("f"), "0.2", payload)
    # The version is a visible segment; the digest (after the version) is the same.
    assert v1 != v2
    assert v1.rsplit("-", 1)[1] == v2.rsplit("-", 1)[1]


def test_upstream_seam_is_omitted_when_absent() -> None:
    # A derived kind that later chains gains the term without moving a scored
    # kind's identifier: omitting the upstream digests differently from passing
    # one, but identical to not passing it at all.
    payload = label_convert_variant_payload("behavior", {})
    without = labels_run_id(label_converter_op("f"), "0.1", payload)
    also_without = labels_run_id(label_converter_op("f"), "0.1", payload, upstream=None)
    with_upstream = labels_run_id(
        label_converter_op("f"), "0.1", payload, upstream="tracks.0.1-aaaaaaaaaa"
    )
    assert without == also_without
    assert without != with_upstream


def test_labels_raw_composition_domain_separated_from_tracks_raw() -> None:
    members = [
        SourceMember(name="a.npy", digest="da", algo="md5"),
        SourceMember(name="b.npy", digest="db", algo="md5"),
    ]
    labels = labels_raw_composition(members)
    tracks = tracks_raw_composition(members)
    assert labels.digest and tracks.digest
    assert labels.digest != tracks.digest
    assert labels.member_count == tracks.member_count == 2


def test_labels_raw_composition_unestablishable_when_a_member_has_no_digest() -> None:
    members = [
        SourceMember(name="a.npy", digest="da", algo="md5"),
        SourceMember(name="b.npy", digest="", algo="md5"),
    ]
    result = labels_raw_composition(members)
    assert result.digest == ""
    assert result.member_count == 2
