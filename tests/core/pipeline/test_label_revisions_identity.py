"""The ``_label_revisions`` term of a feature run's identifier."""

from __future__ import annotations

from dataclasses import replace

from mosaic.cli._features import build_feature
from mosaic.core.pipeline._utils import LabelRevisionTerm, ResolvedScope
from mosaic.core.pipeline.inventory.params import RunParams
from mosaic.core.pipeline.run import build_run_params_payload, compute_run_id

_ORIGIN = "0b7c1d52-aaaa-bbbb-cccc-000000000001"


def _term(
    *, digest: str = "d" * 64, revision: int = 3, payload_digest: str = "p" * 64
) -> LabelRevisionTerm:
    return LabelRevisionTerm(
        series="tags",
        origin_uuid=_ORIGIN,
        key="dataset",
        digest=digest,
        revision=revision,
        payload_digest=payload_digest,
    )


def _run_id(*terms: LabelRevisionTerm) -> str:
    feature = build_feature("speed-angvel", None, None)
    scope = ResolvedScope(label_revisions=terms)
    return compute_run_id(feature, None, None, scope)[0]


def test_a_scope_without_terms_keeps_its_identifier() -> None:
    feature = build_feature("speed-angvel", None, None)
    bare = compute_run_id(feature, None, None, ResolvedScope())[0]

    assert _run_id() == bare
    assert ResolvedScope().label_revisions == ()


def test_a_term_gives_another_identifier() -> None:
    assert _run_id(_term()) != _run_id()


def test_a_changed_digest_gives_another_identifier() -> None:
    assert _run_id(_term(digest="e" * 64)) != _run_id(_term())


def test_the_origin_and_key_of_a_term_enter_the_identifier() -> None:
    base = _run_id(_term())

    assert _run_id(replace(_term(), origin_uuid="another-dataset-uuid")) != base
    assert _run_id(replace(_term(), key="another-key")) != base


def test_the_order_of_the_terms_does_not_matter() -> None:
    first = _term()
    second = replace(_term(digest="f" * 64), key="other")

    assert _run_id(first, second) == _run_id(second, first)


def test_the_revision_number_and_payload_digest_stay_out_of_the_identifier() -> None:
    assert _run_id(_term(revision=9, payload_digest="q" * 64)) == _run_id(_term())


def test_the_params_payload_records_each_term() -> None:
    feature = build_feature("speed-angvel", None, None)
    scope = ResolvedScope(label_revisions=(_term(),))

    payload = build_run_params_payload(feature, None, None, scope, [])

    assert payload["_resolved"] == [
        {
            "where": "series[tags]",
            "feature": "",
            "run_id": None,
            "series": "tags",
            "key": "dataset",
            "origin_uuid": _ORIGIN,
            "revision": 3,
            "digest": "d" * 64,
            "payload_digest": "p" * 64,
        }
    ]


def test_a_payload_without_terms_records_no_series_entry() -> None:
    feature = build_feature("speed-angvel", None, None)

    payload = build_run_params_payload(feature, None, None, ResolvedScope(), [])

    assert payload["_resolved"] == []


def test_a_recorded_term_reads_back_and_names_no_upstream_run() -> None:
    feature = build_feature("speed-angvel", None, None)
    scope = ResolvedScope(label_revisions=(_term(),))
    payload = build_run_params_payload(feature, None, None, scope, [])

    params = RunParams.model_validate(payload)

    assert params.label_revisions() == (_term(),)
    assert params.upstream_references() == (frozenset(), frozenset())
    assert params.consumed_run_ids() == frozenset()


def test_a_document_without_terms_reads_no_terms() -> None:
    assert RunParams.model_validate({"_resolved": []}).label_revisions() == ()
