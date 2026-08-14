"""Which steps may be wired to which, checked with no dataset in hand.

The claim under test is not only that a bad edge is refused, but that it is
refused **without a dataset** -- because a canvas asks this while a wire is being
drawn, before its user has chosen anything to run it against. So the sharpest
case here runs in a subprocess where constructing a dataset raises.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict

import pytest

from mosaic.core.pipeline.graph import (
    ConsumerDecl,
    Declaration,
    DeclarationCatalog,
    ProducerDecl,
    can_connect,
    can_join,
    compatible_consumers,
    compatible_producers,
    declaration_catalog,
    possible_connections,
    resolve_emits,
)

# --- hand-built declarations, so the unit tests need no registry -------------

TRACKS = ProducerDecl(name="tracks", kind="tracks", level="individual")
SPEED = ProducerDecl(name="speed-angvel", kind="feature", level="individual")
PAIRWISE = ProducerDecl(name="pair-position", kind="feature", level="pair")
COLLECTIVE = ProducerDecl(
    name="collective-motion-metrics", kind="feature", level="global"
)
TREX = ProducerDecl(name="trex", kind="op", level="individual", writes_tracks=True)

JOINER = ConsumerDecl(
    name="temporal-stack", kind="feature", accepts_tracks=True, accepts_features=True
)
TRACK_SHAPED = ConsumerDecl(
    name="speed-angvel",
    kind="feature",
    accepts_tracks=True,
    accepts_features=True,
    requires_track_shape=True,
)
FEATURES_ONLY = ConsumerDecl(name="global-tsne", kind="feature", accepts_features=True)
SELF_LOADING = ConsumerDecl(name="kpms", kind="feature", takes_no_inputs=True)
CROSS = ConsumerDecl(
    name="interaction-crop-pipeline",
    kind="feature",
    accepts_tracks=True,
    accepts_features=True,
    cross_joins=True,
)


# --- one wire ----------------------------------------------------------------


def test_a_feature_may_read_tracks_and_another_feature() -> None:
    assert can_connect(TRACKS, JOINER)
    assert can_connect(SPEED, JOINER)


def test_an_op_is_not_a_feature_input() -> None:
    """What a feature reads from a tracker is its tracks variant, not the run."""
    verdict = can_connect(TREX, JOINER)
    assert not verdict
    assert "tracks field" in verdict.reason


def test_a_consumer_taking_no_inputs_refuses_everything() -> None:
    verdict = can_connect(TRACKS, SELF_LOADING)
    assert not verdict
    assert "no pipeline inputs" in verdict.reason


def test_a_track_shaped_consumer_does_not_refuse_a_feature_here() -> None:
    """Advisory, not a refusal, because refusing it would be wrong more often.

    ``TrackInputs`` accepts a ``Result`` from a track-*producing* feature, and
    that set is open -- ``trajectory-smooth -> speed-angvel`` is an ordinary
    chain. Refusing every feature would refuse that, which is a false refusal;
    the real check reads the resolved output's columns at input resolution.
    """
    assert can_connect(TRACKS, TRACK_SHAPED)
    assert can_connect(SPEED, TRACK_SHAPED)
    assert TRACK_SHAPED.requires_track_shape


def test_a_features_only_consumer_refuses_raw_tracks() -> None:
    """``global-tsne`` fits on feature columns; there are none in a tracks table."""
    assert can_connect(SPEED, FEATURES_ONLY)
    verdict = can_connect(TRACKS, FEATURES_ONLY)
    assert not verdict
    assert "does not read tracks" in verdict.reason


def test_a_verdict_is_truthy_and_carries_its_reason() -> None:
    """A greyed-out option a user cannot explain is worse than an absent one."""
    refused = can_connect(TREX, JOINER)
    assert bool(refused) is False
    assert refused.reason
    assert can_connect(SPEED, JOINER).reason == ""


# --- the set ------------------------------------------------------------------


def test_one_input_is_always_alignable() -> None:
    assert can_join([SPEED], JOINER)
    assert can_join([PAIRWISE], JOINER)


def test_joining_an_individual_to_a_pair_is_refused() -> None:
    """The silent cartesian product, refused before it can be produced."""
    verdict = can_join([SPEED, PAIRWISE], JOINER)
    assert not verdict
    assert "pair every row" in verdict.reason


def test_joining_two_inputs_at_one_level_is_allowed() -> None:
    other = ProducerDecl(name="heading", kind="feature", level="individual")
    assert can_join([SPEED, other], JOINER)


def test_an_unidentified_aggregate_broadcasts_rather_than_fans_out() -> None:
    """A per-frame aggregate legitimately joins onto per-id rows on frame."""
    assert can_join([SPEED, COLLECTIVE], JOINER)


def test_the_cross_join_escape_is_honoured() -> None:
    """One consumer joins across levels on purpose and pays in memory."""
    assert not can_join([SPEED, PAIRWISE], JOINER)
    assert can_join([SPEED, PAIRWISE], CROSS)


# --- enumeration --------------------------------------------------------------


@pytest.fixture
def small_catalog() -> DeclarationCatalog:
    def entry(producer: ProducerDecl) -> Declaration:
        return Declaration(
            produces=producer,
            consumes=ConsumerDecl(name=producer.name, kind="feature"),
            emits="individual",
        )

    return DeclarationCatalog(
        entries={p.name: entry(p) for p in (TRACKS, SPEED, PAIRWISE, COLLECTIVE)}
    )


def test_possible_connections_narrows_once_a_wire_is_taken(
    small_catalog: DeclarationCatalog,
) -> None:
    """The case that proves ``existing`` is load-bearing rather than decoration.

    With nothing wired in, a pair-level producer is a valid first input. Once an
    individual-level one is taken, it stops being one -- and a palette computed
    from the consumer alone would keep offering it.
    """
    empty = possible_connections(JOINER, small_catalog)
    assert empty["pair-position"]

    taken = possible_connections(JOINER, small_catalog, existing=[SPEED])
    assert not taken["pair-position"]
    assert taken["speed-angvel"]


def test_possible_connections_answers_for_every_candidate(
    small_catalog: DeclarationCatalog,
) -> None:
    """Refused candidates are reported with a reason, never omitted."""
    answers = possible_connections(JOINER, small_catalog, existing=[PAIRWISE])
    assert set(answers) == set(small_catalog.names())
    assert all(verdict.reason for verdict in answers.values() if not verdict)


def test_an_undeclared_candidate_is_refused_by_name(
    small_catalog: DeclarationCatalog,
) -> None:
    answers = possible_connections(JOINER, small_catalog, candidates=["nope"])
    assert not answers["nope"]
    assert "nothing declares" in answers["nope"].reason


def test_compatible_producers_lists_the_allowed_ones(
    small_catalog: DeclarationCatalog,
) -> None:
    assert compatible_producers(JOINER, small_catalog) == (
        "collective-motion-metrics",
        "pair-position",
        "speed-angvel",
        "tracks",
    )


def test_forward_and_reverse_listings_agree() -> None:
    """``p`` feeds ``c`` in one listing iff ``c`` reads ``p`` in the other."""
    catalog = declaration_catalog()
    names = catalog.names()
    for name in names:
        declared = catalog.get(name)
        assert declared is not None
        forward = set(compatible_consumers(declared.produces, catalog))
        for consumer_name in names:
            consumer = catalog.get(consumer_name)
            assert consumer is not None
            reverse = set(compatible_producers(consumer.consumes, catalog))
            assert (consumer_name in forward) == (name in reverse), (
                f"{name} -> {consumer_name} disagrees between the two listings"
            )


# --- passthrough resolution ---------------------------------------------------


def test_a_passthrough_takes_its_upstream_level() -> None:
    assert resolve_emits("as-input", ["pair"]) == "pair"
    assert resolve_emits("as-input", ["individual"]) == "individual"


def test_a_passthrough_with_no_upstream_reads_tracks() -> None:
    """The only thing a feature with no feature inputs can be reading."""
    assert resolve_emits("as-input") == "individual"


def test_unidentified_is_the_declaration_side_spelling_of_global() -> None:
    assert resolve_emits("unidentified") == "global"
    assert resolve_emits("pair") == "pair"


# --- the catalog --------------------------------------------------------------


def test_the_catalog_covers_every_feature_and_op() -> None:
    from mosaic.behavior.feature_library import FEATURES
    from mosaic.core.pipeline.ops import OPS

    catalog = declaration_catalog()
    for cls in FEATURES.values():
        assert getattr(cls, "name") in catalog
    for kind in OPS:
        assert kind in catalog
    assert "tracks" in catalog


def test_the_catalog_round_trips_through_json() -> None:
    """Phase 5 serves this object, so a client answers with the same code path."""
    catalog = declaration_catalog()
    payload = {name: asdict(declared) for name, declared in catalog.entries.items()}
    assert json.loads(json.dumps(payload)) == payload


def test_declarations_are_read_off_what_a_class_declares() -> None:
    """Spot-checks that the reader is reading, not guessing from a name."""
    catalog = declaration_catalog()

    tracks_reader = catalog.get("speed-angvel")
    assert tracks_reader is not None
    assert tracks_reader.consumes.accepts_tracks
    assert tracks_reader.consumes.requires_track_shape

    pairwise = catalog.get("pair-position")
    assert pairwise is not None
    assert pairwise.produces.level == "pair"

    tracker = catalog.get("trex")
    assert tracker is not None
    assert tracker.produces.writes_tracks
    assert tracker.consumes.reads_media
    assert not tracker.produces.writes_media

    encoder = catalog.get("transcode")
    assert encoder is not None
    assert encoder.produces.writes_media
    assert not encoder.produces.writes_tracks

    trainer = catalog.get("train-pose")
    assert trainer is not None
    assert not trainer.produces.writes_tracks
    assert not trainer.consumes.reads_media


# --- no dataset ---------------------------------------------------------------

_REFUSE_WITH_NO_DATASET = """
import sys

from mosaic.core.dataset import Dataset


def refuse(*args, **kwargs):
    raise AssertionError("a Dataset was constructed to answer a connection question")


Dataset.__init__ = refuse

from mosaic.core.pipeline.graph import ConsumerDecl, ProducerDecl, can_join

individual = ProducerDecl(name="speed-angvel", kind="feature", level="individual")
pairwise = ProducerDecl(name="pair-position", kind="feature", level="pair")
consumer = ConsumerDecl(
    name="temporal-stack", kind="feature", accepts_tracks=True, accepts_features=True
)

verdict = can_join([individual, pairwise], consumer)
assert not verdict, "a mismatched join must be refused"
assert "pair every row" in verdict.reason
print("OK")
"""


def test_a_mismatched_join_is_refused_with_no_dataset_constructed() -> None:
    """The whole point of keeping this dataset-independent, asserted rather than said.

    Constructing a ``Dataset`` is made to raise rather than blocking the module,
    because ``mosaic.core`` imports it unconditionally at package import -- so a
    blocked module would prove only that the package was imported, which is not
    the claim. The claim is that no dataset is *opened* to answer this.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _REFUSE_WITH_NO_DATASET],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"answering a connection question opened a dataset.\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "OK" in proc.stdout
