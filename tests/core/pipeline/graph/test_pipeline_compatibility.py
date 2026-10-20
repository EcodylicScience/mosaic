"""Which steps may be wired to which, checked with no dataset in hand.

The claim under test is not only that a bad edge is refused, but that it is
refused **without a dataset** -- because a canvas asks this while a wire is being
drawn, before its user has chosen anything to run it against. So the sharpest
case here runs in a subprocess where constructing a dataset raises.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from mosaic.core.pipeline.graph import (
    ConsumerDecl,
    Declaration,
    DeclarationCatalog,
    EditorCatalog,
    ProducerDecl,
    can_connect,
    can_join,
    compatible_consumers,
    compatible_producers,
    declaration_catalog,
    editor_catalog,
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


def test_a_track_input_refuses_a_producer_that_drops_the_track_frame() -> None:
    """Decided from the producer's declared output columns.

    A producer passing its input through, or writing ``X`` and ``Y``, keeps the
    track frame. A producer whose declared output lacks both does not. A producer
    with undeclared output columns is allowed, and input resolution checks its
    resolved output at run time. Refusing it would refuse every undeclared track
    producer, ``movement-smooth -> speed-angvel`` among them.
    """
    keeps = ProducerDecl(
        name="trajectory-smooth",
        kind="feature",
        level="individual",
        keeps_track_frame=True,
    )
    drops = ProducerDecl(
        name="nearest-neighbor",
        kind="feature",
        level="individual",
        keeps_track_frame=False,
    )
    undeclared = ProducerDecl(
        name="movement-smooth", kind="feature", level="individual"
    )

    assert can_connect(TRACKS, TRACK_SHAPED)
    assert can_connect(keeps, TRACK_SHAPED)
    assert can_connect(undeclared, TRACK_SHAPED)
    refused = can_connect(drops, TRACK_SHAPED)
    assert not refused
    assert "neither passes its input through nor writes X and Y" in refused.reason
    # A consumer of any table takes it.
    assert can_connect(drops, JOINER)


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
    """Each candidate is answered, a refused one with the reason ``can_join`` gives."""
    answers = possible_connections(JOINER, small_catalog, existing=[PAIRWISE])
    assert set(answers) == set(small_catalog.names())
    assert not all(answers.values())
    for name, verdict in answers.items():
        declared = small_catalog.get(name)
        assert declared is not None
        assert verdict == can_join([PAIRWISE, declared.produces], JOINER), name


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
    """An API serves the declarations, and a client answers with the same code path.

    Each one is served inside its editor-catalog entry, and reads back from that
    JSON equal to the declaration validation reads, column contract included.
    """
    served = EditorCatalog.model_validate_json(editor_catalog().model_dump_json())
    declarations = declaration_catalog()
    for entry in served.entries:
        assert entry.declaration == declarations.get(entry.name), entry.name


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


def test_convert_labels_declares_writes_labels() -> None:
    """The one op a labels field may reference, and it writes no tracks."""
    catalog = declaration_catalog()

    writers = {
        name
        for name, declared in catalog.entries.items()
        if declared.produces.writes_labels
    }

    assert writers == {"convert-labels"}
    assert not catalog.entries["convert-labels"].produces.writes_tracks
    converter = catalog.entries["convert-tracks"]
    assert converter.produces.writes_tracks
    assert not converter.consumes.reads_media
    assert converter.scope_takes == "none"


def test_the_media_domain_ops_are_the_ones_that_write_media() -> None:
    """The answer is read off each op's declared domain, without a name list.

    The three media ops that wrote media before a variant existed keep their
    answer, and ``preprocess``, which declares the media domain under a separate
    category, writes media as well.
    """
    catalog = declaration_catalog()

    writers = {
        name
        for name, declared in catalog.entries.items()
        if declared.produces.kind == "op" and declared.produces.writes_media
    }

    assert writers == {"transcode", "export-store", "export-joined", "preprocess"}
    assert not any(catalog.entries[name].produces.writes_tracks for name in writers)


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


# --- lanes --------------------------------------------------------------------


def test_a_lane_is_decided_from_what_a_step_declares() -> None:
    """GPU work splits into fits and the rest, ffmpeg work has its own lane, and
    everything else takes the default one."""
    from mosaic.core.pipeline.graph import lane_for

    catalog = declaration_catalog()
    expected = {
        "speed-angvel": "feature-compute",
        "prepare-training-data": "feature-compute",
        "lightning-action": "gpu-train",  # a global fitter that declares a GPU, whatever its device
        "transcode": "transcode",
        "export-joined": "transcode",
        "export-store": "transcode",
        "train-pose": "gpu-train",
        "train-sleap": "gpu-train",
        "feral": "gpu-train",
        "sequence-classifier": "gpu-train",
        "kpms": "gpu-train",
        "vame": "gpu-train",
        "global-identity-model": "gpu-train",
        "infer-pose": "gpu-infer",
        "trex": "gpu-infer",
    }
    for name, lane in expected.items():
        declared = catalog.get(name)
        assert declared is not None
        assert lane_for(declared) == lane, name


def test_a_submitted_job_finds_its_lane_by_name() -> None:
    """What a control plane holds is a feature slug or an op kind, not a declaration."""
    from mosaic.core.pipeline.graph import lane_for_step

    assert lane_for_step("train-pose") == "gpu-train"
    assert lane_for_step("speed-angvel") == "feature-compute"
    with pytest.raises(KeyError, match="train-poze"):
        _ = lane_for_step("train-poze")


def test_trex_takes_a_gpu_lane_from_its_own_declaration() -> None:
    """Its category is 'convert', which would not imply a GPU; it says so itself."""
    from mosaic.core.pipeline.graph import resource_class_of

    declared = declaration_catalog().get("trex")
    assert declared is not None
    assert declared.category == "convert"
    assert resource_class_of(declared) == "gpu"


def test_every_declaration_resolves_to_a_lane() -> None:
    """A step planned without a lane has nowhere to be offered."""
    from mosaic.core.pipeline.graph import lane_for

    catalog = declaration_catalog()
    for name in catalog.names():
        declared = catalog.get(name)
        assert declared is not None
        assert lane_for(declared)
